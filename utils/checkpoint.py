"""Full-state checkpointing with best/last tracking and safe resume.

A checkpoint stores *everything* needed to resume reproducibly (requirement
#13 / "checkpoints must include everything needed to resume"):

- model weights,
- optimizer state,
- AMP GradScaler state,
- LR scheduler state,
- epoch / global step,
- best metric seen so far,
- RNG states (python / numpy / torch / cuda),
- the (validated) config used for the run.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[checkpoint] WARNING: could not fully restore RNG state: {exc}")


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional["torch.amp.GradScaler"] = None,
    scheduler: Optional[Any] = None,
    epoch: int,
    global_step: int,
    best_metric: float,
    best_metric_name: str = "psnr",
    config: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "best_metric_name": best_metric_name,
        "config": config,
        "rng": _rng_state(),
        "format_version": 2,
    }
    if extra:
        payload.update(extra)
    # Write to a temp file then atomically replace, so a crash never leaves a
    # half-written checkpoint. The torch.save itself is retried: on Windows an
    # antivirus / indexer briefly locking the freshly written .pth can raise a
    # transient "inline_container ... unexpected pos" / badbit error, which a
    # short backoff resolves.
    tmp = path.with_suffix(path.suffix + ".tmp")
    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            torch.save(payload, tmp)
            tmp.replace(path)  # atomic on the same filesystem
            return path
        except (RuntimeError, OSError) as exc:
            last_err = exc
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt < 3:
                time.sleep(0.5 * attempt)
    raise RuntimeError(f"Failed to save checkpoint to {path} after 3 attempts: {last_err}")


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional["torch.amp.GradScaler"] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint and restore requested components.

    Returns the resume metadata: ``{epoch, global_step, best_metric, ...}``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Backwards compatibility with the old flat / module-pickle format.
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if isinstance(state, nn.Module):
        state = state.state_dict()
    missing, unexpected = _unwrap(model).load_state_dict(state, strict=strict)
    if not strict and (missing or unexpected):
        print(f"[checkpoint] non-strict load: {len(missing)} missing, {len(unexpected)} unexpected keys")

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if restore_rng and isinstance(ckpt, dict) and ckpt.get("rng"):
        _restore_rng(ckpt["rng"])

    return {
        "epoch": int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0,
        "global_step": int(ckpt.get("global_step", 0)) if isinstance(ckpt, dict) else 0,
        "best_metric": float(ckpt.get("best_metric", float("-inf"))) if isinstance(ckpt, dict) else float("-inf"),
        "best_metric_name": ckpt.get("best_metric_name", "psnr") if isinstance(ckpt, dict) else "psnr",
        "config": ckpt.get("config") if isinstance(ckpt, dict) else None,
    }


class BestTracker:
    """Tracks the best validation metric and drives early stopping.

    ``mode='max'`` for PSNR/SSIM (higher is better). Returns whether the current
    epoch is a new best so the caller can save ``best.pth``.
    """

    def __init__(self, *, mode: str = "max", min_delta: float = 0.0, patience: int = 0) -> None:
        self.mode = mode
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.best = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = 0
        self.num_bad_epochs = 0

    def _is_better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def update(self, value: float, epoch: int) -> bool:
        if self._is_better(value):
            self.best = value
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            return True
        self.num_bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.num_bad_epochs >= self.patience
