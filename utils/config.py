"""Configuration loading, merging and validation.

Design goals (see project requirements: "the config system must be clear,
documented, and hard to misuse"):

- A config may declare ``extends: <other.yaml>`` to inherit from a base file.
  This keeps debug configs tiny and avoids copy-paste drift.
- :func:`validate_config` fills documented defaults and raises a clear
  :class:`ConfigError` on anything suspicious (missing dataset root, non
  power-of-two scale, hr_size not divisible by scale, ...).
- ``scale`` is the single source of truth; the number of Haar levels is
  *derived* from it (``levels = log2(scale)``) so the two can never disagree.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration is missing required fields or inconsistent."""


# ---------------------------------------------------------------------------
# scale <-> Haar levels mapping (documented once, used everywhere)
# ---------------------------------------------------------------------------
# scale = 2 ** levels.  The network always operates at the LR spatial size and
# predicts ``3 * 4**levels`` wavelet coefficient channels.
#   scale=4  -> 2 Haar levels -> 48 channels   (original --upscale=2)
#   scale=8  -> 3 Haar levels -> 192 channels  (original --upscale=3, i.e. 8x SR)
#   scale=16 -> 4 Haar levels -> 768 channels  (original --upscale=4)
def scale_to_levels(scale: int) -> int:
    """Return the number of Haar levels for an integer power-of-two ``scale``."""
    scale = int(scale)
    if scale < 1:
        raise ConfigError(f"scale must be >= 1, got {scale}")
    levels = int(round(math.log2(scale)))
    if 2**levels != scale:
        raise ConfigError(
            f"scale={scale} is not a power of two; Haar packet reconstruction "
            f"requires scale in {{2, 4, 8, 16, ...}}."
        )
    return levels


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (override wins)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving an optional ``extends`` chain.

    ``extends`` is resolved relative to the file that declares it, so configs
    can live anywhere and still reference a sibling base file by name.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config {path} must define a mapping at the top level.")

    extends = cfg.pop("extends", None)
    if extends is not None:
        base_path = (path.parent / extends).resolve()
        base = load_config(base_path)
        cfg = _deep_merge(base, cfg)
    return cfg


# ---------------------------------------------------------------------------
# Validation / default-filling
# ---------------------------------------------------------------------------
def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer, got {value!r}") from exc


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number, got {value!r}") from exc


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate, normalise and fill defaults. Returns the same dict mutated.

    Raises :class:`ConfigError` with an actionable message on any problem.
    """
    cfg = copy.deepcopy(cfg)

    # --- top level ---------------------------------------------------------
    cfg.setdefault("seed", 42)
    cfg.setdefault("deterministic", False)
    # NOTE: both default OFF for this model. cuDNN benchmark autotuning is very
    # expensive for its branched grouped convolutions and yields no steady-state
    # gain; channels_last (NHWC) hits a slow cuDNN grouped-conv path (~2.7x
    # slower here). See OPTIMIZATION_REPORT.md section 4.
    cfg.setdefault("cudnn_benchmark", False)
    cfg.setdefault("tf32", True)
    cfg.setdefault("channels_last", False)
    cfg.setdefault("output_dir", "results/8x")
    cfg.setdefault("wavelet_weights_path", None)  # None -> robust generated Haar filters

    # --- data --------------------------------------------------------------
    if "data" not in cfg:
        raise ConfigError("Missing 'data' section in config.")
    data = cfg["data"]
    if not data.get("dataset_root"):
        raise ConfigError(
            "data.dataset_root is required and must point to a folder of images "
            "(e.g. 'img_align_celeba'). It must NOT be a hardcoded Colab path."
        )
    data["scale"] = _as_int(data.get("scale", 8), "data.scale")
    cfg["levels"] = scale_to_levels(data["scale"])  # derived, single source of truth
    data["hr_size"] = _as_int(data.get("hr_size", 128), "data.hr_size")
    if data["hr_size"] % data["scale"] != 0:
        raise ConfigError(
            f"data.hr_size ({data['hr_size']}) must be divisible by data.scale "
            f"({data['scale']}); LR size would be {data['hr_size'] / data['scale']}."
        )
    data["lr_size"] = data["hr_size"] // data["scale"]
    data.setdefault("val_split", 0.1)
    data.setdefault("test_split", 0.1)
    # overfit: if set to N, train and val use the SAME first-N images with
    # augmentation disabled (a true memorisation sanity check). Equivalent to the
    # CLI --overfit N. Leave null for normal training.
    data.setdefault("overfit", None)
    if data["overfit"] is not None:
        data["overfit"] = _as_int(data["overfit"], "data.overfit")
    data.setdefault("train_max_images", None)
    data.setdefault("val_max_images", None)
    data.setdefault("hflip", True)
    data.setdefault("random_crop", False)
    # Cache: {"mode": none|disk|memory, "dir": ..., "size": ...}
    cache = data.setdefault("cache", {})
    cache.setdefault("mode", "disk")
    if cache["mode"] not in {"none", "disk", "memory"}:
        raise ConfigError("data.cache.mode must be one of: none, disk, memory")
    cache.setdefault("dir", ".cache")
    cache.setdefault("size", data["hr_size"])
    cache["size"] = _as_int(cache["size"], "data.cache.size")
    if cache["size"] < data["hr_size"]:
        raise ConfigError(
            f"data.cache.size ({cache['size']}) must be >= data.hr_size "
            f"({data['hr_size']}) so the HR crop fits."
        )

    # --- model -------------------------------------------------------------
    model = cfg.setdefault("model", {})
    model.setdefault("image_channels", 3)
    model.setdefault("embedding_channels", [64, 128, 256, 512, 1024])
    model.setdefault("num_embedding_blocks", 2)
    model.setdefault("wavelet_channels", 32)
    model.setdefault("num_branch_blocks", 1)
    model["wavelet_levels"] = cfg["levels"]  # always derived from scale

    # --- loss --------------------------------------------------------------
    loss = cfg.setdefault("loss", {})
    loss.setdefault("image_channels", model["image_channels"])
    loss.setdefault("lambda_low", 0.01)
    loss.setdefault("lambda_high", 1.0)
    loss.setdefault("texture_weight", 1.0)
    loss.setdefault("image_weight", 0.1)
    loss.setdefault("identity_weight", 0.0)    # ArcFace identity loss weight (0 = off)
    loss.setdefault("identity_arch", "r50")    # ArcFace arch for the in-graph loss (R50 fits 12GB; R100 does not)
    loss.setdefault("texture_alpha", 1.2)
    loss.setdefault("texture_margin", 0.0)
    loss.setdefault("reduction", "mean")

    # --- train -------------------------------------------------------------
    train = cfg.setdefault("train", {})
    train["epochs"] = _as_int(train.get("epochs", 100), "train.epochs")
    train["batch_size"] = _as_int(train.get("batch_size", 64), "train.batch_size")
    if train["batch_size"] < 1:
        raise ConfigError("train.batch_size must be >= 1")
    train.setdefault("val_batch_size", train["batch_size"])
    train.setdefault("num_workers", 4)
    train.setdefault("prefetch_factor", 4)
    train.setdefault("persistent_workers", True)
    train["lr"] = _as_float(train.get("lr", 2e-4), "train.lr")
    train.setdefault("betas", [0.9, 0.999])
    train["weight_decay"] = _as_float(train.get("weight_decay", 5e-4), "train.weight_decay")
    train.setdefault("amp", True)
    train.setdefault("grad_clip_norm", None)
    train.setdefault("data_parallel", False)
    train.setdefault("compile", False)             # torch.compile (experimental on Windows)
    train.setdefault("log_every", 20)
    train.setdefault("val_every", 1)
    train.setdefault("save_every", 1)
    train.setdefault("max_steps", None)            # cap steps/epoch (smoke mode)
    train.setdefault("max_val_batches", None)
    train.setdefault("profile", False)             # per-stage synced timing
    train.setdefault("save_samples", True)
    train.setdefault("num_sample_images", 6)
    # scheduler: {"type": none|cosine|step, ...}
    sched = train.setdefault("scheduler", {})
    sched.setdefault("type", "none")
    if sched["type"] not in {"none", "cosine", "step"}:
        raise ConfigError("train.scheduler.type must be one of: none, cosine, step")
    sched.setdefault("min_lr", 1e-6)
    sched.setdefault("step_size", 50)
    sched.setdefault("gamma", 0.5)
    sched.setdefault("warmup_epochs", 0)
    # early stopping / best tracking
    early = train.setdefault("early_stopping", {})
    early.setdefault("enabled", False)
    early.setdefault("metric", "psnr")             # psnr | ssim
    early.setdefault("patience", 20)
    early.setdefault("min_delta", 0.0)

    # --- test --------------------------------------------------------------
    test = cfg.setdefault("test", {})
    test.setdefault("batch_size", train["val_batch_size"])
    test.setdefault("num_workers", train["num_workers"])

    # --- metrics -----------------------------------------------------------
    metrics = cfg.setdefault("metrics", {})
    metrics.setdefault("lpips", {"enabled": False, "net": "alex"})
    # ArcFace identity metric (SR vs HR cosine similarity). Self-disables if the
    # pretrained IResNet weights cannot be obtained, so training is never blocked.
    arcface = metrics.setdefault("arcface", {})
    arcface.setdefault("enabled", False)
    arcface.setdefault("arch", "r100")         # r18 | r34 | r50 | r100 (r100=Glint360K)
    arcface.setdefault("weights", None)        # local IResNet checkpoint; None -> best-effort download
    arcface.setdefault("download_dir", "weights")

    return cfg
