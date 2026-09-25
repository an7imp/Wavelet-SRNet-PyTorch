"""Device / CUDA / AMP / TF32 environment helpers.

These make it *obvious* whether CUDA is being used correctly (requirement #3)
and centralise the performance-oriented backend flags so every entry point
behaves identically.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Backend flags
# ---------------------------------------------------------------------------
def setup_backends(cfg: dict[str, Any]) -> None:
    """Apply performance backend flags from the config.

    - ``cudnn.benchmark`` lets cuDNN pick the fastest conv algorithm for the
      fixed input sizes used here (huge win because shapes never change).
    - TF32 massively speeds up matmul/conv on Ampere+ (RTX 30xx/40xx, A100)
      with negligible accuracy impact for SR training.
    """
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))
    if bool(cfg.get("tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # PyTorch >= 1.12 unified knob; harmless if the op is unsupported.
        with contextlib.suppress(Exception):
            torch.set_float32_matmul_precision("high")


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve a device string, warning loudly if CUDA was wanted but missing."""
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(
            "[env] WARNING: CUDA device requested but torch.cuda.is_available() "
            "is False -> falling back to CPU. Training will be extremely slow. "
            "Check your PyTorch/CUDA install."
        )
        device = torch.device("cpu")
    return device


# ---------------------------------------------------------------------------
# Memory-format helpers
# ---------------------------------------------------------------------------
def maybe_channels_last(module: nn.Module, enabled: bool, device: torch.device) -> nn.Module:
    """Convert a module to channels_last memory format on CUDA.

    channels_last (NHWC) is the fast path for convolutions with AMP on Ampere+.
    It is a no-op on CPU.
    """
    if enabled and device.type == "cuda":
        module = module.to(memory_format=torch.channels_last)
    return module


# ---------------------------------------------------------------------------
# AMP helpers (version-robust wrappers around torch.amp)
# ---------------------------------------------------------------------------
def make_grad_scaler(enabled: bool, device: torch.device) -> "torch.amp.GradScaler":
    """Create a GradScaler using the non-deprecated torch.amp API."""
    enabled = bool(enabled) and device.type == "cuda"
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except TypeError:  # very old torch fallback
        from torch.cuda.amp import GradScaler  # type: ignore

        return GradScaler(enabled=enabled)


def autocast(device: torch.device, enabled: bool, dtype: torch.dtype | None = None):
    """Return an autocast context manager for the given device."""
    enabled = bool(enabled) and device.type == "cuda"
    if dtype is None:
        dtype = torch.float16
    return torch.amp.autocast(device.type, enabled=enabled, dtype=dtype)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def gpu_memory_mb(device: torch.device) -> dict[str, float]:
    """Return allocated / reserved / total VRAM in MiB (zeros on CPU)."""
    if device.type != "cuda":
        return {"allocated": 0.0, "reserved": 0.0, "total": 0.0}
    idx = device.index or 0
    props = torch.cuda.get_device_properties(idx)
    return {
        "allocated": torch.cuda.memory_allocated(idx) / 1024**2,
        "reserved": torch.cuda.memory_reserved(idx) / 1024**2,
        "total": props.total_memory / 1024**2,
    }


def describe_environment(device: torch.device, *, amp: bool, channels_last: bool) -> dict[str, Any]:
    """Collect a structured description of the compute environment."""
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "amp": bool(amp) and device.type == "cuda",
        "channels_last": bool(channels_last) and device.type == "cuda",
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_capability": f"{props.major}.{props.minor}",
                "gpu_total_mem_mb": round(props.total_memory / 1024**2, 1),
                "gpu_multiprocessors": props.multi_processor_count,
            }
        )
    return info


def print_environment(device: torch.device, *, amp: bool, channels_last: bool) -> dict[str, Any]:
    """Pretty-print :func:`describe_environment` and return it."""
    info = describe_environment(device, amp=amp, channels_last=channels_last)
    print("=" * 70)
    print("COMPUTE ENVIRONMENT")
    print("-" * 70)
    for key, value in info.items():
        print(f"  {key:22s}: {value}")
    print("=" * 70)
    return info
