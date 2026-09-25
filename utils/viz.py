"""Qualitative validation-image saving (requirement #11).

Each saved sample is a 4-row grid for the first N validation images:

    row 0: LR upscaled with bicubic (the naive baseline the model must beat)
    row 1: SR prediction (model output, reconstructed via inverse Haar)
    row 2: HR ground truth
    row 3: amplified error |SR - HR| (brighter = larger error)

Looking at row 1 vs row 0 tells you at a glance whether the network adds real
high-frequency detail beyond bicubic; row 3 localises where it still fails.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.utils import save_image


@torch.no_grad()
def bicubic_upscale(lr: Tensor, size: tuple[int, int]) -> Tensor:
    """Bicubic-upscale an LR batch to ``size`` and clamp to [0, 1]."""
    up = F.interpolate(lr, size=size, mode="bicubic", align_corners=False, antialias=True)
    return up.clamp(0, 1)


@torch.no_grad()
def save_comparison_grid(
    lr: Tensor,
    sr: Tensor,
    hr: Tensor,
    path: str | Path,
    *,
    num_images: int = 6,
    error_gain: float = 5.0,
) -> Path:
    """Save the LR-bicubic / SR / HR / error comparison grid described above."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = min(num_images, lr.shape[0])
    lr, sr, hr = lr[:n].float(), sr[:n].float().clamp(0, 1), hr[:n].float().clamp(0, 1)
    lr_up = bicubic_upscale(lr, hr.shape[-2:])
    error = (sr - hr).abs().mul(error_gain).clamp(0, 1)

    grid = torch.cat([lr_up, sr, hr, error], dim=0)
    save_image(grid, path, nrow=n)
    return path


@torch.no_grad()
def save_triptych_per_image(
    lr: Tensor,
    sr: Tensor,
    hr: Tensor,
    out_dir: str | Path,
    *,
    start_index: int = 0,
    error_gain: float = 5.0,
) -> int:
    """Save one ``LR-bicubic | SR | HR | error`` strip per image; return count."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lr, sr, hr = lr.float(), sr.float().clamp(0, 1), hr.float().clamp(0, 1)
    lr_up = bicubic_upscale(lr, hr.shape[-2:])
    error = (sr - hr).abs().mul(error_gain).clamp(0, 1)
    for i in range(lr.shape[0]):
        strip = torch.stack([lr_up[i], sr[i], hr[i], error[i]], dim=0)
        save_image(strip, out_dir / f"sample_{start_index + i:06d}.png", nrow=4)
    return lr.shape[0]
