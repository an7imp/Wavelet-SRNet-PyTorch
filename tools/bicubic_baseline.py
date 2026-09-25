"""Bicubic baseline: PSNR/SSIM of naive bicubic upscaling on a split.

PSNR/SSIM numbers are meaningless without this reference. A trained model is
only worth keeping if it beats bicubic by a clear margin.

    python tools/bicubic_baseline.py --config configs/config_8x.yaml --split val
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import torch
import torch.nn.functional as F

from dataset import create_dataloader, split_image_lists
from metrics import psnr_per_image, ssim_per_image
from utils import load_config, validate_config
from utils.env import resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bicubic PSNR/SSIM baseline")
    p.add_argument("--config", type=str, default="configs/config_8x.yaml")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--max-images", type=int, default=2000)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    cfg = validate_config(load_config(args.config))
    device = resolve_device(args.device)
    data = cfg["data"]

    image_list = split_image_lists(
        data["dataset_root"], seed=int(cfg["seed"]),
        val_split=float(data["val_split"]), test_split=float(data["test_split"]),
    )[args.split]
    if args.max_images:
        image_list = image_list[: args.max_images]

    loader = create_dataloader(
        root=data["dataset_root"], image_list=image_list, hr_size=data["hr_size"], scale=data["scale"],
        random_crop=False, hflip=False, cache_mode=data["cache"]["mode"], cache_dir=data["cache"]["dir"],
        cache_size=data["hr_size"], batch_size=cfg["test"]["batch_size"], shuffle=False,
        num_workers=cfg["test"]["num_workers"],
    )

    psnrs, ssims = [], []
    for batch in loader:
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        psnrs.append(psnr_per_image(up, hr, luminance=True).cpu())
        ssims.append(ssim_per_image(up, hr, luminance=False).cpu())

    psnr = torch.cat(psnrs)
    ssim = torch.cat(ssims)
    print("=" * 70)
    print(f"BICUBIC BASELINE  ({args.split} split, {psnr.numel()} images, {data['scale']}x)")
    print("-" * 70)
    print(f"  PSNR = {psnr.mean():.3f} dB  (std {psnr.std():.3f})")
    print(f"  SSIM = {ssim.mean():.4f}  (std {ssim.std():.4f})")
    print("=" * 70)
    print("Interpret: a trained Wavelet-SRNet should beat these by a clear margin")
    print("(typically +1-3 dB PSNR). If it does not, training has not converged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
