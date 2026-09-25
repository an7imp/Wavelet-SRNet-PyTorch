"""Evaluate a trained Wavelet-SRNet checkpoint on the test split.

Reports model PSNR/SSIM (and optional LPIPS) **side by side with the bicubic
baseline** so the numbers are interpretable, and saves qualitative comparison
images (LR-bicubic | SR | HR | error).

    python test.py --config configs/config_8x.yaml \
        --checkpoint results/8x/checkpoints/best.pth --save-images
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import create_dataloader, split_image_lists
from metrics import psnr_per_image, ssim_per_image, LPIPSMetric
from models import build_haar_transforms, build_model_from_config
from utils import load_config, validate_config
from utils.env import autocast, maybe_channels_last, print_environment, resolve_device, setup_backends
from utils.checkpoint import load_checkpoint
from utils.viz import save_triptych_per_image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate modern Wavelet-SRNet")
    p.add_argument("--config", type=str, default="configs/config_8x.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="results/test")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--save-images", action="store_true", help="save per-image comparison strips")
    p.add_argument("--save-limit", type=int, default=64, help="max comparison images to save")
    return p.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    cfg = validate_config(load_config(args.config))
    setup_backends(cfg)
    device = resolve_device(args.device)
    data = cfg["data"]
    levels = cfg["levels"]
    channels_last = bool(cfg["channels_last"])
    amp = bool(cfg["train"]["amp"])
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print_environment(device, amp=amp, channels_last=channels_last)

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

    model = build_model_from_config(cfg).to(device).eval()
    model = maybe_channels_last(model, channels_last, device)
    load_checkpoint(args.checkpoint, model=model, device=device, strict=False, restore_rng=False)
    _, wavelet_rec = build_haar_transforms(levels, params_path=cfg.get("wavelet_weights_path"))
    wavelet_rec = wavelet_rec.to(device)

    lpips_metric = None
    if cfg["metrics"]["lpips"].get("enabled", False):
        try:
            lpips_metric = LPIPSMetric(net=cfg["metrics"]["lpips"].get("net", "alex"), device=device)
        except Exception as exc:
            print(f"[test] LPIPS unavailable ({exc})")

    model_psnr, model_ssim, bic_psnr, bic_ssim, lpips_vals = [], [], [], [], []
    saved = 0
    for batch in loader:
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        if channels_last and device.type == "cuda":
            lr = lr.contiguous(memory_format=torch.channels_last)
        with autocast(device, amp):
            sr = wavelet_rec(model(lr)).float().clamp(0, 1)
        up = F.interpolate(lr.float(), size=hr.shape[-2:], mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)

        model_psnr.append(psnr_per_image(sr, hr, luminance=True).cpu())
        model_ssim.append(ssim_per_image(sr, hr, luminance=False).cpu())
        bic_psnr.append(psnr_per_image(up, hr, luminance=True).cpu())
        bic_ssim.append(ssim_per_image(up, hr, luminance=False).cpu())
        if lpips_metric is not None:
            try:
                lpips_vals.append(lpips_metric(sr, hr))
            except Exception:
                pass
        if args.save_images and saved < args.save_limit:
            n = min(lr.shape[0], args.save_limit - saved)
            saved += save_triptych_per_image(lr[:n], sr[:n], hr[:n], output_dir / "comparisons", start_index=saved)

    mp = torch.cat(model_psnr).mean().item()
    ms = torch.cat(model_ssim).mean().item()
    bp = torch.cat(bic_psnr).mean().item()
    bs = torch.cat(bic_ssim).mean().item()
    print("=" * 70)
    print(f"TEST RESULTS  ({args.split} split, {torch.cat(model_psnr).numel()} images, {data['scale']}x SR)")
    print("-" * 70)
    print(f"  {'':10s}{'PSNR (dB)':>12s}{'SSIM':>12s}")
    print(f"  {'bicubic':10s}{bp:>12.3f}{bs:>12.4f}")
    print(f"  {'WaveletSR':10s}{mp:>12.3f}{ms:>12.4f}")
    print(f"  {'gain':10s}{mp - bp:>+12.3f}{ms - bs:>+12.4f}")
    if lpips_vals:
        print(f"  WaveletSR LPIPS = {float(np.mean(lpips_vals)):.4f} (lower is better)")
    print("=" * 70)
    if args.save_images:
        print(f"  saved {saved} comparison strips to {output_dir / 'comparisons'}")
    verdict = "BEATS" if mp > bp else "DOES NOT beat"
    print(f"  Verdict: the model {verdict} bicubic ({mp - bp:+.2f} dB).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
