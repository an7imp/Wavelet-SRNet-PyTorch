"""DataLoader sanity check: are LR/HR pairs correctly aligned and sized?

Pulls a real batch from the configured dataset and verifies:
- shapes:  LR == B x 3 x (hr/scale)^2,  HR == B x 3 x hr^2
- value range in [0, 1]
- alignment: bicubic-upscaling LR back to HR size is close to HR (a misaligned
  or mismatched pair would give a large error). It also checks that the LR is a
  genuine downscale of *its own* HR (lower error than against a shuffled HR).

It saves a comparison grid so you can eyeball the pairing.

    python tools/dataloader_check.py --config configs/config_8x_smoke.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
import torch.nn.functional as F

from dataset import create_dataloader
from train import make_splits  # reuse the exact split logic
from utils import load_config, validate_config
from utils.viz import save_comparison_grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DataLoader alignment / shape check")
    p.add_argument("--config", type=str, default="configs/config_8x_smoke.yaml")
    p.add_argument("--split", type=str, default="train", choices=["train", "val"])
    p.add_argument("--out", type=str, default="results/diagnostics/dataloader_sample.png")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = validate_config(load_config(args.config))

    class _A:  # minimal stand-in for make_splits' argparse namespace
        overfit = None

    train_list, val_list = make_splits(cfg, _A())
    image_list = train_list if args.split == "train" else val_list
    data = cfg["data"]

    # Force a single-worker, no-shuffle loader for deterministic inspection.
    loader = create_dataloader(
        root=data["dataset_root"], image_list=image_list[: cfg["train"]["batch_size"]],
        hr_size=data["hr_size"], scale=data["scale"], random_crop=False, hflip=False,
        cache_mode="memory", cache_dir=data["cache"]["dir"], cache_size=data["hr_size"],
        batch_size=min(8, cfg["train"]["batch_size"]), shuffle=False, num_workers=0, drop_last=False,
    )
    batch = next(iter(loader))
    lr, hr, paths = batch["lr"], batch["hr"], batch["path"]

    print("=" * 70)
    print("DATALOADER CHECK")
    print("=" * 70)
    scale = data["scale"]
    exp_lr = (lr.shape[0], 3, data["hr_size"] // scale, data["hr_size"] // scale)
    exp_hr = (lr.shape[0], 3, data["hr_size"], data["hr_size"])
    shape_ok = tuple(lr.shape) == exp_lr and tuple(hr.shape) == exp_hr
    range_ok = bool(lr.min() >= 0 and lr.max() <= 1.0001 and hr.min() >= 0 and hr.max() <= 1.0001)
    print(f"  LR shape : {tuple(lr.shape)}  expected {exp_lr}  [{'PASS' if tuple(lr.shape)==exp_lr else 'FAIL'}]")
    print(f"  HR shape : {tuple(hr.shape)}  expected {exp_hr}  [{'PASS' if tuple(hr.shape)==exp_hr else 'FAIL'}]")
    print(f"  dtype    : lr={lr.dtype} hr={hr.dtype}")
    print(f"  range    : lr[{lr.min():.3f},{lr.max():.3f}] hr[{hr.min():.3f},{hr.max():.3f}]  [{'PASS' if range_ok else 'FAIL'}]")
    print(f"  paths[:3]: {[str(p) for p in paths[:3]]}")

    # Alignment: each LR matches its OWN HR much better than a shuffled HR.
    up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    self_err = (up - hr).abs().mean().item()
    shifted = torch.roll(hr, shifts=1, dims=0)
    cross_err = (up - shifted).abs().mean().item()
    align_ok = self_err < cross_err
    print(f"  align    : self-pair L1={self_err:.4f}  vs  shuffled-pair L1={cross_err:.4f}  "
          f"[{'PASS (LR matches its HR)' if align_ok else 'FAIL'}]")

    out_path = Path(args.out)
    save_comparison_grid(lr, up, hr, out_path, num_images=min(6, lr.shape[0]))
    print(f"  saved comparison grid -> {out_path}")
    print("=" * 70)
    ok = shape_ok and range_ok and align_ok
    print("[dataloader_check] PASS" if ok else "[dataloader_check] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
