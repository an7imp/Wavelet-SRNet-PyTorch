"""Shape contract check for scale 4 / 8 / 16.

Verifies that for an LR input the network emits exactly ``3 * 4**levels``
coefficient channels at the LR spatial size, and that the inverse Haar transform
reconstructs a ``B x 3 x HR x HR`` image. This is the structural counterpart to
``haar_check`` (which proves numerical invertibility).

    python tools/shapes_check.py
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import torch

from models import WaveletSRNet, build_haar_transforms
from utils.config import scale_to_levels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Network/transform shape check")
    p.add_argument("--scales", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--hr-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 70)
    print("SHAPE CHECK")
    print("=" * 70)
    print(f"{'scale':>5} {'levels':>6} {'lr':>9} {'model_out':>18} {'expect_ch':>10} {'recon':>16} {'status':>7}")
    ok = True
    for scale in args.scales:
        levels = scale_to_levels(scale)
        lr_size = args.hr_size // scale
        model = WaveletSRNet(levels=levels).eval()
        _, rec = build_haar_transforms(levels, params_path=None)
        lr = torch.rand(args.batch_size, 3, lr_size, lr_size)
        with torch.no_grad():
            out = model(lr)
            sr = rec(out)
        expect_ch = 3 * 4**levels
        out_ok = tuple(out.shape) == (args.batch_size, expect_ch, lr_size, lr_size)
        sr_ok = tuple(sr.shape) == (args.batch_size, 3, args.hr_size, args.hr_size)
        passed = out_ok and sr_ok
        ok = ok and passed
        print(f"{scale:>5} {levels:>6} {f'{lr_size}x{lr_size}':>9} {str(tuple(out.shape)):>18} "
              f"{expect_ch:>10} {str(tuple(sr.shape)):>16} {'PASS' if passed else 'FAIL':>7}")
    print("=" * 70)
    print("[shapes_check] PASS" if ok else "[shapes_check] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
