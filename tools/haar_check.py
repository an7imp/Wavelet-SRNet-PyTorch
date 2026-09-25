"""Haar reconstruction correctness check.

Proves the *logic* (not just the shapes): decomposing an image into Haar packet
coefficients and reconstructing it must return the original with ~0 error. This
is run for scale 4 / 8 / 16, with both the generated filters and (if present)
the original pickle, and exits non-zero if the generated filters ever fail.

    python tools/haar_check.py
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import torch

from models import build_haar_transforms, haar_reconstruction_error
from utils.config import scale_to_levels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Haar decompose->reconstruct identity check")
    p.add_argument("--scales", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--pickle", type=str, default="weights/wavelet_weights_c2.pkl")
    p.add_argument("--tol", type=float, default=1e-4)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 70)
    print("HAAR RECONSTRUCTION CHECK  (decompose -> reconstruct == identity)")
    print("=" * 70)
    ok = True
    for scale in args.scales:
        levels = scale_to_levels(scale)

        gen_err = haar_reconstruction_error(levels, params_path=None)
        gen_ok = gen_err <= args.tol
        ok = ok and gen_ok
        print(f"  scale={scale:2d} (levels={levels})  generated filters: max_err={gen_err:.3e}  "
              f"[{'PASS' if gen_ok else 'FAIL'}]")

        try:
            pkl_err = haar_reconstruction_error(levels, params_path=args.pickle)
            status = "PASS" if pkl_err <= args.tol else "broken -> auto-fallback to generated"
            print(f"  scale={scale:2d} (levels={levels})  pickle filters   : max_err={pkl_err:.3e}  [{status}]")
        except Exception as exc:
            print(f"  scale={scale:2d} (levels={levels})  pickle filters   : unavailable ({exc})")

        # Also verify the factory always yields a working pair (with fallback).
        dec, rec = build_haar_transforms(levels, params_path=args.pickle)
        x = torch.rand(2, 3, 2**levels * 8, 2**levels * 8)
        pair_err = (rec(dec(x)) - x).abs().max().item()
        print(f"  scale={scale:2d} (levels={levels})  build_haar_transforms pair: max_err={pair_err:.3e}  "
              f"[{'PASS' if pair_err <= args.tol else 'FAIL'}]")
        ok = ok and pair_err <= args.tol
        print("-" * 70)

    print("[haar_check] PASS" if ok else "[haar_check] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
