"""Pre-build the decoded disk cache for fast training.

Training builds the cache lazily on first use, but doing it ahead of time (with
a visible progress bar) is convenient before a long run. The cache stores every
image decoded+resized to ``cache.size`` as a uint8 memmap, so subsequent epochs
never touch the JPEG decoder again.

    python tools/build_cache.py --config configs/config_8x.yaml
"""

from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401

from dataset import build_disk_cache, scan_images
from utils import load_config, validate_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-build the decoded image cache")
    p.add_argument("--config", type=str, default="configs/config_8x.yaml")
    p.add_argument("--max-images", type=int, default=None, help="cache only the first N images")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = validate_config(load_config(args.config))
    data = cfg["data"]
    paths = scan_images(data["dataset_root"])
    if args.max_images:
        paths = paths[: args.max_images]
    size = data["cache"]["size"]
    print(f"Building cache for {len(paths)} images @ {size}px under {data['cache']['dir']} ...")
    t0 = time.perf_counter()
    dat = build_disk_cache(data["dataset_root"], paths, data["cache"]["dir"], size, verbose=True)
    print(f"Finished in {time.perf_counter() - t0:.1f}s -> {dat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
