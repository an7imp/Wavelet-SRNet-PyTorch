"""Robustness evaluation for Wavelet-SRNet.

Evaluates the trained model and the bicubic baseline under controlled LR
degradations: unknown Gaussian blur, additive Gaussian noise, and LR
misalignment. The blur protocol mirrors the ICCV 2017 Wavelet-SRNet discussion:
LR faces are generated from HR faces with a Gaussian blur kernel before 8x
downsampling, with sigma=0 corresponding to nearest-neighbor downsampling.

Examples:
    # CelebA test split (uses data.dataset_root from the config)
    python tools/eval_robustness.py --checkpoint results/8x/checkpoints/best.pth

    # HELEN* crops produced by tools/prep_helen.py
    python tools/eval_robustness.py --root results/helen/hr \
        --checkpoint results/8x/checkpoints/best.pth --device cuda
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Iterable

import _bootstrap  # noqa: F401
import torch
import torch.nn.functional as F

from dataset import create_dataloader, split_image_lists
from metrics import psnr_per_image, ssim_per_image
from models import build_arcface, build_haar_transforms, build_model_from_config
from utils import load_config, validate_config
from utils.checkpoint import load_checkpoint
from utils.env import autocast, resolve_device, setup_backends


MetricStore = dict[str, list[torch.Tensor]]


def _csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robustness evaluation for Wavelet-SRNet")
    p.add_argument("--config", default="configs/config_8x.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root", default=None, help="image root; default uses data.dataset_root from config")
    p.add_argument("--split", default="test", choices=["val", "test"],
                   help="used only when --root is not supplied")
    p.add_argument("--output", default="results/paper_eval/robustness.json")
    p.add_argument("--csv-output", default=None)
    p.add_argument("--label", default="WaveletSR + Id.")
    p.add_argument("--device", default=None)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--suites", default="blur,noise,shift",
                   help="comma-separated subset of: blur,noise,shift")
    p.add_argument("--blur-sigmas", default="0,0.5,1,2,3,4,5,6",
                   help="Gaussian blur sigmas in HR pixels; 0 means nearest downsampling")
    p.add_argument("--noise-stds", default="0,0.01,0.03,0.05,0.10",
                   help="additive LR Gaussian noise stds in [0,1]")
    p.add_argument("--shift-pixels", default="0,1,2,3",
                   help="LR-pixel shift magnitudes; non-zero magnitudes are averaged over four directions")
    p.add_argument("--id-metric", default="auto", choices=["auto", "on", "off"],
                   help="ArcFace identity metric. auto disables it if weights are unavailable.")
    p.add_argument("--arcface-weights", default=None)
    p.add_argument("--arcface-arch", default=None)
    p.add_argument("--lpips", action="store_true", help="also compute LPIPS if the package is installed")
    return p.parse_args()


def image_list_from_root(root: str | Path) -> list[str]:
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(str(root), "*")))
    return [name for name in names if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}]


def upscale(lr: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(lr, size=size, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)


def bicubic_downsample(hr: torch.Tensor, lr_size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(hr, size=lr_size, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)


def _gaussian_kernel1d(sigma: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def gaussian_blur(hr: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return hr
    kernel = _gaussian_kernel1d(sigma, device=hr.device, dtype=hr.dtype)
    radius = kernel.numel() // 2
    channels = hr.shape[1]
    kx = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    out = F.pad(hr, (radius, radius, 0, 0), mode="reflect")
    out = F.conv2d(out, kx, groups=channels)
    out = F.pad(out, (0, 0, radius, radius), mode="reflect")
    out = F.conv2d(out, ky, groups=channels)
    return out.clamp(0, 1)


def blur_downsample(hr: torch.Tensor, lr_size: tuple[int, int], sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return F.interpolate(hr, size=lr_size, mode="nearest").clamp(0, 1)
    return F.interpolate(gaussian_blur(hr, sigma), size=lr_size, mode="nearest").clamp(0, 1)


def add_lr_noise(lr: torch.Tensor, std: float, generator: torch.Generator) -> torch.Tensor:
    if std <= 0:
        return lr
    noise = torch.randn(lr.shape, generator=generator, device=lr.device, dtype=lr.dtype)
    return (lr + noise * std).clamp(0, 1)


def shift_lr(lr: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    if dx == 0 and dy == 0:
        return lr
    _, _, h, w = lr.shape
    pad_x, pad_y = abs(dx), abs(dy)
    padded = F.pad(lr, (pad_x, pad_x, pad_y, pad_y), mode="reflect")
    x0 = pad_x - dx
    y0 = pad_y - dy
    return padded[:, :, y0:y0 + h, x0:x0 + w].contiguous()


def init_store(include_lpips: bool, include_id: bool) -> MetricStore:
    keys = ["psnr", "ssim"]
    if include_lpips:
        keys.append("lpips")
    if include_id:
        keys.append("id")
    return {key: [] for key in keys}


@torch.no_grad()
def append_metrics(
    store: MetricStore,
    pred: torch.Tensor,
    hr: torch.Tensor,
    *,
    lpips_model,
    embedder,
    hr_feat: torch.Tensor | None,
) -> None:
    store["psnr"].append(psnr_per_image(pred, hr, luminance=True).cpu())
    store["ssim"].append(ssim_per_image(pred, hr, luminance=False).cpu())
    if lpips_model is not None:
        store["lpips"].append(lpips_model(pred * 2 - 1, hr * 2 - 1).flatten().cpu())
    if embedder is not None and hr_feat is not None:
        pred_feat = F.normalize(embedder(pred), dim=1)
        store["id"].append((pred_feat * hr_feat).sum(dim=1).cpu())


def aggregate(store: MetricStore) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, chunks in store.items():
        if not chunks:
            continue
        values = torch.cat(chunks).float().numpy()
        out[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "n": int(values.shape[0]),
        }
    return out


def make_condition_id(suite: str, level: float | int) -> str:
    if suite == "blur":
        return f"blur_sigma_{level:g}"
    if suite == "noise":
        return f"noise_std_{level:g}"
    return f"shift_px_{int(level)}"


def shift_dirs(magnitude: int) -> list[tuple[int, int]]:
    if magnitude == 0:
        return [(0, 0)]
    return [(magnitude, 0), (-magnitude, 0), (0, magnitude), (0, -magnitude)]


def load_optional_arcface(args: argparse.Namespace, cfg, device: torch.device):
    if args.id_metric == "off":
        return None, None
    arch = args.arcface_arch or cfg["metrics"]["arcface"].get("arch", "r100")
    weights = args.arcface_weights or cfg["metrics"]["arcface"].get("weights")
    download_dir = cfg["metrics"]["arcface"].get("download_dir", "weights")
    try:
        embedder = build_arcface(weights_path=weights, arch=arch, device=device, download_dir=download_dir)
        return embedder.eval(), {"arch": arch, "weights": weights, "download_dir": download_dir}
    except Exception as exc:
        if args.id_metric == "on":
            raise
        print(f"[robustness] ArcFace non disponibile ({exc}); salto ID.")
        return None, None


def load_optional_lpips(enabled: bool, device: torch.device):
    if not enabled:
        return None
    try:
        import lpips  # type: ignore
        return lpips.LPIPS(net="alex").to(device).eval()
    except Exception as exc:
        print(f"[robustness] LPIPS non disponibile ({exc}); salto LPIPS.")
        return None


def row_for_csv(condition: dict) -> dict[str, str | int | float]:
    row: dict[str, str | int | float] = {
        "suite": condition["suite"],
        "level": condition["level"],
        "condition": condition["condition"],
        "directions": condition["directions"],
    }
    for method in ("bicubic", "model"):
        for metric, values in condition["aggregates"][method].items():
            row[f"{method}_{metric}_mean"] = values["mean"]
            row[f"{method}_{metric}_std"] = values["std"]
            row[f"{method}_{metric}_n"] = values["n"]
    return row


def write_csv(path: Path, rows: Iterable[dict[str, str | int | float]]) -> None:
    rows = list(rows)
    if not rows:
        return
    import csv

    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def print_markdown(conditions: list[dict], label: str) -> None:
    metrics = ["psnr", "ssim"]
    if any("lpips" in c["aggregates"]["model"] for c in conditions):
        metrics.append("lpips")
    if any("id" in c["aggregates"]["model"] for c in conditions):
        metrics.append("id")

    print("=" * 86)
    print(f"ROBUSTNESS EVAL -- {label}")
    for suite in ("blur", "noise", "shift"):
        suite_rows = [c for c in conditions if c["suite"] == suite]
        if not suite_rows:
            continue
        print("-" * 86)
        print(f"{suite.upper()}")
        header = "| Condition | Method | " + " | ".join(m.upper() for m in metrics) + " |"
        print(header)
        print("|" + "---|" * (2 + len(metrics)))
        for condition in suite_rows:
            for method in ("bicubic", "model"):
                agg = condition["aggregates"][method]
                cells = []
                for metric in metrics:
                    if metric not in agg:
                        cells.append("n/a")
                    else:
                        cells.append(f"{agg[metric]['mean']:.4f}")
                name = label if method == "model" else "Bicubic"
                print(f"| {condition['condition']} | {name} | " + " | ".join(cells) + " |")
    print("=" * 86)


@torch.no_grad()
def main() -> int:
    args = parse_args()
    cfg = validate_config(load_config(args.config))
    setup_backends(cfg)
    device = resolve_device(args.device)
    data = cfg["data"]
    levels = cfg["levels"]
    amp = bool(cfg["train"]["amp"])
    suites = {s.strip().lower() for s in args.suites.split(",") if s.strip()}
    unknown = suites - {"blur", "noise", "shift"}
    if unknown:
        raise SystemExit(f"Unknown suites: {sorted(unknown)}")

    root = args.root or data["dataset_root"]
    if args.root:
        image_list = image_list_from_root(root)
    else:
        image_list = split_image_lists(
            data["dataset_root"], seed=int(cfg["seed"]),
            val_split=float(data["val_split"]), test_split=float(data["test_split"]),
        )[args.split]
    if args.max_images is not None:
        image_list = image_list[: args.max_images]
    if not image_list:
        raise SystemExit(f"No images found in {root}")

    batch_size = args.batch_size or cfg["test"]["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else cfg["test"]["num_workers"]
    loader = create_dataloader(
        root=root,
        image_list=image_list,
        hr_size=data["hr_size"],
        scale=data["scale"],
        random_crop=False,
        hflip=False,
        cache_mode="none" if args.root else data["cache"]["mode"],
        cache_dir=data["cache"]["dir"],
        cache_size=data["hr_size"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_model_from_config(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, model=model, device=device, strict=False, restore_rng=False)
    _, wavelet_rec = build_haar_transforms(levels, params_path=cfg.get("wavelet_weights_path"))
    wavelet_rec = wavelet_rec.to(device).eval()
    lpips_model = load_optional_lpips(args.lpips, device)
    embedder, arcface_info = load_optional_arcface(args, cfg, device)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    blur_sigmas = _csv_floats(args.blur_sigmas)
    noise_stds = _csv_floats(args.noise_stds)
    shift_pixels = _csv_ints(args.shift_pixels)

    conditions: list[dict] = []
    for suite in ("blur", "noise", "shift"):
        if suite not in suites:
            continue
        levels_to_eval: list[float | int]
        if suite == "blur":
            levels_to_eval = blur_sigmas
        elif suite == "noise":
            levels_to_eval = noise_stds
        else:
            levels_to_eval = shift_pixels

        for level in levels_to_eval:
            stores = {
                "model": init_store(lpips_model is not None, embedder is not None),
                "bicubic": init_store(lpips_model is not None, embedder is not None),
            }
            direction_count = 1
            n_images = 0
            for batch in loader:
                hr = batch["hr"].to(device, non_blocking=True)
                lr_size = (hr.shape[-2] // data["scale"], hr.shape[-1] // data["scale"])
                if suite == "blur":
                    lr_variants = [blur_downsample(hr, lr_size, float(level))]
                elif suite == "noise":
                    clean_lr = bicubic_downsample(hr, lr_size)
                    lr_variants = [add_lr_noise(clean_lr, float(level), generator)]
                else:
                    clean_lr = bicubic_downsample(hr, lr_size)
                    dirs = shift_dirs(int(level))
                    direction_count = len(dirs)
                    lr_variants = [shift_lr(clean_lr, dx, dy) for dx, dy in dirs]

                hr_feat = F.normalize(embedder(hr), dim=1) if embedder is not None else None
                for lr in lr_variants:
                    with autocast(device, amp):
                        sr = wavelet_rec(model(lr)).float().clamp(0, 1)
                    bic = upscale(lr.float(), hr.shape[-2:]).clamp(0, 1)
                    append_metrics(stores["model"], sr, hr, lpips_model=lpips_model, embedder=embedder, hr_feat=hr_feat)
                    append_metrics(stores["bicubic"], bic, hr, lpips_model=lpips_model, embedder=embedder, hr_feat=hr_feat)
                n_images += hr.shape[0]

            condition = {
                "suite": suite,
                "level": level,
                "condition": make_condition_id(suite, level),
                "directions": direction_count,
                "num_images": n_images,
                "aggregates": {
                    "model": aggregate(stores["model"]),
                    "bicubic": aggregate(stores["bicubic"]),
                },
            }
            conditions.append(condition)
            mp = condition["aggregates"]["model"]["psnr"]["mean"]
            bp = condition["aggregates"]["bicubic"]["psnr"]["mean"]
            print(f"[robustness] {condition['condition']}: model PSNR {mp:.3f}, bicubic {bp:.3f}")

    payload = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "root": str(root),
        "num_images": len(image_list),
        "scale": data["scale"],
        "hr_size": data["hr_size"],
        "seed": args.seed,
        "suites": sorted(suites),
        "arcface": arcface_info,
        "lpips": lpips_model is not None,
        "conditions": conditions,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_out = Path(args.csv_output) if args.csv_output else out.with_suffix(".csv")
    write_csv(csv_out, [row_for_csv(condition) for condition in conditions])
    print_markdown(conditions, args.label)
    print(f"  saved -> {out}")
    print(f"  saved -> {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
