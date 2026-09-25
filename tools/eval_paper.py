"""Paper-ready evaluation on the test split.

Computes per-image PSNR / SSIM / LPIPS / ArcFace identity-similarity for the
trained model (SR) and for the naive upscaling baselines (bicubic / bilinear /
nearest), reports mean +/- std, prints a markdown table and saves a JSON with
the aggregates plus the per-image ID arrays (for the identity-distribution
figure). Reusable across checkpoints (model vs ablation) via --label/--output.

    python tools/eval_paper.py --config configs/config_8x.yaml \
        --checkpoint results/8x/checkpoints/best.pth --label "WaveletSR (ours)" \
        --output results/paper_eval/ours.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  # adds repo root to sys.path
import torch
import torch.nn.functional as F

from dataset import create_dataloader, split_image_lists
from metrics import psnr_per_image, ssim_per_image
from models import build_arcface, build_haar_transforms, build_model_from_config
from utils import load_config, validate_config
from utils.env import autocast, print_environment, resolve_device, setup_backends
from utils.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper-ready evaluation of Wavelet-SRNet")
    p.add_argument("--config", type=str, default="configs/config_8x.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--label", type=str, default="WaveletSR")
    p.add_argument("--output", type=str, default="results/paper_eval/eval.json")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--lpips-net", type=str, default="alex", choices=["alex", "vgg"])
    return p.parse_args()


def upscale(lr: torch.Tensor, size, mode: str) -> torch.Tensor:
    if mode == "nearest":
        return F.interpolate(lr, size=size, mode="nearest").clamp(0, 1)
    return F.interpolate(lr, size=size, mode=mode, align_corners=False, antialias=True).clamp(0, 1)


@torch.no_grad()
def id_cossim(embedder, pred: torch.Tensor, hr_feat: torch.Tensor) -> torch.Tensor:
    """Per-image cosine similarity between embed(pred) and a precomputed HR embedding."""
    fp = F.normalize(embedder(pred), dim=1)
    return (fp * hr_feat).sum(dim=1).cpu()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    cfg = validate_config(load_config(args.config))
    setup_backends(cfg)
    device = resolve_device(args.device)
    data = cfg["data"]
    levels = cfg["levels"]
    amp = bool(cfg["train"]["amp"])
    print_environment(device, amp=amp, channels_last=False)

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
    load_checkpoint(args.checkpoint, model=model, device=device, strict=False, restore_rng=False)
    _, wavelet_rec = build_haar_transforms(levels, params_path=cfg.get("wavelet_weights_path"))
    wavelet_rec = wavelet_rec.to(device)

    # ArcFace (R100) identity evaluator -- independent of any loss used in training.
    embedder = build_arcface(
        weights_path=cfg["metrics"]["arcface"].get("weights"),
        arch=cfg["metrics"]["arcface"].get("arch", "r100"),
        device=device,
        download_dir=cfg["metrics"]["arcface"].get("download_dir", "weights"),
    )
    import lpips  # type: ignore
    lpips_model = lpips.LPIPS(net=args.lpips_net).to(device).eval()

    methods = ["model", "bicubic", "bilinear", "nearest"]
    metrics = {m: {k: [] for k in ("psnr", "ssim", "lpips", "id")} for m in methods}

    n_done = 0
    for batch in loader:
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        with autocast(device, amp):
            sr = wavelet_rec(model(lr)).float().clamp(0, 1)
        preds = {
            "model": sr,
            "bicubic": upscale(lr.float(), hr.shape[-2:], "bicubic"),
            "bilinear": upscale(lr.float(), hr.shape[-2:], "bilinear"),
            "nearest": upscale(lr.float(), hr.shape[-2:], "nearest"),
        }
        hr_feat = F.normalize(embedder(hr), dim=1)  # reused across methods
        for m, pred in preds.items():
            metrics[m]["psnr"].append(psnr_per_image(pred, hr, luminance=True).cpu())
            metrics[m]["ssim"].append(ssim_per_image(pred, hr, luminance=False).cpu())
            lp = lpips_model(pred * 2 - 1, hr * 2 - 1).flatten().cpu()
            metrics[m]["lpips"].append(lp)
            metrics[m]["id"].append(id_cossim(embedder, pred, hr_feat))
        n_done += lr.shape[0]
        if n_done % (cfg["test"]["batch_size"] * 10) == 0:
            print(f"  evaluated {n_done}/{len(image_list)} ...")

    # aggregate
    agg = {}
    for m in methods:
        agg[m] = {}
        for k in ("psnr", "ssim", "lpips", "id"):
            v = torch.cat(metrics[m][k]).numpy()
            agg[m][k] = {"mean": float(v.mean()), "std": float(v.std())}

    # markdown table
    print("=" * 78)
    print(f"PAPER EVAL  --  {args.label}  ({args.split} split, {n_done} images, {data['scale']}x SR)")
    print("-" * 78)
    hdr = f"| {'Method':<12} | {'PSNR (dB)':>14} | {'SSIM':>14} | {'LPIPS':>14} | {'ID (ArcFace)':>14} |"
    print(hdr)
    print("|" + "-" * 14 + "|" + ("-" * 16 + "|") * 4)
    names = {"model": args.label, "bicubic": "Bicubic", "bilinear": "Bilinear", "nearest": "Nearest"}
    for m in ("nearest", "bilinear", "bicubic", "model"):
        a = agg[m]
        row = (f"| {names[m]:<12} | {a['psnr']['mean']:6.3f}+/-{a['psnr']['std']:<4.2f} | "
               f"{a['ssim']['mean']:.4f}+/-{a['ssim']['std']:<.3f} | "
               f"{a['lpips']['mean']:.4f}+/-{a['lpips']['std']:<.3f} | "
               f"{a['id']['mean']:.4f}+/-{a['id']['std']:<.3f} |")
        print(row)
    print("=" * 78)
    gp = agg["model"]["psnr"]["mean"] - agg["bicubic"]["psnr"]["mean"]
    gi = agg["model"]["id"]["mean"] - agg["bicubic"]["id"]["mean"]
    print(f"  vs bicubic:  PSNR {gp:+.3f} dB  |  ID {gi:+.4f}  |  LPIPS {agg['model']['lpips']['mean'] - agg['bicubic']['lpips']['mean']:+.4f} (lower better)")

    # save JSON (aggregates + per-image id arrays of model & bicubic for the figure)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": args.label, "split": args.split, "num_images": n_done,
        "checkpoint": args.checkpoint, "scale": data["scale"], "lpips_net": args.lpips_net,
        "aggregates": agg,
        "id_per_image": {
            "model": torch.cat(metrics["model"]["id"]).numpy().tolist(),
            "bicubic": torch.cat(metrics["bicubic"]["id"]).numpy().tolist(),
        },
        "psnr_per_image_model": torch.cat(metrics["model"]["psnr"]).numpy().tolist(),
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"  saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
