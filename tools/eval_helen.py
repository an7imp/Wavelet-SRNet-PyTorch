"""Valutazione cross-dataset su HELEN* (100 volti di test).

Carica i crop 128x128 prodotti da ``tools/prep_helen.py``, genera l'LR 16x16 con
lo stesso downscale bicubico antialiasato della pipeline CelebA, applica il
modello (checkpoint ``best.pth``) e calcola PSNR / SSIM / LPIPS / ID-ArcFace del
modello e del baseline bicubico, con lo stesso protocollo del Tab. CelebA. Salva
gli aggregati in JSON e alcune strisce qualitative ``LR-bicubic | SR | HR | err``.

    python tools/eval_helen.py --checkpoint results/8x/checkpoints/best.pth
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
import torch.nn.functional as F

from dataset import create_dataloader
from metrics import psnr_per_image, ssim_per_image
from models import build_arcface, build_haar_transforms, build_model_from_config
from utils import load_config, validate_config
from utils.env import resolve_device
from utils.checkpoint import load_checkpoint
from utils.viz import save_triptych_per_image


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-dataset evaluation on HELEN*")
    ap.add_argument("--config", default="configs/config_8x.yaml")
    ap.add_argument("--checkpoint", default="results/8x/checkpoints/best.pth")
    ap.add_argument("--root", default="results/helen/hr")
    ap.add_argument("--label", default="WaveletSR + Id. (Helen)")
    ap.add_argument("--output", default="results/paper_eval/helen.json")
    ap.add_argument("--strips-out", default="results/helen/comparisons")
    ap.add_argument("--num-strips", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = validate_config(load_config(args.config))
    device = resolve_device(args.device)
    levels = cfg["levels"]

    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.root, "*.png")))
    if not names:
        raise SystemExit(f"Nessun crop trovato in {args.root} (esegui prima tools/prep_helen.py)")
    print(f"[helen] {len(names)} volti da {args.root}")

    loader = create_dataloader(
        root=args.root, image_list=names, hr_size=cfg["data"]["hr_size"], scale=cfg["data"]["scale"],
        random_crop=False, hflip=False, cache_mode="none", batch_size=16, shuffle=False, num_workers=0,
    )

    model = build_model_from_config(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, model=model, device=device, strict=False, restore_rng=False)
    _, wavelet_rec = build_haar_transforms(levels, params_path=cfg.get("wavelet_weights_path"))
    wavelet_rec = wavelet_rec.to(device)

    embedder = build_arcface(
        weights_path=cfg["metrics"]["arcface"].get("weights"),
        arch=cfg["metrics"]["arcface"].get("arch", "r100"),
        device=device, download_dir=cfg["metrics"]["arcface"].get("download_dir", "weights"),
    )
    lpips_model = None
    try:
        import lpips  # type: ignore
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as exc:
        print(f"[helen] LPIPS non disponibile ({exc}); salto LPIPS.")

    def idsim(pred, hr_feat):
        return (F.normalize(embedder(pred), dim=1) * hr_feat).sum(dim=1).cpu()

    M = {k: {m: [] for m in ("psnr", "ssim", "lpips", "id")} for k in ("model", "bicubic")}
    saved = 0
    for batch in loader:
        lr = batch["lr"].to(device); hr = batch["hr"].to(device)
        sr = wavelet_rec(model(lr)).float().clamp(0, 1)
        up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        hr_feat = F.normalize(embedder(hr), dim=1)
        for key, pred in (("model", sr), ("bicubic", up)):
            M[key]["psnr"].append(psnr_per_image(pred, hr, luminance=True).cpu())
            M[key]["ssim"].append(ssim_per_image(pred, hr, luminance=False).cpu())
            M[key]["id"].append(idsim(pred, hr_feat))
            if lpips_model is not None:
                M[key]["lpips"].append(lpips_model(pred * 2 - 1, hr * 2 - 1).flatten().cpu())
        if saved < args.num_strips:
            n = min(lr.shape[0], args.num_strips - saved)
            saved += save_triptych_per_image(lr[:n], sr[:n], hr[:n], Path(args.strips_out), start_index=saved)

    agg = {}
    for key in ("model", "bicubic"):
        agg[key] = {}
        for m in ("psnr", "ssim", "lpips", "id"):
            if M[key][m]:
                v = torch.cat(M[key][m]).numpy()
                agg[key][m] = {"mean": float(v.mean()), "std": float(v.std())}

    print("=" * 72)
    print(f"HELEN* cross-dataset  ({len(names)} volti, 8x)   --  {args.label}")
    print("-" * 72)
    print(f"  {'Metodo':10s}{'PSNR':>10s}{'SSIM':>10s}{'LPIPS':>10s}{'ID':>10s}")
    for key in ("bicubic", "model"):
        a = agg[key]
        lp = f"{a['lpips']['mean']:.4f}" if 'lpips' in a else "  n/a "
        print(f"  {key:10s}{a['psnr']['mean']:>10.3f}{a['ssim']['mean']:>10.4f}{lp:>10s}{a['id']['mean']:>10.4f}")
    print("=" * 72)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"label": args.label, "num_images": len(names),
                               "checkpoint": args.checkpoint, "aggregates": agg}, indent=2), encoding="utf-8")
    print(f"  salvato -> {out}  |  strisce -> {args.strips_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
