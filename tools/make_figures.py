"""Generate paper figures from training logs and the paper-eval JSON.

  python tools/make_figures.py --metrics results/8x/metrics.csv \
      --eval results/paper_eval/ours.json --out results/paper_eval/figures
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_metrics(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    def col(name):
        return [float(r[name]) if r[name] not in ("", None) else float("nan") for r in rows]
    return {
        "epoch": col("epoch"), "psnr": col("val_psnr"), "ssim": col("val_ssim"),
        "id": col("val_id_sim"), "loss": col("loss_total"), "lr": col("lr"),
        "bic_psnr": col("bicubic_psnr")[0], "bic_ssim": col("bicubic_ssim")[0],
        "is_best": [r["is_best"] == "True" for r in rows],
    }


def fig_training_curves(m, out: Path):
    ep = m["epoch"]
    best_ep = ep[[i for i, b in enumerate(m["is_best"]) if b][-1]] if any(m["is_best"]) else None
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("Wavelet-SRNet 8x  --  training curves (validation)", fontsize=13)

    ax[0, 0].plot(ep, m["psnr"], color="#1f77b4", lw=1.6, label="WaveletSR")
    ax[0, 0].axhline(m["bic_psnr"], color="gray", ls="--", lw=1, label="bicubic")
    ax[0, 0].set_title("PSNR (dB)"); ax[0, 0].set_xlabel("epoch"); ax[0, 0].legend(fontsize=8)

    ax[0, 1].plot(ep, m["ssim"], color="#2ca02c", lw=1.6, label="WaveletSR")
    ax[0, 1].axhline(m["bic_ssim"], color="gray", ls="--", lw=1, label="bicubic")
    ax[0, 1].set_title("SSIM"); ax[0, 1].set_xlabel("epoch"); ax[0, 1].legend(fontsize=8)

    ax[1, 0].plot(ep, m["id"], color="#d62728", lw=1.6)
    ax[1, 0].set_title("Identity similarity (ArcFace R100)"); ax[1, 0].set_xlabel("epoch")

    ax[1, 1].plot(ep, m["loss"], color="#9467bd", lw=1.6)
    ax[1, 1].set_title("Training loss (total)"); ax[1, 1].set_xlabel("epoch"); ax[1, 1].set_yscale("log")

    for a in ax.flat:
        if best_ep is not None:
            a.axvline(best_ep, color="orange", ls=":", lw=1, alpha=0.7)
        a.grid(alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = out / "training_curves.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_id_distribution(ev, out: Path):
    sr = ev["id_per_image"]["model"]
    bic = ev["id_per_image"]["bicubic"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(bic, bins=60, range=(-0.2, 0.9), alpha=0.55, color="gray", label=f"Bicubic (mean {sum(bic)/len(bic):.3f})")
    ax.hist(sr, bins=60, range=(-0.2, 0.9), alpha=0.6, color="#1f77b4", label=f"WaveletSR (mean {sum(sr)/len(sr):.3f})")
    ax.axvline(sum(bic)/len(bic), color="gray", ls="--", lw=1)
    ax.axvline(sum(sr)/len(sr), color="#1f77b4", ls="--", lw=1)
    ax.set_title("Identity similarity to HR  (ArcFace cosine, test set)")
    ax.set_xlabel("cosine similarity  (higher = identity better preserved)")
    ax.set_ylabel("# images"); ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout()
    p = out / "id_distribution.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/8x/metrics.csv")
    ap.add_argument("--eval", default="results/paper_eval/ours.json")
    ap.add_argument("--out", default="results/paper_eval/figures")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    m = read_metrics(Path(args.metrics))
    p1 = fig_training_curves(m, out)
    print(f"  saved {p1}")
    if Path(args.eval).exists():
        ev = json.loads(Path(args.eval).read_text(encoding="utf-8"))
        p2 = fig_id_distribution(ev, out)
        print(f"  saved {p2}")


if __name__ == "__main__":
    main()
