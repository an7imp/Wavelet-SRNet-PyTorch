"""Ablation figures: WaveletSRNet with vs without the ArcFace identity loss.

Reads results/paper_eval/{ours.json,noid.json} and produces:
  - id_distribution_ablation.png : 3 overlaid ArcFace-ID histograms (bicubic /
    no-ArcFace / ours), showing how much the identity loss shifts identity right.
  - metrics_bars.png : PSNR / SSIM / LPIPS / ID bar charts for the 3 methods.
"""
from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/paper_eval/figures")
OUT.mkdir(parents=True, exist_ok=True)
ours = json.loads(Path("results/paper_eval/ours.json").read_text(encoding="utf-8"))
noid = json.loads(Path("results/paper_eval/noid.json").read_text(encoding="utf-8"))


def mean(x):
    return sum(x) / len(x)


# --- 3-way ID distribution ---------------------------------------------------
bic = ours["id_per_image"]["bicubic"]
no = noid["id_per_image"]["model"]
ye = ours["id_per_image"]["model"]
fig, ax = plt.subplots(figsize=(7.5, 4.4))
for data, color, lab in [
    (bic, "gray", f"Bicubic (mean {mean(bic):.3f})"),
    (no, "#ff7f0e", f"WaveletSR, no ArcFace (mean {mean(no):.3f})"),
    (ye, "#1f77b4", f"WaveletSR + ArcFace, ours (mean {mean(ye):.3f})"),
]:
    ax.hist(data, bins=60, range=(-0.2, 0.9), alpha=0.55, color=color, label=lab)
    ax.axvline(mean(data), color=color, ls="--", lw=1)
ax.set_title("Identity similarity to HR  (ArcFace cosine, test set)")
ax.set_xlabel("cosine similarity  (higher = identity better preserved)")
ax.set_ylabel("# images"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "id_distribution_ablation.png", dpi=150); plt.close(fig)
print(f"  saved {OUT / 'id_distribution_ablation.png'}")

# --- metrics bar charts ------------------------------------------------------
methods = ["Bicubic", "no ArcFace", "+ArcFace (ours)"]
colors = ["gray", "#ff7f0e", "#1f77b4"]
A = ours["aggregates"]; B = noid["aggregates"]
vals = {
    "PSNR (dB) ^": [A["bicubic"]["psnr"]["mean"], B["model"]["psnr"]["mean"], A["model"]["psnr"]["mean"]],
    "SSIM ^": [A["bicubic"]["ssim"]["mean"], B["model"]["ssim"]["mean"], A["model"]["ssim"]["mean"]],
    "LPIPS v": [A["bicubic"]["lpips"]["mean"], B["model"]["lpips"]["mean"], A["model"]["lpips"]["mean"]],
    "ID ArcFace ^": [A["bicubic"]["id"]["mean"], B["model"]["id"]["mean"], A["model"]["id"]["mean"]],
}
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
for ax, (title, v) in zip(axes, vals.items()):
    ax.bar(methods, v, color=colors)
    ax.set_title(title); ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    for i, val in enumerate(v):
        ax.text(i, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.margins(y=0.18); ax.grid(axis="y", alpha=0.25)
fig.suptitle("Ablation: contribution of the ArcFace identity loss (test set)  [^ higher better, v lower better]", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "metrics_bars.png", dpi=150); plt.close(fig)
print(f"  saved {OUT / 'metrics_bars.png'}")
