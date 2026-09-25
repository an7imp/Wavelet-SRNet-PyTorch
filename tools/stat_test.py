"""Paired statistical test: WaveletSRNet + identity loss (E2) vs baseline (E1).

The two ``results/paper_eval/{ours,noid}.json`` files store *per-image* PSNR and
ArcFace-ID arrays computed on the **same** test split (seed=42, ``shuffle=False``),
so the i-th entry is the same image in both files: the comparison is naturally
**paired**. Reporting "23.14 > 22.69" alone says nothing about significance; this
script turns it into a defensible result with a paired t-test, a non-parametric
Wilcoxon signed-rank test, an effect size and a confidence interval.

    python tools/stat_test.py
    python tools/stat_test.py --a results/paper_eval/ours.json --b results/paper_eval/noid.json

Note: only PSNR and ID are stored per image; SSIM/LPIPS are stored as aggregates
only, so they are not tested here (re-run eval with per-image SSIM/LPIPS dumps to
extend this).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def paired_report(name: str, a: np.ndarray, b: np.ndarray, *, higher_is_better: bool = True) -> dict:
    """Paired comparison of metric arrays a (E2) vs b (E1), aligned per image."""
    assert a.shape == b.shape, f"{name}: shape mismatch {a.shape} vs {b.shape}"
    d = a - b  # per-image improvement of E2 over E1
    n = d.size
    mean_d = float(d.mean())
    sd_d = float(d.std(ddof=1))
    se = sd_d / np.sqrt(n)
    # 95% CI on the mean difference (normal approx; n is huge)
    ci = (mean_d - 1.96 * se, mean_d + 1.96 * se)
    # paired t-test and non-parametric Wilcoxon signed-rank
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        _, w_p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except ValueError:  # all-zero differences
        w_p = 1.0
    cohen_dz = mean_d / sd_d if sd_d > 0 else float("nan")  # paired effect size
    wins = float((d > 0).mean()) if higher_is_better else float((d < 0).mean())
    return {
        "name": name, "n": n,
        "mean_E2": float(a.mean()), "mean_E1": float(b.mean()),
        "mean_diff": mean_d, "median_diff": float(np.median(d)),
        "ci95": ci, "cohen_dz": cohen_dz,
        "t_p": float(t_p), "wilcoxon_p": float(w_p),
        "pct_improved": 100.0 * wins,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired statistical test E2 vs E1")
    ap.add_argument("--a", default="results/paper_eval/ours.json", help="E2 (with identity loss)")
    ap.add_argument("--b", default="results/paper_eval/noid.json", help="E1 (baseline)")
    ap.add_argument("--output", default="results/paper_eval/stat_test.json")
    args = ap.parse_args()

    A, B = _load(args.a), _load(args.b)
    reports = [
        paired_report("PSNR (dB)",
                      np.asarray(A["psnr_per_image_model"], dtype=np.float64),
                      np.asarray(B["psnr_per_image_model"], dtype=np.float64)),
        paired_report("ID ArcFace",
                      np.asarray(A["id_per_image"]["model"], dtype=np.float64),
                      np.asarray(B["id_per_image"]["model"], dtype=np.float64)),
    ]

    n = reports[0]["n"]
    print("=" * 92)
    print(f"PAIRED TEST  E2 (+identity, '{A.get('label')}')  vs  E1 (baseline, '{B.get('label')}')   n={n} images")
    print("-" * 92)
    print(f"| {'Metric':<11} | {'E1':>7} | {'E2':>7} | {'d mean':>8} | {'95% CI':>20} | {'dz':>6} | {'%img+':>6} | {'p (t / Wilcoxon)':>22} |")
    print("|" + "-" * 13 + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 10 + "|" + "-" * 22 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 24 + "|")
    for r in reports:
        ci = f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
        pstr = f"{r['t_p']:.2e} / {r['wilcoxon_p']:.2e}"
        print(f"| {r['name']:<11} | {r['mean_E1']:>7.3f} | {r['mean_E2']:>7.3f} | {r['mean_diff']:>+8.4f} | {ci:>20} | {r['cohen_dz']:>6.3f} | {r['pct_improved']:>5.1f}% | {pstr:>22} |")
    print("=" * 92)
    print("d mean = mean per-image difference (E2 - E1);  dz = Cohen's d for paired samples;")
    print("%img+ = fraction of test images where E2 beats E1.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"saved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
