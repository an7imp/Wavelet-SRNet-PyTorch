"""Prepara i volti di test HELEN* per la valutazione cross-dataset.

Le immagini HELEN* (``helenstar_release/test``) sono foto intere con il volto non
allineato come in CelebA. Usiamo le maschere di parsing fornite (``*_label.png``,
valori 1..9 = pelle/sopracciglia/occhi/naso/labbra) per ritagliare un quadrato
centrato sul volto e con framing simile a CelebA (fronte+mento+un po' di capelli),
quindi ridimensioniamo a 128x128. Nessuna dipendenza esterna (solo PIL/numpy):
non e' un allineamento per landmark, ma per il confronto SR-vs-HR e' sufficiente
perche' SR e HR condividono esattamente lo stesso ritaglio.

    python tools/prep_helen.py
    python tools/prep_helen.py --src helenstar_release/test --out results/helen/hr --size 128
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image

FACE_LABELS = range(1, 10)  # skin, brows, eyes, nose, lips (NON i capelli=10)


def face_crop(img: Image.Image, mask: np.ndarray, *, size: int, expand: float) -> Image.Image:
    """Ritaglia un quadrato centrato sul volto (bbox delle feature, espanso) -> size x size."""
    ys, xs = np.where(np.isin(mask, list(FACE_LABELS)))
    if len(xs) < 50:  # fallback: usa tutto il foreground (incl. capelli) se le feature mancano
        ys, xs = np.where(mask > 0)
    if len(xs) < 50:  # nessuna maschera utile: center-crop quadrato
        w, h = img.size
        s = min(w, h)
        box = ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
        return img.crop(box).resize((size, size), Image.BICUBIC)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * expand
    half = side / 2.0
    left, top, right, bot = cx - half, cy - half, cx + half, cy + half

    # Pad in riflessione se il quadrato esce dai bordi, cosi' il volto resta centrato.
    arr = np.asarray(img)
    pad_l = int(max(0, -left)); pad_t = int(max(0, -top))
    pad_r = int(max(0, right - arr.shape[1])); pad_b = int(max(0, bot - arr.shape[0]))
    if pad_l or pad_t or pad_r or pad_b:
        arr = np.pad(arr, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="reflect")
        left += pad_l; right += pad_l; top += pad_t; bot += pad_t
    crop = Image.fromarray(arr).crop((int(round(left)), int(round(top)), int(round(right)), int(round(bot))))
    return crop.resize((size, size), Image.BICUBIC)


def main() -> int:
    ap = argparse.ArgumentParser(description="HELEN* face crops for cross-dataset SR eval")
    ap.add_argument("--src", default="helenstar_release/test")
    ap.add_argument("--out", default="results/helen/hr")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--expand", type=float, default=1.3)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = sorted(glob.glob(os.path.join(args.src, "*_image.jpg")))
    n = 0
    for img_path in images:
        label_path = img_path.replace("_image.jpg", "_label.png")
        if not os.path.exists(label_path):
            continue
        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(label_path).convert("L"))
        crop = face_crop(img, mask, size=args.size, expand=args.expand)
        name = os.path.basename(img_path).replace("_image.jpg", ".png")
        crop.save(out / name)
        n += 1
    print(f"[prep_helen] salvati {n} volti {args.size}x{args.size} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
