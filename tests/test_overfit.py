"""The pipeline must be able to OVERFIT a handful of images.

This is the strongest end-to-end correctness signal: it exercises the full path
(LR -> predict wavelet coefficients -> inverse Haar -> SR image -> loss ->
backward). If the model cannot drive the reconstruction error down on a few
fixed images, something is fundamentally wrong (wrong targets, broken gradients,
detached reconstruction, ...).
"""

from __future__ import annotations

import torch
from torch.optim import Adam

from dataset import FaceSRDataset
from losses import WaveletSRLoss
from metrics import psnr
from models import WaveletSRNet, build_haar_transforms


def test_overfit_tiny_dataset(synthetic_dataset, tmp_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    ds = FaceSRDataset(
        root=synthetic_dataset, hr_size=128, scale=8, hflip=False,
        cache_mode="memory", cache_dir=tmp_path / "cache",
    )
    lr = torch.stack([ds[i]["lr"] for i in range(4)]).to(device)
    hr = torch.stack([ds[i]["hr"] for i in range(4)]).to(device)

    # Small model keeps the test fast while still using the real architecture.
    model = WaveletSRNet(levels=3, embedding_channels=(16, 32), num_embedding_blocks=1).to(device)
    dec, rec = build_haar_transforms(3, params_path=None)
    dec, rec = dec.to(device), rec.to(device)
    criterion = WaveletSRLoss().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    model.train()
    with torch.no_grad():
        start_psnr = psnr(rec(model(lr)).clamp(0, 1), hr, luminance=True)

    losses = []
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        target = dec(hr)
        pred = model(lr)
        sr = rec(pred)
        loss = criterion(pred, target, sr, hr).total
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        end_psnr = psnr(rec(model(lr)).clamp(0, 1), hr, luminance=True)

    # Loss must drop a lot and reconstruction quality must climb well past the
    # untrained starting point (and past a typical bicubic baseline).
    assert losses[-1] < 0.2 * losses[0], f"loss barely moved: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert end_psnr > start_psnr + 5.0, f"PSNR did not improve enough: {start_psnr:.2f} -> {end_psnr:.2f}"
    assert end_psnr > 25.0, f"final overfit PSNR too low: {end_psnr:.2f}"
