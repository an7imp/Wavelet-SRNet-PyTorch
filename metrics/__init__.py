"""Metriche di valutazione di Wavelet-SRNet — tutte in un unico modulo.

Le metriche misurano (NON addestrano) la qualità dell'immagine super-risolta SR
rispetto alla ground truth HR. Erano tre file separati di poche righe; qui sono
raccolte insieme. Vengono usate in validazione (train.py) e in test (test.py):

- PSNR / SSIM     : fedeltà numerica e similarità strutturale (per-immagine, poi media).
                    Si confrontano sempre col baseline bicubico, "la soglia da battere".
- LPIPS           : similarità percettiva (opzionale: richiede ``pip install lpips``).
- ArcFace ID      : similarità identitaria, coseno tra embedding facciali (R100 indipendente).

Nota di correttezza: PSNR e SSIM sono definiti *per immagine* e poi mediati sul
dataset. Calcolare un singolo PSNR dalla MSE media del batch (come faceva la vecchia
implementazione) è uno stimatore distorto — poche immagini facili possono dominare.

API pubblica (invariata): ``from metrics import psnr_per_image, ssim_per_image, LPIPSMetric, ArcFaceSimilarity, ...``
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ===========================================================================
# PSNR / SSIM  (per-immagine, poi mediati)
# ===========================================================================
def rgb_to_y(x: Tensor) -> Tensor:
    """Convert an RGB image tensor in [0, 1] to luminance Y (BT.601)."""
    if x.shape[1] != 3:
        return x
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


@torch.no_grad()
def psnr_per_image(pred: Tensor, target: Tensor, *, data_range: float = 1.0, luminance: bool = True) -> Tensor:
    """Per-image PSNR in dB. Returns a 1-D tensor of length ``batch``."""
    pred = pred.detach().clamp(0, data_range)
    target = target.detach().clamp(0, data_range)
    if luminance:
        pred = rgb_to_y(pred)
        target = rgb_to_y(target)
    mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1)
    mse = torch.clamp(mse, min=1e-12)  # avoid log(0); identical images -> ~120 dB
    return 10.0 * torch.log10((data_range**2) / mse)


@torch.no_grad()
def psnr(pred: Tensor, target: Tensor, *, data_range: float = 1.0, luminance: bool = True) -> float:
    """Mean per-image PSNR (dB)."""
    return psnr_per_image(pred, target, data_range=data_range, luminance=luminance).mean().item()


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(channels, 1, window_size, window_size).contiguous()


@torch.no_grad()
def ssim_per_image(
    pred: Tensor,
    target: Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    luminance: bool = False,
) -> Tensor:
    """Per-image SSIM. Returns a 1-D tensor of length ``batch``."""
    pred = pred.detach().clamp(0, data_range)
    target = target.detach().clamp(0, data_range)
    if luminance:
        pred = rgb_to_y(pred)
        target = rgb_to_y(target)

    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma, channels, pred.device, pred.dtype)
    padding = window_size // 2

    mu_x = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x.pow(2), mu_y.pow(2), mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
    return ssim_map.flatten(1).mean(dim=1)


@torch.no_grad()
def ssim(
    pred: Tensor,
    target: Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    luminance: bool = False,
) -> float:
    """Mean per-image SSIM."""
    return ssim_per_image(
        pred, target, data_range=data_range, window_size=window_size, sigma=sigma, luminance=luminance
    ).mean().item()


# ===========================================================================
# LPIPS  (percettiva, opzionale)
# ===========================================================================
class LPIPSMetric:
    """Optional LPIPS wrapper.

    Install ``lpips`` to enable it. Inputs are expected in [0, 1] and are
    internally mapped to [-1, 1].
    """

    def __init__(self, net: str = "alex", device: Optional[torch.device | str] = None) -> None:
        # Import pigro: ``import metrics`` non richiede il pacchetto lpips installato.
        try:
            import lpips  # type: ignore
        except ImportError as exc:
            raise ImportError("LPIPS is optional. Install it with: pip install lpips") from exc

        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, pred: Tensor, target: Tensor) -> float:
        pred = pred.to(self.device).clamp(0, 1) * 2 - 1
        target = target.to(self.device).clamp(0, 1) * 2 - 1
        return self.model(pred, target).mean().item()


# ===========================================================================
# ArcFace identity similarity  (coseno tra embedding facciali)
# ===========================================================================
class ArcFaceSimilarity:
    """Cosine similarity wrapper for an ArcFace-compatible backbone.

    The backbone must return one embedding per image. This class deliberately
    does not download weights; pass your own verified ArcFace model.
    """

    def __init__(self, backbone: nn.Module, *, device: Optional[torch.device | str] = None) -> None:
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = backbone.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, pred: Tensor, target: Tensor) -> float:
        pred_features = F.normalize(self.backbone(pred.to(self.device)), dim=1)
        target_features = F.normalize(self.backbone(target.to(self.device)), dim=1)
        return torch.sum(pred_features * target_features, dim=1).mean().item()


__all__ = [
    "psnr",
    "ssim",
    "psnr_per_image",
    "ssim_per_image",
    "rgb_to_y",
    "LPIPSMetric",
    "ArcFaceSimilarity",
]
