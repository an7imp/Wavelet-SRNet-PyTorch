"""Funzioni di costo (loss) di Wavelet-SRNet.

Durante il training la rete NON confronta direttamente i pixel: predice i
coefficienti wavelet di Haar e questi vengono valutati da più termini di loss che,
sommati, formano la loss totale. Questo file raccoglie tutti questi termini (erano
file separati di poche righe: ora sono qui, in ordine di dipendenza).

Composizione della loss totale (vedi :class:`WaveletSRLoss`):

    L_tot = lambda_low * MSE(coeff. bassa freq.)      # struttura globale del volto
          + lambda_high * MSE(coeff. alta freq.)      # bordi / dettagli
          + texture_weight * L_texture                # impedisce dettagli "spenti"
          + image_weight * MSE(immagine ricostruita)  # coerenza pixel finale

A questa, in train.py, si somma a parte il termine identitario:

          + identity_weight * L_identity(ArcFace)     # preserva l'identità (estensione principale)

Posizione nella pipeline:
    dataset (LR/HR) -> WaveletSRNet (coeff.) -> Haar inversa (SR) -> LOSS -> backward

API pubblica: ``from losses import WaveletSRLoss, IdentityLoss, ...``
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ===========================================================================
# Texture loss  (sui coefficienti wavelet ad alta frequenza)
# ===========================================================================
class TextureLoss(nn.Module):
    """Texture loss from Wavelet-SRNet.

    Penalizes high-frequency predicted wavelet coefficients whose energy is too
    small compared with the target coefficients.
    """

    def __init__(self, *, image_channels: int = 3, alpha: float = 1.2, margin: float = 0.0) -> None:
        super().__init__()
        self.image_channels = int(image_channels)
        self.alpha = float(alpha)
        self.margin = float(margin)

    def forward(self, pred_high: Tensor, target_high: Tensor) -> Tensor:
        if pred_high.shape != target_high.shape:
            raise ValueError(f"Shape mismatch: {pred_high.shape} vs {target_high.shape}")
        if pred_high.shape[1] % self.image_channels != 0:
            raise ValueError("Channel dimension must be divisible by image_channels")

        pred = pred_high.contiguous().view(
            pred_high.shape[0], -1, self.image_channels, pred_high.shape[-2], pred_high.shape[-1]
        )
        target = target_high.contiguous().view_as(pred)
        pred_energy = torch.sum(pred * pred, dim=2)
        target_energy = torch.sum(target * target, dim=2)
        return F.relu(target_energy.mul(self.alpha) - pred_energy + self.margin).mean() #texture_loss=mean(max(0, alpha * target_energy - pred_energy + margin)) penalizza i coefficienti wavelet ad alta frequenza previsti (pred) che hanno un'energia (somma dei quadrati) troppo piccola rispetto all'energia dei coefficienti target (target). La formula utilizzata è: loss = mean(max(0, alpha * target_energy - pred_energy + margin)). Se l'energia predetta è sufficientemente grande (superiore a alpha * target_energy + margin), la perdita è zero. Altrimenti, la perdita aumenta proporzionalmente alla differenza tra l'energia target e quella predetta, scalata da alpha e offset da margin.


# ===========================================================================
# Identity loss ArcFace
# ===========================================================================
class IdentityLoss(nn.Module):
    """Cosine identity loss for face SR.

    Pass an already-loaded ArcFace/face-recognition backbone that maps images to
    embeddings. When no backbone is supplied, the module returns zero, making it
    safe to keep in configs before adding ArcFace.
    """

    def __init__(self, backbone: nn.Module | None = None, *, weight: float = 1.0, freeze: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.weight = float(weight)
        if self.backbone is not None and freeze:
            self.backbone.eval()
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
#la funzioen forward viene chiamata durante il training
    def forward(self, pred_image: Tensor, target_image: Tensor) -> Tensor: #questa funzione prende in input l'immagine predetta e quella target, le passa attraverso il backbone per ottenere le rispettive feature, normalizza queste feature e calcola la loss come 1 meno la media del prodotto scalare tra le feature normalizzate. Se il backbone è None o se il peso è 0, restituisce un tensore di zeri.
        if self.backbone is None or self.weight == 0.0:
            return pred_image.new_zeros(())

        pred_features = F.normalize(self.backbone(pred_image), dim=1) #F è torch.nn.functional, contiene funzioni di attivazione, perdita, convoluzione, pooling e altro. normalize normalizza i tensori lungo una dimensione specificata, in questo caso dim=1 (la dimensione dei canali), restituendo un tensore con la stessa forma ma con valori normalizzati.
        with torch.no_grad():
            target_features = F.normalize(self.backbone(target_image), dim=1)
        loss = 1.0 - torch.sum(pred_features * target_features, dim=1).mean()
        return loss * self.weight


# ===========================================================================
# Wavelet SR loss
# ===========================================================================
@dataclass
class WaveletSRLossOutput:
    total: Tensor
    low: Tensor
    high: Tensor
    texture: Tensor
    image: Tensor


class WaveletSRLoss(nn.Module):
    """Unified Wavelet-SRNet loss.

    This combines:
    - low-frequency wavelet MSE with a small weight;
    - high-frequency wavelet MSE;
    - texture loss on high-frequency bands;
    - full-image reconstruction MSE.
    """

    def __init__(
        self,
        *,
        image_channels: int = 3,
        lambda_low: float = 0.01, #peso Wavelet prediction MSE basso per la banda di approssimazione (bassa frequenza) perché è più facile da prevedere e non richiede tanta attenzione durante l'addestramento. La rete può concentrarsi maggiormente sui dettagli ad alta frequenza, che sono più difficili da prevedere e più importanti per la qualità visiva finale dell'immagine super-risoluta.
        lambda_high: float = 1.0, #peso Wavelet prediction MSE più alto per la banda di dettaglio (alta frequenza) perché è più difficile da prevedere e più importante per la qualità visiva finale dell'immagine super-risoluta. La rete deve prestare maggiore attenzione a questa banda durante l'addestramento per migliorare la capacità di catturare i dettagli fini del volto.
        texture_weight: float = 1.0, #peso texture in total loss
        image_weight: float = 0.1, #peso image_loss in total loss
        texture_alpha: float = 1.2,
        texture_margin: float = 0.0, #epsilon sul paper, che permette di avere una zona "morta" in cui non viene penalizzata la perdita di texture. Questo può aiutare a stabilizzare l'addestramento e prevenire che la rete si concentri troppo su dettagli ad alta frequenza che potrebbero essere rumorosi o difficili da prevedere con precisione.
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.image_channels = int(image_channels)
        self.lambda_low = float(lambda_low)
        self.lambda_high = float(lambda_high)
        self.texture_weight = float(texture_weight)
        self.image_weight = float(image_weight)
        self.reduction = reduction
        self.texture = TextureLoss(
            image_channels=image_channels,
            alpha=texture_alpha,
            margin=texture_margin,
        )

    def _mse(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.reduction == "mean":
            return F.mse_loss(pred, target, reduction="mean")
        if self.reduction == "sum_batch":
            return torch.sum((pred - target) ** 2) / (pred.shape[0] * 2.0)
        raise ValueError(f"Unsupported reduction: {self.reduction}")

    def forward(self, pred_wavelets: Tensor, target_wavelets: Tensor, pred_image: Tensor, target_image: Tensor) -> WaveletSRLossOutput:
        if pred_wavelets.shape != target_wavelets.shape:
            raise ValueError(f"Wavelet shape mismatch: {pred_wavelets.shape} vs {target_wavelets.shape}")
        # I primi 3 `image_channels` canali sono la banda di approssimazione (bassa
        # frequenza, struttura globale); i restanti sono i dettagli ad alta frequenza.
        low_pred = pred_wavelets[:, : self.image_channels] #prende i primi image_channels canali dei coefficienti wavelet predetti, che corrispondono alla banda di approssimazione (bassa frequenza, struttura globale)
        low_target = target_wavelets[:, : self.image_channels] 
        high_pred = pred_wavelets[:, self.image_channels :] #prende tutti gli altri canali dei coefficienti wavelet predetti, che corrispondono ai dettagli ad alta frequenza
        high_target = target_wavelets[:, self.image_channels :]

        low = self._mse(low_pred, low_target)
        high = self._mse(high_pred, high_target)
        texture = self.texture(high_pred, high_target) if high_pred.numel() > 0 else high.new_zeros(())
        image = self._mse(pred_image, target_image)

        total = (
            self.lambda_low * low
            + self.lambda_high * high
            + self.texture_weight * texture
            + self.image_weight * image
        )
        return WaveletSRLossOutput(total=total, low=low, high=high, texture=texture, image=image)


__all__ = [
    "TextureLoss",
    "WaveletSRLoss",
    "WaveletSRLossOutput",
    "IdentityLoss",
]
