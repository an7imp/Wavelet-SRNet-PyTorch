"""Implementazione PyTorch moderna di Wavelet-SRNet.

La rete prende in input un volto a bassa risoluzione, prevede i coefficienti
dei pacchetti wavelet di Haar (frequenze alte e basse), e infine ricostruisce
l'immagine ad alta risoluzione (super-risolta) utilizzando una trasformata 
inversa di Haar con pesi congelati (non addestrabili).
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _haar_packet_filters_2d(levels: int) -> Tensor:
    """Genera matematicamente i filtri 2D dei pacchetti di Haar.

    Shape: ``[4**levels, 1, 2**levels, 2**levels]``. 

    L'ordine è ricorsivo: per ogni coefficiente "genitore", genera le 4 sottomappe
    classiche delle wavelet: LL (Basse), LH (Orizzontali), HL (Verticali), HH (Diagonali).
    Il primo filtro generato è sempre quello di approssimazione (frequenza più bassa).
    """
    if levels < 0:
        raise ValueError("levels must be >= 0")

    filters = torch.ones(1, 1, 1, dtype=torch.float32)
    for _ in range(levels):
        children = []
        for filt in filters:
            # LL: Basso-Basso (Approssimazione)
            top = torch.cat((filt, filt), dim=1)
            bottom = torch.cat((filt, filt), dim=1)
            ll = torch.cat((top, bottom), dim=0) * 0.5

            # LH: Basso-Alto (Dettagli orizzontali)
            top = torch.cat((filt, -filt), dim=1)
            bottom = torch.cat((filt, -filt), dim=1)
            lh = torch.cat((top, bottom), dim=0) * 0.5

            # HL: Alto-Basso (Dettagli verticali)
            top = torch.cat((filt, filt), dim=1)
            bottom = torch.cat((-filt, -filt), dim=1)
            hl = torch.cat((top, bottom), dim=0) * 0.5

            # HH: Alto-Alto (Dettagli diagonali)
            top = torch.cat((filt, -filt), dim=1)
            bottom = torch.cat((-filt, filt), dim=1)
            hh = torch.cat((top, bottom), dim=0) * 0.5

            children.extend((ll, lh, hl, hh))
        # Sovrappone tutti i filtri creati per questo livello
        filters = torch.stack(children, dim=0)
    return filters.unsqueeze(1).contiguous()


def _load_haar_weights_from_pickle(levels: int, channels: int, params_path: str | Path) -> Tensor:
    """Carica i filtri Haar originali di Wavelet-SRNet dal file ``wavelet_weights_c2.pkl``.

    La repository originale salva i filtri raggruppati per canale sotto le chiavi
    ``rec2``, ``rec4``, ``rec8`` e ``rec16``. Gli stessi identici filtri vengono usati
    sia per scomporre l'immagine (tramite Conv2d) sia per ricostruirla (ConvTranspose2d).
    """
    params_path = Path(params_path)
    if not params_path.exists():
        raise FileNotFoundError(f"Wavelet weights file not found: {params_path}")

    with params_path.open("rb") as handle:
        # The original pickle was produced under NumPy 1.x; loading it under
        # NumPy 2.x emits a harmless dtype-align deprecation warning that we
        # silence to keep the console clean.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                data = pickle.load(handle, encoding="latin1")
            except TypeError:  # pragma: no cover - Python 2 compatibility fallback
                data = pickle.load(handle)

    scale = 2 ** int(levels)
    key = f"rec{scale}"
    if key not in data:
        raise KeyError(f"Missing key {key!r} in {params_path}; available keys: {sorted(data.keys())}")

    weight = torch.as_tensor(data[key], dtype=torch.float32)
    expected = (int(channels) * (4 ** int(levels)), 1, scale, scale)
    if tuple(weight.shape) != expected:
        raise ValueError(f"Unexpected shape for {key}: {tuple(weight.shape)}; expected {expected}")
    return weight.contiguous()


class HaarWaveletTransform(nn.Module):
    """Modulo PyTorch per la Trasformata e l'Antitrasformata Wavelet di Haar.
    Nota: i pesi di questo modulo sono congelati (non vengono addestrati).

    Argomenti:
        levels: Numero di livelli Wavelet. Il fattore di scala (SR) sarà ``2 ** levels``.
        inverse: Se ``False`` esegue la scomposizione dell'immagine; se ``True`` ricostruisce.
        channels: Numero di canali dell'immagine (3 per RGB).
        coefficient_major: Ordina i canali come ``[coef0_R, coef0_G, coef0_B, coef1_R, ...]``.
            Questo è l'ordine esatto che si aspetta la WaveletSRNet.
        params_path: Percorso opzionale al file ``wavelet_weights_c2.pkl``. Se fornito,
            usa i pesi originali del paper invece di ricalcolarli matematicamente.
    """

    def __init__(
        self,
        levels: int,
        *,
        inverse: bool = False,
        channels: int = 3,
        coefficient_major: bool = True,
        params_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.levels = int(levels)
        self.inverse = bool(inverse)
        self.channels = int(channels)
        self.coefficient_major = bool(coefficient_major)

        # Carica o genera i filtri
        if params_path is not None:
            weight = _load_haar_weights_from_pickle(self.levels, self.channels, params_path)
        else:
            base = _haar_packet_filters_2d(self.levels)  # [Nw, 1, k, k]
            weight = base.repeat(self.channels, 1, 1, 1)  # channel-grouped order
            
        # Registra i pesi come buffer costante (non partecipa al .backward())
        self.register_buffer("weight", weight, persistent=False)

    @property
    def scale_factor(self) -> int:
        return 2**self.levels

    @property
    def num_coefficients(self) -> int:
        return 4**self.levels

    @property
    def output_channels(self) -> int:
        return self.channels * self.num_coefficients

    def _to_coefficient_major(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        return x.view(b, self.channels, -1, h, w).transpose(1, 2).contiguous().view(b, c, h, w)

    def _to_channel_grouped(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        return x.view(b, -1, self.channels, h, w).transpose(1, 2).contiguous().view(b, c, h, w)

    def forward(self, x: Tensor) -> Tensor:
        k = self.scale_factor
        
        # DECOMPOSIZIONE: Da Immagine a Coefficienti Wavelet
        if not self.inverse:
            if x.shape[-2] % k != 0 or x.shape[-1] % k != 0:
                raise ValueError(f"Input H/W must be divisible by {k}; got {tuple(x.shape[-2:])}.")
            # Usa una convoluzione standard per estrarre le frequenze
            out = F.conv2d(x, self.weight.to(dtype=x.dtype), stride=k, groups=self.channels)
            if self.coefficient_major:
                out = self._to_coefficient_major(out)
            return out

        # RICOSTRUZIONE: Da Coefficienti Wavelet a Immagine
        if self.coefficient_major:
            x = self._to_channel_grouped(x)
        # Usa una convoluzione trasposta (deconvoluzione) per unire le frequenze nei pixel
        return F.conv_transpose2d(x, self.weight.to(dtype=x.dtype), stride=k, groups=self.channels)


@torch.no_grad()
def haar_reconstruction_error(
    levels: int,
    *,
    params_path: str | Path | None = None,
    channels: int = 3,
    size: int | None = None,
    device: torch.device | str = "cpu",
) -> float:
    """Decompose then reconstruct a random image; return the max abs error.

    A correct Haar packet transform is *perfectly invertible*: this should be
    ~1e-6 (float32 round-off). It is the cheapest possible proof that the
    decomposition/reconstruction pair is mathematically sound.
    """
    dec = HaarWaveletTransform(levels, inverse=False, channels=channels, params_path=params_path).to(device)
    rec = HaarWaveletTransform(levels, inverse=True, channels=channels, params_path=params_path).to(device)
    if size is None:
        size = 2 ** levels * 16
    x = torch.rand(2, channels, size, size, device=device)
    recon = rec(dec(x))
    return (recon - x).abs().max().item()


def build_haar_transforms(
    levels: int,
    *,
    params_path: str | Path | None = None,
    channels: int = 3,
    verify: bool = True,
    tolerance: float = 1e-3,
) -> tuple["HaarWaveletTransform", "HaarWaveletTransform"]:
    """Build a *consistent* (decompose, reconstruct) pair, with safe fallback.

    The original ``wavelet_weights_c2.pkl`` ships a broken ``rec16`` filter set
    (scale=16 reconstruction error ~0.13). When ``params_path`` is given we
    therefore verify invertibility and transparently fall back to the
    mathematically-generated Haar filters if the pickle fails — both transforms
    fall back together so they always remain a matched pair.
    """
    use_path = params_path
    if params_path is not None and verify:
        try:
            err = haar_reconstruction_error(levels, params_path=params_path, channels=channels)
            if err > tolerance:
                print(
                    f"[haar] WARNING: '{params_path}' reconstruction error={err:.3e} "
                    f"> tol={tolerance:.0e} at scale={2**levels}. Falling back to "
                    f"generated Haar filters (which are exact)."
                )
                use_path = None
        except Exception as exc:  # pragma: no cover - file/format issues
            print(f"[haar] WARNING: could not load '{params_path}' ({exc}); using generated filters.")
            use_path = None

    dec = HaarWaveletTransform(levels, inverse=False, channels=channels, params_path=use_path)
    rec = HaarWaveletTransform(levels, inverse=True, channels=channels, params_path=use_path)
    return dec, rec


class ResidualBlock(nn.Module):
    """Un classico Blocco Residuo (tipo ResNet)."""
    def __init__(self, in_channels: int, out_channels: int, *, groups: int = 1) -> None:
        super().__init__()
        # Se il numero di canali cambia, usa una conv 1x1 per allinearli per la somma finale
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        # Somma l'input originale (skip connection) all'output delle convoluzioni
        return self.act(self.block(x) + self.skip(x))


class InterimBlock(nn.Module):
    """Blocco di transizione usato prima che la rete si divida nei vari rami di predizione Wavelet.
    Prepara le feature per essere elaborate in gruppi indipendenti."""

    def __init__(self, in_channels: int, out_channels: int, *, groups: int) -> None:
        super().__init__()
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


def _make_residual_stage(
    num_blocks: int,
    in_channels: int,
    out_channels: int,
    *,
    groups: int = 1,
) -> nn.Sequential:
    """Crea una sequenza (uno strato profondo) composta da più ResidualBlock concatenati."""
    blocks = [ResidualBlock(in_channels, out_channels, groups=groups)]
    blocks.extend(ResidualBlock(out_channels, out_channels, groups=groups) for _ in range(1, num_blocks))
    return nn.Sequential(*blocks)


class WaveletPredictionBranch(nn.Module):
    """Singolo ramo specializzato della rete.
    Prende le feature globali e prevede un gruppo specifico di coefficienti Wavelet 
    (es. solo le 3 bande a media frequenza, o le 12 bande ad alta frequenza).
    """
    def __init__(
        self,
        in_channels: int,
        bands: int,
        *,
        wavelet_channels: int = 32,
        num_res_blocks: int = 1,
        image_channels: int = 3,
    ) -> None:
        super().__init__()
        hidden = wavelet_channels * bands
        self.net = nn.Sequential(
            # Usa convoluzioni a gruppi (groups=bands) per mantenere indipendenti le predizioni di ogni banda
            InterimBlock(in_channels, hidden, groups=bands),
            _make_residual_stage(num_res_blocks, hidden, hidden * 2, groups=bands),
            nn.Conv2d(hidden * 2, image_channels * bands, kernel_size=3, padding=1, groups=bands),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class WaveletSRNet(nn.Module):
    """Architettura principale Wavelet-SRNet.

    Formato di Input per SR 8x (immagine LR 16x16): ``[Batch, 3, 16, 16]``.
    Formato di Output: ``[Batch, 3 * 4**levels, 16, 16]``.
    Per ottenere l'immagine visibile, passa l'output a ``HaarWaveletTransform(levels, inverse=True)``.
    """

    def __init__(
        self,
        levels: int = 3,
        *,
        image_channels: int = 3,
        embedding_channels: Iterable[int] = (64, 128, 256, 512, 1024),
        num_embedding_blocks: int = 2,
        wavelet_channels: int = 32,
        num_branch_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.levels = int(levels)
        self.image_channels = int(image_channels)
        embedding_channels = tuple(int(c) for c in embedding_channels)
        if not embedding_channels:
            raise ValueError("embedding_channels cannot be empty")

        # 1. STEM: Il primo strato che "legge" i pixel dell'immagine
        first_channels = embedding_channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(image_channels, first_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(first_channels),
            nn.ReLU(inplace=True),
        )

        # 2. EMBEDDING (Backbone): Estrae progressivamente feature sempre più complesse
        stages = []
        in_c = first_channels
        for out_c in embedding_channels:
            stages.append(_make_residual_stage(num_embedding_blocks, in_c, out_c))
            in_c = out_c
        self.embedding = nn.Sequential(*stages)

        # 3. RAMI DI PREDIZIONE (Branches): Si divide per prevedere i coefficienti
        # L'implementazione originale prevede la banda a bassa frequenza (1 banda),
        # e poi le bande ad alta frequenza (3, 12, 48...) separate per ogni livello wavelet.
        bands_per_branch = [1] + [3 * (4 ** (i - 1)) for i in range(1, self.levels + 1)]
        self.branches = nn.ModuleList(
            WaveletPredictionBranch(
                in_c,
                bands,
                wavelet_channels=wavelet_channels,
                num_res_blocks=num_branch_blocks,
                image_channels=image_channels,
            )
            for bands in bands_per_branch
        )

        self._initialize_weights()

    @property
    def scale_factor(self) -> int:
        return 2**self.levels

    @property
    def num_coefficients(self) -> int:
        return 4**self.levels

    @property
    def output_channels(self) -> int:
        return self.image_channels * self.num_coefficients

    def _initialize_weights(self) -> None:
        """Inizializza i pesi della rete con Kaiming Normal per una convergenza ottimale."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        # Estrae le feature dall'immagine LR
        features = self.embedding(self.stem(x))
        # Passa le feature a tutti i rami in parallelo e concatena le predizioni 
        # per formare l'intero set di coefficienti Wavelet.
        return torch.cat([branch(features) for branch in self.branches], dim=1)


def build_model_from_config(cfg: dict) -> WaveletSRNet:
    """Build a WaveletSRNet from a *validated* config dict (its ``model`` section).

    Shared by every entry point (train.py, test.py, tools/*) so the architecture
    hyper-parameters are read in exactly one place.
    """
    m = cfg["model"]
    return WaveletSRNet(
        levels=m["wavelet_levels"],
        image_channels=m["image_channels"],
        embedding_channels=tuple(m["embedding_channels"]),
        num_embedding_blocks=m["num_embedding_blocks"],
        wavelet_channels=m["wavelet_channels"],
        num_branch_blocks=m["num_branch_blocks"],
    )


__all__ = [
    "HaarWaveletTransform",
    "WaveletSRNet",
    "build_model_from_config",
]
