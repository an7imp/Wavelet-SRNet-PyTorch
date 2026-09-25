"""ArcFace face-recognition backbone (IResNet) for identity metrics / loss.

This is the standard **InsightFace ``arcface_torch`` IResNet** architecture in pure
PyTorch. We keep it self-contained (no ``insightface``/ONNX dependency) so the
same module works as:

- a **metric** — cosine identity similarity between the SR output and the HR
  target (the field-standard "ID" number reported for face super-resolution), and
- an optional **identity loss** — because the backbone is differentiable and runs
  natively under the training pipeline's AMP/CUDA path.

Preprocessing matches how ``arcface_torch`` was trained: RGB images in ``[0, 1]``
are resized to ``112x112`` and normalised to ``[-1, 1]`` via ``(x - 0.5) / 0.5``.
Our data pipeline already yields RGB ``[0, 1]`` tensors, so no channel swap is
needed.

The pretrained weights are **not** bundled. ``build_arcface`` loads them from a
local path if given, otherwise best-effort downloads a known IResNet-50 checkpoint
into ``weights/``. Any failure raises a clear error that the caller is expected to
catch and degrade gracefully (training must never be blocked by a missing
face-recognition model).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# IResNet (InsightFace arcface_torch architecture)
# ---------------------------------------------------------------------------
def _conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation,
    )


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    """The "improved" residual block (BN-first, PReLU) used by InsightFace."""

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None, groups: int = 1,
                 base_width: int = 64, dilation: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = _conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    """IResNet backbone mapping a 112x112 face to a ``num_features``-d embedding."""

    fc_scale = 7 * 7

    def __init__(self, block, layers, *, dropout: float = 0.0, num_features: int = 512,
                 zero_init_residual: bool = False, groups: int = 1, width_per_group: int = 64,
                 replace_stride_with_dilation=None) -> None:
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.1)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, IBasicBlock):
                    nn.init.constant_(module.bn3.weight, 0)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1, dilate: bool = False) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


_ARCH_LAYERS = {
    "r18": [2, 2, 2, 2],
    "r34": [3, 4, 6, 3],
    "r50": [3, 4, 14, 3],
    "r100": [3, 13, 30, 3],
}


def _canon_arch(arch: str) -> str:
    """Normalise 'iresnet50' / '50' / 'R50' -> 'r50'."""
    a = arch.lower().replace("iresnet", "").strip("-_ ")
    a = a if a.startswith("r") else f"r{a}"
    if a not in _ARCH_LAYERS:
        raise ValueError(f"Unknown ArcFace arch {arch!r}; expected one of {sorted(_ARCH_LAYERS)}")
    return a


def build_iresnet(arch: str = "r50", *, num_features: int = 512) -> IResNet:
    arch = _canon_arch(arch)
    return IResNet(IBasicBlock, _ARCH_LAYERS[arch], num_features=num_features)


# ---------------------------------------------------------------------------
# Weight loading (local path or best-effort download)
# ---------------------------------------------------------------------------
# Best-effort public sources per architecture (plain arcface_torch IResNet
# state_dicts, verified to load with zero missing keys). Tried in order; the
# first that downloads and loads cleanly wins. A user-supplied local path always
# takes precedence over these.
_ARCH_URLS = {
    # ArcFace R100 trained on Glint360K — strongest identity discrimination.
    "r100": (
        "https://huggingface.co/BooBooWu/Vec2Face/resolve/main/weights/arcface-r100-glint360k.pth",
    ),
    # ArcFace R50 (IResNet-50) — lighter fallback.
    "r50": (
        "https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/pretrained_models/ArcFace/backbone_ir50.pth",
    ),
}


def _strip_prefixes(state: dict) -> dict:
    """Normalise common state_dict key prefixes (DDP / wrapper)."""
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    if all(k.startswith("backbone.") for k in state):
        state = {k[len("backbone."):]: v for k, v in state.items()}
    return state


def _load_state_into(model: IResNet, ckpt_path: Path) -> IResNet:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    state = _strip_prefixes(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The architecture's parameter names must line up; a handful of buffers may
    # legitimately differ, but a wholesale mismatch means wrong weights/arch.
    core_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if len(core_missing) > 10:
        raise RuntimeError(
            f"ArcFace weights at {ckpt_path} do not match the IResNet-50 architecture "
            f"({len(core_missing)} params missing, e.g. {core_missing[:3]})."
        )
    return model


def build_arcface(
    *,
    weights_path: Optional[str | Path] = None,
    arch: str = "r100",
    device: Optional[torch.device | str] = None,
    download_dir: str | Path = "weights",
) -> "ArcFaceEmbedder":
    """Build an :class:`ArcFaceEmbedder` with pretrained weights loaded.

    Resolution order for the checkpoint:
      1. ``weights_path`` if provided (raises if it does not load);
      2. an already-downloaded file under ``download_dir``;
      3. a best-effort download from :data:`_ARCH_URLS` for this ``arch``.

    Raises on total failure so the caller can disable ArcFace and continue.
    """
    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = _canon_arch(arch)
    model = build_iresnet(arch)

    # 1) explicit local path
    if weights_path is not None:
        p = Path(weights_path)
        if not p.exists():
            raise FileNotFoundError(f"ArcFace weights not found: {p}")
        _load_state_into(model, p)
        return ArcFaceEmbedder(model, device=device)

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    cached = download_dir / f"arcface_{arch}.pth"

    # 2) previously downloaded
    if cached.exists():
        _load_state_into(model, cached)
        return ArcFaceEmbedder(model, device=device)

    # 3) best-effort download
    errors = []
    for url in _ARCH_URLS.get(arch, ()):
        try:
            torch.hub.download_url_to_file(url, str(cached), progress=True)
            _load_state_into(model, cached)
            return ArcFaceEmbedder(model, device=device)
        except Exception as exc:  # pragma: no cover - network dependent
            errors.append(f"{url} -> {exc}")
            if cached.exists():
                cached.unlink(missing_ok=True)
    raise RuntimeError(
        f"Could not obtain ArcFace {arch} weights automatically. Provide a local IResNet "
        "checkpoint via metrics.arcface.weights. Tried:\n  " + "\n  ".join(errors or ["(no URL configured)"])
    )


# ---------------------------------------------------------------------------
# Embedder wrapper (preprocessing + frozen backbone)
# ---------------------------------------------------------------------------
class ArcFaceEmbedder(nn.Module):
    """Wrap an IResNet backbone with ArcFace preprocessing.

    Accepts an RGB image batch in ``[0, 1]`` of any spatial size, resizes to
    ``input_size`` (112) and normalises to ``[-1, 1]`` before the backbone, then
    returns raw (un-normalised) embeddings — callers L2-normalise as needed.
    The backbone is frozen and kept in eval mode.
    """

    def __init__(self, backbone: IResNet, *, input_size: int = 112,
                 device: Optional[torch.device | str] = None, freeze: bool = True) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.backbone = backbone
        if freeze:
            self.backbone.eval()
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
        if device is not None:
            self.to(device)

    def forward(self, images: Tensor) -> Tensor:
        if images.shape[-2:] != (self.input_size, self.input_size):
            images = F.interpolate(images, size=(self.input_size, self.input_size),
                                   mode="bilinear", align_corners=False, antialias=True)
        images = (images.clamp(0, 1) - 0.5) / 0.5  # [0,1] RGB -> [-1,1], arcface_torch convention
        return self.backbone(images)
