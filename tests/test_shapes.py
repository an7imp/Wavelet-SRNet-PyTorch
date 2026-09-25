"""Network + transform shape contract for scale 4 / 8 / 16."""

from __future__ import annotations

import pytest
import torch

from models import WaveletSRNet, build_haar_transforms


@pytest.mark.parametrize("scale", [4, 8, 16])
def test_model_output_and_reconstruction_shapes(scale):
    levels = scale.bit_length() - 1
    hr_size = 128
    lr_size = hr_size // scale
    batch = 2

    model = WaveletSRNet(levels=levels).eval()
    _, rec = build_haar_transforms(levels, params_path=None)
    lr = torch.rand(batch, 3, lr_size, lr_size)
    with torch.no_grad():
        coeffs = model(lr)
        sr = rec(coeffs)

    assert tuple(coeffs.shape) == (batch, 3 * 4**levels, lr_size, lr_size)
    assert tuple(sr.shape) == (batch, 3, hr_size, hr_size)
    assert model.output_channels == 3 * 4**levels


def test_8x_canonical_shapes():
    """The headline case from the spec: B x 192 x 16 x 16 -> B x 3 x 128 x 128."""
    model = WaveletSRNet(levels=3).eval()
    _, rec = build_haar_transforms(3, params_path=None)
    lr = torch.rand(4, 3, 16, 16)
    with torch.no_grad():
        coeffs = model(lr)
        sr = rec(coeffs)
    assert tuple(coeffs.shape) == (4, 192, 16, 16)
    assert tuple(sr.shape) == (4, 3, 128, 128)
