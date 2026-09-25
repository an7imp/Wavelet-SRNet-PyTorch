"""Haar transform is a perfect inverse pair (the core of the paper's design)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models import build_haar_transforms, haar_reconstruction_error
from models.waveletsrnet import _haar_packet_filters_2d

# The original pickle is optional (gitignored, only needed to cross-check the
# paper's shipped filters). The generated filters are the default and are always
# tested; pickle tests skip cleanly when the file is absent.
_PICKLE = Path(__file__).resolve().parent.parent / "weights" / "wavelet_weights_c2.pkl"
needs_pickle = pytest.mark.skipif(
    not _PICKLE.exists(), reason="optional weights/wavelet_weights_c2.pkl not present"
)


@pytest.mark.parametrize("scale", [2, 4, 8, 16])
def test_generated_filters_reconstruct_identity(scale):
    levels = scale.bit_length() - 1  # 2->1, 4->2, 8->3, 16->4
    err = haar_reconstruction_error(levels, params_path=None, size=2**levels * 8)
    assert err < 1e-4, f"generated Haar at scale={scale} recon err={err}"


@needs_pickle
@pytest.mark.parametrize("scale", [4, 8])
def test_pickle_filters_reconstruct_identity_low_scales(scale):
    levels = scale.bit_length() - 1
    err = haar_reconstruction_error(levels, params_path=_PICKLE)
    assert err < 1e-4


@needs_pickle
def test_factory_falls_back_for_broken_scale16():
    # The shipped rec16 is broken; the factory must transparently fall back and
    # still deliver a perfectly-invertible pair.
    dec, rec = build_haar_transforms(4, params_path=_PICKLE)
    x = torch.rand(2, 3, 128, 128)
    err = (rec(dec(x)) - x).abs().max().item()
    assert err < 1e-4


def test_filters_are_orthonormal_haar():
    # Each level's 4 base filters (LL/LH/HL/HH) should be orthogonal and unit-norm.
    filters = _haar_packet_filters_2d(1).reshape(4, -1)
    gram = filters @ filters.t()
    assert torch.allclose(gram, torch.eye(4), atol=1e-6)


def test_low_band_is_first():
    # Channel 0 must be the LL (approximation) band: a constant image decomposes
    # to a single non-zero low-frequency coefficient and ~zero high-frequency ones.
    dec, _ = build_haar_transforms(3, params_path=None)
    const = torch.ones(1, 3, 128, 128) * 0.5
    coeffs = dec(const)
    low = coeffs[:, :3].abs().mean()
    high = coeffs[:, 3:].abs().mean()
    assert low > 1e-3 and high < 1e-5
