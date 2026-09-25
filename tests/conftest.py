"""Shared pytest fixtures.

Tests are hermetic: they synthesize small smooth images in a temp directory so
they do not depend on the CelebA dataset being present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Make the repo root importable when pytest is launched from anywhere.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _smooth_image(seed: int, size: int = 128, lr_seed: int = 16) -> np.ndarray:
    """A low-frequency RGB image (distinct per seed) so its 8x-downscaled LR is
    distinguishable — important for the overfit test to be learnable."""
    rng = np.random.default_rng(seed)
    low = rng.uniform(0, 1, size=(lr_seed, lr_seed, 3)).astype(np.float32)
    img = Image.fromarray((low * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


@pytest.fixture
def synthetic_dataset(tmp_path) -> Path:
    """Create a folder of 8 smooth synthetic images; return its path."""
    root = tmp_path / "synthetic"
    root.mkdir()
    for i in range(8):
        Image.fromarray(_smooth_image(i)).save(root / f"img_{i:03d}.png")
    return root
