"""DataLoader returns correctly-shaped, aligned LR/HR pairs (all cache modes)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from dataset import FaceSRDataset, create_dataloader


@pytest.mark.parametrize("cache_mode", ["none", "memory", "disk"])
def test_shapes_and_range(synthetic_dataset, tmp_path, cache_mode):
    ds = FaceSRDataset(
        root=synthetic_dataset, hr_size=128, scale=8, hflip=False,
        cache_mode=cache_mode, cache_dir=tmp_path / "cache",
    )
    item = ds[0]
    assert tuple(item["lr"].shape) == (3, 16, 16)
    assert tuple(item["hr"].shape) == (3, 128, 128)
    assert 0.0 <= float(item["lr"].min()) and float(item["lr"].max()) <= 1.0
    assert 0.0 <= float(item["hr"].min()) and float(item["hr"].max()) <= 1.0


def test_lr_is_downscale_of_its_own_hr(synthetic_dataset, tmp_path):
    ds = FaceSRDataset(
        root=synthetic_dataset, hr_size=128, scale=8, hflip=False,
        cache_mode="memory", cache_dir=tmp_path / "cache",
    )
    item = ds[3]
    lr, hr = item["lr"].unsqueeze(0), item["hr"].unsqueeze(0)
    up = F.interpolate(lr, size=(128, 128), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    # Self pairing must beat a mismatched HR by a wide margin.
    other = ds[5]["hr"].unsqueeze(0)
    assert (up - hr).abs().mean() < (up - other).abs().mean()


def test_dataloader_batches(synthetic_dataset, tmp_path):
    loader = create_dataloader(
        root=synthetic_dataset, hr_size=128, scale=8, hflip=False,
        cache_mode="memory", cache_dir=tmp_path / "cache",
        batch_size=4, shuffle=False, num_workers=0,
    )
    batch = next(iter(loader))
    assert batch["lr"].shape[0] == 4
    assert batch["hr"].shape == (4, 3, 128, 128)
    assert len(batch["path"]) == 4


def test_disk_cache_is_reused(synthetic_dataset, tmp_path):
    cache_dir = tmp_path / "cache"
    paths = [f"img_{i:03d}.png" for i in range(8)]
    # hflip=False so the two reads are deterministically comparable.
    ds1 = FaceSRDataset(root=synthetic_dataset, image_list=paths, hr_size=128, scale=8, hflip=False, cache_mode="disk", cache_dir=cache_dir)
    files_after_first = sorted(p.name for p in cache_dir.glob("*.dat"))
    ds2 = FaceSRDataset(root=synthetic_dataset, image_list=paths, hr_size=128, scale=8, hflip=False, cache_mode="disk", cache_dir=cache_dir)
    files_after_second = sorted(p.name for p in cache_dir.glob("*.dat"))
    assert files_after_first == files_after_second  # same key -> reused, not rebuilt
    assert torch.allclose(ds1[0]["hr"], ds2[0]["hr"])
