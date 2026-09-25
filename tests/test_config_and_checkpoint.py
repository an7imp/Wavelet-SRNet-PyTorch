"""Config validation is strict, and checkpoints round-trip for safe resume."""

from __future__ import annotations

import pytest
import torch
from torch.optim import Adam

from models import WaveletSRNet
from utils.checkpoint import BestTracker, load_checkpoint, save_checkpoint
from utils.config import ConfigError, scale_to_levels, validate_config


def _minimal_cfg():
    return {"data": {"dataset_root": "img_align_celeba", "scale": 8, "hr_size": 128}}


def test_scale_to_levels():
    assert scale_to_levels(4) == 2
    assert scale_to_levels(8) == 3
    assert scale_to_levels(16) == 4


def test_levels_derived_from_scale():
    cfg = validate_config(_minimal_cfg())
    assert cfg["levels"] == 3
    assert cfg["model"]["wavelet_levels"] == 3
    assert cfg["data"]["lr_size"] == 16


def test_rejects_non_power_of_two_scale():
    cfg = _minimal_cfg()
    cfg["data"]["scale"] = 6
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_rejects_indivisible_hr_size():
    cfg = _minimal_cfg()
    cfg["data"]["hr_size"] = 130  # not divisible by 8
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_requires_dataset_root():
    with pytest.raises(ConfigError):
        validate_config({"data": {"scale": 8}})


def test_cache_size_must_fit_hr():
    cfg = _minimal_cfg()
    cfg["data"]["cache"] = {"size": 64}  # < hr_size 128
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_best_tracker_and_early_stopping():
    t = BestTracker(mode="max", patience=2)
    assert t.update(20.0, 1) is True
    assert t.update(21.0, 2) is True
    assert t.update(20.5, 3) is False
    assert not t.should_stop
    assert t.update(20.4, 4) is False
    assert t.should_stop  # 2 bad epochs in a row
    assert t.best == 21.0 and t.best_epoch == 2


def test_checkpoint_roundtrip(tmp_path):
    model = WaveletSRNet(levels=3, embedding_channels=(16, 32), num_embedding_blocks=1)
    opt = Adam(model.parameters(), lr=1e-3)
    # take one step so optimizer/model state is non-trivial
    out = model(torch.rand(1, 3, 16, 16)).sum()
    out.backward()
    opt.step()

    path = tmp_path / "ckpt.pth"
    save_checkpoint(path, model=model, optimizer=opt, epoch=7, global_step=123, best_metric=24.5, config={"levels": 3})

    model2 = WaveletSRNet(levels=3, embedding_channels=(16, 32), num_embedding_blocks=1)
    opt2 = Adam(model2.parameters(), lr=1e-3)
    meta = load_checkpoint(path, model=model2, device=torch.device("cpu"), optimizer=opt2, strict=True)

    assert meta["epoch"] == 7 and meta["global_step"] == 123 and meta["best_metric"] == 24.5
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
