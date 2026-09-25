"""Reproducible seeding helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    Args:
        seed: Base seed.
        deterministic: If ``True`` force deterministic cuDNN algorithms. This is
            slower but guarantees bit-reproducibility; keep it ``False`` for
            fast training and ``True`` only when chasing a non-determinism bug.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # ``warn_only`` keeps ops without a deterministic implementation working.
        torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker a distinct, reproducible seed.

    Without this, all workers share the same NumPy/random state and produce
    correlated augmentations.
    """
    base = torch.initial_seed() % 2**32
    seed = (base + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)
