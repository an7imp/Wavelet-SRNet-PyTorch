"""Lightweight timing utilities for bottleneck analysis.

Two complementary mechanisms (requirement #8 / #performance):

- :class:`AverageMeter` + cheap wall-clock deltas around the training loop tell
  you the *data-vs-compute* split with zero CUDA synchronisation overhead. This
  is what the training loop uses every step.
- :class:`StageTimer` performs *synchronised* per-stage timing (data / forward /
  backward / optimizer / ...). It is used by the benchmark tool and by training
  only when ``profile: true`` because the ``cuda.synchronize()`` it inserts
  serialises the pipeline and slightly slows things down.
"""

from __future__ import annotations

import time
from collections import OrderedDict

import torch


class AverageMeter:
    """Tracks the running average of a scalar."""

    __slots__ = ("total", "count")

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


class StageTimer:
    """Accumulates synchronised timings for named stages.

    Usage::

        timer = StageTimer(device)
        with timer.section("forward"):
            out = model(x)
        ...
        print(timer.summary())

    On CUDA each section is wrapped with ``torch.cuda.synchronize()`` so the
    measured time reflects real GPU work rather than just kernel-launch latency.
    """

    def __init__(self, device: torch.device, *, enabled: bool = True) -> None:
        self.device = device
        self.enabled = enabled
        self.meters: "OrderedDict[str, AverageMeter]" = OrderedDict()

    def _sync(self) -> None:
        if self.enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def section(self, name: str) -> "_Section":
        return _Section(self, name)

    def record(self, name: str, seconds: float, n: int = 1) -> None:
        self.meters.setdefault(name, AverageMeter()).update(seconds, n)

    def averages(self) -> "OrderedDict[str, float]":
        return OrderedDict((name, meter.avg) for name, meter in self.meters.items())

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()

    def summary(self, *, unit: str = "ms") -> str:
        scale = 1000.0 if unit == "ms" else 1.0
        parts = [f"{name}={meter.avg * scale:.2f}{unit}" for name, meter in self.meters.items()]
        return " ".join(parts)


class _Section:
    def __init__(self, timer: StageTimer, name: str) -> None:
        self.timer = timer
        self.name = name
        self.start = 0.0

    def __enter__(self) -> "_Section":
        self.timer._sync()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.timer._sync()
        self.timer.record(self.name, time.perf_counter() - self.start)


class Throughput:
    """Convenience wrapper to report images/second."""

    def __init__(self) -> None:
        self.images = 0
        self.start = time.perf_counter()

    def update(self, n: int) -> None:
        self.images += n

    def rate(self) -> float:
        elapsed = time.perf_counter() - self.start
        return self.images / elapsed if elapsed > 0 else 0.0

    def reset(self) -> None:
        self.images = 0
        self.start = time.perf_counter()
