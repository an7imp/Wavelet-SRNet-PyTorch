"""Structured logging: a CSV metrics logger plus console helpers.

The CSV file (one row per validated epoch) is the artefact you actually plot to
decide whether to keep training or stop early (requirement #14). It records loss
components, validation PSNR/SSIM, the bicubic baseline, timing breakdown,
learning rate, batch size, AMP status and the checkpoint path.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class CSVLogger:
    """Append-only CSV logger with a stable, documented schema.

    New columns can appear over time; the header is (re)written so that older
    rows simply have empty cells for columns added later.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames: list[str] = []
        if self.path.exists():
            with self.path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if header:
                    self.fieldnames = header

    def log(self, row: dict[str, Any]) -> None:
        new_keys = [k for k in row if k not in self.fieldnames]
        if new_keys or not self.path.exists():
            self.fieldnames.extend(new_keys)
            self._rewrite_with_header(row)
        else:
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writerow(self._format(row))

    def _rewrite_with_header(self, row: dict[str, Any]) -> None:
        rows: list[dict[str, str]] = []
        if self.path.exists():
            with self.path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            for old in rows:
                writer.writerow(old)
            writer.writerow(self._format(row))

    @staticmethod
    def _format(row: dict[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, value in row.items():
            if isinstance(value, float):
                out[key] = f"{value:.6g}"
            else:
                out[key] = "" if value is None else str(value)
        return out


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    """Dump a dict as pretty JSON (used for config snapshots / run metadata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


def banner(title: str, width: int = 70) -> str:
    return "\n".join(["=" * width, title, "=" * width])
