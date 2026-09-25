"""Data pipeline for face super-resolution.

The dominant per-epoch cost in the original pipeline was decoding every JPEG and
resizing it *again on every epoch*. This module removes that repeated work with
an optional **decoded cache**:

- ``cache.mode = "disk"``  (recommended for full CelebA): images are decoded and
  resized to ``cache.size`` **once** and stored as a uint8 memmap on disk. Across
  epochs and across runs we then only read raw bytes — no JPEG decode, no PIL
  resize. The memmap is file-backed so multiple DataLoader workers share it
  through the OS page cache without duplicating RAM (important on Windows spawn).
- ``cache.mode = "memory"`` (for tiny overfit/smoke subsets): everything is kept
  in a uint8 RAM tensor. Use ``num_workers = 0`` with this mode.
- ``cache.mode = "none"``: classic on-the-fly decoding (kept for parity / when
  disk space is tight).

Per item we then do only cheap work: read uint8 HR, random crop (if the cache is
larger than ``hr_size``), horizontal flip, convert to float, and produce the LR
image with an antialiased bicubic downscale.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

try:
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # Pillow < 9
    BICUBIC = Image.BICUBIC

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_CACHE_VERSION = 1



#scan_image -> lista di immagini in root, con caching in .image_index.txt
#split_image_lists -> split train/val/test in modo riproducibile, ottengo train:[], val:[], test:[]
#FaceSRDataset -> 
    #build_disk_cache -> costruisce un memmap di immagini decodificate e ridimensionate, [N, size, size, 3] uint8 memmap per velocizzare il caricamento delle immagini durante l'addestramento. La cache viene salvata su disco e può essere riutilizzata in esecuzioni successive.
    #load image → uint8 HR → resize -> center crop -> (salvataggio in cache) -> tensor HR → random crop (No è disattivato) → flip random  → bicubic downsample → LR
    #il dataset restituisce un dizionario tipo {"lr": Tensor (3, 16, 16), "hr": Tensor (3,,128,128), "path": str} per ogni elemento, dove "lr" è l'immagine a bassa risoluzione, "hr" è l'immagine ad alta risoluzione e "path" è il percorso relativo dell'immagine originale. Le immagini sono restituite come tensori float in [0, 1].
    #il dataloader crea i batch facendo stack dei vari tensori richiesti al dataset, in output: {"lr": Tensor (B, 3, 16, 16), "hr": Tensor (B, 3, 128, 128), "path": list[str]} dove B è la dimensione del batch.
    
def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def scan_images(root: str | Path) -> list[str]:
    """Return sorted relative image paths under ``root`` (cached to a sidecar file).

    Scanning 200k+ files repeatedly is wasteful, so the sorted listing is cached
    in ``<root>/.image_index.txt`` and reused while the directory's file count is
    unchanged. Delete that file to force a rescan.
    """
    root = Path(root)
    index_file = root / ".image_index.txt"
    if index_file.exists():
        try:
            lines = index_file.read_text(encoding="utf-8").splitlines()
            count, *paths = lines
            if int(count) == len(paths) and paths:
                return paths
        except Exception:  # pragma: no cover - corrupt index -> rescan
            pass
    paths = [str(p.relative_to(root)) for p in sorted(root.rglob("*")) if is_image_file(p)]
    try:
        index_file.write_text("\n".join([str(len(paths)), *paths]), encoding="utf-8")
    except Exception:  # pragma: no cover - read-only dataset dir
        pass
    return paths


def split_image_lists(root: str | Path, *, seed: int, val_split: float, test_split: float) -> dict[str, list[str]]:
    """Reproducible train/val/test split of the images under ``root``.

    Single source of truth for every entry point (train.py, test.py, tools/*):
    the same ``seed`` always yields the same split, with no manual list files.
    """
    all_images = scan_images(root)
    rng = random.Random(int(seed))
    shuffled = all_images[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * float(val_split))
    n_test = int(len(shuffled) * float(test_split))
    return {
        "val": shuffled[:n_val],
        "test": shuffled[n_val : n_val + n_test],
        "train": shuffled[n_val + n_test :],
    }


def _load_and_fit(image_path: Path, size: int) -> np.ndarray:
    """Open an RGB image, resize the short side to ``size`` and center-crop to a square.

    Returns a ``(size, size, 3)`` uint8 array. This is the single canonical
    pre-processing step shared by the cache builder and the no-cache path.
    """
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if (w, h) != (size, size):
        scale = size / min(w, h)
        new_w, new_h = max(size, round(w * scale)), max(size, round(h * scale))
        image = image.resize((new_w, new_h), BICUBIC)
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        image = image.crop((left, top, left + size, top + size))
    # np.array (not asarray) -> a writable copy, so torch.from_numpy never warns
    # about a non-writable (PIL-backed) buffer downstream.
    return np.array(image, dtype=np.uint8)


def _cache_key(paths: list[str], size: int) -> str:
    hasher = hashlib.sha1()
    hasher.update(f"v{_CACHE_VERSION}|{size}|{len(paths)}|".encode("utf-8"))
    for p in paths:
        hasher.update(p.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()[:16]


def build_disk_cache(
    root: str | Path,
    paths: list[str],
    cache_dir: str | Path,
    size: int,
    *,
    verbose: bool = True,
) -> Path:
    """Build (or reuse) a uint8 memmap of shape ``[N, size, size, 3]``.

    Returns the path to the ``.dat`` file. A sidecar ``.json`` records the
    metadata so a stale/incompatible cache is rebuilt automatically.
    """
    root = Path(root)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(paths, size)
    dat_path = cache_dir / f"celeba_{size}px_{key}.dat"
    meta_path = dat_path.with_suffix(".json")

    if dat_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("count") == len(paths) and meta.get("size") == size and meta.get("complete"):
            return dat_path

    if verbose:
        gib = len(paths) * size * size * 3 / 1024**3
        print(f"[cache] Building decoded cache: {len(paths)} imgs @ {size}px (~{gib:.2f} GiB) -> {dat_path}")

    shape = (len(paths), size, size, 3)
    mmap = np.lib.format.open_memmap(dat_path, mode="w+", dtype=np.uint8, shape=shape)
    report_every = max(1, len(paths) // 20)
    for i, rel in enumerate(paths):
        mmap[i] = _load_and_fit(root / rel, size)
        if verbose and (i + 1) % report_every == 0:
            print(f"[cache]   {i + 1}/{len(paths)} ({100 * (i + 1) / len(paths):.0f}%)")
    mmap.flush()
    del mmap
    meta_path.write_text(
        json.dumps({"count": len(paths), "size": size, "complete": True, "version": _CACHE_VERSION}),
        encoding="utf-8",
    )
    if verbose:
        print(f"[cache] Done: {dat_path}")
    return dat_path


class FaceSRDataset(Dataset[dict[str, Tensor | str]]):
    """Face SR dataset returning ``{"lr", "hr", "path"}`` with values in [0, 1]."""

    def __init__(
        self,
        *,
        root: str | Path,
        image_list: Optional[Iterable[str]] = None,
        hr_size: int = 128,
        scale: int = 8,
        random_crop: bool = False,
        hflip: bool = True,
        cache_mode: str = "disk",
        cache_dir: str | Path = ".cache",
        cache_size: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.image_paths = list(image_list) if image_list is not None else scan_images(self.root)
        if not self.image_paths:
            raise ValueError(f"No images found in {self.root}")

        self.hr_size = int(hr_size)
        self.scale = int(scale)
        if self.hr_size % self.scale != 0:
            raise ValueError(f"hr_size={self.hr_size} must be divisible by scale={self.scale}")
        self.lr_size = self.hr_size // self.scale
        self.cache_size = int(cache_size) if cache_size else self.hr_size
        if self.cache_size < self.hr_size:
            raise ValueError("cache_size must be >= hr_size")
        self.random_crop = bool(random_crop) and self.cache_size > self.hr_size
        self.hflip = bool(hflip)
        self.cache_mode = cache_mode

        self._mmap_path: Optional[Path] = None
        self._mmap: Optional[np.memmap] = None
        self._mem: Optional[np.ndarray] = None

        if cache_mode == "disk":
            self._mmap_path = build_disk_cache(self.root, self.image_paths, cache_dir, self.cache_size)
        elif cache_mode == "memory":
            self._mem = np.empty((len(self.image_paths), self.cache_size, self.cache_size, 3), dtype=np.uint8)
            for i, rel in enumerate(self.image_paths):
                self._mem[i] = _load_and_fit(self.root / rel, self.cache_size)
        elif cache_mode != "none":
            raise ValueError(f"Unknown cache_mode: {cache_mode}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _get_mmap(self) -> np.memmap:
        # Opened lazily so each worker process maps the file itself (no pickling
        # of a giant array through spawn).
        if self._mmap is None:
            self._mmap = np.load(self._mmap_path, mmap_mode="r")
        return self._mmap

    def _hr_uint8(self, index: int) -> np.ndarray:
        if self._mem is not None:
            return self._mem[index]
        if self._mmap_path is not None:
            # np.array(copy) -> a writable copy; the memmap itself is read-only,
            # and we don't want to keep a reference to it past this call.
            return np.array(self._get_mmap()[index])
        return _load_and_fit(self.root / self.image_paths[index], self.cache_size)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        arr = self._hr_uint8(index)  # (cs, cs, 3) uint8
        # HWC uint8 -> CHW float in [0, 1]
        hr = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float().div_(255.0)

        if self.cache_size > self.hr_size:
            max_off = self.cache_size - self.hr_size
            if self.random_crop:
                top = random.randint(0, max_off)
                left = random.randint(0, max_off)
            else:
                top = left = max_off // 2
            hr = hr[:, top : top + self.hr_size, left : left + self.hr_size]

        if self.hflip and random.random() < 0.5:
            hr = torch.flip(hr, dims=[2])

        hr = hr.contiguous()
        lr = TF.resize(
            hr,
            [self.lr_size, self.lr_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).clamp_(0, 1)
        return {"lr": lr, "hr": hr, "path": str(self.image_paths[index])}


def create_dataloader(
    *,
    root: str | Path,
    image_list: Optional[Iterable[str]] = None,
    hr_size: int = 128,
    scale: int = 8,
    random_crop: bool = False,
    hflip: bool = True,
    cache_mode: str = "disk",
    cache_dir: str | Path = ".cache",
    cache_size: Optional[int] = None,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: Optional[bool] = None,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    worker_init_fn=None,
) -> DataLoader:
    dataset = FaceSRDataset(
        root=root,
        image_list=image_list,
        hr_size=hr_size,
        scale=scale,
        random_crop=random_crop,
        hflip=hflip,
        cache_mode=cache_mode,
        cache_dir=cache_dir,
        cache_size=cache_size,
    )
    if drop_last is None:
        drop_last = shuffle
    # persistent_workers/prefetch_factor are only valid when workers exist.
    loader_kwargs: dict = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)
