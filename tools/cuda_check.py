"""CUDA sanity check (requirement: prove CUDA is used correctly).

Prints the device, GPU name, capability and VRAM, builds the actual model and a
real batch, moves them to CUDA, runs one AMP forward+backward, and reports the
device of every key tensor plus VRAM usage. If anything is silently stuck on the
CPU, this makes it obvious.

    python tools/cuda_check.py
    python tools/cuda_check.py --config configs/config_8x.yaml
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
import torch

from models import build_haar_transforms, build_model_from_config
from utils import load_config, validate_config
from utils.env import autocast, gpu_memory_mb, make_grad_scaler, maybe_channels_last, print_environment, resolve_device, setup_backends


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CUDA sanity check")
    p.add_argument("--config", type=str, default="configs/config_8x.yaml")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = validate_config(load_config(args.config))
    setup_backends(cfg)
    device = resolve_device(args.device)

    amp = bool(cfg["train"]["amp"])
    channels_last = bool(cfg["channels_last"])
    print_environment(device, amp=amp, channels_last=channels_last)

    if device.type != "cuda":
        print("\n[cuda_check] FAIL: CUDA was not selected. Training would run on CPU.")
        print("            Install a CUDA build of torch or pass --device cuda.")
        return 1

    levels = cfg["levels"]
    lr_size = cfg["data"]["lr_size"]
    model = build_model_from_config(cfg).to(device)
    model = maybe_channels_last(model, channels_last, device)
    dec, rec = build_haar_transforms(levels, params_path=cfg.get("wavelet_weights_path"))
    rec = rec.to(device)

    lr = torch.rand(args.batch_size, 3, lr_size, lr_size, device=device)
    if channels_last:
        lr = lr.contiguous(memory_format=torch.channels_last)
    hr = torch.rand(args.batch_size, 3, cfg["data"]["hr_size"], cfg["data"]["hr_size"], device=device)

    scaler = make_grad_scaler(amp, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    with autocast(device, amp):
        pred = model(lr)
        sr = rec(pred)
        loss = torch.nn.functional.mse_loss(sr.float(), hr)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    mem = gpu_memory_mb(device)
    grad = next(p.grad for p in model.parameters() if p.grad is not None)
    print("-" * 70)
    print("LIVE TENSOR / MODULE DEVICES")
    print("-" * 70)
    print(f"  model params device : {next(model.parameters()).device}")
    print(f"  inverse-Haar device : {rec.weight.device}")
    print(f"  LR batch device     : {lr.device}  (memory_format channels_last={lr.is_contiguous(memory_format=torch.channels_last)})")
    print(f"  model output device : {pred.device}  dtype={pred.dtype}")
    print(f"  SR output device    : {sr.device}")
    print(f"  a gradient device   : {grad.device}")
    print(f"  AMP enabled         : {amp}")
    print(f"  loss value          : {loss.item():.4f}")
    print(f"  VRAM allocated      : {mem['allocated']:.0f} MB / {mem['total']:.0f} MB")
    print("-" * 70)

    ok = all(t.type == "cuda" for t in (next(model.parameters()).device, lr.device, pred.device, sr.device, grad.device))
    print("[cuda_check] PASS: model, batch, output and gradients are all on CUDA." if ok else "[cuda_check] FAIL: something is not on CUDA.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
