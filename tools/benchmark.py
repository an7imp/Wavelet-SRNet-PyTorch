"""End-to-end benchmark + bottleneck classifier.

Measures, with proper CUDA synchronisation, the per-step cost of every stage and
tells you what the pipeline is bound by (DataLoader / host->device copy / GPU
compute / validation / image saving). This is the tool that answers
"is slow training caused by data, transfers, GPU underuse, validation, or
metrics?".

    python tools/benchmark.py --config configs/config_8x.yaml --batch-size 256
    python tools/benchmark.py --config configs/config_8x.yaml --synthetic   # skip disk entirely

Stages measured per training step:
    data    : waiting for the next batch from the DataLoader
    h2d     : host -> device copy (+ channels_last conversion)
    forward : model forward + inverse-Haar reconstruction (under AMP)
    loss    : loss computation (incl. fp32 wavelet target)
    backward: autograd backward
    optim   : optimizer step / grad scaler update
It also times one validation pass and one comparison-image save.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from dataset import create_dataloader, scan_images
from losses import WaveletSRLoss
from models import build_haar_transforms, build_model_from_config
from utils import load_config, validate_config
from utils.env import autocast, gpu_memory_mb, make_grad_scaler, maybe_channels_last, print_environment, resolve_device, setup_backends
from utils.timing import StageTimer
from utils.viz import save_comparison_grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark + bottleneck classifier")
    p.add_argument("--config", type=str, default="configs/config_8x.yaml")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--warmup", type=int, default=10, help="warmup steps (cuDNN autotune)")
    p.add_argument("--steps", type=int, default=40, help="measured steps")
    p.add_argument("--max-images", type=int, default=4096)
    p.add_argument("--synthetic", action="store_true", help="use random tensors instead of the DataLoader (isolate GPU compute)")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def gpu_utilization(device: torch.device):
    if device.type != "cuda":
        return None
    try:
        return torch.cuda.utilization(device.index or 0)  # needs pynvml
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    cfg = validate_config(load_config(args.config))
    setup_backends(cfg)
    device = resolve_device(args.device)
    data = cfg["data"]
    batch_size = args.batch_size or cfg["train"]["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else cfg["train"]["num_workers"]
    amp = cfg["train"]["amp"] and not args.no_amp
    channels_last = bool(cfg["channels_last"])
    hr_size, lr_size, scale, levels = data["hr_size"], data["lr_size"], data["scale"], cfg["levels"]

    print_environment(device, amp=amp, channels_last=channels_last)
    print(f"benchmark: batch_size={batch_size} num_workers={num_workers} synthetic={args.synthetic}\n")

    model = build_model_from_config(cfg).to(device)
    model = maybe_channels_last(model, channels_last, device)
    model.train()
    dec, rec = build_haar_transforms(levels, params_path=cfg.get("wavelet_weights_path"))
    dec, rec = dec.to(device), rec.to(device)
    # identity_* keys are ArcFace-specific (consumed by train.py); the wavelet
    # loss only takes its own kwargs. The benchmark measures the wavelet path.
    loss_cfg = {k: v for k, v in cfg["loss"].items() if not k.startswith("identity_")}
    criterion = WaveletSRLoss(**loss_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    scaler = make_grad_scaler(amp, device)
    timer = StageTimer(device, enabled=True)

    # --- data source -------------------------------------------------------
    if args.synthetic:
        loader = None
        fixed_lr = torch.rand(batch_size, 3, lr_size, lr_size, device=device)
        fixed_hr = torch.rand(batch_size, 3, hr_size, hr_size, device=device)

        def batches():
            while True:
                yield {"lr": fixed_lr, "hr": fixed_hr}
    else:
        images = scan_images(data["dataset_root"])[: args.max_images]
        loader = create_dataloader(
            root=data["dataset_root"], image_list=images, hr_size=hr_size, scale=scale,
            random_crop=False, hflip=True, cache_mode=data["cache"]["mode"], cache_dir=data["cache"]["dir"],
            cache_size=hr_size, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
        )

        def batches():
            while True:
                yield from loader

    gen = batches()

    def run_step(batch, measure: bool):
        t = timer if measure else StageTimer(device, enabled=False)
        if args.synthetic:
            lr, hr = batch["lr"], batch["hr"]
        else:
            with t.section("data_consume"):
                pass
            with t.section("h2d"):
                lr = batch["lr"].to(device, non_blocking=True)
                hr = batch["hr"].to(device, non_blocking=True)
                if channels_last:
                    lr = lr.contiguous(memory_format=torch.channels_last)
                    hr = hr.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with t.section("loss_target"):
            target = dec(hr.float())
        with t.section("forward"):
            with autocast(device, amp):
                pred = model(lr)
                sr = rec(pred)
        with t.section("loss"):
            out = criterion(pred.float(), target, sr.float(), hr.float())
        with t.section("backward"):
            scaler.scale(out.total).backward()
        with t.section("optim"):
            scaler.step(optimizer)
            scaler.update()
        return lr.size(0)

    # --- warmup (NOT measured; lets cuDNN.benchmark pick algorithms) -------
    data_wait = 0.0
    for _ in range(args.warmup):
        t0 = time.perf_counter()
        batch = next(gen)
        data_wait += time.perf_counter() - t0
        run_step(batch, measure=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # --- measured steps ----------------------------------------------------
    timer.reset()
    data_meter = 0.0
    wall0 = time.perf_counter()
    for _ in range(args.steps):
        t0 = time.perf_counter()
        batch = next(gen)
        data_meter += time.perf_counter() - t0
        run_step(batch, measure=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - wall0

    avgs = timer.averages()
    data_time = data_meter / args.steps  # real wait for the loader (overlaps prefetch)
    step_time = wall / args.steps
    img_s = batch_size / step_time
    mem = gpu_memory_mb(device)

    print("=" * 70)
    print(f"PER-STEP TIMING  (avg over {args.steps} steps, batch={batch_size})")
    print("-" * 70)
    print(f"  {'data (loader wait)':22s}: {data_time*1000:8.2f} ms")
    for name, val in avgs.items():
        if name == "data_consume":
            continue
        print(f"  {name:22s}: {val*1000:8.2f} ms")
    compute = sum(v for n, v in avgs.items() if n in {"loss_target", "forward", "loss", "backward", "optim"})
    print("-" * 70)
    print(f"  {'GPU compute (sum)':22s}: {compute*1000:8.2f} ms")
    print(f"  {'total wall / step':22s}: {step_time*1000:8.2f} ms")
    print(f"  {'throughput':22s}: {img_s:8.0f} img/s")
    print(f"  {'VRAM allocated':22s}: {mem['allocated']:8.0f} MB / {mem['total']:.0f} MB")
    util = gpu_utilization(device)
    if util is not None:
        print(f"  {'GPU utilization (nvml)':22s}: {util:8d} %")

    # --- validation + save timing -----------------------------------------
    if not args.synthetic and loader is not None:
        model.eval()
        with torch.no_grad():
            vt0 = time.perf_counter()
            vb = next(iter(loader))
            vlr = vb["lr"].to(device); vhr = vb["hr"].to(device)
            if channels_last:
                vlr = vlr.contiguous(memory_format=torch.channels_last)
            with autocast(device, amp):
                vsr = rec(model(vlr)).float().clamp(0, 1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            val_time = time.perf_counter() - vt0
            st0 = time.perf_counter()
            save_comparison_grid(vlr, vsr, vhr, Path("results/diagnostics/benchmark_sample.png"), num_images=6)
            save_time = time.perf_counter() - st0
        print(f"  {'one val batch':22s}: {val_time*1000:8.2f} ms")
        print(f"  {'save comparison img':22s}: {save_time*1000:8.2f} ms")

    # --- bottleneck verdict -----------------------------------------------
    print("=" * 70)
    print("BOTTLENECK VERDICT")
    print("-" * 70)
    if args.synthetic:
        print("  Synthetic mode isolates GPU compute. Compare its img/s with the")
        print("  real-data run: if real is much lower, the DataLoader is the bottleneck.")
    else:
        data_frac = data_time / step_time if step_time > 0 else 0.0
        if data_frac > 0.5:
            print(f"  DATA-BOUND: loader wait is {data_frac*100:.0f}% of the step.")
            print("  -> build the disk cache (tools/build_cache.py), raise num_workers,")
            print("     or increase prefetch_factor. GPU is starved.")
        elif compute / step_time > 0.7:
            print(f"  COMPUTE-BOUND: GPU compute is {compute/step_time*100:.0f}% of the step (healthy).")
            print("  -> to go faster: larger batch, AMP (already on), or a smaller model.")
        else:
            print("  MIXED/OVERHEAD-BOUND: neither data nor compute dominates.")
            print("  -> often Python/launch overhead at small batch sizes; increase batch_size.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
