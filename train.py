"""Train modern Wavelet-SRNet.

The network predicts Haar wavelet packet coefficients of the HR face and the SR
image is reconstructed through a *fixed* inverse Haar transform — faithful to
"Wavelet-SRNet: A Wavelet-Based CNN for Multi-scale Face Super Resolution".

PIPELINE END-TO-END (quali file fanno cosa)
-------------------------------------------
    configs/*.yaml          parametri (scale, loss weights, training)   -> utils/config.py
        |
    dataset.py              volto HR -> coppia (LR 16x16, HR 128x128)
        |  lr                                       |  hr
        v                                           v
    models/waveletsrnet.py  WaveletSRNet(lr) = coeff. wavelet predetti
        |                                       Haar diretta su hr = coeff. target
        |  pred_wavelets                            |  target_wavelets
        v                                           |
    Haar inversa (rec)  ->  immagine SR             |
        |                                           |
        +---------------------+---------------------+
                              v
    losses/__init__.py     WaveletSRLoss (low/high/texture/image) + IdentityLoss (ArcFace)
                              |  backward + Adam
                              v
    metrics/__init__.py    in validazione: PSNR / SSIM / LPIPS / ArcFace-ID  vs  baseline bicubico
        |
    utils/checkpoint.py    salva best.pth / last.pth ; utils/logging_utils.py -> metrics.csv

This training script is built for efficient, reproducible RTX-class GPU training:
AMP, channels_last, TF32, big batches, a decoded data cache, per-stage timing,
LR scheduling, best-checkpoint tracking, early stopping, safe resume and CSV
logging. Run modes (smoke / overfit / short / medium / long) are selected via
config files in ``configs/`` and/or the CLI overrides below.

Examples
--------
    python train.py --config configs/config_8x_smoke.yaml          # quick smoke test
    python train.py --config configs/config_8x_overfit.yaml         # tiny overfit
    python train.py --config configs/config_8x.yaml                 # serious run
    python train.py --config configs/config_8x.yaml --resume results/8x/checkpoints/last.pth
    python train.py --config configs/config_8x.yaml --overfit 8     # CLI overfit on 8 imgs
"""



#PIPELINE TENSORI upscaling 8x: r=2^n e N=3 livelli di wavelet, quindi N_w=4^n=64 coefficienti totali,
#TENSORI [B, C, H, W] con B=Batch size, C=canali, H=altezza, W=larghezza

#DATALOADER: LR: (B, 3, 16, 16) HR: (B, 3, 128, 128)
#decomposizione HR: (B, 3, 128, 128) → (B, 192, 16, 16) 192=3*4^3 ovvero le 3 bandergb

#ESTRAZIONE FEATURE: LR (B, 3, 16, 16) -> Feature map (B,1024, 16,16)
#predizione coefficienti Feature map (B, 1024, 16, 16) → HR_predetti (B, 192, 16, 16) 192=3*4^3 ovvero le 3 bandergb
#ricostruzione HR_predetti (B, 192, 16, 16) → SR (B, 3, 128, 128)
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam

from dataset import create_dataloader, scan_images, split_image_lists
from losses import WaveletSRLoss, IdentityLoss
from metrics import psnr_per_image, ssim_per_image, LPIPSMetric, ArcFaceSimilarity
from models import build_arcface, build_haar_transforms, build_model_from_config
from utils import seed_everything, worker_init_fn, load_config, validate_config
from utils.env import (
    autocast,
    gpu_memory_mb,
    make_grad_scaler,
    maybe_channels_last,
    print_environment,
    resolve_device,
    setup_backends,
)
from utils.checkpoint import BestTracker, load_checkpoint, save_checkpoint
from utils.logging_utils import CSVLogger, banner, save_json
from utils.timing import AverageMeter, Throughput
from utils.viz import save_comparison_grid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train modern Wavelet-SRNet", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=str, default="configs/config_8x.yaml")
    p.add_argument("--resume", type=str, default=None, help="checkpoint path to resume from")
    p.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    # --- overrides so modes need no code edits ---
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="cap optimizer steps per epoch")
    p.add_argument("--max-images", type=int, default=None, help="cap number of training images")
    p.add_argument("--overfit", type=int, default=None, help="overfit on N images (val == train)")
    p.add_argument("--val-every", type=int, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-cache", action="store_true", help="disable decoded cache (cache_mode=none)")
    p.add_argument("--profile", action="store_true", help="synced per-stage timing")
    p.add_argument("--tag", type=str, default=None, help="suffix appended to output_dir")
    return p.parse_args()


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
        cfg["train"]["val_batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.max_images is not None:
        cfg["data"]["train_max_images"] = args.max_images
    if args.val_every is not None:
        cfg["train"]["val_every"] = args.val_every
    if args.no_amp:
        cfg["train"]["amp"] = False
    if args.no_cache:
        cfg["data"]["cache"]["mode"] = "none"
    if args.profile:
        cfg["train"]["profile"] = True
    if args.tag:
        cfg["output_dir"] = f"{cfg['output_dir']}_{args.tag}"
    return cfg


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict[str, Any]) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    sched = cfg["train"]["scheduler"]
    if sched["type"] == "none":
        return None
    epochs = cfg["train"]["epochs"]
    warmup = int(sched["warmup_epochs"])
    base_lr = cfg["train"]["lr"]
    min_factor = float(sched["min_lr"]) / base_lr if base_lr > 0 else 0.0

    def lr_lambda(epoch: int) -> float:  # epoch is 0-indexed within the scheduler
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / warmup
        if sched["type"] == "cosine":
            progress = (epoch - warmup) / max(1, epochs - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return min_factor + (1.0 - min_factor) * cosine
        # step
        decays = (epoch - warmup) // max(1, int(sched["step_size"]))
        return max(min_factor, float(sched["gamma"]) ** decays)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def overfit_n(cfg: dict[str, Any], args: argparse.Namespace) -> int | None:
    """Number of images to overfit, from CLI (--overfit) or config (data.overfit)."""
    return args.overfit if args.overfit else cfg["data"].get("overfit")


def make_splits(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """Return (train_list, val_list). Reproducible given the seed."""
    data = cfg["data"]
    all_images = scan_images(data["dataset_root"])
    if not all_images:
        raise SystemExit(f"No images found under data.dataset_root={data['dataset_root']!r}")

    n = overfit_n(cfg, args)
    if n:
        subset = all_images[:n]
        print(f"[overfit] Using {len(subset)} images for BOTH train and val (no augmentation).")
        return subset, subset

    splits = split_image_lists(
        data["dataset_root"], seed=int(cfg["seed"]),
        val_split=float(data["val_split"]), test_split=float(data["test_split"]),
    )
    train_list, val_list = splits["train"], splits["val"]
    if data.get("train_max_images"):
        train_list = train_list[: int(data["train_max_images"])]
    if data.get("val_max_images"):
        val_list = val_list[: int(data["val_max_images"])]
    return train_list, val_list


# ---------------------------------------------------------------------------
# Validation / baseline
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    wavelet_rec,
    loader,
    device: torch.device,
    amp: bool,
    channels_last: bool,
    sample_path: Optional[Path] = None,
    num_sample_images: int = 6,
    max_batches: Optional[int] = None,
    lpips_metric: Optional[LPIPSMetric] = None,
    arcface_metric: Optional[ArcFaceSimilarity] = None,
) -> dict[str, float]:
    model.eval()
    psnrs: list[Tensor] = []
    ssims: list[Tensor] = []
    lpips_vals: list[float] = []
    id_vals: list[float] = []
    t0 = time.perf_counter()
    for batch_idx, batch in enumerate(loader):
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        if channels_last and device.type == "cuda":
            lr = lr.contiguous(memory_format=torch.channels_last)
        with autocast(device, amp):
            sr = wavelet_rec(model(lr)).float().clamp(0, 1)
        psnrs.append(psnr_per_image(sr, hr, luminance=True).cpu())
        ssims.append(ssim_per_image(sr, hr, luminance=False).cpu())
        if lpips_metric is not None:
            try:
                lpips_vals.append(lpips_metric(sr, hr))
            except Exception:
                pass
        if arcface_metric is not None:
            try:
                id_vals.append(arcface_metric(sr, hr))
            except Exception:
                pass
        if sample_path is not None and batch_idx == 0:
            save_comparison_grid(lr, sr, hr, sample_path, num_images=num_sample_images)
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    model.train()
    result = {
        "psnr": torch.cat(psnrs).mean().item() if psnrs else 0.0,
        "ssim": torch.cat(ssims).mean().item() if ssims else 0.0,
        "eval_time": time.perf_counter() - t0,
    }
    if lpips_vals:
        result["lpips"] = float(np.mean(lpips_vals))
    if id_vals:
        result["id_sim"] = float(np.mean(id_vals))
    return result


@torch.no_grad()
def bicubic_baseline(*, loader, device: torch.device, max_batches: Optional[int] = None) -> dict[str, float]:
    """PSNR/SSIM of bicubic-upscaled LR vs HR — the number the model must beat."""
    import torch.nn.functional as F

    psnrs: list[Tensor] = []
    ssims: list[Tensor] = []
    for batch_idx, batch in enumerate(loader):
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        psnrs.append(psnr_per_image(up, hr, luminance=True).cpu())
        ssims.append(ssim_per_image(up, hr, luminance=False).cpu())
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    return {
        "bicubic_psnr": torch.cat(psnrs).mean().item() if psnrs else 0.0,
        "bicubic_ssim": torch.cat(ssims).mean().item() if ssims else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
     #pars degli argomenti e carica il file di config
    args = parse_args()
    cfg = validate_config(load_config(args.config))
    cfg = apply_overrides(cfg, args)

    #seed e gpu
    seed_everything(int(cfg["seed"]), deterministic=bool(cfg.get("deterministic", False)))
    setup_backends(cfg)
    device = resolve_device(args.device)

    #estrae le configurazioni di data e train dal file di config, crea le cartelle di output, checkpoint e sample, e salva una copia del config in output_dir
    data_cfg, train_cfg = cfg["data"], cfg["train"]
    out_dir = Path(cfg["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    sample_dir = out_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config_snapshot.json", cfg)

    #estrae i parametri di training, ovvero il numero di livelli della wavelet, il fattore di scala, se usare channels_last e se usare AMP
    #channel last è un formato di memoria che può essere più efficiente per alcune operazioni su GPU, mentre AMP (Automatic Mixed Precision) permette di usare precisione mista per velocizzare il training e ridurre l'uso di memoria

    levels = cfg["levels"]
    scale = data_cfg["scale"]
    channels_last = bool(cfg["channels_last"])
    amp = bool(train_cfg["amp"])

    # --- data --------------------------------------------------------------
    #overfit è una modalità di training in cui il modello viene addestrato su un piccolo numero di immagini, per verificare che il modello sia in grado di imparare correttamente. Se overfit è attivo, il loader dei dati viene configurato per usare solo un sottoinsieme delle immagini disponibili, e la cache dei dati viene disabilitata per evitare di memorizzare dati non necessari.
    is_overfit = bool(overfit_n(cfg, args))
    cache_mode = "memory" if is_overfit and data_cfg["cache"]["mode"] != "none" else data_cfg["cache"]["mode"]
    #i workers sono processo che caricano i dati in parallelo
    num_workers = 0 if (is_overfit or cache_mode == "memory") else train_cfg["num_workers"]
    train_list, val_list = make_splits(cfg, args)

    train_loader = create_dataloader(
        root=data_cfg["dataset_root"], image_list=train_list,
        hr_size=data_cfg["hr_size"], scale=scale,
        random_crop=bool(data_cfg["random_crop"]) and not is_overfit,
        hflip=bool(data_cfg["hflip"]) and not is_overfit,
        cache_mode=cache_mode, cache_dir=data_cfg["cache"]["dir"], cache_size=data_cfg["cache"]["size"],
        batch_size=train_cfg["batch_size"], shuffle=not is_overfit, num_workers=num_workers,
        persistent_workers=train_cfg["persistent_workers"], prefetch_factor=train_cfg["prefetch_factor"],
        worker_init_fn=worker_init_fn,
    )
    val_loader = create_dataloader(
        root=data_cfg["dataset_root"], image_list=val_list,
        hr_size=data_cfg["hr_size"], scale=scale, random_crop=False, hflip=False,
        cache_mode=cache_mode, cache_dir=data_cfg["cache"]["dir"], cache_size=data_cfg["cache"]["size"],
        batch_size=train_cfg["val_batch_size"], shuffle=False, num_workers=num_workers,
        persistent_workers=train_cfg["persistent_workers"], prefetch_factor=train_cfg["prefetch_factor"],
        worker_init_fn=worker_init_fn,
    )

    # --- model / transforms / loss ----------------------------------------
    model = build_model_from_config(cfg).to(device)
    model = maybe_channels_last(model, channels_last, device)
    if torch.cuda.device_count() > 1 and train_cfg["data_parallel"]:
        model = nn.DataParallel(model)
    if train_cfg.get("compile", False):
        try:
            model = torch.compile(model)
            print("[train] torch.compile enabled.")
        except Exception as exc:  # pragma: no cover
            print(f"[train] torch.compile unavailable ({exc}); continuing eager.")

    wavelet_weights_path = cfg.get("wavelet_weights_path")
    wavelet_dec, wavelet_rec = build_haar_transforms(levels, params_path=wavelet_weights_path)
    wavelet_dec, wavelet_rec = wavelet_dec.to(device), wavelet_rec.to(device)

    # identity_weight / identity_arch 
    loss_cfg = dict(cfg["loss"])
    identity_weight = float(loss_cfg.pop("identity_weight", 0.0))
    identity_arch = str(loss_cfg.pop("identity_arch", "r50"))
    criterion = WaveletSRLoss(**loss_cfg).to(device) #i due asterisco servono per passare i parametri del dizionario come argomenti keyword alla funzione WaveletSRLoss. In questo modo, ogni chiave del dizionario cfg["loss"] diventa un argomento della funzione, e il valore corrispondente viene passato come valore dell'argomento. Ad esempio, se cfg["loss"] contiene {"alpha": 0.5, "beta": 0.3}, allora la chiamata a WaveletSRLoss(**cfg["loss"]) equivale a WaveletSRLoss(alpha=0.5, beta=0.3).
    optimizer = Adam(model.parameters(), lr=train_cfg["lr"], betas=tuple(train_cfg["betas"]), weight_decay=train_cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, cfg)
    scaler = make_grad_scaler(amp, device)

    lpips_metric = None
    if cfg["metrics"]["lpips"].get("enabled", False): #con get se enabled non è presente viene automaticamente settato con il secondo argomento "False"
        try:
            lpips_metric = LPIPSMetric(net=cfg["metrics"]["lpips"].get("net", "alex"), device=device)
            print("[train] LPIPS metric enabled.")
        except Exception as exc:
            print(f"[train] LPIPS unavailable ({exc}); skipping.")

    #ARCFACE
    arcface_metric = None
    identity_criterion = None
    arcface_cfg = cfg["metrics"].get("arcface", {})
    arcface_dl = arcface_cfg.get("download_dir", "weights")
    if arcface_cfg.get("enabled", False):
        try:
            metric_embedder = build_arcface(
                weights_path=arcface_cfg.get("weights"), arch=arcface_cfg.get("arch", "r100"),
                device=device, download_dir=arcface_dl,
            )
            arcface_metric = ArcFaceSimilarity(metric_embedder, device=device)
            print(f"[train] ArcFace identity metric enabled (arch={arcface_cfg.get('arch', 'r100')}).")
        except Exception as exc:
            print(f"[train] ArcFace metric unavailable ({exc}); skipping ID metric.")
    if identity_weight > 0.0:
        try:
            loss_embedder = build_arcface(arch=identity_arch, device=device, download_dir=arcface_dl)
            identity_criterion = IdentityLoss(backbone=loss_embedder, weight=identity_weight).to(device)
            print(f"[train] ArcFace identity loss enabled (arch={identity_arch}, weight={identity_weight}).")
        except Exception as exc:
            print(f"[train] ArcFace identity loss unavailable ({exc}); training without it.")

    # --- resume ------------------------------------------------------------
    start_epoch, global_step = 1, 0
    best_metric_name = cfg["train"]["early_stopping"]["metric"] #recupera la metrica da monitorare per l'early stopping
    tracker = BestTracker(mode="max", min_delta=cfg["train"]["early_stopping"]["min_delta"],
                          patience=cfg["train"]["early_stopping"]["patience"] if cfg["train"]["early_stopping"]["enabled"] else 0) #patience sarebbe per quante epoche non deve migliorare la metrica prima di fermare l'addestramento. Se enabled è False, viene impostato a 0, quindi non ci sarà early stopping.
    #controlla se è stato passato un checkpoint per l'addestramento
    if args.resume:
        meta = load_checkpoint(args.resume, model=model, device=device, optimizer=optimizer, scaler=scaler, scheduler=scheduler, strict=False)
        start_epoch = meta["epoch"] + 1
        global_step = meta["global_step"]
        tracker.best = meta["best_metric"]
        print(f"[train] Resumed from {args.resume} at epoch {meta['epoch']} (best {best_metric_name}={tracker.best:.3f}).")

    epochs = train_cfg["epochs"]
    log_every = train_cfg["log_every"]
    max_steps = train_cfg.get("max_steps") #recupera il num massimo di batch (step) da processare in un epoca
    steps_per_epoch = len(train_loader) if max_steps is None else min(len(train_loader), max_steps)#numero di batch per epoca, len(train_loader) restituisce il numero totale di batch nel dataloader

    # --- run plan + environment -------------------------------------------
    print(banner("RUN PLAN"))
    print(f"  config            : {args.config}")
    print(f"  output_dir        : {out_dir}")
    print(f"  scale / levels    : {scale}x / {levels}  (LR {data_cfg['lr_size']} -> HR {data_cfg['hr_size']})")
    print(f"  wavelet output    : {model_output_channels(model)} ch  ({data_cfg['lr_size']}x{data_cfg['lr_size']})")
    print(f"  train / val imgs  : {len(train_list)} / {len(val_list)}")
    print(f"  batch_size        : {train_cfg['batch_size']}  -> {steps_per_epoch} steps/epoch")
    print(f"  epochs            : {epochs}  (start {start_epoch})  total steps ~ {steps_per_epoch * (epochs - start_epoch + 1)}")
    print(f"  cache_mode        : {cache_mode}  | num_workers {num_workers}")
    print(f"  AMP / channels_last: {amp and device.type=='cuda'} / {channels_last and device.type=='cuda'}")
    print(f"  scheduler         : {cfg['train']['scheduler']['type']}  | early_stop {tracker.patience or 'off'}")
    print(f"  model params      : {sum(p.numel() for p in model.parameters())/1e6:.1f} M")
    print_environment(device, amp=amp, channels_last=channels_last)

    csv_logger = CSVLogger(out_dir / "metrics.csv")

    # --- bicubic baseline (computed once; the bar to beat) -----------------
    print("[train] Computing bicubic baseline on validation split ...")
    baseline = bicubic_baseline(loader=val_loader, device=device, max_batches=train_cfg.get("max_val_batches")) #baseline è un dizionario che contiene i valori di PSNR e SSIM della baseline bicubica calcolata sul set di validazione. Questi valori vengono utilizzati come riferimento per valutare le prestazioni del modello durante l'addestramento. In particolare, il PSNR (Peak Signal-to-Noise Ratio) misura la qualità dell'immagine ricostruita rispetto all'immagine originale, mentre l'SSIM (Structural Similarity Index) valuta la similarità strutturale tra le due immagini. Un valore più alto di PSNR e SSIM indica una migliore qualità dell'immagine ricostruita.
    print(f"[train] Bicubic baseline: PSNR={baseline['bicubic_psnr']:.3f} dB  SSIM={baseline['bicubic_ssim']:.4f}")

    # --- training loop -----------------------------------------------------
    print(banner("TRAINING"))
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        data_meter, step_meter = AverageMeter(), AverageMeter()#dizionario che tiene traccia delle perdite medie per ogni tipo di perdita (total, low, high, texture, image, perceptual) durante l'epoca corrente. Ogni chiave del dizionario è associata a un oggetto AverageMeter che calcola la media delle perdite accumulate.
        loss_meters: dict[str, AverageMeter] = {k: AverageMeter() for k in ("total", "low", "high", "texture", "image", "identity")}
        throughput = Throughput() #velocità di elaborazione
        epoch_start = time.perf_counter()
        end = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader):
            if max_steps is not None and batch_idx >= max_steps:
                break
            data_meter.update(time.perf_counter() - end)

            #sposta i dati sulla gpu, se channels_last è attivo e il device è una GPU, converte i tensori in formato channels_last per ottimizzare le prestazioni
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            if channels_last and device.type == "cuda":
                lr = lr.contiguous(memory_format=torch.channels_last)
                hr = hr.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True) #reset del gradiente
            # Targets computed in fp32 (outside autocast) for numerical stability;
            # the model forward + inverse-Haar reconstruction run under AMP.
            target_wavelets = wavelet_dec(hr.float())
            with autocast(device, amp): #autocast è un contesto che permette di eseguire le operazioni in precisione mista (mixed precision) per migliorare le prestazioni e ridurre l'uso della memoria. Se amp è True, le operazioni all'interno del blocco with verranno eseguite in precisione mista, altrimenti verranno eseguite in precisione completa (float32).
                #pred_wavelets è un tensore che contiene i coefficienti wavelet predetti, è del tipo [batch_size, num_channels, height, width], dove batch_size è il numero di immagini nel batch, num_channels è il numero di canali dei coefficienti wavelet (dipende dal numero di livelli della wavelet), height e width sono le dimensioni spaziali dei coefficienti wavelet. 
                #Per ogni canale iniziale (3 essendo RGB) abbiamo i 4 coefficienti, quindi i canali in uscita sono ad esempio channel_0-> LL Red, channel_1 ->LL Green, channel_2 -> LL blue,
                pred_wavelets = model(lr) 
                pred_image = wavelet_rec(pred_wavelets)
    
                id_loss = identity_criterion(pred_image, hr) if identity_criterion is not None else None #chiama la forward function dell'identity_criterion tramite __call__, che a sua volta chiama la forward function dell'IdentityLoss. La forward function dell'IdentityLoss 
            loss_out = criterion(pred_wavelets.float(), target_wavelets, pred_image.float(), hr.float())
            total_loss = loss_out.total if id_loss is None else loss_out.total + id_loss.float()

            scaler.scale(total_loss).backward() #scale evita che i gradienti diventino troppo piccoli e approssimati a zero
            if train_cfg["grad_clip_norm"] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
            scaler.step(optimizer) #aggiorna i pesi
            scaler.update() #aggiorna il fattore di scaling per la prossima iterazione

            global_step += 1
            throughput.update(lr.size(0)) #calcola il numero di immagini elaborate nel batch corrente e aggiorna il contatore di throughput. lr.size(0) restituisce la dimensione del batch, ovvero il numero di immagini a bassa risoluzione nel batch corrente.
            for key in ("low", "high", "texture", "image"): 
                loss_meters[key].update(getattr(loss_out, key).item()) #aggiorna le loss medie 
            loss_meters["total"].update(total_loss.item())
            if id_loss is not None:
                loss_meters["identity"].update(id_loss.item())
            step_meter.update(time.perf_counter() - end) #aggiorna il tempo impiegato per elaborare il batch corrente, calcolando la differenza tra il tempo corrente e il tempo registrato alla fine del batch precedente. Questo valore viene utilizzato per calcolare la media del tempo di elaborazione per batch durante l'epoca corrente.
            end = time.perf_counter()

            if global_step % log_every == 0: #logging delle informazioni di training
                mem = gpu_memory_mb(device)
                data_frac = 100.0 * data_meter.avg / max(1e-9, step_meter.avg)
                print(
                    f"  e{epoch:03d} s{batch_idx + 1:05d}/{steps_per_epoch:05d} "
                    f"loss={loss_meters['total'].avg:.4f} (low={loss_meters['low'].avg:.4f} "
                    f"high={loss_meters['high'].avg:.4f} tex={loss_meters['texture'].avg:.4f} "
                    f"img={loss_meters['image'].avg:.4f} id={loss_meters['identity'].avg:.4f}) "
                    f"| {throughput.rate():.0f} img/s data={data_frac:.0f}% "
                    f"| vram={mem['allocated']:.0f}/{mem['total']:.0f}MB"
                )

        if scheduler is not None:
            scheduler.step() #aggiorna il learning rate in base alla strategia di scheduling definita nel file di configurazione
        epoch_time = time.perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"] #recupera il learning rate dopo l'aggiornamento dello scheduler

        # --- validation ----------------------------------------------------
        do_val = (epoch % train_cfg["val_every"] == 0) or (epoch == epochs) #check se effettuare validazione
        metrics: dict[str, float] = {}
        if do_val:
            sample_path = (sample_dir / f"epoch_{epoch:04d}.png") if train_cfg["save_samples"] else None
            metrics = evaluate(
                model=model, wavelet_rec=wavelet_rec, loader=val_loader, device=device,
                amp=amp, channels_last=channels_last, sample_path=sample_path,
                num_sample_images=train_cfg["num_sample_images"],
                max_batches=train_cfg.get("max_val_batches"), lpips_metric=lpips_metric,
                arcface_metric=arcface_metric,
            )

        # --- checkpointing -------------------------------------------------
        is_best = False
        if do_val:
            is_best = tracker.update(metrics[best_metric_name], epoch)
        save_checkpoint(ckpt_dir / "last.pth", model=model, optimizer=optimizer, scaler=scaler, scheduler=scheduler,
                        epoch=epoch, global_step=global_step, best_metric=tracker.best, best_metric_name=best_metric_name, config=cfg)
        if is_best:
            save_checkpoint(ckpt_dir / "best.pth", model=model, optimizer=optimizer, scaler=scaler, scheduler=scheduler,
                            epoch=epoch, global_step=global_step, best_metric=tracker.best, best_metric_name=best_metric_name, config=cfg)
        if epoch % train_cfg["save_every"] == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch:04d}.pth", model=model, optimizer=optimizer, scaler=scaler,
                            scheduler=scheduler, epoch=epoch, global_step=global_step, best_metric=tracker.best,
                            best_metric_name=best_metric_name, config=cfg)

        # --- epoch summary + CSV ------------------------------------------
        mem = gpu_memory_mb(device)
        summary = (
            f"[epoch {epoch:03d}/{epochs}] time={epoch_time:.1f}s "
            f"data={100*data_meter.avg/max(1e-9,step_meter.avg):.0f}% "
            f"img/s={throughput.rate():.0f} loss={loss_meters['total'].avg:.4f} lr={current_lr:.2e}"
        )
        if metrics:
            delta = metrics["psnr"] - baseline["bicubic_psnr"]
            summary += (
                f" | PSNR={metrics['psnr']:.3f}({delta:+.2f} vs bicubic) "
                f"SSIM={metrics['ssim']:.4f}"
            )
            if "id_sim" in metrics:
                summary += f" ID={metrics['id_sim']:.4f}"
            summary += " [BEST]" if is_best else ""
        print(summary)

        csv_logger.log({
            "epoch": epoch, "global_step": global_step, "epoch_time_s": epoch_time,
            "data_time_ms": data_meter.avg * 1000, "step_time_ms": step_meter.avg * 1000,
            "img_per_s": throughput.rate(), "lr": current_lr, "batch_size": train_cfg["batch_size"], "amp": amp,
            "loss_total": loss_meters["total"].avg, "loss_low": loss_meters["low"].avg,
            "loss_high": loss_meters["high"].avg, "loss_texture": loss_meters["texture"].avg,
            "loss_image": loss_meters["image"].avg,
            "loss_identity": loss_meters["identity"].avg,
            "val_psnr": metrics.get("psnr"), "val_ssim": metrics.get("ssim"), "val_lpips": metrics.get("lpips"),
            "val_id_sim": metrics.get("id_sim"),
            "bicubic_psnr": baseline["bicubic_psnr"], "bicubic_ssim": baseline["bicubic_ssim"],
            "psnr_gain_vs_bicubic": (metrics.get("psnr") - baseline["bicubic_psnr"]) if metrics else None,
            "is_best": is_best, "vram_mb": mem["allocated"], "ckpt": str(ckpt_dir / "last.pth"),
        })

        if tracker.should_stop:
            print(f"[train] Early stopping: no {best_metric_name} improvement for {tracker.patience} epochs "
                  f"(best={tracker.best:.3f} @ epoch {tracker.best_epoch}).")
            break

    print(banner("DONE"))
    print(f"  best {best_metric_name} = {tracker.best:.4f} @ epoch {tracker.best_epoch}")
    print(f"  checkpoints in {ckpt_dir}  |  metrics in {out_dir / 'metrics.csv'}")


def model_output_channels(model: nn.Module) -> int:
    m = model.module if isinstance(model, nn.DataParallel) else model
    m = getattr(m, "_orig_mod", m)  # unwrap torch.compile
    return getattr(m, "output_channels", -1)


if __name__ == "__main__":
    main()
