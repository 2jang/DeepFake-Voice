#!/usr/bin/env python3
"""Phase 2: 학습 및 평가 스크립트.

사용법:
    python scripts/03_train/train.py --config scripts/03_train/configs/aasist_c1.yaml
    python scripts/03_train/train.py --config scripts/03_train/configs/aasist_c1.yaml --epochs 5
    python scripts/03_train/train.py --all
    python scripts/03_train/train.py --all --epochs 5
    python scripts/03_train/train.py --all --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import AudioDataset
from metrics import compute_all
from models import get_model
from sota import SOTA


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0,
) -> float:
    model.train()
    total_loss = 0.0
    use_amp = device.type == "cuda"
    pbar = tqdm(loader, desc=f"  Train ep{epoch}", leave=False, ncols=80)
    for wav, label in pbar:
        wav = wav.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(wav)
            loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * wav.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_labels, all_scores = [], []
    for wav, label in tqdm(loader, desc="  Eval", leave=False, ncols=80):
        wav = wav.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits = model(wav)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().float().numpy()
        all_scores.extend(probs)
        all_labels.extend(label.numpy())

    labels = np.array(all_labels)
    scores = np.array(all_scores)
    return compute_all(labels, scores)


def measure_inference_time(
    model: nn.Module, device: torch.device, num_samples: int = 64_000, runs: int = 110
) -> float:
    model.eval()
    dummy = torch.randn(1, num_samples, device=device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        for _ in range(10):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        for _ in range(runs - 10):
            start = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times))


def measure_flops(model: nn.Module, device: torch.device, num_samples: int = 64_000) -> int:
    dummy = torch.randn(1, num_samples, device=device)
    from torch.utils.flop_counter import FlopCounterMode
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        model(dummy)
    return flop_counter.get_total_flops()


def measure_peak_memory(model: nn.Module, device: torch.device, num_samples: int = 64_000) -> float:
    if device.type != "cuda":
        return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    dummy = torch.randn(1, num_samples, device=device)
    with torch.no_grad():
        model(dummy)
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

ALL_CONFIGS = sorted(CONFIGS_DIR.glob("*.yaml"))


def run_experiment(config_path: str, epochs_override: int | None, device_str: str, resume: bool = False) -> int:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    max_epochs = epochs_override or cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping_patience"]
    num_samples = cfg["data"]["num_samples"]
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["lr"]
    num_workers = cfg["data"].get("num_workers", 8)
    experiment_name = cfg["experiment"]

    # --- output dirs ---
    out_root = Path(cfg.get("output_root", "experiments"))
    ckpt_dir = out_root / "checkpoints"
    log_dir = out_root / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    tb_dir = out_root / "tb_logs" / experiment_name
    writer = SummaryWriter(log_dir=str(tb_dir))

    # --- data ---
    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Model: {cfg['model']['name']}, Epochs: {max_epochs}, LR: {lr}")
    print(f"{'='*60}")

    train_ds = AudioDataset(cfg["data"]["train_manifest"], num_samples, is_train=True, seed=seed)
    dev_ds = AudioDataset(cfg["data"]["dev_manifest"], num_samples, is_train=False, seed=seed)

    persistent = num_workers > 0
    prefetch = 4 if num_workers > 0 else None
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=persistent, prefetch_factor=prefetch,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent, prefetch_factor=prefetch,
    )

    print(f"  Train: {len(train_ds):,} samples")
    print(f"  Dev:   {len(dev_ds):,} samples")

    # --- model ---
    model = get_model(cfg["model"]["name"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")

    if device.type == "cuda":
        model = torch.compile(model)
        print(f"  torch.compile enabled (bf16 + tf32)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg["training"].get("weight_decay", 1e-4))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg["training"].get("class_weights", [1.0, 1.0]), device=device)
    )

    # --- training loop ---
    best_eer = 1.0
    best_epoch = -1
    wait = 0
    start_epoch = 1
    resume_path = ckpt_dir / f"{experiment_name}_resume.pt"
    log_path = log_dir / f"{experiment_name}.jsonl"

    if resume and resume_path.exists():
        ckpt = torch.load(resume_path, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_eer = ckpt["best_eer"]
        best_epoch = ckpt["best_epoch"]
        wait = ckpt["wait"]
        print(f"  Resumed from epoch {ckpt['epoch']} (best EER: {best_eer:.4f})")

    print(f"\n  {'Epoch':>5}  {'Loss':>10}  {'EER':>8}  {'Acc':>8}  {'F1':>8}  {'AUC':>8}  {'Best':>5}")
    print(f"  {'-'*61}")

    for epoch in range(start_epoch, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        dev_metrics = evaluate(model, dev_loader, device)
        dev_eer = dev_metrics["eer"]

        is_best = dev_eer < best_eer
        if is_best:
            best_eer = dev_eer
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), ckpt_dir / f"{experiment_name}_best.pt")
        else:
            wait += 1

        mark = "*" if is_best else ""
        print(f"  {epoch:5d}  {train_loss:10.4f}  {dev_eer:8.4f}  {dev_metrics['accuracy']:8.4f}  {dev_metrics['f1']:8.4f}  {dev_metrics['auc']:8.4f}  {mark:>5}")

        with open(log_path, "a") as f:
            json.dump({"epoch": epoch, "train_loss": train_loss, **dev_metrics}, f)
            f.write("\n")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Metrics/eer", dev_eer, epoch)
        writer.add_scalar("Metrics/accuracy", dev_metrics["accuracy"], epoch)
        writer.add_scalar("Metrics/f1", dev_metrics["f1"], epoch)
        writer.add_scalar("Metrics/precision", dev_metrics["precision"], epoch)
        writer.add_scalar("Metrics/recall", dev_metrics["recall"], epoch)
        writer.add_scalar("Metrics/auc", dev_metrics["auc"], epoch)
        writer.flush()

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_eer": best_eer,
            "best_epoch": best_epoch,
            "wait": wait,
        }, resume_path)

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    # --- final evaluation on T1/T2/T3 ---
    print(f"\n  Best dev EER: {best_eer:.4f} (epoch {best_epoch})")
    print(f"\n  ── Evaluation on test sets ──")

    model.load_state_dict(torch.load(ckpt_dir / f"{experiment_name}_best.pt", weights_only=True))

    results_row = {
        "experiment": experiment_name,
        "model": cfg["model"]["name"],
        "data_combo": cfg.get("data_combo", ""),
        "best_epoch": best_epoch,
        "best_dev_eer": f"{best_eer:.6f}",
        "params": n_params,
    }

    for eval_name, eval_path in cfg["data"]["eval_manifests"].items():
        if not Path(eval_path).exists():
            print(f"    {eval_name}: manifest not found ({eval_path})")
            continue
        eval_ds = AudioDataset(eval_path, num_samples, is_train=False, seed=seed)
        eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        m = evaluate(model, eval_loader, device)
        print(f"    {eval_name:5s}  EER={m['eer']:.4f}  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  Prec={m['precision']:.4f}  Rec={m['recall']:.4f}  AUC={m['auc']:.4f}")
        for metric_name, metric_val in m.items():
            results_row[f"{eval_name}_{metric_name}"] = f"{metric_val:.6f}"
        writer.add_scalar(f"Eval_{eval_name}/eer", m["eer"], 0)
        writer.add_scalar(f"Eval_{eval_name}/accuracy", m["accuracy"], 0)
        writer.add_scalar(f"Eval_{eval_name}/f1", m["f1"], 0)
        writer.add_scalar(f"Eval_{eval_name}/auc", m["auc"], 0)

    # --- efficiency metrics ---
    inf_ms = measure_inference_time(model, device, num_samples)
    throughput = 1000.0 / inf_ms if inf_ms > 0 else 0.0
    model_mb = n_params * 4 / (1024 * 1024)
    flops_model = get_model(cfg["model"]["name"]).to(device)
    flops = measure_flops(flops_model, device, num_samples)
    del flops_model
    peak_mem_mb = measure_peak_memory(model, device, num_samples)

    results_row["inference_ms"] = f"{inf_ms:.2f}"
    results_row["throughput_sps"] = f"{throughput:.1f}"
    results_row["model_mb"] = f"{model_mb:.2f}"
    results_row["flops_m"] = f"{flops / 1e6:.1f}"
    results_row["peak_mem_mb"] = f"{peak_mem_mb:.1f}"

    print(f"\n  ── Efficiency ──")
    print(f"    Inference:  {inf_ms:.2f} ms/sample")
    print(f"    Throughput: {throughput:.1f} samples/sec")
    print(f"    Model Size: {model_mb:.2f} MB ({n_params:,} params)")
    print(f"    FLOPs:      {flops / 1e6:.1f} M")
    if peak_mem_mb > 0:
        print(f"    Peak Memory:{peak_mem_mb:.1f} MB")

    writer.add_scalar("Efficiency/inference_ms", inf_ms, 0)
    writer.add_scalar("Efficiency/throughput_sps", throughput, 0)
    writer.add_scalar("Efficiency/model_mb", model_mb, 0)
    writer.add_scalar("Efficiency/flops_m", flops / 1e6, 0)
    if peak_mem_mb > 0:
        writer.add_scalar("Efficiency/peak_mem_mb", peak_mem_mb, 0)
    writer.close()

    # --- append to results CSV ---
    results_csv = out_root / "results.csv"
    file_exists = results_csv.exists()
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results_row)

    print(f"\n  Results appended to {results_csv}")

    # --- SOTA comparison ---
    model_name = cfg["model"]["name"].lower()
    if model_name in SOTA:
        ref = SOTA[model_name]
        print(f"\n  ── SOTA 비교 (참고: {ref['dataset']}) ──")
        print(f"  {'':15s} {'SOTA':>8s}  {'Dev':>8s}", end="")
        eval_names = list(cfg["data"]["eval_manifests"].keys())
        for en in eval_names:
            print(f"  {en:>8s}", end="")
        print()
        print(f"  {'EER (%)':15s} {ref['eer']:8.2f}", end="")
        print(f"  {best_eer * 100:8.2f}", end="")
        for en in eval_names:
            key = f"{en}_eer"
            if key in results_row:
                print(f"  {float(results_row[key]) * 100:8.2f}", end="")
            else:
                print(f"  {'N/A':>8s}", end="")
        print()
        print(f"  {'Paper':15s} {ref['paper']}")

    print(f"{'='*60}\n")
    resume_path.unlink(missing_ok=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Anti-spoofing training")
    parser.add_argument("--config", help="YAML config path")
    parser.add_argument("--all", action="store_true", help="Run all configs in configs/")
    parser.add_argument("--epochs", type=int, help="Override max epochs")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    parser.add_argument("--tb-port", type=int, default=6006, help="TensorBoard port (0 to disable)")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted training")
    args = parser.parse_args()

    if args.tb_port > 0 and (args.all or args.config):
        tb_log_dir = Path("experiments") / "tb_logs"
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main",
             "--logdir", str(tb_log_dir),
             "--host", "0.0.0.0",
             "--port", str(args.tb_port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"TensorBoard: http://0.0.0.0:{args.tb_port}")

    if args.all:
        print(f"Running all {len(ALL_CONFIGS)} experiments sequentially...")
        for i, cfg_path in enumerate(ALL_CONFIGS, 1):
            if args.resume:
                with open(cfg_path) as f:
                    _cfg = yaml.safe_load(f)
                _out = Path(_cfg.get("output_root", "experiments"))
                _name = _cfg["experiment"]
                _resume_exists = (_out / "checkpoints" / f"{_name}_resume.pt").exists()
                _best_exists = (_out / "checkpoints" / f"{_name}_best.pt").exists()
                if _best_exists and not _resume_exists:
                    print(f"\n  [{i}/{len(ALL_CONFIGS)}] {cfg_path.stem} — skipped (completed)")
                    continue
            print(f"\n{'#'*60}")
            print(f"  [{i}/{len(ALL_CONFIGS)}] {cfg_path.stem}")
            print(f"{'#'*60}")
            run_experiment(str(cfg_path), args.epochs, args.device, args.resume)
        print(f"\nAll {len(ALL_CONFIGS)} experiments complete.")
        return 0
    elif args.config:
        return run_experiment(args.config, args.epochs, args.device, args.resume)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
