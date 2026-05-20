#!/usr/bin/env python3
"""Phase 2: 학습 및 평가 스크립트.

사용법:
    python scripts/03_train/train.py --config scripts/03_train/configs/aasist_c1.yaml
    python scripts/03_train/train.py --config scripts/03_train/configs/aasist_c1.yaml --epochs 5  # 시범 학습
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import AudioDataset
from metrics import compute_accuracy, compute_eer
from models import get_model


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
    preds = (scores >= 0.5).astype(int)

    eer = compute_eer(labels, scores)
    acc = compute_accuracy(labels, preds)
    return {"eer": eer, "accuracy": acc}


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Anti-spoofing training")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--epochs", type=int, help="Override max epochs")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    max_epochs = args.epochs or cfg["training"]["epochs"]
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
    log_path = log_dir / f"{experiment_name}.jsonl"

    print(f"\n  {'Epoch':>5}  {'Train Loss':>10}  {'Dev EER':>8}  {'Dev Acc':>8}  {'Best':>5}")
    print(f"  {'-'*45}")

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        dev_metrics = evaluate(model, dev_loader, device)
        dev_eer = dev_metrics["eer"]
        dev_acc = dev_metrics["accuracy"]

        is_best = dev_eer < best_eer
        if is_best:
            best_eer = dev_eer
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), ckpt_dir / f"{experiment_name}_best.pt")
        else:
            wait += 1

        mark = "*" if is_best else ""
        print(f"  {epoch:5d}  {train_loss:10.4f}  {dev_eer:8.4f}  {dev_acc:8.4f}  {mark:>5}")

        with open(log_path, "a") as f:
            json.dump({"epoch": epoch, "train_loss": train_loss, "dev_eer": dev_eer, "dev_acc": dev_acc}, f)
            f.write("\n")

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
        print(f"    {eval_name:5s}  EER={m['eer']:.4f}  Acc={m['accuracy']:.4f}")
        results_row[f"{eval_name}_eer"] = f"{m['eer']:.6f}"
        results_row[f"{eval_name}_acc"] = f"{m['accuracy']:.6f}"

    # --- inference time ---
    inf_ms = measure_inference_time(model, device, num_samples)
    model_mb = n_params * 4 / (1024 * 1024)
    results_row["inference_ms"] = f"{inf_ms:.2f}"
    results_row["model_mb"] = f"{model_mb:.2f}"
    print(f"\n  Inference: {inf_ms:.2f} ms/sample, Model: {model_mb:.2f} MB")

    # --- append to results CSV ---
    results_csv = out_root / "results.csv"
    file_exists = results_csv.exists()
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results_row)

    print(f"\n  Results appended to {results_csv}")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
