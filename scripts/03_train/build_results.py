#!/usr/bin/env python3
"""저장된 체크포인트를 평가하여 results.csv를 생성.

train.py와 별도로 실행하여, 학습 완료된 체크포인트들을
테스트셋에서 평가하고 효율성 지표를 측정합니다.

사용법:
    python scripts/03_train/build_results.py --config scripts/03_train/configs/aasist_c1.yaml
    python scripts/03_train/build_results.py --all
    python scripts/03_train/build_results.py --config ... --checkpoint experiments/checkpoints/aasist_c1_ep010.pt
    python scripts/03_train/build_results.py --all --output experiments/results_v2.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import AudioDataset
from metrics import compute_all
from models import get_model
from train import evaluate, load_checkpoint, measure_inference_time, measure_flops, measure_peak_memory


CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
ALL_CONFIGS = sorted(CONFIGS_DIR.glob("*.yaml"))


def evaluate_experiment(
    config_path: str,
    device_str: str,
    checkpoint_path: str | None = None,
) -> dict[str, str | int | float]:
    """단일 실험의 체크포인트를 평가하고 결과 dict를 반환."""

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    experiment_name = cfg["experiment"]
    num_samples = cfg["data"]["num_samples"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["data"].get("num_workers", 8)
    out_root = Path(cfg.get("output_root", "experiments"))
    ckpt_dir = out_root / "checkpoints"

    # --- checkpoint 결정 ---
    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
    else:
        ckpt_path = ckpt_dir / f"{experiment_name}_best.pt"

    if not ckpt_path.exists():
        print(f"  Checkpoint not found: {ckpt_path}")
        return {}

    ckpt_label = ckpt_path.stem.replace(f"{experiment_name}_", "")

    print(f"\n{'='*60}")
    print(f"  Experiment:  {experiment_name}")
    print(f"  Checkpoint:  {ckpt_path.name} ({ckpt_label})")
    print(f"  Device:      {device}")
    print(f"{'='*60}")

    # --- model ---
    model = get_model(cfg["model"]["name"]).to(device)
    load_checkpoint(model, ckpt_path)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")

    if device.type == "cuda":
        model = torch.compile(model)

    # --- evaluate on test sets ---
    print(f"\n  -- Evaluation on test sets --")
    results_row: dict[str, str | int | float] = {
        "experiment": experiment_name,
        "model": cfg["model"]["name"],
        "data_combo": cfg.get("data_combo", ""),
        "checkpoint": ckpt_label,
        "params": n_params,
    }

    for eval_name, eval_path in cfg["data"]["eval_manifests"].items():
        if not Path(eval_path).exists():
            print(f"    {eval_name}: manifest not found ({eval_path})")
            continue
        eval_ds = AudioDataset(eval_path, num_samples, is_train=False, seed=42)
        eval_loader = DataLoader(
            eval_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        m = evaluate(model, eval_loader, device)
        print(f"    {eval_name:5s}  EER={m['eer']:.4f}  minDCF={m['mindcf']:.4f}  "
              f"Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}  "
              f"FPR={m['fpr']:.4f}  FNR={m['fnr']:.4f}  "
              f"FPR@EER={m['fpr_eer']:.4f}  FNR@EER={m['fnr_eer']:.4f}")
        for metric_name, metric_val in m.items():
            results_row[f"{eval_name}_{metric_name}"] = f"{metric_val:.6f}"

    # --- efficiency ---
    print(f"\n  -- Efficiency --")
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

    print(f"    Inference:  {inf_ms:.2f} ms/sample")
    print(f"    Throughput: {throughput:.1f} samples/sec")
    print(f"    Model Size: {model_mb:.2f} MB ({n_params:,} params)")
    print(f"    FLOPs:      {flops / 1e6:.1f} M")
    if peak_mem_mb > 0:
        print(f"    Peak Memory:{peak_mem_mb:.1f} MB")

    print(f"{'='*60}\n")
    return results_row


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints and build results.csv")
    parser.add_argument("--config", help="YAML config path")
    parser.add_argument("--all", action="store_true", help="Evaluate all configs")
    parser.add_argument("--checkpoint", help="Specific checkpoint path (default: best)")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    parser.add_argument("--output", default="experiments/results_v2.csv",
                        help="Output CSV path (default: experiments/results_v2.csv). "
                             "train.py의 results.csv(학습 로그)를 덮어쓰지 않도록 분리.")
    args = parser.parse_args()

    if not args.all and not args.config:
        parser.print_help()
        return 1

    configs = ALL_CONFIGS if args.all else [Path(args.config)]
    rows: list[dict] = []

    for i, cfg_path in enumerate(configs, 1):
        if args.all:
            print(f"\n{'#'*60}")
            print(f"  [{i}/{len(configs)}] {cfg_path.stem}")
            print(f"{'#'*60}")

        try:
            row = evaluate_experiment(str(cfg_path), args.device, args.checkpoint)
            if row:
                rows.append(row)
        except Exception as e:
            print(f"  ERROR: {cfg_path.stem} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not rows:
        print("No results to write.")
        return 1

    # --- write CSV: 모든 row의 키를 합집합으로 수집 ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Results written to {output_path} ({len(rows)} experiments)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
