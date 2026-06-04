#!/usr/bin/env python3
"""실험 결과 분석: results.csv를 읽어 SOTA 대비 비교 테이블 출력.

사용법:
    python scripts/04_analyze/analyze.py
    python scripts/04_analyze/analyze.py --results experiments/results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# 03_train의 공유 모듈 (sota.py) 참조
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_train"))
from sota import SOTA


def print_comparison(df: pd.DataFrame) -> None:
    eer_cols = [c for c in df.columns if c.endswith("_eer")]
    eval_names = [c.replace("_eer", "") for c in eer_cols]

    header = f"  {'Experiment':20s} {'Model':10s} {'Combo':6s} {'SOTA':>7s} {'Dev':>7s}"
    for en in eval_names:
        header += f" {en:>7s}"
    header += f"  {'Best':>5s}"

    print(f"\n{'='*len(header)}")
    print("  EER 비교 (%, SOTA: ASVspoof 2019 LA)")
    print(f"{'='*len(header)}")
    print(header)
    print(f"  {'-'*(len(header)-2)}")

    best_exp = None
    best_dev_eer = 1.0

    for _, row in df.iterrows():
        model = row["model"].lower()
        sota_eer = SOTA[model]["eer"] if model in SOTA else None
        dev_eer = float(row["best_dev_eer"]) * 100

        line = f"  {row['experiment']:20s} {row['model']:10s} {row.get('data_combo', ''):6s}"
        line += f" {sota_eer:7.2f}" if sota_eer else f" {'N/A':>7s}"
        line += f" {dev_eer:7.2f}"

        for col in eer_cols:
            if col in row and pd.notna(row[col]):
                line += f" {float(row[col]) * 100:7.2f}"
            else:
                line += f" {'N/A':>7s}"

        if dev_eer < best_dev_eer:
            best_dev_eer = dev_eer
            best_exp = row["experiment"]

        print(line)

    print(f"  {'-'*(len(header)-2)}")
    if best_exp:
        print(f"  Best: {best_exp} (Dev EER {best_dev_eer:.2f}%)")
    print()


def print_delta(df: pd.DataFrame) -> None:
    eer_cols = [c for c in df.columns if c.endswith("_eer")]
    eval_names = [c.replace("_eer", "") for c in eer_cols]

    header = f"  {'Experiment':20s} {'Model':10s} {'Dev-SOTA':>9s}"
    for en in eval_names:
        header += f" {en+'-SOTA':>9s}"

    print(f"  SOTA 대비 차이 (pp, 음수 = SOTA보다 우수)")
    print(f"  {'-'*(len(header)-2)}")
    print(header)
    print(f"  {'-'*(len(header)-2)}")

    for _, row in df.iterrows():
        model = row["model"].lower()
        if model not in SOTA:
            continue
        sota_eer = SOTA[model]["eer"]
        dev_delta = float(row["best_dev_eer"]) * 100 - sota_eer

        line = f"  {row['experiment']:20s} {row['model']:10s} {dev_delta:+9.2f}"
        for col in eer_cols:
            if col in row and pd.notna(row[col]):
                delta = float(row[col]) * 100 - sota_eer
                line += f" {delta:+9.2f}"
            else:
                line += f" {'N/A':>9s}"
        print(line)

    print()


def print_sota_ref() -> None:
    print(f"  SOTA 레퍼런스 (ASVspoof 2019 LA)")
    print(f"  {'-'*55}")
    for name, ref in SOTA.items():
        print(f"  {name:10s}  EER {ref['eer']:.2f}%  {ref['paper']}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze experiment results vs SOTA")
    parser.add_argument("--results", default="experiments/results.csv", help="Path to results.csv")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"  results.csv not found: {results_path}")
        print(f"  Run training first: python scripts/03_train/train.py --all")
        return 1

    df = pd.read_csv(results_path)
    print(f"\n  Loaded {len(df)} experiments from {results_path}")

    print_sota_ref()
    print_comparison(df)
    print_delta(df)

    return 0


if __name__ == "__main__":
    sys.exit(main())
