#!/usr/bin/env python3
"""실험 결과 분석: results.csv를 읽어 SOTA 대비 비교 테이블 출력 및 CSV/텍스트 저장.

사용법:
    python scripts/04_analyze/analyze.py --results experiments/results_v2.csv
    python scripts/04_analyze/analyze.py --results experiments/results_v2.csv --outdir experiments/analysis
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import pandas as pd

# 03_train의 공유 모듈 (sota.py) 참조
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_train"))
from sota import SOTA

METRICS = ["eer", "mindcf", "accuracy", "f1", "precision", "recall", "auc",
           "fpr", "fnr", "fpr_eer", "fnr_eer"]


def _eer_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c.endswith("_eer") and "_fpr_eer" not in c and "_fnr_eer" not in c]


def _eval_names(df: pd.DataFrame) -> list[str]:
    return [c.replace("_eer", "") for c in _eer_cols(df)]


# ---------------------------------------------------------------------------
# 콘솔 출력 함수 (텍스트 리포트용)
# ---------------------------------------------------------------------------

def print_comparison(df: pd.DataFrame) -> None:
    eer_cols = _eer_cols(df)
    eval_names = [c.replace("_eer", "") for c in eer_cols]
    has_dev = "best_dev_eer" in df.columns

    header = f"  {'Experiment':20s} {'Model':10s} {'Combo':6s} {'SOTA':>7s}"
    if has_dev:
        header += f" {'Dev':>7s}"
    for en in eval_names:
        header += f" {en:>7s}"
    header += f"  {'Best':>5s}"

    print(f"\n{'='*len(header)}")
    print("  EER 비교 (%, SOTA: ASVspoof 2019 LA)")
    print(f"{'='*len(header)}")
    print(header)
    print(f"  {'-'*(len(header)-2)}")

    best_exp = None
    best_eer_val = 1.0

    for _, row in df.iterrows():
        model = row["model"].lower()
        sota_eer = SOTA[model]["eer"] if model in SOTA else None

        dev_eer = None
        if has_dev and pd.notna(row.get("best_dev_eer")) and row["best_dev_eer"] != "":
            dev_eer = float(row["best_dev_eer"]) * 100

        line = f"  {row['experiment']:20s} {row['model']:10s} {row.get('data_combo', ''):6s}"
        line += f" {sota_eer:7.2f}" if sota_eer else f" {'N/A':>7s}"
        if has_dev:
            line += f" {dev_eer:7.2f}" if dev_eer is not None else f" {'N/A':>7s}"

        for col in eer_cols:
            if col in row and pd.notna(row[col]):
                line += f" {float(row[col]) * 100:7.2f}"
            else:
                line += f" {'N/A':>7s}"

        compare_eer = dev_eer if dev_eer is not None else (
            float(row[eer_cols[0]]) * 100 if eer_cols and pd.notna(row.get(eer_cols[0])) else None)
        if compare_eer is not None and compare_eer < best_eer_val:
            best_eer_val = compare_eer
            best_exp = row["experiment"]

        print(line)

    print(f"  {'-'*(len(header)-2)}")
    if best_exp:
        label = "Dev EER" if has_dev else "T1 EER"
        print(f"  Best: {best_exp} ({label} {best_eer_val:.2f}%)")
    print()


def print_delta(df: pd.DataFrame) -> None:
    eer_cols = _eer_cols(df)
    eval_names = [c.replace("_eer", "") for c in eer_cols]
    has_dev = "best_dev_eer" in df.columns

    header = f"  {'Experiment':20s} {'Model':10s}"
    if has_dev:
        header += f" {'Dev-SOTA':>9s}"
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

        line = f"  {row['experiment']:20s} {row['model']:10s}"
        if has_dev:
            if pd.notna(row.get("best_dev_eer")) and row["best_dev_eer"] != "":
                dev_delta = float(row["best_dev_eer"]) * 100 - sota_eer
                line += f" {dev_delta:+9.2f}"
            else:
                line += f" {'N/A':>9s}"

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


# ---------------------------------------------------------------------------
# CSV 출력 함수
# ---------------------------------------------------------------------------

def save_eer_comparison_csv(df: pd.DataFrame, path: Path) -> None:
    """EER 비교 테이블 CSV (%, SOTA 포함)."""
    eer_cols = _eer_cols(df)
    eval_names = [c.replace("_eer", "") for c in eer_cols]
    has_dev = "best_dev_eer" in df.columns

    rows = []
    for _, row in df.iterrows():
        model = row["model"].lower()
        r: dict = {
            "experiment": row["experiment"],
            "model": row["model"],
            "data_combo": row.get("data_combo", ""),
            "sota_eer_pct": SOTA[model]["eer"] if model in SOTA else "",
        }
        if has_dev and pd.notna(row.get("best_dev_eer")) and row["best_dev_eer"] != "":
            r["dev_eer_pct"] = f"{float(row['best_dev_eer']) * 100:.4f}"
        else:
            r["dev_eer_pct"] = ""

        for en, col in zip(eval_names, eer_cols):
            if col in row and pd.notna(row[col]):
                r[f"{en}_eer_pct"] = f"{float(row[col]) * 100:.4f}"
            else:
                r[f"{en}_eer_pct"] = ""
        rows.append(r)

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_all_metrics_csv(df: pd.DataFrame, path: Path) -> None:
    """전체 지표 CSV — 실험 × 평가셋 × 지표 (long format)."""
    eval_names = _eval_names(df)
    has_dev = "best_dev_eer" in df.columns

    rows = []
    for _, row in df.iterrows():
        for en in eval_names:
            r: dict = {
                "experiment": row["experiment"],
                "model": row["model"],
                "data_combo": row.get("data_combo", ""),
                "eval_set": en,
            }
            if has_dev and pd.notna(row.get("best_dev_eer")) and row["best_dev_eer"] != "":
                r["best_dev_eer"] = f"{float(row['best_dev_eer']):.6f}"
            else:
                r["best_dev_eer"] = ""

            for m in METRICS:
                key = f"{en}_{m}"
                if key in row and pd.notna(row[key]):
                    r[m] = f"{float(row[key]):.6f}"
                else:
                    r[m] = ""
            rows.append(r)

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_sota_delta_csv(df: pd.DataFrame, path: Path) -> None:
    """SOTA 대비 EER 차이 CSV (pp 단위)."""
    eer_cols = _eer_cols(df)
    eval_names = [c.replace("_eer", "") for c in eer_cols]
    has_dev = "best_dev_eer" in df.columns

    rows = []
    for _, row in df.iterrows():
        model = row["model"].lower()
        if model not in SOTA:
            continue
        sota_eer = SOTA[model]["eer"]

        r: dict = {
            "experiment": row["experiment"],
            "model": row["model"],
            "data_combo": row.get("data_combo", ""),
            "sota_eer_pct": sota_eer,
        }
        if has_dev and pd.notna(row.get("best_dev_eer")) and row["best_dev_eer"] != "":
            r["dev_delta_pp"] = f"{float(row['best_dev_eer']) * 100 - sota_eer:+.4f}"
        else:
            r["dev_delta_pp"] = ""

        for en, col in zip(eval_names, eer_cols):
            if col in row and pd.notna(row[col]):
                r[f"{en}_delta_pp"] = f"{float(row[col]) * 100 - sota_eer:+.4f}"
            else:
                r[f"{en}_delta_pp"] = ""
        rows.append(r)

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_efficiency_csv(df: pd.DataFrame, path: Path) -> None:
    """효율성 지표 CSV."""
    eff_cols = ["inference_ms", "throughput_sps", "model_mb", "flops_m", "peak_mem_mb", "params"]
    present = [c for c in eff_cols if c in df.columns]
    if not present:
        return

    rows = []
    for _, row in df.iterrows():
        r: dict = {
            "experiment": row["experiment"],
            "model": row["model"],
            "data_combo": row.get("data_combo", ""),
        }
        for c in present:
            r[c] = row[c] if pd.notna(row.get(c)) else ""
        rows.append(r)

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze experiment results vs SOTA")
    parser.add_argument("--results", default="experiments/results.csv",
                        help="Path to results CSV")
    parser.add_argument("--outdir", default="experiments/analysis",
                        help="분석 결과 출력 디렉터리 (기본: experiments/analysis)")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"  results.csv not found: {results_path}")
        print(f"  Run training first: python scripts/03_train/train.py --all")
        return 1

    df = pd.read_csv(results_path)

    # --- 콘솔 + 텍스트 리포트 ---
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf

    print(f"\n  Loaded {len(df)} experiments from {results_path}")
    print_sota_ref()
    print_comparison(df)
    print_delta(df)

    sys.stdout = old_stdout
    text = buf.getvalue()
    print(text, end="")

    # --- 파일 출력 ---
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    report_path = outdir / "analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)

    save_eer_comparison_csv(df, outdir / "eer_comparison.csv")
    save_sota_delta_csv(df, outdir / "sota_delta.csv")
    save_all_metrics_csv(df, outdir / "all_metrics.csv")
    save_efficiency_csv(df, outdir / "efficiency.csv")

    print(f"  출력 파일:")
    for p in sorted(outdir.glob("*")):
        print(f"    {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
