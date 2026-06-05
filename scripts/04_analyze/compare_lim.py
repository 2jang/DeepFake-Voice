#!/usr/bin/env python3
"""임동훈(2024) 직접 비교표 생성.

임동훈, 한종기 (2024) "LCNN 기반 Deepfake Audio Detection의 개선 알고리즘"의
Table 4 (KoAAD EER/Accuracy/F1) 및 Table 5 (KoAAD FPR)와
본 연구의 C_all 결과를 대조한다.

사용법:
    python scripts/04_analyze/compare_lim.py
    python scripts/04_analyze/compare_lim.py --results experiments/results_v2.csv
    python scripts/04_analyze/compare_lim.py --output experiments/lim_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# 임동훈(2024) 보고 수치
# ---------------------------------------------------------------------------

# Table 4: KoAAD test (다국어 학습 → 한국어 테스트)
LIM_TABLE4 = {
    "LCNN-LSTM": {
        "eer": 50.85, "accuracy": 52.46, "f1": 42.09,
        "note": "기본 LCNN-LSTM",
    },
    "Whisper-LCNN": {
        "eer": 66.36, "accuracy": 47.42, "f1": 25.58,
        "note": "Whisper feature + LCNN",
    },
    "SpecRNet": {
        "eer": 57.42, "accuracy": 51.28, "f1": 47.07,
        "note": "Kawa(2022) 원 모델",
    },
    "HPF+Mean+Enhance": {
        "eer": 54.55, "accuracy": 52.69, "f1": 42.70,
        "note": "임동훈 제안 3가지 개선 모두 적용",
    },
    "RawNet2": {
        "eer": 47.66, "accuracy": 53.21, "f1": 48.99,
        "note": "가장 낮은 EER (임동훈 기준)",
    },
}

# Table 5: KoAAD FPR (real → fake 오판률, %)
LIM_TABLE5 = {
    "LCNN-LSTM":        {"fpr": 65.36},
    "Whisper-LCNN":     {"fpr": 93.47},
    "SpecRNet":         {"fpr": 68.38},
    "HPF+Mean+Enhance": {"fpr": 78.04},
    "RawNet2":          {"fpr": 72.30},
}

# 임동훈 실험 조건
LIM_CONDITIONS = {
    "학습 데이터": "MLAAD + M-AILABS (5,000개)",
    "한국어 학습": "미포함",
    "Epoch": "최대 10",
    "Batch": "4",
}


def load_our_results(results_path: str) -> dict[str, dict]:
    """본 연구 results.csv에서 C_all 모델의 T2(KoAAD) 결과 추출."""
    df = pd.read_csv(results_path)

    # C_all 실험만 필터
    call_df = df[df["data_combo"] == "C_all"]
    if call_df.empty:
        # experiment 이름으로 필터
        call_df = df[df["experiment"].str.contains("call", na=False)]

    results = {}
    for _, row in call_df.iterrows():
        model = row["model"].lower()
        entry = {"experiment": row["experiment"]}

        # T2 (KoAAD) 지표
        for metric in ["eer", "mindcf", "accuracy", "f1", "fpr", "fnr", "auc"]:
            key = f"T2_{metric}"
            if key in row and pd.notna(row[key]):
                val = float(row[key])
                # EER/FPR/FNR은 0-1 범위를 % 변환
                if metric in ("eer", "fpr", "fnr"):
                    entry[f"T2_{metric}_pct"] = val * 100
                else:
                    entry[f"T2_{metric}"] = val
            else:
                entry[f"T2_{metric}_pct" if metric in ("eer", "fpr", "fnr") else f"T2_{metric}"] = None

        results[model] = entry

    return results


def print_comparison(our: dict[str, dict]) -> None:
    """대조표 출력."""

    print(f"\n{'='*80}")
    print(f"  임동훈(2024) vs 본 연구 — KoAAD(T2) 성능 비교")
    print(f"{'='*80}")

    # 조건 비교
    print(f"\n  [실험 조건]")
    print(f"  {'':25s} {'임동훈(2024)':>20s}  {'본 연구':>20s}")
    print(f"  {'-'*70}")
    print(f"  {'학습 데이터':25s} {'MLAAD+M-AILABS 5K':>20s}  {'C_all 1,067,960':>20s}")
    print(f"  {'한국어 학습 포함':25s} {'X':>20s}  {'O':>20s}")
    print(f"  {'Epoch':25s} {'10':>20s}  {'50 (early stop)':>20s}")

    # EER 비교
    print(f"\n  [EER 비교 (%, 낮을수록 좋음)]")
    print(f"  {'모델':25s} {'임동훈':>10s}  {'본 연구':>10s}  {'개선 배율':>10s}")
    print(f"  {'-'*60}")

    comparisons = [
        ("LCNN-LSTM → LCNN", "LCNN-LSTM", "lcnn"),
        ("SpecRNet → SpecRNet", "SpecRNet", "specrnet"),
        ("RawNet2 → RawNet2", "RawNet2", "rawnet2"),
        ("HPF+Mean+Enhance → LCNN", "HPF+Mean+Enhance", "lcnn"),
    ]

    for label, lim_key, our_key in comparisons:
        lim_eer = LIM_TABLE4[lim_key]["eer"]
        our_entry = our.get(our_key, {})
        our_eer = our_entry.get("T2_eer_pct")

        if our_eer is not None and our_eer > 0:
            ratio = lim_eer / our_eer
            print(f"  {label:25s} {lim_eer:10.2f}  {our_eer:10.2f}  {ratio:10.0f}x")
        else:
            print(f"  {label:25s} {lim_eer:10.2f}  {'N/A':>10s}  {'—':>10s}")

    # AASIST (임동훈에 없는 모델)
    if "aasist" in our:
        our_eer = our["aasist"].get("T2_eer_pct")
        if our_eer is not None:
            print(f"  {'(신규) AASIST':25s} {'—':>10s}  {our_eer:10.2f}  {'—':>10s}")

    # FPR 비교
    print(f"\n  [FPR 비교 — real→fake 오판률 (%, 낮을수록 좋음)]")
    print(f"  {'모델':25s} {'임동훈':>10s}  {'본 연구':>10s}")
    print(f"  {'-'*50}")

    fpr_comparisons = [
        ("LCNN-LSTM → LCNN", "LCNN-LSTM", "lcnn"),
        ("SpecRNet → SpecRNet", "SpecRNet", "specrnet"),
        ("RawNet2 → RawNet2", "RawNet2", "rawnet2"),
    ]

    for label, lim_key, our_key in fpr_comparisons:
        lim_fpr = LIM_TABLE5[lim_key]["fpr"]
        our_entry = our.get(our_key, {})
        our_fpr = our_entry.get("T2_fpr_pct")

        if our_fpr is not None:
            print(f"  {label:25s} {lim_fpr:10.2f}  {our_fpr:10.2f}")
        else:
            print(f"  {label:25s} {lim_fpr:10.2f}  {'N/A':>10s}")

    print(f"\n{'='*80}")


def save_comparison_csv(our: dict[str, dict], output_path: str) -> None:
    """대조표를 CSV로 저장."""
    rows = []

    for lim_key, lim_data in LIM_TABLE4.items():
        row = {
            "source": "임동훈(2024)",
            "model": lim_key,
            "training_data": "MLAAD+M-AILABS (5K)",
            "korean_training": "N",
            "T2_eer_pct": lim_data["eer"],
            "T2_accuracy": lim_data.get("accuracy"),
            "T2_f1": lim_data.get("f1"),
            "T2_fpr_pct": LIM_TABLE5.get(lim_key, {}).get("fpr"),
            "note": lim_data.get("note", ""),
        }
        rows.append(row)

    for model_key, entry in our.items():
        row = {
            "source": "본 연구",
            "model": f"{model_key} (C_all)",
            "training_data": "C_all (1,067,960)",
            "korean_training": "Y",
            "T2_eer_pct": entry.get("T2_eer_pct"),
            "T2_accuracy": entry.get("T2_accuracy"),
            "T2_f1": entry.get("T2_f1"),
            "T2_fpr_pct": entry.get("T2_fpr_pct"),
            "note": entry.get("experiment", ""),
        }
        rows.append(row)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"  비교표 저장: {output} ({len(rows)}개 행)")


def main() -> int:
    parser = argparse.ArgumentParser(description="임동훈(2024) 직접 비교표 생성")
    parser.add_argument("--results", default="experiments/results.csv",
                        help="본 연구 results.csv 경로")
    parser.add_argument("--output", default="experiments/lim_comparison.csv",
                        help="비교표 CSV 출력 경로")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"  results.csv를 찾을 수 없습니다: {results_path}")
        print(f"  먼저 build_results.py를 실행하세요:")
        print(f"    python scripts/03_train/build_results.py --all")
        return 1

    our = load_our_results(str(results_path))
    if not our:
        print("  C_all 실험 결과를 찾을 수 없습니다.")
        return 1

    print_comparison(our)
    save_comparison_csv(our, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
