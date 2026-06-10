#!/usr/bin/env python3
"""Phase 6: 논문용 그래프 및 테이블 생성.

사용법:
    # 전체 생성
    python scripts/04_analyze/visualize.py --all

    # 개별 그래프
    python scripts/04_analyze/visualize.py --graph G1
    python scripts/04_analyze/visualize.py --graph G2
    python scripts/04_analyze/visualize.py --graph G3
    python scripts/04_analyze/visualize.py --graph G4

    # 테이블만
    python scripts/04_analyze/visualize.py --tables

    # 커스텀 입력
    python scripts/04_analyze/visualize.py --all --results experiments/results_v2.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 04_analyze 내 모듈 (compare_lim.py) + 03_train 공유 모듈 참조
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_train"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 한글 폰트 설정 — 실제 사용 가능 여부를 확인한 뒤 설정
_font_set = False
for _candidate in ["Malgun Gothic", "NanumGothic", "NanumBarunGothic",
                    "AppleGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
    from matplotlib.font_manager import findfont, FontProperties
    _fp = findfont(FontProperties(family=_candidate), fallback_to_default=False)
    if _fp and "DejaVu" not in _fp and "dejavu" not in _fp.lower():
        matplotlib.rcParams["font.family"] = _candidate
        _font_set = True
        break

if not _font_set:
    # 한글 폰트가 전혀 없으면 영문 라벨만 깨짐 없이 표시
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    print("  [WARN] 한글 폰트 미설치 — 그래프 라벨이 영문 fallback 됩니다.")
    print("         설치: sudo apt install fonts-nanum && fc-cache -fv")

matplotlib.rcParams["axes.unicode_minus"] = False

# 논문 스타일
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

FIGURE_DIR = Path("experiments/figures")
TABLE_DIR = Path("experiments/tables")

# 모델별 색상/마커
MODEL_STYLES = {
    "aasist":   {"color": "#1f77b4", "marker": "o", "label": "AASIST (229K)"},
    "specrnet": {"color": "#2ca02c", "marker": "s", "label": "SpecRNet (416K)"},
    "lcnn":     {"color": "#ff7f0e", "marker": "^", "label": "LCNN (1.37M)"},
    "rawnet2":  {"color": "#d62728", "marker": "D", "label": "RawNet2 (18.5M)"},
}

DATA_COMBOS = ["C1", "C2", "C3", "C_all"]
EVAL_SETS = ["T1", "T2", "T3"]


# ═══════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════

def load_results(path: str = "experiments/results.csv") -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        print(f"  [SKIP] {p} 없음")
        return None
    df = pd.read_csv(p)
    return df


def load_noise_results(path: str = "experiments/noise_robustness.csv") -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        print(f"  [SKIP] {p} 없음")
        return None
    return pd.read_csv(p)


# ═══════════════════════════════════════════════════════════════
# G1: 데이터 커버리지별 EER (Grouped Bar Chart)
# ═══════════════════════════════════════════════════════════════

def plot_g1(df: pd.DataFrame, output_dir: Path) -> None:
    """데이터 조합(C1/C2/C3/C_all) × 모델별 EER 비교.

    3개 서브플롯 (T1, T2, T3).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    # Phase 1 모델 (C1/C2/C3 결과가 있는 모델)
    phase1_models = ["aasist", "lcnn", "rawnet2"]
    # C_all에는 specrnet도 포함
    all_models = ["aasist", "specrnet", "lcnn", "rawnet2"]

    for ax_idx, eval_set in enumerate(EVAL_SETS):
        ax = axes[ax_idx]
        eer_col = f"{eval_set}_eer"

        if eer_col not in df.columns:
            ax.set_title(f"{eval_set} (데이터 없음)")
            continue

        combos = DATA_COMBOS
        x = np.arange(len(combos))
        width = 0.18
        n_models = len(all_models)
        offsets = np.linspace(-(n_models - 1) * width / 2, (n_models - 1) * width / 2, n_models)

        for m_idx, model in enumerate(all_models):
            eers = []
            for combo in combos:
                # specrnet은 Phase 1(C1/C2/C3) 미수행 → row.empty → NaN
                row = df[(df["model"] == model) & (df["data_combo"] == combo)]
                eers.append(float(row.iloc[0][eer_col]) * 100 if not row.empty else np.nan)

            style = MODEL_STYLES.get(model, {"color": "gray", "label": model})
            # NaN(미수행 조합)은 막대를 그리지 않아 실제 0.00% EER과 시각적으로 구분된다.
            heights = np.array(eers, dtype=float)
            bars = ax.bar(x + offsets[m_idx], heights,
                          width, label=style["label"], color=style["color"],
                          alpha=0.85, edgecolor="white", linewidth=0.5)

            # 값 라벨 (5% 이하일 때)
            for bar, val in zip(bars, eers):
                if not np.isnan(val) and val <= 5:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xlabel("데이터 조합")
        ax.set_ylabel("EER (%)")
        ax.set_title(f"{eval_set}")
        ax.set_xticks(x)
        ax.set_xticklabels(combos)
        ax.set_ylim(bottom=0)

    axes[0].legend(loc="upper left", framealpha=0.9)
    fig.suptitle("데이터 커버리지별 EER 비교", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "G1_coverage_ablation.pdf")
    fig.savefig(output_dir / "G1_coverage_ablation.png")
    plt.close(fig)
    print(f"  G1 저장: {output_dir}/G1_coverage_ablation.pdf")


# ═══════════════════════════════════════════════════════════════
# G2: 성능-파라미터 트레이드오프 (Scatter Plot)
# ═══════════════════════════════════════════════════════════════

def plot_g2(df: pd.DataFrame, output_dir: Path) -> None:
    """C_all 기준: 파라미터 수 vs EER, 버블 크기 = 추론 속도."""
    call_df = df[df["data_combo"] == "C_all"]
    if call_df.empty:
        call_df = df[df["experiment"].str.contains("call", na=False)]

    if call_df.empty:
        print("  [SKIP] G2: C_all 결과 없음")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax_idx, eval_set in enumerate(EVAL_SETS):
        ax = axes[ax_idx]
        eer_col = f"{eval_set}_eer"

        for _, row in call_df.iterrows():
            model = row["model"].lower()
            style = MODEL_STYLES.get(model, {"color": "gray", "marker": "o", "label": model})

            params = int(row["params"])
            eer = float(row[eer_col]) * 100 if eer_col in row and pd.notna(row[eer_col]) else None
            inf_ms = float(row.get("inference_ms", 1))

            if eer is None:
                continue

            # 버블 크기: 추론 속도에 비례 (느릴수록 큰 원)
            size = max(inf_ms * 80, 50)

            ax.scatter(params, eer, s=size, c=style["color"],
                       marker=style["marker"], label=style["label"],
                       edgecolors="black", linewidth=0.5, alpha=0.8, zorder=5)
            ax.annotate(f"{eer:.2f}%\n{inf_ms:.1f}ms",
                        (params, eer), textcoords="offset points",
                        xytext=(10, 5), fontsize=8)

        ax.set_xscale("log")
        ax.set_xlabel("파라미터 수")
        ax.set_ylabel("EER (%)")
        ax.set_title(f"{eval_set}")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    # 중복 제거
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), loc="upper right")

    fig.suptitle("성능-파라미터 트레이드오프 (C_all)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "G2_efficiency_tradeoff.pdf")
    fig.savefig(output_dir / "G2_efficiency_tradeoff.png")
    plt.close(fig)
    print(f"  G2 저장: {output_dir}/G2_efficiency_tradeoff.pdf")


# ═══════════════════════════════════════════════════════════════
# G3: SNR별 성능 하락 (Line Chart)
# ═══════════════════════════════════════════════════════════════

def plot_g3(noise_df: pd.DataFrame, output_dir: Path) -> None:
    """SNR별 EER 변화 라인 차트."""
    # X축 순서: clean → 20dB → 10dB → 5dB
    condition_order = ["clean", "gaussian_001", "musan_20db", "musan_10db", "musan_5db"]
    condition_labels = ["Clean", "Gauss\nα=0.001", "MUSAN\n20dB", "MUSAN\n10dB", "MUSAN\n5dB"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax_idx, eval_set in enumerate(EVAL_SETS):
        ax = axes[ax_idx]
        es_df = noise_df[noise_df["eval_set"] == eval_set]

        for model in es_df["model"].unique():
            m_df = es_df[es_df["model"] == model]
            style = MODEL_STYLES.get(model, {"color": "gray", "marker": "o", "label": model})

            x_vals, y_vals = [], []
            for i, cond in enumerate(condition_order):
                row = m_df[m_df["noise_condition"] == cond]
                if not row.empty:
                    x_vals.append(i)
                    y_vals.append(float(row.iloc[0]["eer"]) * 100)

            if x_vals:
                ax.plot(x_vals, y_vals, marker=style["marker"], color=style["color"],
                        label=style["label"], linewidth=2, markersize=8)

        ax.set_xticks(range(len(condition_labels)))
        ax.set_xticklabels(condition_labels, fontsize=9)
        ax.set_ylabel("EER (%)")
        ax.set_title(f"{eval_set}")
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper left")
    fig.suptitle("노이즈 조건별 성능 변화", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "G3_noise_degradation.pdf")
    fig.savefig(output_dir / "G3_noise_degradation.png")
    plt.close(fig)
    print(f"  G3 저장: {output_dir}/G3_noise_degradation.pdf")


# ═══════════════════════════════════════════════════════════════
# G4: Phase 1 → Phase 2 개선 (Before/After Bar Chart)
# ═══════════════════════════════════════════════════════════════

def plot_g4(df: pd.DataFrame, output_dir: Path) -> None:
    """Phase 1 최악 EER vs C_all EER 비교."""
    # 각 모델 × 테스트셋에서 Phase 1 최악 vs C_all
    models_phase1 = ["aasist", "lcnn", "rawnet2"]
    phase1_combos = ["C1", "C2", "C3"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax_idx, eval_set in enumerate(EVAL_SETS):
        ax = axes[ax_idx]
        eer_col = f"{eval_set}_eer"

        if eer_col not in df.columns:
            continue

        labels_list = []
        worst_eers = []
        call_eers = []
        colors = []

        for model in models_phase1:
            style = MODEL_STYLES.get(model, {"color": "gray"})

            # Phase 1 최악
            phase1 = df[(df["model"] == model) & (df["data_combo"].isin(phase1_combos))]
            if phase1.empty or eer_col not in phase1.columns:
                continue
            worst = phase1[eer_col].apply(lambda x: float(x) if pd.notna(x) else 0).max() * 100

            # C_all
            call = df[(df["model"] == model) & (df["data_combo"] == "C_all")]
            if call.empty:
                continue
            call_eer = float(call.iloc[0][eer_col]) * 100

            labels_list.append(model.upper())
            worst_eers.append(worst)
            call_eers.append(call_eer)
            colors.append(style["color"])

        # SpecRNet은 Phase 1 없으므로 C_all만
        spec_call = df[(df["model"] == "specrnet") & (df["data_combo"] == "C_all")]
        if not spec_call.empty:
            labels_list.append("SPECRNET")
            worst_eers.append(0)  # Phase 1 없음
            call_eers.append(float(spec_call.iloc[0][eer_col]) * 100)
            colors.append(MODEL_STYLES["specrnet"]["color"])

        x = np.arange(len(labels_list))
        width = 0.35

        bars1 = ax.bar(x - width / 2, worst_eers, width, label="Phase 1 최악",
                        color=[c for c in colors], alpha=0.4, edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x + width / 2, call_eers, width, label="C_all",
                        color=[c for c in colors], alpha=0.9, edgecolor="black", linewidth=0.5)

        # 개선 배율 표시
        for i, (w, c) in enumerate(zip(worst_eers, call_eers)):
            if w > 0 and c > 0:
                ratio = w / c
                ax.text(x[i], max(w, c) + 1, f"{ratio:.0f}x",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_ylabel("EER (%)")
        ax.set_title(f"{eval_set}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, fontsize=9)

    axes[0].legend()
    fig.suptitle("Phase 1 최악 → C_all 개선 비교", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "G4_improvement.pdf")
    fig.savefig(output_dir / "G4_improvement.png")
    plt.close(fig)
    print(f"  G4 저장: {output_dir}/G4_improvement.pdf")


# ═══════════════════════════════════════════════════════════════
# Tables (LaTeX + Markdown)
# ═══════════════════════════════════════════════════════════════

def generate_tables(df: pd.DataFrame, noise_df: pd.DataFrame | None, output_dir: Path) -> None:
    """T1~T6 논문용 테이블 생성."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- T1: 전체 결과표 ---
    _generate_t1(df, output_dir)

    # --- T2: C_all 모델 비교 ---
    _generate_t2(df, output_dir)

    # --- T3: 임동훈 비교표 ---
    _generate_t3(df, output_dir)

    # --- T4: 노이즈 강건성표 ---
    if noise_df is not None:
        _generate_t4(noise_df, output_dir)


def _generate_t1(df: pd.DataFrame, output_dir: Path) -> None:
    """T1: 13개 실험 전체 결과표."""
    metrics = ["eer", "mindcf", "accuracy", "f1", "auc"]
    lines = []
    lines.append("# Table 1: 전체 실험 결과")
    lines.append("")
    header = "| 실험 | 모델 | 데이터 |"
    for es in EVAL_SETS:
        for m in ["EER", "minDCF"]:
            header += f" {es} {m} |"
    lines.append(header)
    lines.append("|" + "---|" * (3 + len(EVAL_SETS) * 2))

    for _, row in df.iterrows():
        line = f"| {row['experiment']} | {row['model']} | {row.get('data_combo', '')} |"
        for es in EVAL_SETS:
            for m in ["eer", "mindcf"]:
                key = f"{es}_{m}"
                if key in row and pd.notna(row[key]):
                    val = float(row[key])
                    if m == "eer":
                        line += f" {val*100:.2f}% |"
                    else:
                        line += f" {val:.4f} |"
                else:
                    line += " — |"
        lines.append(line)

    (output_dir / "T1_full_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  T1 저장: {output_dir}/T1_full_results.md")


def _generate_t2(df: pd.DataFrame, output_dir: Path) -> None:
    """T2: C_all 모델 비교 (성능 + 경량화 지표)."""
    call_df = df[df["data_combo"] == "C_all"]
    if call_df.empty:
        call_df = df[df["experiment"].str.contains("call", na=False)]

    lines = []
    lines.append("# Table 2: C_all 모델 비교")
    lines.append("")
    lines.append("| 모델 | Params | 크기(MB) | 추론(ms) | 처리량 |"
                 " T1 EER | T2 EER | T3 EER | T1 minDCF | T2 minDCF | T3 minDCF |")
    lines.append("|" + "---|" * 11)

    for _, row in call_df.iterrows():
        params = int(row["params"])
        if params >= 1_000_000:
            params_str = f"{params/1e6:.2f}M"
        else:
            params_str = f"{params/1e3:.0f}K"

        line = f"| {row['model']} | {params_str} | {row.get('model_mb', '—')} | {row.get('inference_ms', '—')} | {row.get('throughput_sps', '—')} |"

        for es in EVAL_SETS:
            eer_key = f"{es}_eer"
            val = float(row[eer_key]) * 100 if eer_key in row and pd.notna(row[eer_key]) else None
            line += f" {val:.2f}% |" if val is not None else " — |"

        for es in EVAL_SETS:
            dcf_key = f"{es}_mindcf"
            val = float(row[dcf_key]) if dcf_key in row and pd.notna(row[dcf_key]) else None
            line += f" {val:.4f} |" if val is not None else " — |"

        lines.append(line)

    (output_dir / "T2_call_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  T2 저장: {output_dir}/T2_call_comparison.md")


def _generate_t3(df: pd.DataFrame, output_dir: Path) -> None:
    """T3: 임동훈(2024) 비교표."""
    from compare_lim import LIM_TABLE4, LIM_TABLE5

    lines = []
    lines.append("# Table 3: 임동훈(2024) vs 본 연구 — KoAAD(T2) 성능 비교")
    lines.append("")
    lines.append("| 출처 | 모델 | 학습 데이터 | T2 EER(%) | T2 FPR(%) | 비고 |")
    lines.append("|---|---|---|---|---|---|")

    for name, data in LIM_TABLE4.items():
        fpr = LIM_TABLE5.get(name, {}).get("fpr", "—")
        lines.append(f"| 임동훈(2024) | {name} | MLAAD 5K | {data['eer']:.2f} | {fpr} | {data.get('note', '')} |")

    call_df = df[df["data_combo"] == "C_all"]
    if call_df.empty:
        call_df = df[df["experiment"].str.contains("call", na=False)]

    for _, row in call_df.iterrows():
        eer = float(row["T2_eer"]) * 100 if "T2_eer" in row and pd.notna(row["T2_eer"]) else "—"
        fpr = float(row["T2_fpr"]) * 100 if "T2_fpr" in row and pd.notna(row["T2_fpr"]) else "—"
        eer_str = f"{eer:.2f}" if isinstance(eer, float) else eer
        fpr_str = f"{fpr:.2f}" if isinstance(fpr, float) else fpr
        lines.append(f"| **본 연구** | **{row['model']} (C_all)** | **C_all 1.07M** | **{eer_str}** | **{fpr_str}** | {row['experiment']} |")

    (output_dir / "T3_lim_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  T3 저장: {output_dir}/T3_lim_comparison.md")


def _generate_t4(noise_df: pd.DataFrame, output_dir: Path) -> None:
    """T4: 노이즈 강건성표."""
    lines = []
    lines.append("# Table 4: 노이즈 강건성 (EER %)")
    lines.append("")

    conditions = noise_df["noise_condition"].unique()
    models = noise_df["model"].unique()

    for eval_set in EVAL_SETS:
        lines.append(f"\n## {eval_set}")
        header = "| 모델 |"
        for c in conditions:
            header += f" {c} |"
        lines.append(header)
        lines.append("|" + "---|" * (1 + len(conditions)))

        for model in models:
            line = f"| {model} |"
            for c in conditions:
                row = noise_df[
                    (noise_df["model"] == model) &
                    (noise_df["noise_condition"] == c) &
                    (noise_df["eval_set"] == eval_set)
                ]
                if not row.empty:
                    eer = float(row.iloc[0]["eer"]) * 100
                    line += f" {eer:.2f} |"
                else:
                    line += " — |"
            lines.append(line)

    (output_dir / "T4_noise_robustness.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  T4 저장: {output_dir}/T4_noise_robustness.md")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="논문용 그래프/테이블 생성")
    parser.add_argument("--results", default="experiments/results.csv")
    parser.add_argument("--noise", default="experiments/noise_robustness.csv")
    parser.add_argument("--graph", choices=["G1", "G2", "G3", "G4"],
                        help="특정 그래프만 생성")
    parser.add_argument("--tables", action="store_true", help="테이블만 생성")
    parser.add_argument("--all", action="store_true", help="전체 생성")
    parser.add_argument("--output-dir", default="experiments",
                        help="출력 루트 디렉토리")
    args = parser.parse_args()

    fig_dir = Path(args.output_dir) / "figures"
    tbl_dir = Path(args.output_dir) / "tables"

    df = load_results(args.results)
    noise_df = load_noise_results(args.noise)

    if df is None:
        print("결과 파일이 없습니다. 먼저 실험을 실행하세요.")
        return 1

    print(f"\n  Loaded: {len(df)} experiments from {args.results}")
    if noise_df is not None:
        print(f"  Loaded: {len(noise_df)} noise evaluations from {args.noise}")

    if args.graph:
        graph_map = {
            "G1": lambda: plot_g1(df, fig_dir),
            "G2": lambda: plot_g2(df, fig_dir),
            "G3": lambda: plot_g3(noise_df, fig_dir) if noise_df is not None else print("  [SKIP] G3: noise 결과 없음"),
            "G4": lambda: plot_g4(df, fig_dir),
        }
        graph_map[args.graph]()
    elif args.tables:
        generate_tables(df, noise_df, tbl_dir)
    elif args.all:
        print("\n  === 그래프 생성 ===")
        plot_g1(df, fig_dir)
        plot_g2(df, fig_dir)
        if noise_df is not None:
            plot_g3(noise_df, fig_dir)
        else:
            print("  [SKIP] G3: noise 결과 없음")
        plot_g4(df, fig_dir)

        print("\n  === 테이블 생성 ===")
        generate_tables(df, noise_df, tbl_dir)
    else:
        parser.print_help()
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
