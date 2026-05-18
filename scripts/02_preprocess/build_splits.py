#!/usr/bin/env python3
"""Phase 1.3–1.4: 데이터 조합별 분할 및 평가셋 생성

processed_metadata.csv를 기반으로 C1/C2/C3 학습 데이터 조합과
T1/T2/T3 고정 평가셋을 생성합니다.

조합 정의 (연구계획서 v5):
    C1: MLAAD v9 (ko 제외) + M-AILABS                      — 다국어 only
    C2: MLAAD v9 (전체) + M-AILABS + SpeechFake-MD ko       — 다국어+한국어
    C3: SpeechFake-MD ko + KoAAD                            — 한국어 only

평가셋 정의:
    T1: SpeechFake-MD ko test split            — 한국어 기본 평가
    T2: KoAAD holdout                          — 한국어 cross-dataset
    T3: MLAAD test split (ko 제외)             — 다국어 성능 유지

사용법:
    python scripts/02_preprocess/build_splits.py --meta-dir metadata --output-dir manifests
    python scripts/02_preprocess/build_splits.py --meta-dir metadata --output-dir manifests --seed 42

필요한 패키지:
    pip install pandas numpy scikit-learn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ════════════════════════════════════════════════════════════
#  설정
# ════════════════════════════════════════════════════════════

METADATA_DIR = Path("metadata")
OUTPUT_DIR = Path("manifests")
SEED = 42
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1

T1_EVAL_RATIO = 0.2
T2_EVAL_RATIO = 0.2
T3_EVAL_RATIO = 0.2


# ════════════════════════════════════════════════════════════
#  유틸리티
# ════════════════════════════════════════════════════════════

def speaker_disjoint_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """화자(speaker_id) 기준 겹침 없이 train/dev/test 분할.

    화자를 먼저 분할한 후 해당 화자의 모든 발화를 해당 세트에 배치합니다.
    """
    speakers = df["speaker_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(speakers)

    n_total = len(speakers)
    n_train = int(n_total * train_ratio)
    n_dev = int(n_total * dev_ratio)

    train_speakers = set(speakers[:n_train])
    dev_speakers = set(speakers[n_train:n_train + n_dev])
    test_speakers = set(speakers[n_train + n_dev:])

    train_df = df[df["speaker_id"].isin(train_speakers)].copy()
    dev_df = df[df["speaker_id"].isin(dev_speakers)].copy()
    test_df = df[df["speaker_id"].isin(test_speakers)].copy()

    return train_df, dev_df, test_df


def balance_labels(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """real/fake 비율을 1:1에 가깝게 맞춤 (다수 클래스 다운샘플링)."""
    real_df = df[df["label"] == "real"]
    fake_df = df[df["label"] == "fake"]

    min_count = min(len(real_df), len(fake_df))
    if min_count == 0:
        return df

    real_sampled = real_df.sample(n=min_count, random_state=seed)
    fake_sampled = fake_df.sample(n=min_count, random_state=seed)

    return pd.concat([real_sampled, fake_sampled]).sample(frac=1, random_state=seed)


def save_manifest(df: pd.DataFrame, path: Path, name: str) -> None:
    """매니페스트 CSV 저장 + 요약 출력."""
    df.to_csv(path, index=False)
    n_real = len(df[df["label"] == "real"])
    n_fake = len(df[df["label"] == "fake"])
    n_speakers = df["speaker_id"].nunique()
    print(f"    {name:25s} — {len(df):>8,}개 (real={n_real:,}, fake={n_fake:,}, speakers={n_speakers:,})")


# ════════════════════════════════════════════════════════════
#  데이터 조합 빌더
# ════════════════════════════════════════════════════════════

def build_c1(df: pd.DataFrame) -> pd.DataFrame:
    """C1: MLAAD v9 (ko 제외) + M-AILABS — 다국어 only."""
    mlaad_non_ko = df[(df["dataset"] == "mlaad") & (df["language"] != "ko")]
    mailabs = df[df["dataset"] == "mailabs"]
    return pd.concat([mlaad_non_ko, mailabs])


def build_c2(df: pd.DataFrame) -> pd.DataFrame:
    """C2: MLAAD v9 (전체) + M-AILABS + SpeechFake-MD ko — 다국어+한국어."""
    mlaad = df[df["dataset"] == "mlaad"]
    mailabs = df[df["dataset"] == "mailabs"]
    sf_md_ko = df[(df["dataset"] == "speechfake-md") & (df["language"] == "ko")]

    return pd.concat([mlaad, mailabs, sf_md_ko])


def build_c3(df: pd.DataFrame) -> pd.DataFrame:
    """C3: SpeechFake-MD ko + KoAAD — 한국어 only."""
    sf_md_ko = df[(df["dataset"] == "speechfake-md") & (df["language"] == "ko")]
    koaad = df[df["dataset"] == "koaad"]

    return pd.concat([sf_md_ko, koaad])


# ════════════════════════════════════════════════════════════
#  평가셋 빌더
# ════════════════════════════════════════════════════════════

def build_t1(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """T1: SpeechFake-MD ko test split — 한국어 기본 평가."""
    sf_md_ko = df[
        (df["dataset"] == "speechfake-md") & (df["language"] == "ko")
    ]
    if len(sf_md_ko) == 0:
        print("    [경고] SpeechFake-MD ko 데이터가 없습니다.")
        return pd.DataFrame(columns=df.columns)

    _, _, test = speaker_disjoint_split(sf_md_ko, 0.6, 0.2, seed)
    return test


def build_t2(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """T2: KoAAD holdout — 한국어 cross-dataset 평가."""
    koaad = df[df["dataset"] == "koaad"]
    if len(koaad) == 0:
        print("    [경고] KoAAD 데이터가 없습니다.")
        return pd.DataFrame(columns=df.columns)

    _, _, test = speaker_disjoint_split(koaad, 0.7, 0.1, seed)
    return test


def build_t3(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """T3: MLAAD test split (ko 제외) — 다국어 성능 유지 확인."""
    mlaad_non_ko = df[(df["dataset"] == "mlaad") & (df["language"] != "ko")]
    if len(mlaad_non_ko) == 0:
        print("    [경고] MLAAD (ko 제외) 데이터가 없습니다.")
        return pd.DataFrame(columns=df.columns)

    _, _, test = speaker_disjoint_split(mlaad_non_ko, 0.7, 0.1, seed)

    mailabs = df[df["dataset"] == "mailabs"]
    if len(mailabs) > 0:
        _, _, mailabs_test = speaker_disjoint_split(mailabs, 0.7, 0.1, seed)
        n_needed = len(test)
        if len(mailabs_test) > n_needed:
            mailabs_test = mailabs_test.sample(n=n_needed, random_state=seed)
        test = pd.concat([test, mailabs_test])

    return test


# ════════════════════════════════════════════════════════════
#  메인 실행
# ════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1.3-1.4: 데이터 조합 분할 및 평가셋 생성",
    )
    parser.add_argument("--meta-dir", default=str(METADATA_DIR), help="메타데이터 CSV 경로")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="매니페스트 출력 경로")
    parser.add_argument("--seed", type=int, default=SEED, help="랜덤 시드 (기본: 42)")
    parser.add_argument("--no-balance", action="store_true", help="클래스 밸런싱 비활성화")
    args = parser.parse_args()

    meta_dir = Path(args.meta_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = meta_dir / "processed_metadata.csv"
    if not csv_path.exists():
        csv_path = meta_dir / "raw_metadata.csv"
    if not csv_path.exists():
        print(f"오류: 메타데이터 CSV가 없습니다: {meta_dir}")
        return 1

    print(f"\n{'='*60}")
    print(f"  데이터 분할 시작")
    print(f"  소스: {csv_path}")
    print(f"  시드: {args.seed}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)

    if "quality_ok" in df.columns:
        before = len(df)
        df = df[df["quality_ok"].astype(str) == "True"]
        print(f"\n  품질 필터링: {before:,} → {len(df):,} (제외: {before - len(df):,})")

    print(f"  전체 데이터: {len(df):,}개")
    print(f"    real: {len(df[df['label'] == 'real']):,}")
    print(f"    fake: {len(df[df['label'] == 'fake']):,}")
    print(f"    화자 수: {df['speaker_id'].nunique():,}")

    # ── 평가셋 먼저 분리 (학습에서 제외) ──
    print(f"\n  ── 평가셋 생성 (T1/T2/T3) ──")

    t1 = build_t1(df, args.seed)
    t2 = build_t2(df, args.seed)
    t3 = build_t3(df, args.seed)

    save_manifest(t1, output_dir / "T1_speechfake_ko_test.csv", "T1 (SpeechFake-MD ko)")
    save_manifest(t2, output_dir / "T2_koaad_holdout.csv", "T2 (KoAAD holdout)")
    save_manifest(t3, output_dir / "T3_mlaad_test.csv", "T3 (MLAAD test, ko제외)")

    eval_indices = set(t1.index) | set(t2.index) | set(t3.index)
    eval_speakers = set(t1["speaker_id"]) | set(t2["speaker_id"]) | set(t3["speaker_id"])

    df_train_pool = df[~df.index.isin(eval_indices)]
    df_train_pool = df_train_pool[~df_train_pool["speaker_id"].isin(eval_speakers)]

    print(f"\n  평가셋 화자 제외 후 학습 풀: {len(df_train_pool):,}개")

    # ── 학습 조합 (C1/C2/C3) ──
    print(f"\n  ── 학습 조합 생성 (C1/C2/C3) ──")

    combinations = [
        ("C1", build_c1),
        ("C2", build_c2),
        ("C3", build_c3),
    ]

    for combo_name, builder in combinations:
        combo_df = builder(df_train_pool)

        if len(combo_df) == 0:
            print(f"    {combo_name}: 데이터 없음 (건너뜀)")
            continue

        train, dev, test = speaker_disjoint_split(combo_df, TRAIN_RATIO, DEV_RATIO, args.seed)

        if not args.no_balance:
            train = balance_labels(train, args.seed)

        combo_dir = output_dir / combo_name
        combo_dir.mkdir(exist_ok=True)

        save_manifest(train, combo_dir / "train.csv", f"{combo_name}/train")
        save_manifest(dev, combo_dir / "dev.csv", f"{combo_name}/dev")
        save_manifest(test, combo_dir / "test.csv", f"{combo_name}/test")

    # ── 검증 보고 ──
    print(f"\n  ── 화자 겹침 검증 ──")
    _verify_speaker_disjoint(output_dir)

    print(f"\n{'='*60}")
    print(f"  완료! 매니페스트 저장 위치: {output_dir}")
    print(f"{'='*60}\n")

    return 0


def _verify_speaker_disjoint(output_dir: Path) -> None:
    """C1/C2/C3 내 train/dev/test 화자 겹침 검증."""
    for combo in ["C1", "C2", "C3"]:
        combo_dir = output_dir / combo
        if not combo_dir.exists():
            continue

        sets = {}
        for split in ["train", "dev", "test"]:
            csv_path = combo_dir / f"{split}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                sets[split] = set(df["speaker_id"].unique())

        overlaps = []
        for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
            if a in sets and b in sets:
                overlap = sets[a] & sets[b]
                if overlap:
                    overlaps.append(f"{a}∩{b}={len(overlap)}")

        if overlaps:
            print(f"    [경고] {combo}: 화자 겹침 발견 — {', '.join(overlaps)}")
        else:
            print(f"    [OK] {combo}: 화자 겹침 없음 ✓")


if __name__ == "__main__":
    sys.exit(main())
