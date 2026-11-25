#!/usr/bin/env python3
"""C1+C2+C3 manifest를 합쳐 C_all manifest를 생성.

기본(수정판) 동작:
    1. C1/C2/C3의 train+dev를 하나의 풀로 모은다.
    2. file_path 기준으로 중복을 제거한다.
    3. 통합 풀에서 speaker_id 그룹 단위로 train/dev를 다시 나눈다.

--legacy 는 수정 전 동작(단순 concat)을 재현한다. 기존 실험 결과와의
대조가 필요할 때만 사용한다.

수정 이유
---------
기존 구현은 C1/C2/C3의 train/dev를 각각 이어붙이기만 했다. 세 조합은
구성 요소가 겹치므로(MLAAD+M-AILABS는 C1·C2 공통, SpeechFake ko는 C2·C3 공통)
다음 두 문제가 발생했다.

    - C_all/train 행의 31.2%가 중복 파일 (1,067,960행 / 고유 734,862개)
    - C_all/dev 파일의 64.6%가 C_all/train에도 존재
      → 조합마다 분할을 독립 수행했기 때문에, 같은 파일이 C1에서는 train,
        C2에서는 dev로 배정된 뒤 양쪽에 모두 남는다.

dev 오염은 early stopping과 best checkpoint 선택의 근거를 무너뜨린다.
자세한 내용은 데이터_분할_명세.md 참조.

사용법:
    python scripts/03_train/merge_manifests.py
    python scripts/03_train/merge_manifests.py --dev-ratio 0.1 --seed 42
    python scripts/03_train/merge_manifests.py --legacy      # 수정 전 동작
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ROOT = ROOT / "manifests"
SOURCES = ("C1", "C2", "C3")


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  SKIP {path} (없음)")
        return None
    df = pd.read_csv(path)
    df["file_path"] = df["file_path"].astype(str).str.replace("\\", "/", regex=False)
    return df


def build_pool(root: Path) -> pd.DataFrame:
    """C1/C2/C3의 train+dev를 모아 file_path 기준 중복 제거."""
    frames = []
    for combo in SOURCES:
        for split in ("train", "dev"):
            df = _read(root / combo / f"{split}.csv")
            if df is not None:
                print(f"  {combo}/{split}: {len(df):,}행")
                frames.append(df)
    if not frames:
        raise SystemExit("C1/C2/C3 manifest를 찾을 수 없습니다.")

    pool = pd.concat(frames, ignore_index=True)
    before = len(pool)
    pool = pool.drop_duplicates(subset=["file_path"]).reset_index(drop=True)
    print(f"\n  통합 {before:,}행 → 중복 제거 후 {len(pool):,}개 "
          f"(제거 {before - len(pool):,}, {(before - len(pool)) / before * 100:.1f}%)")
    return pool


def split_by_speaker(pool: pd.DataFrame, dev_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """speaker_id 그룹 단위로 train/dev 분할.

    한쪽에 라벨이 빠지면 그룹 분할을 유지한 채 라벨별로 따로 나눈다.
    build_splits.py의 fallback과 달리 파일 단위 무작위 분할로 되돌아가지 않는다.
    """
    rng = np.random.RandomState(seed)
    dev_speakers: set = set()

    # 라벨별로 그룹을 나눠 양쪽 split에 두 라벨이 모두 들어가도록 보장
    for label in sorted(pool["label"].unique()):
        # pandas 확장 dtype(ArrowStringArray)은 in-place shuffle이 보장되지 않으므로
        # 파이썬 객체 배열로 변환한 뒤 섞는다.
        speakers = np.asarray(
            sorted(pool.loc[pool["label"] == label, "speaker_id"].unique().tolist()),
            dtype=object,
        )
        rng.shuffle(speakers)
        n_dev = max(1, int(len(speakers) * dev_ratio)) if len(speakers) > 1 else 0
        dev_speakers |= set(speakers[:n_dev])

    dev = pool[pool["speaker_id"].isin(dev_speakers)]
    train = pool[~pool["speaker_id"].isin(dev_speakers)]
    return train.reset_index(drop=True), dev.reset_index(drop=True)


def report(name: str, df: pd.DataFrame) -> None:
    n_real = int((df["label"] == "real").sum())
    n_fake = int((df["label"] == "fake").sum())
    print(f"    {name:12s} {len(df):>10,}개  (real {n_real:,} / fake {n_fake:,}, "
          f"speaker_id {df['speaker_id'].nunique():,})")


def merge_legacy(root: Path, out_dir: Path) -> None:
    """수정 전 동작: 단순 concat (중복 제거·재분할 없음)."""
    print("  [legacy] 단순 concat — 중복과 train/dev 누수가 그대로 남습니다.")
    for split in ("train", "dev", "test"):
        header, rows = None, []
        for combo in SOURCES:
            src = root / combo / f"{split}.csv"
            if not src.exists():
                continue
            lines = src.read_text(encoding="utf-8").splitlines()
            if header is None:
                header = lines[0]
            rows.extend(lines[1:])
        if header is None:
            continue
        out = out_dir / f"{split}.csv"
        out.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
        print(f"    {split}: {len(rows):,}행 → {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="C_all manifest 생성")
    ap.add_argument("--manifests", default=str(MANIFEST_ROOT), help="manifest 루트")
    ap.add_argument("--dev-ratio", type=float, default=0.1, help="dev 비율 (speaker_id 그룹 기준, 기본 0.1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-name", default="C_all",
                    help="출력 디렉토리 이름 (기본 C_all). 기존 결과를 보존하려면 C_all_v2 등으로 지정")
    ap.add_argument("--legacy", action="store_true", help="수정 전 동작(단순 concat) 재현")
    args = ap.parse_args()

    root = Path(args.manifests)
    out_dir = root / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.legacy:
        merge_legacy(root, out_dir)
        return 0

    print("[1] 소스 manifest 수집")
    pool = build_pool(root)

    print("\n[2] speaker_id 그룹 단위 train/dev 재분할")
    train, dev = split_by_speaker(pool, args.dev_ratio, args.seed)
    report(f"{args.out_name}/train", train)
    report(f"{args.out_name}/dev", dev)

    overlap = set(train["file_path"]) & set(dev["file_path"])
    sp_overlap = set(train["speaker_id"]) & set(dev["speaker_id"])
    print(f"\n[3] 검증")
    print(f"    train↔dev 파일 중복   : {len(overlap):,}")
    print(f"    train↔dev 그룹 ID(speaker_id) 중복: {len(sp_overlap):,}")
    if overlap or sp_overlap:
        print("    [경고] 중복이 남아 있습니다.")
        return 1

    train.to_csv(out_dir / "train.csv", index=False)
    dev.to_csv(out_dir / "dev.csv", index=False)
    print(f"\n  저장: {out_dir}/train.csv, {out_dir}/dev.csv")
    print("  평가셋 누수 재확인: python scripts/04_analyze/verify_splits.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
