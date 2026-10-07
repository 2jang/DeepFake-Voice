#!/usr/bin/env python3
"""데이터 분할 검증 — manifest 기준 누수·중복 실측.

데이터_분할_명세.md에 기재한 수치를 재현하는 스크립트.
manifest CSV만 읽으므로 오디오 파일이나 GPU 없이 실행된다.

사용법:
    python scripts/04_analyze/verify_splits.py
    python scripts/04_analyze/verify_splits.py --manifests manifests --json out.json

종료 코드:
    0  중대한 누수 없음 (평가셋 ↔ 학습셋 파일 중복 0)
    1  평가셋에 학습 파일이 포함됨 (결과 무효)
    2  manifest 없음
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

COMBOS = ["C1", "C2", "C3", "C_all"]
EVAL_SETS = {  # noqa: E305
    "T1": "T1_speechfake_ko_test.csv",
    "T2": "T2_koaad_holdout.csv",
    "T3": "T3_mlaad_test.csv",
}


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["file_path"] = df["file_path"].astype(str).str.replace("\\", "/", regex=False)
    return df


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def check_counts(root: Path, report: dict) -> None:
    section("1. 수량 — manifest 행 수 vs 고유 파일 수")
    print(f"  {'manifest':<18} {'행 수':>12} {'고유 파일':>12} {'중복 행':>12}  {'중복률':>7}")
    print(f"  {'-' * 68}")
    for combo in COMBOS:
        for split in ("train", "dev"):
            df = _load(root / combo / f"{split}.csv")
            if df is None:
                continue
            n, u = len(df), df["file_path"].nunique()
            dup = n - u
            rate = dup / n * 100 if n else 0.0
            flag = "  <-- 중복" if dup else ""
            print(f"  {combo + '/' + split:<18} {n:>12,} {u:>12,} {dup:>12,}  {rate:>6.1f}%{flag}")
            report["counts"][f"{combo}/{split}"] = {"rows": n, "unique": u, "dupes": dup}


def check_train_dev(root: Path, report: dict) -> None:
    section("2. train <-> dev 누수 (모델 선택 신뢰도)")
    print(f"  {'조합':<8} {'dev 고유':>12} {'train과 중복':>14} {'오염률':>9}")
    print(f"  {'-' * 48}")
    for combo in COMBOS:
        tr, dv = _load(root / combo / "train.csv"), _load(root / combo / "dev.csv")
        if tr is None or dv is None:
            continue
        s_tr, s_dv = set(tr["file_path"]), set(dv["file_path"])
        ov = s_tr & s_dv
        rate = len(ov) / len(s_dv) * 100 if s_dv else 0.0
        flag = "  <-- 오염" if rate > 1 else ""
        print(f"  {combo:<8} {len(s_dv):>12,} {len(ov):>14,} {rate:>8.1f}%{flag}")
        report["train_dev"][combo] = {"dev_unique": len(s_dv), "overlap": len(ov), "rate_pct": round(rate, 2)}


def check_eval_leak(root: Path, report: dict) -> int:
    section("3. 평가셋 <-> 학습셋 누수")
    critical = 0
    for combo in COMBOS:
        tr = _load(root / combo / "train.csv")
        if tr is None:
            continue
        f_tr, s_tr = set(tr["file_path"]), set(tr["speaker_id"])
        print(f"\n  [{combo}/train]  파일 {len(f_tr):,} · speaker_id {len(s_tr):,}")
        print(f"    {'평가셋':<6} {'파일':>10} {'파일 중복':>11} {'speaker_id':>10} {'ID 중복':>9} {'비율':>8}")
        print(f"    {'-' * 62}")
        for name, fn in EVAL_SETS.items():
            ev = _load(root / fn)
            if ev is None:
                continue
            f_ov = f_tr & set(ev["file_path"])
            s_ev = set(ev["speaker_id"])
            s_ov = s_tr & s_ev
            rate = len(s_ov) / len(s_ev) * 100 if s_ev else 0.0
            mark = ""
            if f_ov:
                mark, critical = "  <-- 파일 누수(치명)", critical + 1
            elif rate > 50:
                mark = "  <-- speaker_id 대부분 공유"
            print(f"    {name:<6} {len(ev):>10,} {len(f_ov):>11,} {len(s_ev):>10,} "
                  f"{len(s_ov):>9,} {rate:>7.1f}%{mark}")
            report["eval_leak"].setdefault(combo, {})[name] = {
                "eval_files": len(ev), "file_overlap": len(f_ov),
                "eval_speakers": len(s_ev), "speaker_overlap": len(s_ov),
                "speaker_overlap_pct": round(rate, 2),
            }
    return critical


def check_speaker_semantics(root: Path, report: dict) -> None:
    section("4. speaker_id의 실제 단위 (데이터셋별)")
    frames = []
    for combo in COMBOS:
        for split in ("train", "dev"):
            df = _load(root / combo / f"{split}.csv")
            if df is not None:
                frames.append(df)
    for fn in EVAL_SETS.values():
        df = _load(root / fn)
        if df is not None:
            frames.append(df)
    if not frames:
        return
    allm = pd.concat(frames).drop_duplicates(subset=["file_path"])
    print(f"  {'dataset':<18} {'고유 파일':>11} {'speaker_id':>11} {'파일/speaker':>13}  판정")
    print(f"  {'-' * 70}")
    for ds, g in allm.groupby("dataset"):
        n_sp = g["speaker_id"].nunique()
        ratio = len(g) / n_sp if n_sp else 0
        verdict = "파일 단위(화자 분할 아님)" if ratio < 3 else (
            "화자/시스템 단위" if ratio < 5000 else "시스템 단위(소수)")
        print(f"  {ds:<18} {len(g):>11,} {n_sp:>11,} {ratio:>13.1f}  {verdict}")
        report["speaker_semantics"][ds] = {
            "files": len(g), "speakers": n_sp, "files_per_speaker": round(ratio, 1), "verdict": verdict,
        }


def main() -> int:
    ap = argparse.ArgumentParser(description="데이터 분할 누수·중복 검증")
    ap.add_argument("--manifests", default="manifests", help="manifest 디렉토리")
    ap.add_argument("--json", help="결과를 JSON으로 저장할 경로")
    args = ap.parse_args()

    root = Path(args.manifests)
    if not root.exists():
        print(f"오류: manifest 디렉토리가 없습니다: {root}")
        return 2

    # train.csv를 가진 디렉토리를 모두 검사 대상으로 삼는다 (C_all_v2 등 포함).
    global COMBOS
    discovered = sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "train.csv").exists())
    COMBOS = [c for c in COMBOS if c in discovered] + [c for c in discovered if c not in COMBOS]

    report: dict = {"counts": {}, "train_dev": {}, "eval_leak": {}, "speaker_semantics": {}}

    check_counts(root, report)
    check_train_dev(root, report)
    critical = check_eval_leak(root, report)
    check_speaker_semantics(root, report)

    section("판정")
    if critical:
        print(f"  [치명] 평가셋에 학습 파일이 포함된 조합 {critical}건 — 해당 결과는 무효입니다.")
    else:
        print("  [OK] 평가셋과 학습셋 사이 파일 중복 없음.")
        print("  단, speaker_id 중복률과 train<->dev 오염률은 위 표를 근거로 해석해야 합니다.")

    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  JSON 저장: {args.json}")

    return 1 if critical else 0


if __name__ == "__main__":
    sys.exit(main())
