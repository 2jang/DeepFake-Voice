#!/usr/bin/env python3
"""Codec 증강 사전 계산 — 학습 시 on-the-fly 인코딩 병목 제거.

apply_codec()은 샘플당 ~5ms (PyAV encode/decode)로, 학습 속도를 40→9 it/s로
떨어뜨린다. 이 스크립트로 전체 학습 파일을 미리 인코딩하면 학습 시에는
일반 WAV 로드와 동일한 속도로 증강 데이터를 사용할 수 있다.

멀티프로세싱으로 CPU 코어 수만큼 병렬 처리 (기본 --workers 8).

사용법:
    # C_all 학습 데이터 사전 인코딩 (기본: MP3 64kbps, 8 workers)
    python scripts/03_train/precompute_codec.py manifests/C_all/train.csv

    # 워커 수 조절
    python scripts/03_train/precompute_codec.py manifests/C_all/train.csv --workers 16

    # 커스텀 설정
    python scripts/03_train/precompute_codec.py manifests/C_all/train.csv \
        --codec mp3 --bitrate 128k --cache-dir data/codec_cache/mp3_128k
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_CACHE_DIR = "data/codec_cache/mp3_64k"


def cache_path_for(file_path: str, cache_dir: Path) -> Path:
    """원본 파일 경로 → 캐시 경로. 원본의 상대 경로 구조를 유지."""
    p = Path(file_path.replace("\\", "/"))
    if p.is_absolute():
        rel = p.relative_to(Path.cwd())
    else:
        rel = p
    return cache_dir / rel


def _process_one(args_tuple: tuple) -> str:
    """단일 파일 codec 처리. 워커 프로세스에서 실행.

    Returns: "ok" | "skip" | "missing" | "error: ..."
    """
    fpath, cache_dir, codec, bitrate, sr, skip_existing = args_tuple
    fpath_norm = fpath.replace("\\", "/")
    out_path = cache_path_for(fpath_norm, Path(cache_dir))

    if skip_existing and out_path.exists():
        return "skip"

    if not Path(fpath_norm).exists():
        return "missing"

    try:
        from augment import apply_codec

        wav, _sr = sf.read(fpath_norm, dtype="float32")
        if wav.ndim > 1:
            wav = wav[:, 0]

        encoded = apply_codec(wav, sr=sr, codec=codec, bitrate=bitrate)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), encoded, sr)
        return "ok"
    except Exception as e:
        return f"error: {fpath_norm}: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Codec 증강 사전 계산 (멀티프로세싱)")
    parser.add_argument("manifest", help="CSV manifest (file_path, label 컬럼)")
    parser.add_argument("--codec", default="mp3")
    parser.add_argument("--bitrate", default="64k")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--sr", type=int, default=16_000)
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="이미 캐시된 파일 건너뛰기 (기본: True)")
    parser.add_argument("--workers", type=int, default=None,
                        help="병렬 워커 수 (기본: CPU 코어 수)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    num_workers = args.workers or os.cpu_count() or 4
    df = pd.read_csv(args.manifest)
    files = df["file_path"].unique().tolist()
    print(f"고유 파일: {len(files):,}개 | 워커: {num_workers}개 (CPU: {os.cpu_count()})")

    # 워커에 전달할 인자 튜플 리스트
    tasks = [
        (f, str(cache_dir), args.codec, args.bitrate, args.sr, args.skip_existing)
        for f in files
    ]

    ok = 0
    skipped = 0
    failed = 0
    errors: list[str] = []

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_process_one, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"Codec({args.codec} {args.bitrate})"):
            result = future.result()
            if result == "ok":
                ok += 1
            elif result == "skip":
                skipped += 1
            elif result == "missing":
                failed += 1
            else:  # error: ...
                failed += 1
                errors.append(result)

    if errors and len(errors) <= 20:
        print("\n에러 목록:")
        for e in errors:
            print(f"  {e}")
    elif errors:
        print(f"\n에러 {len(errors)}개 (처음 10개):")
        for e in errors[:10]:
            print(f"  {e}")

    print(f"\n완료: {ok:,}개 인코딩, {skipped:,}개 스킵, {failed:,}개 실패 / 전체 {len(files):,}개")
    print(f"캐시 디렉토리: {cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
