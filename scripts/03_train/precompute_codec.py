#!/usr/bin/env python3
"""Codec 증강 사전 계산 — 학습 시 on-the-fly 인코딩 병목 제거.

apply_codec()은 샘플당 ~5ms (PyAV encode/decode)로, 학습 속도를 40→9 it/s로
떨어뜨린다. 이 스크립트로 전체 학습 파일을 미리 인코딩하면 학습 시에는
일반 WAV 로드와 동일한 속도로 증강 데이터를 사용할 수 있다.

사용법:
    # C_all 학습 데이터 사전 인코딩 (기본: MP3 64kbps)
    python scripts/03_train/precompute_codec.py manifests/C_all/train.csv

    # 커스텀 설정
    python scripts/03_train/precompute_codec.py manifests/C_all/train.csv \
        --codec mp3 --bitrate 128k --cache-dir data/codec_cache/mp3_128k

    # dev set도 사전 인코딩 (eval_noise.py용)
    python scripts/03_train/precompute_codec.py manifests/C_all/dev.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from augment import apply_codec

DEFAULT_CACHE_DIR = "data/codec_cache/mp3_64k"


def cache_path_for(file_path: str, cache_dir: Path) -> Path:
    """원본 파일 경로 → 캐시 경로. 원본의 상대 경로 구조를 유지."""
    p = Path(file_path.replace("\\", "/"))
    if p.is_absolute():
        rel = p.relative_to(Path.cwd())
    else:
        rel = p
    return cache_dir / rel


def main() -> int:
    parser = argparse.ArgumentParser(description="Codec 증강 사전 계산")
    parser.add_argument("manifest", help="CSV manifest (file_path, label 컬럼)")
    parser.add_argument("--codec", default="mp3")
    parser.add_argument("--bitrate", default="64k")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--sr", type=int, default=16_000)
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="이미 캐시된 파일 건너뛰기 (기본: True)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    df = pd.read_csv(args.manifest)
    files = df["file_path"].unique().tolist()
    print(f"고유 파일: {len(files)}개")

    skipped = 0
    failed = 0

    for fpath in tqdm(files, desc=f"Codec({args.codec} {args.bitrate})"):
        fpath_norm = fpath.replace("\\", "/")
        out_path = cache_path_for(fpath_norm, cache_dir)

        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        if not Path(fpath_norm).exists():
            failed += 1
            continue

        try:
            wav, sr = sf.read(fpath_norm, dtype="float32")
            if wav.ndim > 1:
                wav = wav[:, 0]

            encoded = apply_codec(wav, sr=args.sr, codec=args.codec, bitrate=args.bitrate)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), encoded, args.sr)
        except Exception as e:
            print(f"\n  실패: {fpath_norm}: {e}")
            failed += 1

    total = len(files)
    processed = total - skipped - failed
    print(f"\n완료: {processed}개 인코딩, {skipped}개 스킵, {failed}개 실패 / 전체 {total}개")
    print(f"캐시 디렉토리: {cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
