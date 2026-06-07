#!/usr/bin/env python3
"""외부 벤치마크 데이터셋 → manifest CSV 변환.

다운로드된 외부 데이터셋을 스캔하여 기존 manifest 형식의 CSV를 생성합니다.
오디오를 16kHz mono WAV로 변환(리샘플)하고, 프로토콜/메타 파일에서 라벨을 읽습니다.

지원 데이터셋:
    T4: ASVspoof 2019 LA eval     — 71,237 samples (영어)
    T5: ASVspoof 2021 DF eval     — 593,253 samples (영어, 코덱 열화)
    T6: ASVspoof 2021 LA eval     — telephony/VoIP (영어)
    T7: CD-ADD                    — 300h+ (중국어, zero-shot TTS)
    T8: InTheWild                 — 37.9h (영어, 실환경)

manifest 형식 (기존과 동일):
    file_path,dataset,label,language,speaker_id,tts_type,duration_sec,original_sr,quality_ok,quality_reason

사용법:
    # 개별 데이터셋
    python scripts/02_preprocess/prepare_benchmarks.py --dataset asvspoof19 --data-dir data/asvspoof19
    python scripts/02_preprocess/prepare_benchmarks.py --dataset asvspoof21_df --data-dir data/asvspoof21
    python scripts/02_preprocess/prepare_benchmarks.py --dataset asvspoof21_la --data-dir data/asvspoof21
    python scripts/02_preprocess/prepare_benchmarks.py --dataset cdadd --data-dir data/cdadd
    python scripts/02_preprocess/prepare_benchmarks.py --dataset inthewild --data-dir data/inthewild

    # 전체 (다운로드된 것만)
    python scripts/02_preprocess/prepare_benchmarks.py --all

    # 리샘플링 건너뛰기 (이미 16kHz인 경우)
    python scripts/02_preprocess/prepare_benchmarks.py --dataset asvspoof19 --data-dir data/asvspoof19 --skip-resample

필요한 패키지:
    pip install soundfile librosa numpy tqdm
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
    import librosa
    from tqdm import tqdm
except ImportError:
    print("필수 패키지를 설치해주세요:")
    print("  pip install soundfile librosa numpy tqdm")
    sys.exit(1)


# ════════════════════════════════════════════════════════════
#  설정
# ════════════════════════════════════════════════════════════

TARGET_SR = 16_000
OUTPUT_DIR = Path("processed")
MANIFEST_DIR = Path("manifests")

FIELDNAMES = [
    "file_path", "dataset", "label", "language", "speaker_id",
    "tts_type", "duration_sec", "original_sr", "quality_ok", "quality_reason",
]

# 데이터셋 → manifest 파일명 매핑
DATASET_MANIFEST = {
    "asvspoof19":   "T4_asvspoof19_la_eval.csv",
    "asvspoof21_df": "T5_asvspoof21_df_eval.csv",
    "asvspoof21_la": "T6_asvspoof21_la_eval.csv",
    "cdadd":        "T7_cdadd.csv",
    "inthewild":    "T8_inthewild.csv",
}

# 데이터셋 → 기본 데이터 경로
DEFAULT_DATA_DIRS = {
    "asvspoof19":   Path("data/asvspoof19"),
    "asvspoof21_df": Path("data/asvspoof21"),
    "asvspoof21_la": Path("data/asvspoof21"),
    "cdadd":        Path("data/cdadd"),
    "inthewild":    Path("data/inthewild"),
}


# ════════════════════════════════════════════════════════════
#  오디오 처리
# ════════════════════════════════════════════════════════════

def process_audio(
    src: Path, dst: Path, skip_resample: bool = False
) -> tuple[float, int]:
    """오디오를 16kHz mono WAV로 변환. (duration_sec, original_sr) 반환."""
    wav, sr = sf.read(str(src), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]  # mono

    original_sr = sr
    duration = len(wav) / sr

    if not skip_resample and sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), wav, sr, subtype="PCM_16")
    return duration, original_sr


# ════════════════════════════════════════════════════════════
#  ASVspoof 2019 LA
# ════════════════════════════════════════════════════════════

def prepare_asvspoof19(data_dir: Path, skip_resample: bool, workers: int) -> list[dict]:
    """ASVspoof 2019 LA eval set → manifest rows."""
    la_dir = data_dir / "LA"
    flac_dir = la_dir / "ASVspoof2019_LA_eval" / "flac"
    protocol = la_dir / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"

    if not protocol.exists():
        print(f"  ⚠ 프로토콜 파일 없음: {protocol}")
        return []

    # 프로토콜 파싱: LA_0069 LA_E_1000147 - A07 spoof
    entries = []
    with open(protocol) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            speaker_id = parts[0]     # LA_0069
            utt_id = parts[1]         # LA_E_1000147
            system_id = parts[3]      # A07 or -
            key = parts[4]            # bonafide or spoof
            entries.append({
                "speaker_id": speaker_id,
                "utt_id": utt_id,
                "system_id": system_id,
                "label": "real" if key == "bonafide" else "fake",
                "tts_type": system_id if key == "spoof" else "none",
            })

    print(f"  ASVspoof 2019 LA eval: {len(entries)} 항목")
    out_dir = OUTPUT_DIR / "asvspoof19_la_eval"

    rows = []
    items = []
    for e in entries:
        src = flac_dir / f"{e['utt_id']}.flac"
        if not src.exists():
            continue
        dst = out_dir / f"{e['utt_id']}.wav"
        items.append((src, dst, e))

    def _proc(item):
        src, dst, e = item
        if dst.exists() and not skip_resample:
            info = sf.info(str(dst))
            dur, orig_sr = info.duration, info.samplerate
        else:
            dur, orig_sr = process_audio(src, dst, skip_resample)
        return {
            "file_path": str(dst),
            "dataset": "asvspoof19_la",
            "label": e["label"],
            "language": "en",
            "speaker_id": e["speaker_id"],
            "tts_type": e["tts_type"],
            "duration_sec": f"{dur:.3f}",
            "original_sr": orig_sr,
            "quality_ok": True,
            "quality_reason": "",
        }

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_proc, item): item for item in items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  ASVspoof19"):
            rows.append(fut.result())

    return rows


# ════════════════════════════════════════════════════════════
#  ASVspoof 2021 DF
# ════════════════════════════════════════════════════════════

def prepare_asvspoof21_df(data_dir: Path, skip_resample: bool, workers: int) -> list[dict]:
    """ASVspoof 2021 DF eval set → manifest rows."""
    df_dir = data_dir / "DF"
    flac_dir = df_dir / "ASVspoof2021_DF_eval" / "flac"

    # keys 디렉토리 탐색 (여러 가능한 경로)
    key_candidates = [
        df_dir / "keys" / "DF" / "CM" / "trial_metadata.txt",
        df_dir / "keys" / "CM" / "trial_metadata.txt",
        df_dir / "trial_metadata.txt",
    ]
    key_file = None
    for kf in key_candidates:
        if kf.exists():
            key_file = kf
            break

    if key_file is None:
        print("  ⚠ ASVspoof 2021 DF eval keys 없음.")
        print("    https://www.asvspoof.org/index2021.html 에서 다운로드 후")
        print(f"    {df_dir}/keys/ 에 배치하세요.")
        return []

    if not flac_dir.exists():
        print(f"  ⚠ FLAC 디렉토리 없음: {flac_dir}")
        return []

    # 프로토콜 파싱 (2021 형식은 복잡 — 최소한 utt_id와 key만 추출)
    entries = {}
    with open(key_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            speaker_id = parts[0]
            utt_id = parts[1]
            key = parts[4]  # bonafide or spoof
            system_id = parts[2] if parts[2] != "-" else "none"
            entries[utt_id] = {
                "speaker_id": speaker_id,
                "label": "real" if key == "bonafide" else "fake",
                "tts_type": system_id,
            }

    print(f"  ASVspoof 2021 DF eval: {len(entries)} 항목")
    out_dir = OUTPUT_DIR / "asvspoof21_df_eval"

    items = []
    for utt_id, e in entries.items():
        src = flac_dir / f"{utt_id}.flac"
        if not src.exists():
            continue
        dst = out_dir / f"{utt_id}.wav"
        items.append((src, dst, utt_id, e))

    rows = []

    def _proc(item):
        src, dst, utt_id, e = item
        if dst.exists():
            info = sf.info(str(dst))
            dur, orig_sr = info.duration, info.samplerate
        else:
            dur, orig_sr = process_audio(src, dst, skip_resample)
        return {
            "file_path": str(dst),
            "dataset": "asvspoof21_df",
            "label": e["label"],
            "language": "en",
            "speaker_id": e["speaker_id"],
            "tts_type": e["tts_type"],
            "duration_sec": f"{dur:.3f}",
            "original_sr": orig_sr,
            "quality_ok": True,
            "quality_reason": "",
        }

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_proc, item): item for item in items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  ASV21-DF"):
            rows.append(fut.result())

    return rows


# ════════════════════════════════════════════════════════════
#  ASVspoof 2021 LA
# ════════════════════════════════════════════════════════════

def prepare_asvspoof21_la(data_dir: Path, skip_resample: bool, workers: int) -> list[dict]:
    """ASVspoof 2021 LA eval set → manifest rows."""
    la_dir = data_dir / "LA"
    flac_dir = la_dir / "ASVspoof2021_LA_eval" / "flac"

    key_candidates = [
        la_dir / "keys" / "LA" / "CM" / "trial_metadata.txt",
        la_dir / "keys" / "CM" / "trial_metadata.txt",
        la_dir / "trial_metadata.txt",
    ]
    key_file = None
    for kf in key_candidates:
        if kf.exists():
            key_file = kf
            break

    if key_file is None:
        print("  ⚠ ASVspoof 2021 LA eval keys 없음.")
        print("    https://www.asvspoof.org/index2021.html 에서 다운로드 후")
        print(f"    {la_dir}/keys/ 에 배치하세요.")
        return []

    if not flac_dir.exists():
        print(f"  ⚠ FLAC 디렉토리 없음: {flac_dir}")
        return []

    entries = {}
    with open(key_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            speaker_id = parts[0]
            utt_id = parts[1]
            key = parts[4]
            system_id = parts[2] if parts[2] != "-" else "none"
            entries[utt_id] = {
                "speaker_id": speaker_id,
                "label": "real" if key == "bonafide" else "fake",
                "tts_type": system_id,
            }

    print(f"  ASVspoof 2021 LA eval: {len(entries)} 항목")
    out_dir = OUTPUT_DIR / "asvspoof21_la_eval"

    items = []
    for utt_id, e in entries.items():
        src = flac_dir / f"{utt_id}.flac"
        if not src.exists():
            continue
        dst = out_dir / f"{utt_id}.wav"
        items.append((src, dst, utt_id, e))

    rows = []

    def _proc(item):
        src, dst, utt_id, e = item
        if dst.exists():
            info = sf.info(str(dst))
            dur, orig_sr = info.duration, info.samplerate
        else:
            dur, orig_sr = process_audio(src, dst, skip_resample)
        return {
            "file_path": str(dst),
            "dataset": "asvspoof21_la",
            "label": e["label"],
            "language": "en",
            "speaker_id": e["speaker_id"],
            "tts_type": e["tts_type"],
            "duration_sec": f"{dur:.3f}",
            "original_sr": orig_sr,
            "quality_ok": True,
            "quality_reason": "",
        }

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_proc, item): item for item in items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  ASV21-LA"):
            rows.append(fut.result())

    return rows


# ════════════════════════════════════════════════════════════
#  CD-ADD
# ════════════════════════════════════════════════════════════

def prepare_cdadd(data_dir: Path, skip_resample: bool, workers: int) -> list[dict]:
    """CD-ADD → manifest rows. 디렉토리 이름으로 라벨 판별."""
    # 가능한 구조: bonafide/ + fake/{model}/ 또는 real/ + fake/
    audio_exts = {".wav", ".flac", ".mp3"}
    items = []

    for f in data_dir.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in audio_exts:
            continue

        rel = f.relative_to(data_dir)
        parts = rel.parts

        # 라벨 판별: 경로에 bonafide/real → real, fake/spoof → fake
        label = None
        tts_type = "none"
        for p in parts:
            pl = p.lower()
            if pl in ("bonafide", "real", "bona_fide"):
                label = "real"
                break
            elif pl in ("fake", "spoof", "spoofed"):
                label = "fake"
                # TTS 모델명 = fake/ 다음 디렉토리
                idx = list(parts).index(p)
                if idx + 1 < len(parts) - 1:  # 파일명 제외
                    tts_type = parts[idx + 1]
                break

        if label is None:
            continue

        out_name = "_".join(rel.parts).replace(" ", "_")
        if not out_name.endswith(".wav"):
            out_name = Path(out_name).stem + ".wav"
        dst = OUTPUT_DIR / "cdadd" / out_name
        items.append((f, dst, label, tts_type))

    print(f"  CD-ADD: {len(items)} 오디오 파일")
    if not items:
        print("  ⚠ 오디오 파일을 찾을 수 없습니다.")
        print(f"    data_dir: {data_dir}")
        return []

    rows = []

    def _proc(item):
        src, dst, label, tts_type = item
        if dst.exists():
            info = sf.info(str(dst))
            dur, orig_sr = info.duration, info.samplerate
        else:
            dur, orig_sr = process_audio(src, dst, skip_resample)
        return {
            "file_path": str(dst),
            "dataset": "cdadd",
            "label": label,
            "language": "zh",
            "speaker_id": "unknown",
            "tts_type": tts_type,
            "duration_sec": f"{dur:.3f}",
            "original_sr": orig_sr,
            "quality_ok": True,
            "quality_reason": "",
        }

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_proc, item): item for item in items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  CD-ADD"):
            rows.append(fut.result())

    return rows


# ════════════════════════════════════════════════════════════
#  InTheWild
# ════════════════════════════════════════════════════════════

def prepare_inthewild(data_dir: Path, skip_resample: bool, workers: int) -> list[dict]:
    """InTheWild → manifest rows. meta.csv에서 라벨 읽기."""
    release_dir = data_dir / "release_in_the_wild"
    meta_csv = release_dir / "meta.csv"

    if not meta_csv.exists():
        # 다른 경로 시도
        for candidate in [data_dir / "meta.csv", data_dir]:
            if (candidate / "meta.csv").exists():
                release_dir = candidate
                meta_csv = candidate / "meta.csv"
                break
        else:
            print(f"  ⚠ meta.csv 없음: {meta_csv}")
            return []

    # meta.csv 파싱: file,speaker,label
    entries = []
    with open(meta_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)

    print(f"  InTheWild: {len(entries)} 항목")
    out_dir = OUTPUT_DIR / "inthewild"

    items = []
    for e in entries:
        src = release_dir / e["file"]
        if not src.exists():
            continue
        dst = out_dir / e["file"]
        if not str(dst).endswith(".wav"):
            dst = dst.with_suffix(".wav")
        items.append((src, dst, e))

    rows = []

    def _proc(item):
        src, dst, e = item
        label_raw = e.get("label", "").lower().strip()
        label = "real" if label_raw in ("bona-fide", "bonafide", "real") else "fake"

        if dst.exists():
            info = sf.info(str(dst))
            dur, orig_sr = info.duration, info.samplerate
        else:
            dur, orig_sr = process_audio(src, dst, skip_resample)
        return {
            "file_path": str(dst),
            "dataset": "inthewild",
            "label": label,
            "language": "en",
            "speaker_id": e.get("speaker", "unknown").replace(" ", "_"),
            "tts_type": "none" if label == "real" else "unknown",
            "duration_sec": f"{dur:.3f}",
            "original_sr": orig_sr,
            "quality_ok": True,
            "quality_reason": "",
        }

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_proc, item): item for item in items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  InTheWild"):
            rows.append(fut.result())

    return rows


# ════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════

PREPARE_FNS = {
    "asvspoof19":   prepare_asvspoof19,
    "asvspoof21_df": prepare_asvspoof21_df,
    "asvspoof21_la": prepare_asvspoof21_la,
    "cdadd":        prepare_cdadd,
    "inthewild":    prepare_inthewild,
}


def write_manifest(rows: list[dict], output_path: Path):
    """manifest CSV 저장."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    # 통계
    n_real = sum(1 for r in rows if r["label"] == "real")
    n_fake = sum(1 for r in rows if r["label"] == "fake")
    print(f"  → {output_path}")
    print(f"    전체: {len(rows)} | real: {n_real} | fake: {n_fake}")


def main():
    parser = argparse.ArgumentParser(description="외부 벤치마크 데이터셋 manifest 생성")
    parser.add_argument("--dataset", choices=list(PREPARE_FNS.keys()),
                        help="처리할 데이터셋")
    parser.add_argument("--data-dir", type=str,
                        help="데이터 디렉토리 (미지정 시 기본 경로)")
    parser.add_argument("--all", action="store_true",
                        help="다운로드된 모든 데이터셋 처리")
    parser.add_argument("--skip-resample", action="store_true",
                        help="리샘플링 건너뛰기 (이미 16kHz인 경우)")
    parser.add_argument("--workers", type=int, default=8,
                        help="병렬 처리 워커 수")
    parser.add_argument("--manifest-dir", type=str, default="manifests",
                        help="manifest 출력 디렉토리")
    args = parser.parse_args()

    global MANIFEST_DIR
    MANIFEST_DIR = Path(args.manifest_dir)

    if not args.dataset and not args.all:
        parser.error("--dataset 또는 --all 중 하나를 지정하세요.")

    datasets = list(PREPARE_FNS.keys()) if args.all else [args.dataset]

    for ds_name in datasets:
        data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIRS[ds_name]
        if not data_dir.exists():
            if args.all:
                print(f"  건너뛰기: {ds_name} (데이터 없음: {data_dir})")
                continue
            else:
                print(f"  ⚠ 데이터 디렉토리 없음: {data_dir}")
                return 1

        print(f"\n{'='*60}")
        print(f"  {ds_name}")
        print(f"{'='*60}")

        prepare_fn = PREPARE_FNS[ds_name]
        rows = prepare_fn(data_dir, args.skip_resample, args.workers)

        if rows:
            manifest_path = MANIFEST_DIR / DATASET_MANIFEST[ds_name]
            write_manifest(rows, manifest_path)
        else:
            print(f"  결과 없음. 데이터를 확인하세요.")

    print("\n  완료!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
