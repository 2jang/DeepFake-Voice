#!/usr/bin/env python3
"""Phase 1.2: 전처리 파이프라인

원본 오디오를 스캔하여 메타데이터를 생성하고, 16kHz mono WAV로 변환하며, 품질 검증을 수행합니다.

데이터셋 (연구계획서 v5):
    - MLAAD v9: 다국어 딥페이크 (data/mlaad/)
    - M-AILABS: 다국어 실제 음성 (data/mailabs/)
    - SpeechFake: 한국어 딥페이크+실제 (data/speechfake/MD/, data/speechfake/Real/)
    - KoAAD: 한국어 딥페이크 (data/koaad/) + 실제 (data/aihub/)

사용법:
    # 1단계: 스캔 — 메타데이터 CSV 생성 (오디오 처리 없음, 빠름)
    python scripts/02_preprocess/preprocess.py scan

    # 2단계: 처리 — 리샘플링 + 품질 검증 (병렬 처리)
    python scripts/02_preprocess/preprocess.py process --workers 8

    # 3단계: 통계 리포트
    python scripts/02_preprocess/preprocess.py report

필요한 패키지:
    pip install soundfile librosa numpy tqdm pandas
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterator

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("preprocess")

try:
    import soundfile  # noqa: F401
    import librosa  # noqa: F401
    import numpy  # noqa: F401
    from tqdm import tqdm
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    tqdm = None

# ════════════════════════════════════════════════════════════
#  경로 설정 — 실행 환경에 맞게 수정하세요
# ════════════════════════════════════════════════════════════

DATA_ROOT = Path(r"data")
OUTPUT_ROOT = Path(r"processed")
METADATA_DIR = Path(r"metadata")

# ════════════════════════════════════════════════════════════
#  전처리 파라미터
# ════════════════════════════════════════════════════════════

TARGET_SR = 16_000
SEED = 42
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

SILENCE_THRESHOLD_RMS = 0.005
MAX_SILENCE_RATIO = 0.95
CLIP_THRESHOLD = 0.999
MAX_CLIP_RATIO = 0.01
MIN_DURATION_SEC = 0.5

# ════════════════════════════════════════════════════════════
#  코퍼스 → 언어 매핑
# ════════════════════════════════════════════════════════════

CORPUS_LANG: dict[str, str] = {}


# ════════════════════════════════════════════════════════════
#  메타데이터 구조
# ════════════════════════════════════════════════════════════

@dataclass
class AudioMeta:
    file_path: str
    dataset: str
    label: str       # real / fake
    language: str
    speaker_id: str
    tts_type: str    # none for real speech
    duration_sec: float = 0.0
    original_sr: int = 0
    quality_ok: bool = True
    quality_reason: str = ""

    @property
    def fieldnames(self) -> list[str]:
        return list(asdict(self).keys())


FIELDNAMES = [
    "file_path", "dataset", "label", "language", "speaker_id",
    "tts_type", "duration_sec", "original_sr", "quality_ok", "quality_reason",
]


# ════════════════════════════════════════════════════════════
#  1. 데이터 소스별 스캐너
# ════════════════════════════════════════════════════════════

def _find_audio(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
            yield f


def scan_mailabs(data_root: Path) -> Iterator[AudioMeta]:
    """data/mailabs/{locale}/[by_book/]{gender}/{speaker}/{book}/wavs/*.wav"""
    mailabs_root = data_root / "mailabs"
    if not mailabs_root.exists():
        logger.warning("M-AILABS 경로 없음: %s", mailabs_root)
        return

    for f in _find_audio(mailabs_root):
        parts = f.relative_to(mailabs_root).parts
        locale = parts[0]
        lang = locale.split("_")[0]

        speaker_id = "unknown"
        for i, p in enumerate(parts):
            if p in ("female", "male", "mix") and i + 1 < len(parts):
                speaker_id = parts[i + 1]
                break

        yield AudioMeta(
            file_path=str(f),
            dataset="mailabs",
            label="real",
            language=lang,
            speaker_id=f"mailabs_{lang}_{speaker_id}",
            tts_type="none",
        )


def scan_mlaad(data_root: Path) -> Iterator[AudioMeta]:
    """data/mlaad/fake/{lang}/{tts_model}/*.{wav,mp3}"""
    mlaad_root = data_root / "mlaad"
    if not mlaad_root.exists():
        logger.warning("MLAAD 경로 없음: %s", mlaad_root)
        return

    for f in _find_audio(mlaad_root):
        parts = f.relative_to(mlaad_root).parts
        if len(parts) < 4 or parts[0] != "fake":
            continue
        lang = parts[1]
        tts_model = parts[2]

        yield AudioMeta(
            file_path=str(f),
            dataset="mlaad",
            label="fake",
            language=lang,
            speaker_id=f"mlaad_{lang}_{tts_model}",
            tts_type=tts_model,
        )


def scan_speechfake(data_root: Path) -> Iterator[AudioMeta]:
    """data/speechfake/MD/{tts_system}/{lang}/*.wav — fake
    data/speechfake/Real/{corpus}/{lang}/**/*.wav — real"""
    sf_root = data_root / "speechfake"
    if not sf_root.exists():
        logger.warning("SpeechFake 경로 없음: %s", sf_root)
        return

    md_root = sf_root / "MD"
    if md_root.exists():
        for f in _find_audio(md_root):
            parts = f.relative_to(md_root).parts
            tts_system = parts[0] if len(parts) > 1 else "unknown"

            raw_lang = parts[1] if len(parts) > 2 else "unknown"
            lang = raw_lang.split("-")[0] if "-" in raw_lang else raw_lang

            if len(parts) > 3:
                speaker_id = f"sfmd_{tts_system}_{parts[2]}"
            else:
                speaker_id = f"sfmd_{tts_system}_{lang}"

            yield AudioMeta(
                file_path=str(f),
                dataset="speechfake-md",
                label="fake",
                language=lang,
                speaker_id=speaker_id,
                tts_type=tts_system,
            )

    real_root = sf_root / "Real"
    if real_root.exists():
        for f in _find_audio(real_root):
            parts = f.relative_to(real_root).parts
            corpus = parts[0] if parts else "unknown"

            lang = "unknown"
            for p in parts:
                if len(p) == 2 or (len(p) == 5 and p[2] == "_"):
                    lang = p[:2]
                    break

            speaker_id = f"sfreal_{corpus}_{f.stem}"

            yield AudioMeta(
                file_path=str(f),
                dataset="speechfake-real",
                label="real",
                language=lang,
                speaker_id=speaker_id,
                tts_type="none",
            )


def scan_koaad(data_root: Path) -> Iterator[AudioMeta]:
    """data/koaad/ — 한국어 딥페이크+실제 음성.
    폴더 구조에 따라 real/fake 판별."""
    koaad_root = data_root / "koaad"
    if not koaad_root.exists():
        logger.warning("KoAAD 경로 없음: %s", koaad_root)
        return

    for f in _find_audio(koaad_root):
        parts = f.relative_to(koaad_root).parts
        folder = parts[0] if parts else ""

        speaker_id = f"koaad_{folder}" if folder else f"koaad_{f.stem}"

        yield AudioMeta(
            file_path=str(f),
            dataset="koaad",
            label="fake",
            language="ko",
            speaker_id=speaker_id,
            tts_type=folder or "unknown",
        )


def scan_aihub(data_root: Path) -> Iterator[AudioMeta]:
    """data/aihub/Training/01.원천데이터/TS_{style}_{id}/*.wav
    data/aihub/Validation/01.원천데이터/VS_{style}_{id}/*.wav
    KoAAD의 real 파트 — 한국어 실제 음성."""
    aihub_root = data_root / "aihub"
    if not aihub_root.exists():
        logger.warning("AIHub 경로 없음: %s", aihub_root)
        return

    for f in _find_audio(aihub_root):
        parts = f.relative_to(aihub_root).parts

        speaker_folder = ""
        for p in parts:
            if p.startswith(("TS_", "VS_")):
                speaker_folder = p
                break

        speaker_id = f"koaad_aihub_{speaker_folder}" if speaker_folder else f"koaad_aihub_{f.stem}"

        yield AudioMeta(
            file_path=str(f),
            dataset="koaad",
            label="real",
            language="ko",
            speaker_id=speaker_id,
            tts_type="none",
        )


# ════════════════════════════════════════════════════════════
#  2. 품질 검증
# ════════════════════════════════════════════════════════════

def check_quality(audio, sr: int) -> tuple[bool, str]:
    import numpy as np

    if len(audio) == 0:
        return False, "empty"

    duration = len(audio) / sr
    if duration < MIN_DURATION_SEC:
        return False, f"too_short:{duration:.2f}s"

    rms = np.sqrt(np.mean(audio ** 2))
    if rms < SILENCE_THRESHOLD_RMS:
        return False, "near_silent"

    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    silent_frames = 0
    total_frames = 0
    for start in range(0, len(audio) - frame_len, hop):
        frame = audio[start:start + frame_len]
        frame_rms = np.sqrt(np.mean(frame ** 2))
        total_frames += 1
        if frame_rms < SILENCE_THRESHOLD_RMS:
            silent_frames += 1
    if total_frames > 0 and (silent_frames / total_frames) > MAX_SILENCE_RATIO:
        return False, f"silence_ratio:{silent_frames / total_frames:.2f}"

    clip_count = int(np.sum(np.abs(audio) > CLIP_THRESHOLD))
    clip_ratio = clip_count / len(audio)
    if clip_ratio > MAX_CLIP_RATIO:
        return False, f"clipping:{clip_ratio:.4f}"

    return True, ""


# ════════════════════════════════════════════════════════════
#  3. 오디오 처리 (리샘플링 + 품질 검증 + 저장)
# ════════════════════════════════════════════════════════════

def process_single(row: dict, output_root: Path) -> dict:
    """단일 파일 처리: 리샘플링 → 품질 검증 → 저장."""
    import librosa
    import soundfile as sf
    import numpy as np

    src = Path(row["file_path"])
    if not src.exists():
        row["quality_ok"] = "False"
        row["quality_reason"] = "file_not_found"
        return row

    try:
        audio, sr = librosa.load(str(src), sr=TARGET_SR, mono=True)
    except Exception as e:
        row["quality_ok"] = "False"
        row["quality_reason"] = f"load_error:{e}"
        return row

    row["original_sr"] = str(sr)
    row["duration_sec"] = f"{len(audio) / TARGET_SR:.3f}"

    ok, reason = check_quality(audio, TARGET_SR)
    if not ok:
        row["quality_ok"] = "False"
        row["quality_reason"] = reason
        return row

    try:
        src_path = Path(row["file_path"])
        rel = src_path.name
        dataset = row["dataset"]
        speaker = row["speaker_id"]
        out_dir = output_root / dataset / speaker
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = src_path.stem
        out_path = out_dir / f"{stem}.wav"

        counter = 1
        while out_path.exists():
            out_path = out_dir / f"{stem}_{counter}.wav"
            counter += 1

        sf.write(str(out_path), audio, TARGET_SR)
        row["file_path"] = str(out_path)
        row["quality_ok"] = "True"
        row["quality_reason"] = ""
    except Exception as e:
        row["quality_ok"] = "False"
        row["quality_reason"] = f"save_error:{e}"

    return row


def _process_wrapper(args):
    return process_single(*args)


def _process_batch(args):
    rows, output_root = args
    return [process_single(row, output_root) for row in rows]


# ════════════════════════════════════════════════════════════
#  CLI 명령어
# ════════════════════════════════════════════════════════════

def cmd_scan(args: argparse.Namespace) -> int:
    """모든 데이터 소스를 스캔하고 raw_metadata.csv 생성."""
    data_root = Path(args.data_root) if args.data_root else DATA_ROOT
    meta_dir = Path(args.meta_dir) if args.meta_dir else METADATA_DIR
    meta_dir.mkdir(parents=True, exist_ok=True)

    out_path = meta_dir / "raw_metadata.csv"
    logger.info("스캔 시작: data_root=%s", data_root)

    scanners = [
        ("MLAAD v9 (다국어 가짜)", scan_mlaad, (data_root,)),
        ("M-AILABS (다국어 실제)", scan_mailabs, (data_root,)),
        ("SpeechFake (한국어 가짜+실제)", scan_speechfake, (data_root,)),
        ("KoAAD (한국어 가짜)", scan_koaad, (data_root,)),
        ("AIHub (한국어 실제)", scan_aihub, (data_root,)),
    ]

    total = 0
    stats: dict[str, dict[str, int]] = {}

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for name, scanner, scanner_args in scanners:
            count = 0
            print(f"\n  스캔 중: {name} ...", end="", flush=True)
            for meta in scanner(*scanner_args):
                writer.writerow(asdict(meta))
                count += 1
                if count % 100_000 == 0:
                    print(f"\r  스캔 중: {name} ... {count:,}개", end="", flush=True)

            print(f"\r  스캔 완료: {name} — {count:,}개")
            ds = meta.dataset if count > 0 else name
            if ds not in stats:
                stats[ds] = {"real": 0, "fake": 0}
            total += count

    print(f"\n{'='*60}")
    print(f"  전체 파일 수: {total:,}개")
    print(f"  메타데이터 저장: {out_path}")
    print(f"{'='*60}")

    _print_scan_summary(out_path)
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    """raw_metadata.csv에서 조합별 필요량 기반 샘플링 → sampled_metadata.csv"""
    import pandas as pd

    meta_dir = Path(args.meta_dir) if args.meta_dir else METADATA_DIR
    raw_csv = meta_dir / "raw_metadata.csv"
    if not raw_csv.exists():
        print("오류: raw_metadata.csv가 없습니다. 먼저 scan을 실행하세요.")
        return 1

    seed = args.seed
    buffer = args.buffer

    df = pd.read_csv(raw_csv)
    print(f"\n{'='*60}")
    print(f"  스마트 샘플링")
    print(f"  원본: {len(df):,}개  (real={len(df[df['label']=='real']):,}, fake={len(df[df['label']=='fake']):,})")
    print(f"  버퍼: {buffer:.0%}")
    print(f"{'='*60}")

    # 조합별 fake 총량 계산 → real 캡 결정
    c2_fake = len(df[df["dataset"] == "mlaad"]) + \
              len(df[(df["dataset"] == "speechfake-md") & (df["language"] == "ko")])
    c3_fake = len(df[(df["dataset"] == "speechfake-md") & (df["language"] == "ko")]) + \
              len(df[(df["dataset"] == "koaad") & (df["label"] == "fake")])

    # mailabs: C2에서 가장 많이 필요 (C2 fake 기준)
    # koaad real (aihub): C3에서 필요 (C3 fake 기준)
    caps = {
        ("mailabs", "real"): int(c2_fake * buffer),
        ("koaad", "real"): int(c3_fake * buffer),
    }

    print(f"\n  조합별 fake 규모:")
    print(f"    C2 fake: {c2_fake:,} → mailabs real 캡: {caps[('mailabs', 'real')]:,}")
    print(f"    C3 fake: {c3_fake:,} → koaad real 캡: {caps[('koaad', 'real')]:,}")

    # 샘플링 실행
    sampled_parts = []
    print(f"\n  샘플링 결과:")
    for (ds, label), group in df.groupby(["dataset", "label"]):
        cap = caps.get((ds, label))
        if cap and len(group) > cap:
            sampled = group.sample(n=cap, random_state=seed)
            print(f"    {ds:20s} {label:5s}: {len(group):>10,} → {len(sampled):>10,}")
        else:
            sampled = group
            print(f"    {ds:20s} {label:5s}: {len(group):>10,}    (유지)")
        sampled_parts.append(sampled)

    result = pd.concat(sampled_parts).sample(frac=1, random_state=seed)

    out_csv = meta_dir / "sampled_metadata.csv"
    result.to_csv(out_csv, index=False)

    n_real = len(result[result["label"] == "real"])
    n_fake = len(result[result["label"] == "fake"])
    saved = len(df) - len(result)

    print(f"\n  샘플링 후: {len(result):,}개 (real={n_real:,}, fake={n_fake:,})")
    print(f"  절감: {saved:,}개 ({saved/len(df):.0%})")
    print(f"  저장: {out_csv}")
    print(f"{'='*60}\n")

    return 0


def cmd_process(args: argparse.Namespace) -> int:
    """sampled/raw_metadata.csv를 읽고 리샘플링 + 품질 검증 수행."""
    if not HAS_AUDIO_LIBS:
        logger.error("필요한 패키지를 설치하세요: pip install soundfile librosa numpy tqdm")
        return 1

    meta_dir = Path(args.meta_dir) if args.meta_dir else METADATA_DIR
    output_root = Path(args.output_root) if args.output_root else OUTPUT_ROOT

    input_csv = meta_dir / "sampled_metadata.csv"
    if not input_csv.exists():
        input_csv = meta_dir / "raw_metadata.csv"
    processed_csv = meta_dir / "processed_metadata.csv"

    if not input_csv.exists():
        logger.error("메타데이터 CSV가 없습니다. 먼저 scan (→ sample)을 실행하세요.")
        return 1

    print(f"\n  입력: {input_csv.name}")

    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    logger.info("처리 대상: %d개 파일, workers=%d", len(rows), args.workers)
    print(f"\n  처리 대상: {len(rows):,}개 파일")
    print(f"  출력 경로: {output_root}")
    print(f"  병렬 워커: {args.workers}")

    results = []
    batch_size = max(1, len(rows) // (args.workers * 4))
    batch_size = min(batch_size, 200)
    batches = []
    for i in range(0, len(rows), batch_size):
        batches.append((rows[i:i + batch_size], output_root))

    print(f"  배치 크기: {batch_size} (총 {len(batches)}개 배치)")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_batch, batch): i for i, batch in enumerate(batches)}
        with tqdm(total=len(rows), desc="처리 중", unit="file") as pbar:
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    pbar.update(len(batch_results))
                except Exception as e:
                    idx = futures[future]
                    batch_rows = batches[idx][0]
                    for row in batch_rows:
                        failed = row.copy()
                        failed["quality_ok"] = "False"
                        failed["quality_reason"] = f"worker_error:{e}"
                        results.append(failed)
                    pbar.update(len(batch_rows))

    with open(processed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    ok_count = sum(1 for r in results if r.get("quality_ok") == "True")
    fail_count = len(results) - ok_count

    print(f"\n{'='*60}")
    print(f"  처리 완료")
    print(f"  통과: {ok_count:,}개")
    print(f"  제외: {fail_count:,}개")
    print(f"  결과 저장: {processed_csv}")
    print(f"{'='*60}")

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """처리 결과 통계 리포트."""
    import pandas as pd

    meta_dir = Path(args.meta_dir) if args.meta_dir else METADATA_DIR

    for csv_name in ["processed_metadata.csv", "raw_metadata.csv"]:
        csv_path = meta_dir / csv_name
        if csv_path.exists():
            break
    else:
        logger.error("메타데이터 CSV가 없습니다. scan 또는 process를 먼저 실행하세요.")
        return 1

    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"  데이터 리포트 — {csv_path.name}")
    print(f"{'='*60}")

    print(f"\n  전체 파일 수: {len(df):,}")

    print("\n  ── 데이터셋별 ──")
    ds_stats = df.groupby(["dataset", "label"]).size().unstack(fill_value=0)
    print(ds_stats.to_string(col_space=10))

    print("\n  ── 언어별 ──")
    lang_stats = df.groupby(["language", "label"]).size().unstack(fill_value=0)
    print(lang_stats.to_string(col_space=10))

    if "quality_ok" in df.columns:
        print("\n  ── 품질 검증 ──")
        qc = df["quality_ok"].value_counts()
        print(f"    통과: {qc.get(True, qc.get('True', 0)):,}")
        print(f"    제외: {qc.get(False, qc.get('False', 0)):,}")

        fails = df[df["quality_ok"].astype(str) == "False"]
        if len(fails) > 0:
            print("\n    제외 사유:")
            for reason, count in fails["quality_reason"].value_counts().head(10).items():
                print(f"      {reason}: {count:,}")

    if "duration_sec" in df.columns:
        dur = pd.to_numeric(df["duration_sec"], errors="coerce")
        valid_dur = dur.dropna()
        if len(valid_dur) > 0:
            print(f"\n  ── 길이 통계 (초) ──")
            print(f"    총 시간: {valid_dur.sum() / 3600:.1f}h")
            print(f"    평균: {valid_dur.mean():.2f}s")
            print(f"    중앙값: {valid_dur.median():.2f}s")
            print(f"    최소: {valid_dur.min():.2f}s / 최대: {valid_dur.max():.2f}s")

    ko_df = df[df["language"] == "ko"]
    if len(ko_df) > 0:
        print(f"\n  ── 한국어 상세 ──")
        ko_stats = ko_df.groupby(["dataset", "label"]).size().unstack(fill_value=0)
        print(ko_stats.to_string(col_space=10))
        ko_real = len(ko_df[ko_df["label"] == "real"])
        ko_fake = len(ko_df[ko_df["label"] == "fake"])
        print(f"    한국어 실제: {ko_real:,} / 가짜: {ko_fake:,}")

    print(f"\n  ── TTS 시스템별 (fake만) ──")
    fake_df = df[df["label"] == "fake"]
    if len(fake_df) > 0:
        tts_counts = fake_df.groupby("tts_type").size().sort_values(ascending=False)
        for tts, cnt in tts_counts.head(20).items():
            print(f"    {tts}: {cnt:,}")
        if len(tts_counts) > 20:
            print(f"    ... 외 {len(tts_counts) - 20}개 시스템")

    print(f"\n{'='*60}\n")
    return 0


def _print_scan_summary(csv_path: Path) -> None:
    """스캔 직후 간단한 요약 출력."""
    counts: dict[str, dict[str, int]] = {}
    lang_counts: dict[str, dict[str, int]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row["dataset"]
            label = row["label"]
            lang = row.get("language", "unknown")

            if ds not in counts:
                counts[ds] = {"real": 0, "fake": 0, "unknown": 0}
            if label in counts[ds]:
                counts[ds][label] += 1
            else:
                counts[ds]["unknown"] += 1

            if lang not in lang_counts:
                lang_counts[lang] = {"real": 0, "fake": 0}
            if label in lang_counts[lang]:
                lang_counts[lang][label] += 1

    print("\n  데이터셋별 분포:")
    total_real = 0
    total_fake = 0
    total_unknown = 0
    for ds in sorted(counts):
        r = counts[ds]["real"]
        fk = counts[ds]["fake"]
        unk = counts[ds]["unknown"]
        total_real += r
        total_fake += fk
        total_unknown += unk
        line = f"    {ds:20s}  real={r:>10,}  fake={fk:>10,}"
        if unk > 0:
            line += f"  unknown={unk:>10,}"
        print(line)
    line = f"    {'합계':20s}  real={total_real:>10,}  fake={total_fake:>10,}"
    if total_unknown > 0:
        line += f"  unknown={total_unknown:>10,}"
    print(line)

    ko = lang_counts.get("ko", {"real": 0, "fake": 0})
    non_ko_real = total_real - ko["real"]
    non_ko_fake = total_fake - ko["fake"]
    print(f"\n  언어별 분포:")
    print(f"    {'한국어 (ko)':20s}  real={ko['real']:>10,}  fake={ko['fake']:>10,}")
    print(f"    {'기타 언어':20s}  real={non_ko_real:>10,}  fake={non_ko_fake:>10,}")


# ════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1.2: 전처리 파이프라인 — 리샘플링, 품질 검증, 메타데이터 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # scan
    p_scan = sub.add_parser("scan", help="데이터 소스 스캔 → raw_metadata.csv 생성")
    p_scan.add_argument("--data-root", default=None, help=f"data/ 경로 (기본: {DATA_ROOT})")
    p_scan.add_argument("--meta-dir", default=None, help=f"메타데이터 출력 경로 (기본: {METADATA_DIR})")

    # sample
    p_sample = sub.add_parser("sample", help="조합별 필요량 기반 샘플링 → sampled_metadata.csv")
    p_sample.add_argument("--meta-dir", default=None, help=f"메타데이터 경로 (기본: {METADATA_DIR})")
    p_sample.add_argument("--seed", type=int, default=SEED, help="랜덤 시드 (기본: 42)")
    p_sample.add_argument("--buffer", type=float, default=1.3, help="버퍼 비율 (기본: 1.3 = 30%%)")

    # process
    p_proc = sub.add_parser("process", help="리샘플링 + 품질 검증 → processed_metadata.csv")
    p_proc.add_argument("--workers", type=int, default=4, help="병렬 워커 수 (기본: 4)")
    p_proc.add_argument("--output-root", default=None, help=f"처리된 파일 저장 경로 (기본: {OUTPUT_ROOT})")
    p_proc.add_argument("--meta-dir", default=None, help=f"메타데이터 경로 (기본: {METADATA_DIR})")

    # report
    p_rep = sub.add_parser("report", help="통계 리포트 출력")
    p_rep.add_argument("--meta-dir", default=None, help=f"메타데이터 경로 (기본: {METADATA_DIR})")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "scan":
        return cmd_scan(args)
    elif args.command == "sample":
        return cmd_sample(args)
    elif args.command == "process":
        return cmd_process(args)
    elif args.command == "report":
        return cmd_report(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
