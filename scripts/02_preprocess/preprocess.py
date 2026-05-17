#!/usr/bin/env python3
"""Phase 1.2: 전처리 파이프라인

원본 오디오를 스캔하여 메타데이터를 생성하고, 16kHz mono WAV로 변환하며, 품질 검증을 수행합니다.

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
SPEECHFAKE_ROOT = Path(r"D:\speechfake")
OUTPUT_ROOT = Path(r"processed")
METADATA_DIR = Path(r"metadata")

# ════════════════════════════════════════════════════════════
#  전처리 파라미터
# ════════════════════════════════════════════════════════════

TARGET_SR = 16_000
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
for _name, _lang in [
    ("aishell", "zh"), ("Aishell", "zh"), ("Aishell3", "zh"), ("aishell3", "zh"),
    ("LibriTTS", "en"), ("LibriTTS_train-clean-100_16k", "en"),
    ("VCTK", "en"), ("VCTK_Corpus", "en"), ("VCTK-Corpus", "en"),
    ("CommonVoice", "multi"),
]:
    CORPUS_LANG[_name] = _lang


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


def scan_aihub(data_root: Path) -> Iterator[AudioMeta]:
    """data/aihub/Training/01.원천데이터/TS_{체}_{번호}/*.wav"""
    aihub_root = data_root / "aihub"
    if not aihub_root.exists():
        logger.warning("AI Hub 경로 없음: %s", aihub_root)
        return

    for f in _find_audio(aihub_root):
        speaker_id = "unknown"
        for p in f.relative_to(aihub_root).parts:
            if p.startswith(("TS_", "VS_")):
                speaker_id = p
                break

        yield AudioMeta(
            file_path=str(f),
            dataset="aihub",
            label="real",
            language="ko",
            speaker_id=speaker_id,
            tts_type="none",
        )


def scan_speechfake_bd(data_root: Path) -> Iterator[AudioMeta]:
    """data/speechfake-BD/{tts_system}/{corpus}[/{voice}]/*.wav"""
    bd_root = data_root / "speechfake-BD"
    if not bd_root.exists():
        logger.warning("SpeechFake-BD 경로 없음: %s", bd_root)
        return

    for f in _find_audio(bd_root):
        parts = f.relative_to(bd_root).parts
        tts_system = parts[0] if len(parts) > 1 else "unknown"
        corpus = parts[1] if len(parts) > 2 else "unknown"
        lang = CORPUS_LANG.get(corpus, "unknown")

        if len(parts) > 3:
            speaker_id = f"sfbd_{tts_system}_{parts[2]}"
        else:
            speaker_id = f"sfbd_{tts_system}_{corpus}"

        yield AudioMeta(
            file_path=str(f),
            dataset="speechfake-bd",
            label="fake",
            language=lang,
            speaker_id=speaker_id,
            tts_type=tts_system,
        )


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


def scan_speechfake_md(sf_root: Path) -> Iterator[AudioMeta]:
    """D:/speechfake/MD/{tts_system}/{lang}[/{voice}]/*.wav"""
    md_root = sf_root / "MD"
    if not md_root.exists():
        logger.warning("SpeechFake-MD 경로 없음: %s", md_root)
        return

    for f in _find_audio(md_root):
        parts = f.relative_to(md_root).parts
        tts_system = parts[0] if len(parts) > 1 else "unknown"

        raw_lang = parts[1] if len(parts) > 2 else "unknown"
        lang = raw_lang.split("_")[-1] if "_" in raw_lang else raw_lang

        if len(parts) > 3:
            voice = parts[2]
            speaker_id = f"sfmd_{tts_system}_{voice}"
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


def scan_speechfake_real(sf_root: Path) -> Iterator[AudioMeta]:
    """D:/speechfake/Real/{corpus}[/{split}]/{speaker}/*.wav"""
    real_root = sf_root / "Real"
    if not real_root.exists():
        logger.warning("SpeechFake-Real 경로 없음: %s", real_root)
        return

    for f in _find_audio(real_root):
        parts = f.relative_to(real_root).parts
        corpus = parts[0] if parts else "unknown"
        lang = CORPUS_LANG.get(corpus, "unknown")

        speaker_id = "unknown"
        skip = {"train", "dev", "test", "wav", "wavs"}
        for p in parts[1:]:
            if p not in skip and not p.endswith((".wav", ".mp3")):
                speaker_id = f"sfreal_{corpus}_{p}"
                break

        yield AudioMeta(
            file_path=str(f),
            dataset="speechfake-real",
            label="real",
            language=lang,
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
    sf_root = Path(args.speechfake_root) if args.speechfake_root else SPEECHFAKE_ROOT
    meta_dir = Path(args.meta_dir) if args.meta_dir else METADATA_DIR
    meta_dir.mkdir(parents=True, exist_ok=True)

    out_path = meta_dir / "raw_metadata.csv"
    logger.info("스캔 시작: data_root=%s, speechfake_root=%s", data_root, sf_root)

    scanners = [
        ("AI Hub (한국어 실제)", scan_aihub, (data_root,)),
        ("SpeechFake-BD (가짜)", scan_speechfake_bd, (data_root,)),
        ("M-AILABS (다국어 실제)", scan_mailabs, (data_root,)),
        ("MLAAD v9 (다국어 가짜)", scan_mlaad, (data_root,)),
        ("SpeechFake-MD (다국어 가짜)", scan_speechfake_md, (sf_root,)),
        ("SpeechFake-Real (실제)", scan_speechfake_real, (sf_root,)),
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


def cmd_process(args: argparse.Namespace) -> int:
    """raw_metadata.csv를 읽고 리샘플링 + 품질 검증 수행."""
    if not HAS_AUDIO_LIBS:
        logger.error("필요한 패키지를 설치하세요: pip install soundfile librosa numpy tqdm")
        return 1

    meta_dir = Path(args.meta_dir) if args.meta_dir else METADATA_DIR
    output_root = Path(args.output_root) if args.output_root else OUTPUT_ROOT
    raw_csv = meta_dir / "raw_metadata.csv"
    processed_csv = meta_dir / "processed_metadata.csv"

    if not raw_csv.exists():
        logger.error("raw_metadata.csv가 없습니다. 먼저 scan을 실행하세요.")
        return 1

    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(raw_csv, "r", encoding="utf-8") as f:
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
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row["dataset"]
            label = row["label"]
            if ds not in counts:
                counts[ds] = {"real": 0, "fake": 0}
            counts[ds][label] += 1

    print("\n  데이터셋별 분포:")
    total_real = 0
    total_fake = 0
    for ds in sorted(counts):
        r, fk = counts[ds]["real"], counts[ds]["fake"]
        total_real += r
        total_fake += fk
        print(f"    {ds:20s}  real={r:>10,}  fake={fk:>10,}")
    print(f"    {'합계':20s}  real={total_real:>10,}  fake={total_fake:>10,}")


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
    p_scan.add_argument("--speechfake-root", default=None, help=f"speechfake 경로 (기본: {SPEECHFAKE_ROOT})")
    p_scan.add_argument("--meta-dir", default=None, help=f"메타데이터 출력 경로 (기본: {METADATA_DIR})")

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
    elif args.command == "process":
        return cmd_process(args)
    elif args.command == "report":
        return cmd_report(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
