"""
MLAAD v9 다운로드 스크립트
=========================
Multi-Language Audio Anti-Spoofing Dataset v9
- 출처: https://huggingface.co/datasets/mueller91/MLAAD
- 규모: 109GB (전체), 51개 언어, 140개 TTS, 687.4시간
- 라이선스: CC-BY-NC 4.0

사용법:
    # 전체 다운로드
    python scripts/download_mlaad.py --output data/mlaad

    # 한국어만 다운로드
    python scripts/download_mlaad.py --output data/mlaad --languages ko

    # 한국어 + 영어 + 일본어
    python scripts/download_mlaad.py --output data/mlaad --languages ko en ja

    # tiny 버전 (테스트용)
    python scripts/download_mlaad.py --output data/mlaad --tiny

    # HuggingFace 미러 사용 (중국/느린 네트워크)
    HF_ENDPOINT=https://hf-mirror.com python scripts/download_mlaad.py --output data/mlaad

사전 준비:
    pip install huggingface_hub tqdm
    huggingface-cli login
"""

import argparse
import os
import sys
import csv
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from tqdm import tqdm
except ImportError:
    print("필수 패키지를 설치해주세요:")
    print("  pip install huggingface_hub tqdm")
    sys.exit(1)


REPO_FULL = "mueller91/MLAAD"
REPO_TINY = "mueller91/MLAAD-tiny"


def get_file_list(repo_id: str) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    return list(files)


def filter_files_by_languages(files: list[str], languages: list[str]) -> list[str]:
    """fake/<language>/... 구조에서 특정 언어 폴더만 필터링."""
    filtered = []
    lang_set = set(languages)
    for f in files:
        parts = f.split("/")
        # fake/<lang>/... 구조
        if len(parts) >= 2 and parts[0] == "fake" and parts[1] in lang_set:
            filtered.append(f)
        # 메타데이터, README 등 루트 파일은 항상 포함
        if "/" not in f:
            filtered.append(f)
    return filtered


def download_full(repo_id: str, output_dir: str, languages: list[str] | None = None,
                  max_workers: int = 4):
    """snapshot_download로 전체(또는 필터링된) 데이터셋 다운로드."""
    allow_patterns = None
    if languages:
        allow_patterns = []
        for lang in languages:
            allow_patterns.append(f"fake/{lang}/**")
        allow_patterns.append("*.md")
        allow_patterns.append("*.csv")
        allow_patterns.append("*.txt")

    print(f"다운로드 시작: {repo_id}")
    if languages:
        print(f"필터링 언어: {languages}")
    print(f"저장 경로: {output_dir}")
    print()

    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=output_dir,
        allow_patterns=allow_patterns,
        max_workers=max_workers,
    )
    return path


def download_selective(repo_id: str, output_dir: str, languages: list[str],
                       max_workers: int = 4):
    """파일 목록을 먼저 조회 후 선택적 다운로드 (세밀한 제어용)."""
    print("파일 목록 조회 중...")
    all_files = get_file_list(repo_id)
    print(f"전체 파일 수: {len(all_files)}")

    target_files = filter_files_by_languages(all_files, languages)
    print(f"다운로드 대상 ({languages}): {target_files.__len__()}개 파일")

    if not target_files:
        print(f"경고: 언어 '{languages}'에 해당하는 파일이 없습니다.")
        print("사용 가능한 언어 목록:")
        langs = set()
        for f in all_files:
            parts = f.split("/")
            if len(parts) >= 2 and parts[0] == "fake":
                langs.add(parts[1])
        for lang in sorted(langs):
            print(f"  - {lang}")
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    failed = []

    def _download_one(filename: str):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=str(output_path),
            )
            return filename, True
        except Exception as e:
            return filename, False

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_one, f): f for f in target_files}
        with tqdm(total=len(target_files), desc="다운로드 진행", unit="file") as pbar:
            for future in as_completed(futures):
                fname, success = future.result()
                if not success:
                    failed.append(fname)
                pbar.update(1)

    if failed:
        print(f"\n실패한 파일 ({len(failed)}개):")
        for f in failed:
            print(f"  - {f}")
        fail_log = output_path / "download_failures.txt"
        fail_log.write_text("\n".join(failed))
        print(f"실패 목록 저장: {fail_log}")
    else:
        print("\n모든 파일 다운로드 완료!")

    return str(output_path)


def print_dataset_info(output_dir: str):
    """다운로드된 데이터셋 기본 통계 출력."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return

    fake_dir = output_path / "fake"
    if not fake_dir.exists():
        print("fake/ 디렉토리가 없습니다.")
        return

    print("\n" + "=" * 60)
    print("다운로드 결과 요약")
    print("=" * 60)

    total_files = 0
    total_size = 0
    lang_stats = {}

    for lang_dir in sorted(fake_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        wav_files = list(lang_dir.rglob("*.wav"))
        lang_size = sum(f.stat().st_size for f in wav_files)
        lang_stats[lang] = {"files": len(wav_files), "size_mb": lang_size / 1e6}
        total_files += len(wav_files)
        total_size += lang_size

    for lang, stats in lang_stats.items():
        print(f"  {lang}: {stats['files']:,}개 파일, {stats['size_mb']:.1f} MB")

    print(f"\n총 {len(lang_stats)}개 언어, {total_files:,}개 파일, {total_size / 1e9:.2f} GB")

    # meta.csv 파일 수 확인
    meta_files = list(fake_dir.rglob("meta.csv"))
    if meta_files:
        print(f"메타데이터 파일: {len(meta_files)}개 (meta.csv)")

        sample_meta = meta_files[0]
        with open(sample_meta, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                print(f"메타데이터 컬럼: {header}")


def main():
    parser = argparse.ArgumentParser(
        description="MLAAD v9 다운로드 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python scripts/download_mlaad.py --output data/mlaad --languages ko
  python scripts/download_mlaad.py --output data/mlaad --tiny
  python scripts/download_mlaad.py --output data/mlaad --languages ko en ja --workers 8
        """,
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/mlaad",
        help="저장 경로 (기본: data/mlaad)",
    )
    parser.add_argument(
        "--languages", "-l",
        nargs="+",
        default=None,
        help="다운로드할 언어 코드 (예: ko en ja). 미지정 시 전체 다운로드",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="MLAAD-tiny (테스트용 소규모 버전) 다운로드",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="동시 다운로드 스레드 수 (기본: 4)",
    )
    parser.add_argument(
        "--selective",
        action="store_true",
        help="파일 단위 선택적 다운로드 (파일 목록 먼저 조회)",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="사용 가능한 언어 목록만 출력",
    )

    args = parser.parse_args()
    repo_id = REPO_TINY if args.tiny else REPO_FULL

    if args.list_languages:
        print(f"파일 목록 조회 중... ({repo_id})")
        all_files = get_file_list(repo_id)
        langs = set()
        for f in all_files:
            parts = f.split("/")
            if len(parts) >= 2 and parts[0] == "fake":
                langs.add(parts[1])
        print(f"\n사용 가능한 언어 ({len(langs)}개):")
        for lang in sorted(langs):
            count = sum(1 for f in all_files if f.startswith(f"fake/{lang}/") and f.endswith(".wav"))
            print(f"  {lang}: ~{count}개 WAV")
        return

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MLAAD v9 다운로드")
    print("=" * 60)
    print(f"리포지토리: {repo_id}")
    print(f"저장 경로:  {output_dir}")
    print(f"언어 필터:  {args.languages or '전체'}")
    print(f"스레드:     {args.workers}")
    print("=" * 60)
    print()

    if args.selective and args.languages:
        result = download_selective(
            repo_id=repo_id,
            output_dir=output_dir,
            languages=args.languages,
            max_workers=args.workers,
        )
    else:
        result = download_full(
            repo_id=repo_id,
            output_dir=output_dir,
            languages=args.languages,
            max_workers=args.workers,
        )

    if result:
        print_dataset_info(output_dir)
        print(f"\n저장 완료: {output_dir}")


if __name__ == "__main__":
    main()
