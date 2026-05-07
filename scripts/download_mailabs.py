"""
M-AILABS Speech Dataset 다운로드 스크립트
========================================
- 출처: https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/
- 규모: ~1,000시간, 9개 언어, 16kHz mono WAV
- 용도: 실제(bonafide) 음성 — MLAAD 딥페이크의 대응 데이터
- 라이선스: BSD 3-Clause (상업적 사용 가능)
- 인증: 불필요

사용법:
    # 전체 다운로드 (9개 언어, ~100GB+ 압축)
    python scripts/download_mailabs.py --output data/mailabs

    # 특정 언어만 다운로드
    python scripts/download_mailabs.py --output data/mailabs --languages en_US en_UK

    # 독일어 + 프랑스어
    python scripts/download_mailabs.py --output data/mailabs --languages de_DE fr_FR

    # 사용 가능한 언어 목록
    python scripts/download_mailabs.py --list-languages

    # 다운로드만 (압축 해제 안 함)
    python scripts/download_mailabs.py --output data/mailabs --no-extract

사전 준비:
    pip install requests tqdm
"""

import argparse
import hashlib
import os
import sys
import tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("필수 패키지를 설치해주세요:")
    print("  pip install requests tqdm")
    sys.exit(1)


BASE_URL = "http://www.caito.de/data/Training/stt_tts"

LANGUAGES = {
    "de_DE": {"name": "German", "duration": "237h 22m", "size_gb_approx": 30.0},
    "fr_FR": {"name": "French", "duration": "190h 30m", "size_gb_approx": 24.0},
    "it_IT": {"name": "Italian", "duration": "127h 40m", "size_gb_approx": 16.0},
    "es_ES": {"name": "Spanish", "duration": "108h 34m", "size_gb_approx": 14.0},
    "en_US": {"name": "English (US)", "duration": "102h 7m", "size_gb_approx": 13.0},
    "uk_UK": {"name": "Ukrainian", "duration": "87h 8m", "size_gb_approx": 11.0},
    "pl_PL": {"name": "Polish", "duration": "53h 50m", "size_gb_approx": 7.0},
    "ru_RU": {"name": "Russian", "duration": "46h 47m", "size_gb_approx": 6.0},
    "en_UK": {"name": "English (UK)", "duration": "45h 35m", "size_gb_approx": 6.0},
}


def get_download_url(lang_code: str) -> str:
    return f"{BASE_URL}/{lang_code}.tgz"


def download_file(url: str, dest_path: str, chunk_size: int = 8192) -> bool:
    """HTTP 다운로드 with progress bar and resume support."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    existing_size = dest.stat().st_size if dest.exists() else 0
    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)

        if response.status_code == 416:
            print(f"  이미 다운로드 완료: {dest.name}")
            return True

        if response.status_code not in (200, 206):
            print(f"  HTTP 오류 {response.status_code}: {url}")
            return False

        total_size = int(response.headers.get("content-length", 0))
        if response.status_code == 206:
            total_size += existing_size

        mode = "ab" if response.status_code == 206 else "wb"
        initial = existing_size if response.status_code == 206 else 0

        with open(dest, mode) as f:
            with tqdm(
                total=total_size,
                initial=initial,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"  {dest.name}",
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True

    except requests.exceptions.ConnectionError:
        print(f"  연결 실패: {url}")
        print("  caito.de 서버가 응답하지 않습니다. 잠시 후 재시도하세요.")
        return False
    except requests.exceptions.Timeout:
        print(f"  타임아웃: {url}")
        return False
    except Exception as e:
        print(f"  다운로드 오류: {e}")
        return False


def extract_tgz(tgz_path: str, output_dir: str) -> bool:
    """tgz 파일 압축 해제."""
    print(f"  압축 해제 중: {Path(tgz_path).name}")
    try:
        with tarfile.open(tgz_path, "r:gz") as tar:
            members = tar.getmembers()
            with tqdm(total=len(members), desc="  추출", unit="file") as pbar:
                for member in members:
                    tar.extract(member, path=output_dir)
                    pbar.update(1)
        return True
    except tarfile.TarError as e:
        print(f"  압축 해제 오류: {e}")
        return False


def check_server_available() -> bool:
    """caito.de 서버 접근 가능 여부 확인."""
    try:
        response = requests.head(
            f"{BASE_URL}/en_UK.tgz", timeout=10, allow_redirects=True
        )
        return response.status_code in (200, 206, 302)
    except Exception:
        return False


def print_dataset_info(output_dir: str):
    """다운로드된 데이터셋 통계 출력."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return

    print("\n" + "=" * 60)
    print("다운로드 결과 요약")
    print("=" * 60)

    total_files = 0
    total_size = 0

    for lang_dir in sorted(output_path.iterdir()):
        if not lang_dir.is_dir():
            continue
        if lang_dir.name.startswith("."):
            continue

        wav_files = list(lang_dir.rglob("*.wav"))
        lang_size = sum(f.stat().st_size for f in wav_files)
        total_files += len(wav_files)
        total_size += lang_size

        lang_info = LANGUAGES.get(lang_dir.name, {})
        lang_name = lang_info.get("name", lang_dir.name)
        print(f"  {lang_dir.name} ({lang_name}): {len(wav_files):,}개 WAV, {lang_size / 1e9:.2f} GB")

    print(f"\n총 {total_files:,}개 파일, {total_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="M-AILABS Speech Dataset 다운로드 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python scripts/download_mailabs.py --output data/mailabs --languages en_US
  python scripts/download_mailabs.py --output data/mailabs --languages en_US de_DE fr_FR
  python scripts/download_mailabs.py --list-languages
        """,
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/mailabs",
        help="저장 경로 (기본: data/mailabs)",
    )
    parser.add_argument(
        "--languages", "-l",
        nargs="+",
        default=None,
        help="다운로드할 언어 코드 (예: en_US de_DE). 미지정 시 전체 다운로드",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="압축 해제 안 함 (tgz만 다운로드)",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="압축 해제 후 tgz 파일 유지",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="사용 가능한 언어 목록만 출력",
    )

    args = parser.parse_args()

    if args.list_languages:
        print("M-AILABS 사용 가능한 언어:")
        print()
        print(f"{'코드':<8} {'언어':<18} {'시간':<12} {'예상 크기'}")
        print("-" * 55)
        total_hours = 0
        total_gb = 0
        for code, info in LANGUAGES.items():
            print(f"{code:<8} {info['name']:<18} {info['duration']:<12} ~{info['size_gb_approx']:.0f} GB")
            total_gb += info["size_gb_approx"]
        print("-" * 55)
        print(f"{'합계':<8} {'9개 언어':<18} {'~1,000h':<12} ~{total_gb:.0f} GB")
        print()
        print("URL 형식: http://www.caito.de/data/Training/stt_tts/<code>.tgz")
        return

    target_langs = args.languages if args.languages else list(LANGUAGES.keys())

    invalid_langs = [l for l in target_langs if l not in LANGUAGES]
    if invalid_langs:
        print(f"오류: 유효하지 않은 언어 코드: {invalid_langs}")
        print(f"사용 가능: {list(LANGUAGES.keys())}")
        sys.exit(1)

    output_dir = os.path.abspath(args.output)
    archive_dir = os.path.join(output_dir, "_archives")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)

    total_size = sum(LANGUAGES[l]["size_gb_approx"] for l in target_langs)

    print("=" * 60)
    print("M-AILABS Speech Dataset 다운로드")
    print("=" * 60)
    print(f"저장 경로:   {output_dir}")
    print(f"대상 언어:   {target_langs}")
    print(f"예상 크기:   ~{total_size:.0f} GB (압축)")
    print(f"압축 해제:   {'예' if not args.no_extract else '아니오'}")
    print("=" * 60)
    print()

    print("서버 접근 확인 중...")
    if not check_server_available():
        print("경고: caito.de 서버에 접근할 수 없습니다.")
        print("서버가 일시적으로 다운되었거나 네트워크 문제일 수 있습니다.")
        print()
        print("대안:")
        print("  1. 잠시 후 재시도")
        print("  2. HuggingFace 미러 사용 (일부 언어만 가능):")
        print("     - FR: huggingface.co/datasets/gigant/m-ailabs_speech_dataset_fr")
        print("     - IT: huggingface.co/datasets/psiyou/m-ailabs-it_IT")
        sys.exit(1)
    print("서버 접근 확인 완료!")
    print()

    results = {}
    for lang in target_langs:
        info = LANGUAGES[lang]
        url = get_download_url(lang)
        tgz_path = os.path.join(archive_dir, f"{lang}.tgz")

        print(f"[{lang}] {info['name']} ({info['duration']}, ~{info['size_gb_approx']:.0f} GB)")

        # 다운로드
        success = download_file(url, tgz_path)
        if not success:
            results[lang] = "다운로드 실패"
            continue

        # 압축 해제
        if not args.no_extract:
            extract_success = extract_tgz(tgz_path, output_dir)
            if not extract_success:
                results[lang] = "압축 해제 실패"
                continue

            if not args.keep_archive:
                os.remove(tgz_path)
                print(f"  아카이브 삭제: {tgz_path}")

        results[lang] = "완료"
        print()

    # 빈 아카이브 디렉토리 정리
    if not args.no_extract and not args.keep_archive:
        try:
            os.rmdir(archive_dir)
        except OSError:
            pass

    # 결과 요약
    print("\n" + "=" * 60)
    print("다운로드 결과")
    print("=" * 60)
    for lang, status in results.items():
        symbol = "O" if status == "완료" else "X"
        print(f"  [{symbol}] {lang} ({LANGUAGES[lang]['name']}): {status}")

    failed = [l for l, s in results.items() if s != "완료"]
    if failed:
        print(f"\n실패한 언어: {failed}")
        print("재실행 시 이어받기(resume) 지원됩니다.")
    else:
        print("\n모든 언어 다운로드 완료!")

    if not args.no_extract:
        print_dataset_info(output_dir)

    print(f"\n저장 경로: {output_dir}")


if __name__ == "__main__":
    main()
