"""
MUSAN (A Music, Speech, and Noise Corpus) 다운로드 스크립트
==========================================================
- 출처: https://www.openslr.org/17/
- 논문: Snyder et al. (2015) "MUSAN: A Music, Speech, and Noise Corpus" (arXiv:1510.08484)
- 규모: ~11GB (압축), ~109시간
- 용도: 잡음/음악/환경음 증강 (실험 B4: MUSAN Noise Addition)
- 라이선스: CC BY 4.0
- 인증: 불필요

구성:
  - speech (~60시간): Librivox 낭독 + 미국 정부 회의록
  - music  (~42시간): 클래식, 재즈, 힙합 등 다양한 장르
  - noise  (~6시간):  929개 환경음/기계음 (DTMF, 바람, 비, 발소리 등)

사용법:
    # 전체 다운로드 (speech + music + noise)
    python scripts/01_download/download_musan.py --output data/musan

    # noise만 다운로드 (증강 목적이면 이것만으로 충분)
    python scripts/01_download/download_musan.py --output data/musan --categories noise

    # noise + music
    python scripts/01_download/download_musan.py --output data/musan --categories noise music

    # EU 미러 사용 (아시아에서 느릴 경우)
    python scripts/01_download/download_musan.py --output data/musan --mirror eu

    # 데이터셋 정보만 확인
    python scripts/01_download/download_musan.py --info

사전 준비:
    pip install requests tqdm
"""

import argparse
import os
import sys
import tarfile
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("필수 패키지를 설치해주세요:")
    print("  pip install requests tqdm")
    sys.exit(1)


MIRRORS = {
    "default": "https://www.openslr.org/resources/17/musan.tar.gz",
    "eu": "https://openslr.elda.org/resources/17/musan.tar.gz",
    "cn": "https://cn-mirror.openslr.org/resources/17/musan.tar.gz",
    "cn2": "https://openslr.magicdatatech.com/resources/17/musan.tar.gz",
}

DATASET_INFO = """
MUSAN (A Music, Speech, and Noise Corpus)
==========================================

1. 개요
   - 저자: David Snyder, Guoguo Chen, Daniel Povey (2015)
   - 용도: 음성 처리 연구용 증강 데이터
   - 크기: ~11GB (압축), 추출 후 ~30GB
   - 포맷: WAV, 16kHz, 16-bit PCM
   - 라이선스: CC BY 4.0

2. 카테고리별 구성

   speech (~60시간)
   ├── librivox/     20시간, 12개 언어 낭독
   └── us-gov/       40시간, 미국 정부 회의록

   music (~42시간)
   ├── jamendo/      클래식, 재즈, 팝 등
   ├── fma/          Free Music Archive
   └── ...           기타 소스

   noise (~6시간, 929개 파일)
   ├── free-sound/   환경음, 기계음
   └── sound-bible/  자연음, 효과음
   → DTMF, 다이얼톤, 자동차, 천둥, 바람, 발소리, 비, 동물 소리 등

3. 본 연구에서의 용도
   - 실험 B4: MUSAN Noise Addition
   - SNR 5~20dB 범위에서 랜덤 잡음 혼합
   - noise 카테고리만 사용해도 충분
"""


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
        return False
    except requests.exceptions.Timeout:
        print(f"  타임아웃: {url}")
        return False
    except Exception as e:
        print(f"  다운로드 오류: {e}")
        return False


def extract_tar(tar_path: str, output_dir: str, categories: list[str] | None = None) -> bool:
    """tar.gz 압축 해제. categories 지정 시 해당 폴더만 추출."""
    print(f"  압축 해제 중: {Path(tar_path).name}")

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()

            if categories:
                filtered = []
                for m in members:
                    parts = m.name.split("/")
                    # musan/<category>/... 구조
                    if len(parts) >= 2 and parts[1] in categories:
                        filtered.append(m)
                    elif len(parts) <= 1:
                        filtered.append(m)
                members = filtered
                print(f"  필터링: {categories} → {len(members)}개 항목")

            with tqdm(total=len(members), desc="  추출", unit="file") as pbar:
                for member in members:
                    tar.extract(member, path=output_dir)
                    pbar.update(1)

        return True
    except tarfile.TarError as e:
        print(f"  압축 해제 오류: {e}")
        return False


def check_mirror(url: str) -> bool:
    """미러 접근 가능 여부 확인."""
    try:
        response = requests.head(url, timeout=10, allow_redirects=True)
        return response.status_code in (200, 206, 302)
    except Exception:
        return False


def find_working_mirror(preferred: str) -> str | None:
    """접근 가능한 미러를 찾아 반환."""
    preferred_url = MIRRORS.get(preferred)
    if preferred_url:
        print(f"  미러 확인 중: {preferred}...")
        if check_mirror(preferred_url):
            return preferred_url
        print(f"  {preferred} 미러 접근 불가, 다른 미러 탐색 중...")

    for name, url in MIRRORS.items():
        if name == preferred:
            continue
        print(f"  미러 확인 중: {name}...")
        if check_mirror(url):
            print(f"  {name} 미러 사용!")
            return url

    return None


def print_structure(dataset_dir: str):
    """다운로드된 데이터셋 구조 출력."""
    musan_path = Path(dataset_dir) / "musan"
    if not musan_path.exists():
        musan_path = Path(dataset_dir)
    if not musan_path.exists():
        print(f"경로가 존재하지 않습니다: {dataset_dir}")
        return

    print("\n" + "=" * 60)
    print("데이터셋 구조")
    print("=" * 60)

    total_files = 0
    total_size = 0

    for category in ["speech", "music", "noise"]:
        cat_dir = musan_path / category
        if not cat_dir.exists():
            print(f"  {category}/: (없음)")
            continue

        wav_files = list(cat_dir.rglob("*.wav"))
        cat_size = sum(f.stat().st_size for f in wav_files)
        total_files += len(wav_files)
        total_size += cat_size

        print(f"  {category}/: {len(wav_files):,}개 WAV, {cat_size / 1e9:.2f} GB")

        for sub_dir in sorted(cat_dir.iterdir()):
            if sub_dir.is_dir():
                sub_wavs = list(sub_dir.rglob("*.wav"))
                sub_size = sum(f.stat().st_size for f in sub_wavs)
                print(f"    └─ {sub_dir.name}/: {len(sub_wavs):,}개, {sub_size / 1e6:.0f} MB")

    print(f"\n총 {total_files:,}개 WAV, {total_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="MUSAN 데이터셋 다운로드 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python scripts/01_download/download_musan.py --output data/musan
  python scripts/01_download/download_musan.py --output data/musan --categories noise
  python scripts/01_download/download_musan.py --output data/musan --categories noise music
  python scripts/01_download/download_musan.py --output data/musan --mirror eu
  python scripts/01_download/download_musan.py --info
        """,
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/musan",
        help="저장 경로 (기본: data/musan)",
    )
    parser.add_argument(
        "--categories", "-c",
        nargs="+",
        choices=["speech", "music", "noise"],
        default=None,
        help="추출할 카테고리 (미지정 시 전체). 증강용이면 noise만으로 충분",
    )
    parser.add_argument(
        "--mirror", "-m",
        choices=list(MIRRORS.keys()),
        default="default",
        help="다운로드 미러 (기본: default=openslr.org)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="압축 해제 안 함 (tar.gz만 다운로드)",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="압축 해제 후 tar.gz 파일 유지",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="데이터셋 정보만 출력",
    )
    parser.add_argument(
        "--structure",
        action="store_true",
        help="다운로드된 데이터셋 구조 확인",
    )

    args = parser.parse_args()

    if args.info:
        print(DATASET_INFO)
        print("다운로드 미러:")
        for name, url in MIRRORS.items():
            print(f"  {name:<10} {url}")
        return

    output_dir = os.path.abspath(args.output)

    if args.structure:
        print_structure(output_dir)
        return

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MUSAN 데이터셋 다운로드")
    print("=" * 60)
    print(f"저장 경로:   {output_dir}")
    print(f"카테고리:    {args.categories or '전체 (speech, music, noise)'}")
    print(f"미러:        {args.mirror}")
    print(f"압축 해제:   {'예' if not args.no_extract else '아니오'}")
    print(f"예상 크기:   ~11GB (압축), ~30GB (추출)")
    print("=" * 60)
    print()

    # 미러 탐색
    url = find_working_mirror(args.mirror)
    if not url:
        print("모든 미러에 접근할 수 없습니다.")
        print("네트워크 연결을 확인하고 다시 시도해주세요.")
        print()
        print("수동 다운로드:")
        for name, mirror_url in MIRRORS.items():
            print(f"  wget {mirror_url}")
        sys.exit(1)

    print(f"다운로드 URL: {url}")
    print()

    # 다운로드
    tar_path = os.path.join(output_dir, "musan.tar.gz")
    success = download_file(url, tar_path)

    if not success:
        print()
        print("다운로드 실패. 다른 미러를 시도해주세요:")
        for name, mirror_url in MIRRORS.items():
            print(f"  python scripts/01_download/download_musan.py -o {output_dir} --mirror {name}")
        print()
        print("또는 직접 다운로드:")
        print(f"  wget -c {url} -O {tar_path}")
        sys.exit(1)

    # 압축 해제
    if not args.no_extract:
        print()
        extract_success = extract_tar(tar_path, output_dir, args.categories)

        if not extract_success:
            print("압축 해제 실패.")
            print(f"수동 해제: tar xzf {tar_path} -C {output_dir}")
            sys.exit(1)

        if not args.keep_archive:
            os.remove(tar_path)
            print(f"  아카이브 삭제: {tar_path}")

    # 구조 출력
    if not args.no_extract:
        print_structure(output_dir)

    print(f"\n저장 완료: {output_dir}")


if __name__ == "__main__":
    main()
