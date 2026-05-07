"""
KoAAD (Korean Audio Anti-spoofing Dataset) 다운로드 스크립트
============================================================
- GitHub: https://github.com/ldh-Hoon/KoAAD
- 데이터 호스팅: Google Drive
- 규모: ~10GB, ~30,000 음성 파일
- 라이선스: 연구 목적 (세종대 캡스톤 프로젝트)

구성:
  - Fake 음성 (KoAAD): 6개 TTS 시스템으로 생성
    * Google Cloud TTS
    * Melo TTS
    * XTTS2
    * Audio Pub
    * 네이버 기사 음성 요약
    * 클로바더빙
  - Real 음성: AIHUB "감성 및 발화스타일 동시고려 음성합성 데이터"
    * https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71349
    * AIHUB는 회원가입 + 승인 필요 (별도 수동 다운로드)

사용법:
    # KoAAD Google Drive 폴더 다운로드 (fake 음성)
    python scripts/download_koaad.py --output data/koaad

    # GitHub 코드도 함께 클론
    python scripts/download_koaad.py --output data/koaad --clone-repo

    # 데이터셋 정보만 확인
    python scripts/download_koaad.py --info

    # 다운로드 후 구조 확인
    python scripts/download_koaad.py --output data/koaad --structure

사전 준비:
    pip install gdown tqdm
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm 패키지를 설치해주세요: pip install tqdm")
    sys.exit(1)


GDRIVE_FOLDER_ID = "1RihDtMWUc5sPg5fNqJ_L015U6HMgwF2u"
GDRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"
PRETRAINED_FOLDER_URL = "https://drive.google.com/drive/folders/1YWMC64MW4HjGUX1fnBaMkMIGgAJde9Ch"
GITHUB_REPO = "https://github.com/ldh-Hoon/KoAAD.git"
AIHUB_URL = "https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71349"

DATASET_INFO = """
KoAAD (Korean Audio Anti-spoofing Dataset)
==========================================

1. 개요
   - 세종대학교 2024-1 캡스톤디자인 프로젝트
   - 한국어 음성 딥페이크 탐지를 위한 데이터셋
   - 논문: 임동훈, 한종기 (2024) "LCNN 기반 Deepfake Audio Detection의 개선 알고리즘"

2. Fake 음성 (KoAAD) — Google Drive
   - 규모: ~10GB, ~30,000 파일
   - 포맷: WAV/MP3, 16kHz
   - TTS 시스템 6종:
     * Google Cloud TTS
     * Melo TTS
     * XTTS2
     * Audio Pub
     * 네이버 기사 음성 요약
     * 클로바더빙
   - 라벨: 모두 spoof (fake)

3. Real 음성 (AIHUB) — 별도 다운로드 필요
   - 출처: AIHUB "감성 및 발화스타일 동시고려 음성합성 데이터"
   - AIHUB 회원가입 + 데이터 이용 신청 필요 (수동)
   - 라벨: 모두 bonafide (real)

4. 디렉토리 구조 (예상)
   KoAAD/
   ├── google_cloud_tts/
   │   ├── sample_001.wav
   │   └── ...
   ├── melo_tts/
   │   ├── sample_001.wav
   │   └── ...
   ├── xtts2/
   ├── audio_pub/
   ├── naver_news/
   └── clova_dubbing/

5. 코드에서의 분할
   - Train: 폴더별 파일의 앞 70%
   - Test: 폴더별 파일의 뒤 30%
   - Speaker-disjoint 아님 (알파벳 순 분할)
"""


def download_via_gdown(folder_id: str, output_dir: str) -> bool:
    """gdown으로 Google Drive 폴더 다운로드."""
    try:
        import gdown
    except ImportError:
        print("gdown 패키지를 설치해주세요:")
        print("  pip install gdown")
        return False

    os.makedirs(output_dir, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"

    print(f"Google Drive 폴더 다운로드 중...")
    print(f"  URL: {url}")
    print(f"  저장: {output_dir}")
    print()

    try:
        gdown.download_folder(
            url=url,
            output=output_dir,
            quiet=False,
            use_cookies=False,
        )
        return True
    except Exception as e:
        print(f"\ngdown 다운로드 실패: {e}")
        return False


def download_via_gdown_cli(folder_id: str, output_dir: str) -> bool:
    """gdown CLI로 다운로드."""
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    cmd = ["gdown", "--folder", url, "-O", output_dir]

    print(f"gdown CLI로 다운로드 중...")
    print(f"  명령어: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True, text=True)
        return True
    except FileNotFoundError:
        print("gdown CLI가 설치되지 않았습니다: pip install gdown")
        return False
    except subprocess.CalledProcessError as e:
        print(f"gdown CLI 실패: {e}")
        return False


def clone_repo(output_dir: str) -> bool:
    """KoAAD GitHub 리포지토리 클론."""
    repo_dir = os.path.join(output_dir, "KoAAD_repo")

    if os.path.exists(repo_dir):
        print(f"리포지토리가 이미 존재합니다: {repo_dir}")
        return True

    print(f"GitHub 리포지토리 클론 중...")
    print(f"  {GITHUB_REPO}")
    print(f"  → {repo_dir}")

    try:
        subprocess.run(
            ["git", "clone", GITHUB_REPO, repo_dir],
            check=True,
            text=True,
        )
        print("클론 완료!")
        return True
    except FileNotFoundError:
        print("git이 설치되지 않았습니다.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"클론 실패: {e}")
        return False


def print_structure(dataset_dir: str):
    """다운로드된 데이터셋 구조 출력."""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"경로가 존재하지 않습니다: {dataset_dir}")
        return

    print("\n" + "=" * 60)
    print("데이터셋 구조")
    print("=" * 60)

    audio_exts = {".wav", ".mp3", ".wv3", ".wap", ".map", ".m4a", ".flac"}
    total_files = 0
    total_size = 0

    for item in sorted(dataset_path.iterdir()):
        if item.is_dir() and not item.name.startswith("."):
            audio_files = [
                f for f in item.rglob("*")
                if f.is_file() and f.suffix.lower() in audio_exts
            ]
            dir_size = sum(f.stat().st_size for f in audio_files)
            total_files += len(audio_files)
            total_size += dir_size

            if audio_files:
                exts = set(f.suffix.lower() for f in audio_files)
                print(f"  {item.name}/: {len(audio_files):,}개 오디오 ({', '.join(exts)}), {dir_size / 1e6:.1f} MB")
            else:
                sub_count = sum(1 for _ in item.rglob("*") if _.is_file())
                print(f"  {item.name}/: {sub_count}개 파일 (오디오 아님)")

        elif item.is_file():
            size = item.stat().st_size
            print(f"  {item.name}: {size / 1e6:.1f} MB")

    if total_files > 0:
        print(f"\n총 오디오: {total_files:,}개, {total_size / 1e9:.2f} GB")
    else:
        print("\n오디오 파일을 찾을 수 없습니다.")
        print("Google Drive 폴더 구조를 확인해주세요.")


def main():
    parser = argparse.ArgumentParser(
        description="KoAAD 데이터셋 다운로드 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python scripts/download_koaad.py --output data/koaad
  python scripts/download_koaad.py --output data/koaad --clone-repo
  python scripts/download_koaad.py --info

참고:
  - Real 음성(AIHUB)은 수동 다운로드 필요:
    https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71349
  - AIHUB 회원가입 후 데이터 이용 신청 → 승인 후 다운로드
        """,
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/koaad",
        help="저장 경로 (기본: data/koaad)",
    )
    parser.add_argument(
        "--clone-repo",
        action="store_true",
        help="KoAAD GitHub 리포지토리도 함께 클론",
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
    parser.add_argument(
        "--method",
        choices=["gdown", "gdown-cli"],
        default="gdown",
        help="다운로드 방법 (기본: gdown)",
    )

    args = parser.parse_args()

    if args.info:
        print(DATASET_INFO)
        return

    output_dir = os.path.abspath(args.output)

    if args.structure:
        print_structure(output_dir)
        return

    print("=" * 60)
    print("KoAAD 데이터셋 다운로드")
    print("=" * 60)
    print(f"Google Drive: {GDRIVE_FOLDER_URL}")
    print(f"저장 경로:    {output_dir}")
    print("=" * 60)
    print()

    # Google Drive 데이터 다운로드
    if args.method == "gdown":
        success = download_via_gdown(GDRIVE_FOLDER_ID, output_dir)
    else:
        success = download_via_gdown_cli(GDRIVE_FOLDER_ID, output_dir)

    if not success:
        print()
        print("=" * 60)
        print("자동 다운로드 실패. 대안을 시도해주세요:")
        print("=" * 60)
        print()
        print("방법 1 — gdown 재시도:")
        print("  pip install --upgrade gdown")
        print(f"  gdown --folder '{GDRIVE_FOLDER_URL}' -O {output_dir}")
        print()
        print("방법 2 — 수동 다운로드:")
        print(f"  1. 브라우저에서 열기: {GDRIVE_FOLDER_URL}")
        print(f"  2. 폴더 우클릭 → 다운로드")
        print(f"  3. {output_dir}에 압축 해제")
        print()
        print("방법 3 — rclone 사용:")
        print("  rclone copy gdrive:KoAAD data/koaad --drive-shared-with-me")
        print()
        print("참고: Google Drive는 대용량 폴더 다운로드 시 제한이 있을 수 있습니다.")
        print("      파일이 많은 경우 하위 폴더별로 나누어 다운로드하세요.")

    # GitHub 리포 클론
    if args.clone_repo:
        print()
        clone_repo(output_dir)

    # 구조 출력
    if success:
        print_structure(output_dir)
        print(f"\n저장 완료: {output_dir}")

    # AIHUB 안내
    print()
    print("=" * 60)
    print("주의: Real 음성(AIHUB)은 별도 수동 다운로드 필요")
    print("=" * 60)
    print(f"  URL: {AIHUB_URL}")
    print("  절차:")
    print("    1. AIHUB 회원가입 (https://www.aihub.or.kr)")
    print("    2. '감성 및 발화스타일 동시고려 음성합성 데이터' 검색")
    print("    3. 데이터 이용 신청 → 승인 대기 (1~3일)")
    print("    4. 승인 후 다운로드")
    print("    5. data/aihub/ 경로에 저장")


if __name__ == "__main__":
    main()
