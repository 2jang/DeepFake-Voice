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
    * ※ sounds.zip (Google Drive)은 다운로드하지 않음 — AIHUB 원본 사용

사용법:
    # KoAAD Google Drive 폴더 다운로드 (fake 음성만, sounds.zip 제외)
    python scripts/01_download/download_koaad.py --output data/koaad

    # AIHUB 데이터 경로 지정 (real 음성)
    python scripts/01_download/download_koaad.py --output data/koaad --aihub-dir data/aihub

    # GitHub 코드도 함께 클론
    python scripts/01_download/download_koaad.py --output data/koaad --clone-repo

    # 데이터셋 정보만 확인
    python scripts/01_download/download_koaad.py --info

    # 다운로드 후 구조 확인
    python scripts/01_download/download_koaad.py --output data/koaad --structure

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


GITHUB_REPO = "https://github.com/ldh-Hoon/KoAAD.git"
AIHUB_URL = "https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71349"

# 다운로드할 fake 음성 파일 (Google Drive 파일 ID + 저장 파일명)
# sounds.zip은 제외 — AIHUB real 음성 사용
FAKE_FILES = [
    {"id": "1AERV-X9NKz_ejQUKkodBDDtbvLp_YD7Z", "name": "Google_Cloud_TTS.zip"},
    {"id": "1NAz89M91MApWqvoHT4xZp-xysW0dUOMt",  "name": "XTTS2.zip"},
    {"id": "1BQ4eWU4Ge7fjsBYxoPzTZh60bZ2F1_5S",  "name": "네이버기사TTS.zip"},
]


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

3. Real 음성 — AIHUB (별도 수동 다운로드)
   ※ sounds.zip은 다운로드하지 않음 — AIHUB 원본 데이터 직접 사용
   - 출처: AIHUB "감성 및 발화스타일 동시고려 음성합성 데이터"
   - AIHUB 회원가입 + 데이터 이용 신청 필요 (수동)
   - 라벨: 모두 bonafide (real)

4. 디렉토리 구조 (예상)
   data/
   ├── koaad/                    ← fake 음성 (Google Drive)
   │   ├── google_cloud_tts/
   │   ├── melo_tts/
   │   ├── xtts2/
   │   ├── audio_pub/
   │   ├── naver_news/
   │   └── clova_dubbing/
   └── aihub/                    ← real 음성 (AIHUB 수동 다운로드)
       └── ...

5. 코드에서의 분할
   - Train: 폴더별 파일의 앞 70%
   - Test: 폴더별 파일의 뒤 30%
   - Speaker-disjoint 아님 (알파벳 순 분할)
"""


def download_fake_files(output_dir: str, fake_files: list = None) -> bool:
    """FAKE_FILES 목록의 파일을 ID로 직접 다운로드.

    폴더 전체 다운로드(download_folder) 대신 파일별 개별 다운로드를 사용해
    sounds.zip 같은 접근 제한 파일을 처음부터 건너뜁니다.
    """
    try:
        import gdown
    except ImportError:
        print("gdown 패키지를 설치해주세요: pip install gdown")
        return False

    if fake_files is None:
        fake_files = FAKE_FILES

    os.makedirs(output_dir, exist_ok=True)
    print(f"fake 음성 파일 {len(fake_files)}개 다운로드 중...")
    print(f"  저장: {output_dir}")
    print()

    all_ok = True
    for i, f in enumerate(fake_files, 1):
        fid, fname = f["id"], f["name"]
        dest = os.path.join(output_dir, fname)

        if os.path.exists(dest):
            size_mb = os.path.getsize(dest) / 1e6
            print(f"  [{i}/{len(fake_files)}] {fname} — 이미 존재 ({size_mb:.1f} MB), 건너뜀")
            continue

        print(f"  [{i}/{len(fake_files)}] {fname}")
        url = f"https://drive.google.com/uc?id={fid}"
        try:
            result = gdown.download(url, dest, quiet=False)
            if result:
                size_mb = os.path.getsize(dest) / 1e6
                print(f"    ✓ 완료 ({size_mb:.1f} MB)")
            else:
                print(f"    ✗ 실패: {fname}")
                all_ok = False
        except Exception as e:
            print(f"    ✗ 오류: {e}")
            all_ok = False

    return all_ok

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


def verify_aihub_dir(aihub_dir: str) -> bool:
    """AIHUB 데이터 디렉토리 존재 및 오디오 파일 확인."""
    aihub_path = Path(aihub_dir)

    if not aihub_path.exists():
        print(f"\n[경고] AIHUB 데이터 경로가 존재하지 않습니다: {aihub_dir}")
        print("  AIHUB에서 수동 다운로드 후 해당 경로에 저장해주세요.")
        return False

    audio_exts = {".wav", ".mp3", ".m4a", ".flac"}
    audio_files = [
        f for f in aihub_path.rglob("*")
        if f.is_file() and f.suffix.lower() in audio_exts
    ]

    if not audio_files:
        print(f"\n[경고] AIHUB 경로에 오디오 파일이 없습니다: {aihub_dir}")
        print("  다운로드한 데이터를 압축 해제했는지 확인해주세요.")
        return False

    total_size = sum(f.stat().st_size for f in audio_files)
    print(f"\n[확인] AIHUB real 음성: {len(audio_files):,}개 파일, {total_size / 1e9:.2f} GB")
    print(f"  경로: {aihub_dir}")
    return True


def create_symlink(aihub_dir: str, koaad_dir: str):
    """AIHUB 데이터를 KoAAD 디렉토리 내에 심볼릭 링크로 연결."""
    link_path = os.path.join(koaad_dir, "real_audio")
    aihub_abs = os.path.abspath(aihub_dir)

    if os.path.exists(link_path):
        if os.path.islink(link_path):
            existing_target = os.readlink(link_path)
            if existing_target == aihub_abs:
                print(f"심볼릭 링크가 이미 존재합니다: {link_path} → {aihub_abs}")
                return
            os.remove(link_path)
        else:
            print(f"[경고] {link_path}가 이미 존재하며 심볼릭 링크가 아닙니다. 건너뜁니다.")
            return

    try:
        os.symlink(aihub_abs, link_path)
        print(f"심볼릭 링크 생성: {link_path} → {aihub_abs}")
    except OSError as e:
        print(f"심볼릭 링크 생성 실패: {e}")
        print(f"  수동으로 연결하거나 코드에서 직접 경로를 지정해주세요.")


def print_structure(dataset_dir: str, aihub_dir: str = None):
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

    print(f"\n[Fake 음성] {dataset_dir}")
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
            # sounds.zip이 남아있는 경우 경고
            if item.name.lower() in {"sounds.zip"}:
                print(f"  [!] {item.name}: {size / 1e6:.1f} MB — 삭제 권장 (AIHUB 원본 사용)")
            else:
                print(f"  {item.name}: {size / 1e6:.1f} MB")

    if total_files > 0:
        print(f"\n  Fake 음성 합계: {total_files:,}개, {total_size / 1e9:.2f} GB")
    else:
        print("\n  오디오 파일을 찾을 수 없습니다.")

    # AIHUB real 음성 구조
    if aihub_dir:
        aihub_path = Path(aihub_dir)
        if aihub_path.exists():
            real_files = [
                f for f in aihub_path.rglob("*")
                if f.is_file() and f.suffix.lower() in audio_exts
            ]
            real_size = sum(f.stat().st_size for f in real_files)
            print(f"\n[Real 음성 — AIHUB] {aihub_dir}")
            print(f"  오디오 파일: {len(real_files):,}개, {real_size / 1e9:.2f} GB")
        else:
            print(f"\n[Real 음성 — AIHUB] 경로 없음: {aihub_dir}")
            print("  AIHUB에서 수동 다운로드 필요")


def main():
    parser = argparse.ArgumentParser(
        description="KoAAD 데이터셋 다운로드 스크립트 (sounds.zip 제외, AIHUB 사용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # fake 음성만 다운로드 (sounds.zip 건너뜀)
  python scripts/01_download/download_koaad.py --output data/koaad

  # AIHUB real 음성 경로 지정 + 심볼릭 링크 생성
  python scripts/01_download/download_koaad.py --output data/koaad --aihub-dir data/aihub

  # 정보 확인
  python scripts/01_download/download_koaad.py --info

참고:
  - sounds.zip은 다운로드하지 않습니다 (AIHUB 원본 데이터 사용)
  - Real 음성(AIHUB)은 수동 다운로드 필요:
    https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71349
  - AIHUB 회원가입 후 데이터 이용 신청 → 승인 후 다운로드
        """,
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/koaad",
        help="fake 음성 저장 경로 (기본: data/koaad)",
    )
    parser.add_argument(
        "--aihub-dir",
        type=str,
        default="data/aihub",
        help="AIHUB real 음성 경로 (기본: data/aihub)",
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
        "--link-aihub",
        action="store_true",
        help="AIHUB 데이터를 KoAAD 디렉토리에 심볼릭 링크로 연결",
    )

    args = parser.parse_args()

    if args.info:
        print(DATASET_INFO)
        return

    output_dir = os.path.abspath(args.output)
    aihub_dir = os.path.abspath(args.aihub_dir)

    if args.structure:
        print_structure(output_dir, aihub_dir)
        return

    files_str = ", ".join(f["name"] for f in FAKE_FILES)
    print("=" * 60)
    print("KoAAD fake 음성 다운로드 (파일 직접 지정)")
    print("=" * 60)
    print(f"다운로드 파일: {files_str}")
    print(f"저장:          {output_dir}")
    print(f"Real (AIHUB):  {aihub_dir}")
    print("=" * 60)
    print()

    # 파일 ID 직접 지정 방식 — sounds.zip 문제 없음
    success = download_fake_files(output_dir)

    if not success:
        print()
        print("=" * 60)
        print("일부 파일 다운로드 실패")
        print("=" * 60)
        print()
        print("수동 다운로드 링크:")
        for f in FAKE_FILES:
            url = f"https://drive.google.com/uc?id={f['id']}&export=download"
            print(f"  {f['name']}: {url}")
        print()
        print(f"다운로드 후 {output_dir} 에 저장해주세요.")

    # GitHub 리포 클론
    if args.clone_repo:
        print()
        clone_repo(output_dir)

    # AIHUB 데이터 확인
    print()
    print("=" * 60)
    print("Real 음성 (AIHUB) 확인")
    print("=" * 60)
    aihub_exists = verify_aihub_dir(aihub_dir)

    # 심볼릭 링크 생성
    if args.link_aihub and aihub_exists:
        create_symlink(aihub_dir, output_dir)

    # 구조 출력
    if success:
        print_structure(output_dir, aihub_dir)
        print(f"\nFake 음성 저장 완료: {output_dir}")

    # AIHUB 안내 (아직 없는 경우)
    if not aihub_exists:
        print()
        print("=" * 60)
        print("Real 음성(AIHUB) 수동 다운로드 안내")
        print("=" * 60)
        print(f"  URL: {AIHUB_URL}")
        print("  절차:")
        print("    1. AIHUB 회원가입 (https://www.aihub.or.kr)")
        print("    2. '감성 및 발화스타일 동시고려 음성합성 데이터' 검색")
        print("    3. 데이터 이용 신청 → 승인 대기 (1~3일)")
        print("    4. 승인 후 다운로드")
        print(f"    5. {aihub_dir} 경로에 저장")
        print()
        print("  다운로드 후 다시 실행하여 확인:")
        print(f"    python scripts/01_download/download_koaad.py --structure --aihub-dir {args.aihub_dir}")


if __name__ == "__main__":
    main()