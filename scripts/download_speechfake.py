"""
SpeechFake 데이터셋 다운로드 스크립트
=====================================
- 논문: Huang et al. (2025) "SpeechFake: A Large-Scale Multilingual Speech Deepfake Dataset" (ACL 2025)
- GitHub: https://github.com/YMLLG/SpeechFake
- 호스팅: ModelScope (inclusionAI/SPEECHFAKE), HuggingFace (DeepFense/SpeechFake)
- 규모: 3M+ 샘플, 3,000+ 시간, 46개 언어, 40개 TTS/VC/NV
- 라이선스: CC BY 4.0
- 한국어(ko): 9대 주요 언어 중 하나 (test set에 독립 subset)

구조:
  - SpeechFake-BD (Bilingual Dataset): 영어+중국어, 40개 생성기, 2M utterances
  - SpeechFake-MD (Multilingual Dataset): 46개 언어, 6개 생성기, 1.3M utterances
    → 한국어(ko)는 MD의 9대 주요 언어에 포함

사용법:
    # ModelScope에서 전체 다운로드
    python scripts/download_speechfake.py --output data/speechfake --method modelscope

    # HuggingFace에서 전체 다운로드
    python scripts/download_speechfake.py --output data/speechfake --method huggingface

    # Git LFS 클론 (ModelScope)
    python scripts/download_speechfake.py --output data/speechfake --method git

    # 다운로드 후 한국어만 추출
    python scripts/download_speechfake.py --output data/speechfake --extract-languages ko

    # 데이터셋 정보만 확인
    python scripts/download_speechfake.py --info

사전 준비:
    # ModelScope 방식
    pip install modelscope

    # HuggingFace 방식
    pip install huggingface_hub
    huggingface-cli login

    # Git LFS 방식
    sudo apt install git-lfs && git lfs install
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm 패키지를 설치해주세요: pip install tqdm")
    sys.exit(1)


MODELSCOPE_REPO = "inclusionAI/SPEECHFAKE"
HUGGINGFACE_REPO = "DeepFense/SpeechFake"
GITHUB_REPO = "https://github.com/YMLLG/SpeechFake.git"
MODELSCOPE_GIT = "https://www.modelscope.cn/datasets/inclusionAI/SPEECHFAKE.git"

PRIMARY_LANGUAGES = ["en", "zh", "es", "fr", "hi", "ja", "ko", "fa", "it"]

DATASET_INFO = """
SpeechFake 데이터셋 구조
========================

1. SpeechFake-BD (Bilingual Dataset)
   - 언어: 영어(en) + 중국어(zh)
   - 샘플 수: 2,003,016 utterances
   - 생성기: 40개 TTS/VC/NV
   - 분할: train / dev / test

2. SpeechFake-MD (Multilingual Dataset)
   - 언어: 46개 (9개 주요 + 37개 추가)
   - 샘플 수: 1,335,492 utterances
   - 생성기: 6개 다국어 TTS
   - 분할:
     * train/dev: 영어+중국어만 포함
     * test: 10개 subset
       - 9개: 주요 언어별 (en, zh, es, fr, hi, ja, ko, fa, it)
       - 1개: 나머지 37개 언어 통합

3. 한국어(ko) 데이터 위치
   - SpeechFake-MD test subset 중 ko 전용 subset
   - 9대 주요 언어로 분류 (더 큰 샘플 볼륨)

4. 기술 사양
   - 포맷: WAV, 16kHz, mono
   - 메타데이터: real/fake 라벨, 생성기 종류, 화자ID, 언어ID, 텍스트
"""


def download_via_modelscope(output_dir: str):
    """ModelScope SDK로 다운로드."""
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError:
        print("modelscope 패키지를 설치해주세요:")
        print("  pip install modelscope")
        sys.exit(1)

    print("ModelScope에서 SpeechFake 다운로드 시작...")
    print(f"리포지토리: {MODELSCOPE_REPO}")
    print(f"저장 경로: {output_dir}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    try:
        ds = MsDataset.load(
            MODELSCOPE_REPO,
            cache_dir=output_dir,
        )
        print("다운로드 완료!")
        return True
    except Exception as e:
        print(f"ModelScope SDK 다운로드 실패: {e}")
        print()
        print("대안: modelscope CLI 사용")
        print(f"  modelscope download --dataset {MODELSCOPE_REPO} --local_dir {output_dir}")
        return False


def download_via_modelscope_cli(output_dir: str):
    """ModelScope CLI로 다운로드."""
    cmd = [
        "modelscope", "download",
        "--dataset", MODELSCOPE_REPO,
        "--local_dir", output_dir,
    ]

    print("ModelScope CLI로 다운로드 시작...")
    print(f"명령어: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True, text=True)
        print("다운로드 완료!")
        return True
    except FileNotFoundError:
        print("modelscope CLI가 설치되지 않았습니다.")
        print("  pip install modelscope")
        return False
    except subprocess.CalledProcessError as e:
        print(f"다운로드 실패: {e}")
        return False


def download_via_huggingface(output_dir: str):
    """HuggingFace에서 다운로드."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub 패키지를 설치해주세요:")
        print("  pip install huggingface_hub")
        sys.exit(1)

    print("HuggingFace에서 SpeechFake 다운로드 시작...")
    print(f"리포지토리: {HUGGINGFACE_REPO}")
    print(f"저장 경로: {output_dir}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    try:
        path = snapshot_download(
            repo_id=HUGGINGFACE_REPO,
            repo_type="dataset",
            local_dir=output_dir,
            max_workers=4,
        )
        print(f"다운로드 완료: {path}")
        return True
    except Exception as e:
        print(f"HuggingFace 다운로드 실패: {e}")
        print()
        print("HuggingFace 로그인이 필요할 수 있습니다:")
        print("  huggingface-cli login")
        return False


def download_via_git(output_dir: str):
    """Git LFS로 ModelScope에서 클론."""
    print("Git LFS로 SpeechFake 클론 시작...")
    print(f"소스: {MODELSCOPE_GIT}")
    print(f"저장 경로: {output_dir}")
    print()
    print("주의: 대용량 데이터셋이므로 시간이 오래 걸릴 수 있습니다.")
    print()

    # git lfs 확인
    try:
        subprocess.run(["git", "lfs", "version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("git-lfs가 설치되지 않았습니다:")
        print("  sudo apt install git-lfs && git lfs install")
        return False

    cmd = ["git", "clone", MODELSCOPE_GIT, output_dir]
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print("클론 완료!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Git 클론 실패: {e}")
        return False


def extract_language(dataset_dir: str, output_dir: str, languages: list[str]):
    """다운로드된 데이터셋에서 특정 언어만 추출."""
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)

    print(f"언어 필터링: {languages}")
    print(f"원본: {dataset_path}")
    print(f"출력: {output_path}")
    print()

    extracted_count = 0

    # 방법 1: 메타데이터 기반 필터링
    meta_files = list(dataset_path.rglob("*.json")) + list(dataset_path.rglob("*.csv"))
    wav_dirs = list(dataset_path.rglob("*.wav"))

    if not wav_dirs and not meta_files:
        print("경고: WAV 파일이나 메타데이터를 찾을 수 없습니다.")
        print("데이터셋 구조를 확인해주세요:")
        print()
        # 디렉토리 구조 출력
        for item in sorted(dataset_path.rglob("*")):
            if item.is_dir():
                depth = len(item.relative_to(dataset_path).parts)
                if depth <= 3:
                    print(f"  {'  ' * depth}{item.name}/")
        return

    # 디렉토리 기반 탐색 (언어 코드 폴더)
    for lang in languages:
        lang_dirs = list(dataset_path.rglob(f"*/{lang}")) + \
                    list(dataset_path.rglob(f"*/{lang}_*")) + \
                    list(dataset_path.rglob(f"*/*{lang}*"))

        for lang_dir in lang_dirs:
            if lang_dir.is_dir():
                dest = output_path / lang_dir.relative_to(dataset_path)
                dest.mkdir(parents=True, exist_ok=True)

                wav_files = list(lang_dir.rglob("*.wav"))
                print(f"  [{lang}] {lang_dir.name}: {len(wav_files)}개 WAV 발견")

                for wav in tqdm(wav_files, desc=f"  복사 ({lang})", unit="file"):
                    rel = wav.relative_to(lang_dir)
                    dest_file = dest / rel
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(wav, dest_file)
                    extracted_count += 1

    print(f"\n추출 완료: {extracted_count}개 파일 → {output_path}")


def print_dataset_structure(dataset_dir: str):
    """다운로드된 데이터셋 구조 출력."""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"경로가 존재하지 않습니다: {dataset_dir}")
        return

    print("\n" + "=" * 60)
    print("데이터셋 구조")
    print("=" * 60)

    total_wav = 0
    total_size = 0

    for item in sorted(dataset_path.iterdir()):
        if item.is_dir():
            wav_count = len(list(item.rglob("*.wav")))
            dir_size = sum(f.stat().st_size for f in item.rglob("*.wav"))
            total_wav += wav_count
            total_size += dir_size
            if wav_count > 0:
                print(f"  {item.name}/: {wav_count:,}개 WAV, {dir_size / 1e9:.2f} GB")
            else:
                print(f"  {item.name}/")

    if total_wav > 0:
        print(f"\n총 {total_wav:,}개 WAV, {total_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="SpeechFake 데이터셋 다운로드 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # ModelScope SDK (권장)
  python scripts/download_speechfake.py -o data/speechfake --method modelscope

  # HuggingFace
  python scripts/download_speechfake.py -o data/speechfake --method huggingface

  # Git LFS 클론
  python scripts/download_speechfake.py -o data/speechfake --method git

  # 한국어 추출
  python scripts/download_speechfake.py -o data/speechfake --extract-languages ko
        """,
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/speechfake",
        help="저장 경로 (기본: data/speechfake)",
    )
    parser.add_argument(
        "--method", "-m",
        choices=["modelscope", "modelscope-cli", "huggingface", "git"],
        default="modelscope",
        help="다운로드 방법 (기본: modelscope)",
    )
    parser.add_argument(
        "--extract-languages",
        nargs="+",
        default=None,
        help="다운로드 후 특정 언어만 추출 (예: ko en ja)",
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
        print()
        print("다운로드 방법:")
        print("  1. ModelScope SDK: pip install modelscope")
        print(f"     → modelscope download --dataset {MODELSCOPE_REPO}")
        print()
        print("  2. HuggingFace: pip install huggingface_hub")
        print(f"     → huggingface.co/datasets/{HUGGINGFACE_REPO}")
        print()
        print("  3. Git LFS:")
        print(f"     → git clone {MODELSCOPE_GIT}")
        return

    output_dir = os.path.abspath(args.output)

    if args.structure:
        print_dataset_structure(output_dir)
        return

    print("=" * 60)
    print("SpeechFake 데이터셋 다운로드")
    print("=" * 60)
    print(f"리포지토리:  {MODELSCOPE_REPO}")
    print(f"저장 경로:   {output_dir}")
    print(f"다운로드 방법: {args.method}")
    if args.extract_languages:
        print(f"추출 언어:   {args.extract_languages}")
    print("=" * 60)
    print()

    # 다운로드
    success = False
    if args.method == "modelscope":
        success = download_via_modelscope(output_dir)
    elif args.method == "modelscope-cli":
        success = download_via_modelscope_cli(output_dir)
    elif args.method == "huggingface":
        success = download_via_huggingface(output_dir)
    elif args.method == "git":
        success = download_via_git(output_dir)

    if not success:
        print()
        print("=" * 60)
        print("다운로드 실패. 대안을 시도해주세요:")
        print("=" * 60)
        print()
        print("방법 1 — ModelScope CLI (가장 안정적):")
        print(f"  pip install modelscope")
        print(f"  modelscope download --dataset {MODELSCOPE_REPO} --local_dir {output_dir}")
        print()
        print("방법 2 — HuggingFace:")
        print(f"  pip install huggingface_hub")
        print(f"  huggingface-cli login")
        print(f"  python scripts/download_speechfake.py -o {output_dir} --method huggingface")
        print()
        print("방법 3 — Git LFS:")
        print(f"  git lfs install")
        print(f"  git clone {MODELSCOPE_GIT} {output_dir}")
        print()
        print("방법 4 — 수동 다운로드:")
        print(f"  https://www.modelscope.cn/datasets/inclusionAI/SPEECHFAKE")
        print(f"  https://huggingface.co/datasets/{HUGGINGFACE_REPO}")
        sys.exit(1)

    # 언어 추출
    if args.extract_languages:
        print()
        extract_dir = output_dir + "_ko" if args.extract_languages == ["ko"] else output_dir + "_filtered"
        extract_language(output_dir, extract_dir, args.extract_languages)

    # 구조 출력
    print_dataset_structure(output_dir)
    print(f"\n저장 완료: {output_dir}")


if __name__ == "__main__":
    main()
