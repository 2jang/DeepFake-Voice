#!/usr/bin/env python3
"""
InTheWild 오디오 딥페이크 데이터셋 다운로드 스크립트
=====================================================
- 논문: Müller et al. (2022) "Does Audio Deepfake Detection Generalize?"
        (Interspeech 2022)
- HuggingFace: https://huggingface.co/datasets/mueller91/In-The-Wild
- 규모: 37.9시간 (20.8h bonafide + 17.2h spoof), 58명 유명인
- 라이선스: CC-BY-SA-4.0

구조 (다운로드 후):
    data/inthewild/
    ├── release_in_the_wild/
    │   ├── meta.csv              # 라벨 파일
    │   └── *.wav                 # 오디오 파일
    └── release_in_the_wild.zip   # 원본 (삭제 가능)

meta.csv 형식:
    file,speaker,label
    ljspeech_0.wav,Linda Johnson,spoof
    ...

사용법:
    # HuggingFace에서 다운로드 (wget)
    python scripts/01_download/download_inthewild.py --output data/inthewild

    # huggingface_hub 사용
    python scripts/01_download/download_inthewild.py --output data/inthewild --method hf

사전 준비 (hf 방식):
    pip install huggingface_hub
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


HF_ZIP_URL = "https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip"
HF_REPO = "mueller91/In-The-Wild"
ZIP_SIZE_GB = 8.16


def download_wget(url: str, output: Path) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-c", "-O", str(output), url]
    print(f"  다운로드 시작: {url}")
    print(f"  예상 크기: ~{ZIP_SIZE_GB} GB")
    result = subprocess.run(cmd)
    return result.returncode == 0


def download_hf(output_dir: Path) -> bool:
    try:
        from huggingface_hub import hf_hub_download
        print(f"  HuggingFace Hub에서 다운로드: {HF_REPO}")
        hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename="release_in_the_wild.zip",
            local_dir=str(output_dir),
        )
        return True
    except ImportError:
        print("  huggingface_hub 패키지가 필요합니다:")
        print("    pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"  다운로드 실패: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="InTheWild 데이터셋 다운로드")
    parser.add_argument("--output", type=str, default="data/inthewild",
                        help="저장 디렉토리")
    parser.add_argument("--method", choices=["wget", "curl", "hf"],
                        default="wget", help="다운로드 방법")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "release_in_the_wild.zip"

    # --- 다운로드 ---
    if zip_path.exists():
        print(f"  이미 존재: {zip_path}")
    elif args.method == "hf":
        if not download_hf(output_dir):
            return 1
    elif args.method == "wget":
        if not download_wget(HF_ZIP_URL, zip_path):
            return 1
    else:  # curl
        cmd = ["curl", "-C", "-", "-L", "-o", str(zip_path), HF_ZIP_URL]
        if subprocess.run(cmd).returncode != 0:
            return 1

    # --- 압축 해제 ---
    release_dir = output_dir / "release_in_the_wild"
    if release_dir.exists():
        print(f"  이미 압축 해제됨: {release_dir}")
    else:
        print("  압축 해제 중...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        print("  압축 해제 완료")

    # --- 확인 ---
    meta_csv = release_dir / "meta.csv"
    if meta_csv.exists():
        n_lines = sum(1 for _ in open(meta_csv)) - 1  # header 제외
        print(f"  샘플 수: {n_lines}")
    else:
        print(f"  ⚠ meta.csv 없음: {meta_csv}")

    n_wav = sum(1 for _ in release_dir.glob("*.wav"))
    print(f"  WAV 파일: {n_wav}개")

    print(f"\n  완료! 다음 단계:")
    print(f"    python scripts/02_preprocess/prepare_benchmarks.py --dataset inthewild --data-dir {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
