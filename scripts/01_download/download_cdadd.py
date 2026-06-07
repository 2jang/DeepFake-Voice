#!/usr/bin/env python3
"""
CD-ADD (Cross-Domain Audio Deepfake Detection) 다운로드 스크립트
================================================================
- 논문: Li et al. (2024) "Cross-Domain Audio Deepfake Detection:
        Dataset and Analysis" (EMNLP 2024)
- GitHub: https://github.com/leolya/CD-ADD
- 규모: 300+ 시간, 5개 zero-shot TTS 모델
- 다운로드: OpenXLab 또는 Baidu Cloud
- 라이선스: 연구 목적

TTS 모델:
    - XTTS-v2, OpenVoice, CosyVoice, FishSpeech, GPT-SoVITS

구조 (다운로드 후):
    data/cdadd/
    ├── bonafide/          # 실제 음성
    └── fake/              # TTS 생성 음성
        ├── XTTS-v2/
        ├── OpenVoice/
        ├── CosyVoice/
        ├── FishSpeech/
        └── GPT-SoVITS/

사용법:
    # OpenXLab에서 다운로드 (추천)
    python scripts/01_download/download_cdadd.py --output data/cdadd --method openxlab

    # 수동 다운로드 안내
    python scripts/01_download/download_cdadd.py --output data/cdadd --method manual

사전 준비 (OpenXLab):
    pip install openxlab
"""

import argparse
import subprocess
import sys
from pathlib import Path


OPENXLAB_DATASET = "lyaleo/CD-ADD"
BAIDU_URL = "https://pan.baidu.com/s/1ORGwECKKnLM-1oY2HvGiqw"
BAIDU_PWD = "agip"
GITHUB_URL = "https://github.com/leolya/CD-ADD"


def download_openxlab(output_dir: Path) -> bool:
    """OpenXLab CLI로 다운로드."""
    try:
        from openxlab.dataset import get as oxl_get
        print(f"  OpenXLab에서 다운로드: {OPENXLAB_DATASET}")
        print(f"  저장 위치: {output_dir}")
        oxl_get(dataset_repo=OPENXLAB_DATASET, target_path=str(output_dir))
        return True
    except ImportError:
        print("  openxlab 패키지가 필요합니다:")
        print("    pip install openxlab")
        return False
    except Exception as e:
        print(f"  OpenXLab 다운로드 실패: {e}")
        return False


def show_manual_instructions(output_dir: Path):
    """수동 다운로드 안내."""
    print(f"""
{'='*60}
  CD-ADD 수동 다운로드 안내
{'='*60}

  방법 1: OpenXLab (추천)
    pip install openxlab
    python -c "from openxlab.dataset import get; get('{OPENXLAB_DATASET}', target_path='{output_dir}')"

  방법 2: Baidu Cloud (중국 계정 필요)
    URL: {BAIDU_URL}
    비밀번호: {BAIDU_PWD}
    다운로드 후 {output_dir}/ 에 압축 해제

  방법 3: GitHub 클론 후 안내 참조
    git clone {GITHUB_URL}
    → README의 다운로드 링크 확인

  다운로드 완료 후:
    python scripts/02_preprocess/prepare_benchmarks.py --dataset cdadd --data-dir {output_dir}
{'='*60}
""")


def main():
    parser = argparse.ArgumentParser(description="CD-ADD 다운로드")
    parser.add_argument("--output", type=str, default="data/cdadd",
                        help="저장 디렉토리")
    parser.add_argument("--method", choices=["openxlab", "manual"],
                        default="manual", help="다운로드 방법")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "openxlab":
        if not download_openxlab(output_dir):
            print("\n  OpenXLab 실패. 수동 다운로드를 시도하세요:")
            show_manual_instructions(output_dir)
            return 1
    else:
        show_manual_instructions(output_dir)
        return 0

    # 확인
    n_files = sum(1 for _ in output_dir.rglob("*.wav"))
    n_flac = sum(1 for _ in output_dir.rglob("*.flac"))
    print(f"  오디오 파일: {n_files} WAV + {n_flac} FLAC")
    print(f"\n  완료! 다음 단계:")
    print(f"    python scripts/02_preprocess/prepare_benchmarks.py --dataset cdadd --data-dir {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
