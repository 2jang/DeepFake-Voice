#!/usr/bin/env python3
"""
ASVspoof 2021 DF/LA 다운로드 스크립트
=====================================
- DF (DeepFake): Zenodo https://zenodo.org/records/4835108
  → ~34.5 GB (4 parts), 20,637 bonafide + 572,616 spoof
- LA (Logical Access): Zenodo https://zenodo.org/records/4837263
  → ~8 GB, telephony/VoIP 채널 효과 포함
- Eval keys: https://www.asvspoof.org/index2021.html 에서 별도 다운로드
- 오디오: FLAC, 16kHz mono
- 라이선스: Open Data Commons Attribution Licence

구조 (다운로드 후):
    data/asvspoof21/
    ├── DF/
    │   ├── ASVspoof2021_DF_eval/
    │   │   └── flac/
    │   └── keys/               # asvspoof.org에서 수동 다운로드
    │       └── DF/
    │           └── CM/
    │               └── trial_metadata.txt
    └── LA/
        ├── ASVspoof2021_LA_eval/
        │   └── flac/
        └── keys/
            └── LA/
                └── CM/
                    └── trial_metadata.txt

프로토콜 형식 (2021, 공백 구분):
    LA_0001 LA_E_0000001 - - bonafide notrim  prog A01 alaw
    LA_0001 LA_E_0000002 A07 - spoof notrim  prog A01 alaw

사용법:
    # DF만 다운로드
    python scripts/01_download/download_asvspoof21.py --output data/asvspoof21 --track DF

    # LA만 다운로드
    python scripts/01_download/download_asvspoof21.py --output data/asvspoof21 --track LA

    # 둘 다
    python scripts/01_download/download_asvspoof21.py --output data/asvspoof21 --track DF LA

주의:
    eval keys는 asvspoof.org에서 수동 다운로드 후 keys/ 디렉토리에 배치해야 합니다.
    https://www.asvspoof.org/index2021.html → "Keys and samples" 섹션
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Zenodo direct download URLs
ZENODO_URLS = {
    "DF": {
        "base": "https://zenodo.org/records/4835108/files",
        "files": [
            "ASVspoof2021_DF_eval_part00.tar.gz",
            "ASVspoof2021_DF_eval_part01.tar.gz",
            "ASVspoof2021_DF_eval_part02.tar.gz",
            "ASVspoof2021_DF_eval_part03.tar.gz",
        ],
        "size_gb": 34.5,
    },
    "LA": {
        "base": "https://zenodo.org/records/4837263/files",
        "files": [
            "ASVspoof2021_LA_eval.tar.gz",
        ],
        "size_gb": 8.0,
    },
}


def download_file(url: str, output: Path, method: str = "wget") -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"    이미 존재: {output.name}")
        return True

    if method == "wget":
        cmd = ["wget", "-c", "-O", str(output), url]
    else:
        cmd = ["curl", "-C", "-", "-L", "-o", str(output), url]

    print(f"    다운로드: {output.name}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def extract_tar(filepath: Path, output_dir: Path) -> bool:
    print(f"    압축 해제: {filepath.name}")
    cmd = ["tar", "-xzf", str(filepath), "-C", str(output_dir)]
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="ASVspoof 2021 DF/LA 다운로드")
    parser.add_argument("--output", type=str, default="data/asvspoof21",
                        help="저장 디렉토리")
    parser.add_argument("--track", nargs="+", default=["DF", "LA"],
                        choices=["DF", "LA"],
                        help="다운로드할 트랙")
    parser.add_argument("--method", choices=["wget", "curl"], default="wget")
    args = parser.parse_args()

    output_dir = Path(args.output)

    for track in args.track:
        info = ZENODO_URLS[track]
        track_dir = output_dir / track
        track_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  ASVspoof 2021 {track}")
        print(f"  예상 크기: ~{info['size_gb']} GB")
        print(f"  파일 수: {len(info['files'])}")
        print(f"{'='*60}")

        # 다운로드
        for fname in info["files"]:
            url = f"{info['base']}/{fname}"
            fpath = track_dir / fname
            if not download_file(url, fpath, args.method):
                print(f"    실패: {fname}")
                return 1

        # 압축 해제
        eval_dir = track_dir / f"ASVspoof2021_{track}_eval"
        if eval_dir.exists():
            print(f"    이미 압축 해제됨: {eval_dir}")
        else:
            for fname in info["files"]:
                fpath = track_dir / fname
                if not extract_tar(fpath, track_dir):
                    print(f"    압축 해제 실패: {fname}")
                    return 1

        # 확인
        flac_dir = eval_dir / "flac"
        if flac_dir.exists():
            n_files = sum(1 for _ in flac_dir.glob("*.flac"))
            print(f"    FLAC 파일: {n_files}개")
        else:
            print(f"    ⚠ flac 디렉토리 없음: {flac_dir}")

    # eval keys 안내
    print(f"\n{'='*60}")
    print("  ⚠ Eval Keys 수동 다운로드 필요:")
    print("    1. https://www.asvspoof.org/index2021.html 접속")
    print("    2. 'Keys and samples' 섹션에서 keys 다운로드")
    print(f"    3. {output_dir}/DF/keys/ 및 {output_dir}/LA/keys/ 에 배치")
    print(f"{'='*60}")

    print("\n  완료! 다음 단계:")
    print(f"    python scripts/02_preprocess/prepare_benchmarks.py --dataset asvspoof21_df --data-dir {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
