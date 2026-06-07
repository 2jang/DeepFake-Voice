#!/usr/bin/env python3
"""
ASVspoof 2019 LA 다운로드 스크립트
==================================
- 출처: Edinburgh DataShare (https://datashare.ed.ac.uk/handle/10283/3336)
- 규모: LA eval — 7,355 bonafide + 63,882 spoof = 71,237 samples
- 오디오: FLAC, 16kHz mono
- 라이선스: CC BY 4.0
- 평가용(테스트 전용) — 학습에 사용하지 않음

구조 (다운로드 후):
    data/asvspoof19/
    ├── LA/
    │   ├── ASVspoof2019_LA_eval/
    │   │   └── flac/           # 71,237 FLAC files
    │   └── ASVspoof2019_LA_cm_protocols/
    │       └── ASVspoof2019.LA.cm.eval.trl.txt
    └── LA.zip                  # 원본 (삭제 가능)

프로토콜 형식 (공백 구분):
    LA_0069 LA_E_1000147 - A07 spoof
    LA_0069 LA_E_1000273 - -   bonafide

사용법:
    python scripts/01_download/download_asvspoof19.py --output data/asvspoof19
    python scripts/01_download/download_asvspoof19.py --output data/asvspoof19 --eval-only
"""

import argparse
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

# Edinburgh DataShare direct download
LA_ZIP_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"
LA_ZIP_MD5 = "30c98f11d8b2bc21f2c257bfd78bb5c5"
LA_ZIP_SIZE_GB = 7.1


def md5_check(filepath: Path, expected: str) -> bool:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def download_wget(url: str, output: Path) -> bool:
    """wget으로 다운로드 (resume 지원)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-c", "-O", str(output), url]
    print(f"  다운로드 시작: {url}")
    print(f"  저장 위치: {output}")
    print(f"  예상 크기: ~{LA_ZIP_SIZE_GB} GB")
    result = subprocess.run(cmd)
    return result.returncode == 0


def download_curl(url: str, output: Path) -> bool:
    """curl로 다운로드 (resume 지원)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-C", "-", "-L", "-o", str(output), url]
    print(f"  다운로드 시작: {url}")
    print(f"  저장 위치: {output}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="ASVspoof 2019 LA 다운로드")
    parser.add_argument("--output", type=str, default="data/asvspoof19",
                        help="저장 디렉토리")
    parser.add_argument("--eval-only", action="store_true",
                        help="eval 파티션만 유지 (train/dev 삭제)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="MD5 검증 건너뛰기")
    parser.add_argument("--method", choices=["wget", "curl"], default="wget",
                        help="다운로드 방법")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "LA.zip"

    # --- 다운로드 ---
    if zip_path.exists():
        print(f"  이미 존재: {zip_path}")
    else:
        if args.method == "wget":
            ok = download_wget(LA_ZIP_URL, zip_path)
        else:
            ok = download_curl(LA_ZIP_URL, zip_path)
        if not ok:
            print("  다운로드 실패!")
            return 1

    # --- MD5 검증 ---
    if not args.skip_verify:
        print("  MD5 검증 중...")
        if md5_check(zip_path, LA_ZIP_MD5):
            print("  MD5 OK ✓")
        else:
            print("  MD5 불일치! 파일이 손상되었을 수 있습니다.")
            print(f"  기대: {LA_ZIP_MD5}")
            return 1

    # --- 압축 해제 ---
    la_dir = output_dir / "LA"
    if la_dir.exists():
        print(f"  이미 압축 해제됨: {la_dir}")
    else:
        print("  압축 해제 중...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        print("  압축 해제 완료")

    # --- eval-only: train/dev 삭제 ---
    if args.eval_only:
        for subdir in ["ASVspoof2019_LA_train", "ASVspoof2019_LA_dev"]:
            p = la_dir / subdir
            if p.exists():
                import shutil
                shutil.rmtree(p)
                print(f"  삭제: {p}")

    # --- 확인 ---
    eval_flac = la_dir / "ASVspoof2019_LA_eval" / "flac"
    protocol = la_dir / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"

    if eval_flac.exists():
        n_files = sum(1 for _ in eval_flac.glob("*.flac"))
        print(f"  Eval FLAC 파일: {n_files}개")
    else:
        print(f"  ⚠ eval 디렉토리 없음: {eval_flac}")

    if protocol.exists():
        n_lines = sum(1 for _ in open(protocol))
        print(f"  프로토콜 라인: {n_lines}개")
    else:
        print(f"  ⚠ 프로토콜 없음: {protocol}")

    print("\n  완료! 다음 단계:")
    print(f"    python scripts/02_preprocess/prepare_benchmarks.py --dataset asvspoof19 --data-dir {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
