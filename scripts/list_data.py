#!/usr/bin/env python3
"""data/ 폴더의 모든 하위 디렉토리와 wav 파일을 제외한 모든 파일을 표시.

사용법:
    python scripts/list_data.py                      # 기본 data/ 경로
    python scripts/list_data.py path/to/dir          # 특정 경로 지정
    python scripts/list_data.py --tree               # tree 형태로 표시
    python scripts/list_data.py --exclude wav mp3    # 추가 확장자 제외
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SEP = "=" * 60
SUB = "-" * 60


def collect(root: Path, exclude_exts: set[str]) -> tuple[list[Path], list[Path], int]:
    dirs: list[Path] = []
    files: list[Path] = []
    excluded = 0
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            dirs.append(path)
        elif path.is_file():
            if path.suffix.lower().lstrip(".") in exclude_exts:
                excluded += 1
            else:
                files.append(path)
    return dirs, files, excluded


def print_flat(root: Path, exclude_exts: set[str]) -> None:
    dirs, files, excluded = collect(root, exclude_exts)
    excluded_label = ", ".join(f"*.{e}" for e in sorted(exclude_exts)) or "(없음)"

    print()
    print("[디렉토리]")
    print(SUB)
    for d in dirs:
        print(d)

    print()
    print(f"[파일 — {excluded_label} 제외]")
    print(SUB)
    for f in files:
        print(f)

    print()
    print(SEP)
    print(
        f"요약: 디렉토리 {len(dirs)}개 / 비제외 파일 {len(files)}개 / "
        f"제외된 파일 {excluded}개"
    )


def print_tree(root: Path, exclude_exts: set[str]) -> None:
    def walk(directory: Path, prefix: str = "") -> None:
        entries = sorted(
            (p for p in directory.iterdir() if p.is_dir() or p.suffix.lower().lstrip(".") not in exclude_exts),
            key=lambda p: (not p.is_dir(), p.name),
        )
        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            print(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                extension = "    " if i == len(entries) - 1 else "│   "
                walk(entry, prefix + extension)

    print(f"{root}/")
    walk(root)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="data/ 폴더의 모든 하위 디렉토리와 wav 파일을 제외한 파일들을 표시",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="대상 디렉토리 (기본: <repo>/data)",
    )
    parser.add_argument(
        "-t", "--tree",
        action="store_true",
        help="tree 형태로 출력",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=["wav"],
        metavar="EXT",
        help="제외할 확장자 (기본: wav). 예: --exclude wav mp3 flac",
    )
    args = parser.parse_args()

    target = Path(args.target) if args.target else Path(__file__).resolve().parent.parent / "data"

    if not target.exists():
        print(f"오류: 디렉토리가 존재하지 않습니다 — {target}", file=sys.stderr)
        return 1
    if not target.is_dir():
        print(f"오류: 디렉토리가 아닙니다 — {target}", file=sys.stderr)
        return 1

    exclude_exts = {e.lower().lstrip(".") for e in args.exclude}

    print(f"대상 경로: {target}")
    print(SEP)

    if args.tree:
        print_tree(target, exclude_exts)
    else:
        print_flat(target, exclude_exts)

    return 0


if __name__ == "__main__":
    sys.exit(main())
