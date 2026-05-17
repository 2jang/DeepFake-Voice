#!/usr/bin/env python3
"""data/ 폴더의 모든 하위 디렉토리와 특정 확장자(wav, mp3, json)를 제외한 모든 파일을 표시.

사용법:
    python scripts/utils/list_data.py                      # 기본 data/ 경로
    python scripts/utils/list_data.py path/to/dir          # 특정 경로 지정
    python scripts/utils/list_data.py --tree               # tree 형태로 표시
    python scripts/utils/list_data.py --exclude wav flac   # 제외할 확장자 수동 지정
    python scripts/utils/list_data.py -o output.txt        # 결과를 txt 파일로 저장
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path


SEP = "=" * 60
SUB = "-" * 60


def collect(root: Path, exclude_exts: set[str]) -> tuple[list[Path], list[Path], int, dict[Path, set[str]]]:
    dirs: list[Path] = []
    files: list[Path] = []
    excluded = 0
    excluded_in_dir: dict[Path, set[str]] = {}

    for path in sorted(root.rglob("*")):
        if path.is_dir():
            dirs.append(path)
        elif path.is_file():
            ext = path.suffix.lower().lstrip(".")
            if ext in exclude_exts:
                excluded += 1
                if path.parent not in excluded_in_dir:
                    excluded_in_dir[path.parent] = set()
                excluded_in_dir[path.parent].add(ext)
            else:
                files.append(path)
    return dirs, files, excluded, excluded_in_dir


def print_flat(root: Path, exclude_exts: set[str]) -> None:
    dirs, files, excluded, excluded_in_dir = collect(root, exclude_exts)
    excluded_label = ", ".join(f"*.{e}" for e in sorted(exclude_exts)) or "(없음)"

    print()
    print("[디렉토리]")
    print(SUB)
    for d in dirs:
        exts = excluded_in_dir.get(d)
        if exts:
            ext_str = ", ".join(sorted(exts))
            print(f"{d} ({ext_str} files)")
        else:
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
    def get_excluded_exts(directory: Path) -> set[str]:
        return {
            p.suffix.lower().lstrip(".")
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower().lstrip(".") in exclude_exts
        }

    def walk(directory: Path, prefix: str = "") -> None:
        entries = sorted(
            (p for p in directory.iterdir() if p.is_dir() or p.suffix.lower().lstrip(".") not in exclude_exts),
            key=lambda p: (not p.is_dir(), p.name),
        )
        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            if entry.is_dir():
                exts = get_excluded_exts(entry)
                ext_str = f" ({', '.join(sorted(exts))} files)" if exts else ""
                print(f"{prefix}{connector}{entry.name}/{ext_str}")
                
                extension = "    " if i == len(entries) - 1 else "│   "
                walk(entry, prefix + extension)
            else:
                print(f"{prefix}{connector}{entry.name}")

    # 루트 디렉토리 처리
    exts = get_excluded_exts(root)
    ext_str = f" ({', '.join(sorted(exts))} files)" if exts else ""
    print(f"{root}/{ext_str}")
    walk(root)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="data/ 폴더의 모든 하위 디렉토리와 지정된 확장자(기본: wav, mp3, json)를 제외한 파일들을 표시",
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
        default=["wav", "mp3", "json"],
        metavar="EXT",
        help="제외할 확장자 (기본: wav mp3 json). 예: --exclude wav mp3 flac",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="결과를 저장할 txt 파일 경로 (예: result.txt)",
    )
    args = parser.parse_args()

    target = Path(args.target) if args.target else Path(__file__).resolve().parent.parent.parent / "data"

    if not target.exists():
        print(f"오류: 디렉토리가 존재하지 않습니다 — {target}", file=sys.stderr)
        return 1
    if not target.is_dir():
        print(f"오류: 디렉토리가 아닙니다 — {target}", file=sys.stderr)
        return 1

    exclude_exts = {e.lower().lstrip(".") for e in args.exclude}

    if args.output:
        print(f"결과를 '{args.output}' 파일에 저장 중입니다...")

    # contextlib.redirect_stdout을 사용하여 파일 출력과 터미널 출력을 제어
    with contextlib.ExitStack() as stack:
        if args.output:
            f = stack.enter_context(open(args.output, "w", encoding="utf-8"))
            stack.enter_context(contextlib.redirect_stdout(f))

        print(f"대상 경로: {target}")
        print(SEP)

        if args.tree:
            print_tree(target, exclude_exts)
        else:
            print_flat(target, exclude_exts)

    if args.output:
        print("저장이 완료되었습니다!")

    return 0


if __name__ == "__main__":
    sys.exit(main())