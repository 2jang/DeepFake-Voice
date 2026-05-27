#!/usr/bin/env python3
"""C1+C2+C3 manifest를 합쳐서 C_all manifest를 생성."""

from pathlib import Path

MANIFEST_ROOT = Path(__file__).resolve().parents[2] / "manifests"
OUT_DIR = MANIFEST_ROOT / "C_all"


def merge(split: str) -> None:
    sources = [MANIFEST_ROOT / f"C{i}" / f"{split}.csv" for i in (1, 2, 3)]
    header = None
    rows: list[str] = []

    for src in sources:
        if not src.exists():
            print(f"  SKIP {src} (not found)")
            continue
        lines = src.read_text(encoding="utf-8").splitlines()
        if header is None:
            header = lines[0]
        rows.extend(lines[1:])
        print(f"  {src.name} from {src.parent.name}: {len(lines) - 1:,} rows")

    out_path = OUT_DIR / f"{split}.csv"
    out_path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"  -> {out_path}: {len(rows):,} total\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        print(f"[{split}]")
        merge(split)
    print("Done.")


if __name__ == "__main__":
    main()
