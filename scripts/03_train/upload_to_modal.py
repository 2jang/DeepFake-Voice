#!/usr/bin/env python3
"""Upload processed data to Modal Volume in chunks to avoid MemoryError.

Usage (from project root on Windows):
    python scripts/03_train/upload_to_modal.py
    python scripts/03_train/upload_to_modal.py --dataset koaad
    python scripts/03_train/upload_to_modal.py --manifests-only
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DATASETS = ["koaad", "mailabs", "mlaad", "speechfake-md", "speechfake-real"]
VOLUME = "deepfake-voice-data"


def run_cmd(cmd: list[str]) -> bool:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        return False
    return True


def upload_manifests() -> None:
    print("\n=== Uploading manifests ===")
    run_cmd(["modal", "volume", "put", VOLUME, "manifests", "manifests", "--force"])


BATCH_SIZE = 20


def upload_dataset(dataset: str, processed_root: Path) -> None:
    dataset_dir = processed_root / dataset
    if not dataset_dir.exists():
        print(f"  Skip {dataset} (not found: {dataset_dir})")
        return

    subdirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    if not subdirs:
        print(f"  Skip {dataset} (no subdirectories)")
        return

    total_batches = (len(subdirs) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n=== {dataset}: {len(subdirs)} speaker dirs, {total_batches} batches (×{BATCH_SIZE}) ===")

    for b in range(total_batches):
        batch = subdirs[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        names = [d.name for d in batch]
        print(f"  [batch {b+1}/{total_batches}] {names[0]} ~ {names[-1]}")

        for speaker_dir in batch:
            ok = run_cmd([
                "modal", "volume", "put", VOLUME,
                str(speaker_dir), f"processed/{dataset}/{speaker_dir.name}", "--force",
            ])
            if not ok:
                print(f"  ERROR on {speaker_dir.name} — stopping {dataset}")
                return

    print(f"  Done: {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="processed",
                        help="Path to processed/ directory")
    parser.add_argument("--dataset", default=None,
                        help="Upload a single dataset (e.g. koaad)")
    parser.add_argument("--manifests-only", action="store_true",
                        help="Only upload manifests/")
    args = parser.parse_args()

    processed_root = Path(args.processed_root)

    if args.manifests_only:
        upload_manifests()
        return

    upload_manifests()

    datasets = [args.dataset] if args.dataset else DATASETS
    for ds in datasets:
        upload_dataset(ds, processed_root)

    print("\n=== All uploads complete ===")
    print("Verify: modal volume ls deepfake-voice-data processed/")


if __name__ == "__main__":
    main()
