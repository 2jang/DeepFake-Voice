#!/usr/bin/env python3
"""TensorBoard 서버 단독 실행.

사용법:
    python scripts/03_train/launch_tb.py
    python scripts/03_train/launch_tb.py --port 8080
    python scripts/03_train/launch_tb.py --logdir experiments/tb_logs
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch TensorBoard server")
    parser.add_argument("--logdir", default="experiments/tb_logs",
                        help="TensorBoard log directory (default: experiments/tb_logs)")
    parser.add_argument("--port", type=int, default=6006, help="Port (default: 6006)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0 for remote access)")
    args = parser.parse_args()

    logdir = Path(args.logdir)
    if not logdir.exists():
        logdir.mkdir(parents=True, exist_ok=True)
        print(f"Created log directory: {logdir}")

    # 실험별 로그 목록 출력
    subdirs = sorted(p.name for p in logdir.iterdir() if p.is_dir())
    if subdirs:
        print(f"Experiments found: {', '.join(subdirs)}")
    else:
        print(f"No experiment logs yet in {logdir}")

    print(f"TensorBoard: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.\n")

    try:
        return subprocess.call(
            [sys.executable, "-m", "tensorboard.main",
             "--logdir", str(logdir),
             "--host", args.host,
             "--port", str(args.port)],
        )
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
