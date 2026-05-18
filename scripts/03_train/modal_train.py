#!/usr/bin/env python3
"""Modal serverless GPU training for deepfake voice detection.

Setup (one-time):
    pip install modal
    modal setup

Upload data to Modal Volume (from project root):
    modal volume create deepfake-voice-data
    modal volume put deepfake-voice-data ./processed processed
    modal volume put deepfake-voice-data ./manifests manifests

Training:
    modal run scripts/03_train/modal_train.py --config aasist_c1
    modal run scripts/03_train/modal_train.py --config aasist_c1 --epochs 5
    modal run scripts/03_train/modal_train.py --all
    modal run scripts/03_train/modal_train.py --all --parallel

Download results:
    modal volume get deepfake-voice-data experiments ./experiments
"""

from __future__ import annotations

from pathlib import Path

import modal

SCRIPT_DIR = Path(__file__).resolve().parent

app = modal.App("deepfake-voice")

volume = modal.Volume.from_name("deepfake-voice-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "soundfile", "numpy", "pandas",
        "scikit-learn", "scipy", "pyyaml", "tqdm",
    )
)

ALL_CONFIGS = [
    "aasist_c1", "aasist_c2", "aasist_c3",
    "lcnn_c1", "lcnn_c2", "lcnn_c3",
    "rawnet2_c1", "rawnet2_c2", "rawnet2_c3",
]

# A10G (24GB VRAM) batch sizes — larger than local 8GB GPU
MODAL_BATCH_SIZES = {
    "aasist": 128,
    "lcnn": 256,
    "rawnet2": 128,
}


@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": volume},
    mounts=[modal.Mount.from_local_dir(str(SCRIPT_DIR), remote_path="/app/train")],
    timeout=3600 * 8,
)
def run_experiment(config_name: str, epochs: int | None = None):
    import os
    import subprocess
    import sys

    import yaml

    os.chdir("/data")

    config_path = f"/app/train/configs/{config_name}.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    if model_name in MODAL_BATCH_SIZES:
        cfg["training"]["batch_size"] = MODAL_BATCH_SIZES[model_name]
    if epochs is not None:
        cfg["training"]["epochs"] = epochs

    tmp_config = "/tmp/config.yaml"
    with open(tmp_config, "w") as f:
        yaml.dump(cfg, f)

    cmd = [sys.executable, "/app/train/train.py", "--config", tmp_config]
    print(f"\n  Config: {config_name} | batch_size: {cfg['training']['batch_size']}")

    subprocess.run(cmd, check=True)
    volume.commit()
    return config_name


@app.local_entrypoint()
def main(
    config: str = "",
    epochs: int = 0,
    all: bool = False,
    parallel: bool = False,
):
    ep = epochs if epochs > 0 else None

    if all:
        if parallel:
            print(f"Running {len(ALL_CONFIGS)} experiments in parallel (9 GPUs)...")
            for result in run_experiment.starmap([(c, ep) for c in ALL_CONFIGS]):
                print(f"  Done: {result}")
        else:
            print(f"Running {len(ALL_CONFIGS)} experiments sequentially...")
            for cfg in ALL_CONFIGS:
                print(f"\n  Starting: {cfg}")
                run_experiment.remote(cfg, ep)
                print(f"  Done: {cfg}")
    elif config:
        run_experiment.remote(config, ep)
    else:
        print("Usage:")
        print("  modal run scripts/03_train/modal_train.py --config aasist_c1")
        print("  modal run scripts/03_train/modal_train.py --config aasist_c1 --epochs 5")
        print("  modal run scripts/03_train/modal_train.py --all")
        print("  modal run scripts/03_train/modal_train.py --all --parallel")
        print()
        print("Data upload (first time):")
        print("  modal volume put deepfake-voice-data ./processed processed")
        print("  modal volume put deepfake-voice-data ./manifests manifests")
        print()
        print("Download results:")
        print("  modal volume get deepfake-voice-data experiments ./experiments")
