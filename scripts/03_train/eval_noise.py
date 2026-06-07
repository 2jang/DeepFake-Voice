#!/usr/bin/env python3
"""Phase 4-A: 테스트 시 노이즈 강건성 평가.

clean/noise 학습 모델 모두를 대상으로, 테스트 데이터에 다양한 노이즈를
추가하여 평가한다 (재학습 없음).

  학습 조건 × 평가 조건 2×2 매트릭스:
    clean 학습 + clean 평가  → 기본 성능 (Phase 4에서 이미 측정)
    clean 학습 + noise 평가  → 강건성 하락 측정
    noise 학습 + clean 평가  → 증강 효과 (Phase 4-B train.py에서 측정)
    noise 학습 + noise 평가  → 증강이 실전 노이즈에서 얼마나 도움되는지

노이즈 조건: gaussian, MUSAN(20/10/5dB), codec(MP3 64kbps)
모델: clean 2개 + augmentation 6개 = 8개 프리셋 + --config로 임의 추가

사용법:
    # 전체 실행 (8모델 × 6조건 × 3테스트셋 = 144개 평가)
    python scripts/03_train/eval_noise.py --musan-dir data/musan/musan

    # clean 학습 모델만
    python scripts/03_train/eval_noise.py --models specrnet lcnn

    # augmentation 모델만
    python scripts/03_train/eval_noise.py --models specrnet_aug_musan specrnet_aug_both

    # 임의 config 경로로 평가
    python scripts/03_train/eval_noise.py --config scripts/03_train/configs/my_model.yaml

    # 특정 조건/테스트셋만
    python scripts/03_train/eval_noise.py --conditions clean musan_10db codec_mp3_64
    python scripts/03_train/eval_noise.py --eval-sets T1 T2
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from augment import MUSANNoisePool, add_gaussian_noise, add_noise_snr, apply_codec
from dataset import AudioDataset
from models import get_model

# ---------------------------------------------------------------------------
# Noise conditions
# ---------------------------------------------------------------------------

NOISE_CONDITIONS = {
    "clean":        {"type": "none"},
    # α=0.001 → 실효 SNR ~50dB로 거의 clean. 임동훈(2024) Table 3 조건 재현용이며,
    # 강건성 '하락'을 보기보다 동일 조건 재현이 목적 (결과는 clean과 비슷하게 나옴).
    "gaussian_001": {"type": "gaussian", "alpha": 0.001},
    "musan_20db":   {"type": "musan", "snr_db": 20},
    "musan_10db":   {"type": "musan", "snr_db": 10},
    "musan_5db":    {"type": "musan", "snr_db": 5},
    "codec_mp3_64": {"type": "codec", "codec": "mp3", "bitrate": "64k"},
}

# Models to evaluate: clean-trained + augmentation-trained
MODEL_CONFIGS = {
    # clean 학습
    "specrnet":           "scripts/03_train/configs/specrnet_call.yaml",
    "lcnn":               "scripts/03_train/configs/lcnn_call.yaml",
    # noise 학습 (augmentation)
    "specrnet_aug_musan": "scripts/03_train/configs/specrnet_call_aug_musan.yaml",
    "specrnet_aug_codec": "scripts/03_train/configs/specrnet_call_aug_codec.yaml",
    "specrnet_aug_both":  "scripts/03_train/configs/specrnet_call_aug_both.yaml",
    "lcnn_aug_musan":     "scripts/03_train/configs/lcnn_call_aug_musan.yaml",
    "lcnn_aug_codec":     "scripts/03_train/configs/lcnn_call_aug_codec.yaml",
    "lcnn_aug_both":      "scripts/03_train/configs/lcnn_call_aug_both.yaml",
}


# ---------------------------------------------------------------------------
# Noisy dataset wrapper
# ---------------------------------------------------------------------------

class NoisyAudioDataset(Dataset):
    """AudioDataset 위에 테스트 시 노이즈를 추가하는 래퍼.

    학습은 하지 않으므로 is_train=False 고정.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        num_samples: int,
        noise_config: dict,
        musan_pool: MUSANNoisePool | None = None,
        sr: int = 16_000,
        seed: int = 42,
    ):
        self.base = AudioDataset(manifest_path, num_samples, is_train=False, seed=seed)
        self.noise_config = noise_config
        self.musan_pool = musan_pool
        self.sr = sr
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        wav_tensor, label = self.base[idx]
        wav = wav_tensor.numpy()

        ntype = self.noise_config["type"]
        if ntype == "none":  # 원본 유지
            return torch.from_numpy(wav), label

        # 샘플 인덱스 기반 결정적 RNG: DataLoader 워커 수·모델·실행 순서와
        # 무관하게 동일 인덱스엔 항상 동일한 노이즈가 적용된다.
        #   → 재현 가능한 강건성 표 + SpecRNet/LCNN이 동일 입력으로 평가됨.
        rng = np.random.default_rng(self.seed + idx)

        if ntype == "gaussian":
            wav = add_gaussian_noise(wav, alpha=self.noise_config["alpha"], rng=rng)
        elif ntype == "musan":
            if self.musan_pool is None:
                raise RuntimeError("MUSAN pool이 필요합니다 (--musan-dir 지정)")
            noise = self.musan_pool.sample(len(wav), rng=rng)
            wav = add_noise_snr(wav, noise, self.noise_config["snr_db"])
        elif ntype == "codec":
            wav = apply_codec(
                wav, sr=self.sr,
                codec=self.noise_config["codec"],
                bitrate=self.noise_config["bitrate"],
            )

        return torch.from_numpy(wav), label


# ---------------------------------------------------------------------------
# Evaluate function (train.py에서 임포트)
# ---------------------------------------------------------------------------

from train import evaluate as evaluate_model
from train import load_checkpoint


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="테스트 시 노이즈 강건성 평가")
    parser.add_argument("--musan-dir", type=str, default="data/musan/musan",
                        help="MUSAN 디렉토리 (noise 하위폴더 포함)")
    parser.add_argument("--conditions", nargs="+", default=list(NOISE_CONDITIONS.keys()),
                        choices=list(NOISE_CONDITIONS.keys()),
                        help="평가할 노이즈 조건")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--models", nargs="+", default=None,
                       choices=list(MODEL_CONFIGS.keys()),
                       help="평가할 모델 (프리셋)")
    group.add_argument("--config", nargs="+", type=str,
                       help="임의 YAML config 경로 (프리셋 외 모델 평가 시)")
    parser.add_argument("--eval-sets", nargs="+",
                        default=["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"],
                        help="평가할 테스트셋 (manifest 없으면 자동 건너뜀)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="experiments/noise_robustness.csv")
    args = parser.parse_args()

    # --models/--config 모두 미지정 시 전체 프리셋
    if args.models is None and args.config is None:
        args.models = list(MODEL_CONFIGS.keys())

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )

    # MUSAN pool (MUSAN 조건이 포함된 경우만)
    musan_pool = None
    need_musan = any(
        NOISE_CONDITIONS[c]["type"] == "musan" for c in args.conditions
    )
    if need_musan:
        musan_pool = MUSANNoisePool(args.musan_dir, category="noise")

    all_rows: list[dict] = []

    # --config: 임의 경로 → (이름, 경로) 리스트 구성
    if args.config:
        config_entries = []
        for p in args.config:
            name = Path(p).stem
            config_entries.append((name, p))
    else:
        config_entries = [(name, MODEL_CONFIGS[name]) for name in args.models]

    for model_name, config_path in config_entries:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        experiment_name = cfg["experiment"]
        num_samples = cfg["data"]["num_samples"]
        batch_size = cfg["training"]["batch_size"]
        num_workers = cfg["data"].get("num_workers", 4)
        out_root = Path(cfg.get("output_root", "experiments"))
        ckpt_path = out_root / "checkpoints" / f"{experiment_name}_best.pt"

        if not ckpt_path.exists():
            print(f"  체크포인트 없음: {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  Model: {model_name} ({experiment_name})")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        # 모델 로드
        model = get_model(cfg["model"]["name"]).to(device)
        load_checkpoint(model, ckpt_path)
        if device.type == "cuda":
            model = torch.compile(model)

        for cond_name in args.conditions:
            cond = NOISE_CONDITIONS[cond_name]
            print(f"\n  --- Condition: {cond_name} ({cond}) ---")

            for eval_name in args.eval_sets:
                eval_path = cfg["data"]["eval_manifests"].get(eval_name)
                if not eval_path or not Path(eval_path).exists():
                    print(f"    {eval_name}: manifest 없음")
                    continue

                ds = NoisyAudioDataset(
                    eval_path, num_samples, cond,
                    musan_pool=musan_pool,
                )
                loader = DataLoader(
                    ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                )

                m = evaluate_model(model, loader, device)

                print(f"    {eval_name:5s}  EER={m['eer']:.4f}  minDCF={m['mindcf']:.4f}  "
                      f"Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  "
                      f"FPR={m['fpr']:.4f}  FNR={m['fnr']:.4f}")

                row = {
                    "model": model_name,
                    "experiment": experiment_name,
                    "noise_condition": cond_name,
                    "eval_set": eval_name,
                }
                for k, v in m.items():
                    row[k] = f"{v:.6f}"
                all_rows.append(row)

    # --- 결과 저장 ---
    if not all_rows:
        print("\n결과 없음.")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(all_rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\n{'='*60}")
    print(f"  결과 저장: {output_path} ({len(all_rows)}개 평가)")
    print(f"{'='*60}")

    # --- 요약 테이블 ---
    print(f"\n  === 노이즈 강건성 요약 (EER %) ===")
    header = f"  {'Model':22s} {'Condition':15s}"
    for es in args.eval_sets:
        header += f" {es:>8s}"
    print(header)
    print(f"  {'-'*(len(header)-2)}")

    for row in all_rows:
        if row["eval_set"] == args.eval_sets[0]:
            line = f"  {row['model']:22s} {row['noise_condition']:15s}"
            for es in args.eval_sets:
                matched = [r for r in all_rows
                           if r["model"] == row["model"]
                           and r["noise_condition"] == row["noise_condition"]
                           and r["eval_set"] == es]
                if matched:
                    line += f" {float(matched[0]['eer'])*100:8.2f}"
                else:
                    line += f" {'N/A':>8s}"
            print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
