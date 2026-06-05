"""Audio dataset for anti-spoofing training and evaluation.

증강 지원:
  augment_config dict를 전달하면 학습 시 on-the-fly 증강을 적용한다.
  augment_config 예시:
    {
      "musan_dir": "data/musan/musan",
      "musan_snr_range": [10, 20],
      "musan_prob": 0.5,
      "codec": "mp3",
      "codec_bitrate": "64k",
      "codec_prob": 0.3,
    }
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """Loads waveform from manifest CSV, pads/crops to fixed length."""

    LABEL_MAP = {"real": 1, "fake": 0}

    def __init__(
        self,
        manifest_path: str | Path,
        num_samples: int = 64_000,
        is_train: bool = True,
        seed: int = 42,
        augment_config: dict | None = None,
    ):
        self.df = pd.read_csv(manifest_path)
        self.num_samples = num_samples
        self.is_train = is_train
        self.rng = random.Random(seed)
        self.augment_config = augment_config if (is_train and augment_config) else None
        self.noise_pool = None

        # MUSAN noise pool 초기화 (학습 시 + config에 musan_dir이 있을 때만)
        if self.augment_config and self.augment_config.get("musan_dir"):
            from augment import MUSANNoisePool
            self.noise_pool = MUSANNoisePool(
                self.augment_config["musan_dir"],
                category=self.augment_config.get("musan_category", "noise"),
                seed=seed,
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        label = self.LABEL_MAP[row["label"]]

        file_path = row["file_path"].replace("\\", "/")
        wav, sr = sf.read(file_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav[:, 0]

        if len(wav) > self.num_samples:
            if self.is_train:
                start = self.rng.randint(0, len(wav) - self.num_samples)
            else:
                start = (len(wav) - self.num_samples) // 2
            wav = wav[start : start + self.num_samples]
        elif len(wav) < self.num_samples:
            wav = np.pad(wav, (0, self.num_samples - len(wav)))

        # 학습 시 증강 적용
        if self.augment_config:
            wav = self._apply_augmentation(wav)

        return torch.from_numpy(wav), label

    def _apply_augmentation(self, wav: np.ndarray) -> np.ndarray:
        """설정에 따라 확률적으로 증강 적용."""
        from augment import add_noise_snr, apply_codec

        cfg = self.augment_config

        # MUSAN noise
        if self.noise_pool and self.rng.random() < cfg.get("musan_prob", 0.5):
            snr_lo, snr_hi = cfg.get("musan_snr_range", [10, 20])
            snr_db = self.rng.uniform(snr_lo, snr_hi)
            noise = self.noise_pool.sample(len(wav))
            wav = add_noise_snr(wav, noise, snr_db)

        # Codec compression
        if cfg.get("codec") and self.rng.random() < cfg.get("codec_prob", 0.3):
            wav = apply_codec(
                wav,
                sr=16_000,
                codec=cfg["codec"],
                bitrate=cfg.get("codec_bitrate", "64k"),
            )

        return wav
