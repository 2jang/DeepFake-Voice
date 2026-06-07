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
      "codec_cache_dir": "data/codec_cache/mp3_64k",  # 사전 계산 캐시
    }

  codec_cache_dir가 있으면 사전 계산된 파일을 로드한다 (precompute_codec.py로 생성).
  없으면 on-the-fly 인코딩 (느림: ~40 it/s → ~9 it/s).
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
        self.codec_cache_dir = None

        if self.augment_config:
            if self.augment_config.get("musan_dir"):
                from augment import MUSANNoisePool
                self.noise_pool = MUSANNoisePool(
                    self.augment_config["musan_dir"],
                    category=self.augment_config.get("musan_category", "noise"),
                    seed=seed,
                )
            if self.augment_config.get("codec_cache_dir"):
                self.codec_cache_dir = Path(self.augment_config["codec_cache_dir"])
                if self.codec_cache_dir.exists():
                    print(f"  Codec cache: {self.codec_cache_dir}")
                else:
                    print(f"  Codec cache 없음: {self.codec_cache_dir} → on-the-fly 인코딩 (느림)")
                    self.codec_cache_dir = None

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_codec_path(self, file_path: str) -> Path | None:
        """캐시 디렉토리에서 codec 사전 계산 파일 경로를 반환. 없으면 None."""
        if self.codec_cache_dir is None:
            return None
        p = Path(file_path)
        rel = p.relative_to(Path.cwd()) if p.is_absolute() else p
        cached = self.codec_cache_dir / rel
        return cached if cached.exists() else None

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        label = self.LABEL_MAP[row["label"]]

        file_path = row["file_path"].replace("\\", "/")

        # codec 캐시: codec_cache_dir가 설정된 경우에만 여기서 codec 결정.
        # 캐시가 없으면 _apply_augmentation에서 on-the-fly 처리 (기존 동작 유지).
        codec_decided = False
        use_codec_cache = False
        if self.codec_cache_dir and self.augment_config and self.augment_config.get("codec"):
            codec_decided = True
            if self.rng.random() < self.augment_config.get("codec_prob", 0.3):
                cached = self._resolve_codec_path(file_path)
                if cached is not None:
                    wav, sr = sf.read(str(cached), dtype="float32")
                    if wav.ndim > 1:
                        wav = wav[:, 0]
                    use_codec_cache = True

        if not use_codec_cache:
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

        if self.augment_config:
            wav = self._apply_augmentation(wav, skip_codec=codec_decided)

        return torch.from_numpy(wav), label

    def _apply_augmentation(self, wav: np.ndarray, skip_codec: bool = False) -> np.ndarray:
        """설정에 따라 확률적으로 증강 적용."""
        cfg = self.augment_config

        # MUSAN noise
        if self.noise_pool and self.rng.random() < cfg.get("musan_prob", 0.5):
            from augment import add_noise_snr
            snr_lo, snr_hi = cfg.get("musan_snr_range", [10, 20])
            snr_db = self.rng.uniform(snr_lo, snr_hi)
            noise = self.noise_pool.sample(len(wav))
            wav = add_noise_snr(wav, noise, snr_db)

        # Codec compression (캐시에서 이미 로드한 경우 건너뜀)
        if not skip_codec and cfg.get("codec") and self.rng.random() < cfg.get("codec_prob", 0.3):
            from augment import apply_codec
            wav = apply_codec(
                wav,
                sr=16_000,
                codec=cfg["codec"],
                bitrate=cfg.get("codec_bitrate", "64k"),
            )

        return wav
