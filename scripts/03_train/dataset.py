"""Audio dataset for anti-spoofing training and evaluation."""

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
    ):
        self.df = pd.read_csv(manifest_path)
        self.num_samples = num_samples
        self.is_train = is_train
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        label = self.LABEL_MAP[row["label"]]

        wav, sr = sf.read(row["file_path"], dtype="float32")
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

        return torch.from_numpy(wav), label
