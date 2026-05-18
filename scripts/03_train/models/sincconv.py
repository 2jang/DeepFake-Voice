"""Sinc-based convolution for raw audio processing.

Shared by AASIST and RawNet2. Based on SincNet (Ravanelli & Bengio, 2018).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv(nn.Module):
    """Learnable sinc-based bandpass filters applied to raw waveform."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 129,
        in_channels: int = 1,
        sample_rate: int = 16_000,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        mel = np.linspace(
            self._hz2mel(min_low_hz), self._hz2mel(high_hz), out_channels + 1
        )
        hz = self._mel2hz(mel)

        self.low_hz_ = nn.Parameter(
            torch.tensor(hz[:-1], dtype=torch.float32).view(-1, 1)
        )
        self.band_hz_ = nn.Parameter(
            torch.tensor(np.diff(hz), dtype=torch.float32).view(-1, 1)
        )

        n_half = self.kernel_size // 2
        n = 2 * math.pi * torch.arange(-n_half, 0, dtype=torch.float32) / sample_rate
        self.register_buffer("n_", n.view(1, -1))

        n_lin = torch.linspace(0, n_half - 1, steps=n_half)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size)
        self.register_buffer("window_", window.view(1, -1))

    @staticmethod
    def _hz2mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def _mel2hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            min=self.min_low_hz,
            max=self.sample_rate / 2,
        )

        f_low = torch.matmul(low, self.n_)
        f_high = torch.matmul(high, self.n_)

        band_pass_left = (
            (torch.sin(f_high) - torch.sin(f_low)) / (self.n_ / 2)
        ) * self.window_

        band_pass_center = 2 * (high - low)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1
        )
        band = (high - low).squeeze(1)
        band_pass = band_pass / (2 * band.unsqueeze(1))

        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=1, padding=0, groups=1)
