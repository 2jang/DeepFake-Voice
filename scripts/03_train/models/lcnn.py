"""LCNN: Light CNN with Max Feature Map activation for anti-spoofing.

Computes log power spectrogram internally, then applies LCNN backbone.
Reference: Wu et al., "Light CNN for Deep Face Recognition" (adapted for audio).
~1.4M parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MFM2d(nn.Module):
    """Max Feature Map activation for Conv2d output (channel-wise max of two halves)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.size(1)
        return torch.max(x[:, : C // 2], x[:, C // 2 :])


class MFM1d(nn.Module):
    """Max Feature Map activation for Linear output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        D = x.size(-1)
        return torch.max(x[..., : D // 2], x[..., D // 2 :])


class LCNN(nn.Module):
    """LCNN anti-spoofing model (~1.4M params).

    Internally extracts log power spectrogram from raw waveform, then
    applies a light CNN backbone with MFM activations.
    """

    def __init__(
        self,
        num_classes: int = 2,
        n_fft: int = 512,
        hop_length: int = 256,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(1, 64, 5, padding=2),
            MFM2d(),  # → 32
            nn.MaxPool2d(2, 2),
            # block 2
            nn.Conv2d(32, 64, 1),
            MFM2d(),  # → 32
            nn.BatchNorm2d(32),
            # block 3
            nn.Conv2d(32, 96, 3, padding=1),
            MFM2d(),  # → 48
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(48),
            # block 4
            nn.Conv2d(48, 96, 1),
            MFM2d(),  # → 48
            nn.BatchNorm2d(48),
            # block 5
            nn.Conv2d(48, 128, 3, padding=1),
            MFM2d(),  # → 64
            nn.MaxPool2d(2, 2),
            # block 6
            nn.Conv2d(64, 128, 1),
            MFM2d(),  # → 64
            nn.BatchNorm2d(64),
            # block 7
            nn.Conv2d(64, 64, 3, padding=1),
            MFM2d(),  # → 32
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
        )

        # After 4 MaxPool2d(2,2): (B, 32, 16, 15) for 64000 samples
        # Flatten → 32*16*15 = 7680 → FC with MFM → output
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 15, 160),
            MFM1d(),  # → 80
            nn.Linear(80, num_classes),
        )

    def _spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log power spectrogram from raw waveform."""
        window = torch.hann_window(self.n_fft, device=x.device)
        spec = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            self.n_fft,
            window,
            center=False,
            return_complex=True,
        )
        return torch.log(spec.abs().pow(2) + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)  # (B, T)

        spec = self._spectrogram(x).unsqueeze(1)  # (B, 1, F, T)
        feat = self.features(spec).flatten(1)
        return self.classifier(feat)
