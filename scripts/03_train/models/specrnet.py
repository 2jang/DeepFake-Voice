"""SpecRNet: Spectrogram Residual Network for audio deepfake detection.

Reference: Kawa et al., "SpecRNet: Real-Time Audio Deepfake Detection", Interspeech 2022.
Key ideas:
  - Log power spectrogram input (same as LCNN)
  - Residual blocks with FMS (Feature Map Scaling) attention
  - SELU activation for self-normalizing properties
  - Global Average Pooling → FC classifier
~0.9M parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FMS(nn.Module):
    """Feature Map Scaling — channel-wise attention via squeeze-excite."""

    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Linear(channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        scale = x.mean(dim=(2, 3))        # (B, C) — global average pool
        scale = self.sigmoid(self.fc(scale))  # (B, C)
        return x * scale.unsqueeze(-1).unsqueeze(-1)


class SpecRNetBlock(nn.Module):
    """Residual block with FMS attention.

    Conv → BN → SELU → Conv → BN → (+ skip) → SELU → FMS
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.selu = nn.SELU(inplace=True)
        self.fms = FMS(out_ch)

        self.skip = nn.Identity()
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.selu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.selu(out + identity)
        out = self.fms(out)
        return out


class SpecRNet(nn.Module):
    """SpecRNet anti-spoofing model (~0.9M params).

    Internally extracts log power spectrogram from raw waveform, then
    applies residual blocks with FMS attention.
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

        # Initial conv to lift 1-channel spectrogram
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SELU(inplace=True),
        )

        # Residual blocks with FMS + downsampling via MaxPool
        self.blocks = nn.Sequential(
            SpecRNetBlock(32, 32),
            nn.MaxPool2d(2, 2),

            SpecRNetBlock(32, 64),
            nn.MaxPool2d(2, 2),

            SpecRNetBlock(64, 64),
            nn.MaxPool2d(2, 2),

            SpecRNetBlock(64, 128),
            nn.MaxPool2d(2, 2),
        )

        # Global Average Pooling → FC
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.SELU(inplace=True),
            nn.AlphaDropout(0.1),
            nn.Linear(64, num_classes),
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
        out = self.stem(spec)
        out = self.blocks(out)
        return self.classifier(out)
