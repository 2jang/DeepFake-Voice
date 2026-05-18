"""RawNet2: End-to-end anti-spoofing from raw waveform.

Reference: Tak et al., "End-to-End Spectro-Temporal Graph Attention Networks
for Speaker Verification Anti-Spoofing", 2021. ~17M parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sincconv import SincConv


class FMS(nn.Module):
    """Feature Map Scaling."""

    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(self.fc(x.mean(dim=-1)))
        return x * s.unsqueeze(-1) + x


class ResBlock1d(nn.Module):
    """1-D residual block with FMS and max-pooling."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.mp = nn.MaxPool1d(3)
        self.fms = FMS(out_ch)

        self.downsample = None
        if in_ch != out_ch:
            self.downsample = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.selu(self.bn1(x))
        out = F.selu(self.bn2(self.conv1(out)))
        out = self.conv2(out)

        identity = self.downsample(x) if self.downsample else x
        out = self.mp(out + identity)
        return self.fms(out)


class RawNet2(nn.Module):
    """RawNet2 anti-spoofing model (~17M params)."""

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.sinc_conv = SincConv(out_channels=128, kernel_size=129)
        self.sinc_bn = nn.BatchNorm1d(128)

        self.group1 = nn.Sequential(
            ResBlock1d(128, 128),
            ResBlock1d(128, 128),
        )
        self.group2 = nn.Sequential(
            ResBlock1d(128, 256),
            ResBlock1d(256, 256),
            ResBlock1d(256, 256),
            ResBlock1d(256, 256),
        )

        self.bn_before_gru = nn.BatchNorm1d(256)
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=1024,
            num_layers=3,
            batch_first=True,
        )
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, T)

        x = F.selu(self.sinc_bn(self.sinc_conv(x)))
        x = F.max_pool1d(torch.abs(x), 3)

        x = self.group1(x)
        x = self.group2(x)

        x = F.leaky_relu(self.bn_before_gru(x), 0.3)
        x = x.permute(0, 2, 1)  # (B, T, C)
        self.gru.flatten_parameters()
        _, h_n = self.gru(x)
        out = h_n[-1]  # last layer hidden state: (B, 1024)

        return self.classifier(out)
