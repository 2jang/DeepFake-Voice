"""XLS-R (wav2vec2-xls-r-300m) frozen feature extractor + trainable head.

D-1: xlsr_aasist — attention-pooled head (~296K trainable params)
D-2: xlsr_fc     — mean-pooled FC head (~2K trainable params)

Input: raw waveform (B, num_samples) at 16 kHz, same as other models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

XLSR_MODEL_ID = "facebook/wav2vec2-xls-r-300m"
XLSR_DIM = 1024


class AttentionPoolHead(nn.Module):
    """Attention-weighted temporal pooling + MLP classifier (~296K params)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.ln(self.proj(x)))
        w = torch.softmax(self.attn(h), dim=1)
        pooled = (h * w).sum(dim=1)
        return self.classifier(pooled)


class FCHead(nn.Module):
    """Mean-pool + single linear layer (~2K params)."""

    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.mean(dim=1))


class _XLSRBase(nn.Module):
    """Frozen XLS-R feature extractor + trainable head.

    state_dict only includes head parameters (checkpoint ~KB, not ~1.2 GB).
    """

    def __init__(self, head: nn.Module):
        super().__init__()
        self.feature_extractor = Wav2Vec2Model.from_pretrained(XLSR_MODEL_ID)
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        with torch.no_grad():
            features = self.feature_extractor(x).last_hidden_state
        return self.head(features)

    def state_dict(self, *args, prefix="", keep_vars=False, **kwargs):
        return self.head.state_dict(
            *args, prefix=prefix + "head.", keep_vars=keep_vars, **kwargs
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        head_state = {}
        for k, v in state_dict.items():
            head_state[k[5:] if k.startswith("head.") else k] = v
        return self.head.load_state_dict(head_state, strict=strict, assign=assign)

    def train(self, mode=True):
        super().train(mode)
        self.feature_extractor.eval()
        return self


class XLSRAasist(_XLSRBase):
    """D-1: XLS-R (frozen) → attention-pooled AASIST-style head."""

    def __init__(self, num_classes: int = 2):
        super().__init__(AttentionPoolHead(XLSR_DIM, 256, num_classes))


class XLSRfc(_XLSRBase):
    """D-2: XLS-R (frozen) → mean-pool + FC."""

    def __init__(self, num_classes: int = 2):
        super().__init__(FCHead(XLSR_DIM, num_classes))
