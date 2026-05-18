"""AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks.

Reference: Jung et al., ICASSP 2022. ~297K parameters.
Input: raw waveform (B, 64000).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sincconv import SincConv


class Residual_block_2d(nn.Module):
    """2-D residual block with Feature Map Scaling (FMS)."""

    def __init__(self, nb_filts: list[int], first: bool = False):
        super().__init__()
        self.first = first

        if not first:
            self.bn1 = nn.BatchNorm2d(nb_filts[0])

        self.conv1 = nn.Conv2d(nb_filts[0], nb_filts[1], (2, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(nb_filts[1])
        self.conv2 = nn.Conv2d(nb_filts[1], nb_filts[1], (2, 3), padding=(0, 1))

        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_ds = nn.Conv2d(nb_filts[0], nb_filts[1], 1)

        self.mp = nn.MaxPool2d((1, 3))
        self.fc_fms = nn.Linear(nb_filts[1], nb_filts[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x) if not self.first else x
        out = F.selu(out)
        out = F.selu(self.bn2(self.conv1(out)))
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_ds(identity)

        out = self.mp(out + identity)

        # FMS
        t = torch.sigmoid(self.fc_fms(out.mean(dim=(2, 3))))
        return out * t.unsqueeze(2).unsqueeze(3) + out


class GraphAttentionLayer(nn.Module):
    """GAT: all nodes attend to a single master node."""

    def __init__(self, node_dim: int, out_dim: int, temperature: float = 1.0):
        super().__init__()
        self.fc = nn.Linear(node_dim, out_dim)
        self.attn = nn.Linear(2 * out_dim, 1)
        self.temperature = temperature

    def forward(
        self, h: torch.Tensor, master: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.fc(h)
        z_m = self.fc(master).expand_as(z)
        e = F.leaky_relu(self.attn(torch.cat([z, z_m], dim=-1)))
        alpha = F.softmax(e / self.temperature, dim=1)
        return (alpha * z).sum(dim=1, keepdim=True), alpha


class HtrgGraphAttentionLayer(nn.Module):
    """Heterogeneous cross-domain graph attention."""

    def __init__(
        self,
        node_dim: int,
        master_dim: int,
        out_dim: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.fc_node = nn.Linear(node_dim, out_dim)
        self.fc_master = nn.Linear(master_dim, out_dim)
        self.attn = nn.Linear(2 * out_dim, 1)
        self.temperature = temperature

    def forward(
        self, h_from: torch.Tensor, master_to: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_n = self.fc_node(h_from)
        z_m = self.fc_master(master_to).expand_as(z_n)
        e = F.leaky_relu(self.attn(torch.cat([z_n, z_m], dim=-1)))
        alpha = F.softmax(e / self.temperature, dim=1)
        return (alpha * z_n).sum(dim=1, keepdim=True), alpha


class GraphPool(nn.Module):
    """Top-k graph pooling with sigmoid gating."""

    def __init__(self, k_ratio: float, in_dim: int):
        super().__init__()
        self.k_ratio = k_ratio
        self.proj = nn.Linear(in_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scores = self.proj(h).squeeze(-1)
        k = max(1, int(h.size(1) * self.k_ratio))
        top_scores, top_idx = torch.topk(scores, k, dim=1)
        gate = torch.sigmoid(top_scores).unsqueeze(-1)
        idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, h.size(-1))
        return torch.gather(h, 1, idx_exp) * gate


class AASIST(nn.Module):
    """AASIST anti-spoofing model (~297K params)."""

    def __init__(self, num_classes: int = 2):
        super().__init__()

        sinc_ch = 70
        filts = [[1, 32], [32, 32], [32, 64], [64, 64]]
        enc_dim = filts[-1][-1]  # 64
        gat_dims = [128, 64]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temps = [2.0, 2.0, 100.0, 100.0]

        # --- encoder ---
        self.sinc_conv = SincConv(out_channels=sinc_ch, kernel_size=129)
        self.first_bn = nn.BatchNorm2d(1)
        self.block0 = Residual_block_2d(filts[0], first=True)
        self.block1 = Residual_block_2d(filts[1])
        self.block2 = Residual_block_2d(filts[2])
        self.block3 = Residual_block_2d(filts[3])
        self.bn_enc = nn.BatchNorm2d(enc_dim)

        # attention gate on encoder output
        self.attention = nn.Sequential(
            nn.Conv2d(enc_dim, 128, 1),
            nn.SELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, enc_dim, 1),
        )

        # --- graph attention ---
        g0, g1 = gat_dims

        self.master_S = nn.Parameter(torch.randn(1, 1, enc_dim))
        self.master_T = nn.Parameter(torch.randn(1, 1, enc_dim))

        self.GAT_S = GraphAttentionLayer(enc_dim, g0, temps[0])
        self.GAT_T = GraphAttentionLayer(enc_dim, g0, temps[0])

        self.pool_S = GraphPool(pool_ratios[0], enc_dim)
        self.pool_T = GraphPool(pool_ratios[0], enc_dim)

        # cross-domain attention (2 rounds x 2 steps)
        self.HtrgGAT_ST_11 = HtrgGraphAttentionLayer(enc_dim, g0, g1, temps[1])
        self.HtrgGAT_TS_11 = HtrgGraphAttentionLayer(enc_dim, g0, g1, temps[1])
        self.pool_hS1 = GraphPool(pool_ratios[1], enc_dim)
        self.pool_hT1 = GraphPool(pool_ratios[1], enc_dim)

        self.HtrgGAT_ST_12 = HtrgGraphAttentionLayer(enc_dim, g1, g1, temps[2])
        self.HtrgGAT_TS_12 = HtrgGraphAttentionLayer(enc_dim, g1, g1, temps[2])
        self.pool_hS2 = GraphPool(pool_ratios[2], enc_dim)
        self.pool_hT2 = GraphPool(pool_ratios[2], enc_dim)

        self.HtrgGAT_ST_21 = HtrgGraphAttentionLayer(enc_dim, g1, g1, temps[2])
        self.HtrgGAT_TS_21 = HtrgGraphAttentionLayer(enc_dim, g1, g1, temps[2])

        self.HtrgGAT_ST_22 = HtrgGraphAttentionLayer(enc_dim, g1, g1, temps[3])
        self.HtrgGAT_TS_22 = HtrgGraphAttentionLayer(enc_dim, g1, g1, temps[3])

        # --- classifier ---
        self.drop = nn.Dropout(0.5)
        self.drop_way = nn.Dropout(0.2)
        self.out_layer = nn.Linear(g1 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, T)

        # encoder — pool(9) to keep temporal dim manageable for 2D blocks
        x = F.max_pool1d(torch.abs(self.sinc_conv(x)), 9)
        x = F.selu(self.first_bn(x.unsqueeze(1)))
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.bn_enc(self.block3(x))

        # attention weighting
        w = torch.sigmoid(self.attention(x))
        x = x * w

        B, C, S, T = x.shape

        # spectro-temporal decomposition
        h_S = x.mean(dim=-1).permute(0, 2, 1)  # (B, S, C)
        h_T = x.mean(dim=-2).permute(0, 2, 1)  # (B, T, C)

        # initial GAT
        m_S = self.master_S.expand(B, -1, -1)
        m_T = self.master_T.expand(B, -1, -1)

        m_S, _ = self.GAT_S(h_S, m_S)
        m_T, _ = self.GAT_T(h_T, m_T)

        h_S = self.pool_S(h_S)
        h_T = self.pool_T(h_T)

        # cross-domain round 1
        m_S, _ = self.HtrgGAT_ST_11(h_T, m_S)
        m_T, _ = self.HtrgGAT_TS_11(h_S, m_T)
        h_S = self.pool_hS1(h_S)
        h_T = self.pool_hT1(h_T)

        m_S, _ = self.HtrgGAT_ST_12(h_T, m_S)
        m_T, _ = self.HtrgGAT_TS_12(h_S, m_T)
        h_S = self.pool_hS2(h_S)
        h_T = self.pool_hT2(h_T)

        # cross-domain round 2
        m_S, _ = self.HtrgGAT_ST_21(self.drop_way(h_T), m_S)
        m_T, _ = self.HtrgGAT_TS_21(self.drop_way(h_S), m_T)

        m_S, _ = self.HtrgGAT_ST_22(self.drop_way(h_T), m_S)
        m_T, _ = self.HtrgGAT_TS_22(self.drop_way(h_S), m_T)

        out = self.drop(torch.cat([m_S.squeeze(1), m_T.squeeze(1)], dim=-1))
        return self.out_layer(out)
