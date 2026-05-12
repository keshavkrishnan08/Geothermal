"""Attention-weighted multi-modal fusion."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """Learns a per-sample attention weight for each modality and projects the
    weighted concatenation to ``fused_dim``.

    With ``use_attention=False`` the module falls back to equal-weight average
    pooling (used by ablation A2).
    """

    def __init__(self, embed_dim: int = 128, n_modalities: int = 4,
                 fused_dim: int = 256, dropout: float = 0.2,
                 use_attention: bool = True):
        super().__init__()
        self.n_modalities = n_modalities
        self.use_attention = use_attention
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        self.projection = nn.Sequential(
            nn.Linear(embed_dim * n_modalities, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
        )

    def forward(self, embeddings: List[torch.Tensor],
                masks: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        B = embeddings[0].size(0)
        if self.use_attention:
            scores = []
            for z, m in zip(embeddings, masks):
                s = self.attention(z).squeeze(-1)
                s = s.masked_fill(~m, -1e9)
                scores.append(s)
            scores = torch.stack(scores, dim=1)
            # If a row has no available modalities (all masked out) we softmax
            # over -inf and get NaNs. Guard with a uniform fallback.
            all_missing = (~torch.stack(masks, dim=1)).all(dim=1, keepdim=True)
            weights = F.softmax(scores, dim=1)
            uniform = scores.new_full(scores.shape, 1.0 / self.n_modalities)
            weights = torch.where(all_missing, uniform, weights)
        else:
            weights = scores = torch.stack(
                [m.float() for m in masks], dim=1
            )
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1.0)

        weighted = [z * weights[:, i:i + 1] for i, z in enumerate(embeddings)]
        concat = torch.cat(weighted, dim=1)
        z_fused = self.projection(concat)
        return z_fused, weights
