"""Learned default embeddings for missing modalities."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedDefault(nn.Module):
    """A trainable embedding substituted when a modality is absent.

    Using zeros would corrupt both the contrastive loss (defaults would align
    to defaults — meaningless) and the attention fusion. A learned default
    lets the model encode "no information" explicitly.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.default = nn.Parameter(torch.randn(embed_dim) * 0.01)

    def forward(self, batch_size: int = 1) -> torch.Tensor:
        return F.normalize(
            self.default.unsqueeze(0).expand(batch_size, -1), dim=-1
        )
