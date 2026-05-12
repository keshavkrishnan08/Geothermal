"""Binary classification head — returns raw logits."""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, fused_dim: int = 256, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z).squeeze(-1)
