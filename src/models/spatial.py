"""Spatial neighbor averaging for geological continuity."""
from __future__ import annotations

import torch
import torch.nn as nn


class SpatialSmoothing(nn.Module):
    """Convex mix of each cell's fused embedding with the average of its k
    spatial neighbours: z' = (1 - alpha) * z + alpha * mean(neighbors).

    The module accepts an OPTIONAL neighbor-embedding tensor of shape
    ``[B, k, D]``. Caller is responsible for gathering neighbor embeddings
    using pre-computed indices (see data/build_neighbors.py).
    """

    def __init__(self, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha

    def forward(self, z_fused: torch.Tensor,
                neighbor_embeddings: torch.Tensor | None = None) -> torch.Tensor:
        if self.alpha == 0 or neighbor_embeddings is None:
            return z_fused
        neighbor_mean = neighbor_embeddings.mean(dim=1)
        return (1 - self.alpha) * z_fused + self.alpha * neighbor_mean
