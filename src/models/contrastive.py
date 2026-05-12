"""Multi-modal contrastive (InfoNCE) alignment loss.

For each pair (i, j) of modalities, we compute symmetric CLIP-style InfoNCE
on the subset of the batch where BOTH modalities are present. Gating on the
masks is critical: if we let learned defaults participate, the model trivially
aligns "no information" to itself across modalities.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

    def forward(self, embeddings: List[torch.Tensor],
                masks: List[torch.Tensor]) -> torch.Tensor:
        # Wider clamp than CLIP-tight 1.0 — the contrastive head may need to
        # soften when modality alignment is genuinely hard at a location.
        temp = torch.exp(self.log_temperature).clamp(min=0.01, max=10.0)
        device = embeddings[0].device
        total = embeddings[0].new_zeros(())
        n_pairs = 0

        n = len(embeddings)
        for i in range(n):
            for j in range(i + 1, n):
                valid = masks[i] & masks[j]
                if valid.sum().item() < 2:
                    continue
                z_i = embeddings[i][valid]
                z_j = embeddings[j][valid]
                logits = z_i @ z_j.T / temp
                labels = torch.arange(z_i.size(0), device=device)
                loss_ij = F.cross_entropy(logits, labels)
                loss_ji = F.cross_entropy(logits.T, labels)
                total = total + (loss_ij + loss_ji) / 2
                n_pairs += 1

        if n_pairs == 0:
            return embeddings[0].new_zeros((), requires_grad=True)
        return total / n_pairs
