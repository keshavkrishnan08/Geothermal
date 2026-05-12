"""Loss functions for GeoProspectNet.

- FocalLoss: original focal loss (Lin et al. 2017).
- MarginLoss: explicit positive-negative separation. Pushes positives above
  pos_margin and negatives below neg_margin; cells in the gap pay a quadratic
  penalty. Combined with BCE for gradient richness near the boundary.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017).

    Defaults are now SYMMETRIC (alpha=0.5, gamma=0 = plain BCE). The original
    asymmetric defaults (0.75, 2.0) combined with positive oversampling caused
    a model collapse where 78% of cells scored above 0.9. Use asymmetry
    deliberately and document the choice.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.gamma == 0 and self.alpha == 0.5:
            return bce.mean()  # equivalent to plain BCE
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class MarginLoss(nn.Module):
    """Push positive scores above pos_margin and negative scores below neg_margin.

    For positives:  loss = max(0, pos_margin - sigmoid(logit)) ** 2
    For negatives:  loss = max(0, sigmoid(logit) - neg_margin) ** 2

    Combined with focal/BCE this gives a strong explicit separation gradient
    even when the per-example loss is otherwise small.
    """

    def __init__(self, pos_margin: float = 0.7, neg_margin: float = 0.3,
                 weight: float = 1.0):
        super().__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        p = torch.sigmoid(logits)
        pos_loss = targets * F.relu(self.pos_margin - p) ** 2
        neg_loss = (1 - targets) * F.relu(p - self.neg_margin) ** 2
        return self.weight * (pos_loss + neg_loss).mean()


class CombinedLoss(nn.Module):
    """Sum of FocalLoss + MarginLoss."""

    def __init__(self, focal_alpha: float = 0.5, focal_gamma: float = 0.0,
                 margin_pos: float = 0.7, margin_neg: float = 0.3,
                 margin_weight: float = 1.0):
        super().__init__()
        self.focal = FocalLoss(focal_alpha, focal_gamma)
        self.margin = MarginLoss(margin_pos, margin_neg, margin_weight)

    def forward(self, logits, targets):
        return self.focal(logits, targets) + self.margin(logits, targets)
