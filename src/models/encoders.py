"""Modality encoders: each produces an L2-normalized 128-d embedding."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoEncoder(nn.Module):
    """32x32x6 geophysical patch -> 128-d L2-normalized embedding."""

    def __init__(self, in_channels: int = 6, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z = self.projection(h)
        return F.normalize(z, dim=-1)


class ChemEncoder(nn.Module):
    """9-d geochemistry vector -> 128-d L2-normalized embedding.

    Uses LayerNorm (not BatchNorm1d) so single-sample eval is safe.
    """

    def __init__(self, in_features: int = 9, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.LayerNorm(128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(x), dim=-1)


class ThermEncoder(nn.Module):
    """64x64x1 thermal-anomaly patch -> 128-d L2-normalized embedding."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.encoder(x)), dim=-1)


class GeoStructEncoder(nn.Module):
    """11-d geological-structure vector -> 128-d L2-normalized embedding.

    Uses LayerNorm (not BatchNorm1d) so single-sample eval is safe.
    """

    def __init__(self, in_features: int = 11, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.LayerNorm(128), nn.ReLU(inplace=True),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(x), dim=-1)
