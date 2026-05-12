"""Full GeoProspectNet model — 4 encoders + contrastive + fusion + spatial + classifier."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.models.classifier import ClassificationHead
from src.models.contrastive import MultiModalContrastiveLoss
from src.models.defaults import LearnedDefault
from src.models.encoders import (
    ChemEncoder, GeoEncoder, GeoStructEncoder, ThermEncoder,
)
from src.models.fusion import AttentionFusion
from src.models.spatial import SpatialSmoothing


# Canonical modality order — used everywhere so attention weights align across
# the codebase (figures, ablation, paper).
MODALITY_ORDER = ("geophysics", "geochemistry", "thermal", "geology")


class GeoProspectNet(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        m_cfg = config["model"]
        mod_cfg = config["modalities"]
        embed_dim = m_cfg["embed_dim"]
        fused_dim = m_cfg["fused_dim"]
        use_thermal = mod_cfg["use_thermal"]
        dropped = mod_cfg.get("drop", None)
        # Normalise to a set so call-sites can pass a string (one modality)
        # or a list (single-modality baselines that drop everything else).
        if dropped is None:
            dropped_set: set = set()
        elif isinstance(dropped, str):
            dropped_set = {dropped}
        else:
            dropped_set = set(dropped)
        use_attention = m_cfg.get("use_attention", True)
        encoder_dropout = m_cfg.get("encoder_dropout", 0.1)
        cls_dropout = m_cfg.get("classifier_dropout", 0.2)

        self.use_thermal = use_thermal
        self.dropped = dropped_set

        self.geo_encoder = GeoEncoder(
            in_channels=mod_cfg["n_geophys_channels"], embed_dim=embed_dim,
            dropout=encoder_dropout,
        )
        self.chem_encoder = ChemEncoder(
            in_features=mod_cfg["n_geochem_features"], embed_dim=embed_dim,
            dropout=encoder_dropout,
        )
        if use_thermal:
            self.therm_encoder = ThermEncoder(
                in_channels=1, embed_dim=embed_dim, dropout=encoder_dropout,
            )
        self.struct_encoder = GeoStructEncoder(
            in_features=mod_cfg["n_geostruct_features"], embed_dim=embed_dim,
            dropout=encoder_dropout,
        )

        self.default_geo = LearnedDefault(embed_dim)
        self.default_chem = LearnedDefault(embed_dim)
        if use_thermal:
            self.default_therm = LearnedDefault(embed_dim)
        self.default_struct = LearnedDefault(embed_dim)

        n_modalities = 4 if use_thermal else 3
        self.contrastive_loss = MultiModalContrastiveLoss(
            temperature=m_cfg["contrastive_temperature"]
        )
        self.fusion = AttentionFusion(
            embed_dim=embed_dim, n_modalities=n_modalities,
            fused_dim=fused_dim, dropout=cls_dropout,
            use_attention=use_attention,
        )
        self.spatial = SpatialSmoothing(alpha=m_cfg["spatial_alpha"])
        self.classifier = ClassificationHead(fused_dim=fused_dim, dropout=cls_dropout)

    @staticmethod
    def _mix_default(z_real: torch.Tensor, mask: torch.Tensor,
                     default: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(1).float()
        return m * z_real + (1.0 - m) * default

    def encode(self, batch: Dict[str, torch.Tensor]):
        B = batch["geophysics"].size(0)
        device = batch["geophysics"].device

        embeddings: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []

        # Geophysics
        if "geophysics" in self.dropped:
            embeddings.append(self.default_geo(B))
            masks.append(torch.zeros(B, dtype=torch.bool, device=device))
        else:
            z = self.geo_encoder(batch["geophysics"])
            d = self.default_geo(B)
            embeddings.append(self._mix_default(z, batch["geo_mask"], d))
            masks.append(batch["geo_mask"])

        # Geochemistry
        if "geochemistry" in self.dropped:
            embeddings.append(self.default_chem(B))
            masks.append(torch.zeros(B, dtype=torch.bool, device=device))
        else:
            z = self.chem_encoder(batch["geochemistry"])
            d = self.default_chem(B)
            embeddings.append(self._mix_default(z, batch["chem_mask"], d))
            masks.append(batch["chem_mask"])

        # Thermal (only if model was constructed with thermal AND it isn't dropped)
        if self.use_thermal:
            if "thermal" in self.dropped:
                embeddings.append(self.default_therm(B))
                masks.append(torch.zeros(B, dtype=torch.bool, device=device))
            else:
                z = self.therm_encoder(batch["thermal"])
                d = self.default_therm(B)
                embeddings.append(self._mix_default(z, batch["therm_mask"], d))
                masks.append(batch["therm_mask"])

        # Geological structure
        if "geology" in self.dropped:
            embeddings.append(self.default_struct(B))
            masks.append(torch.zeros(B, dtype=torch.bool, device=device))
        else:
            z = self.struct_encoder(batch["geology"])
            d = self.default_struct(B)
            embeddings.append(self._mix_default(z, batch["struct_mask"], d))
            masks.append(batch["struct_mask"])

        return embeddings, masks

    def forward(self, batch: Dict[str, torch.Tensor],
                neighbor_embeddings: Optional[torch.Tensor] = None,
                return_extras: bool = False) -> Dict[str, torch.Tensor]:
        embeddings, masks = self.encode(batch)
        l_contrastive = self.contrastive_loss(embeddings, masks)
        z_fused, attn = self.fusion(embeddings, masks)
        z_spatial = self.spatial(z_fused, neighbor_embeddings)
        logits = self.classifier(z_spatial)

        out = {
            "logits": logits,
            "l_contrastive": l_contrastive,
            "attention_weights": attn,
            "z_fused": z_fused,
        }
        if return_extras:
            out["embeddings"] = embeddings
        return out
