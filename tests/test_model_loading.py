"""Every checkpoint we ship must load and produce finite scores."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.models.geoprospectnet import GeoProspectNet


def _ckpt_config_pairs(root: Path):
    """Map checkpoint files to the configs they were trained with."""
    return [
        ("random_cpu_max_seed42.pt",            "configs/cpu_max.yaml"),
        ("random_cpu_margin_seed42.pt",          "configs/cpu_margin.yaml"),
        ("random_cpu_calibrated_seed42.pt",      "configs/cpu_calibrated.yaml"),
        ("random_cpu_margin_no_contrastive_seed42.pt",
                                                 "configs/cpu_margin_no_contrastive.yaml"),
        ("random_cpu_margin_no_attention_seed42.pt",
                                                 "configs/cpu_margin_no_attention.yaml"),
        ("random_cpu_margin_no_spatial_seed42.pt",
                                                 "configs/cpu_margin_no_spatial.yaml"),
    ]


def _make_dummy_batch(cfg, batch_size: int = 4):
    geo = torch.randn(batch_size, cfg["modalities"]["n_geophys_channels"], 32, 32)
    chem = torch.randn(batch_size, cfg["modalities"]["n_geochem_features"])
    geol = torch.randn(batch_size, cfg["modalities"]["n_geostruct_features"])
    therm = torch.randn(batch_size, 1, 64, 64)
    return {
        "geophysics": geo,
        "geo_mask": torch.ones(batch_size, dtype=torch.bool),
        "geochemistry": chem,
        "chem_mask": torch.ones(batch_size, dtype=torch.bool),
        "thermal": therm,
        "therm_mask": torch.zeros(batch_size, dtype=torch.bool),
        "geology": geol,
        "struct_mask": torch.ones(batch_size, dtype=torch.bool),
        "cell_id": torch.arange(batch_size, dtype=torch.long),
        "label": torch.zeros(batch_size),
    }


@pytest.mark.parametrize("ckpt_name,cfg_path", _ckpt_config_pairs(Path(".")))
def test_checkpoint_loads_and_scores(root, ckpt_name, cfg_path):
    cp = root / "outputs/checkpoints" / ckpt_name
    cf = root / cfg_path
    if not cp.exists():
        pytest.skip(f"missing {cp.name}")
    cfg = yaml.safe_load(open(cf))
    state = torch.load(cp, map_location="cpu", weights_only=False)["state_dict"]
    model = GeoProspectNet(cfg).to("cpu")
    model.load_state_dict(state)
    model.eval()
    batch = _make_dummy_batch(cfg)
    with torch.no_grad():
        out = model(batch)
    scores = torch.sigmoid(out["logits"]).numpy()
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all() and (scores <= 1).all()


def test_cpu_max_real_data_inference(root, cfg_max, cpu_max_model, processed):
    """Run cpu_max on 32 real labeled cells and confirm finite scores."""
    from src.data.dataset import GeoProspectDataset, make_loader
    train_mask = np.load(processed / "train_mask.npy").astype(bool)
    idx = np.where(train_mask)[0][:32]
    ds = GeoProspectDataset(processed, indices=idx,
                            use_thermal=cfg_max["modalities"]["use_thermal"])
    loader = make_loader(ds, batch_size=32, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    with torch.no_grad():
        out = cpu_max_model(batch)
    s = torch.sigmoid(out["logits"]).numpy()
    assert s.shape == (32,)
    assert np.isfinite(s).all()
