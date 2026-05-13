"""Determinism checks: seeded forward passes must be bit-identical, and
MC-Dropout must reproduce after my seeding fix."""
from __future__ import annotations

import numpy as np
import torch

from src.evaluation.discovery import mc_dropout_inference
from src.models.geoprospectnet import GeoProspectNet


def _make_dummy_loader(cfg, n_cells: int = 32, batch_size: int = 16):
    """Tiny in-memory loader for fast unit-style tests."""
    geo = torch.randn(n_cells, cfg["modalities"]["n_geophys_channels"], 32, 32)
    chem = torch.randn(n_cells, cfg["modalities"]["n_geochem_features"])
    geol = torch.randn(n_cells, cfg["modalities"]["n_geostruct_features"])
    therm = torch.zeros(n_cells, 1, 64, 64)
    masks_geo = torch.ones(n_cells, dtype=torch.bool)
    masks_chem = torch.ones(n_cells, dtype=torch.bool)
    masks_therm = torch.zeros(n_cells, dtype=torch.bool)
    masks_struct = torch.ones(n_cells, dtype=torch.bool)
    cells = torch.arange(n_cells, dtype=torch.long)
    labels = torch.zeros(n_cells)

    batches = []
    for i in range(0, n_cells, batch_size):
        j = min(i + batch_size, n_cells)
        batches.append({
            "geophysics": geo[i:j], "geo_mask": masks_geo[i:j],
            "geochemistry": chem[i:j], "chem_mask": masks_chem[i:j],
            "thermal": therm[i:j], "therm_mask": masks_therm[i:j],
            "geology": geol[i:j], "struct_mask": masks_struct[i:j],
            "cell_id": cells[i:j], "label": labels[i:j],
        })
    return batches


def test_mc_dropout_is_seeded(cfg_max, cpu_max_model):
    """Two MC-Dropout runs with the same seed must match exactly. This is the
    contract that broke our Kaggle vs local determinism (29 vs 33 discoveries)."""
    loader = _make_dummy_loader(cfg_max, n_cells=32, batch_size=16)
    mu1, std1 = mc_dropout_inference(cpu_max_model, loader, device="cpu",
                                      n_passes=4, seed=123)
    mu2, std2 = mc_dropout_inference(cpu_max_model, loader, device="cpu",
                                      n_passes=4, seed=123)
    np.testing.assert_array_equal(mu1, mu2)
    np.testing.assert_array_equal(std1, std2)


def test_mc_dropout_is_stochastic(cfg_max, cpu_max_model):
    """Different seeds should give different (non-trivially) draws."""
    loader = _make_dummy_loader(cfg_max, n_cells=32, batch_size=16)
    mu1, _ = mc_dropout_inference(cpu_max_model, loader, device="cpu",
                                   n_passes=4, seed=1)
    mu2, _ = mc_dropout_inference(cpu_max_model, loader, device="cpu",
                                   n_passes=4, seed=2)
    assert np.any(np.abs(mu1 - mu2) > 1e-6), \
        "different MC-Dropout seeds produced identical means"


def test_eval_forward_is_deterministic(cfg_max, cpu_max_model):
    """In .eval() mode the model must produce bit-exact outputs across calls."""
    loader = _make_dummy_loader(cfg_max, n_cells=8, batch_size=8)
    cpu_max_model.eval()
    with torch.no_grad():
        a = cpu_max_model(loader[0])["logits"].numpy()
        b = cpu_max_model(loader[0])["logits"].numpy()
    np.testing.assert_array_equal(a, b)


def test_seeding_reproduces_initial_weights(cfg_max):
    """A fresh model with seed_everything(N) must match weight-by-weight
    another fresh model with seed_everything(N)."""
    from src.training.utils import seed_everything
    seed_everything(99)
    m1 = GeoProspectNet(cfg_max)
    seed_everything(99)
    m2 = GeoProspectNet(cfg_max)
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        np.testing.assert_array_equal(p1.detach().numpy(), p2.detach().numpy(),
                                       err_msg=f"weights differ for {n1}")
