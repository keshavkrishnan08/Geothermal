"""Every shipped module must import cleanly. Catches accidental syntax errors
or missing optional dependencies before they break the overnight pipeline."""
from __future__ import annotations

import importlib

MODULES = [
    # data
    "src.data.dataset",
    # models
    "src.models.geoprospectnet",
    "src.models.encoders",
    "src.models.fusion",
    "src.models.spatial",
    "src.models.contrastive",
    "src.models.classifier",
    "src.models.defaults",
    "src.models.baselines",
    # training
    "src.training.train",
    "src.training.train_baselines",
    "src.training.losses",
    "src.training.utils",
    "src.training.multi_seed",
    "src.training.neg_pool_sweep",
    "src.training.tune_separation",
    "src.training.single_modality_ablation",
    "src.training.east_coast_finetune",
    # evaluation
    "src.evaluation.discovery",
    "src.evaluation.ablation",
    "src.evaluation.baselines",
    "src.evaluation.calibration",
    "src.evaluation.cross_basin",
    "src.evaluation.cross_validation",
    "src.evaluation.economic_analysis",
    "src.evaluation.embedding_viz",
    "src.evaluation.field_validation",
    "src.evaluation.hard_negatives",
    "src.evaluation.lofcv_buffer_audit",
    "src.evaluation.metrics",
    "src.evaluation.modality_analysis",
    "src.evaluation.mordensky_comparison",
    "src.evaluation.mwe_estimation_v2",
    "src.evaluation.negative_controls",
    "src.evaluation.ood_eastern_us",
    "src.evaluation.preregister",
    "src.evaluation.reservoir_thickness",
    "src.evaluation.spatial_analysis",
    "src.evaluation.statistical_tests",
    "src.evaluation.temporal_holdout",
    # visualization
    "src.visualization.make_figures",
]


def test_all_modules_import():
    failures = []
    for m in MODULES:
        try:
            importlib.import_module(m)
        except Exception as e:
            failures.append((m, repr(e)))
    assert not failures, "\n".join(f"  {m}: {err}" for m, err in failures)
