"""Evaluation metric utilities — thin wrappers around scikit-learn for code reuse."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve, roc_auc_score,
)


def capture_rate_at_top_pct(y_true: np.ndarray, y_score: np.ndarray,
                            pct: float = 0.10) -> float:
    """Mordensky et al. (2025) headline metric: of all positive sites, what
    fraction are ranked in the top `pct` of predicted favourability?

    Mordensky 2025 reports 85% capture at top-10% for the Great Basin. This is
    the metric we report directly against that benchmark.
    """
    n = len(y_score)
    n_top = max(1, int(np.ceil(pct * n)))
    threshold_idx = np.argpartition(y_score, -n_top)[-n_top:]
    top_mask = np.zeros(n, dtype=bool)
    top_mask[threshold_idx] = True
    n_pos = max(1, int(y_true.sum()))
    return float(y_true[top_mask].sum() / n_pos)


def fold_metrics(y_true: np.ndarray, y_score: np.ndarray, k: int = 50) -> Dict[str, float]:
    out = {
        "auroc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y_true, y_score)) if y_true.sum() else float("nan"),
    }
    # F1 at optimal threshold
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    if len(f1) > 0:
        out["f1"] = float(f1.max())
        out["best_threshold"] = float(thr[np.argmax(f1)]) if len(thr) else 0.5
    else:
        out["f1"] = float("nan")
        out["best_threshold"] = 0.5

    k = min(k, len(y_score))
    top = np.argsort(y_score)[-k:]
    out["precision_at_k"] = float(y_true[top].mean())
    out["recall_at_k"] = float(y_true[top].sum() / max(1, y_true.sum()))

    # Mordensky 2025 head-to-head metrics
    out["capture_top_1pct"] = capture_rate_at_top_pct(y_true, y_score, 0.01)
    out["capture_top_5pct"] = capture_rate_at_top_pct(y_true, y_score, 0.05)
    out["capture_top_10pct"] = capture_rate_at_top_pct(y_true, y_score, 0.10)
    out["capture_top_25pct"] = capture_rate_at_top_pct(y_true, y_score, 0.25)
    return out


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray,
                 metric_fn=roc_auc_score, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_score[idx]))
    scores = np.array(scores)
    return {
        "mean": float(scores.mean()),
        "ci_low": float(np.quantile(scores, alpha / 2)),
        "ci_high": float(np.quantile(scores, 1 - alpha / 2)),
    }
