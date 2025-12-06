"""Classification metrics helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute PR-AUC (average precision)."""

    return float(average_precision_score(y_true, y_score))


def roc_auc_score_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC, returning NaN when only one class is present."""

    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Return a dictionary with PR-AUC and ROC-AUC."""

    return {
        "pr_auc": pr_auc_score(y_true, y_score),
        "roc_auc": roc_auc_score_safe(y_true, y_score),
    }
