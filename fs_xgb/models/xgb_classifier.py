"""XGBoost classifier training utilities."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import xgboost as xgb


def _default_n_jobs() -> int:
    try:
        import os

        max_workers = os.cpu_count() or 2
        return max(1, max_workers - 1)
    except Exception:  # pragma: no cover - fallback path
        return 1


def train_xgb_classifier(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    params: Optional[Dict] = None,
    early_stopping_rounds: Optional[int] = None,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier with sensible defaults."""

    params = dict(params or {})
    params.setdefault("objective", "binary:logistic")
    params.setdefault("eval_metric", "aucpr")
    params.setdefault("tree_method", "hist")
    params.setdefault("use_label_encoder", False)
    params.setdefault("n_jobs", _default_n_jobs())
    params.setdefault("random_state", 42)
    if early_stopping_rounds is not None:
        params.setdefault("early_stopping_rounds", early_stopping_rounds)

    model = xgb.XGBClassifier(**params)
    eval_set = []
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))

    fit_kwargs = {}
    if eval_set:
        fit_kwargs["eval_set"] = eval_set
    model.fit(X_train, y_train, verbose=False, **fit_kwargs)
    return model


def predict_proba(model: xgb.XGBClassifier, X) -> np.ndarray:
    """Return positive-class probabilities for the given data."""

    probs = model.predict_proba(X)
    if probs.shape[1] == 2:
        return probs[:, 1]
    # Fallback for binary logistic when only one column returned
    return probs.ravel()
