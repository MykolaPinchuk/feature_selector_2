"""Implementation of the permutation- and SHAP-based feature selection workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fs_xgb.metrics import pr_auc_score
from fs_xgb.models import predict_proba, train_xgb_classifier

try:
    import shap
except ImportError as exc:  # pragma: no cover - shap is part of project deps
    raise RuntimeError("shap must be installed to run feature selection.") from exc


@dataclass
class FeatureSelectionResult:
    """Outputs from the feature-selection routine."""

    kept_features: List[str]
    dropped_features: List[str]
    permutation_table: pd.DataFrame
    shap_importance: pd.Series


def _make_inner_split(X: pd.DataFrame, y: pd.Series, holdout_frac: float, random_state: int) -> Tuple:
    return train_test_split(
        X,
        y,
        test_size=holdout_frac,
        stratify=y,
        random_state=random_state,
    )


def _create_fs_eval(X: pd.DataFrame, y: pd.Series, neg_pos_ratio: float, random_state: int) -> Tuple[pd.DataFrame, pd.Series]:
    positives = X[y == 1]
    negatives = X[y == 0]
    rng = np.random.default_rng(random_state)
    target_negatives = int(len(positives) * neg_pos_ratio)
    sampled_negatives = negatives.sample(
        n=min(target_negatives, len(negatives)),
        replace=False,
        random_state=random_state,
    )
    X_eval = pd.concat([positives, sampled_negatives], axis=0).reset_index(drop=True)
    y_eval = pd.Series([1] * len(positives) + [0] * len(sampled_negatives))
    return X_eval, y_eval


def _compute_shap_importance(model, X_eval: pd.DataFrame) -> pd.Series:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_eval)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs, index=X_eval.columns)


def _aggregate_shap(shap_list: List[pd.Series]) -> pd.Series:
    stacked = pd.concat(shap_list, axis=1)
    return stacked.mean(axis=1).sort_values(ascending=False)


def _permutation_table(
    models,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    features: List[str],
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    for feature in features:
        deltas = []
        for model in models:
            baseline_pred = predict_proba(model, X_eval)
            baseline_metric = pr_auc_score(y_eval, baseline_pred)
            X_perm = X_eval.copy()
            shuffled = X_perm[feature].values.copy()
            rng.shuffle(shuffled)
            X_perm[feature] = shuffled
            perm_pred = predict_proba(model, X_perm)
            perm_metric = pr_auc_score(y_eval, perm_pred)
            deltas.append(baseline_metric - perm_metric)
        rows.append(
            {
                "feature": feature,
                "delta_mean": float(np.mean(deltas)),
                "delta_std": float(np.std(deltas)),
            }
        )
    return pd.DataFrame(rows).sort_values("delta_mean", ascending=False).reset_index(drop=True)


def _apply_selection_rules(fi_table: pd.DataFrame, shap_importance: pd.Series, config: Dict) -> Tuple[List[str], List[str]]:
    thresholds = config.get("thresholds", {})
    delta_abs_min = thresholds.get("delta_abs_min", 0.0)
    k_noise_std = thresholds.get("k_noise_std", 2.0)
    noise_std = fi_table["delta_mean"].std()
    if np.isnan(noise_std) or noise_std == 0:
        noise_std = 1e-6
    dynamic_threshold = k_noise_std * noise_std
    threshold = max(delta_abs_min, dynamic_threshold)
    keep = fi_table[fi_table["delta_mean"] >= threshold]["feature"].tolist()
    drop = [feat for feat in fi_table["feature"] if feat not in keep]

    rest_policy = config.get("rest_policy", "keep_all")
    shap_rank = shap_importance.rank(ascending=False, method="min")
    rest_features = [feat for feat in shap_importance.index if feat not in fi_table["feature"]]

    if rest_policy == "keep_all":
        keep.extend(rest_features)
    elif rest_policy == "keep_above_min_shap":
        min_rank = config.get("rest_min_shap_rank", len(shap_rank))
        keep.extend([feat for feat in rest_features if shap_rank[feat] <= min_rank])
    # rest_policy == "drop_all" defaults to dropping the rest

    keep = sorted(dict.fromkeys(keep))
    drop = [feat for feat in shap_importance.index if feat not in keep]

    if config.get("drop_negative_features", True):
        negative_features = fi_table[fi_table["delta_mean"] <= 0]["feature"].tolist()
        drop = sorted(set(drop).union(negative_features))
        keep = [feat for feat in keep if feat not in drop]

    return keep, drop


def run_feature_selection(X: pd.DataFrame, y: pd.Series, config: Dict, random_state: int = 42) -> FeatureSelectionResult:
    """Execute the SHAP + permutation-based FS pipeline."""

    if X.empty:
        raise ValueError("Feature matrix is empty; cannot run feature selection.")

    holdout_frac = config.get("holdout_fraction", 0.25)
    n_models = config.get("n_fs_models", 3)
    topk = config.get("topk_shap", min(50, X.shape[1]))
    neg_pos_ratio = config.get("fs_eval", {}).get("neg_pos_ratio", 5)
    fs_params = config.get("xgb_fs_params", {})
    inner_random_state = config.get("random_state", random_state)

    X_train_fs, X_holdout_fs, y_train_fs, y_holdout_fs = _make_inner_split(
        X, y, holdout_frac=holdout_frac, random_state=inner_random_state
    )

    X_eval, y_eval = _create_fs_eval(X_holdout_fs, y_holdout_fs, neg_pos_ratio, random_state=inner_random_state)

    models = []
    shap_values = []
    for seed in range(n_models):
        model_params = dict(fs_params)
        model_params["random_state"] = inner_random_state + seed
        model = train_xgb_classifier(
            X_train_fs,
            y_train_fs,
            X_holdout_fs,
            y_holdout_fs,
            params=model_params,
            early_stopping_rounds=model_params.get("early_stopping_rounds", 50),
        )
        models.append(model)
        shap_values.append(_compute_shap_importance(model, X_eval))

    shap_mean = _aggregate_shap(shap_values)
    top_features = shap_mean.head(topk).index.tolist()
    fi_table = _permutation_table(models, X_eval, y_eval, top_features, random_state=inner_random_state)
    kept, dropped = _apply_selection_rules(fi_table, shap_mean, config)

    if not kept:
        kept = list(X.columns)
        dropped = []

    return FeatureSelectionResult(
        kept_features=kept,
        dropped_features=dropped,
        permutation_table=fi_table,
        shap_importance=shap_mean,
    )
