"""Implementation of the permutation- and SHAP-based feature selection workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fs_xgb.metrics import pr_auc_score
from fs_xgb.models import predict_proba, train_xgb_classifier

try:  # pragma: no cover - optional import for unit tests
    import shap
except ImportError:  # pragma: no cover - tests without shap
    shap = None


@dataclass
class FeatureSelectionResult:
    """Outputs from the feature-selection routine."""

    kept_features: List[str]
    dropped_features: List[str]
    permutation_table: pd.DataFrame
    shap_importance: pd.Series
    gain_importance: pd.Series
    shap_overfit_features: List[str] = field(default_factory=list)
    gain_overfit_features: List[str] = field(default_factory=list)

    @property
    def overfit_features(self) -> List[str]:
        """Union of SHAP- and gain-based mismatch flags."""

        return sorted(set(self.shap_overfit_features).union(self.gain_overfit_features))


@dataclass
class FeatureImportanceArtifacts:
    """Reusable SHAP + permutation outputs for multiple selection passes."""

    permutation_table: pd.DataFrame
    shap_importance: pd.Series
    gain_importance: pd.Series


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
    if shap is None:
        raise RuntimeError("shap must be installed to run feature selection.")
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


def _flag_overfit_features(fi_table: pd.DataFrame, shap_importance: pd.Series, config: Dict) -> List[str]:
    cfg = config or {}
    if not cfg.get("enabled"):
        return []
    max_rank = cfg.get("max_shap_rank")
    delta_max = cfg.get("delta_max", 0.0)
    delta_std_max = cfg.get("delta_std_max")
    shap_rank = shap_importance.rank(ascending=False, method="min")
    if fi_table.empty:
        fi_lookup = pd.DataFrame(columns=["delta_mean", "delta_std"]).set_index(pd.Index([]))
    else:
        fi_lookup = fi_table.set_index("feature")
    flagged = []
    for feature, rank in shap_rank.items():
        if max_rank is not None and rank > max_rank:
            continue
        delta_mean = float(fi_lookup.at[feature, "delta_mean"]) if feature in fi_lookup.index else 0.0
        delta_std = float(fi_lookup.at[feature, "delta_std"]) if feature in fi_lookup.index else 0.0
        if delta_mean <= delta_max and (delta_std_max is None or delta_std <= delta_std_max):
            flagged.append(feature)
    return flagged


def _flag_gain_overfit_features(fi_table: pd.DataFrame, gain_importance: pd.Series | None, config: Dict) -> List[str]:
    if gain_importance is None:
        return []
    cfg = config or {}
    if not cfg.get("enabled"):
        return []
    max_rank = cfg.get("max_gain_rank")
    delta_max = cfg.get("delta_max", 0.0)
    delta_std_max = cfg.get("delta_std_max")
    gain_rank = gain_importance.rank(ascending=False, method="min")
    if fi_table.empty:
        fi_lookup = pd.DataFrame(columns=["delta_mean", "delta_std"]).set_index(pd.Index([]))
    else:
        fi_lookup = fi_table.set_index("feature")
    flagged = []
    for feature, rank in gain_rank.items():
        if max_rank is not None and rank > max_rank:
            continue
        delta_mean = float(fi_lookup.at[feature, "delta_mean"]) if feature in fi_lookup.index else 0.0
        delta_std = float(fi_lookup.at[feature, "delta_std"]) if feature in fi_lookup.index else 0.0
        if delta_mean <= delta_max and (delta_std_max is None or delta_std <= delta_std_max):
            flagged.append(feature)
    return flagged


def _apply_selection_rules(
    fi_table: pd.DataFrame,
    shap_importance: pd.Series,
    gain_importance: pd.Series | None,
    config: Dict,
) -> Tuple[List[str], List[str], List[str], List[str]]:
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

    shap_cfg = config.get("overfit_filter", {})
    shap_flags = set(_flag_overfit_features(fi_table, shap_importance, shap_cfg))
    shap_action = shap_cfg.get("action", "drop")
    if shap_action not in {"drop", "demote"}:
        shap_action = "drop"

    gain_cfg = config.get("gain_overfit_filter", {})
    gain_flags = set(_flag_gain_overfit_features(fi_table, gain_importance, gain_cfg))
    gain_action = gain_cfg.get("action", "drop")
    if gain_action not in {"drop", "demote"}:
        gain_action = "drop"

    drop_overfit = set()
    demote_overfit = set()
    if shap_action == "drop":
        drop_overfit.update(shap_flags)
    else:
        demote_overfit.update(shap_flags)
    if gain_action == "drop":
        drop_overfit.update(gain_flags)
    else:
        demote_overfit.update(gain_flags)

    if drop_overfit:
        keep = [feat for feat in keep if feat not in drop_overfit]
    if demote_overfit:
        keep = [feat for feat in keep if feat not in demote_overfit]

    rest_policy = config.get("rest_policy", "keep_all")
    shap_rank = shap_importance.rank(ascending=False, method="min")
    permuted_features = set(fi_table["feature"]).difference(demote_overfit)
    rest_features = [
        feat
        for feat in shap_importance.index
        if feat not in permuted_features and feat not in drop_overfit
    ]

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
    if drop_overfit:
        drop = sorted(set(drop).union(drop_overfit))
        keep = [feat for feat in keep if feat not in drop_overfit]
    return keep, drop, sorted(shap_flags), sorted(gain_flags)


def compute_fs_artifacts(X: pd.DataFrame, y: pd.Series, config: Dict, random_state: int = 42) -> FeatureImportanceArtifacts:
    """Run the expensive SHAP + permutation steps once and return reusable artifacts."""

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
    gain_values = []
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
        booster = model.get_booster()
        gain_dict = booster.get_score(importance_type="gain")
        gain_series = pd.Series(gain_dict, dtype=float)
        gain_series = gain_series.reindex(X_train_fs.columns, fill_value=0.0)
        gain_values.append(gain_series)

    shap_mean = _aggregate_shap(shap_values)
    gain_mean = pd.concat(gain_values, axis=1).mean(axis=1).sort_values(ascending=False)
    top_features = shap_mean.head(topk).index.tolist()
    fi_table = _permutation_table(models, X_eval, y_eval, top_features, random_state=inner_random_state)
    return FeatureImportanceArtifacts(permutation_table=fi_table, shap_importance=shap_mean, gain_importance=gain_mean)


def build_fs_result_from_artifacts(artifacts: FeatureImportanceArtifacts, config: Dict, all_features: List[str]) -> FeatureSelectionResult:
    """Apply selection thresholds using precomputed artifacts."""

    fi_table = artifacts.permutation_table.copy()
    shap_importance = artifacts.shap_importance.copy()
    gain_importance = artifacts.gain_importance.copy()
    kept, dropped, shap_flags, gain_flags = _apply_selection_rules(
        fi_table,
        shap_importance,
        gain_importance,
        config,
    )
    if not kept:
        kept = list(all_features)
        dropped = []
    return FeatureSelectionResult(
        kept_features=kept,
        dropped_features=dropped,
        permutation_table=fi_table,
        shap_importance=shap_importance,
        gain_importance=gain_importance,
        shap_overfit_features=shap_flags,
        gain_overfit_features=gain_flags,
    )


def run_feature_selection(X: pd.DataFrame, y: pd.Series, config: Dict, random_state: int = 42) -> FeatureSelectionResult:
    """Execute the SHAP + permutation-based FS pipeline."""

    artifacts = compute_fs_artifacts(X, y, config, random_state=random_state)
    return build_fs_result_from_artifacts(artifacts, config, list(X.columns))
