"""End-to-end experiment runner."""

from __future__ import annotations

import json
import itertools
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import copy
import pandas as pd
import yaml

from fs_xgb.data import load_dataset
from fs_xgb.eval.reporting import write_eda_report, write_experiment_report
from fs_xgb.fs_logic import (
    FeatureImportanceArtifacts,
    FeatureSelectionResult,
    build_fs_result_from_artifacts,
    compute_fs_artifacts,
    run_feature_selection,
)
from fs_xgb.metrics import compute_classification_metrics
from fs_xgb.models import predict_proba, train_xgb_classifier
from fs_xgb.preprocessing import FeatureEngineer, FeatureEngineerConfig, TargetEncodingConfig
from fs_xgb.splitting import RandomSplitConfig, create_random_splits
from fs_xgb.types import ExperimentResult, ModeResult, ModelResult


def _deep_update(base: Dict, updates: Dict) -> Dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Optional[Path]) -> Dict:
    default_path = Path("fs_xgb/config/default_config.yaml")
    with default_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if path:
        with Path(path).open("r", encoding="utf-8") as fh:
            user_config = yaml.safe_load(fh)
        config = _deep_update(config, user_config or {})
    return config


def _prepare_feature_engineer(config: Dict) -> FeatureEngineer:
    fe_config = config.get("feature_engineering", {})
    target_encoding_cfg = fe_config.get("target_encoding", {})
    return FeatureEngineer(
        FeatureEngineerConfig(
            categorical_columns=fe_config.get("categorical_columns"),
            binary_missing_value=fe_config.get("binary_missing_value", 0.5),
            target_encoding=TargetEncodingConfig(**target_encoding_cfg),
        )
    )


def _build_fs_mode_configs(config: Dict) -> Dict[str, Dict]:
    base = copy.deepcopy(config.get("fs", {}))
    modes_cfg = config.get("fs_modes")
    if not modes_cfg:
        return {"moderate": base}
    mode_configs = {}
    for name, overrides in modes_cfg.items():
        merged = copy.deepcopy(base)
        if overrides:
            merged = _deep_update(merged, copy.deepcopy(overrides))
        mode_configs[name] = merged
    return mode_configs


def _frontier_param_grid(base_fs_config: Dict, frontier_cfg: Dict) -> List[Dict]:
    thresholds = base_fs_config.get("thresholds", {})
    delta_values = frontier_cfg.get("delta_abs_min_values") or [thresholds.get("delta_abs_min", 0.0)]
    k_values = frontier_cfg.get("k_noise_std_values") or [thresholds.get("k_noise_std", 1.0)]
    rest_policies = frontier_cfg.get("rest_policies") or [base_fs_config.get("rest_policy", "keep_all")]
    drop_options = frontier_cfg.get("drop_negative_options")
    if drop_options is None:
        drop_options = [base_fs_config.get("drop_negative_features", True)]
    grid = []
    for delta, k_noise, rest_policy, drop_neg in itertools.product(delta_values, k_values, rest_policies, drop_options):
        grid.append(
            {
                "thresholds": {"delta_abs_min": delta, "k_noise_std": k_noise},
                "rest_policy": rest_policy,
                "drop_negative_features": drop_neg,
            }
        )
    return grid


def _models_equal(models_a: List[ModelResult], models_b: List[ModelResult]) -> bool:
    if len(models_a) != len(models_b):
        return False
    for model_a, model_b in zip(models_a, models_b):
        if model_a.name != model_b.name:
            return False
        if model_a.feature_names != model_b.feature_names:
            return False
        if model_a.selected != model_b.selected:
            return False
        if model_a.metrics != model_b.metrics:
            return False
    return True


def _mode_results_equal(mode_a: ModeResult, mode_b: ModeResult) -> bool:
    fs_a = mode_a.fs_result
    fs_b = mode_b.fs_result
    if fs_a.kept_features != fs_b.kept_features:
        return False
    if fs_a.dropped_features != fs_b.dropped_features:
        return False
    if not fs_a.permutation_table.equals(fs_b.permutation_table):
        return False
    if not fs_a.shap_importance.equals(fs_b.shap_importance):
        return False
    return _models_equal(mode_a.models, mode_b.models)


def _evaluate_model(model, X_dict: Dict[str, pd.DataFrame], y_dict: Dict[str, pd.Series]) -> Dict[str, Dict[str, float]]:
    metrics = {}
    for split in ["train", "val", "test"]:
        y_true = y_dict[split]
        y_score = predict_proba(model, X_dict[split])
        metrics[split] = compute_classification_metrics(y_true, y_score)
    return metrics


def _train_and_eval_models(
    config: Dict,
    X_splits: Dict[str, pd.DataFrame],
    y_splits: Dict[str, pd.Series],
    fs_result: FeatureSelectionResult,
    include_all_features: bool = True,
) -> List[ModelResult]:
    params = config.get("xgb_final_params", {})
    selection_cfg = config.get("selection", {})
    tolerance = selection_cfg.get("val_tolerance_relative", 0.01)
    feature_sets = []
    if include_all_features:
        feature_sets.append(("all_features", list(X_splits["train"].columns)))
    feature_sets.append(("fs_filtered", fs_result.kept_features))

    models = []
    val_scores = []
    for name, features in feature_sets:
        X_train = X_splits["train"][features]
        X_val = X_splits["val"][features]
        X_test = X_splits["test"][features]
        model_params = dict(params)
        model_params.setdefault("random_state", selection_cfg.get("final_model_random_state", 42))
        model_params["subsample"] = 1.0
        model = train_xgb_classifier(
            X_train,
            y_splits["train"],
            X_val,
            y_splits["val"],
            params=model_params,
            early_stopping_rounds=model_params.get("early_stopping_rounds", 100),
        )
        metrics = _evaluate_model(
            model,
            {"train": X_train, "val": X_val, "test": X_test},
            y_splits,
        )
        models.append(ModelResult(name=name, feature_names=features, metrics=metrics))
        val_scores.append(metrics["val"]["pr_auc"])

    best_val = max(val_scores)
    cutoff = (1 - tolerance) * best_val
    eligible = [m for m in models if m.metrics["val"]["pr_auc"] >= cutoff]
    eligible.sort(key=lambda m: (len(m.feature_names), -m.metrics["val"]["pr_auc"]))
    selected = eligible[0]
    for model in models:
        model.selected = model is selected
    return models


def _persist_results(
    dataset: str,
    config: Dict,
    mode_results: Dict[str, ModeResult],
    primary_mode: str,
    results_root: Path,
    frontier_log: Optional[List[Dict]] = None,
) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = results_root / dataset / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)

    for mode_name, mode_result in mode_results.items():
        if mode_name != primary_mode and mode_result.identical_to_primary:
            continue
        suffix = "" if mode_name == primary_mode else f"_{mode_name}"
        mode_result.fs_result.permutation_table.to_csv(
            run_dir / f"permutation_fi{suffix}.csv", index=False
        )
        mode_result.fs_result.shap_importance.rename("mean_abs_shap").to_csv(
            run_dir / f"shap_importance{suffix}.csv"
        )
        metrics_payload = [
            {"model": m.name, "selected": m.selected, "feature_count": len(m.feature_names), "metrics": m.metrics}
            for m in mode_result.models
        ]
        with (run_dir / f"metrics{suffix}.json").open("w", encoding="utf-8") as fh:
            json.dump(metrics_payload, fh, indent=2)
        with (run_dir / f"features{suffix}.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "kept_features": mode_result.fs_result.kept_features,
                    "dropped_features": mode_result.fs_result.dropped_features,
                },
                fh,
                indent=2,
            )

    if frontier_log:
        with (run_dir / "frontier_candidates.json").open("w", encoding="utf-8") as fh:
            json.dump(frontier_log, fh, indent=2)

    return run_dir


def _run_profile_modes(
    config: Dict,
    X_splits: Dict[str, pd.DataFrame],
    y_splits: Dict[str, pd.Series],
    split_config: RandomSplitConfig,
) -> Tuple[Dict[str, ModeResult], str, Optional[List[Dict]]]:
    mode_configs = _build_fs_mode_configs(config)
    mode_results: Dict[str, ModeResult] = {}
    for mode_name, fs_mode_cfg in mode_configs.items():
        fs_config = copy.deepcopy(fs_mode_cfg)
        fs_config["xgb_fs_params"] = dict(config.get("xgb_fs_params", {}))
        fs_result = run_feature_selection(
            X_splits["train"], y_splits["train"], fs_config, random_state=split_config.random_state
        )
        model_results = _train_and_eval_models(config, X_splits, y_splits, fs_result)
        mode_results[mode_name] = ModeResult(fs_result=fs_result, models=model_results)

    primary_mode = config.get("fs_primary_mode", "moderate")
    if primary_mode not in mode_results:
        primary_mode = next(iter(mode_results.keys()))
    primary_result = mode_results[primary_mode]
    for mode_name, mode_result in mode_results.items():
        if mode_name == primary_mode:
            continue
        mode_result.identical_to_primary = _mode_results_equal(primary_result, mode_result)
    return mode_results, primary_mode, None


def _select_frontier_primary(mode_results: Dict[str, ModeResult], selection_cfg: Dict) -> str:
    tolerance = selection_cfg.get("val_tolerance_relative", 0.01)
    ranked = []
    for mode_name, mode_result in mode_results.items():
        fs_model = next((m for m in mode_result.models if m.name == "fs_filtered"), None)
        if not fs_model:
            continue
        val_score = fs_model.metrics["val"]["pr_auc"]
        train_score = fs_model.metrics["train"]["pr_auc"]
        gap = train_score - val_score
        if math.isnan(gap):
            gap = float("inf")
        ranked.append(
            {
                "mode": mode_name,
                "val": val_score,
                "gap": gap,
                "feature_count": len(fs_model.feature_names),
            }
        )
    if not ranked:
        raise ValueError("No candidate FS models evaluated during frontier search.")
    best_val = max(row["val"] for row in ranked)
    cutoff = (1 - tolerance) * best_val
    eligible = [row for row in ranked if row["val"] >= cutoff]
    eligible.sort(key=lambda row: (row["feature_count"], row["gap"], -row["val"], row["mode"]))
    return eligible[0]["mode"]


def _run_frontier_mode(
    config: Dict,
    X_splits: Dict[str, pd.DataFrame],
    y_splits: Dict[str, pd.Series],
    split_config: RandomSplitConfig,
) -> Tuple[Dict[str, ModeResult], str, List[Dict]]:
    base_fs_config = copy.deepcopy(config.get("fs", {}))
    base_fs_config["xgb_fs_params"] = dict(config.get("xgb_fs_params", {}))
    artifacts = compute_fs_artifacts(
        X_splits["train"], y_splits["train"], base_fs_config, random_state=split_config.random_state
    )
    frontier_cfg = config.get("fs_search", {}).get("frontier", {})
    grid = _frontier_param_grid(base_fs_config, frontier_cfg)
    max_candidates = frontier_cfg.get("max_candidates")
    if not max_candidates or max_candidates <= 0:
        max_candidates = len(grid) or 1
    feature_to_mode: Dict[Tuple[str, ...], str] = {}
    mode_results: Dict[str, ModeResult] = {}
    frontier_log: List[Dict] = []

    for idx, candidate_cfg in enumerate(grid, start=1):
        fs_config = copy.deepcopy(base_fs_config)
        fs_config.setdefault("thresholds", {})
        fs_config["thresholds"]["delta_abs_min"] = candidate_cfg["thresholds"]["delta_abs_min"]
        fs_config["thresholds"]["k_noise_std"] = candidate_cfg["thresholds"]["k_noise_std"]
        fs_config["rest_policy"] = candidate_cfg["rest_policy"]
        fs_config["drop_negative_features"] = candidate_cfg["drop_negative_features"]
        fs_result = build_fs_result_from_artifacts(artifacts, fs_config, list(X_splits["train"].columns))
        feature_key = tuple(fs_result.kept_features)
        metadata = {
            "delta_abs_min": fs_config["thresholds"]["delta_abs_min"],
            "k_noise_std": fs_config["thresholds"]["k_noise_std"],
            "rest_policy": fs_config.get("rest_policy", "keep_all"),
            "drop_negative_features": fs_config.get("drop_negative_features", True),
            "kept_features": len(fs_result.kept_features),
            "dropped_features": len(fs_result.dropped_features),
        }
        if feature_key in feature_to_mode:
            candidate_entry = {
                "candidate_index": idx,
                "mode": feature_to_mode[feature_key],
                **metadata,
                "is_duplicate": True,
                "skipped_due_to_limit": False,
                "fs_filtered_metrics": None,
            }
            frontier_log.append(candidate_entry)
            continue
        if len(mode_results) >= max_candidates:
            frontier_log.append(
                {
                    "candidate_index": idx,
                    "mode": None,
                    **metadata,
                    "is_duplicate": False,
                    "skipped_due_to_limit": True,
                    "fs_filtered_metrics": None,
                }
            )
            continue
        models = _train_and_eval_models(config, X_splits, y_splits, fs_result)
        mode_name = f"frontier_{len(mode_results) + 1}"
        mode_results[mode_name] = ModeResult(fs_result=fs_result, models=models, metadata=metadata)
        feature_to_mode[feature_key] = mode_name
        fs_model = next((m for m in models if m.name == "fs_filtered"), None)
        frontier_log.append(
            {
                "candidate_index": idx,
                "mode": mode_name,
                **metadata,
                "is_duplicate": False,
                "skipped_due_to_limit": False,
                "fs_filtered_metrics": fs_model.metrics if fs_model else None,
            }
        )

    if not mode_results:
        raise ValueError("Frontier search did not produce any unique feature sets.")

    primary_mode = _select_frontier_primary(mode_results, config.get("selection", {}))
    for mode_name, mode_result in mode_results.items():
        for model in mode_result.models:
            model.selected = mode_name == primary_mode and model.name == "fs_filtered"
    return mode_results, primary_mode, frontier_log


def run_experiment(config_path: Optional[Path], results_root: Path = Path("results")) -> ExperimentResult:
    config = load_config(config_path)
    dataset_name = config["dataset"]
    dataset_path = config.get("dataset_path")
    data_path = Path(dataset_path) if dataset_path else None
    X, y, metadata = load_dataset(dataset_name, data_path)
    write_eda_report(dataset_name, X, y, metadata, results_root)

    splits_cfg = config.get("splits", {})
    split_config = RandomSplitConfig(
        test_size=splits_cfg.get("test_size", 0.2),
        val_size=splits_cfg.get("val_size", 0.2),
        random_state=splits_cfg.get("random_state", 42),
    )
    splits = create_random_splits(X, y, split_config)

    feature_engineer = _prepare_feature_engineer(config)
    X_train = feature_engineer.fit_transform(splits.X_train, splits.y_train)
    X_val = feature_engineer.transform(splits.X_val)
    X_test = feature_engineer.transform(splits.X_test)

    X_splits = {"train": X_train, "val": X_val, "test": X_test}
    y_splits = {"train": splits.y_train, "val": splits.y_val, "test": splits.y_test}

    fs_search_mode = config.get("fs_search", {}).get("mode", "profiles")
    if fs_search_mode == "frontier":
        mode_results, primary_mode, frontier_log = _run_frontier_mode(config, X_splits, y_splits, split_config)
    else:
        mode_results, primary_mode, frontier_log = _run_profile_modes(config, X_splits, y_splits, split_config)

    run_dir = _persist_results(dataset_name, config, mode_results, primary_mode, results_root, frontier_log)
    experiment_result = ExperimentResult(
        dataset=dataset_name,
        run_dir=run_dir,
        mode_results=mode_results,
        primary_mode=primary_mode,
    )
    write_experiment_report(experiment_result, config)

    return experiment_result
