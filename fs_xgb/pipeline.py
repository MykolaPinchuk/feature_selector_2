"""End-to-end experiment runner."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import copy
import pandas as pd
import yaml

from fs_xgb.data import load_dataset
from fs_xgb.eval.reporting import write_eda_report, write_experiment_report
from fs_xgb.fs_logic import FeatureSelectionResult, run_feature_selection
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
) -> List[ModelResult]:
    params = config.get("xgb_final_params", {})
    selection_cfg = config.get("selection", {})
    tolerance = selection_cfg.get("val_tolerance_relative", 0.01)
    feature_sets = [
        ("all_features", list(X_splits["train"].columns)),
        ("fs_filtered", fs_result.kept_features),
    ]

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
) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = results_root / dataset / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)

    for mode_name, mode_result in mode_results.items():
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

    return run_dir


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

    mode_configs = _build_fs_mode_configs(config)
    X_splits = {"train": X_train, "val": X_val, "test": X_test}
    y_splits = {"train": splits.y_train, "val": splits.y_val, "test": splits.y_test}

    mode_results: Dict[str, ModeResult] = {}
    for mode_name, fs_mode_cfg in mode_configs.items():
        fs_config = copy.deepcopy(fs_mode_cfg)
        fs_config["xgb_fs_params"] = dict(config.get("xgb_fs_params", {}))
        fs_result = run_feature_selection(
            X_train, splits.y_train, fs_config, random_state=split_config.random_state
        )
        model_results = _train_and_eval_models(config, X_splits, y_splits, fs_result)
        mode_results[mode_name] = ModeResult(fs_result=fs_result, models=model_results)

    primary_mode = config.get("fs_primary_mode", "moderate")
    if primary_mode not in mode_results:
        primary_mode = next(iter(mode_results.keys()))

    run_dir = _persist_results(dataset_name, config, mode_results, primary_mode, results_root)
    experiment_result = ExperimentResult(
        dataset=dataset_name,
        run_dir=run_dir,
        mode_results=mode_results,
        primary_mode=primary_mode,
    )
    write_experiment_report(experiment_result, config)

    return experiment_result
