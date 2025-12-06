"""End-to-end experiment runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from fs_xgb.data import load_dataset
from fs_xgb.fs_logic import FeatureSelectionResult, run_feature_selection
from fs_xgb.metrics import compute_classification_metrics
from fs_xgb.models import predict_proba, train_xgb_classifier
from fs_xgb.preprocessing import FeatureEngineer, FeatureEngineerConfig, TargetEncodingConfig
from fs_xgb.splitting import RandomSplitConfig, create_random_splits


@dataclass
class ModelResult:
    name: str
    feature_names: List[str]
    metrics: Dict[str, Dict[str, float]]
    selected: bool = False


@dataclass
class ExperimentResult:
    dataset: str
    run_dir: Path
    fs_result: FeatureSelectionResult
    models: List[ModelResult]


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
        model = train_xgb_classifier(
            X_train,
            y_splits["train"],
            X_val,
            y_splits["val"],
            params=params,
            early_stopping_rounds=params.get("early_stopping_rounds", 100),
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
    fs_result: FeatureSelectionResult,
    models: List[ModelResult],
    results_root: Path,
) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = results_root / dataset / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)

    fs_result.permutation_table.to_csv(run_dir / "permutation_fi.csv", index=False)
    fs_result.shap_importance.rename("mean_abs_shap").to_csv(run_dir / "shap_importance.csv")

    metrics_payload = [
        {"model": m.name, "selected": m.selected, "feature_count": len(m.feature_names), "metrics": m.metrics}
        for m in models
    ]
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2)

    with (run_dir / "features.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "kept_features": fs_result.kept_features,
                "dropped_features": fs_result.dropped_features,
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
    X, y, _ = load_dataset(dataset_name, data_path)

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

    fs_config = dict(config.get("fs", {}))
    fs_config["xgb_fs_params"] = dict(config.get("xgb_fs_params", {}))
    fs_result = run_feature_selection(X_train, splits.y_train, fs_config, random_state=split_config.random_state)

    X_splits = {"train": X_train, "val": X_val, "test": X_test}
    y_splits = {"train": splits.y_train, "val": splits.y_val, "test": splits.y_test}

    model_results = _train_and_eval_models(config, X_splits, y_splits, fs_result)
    run_dir = _persist_results(dataset_name, config, fs_result, model_results, results_root)

    return ExperimentResult(
        dataset=dataset_name,
        run_dir=run_dir,
        fs_result=fs_result,
        models=model_results,
    )
