"""Markdown reporting utilities for experiments and datasets."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import math

from fs_xgb.fs_logic import FeatureSelectionResult
from fs_xgb.types import ExperimentResult, ModeResult, ModelResult


def _markdown_table(headers: List[str], rows: Iterable[Iterable]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = ["| " + " | ".join(str(item) for item in row) + " |" for row in rows]
    return "\n".join([header_line, separator, *row_lines])


def _format_float(value: float, decimals: int = 4) -> str:
    if value != value:  # NaN check
        return "nan"
    return f"{value:.{decimals}f}"


def _build_top_feature_correlations(numeric_df: pd.DataFrame, top_n: int = 10) -> List[List[str]]:
    corr_matrix = numeric_df.corr()
    pairs: List[tuple] = []
    columns = list(corr_matrix.columns)
    for i, col_a in enumerate(columns):
        for j in range(i + 1, len(columns)):
            col_b = columns[j]
            value = corr_matrix.iloc[i, j]
            if pd.isna(value):
                continue
            pairs.append((col_a, col_b, value))
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    top_pairs = pairs[:top_n]
    return [[a, b, _format_float(abs(val))] for a, b, val in top_pairs]


def build_correlation_section(X: pd.DataFrame, y: pd.Series, top_n: int = 10) -> str:
    numeric_df = X.select_dtypes(include="number")
    lines = ["## Correlation Analysis"]
    if numeric_df.empty:
        lines.append("_No numeric features available for correlation analysis._")
        return "\n".join(lines)

    # Feature vs target correlations
    target_series = y.reset_index(drop=True).astype(float)
    corr_with_target = (
        numeric_df.assign(__target__=target_series)
        .corr()
        .loc[:, "__target__"]
        .drop("__target__", errors="ignore")
    )
    corr_with_target = corr_with_target.dropna()
    if not corr_with_target.empty:
        sorted_features = corr_with_target.abs().sort_values(ascending=False).index
        rows = [
            [feature, _format_float(corr_with_target[feature])]
            for feature in sorted_features[:top_n]
        ]
        target_table = _markdown_table(["Feature", "Correlation"], rows)
    else:
        target_table = "_Not enough numeric-target overlap for correlation._"

    # Feature-feature correlations
    pair_rows = _build_top_feature_correlations(numeric_df, top_n)
    if pair_rows:
        feature_table = _markdown_table(
            ["Feature A", "Feature B", "|corr|"], pair_rows
        )
    else:
        feature_table = "_Not enough numeric features for pairwise correlation._"

    lines.extend(
        [
            "### Numeric Features vs Target",
            target_table,
            "",
            "### Top Numeric Feature Pairs (absolute correlation)",
            feature_table,
        ]
    )
    return "\n".join(lines)


def build_metrics_table(models: List[ModelResult]) -> str:
    rows = []
    for model in models:
        rows.append(
            [
                model.name,
                "yes" if model.selected else "no",
                len(model.feature_names),
                _format_float(model.metrics["train"]["pr_auc"]),
                _format_float(model.metrics["val"]["pr_auc"]),
                _format_float(model.metrics["test"]["pr_auc"]),
            ]
        )
    headers = ["Model", "Selected", "Feature Count", "Train PR-AUC", "Val PR-AUC", "Test PR-AUC"]
    return _markdown_table(headers, rows)


def _gap_penalized_score(train_pr: float, val_pr: float, weight: float) -> float:
    if math.isnan(val_pr):
        return float("-inf")
    if math.isnan(train_pr):
        train_pr = val_pr
    gap = train_pr - val_pr
    if math.isnan(gap):
        gap = 0.0
    return val_pr - weight * gap


def build_feature_summary(fs_result: FeatureSelectionResult) -> str:
    kept = len(fs_result.kept_features)
    dropped = len(fs_result.dropped_features)
    lines = [
        f"- Total features (post FE): {kept + dropped}",
        f"- Kept after FS: {kept}",
        f"- Dropped after FS: {dropped}",
    ]
    if fs_result.shap_overfit_features:
        lines.append(
            f"- SHAP overfit flags (high SHAP, low ΔPR-AUC): {len(fs_result.shap_overfit_features)}"
        )
    if fs_result.gain_overfit_features:
        lines.append(
            f"- Gain overfit flags (high gain, low ΔPR-AUC): {len(fs_result.gain_overfit_features)}"
        )
    return "\n".join(lines)


def build_overfit_section(features: List[str]) -> str:
    if not features:
        return "_None._"
    return _markdown_table(["Feature"], [[feature] for feature in features])


def build_top_tables(fs_result: FeatureSelectionResult, top_n: int = 15) -> str:
    permutation = fs_result.permutation_table.head(top_n)
    permutation_rows = [
        [row["feature"], _format_float(row["delta_mean"]), _format_float(row["delta_std"])]
        for _, row in permutation.iterrows()
    ]
    perm_table = _markdown_table(["Feature", "ΔPR-AUC (mean)", "ΔPR-AUC (std)"], permutation_rows)

    shap_series = fs_result.shap_importance.head(top_n)
    shap_table = _markdown_table(
        ["Feature", "Mean abs(SHAP)"], [[feature, _format_float(value)] for feature, value in shap_series.items()]
    )
    return f"### Top Permutation FI (ΔPR-AUC)\n\n{perm_table}\n\n### Top SHAP Importance\n\n{shap_table}"


def build_feature_details_table(fs_result: FeatureSelectionResult) -> str:
    """Build a comprehensive table listing FI metrics for all features."""

    perm_table = fs_result.permutation_table.set_index("feature")
    shap_series = fs_result.shap_importance
    gain_series = fs_result.gain_importance
    shap_flags = set(fs_result.shap_overfit_features)
    gain_flags = set(fs_result.gain_overfit_features)
    threshold = fs_result.selection_threshold

    def _fmt(value: float | str | None) -> str:
        if isinstance(value, str):
            return value
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            return "n/a"
        return _format_float(value)

    rows = []
    for feature in shap_series.index:
        permuted = feature in perm_table.index
        delta_mean = float(perm_table.at[feature, "delta_mean"]) if permuted else float("nan")
        delta_std = float(perm_table.at[feature, "delta_std"]) if permuted else float("nan")
        shap_val = float(shap_series.get(feature, float("nan")))
        gain_val = float(gain_series.get(feature, float("nan")))
        kept = feature in fs_result.kept_features
        gap = None
        if threshold is not None and permuted and not math.isnan(delta_mean):
            gap = delta_mean - threshold
        overfit_labels = []
        if feature in shap_flags:
            overfit_labels.append("SHAP")
        if feature in gain_flags:
            overfit_labels.append("Gain")
        overfit_text = ", ".join(overfit_labels) if overfit_labels else "-"
        rows.append(
            [
                feature,
                "yes" if kept else "no",
                "yes" if permuted else "no",
                _fmt(delta_mean),
                _fmt(delta_std),
                _fmt(gap),
                _fmt(shap_val),
                _fmt(gain_val),
                overfit_text,
            ]
        )

    headers = [
        "Feature",
        "Included",
        "Permuted",
        "ΔPR-AUC (mean)",
        "ΔPR-AUC (std)",
        "Δ - threshold",
        "Mean abs(SHAP)",
        "Mean gain",
        "Overfit flag",
    ]
    return _markdown_table(headers, rows)


def generate_experiment_report_markdown(result: ExperimentResult, config: dict) -> str:
    dataset = result.dataset
    timestamp = result.run_dir.name
    primary_mode = result.primary_mode
    primary = result.mode_results[primary_mode]
    xgb_params = config.get("xgb_final_params", {})
    hyperparam_summary = ", ".join(f"{k}={v}" for k, v in sorted(xgb_params.items())) or "Default XGBoost parameters"
    gap_weight = config.get("selection", {}).get("gap_penalty_weight", 0.0)
    lines = [
        f"# Feature Selection Report – {dataset}",
        f"- Run timestamp: `{timestamp}`",
        f"- Results directory: `{result.run_dir}`",
        f"- Final model hyperparameters: {hyperparam_summary}",
        "",
        "## Mode Summary (thresholds & performance)",
        "_ΔPR min_: minimum absolute PR-AUC drop from permutation importance required to keep a feature. "
        "_k_noise_std_: multiplier applied to the standard deviation of permutation deltas when computing the dynamic threshold. "
        "_Rest policy_: governs what happens to features outside the permutation-evaluated set (keep/drop/SHAP rank filter).",
        build_mode_summary_table(result, config),
        "",
        f"## Mode Rankings (Val PR-AUC − {gap_weight} × gap)",
        build_mode_ranking_table(result, config),
        "",
    ]
    frontier_plot = result.run_dir / "frontier_curve.png"
    train_val_plot = result.run_dir / "train_val_vs_features.png"
    if frontier_plot.exists() and train_val_plot.exists():
        lines.extend(
            [
                f"_Plots saved to `{frontier_plot.name}` and `{train_val_plot.name}`._",
                "",
            ]
        )
    lines.extend(
        [
        "## Model Performance",
        build_metrics_table(primary.models),
        "",
        "## Feature Summary",
        build_feature_summary(primary.fs_result),
        "",
        "## SHAP vs Permutation Overfit Flags",
        build_overfit_section(primary.fs_result.shap_overfit_features),
        "",
        "## Gain vs Permutation Overfit Flags",
        build_overfit_section(primary.fs_result.gain_overfit_features),
        "",
        build_top_tables(primary.fs_result),
        "",
        "## Full Feature Table",
        build_feature_details_table(primary.fs_result),
    ]

    other_modes = [name for name in result.mode_results.keys() if name != primary_mode]
    other_modes.sort(
        key=lambda name: _mode_gap_score(result.mode_results[name], gap_weight),
        reverse=True,
    )
    if other_modes:
        lines.append("")
        lines.append("## Additional Feature-Selection Modes")
    for mode_name in other_modes:
        mode_result = result.mode_results[mode_name]
        lines.append(f"### Mode: {mode_name}")
        if mode_result.identical_to_primary:
            lines.append("_Identical to primary mode results; no additional files generated._")
            lines.append("")
            continue
        lines.extend(
            [
                "#### Feature Set",
                build_feature_list_table(mode_result.fs_result),
                "",
                "#### Overfit Flags",
                "_SHAP vs permutation:_",
                build_overfit_section(mode_result.fs_result.shap_overfit_features),
                "",
                "_Gain vs permutation:_",
                build_overfit_section(mode_result.fs_result.gain_overfit_features),
                "",
                "#### Performance",
                build_metrics_table(mode_result.models),
                "",
            ]
        )
    return "\n".join(lines)


def build_feature_list_table(fs_result: FeatureSelectionResult) -> str:
    kept = ", ".join(fs_result.kept_features) if fs_result.kept_features else "-"
    dropped = ", ".join(fs_result.dropped_features) if fs_result.dropped_features else "-"
    return _markdown_table(
        ["Type", "Features"],
        [
            ["Kept", kept],
            ["Dropped", dropped],
        ],
    )


def build_mode_ranking_table(result: ExperimentResult, config: dict) -> str:
    weight = config.get("selection", {}).get("gap_penalty_weight", 0.0)
    entries = []
    primary_mode = result.mode_results[result.primary_mode]
    baseline_model = next((m for m in primary_mode.models if m.name == "all_features"), None)
    if baseline_model:
        train_pr = baseline_model.metrics["train"]["pr_auc"]
        val_pr = baseline_model.metrics["val"]["pr_auc"]
        entries.append(
            {
                "label": "all_features (baseline)",
                "count": len(baseline_model.feature_names),
                "train": train_pr,
                "val": val_pr,
                "gap": train_pr - val_pr,
                "score": _gap_penalized_score(train_pr, val_pr, weight),
            }
        )
    for mode_name, mode_result in result.mode_results.items():
        fs_model = next((m for m in mode_result.models if m.name == "fs_filtered"), None)
        if not fs_model:
            continue
        train_pr = fs_model.metrics["train"]["pr_auc"]
        val_pr = fs_model.metrics["val"]["pr_auc"]
        entries.append(
            {
                "label": mode_name,
                "count": len(fs_model.feature_names),
                "train": train_pr,
                "val": val_pr,
                "gap": train_pr - val_pr,
                "score": _gap_penalized_score(train_pr, val_pr, weight),
            }
        )
    entries.sort(key=lambda entry: entry["score"], reverse=True)
    rows = [
        [
            entry["label"],
            entry["count"],
            _format_float(entry["train"]),
            _format_float(entry["val"]),
            _format_float(entry["gap"]),
            _format_float(entry["score"]),
        ]
        for entry in entries
    ]
    headers = [
        "Mode / Feature set",
        "Feature count",
        "Train PR-AUC",
        "Val PR-AUC",
        "Gap",
        f"Penalty (w={weight})",
    ]
    return _markdown_table(headers, rows)


def _mode_gap_score(mode_result: ModeResult, weight: float) -> float:
    fs_model = next((m for m in mode_result.models if m.name == "fs_filtered"), None)
    if not fs_model:
        return float("-inf")
    return _gap_penalized_score(
        fs_model.metrics["train"]["pr_auc"],
        fs_model.metrics["val"]["pr_auc"],
        weight,
    )


def write_experiment_report(result: ExperimentResult, config: dict) -> Path:
    markdown = generate_experiment_report_markdown(result, config)
    report_path = result.run_dir / "report.md"
    report_path.write_text(markdown, encoding="utf-8")
    return report_path


def _deep_merge(base: Dict, updates: Dict) -> Dict:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_mode_configs(config: dict) -> Dict[str, Dict]:
    base = copy.deepcopy(config.get("fs", {}))
    modes_cfg = config.get("fs_modes")
    if not modes_cfg:
        return {"moderate": base}
    resolved = {}
    for name, overrides in modes_cfg.items():
        resolved[name] = _deep_merge(base, overrides or {})
    return resolved


def build_mode_summary_table(result: ExperimentResult, config: dict) -> str:
    mode_configs = _resolve_mode_configs(config)
    selection_cfg = config.get("selection", {})
    gap_weight = selection_cfg.get("gap_penalty_weight", 0.0)
    rows = []
    for mode_name, mode_result in result.mode_results.items():
        cfg = mode_configs.get(mode_name, {})
        thresholds = cfg.get("thresholds", {})
        rest_policy = cfg.get("rest_policy", config.get("fs", {}).get("rest_policy", "keep_all"))
        metadata = mode_result.metadata or {}
        delta_value = metadata.get("delta_abs_min", thresholds.get("delta_abs_min", "n/a"))
        k_noise = metadata.get("k_noise_std", thresholds.get("k_noise_std", "n/a"))
        rest_policy = metadata.get("rest_policy", rest_policy)
        fs_filtered = next((m for m in mode_result.models if m.name == "fs_filtered"), None)
        train_pr = fs_filtered.metrics["train"]["pr_auc"] if fs_filtered else float("nan")
        val_pr = fs_filtered.metrics["val"]["pr_auc"] if fs_filtered else float("nan")
        test_pr = fs_filtered.metrics["test"]["pr_auc"] if fs_filtered else float("nan")
        penalty_score = _gap_penalized_score(train_pr, val_pr, gap_weight)
        rows.append(
            [
                f"{mode_name} {'(primary)' if mode_name == result.primary_mode else ''}".strip(),
                len(mode_result.fs_result.kept_features),
                len(mode_result.fs_result.dropped_features),
                delta_value,
                k_noise,
                rest_policy,
                _format_float(train_pr),
                _format_float(val_pr),
                _format_float(test_pr),
                _format_float(penalty_score),
            ]
        )
    headers = [
        "Mode",
        "Kept",
        "Dropped",
        "ΔPR min",
        "k_noise_std",
        "Rest policy",
        "Train PR-AUC",
        "Val PR-AUC",
        "Test PR-AUC",
        f"Penalty (w={gap_weight})",
    ]
    return _markdown_table(headers, rows)


def generate_eda_report_markdown(dataset: str, X: pd.DataFrame, y: pd.Series, metadata: dict) -> str:
    total_rows = len(X)
    positive = int(y.sum())
    negative = total_rows - positive
    class_table = _markdown_table(
        ["Class", "Count", "Percent"],
        [
            ["Positive (1)", positive, _format_float(positive / total_rows * 100, 2)],
            ["Negative (0)", negative, _format_float(negative / total_rows * 100, 2)],
        ],
    )

    summary_rows = []
    for col in X.columns:
        non_null = X[col].notna().sum()
        missing_pct = _format_float((1 - non_null / total_rows) * 100, 2)
        summary_rows.append(
            [
                col,
                str(X[col].dtype),
                X[col].nunique(dropna=True),
                missing_pct,
            ]
        )
    summary_table = _markdown_table(["Feature", "dtype", "Unique", "Missing %"], summary_rows)

    numeric_cols = X.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        numeric_stats = X[numeric_cols].describe().transpose().reset_index().rename(columns={"index": "feature"})
        numeric_table = _markdown_table(
            ["Feature", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"],
            [
                [
                    row["feature"],
                    _format_float(row["mean"]),
                    _format_float(row["std"]),
                    _format_float(row["min"]),
                    _format_float(row["25%"]),
                    _format_float(row["50%"]),
                    _format_float(row["75%"]),
                    _format_float(row["max"]),
                ]
                for _, row in numeric_stats.iterrows()
            ],
        )
        numeric_section = ["## Numeric Feature Stats", numeric_table]
    else:
        numeric_section = ["## Numeric Feature Stats", "_No numeric features available._"]

    correlation_section = build_correlation_section(X, y)

    lines = [
        f"# EDA Report – {dataset}",
        f"- Total rows: {total_rows}",
        f"- Source: {metadata.get('source_url', 'N/A')}",
    ]
    target_definition = metadata.get("target_definition")
    if target_definition:
        lines.append(f"- Target: {target_definition}")
    lines.extend(
        [
            "",
            "## Target Distribution",
            class_table,
            "",
            "## Feature Summary",
            summary_table,
            "",
            *numeric_section,
            "",
            correlation_section,
        ]
    )
    return "\n".join(lines)


def write_eda_report(dataset: str, X: pd.DataFrame, y: pd.Series, metadata: dict, results_root: Path) -> Path:
    markdown = generate_eda_report_markdown(dataset, X, y, metadata)
    report_path = results_root / dataset / "eda.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown, encoding="utf-8")
    return report_path
