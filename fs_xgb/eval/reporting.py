"""Markdown reporting utilities for experiments and datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

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


def build_feature_summary(fs_result: FeatureSelectionResult) -> str:
    kept = len(fs_result.kept_features)
    dropped = len(fs_result.dropped_features)
    return (
        f"- Total features (post FE): {kept + dropped}\n"
        f"- Kept after FS: {kept}\n"
        f"- Dropped after FS: {dropped}"
    )


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


def generate_experiment_report_markdown(result: ExperimentResult, config: dict) -> str:
    dataset = result.dataset
    timestamp = result.run_dir.name
    primary_mode = result.primary_mode
    primary = result.mode_results[primary_mode]
    lines = [
        f"# Feature Selection Report – {dataset}",
        f"- Run timestamp: `{timestamp}`",
        f"- Results directory: `{result.run_dir}`",
        "",
        "## Model Performance",
        build_metrics_table(primary.models),
        "",
        "## Feature Summary",
        build_feature_summary(primary.fs_result),
        "",
        build_top_tables(primary.fs_result),
    ]

    other_modes = [name for name in result.mode_results.keys() if name != primary_mode]
    if other_modes:
        lines.append("")
        lines.append("## Additional Feature-Selection Modes")
    for mode_name in other_modes:
        mode_result = result.mode_results[mode_name]
        lines.extend(
            [
                f"### Mode: {mode_name}",
                "#### Feature Set",
                build_feature_list_table(mode_result.fs_result),
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


def write_experiment_report(result: ExperimentResult, config: dict) -> Path:
    markdown = generate_experiment_report_markdown(result, config)
    report_path = result.run_dir / "report.md"
    report_path.write_text(markdown, encoding="utf-8")
    return report_path


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

    lines = [
        f"# EDA Report – {dataset}",
        f"- Total rows: {total_rows}",
        f"- Source: {metadata.get('source_url', 'N/A')}",
        "",
        "## Target Distribution",
        class_table,
        "",
        "## Feature Summary",
        summary_table,
        "",
        *numeric_section,
    ]
    return "\n".join(lines)


def write_eda_report(dataset: str, X: pd.DataFrame, y: pd.Series, metadata: dict, results_root: Path) -> Path:
    markdown = generate_eda_report_markdown(dataset, X, y, metadata)
    report_path = results_root / dataset / "eda.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown, encoding="utf-8")
    return report_path
