"""Plotting helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def _unique_frontier_entries(frontier_log: List[Dict]) -> List[Dict]:
    entries: List[Dict] = []
    seen_modes: set[str] = set()
    for entry in frontier_log:
        mode = entry.get("mode")
        if not mode or entry.get("is_duplicate") or entry.get("skipped_due_to_limit"):
            continue
        if mode in seen_modes:
            continue
        metrics = entry.get("fs_filtered_metrics")
        if not metrics:
            continue
        seen_modes.add(mode)
        entries.append(entry)
    return entries


def _gap_penalized_score(train_pr: float, val_pr: float, weight: float) -> float:
    gap = (train_pr or 0.0) - (val_pr or 0.0)
    return (val_pr or 0.0) - weight * gap


def _plot_frontier_curve(entries: List[Dict], run_dir: Path, gap_weight: float) -> None:
    entries = sorted(entries, key=lambda e: e.get("kept_features", 0))
    x_vals = [entry.get("kept_features", 0) for entry in entries]
    val_scores = [entry["fs_filtered_metrics"]["val"]["pr_auc"] for entry in entries]
    best_entry = max(
        entries,
        key=lambda e: _gap_penalized_score(
            e["fs_filtered_metrics"]["train"]["pr_auc"],
            e["fs_filtered_metrics"]["val"]["pr_auc"],
            gap_weight,
        ),
    )
    best_idx = entries.index(best_entry)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_vals, val_scores, marker="o", label="Validation PR-AUC")
    ax.scatter(
        [x_vals[best_idx]],
        [val_scores[best_idx]],
        color="crimson",
        label="Best (penalized)",
    )
    ax.set_xlabel("Feature count")
    ax.set_ylabel("Validation PR-AUC")
    ax.set_title("Frontier – Validation PR-AUC vs Feature Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "frontier_curve.png", dpi=200)
    plt.close(fig)


def _plot_train_val(entries: List[Dict], run_dir: Path) -> None:
    entries = sorted(entries, key=lambda e: e.get("kept_features", 0))
    x_vals = [entry.get("kept_features", 0) for entry in entries]
    val_scores = [entry["fs_filtered_metrics"]["val"]["pr_auc"] for entry in entries]
    train_scores = [entry["fs_filtered_metrics"]["train"]["pr_auc"] for entry in entries]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_vals, train_scores, marker="o", label="Train PR-AUC")
    ax.plot(x_vals, val_scores, marker="o", label="Val PR-AUC")
    ax.set_xlabel("Feature count")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Train vs Validation PR-AUC by Feature Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "train_val_vs_features.png", dpi=200)
    plt.close(fig)


def generate_frontier_plots(frontier_log: List[Dict] | None, run_dir: Path, selection_cfg: Dict) -> None:
    """Generate helpful plots when a frontier search has been executed."""

    if not frontier_log:
        return
    entries = _unique_frontier_entries(frontier_log)
    if not entries:
        return
    gap_weight = selection_cfg.get("gap_penalty_weight", 0.0)
    _plot_frontier_curve(entries, run_dir, gap_weight)
    _plot_train_val(entries, run_dir)
