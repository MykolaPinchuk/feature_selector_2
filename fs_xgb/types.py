"""Shared dataclasses for experiment results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from fs_xgb.fs_logic import FeatureSelectionResult


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
