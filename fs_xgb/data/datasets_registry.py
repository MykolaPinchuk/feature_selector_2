"""Dataset registry and discovery utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import pandas as pd

DatasetLoader = Callable[[Optional[Path]], Tuple[pd.DataFrame, pd.Series, Dict[str, str]]]


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata describing a dataset available to the FS framework."""

    name: str
    loader: DatasetLoader
    default_path: Path
    description: str
    target: str
    positive_class: int
    timestamp_col: Optional[str] = None
    source_url: Optional[str] = None


_DATASETS: Dict[str, DatasetMetadata] = {}


def register_dataset(metadata: DatasetMetadata) -> None:
    """Register a dataset loader so it can be referenced by name."""

    if metadata.name in _DATASETS:
        raise ValueError(f"Dataset '{metadata.name}' already registered.")
    _DATASETS[metadata.name] = metadata


def list_datasets() -> Iterable[str]:
    """Return the registered dataset names."""

    return tuple(_DATASETS.keys())


def get_dataset_metadata(name: str) -> DatasetMetadata:
    """Return metadata for a given dataset."""

    try:
        return _DATASETS[name]
    except KeyError as exc:  # pragma: no cover - simple defensive branch
        raise KeyError(f"Unknown dataset '{name}'.") from exc


def load_dataset(name: str, data_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
    """Load a dataset by name and return feature matrix, target vector, and metadata."""

    metadata = get_dataset_metadata(name)
    effective_path = data_path or metadata.default_path
    return metadata.loader(effective_path)
