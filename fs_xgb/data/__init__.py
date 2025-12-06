"""Data loading utilities and dataset registry."""

from .datasets_registry import DatasetMetadata, load_dataset, list_datasets

# Ensure dataset loaders register themselves via import side-effects.
from . import loaders as _loaders  # noqa: F401

__all__ = ["DatasetMetadata", "load_dataset", "list_datasets"]
