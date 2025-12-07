"""Loader for the Santander Customer Satisfaction dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from ..datasets_registry import DatasetMetadata, register_dataset

DATASET_NAME = "santander_customer_satisfaction"
TRAIN_FILENAME = "train.csv"
SOURCE_URL = "https://www.kaggle.com/c/santander-customer-satisfaction"
TARGET_COLUMN = "TARGET"
ID_COLUMN = "ID"
BINARY_TARGET = "is_unsatisfied"

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "santander_customer_satisfaction"


def _resolve_train_path(path: Optional[Path]) -> Path:
    resolved = Path(path) if path else DEFAULT_DATA_PATH
    if resolved.is_dir():
        resolved = resolved / TRAIN_FILENAME
    return resolved


def load_santander_dataset(csv_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
    """Load Santander customer satisfaction data and return (features, target, metadata)."""

    train_path = _resolve_train_path(csv_path)
    if not train_path.exists():
        raise FileNotFoundError(
            f"Santander dataset not found at {train_path}. "
            "Download the Kaggle competition files and place train.csv under data/raw/santander_customer_satisfaction/."
        )

    df = pd.read_csv(train_path)
    missing_columns = {TARGET_COLUMN, ID_COLUMN} - set(df.columns)
    if missing_columns:
        raise ValueError(f"Santander dataset is missing required columns: {missing_columns}")

    df = df.drop_duplicates().reset_index(drop=True)
    target = df[TARGET_COLUMN].astype(int).rename(BINARY_TARGET)
    features = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])

    metadata = {
        "task": "binary_classification",
        "positive_class": "TARGET == 1 (dissatisfied customers)",
        "total_rows": str(len(df)),
        "source_url": SOURCE_URL,
    }

    return features, target, metadata


register_dataset(
    DatasetMetadata(
        name=DATASET_NAME,
        loader=load_santander_dataset,
        default_path=DEFAULT_DATA_PATH,
        description="Kaggle competition dataset with anonymized customer metrics and a dissatisfaction flag.",
        target=BINARY_TARGET,
        positive_class=1,
        source_url=SOURCE_URL,
    )
)
