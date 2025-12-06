"""Loader for the 'Factors Affecting University Student Grades' dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from ..datasets_registry import DatasetMetadata, register_dataset

DATASET_NAME = "student_grades"
DATA_FILENAME = "factors_affecting_university_student_grades.csv"
SOURCE_URL = "https://www.kaggle.com/datasets/atifmasih/factors-affecting-university-student-grades"
TARGET_COLUMN = "Grade"
BINARY_TARGET = "is_high_grade"

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / DATA_FILENAME


def _normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace from column names and string cells."""

    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda value: value.strip() if isinstance(value, str) else value)
        df[col] = df[col].replace({"": pd.NA})
    return df


def _build_target(df: pd.DataFrame) -> pd.Series:
    """Construct the binary target used for FS experiments."""

    normalized = df[TARGET_COLUMN].str.upper()
    return (normalized == "A").astype(int)


def load_student_grades(csv_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
    """Load the dataset and return (features, target, metadata)."""

    path = Path(csv_path) if csv_path else DEFAULT_DATA_PATH
    if path.is_dir():
        path = path / DATA_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Student grades dataset not found at {path}. "
            "Download the CSV and place it under data/raw/."
        )

    df = pd.read_csv(path)
    df = _normalize_strings(df).dropna(subset=[TARGET_COLUMN]).drop_duplicates().reset_index(drop=True)

    target = _build_target(df)
    features = df.drop(columns=[TARGET_COLUMN])

    metadata = {
        "task": "binary_classification",
        "positive_class": "Grade == 'A'",
        "total_rows": str(len(df)),
        "source_url": SOURCE_URL,
    }

    return features, target, metadata


register_dataset(
    DatasetMetadata(
        name=DATASET_NAME,
        loader=load_student_grades,
        default_path=DEFAULT_DATA_PATH,
        description="Kaggle dataset capturing demographic, academic, and lifestyle factors that influence student grades.",
        target=BINARY_TARGET,
        positive_class=1,
        source_url=SOURCE_URL,
    )
)
