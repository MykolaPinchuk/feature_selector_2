"""Loader for the Census Income (KDD) dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from ..datasets_registry import DatasetMetadata, register_dataset

DATASET_NAME = "census_income"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/117/census%2Bincome%2Bkdd"
TRAIN_FILENAME = "census-income.data"
TEST_FILENAME = "census-income.test"
TARGET_COLUMN = "income"
TARGET_NAME = "is_high_income"
POSITIVE_LABEL = "50000+"
INSTANCE_WEIGHT_COLUMN = "instance_weight"

COLUMN_NAMES = [
    "age",
    "class_of_worker",
    "detailed_industry_recode",
    "detailed_occupation_recode",
    "education",
    "wage_per_hour",
    "enroll_in_edu_inst_last_wk",
    "marital_status",
    "major_industry_code",
    "major_occupation_code",
    "race",
    "hispanic_origin",
    "sex",
    "member_of_labor_union",
    "reason_for_unemployment",
    "employment_status",
    "capital_gains",
    "capital_losses",
    "dividends_from_stocks",
    "tax_filer_status",
    "region_of_previous_residence",
    "state_of_previous_residence",
    "detailed_household_family_stat",
    "detailed_household_summary",
    INSTANCE_WEIGHT_COLUMN,
    "migration_code_change_in_msa",
    "migration_code_change_in_reg",
    "migration_code_move_within_reg",
    "live_in_this_house_1_year_ago",
    "migration_prev_res_in_sunbelt",
    "num_persons_worked_for_employer",
    "family_members_under_18",
    "country_of_birth_father",
    "country_of_birth_mother",
    "country_of_birth_self",
    "citizenship",
    "own_business_or_self_employed",
    "fill_inc_questionnaire_for_veterans_admin",
    "veterans_benefits",
    "weeks_worked_in_year",
    "year",
    TARGET_COLUMN,
]

NUMERIC_COLUMNS = [
    "age",
    "wage_per_hour",
    "capital_gains",
    "capital_losses",
    "dividends_from_stocks",
    "num_persons_worked_for_employer",
    "weeks_worked_in_year",
]

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "census_income_kdd"


def _read_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Census Income file not found at {path}.")
    return pd.read_csv(
        path,
        names=COLUMN_NAMES,
        header=None,
        na_values=["?"],
        skipinitialspace=True,
    )


def _normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].str.strip()
    return df


def load_census_income(data_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
    """Load and preprocess the Census Income dataset."""

    base_path = Path(data_path) if data_path else DEFAULT_DATA_DIR
    if base_path.is_dir():
        train_path = base_path / TRAIN_FILENAME
        test_path = base_path / TEST_FILENAME
    elif base_path.is_file():
        if base_path.name == TRAIN_FILENAME:
            train_path = base_path
            test_path = base_path.with_name(TEST_FILENAME)
        elif base_path.name == TEST_FILENAME:
            test_path = base_path
            train_path = base_path.with_name(TRAIN_FILENAME)
        else:
            raise FileNotFoundError(
                f"Expected either {TRAIN_FILENAME} or {TEST_FILENAME}, received {base_path.name}."
            )
    else:
        raise FileNotFoundError(f"Census Income path does not exist: {base_path}")

    train_df = _read_split(train_path)
    test_df = _read_split(test_path)

    df = pd.concat([train_df, test_df], ignore_index=True)
    df = _normalize_strings(df)
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.replace(".", "", regex=False).str.strip()
    df = df.drop_duplicates().reset_index(drop=True)

    target = (df[TARGET_COLUMN] == POSITIVE_LABEL).astype(int)
    target.name = TARGET_NAME
    features = df.drop(columns=[TARGET_COLUMN, INSTANCE_WEIGHT_COLUMN]).reset_index(drop=True)

    for col in NUMERIC_COLUMNS:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce")
    features["year"] = pd.to_numeric(features["year"], errors="coerce").astype(int)

    metadata = {
        "task": "binary_classification",
        "positive_class": "income >= $50K",
        "target_definition": "is_high_income = 1 when total person income >= $50K, otherwise 0.",
        "total_rows": str(len(features)),
        "positive_rate": f"{target.mean():.4f}",
        "source_url": SOURCE_URL,
        "timestamp_column": "year",
    }

    return features, target, metadata


register_dataset(
    DatasetMetadata(
        name=DATASET_NAME,
        loader=load_census_income,
        default_path=DEFAULT_DATA_DIR,
        description="KDD Cup (Census Income) dataset with demographic and employment attributes.",
        target=TARGET_NAME,
        positive_class=1,
        source_url=SOURCE_URL,
        timestamp_col="year",
    )
)
