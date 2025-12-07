"""Loader for the 2015 BRFSS (CDC) dataset."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..datasets_registry import DatasetMetadata, register_dataset


DATASET_NAME = "brfss_2015"
SOURCE_URL = "https://www.cdc.gov/brfss/annual_data/annual_2015.html"
TARGET_COLUMN = "DIABETE3"
TARGET_NAME = "is_diabetic"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "brfss_2015"
PARQUET_FILENAME = "brfss_2015.parquet"
SAMPLE_FILENAME = "brfss_2015_sample.parquet"
XPT_FILENAME = "LLCP2015.XPT"
ZIP_FILENAME = "LLCP2015XPT.zip"

TARGET_MAPPING = {
    1.0: 1,  # Diagnosed with diabetes
    2.0: 0,  # Gestational diabetes only
    3.0: 0,  # No
    4.0: 0,  # No, borderline or pre-diabetic
}

# Common sentinel values used throughout BRFSS to indicate non-response/unknown.
MISSING_CODE_BASE = [7, 8, 9, 77, 88, 99, 777, 888, 999, 7777, 8888, 9999, 77777, 88888, 99999]
MISSING_CODES = {float(code) for code in MISSING_CODE_BASE}
LOW_CARDINALITY_MAX_UNIQUE = 15
HIGH_MISSING_THRESHOLD = 0.98
DROP_COLUMNS = {"SEQNO"}
LEAKY_SUBSTRINGS = ("DIAB", "INSUL", "PREDIAB")
TIMESTAMP_COLUMN = "interview_timestamp"
RANDOM_STATE = 42


def _extract_xpt_from_zip(zip_path: Path) -> Path:
    base_dir = zip_path.parent
    with zipfile.ZipFile(zip_path) as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(".xpt")]
        if not members:
            raise FileNotFoundError(f"No XPT files found inside {zip_path}.")
        member = members[0]
        target_name = Path(member).name
        extracted_path = base_dir / target_name
        zf.extract(member, path=base_dir)
        cleaned_path = base_dir / XPT_FILENAME
        if extracted_path != cleaned_path:
            if cleaned_path.exists():
                cleaned_path.unlink()
            extracted_path.rename(cleaned_path)
        return cleaned_path


def _load_from_path(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".xpt":
        try:
            import pyreadstat
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "pyreadstat is required to read .XPT files. Install it via `pip install pyreadstat`."
            ) from exc
        df, _ = pyreadstat.read_xport(path)
        return df
    raise FileNotFoundError(f"Unsupported BRFSS file format: {path}")


def _resolve_dataframe(data_path: Path) -> Tuple[pd.DataFrame, Path]:
    if data_path.is_file():
        return _load_from_path(data_path), data_path.parent

    base_dir = data_path
    candidates = [base_dir / PARQUET_FILENAME, base_dir / SAMPLE_FILENAME, base_dir / XPT_FILENAME]
    for candidate in candidates:
        if candidate.exists():
            return _load_from_path(candidate), base_dir

    zip_candidate = base_dir / ZIP_FILENAME
    if zip_candidate.exists():
        xpt_path = _extract_xpt_from_zip(zip_candidate)
        return _load_from_path(xpt_path), base_dir

    raise FileNotFoundError(
        f"Could not locate BRFSS data under {data_path}. Expected a directory containing "
        f"{PARQUET_FILENAME}, {XPT_FILENAME}, or {ZIP_FILENAME}."
    )


def _replace_missing_codes(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        series = df[col]
        mask = series.isin(MISSING_CODES)
        if mask.any():
            df[col] = series.mask(mask)
    return df


def _drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    upper = {col: col.upper() for col in df.columns}
    to_drop = [
        col
        for col, upper_name in upper.items()
        if col != TARGET_COLUMN and any(token in upper_name for token in LEAKY_SUBSTRINGS)
    ]
    if to_drop:
        df = df.drop(columns=to_drop)
    return df


def _is_integral(series: pd.Series) -> bool:
    values = series.to_numpy()
    if values.size == 0:
        return True
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return True
    return np.all(np.isclose(finite, np.round(finite)))


def _convert_low_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        series = df[col]
        non_null = series.dropna()
        unique_count = non_null.nunique()
        if 2 <= unique_count <= LOW_CARDINALITY_MAX_UNIQUE and _is_integral(non_null):
            int_series = series.round().astype("Int64")
            string_values = int_series.astype(str).replace({"<NA>": pd.NA})
            categorical = pd.Categorical(string_values)
            if "__MISSING__" not in categorical.categories:
                categorical = categorical.add_categories(["__MISSING__"])
            df[col] = categorical
    return df


def _build_timestamp_features(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    if "IDATE" in df.columns:
        date_series = pd.to_datetime(df["IDATE"], format="%m%d%Y", errors="coerce")
    else:
        date_series = pd.Series(pd.NaT, index=df.index)

    missing_mask = date_series.isna()
    if missing_mask.any():
        month = pd.to_numeric(df.get("IMONTH"), errors="coerce").astype("Int64")
        day = pd.to_numeric(df.get("IDAY"), errors="coerce").astype("Int64")
        year = pd.to_numeric(df.get("IYEAR"), errors="coerce").astype("Int64")
        fallback = pd.to_datetime(
            month.astype(str).str.zfill(2)
            + day.astype(str).str.zfill(2)
            + year.astype(str),
            format="%m%d%Y",
            errors="coerce",
        )
        date_series = date_series.fillna(fallback)
    else:
        fallback = None

    min_date = date_series.min()
    if pd.isna(min_date):
        min_date = pd.Timestamp("2015-01-01")
    date_series = date_series.fillna(min_date)
    timestamp = (date_series.astype("int64") // 86_400_000_000_000).astype("Int64")
    return date_series, timestamp


def _downsample_majority(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    positives = y[y == 1].index
    negatives = y[y == 0].index
    if len(negatives) <= len(positives):
        return X.reset_index(drop=True), y.reset_index(drop=True)
    rng = np.random.default_rng(RANDOM_STATE)
    sampled_neg = rng.choice(negatives.to_numpy(), size=len(positives), replace=False)
    keep_idx = np.concatenate([positives.to_numpy(), sampled_neg])
    keep_idx.sort()
    return X.loc[keep_idx].reset_index(drop=True), y.loc[keep_idx].reset_index(drop=True)


def _drop_high_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing_fraction = df.isna().mean()
    to_drop = missing_fraction[missing_fraction >= HIGH_MISSING_THRESHOLD].index
    if len(to_drop) > 0:
        df = df.drop(columns=list(to_drop))
    return df


def load_brfss_2015(data_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
    base_path = Path(data_path) if data_path else DEFAULT_DATA_DIR
    raw_df, resolved_dir = _resolve_dataframe(base_path)

    # Persist an on-the-fly Parquet cache for faster reloads when starting from XPT.
    parquet_path = resolved_dir / PARQUET_FILENAME
    if not parquet_path.exists():  # pragma: no cover - cache helper
        try:
            raw_df.to_parquet(parquet_path, index=False)
        except Exception:
            pass

    df = raw_df.copy()
    df.columns = [col.strip() for col in df.columns]
    for column in DROP_COLUMNS.intersection(df.columns):
        df = df.drop(columns=column)

    df = _drop_leaky_columns(df)

    _, interview_ts = _build_timestamp_features(df)
    df[TIMESTAMP_COLUMN] = interview_ts

    target_series = df[TARGET_COLUMN]
    valid_mask = target_series.isin(TARGET_MAPPING.keys())
    df = df.loc[valid_mask].reset_index(drop=True)
    y = target_series.loc[valid_mask].map(TARGET_MAPPING).astype(int).reset_index(drop=True)

    X = df.drop(columns=[TARGET_COLUMN])
    X = _replace_missing_codes(X)
    X = _drop_high_missing(X)
    X = _convert_low_cardinality(X)
    X[TIMESTAMP_COLUMN] = X[TIMESTAMP_COLUMN].astype("float64")

    X, y = _downsample_majority(X, y)

    metadata = {
        "task": "binary_classification",
        "positive_class": "Respondent reported a diabetes diagnosis (DIABETE3 == 1)",
        "target_definition": "is_diabetic = 1 for DIABETE3 == 1, 0 otherwise (excluding unknown/refused)",
        "total_rows": str(len(X)),
        "positive_rate": f"{y.mean():.4f}",
        "source_url": SOURCE_URL,
        "description": "CDC Behavioral Risk Factor Surveillance System (2015) with 330 health & lifestyle variables.",
        "timestamp_column": TIMESTAMP_COLUMN,
    }

    return X, y, metadata


register_dataset(
    DatasetMetadata(
        name=DATASET_NAME,
        loader=load_brfss_2015,
        default_path=DEFAULT_DATA_DIR,
        description="CDC BRFSS 2015 survey (330 features) predicting diabetes diagnosis.",
        target=TARGET_NAME,
        positive_class=1,
        source_url=SOURCE_URL,
    )
)
