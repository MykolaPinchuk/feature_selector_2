"""Feature engineering pipeline that applies categorical encodings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
from pandas.api.types import is_categorical_dtype, is_object_dtype

from .encoding import BinaryCategoryEncoder, TargetEncoder, TargetEncodingConfig


@dataclass
class FeatureEngineerConfig:
    """Configuration for the feature engineering pipeline."""

    categorical_columns: Optional[List[str]] = None
    target_encoding: TargetEncodingConfig = field(default_factory=TargetEncodingConfig)
    binary_missing_value: float = 0.5


class FeatureEngineer:
    """Apply binary and target encoding to categorical columns."""

    def __init__(self, config: FeatureEngineerConfig | None = None) -> None:
        self.config = config or FeatureEngineerConfig()
        self.binary_encoders: Dict[str, BinaryCategoryEncoder] = {}
        self.target_encoders: Dict[str, TargetEncoder] = {}
        self.categorical_columns: List[str] = []
        self.constant_fills: Dict[str, float] = {}

    def _select_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        if self.config.categorical_columns is not None:
            return self.config.categorical_columns
        return [
            col
            for col in df.columns
            if is_object_dtype(df[col]) or is_categorical_dtype(df[col])
        ]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        X_transformed = X.copy()
        self.categorical_columns = self._select_categorical_columns(X)
        for col in self.categorical_columns:
            unique_values = X[col].dropna().unique()
            n_unique = len(unique_values)
            if n_unique == 0:
                fill_value = self.config.binary_missing_value
                X_transformed[col] = fill_value
                self.constant_fills[col] = fill_value
            elif n_unique == 1:
                X_transformed[col] = 0.0
                self.constant_fills[col] = 0.0
            elif n_unique == 2:
                encoder = BinaryCategoryEncoder(missing_value=self.config.binary_missing_value)
                X_transformed[col] = encoder.fit_transform(X[col], y)
                self.binary_encoders[col] = encoder
            else:
                encoder = TargetEncoder(config=self.config.target_encoding)
                X_transformed[col] = encoder.fit_transform(X[col], y)
                self.target_encoders[col] = encoder
        return X_transformed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()
        for col, encoder in self.binary_encoders.items():
            X_transformed[col] = encoder.transform(X[col])
        for col, encoder in self.target_encoders.items():
            X_transformed[col] = encoder.transform(X[col])
        for col, fill_value in self.constant_fills.items():
            if col in X_transformed.columns:
                X_transformed[col] = fill_value
        return X_transformed
