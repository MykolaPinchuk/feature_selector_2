"""Encoding primitives for categorical features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass
class TargetEncodingConfig:
    """Configuration for the target encoder."""

    n_splits: int = 5
    smoothing: float = 10.0
    min_category_size: int = 30
    random_state: int = 42


class BinaryCategoryEncoder:
    """Encode binary categorical columns into 0/1 features."""

    def __init__(self, missing_value: float = 0.5) -> None:
        self.missing_value = missing_value
        self.mapping: Dict[Optional[str], float] = {}
        self.positive_category: Optional[str] = None
        self.negative_category: Optional[str] = None

    def fit(self, series: pd.Series, target: pd.Series) -> "BinaryCategoryEncoder":
        data = pd.DataFrame({"feature": series, "target": target})
        observed = data.dropna(subset=["feature"])
        categories = observed["feature"].unique()
        if len(categories) != 2:
            raise ValueError(f"Expected exactly 2 categories, found {len(categories)} for '{series.name}'.")

        means = observed.groupby("feature")["target"].mean()
        self.positive_category = means.idxmax()
        neg_candidates = [cat for cat in categories if cat != self.positive_category]
        self.negative_category = neg_candidates[0]
        self.mapping = {self.negative_category: 0.0, self.positive_category: 1.0}
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        encoded = series.map(self.mapping)
        return encoded.fillna(self.missing_value)

    def fit_transform(self, series: pd.Series, target: pd.Series) -> pd.Series:
        return self.fit(series, target).transform(series)


class TargetEncoder:
    """Out-of-fold target encoder with smoothing and rarity thresholding."""

    def __init__(self, config: TargetEncodingConfig) -> None:
        self.config = config
        self.global_mean: float = 0.0
        self.mapping: Dict[str, float] = {}

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        return series.fillna("__MISSING__").astype(str)

    def _build_mapping(self, series: pd.Series, target: pd.Series) -> Dict[str, float]:
        combined = pd.DataFrame({"feature": self._normalize(series), "target": target})
        grouped = combined.groupby("feature")["target"].agg(["mean", "count"])
        lambda_ = grouped["count"] / (grouped["count"] + self.config.smoothing)
        encoding = lambda_ * grouped["mean"] + (1 - lambda_) * self.global_mean
        encoding[grouped["count"] < self.config.min_category_size] = self.global_mean
        return encoding.to_dict()

    def fit_transform(self, series: pd.Series, target: pd.Series) -> pd.Series:
        if self.config.n_splits < 2:
            raise ValueError("TargetEncoder requires at least 2 folds.")

        normalized = self._normalize(series)
        target = target.reset_index(drop=True)
        normalized = normalized.reset_index(drop=True)

        self.global_mean = float(target.mean())
        encoded = pd.Series(index=normalized.index, dtype=float)

        splitter = StratifiedKFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )

        for train_idx, valid_idx in splitter.split(normalized, target):
            train_series = normalized.iloc[train_idx]
            train_target = target.iloc[train_idx]
            mapping = self._build_mapping(train_series, train_target)
            valid_series = normalized.iloc[valid_idx]
            encoded.iloc[valid_idx] = valid_series.map(mapping).fillna(self.global_mean)

        self.mapping = self._build_mapping(normalized, target)
        return encoded

    def transform(self, series: pd.Series) -> pd.Series:
        normalized = self._normalize(series)
        return normalized.map(self.mapping).fillna(self.global_mean)
