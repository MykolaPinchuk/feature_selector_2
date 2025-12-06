"""Random splitting strategy used when no timestamp is available."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class RandomSplitConfig:
    """Configuration for random stratified splits."""

    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42


@dataclass
class RandomDatasetSplits:
    """Container holding TRAIN/VAL/TEST matrices."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def create_random_splits(X: pd.DataFrame, y: pd.Series, config: RandomSplitConfig) -> RandomDatasetSplits:
    """Generate random TRAIN/VAL/TEST splits while preserving class ratios."""

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )

    val_fraction = config.val_size / (1.0 - config.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_fraction,
        stratify=y_temp,
        random_state=config.random_state,
    )

    return RandomDatasetSplits(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )
