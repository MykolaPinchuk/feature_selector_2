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
class ChronoSplitConfig:
    """Configuration for chronological (time-ordered) splits."""

    test_size: float = 0.2
    val_size: float = 0.2
    timestamp_column: str = "timestamp"
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


def create_chronological_splits(X: pd.DataFrame, y: pd.Series, config: ChronoSplitConfig) -> RandomDatasetSplits:
    """Generate TRAIN/VAL/TEST splits ordered by timestamp."""

    if config.timestamp_column not in X.columns:
        raise ValueError(f"Timestamp column '{config.timestamp_column}' not found in feature matrix.")

    combined = X.copy()
    combined["_target_"] = y.values
    combined = combined.sort_values(by=config.timestamp_column, kind="mergesort").reset_index(drop=True)

    total_rows = len(combined)
    if total_rows == 0:
        raise ValueError("Cannot create splits from an empty dataset.")

    timestamp_values = combined[config.timestamp_column].to_numpy()
    groups: list[tuple[int, int]] = []
    start = 0
    while start < total_rows:
        end = start + 1
        while end < total_rows and timestamp_values[end] == timestamp_values[start]:
            end += 1
        groups.append((start, end))
        start = end

    if len(groups) < 2:
        raise ValueError(
            "Chronological splitting requires at least two distinct timestamp values to form OOT splits."
        )

    target_train_fraction = max(0.0, 1.0 - config.val_size - config.test_size)
    target_train_rows = int(round(total_rows * target_train_fraction))
    min_rest_rows = 2  # Need at least VAL + TEST rows
    train_cut = 0
    assigned = 0
    for idx, (group_start, group_end) in enumerate(groups):
        group_size = group_end - group_start
        # Ensure we leave at least two rows for VAL/TEST
        remaining_rows_after_group = total_rows - group_end
        if remaining_rows_after_group < min_rest_rows:
            break
        train_cut = group_end
        assigned += group_size
        # Stop once we meet the desired train size while still leaving future timestamps
        if assigned >= target_train_rows:
            # If there are remaining groups beyond this one, we can stop.
            if idx < len(groups) - 1:
                break
    if train_cut == 0:
        group_start, group_end = groups[0]
        remaining_rows_after_group = total_rows - group_end
        if remaining_rows_after_group < min_rest_rows:
            raise ValueError("Not enough timestamp diversity to create OOT splits with the requested sizes.")
        train_cut = group_end

    train_df = combined.iloc[:train_cut]
    rest_df = combined.iloc[train_cut:]
    rest_rows = len(rest_df)
    if rest_rows < min_rest_rows:
        raise ValueError("Not enough rows left for validation/test after assigning chronological train split.")

    val_test_total = config.val_size + config.test_size
    if val_test_total <= 0:
        val_fraction = 0.5
    else:
        val_fraction = config.val_size / val_test_total
    val_count = int(round(rest_rows * val_fraction))
    val_count = min(max(val_count, 1), rest_rows - 1)
    test_count = rest_rows - val_count

    val_df = rest_df.iloc[:val_count]
    test_df = rest_df.iloc[val_count:]

    def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        features = df.drop(columns=["_target_"]).reset_index(drop=True)
        target = df["_target_"].reset_index(drop=True)
        return features, target

    X_train, y_train = _split(train_df)
    X_val, y_val = _split(val_df)
    X_test, y_test = _split(test_df)

    return RandomDatasetSplits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
