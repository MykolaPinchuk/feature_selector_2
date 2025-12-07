"""Utility to build a balanced Census Income sample (1:1 class ratio)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

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
    "instance_weight",
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
    "income",
]

TARGET_COLUMN = "income"
POS_LABEL = "50000+"
NEG_LABEL = "- 50000"


def _read_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(
        path,
        names=COLUMN_NAMES,
        header=None,
        skipinitialspace=True,
    )


def _load_combined(source_dir: Path) -> pd.DataFrame:
    train = _read_split(source_dir / "census-income.data")
    test = _read_split(source_dir / "census-income.test")
    df = pd.concat([train, test], ignore_index=True)
    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        df[col] = df[col].str.strip()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.replace(".", "", regex=False).str.strip()
    df = df.dropna(subset=[TARGET_COLUMN])
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def build_balanced_sample(source_dir: Path, output_dir: Path, train_fraction: float = 2 / 3) -> None:
    df = _load_combined(source_dir)
    positives = df[df[TARGET_COLUMN] == POS_LABEL]
    negatives = df[df[TARGET_COLUMN] == NEG_LABEL]
    if positives.empty:
        raise ValueError("No positive class examples found in source data.")
    if len(negatives) < len(positives):
        raise ValueError("Not enough negatives to downsample to a 1:1 ratio.")

    negative_sample = negatives.sample(n=len(positives), random_state=42)
    balanced = pd.concat([positives, negative_sample], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

    split_idx = int(len(balanced) * train_fraction)
    train = balanced.iloc[:split_idx]
    test = balanced.iloc[split_idx:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "census-income.data", index=False, header=False)
    test.to_csv(output_dir / "census-income.test", index=False, header=False)

    print(
        f"Balanced sample written to {output_dir}. "
        f"Train rows: {len(train)}, Test rows: {len(test)}, Positives per split: "
        f"{int(train[train[TARGET_COLUMN] == POS_LABEL].shape[0])}/{int(test[test[TARGET_COLUMN] == POS_LABEL].shape[0])}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a balanced Census Income sample (1:1 class ratio).")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/census_income_kdd"),
        help="Directory containing census-income.data/test files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/census_income_kdd_balanced"),
        help="Destination directory for the balanced sample files.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=2 / 3,
        help="Fraction of rows to place in census-income.data (rest go to .test).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_balanced_sample(args.source, args.output, train_fraction=args.train_fraction)


if __name__ == "__main__":
    main()
