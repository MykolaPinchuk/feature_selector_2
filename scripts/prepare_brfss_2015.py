"""Utility script to download/prepare the 2015 BRFSS dataset."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


try:  # pragma: no cover - optional import guard
    import pyreadstat
except ImportError as exc:  # pragma: no cover - simple runtime protection
    raise ImportError(
        "pyreadstat is required to prepare BRFSS data. Install it via `pip install pyreadstat`."
    ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BRFSS 2015 XPT data to Parquet + metadata.")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to LLCP2015.XPT (or a directory containing it).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/brfss_2015"),
        help="Directory where processed files will be written.",
    )
    parser.add_argument(
        "--sas-labels",
        type=Path,
        default=None,
        help="Optional path to SASOUT15_LLCP.SAS for richer variable descriptions.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=60000,
        help="Number of rows to include in the lightweight sample Parquet (0 to skip).",
    )
    parser.add_argument(
        "--preview-csv",
        type=int,
        default=1000,
        help="If >0, write the first N rows to brfss_2015_sample.csv for quick inspection.",
    )
    return parser.parse_args()


def _resolve_source(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [path / "LLCP2015.XPT", path / "brfss_2015.parquet"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate BRFSS source file under {path}.")


def _parse_sas_labels(path: Optional[Path]) -> Dict[str, str]:
    if not path or not path.exists():
        return {}
    pattern = re.compile(r"^(\w+)\s*=\s*['\"](.+?)['\"]")
    labels: Dict[str, str] = {}
    with path.open("r", encoding="latin-1") as fh:
        for line in fh:
            match = pattern.match(line.strip())
            if match:
                column = match.group(1).upper()
                labels[column] = match.group(2)
    return labels


def main() -> None:
    args = _parse_args()
    source_path = _resolve_source(args.source)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if source_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(source_path)
        meta = None
    else:
        df, meta = pyreadstat.read_xport(source_path)

    parquet_path = output_dir / "brfss_2015.parquet"
    print(f"Writing full dataset to {parquet_path} ...")
    df.to_parquet(parquet_path, index=False)

    if args.sample_rows and args.sample_rows > 0:
        sample_path = output_dir / "brfss_2015_sample.parquet"
        print(f"Writing sample Parquet ({args.sample_rows} rows) to {sample_path} ...")
        df.head(args.sample_rows).to_parquet(sample_path, index=False)

    if args.preview_csv and args.preview_csv > 0:
        csv_path = output_dir / "brfss_2015_sample.csv"
        print(f"Writing preview CSV ({args.preview_csv} rows) to {csv_path} ...")
        df.head(args.preview_csv).to_csv(csv_path, index=False)

    labels = {col: "" for col in df.columns}
    if meta is not None:
        for name, label in zip(meta.column_names, meta.column_labels):
            labels[name] = label or labels.get(name, "")
    sas_labels = _parse_sas_labels(args.sas_labels)
    for column, description in sas_labels.items():
        labels[column] = description

    info = pd.DataFrame(
        {
            "column": df.columns,
            "description": [labels.get(col, "") for col in df.columns],
            "dtype": [str(df[col].dtype) for col in df.columns],
            "non_null_count": [df[col].notna().sum() for col in df.columns],
        }
    )
    descriptor_path = output_dir / "variable_descriptions.csv"
    print(f"Writing variable descriptions to {descriptor_path} ...")
    info.to_csv(descriptor_path, index=False)

    print("\nSummary")
    print("=" * 72)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    positives = int((df["DIABETE3"] == 1.0).sum()) if "DIABETE3" in df.columns else 0
    print(f"Reported diabetes cases (DIABETE3 == 1): {positives:,}")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
