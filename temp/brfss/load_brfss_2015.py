"""
Script to load 2015 BRFSS dataset and extract variable descriptions.
"""

import pandas as pd
import pyreadstat
import re
import os

# Load the XPT file with metadata
print("Loading 2015 BRFSS data from XPT file...")
df, meta = pyreadstat.read_xport("/home/mykola/repos/temp_brfss/LLCP2015.XPT")

print(f"\nDataset loaded successfully!")
print(f"Number of records: {len(df):,}")
print(f"Number of variables: {len(df.columns)}")

# Extract variable labels from metadata
print("\n" + "="*80)
print("VARIABLE LABELS FROM XPT FILE METADATA")
print("="*80)

# Create a dictionary of variable labels
variable_labels = {}
for col in df.columns:
    label = meta.column_labels[meta.column_names.index(col)] if col in meta.column_names else ""
    variable_labels[col] = label

# Parse SAS file for additional variable labels
print("\nParsing SAS file for additional variable descriptions...")
sas_labels = {}
with open("/home/mykola/repos/temp_brfss/SASOUT15_LLCP.SAS", "r", encoding="latin-1") as f:
    sas_content = f.read()
    
    # Find LABEL statements
    label_pattern = r"^\s*(\w+)\s*=\s*['\"](.+?)['\"]"
    for line in sas_content.split("\n"):
        match = re.match(label_pattern, line.strip())
        if match:
            var_name = match.group(1).upper()
            var_label = match.group(2)
            sas_labels[var_name] = var_label

# Combine labels (prefer SAS labels as they're often more complete)
print("\nCombining variable descriptions...")
all_labels = {}
for col in df.columns:
    xpt_label = variable_labels.get(col, "")
    sas_label = sas_labels.get(col, "")
    # Use SAS label if available and non-empty, otherwise use XPT label
    all_labels[col] = sas_label if sas_label else xpt_label

# Create a DataFrame with variable information
var_info = pd.DataFrame({
    "Variable Name": df.columns,
    "Description": [all_labels.get(col, "") for col in df.columns],
    "Data Type": [str(df[col].dtype) for col in df.columns],
    "Non-Null Count": [df[col].notna().sum() for col in df.columns],
    "Sample Values": [str(df[col].dropna().head(3).tolist()) for col in df.columns]
})

# Save variable descriptions to CSV
var_info.to_csv("/home/mykola/repos/temp_brfss/variable_descriptions.csv", index=False)
print(f"\nVariable descriptions saved to: variable_descriptions.csv")

# Save the dataset to a more efficient format (parquet)
print("\nSaving dataset to Parquet format for faster loading...")
df.to_parquet("/home/mykola/repos/temp_brfss/brfss_2015.parquet", index=False)
print("Dataset saved to: brfss_2015.parquet")

# Also save as CSV (optional, but useful for inspection)
print("\nSaving first 1000 rows to CSV for quick inspection...")
df.head(1000).to_csv("/home/mykola/repos/temp_brfss/brfss_2015_sample.csv", index=False)
print("Sample saved to: brfss_2015_sample.csv")

# Display summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total variables: {len(df.columns)}")
print(f"Total records: {len(df):,}")
print(f"\nFiles created:")
print(f"  - brfss_2015.parquet (full dataset)")
print(f"  - brfss_2015_sample.csv (first 1000 rows)")
print(f"  - variable_descriptions.csv (all variable names and descriptions)")

# Display first 50 variables with descriptions
print("\n" + "="*80)
print("FIRST 50 VARIABLES WITH DESCRIPTIONS")
print("="*80)
for i, row in var_info.head(50).iterrows():
    print(f"{row['Variable Name']:15} | {row['Description'][:60] if row['Description'] else 'No description'}")

# Display diabetes-related variables
print("\n" + "="*80)
print("DIABETES-RELATED VARIABLES")
print("="*80)
diabetes_vars = var_info[var_info['Variable Name'].str.contains('DIAB|BLDSUGAR|INSULIN|PREDIAB', case=False)]
for i, row in diabetes_vars.iterrows():
    print(f"{row['Variable Name']:15} | {row['Description'][:60] if row['Description'] else 'No description'}")

print("\n" + "="*80)
print("DONE!")
print("="*80)
