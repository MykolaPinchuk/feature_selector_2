# Feature Selection Framework for XGBoost

This repository hosts a reusable Python codebase for permutation- and SHAP-based feature selection tailored to XGBoost classifiers. The framework follows the Product Requirements Document in `prd.md` and currently focuses on the first dataset in the roadmap: **Factors Affecting University Student Grades**.

## Getting Started

1. **Install dependencies**

   ```bash
   pip install -e .
   ```

2. **Download the student grades dataset**

   The Kaggle dataset is mirrored on GitHub, which allows automated downloads without credentials. Run:

   ```bash
   mkdir -p data/raw
   curl -L -o data/raw/factors_affecting_university_student_grades.csv \
     https://raw.githubusercontent.com/J-Disara/Data-Warehousing-Data-Mining-Project/main/Factors_%20Affecting_%20University_Student_Grades_Dataset.csv
   ```

   The loader expects the CSV at `data/raw/factors_affecting_university_student_grades.csv`. Adjust the path via `fs_xgb.data.load_dataset(..., data_path=Path(...))` if you store it elsewhere.

3. **Quick smoke test**

   ```bash
   python - <<'PY'
   from fs_xgb.data import load_dataset
   from fs_xgb.preprocessing import FeatureEngineer, FeatureEngineerConfig, TargetEncodingConfig
   from fs_xgb.splitting import RandomSplitConfig, create_random_splits

   X, y, info = load_dataset("student_grades")
   print(f"Rows: {len(X)}, Positives: {int(y.sum())}")
   splits = create_random_splits(X, y, RandomSplitConfig())
   print(f"Train shape: {splits.X_train.shape}, Test shape: {splits.X_test.shape}")

   fe_config = FeatureEngineerConfig(
       target_encoding=TargetEncodingConfig(n_splits=5, smoothing=10.0, min_category_size=25)
   )
   engineer = FeatureEngineer(fe_config)
   X_train_enc = engineer.fit_transform(splits.X_train, splits.y_train)
   X_val_enc = engineer.transform(splits.X_val)
   print(f"Encoded train dtypes: {X_train_enc.dtypes.unique()}")
   PY
   ```

## Feature Engineering Strategy

- **Binary categorical columns** (exactly two categories) become numeric indicator features using a binary encoder. The encoder chooses the category with higher target mean as `1` and applies a configurable fill value (default `0.5`) for missing data.
- **Categoricals with >2 categories** are fed through an **out-of-fold target encoder** with:
  - Stratified K-fold (default 5) to avoid leakage.
  - Smoothing toward the global target rate.
  - A minimum category frequency threshold to default rare categories back to the global prior.

All encoders live under `fs_xgb.preprocessing` and can be composed via `FeatureEngineer` for TRAIN/VAL/TEST splits.

## Running Experiments

Use the CLI to run the full pipeline (feature engineering → FS → baseline comparisons). Results are written to `results/<dataset>/<timestamp>/` and include metrics, feature importances, and the resolved configuration.

```bash
python -m fs_xgb.cli.main \
  --config fs_xgb/experiments/configs/student_grades.yaml \
  --results-dir results
```

After the run completes you can inspect:

- `metrics.json`: PR-AUC/ROC-AUC on train/val/test for each model variant plus which one satisfied the tolerance criterion.
- `permutation_fi.csv`: ΔPR-AUC per permuted feature aggregated across FS models.
- `shap_importance.csv`: Mean |SHAP| values used for permutation triage.
- `features.json`: Lists of kept vs dropped features.
- `config.yaml`: The resolved configuration (defaults + overrides) used for the run.

## Project Layout

```
fs_xgb/
  config/default_config.yaml    # Baseline config (dataset + FS knobs)
  data/                         # Dataset registry and loaders
  splitting/                    # Train/val/test split utilities
data/raw/                       # (gitignored) source CSVs
logs/agents/                    # Agent run logs (see below)
prd.md                          # Product requirements document
```

The next steps in the roadmap are to flesh out the FS models, SHAP triage, permutation FI, and CLI surfaces described in `prd.md`.

## Agent Logs

Per the repo instructions, each automation run should leave a short log under `logs/agents/`. The log files record the agent/model used and summarize key actions for traceability.
