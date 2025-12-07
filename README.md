# Feature Selection Framework for XGBoost

This repository hosts a reusable Python codebase for permutation- and SHAP-based feature selection tailored to XGBoost classifiers. The framework follows the Product Requirements Document in `prd.md` and currently focuses on the first dataset in the roadmap: **Factors Affecting University Student Grades**.

## Getting Started

1. **Install dependencies**

   ```bash
   pip install -e .
   ```

2. **Download datasets**

   _Student grades (Factors Affecting University Student Grades)_:

   ```bash
   mkdir -p data/raw
   curl -L -o data/raw/factors_affecting_university_student_grades.csv \
     https://raw.githubusercontent.com/J-Disara/Data-Warehousing-Data-Mining-Project/main/Factors_%20Affecting_%20University_Student_Grades_Dataset.csv
   ```

   The loader expects the CSV at `data/raw/factors_affecting_university_student_grades.csv`. Adjust the path via `fs_xgb.data.load_dataset(..., data_path=Path(...))` if you store it elsewhere.

   _Census Income (KDD)_:

   ```bash
   mkdir -p data/raw/census_income_kdd
   curl -L -o data/raw/census_income_kdd.zip \
     https://archive.ics.uci.edu/static/public/117/census%2Bincome%2Bkdd.zip
   unzip -o data/raw/census_income_kdd.zip -d data/raw/census_income_kdd
   tar -xzf data/raw/census_income_kdd/census.tar.gz -C data/raw/census_income_kdd
   ```

   The loader looks for `census-income.data` and `census-income.test` under `data/raw/census_income_kdd/`. Feel free to point the loader to another directory via the `data_path` argument.

   _Balanced subsample (1:1 class ratio, used for Iterations 4–5)_:

   ```bash
   python scripts/create_census_balanced_sample.py \
     --source data/raw/census_income_kdd \
     --output data/raw/census_income_kdd_balanced
   ```

   All Census configs set `splits.strategy: "chronological"`, which means TRAIN consumes the earliest year (1994) and VAL/TEST begin with 1995 to enforce out-of-time evaluation.

   _BRFSS 2015 (CDC)_:

   ```bash
   mkdir -p data/raw/brfss_2015
   curl -L -o data/raw/brfss_2015/LLCP2015XPT.zip \
     https://www.cdc.gov/brfss/annual_data/2015/files/LLCP2015XPT.zip
   unzip -o data/raw/brfss_2015/LLCP2015XPT.zip -d data/raw/brfss_2015
   # Optional but recommended for human-friendly column descriptions
   curl -L -o data/raw/brfss_2015/SASOUT15_LLCP.SAS \
     https://www.cdc.gov/brfss/annual_data/2015/files/SASOUT15_LLCP.SAS

   python scripts/prepare_brfss_2015.py \
     --source data/raw/brfss_2015/LLCP2015.XPT \
     --sas-labels data/raw/brfss_2015/SASOUT15_LLCP.SAS \
     --output data/raw/brfss_2015 \
     --sample-rows 60000
   ```

   The prep script emits `brfss_2015.parquet` (full dataset), `brfss_2015_sample.parquet` (faster experiments), and a `variable_descriptions.csv` that combines XPT metadata with SAS label statements. The loader automatically downsamples the majority class to a 50/50 ratio and exposes an interview-timestamp column so configs can request chronological splits.

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

  Set `TargetEncodingConfig.strategy` to `"naive"` to disable smoothing and frequency thresholds, which intentionally overfits high-cardinality columns so the FS pipeline can stress-test its ability to drop them. Keeping the default `"smoothed"` mode preserves the original regularized behavior.

  Regardless of strategy, the encoder uses out-of-fold averages during `fit_transform`, so validation rows never see their own labels—this prevents leakage even when naive encoding is requested.

All encoders live under `fs_xgb.preprocessing` and can be composed via `FeatureEngineer` for TRAIN/VAL/TEST splits.

## Running Experiments

Use the CLI to run the full pipeline (feature engineering → FS → baseline comparisons). Results are written to `results/<dataset>/<timestamp>/` and include metrics, feature importances, and the resolved configuration.

**Profile loop (aggressive/moderate/mild):**

```bash
python -m fs_xgb.cli.main \
  --config fs_xgb/experiments/configs/student_grades.yaml \
  --results-dir results
```

**Census Income profile loop:**

```bash
python -m fs_xgb.cli.main \
  --config fs_xgb/experiments/configs/census_income.yaml \
  --results-dir results
```

**Census Income balanced subsample:**

```bash
python -m fs_xgb.cli.main \
  --config fs_xgb/experiments/configs/census_income_balanced.yaml \
  --results-dir results
```

**BRFSS 2015 (full dataset):**

```bash
python -m fs_xgb.cli.main \
  --config fs_xgb/experiments/configs/brfss_2015.yaml \
  --results-dir results
```

**BRFSS 2015 sample (fast smoke test):**

```bash
python -m fs_xgb.cli.main \
  --config fs_xgb/experiments/configs/brfss_2015_sample.yaml \
  --results-dir results
```

**Frontier sweep (reuse SHAP/permutation pass across many threshold combos):**

```bash
python -m fs_xgb.cli.main \
  --config fs_xgb/experiments/configs/student_grades_frontier.yaml \
  --results-dir results
```

Swap `student_grades_frontier.yaml` with `census_income_frontier.yaml` (full dataset), `census_income_balanced_frontier.yaml` (balanced subset), or `brfss_2015_frontier.yaml` (diabetes target) to run the comprehensive search on those datasets.

After the run completes you can inspect:

- `metrics.json`: PR-AUC/ROC-AUC on train/val/test for each model variant plus which one satisfied the tolerance criterion for the **primary** FS mode (default: moderate).
- `metrics_aggressive.json` / `metrics_mild.json`: Additional summaries for the alternate FS modes.
- `permutation_fi*.csv`, `shap_importance*.csv`, `features*.json`: Mode-specific feature-importance outputs.
- `report.md`: Human-readable summary covering the primary mode, alternate modes, and a comprehensive feature table listing every column’s permutation/SHAP/gain metrics, threshold gaps, and inclusion status.
- `config.yaml`: The resolved configuration (defaults + overrides) used for the run.

A dataset-specific EDA summary is written (or refreshed) at `results/<dataset>/eda.md`, covering target distribution, per-column missingness, and numeric stats. This lets you review the dataset once and keep it alongside experiment outputs.

### Feature-Selection Modes

`fs_xgb/config/default_config.yaml` defines three FS aggressiveness profiles:

1. **Aggressive** – high ΔPR-AUC threshold + drop-all rest policy; expects substantial feature pruning.
2. **Moderate** – balanced thresholds; this is the `fs_primary_mode` that drives final model selection/reporting.
3. **Mild** – low thresholds to retain most features.

Each run loops through all configured modes, producing feature lists and final-model metrics for each. Tweak `fs_modes` in your experiment config to experiment with custom thresholds or rest policies.

### Overfit Filters (SHAP & Gain)

Permutation ΔPR-AUC reflects generalization utility while SHAP and XGBoost gain highlight model reliance. Starting in Iteration 3 you can enable `fs.overfit_filter` (SHAP-focused) and `fs.gain_overfit_filter` to automatically penalize features that rank highly by SHAP/gain but fail to register meaningful permutation gains. Configure the rank windows, maximum acceptable ΔPR-AUC (and optional noise constraints), and whether each filter should hard-drop or simply demote those columns back into the "rest" bucket. The logic runs before final-model training and is compatible with both the fast profile loop and the comprehensive frontier search.

### Comprehensive Frontier Search (opt-in)

Set `fs_search.mode: "frontier"` plus the desired sweep values under `fs_search.frontier` to reuse one SHAP/permutation pass while evaluating multiple threshold combinations. The framework deduplicates feature sets, scores each candidate on the outer VAL/TEST splits, and picks the smallest set whose validation PR-AUC is within tolerance of the best-performing candidate. The run directory will include `frontier_candidates.json` summarizing every evaluated point on the frontier.
## Project Layout

```
fs_xgb/
  config/default_config.yaml    # Baseline config (dataset + FS knobs)
  data/                         # Dataset registry and loaders
  splitting/                    # Train/val/test split utilities
data/raw/                       # (gitignored) source CSVs
logs/agents/                    # Agent run logs (see below)
docs/iteration_plan.md          # Iteration roadmap and future FS enhancements
prd.md                          # Product requirements document
```

The next steps in the roadmap are to flesh out the FS models, SHAP triage, permutation FI, and CLI surfaces described in `prd.md`.

## Agent Logs

Per the repo instructions, each automation run should leave a short log under `logs/agents/`. The log files record the agent/model used and summarize key actions for traceability.
