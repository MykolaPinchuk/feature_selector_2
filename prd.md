# PRD: Permutation- and SHAP-based Feature Selection Framework for XGBoost

This is an original PRD for a feature selection framework. It started Iteration 1.

## 1. Overview

### 1.1 Objective

Build a reusable Python codebase that implements a **feature selection (FS) framework for XGBoost classifiers**, suitable for:

- Binary classification problems with mild to strong class imbalance.
- Datasets with **30–500+ candidate features**.
- Settings with or without a **time dimension**, with a strong preference for time-respecting splits when timestamps are available.

The core idea:

- Use **permutation importance on an out-of-time (OOT) validation slice** as the *primary* measure of feature usefulness.
- Use **SHAP** only for:
  - Triaging which features to permute individually vs in bulk.
  - Diagnosing overfitting (gain/SHAP vs permutation discrepancies).
- Avoid full RFE / exhaustive wrappers in favor of a **small number of model fits** and OOT FI.

Metric: **PR-AUC** is the primary optimization and evaluation metric across all experiments.

### 1.2 Non-goals

- Not focused on fraud/credit-specific domain logic; framework must be generic.
- Not building a GUI; CLI + Python API + simple reporting (tables/plots) is sufficient.
- Not implementing full time-series CV; we will use **single inner split** for FS and **single OOT val/test** for final evaluation.

---

## 2. Problem Statement and Context

Tree-based models (XGBoost) handle many irrelevant features reasonably well, but:

- Real-world datasets often have:
  - Potential **leakage** features.
  - **ID-like** or high-cardinality overfitting features.
  - Many **redundant** or weak predictors.
- Naive FS approaches are problematic:
  - **Gain-based FI** is computed on **train** and often promotes overfit features.
  - **SHAP-based FI** measures **model reliance**, not **predictive usefulness on future data**.
  - Full **RFE** with re-training is computationally expensive for large datasets and high-d models.

We need a framework that:

- Is **general** (works across multiple public datasets).
- Uses **OOT performance** as the primary signal for “keep vs drop”.
- Is **computationally tractable** and can be used repeatedly.

---

## 3. Design Principles

1. **OOT performance is the ground truth for usefulness**
   - Feature selection decisions should be driven primarily by **permutation-based PR-AUC drops on an OOT slice**.

2. **Separate “how much the model uses it” from “how useful it is”**
   - Gain/SHAP indicate **usage**, not **benefit**.
   - Permutation FI on OOT indicates **benefit for generalization**.

3. **Time-respecting where possible**
   - When timestamps exist, all splits must preserve temporal order:
     - TRAIN (earlier), VAL (later), TEST (latest).
     - Inner split for FS: TRAIN_FS (earliest part of TRAIN), HOLDOUT_FS (later part of TRAIN).

4. **Compute efficiency over exhaustive search**
   - Use a **small number of FS models** (3–5) on subsampled data.
   - Use **SHAP to triage** features before permutation to limit cost.
   - Do only **2–3 final full-model ablations**, not 100+ RFE steps.

5. **Metric alignment**
   - Use **PR-AUC** for permutation FI and final evaluation to reflect imbalanced-class behavior.
   - For balanced problems, we can additionally log ROC-AUC for interpretability, but PR-AUC remains primary for FS.

6. **Extensibility**
   - Core design should allow:
     - Swapping metrics.
     - Plugging in other tree-based models (LightGBM, CatBoost) later.
     - Adding group-level operations (e.g., correlated clusters) in future versions.

---

## 4. Data Model and Assumptions

### 4.1 Task

- **Binary classification**:
  - Target `y ∈ {0,1}`.
  - Positive class is `1`.
- Multi-class datasets will be used via **one-vs-rest tasks** defined in the dataset harness (e.g., choose a focal class as positive).

### 4.2 Features

- Numeric and categorical features:
  - Categorical handled by either:
    - Pre-encoding (e.g. one-hot, target encoding, ordinal) outside the FS core, or
    - Letting XGBoost handle them if the version supports native categorical splits.
- We assume **no heavy leakage engineering in this framework itself**; leakage detection is mostly based on user-provided metadata.

### 4.3 Splits

Generic split definition:

- `TRAIN`: used to fit FS models and final models.
- `VAL`: OOT validation used for hyperparameter selection and FS confirmation.
- `TEST`: final untouched OOT set.

Inner FS-specific split inside TRAIN (when time dimension available):

- `TRAIN_FS`: earlier part of TRAIN.
- `HOLDOUT_FS`: later part of TRAIN (still before VAL).

If no time dimension:

- Use stratified random splits to define TRAIN/VAL/TEST and TRAIN_FS/HOLDOUT_FS.

---

## 5. High-level FS Algorithm

1. **Static pre-filtering** on TRAIN:
   - Remove obvious leakage, constants, extreme-missing, exact duplicates.

2. **FS model training**:
   - Train 3–5 XGB models on TRAIN_FS (possibly subsampled) with moderate regularization.

3. **SHAP triage on OOT slice (HOLDOUT_FS)**:
   - Compute SHAP on OOT slice.
   - Rank features by mean |SHAP|.
   - Select **TopK** features as candidates for individual permutation.
   - Bottom features are potential drop candidates and/or bulk-handled.

4. **Permutation FI on OOT using PR-AUC**:
   - On a carefully constructed FS_EVAL sample (all positives + sampled negatives), for each FS model:
     - Compute baseline PR-AUC.
     - For each feature in TopK (or TopK∪Middle), permute its values and recompute PR-AUC.
   - Aggregate ΔPR-AUC across FS models for each feature.

5. **Decide keep/drop**:
   - Use ΔPR-AUC distribution to:
     - Identify features with non-trivial positive ΔPR-AUC.
     - Identify features with ΔPR-AUC ≈ 0 (useless).
     - Optionally use train gain vs OOT permutation discrepancies to flag overfit features.

6. **Final ablation models on TRAIN/VAL**:
   - Train 2–3 full models:
     - All features.
     - Filtered feature set.
     - Optionally an aggressive subset.
   - Choose smallest feature set whose VAL PR-AUC is within tolerance of best.

7. **Final model on TRAIN(+VAL) and evaluation on TEST**.

---

## 6. Detailed FS Algorithm Specification

### 6.1 Static Pre-filtering

Input: TRAIN (X_train, y_train) with feature set F_all.

Steps:

1. **Leakage filter**
   - Use user-provided metadata (e.g., `is_leakage=True`, feature lineage, or simple rules) to drop:
     - Post-outcome indicators.
     - Features constructed using future labels or outcomes.
   - Rationale: leakage can dominate FI rankings and break generalization; must be removed before any FS.

2. **Constant / quasi-constant**
   - For each feature j:
     - If variance == 0 on TRAIN → drop.
     - Optionally: if the top category/value covers > 99–99.5% of rows → drop unless explicitly whitelisted.
   - Rationale: these carry no useful information and waste capacity.

3. **Extreme missingness**
   - For each feature j:
     - Compute missing rate on TRAIN.
     - If missing rate > threshold (e.g., 98–99%) and no explicit whitelist → drop.
   - Rationale: extremely sparse features rarely add stable predictive signal.

4. **Exact duplicates**
   - Detect features that are equal on all TRAIN rows (e.g., via column-wise hashing).
   - Keep one representative and drop the rest.
   - Rationale: redundant exact copies confuse FI and waste compute.

Output: cleaned feature set F0.

### 6.2 Inner FS Split

Given TRAIN (with F0),

- If time available:
  - Sort by time.
  - Split TRAIN into TRAIN_FS (earlier) and HOLDOUT_FS (later, e.g. last 20–30% by time).
- Else:
  - Random stratified split of TRAIN into TRAIN_FS and HOLDOUT_FS (preserve class ratio).

Rationale:

- HOLDOUT_FS approximates an OOT slice for measuring FI, without touching the main VAL.

### 6.3 FS Models and Subsampling

For computational efficiency:

1. **Row subsampling for FS models**
   - If TRAIN_FS is very large:
     - Sample N rows (e.g., 100k–300k), potentially time-stratified.
     - Name this TRAIN_FS_SUB.

2. **FS model hyperparameters (XGBClassifier)**
   - Reasonable defaults (configurable):
     - `max_depth`: 4–6
     - `min_child_weight`: 5–20
     - `subsample`: 0.7–0.9
     - `colsample_bytree`: 0.7–1.0
     - `lambda`: 1–5
     - `eta`: 0.05–0.2
     - `n_estimators`: 200–400 (or use early stopping on HOLDOUT_FS)
   - Rationale:
     - Enough capacity to learn signal, but not massive ensembles.
     - Cheap enough to run 3–5 times.

3. **Train M FS models**
   - For m in {1,…,M} (M configurable, default 3–5):
     - Optionally sample another subset of TRAIN_FS_SUB or just change `random_state`.
     - Train XGB on TRAIN_FS_SUB with early stopping on HOLDOUT_FS or a fixed number of rounds.
     - Save model f^(m).

Rationale:

- Multiple seeds/subsamples reduce FI noise and allow stability-based aggregation.

### 6.4 SHAP Triaging on OOT Slice

1. **Construct FS_EVAL**
   - From HOLDOUT_FS:
     - Include **all positives** (y=1).
     - Sample negatives to achieve a target size (e.g., neg:pos ratio 5–10:1).
   - Rationale:
     - Enriches positives so that PR-AUC and ΔPR-AUC are more sensitive to features affecting the minority class.
     - Reduces cost of repeated predictions.

2. **Compute SHAP for each FS model**
   - For each model f^(m):
     - Compute tree SHAP values on FS_EVAL.
     - For each feature j in F0:
       - shap_imp^(m)_j = mean(|SHAP_j(x)|) over x in FS_EVAL.

3. **Aggregate SHAP**
   - mean_shap_j = mean_m shap_imp^(m)_j.
   - Rank features by mean_shap_j descending.

4. **Define permutation candidate set**
   - Choose K (e.g. 60 or configurable).
   - TopK = first K features by mean_shap_j.
   - Rest = remaining features; they are considered low-priority for permutation in v1, but may be retained or dropped based on later heuristics.

Rationale:

- SHAP triage reduces the number of features for expensive per-feature permutation, while still reflecting how the model uses features on an OOT slice.

### 6.5 Permutation FI on OOT using PR-AUC

For each model f^(m):

1. **Baseline PR-AUC**
   - Compute baseline PR-AUC on FS_EVAL: PR_base^(m).

2. **Per-feature permutation for TopK**
   - For each feature j in TopK:
     - Shuffle X_j across rows in FS_EVAL (labels and other features unchanged).
     - Compute PR-perm^(m)(j).
     - Define Δ^(m)_j = PR_base^(m) – PR_perm^(m)(j).

3. **Aggregate across models**
   - For each j in TopK:
     - mean_perm_j = mean_m Δ^(m)_j.
     - std_perm_j = std_m Δ^(m)_j.

4. **Estimate noise band**
   - Use a subset of obviously low-SHAP features (from Rest) as noise reference:
     - For those j, approximate the distribution of Δ^(m)_j.
     - Derive `noise_std` as the standard deviation of Δ for near-zero features.
   - Rationale:
     - Enables statistical thresholding beyond arbitrary absolute cutoffs.

Rationale:

- PR-AUC drop directly measures how much each feature improves ranking quality under the class imbalance structure.
- Aggregating across models and using a noise band mitigates randomness.

### 6.6 Keep/Drop Decision Rules

Define decision rules for TopK features:

1. **Thresholds / criteria**
   - Let Δ_j = mean_perm_j.
   - Let `noise_std` be estimated from near-zero features.
   - Candidate rule:
     - Keep feature j if:
       - Δ_j ≥ Δ_abs_min (configurable, e.g. 0.001 PR-AUC), or
       - Δ_j ≥ k * noise_std (k ≈ 2), or
       - j is in top N_perm by Δ_j (e.g., top 20), or
       - j is on a user-provided “whitelist”.

2. **Feature groups**
   - K_keep = {j ∈ TopK | passes keep rule}.
   - K_drop = TopK \ K_keep.

3. **Handling features in Rest**
   - v1: conservative option:
     - Keep Rest as “neutral” features, or
     - Drop Rest by default if static pre-filter + low SHAP is trusted.
   - The specific policy should be configurable:
     - `rest_policy ∈ {keep_all, drop_all, keep_above_min_shap}`.

Rationale:

- Combines absolute and statistical thresholds.
- Allows rare but consistently useful features to be retained even if Δ_j is small relative to max.

### 6.7 Overfitting Diagnostics (Optional, for Reporting)

For at least one FS model:

1. **Gain FI on TRAIN_FS_SUB**
   - gain_train_j from f^(m).get_score(importance_type="gain").

2. **Rank percentiles**
   - Convert gain, mean_shap, and mean_perm into rank percentiles in [0,1].

3. **Overfit score**
   - overfit_score_j = rank_perm_j – rank_gain_j.
   - Large positive: high gain, low permutation.

Use this only for:

- Reporting.
- Manually flagging suspicious features (e.g., ID-like features).

No automated dropping by overfit_score in v1 unless explicitly configured.

### 6.8 Final Ablation Models

Using full TRAIN and VAL:

Define feature sets:

- F_all = F0 (after static pre-filters).
- F_fs = K_keep combined with a subset of Rest, according to configuration:
  - Option A: F_fs = K_keep ∪ Rest (drop only TopK features classified as K_drop).
  - Option B: F_fs = K_keep (aggressive).

Train models:

1. **Model A (baseline)**
   - Features: F_all.
   - Train XGB with “production-like” hyperparams on TRAIN, early stopping on VAL.
   - Record PR-AUC on VAL: PR_A.

2. **Model B (filtered)**
   - Features: F_fs.
   - Same training procedure.
   - Record PR-B on VAL.

3. (Optional) **Model C (more aggressive)**
   - Features: stricter subset (e.g., only top N_perm by Δ_j).
   - Record PR-C.

Selection:

- Let PR_best = max(PR-A, PR-B, PR-C).
- Define tolerance τ (e.g., 0.5–1% relative).
- Choose the smallest feature set S* such that:
  - PR(S*) ≥ (1 – τ) * PR_best.

Rationale:

- Enforces that FS does not significantly degrade performance on OOT validation, while preferring simpler models.

### 6.9 Final Model and TEST

1. Retrain final model with features F* = S* on TRAIN (+VAL if desired).
2. Evaluate exactly once on TEST.
3. Output:
   - TEST PR-AUC.
   - Final feature list, sorted by permutation FI.
   - Diagnostic plots/tables.

---

## 7. Implementation Plan and Code Structure

### 7.1 Tech Stack

- Python 3.10+
- Core libs:
  - `pandas`, `numpy`
  - `scikit-learn`
  - `xgboost`
  - `shap`
  - `pyyaml` or `omegaconf` for configs
  - `matplotlib` / `seaborn` for basic plots (optional)

### 7.2 Package Layout

Proposed structure:

```text
fs_xgb/
  __init__.py
  config/
    default_config.yaml
  data/
    datasets_registry.py
    loaders/
      covertype.py
      spambase.py
      har_smartphones.py
      internet_ads.py
      ...
  splitting/
    time_splits.py
    random_splits.py
  models/
    xgb_fs_model.py        # FS model training
    xgb_final_model.py     # final models
  fi/
    shap_triage.py
    permutation_fi.py
    gain_overfit_diag.py
  fs_logic/
    static_filters.py
    fs_pipeline.py         # orchestrates full FS flow
  eval/
    metrics.py             # PR-AUC, ROC-AUC
    reporting.py           # tables, plots
  cli/
    main.py                # entry point: python -m fs_xgb.cli ...
  experiments/
    configs/
      spambase_baseline.yaml
      covertype_baseline.yaml
      ...
    notebooks/
      exploratory_analysis.ipynb
````

### 7.3 Core APIs

Key functions / classes:

* `load_dataset(name: str) -> (X, y, meta)`
* `make_splits(meta, strategy_config) -> (TRAIN, VAL, TEST)`
* `run_static_filters(X_train, meta, config) -> (X_train_filtered, feature_list)`
* `train_fs_models(X_train_fs, y_train_fs, X_holdout_fs, y_holdout_fs, config) -> List[models]`
* `compute_shap_triage(models, X_eval, config) -> ranked_features`
* `compute_permutation_fi(models, X_eval, y_eval, topk_features, config) -> fi_table`
* `select_features(fi_table, shap_info, config) -> feature_sets`
* `train_and_eval_final_models(feature_sets, data_splits, config) -> results`
* `run_full_fs_experiment(dataset_config) -> experiment_report`

---

## 8. Configuration and CLI

Use YAML configs to describe experiments:

Example `experiments/configs/spambase_baseline.yaml`:

```yaml
dataset: "spambase"
target: "is_spam"
metric: "prauc"

splits:
  strategy: "random"
  test_size: 0.2
  val_size: 0.2
  random_state: 42

fs:
  n_fs_models: 3
  topk_shap: 60
  fs_eval:
    neg_pos_ratio: 10
  thresholds:
    delta_abs_min: 0.001
    k_noise_std: 2.0
  rest_policy: "keep_all"  # or "drop_all", "keep_above_min_shap"

xgb_fs_params:
  max_depth: 5
  min_child_weight: 10
  subsample: 0.8
  colsample_bytree: 0.8
  lambda: 1.0
  eta: 0.1
  n_estimators: 300

xgb_final_params:
  max_depth: 6
  min_child_weight: 10
  subsample: 0.8
  colsample_bytree: 0.8
  lambda: 2.0
  eta: 0.05
  n_estimators: 2000
  early_stopping_rounds: 100

selection:
  val_tolerance_relative: 0.01
  max_final_feature_sets: 3
```

CLI:

* `python -m fs_xgb.cli run --config experiments/configs/spambase_baseline.yaml`

Output:

* JSON/YAML report with:

  * FS model FI summary.
  * Selected feature set(s).
  * TRAIN/VAL/TEST metrics.
* Optional HTML or Markdown summary.

---

## 9. Dataset Harness

Implement dataset loaders and standard splits for a small curated set:

For each dataset:

* Loader:

  * Path or download instructions.
  * Feature/target columns.
  * Any necessary preprocessing (e.g., type casting, simple encoding).
* Splits:

  * Predefined TRAIN/VAL/TEST splits consistent across runs (e.g., fixed random_state or chronological split).

Initial target dataset list:

1. Factors Affecting University Student Grades (Kaggle) 
https://www.kaggle.com/datasets/atifmasih/factors-affecting-university-student-grades?utm_source=chatgpt.com

2. Census-Income (KDD) – UCI 
https://archive.ics.uci.edu/dataset/117/census%2Bincome%2Bkdd?utm_source=chatgpt.com

3. BRFSS 2015 (CDC)
https://www.cdc.gov/brfss/annual_data/annual_2015.html?utm_source=chatgpt.com


All problems should be framed as imbalanced binary classification. So some transformations of the target from the original datasets may be needed.

---

## 10. Evaluation, Reporting, and Acceptance Criteria

### 10.1 Evaluation Outputs

For each experiment:

* Metrics:

  * PR-AUC and ROC-AUC on TRAIN, VAL, TEST for each model (A/B/C).
* Feature selection outcomes:

  * Final feature set(s) F*.
  * Table of ΔPR-AUC per feature (TopK).
  * SHAP vs permutation vs gain comparisons for diagnostics.

### 10.2 Acceptance Criteria (Functional)

1. **End-to-end run**

   * Given a valid dataset config, `run_full_fs_experiment` runs to completion and returns:

     * A final feature list.
     * Final TRAIN/VAL/TEST metrics.
     * Intermediate FI diagnostics.

2. **Performance tolerance**

   * For test datasets (e.g., Spambase, Covertype), using F* must achieve:

     * VAL PR-AUC within 1% relative of using all features (F_all), in default configs.

3. **FS non-triviality**

   * On at least one high-d dataset (HAR or Internet Ads), the number of selected features |F*| should be substantially lower than |F_all| (e.g., < 30–50% of original) while preserving VAL PR-AUC within tolerance.

4. **Reproducibility**

   * Running the same config twice with the same `random_state` yields the same selected feature set and metrics (within minor floating-point noise).

### 10.3 Acceptance Criteria (Design)

* FS logic must:

  * Use **permutation FI on an OOT slice** as the primary keep/drop criterion.
  * Use **SHAP only for triage and diagnostics**, not as the final usefulness metric.
  * Avoid dependence on gain FI for FS (gain used only for optional overfitting diagnostics).

---

## 11. Future Extensions (Non-blocking)

* Group-level operations:

  * Correlation-based clustering and group permutation FI.
* Multi-class native support:

  * Micro/macro PR-AUC and class-conditional FS.
* Support for non-XGB models (LightGBM, CatBoost).
* Automated hyperparameter tuning integrated with FS.



