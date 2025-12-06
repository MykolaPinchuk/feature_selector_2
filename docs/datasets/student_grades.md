# Factors Affecting University Student Grades

- **Source:** [Kaggle](https://www.kaggle.com/datasets/atifmasih/factors-affecting-university-student-grades)
- **Local file:** `data/raw/factors_affecting_university_student_grades.csv`
- **Task:** Binary classification (`target = 1` when `Grade == 'A'`, else `0`)
- **Rows:** 4,491 after deduplication and dropping rows without `Grade`
- **Features:** 26 columns (demographics, academics, lifestyle, resources)
- **Time column:** Not provided → random stratified splits are used.

## Loader Rules

1. Trim whitespace from column names and string fields.
2. Drop duplicate rows and examples without a `Grade`.
3. Derive `is_high_grade` as the binary target.
4. Return `(X, y, metadata)` where:
   - `X` includes all columns except `Grade`
   - `y` is the derived binary Series
   - `metadata` tracks source URL, task type, and row counts

## Default Split Configuration

```
test_size = 0.2
val_size  = 0.2
random_state = 42
splitting strategy = stratified random (see fs_xgb.splitting.random_splits)
```

## Notes for FS Experiments

- Class imbalance: ~34% of samples are positive (Grade A).
- Several columns contain categorical strings; encoding happens upstream of FS modeling.
- Missing values exist in attendance/study hour columns; treat NaNs as-is so tree models can infer splits.
- Feature engineering policy:
  - Binary categoricals → deterministic 0/1 indicator.
  - >2 category columns → out-of-fold target encoding with smoothing + rare-category thresholding for stability.
