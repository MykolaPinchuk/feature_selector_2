# Feature Selection Iteration Roadmap

This document tracks the multi-iteration plan for the Feature Selector beyond Iteration&nbsp;1. Keep it updated as priorities shift so future agents can pick up the context quickly.

## Iteration 2 (complete)

- **Goal:** Add an optional “comprehensive” search mode that sweeps the feature-selection threshold space to approximate the full performance/feature-count frontier while preserving the existing fast profiles.
- **Fast mode (default):** Keep the present aggressive/moderate/mild profile loop untouched for quick experiments.
- **Comprehensive mode:**
  - Run the SHAP + permutation pipeline once to obtain importance artifacts.
  - Re-apply selection rules across a grid of (`delta_abs_min`, `k_noise_std`, rest-policy) combinations to generate multiple distinct kept/dropped sets without re-running SHAP.
  - Deduplicate identical feature sets; cap evaluations via config (e.g., `max_frontier_candidates`).
  - For each unique set, reuse the existing TRAIN/VAL/TEST splits to fit final XGB models and record metrics.
  - Pick the “optimal” set by enforcing a validation PR-AUC tolerance (relative to the frontier best) and minimizing feature count / overfitting gap.
  - Persist artifacts (metrics, feature lists, chosen thresholds) so users can inspect the full frontier.
- **Config surface:** Introduce an opt-in flag (e.g., `fs_search.mode: profiles|frontier`) plus knobs for sweep values, candidate cap, and selection criteria.
- **Reporting:** Extend `report.md` to summarize the selected frontier point and link to the comprehensive metrics file while keeping the legacy tables for fast mode runs.

## Iteration 3 (current)

- **Goal:** Ship a configurable overfit detector that cross-references SHAP ranks with permutation ΔPR-AUC so FS can suppress high-usage/low-value columns.
- **Config surface:** Introduce `fs.overfit_filter` (SHAP) and `fs.gain_overfit_filter` with knobs for enabling/disabling, rank windows, ΔPR-AUC and noise ceilings, and an action (`drop` vs `demote`). Allow overrides per `fs_modes`.
- **Selection logic:** Extend `_apply_selection_rules` so the flagged features are either forcibly dropped or demoted back into the rest-policy bucket before final kept/dropped lists are emitted. Features outside the permutation Top-K should still be eligible for flagging by treating their ΔPR-AUC as ~0.
- **Metadata & reporting:** Surface both SHAP- and gain-mismatch sets in `ModeResult.metadata`, list them in `report.md`, and log counts in `frontier_candidates.json` so users understand why features disappeared.
- **Tests:** Unit tests for the detector plus an integration smoke test that validates at least one feature is filtered when the heuristic criteria are met.


## Iteration 4 (in progress)

- **Onboard Census Income (KDD) dataset**
  - ✅ Add dataset loader + registry entry with `is_high_income` target (≈6% positives).
  - ✅ Provide CLI configs (`census_income*.yaml`) tuned for faster FS models and document download steps.
  - ✅ Produce balanced (1:1) subsample via `scripts/create_census_balanced_sample.py` and focus Iterations 4–5 on this downsample for faster iteration.
  - ✅ Enforce chronological splits so TRAIN uses 1994 rows while VAL/TEST begin at 1995 (OOT evaluation).
  - ☐ Run first full experiments + reports on the full dataset once the balanced workflow stabilizes.
- **EDA upgrades**
  - ✅ Extend `eda.md` with correlation analysis for numeric features + target lift.
  - ☐ Explore additional pre-filters for FS pipeline if runtime pressure grows.


## Iteration 5 

- ✅ For previously added datasets, swap the target encoder to the new `strategy="naive"` mode (no smoothing / frequency thresholding) so these columns overfit and provide a stronger signal for the FS heuristics.
- ✅ Add BRFSS 2015 (CDC) dataset support: loader + prep script, configs (full + sample), README instructions, and EDA/reporting artifacts for the diabetes target, with downsampled 50/50 class ratio and interview-date chronological splits to avoid leakage.

## Iteration 6 

- **Santander Customer Satisfaction dataset**
  - ✅ Add dataset loader + registry entry, configs (profiles + frontier), and README instructions.
  - ☐ Run first full experiments + report to baseline FS heuristics.
- **Permutation FI upgrades**
  - ☐ Add stat test against <=0 null hypothesis to better filter noise-dominated features.

## Iteration 7 

Try synthetic dataset as suggested by Opus.

## Iterations 8-9

- Add 2 more datasets. See 12/6 chatgpt chat.
- The hope here is that FS will provide large gains on these datasets.



## Iterations 10+
- Benchmark the framework (fast + comprehensive + future greedy mode) against established FS approaches (RFE, Boruta, filter-based methods, and more). This will require:
  - Shared split definitions and evaluation contracts so external algorithms can plug in.
  - Runtime tracking to compare cost vs quality.
  - Report sections or separate notebooks summarizing head-to-head results.
- Consider integrating the best-performing external FS outputs as additional candidates in the comprehensive search pipeline.



## Long-term

- **Option 2 (greedy pruning):** Implement a rank-guided, incremental pruning loop that removes the weakest SHAP-ranked feature, retrains a lightweight surrogate model, and stops once validation PR-AUC drops meaningfully. This should be configurable as another `fs_search.mode` but is lower priority than the frontier sweep.
