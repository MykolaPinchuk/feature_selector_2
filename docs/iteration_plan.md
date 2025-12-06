# Feature Selection Iteration Roadmap

This document tracks the multi-iteration plan for the Feature Selector beyond Iteration&nbsp;1. Keep it updated as priorities shift so future agents can pick up the context quickly.

## Iteration 2 (current)

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

## Iteration 3

- **SHAP vs permutation overfit filter (top priority):** Implement a mismatch detector that flags features with high SHAP/gain importance but low permutation ΔPR-AUC so the selection rules can penalize or drop overfit signals automatically.



## Iterations 5+
- Benchmark the framework (fast + comprehensive + future greedy mode) against established FS approaches (RFE, Boruta, filter-based methods, and more). This will require:
  - Shared split definitions and evaluation contracts so external algorithms can plug in.
  - Runtime tracking to compare cost vs quality.
  - Report sections or separate notebooks summarizing head-to-head results.
- Consider integrating the best-performing external FS outputs as additional candidates in the comprehensive search pipeline.



## Long-term

- **Option 2 (greedy pruning):** Implement a rank-guided, incremental pruning loop that removes the weakest SHAP-ranked feature, retrains a lightweight surrogate model, and stops once validation PR-AUC drops meaningfully. This should be configurable as another `fs_search.mode` but is lower priority than the frontier sweep.