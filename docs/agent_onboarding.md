# Agent Onboarding Guide

The repository has grown considerably across Iterations 1–5. To keep future agents fast and focused, follow this read-order checklist instead of scanning the whole tree on every handoff.

1. **High-level context**
   - `README.md`: only skim the “Getting Started” + “Running Experiments” sections for CLI usage; skip dataset download blocks if you already have the sources.
   - `prd.md`: skim the Overview/Design Principles to understand the original intent, but defer to core codebase below + recent results for the authoritative implementation. Look through `docs/iteration_plan.md` too (Iterations 2–5 added features that are not captured in the initial PRD spec).
   - `docs/iteration_plan.md`: read the active iteration section and the next one on deck; earlier iterations can be skimmed or skipped.
   - Latest run report: open the most recent `results/<dataset>/<timestamp>/report.md` for the dataset you’re working on (e.g., the newest BRFSS run). This gives the freshest status on FS decisions, heuristics, and runtime notes without re-reading the entire PRD.
2. **Core codepaths**
   - `fs_xgb/pipeline.py`: main analysis flow pipeline. This is the entrypoint for most tasks.
   - `fs_xgb/eval/reporting.py`.
   - `fs_xgb/fs_logic/fs_pipeline.py`: only read when modifying permutation/SHAP logic; otherwise trust the existing helpers.
   - `fs_xgb/config/default_config.yaml` and any dataset-specific configs under `fs_xgb/experiments/configs/` relevant to your task.
3. **Supporting modules**
   - Only open `fs_xgb/eval/reporting.py` or `fs_xgb/eval/plots.py` if your task touches reporting/visualization.
   - `fs_xgb/data/*` and `scripts/*` should be read only when working on new datasets.
   - Ignore historical results under `results/` unless the task explicitly references a run directory—just read the latest report for the dataset in scope.
4. **Logs & temp files**
   - Do **not** open `logs/agents/` or old `results/*` directories unless required; they add little context and inflate token usage.
5. **PR/issue context**
   - If you need more background, prefer asking for clarification or referring to the latest iteration notes rather than re-reading the entire PRD.

Following this checklist keeps onboarding concise (usually 5–6 files) while still covering the essentials. Update this guide whenever the repo structure or runbook changes.
