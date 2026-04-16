# MVA Optimization Study

This directory contains the active multivariate-selection study used by the
top-level workflow, plus some local comparison material that is intentionally
kept study-local.

## What Is Active

The active production outputs consumed by the main pipeline are the per-category
CatBoost models and related optimization products written under
`generated/output/studies/mva_optimization/`:

- `generated/output/studies/mva_optimization/models/catboost_bdt_LL.cbm`
- `generated/output/studies/mva_optimization/models/catboost_bdt_DD.cbm`

The authoritative orchestration path is the top-level `analysis/Snakefile`
through `make study-mva` or `uv run snakemake study_mva`.

## What Is Study-Local

This directory still contains helper code and comparison paths that are kept
for study-local reproducibility, but are not shared source-of-truth for the
rest of the analysis:

- internal model-comparison logic used while choosing CatBoost
- the TMVA comparison path under `tmva/`
- local helper scripts that should not be imported as general shared modules

Those files remain here intentionally so the study stays reproducible without
duplicating the active pipeline logic elsewhere.

## Guidance

- use the top-level `analysis/Snakefile` for normal workflow execution
- treat this directory as the owned implementation area for the MVA study
- do not treat its local comparison helpers as shared framework code unless
  they are deliberately promoted into `analysis/modules/` later

## TMVA Comparison

The TMVA path is retained for cross-checks and historical comparison only. It
is not part of the active production run and may require an environment with a
full ROOT installation that includes TMVA.
