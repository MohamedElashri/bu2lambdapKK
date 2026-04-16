# PID Optimization Study

This is a standalone exploratory study for the B+ → Λ̄pK⁻K⁺ analysis. It is
kept under `analysis/studies/standalone/` because it records an important
decision trail around PID optimization, but it is not part of the active
reproducible workflow driven by `analysis/Snakefile`.

## Role

- documents why proxy-based PID optimization is biased
- preserves the fit-based PID-cut scan and related comparison material
- remains useful as provenance and for targeted re-checks

## Status

- not in the default `make` or top-level Snakemake run graph
- active selection config may point here as historical context only
- outputs from this study are not treated as authoritative pipeline products

## Layout

```text
studies/standalone/pid_optimization_study/
├── config.toml
├── run_all.py
├── scripts/
│   ├── box_scan_proxy.py
│   ├── fit_based_scan.py
│   ├── mva_pid_study.py
│   └── compare_results.py
└── output/
```

## How To Run

From `analysis/`:

```bash
cd studies/standalone/pid_optimization_study
uv run python run_all.py
```

Examples:

```bash
uv run python run_all.py --steps proxy
uv run python run_all.py --steps fit
uv run python run_all.py --steps mva
uv run python run_all.py --steps compare
```

## Interpretation

- proxy-based results are expected to prefer little or no PID tightening
- fit-based scans are the authoritative result inside this standalone study
- MVA-with-PID comparisons are decision-support material, not production truth

## Relationship To The Main Workflow

- active config ownership now lives under `analysis/config/` and
  `analysis/modules/config_loader.py`
- active PID bootstrap/systematics live under `analysis/studies/pid_cancellation/`
- this study stays separate so the main workflow does not inherit exploratory
  assumptions or duplicate logic
