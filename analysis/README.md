# B⁺ → Λ̄pK⁻K⁺ Charmonium Analysis — Snakemake Pipeline

Snakemake-based analysis pipeline for measuring branching fraction ratios of charmonium states
(J/ψ, ηc, χc0, χc1, ηc(2S)) in B⁺ → Λ̄pK⁻K⁺ decays at LHCb.


## Prerequisites

- **Python ≥ 3.11**
- **[uv](https://docs.astral.sh/uv/)** — for dependency management

All Python packages (including ROOT/PyROOT) are managed by `uv` and installed automatically.

## Setup

```bash
# From the repository root:
uv sync
```

This installs all dependencies into the repo-root `.venv/`, including:
- **HEP stack:** `uproot`, `awkward`, `ROOT` (PyROOT), `mplhep`, `vector`
- **Core:** `numpy`, `scipy`, `matplotlib`, `pandas`
- **Workflow:** `snakemake`
- **Utilities:** `tqdm`, `uncertainties`, `psutil`, `pyyaml`, `tomli`

## Running the Pipeline

### Full pipeline

All `uv run` commands below should be run from the `analysis/` directory.

The pipeline is parameterized by the optimization method (`mva` or `box`). We recommend using the provided Makefile for convenience:

```bash
cd analysis/
make OPT_METHOD=mva           # Run the full pipeline using MVA optimization
make clean                    # Clean all outputs and caches
```

Or you can run Snakemake directly:

```bash
uv run snakemake -j1 --config opt_method=mva  # Single-core (recommended for first run)
uv run snakemake -j2 --config opt_method=mva  # Steps 5 & 6 can run in parallel
```

### Dry run

```bash
uv run snakemake -n           # Show what would be executed (no actual work)
```

### Run individual steps

```bash
uv run snakemake validate_config       # Step 1 only
uv run snakemake load_data             # Steps 1–2
uv run snakemake optimize_selection    # Steps 1–3
uv run snakemake apply_cuts            # Steps 1–4
uv run snakemake mass_fitting          # Steps 1–5
uv run snakemake efficiency_calculation  # Steps 1–4, 6
uv run snakemake branching_ratios      # Full pipeline (all steps)
```

### Visualize the DAG

```bash
uv run snakemake --dag | dot -Tpdf > dag.pdf
```

## Pipeline Steps

| Step | Rule | Description | Key Outputs |
|------|------|-------------|-------------|
| 1 | `validate_config` | Validate all 11 TOML config files | `.config_validated` sentinel |
| 2 | `load_data` | Load data/MC ROOT files, apply Λ pre-selection | Cached pickle files (~1.2 GB) |
| 3 | `optimize_selection` | Run configured optimization (`box` grid scan or `mva` BDT cut) | `models/optimized_cuts.json` |
| 4 | `apply_cuts` | Apply optimized cuts to MC (data unchanged) | `cut_summary.json` |
| 5 | `mass_fitting` | Simultaneous RooFit mass fit (all charmonium states) | `fitted_yields.csv`, fit plots |
| 6 | `efficiency_calculation` | MC selection efficiency ε_sel and ratios vs J/ψ | `efficiencies.csv`, `efficiency_ratios.csv` |
| 7 | `branching_ratios` | Branching fraction ratios relative to J/ψ | `branching_fraction_ratios.csv`, `final_results.md` |
| 8 | `compare_branches` | Compare outputs between high/low yield branches | `branch_comparison.md` |
| 9 | `export_latex_results` | Generate LaTeX formatted table of ultimate BF products | `bf_products.tex` |

Steps 5 and 6 are independent and can run in parallel (both depend only on Step 4).

### Step details

**Step 2 — Data loading:**
Loads real data and Monte Carlo from ROOT files for all configured years, track types (LL/DD),
and magnet polarities (MagDown/MagUp). Applies fixed Λ pre-selection cuts (mass window, flight
distance, proton PID). Caches 4 outputs: `data_dict`, `mc_dict`, `phase_space_dict`,
`mc_generated_counts`.

**Step 3 — Selection optimization:**
Uses objects native to `analysis/modules/` (e.g. `BoxOptimizer`) to dynamically apply either a Box grid-scan or MVA BDT thresholding dynamically based on the `opt_method` configuration parameter.

**Step 5 — Mass fitting:**
Extracts a universal `MassFitter` from `analysis/modules/mass_fitter.py`. Performs a simultaneous binned maximum-likelihood fit to M(Λ̄pK⁻) in [2800, 4000] MeV using RooFit. Models 5 charmonium states with Double Crystal Ball signals and ARGUS background. Output paths isolate neatly by `opt_method` and `branch`.

**Step 6 — Efficiency:**
Calculates selection efficiency ε_sel = N_pass / N_generated from MC. Other efficiencies
(reconstruction, stripping, trigger, PID) cancel in ratios because all channels share an
identical final state (Λ̄pK⁻K⁺). Uses χc1 as proxy for ηc(2S) (no dedicated MC).

## Snakemake Configuration

All pipeline parameters are set in `snakemake_config.yaml` and can be overridden at the
command line with `--config key=value`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `opt_method` | str | `"box"` | Selection optimization strategy (`box` or `mva`) |
| `years` | list | `["2016", "2017", "2018"]` | LHCb data-taking years to process |
| `track_types` | list | `["LL", "DD"]` | Λ reconstruction categories |
| `magnets` | list | `["MD", "MU"]` | Magnet polarities (MagDown, MagUp) |
| `states` | list | `["jpsi", "etac", "chic0", "chic1"]` | Signal MC charmonium states |
| `use_manual_cuts` | bool | `false` | Skip optimization, use manual cuts from config |
| `no_cache` | bool | `false` | Force reprocessing (ignore cached intermediate results) |
| `config_dir` | str | `"config"` | Path to TOML configuration directory |
| `cache_dir` | str | `"cache"` | Path to intermediate cache directory |
| `output_dir` | str | `"analysis_output"` | Path to output directory |

### Common overrides

```bash
# Process only 2016 data (fast test)
uv run snakemake -j1 --config years='["2016"]'

# Use manual cuts (skip ~5 min optimization)
uv run snakemake -j1 --config use_manual_cuts=true

# Force full reprocessing (ignore all caches)
uv run snakemake -j1 --config no_cache=true

# Combine overrides
uv run snakemake -j1 --config years='["2016"]' use_manual_cuts=true no_cache=true
```

## TOML Configuration Files

The `config/` directory contains 11 TOML files controlling the physics analysis:

| File | Purpose |
|------|---------|
| `physics.toml` | PDG masses, widths, branching fractions, analysis method |
| `detector.toml` | Mass windows, signal regions, integrated luminosity per year |
| `fitting.toml` | Fit method (binned/unbinned), signal model (DCB), background (ARGUS), strategy |
| `selection.toml` | Λ cuts, B⁺ fixed cuts, optimizable variables, manual cuts, optimization strategy |
| `triggers.toml` | L0, HLT1, HLT2 trigger requirements |
| `data.toml` | Input ROOT file paths, output directories, cache settings, verbosity |
| `efficiencies.toml` | Efficiency components (which cancel in ratios, which are calculated) |
| `paths.toml` | Legacy path definitions (superseded by `data.toml`) |
| `luminosity.toml` | Legacy luminosity (superseded by `detector.toml`) |
| `branching_fractions.toml` | Legacy BR values (superseded by `physics.toml`) |
| `particles.toml` | Legacy particle properties (superseded by `physics.toml` + `detector.toml`) |

### Key analysis choices in config

- **Fit range:** M(Λ̄pK⁻) ∈ [2800, 4000] MeV (`detector.toml`)
- **B⁺ mass window:** M_corr ∈ [5255, 5305] MeV (`selection.toml`)
- **Λ mass window:** [1111, 1121] MeV (`selection.toml`)
- **Fit type:** Binned ML, 5 MeV/bin (`fitting.toml`)
- **Signal model:** Double Crystal Ball (`fitting.toml`)
- **Background model:** ARGUS function (`fitting.toml`)
- **Optimization:** Unbiased data-driven, universal cuts (`selection.toml`)
- **Efficiency:** Only ε_sel calculated; ε_reco, ε_strip, ε_trig, ε_PID cancel in ratios (`efficiencies.toml`)
