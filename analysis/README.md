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
- **HEP stack:** `uproot`, `awkward`, `ROOT` (PyROOT via CERN pip package), `mplhep`, `vector`
- **Core:** `numpy`, `scipy`, `matplotlib`, `pandas`
- **Workflow:** `snakemake`
- **Utilities:** `tqdm`, `uncertainties`, `psutil`, `pyyaml`, `tomli`

## Running the Pipeline

### Full pipeline

All `uv run` commands below should be run from the `analysis/` directory:

```bash
cd analysis/
uv run snakemake -j1          # Single-core (recommended for first run)
uv run snakemake -j2          # Steps 5 & 6 can run in parallel
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
| 3 | `optimize_selection` | N-D grid scan FOM optimization (or manual cuts) | `optimized_cuts.csv` |
| 4 | `apply_cuts` | Apply optimized cuts to MC (data unchanged) | `step4_summary.json` |
| 5 | `mass_fitting` | Simultaneous RooFit mass fit (all charmonium states) | `step5_yields.csv`, fit plots |
| 6 | `efficiency_calculation` | MC selection efficiency ε_sel and ratios vs J/ψ | `efficiencies.csv`, `efficiency_ratios.csv` |
| 7 | `branching_ratios` | Branching fraction ratios relative to J/ψ | `branching_fraction_ratios.csv`, `final_results.md` |

Steps 5 and 6 are independent and can run in parallel (both depend only on Step 4).

### Step details

**Step 2 — Data loading:**
Loads real data and Monte Carlo from ROOT files for all configured years, track types (LL/DD),
and magnet polarities (MagDown/MagUp). Applies fixed Λ pre-selection cuts (mass window, flight
distance, proton PID). Caches 4 outputs: `data_dict`, `mc_dict`, `phase_space_dict`,
`mc_generated_counts`.

**Step 3 — Selection optimization:**
Two modes controlled by `use_manual_cuts`:
- **Grid scan** (default): Exhaustive N-D search over 7 variables (3,888 combinations).
  Uses unbiased data-driven FOM with B⁺ mass sidebands for background and no-charmonium
  region for signal proxy. Produces universal or state-specific cuts.
- **Manual cuts**: Uses predefined cuts from `config/selection.toml [manual_cuts]`.

**Step 5 — Mass fitting:**
Simultaneous binned maximum-likelihood fit to M(Λ̄pK⁻) in [2800, 4000] MeV using RooFit.
Models 5 charmonium states (ηc, J/ψ, χc0, χc1, ηc(2S)) + ψ(3770) with Double Crystal Ball
signals and ARGUS background. Fits per-year and combined. All masses/widths fixed to PDG values.

**Step 6 — Efficiency:**
Calculates selection efficiency ε_sel = N_pass / N_generated from MC. Other efficiencies
(reconstruction, stripping, trigger, PID) cancel in ratios because all channels share an
identical final state (Λ̄pK⁻K⁺). Uses χc1 as proxy for ηc(2S) (no dedicated MC).

## Snakemake Configuration

All pipeline parameters are set in `snakemake_config.yaml` and can be overridden at the
command line with `--config key=value`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `years` | list | `["2016", "2017", "2018"]` | LHCb data-taking years to process |
| `track_types` | list | `["LL", "DD"]` | Λ reconstruction categories (Long-Long, Downstream-Downstream) |
| `magnets` | list | `["MD", "MU"]` | Magnet polarities (MagDown, MagUp) |
| `states` | list | `["jpsi", "etac", "chic0", "chic1"]` | Signal MC charmonium states |
| `use_manual_cuts` | bool | `false` | Skip grid scan, use manual cuts from config |
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

## Output Files

After a successful run, outputs are organized under `analysis_output/`:

```
analysis_output/
├── tables/
│   ├── optimized_cuts.csv              # Step 3: optimal cut values per variable
│   ├── step4_summary.json              # Step 4: cut application summary
│   ├── step5_yields.csv                # Step 5: fitted yields per state and year
│   ├── efficiencies.csv                # Step 6: selection efficiencies per state/year
│   ├── efficiencies.md                 # Step 6: formatted efficiency table
│   ├── efficiency_ratios.csv           # Step 6: ε(J/ψ)/ε(state) ratios
│   ├── branching_fraction_ratios.csv   # Step 7: final BR ratios
│   └── yield_consistency.csv           # Step 7: N/(L×ε) per year
├── plots/
│   ├── fits/
│   │   ├── mass_fit_2016.pdf           # Step 5: per-year fit projections
│   │   ├── mass_fit_2017.pdf
│   │   ├── mass_fit_2018.pdf
│   │   └── mass_fit_combined.pdf       # Step 5: combined fit
│   └── yield_consistency_check.pdf     # Step 7: yield vs year plot
└── results/
    └── final_results.md                # Step 7: summary with BR ratios and discussion
```

## Caching

Intermediate results are cached as pickle files in `cache/` using SHA256 content hashing
(`CacheManager`). Cache entries are automatically invalidated when config files or code
dependencies change.

Cached entries:
- **Step 2:** `step2_data_after_lambda`, `step2_mc_after_lambda`, `step2_phase_space_after_lambda`, `step2_mc_generated_counts` (~1.2 GB)
- **Step 3:** `step3_optimized_cuts`
- **Step 4:** `step4_data_final`, `step4_mc_final`
- **Step 5:** `step5_fit_results`
- **Step 6:** `step6_efficiencies`

On re-runs with unchanged config, cached steps complete in seconds instead of minutes.

## Cleaning Up

```bash
# Remove cached intermediate results only
uv run snakemake clean_cache -j1 -f

# Remove output files only (tables, plots, results)
uv run snakemake clean_outputs -j1 -f

# Remove everything (cache + outputs)
uv run snakemake clean_all -j1 -f

# Or use Snakemake's built-in output cleanup
uv run snakemake --delete-all-output -j1
```

## Directory Structure

```
analysis/
├── Snakefile                 # Main workflow definition (7 analysis rules + 3 clean rules)
├── snakemake_config.yaml     # Pipeline parameters (years, states, flags)
├── config/                   # 11 TOML analysis configuration files
├── modules/                  # Core analysis modules (1:1 copy from analysis/)
│   ├── data_handler.py       #   TOMLConfig, DataManager, FourMomentumCalculator
│   ├── lambda_selector.py    #   LambdaSelector — fixed Λ pre-selection
│   ├── cache_manager.py      #   CacheManager — SHA256 pickle caching
│   ├── selection_optimizer.py #  SelectionOptimizer — N-D grid scan FOM
│   ├── mass_fitter.py        #   MassFitter — RooFit simultaneous fitting
│   ├── efficiency_calculator.py # EfficiencyCalculator — ε_sel and ratios
│   ├── branching_fraction_calculator.py # BR ratios relative to J/ψ
│   ├── branch_config.py      #   BranchConfig — ROOT branch name management
│   └── exceptions.py         #   Custom exception hierarchy
├── utils/                    # Utility modules
│   ├── validate_config.py    #   ConfigValidator — pre-flight validation
│   └── logging_config.py     #   Warning suppression, progress bar config
├── scripts/                  # Snakemake rule wrapper scripts (one per step)
├── tests/                    # Test suite (unit, integration, validation)
├── cache/                    # Intermediate cached results (generated, ~1.2 GB)
├── analysis_output/          # Final outputs (generated)
│   ├── tables/               #   CSV/JSON result tables
│   ├── plots/                #   Fit plots, consistency plots
│   └── results/              #   Final summary documents
├── pyproject.toml            # Project config, dependencies, tool settings
└── plan.md                   # Development plan and implementation notes
```

## Testing

```bash
uv run pytest tests/ -v
```
