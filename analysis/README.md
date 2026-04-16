# B⁺ → Λ̄pK⁻K⁺ Charmonium Analysis Pipeline

Snakemake-based analysis for measuring branching fraction ratios of charmonium states
(J/ψ, η_c, χ_c0, χ_c1, η_c(2S)) in B⁺ → Λ̄pK⁻K⁺ decays at LHCb.

---

## Prerequisites

- **Python ≥ 3.11**
- **[uv](https://docs.astral.sh/uv/)** — dependency manager
- **graphviz** — optional, only needed for `make dag`

All Python packages (including ROOT/PyROOT) are managed by `uv`:

```bash
cd /path/to/bu2lambdapKK
uv sync
```

---

## Quick start

All commands run from the `analysis/` directory.

```bash
cd analysis/

make          # Full unified run (studies -> pipeline -> systematics -> reports)
make plots    # Optional presentation plots
make help     # Print all available targets
make clean    # Remove all outputs before a fresh run
```

---

## Full Workflow

The top-level `Snakefile` is now the authoritative DAG. The `Makefile` is a
thin wrapper around named workflow families. Running `make` executes the four
core families:

```
studies  →  main-pipeline  →  systematics  →  reports
```

| Workflow family | Make target | What it does |
|-----------------|-------------|--------------|
| Studies | `make studies` | Train MVA, compute trigger ratios, derive kinematic weights, compute efficiency table |
| Main pipeline | `make main-pipeline` | Load data, optimize selection, fit mass spectra, compute efficiencies, calculate BF ratios, and compare branches |
| Systematics | `make systematics` | Run PID bootstrap and aggregate the fit, selection, and PID systematics into final uncertainty products |
| Reports | `make reports` | Collect key outputs into `generated/output/reports/` and generate the HEP-style final report |

Presentation plots are intentionally separate from the default physics run:

```bash
make plots
```

### Running individual workflow families

```bash
make studies           # Studies only
make main-pipeline     # Main pipeline only
make systematics       # Systematics only
make reports           # Reporting only
make plots             # Optional note/slides figures
```

### Controlling parallelism and method

```bash
make CORES=8               # Use 8 parallel Snakemake jobs (default: 4)
make OPT_METHOD=box        # Use box-cut optimization instead of MVA (default: mva)
make OPT_METHOD=box CORES=2
```

---

## Make Targets Reference

### Full workflow

| Target | Description |
|--------|-------------|
| `make` | **Full unified run**: studies → pipeline → systematics → reports |
| `make full-analysis` | Same as `make` |
| `make dry-run` | Show what Snakemake would execute without running anything |
| `make dag` | Render pipeline DAG to `dag.pdf` (requires graphviz) |
| `make rerun` | Force complete full-analysis re-run (ignores existing outputs) |

### Prerequisite studies

| Target | Description |
|--------|-------------|
| `make studies` | Run all four prerequisite studies in order |
| `make study-mva` | Train CatBoost BDTs for LL and DD separately |
| `make study-trigger` | Compute TIS/TOS trigger efficiency ratios |
| `make study-reweighting` | Derive per-category kinematic reweighting weights |
| `make study-efficiency` | Compute cumulative efficiency table |

### Main pipeline steps

Each target runs all preceding steps automatically.

| Target | Steps | Key outputs |
|--------|-------|-------------|
| `make load-data` | 1–2 | Cached data/MC pickle files |
| `make optimize` | 1–3 | `models/optimized_cuts.json` per branch×category |
| `make apply-cuts` | 1–4 | `tables/cut_summary.json` |
| `make mass-fitting` | 1–5 | `tables/fitted_yields.csv`, fit plots |
| `make efficiency` | 6 | `tables/efficiencies.csv`, `efficiency_ratios.csv` |
| `make branching-ratios` | 7 | `generated/output/pipeline/<opt_method>/<branch>/tables/branching_fraction_ratios.csv`, `generated/output/pipeline/<opt_method>/<branch>/results/final_results.md` |
| `make compare` | 8 | `comparison/branch_comparison.md` |
| `make export-latex` | 9–10 | `generated/output/pipeline/<opt_method>/results/bf_products.tex` after the required systematics summary is built |
| `make main-pipeline` | 1–8 | All main-physics outputs through branch comparison |

### Systematic studies

| Target | Description |
|--------|-------------|
| `make systematics` | Run the PID bootstrap and aggregate the full systematic workflow |
| `make study-fit-syst` | Fit model variations (ARGUS→poly2, endpoint ±50 MeV, resolution ±2 MeV) |
| `make study-sel-syst` | Selection threshold systematic (BDT cut ±1 step) |
| `make study-pid-bootstrap` | PID efficiency bootstrap (100 Gaussian-smeared iterations) |

### Presentation workflows

| Target | Description |
|--------|-------------|
| `make plots` | Run all optional presentation plots |
| `make plot-backgrounds` | Background-study plots |
| `make plot-reweighting` | Reweighting validation plots |
| `make plot-note` | Analysis-note plots |

Presentation scripts now follow one shared runtime contract:

- plots that can consume stable pipeline products should do that directly
  - examples: `plot_datafit.py`, `plot_bdt_variables.py`
- plots that still need raw tuples for note-only diagnostics should read paths,
  windows, trigger assumptions, and fixed PID cuts from
  `modules/presentation_config.py` instead of embedding their own copies
- active presentation workflows now live under `analysis/presentation/`
- exploratory one-off studies now live under `analysis/studies/standalone/`
- `analysis/studies/pid_cancellation/pidcalib2/` is intentionally kept as a
  vendored study-local dependency

### Reporting

| Target | Description |
|--------|-------------|
| `make collect-results` | Copy all key outputs into `generated/output/reports/collected/` |
| `make report-hep` | Generate the HEP-style LaTeX and text summary |
| `make reports` | Run `collect-results` and `report-hep` together |

### Validation

| Target | Description |
|--------|-------------|
| `make dry-run` | Full-analysis dry-run against the active DAG |
| `make validate-config` | Run `validate_config` into `generated/output/validation/pipeline/<opt_method>/` |
| `make smoke-imports` | Import shared modules and instantiate shared config helpers |
| `make smoke-subset` | Execute a real one-year/one-category `load_data` smoke run into `generated/output/validation/pipeline/<opt_method>/` |
| `make smoke` | Run the structural smoke suite: dry-run, imports, config validation, subset run, and presentation dry-run |

The smoke subset is intentionally isolated from the real outputs. By default it
uses:

- `VALIDATION_OUTPUT_ROOT=generated/output/validation`
- `VALIDATION_CACHE_ROOT=generated/cache/validation`
- `SMOKE_YEAR=2016`
- `SMOKE_TRACK=LL`
- `SMOKE_MAGNET=MD`
- `SMOKE_STATE=Jpsi`

Example:

```bash
make smoke SMOKE_YEAR=2017 SMOKE_TRACK=DD SMOKE_MAGNET=MU
```

### Cleaning

| Target | What is removed |
|--------|-----------------|
| `make clean` | Everything generated under `analysis/generated/` plus `.snakemake/`. **`studies/pid_cancellation/pidcalib_output/` is preserved** (PIDCalib2 histograms from lxplus) |
| `make clean-generated` | Alias for `make clean` |
| `make clean-main` | `generated/output/` and `generated/cache/` |
| `make clean-studies` | `generated/output/studies/` only |
| `make clean-snakemake` | Top-level and study-local `.snakemake/` directories |

## Tree Boundaries

The tree is separated into three conceptual areas:

- active source
  - `analysis/config/`
  - `analysis/modules/`
  - `analysis/scripts/`
  - active study code that is still referenced by the top-level `Snakefile`
  - presentation workflows under `analysis/presentation/` that either consume stable
    products or use the shared presentation config layer
  - exploratory but intentionally kept one-off studies under
    `analysis/studies/standalone/`
- generated outputs
  - `analysis/generated/output/pipeline/`
  - `analysis/generated/output/studies/`
  - `analysis/generated/output/presentation/`
  - `analysis/generated/output/reports/`
  - `analysis/generated/output/validation/`
  - `analysis/generated/cache/`
  - `analysis/.snakemake/`
- archived historical/reference material
  - `analysis/archive/`

Archive examples:

- `analysis/archive/reference_workflows/`

Additional ownership notes:

- `analysis/presentation/` contains optional note, slide, and validation plots
- `analysis/studies/standalone/` contains exploratory or documentation-only
  studies that are intentionally kept outside the active DAG
- `analysis/studies/pid_cancellation/` owns the vendored `pidcalib2/` payload
  and the externally prepared `pidcalib_output/` area

### Generated layout

Generated material is now centralized under one output root and one cache root:

- `generated/output/pipeline/<opt_method>/` for the main physics workflow
- `generated/output/studies/<study_name>/` for active study products
- `generated/output/presentation/<workflow_name>/` for optional figures
- `generated/output/reports/` for collected tables and final report products
- `generated/output/validation/` for smoke/validation runs
- `generated/cache/pipeline/<opt_method>/` for the main cached arrays
- `generated/cache/validation/` for smoke/validation cache material

---

## Pipeline Steps Detail

### Steps 1–2: Config validation + data loading

Validates the active analysis configuration, then loads data and MC from ROOT files for all configured years (2016–2018), track types (LL/DD), and magnet polarities (MagDown/MagUp). Applies fixed Λ pre-selection (mass window, flight distance, proton PID). Results are cached for downstream optimization and fitting.

**Cache note:** The load-data cache depends on the active `config/*.toml` files plus the loader path (`scripts/load_data.py`, `modules/clean_data_loader.py`). Rebuild with `make load-data` after changing those inputs.

### Step 3: Selection optimization

Per (branch, category): either a box grid-scan (`OPT_METHOD=box`) or MVA BDT threshold scan (`OPT_METHOD=mva`, default). Outputs `optimized_cuts.json`.

- **high_yield** branch: FOM = S/√B (J/ψ + η_c)
- **low_yield** branch: FOM = S/√(S+B) (χ_c0, χ_c1, η_c(2S))

### Step 5: Mass fitting

Simultaneous binned maximum-likelihood fit to M(Λ̄pK⁻) in [2800, 4000] MeV using RooFit. The active fitter models the charmonium signals with RooVoigtian shapes (PDG mass/width convolved with a shared Gaussian resolution) and uses an ARGUS background in the nominal fit. Fit plots are saved to `{branch}/{category}/plots/fits/mass_fit_{year}.pdf`.

### Step 6: Efficiency

Builds per-category efficiency tables from the standalone `studies/efficiency_steps/` output, then writes `efficiencies.csv` and `efficiency_ratios.csv` for the main pipeline. The active implementation consumes the study-level total efficiencies rather than deriving a fresh `ε_sel = N_pass / N_generated` inside this step. `η_c(2S)` currently remains a placeholder state until MC is available.

### Steps 7–10: Branching fractions

LL and DD yields summed; efficiency ratios yield-weighted. Normalization channel: B⁺ → J/ψ K⁺. Systematic uncertainties loaded from `systematics.json` after the systematic studies run. Final results exported to LaTeX.

## Snakemake Configuration

All pipeline parameters are in `snakemake_config.yaml` and overridable at the command line.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `opt_method` | `"mva"` | Selection optimization strategy (`mva` or `box`) |
| `years` | `["2016","2017","2018"]` | LHCb data-taking years |
| `track_types` | `["LL","DD"]` | Λ reconstruction categories |
| `magnets` | `["MD","MU"]` | Magnet polarities |
| `states` | `["jpsi","etac","chic0","chic1"]` | Signal MC states |
| `use_manual_cuts` | `false` | Skip optimization, use manual cuts |
| `no_cache` | `false` | Force reprocessing (ignore cached results) |

### Useful Snakemake overrides

```bash
# Process only 2016 (fast test)
uv run snakemake -j1 --config years='["2016"]' opt_method=mva

# Use manual cuts (skip ~10 min optimization)
uv run snakemake -j1 --config use_manual_cuts=true opt_method=mva

# Force full reprocessing
uv run snakemake -j4 --config no_cache=true opt_method=mva
```

---

## TOML Configuration Files

Located in `config/`:

| File | Purpose |
|------|---------|
| `physics.toml` | PDG masses, widths, branching fractions |
| `detector.toml` | Mass windows, signal regions, luminosity |
| `fitting.toml` | Reference fit/plot configuration; not yet the single operational source for the active pipeline |
| `selection.toml` | Λ cuts, B⁺ cuts, optimizable variables, optimization strategy |
| `triggers.toml` | L0, HLT1, HLT2 trigger requirements |
| `data.toml` | Input ROOT file paths, cache settings |
| `generator_effs.toml` | Generator-level efficiencies used by the efficiency study |
| `efficiencies.toml` | Which efficiency components cancel in ratios |

**Operational note:** active pipeline rules and active prerequisite studies now load config through the shared `modules.config_loader.StudyConfig` layer rooted at `analysis/config/`. The current ownership split inside that shared layer is:

- `selection.toml` owns active cuts, optimization settings, and operational fitter knobs
- `physics.toml` owns normalization constants and PDG masses/widths
- `data.toml` owns input paths, years, and magnet lists
- `generator_effs.toml` owns generator efficiencies consumed by the efficiency study
- `fitting.toml` contributes plotting/reference metadata that is merged into the active config view

### Key analysis choices

| Parameter | Value | Location |
|-----------|-------|----------|
| Fit range | M(Λ̄pK⁻) ∈ [2800, 4000] MeV | `detector.toml` |
| B⁺ mass window | M_corr ∈ [5255, 5305] MeV | `selection.toml` |
| Λ mass window | [1111, 1121] MeV | `selection.toml` |
| Fit type | Binned ML, 5 MeV/bin | `fitting.toml` |
| Signal model | RooVoigtian (nominal fitter implementation) | `modules/mass_fitter.py` |
| Background model | ARGUS function | `fitting.toml` |
| PID cut | PID_product > 0.25 (fixed) | `selection.toml` |
| Normalization channel | B⁺ → J/ψ K⁺ | `physics.toml` |
