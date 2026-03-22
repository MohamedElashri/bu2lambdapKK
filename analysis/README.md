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

make          # Full A-Z run (recommended for a clean slate)
make help     # Print all available targets
make clean    # Remove all outputs before a fresh run
```

---

## Full A-Z Workflow

The analysis has four sequential phases. Running `make` executes all four:

```
studies  →  main-pipeline  →  systematics  →  collect-results
```

| Phase | Make target | What it does |
|-------|-------------|--------------|
| 1 | `make studies` | Train MVA, compute trigger ratios, derive kinematic weights, compute efficiency table |
| 2 | `make main-pipeline` | Load data, optimize selection, fit mass spectra, compute efficiencies, calculate BF ratios |
| 3 | `make systematics` | Fit model + selection + PID systematics; aggregate into final uncertainties |
| 4 | `make collect-results` | Gather all key outputs into `results/` |

### Running individual phases

```bash
make studies           # Phase 1 only
make main-pipeline     # Phase 2 only
make systematics       # Phase 3 only
make collect-results   # Phase 4 only
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
| `make` | **Full A-Z**: studies → pipeline → systematics → collect |
| `make dry-run` | Show what Snakemake would execute without running anything |
| `make dag` | Render pipeline DAG to `dag.pdf` (requires graphviz) |
| `make rerun` | Force complete main pipeline re-run (ignores existing outputs) |

### Prerequisite studies (Phase 1)

| Target | Description |
|--------|-------------|
| `make studies` | Run all four prerequisite studies in order |
| `make study-mva` | Train CatBoost BDTs for LL and DD separately |
| `make study-trigger` | Compute TIS/TOS trigger efficiency ratios |
| `make study-reweighting` | Derive per-category kinematic reweighting weights |
| `make study-efficiency` | Compute cumulative efficiency table |

### Main pipeline steps (Phase 2)

Each target runs all preceding steps automatically.

| Target | Steps | Key outputs |
|--------|-------|-------------|
| `make load-data` | 1–2 | Cached data/MC pickle files |
| `make optimize` | 1–3 | `models/optimized_cuts.json` per branch×category |
| `make apply-cuts` | 1–4 | `tables/cut_summary.json` |
| `make mass-fitting` | 1–5 | `tables/fitted_yields.csv`, fit plots |
| `make efficiency` | 6 | `tables/efficiencies.csv`, `efficiency_ratios.csv` |
| `make branching-ratios` | 7 | `tables/branching_fraction_ratios.csv`, `results/final_results.md` |
| `make compare` | 8 | `comparison/branch_comparison.md` |
| `make export-latex` | 9–10 | `results/bf_products.tex`, `results/systematics_summary.md` |
| `make main-pipeline` | 1–10 | All of the above |

### Systematic studies (Phase 3)

| Target | Description |
|--------|-------------|
| `make systematics` | Run all three studies then aggregate into `systematics.json` |
| `make study-fit-syst` | Fit model variations (ARGUS→poly2, endpoint ±50 MeV, resolution ±2 MeV) |
| `make study-sel-syst` | Selection threshold systematic (BDT cut ±1 step) |
| `make study-pid-bootstrap` | PID efficiency bootstrap (100 Gaussian-smeared iterations) |

### Collect results (Phase 4)

| Target | Description |
|--------|-------------|
| `make collect-results` | Copy all key outputs into `results/` |

### Cleaning

| Target | What is removed |
|--------|-----------------|
| `make clean` | Everything — analysis_output/, results/, study outputs, .snakemake metadata. **`studies/pid_cancellation/pidcalib_output/` is preserved** (PIDCalib2 histograms from lxplus) |
| `make clean-main` | `analysis_output/` and `results/` only |
| `make clean-studies` | Study outputs only (same PKL preservation) |
| `make clean-snakemake` | `.snakemake/metadata` and locks only |

---

## Pipeline Steps Detail

### Steps 1–2: Config validation + data loading

Validates all TOML config files, then loads data and MC from ROOT files for all configured years (2016–2018), track types (LL/DD), and magnet polarities (MagDown/MagUp). Applies fixed Λ pre-selection (mass window, flight distance, proton PID). Results cached as pickle files (~19 MB).

**Cache note:** Modifying `data_handler.py` or `selection.toml` invalidates this cache. Rebuild with `make load-data`.

### Step 3: Selection optimization

Per (branch, category): either a box grid-scan (`OPT_METHOD=box`) or MVA BDT threshold scan (`OPT_METHOD=mva`, default). Outputs `optimized_cuts.json`.

- **high_yield** branch: FOM = S/√B (J/ψ + η_c)
- **low_yield** branch: FOM = S/√(S+B) (χ_c0, χ_c1, η_c(2S))

### Step 5: Mass fitting

Simultaneous binned maximum-likelihood fit to M(Λ̄pK⁻) in [2800, 4000] MeV using RooFit. Models 5 charmonium states with Double Crystal Ball signals and ARGUS background. Fit plots saved to `{branch}/{category}/plots/fits/mass_fit_{year}.pdf`.

### Step 6: Efficiency

Calculates ε_sel = N_pass / N_generated from MC. All other efficiency components (ε_reco, ε_strip, ε_trig, ε_PID) cancel in ratios because all channels share the identical Λ̄pK⁻K⁺ final state.

### Steps 7–10: Branching fractions

LL and DD yields summed; efficiency ratios yield-weighted. Normalization channel: B⁺ → J/ψ K⁺. Systematic uncertainties loaded from `systematics.json` (populated by Phase 3). Final results exported to LaTeX.

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
| `fitting.toml` | Fit method (binned), signal model (DCB), background (ARGUS) |
| `selection.toml` | Λ cuts, B⁺ cuts, optimizable variables, optimization strategy |
| `triggers.toml` | L0, HLT1, HLT2 trigger requirements |
| `data.toml` | Input ROOT file paths, cache settings |
| `efficiencies.toml` | Which efficiency components cancel in ratios |

### Key analysis choices

| Parameter | Value | Location |
|-----------|-------|----------|
| Fit range | M(Λ̄pK⁻) ∈ [2800, 4000] MeV | `detector.toml` |
| B⁺ mass window | M_corr ∈ [5255, 5305] MeV | `selection.toml` |
| Λ mass window | [1111, 1121] MeV | `selection.toml` |
| Fit type | Binned ML, 5 MeV/bin | `fitting.toml` |
| Signal model | Double Crystal Ball | `fitting.toml` |
| Background model | ARGUS function | `fitting.toml` |
| PID cut | PID_product > 0.20 (fixed) | `selection.toml` |
| Normalization channel | B⁺ → J/ψ K⁺ | `physics.toml` |
