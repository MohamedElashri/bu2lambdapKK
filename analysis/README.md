# B⁺ → Λ̄pK⁻K⁺ Charmonium Analysis Pipeline

## Overview

Complete analysis pipeline for measuring branching fraction ratios of charmonium states (J/ψ, ηc, χc0, χc1) in B⁺ → Λ̄pK⁻K⁺ decays. The pipeline processes data from 2016-2018, performs mass fitting with ROOT/RooFit, calculates efficiencies, and extracts branching fraction ratios.

**Key Feature:** Self-normalizing to J/ψ - we measure RATIOS, no absolute branching fractions needed!

---

## Quick Start

### Essential Commands

```bash
# 1. Activate virtual environment
source ../.venv/bin/activate

# 2. Run complete pipeline (all years: 2016, 2017, 2018)
make pipeline

# 3. View results
make show-results
make show-yields
```

### More Options

```bash
# Show all available commands
make help

# Test with single year
make pipeline-2016

# Clean cache and rerun
make clean-cache
make pipeline

# View outputs
make show-efficiencies
make list-outputs
```

### Direct Python Commands

```bash
# Full pipeline with all years (per-year + combined fits)
python run_pipeline.py --years 2016,2017,2018

# Single year for testing
python run_pipeline.py --years 2016

# Force reprocessing (no cache)
python run_pipeline.py --years 2016,2017,2018 --no-cache
```

---

## Pipeline Workflow

The pipeline runs all phases automatically through `run_pipeline.py`:

**Integrated Phases:**
1. Configuration validation
2. Data/MC loading with Lambda pre-selection
3. Selection optimization (optional, skipped by default)
4. Apply selection cuts
5. Mass fitting
6. Efficiency calculation
7. Branching fraction ratios

### Cache Management

The pipeline caches intermediate results for efficiency:

```bash
# Use cache (fast, default)
make pipeline

# Force reprocessing (ignores cache)
make pipeline-no-cache
# or
python run_pipeline.py --no-cache --years 2016,2017,2018

# Clear specific cached results
rm cache/phase5_fit_results.pkl  # Refit masses
rm cache/phase2_*.pkl            # Reload data
make clean-cache                 # Clear all cache
```

## Manual Cuts (Skip Optimization)

### Overview

Use manual cuts to **skip the 10-20 minute grid scan optimization** (Phase 3). Perfect for:
- **Quick testing**: Rapid iteration during development
- **Fixed cuts**: Physics-motivated values
- **Validation**: Compare manual vs. optimized cuts

### Usage

**Method 1: Automatic Detection** (config-based)
```bash
# 1. Edit config/selection.toml and uncomment manual cuts
vim config/selection.toml

# 2. Run pipeline (auto-detects manual cuts)
python run_pipeline.py --years 2016

# Or use Makefile
make pipeline-manual-2016
```

**Method 2: Explicit Flag**
```bash
python run_pipeline.py --use-manual-cuts --years 2016
```

### Configuration Format

Edit `config/selection.toml`:

```toml
[manual_cuts]
Bu_PT = { cut_type = "greater", value = 5000.0 }
Bu_FDCHI2_OWNPV = { cut_type = "greater", value = 150.0 }
Bu_IPCHI2_OWNPV = { cut_type = "less", value = 9.0 }
Bu_DTF_chi2 = { cut_type = "less", value = 20.0 }
h1_ProbNNk = { cut_type = "greater", value = 0.2 }
h2_ProbNNk = { cut_type = "greater", value = 0.2 }
p_ProbNNp = { cut_type = "greater", value = 0.25 }
```

### Available Variables

| Branch Name | Description | Typical Range | Cut Type |
|-------------|-------------|---------------|----------|
| `Bu_PT` | B+ transverse momentum (MeV/c) | 3000-10000 | greater |
| `Bu_FDCHI2_OWNPV` | B+ flight distance χ² | 100-500 | greater |
| `Bu_IPCHI2_OWNPV` | B+ impact parameter χ² | 0-25 | less |
| `Bu_DTF_chi2` | B+ decay tree fit χ² | 0-50 | less |
| `h1_ProbNNk` | K+ PID probability | 0.0-0.5 | greater |
| `h2_ProbNNk` | K- PID probability | 0.0-0.5 | greater |
| `p_ProbNNp` | Bachelor p̄ PID probability | 0.0-0.5 | greater |

### What Happens

| Phase | With Manual Cuts | With Grid Scan |
|-------|------------------|----------------|
| Phase 2 | Load data  | Load data  |
| Phase 3 | Use manual cuts ⚡ **(skips 10-20 min!)** | Run optimization  |
| Phases 4-7 | Continue normally  | Continue normally  |

### Output

Manual cuts create the same `tables/optimized_cuts.csv` format:
- Compatible with all downstream phases
- `max_fom = 0.0` (not optimized)
- Applies same cuts to **all states** (not state-dependent)

### When to Use

| Scenario | Use Manual Cuts? | Use Grid Scan? |
|----------|------------------|----------------|
| Quick testing | ✅ Yes | ❌ No |
| Production analysis | ❌ No | ✅ Yes |
| State-dependent cuts | ❌ No | ✅ Yes (required) |
| Development/iteration | ✅ Yes | ❌ No |
| Physics-motivated cuts | ✅ Yes (with justification) | Optional |

### Limitations

- **State-independence**: Manual cuts apply to ALL states (J/ψ, ηc, χc0, χc1)
- **No FOM**: Manual cuts don't have optimization metrics
- **Lambda cuts still fixed**: Only affects 7 optimizable variables

For **state-dependent optimization**, use the grid scan (comment out manual cuts).

### Makefile Targets

```bash
make pipeline-manual         # Manual cuts, all years
make pipeline-manual-2016    # Manual cuts, 2016 only
make pipeline-manual-test    # Manual cuts, quick test
```

### Switching Back to Grid Scan

```toml
[manual_cuts]
# Comment out all cuts to use grid scan optimization
# Bu_PT = { cut_type = "greater", value = 5000.0 }
# Bu_FDCHI2_OWNPV = { cut_type = "greater", value = 150.0 }
# ...
```

Then run normally:
```bash
make pipeline
```

---

## Pipeline Phases

### Phase 0: Branch Discovery (Manual)
**Status:** Already completed
- Branch names identified and documented
- Configuration files updated

### Phase 1: Configuration Validation
**Status:** Automatic
- Loads all TOML configuration files
- Validates paths and parameters
- Creates output directories

### Phase 2: Data/MC Loading + Lambda Pre-Selection
**Purpose:** Load raw data and apply fixed Lambda cuts
**Time:** 5-15 minutes (depending on data size)
**Caching:** Yes - saves intermediate results to `cache/`

**What it does:**
- Loads data for all years and track types (LL/DD)
- Loads MC for all 4 states (J/ψ, ηc, χc0, χc1)
- Applies fixed Lambda selection cuts:
  - Mass window: |M_Λ - 1115.683| < 5 MeV
  - Flight distance χ²: FD_CHI2 > 100
  - Impact parameter: IPCHI2_OWNPV > 9
  - PID: ProbNNp > 0.2
- Saves filtered events to cache

**Typical efficiencies:**
- Data: ~27% (harsh cuts due to background)
- MC: ~52% (cleaner signal events)

**Output:**
- `cache/phase2_data_after_lambda.pkl` - Cached data
- `cache/phase2_mc_after_lambda.pkl` - Cached MC

### Phase 3: Selection Optimization (Optional)
**Purpose:** Find optimal Bu-level cuts using 2D FOM scans
**Time:** 30-60 minutes
**Caching:** Yes
**Can Skip:** Yes - use default cuts

**What it does:**
- Scans Bu_PT, IPCHI2_OWNPV, and other variables
- Computes Figure of Merit (FOM) for each cut value
- Identifies optimal cuts per state
- Creates optimization plots

**Output:**
- `tables/optimized_cuts.csv` - Optimal cuts
- `plots/optimization/*.pdf` - FOM scan plots

**Note:** For draft analysis, you can skip this and use simple cuts (e.g., Bu_PT > 2000 MeV)

### Phase 4: Apply Optimized Cuts
**Purpose:** Apply Bu-level cuts to data and MC
**Time:** < 1 minute
**Status:** Currently simplified (Lambda cuts only)

**What it does:**
- Takes cuts from Phase 3
- Applies to both data and MC
- Creates final datasets for fitting and efficiency

**Note:** Currently just passes through Lambda-cut data. Bu-level cut application can be added later.

### Phase 5: Mass Fitting
**Purpose:** Fit charmonium mass spectrum to extract yields
**Time:** 5-10 minutes
**Caching:** Yes

**What it does:**
- Applies B+ mass cut: [5255, 5305] MeV using Bu_MM_corrected
- Sets up RooFit mass observable M(Λ̄pK⁻) ∈ [2800, 4000] MeV
- Creates signal PDFs: RooVoigtian for each state
  - J/ψ: M=3096.92 MeV, Γ=0.093 MeV (fixed)
  - ηc: M=2983.90 MeV, Γ=32.0 MeV (floating)
  - χc0: M=3414.75 MeV, Γ=10.5 MeV (floating)
  - χc1: M=3510.66 MeV, Γ=0.84 MeV (fixed)
- Creates background PDF: Exponential per year
- Performs extended likelihood fit per year
- **NEW:** Also fits combined dataset (all years)
- Shares physics parameters across years
- Extracts yields with uncertainties
- Creates publication-quality fit plots

**Output:**
- `tables/phase5_yields.csv` - Yields per state/year + combined
- `plots/fits/mass_fit_2016.pdf` - Individual year fits
- `plots/fits/mass_fit_2017.pdf`
- `plots/fits/mass_fit_2018.pdf`
- `plots/fits/mass_fit_combined.pdf` - **Combined 2016-2018 fit**
- `cache/phase5_fit_results.pkl` - Complete fit results

**Typical yields (2016-2018 combined):**
```
combined:
  J/ψ     :  232 ±   2
  ηc      :  532 ±   3
  χc0     :   47 ±   1
  χc1     :   44 ±   1
```

### Phase 6: Efficiency Calculation
**Purpose:** Calculate selection efficiencies from MC
**Time:** 2-5 minutes
**Caching:** Yes

**What it does:**
- Takes MC after Lambda cuts (from Phase 2)
- Applies optimized cuts (from Phase 3)
- Calculates selection efficiency: ε_sel = N_after / N_before
- Computes binomial errors: σ_eff = sqrt(ε × (1-ε) / N)
- Calculates efficiency ratios: ε_J/ψ / ε_state
- Propagates errors through ratios

**Output:**
- `tables/efficiencies.csv` - Efficiencies per state/year
- `tables/efficiency_ratios.csv` - Ratios ε_J/ψ / ε_state
- `tables/efficiencies.md` - Human-readable table
- `cache/phase6_efficiencies.pkl` - Cached results

**Expected efficiencies:**
```
All states: ~85-90% (very similar)
Ratios: ~0.96-1.11 (close to 1.0)
```

### Phase 7: Branching Fraction Ratios
**Purpose:** Calculate final physics results
**Time:** < 1 minute
**Caching:** No (fast calculation)

**What it does:**
- Combines yields from Phase 5 (per-year only, skips "combined")
- Combines efficiencies from Phase 6
- Calculates efficiency-corrected yields: Σ(N/ε) per state
- Computes BR ratios: R = Σ(N_state/ε_state) / Σ(N_J/ψ/ε_J/ψ)
- Full error propagation including efficiency uncertainties
- Yield consistency check: N/(L×ε) vs year
- Generates final summary

**Output:**
- `tables/branching_fraction_ratios.csv` - Final BR ratios
- `tables/yield_consistency.csv` - Consistency check
- `plots/yield_consistency_check.pdf` - Consistency plot
- `results/final_results.md` - Complete summary

**Actual results (2016-2018):**
```
ηc/J/ψ ratio:    2.299 ± 0.238  (ηc ~2.3× J/ψ)
χc0/J/ψ ratio:   0.279 ± 0.107  (χc0 ~28% of J/ψ)
χc1/J/ψ ratio:   0.201 ± 0.058  (χc1 ~20% of J/ψ)
χc1/χc0 ratio:   0.721 ± 0.346  (not NRQCD predicted ~3)
```

**Note:** These are statistical uncertainties only. Systematics to be added.

## Caching System

The pipeline uses intelligent caching to avoid reprocessing:

### Cache Location
All cached results stored in `cache/` directory:
```
cache/
├── phase2_data_after_lambda.pkl    # Data after Lambda cuts
├── phase2_mc_after_lambda.pkl      # MC after Lambda cuts
├── phase3_optimized_cuts.pkl       # Optimized cuts
├── phase5_fit_results.pkl          # Fit results
└── phase6_efficiencies.pkl         # Efficiencies
```

### Cache Usage
```bash
# Use cache when available (default)
python run_pipeline.py --use-cached

# Force reprocessing (ignore cache)
python run_pipeline.py --no-cache

# Clear cache manually
rm cache/*.pkl
```

### When to Clear Cache
- After updating configuration files
- After modifying selection cuts
- After discovering data issues
- When you want fresh results

## Error Handling

### Common Issues

**Issue:** "No cached data found"
```
Solution: Run the pipeline to generate cache: python run_pipeline.py --years 2016
```

**Issue:** "Data root directory not found"
```
Solution: Check config/paths.toml and update data paths
```

**Issue:** "Memory error during loading"
```
Solution: Process one year at a time: --years 2016
```

**Issue:** "RooFit segmentation fault"
```
Solution: This is a known ROOT issue. Try:
  1. Reduce data size (single year)
  2. Run infrastructure tests instead
  3. Update ROOT version
```

## Data Requirements

### Expected File Structure
```
├── data
│   │   ├── dataBu2L0barPHH_16MD.root
│   │   ├── dataBu2L0barPHH_16MU.root
│   │   ├── dataBu2L0barPHH_17MD.root
│   │   ├── dataBu2L0barPHH_17MU.root
│   │   ├── dataBu2L0barPHH_18MD.root
│   │   └── dataBu2L0barPHH_18MU.root
└── mc
        ├── chic0
    │       │   ├── chic0_16_MD.root
    │       │   ├── chic0_16_MU.root
    │       │   ├── chic0_17_MD.root
    │       │   ├── chic0_18_MD.root
    │       │   └── chic0_18_MU.root
        ├── chic1
    │       │   ├── chic1_16_MD.root
    │       │   ├── chic1_16_MU.root
    │       │   ├── chic1_17_MD.root
    │       │   ├── chic1_17_MU.root
    │       │   ├── chic1_18_MD.root
    │       │   └── chic1_18_MU.root
        ├── chic2
    │       │   ├── chic2_16_MD.root
    │       │   ├── chic2_16_MU.root
    │       │   ├── chic2_17_MD.root
    │       │   ├── chic2_17_MU.root
    │       │   ├── chic2_18_MD.root
    │       │   └── chic2_18_MU.root
        ├── etac
    │       │   ├── etac_16_MD.root
    │       │   ├── etac_16_MU.root
    │       │   ├── etac_17_MD.root
    │       │   ├── etac_17_MU.root
    │       │   ├── etac_18_MD.root
    │       │   └── etac_18_MU.root
        ├── Jpsi
    │       │   ├── Jpsi_16_MD.root
    │       │   ├── Jpsi_16_MU.root
    │       │   ├── Jpsi_17_MD.root
    │       │   ├── Jpsi_17_MU.root
    │       │   ├── Jpsi_18_MD.root
    │       │   └── Jpsi_18_MU.root
        └── KpKm
                ├── KpKm_16_MD.root
                ├── KpKm_16_MU.root
                ├── KpKm_17_MD.root
                ├── KpKm_17_MU.root
                ├── KpKm_18_MD.root
                ├── KpKm_18_MU.root
```

### File Contents
Each ROOT file should contain:
- TTree: `B2L0barPKpKm_LL/DecayTree` (or `_DD`)
- Branches: Lambda 4-momentum, bachelor tracks, Bu variables
- See `config/branch_config.toml` for complete list

## Output Files

### Tables (CSV format)
- `phase5_yields.csv` - Fitted yields
- `efficiencies.csv` - Selection efficiencies
- `efficiency_ratios.csv` - ε_J/ψ / ε_state
- `branching_fraction_ratios.csv` - Final BR ratios
- `yield_consistency.csv` - N/(L×ε) per year
- `optimized_cuts.csv` - Optimal cuts (if optimization run)

### Plots (pdf format)
- `fit_*.pdf` - Mass fit results
- `yield_consistency_check.pdf` - Consistency across years

### Results (Markdown)
- `final_results.md` - Complete analysis summary with:
  - BR ratio results
  - Comparison with theory
  - Statistical uncertainties
  - Next steps for full analysis

## Workflow Examples

### First Time Running
```bash
# 1. Test with single year (fast)
make pipeline-2016
# or
python run_pipeline.py --years 2016

# 2. View results
make show-results

# 3. If successful, run full pipeline (all years)
make pipeline
# or
python run_pipeline.py --years 2016,2017,2018
```

### Re-running After Updates
```bash
# Clear cache and reprocess
make clean-cache
make pipeline

# Or force reprocessing directly
python run_pipeline.py --no-cache --years 2016,2017,2018
```

### Quick Test
```bash
# Test with single year
make pipeline-2016

# View outputs
make show-yields
make show-efficiencies
make show-results
```

### Production Run
```bash
# Full analysis with all data
python run_pipeline.py --skip-optimization
```

## Performance Tips

1. **Use caching:** Default behavior, saves hours of reprocessing
2. **Start small:** Test with one year first
3. **Skip optimization:** Use default cuts for draft analysis
4. **Monitor memory:** Large datasets may need subset processing
5. **Parallel processing:** Not yet implemented, but possible for Phase 3

## Next Steps

After completing the pipeline:

1. **Review results:** Check `results/final_results.md`
2. **Validate fits:** Inspect fit plots in `plots/`
3. **Check consistency:** Review yield consistency across years
4. **Add systematics:** Implement systematic uncertainty studies
5. **Full analysis:** Add reconstruction, PID, trigger efficiencies

## Standalone Plotting Scripts

Scripts for visualization are located in the `scripts/` directory.

### Lambda Mass Distribution Plotter

**Purpose:** Visualize Lambda mass distributions (full range, not just cut region)

**Script:** `scripts/plot_lambda_mass.py`

**What it does:**
- Loads MC and data for specified years
- Shows full Lambda mass distribution (no mass cut applied)
- Creates side-by-side plots (MC left, data right)
- Indicates cut windows and signal regions with vertical lines
- Generates separate PDFs per year + combined

**Usage:**
```bash
# From ana/scripts directory
cd scripts

# Plot all years (default: 2016, 2017, 2018)
python plot_lambda_mass.py

# Plot specific years
python plot_lambda_mass.py --years 2016,2017

# Use different MC state for comparison
python plot_lambda_mass.py --mc-state etac
```

**Output:**
```
plots/lambda_mass/
├── lambda_mass_2016_Jpsi.pdf
├── lambda_mass_2017_Jpsi.pdf
├── lambda_mass_2018_Jpsi.pdf
└── lambda_mass_combined_Jpsi.pdf
```

**Plot features:**
- MC (left): Shows signal shape across full mass range
- Data (right): $B^+ \to \bar{\Lambda} p K^+ K^-$ decay
- Red dashed lines: Cut boundaries [1111, 1121] MeV
- Green shaded region: Signal window used in analysis
- Full distribution shown to visualize background outside cuts

**Typical results:**
- MC: ~243k events total (clean Lambda peak)
- Data: ~1.28M events total (Lambda peak with background)

---

### Optimization Variables Distribution Plotter

**Purpose:** Visualize distributions of the 7 optimization cut variables with optimal cut values

**Script:** `scripts/plot_optimization_variables.py`

**What it does:**
- Loads MC and data for specified years
- Plots distributions of all 7 optimization variables:
  - `Bu_PT`: B+ transverse momentum
  - `Bu_FDCHI2_OWNPV`: B+ flight distance χ²
  - `Bu_IPCHI2_OWNPV`: B+ impact parameter χ²
  - `Bu_DTF_chi2`: B+ decay tree fit χ²
  - `h1_ProbNNk`: K+ PID probability
  - `h2_ProbNNk`: K- PID probability
  - `p_ProbNNp`: Bachelor proton PID probability
- Shows optimal cut values from ALL 4 states (J/ψ, ηc, χc0, χc1) as vertical lines
- Shades accepted regions (green transparent)
- Creates side-by-side plots (MC left, data right)
- Generates **one PDF per variable** (7 PDFs per year/combined)
- Organized in year-specific folders

**Usage:**
```bash
# From ana/scripts directory
cd scripts

# Plot all years with optimal cuts from all states (default)
python plot_optimization_variables.py

# Plot specific years
python plot_optimization_variables.py --years 2016,2017

# Use different MC state for distributions
python plot_optimization_variables.py --mc-state etac
```

**Output:**
```
plots/optimization_variables/
├── 2016/
│   ├── Bu_PT_2016.pdf
│   ├── Bu_FDCHI2_OWNPV_2016.pdf
│   ├── Bu_IPCHI2_OWNPV_2016.pdf
│   ├── Bu_DTF_chi2_2016.pdf
│   ├── h1_ProbNNk_2016.pdf
│   ├── h2_ProbNNk_2016.pdf
│   └── p_ProbNNp_2016.pdf
├── 2017/
│   └── [7 PDFs]
├── 2018/
│   └── [7 PDFs]
└── combined/
    ├── Bu_PT_combined.pdf
    ├── Bu_FDCHI2_OWNPV_combined.pdf
    ├── Bu_IPCHI2_OWNPV_combined.pdf
    ├── Bu_DTF_chi2_combined.pdf
    ├── h1_ProbNNk_combined.pdf
    ├── h2_ProbNNk_combined.pdf
    └── p_ProbNNp_combined.pdf
```
