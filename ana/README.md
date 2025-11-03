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


### Run Individual Phases (with Automatic Dependencies)
```bash
# Phase 2: Load data/MC with Lambda cuts
python run_phase.py 2 --years 2016,2017,2018

# Phase 5: Mass fitting (auto-runs Phase 2 if needed)
python run_phase.py 5

# Phase 6: Efficiency calculation (auto-runs Phase 2 if needed)
python run_phase.py 6

# Phase 7: Branching ratios (auto-runs Phases 2, 5, 6 if needed)
python run_phase.py 7

# Force reprocessing (ignore cache)
python run_phase.py 7 --no-cache
```

**Key Feature:** Each phase automatically checks for and runs prerequisite phases if their cached results are not found. You can run any phase directly without worrying about dependencies!

## Automatic Dependency Resolution

The pipeline now features **automatic dependency resolution**. Each phase checks if prerequisite phases have been run and automatically executes them if needed.

### Phase Dependencies
```
Phase 2: None (loads raw data/MC)
         ↓
Phase 3: Phase 2 (optional optimization)
         ↓
Phase 5: Phase 2 (fits data mass spectra)
         ↓
Phase 6: Phase 2 (calculates MC efficiencies)
         ↓
Phase 7: Phases 5 + 6 (computes BR ratios)
```

### Example Workflows

**Scenario 1: Fresh Start**
```bash
# Just run Phase 7 - it will automatically run 2, 5, and 6 first
make phase7
```

**Scenario 2: Reprocess Only Fitting**
```bash
# Delete Phase 5 cache, then run Phase 7
rm cache/phase5_fit_results.pkl
make phase7  # Will rerun Phase 5, skip 2 and 6 (cache exists)
```

**Scenario 3: Full Reprocessing**
```bash
# Clear all cache
make clean-cache

# Run any phase - all prerequisites will be executed
make phase7  # Runs 2 → 5 → 6 → 7
```

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
- `plots/optimization/*.png` - FOM scan plots

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
- `plots/yield_consistency_check.png` - Consistency plot
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
Solution: Run Phase 2 first: python run_phase.py 2
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
- `yield_consistency_check.png` - Consistency across years

### Results (Markdown)
- `final_results.md` - Complete analysis summary with:
  - BR ratio results
  - Comparison with theory
  - Statistical uncertainties
  - Next steps for full analysis

## Workflow Examples

### First Time Running
```bash
# 1. Validate everything works with small dataset
python run_phase.py 2 --years 2016 --track-types LL

# 2. If successful, run full Phase 2
python run_phase.py 2

# 3. Skip optimization, use default cuts
# 4. Run fitting on data
python run_phase.py 5 --use-cached

# 5. Calculate efficiencies from MC
python run_phase.py 6 --use-cached

# 6. Compute final BR ratios
python run_phase.py 7
```

### Re-running After Updates
```bash
# Clear cache and reprocess
rm cache/*.pkl
python run_pipeline.py --no-cache
```

### Quick Test on Subset
```bash
# Test with 2016 LL only (fast)
python run_phase.py 2 --years 2016 --track-types LL
python run_phase.py 5 --use-cached
python run_phase.py 6 --use-cached
python run_phase.py 7
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

## Support

For issues or questions:
1. Check this README
2. Review `plan.md` for detailed phase specifications
3. Run individual phase tests in `tests/test_phase*.py`
4. Check intermediate cache files for debugging
