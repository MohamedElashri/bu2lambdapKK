# Selection Optimization Study for B+ → pK⁻Λ̄ K+

**Grid Search Optimization for J/ψ Signal Selection**

A two-phase selection optimization study that maximizes S/√B on MC (Phase 1), then applies optimal cuts to data (Phase 2) with comprehensive cut visualizations.

---

## Overview

This study implements an automated **grid search optimization** to find optimal selection cuts for J/ψ signal in the B+ → pK⁻Λ̄ K+ decay:

### **Two-Phase Workflow**

1. **Phase 1: MC Optimization** 
   - Scan cut values on J/ψ signal MC
   - Calculate S/√B for each cut combination
   - Find optimal cuts that maximize S/√B
   - Generate cut visualization plots showing where cuts are applied

2. **Phase 2: Data Application**
   - Apply optimal cuts (from Phase 1) to real data
   - Extract signal and background yields
   - Generate cut visualization plots on data
   - Produce final J/ψ mass spectrum

### **Key Features**

✅ **Grid Search Optimization**: Automated 1D scans for each variable  
✅ **2D Grid Optimization**: NEW! Comprehensive multi-dimensional optimization  
✅ **S/√B Maximization**: Physics-motivated figure of merit  
✅ **Cut Visualizations**: See exactly where cuts are applied (MC and Data)  
✅ **Centralized Plotting**: Consistent LHCb styling via `plot.py`  
✅ **MC vs Data Comparison**: Validate cuts with side-by-side optimization  
✅ **No Hardcoded Cuts**: All cuts determined from data-driven optimization  

---

## Quick Start

```bash
# Run the full two-phase study
./run_study.sh

# Or run directly with Python
python main.py

# With custom config
python main.py --config my_config.toml
```

**Output**: 25 PDF plots + tables in `output/` directory

---

## Study Structure

```
selection/
├── main.py                          # Main study coordinator
├── plot.py                          # Centralized plotting (LHCb style)
├── jpsi_analyzer.py                 # J/ψ region analysis
├── variable_analyzer.py             # Variable distribution analysis
├── selection_efficiency.py          # Efficiency calculations
├── grid_optimizer.py                # NEW! 2D grid search optimization
├── config.toml                      # Configuration file
├── run_study.sh                     # Execution script with checks
├── README.md                        # This file
├── GRID_OPTIMIZATION.md             # NEW! 2D grid optimization documentation
└── output/                          # Generated outputs
    ├── mc/                          # Phase 1 outputs
    │   ├── cut_tables/
    │   │   ├── optimization_scan_full.csv
    │   │   ├── optimal_cuts_summary.txt
    │   │   └── optimization_results.md
    │   ├── grid_optimization/       # NEW! 2D grid results
    │   │   ├── grid_optimization_mc.csv
    │   │   ├── grid_optimization_data.csv
    │   │   ├── top_20_cuts_mc.txt
    │   │   ├── top_20_cuts_data.txt
    │   │   ├── grid_optimization_mc.md
    │   │   ├── grid_optimization_data.md
    │   │   └── grid_comparison/
    │   │       ├── s_over_sqrtb_comparison.pdf
    │   │       ├── best_cuts_comparison.pdf
    │   │       └── efficiency_rejection_scatter.pdf
    │   └── cut_visualizations/      # 12 MC cut plots
    │       ├── lambda_fdchi2_cut_visualization.pdf
    │       ├── p_probnnp_cut_visualization.pdf
    │       └── ...
    └── data/                        # Phase 2 outputs
        ├── cut_tables/
        │   ├── data_cuts_applied.csv
        │   └── data_yields_summary.txt
        ├── cut_visualizations/      # 12 Data cut plots
        │   ├── lambda_fdchi2_cut_visualization.pdf
        │   └── ...
        └── jpsi_analysis/
            └── jpsi_mass_data_after_cuts.pdf
```

```

---

## Configuration

All settings in `config.toml`:

### 1. Study Workflow

```toml
[study_workflow]
run_mc_optimization = true     # Phase 1: Grid search on MC
run_data_application = true    # Phase 2: Apply cuts to data
```

### 2. Data and MC Selection

```toml
[data_selection]
years = ["16", "17", "18"]
polarities = ["MD", "MU"]
track_types = ["LL", "DD"]
channel_name = "B2L0barPKpKm"

[mc_selection]
signal_sample = "Jpsi"         # J/ψ signal MC (SS + OS combined)
channel_name = "B2L0barPKpKm"
```

### 3. J/ψ Study Regions

```toml
[study_regions]
jpsi_range = [3000, 3200]        # Full region around J/ψ
jpsi_window = [3070, 3120]       # Signal window (±2σ)
sideband_left = [3000, 3050]     # Left sideband
sideband_right = [3150, 3200]    # Right sideband
```

### 4. Variables to Optimize

The study optimizes **12 variables** across three categories:

#### Lambda Variables (Λ̄ reconstruction quality)
- `lambda_fdchi2`: Λ flight distance χ²
- `delta_z`: ΔZ (L0_Z - Bu_Z), must be positive
- `lp_probnnp`: Lambda daughter proton PID

#### PID Variables (THE CROWN JEWEL 👑)
- `p_probnnp`: Proton PID probability
- `h1_probnnk`: Kaon 1 PID probability
- `h2_probnnk`: Kaon 2 PID probability
- `kk_product`: Combined K×K PID product
- `pid_product`: Combined p×K₁×K₂ PID product

#### B+ Variables (B+ quality)
- `bu_pt`: B+ transverse momentum
- `bu_dtf_chi2`: B+ DTF χ²
- `bu_ipchi2`: B+ impact parameter χ²
- `bu_fdchi2`: B+ flight distance χ²

Each variable has:
```toml
[lambda_variables.delta_z]
branch = "delta_z"
description = "ΔZ (L0_Z - Bu_Z), must be positive"
scan_range = [0, 30]           # Range to scan
scan_steps = 51                # Number of scan points
operator = ">"                 # Cut operator (>, <, >=, <=)
plot_range = [0, 30]           # Range for plotting
enabled = true                 # Include in optimization
```

### 5. Plot Settings

```toml
[plot_settings]
style = "LHCb2"
dpi = 300
format = "pdf"

[plot_settings.colors]
jpsi_signal = "#E41A1C"        # Red
data = "#000000"               # Black
data_sideband = "#377EB8"      # Blue
optimal_cut = "#9467BD"        # Purple

[plot_settings.labels]
# LaTeX-formatted axis labels
p_probnnp = "Proton ProbNNp"
jpsi_mass = "$M(pK^-\\bar{\\Lambda})$ [MeV/$c^2$]"
```

---

## 2D Grid Search Optimization

In addition to the 1D optimization, the study now includes a **comprehensive 2D grid search** that tests all combinations of cuts simultaneously.

### What's New?

- **Multi-dimensional optimization**: Tests all possible combinations of cut values
- **MC and Data tables**: Generates optimization tables for both datasets
- **Comparison plots**: Visualize differences between MC and Data optima
- **Ranked results**: Tables sorted by S/√B with top combinations highlighted

### Output Tables

The 2D grid optimization creates tables where:
- **Rows** = Different combinations of cuts
- **Columns** = Variables (with cut values) + metrics
- **Last columns** = Signal, Background, S/√B, efficiencies

Example table structure:

| p_probnnp | h1_probnnk | pid_product | bu_pt | Signal | Background | S/√B | Sig Eff | Bkg Rej |
|-----------|------------|-------------|-------|--------|------------|------|---------|---------|
| 0.550 | 0.250 | 0.600 | 3200 | 1198.0 | 198.0 | 85.14 | 82% | 79% |
| 0.600 | 0.250 | 0.550 | 3200 | 1156.0 | 182.0 | 85.69 | 79% | 81% |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

### Configuration

Enable/disable and control in `config.toml`:

```toml
[optimization]
perform_2d_grid = true         # Enable 2D grid optimization
grid_scan_steps = 10           # Points per variable (10^n_vars combinations!)
```

⚠️ **Warning**: Computation grows exponentially!
- 5 variables × 10 steps = 100,000 combinations (~2 min)
- 5 variables × 15 steps = 759,375 combinations (~10 min)
- 5 variables × 20 steps = 3,200,000 combinations (~1 hour)

### Files Generated

In `output/mc/grid_optimization/`:

**CSV Tables:**
- `grid_optimization_mc.csv` - Full MC optimization results
- `grid_optimization_data.csv` - Full Data optimization results

**Summaries:**
- `top_20_cuts_mc.txt` - Top 20 cut combinations for MC
- `top_20_cuts_data.txt` - Top 20 cut combinations for Data

**Markdown:**
- `grid_optimization_mc.md` - Human-readable MC results
- `grid_optimization_data.md` - Human-readable Data results

**Comparison Plots** (in `grid_comparison/`):
- `s_over_sqrtb_comparison.pdf` - S/√B distributions
- `best_cuts_comparison.pdf` - Optimal cut values MC vs Data
- `efficiency_rejection_scatter.pdf` - Efficiency vs rejection trade-off

### Reading the Results

The tables are sorted by S/√B (highest first):

1. **Rank 1** = Best overall combination
2. **Rank 2-20** = Close alternatives

---

## Phase 1: MC Optimization (Grid Search)

### What It Does

1. **Loads J/ψ signal MC** into signal window and sidebands
2. **Grid search for each variable**:
   - Scan cut values across configured range
   - For each cut value, calculate:
     - S (signal events passing cut)
     - B (background events passing cut)
     - S/√B (figure of merit)
     - Signal efficiency
     - Background rejection
3. **Find optimal cut** that maximizes S/√B
4. **Save results**:
   - Full scan table: `optimization_scan_full.csv`
   - Optimal cuts: `optimal_cuts_summary.txt`
   - Markdown report: `optimization_results.md`
5. **Generate cut visualizations**:
   - One plot per variable showing:
     - Signal MC distribution (red)
     - Background MC distribution (blue)
     - Optimal cut line (purple dashed)
     - Accepted region (green transparent box)
     - S/√B, efficiency, rejection metrics

### Example Output

```
=== Cut Optimization via Grid Search ===
Signal events (J/ψ window): 49,467
Background events (sidebands): 4,198

Scanning 12 variables:
  lambda_fdchi2: [0, 500] (20 steps)
  delta_z: [0, 30] (51 steps)
  p_probnnp: [0.0, 1.0] (51 steps)
  ...

Optimization complete! Found optimal cuts for 12 variables.

Creating MC Cut Visualization Plots:
  Plotting lambda_fdchi2 (L0_FDCHI2_OWNPV > 78.9474)
  Plotting p_probnnp (p_ProbNNp > 0.7400)
  ...
Saved 12 cut visualization plots to output/mc/cut_visualizations
```

### Optimal Cuts Table (Example)

```
OPTIMAL CUTS FROM GRID SEARCH (Maximizing S/√B)

p_probnnp:
  Optimal cut value: 0.7400
  S/√B at optimum: 31.45
  Signal efficiency: 87.3%
  Background rejection: 94.2%
  Signal (S): 43,156.0
  Background (B): 189.0
```

---

## Phase 2: Data Application

### What It Does

1. **Loads real data** in J/ψ region
2. **Applies optimal cuts sequentially**:
   - Start with all data in J/ψ region
   - Apply each cut one by one
   - Track efficiency at each step
3. **Calculate final yields**:
   - Events in signal window
   - Background estimate from sidebands
   - Signal purity
4. **Save results**:
   - Cut summary: `data_cuts_applied.csv`
   - Yields: `data_yields_summary.txt`
5. **Generate visualizations**:
   - Cut visualization plots (data distributions with cut lines)
   - Final J/ψ mass spectrum after all cuts

### Example Output

```
=== Applying Optimal Cuts to Data ===
Starting with 50,977 events in J/ψ region

Applying optimal cuts:
  lambda_fdchi2: L0_FDCHI2_OWNPV > 78.9474
    Events: 50,977 → 41,398 (eff: 81.21%)
  p_probnnp: p_ProbNNp > 0.7400
    Events: 21,569 → 6,768 (eff: 31.38%)
  ...

Final Data Yields:
  Initial events: 50,977
  After optimal cuts: 517
  Overall efficiency: 1.01%

Signal Window [3070-3120 MeV]:
  Events in signal window: 186
  Expected background: 118.5
  Estimated signal: 67.5
  Signal purity: 36.3%
```

---

## Output Files

### Tables

#### `mc/cut_tables/optimization_scan_full.csv`
Complete grid search results for all variables

#### `mc/cut_tables/optimal_cuts_summary.txt`
Best cut for each variable with performance metrics

#### `mc/cut_tables/optimization_results.md`
Markdown-formatted tables for easy viewing

#### `data/cut_tables/data_cuts_applied.csv`
Sequential cut application on data with efficiencies

#### `data/cut_tables/data_yields_summary.txt`
Final yields and signal purity

### Plots

#### MC Cut Visualizations (`mc/cut_visualizations/*.pdf`)
- 12 plots showing MC distributions with optimal cuts
- **Red**: J/ψ signal MC
- **Blue**: Sideband background
- **Purple dashed line**: Optimal cut value
- **Green box**: Accepted region
- **Info box**: S/√B, efficiency, rejection

#### Data Cut Visualizations (`data/cut_visualizations/*.pdf`)
- 12 plots showing data distributions with optimal cuts
- **Black**: Real data
- **Purple dashed line**: Optimal cut (from MC)
- **Green box**: Accepted region
- **Info box**: Cut performance from MC optimization

#### J/ψ Mass Spectrum (`data/jpsi_analysis/jpsi_mass_data_after_cuts.pdf`)
- Final mass spectrum after all cuts
- Signal window highlighted
- Sideband regions shown

---

## Plotting Infrastructure

### Centralized in `plot.py`

All plotting uses the `StudyPlotter` class with:

✅ **Consistent LHCb styling**: `mplhep.style.LHCb2`  
✅ **Proper fonts**: Serif (Computer Modern Roman), size 14-18  
✅ **Publication quality**: 300 DPI, vector PDF format  
✅ **Automatic styling**: Font sizes, line widths, colors configured once  

```python
class StudyPlotter:
    def plot_cut_visualizations_mc(...)      # MC cut plots
    def plot_cut_visualizations_data(...)    # Data cut plots
    def plot_data_mass_spectrum(...)         # Mass spectrum
    def plot_cutflow(...)                    # Cutflow comparison
    def plot_2d_correlation(...)             # 2D efficiency maps
```

### Font Configuration

```python
Font settings applied to ALL plots:
- Base font size: 14
- Axis labels: 16
- Titles: 18
- Legend: 13
- Font family: Serif (Computer Modern Roman)
- Line widths: 2.5 (data), 1.5 (axes)
- LaTeX math: Computer Modern
```

**No more inconsistent fonts or inline plotting code!**

---

## Implementation Details

### Module Structure

```
main.py (SelectionStudy)
├── Coordinates two-phase workflow
├── Calls grid search optimization
└── Manages output directories

plot.py (StudyPlotter)
├── All plotting functions
├── Font configuration
└── LHCb styling

jpsi_analyzer.py (JPsiAnalyzer)
├── Mass calculation
├── Region application (signal/sidebands)
└── Purity calculations

variable_analyzer.py (VariableAnalyzer)
├── Variable distribution analysis
└── Cut efficiency calculations

selection_efficiency.py (EfficiencyCalculator)
├── Single cut efficiency
├── Efficiency scans
└── Cutflow generation
```

---

## Extending the Study

### Add New Variables

1. **Add to config** (`config.toml`):

```toml
[bplus_variables.new_variable]
branch = "Bu_NewBranch"
description = "Description of new variable"
operator = ">"
scan_range = [0, 100]
scan_steps = 51
plot_range = [0, 120]
enabled = true
```

2. **Run study** - automatic optimization!

### Study Other Resonances

Change the mass regions in config:

```toml
[study_regions]
# For ψ(2S): M ≈ 3686 MeV
jpsi_range = [3600, 3800]
jpsi_window = [3666, 3706]
sideband_left = [3600, 3650]
sideband_right = [3750, 3800]
```

---

## Physics Notes

### S/√B Figure of Merit

**Why S/√B?**
- Standard in HEP for optimizing cuts
- Balance between signal efficiency and background rejection
- Maximizing S/√B ≈ maximizing discovery significance

### Delta_z Sign

The `delta_z` variable **must be positive**:
```
delta_z = z_decay(Λ) - z_production(B+)
```
- Λ travels forward before decaying → positive ΔZ
- Negative values indicate reconstruction issues

### Background Estimation

Background in signal window estimated from sidebands:
```python
bkg_in_signal = N_sidebands × (width_signal / width_sidebands)
```

Assumes flat background across J/ψ region.

---

## Troubleshooting

### Low Statistics Warning

**Issue**: Few events after cuts  
**Solution**: Check optimal cut values in `optimal_cuts_summary.txt`

### Branch Not Found

**Issue**: `Branch X not found in data`  
**Solution**: Verify branch names in config match ROOT branches

### Fonts Look Wrong

**Issue**: Plots have inconsistent fonts  
**Solution**: ✅ Already fixed! All plotting goes through `plot.py`

---

## Dependencies

### Python Packages
- `awkward` ≥ 2.0
- `uproot` ≥ 5.0
- `numpy`, `pandas`
- `matplotlib`, `mplhep`
- `toml`

---

## Quick Reference Commands

```bash
# Full study
./run_study.sh

# Check outputs
ls -R output/

# View optimal cuts
cat output/mc/cut_tables/optimal_cuts_summary.txt

# View data yields
cat output/data/cut_tables/data_yields_summary.txt
```

---

**Version**: 2.0  
**Last Updated**: October 28, 2025  
**Author**: Mohamed Elashri  
**Study Focus**: J/ψ selection optimization via grid search

## Outputs

### Plots Generated

1. **Distribution Plots** (`*_dist.pdf`):
   - Show variable distributions for J/ψ signal MC, KpKm background MC, and real data
   - Mark current tight/loose cut values

2. **Efficiency Scans** (`*_effscan.pdf`):
   - Plot efficiency vs cut value
   - Show current cuts and their efficiencies

3. **2D Efficiency Maps** (`pid_2d_*.pdf`):
   - Show combined efficiency for two variables
   - Contour lines at 50%, 60%, 70%, 80%, 90% efficiency

4. **Mass Spectrum** (`jpsi_mass.pdf`):
   - M(pK⁻Λ̄) distribution in J/ψ region
   - Compare signal MC, background MC, and data
   - Mark signal window and sidebands

### Log File

The `selection_study.log` contains:
- Detailed execution information
- Event counts at each stage
- Efficiency metrics
- Signal purity calculations
- S/B ratios

Example output:
```
Phase 2: PID Selection Study - THE CROWN JEWEL
Studying variable: p_ProbNNp
  Branch: p_ProbNNp
  Operator: >
  Tight cut (0.5): ε = 87.3%
  Loose cut (0.2): ε = 95.8%
  
Signal Purity Metrics:
  Signal window [3070-3120 MeV]:
    Signal events: 15234
    Background events: 1842
    Purity: 89.23%
    S/B ratio: 8.27
```

## Running Individual Phases

You can run specific study phases independently:

```bash
# Phase 1: Lambda selection
python3 selection_study.py --phase lambda

# Phase 2: PID selection (THE CROWN JEWEL)
python3 selection_study.py --phase pid

# Phase 3: B+ quality
python3 selection_study.py --phase bplus

# Phase 4: J/ψ region
python3 selection_study.py --phase jpsi

# All phases (default)
python3 selection_study.py --phase all
```

## Extending to Other Resonances

This study is designed to be easily extended to other charmonium resonances (ψ(2S), χc states, etc.):

### Step 1: Update Configuration

Edit `config.toml`:

```toml
[jpsi_study]
# For ψ(2S): M ≈ 3686 MeV
jpsi_range = [3600, 3800]
jpsi_window = [3666, 3706]
left_sideband = [3600, 3650]
right_sideband = [3750, 3800]

# Update MC samples if available
[mc_samples]
signal_samples = ["Psi2S_SS", "Psi2S_OS"]
background_samples = ["KpKm"]
```

### Step 2: Update Metadata

```toml
[metadata]
study_name = "Psi2S_Selection_Study"
description = "Selection optimization for ψ(2S) → pK⁻Λ̄ region"
```

### Step 3: Run Study

```bash
./run_study.sh
```

The same analysis framework will work for any resonance in the M(pK⁻Λ̄) spectrum!

## Physics Notes

### Delta_z Sign Convention

The `delta_z` variable (vertex separation) **must be positive** by physics construction:
- delta_z = z_decay(Λ) - z_production(B+)
- Λ travels forward before decaying
- Negative values indicate reconstruction issues

The study enforces `enforce_positive = true` in the configuration.

### B+ Mass Window

The B+ mass window cut can be toggled:

```toml
[bplus_selection.Bu_mass_window]
enabled = false  # Set to true to apply B+ mass cut
```

This allows studying:
- **enabled = false**: Full M(pK⁻Λ̄) spectrum, all B+ candidates
- **enabled = true**: Only B+ mass window, reduced backgrounds

### MC Sample Combination

J/ψ signal MC combines two samples:
- **Jpsi_SS**: J/ψ → p⁺p⁻ (same-sign, charge conjugation state)
- **Jpsi_OS**: J/ψ → p⁺p̄ (opposite-sign, dominant state)

Both are loaded and concatenated to represent the full signal.

## Implementation Details

### Class Structure

```
SelectionStudy (main coordinator)
├── EfficiencyCalculator
│   ├── calculate_single_cut()
│   ├── scan_efficiency()
│   ├── find_optimal_cut()
│   └── generate_cutflow()
├── VariableAnalyzer
│   ├── plot_distribution()
│   ├── plot_efficiency_scan()
│   └── plot_2d_efficiency()
├── StudyPlotter
│   ├── plot_cutflow()
│   └── plot_signal_to_background()
└── JPsiAnalyzer
    ├── calculate_mass()
    ├── apply_jpsi_region()
    ├── apply_signal_window()
    ├── apply_sidebands()
    ├── plot_mass_spectrum()
    └── calculate_purity()
```

### Dependencies

The study uses existing infrastructure:
- `DataLoader`: Load real data by year/polarity/track type
- `MCLoader`: Load MC samples by name
- `SelectionProcessor`: Apply selection cuts
- `MassCalculator`: Calculate M(pK⁻Λ̄) invariant mass
- `BranchConfig`: Branch name configuration

## Troubleshooting

### Issue: Missing branches in data

**Error:** `Branch X not found in data`

**Solution:** Check branch names in configuration match actual ROOT branch names. Use `branch_config.py` to verify available branches.

### Issue: Low statistics in plots

**Warning:** `No data found for branch X`

**Solution:** 
1. Verify data loading succeeded (check log file)
2. Ensure preselection isn't too tight (modify `selection.toml`)
3. Check year/polarity/track type combinations exist

### Issue: Delta_z negative values

**Warning:** `Delta_z has X negative values`

**Solution:** This is expected if reconstruction has issues. The study validates and reports this. Consider tightening Λ quality cuts.

### Issue: Low J/ψ signal purity

**Result:** S/B < 2 in signal window

**Solution:**
1. Tighten PID cuts (Phase 2 - THE CROWN JEWEL)
2. Apply stricter Λ quality cuts (Phase 1)
3. Tighten B+ vertex quality (Phase 3)
4. Consider narrower signal window

## Performance Tips

### Speed Up Development

1. **Use subsamples:** Modify data loader to use fewer events during testing
2. **Reduce scan points:** Use `scan_points = 20` instead of 50
3. **Disable plots:** Comment out plotting calls for quick efficiency scans
4. **Run single phase:** Use `--phase pid` to test one phase

### Memory Optimization

For large datasets:
1. Process years individually
2. Use only required branches (configure in `branch_config.py`)
3. Apply loose preselection before study

## References

- Study plan: `plan.md` (comprehensive 15-section document)
- Branch configuration: `../../branch_config.py`
- Selection implementation: `../../selection.py`
- MC sample definitions: `../../mc_loader.py`

## Contact

For questions or issues with this study:
- Check `plan.md` for detailed methodology
- Review log file `output/selection_study.log`
- Examine generated plots in `output/`

---

**Version:** 1.0  
**Last Updated:** October 28, 2025  
**Study Focus:** J/ψ → pK⁻Λ̄ in B+ → pK⁻Λ̄ K+
