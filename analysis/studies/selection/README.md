# Selection Optimization Study for B+ → pK⁻Λ̄ K+

**Focus: J/ψ Signal vs Background Discrimination**

This study implements a comprehensive hierarchical selection optimization for the B+ → pK⁻Λ̄ K+ analysis, focusing on maximizing J/ψ signal purity while maintaining high efficiency.

## Overview

The study follows a hierarchical approach to selection optimization:

1. **Phase 1: Lambda Quality** - Ensure good Λ reconstruction
2. **Phase 2: PID Selection** - Maximize signal/background discrimination
3. **Phase 3: B+ Quality** - Select well-reconstructed B+ candidates
4. **Phase 4: J/ψ Region Analysis** - Study signal purity in J/ψ mass region

## Quick Start

```bash
# Run the full study
./run_study.sh

# Or with custom config
./run_study.sh my_config.toml

# Run specific phase only
python3 selection_study.py --phase pid
```

## Study Structure

```
selection/
├── selection_study_config.toml   # Configuration file
├── selection_study.py             # Main implementation
├── run_study.sh                   # Execution script
├── README.md                      # This file
├── plan.md                        # Detailed study plan
└── output/                        # Generated outputs
    ├── selection_study.log        # Execution log
    ├── lambda_*.pdf               # Phase 1 plots
    ├── pid_*.pdf                  # Phase 2 plots 
    ├── bplus_*.pdf                # Phase 3 plots
    └── jpsi_mass.pdf              # Phase 4 mass spectrum
```

## Configuration

The study is configured via `selection_study_config.toml`. Key sections:

### 1. Data and MC Samples

```toml
[mc_samples]
signal_samples = ["Jpsi_SS", "Jpsi_OS"]     # J/ψ signal MC
background_samples = ["KpKm"]                # Non-resonant background

[data_samples]
years = [2016, 2017, 2018]
polarities = ["MD", "MU"]
track_types = ["LL", "DD"]
channel = "B2L0barPKpKm"
```

### 2. Selection Variables

#### Phase 1: Lambda Variables
- `L0_FD_CHISQ`: Λ flight distance χ²
- `delta_z`: Vertex separation (must be positive!)
- `Lp_ProbNNp`: Proton PID from Λ
- `Lambda_mass_window`: Λ mass window cut

#### Phase 2: PID Variables
- `p_ProbNNp`: Prompt proton PID probability
- `h1_ProbNNk`: K⁻ PID probability (from pK⁻)
- `h2_ProbNNk`: K+ PID probability (bachelor)
- `kk_product`: Combined K⁻K+ PID
- `pid_product`: Combined p × K⁻ × K+ PID


#### Phase 3: B+ Variables
- `Bu_PT`: B+ transverse momentum
- `Bu_DTF_chi2`: B+ decay tree fit χ²
- `Bu_IPCHI2`: B+ impact parameter χ²
- `Bu_FDCHI2`: B+ flight distance χ²

### 3. J/ψ Study Region

```toml
[jpsi_study]
jpsi_range = [3000, 3200]          # Full study region (MeV)
jpsi_window = [3070, 3120]         # Signal window (MeV)
left_sideband = [3000, 3050]       # Left sideband (MeV)
right_sideband = [3150, 3200]      # Right sideband (MeV)
```

### 4. Cut Optimization

Each variable has:
- `current_tight`: Current tight cut value
- `current_loose`: Current loose cut value
- `scan_range`: Range to scan for optimization
- `scan_points`: Number of points in scan
- `operator`: Comparison operator (>, <, >=, <=)

Example:
```toml
[pid_selection.p_ProbNNp]
branch = "p_ProbNNp"
description = "Prompt proton PID"
operator = ">"
current_tight = 0.5
current_loose = 0.2
scan_range = [0.0, 0.95]
scan_points = 50
plot_range = [0.0, 1.0]
enabled = true
```

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

Edit `selection_study_config.toml`:

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
