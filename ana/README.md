# B⁺ → Λ̄pK⁻K⁺ Charmonium Analysis

Draft analysis for measuring branching fraction ratios of charmonium states (J/ψ, ηc(1S), χc0, χc1) in B⁺ → Λ̄pK⁻K⁺ decays.

## 🎯 Current Status

### ✅ Phase 0: COMPLETED
Data/MC loading infrastructure updated to use proven `BranchConfig` system from `analysis/` folder.

**Key improvements:**
- Automatic handling of data vs MC branch name differences
- Support for both LL and DD Lambda reconstruction categories
- Branch name normalization (your code uses common names)
- Proper ROOT file structure handling
- Derived branch calculations (Bu_MM_corrected, delta_z, M_LpKm, etc.)

### 🔄 Next: Phase 2 (Data Loading Execution)
Ready to load all data and MC files.

## Quick Start

### 1. Test the Implementation
```bash
cd ana
python test_phase0.py
```

This will verify:
- ✓ Imports work
- ✓ Configuration loads
- ✓ BranchConfig functions correctly
- ✓ File structure is accessible
- ✓ Derived branch logic is sound

### 2. Test Data Loading (Small Sample)
```python
from modules.data_handler import TOMLConfig, DataManager

config = TOMLConfig("./config")
dm = DataManager(config)

# Load one file
events = dm.load_tree("data", 2016, "MD", "LL")
print(f"Loaded {len(events)} events")
print(f"Fields: {list(events.fields)[:10]}")

# Check derived branches
print(f"Bu_MM_corrected: {events['Bu_MM_corrected'][:5]}")
print(f"M_LpKm_h1: {events['M_LpKm_h1'][:5]}")
```

### 3. Load All Data (Full Pipeline)
```python
# Load all years, combining magnets and track types
data_by_year = dm.load_all_data_combined_magnets("data")

# Load MC for all states
mc_jpsi = dm.load_all_data_combined_magnets("Jpsi")
mc_etac = dm.load_all_data_combined_magnets("etac")
mc_chic0 = dm.load_all_data_combined_magnets("chic0")
mc_chic1 = dm.load_all_data_combined_magnets("chic1")
mc_kpkm = dm.load_all_data_combined_magnets("KpKm")  # Phase space
```

## Project Structure

```
ana/
├── config/                      # TOML configuration files
│   ├── paths.toml              # File paths, years, magnets
│   ├── particles.toml          # PDG values, mass windows
│   ├── selection.toml          # Lambda cuts, optimization ranges
│   └── ...                     # Other config files
├── modules/
│   ├── data_handler.py         # ✓ Data/MC loading with BranchConfig
│   ├── lambda_selector.py      # ✓ Lambda selection cuts
│   ├── selection_optimizer.py  # ✓ FOM optimization
│   ├── mass_fitter.py          # TODO: RooFit mass fitting
│   ├── efficiency_calculator.py # TODO: Efficiency calculation
│   └── branching_fraction_calculator.py  # TODO: BR ratios
├── test_phase0.py              # ✓ Test script for Phase 0
├── MIGRATION_NOTES.md          # ✓ Documentation of Phase 0 changes
├── plan.md                     # ✓ Complete analysis plan
└── main_analysis.py            # Master execution script

# Reused from analysis/:
../analysis/
├── branches_config.toml        # Complete branch configuration
├── branch_config.py            # Branch configuration manager
├── data_loader.py              # Reference implementation
└── mc_loader.py                # Reference implementation
```

## Branch Name Examples

After loading, your analysis code uses these **common names** (BranchConfig handles the rest):

| Common Name | Description | Value Type |
|-------------|-------------|------------|
| `Bu_M` | B+ invariant mass | Float |
| `Bu_MM` | B+ mass (alternative) | Float |
| `Bu_PT` | B+ transverse momentum | Float |
| `L0_MM` | Lambda mass | Float |
| `L0_FDCHI2_OWNPV` | Lambda FD χ² | Float |
| `Lp_ProbNNp` | Lambda proton PID | Float |
| `p_ProbNNp` | Bachelor proton PID | Float |
| `h1_ProbNNk` | K± PID (normalized) | Float |
| `h2_ProbNNk` | K± PID (normalized) | Float |
| `Bu_MM_corrected` | Lambda-corrected B+ mass | Float (derived) |
| `delta_z` | Z vertex separation significance | Float (derived) |
| `M_LpKm_h1` | M(Λ̄p h1) invariant mass | Float (derived) |
| `M_LpKm_h2` | M(Λ̄p h2) invariant mass | Float (derived) |
| `M_KK` | M(K+K-) invariant mass | Float (derived) |

**Data/MC differences handled automatically**:
- Data uses `h1_MC15TuneV1_ProbNNk` → normalized to `h1_ProbNNk`
- MC uses `h1_MC12TuneV4_ProbNNk` → normalized to `h1_ProbNNk`
- Your code just uses `h1_ProbNNk` everywhere!

## Analysis Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ DONE | Branch configuration & data loading infrastructure |
| 1 | ✅ DONE | Configuration setup (TOML files exist) |
| 2 | ⏳ NEXT | Data loading execution |
| 3 | 📋 TODO | Lambda pre-selection (fixed cuts) |
| 4 | 📋 TODO | Selection optimization (2D FOM scan) |
| 5 | 📋 TODO | Mass fitting (RooFit) |
| 6 | 📋 TODO | Efficiency calculation |
| 7 | 📋 TODO | Branching fraction ratios |

## Key Features

### ✅ What's Working
1. **Unified branch handling** - Reuses proven BranchConfig system
2. **Data/MC compatibility** - Automatic alias resolution
3. **LL/DD support** - Combines both Lambda categories
4. **Derived branches** - Bu_MM_corrected, delta_z, M_LpKm, etc.
5. **Trigger selection** - (L0_TIS) AND (HLT1_TOS) AND (HLT2_TOS)
6. **Modular design** - Each phase is independent module

### 🔄 What's Next
1. **Test with real data** - Load a small sample
2. **Apply Lambda cuts** - Phase 3 selection
3. **Optimize cuts** - Phase 4 FOM maximization
4. **Fit masses** - Phase 5 RooFit
5. **Calculate efficiencies** - Phase 6
6. **Extract ratios** - Phase 7 final results

## Important Notes

### ⚠️ Draft Analysis Scope
This is a **draft analysis** focusing on:
- ✅ Statistical precision
- ✅ Analysis framework
- ✅ Branching fraction **ratios** (not absolute)

**Not yet included** (for full analysis):
- Systematic uncertainties
- Full efficiency breakdown (only selection efficiency)
- Multiple candidate handling
- Background composition studies

### 🎯 Physics Goal
Measure **ratios** of branching fractions:

```
Br(B⁺ → ηc X) × Br(ηc → Λ̄pK⁻)
────────────────────────────────── = ?
Br(B⁺ → J/ψ X) × Br(J/ψ → Λ̄pK⁻)

Br(B⁺ → χc0 X) × Br(χc0 → Λ̄pK⁻)
────────────────────────────────── = ?
Br(B⁺ → J/ψ X) × Br(J/ψ → Λ̄pK⁻)

Br(B⁺ → χc1 X) × Br(χc1 → Λ̄pK⁻)
────────────────────────────────── = ?
Br(B⁺ → J/ψ X) × Br(J/ψ → Λ̄pK⁻)
```

**Key advantage**: Ratios don't require absolute branching fractions!

## Documentation

- `plan.md` - Complete analysis plan with pseudocode
- `MIGRATION_NOTES.md` - Phase 0 implementation details
- `test_phase0.py` - Validation tests
- `../analysis/branches_config.toml` - Branch configuration reference

## Questions?

Check the migration notes: `MIGRATION_NOTES.md`

Or review the proven implementation: `../analysis/studies/selection/`
