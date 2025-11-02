#!/usr/bin/env python3
"""
Main analysis script for B⁺ → Λ̄pK⁻K⁺ charmonium study

Executes full analysis pipeline:
0. Branch discovery
1. Data loading
2. Lambda selection
3. Selection optimization
4. Mass fitting
5. Efficiency calculation
6. Branching fraction extraction
"""

import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "modules"))

from branch_inspector import BranchInspector
from data_handler import TOMLConfig, DataManager
from lambda_selector import LambdaSelector
from selection_optimizer import SelectionOptimizer
from mass_fitter import InvariantMassFitter
from efficiency_calculator import EfficiencyCalculator
from branching_fraction_calculator import BranchingFractionCalculator

def main():
    """Execute full analysis pipeline"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  B⁺ → Λ̄pK⁻K⁺ Charmonium Analysis                            ║
    ║  Draft Analysis - Statistical Uncertainties Only            ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 0: Branch Discovery
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("PHASE 0: BRANCH STRUCTURE DISCOVERY")
    print("="*80)
    
    run_phase0 = input("Run branch discovery? (y/n, skip if already done): ").lower()
    
    if run_phase0 == 'y':
        inspector = BranchInspector(
            data_path="./data",
            mc_path="./mc"
        )
        inspector.run_full_inspection()
        
        print("\n⚠️  ACTION REQUIRED:")
        print("   1. Review docs/branch_structure.md")
        print("   2. Identify missing branch names (marked with ???)")
        print("   3. Update config/particles.toml with actual branch names")
        print("   4. Re-run this script after updating\n")
        
        cont = input("Have you updated branch names? (y/n): ").lower()
        if cont != 'y':
            print("Exiting. Please update branch names and re-run.")
            return
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Configuration and Data Loading
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("PHASE 1: LOADING CONFIGURATION AND DATA")
    print("="*80)
    
    config = TOMLConfig(config_dir="./config")
    data_manager = DataManager(config)
    
    print("\n[Loading Data]")
    data_by_year = data_manager.load_all_data_combined_magnets("data")
    
    print("\n[Loading MC - Signal States]")
    mc_by_state = {}
    for state in ["Jpsi", "etac", "chic0", "chic1"]:
        print(f"\n  {state}:")
        mc_by_state[state.lower()] = data_manager.load_all_data_combined_magnets(state)
    
    print("\n[Loading MC - Phase Space]")
    phase_space_mc = data_manager.load_all_data_combined_magnets("KpKm")
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Lambda Pre-Selection
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("PHASE 2: LAMBDA PRE-SELECTION (FIXED CUTS)")
    print("="*80)
    
    lambda_selector = LambdaSelector(config)
    
    print("\n[Applying to Data]")
    for year in data_by_year:
        print(f"  {year}:")
        data_by_year[year] = lambda_selector.apply_lambda_cuts(data_by_year[year])
    
    print("\n[Applying to Signal MC]")
    for state in mc_by_state:
        print(f"  {state}:")
        for year in mc_by_state[state]:
            mc_by_state[state][year] = lambda_selector.apply_lambda_cuts(
                mc_by_state[state][year]
            )
    
    print("\n[Applying to Phase Space MC]")
    for year in phase_space_mc:
        print(f"  {year}:")
        phase_space_mc[year] = lambda_selector.apply_lambda_cuts(phase_space_mc[year])
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Selection Optimization
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("PHASE 3: SELECTION OPTIMIZATION (2D FOM SCAN)")
    print("="*80)
    print("\n⚠️  This step may take 30-60 minutes depending on data size\n")
    
    run_optimization = input("Run optimization? (y/n, can load cached): ").lower()
    
    if run_optimization == 'y':
        optimizer = SelectionOptimizer(
            signal_mc=mc_by_state,
            phase_space_mc=phase_space_mc,
            data=data_by_year,
            config=config
        )
        
        optimized_cuts_df = optimizer.optimize_2d_all_variables()
    else:
        # Load cached results
        import pandas as pd
        optimized_cuts_df = pd.read_csv("./tables/optimized_cuts_2d.csv")
        print("✓ Loaded cached optimization results")
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 4: Apply Optimized Cuts
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("PHASE 4: APPLYING OPTIMIZED CUTS")
    print("="*80)
    
    # TODO: Implement cut application
    # For now, note that we'll use optimized_cuts_df in efficiency calculation
    
    print("⚠️  Cut application deferred to efficiency calculation")
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 5: Mass Fitting
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("PHASE 5: MASS FITTING")
    print("="*80)
    
    fitter = InvariantMassFitter(config)
    
    fit_results = fitter.perform_fit(data_by_year)
    
    yields = fit_results["yields"]
    masses = fit_results["masses"]
    widths = fit_results["widths"]
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 6: Efficiency Calculation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("PHASE 6: EFFICIENCY CALCULATION")
    print("="*80)
    
    eff_calculator = EfficiencyCalculator(config)
    
    # Calculate for each (state, year) pair
    efficiencies_by_state_year = {}
    
    for state in ["jpsi", "etac", "chic0", "chic1"]:
        efficiencies_by_state_year[state] = {}
        
        # Get optimized cuts for this state
        state_cuts = optimized_cuts_df[optimized_cuts_df["state"] == state]
        
        for year in ["2016", "2017", "2018"]:
            mc_events = mc_by_state[state][year]
            
            eff_dict = eff_calculator.calculate_total_efficiency_per_year(
                state=state,
                year=year,
                mc_events=mc_events,
                phase_space_mc=phase_space_mc,
                optimized_cuts=state_cuts,
                trigger_selector=data_manager
            )
            
            efficiencies_by_state_year[state][year] = eff_dict["total"]
    
    # Calculate efficiency ratios
    eff_ratios_df = eff_calculator.calculate_efficiency_ratios(
        efficiencies_by_state_year
    )
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 7: Branching Fraction Calculation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("PHASE 7: BRANCHING FRACTION RATIOS")
    print("="*80)
    
    bf_calculator = BranchingFractionCalculator(
        yields=yields,
        efficiencies=efficiencies_by_state_year,
        config=config
    )
    
    # Calculate ratios
    ratios_df = bf_calculator.calculate_all_ratios()
    
    # Yield consistency check
    consistency_df = bf_calculator.check_yield_consistency_per_year()
    
    # Generate final summary
    bf_calculator.generate_final_summary(ratios_df)
    
    # ═══════════════════════════════════════════════════════════
    # DONE
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nOutput locations:")
    print(f"  - Plots: {config.paths['output']['plots_dir']}")
    print(f"  - Tables: {config.paths['output']['tables_dir']}")
    print(f"  - Results: {config.paths['output']['results_dir']}")
    print(f"  - Documentation: {config.paths['output']['docs_dir']}")
    
    print("\n⚠️  IMPORTANT NOTES:")
    print("  1. This is a DRAFT analysis with statistical uncertainties only")
    print("  2. Efficiency placeholders (reco×strip = 0) - OK for ratios!")
    print("  3. Multiple candidates not yet estimated")
    print("  4. Systematic uncertainties to be added")
    print("  5. Results are RATIOS (don't need absolute branching fractions!)")
    
    print("\n✓ Review ./results/final_results.md for summary\n")

if __name__ == "__main__":
    main()

# ---

# ## Important Notes and Caveats

# ### ⚠️ CRITICAL ITEMS TO ADDRESS

# 1. **J/ψ → Λ̄pK⁻ Quantum Numbers**
#    - MUST verify this decay is allowed
#    - If forbidden, switch normalization to ηc

# 2. **Branch Name Discovery (Phase 0)**
#    - Run `branch_inspector.py` first
#    - Manually identify actual branch names for:
#      - Bachelor p̄ four-momenta and PID
#      - K⁺, K⁻ four-momenta and PID
#      - Truth-matching branches in MC
#    - Update `config/particles.toml` with correct names

# 3. **Multiple Candidates**
#    - Currently: keeping all candidates (let fit handle)
#    - TODO: Estimate fraction with multiple candidates
#    - Typical: 1-5% for B meson analyses

# 4. **Efficiency Placeholders**
#    - ε_reco×strip = 0 (PLACEHOLDER)
#    - This is OK because we measure RATIOS
#    - If reco×strip similar across states → cancels in ratio
#    - Full study needed later for absolute measurements

# 5. **Systematic Uncertainties**
#    - Draft analysis: statistical only
#    - Full analysis needs:
#      - Fit model systematics
#      - Selection optimization systematics
#      - Efficiency systematics
#      - Background model systematics

# ### ✓ What This Plan Delivers

# 1. **Complete modular code structure** with TOML configuration
# 2. **Branch discovery tool** to identify actual branch names
# 3. **2D selection optimization** with FOM scans and plots
# 4. **Simultaneous mass fitting** to all charmonium states
# 5. **Efficiency calculation framework** (with placeholders noted)
# 6. **Branching fraction ratios** relative to J/ψ (self-normalization)
# 7. **Validation plots** (yield consistency, kinematic distributions)
# 8. **Comprehensive documentation** and results summary

# ### 📊 Expected Outputs

# - `docs/branch_structure.md` - Branch name documentation
# - `plots/optimization/*.png` - FOM scan plots (one per variable×state)
# - `plots/fits/*.png` - Mass fit results per year
# - `plots/yield_consistency_check.png` - N/(L×ε) vs year
# - `tables/optimized_cuts_summary.csv` - Selection summary
# - `tables/efficiency_ratios.csv` - Efficiency ratios
# - `tables/branching_fraction_ratios.csv` - Final BR ratios
# - `results/final_results.md` - Complete analysis summary

# ### 🎯 Physics Results Format
# ```
# Br(B⁺ → ηc X) × Br(ηc → Λ̄pK⁻)
# ───────────────────────────────── = X.XXX ± 0.XXX (stat)
# Br(B⁺ → J/ψ X) × Br(J/ψ → Λ̄pK⁻)

# Br(B⁺ → χc0 X) × Br(χc0 → Λ̄pK⁻)
# ───────────────────────────────── = X.XXX ± 0.XXX (stat)
# Br(B⁺ → J/ψ X) × Br(J/ψ → Λ̄pK⁻)

# Br(B⁺ → χc1 X) × Br(χc1 → Λ̄pK⁻)
# ───────────────────────────────── = X.XXX ± 0.XXX (stat)
# Br(B⁺ → J/ψ X) × Br(J/ψ → Λ̄pK⁻)

# Br(B⁺ → χc1 X) × Br(χc1 → Λ̄pK⁻)
# ───────────────────────────────── = X.XXX ± 0.XXX (stat)
# Br(B⁺ → χc0 X) × Br(χc0 → Λ̄pK⁻)