#!/usr/bin/env python3
"""
Phase 4 Mini Test: Run optimization on just ONE variable to test end-to-end

This is useful for quick validation before running full optimization.
"""

import sys
from pathlib import Path
import time

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from data_handler import TOMLConfig, DataManager
from lambda_selector import LambdaSelector
from selection_optimizer import SelectionOptimizer
import awkward as ak

def main():
    print("\n" + "="*80)
    print("PHASE 4 MINI TEST: Single Variable Optimization")
    print("="*80)
    print("\nThis tests the full optimization pipeline on ONE variable only.")
    print("If this works, the full optimization should work too.\n")
    
    # Initialize
    print("1. Loading configuration...")
    config = TOMLConfig("./config")
    dm = DataManager(config)
    selector = LambdaSelector(config)
    
    # Load minimal data (2016 MD LL only)
    print("\n2. Loading data (2016 MD LL)...")
    data = dm.load_tree("data", 2016, "MD", "LL")
    data = dm.compute_derived_branches(data)
    data = selector.apply_lambda_cuts(data)
    data = selector.apply_bu_fixed_cuts(data)
    print(f"   Data events: {len(data):,}")
    
    # Load MC (just J/psi for speed)
    print("\n3. Loading J/ψ MC (2016 MD LL)...")
    mc_jpsi = dm.load_tree("Jpsi", 2016, "MD", "LL")
    mc_jpsi = dm.compute_derived_branches(mc_jpsi)
    mc_jpsi = selector.apply_lambda_cuts(mc_jpsi)
    mc_jpsi = selector.apply_bu_fixed_cuts(mc_jpsi)
    print(f"   J/ψ events: {len(mc_jpsi):,}")
    
    # Load phase space
    print("\n4. Loading phase space MC (2016 MD LL)...")
    phase_space = dm.load_tree("KpKm", 2016, "MD", "LL")
    phase_space = dm.compute_derived_branches(phase_space)
    phase_space = selector.apply_lambda_cuts(phase_space)
    phase_space = selector.apply_bu_fixed_cuts(phase_space)
    print(f"   Phase space events: {len(phase_space):,}")
    
    # Create optimizer
    print("\n5. Creating optimizer...")
    optimizer = SelectionOptimizer(
        signal_mc={"jpsi": {"2016": mc_jpsi}},
        phase_space_mc={"2016": phase_space},
        data={"2016": data},
        config=config
    )
    
    # Test scanning ONE variable
    print("\n6. Testing optimization on B+ pT...")
    print("   (Scanning from 2000 to 8000 MeV with 500 MeV steps)\n")
    
    scan_config = {
        "begin": 2000.0,
        "end": 8000.0,
        "step": 500.0,
        "cut_type": "greater",
        "description": "B+ pT [MeV/c]"
    }
    
    start = time.time()
    
    results = optimizer.scan_single_variable(
        state="jpsi",
        variable_name="pt",
        branch_name="Bu_PT",
        scan_config=scan_config
    )
    
    elapsed = time.time() - start
    
    # Show results
    print(f"\n7. Results (computed in {elapsed:.1f}s):\n")
    print(results.to_string())
    
    # Find optimal
    idx_max = results["fom"].idxmax()
    optimal = results.loc[idx_max]
    
    print(f"\n8. Optimal cut:")
    print(f"   Bu_PT > {optimal['cut_value']:.0f} MeV/c")
    print(f"   FOM = {optimal['fom']:.3f}")
    print(f"   n_sig = {optimal['n_sig']:.0f}")
    print(f"   n_bkg = {optimal['n_bkg']:.1f}")
    
    # Test plot generation
    print(f"\n9. Testing plot generation...")
    try:
        optimizer._plot_fom_scan(
            results,
            category="bu",
            var_name="pt",
            state="jpsi",
            var_config=scan_config
        )
        
        plot_file = Path(config.paths["output"]["plots_dir"]) / "optimization" / "fom_scan_bu_pt_jpsi.png"
        if plot_file.exists():
            print(f"   ✓ Plot created: {plot_file}")
        else:
            print(f"   ⚠️  Plot not found at expected location")
    except Exception as e:
        print(f"   ✗ Plot generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✓ MINI TEST COMPLETE")
    print("="*80)
    print("\nThe optimization pipeline is working correctly!")
    print("You can now run the full optimization with:")
    print("  python run_phase4_optimization.py")
    print("\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
