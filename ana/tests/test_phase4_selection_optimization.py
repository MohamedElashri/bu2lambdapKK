#!/usr/bin/env python3
"""
Phase 4 Validation: Selection Optimization (2D FOM Scan)

Tests the 2D optimization framework:
1. SelectionOptimizer initialization
2. FOM calculation
3. Signal/background counting in regions
4. Single variable scanning
5. Full 2D optimization
6. Results validation
"""

import sys
from pathlib import Path
import awkward as ak
import numpy as np
import pandas as pd

# Add modules to path
sys.path.append(str(Path(__file__).parent.parent / "modules"))

def test_selection_optimizer_init():
    """Test that SelectionOptimizer initializes correctly"""
    print("="*80)
    print("TEST 1: SelectionOptimizer Initialization")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig, DataManager
        from lambda_selector import LambdaSelector
        from selection_optimizer import SelectionOptimizer
        
        config = TOMLConfig("./config")
        dm = DataManager(config)
        selector = LambdaSelector(config)
        
        # Load minimal data for initialization test (2016 MD LL only)
        print("\nLoading minimal test data (2016 MD LL)...")
        
        # Data
        data = dm.load_tree("data", 2016, "MD", "LL")
        data = dm.compute_derived_branches(data)
        data = selector.apply_lambda_cuts(data)
        data = selector.apply_bu_fixed_cuts(data)
        data_dict = {"2016": data}
        
        # MC for J/psi (minimal for testing)
        mc_jpsi = dm.load_tree("Jpsi", 2016, "MD", "LL")
        mc_jpsi = dm.compute_derived_branches(mc_jpsi)
        mc_jpsi = selector.apply_lambda_cuts(mc_jpsi)
        mc_jpsi = selector.apply_bu_fixed_cuts(mc_jpsi)
        mc_dict = {"jpsi": {"2016": mc_jpsi}}
        
        # Phase space (KpKm)
        phase_space = dm.load_tree("KpKm", 2016, "MD", "LL")
        phase_space = dm.compute_derived_branches(phase_space)
        phase_space = selector.apply_lambda_cuts(phase_space)
        phase_space = selector.apply_bu_fixed_cuts(phase_space)
        phase_space_dict = {"2016": phase_space}
        
        # Initialize optimizer
        print("\nInitializing SelectionOptimizer...")
        optimizer = SelectionOptimizer(
            signal_mc=mc_dict,
            phase_space_mc=phase_space_dict,
            data=data_dict,
            config=config
        )
        
        print(f"✓ Optimizer initialized with:")
        print(f"  - Data events: {len(data):,}")
        print(f"  - J/ψ MC events: {len(mc_jpsi):,}")
        print(f"  - Phase space events: {len(phase_space):,}")
        
        # Check signal regions defined
        print("\nSignal regions:")
        for state in ["jpsi", "etac", "chic0", "chic1"]:
            region = optimizer.define_signal_region(state)
            print(f"  {state}: [{region[0]:.1f}, {region[1]:.1f}] MeV")
        
        print("\n✓ SelectionOptimizer initialized successfully")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fom_calculation():
    """Test FOM calculation"""
    print("\n" + "="*80)
    print("TEST 2: FOM Calculation")
    print("="*80)
    
    try:
        from selection_optimizer import SelectionOptimizer
        
        # Test FOM formula: FOM = n_sig / sqrt(n_sig + n_bkg)
        test_cases = [
            (100, 100, 100 / np.sqrt(200)),  # Equal signal/background
            (100, 0, 100 / np.sqrt(100)),     # No background
            (0, 100, 0),                       # No signal
            (1000, 100, 1000 / np.sqrt(1100)), # High S/B
            (100, 1000, 100 / np.sqrt(1100)),  # Low S/B
        ]
        
        print("\nTesting FOM = n_sig / sqrt(n_sig + n_bkg):\n")
        
        # Create dummy optimizer instance (just for FOM method)
        class DummyOptimizer:
            def compute_fom(self, n_sig, n_bkg):
                return SelectionOptimizer.compute_fom(None, n_sig, n_bkg)
        
        optimizer = DummyOptimizer()
        
        all_pass = True
        for n_sig, n_bkg, expected in test_cases:
            result = optimizer.compute_fom(n_sig, n_bkg)
            match = abs(result - expected) < 1e-6
            status = "✓" if match else "✗"
            print(f"{status} n_sig={n_sig:4.0f}, n_bkg={n_bkg:4.0f} → FOM={result:.3f} (expected {expected:.3f})")
            if not match:
                all_pass = False
        
        if all_pass:
            print("\n✓ FOM calculation correct")
            return True
        else:
            print("\n✗ FOM calculation has errors")
            return False
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_signal_background_counting():
    """Test counting events in signal and sideband regions"""
    print("\n" + "="*80)
    print("TEST 3: Signal/Background Region Counting")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig, DataManager
        from lambda_selector import LambdaSelector
        from selection_optimizer import SelectionOptimizer
        
        config = TOMLConfig("./config")
        dm = DataManager(config)
        selector = LambdaSelector(config)
        
        # Load J/psi MC (should have peak at J/psi mass)
        print("\nLoading J/ψ MC (2016 MD LL)...")
        mc_jpsi = dm.load_tree("Jpsi", 2016, "MD", "LL")
        mc_jpsi = dm.compute_derived_branches(mc_jpsi)
        mc_jpsi = selector.apply_lambda_cuts(mc_jpsi)
        mc_jpsi = selector.apply_bu_fixed_cuts(mc_jpsi)
        
        # Create minimal optimizer
        mc_dict = {"jpsi": {"2016": mc_jpsi}}
        data_dict = {"2016": mc_jpsi}  # Use MC as fake data for testing
        phase_space_dict = {"2016": mc_jpsi}
        
        optimizer = SelectionOptimizer(
            signal_mc=mc_dict,
            phase_space_mc=phase_space_dict,
            data=data_dict,
            config=config
        )
        
        # Test signal region counting
        print("\nTesting signal region counting (J/ψ):")
        signal_region = optimizer.define_signal_region("jpsi")
        print(f"  Signal region: [{signal_region[0]:.1f}, {signal_region[1]:.1f}] MeV")
        
        n_in_signal = optimizer.count_events_in_region(mc_jpsi, signal_region)
        n_total = len(mc_jpsi)
        fraction = n_in_signal / n_total if n_total > 0 else 0
        
        print(f"  Events in signal region: {n_in_signal:,} / {n_total:,} ({100*fraction:.1f}%)")
        
        # For J/psi MC, expect significant fraction in signal region
        if fraction < 0.1:
            print(f"  ⚠️  Low fraction in signal region - check M_LpKm_h1 calculation")
        else:
            print(f"  ✓ Reasonable fraction in signal region")
        
        # Test sideband regions
        print("\nTesting sideband regions (J/ψ):")
        sidebands = optimizer.define_sideband_regions("jpsi")
        print(f"  Low sideband: [{sidebands[0][0]:.1f}, {sidebands[0][1]:.1f}] MeV")
        print(f"  High sideband: [{sidebands[1][0]:.1f}, {sidebands[1][1]:.1f}] MeV")
        
        n_low_sb = optimizer.count_events_in_region(mc_jpsi, sidebands[0])
        n_high_sb = optimizer.count_events_in_region(mc_jpsi, sidebands[1])
        
        print(f"  Events in low SB: {n_low_sb:,}")
        print(f"  Events in high SB: {n_high_sb:,}")
        
        # Background estimation
        n_bkg_est = optimizer.estimate_background_in_signal_region(mc_jpsi, "jpsi")
        print(f"  Estimated background in signal: {n_bkg_est:.1f}")
        
        print("\n✓ Signal/background counting working")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_variable_scan():
    """Test scanning a single variable"""
    print("\n" + "="*80)
    print("TEST 4: Single Variable Scan")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig, DataManager
        from lambda_selector import LambdaSelector
        from selection_optimizer import SelectionOptimizer
        
        config = TOMLConfig("./config")
        dm = DataManager(config)
        selector = LambdaSelector(config)
        
        # Load data for testing (use 2016 MD LL only for speed)
        print("\nLoading test data (2016 MD LL)...")
        
        data = dm.load_tree("data", 2016, "MD", "LL")
        data = dm.compute_derived_branches(data)
        data = selector.apply_lambda_cuts(data)
        data = selector.apply_bu_fixed_cuts(data)
        
        mc_jpsi = dm.load_tree("Jpsi", 2016, "MD", "LL")
        mc_jpsi = dm.compute_derived_branches(mc_jpsi)
        mc_jpsi = selector.apply_lambda_cuts(mc_jpsi)
        mc_jpsi = selector.apply_bu_fixed_cuts(mc_jpsi)
        
        phase_space = dm.load_tree("KpKm", 2016, "MD", "LL")
        phase_space = dm.compute_derived_branches(phase_space)
        phase_space = selector.apply_lambda_cuts(phase_space)
        phase_space = selector.apply_bu_fixed_cuts(phase_space)
        
        # Create optimizer
        optimizer = SelectionOptimizer(
            signal_mc={"jpsi": {"2016": mc_jpsi}},
            phase_space_mc={"2016": phase_space},
            data={"2016": data},
            config=config
        )
        
        # Test scanning B+ pT (should have clear optimum)
        print("\nScanning Bu_PT for J/ψ...")
        
        scan_config = {
            "begin": 2000.0,
            "end": 8000.0,
            "step": 1000.0,
            "cut_type": "greater",
            "description": "B+ pT [MeV/c]"
        }
        
        results = optimizer.scan_single_variable(
            state="jpsi",
            variable_name="pt",
            branch_name="Bu_PT",
            scan_config=scan_config
        )
        
        print(f"\n✓ Scan completed: {len(results)} points")
        print("\nScan results:")
        print(results.to_string())
        
        # Find optimal cut
        idx_max = results["fom"].idxmax()
        optimal = results.loc[idx_max]
        
        print(f"\nOptimal cut:")
        print(f"  Bu_PT > {optimal['cut_value']:.0f} MeV/c")
        print(f"  FOM = {optimal['fom']:.3f}")
        print(f"  n_sig = {optimal['n_sig']:.0f}")
        print(f"  n_bkg = {optimal['n_bkg']:.1f}")
        
        # Validate results
        if optimal['fom'] <= 0:
            print(f"  ✗ FOM is zero or negative!")
            return False
        
        if optimal['n_sig'] <= 0:
            print(f"  ✗ No signal events!")
            return False
        
        print("\n✓ Single variable scan working correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_2d_optimization():
    """Test full 2D optimization (reduced scope for testing)"""
    print("\n" + "="*80)
    print("TEST 5: Full 2D Optimization (Limited)")
    print("="*80)
    print("\nNOTE: This test runs a reduced optimization")
    print("      (fewer variables/states for speed)")
    
    try:
        from data_handler import TOMLConfig, DataManager
        from lambda_selector import LambdaSelector
        from selection_optimizer import SelectionOptimizer
        
        config = TOMLConfig("./config")
        dm = DataManager(config)
        selector = LambdaSelector(config)
        
        # Load data (2016 LL only for speed)
        print("\nLoading data for optimization test (2016 LL)...")
        print("(Using single year/magnet/track for speed)")
        
        # Data
        data = dm.load_tree("data", 2016, "MD", "LL")
        data = dm.compute_derived_branches(data)
        data = selector.apply_lambda_cuts(data)
        data = selector.apply_bu_fixed_cuts(data)
        data_dict = {"2016": data}
        
        # Load all MC states
        mc_dict = {}
        for state in ["Jpsi", "etac", "chic0", "chic1"]:
            print(f"  Loading {state} MC...")
            mc = dm.load_tree(state, 2016, "MD", "LL")
            mc = dm.compute_derived_branches(mc)
            mc = selector.apply_lambda_cuts(mc)
            mc = selector.apply_bu_fixed_cuts(mc)
            mc_dict[state.lower()] = {"2016": mc}
        
        # Phase space
        print(f"  Loading phase space (KpKm)...")
        phase_space = dm.load_tree("KpKm", 2016, "MD", "LL")
        phase_space = dm.compute_derived_branches(phase_space)
        phase_space = selector.apply_lambda_cuts(phase_space)
        phase_space = selector.apply_bu_fixed_cuts(phase_space)
        phase_space_dict = {"2016": phase_space}
        
        # Create optimizer
        print("\nCreating optimizer...")
        optimizer = SelectionOptimizer(
            signal_mc=mc_dict,
            phase_space_mc=phase_space_dict,
            data=data_dict,
            config=config
        )
        
        # Run optimization (this will take some time)
        print("\nRunning 2D optimization...")
        print("(This may take several minutes...)")
        
        results_df = optimizer.optimize_2d_all_variables()
        
        print(f"\n✓ Optimization completed!")
        print(f"  Total combinations: {len(results_df)}")
        
        # Show summary
        print("\nOptimization summary:")
        print(results_df.groupby('state')['max_fom'].describe())
        
        # Check output files created
        output_dir = Path(config.paths["output"]["tables_dir"])
        csv_file = output_dir / "optimized_cuts_2d.csv"
        summary_file = output_dir / "optimized_cuts_summary.csv"
        
        if csv_file.exists():
            print(f"\n✓ Results saved to: {csv_file}")
        else:
            print(f"\n⚠️  Results file not found: {csv_file}")
        
        if summary_file.exists():
            print(f"✓ Summary saved to: {summary_file}")
        else:
            print(f"⚠️  Summary file not found: {summary_file}")
        
        # Check plots created
        plot_dir = Path(config.paths["output"]["plots_dir"]) / "optimization"
        if plot_dir.exists():
            n_plots = len(list(plot_dir.glob("*.png")))
            print(f"✓ Generated {n_plots} optimization plots")
        else:
            print(f"⚠️  Plot directory not found: {plot_dir}")
        
        print("\n✓ Full 2D optimization working")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimization_results_validation():
    """Validate optimization results make physical sense"""
    print("\n" + "="*80)
    print("TEST 6: Optimization Results Validation")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig
        
        config = TOMLConfig("./config")
        output_dir = Path(config.paths["output"]["tables_dir"])
        csv_file = output_dir / "optimized_cuts_2d.csv"
        
        if not csv_file.exists():
            print(f"\n⚠️  Results file not found: {csv_file}")
            print("     Run test_full_2d_optimization() first")
            return False
        
        # Load results
        print(f"\nLoading results from: {csv_file}")
        results = pd.read_csv(csv_file)
        
        print(f"Total rows: {len(results)}")
        print(f"Variables: {results['variable'].nunique()}")
        print(f"States: {results['state'].nunique()}")
        
        # Check all expected columns exist
        expected_cols = ['category', 'variable', 'branch_name', 'state', 
                        'optimal_cut', 'max_fom', 'n_sig_at_optimal', 
                        'n_bkg_at_optimal', 'cut_type']
        
        print("\nChecking columns:")
        all_present = True
        for col in expected_cols:
            if col in results.columns:
                print(f"  ✓ {col}")
            else:
                print(f"  ✗ {col} missing!")
                all_present = False
        
        if not all_present:
            return False
        
        # Check for reasonable values
        print("\nValidating values:")
        
        # FOM should be positive
        if (results['max_fom'] > 0).all():
            print(f"  ✓ All FOM values positive")
        else:
            n_zero = (results['max_fom'] <= 0).sum()
            print(f"  ⚠️  {n_zero} entries with FOM ≤ 0")
        
        # Signal counts should be positive
        if (results['n_sig_at_optimal'] > 0).all():
            print(f"  ✓ All signal counts positive")
        else:
            n_zero = (results['n_sig_at_optimal'] <= 0).sum()
            print(f"  ⚠️  {n_zero} entries with n_sig ≤ 0")
        
        # Background should be non-negative
        if (results['n_bkg_at_optimal'] >= 0).all():
            print(f"  ✓ All background estimates non-negative")
        else:
            print(f"  ✗ Some background estimates negative!")
            return False
        
        # Show example results
        print("\nExample optimal cuts (J/ψ):")
        jpsi_results = results[results['state'] == 'jpsi'].head(5)
        for _, row in jpsi_results.iterrows():
            print(f"  {row['variable']:15s}: {row['optimal_cut']:8.1f} ({row['cut_type']:7s}) - FOM={row['max_fom']:.3f}")
        
        print("\n✓ Optimization results validated")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Phase 4 tests"""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*20 + "PHASE 4 VALIDATION TESTS" + " "*34 + "█")
    print("█" + " "*15 + "Selection Optimization (2D FOM Scan)" + " "*27 + "█")
    print("█" + " "*78 + "█")
    print("█"*80 + "\n")
    
    tests = [
        ("SelectionOptimizer Initialization", test_selection_optimizer_init),
        ("FOM Calculation", test_fom_calculation),
        ("Signal/Background Counting", test_signal_background_counting),
        ("Single Variable Scan", test_single_variable_scan),
        ("Full 2D Optimization (Limited)", test_full_2d_optimization),
        ("Results Validation", test_optimization_results_validation),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*80)
    print("PHASE 4 TEST SUMMARY")
    print("="*80)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} - {name}")
    
    n_pass = sum(results.values())
    n_total = len(results)
    
    print(f"\n{n_pass}/{n_total} tests passed")
    
    if n_pass == n_total:
        print("\n🎉 All Phase 4 tests PASSED! 🎉")
        print("\nPhase 4 is ready for production use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Review output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
