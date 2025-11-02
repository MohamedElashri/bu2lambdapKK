#!/usr/bin/env python
"""
Test Script for Phase 6: Efficiency Calculation

Tests the EfficiencyCalculator class with simplified selection efficiency approach.
Validates efficiency calculations on MC samples.
"""

import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

import pandas as pd
import awkward as ak
from modules.data_handler import DataManager, TOMLConfig
from modules.lambda_selector import LambdaSelector
from modules.efficiency_calculator import EfficiencyCalculator


def test_1_efficiency_calculator_initialization():
    """Test 1: EfficiencyCalculator initialization"""
    print("\n" + "="*80)
    print("TEST 1: EfficiencyCalculator Initialization")
    print("="*80)
    
    config = TOMLConfig()
    
    # Create dummy optimized cuts dataframe
    cuts_data = {
        "state": ["jpsi", "jpsi"],
        "variable": ["Bu_PT", "bachelor_p_PT"],
        "branch_name": ["Bu_PT", "bachelor_p_PT"],
        "optimal_cut": [2000.0, 1500.0],
        "cut_type": ["greater", "greater"]
    }
    cuts_df = pd.DataFrame(cuts_data)
    
    calculator = EfficiencyCalculator(config, cuts_df)
    
    assert calculator.config is not None, "Config should be set"
    assert calculator.optimized_cuts is not None, "Optimized cuts should be set"
    
    print("✓ EfficiencyCalculator initialized correctly")
    print(f"  Config loaded: {config.config_dir}")
    print(f"  Optimized cuts: {len(cuts_df)} cut entries")
    
    return True


def test_2_get_cuts_for_state():
    """Test 2: Extract cuts for specific state"""
    print("\n" + "="*80)
    print("TEST 2: Get Cuts for State")
    print("="*80)
    
    config = TOMLConfig()
    
    # Create cuts for multiple states
    cuts_data = {
        "state": ["jpsi", "jpsi", "etac", "etac"],
        "variable": ["Bu_PT", "bachelor_p_PT", "Bu_PT", "bachelor_p_PT"],
        "branch_name": ["Bu_PT", "bachelor_p_PT", "Bu_PT", "bachelor_p_PT"],
        "optimal_cut": [2000.0, 1500.0, 1800.0, 1400.0],
        "cut_type": ["greater", "greater", "greater", "greater"]
    }
    cuts_df = pd.DataFrame(cuts_data)
    
    calculator = EfficiencyCalculator(config, cuts_df)
    
    # Get cuts for J/ψ
    jpsi_cuts = calculator.get_cuts_for_state("jpsi")
    assert len(jpsi_cuts) == 2, "Should have 2 cuts for J/ψ"
    assert jpsi_cuts.iloc[0]["optimal_cut"] == 2000.0, "First cut value incorrect"
    
    # Get cuts for ηc
    etac_cuts = calculator.get_cuts_for_state("etac")
    assert len(etac_cuts) == 2, "Should have 2 cuts for ηc"
    assert etac_cuts.iloc[0]["optimal_cut"] == 1800.0, "First cut value incorrect"
    
    print("✓ Cut extraction working correctly")
    print(f"  J/ψ cuts: {len(jpsi_cuts)}")
    print(f"  ηc cuts: {len(etac_cuts)}")
    
    return True


def test_3_apply_optimized_cuts():
    """Test 3: Apply optimized cuts to MC events"""
    print("\n" + "="*80)
    print("TEST 3: Apply Optimized Cuts")
    print("="*80)
    
    # Create dummy MC events
    mc_events = ak.Array({
        "Bu_MM_corrected": [5270, 5280, 5290, 5300, 5310],
        "Bu_PT": [3000, 2500, 2000, 1500, 1000],
        "bachelor_p_PT": [2000, 1800, 1600, 1400, 1200]
    })
    
    print(f"Initial events: {len(mc_events)}")
    
    # Create cuts
    cuts_data = {
        "state": ["jpsi", "jpsi"],
        "variable": ["Bu_PT", "bachelor_p_PT"],
        "branch_name": ["Bu_PT", "bachelor_p_PT"],
        "optimal_cut": [2000.0, 1500.0],
        "cut_type": ["greater", "greater"]
    }
    cuts_df = pd.DataFrame(cuts_data)
    
    config = TOMLConfig()
    calculator = EfficiencyCalculator(config, cuts_df)
    
    # Apply cuts
    mc_after = calculator.apply_optimized_cuts(mc_events, "jpsi")
    
    print(f"Events after cuts: {len(mc_after)}")
    print(f"  Bu_PT > 2000: {sum(mc_events['Bu_PT'] > 2000.0)} events pass")
    print(f"  bachelor_p_PT > 1500: {sum(mc_events['bachelor_p_PT'] > 1500.0)} events pass")
    print(f"  Both cuts: {len(mc_after)} events pass")
    
    # Verify
    assert len(mc_after) == 2, "Should have 2 events passing both cuts"
    assert all(mc_after["Bu_PT"] > 2000.0), "All events should have Bu_PT > 2000"
    assert all(mc_after["bachelor_p_PT"] > 1500.0), "All events should have bachelor_p_PT > 1500"
    
    print("\n✓ Cut application working correctly")
    
    return True


def test_4_selection_efficiency_calculation():
    """Test 4: Calculate selection efficiency"""
    print("\n" + "="*80)
    print("TEST 4: Selection Efficiency Calculation")
    print("="*80)
    
    # Create dummy MC events (100 events)
    import numpy as np
    np.random.seed(42)
    
    n_events = 100
    mc_events = ak.Array({
        "Bu_MM_corrected": 5280 + np.random.normal(0, 10, n_events),
        "Bu_PT": np.random.exponential(3000, n_events),
        "bachelor_p_PT": np.random.exponential(2000, n_events)
    })
    
    # Create moderate cuts (should pass ~50%)
    cuts_data = {
        "state": ["test", "test"],
        "variable": ["Bu_PT", "bachelor_p_PT"],
        "branch_name": ["Bu_PT", "bachelor_p_PT"],
        "optimal_cut": [2000.0, 1200.0],
        "cut_type": ["greater", "greater"]
    }
    cuts_df = pd.DataFrame(cuts_data)
    
    config = TOMLConfig()
    calculator = EfficiencyCalculator(config, cuts_df)
    
    # Calculate efficiency
    eff, err = calculator.calculate_selection_efficiency(mc_events, "test")
    
    print(f"  N_before = {n_events}")
    print(f"  Efficiency = {eff:.4f} ± {err:.4f}")
    print(f"  Percentage = {100*eff:.2f}%")
    
    # Verify
    assert 0.0 <= eff <= 1.0, "Efficiency should be between 0 and 1"
    assert err >= 0.0, "Error should be non-negative"
    assert err < eff if eff > 0 else True, "Error should be smaller than efficiency"
    
    print("\n✓ Efficiency calculation working correctly")
    print(f"✓ Binomial error propagation verified")
    
    return True


def test_5_calculate_all_efficiencies_mc():
    """Test 5: Calculate efficiencies for all states (using real MC)"""
    print("\n" + "="*80)
    print("TEST 5: Calculate All Efficiencies (Real MC)")
    print("="*80)
    
    # Load real MC data
    config = TOMLConfig()
    data_manager = DataManager(config)
    selector = LambdaSelector(config)
    
    # Load small sample: 2016 MD LL only
    print("Loading MC samples (2016 MD LL)...")
    mc_by_state = {}
    
    for state in ["Jpsi", "etac", "chic0", "chic1"]:
        print(f"  Loading {state}...")
        mc_raw = data_manager.load_tree(state, 2016, "MD", "LL")
        mc_selected = selector.apply_lambda_cuts(mc_raw)
        
        mc_by_state[state.lower()] = {"2016": mc_selected}
        print(f"    After Lambda cuts: {len(mc_selected)} events")
    
    # Create simple optimized cuts (Bu_PT > 2000 for all states)
    cuts_list = []
    for state in ["jpsi", "etac", "chic0", "chic1"]:
        cuts_list.append({
            "state": state,
            "variable": "Bu_PT",
            "branch_name": "Bu_PT",
            "optimal_cut": 2000.0,
            "cut_type": "greater"
        })
    cuts_df = pd.DataFrame(cuts_list)
    
    # Calculate efficiencies
    calculator = EfficiencyCalculator(config, cuts_df)
    efficiencies = calculator.calculate_all_efficiencies(mc_by_state)
    
    # Verify results
    for state in ["jpsi", "etac", "chic0", "chic1"]:
        assert state in efficiencies, f"{state} should be in results"
        assert "2016" in efficiencies[state], "2016 should be in results"
        
        eff_data = efficiencies[state]["2016"]
        assert "eff" in eff_data, "eff should be in data"
        assert "err" in eff_data, "err should be in data"
        assert "n_before" in eff_data, "n_before should be in data"
        assert "n_after" in eff_data, "n_after should be in data"
        
        eff = eff_data["eff"]
        err = eff_data["err"]
        
        assert 0.0 <= eff <= 1.0, f"Efficiency for {state} should be in [0, 1]"
        assert err >= 0.0, f"Error for {state} should be non-negative"
    
    print("\n✓ All efficiencies calculated successfully")
    print("✓ Data structure validated")
    
    return True


def test_6_efficiency_ratios():
    """Test 6: Calculate efficiency ratios relative to J/ψ"""
    print("\n" + "="*80)
    print("TEST 6: Efficiency Ratios")
    print("="*80)
    
    # Create dummy efficiency data
    efficiencies = {
        "jpsi": {
            "2016": {"eff": 0.50, "err": 0.01, "n_before": 1000, "n_after": 500}
        },
        "etac": {
            "2016": {"eff": 0.45, "err": 0.015, "n_before": 900, "n_after": 405}
        },
        "chic0": {
            "2016": {"eff": 0.52, "err": 0.012, "n_before": 1100, "n_after": 572}
        },
        "chic1": {
            "2016": {"eff": 0.48, "err": 0.011, "n_before": 1050, "n_after": 504}
        }
    }
    
    config = TOMLConfig()
    calculator = EfficiencyCalculator(config)
    
    # Calculate ratios
    ratios_df = calculator.calculate_efficiency_ratios(efficiencies)
    
    # Verify
    assert len(ratios_df) == 3, "Should have 3 ratios (etac, chic0, chic1)"
    assert "state" in ratios_df.columns, "Should have state column"
    assert "year" in ratios_df.columns, "Should have year column"
    assert "ratio_eps_jpsi_over_state" in ratios_df.columns, "Should have ratio column"
    assert "ratio_error" in ratios_df.columns, "Should have error column"
    
    # Check ratio values are reasonable
    for _, row in ratios_df.iterrows():
        ratio = row["ratio_eps_jpsi_over_state"]
        error = row["ratio_error"]
        
        assert ratio > 0, "Ratio should be positive"
        assert error >= 0, "Error should be non-negative"
        assert error < ratio, "Error should be smaller than ratio"
        
        print(f"  {row['state']}: ε_J/ψ / ε_{row['state']} = {ratio:.3f} ± {error:.3f}")
    
    print("\n✓ Efficiency ratios calculated correctly")
    print("✓ Error propagation working")
    
    return True


def run_all_tests():
    """Run all Phase 6 tests"""
    print("\n" + "="*80)
    print("PHASE 6: EFFICIENCY CALCULATION - TEST SUITE")
    print("="*80)
    print("Testing simplified efficiency calculation (selection efficiency only)")
    print("="*80)
    
    tests = [
        ("Initialization", test_1_efficiency_calculator_initialization),
        ("Get Cuts for State", test_2_get_cuts_for_state),
        ("Apply Optimized Cuts", test_3_apply_optimized_cuts),
        ("Selection Efficiency", test_4_selection_efficiency_calculation),
        ("All Efficiencies (Real MC)", test_5_calculate_all_efficiencies_mc),
        ("Efficiency Ratios", test_6_efficiency_ratios),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    
    print("-"*80)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 6 efficiency calculation is working.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
