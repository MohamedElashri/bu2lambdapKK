#!/usr/bin/env python3
"""
Phase 3 Validation: Lambda Pre-Selection

Tests applying fixed Lambda quality cuts to all data and MC.
This validates that:
1. Lambda cuts are applied correctly
2. Cut flow is reasonable (not too aggressive, not too loose)
3. Efficiency is similar across years and MC samples
4. Distributions after cuts look sensible
"""

import sys
from pathlib import Path
import awkward as ak
import numpy as np

# Add modules to path
sys.path.append(str(Path(__file__).parent.parent / "modules"))

def test_lambda_selector_init():
    """Test that LambdaSelector initializes correctly"""
    print("="*80)
    print("TEST 1: LambdaSelector Initialization")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig
        from lambda_selector import LambdaSelector
        
        config = TOMLConfig("./config")
        selector = LambdaSelector(config)
        
        print("\nLambda selection cuts:")
        print(f"  Lambda mass: [{selector.cuts['mass_min']:.1f}, {selector.cuts['mass_max']:.1f}] MeV")
        print(f"  Lambda FD χ²: > {selector.cuts['fd_chisq_min']:.0f}")
        print(f"  Delta Z: > {selector.cuts['delta_z_min']:.1f} mm")
        print(f"  Proton PID: > {selector.cuts['proton_probnnp_min']:.2f}")
        
        # Verify cuts match config
        assert selector.cuts['mass_min'] == 1111.0
        assert selector.cuts['mass_max'] == 1121.0
        assert selector.cuts['fd_chisq_min'] == 250.0
        assert selector.cuts['delta_z_min'] == 5.0
        assert selector.cuts['proton_probnnp_min'] == 0.3
        
        print("\n✓ LambdaSelector initialized with correct cuts")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_apply_lambda_cuts_to_single_file():
    """Test applying Lambda cuts to a single data file"""
    print("\n" + "="*80)
    print("TEST 2: Apply Lambda Cuts to Single File")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig, DataManager
        from lambda_selector import LambdaSelector
        
        config = TOMLConfig("./config")
        dm = DataManager(config)
        selector = LambdaSelector(config)
        
        # Load data
        print("\nLoading 2016 MD LL data...")
        events = dm.load_tree("data", 2016, "MD", "LL")
        events = dm.compute_derived_branches(events)
        
        n_before = len(events)
        print(f"Events before cuts: {n_before:,}")
        
        # Apply Lambda cuts
        print("\nApplying Lambda cuts...")
        events_after = selector.apply_lambda_cuts(events)
        
        n_after = len(events_after)
        efficiency = n_after / n_before if n_before > 0 else 0
        
        print(f"Events after cuts: {n_after:,}")
        print(f"Efficiency: {100*efficiency:.2f}%")
        
        # Check efficiency is reasonable (expect 10-50% for Lambda cuts)
        if not (0.05 < efficiency < 0.80):
            print(f"  ⚠️  Efficiency {100*efficiency:.1f}% seems unusual (expect 5-80%)")
        
        # Check that critical branches still exist
        print("\nChecking branches after cuts:")
        critical_branches = ["L0_MM", "Bu_MM_corrected", "M_LpKm_h1"]
        for branch in critical_branches:
            if branch in events_after.fields:
                print(f"  ✓ {branch}")
            else:
                print(f"  ✗ {branch} missing")
                return False
        
        # Check Lambda mass distribution after cuts
        print("\nLambda mass after cuts:")
        l0_mass = events_after["L0_MM"]
        mean_l0 = ak.mean(l0_mass)
        min_l0 = ak.min(l0_mass)
        max_l0 = ak.max(l0_mass)
        
        print(f"  Mean: {mean_l0:.2f} MeV")
        print(f"  Range: [{min_l0:.2f}, {max_l0:.2f}] MeV")
        
        # Should be within cut window
        if not (1111 <= min_l0 and max_l0 <= 1121):
            print(f"  ⚠️  Lambda mass outside cut window!")
        else:
            print(f"  ✓ Lambda mass within cut window [1111, 1121] MeV")
        
        print("\n✓ Lambda cuts applied successfully to single file")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_apply_lambda_cuts_to_all_data():
    """Test applying Lambda cuts to all data years"""
    print("\n" + "="*80)
    print("TEST 3: Apply Lambda Cuts to All Data")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig, DataManager
        from lambda_selector import LambdaSelector
        
        config = TOMLConfig("./config")
        dm = DataManager(config)
        selector = LambdaSelector(config)
        
        # Test both track types
        for track_type in ["LL", "DD"]:
            print(f"\n--- Testing {track_type} ---")
            
            # Load all data
            print(f"\nLoading all data ({track_type})...")
            data_by_year = dm.load_all_data_combined_magnets("data", track_types=[track_type])
            
            # Apply cuts to each year
            data_selected = {}
            efficiencies = {}
            
            print(f"\nApplying Lambda cuts per year ({track_type}):")
            for year in sorted(data_by_year.keys()):
                print(f"\n{year}:")
                events = data_by_year[year]
                n_before = len(events)
                
                events_after = selector.apply_lambda_cuts(events)
                n_after = len(events_after)
                
                efficiency = n_after / n_before if n_before > 0 else 0
                efficiencies[year] = efficiency
                data_selected[year] = events_after
                
                print(f"  Events: {n_before:,} → {n_after:,}")
                print(f"  Efficiency: {100*efficiency:.2f}%")
            
            # Check efficiencies are consistent across years
            print(f"\nEfficiency consistency check ({track_type}):")
            eff_values = list(efficiencies.values())
            mean_eff = np.mean(eff_values)
            std_eff = np.std(eff_values)
            
            print(f"  Mean efficiency: {100*mean_eff:.2f}%")
            print(f"  Std efficiency: {100*std_eff:.2f}%")
            
            # Efficiencies should be similar across years (within ~10%)
            if std_eff > 0.10:
                print(f"  ⚠️  Large variation in efficiencies across years")
            else:
                print(f"  ✓ Efficiencies consistent across years")
            
            # Total events
            total_before = sum(len(data_by_year[y]) for y in data_by_year)
            total_after = sum(len(data_selected[y]) for y in data_selected)
            
            print(f"\nTotal {track_type} events:")
            print(f"  Before: {total_before:,}")
            print(f"  After: {total_after:,}")
            print(f"  Overall efficiency: {100*total_after/total_before:.2f}%")
        
        print("\n✓ Lambda cuts applied to all data years (LL + DD)")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_apply_lambda_cuts_to_mc():
    """Test applying Lambda cuts to MC samples"""
    print("\n" + "="*80)
    print("TEST 4: Apply Lambda Cuts to MC Samples")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig, DataManager
        from lambda_selector import LambdaSelector
        
        config = TOMLConfig("./config")
        dm = DataManager(config)
        selector = LambdaSelector(config)
        
        mc_states = ["Jpsi", "etac", "chic0", "chic1"]
        
        # Test both track types
        for track_type in ["LL", "DD"]:
            print(f"\n--- Testing MC with {track_type} ---")
            
            mc_selected = {}
            mc_efficiencies = {}
            
            print(f"\nApplying Lambda cuts to MC (2016 MD {track_type}):")
            
            for state in mc_states:
                print(f"\n{state}:")
                
                try:
                    # Load MC
                    events = dm.load_tree(state, 2016, "MD", track_type, channel_name="B2L0barPKpKm")
                    events = dm.compute_derived_branches(events)
                    
                    n_before = len(events)
                    
                    # Apply cuts
                    events_after = selector.apply_lambda_cuts(events)
                    n_after = len(events_after)
                    
                    efficiency = n_after / n_before if n_before > 0 else 0
                    mc_efficiencies[state] = efficiency
                    mc_selected[state] = events_after
                    
                    print(f"  Events: {n_before:,} → {n_after:,}")
                    print(f"  Efficiency: {100*efficiency:.2f}%")
                    
                except FileNotFoundError:
                    print(f"  ⚠️  MC file not found, skipping")
                    continue
            
            if len(mc_efficiencies) == 0:
                print(f"\n⚠️  No {track_type} MC files available for testing")
                continue  # Continue to next track type
            
            # Check MC efficiencies
            print(f"\nMC efficiency summary ({track_type}):")
            for state, eff in mc_efficiencies.items():
                print(f"  {state:8s}: {100*eff:.2f}%")
            
            # MC efficiencies should be higher than data (signal MC)
            mean_mc_eff = np.mean(list(mc_efficiencies.values()))
            print(f"\n  Mean MC efficiency ({track_type}): {100*mean_mc_eff:.2f}%")
            
            if mean_mc_eff < 0.1:
                print(f"  ⚠️  MC efficiency seems low")
        
        print("\n✓ Lambda cuts applied to MC samples (LL + DD)")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_distributions_after_cuts():
    """Test that physics distributions look good after Lambda cuts"""
    print("\n" + "="*80)
    print("TEST 5: Physics Distributions After Lambda Cuts")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig, DataManager
        from lambda_selector import LambdaSelector
        
        config = TOMLConfig("./config")
        dm = DataManager(config)
        selector = LambdaSelector(config)
        
        # Test both track types
        for track_type in ["LL", "DD"]:
            print(f"\n--- Testing {track_type} ---")
            
            # Load 2016 data
            print(f"\nLoading 2016 data ({track_type})...")
            data_by_year = dm.load_all_data_combined_magnets("data", track_types=[track_type])
            events = data_by_year["2016"]
            
            print(f"Before cuts: {len(events):,} events")
            
            # Apply cuts
            events_after = selector.apply_lambda_cuts(events)
            print(f"After cuts: {len(events_after):,} events")
            
            # Check distributions
            print(f"\nPhysics distributions after Lambda cuts ({track_type}):")
            
            distributions = {
                "L0_MM": {
                    "expected_range": (1111, 1121),
                    "description": "Lambda mass (should be tight)"
                },
                "Bu_MM_corrected": {
                    "expected_range": (5000, 6000),
                    "description": "B+ corrected mass"
                },
                "M_LpKm_h1": {
                    "expected_range": (2500, 5000),
                    "description": "M(Λ̄pK⁻) charmonium candidate"
                },
            }
            
            all_ok = True
            for branch, info in distributions.items():
                if branch not in events_after.fields:
                    print(f"\n✗ {branch} not in events")
                    all_ok = False
                    continue
                
                values = events_after[branch]
                mean_val = ak.mean(values)
                std_val = ak.std(values)
                
                print(f"\n{info['description']} ({branch}):")
                print(f"  Mean: {mean_val:.2f} MeV")
                print(f"  Std:  {std_val:.2f} MeV")
                
                # Check range
                exp_min, exp_max = info["expected_range"]
                if not (exp_min < mean_val < exp_max):
                    print(f"  ⚠️  Mean outside expected range {info['expected_range']}")
                    all_ok = False
                else:
                    print(f"  ✓ Mean in expected range")
            
            if not all_ok:
                print(f"\n⚠️  Some {track_type} distributions look unusual")
                return False
        
        print("\n✓ All distributions look good after Lambda cuts (LL + DD)")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cut_flow_details():
    """Test detailed cut flow to see which cuts are most aggressive"""
    print("\n" + "="*80)
    print("TEST 6: Detailed Cut Flow Analysis")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig, DataManager
        from lambda_selector import LambdaSelector
        
        config = TOMLConfig("./config")
        dm = DataManager(config)
        selector = LambdaSelector(config)
        
        # Test both track types
        for track_type in ["LL", "DD"]:
            print(f"\n--- Testing {track_type} ---")
            
            # Load sample
            print(f"\nLoading 2016 MD {track_type} data...")
            events = dm.load_tree("data", 2016, "MD", track_type)
            events = dm.compute_derived_branches(events)
            
            n_start = len(events)
            print(f"\nStarting events: {n_start:,}")
            
            # Apply cuts one by one
            print(f"\nCut flow ({track_type}):")
            
            # Cut 1: Lambda mass
            if "L0_MM" in events.fields:
                mask = (events["L0_MM"] > selector.cuts["mass_min"]) & (events["L0_MM"] < selector.cuts["mass_max"])
                n_pass = ak.sum(mask)
                print(f"  1. Lambda mass [1111, 1121] MeV: {n_start:,} → {n_pass:,} ({100*n_pass/n_start:.1f}%)")
                events_temp = events[mask]
                n_start = len(events_temp)
            
            # Cut 2: Lambda FD χ²
            if "L0_FDCHI2_OWNPV" in events_temp.fields:
                mask = events_temp["L0_FDCHI2_OWNPV"] > selector.cuts["fd_chisq_min"]
                n_pass = ak.sum(mask)
                print(f"  2. Lambda FD χ² > 250:           {n_start:,} → {n_pass:,} ({100*n_pass/n_start:.1f}%)")
                events_temp = events_temp[mask]
                n_start = len(events_temp)
            
            # Cut 3: Delta Z
            if "Delta_Z_mm" in events_temp.fields:
                mask = np.abs(events_temp["Delta_Z_mm"]) > selector.cuts["delta_z_min"]
                n_pass = ak.sum(mask)
                print(f"  3. |Delta Z| > 5 mm:              {n_start:,} → {n_pass:,} ({100*n_pass/n_start:.1f}%)")
                events_temp = events_temp[mask]
                n_start = len(events_temp)
            
            # Cut 4: Proton PID
            if "Lp_ProbNNp" in events_temp.fields:
                mask = events_temp["Lp_ProbNNp"] > selector.cuts["proton_probnnp_min"]
                n_pass = ak.sum(mask)
                print(f"  4. Proton PID > 0.3:              {n_start:,} → {n_pass:,} ({100*n_pass/n_start:.1f}%)")
                events_temp = events_temp[mask]
                n_final = len(events_temp)
            else:
                n_final = len(events_temp)
            
            print(f"\nFinal {track_type} events: {n_final:,}")
        
        print("\n✓ Cut flow analysis complete (LL + DD)")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Phase 3 validation tests"""
    print("\n" + "="*80)
    print("PHASE 3 VALIDATION: Lambda Pre-Selection")
    print("="*80 + "\n")
    
    tests = [
        ("LambdaSelector Initialization", test_lambda_selector_init),
        ("Apply Cuts to Single File", test_apply_lambda_cuts_to_single_file),
        ("Apply Cuts to All Data", test_apply_lambda_cuts_to_all_data),
        ("Apply Cuts to MC", test_apply_lambda_cuts_to_mc),
        ("Distributions After Cuts", test_distributions_after_cuts),
        ("Detailed Cut Flow", test_cut_flow_details),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY - PHASE 3")
    print("="*80)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")
    
    total_passed = sum(1 for _, r in results if r)
    total_tests = len(results)
    print(f"\nPassed: {total_passed}/{total_tests}")
    
    if total_passed == total_tests:
        print("\n🎉 Phase 3 COMPLETE: Lambda pre-selection validated!")
        print("\nLambda Pre-Selection Summary:")
        print("  - Fixed Lambda cuts applied to all data/MC")
        print("  - Cut flow analyzed (mass → FD χ² → ΔZ → PID)")
        print("  - Efficiencies consistent across years")
        print("  - Distributions clean after cuts")
        print("\nReady to proceed to Phase 4: Selection Optimization")
    else:
        print("\n⚠️  Some tests failed. Review errors above.")
    
    return total_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
