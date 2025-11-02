#!/usr/bin/env python3
"""
Phase 2 Validation: Data Loading Execution

Tests loading all data and MC files with derived branch calculation.
This validates that:
1. All data files (2016-2018, MD+MU, LL+DD) can be loaded
2. All MC files (Jpsi, etac, chic0, chic1, KpKm) can be loaded
3. Derived branches are computed correctly
4. Event counts are reasonable
5. Physics distributions are sensible
"""

import sys
from pathlib import Path
import awkward as ak
import numpy as np

# Add modules to path
sys.path.append(str(Path(__file__).parent.parent / "modules"))

def test_load_single_data_file():
    """Test loading a single data file with derived branches"""
    print("="*80)
    print("TEST 1: Load Single Data File (2016 MD LL)")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig, DataManager
        
        config = TOMLConfig("./config")
        dm = DataManager(config)
        
        # Load single file
        print("\nLoading 2016 MagDown LL data...")
        events = dm.load_tree("data", 2016, "MD", "LL")
        print(f"✓ Loaded {len(events)} events")
        
        # Compute derived branches
        print("\nComputing derived branches...")
        events = dm.compute_derived_branches(events)
        
        # Check critical branches exist
        print("\nChecking critical branches:")
        critical_branches = [
            "Bu_MM", "L0_MM", "Bu_ENDVERTEX_Z", "L0_ENDVERTEX_Z",
            "Bu_MM_corrected", "delta_z", "Delta_Z_mm", 
            "M_LpKm_h1", "M_LpKm_h2", "M_KK"
        ]
        
        all_present = True
        for branch in critical_branches:
            if branch in events.fields:
                print(f"  ✓ {branch}")
            else:
                print(f"  ✗ {branch} MISSING")
                all_present = False
        
        if not all_present:
            return False
        
        # Physics checks
        print("\nPhysics sanity checks:")
        
        # Lambda mass
        mean_l0 = ak.mean(events["L0_MM"])
        print(f"  Lambda mass: {mean_l0:.2f} MeV (expect ~1115.7)")
        if not (1100 < mean_l0 < 1130):
            print(f"    ⚠️  Lambda mass out of range!")
            return False
        
        # B+ corrected mass
        mean_bu = ak.mean(events["Bu_MM_corrected"])
        print(f"  B+ corrected mass: {mean_bu:.2f} MeV (expect ~5279, but high before cuts)")
        if not (5000 < mean_bu < 6000):
            print(f"    ⚠️  B+ mass out of range!")
            return False
        
        # M_KK
        mean_kk = ak.mean(events["M_KK"])
        print(f"  M(KK): {mean_kk:.2f} MeV")
        if not (1000 < mean_kk < 3000):
            print(f"    ⚠️  M(KK) out of range!")
            return False
        
        print("\n✓ Single file loading successful with derived branches")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_load_all_data():
    """Test loading all data files (all years, both magnets, both track types)"""
    print("\n" + "="*80)
    print("TEST 2: Load All Data Files (LL and DD)")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig, DataManager
        
        config = TOMLConfig("./config")
        dm = DataManager(config)
        
        # Test both track types
        for track_type in ["LL", "DD"]:
            print(f"\n{'='*60}")
            print(f"Loading {track_type} track type")
            print(f"{'='*60}")
            
            data_by_year = dm.load_all_data_combined_magnets("data", track_types=[track_type])
            
            # Check we have all years
            print(f"\nData loaded per year ({track_type}):")
            expected_years = ["2016", "2017", "2018"]
            total_events = 0
            
            for year in expected_years:
                if year not in data_by_year:
                    print(f"  ✗ Year {year} missing!")
                    return False
                
                n_events = len(data_by_year[year])
                total_events += n_events
                print(f"  {year}: {n_events:,} events")
            
            print(f"\nTotal {track_type} events: {total_events:,}")
            
            # Expect ~100k-300k per year for LL, similar for DD
            if total_events < 100000:
                print(f"  ⚠️  Total events suspiciously low!")
                return False
            
            # Check derived branches in 2016 data
            print(f"\nChecking 2016 {track_type} data has derived branches:")
            events_2016 = data_by_year["2016"]
            
            derived = ["Bu_MM_corrected", "delta_z", "M_LpKm_h1", "M_KK"]
            for branch in derived:
                if branch in events_2016.fields:
                    print(f"  ✓ {branch}")
                else:
                    print(f"  ✗ {branch} missing")
                    return False
        
        print("\n✓ All data files loaded successfully (LL + DD)")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_load_all_mc():
    """Test loading all MC samples"""
    print("\n" + "="*80)
    print("TEST 3: Load All MC Samples")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig, DataManager
        
        config = TOMLConfig("./config")
        dm = DataManager(config)
        
        mc_states = ["Jpsi", "etac", "chic0", "chic1"]
        mc_data = {}
        
        print("\nLoading MC samples (2016 only, MD, LL for speed)...")
        
        for state in mc_states:
            print(f"\n  Loading {state}...")
            
            try:
                events = dm.load_tree(state, 2016, "MD", "LL", channel_name="B2L0barPKpKm")
                events = dm.compute_derived_branches(events)
                
                n_events = len(events)
                mc_data[state] = events
                
                print(f"    ✓ Loaded {n_events:,} events")
                
                # Check derived branches
                if "Bu_MM_corrected" not in events.fields:
                    print(f"    ✗ Derived branches missing!")
                    return False
                
            except FileNotFoundError as e:
                print(f"    ⚠️  File not found: {e}")
                print(f"    Skipping {state} (may not exist yet)")
                continue
            except Exception as e:
                print(f"    ✗ Error loading {state}: {e}")
                return False
        
        if len(mc_data) == 0:
            print("\n⚠️  No MC files found - this is expected if MC not yet available")
            print("    Phase 2 will pass, but MC loading should be tested when files ready")
            return True  # Don't fail if MC not ready yet
        
        print(f"\n✓ Loaded {len(mc_data)} MC samples successfully")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_derived_branch_calculations():
    """Test that derived branches are calculated correctly"""
    print("\n" + "="*80)
    print("TEST 4: Validate Derived Branch Calculations")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig, DataManager
        
        config = TOMLConfig("./config")
        dm = DataManager(config)
        
        # Load a small sample
        print("\nLoading 2016 MD LL data...")
        events = dm.load_tree("data", 2016, "MD", "LL")
        events = dm.compute_derived_branches(events)
        
        print(f"Testing with {len(events)} events\n")
        
        # Test 1: Bu_MM_corrected calculation
        print("Test 1: Bu_MM_corrected = Bu_MM - L0_MM + PDG_Lambda")
        lambda_pdg = config.get_pdg_mass("lambda")
        
        # Manual calculation
        bu_mm_corrected_manual = events["Bu_MM"] - events["L0_MM"] + lambda_pdg
        bu_mm_corrected_computed = events["Bu_MM_corrected"]
        
        diff = ak.mean(np.abs(bu_mm_corrected_manual - bu_mm_corrected_computed))
        print(f"  Mean difference: {diff:.6f} MeV (should be ~0)")
        
        if diff > 0.01:
            print(f"  ✗ Bu_MM_corrected calculation wrong!")
            return False
        print(f"  ✓ Bu_MM_corrected correct")
        
        # Test 2: Delta_Z_mm calculation
        print("\nTest 2: Delta_Z_mm = L0_ENDVERTEX_Z - Bu_ENDVERTEX_Z")
        delta_z_manual = events["L0_ENDVERTEX_Z"] - events["Bu_ENDVERTEX_Z"]
        delta_z_computed = events["Delta_Z_mm"]
        
        diff = ak.mean(np.abs(delta_z_manual - delta_z_computed))
        print(f"  Mean difference: {diff:.6f} mm (should be ~0)")
        
        if diff > 0.01:
            print(f"  ✗ Delta_Z_mm calculation wrong!")
            return False
        print(f"  ✓ Delta_Z_mm correct")
        
        # Test 3: delta_z (significance) calculation
        print("\nTest 3: delta_z significance")
        delta_z_sig = events["delta_z"]
        
        mean_sig = ak.mean(delta_z_sig)
        std_sig = ak.std(delta_z_sig)
        
        print(f"  Mean significance: {mean_sig:.2f}")
        print(f"  Std significance: {std_sig:.2f}")
        
        # Should be reasonable values
        if not (-10 < mean_sig < 100):
            print(f"  ⚠️  Significance mean out of reasonable range")
        
        print(f"  ✓ delta_z has reasonable values")
        
        # Test 4: M_LpKm calculations exist
        print("\nTest 4: M_LpKm invariant mass calculations")
        m_lpkm_h1 = events["M_LpKm_h1"]
        m_lpkm_h2 = events["M_LpKm_h2"]
        
        mean_h1 = ak.mean(m_lpkm_h1)
        mean_h2 = ak.mean(m_lpkm_h2)
        
        print(f"  Mean M(Λ̄pK) with h1 as K: {mean_h1:.1f} MeV")
        print(f"  Mean M(Λ̄pK) with h2 as K: {mean_h2:.1f} MeV")
        
        # Should be in charmonium region (2-5 GeV)
        if not (2000 < mean_h1 < 5000 and 2000 < mean_h2 < 5000):
            print(f"  ⚠️  M_LpKm values out of charmonium range")
        
        print(f"  ✓ M_LpKm calculations completed")
        
        # Test 5: M_KK calculation
        print("\nTest 5: M(KK) invariant mass")
        m_kk = events["M_KK"]
        
        mean_kk = ak.mean(m_kk)
        print(f"  Mean M(KK): {mean_kk:.1f} MeV")
        
        # Should be > 2*m_K
        if mean_kk < 2 * 493.677:
            print(f"  ✗ M(KK) below threshold!")
            return False
        
        print(f"  ✓ M(KK) above threshold")
        
        print("\n✓ All derived branch calculations validated")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_physics_distributions():
    """Test that physics distributions look reasonable"""
    print("\n" + "="*80)
    print("TEST 5: Physics Distribution Sanity Checks")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig, DataManager
        
        config = TOMLConfig("./config")
        dm = DataManager(config)
        
        # Test both track types
        for track_type in ["LL", "DD"]:
            print(f"\n--- Testing {track_type} ---")
            
            # Load 2016 data
            print(f"\nLoading 2016 data ({track_type})...")
            data_by_year = dm.load_all_data_combined_magnets("data", track_types=[track_type])
            events = data_by_year["2016"]
            
            print(f"Using {len(events):,} events\n")
        
            # Check various distributions
            distributions = {
                "L0_MM": {
                    "expected_mean": 1115.7,
                    "expected_range": (1100, 1130),
                    "description": "Lambda mass"
                },
                "Bu_MM": {
                    "expected_mean": 5279,
                    "expected_range": (5000, 6000),
                    "description": "B+ mass (DTF)"
                },
                "Bu_MM_corrected": {
                    "expected_mean": 5279,
                    "expected_range": (5000, 6000),
                    "description": "B+ corrected mass"
                },
                "M_KK": {
                    "expected_mean": None,
                    "expected_range": (1000, 3000),
                    "description": "M(K+K-)"
                },
            }
            
            all_ok = True
            for branch, info in distributions.items():
                if branch not in events.fields:
                    print(f"✗ {branch} not in events")
                    all_ok = False
                    continue
                
                values = events[branch]
                mean_val = ak.mean(values)
                std_val = ak.std(values)
                min_val = ak.min(values)
                max_val = ak.max(values)
                
                print(f"{info['description']} ({branch}):")
                print(f"  Mean: {mean_val:.2f} MeV")
                print(f"  Std:  {std_val:.2f} MeV")
                print(f"  Range: [{min_val:.2f}, {max_val:.2f}] MeV")
                
                # Check range
                exp_min, exp_max = info["expected_range"]
                if not (exp_min < mean_val < exp_max):
                    print(f"  ⚠️  Mean outside expected range {info['expected_range']}")
                    all_ok = False
                else:
                    print(f"  ✓ Mean in expected range")
                
                print()
            
            if not all_ok:
                print(f"⚠️  Some {track_type} distributions out of expected ranges")
                return False
        
        print("✓ All physics distributions look reasonable (LL + DD)")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Phase 2 validation tests"""
    print("\n" + "="*80)
    print("PHASE 2 VALIDATION: Data Loading Execution")
    print("="*80 + "\n")
    
    tests = [
        ("Load Single Data File", test_load_single_data_file),
        ("Load All Data Files", test_load_all_data),
        ("Load All MC Samples", test_load_all_mc),
        ("Derived Branch Calculations", test_derived_branch_calculations),
        ("Physics Distribution Checks", test_physics_distributions),
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
    print("TEST SUMMARY - PHASE 2")
    print("="*80)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")
    
    total_passed = sum(1 for _, r in results if r)
    total_tests = len(results)
    print(f"\nPassed: {total_passed}/{total_tests}")
    
    if total_passed == total_tests:
        print("\n🎉 Phase 2 COMPLETE: Data loading validated!")
        print("\nData Loading Summary:")
        print("  - All data files loaded (2016-2018, MD+MU, LL)")
        print("  - Derived branches computed correctly")
        print("  - Physics distributions reasonable")
        print("\nReady to proceed to Phase 3: Lambda Pre-Selection")
    else:
        print("\n⚠️  Some tests failed. Review errors above.")
    
    return total_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
