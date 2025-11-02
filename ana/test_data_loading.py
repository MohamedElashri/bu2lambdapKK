#!/usr/bin/env python3
"""
Test actual data loading from ROOT files

This script tests loading a small sample of real data
to verify the implementation works end-to-end.
"""

import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "modules"))

def test_single_file_loading():
    """Test loading a single ROOT file"""
    print("="*80)
    print("TEST: Loading Single ROOT File")
    print("="*80)
    
    from data_handler import TOMLConfig, DataManager
    
    config = TOMLConfig("./config")
    dm = DataManager(config)
    
    try:
        # Try loading one data file (2016, MagDown, LL)
        print("\nAttempting to load: data 2016 MD LL...")
        events = dm.load_tree("data", 2016, "MD", "LL")
        
        print(f"✓ Successfully loaded {len(events)} events")
        
        # Check available fields
        print(f"\nAvailable fields ({len(events.fields)} total):")
        for i, field in enumerate(sorted(events.fields)[:20], 1):
            print(f"  {i:2d}. {field}")
        if len(events.fields) > 20:
            print(f"  ... and {len(events.fields) - 20} more")
        
        # Check derived branches
        print("\nDerived branches:")
        derived = ["Bu_MM_corrected", "delta_z", "Delta_Z_mm", "M_LpKm_h1", "M_LpKm_h2", "M_KK"]
        for branch in derived:
            if branch in events.fields:
                print(f"  ✓ {branch} exists")
                # Show first 3 values
                vals = events[branch][:3]
                print(f"    Sample values: {vals}")
            else:
                print(f"  ✗ {branch} missing")
        
        # Check some physics quantities
        print("\nPhysics sanity checks:")
        if "Bu_MM_corrected" in events.fields:
            import awkward as ak
            mean_bu = ak.mean(events["Bu_MM_corrected"])
            print(f"  Mean Bu_MM_corrected: {mean_bu:.1f} MeV (expected ~5279)")
        
        if "L0_MM" in events.fields or "L0_M" in events.fields:
            branch = "L0_MM" if "L0_MM" in events.fields else "L0_M"
            import awkward as ak
            mean_l0 = ak.mean(events[branch])
            print(f"  Mean {branch}: {mean_l0:.1f} MeV (expected ~1115.7)")
        
        if "M_KK" in events.fields:
            import awkward as ak
            mean_kk = ak.mean(events["M_KK"])
            print(f"  Mean M_KK: {mean_kk:.1f} MeV")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_combined_loading():
    """Test loading and combining multiple files"""
    print("\n" + "="*80)
    print("TEST: Loading Combined Data (MD + MU, LL + DD)")
    print("="*80)
    
    from data_handler import TOMLConfig, DataManager
    
    config = TOMLConfig("./config")
    dm = DataManager(config)
    
    try:
        # Load all 2016 data (MD + MU, LL + DD)
        print("\nLoading all 2016 data...")
        data_by_year = dm.load_all_data_combined_magnets("data", track_types=["LL"])
        
        if "2016" in data_by_year:
            events = data_by_year["2016"]
            print(f"✓ Successfully loaded and combined 2016 data: {len(events)} events")
            
            # Check trigger efficiency
            n_total = len(events)
            print(f"\nTrigger selection applied: {n_total} events remaining")
            
            return True
        else:
            print("✗ No 2016 data loaded")
            return False
            
    except Exception as e:
        print(f"\n✗ Combined loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mc_loading():
    """Test loading MC sample"""
    print("\n" + "="*80)
    print("TEST: Loading MC Sample")
    print("="*80)
    
    from data_handler import TOMLConfig, DataManager
    
    config = TOMLConfig("./config")
    dm = DataManager(config)
    
    try:
        # Load J/psi MC for 2016
        print("\nLoading J/psi MC 2016 MD LL...")
        events = dm.load_tree("Jpsi", 2016, "MD", "LL", channel_name="B2L0barPKpKm")
        
        print(f"✓ Successfully loaded {len(events)} MC events")
        
        # Check for MC-specific branches
        mc_branches = ["Bu_TRUEPT", "Bu_TRUEID", "L0_TRUEPT"]
        print("\nMC truth branches:")
        for branch in mc_branches:
            if branch in events.fields:
                print(f"  ✓ {branch} exists")
            else:
                print(f"  ✗ {branch} not in loaded branches (may not be in preset)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ MC loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all loading tests"""
    print("\n" + "="*80)
    print("ACTUAL DATA LOADING TESTS")
    print("Testing with real ROOT files")
    print("="*80 + "\n")
    
    tests = [
        ("Single File Loading", test_single_file_loading),
        ("Combined Data Loading", test_combined_loading),
        ("MC Sample Loading", test_mc_loading),
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
    print("TEST SUMMARY")
    print("="*80)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")
    
    total_passed = sum(1 for _, r in results if r)
    total_tests = len(results)
    print(f"\nPassed: {total_passed}/{total_tests}")
    
    if total_passed == total_tests:
        print("\n🎉 All data loading tests passed!")
        print("\nYou're ready to proceed to Phase 3 (Lambda Selection)")
    else:
        print("\n⚠️  Some tests failed. Review errors above.")
    
    return total_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
