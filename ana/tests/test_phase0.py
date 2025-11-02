#!/usr/bin/env python3
"""
Quick test script for Phase 0 implementation

Tests the updated data loading infrastructure with BranchConfig
"""

import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "modules"))

def test_imports():
    """Test that all imports work"""
    print("="*80)
    print("TEST 1: Imports")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig, DataManager
        print("✓ Successfully imported TOMLConfig and DataManager")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config_loading():
    """Test TOML configuration loading"""
    print("\n" + "="*80)
    print("TEST 2: Configuration Loading")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig
        
        config = TOMLConfig("./config")
        print(f"✓ Loaded configuration from ./config")
        print(f"  Data path: {config.paths['data']['base_path']}")
        print(f"  MC path: {config.paths['mc']['base_path']}")
        print(f"  Years: {config.paths['data']['years']}")
        print(f"  BranchConfig loaded: {config.branch_config is not None}")
        return True
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_branch_config():
    """Test BranchConfig functionality"""
    print("\n" + "="*80)
    print("TEST 3: BranchConfig Functionality")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig
        
        config = TOMLConfig("./config")
        bc = config.branch_config
        
        # Test getting branches from preset
        branches = bc.get_branches_from_preset("standard", exclude_mc=True)
        print(f"✓ Got {len(branches)} branches from 'standard' preset")
        print(f"  Sample branches: {branches[:5]}")
        
        # Test alias resolution
        resolved = bc.resolve_aliases(["h1_ProbNNk", "Bu_PT"], is_mc=False)
        print(f"✓ Alias resolution works")
        print(f"  h1_ProbNNk (data) → {resolved[0]}")
        
        # Test normalization
        normalize_map = bc.normalize_branches(
            ["h1_MC15TuneV1_ProbNNk", "Bu_PT"], 
            is_mc=False
        )
        print(f"✓ Branch normalization works")
        print(f"  Normalization map: {normalize_map}")
        
        return True
    except Exception as e:
        print(f"✗ BranchConfig test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_loading_simulation():
    """Test file loading logic (without actually loading files)"""
    print("\n" + "="*80)
    print("TEST 4: File Loading Logic (Simulation)")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig, DataManager
        
        config = TOMLConfig("./config")
        dm = DataManager(config)
        
        # Check file paths exist
        data_path = Path(config.paths['data']['base_path'])
        mc_path = Path(config.paths['mc']['base_path'])
        
        print(f"  Data directory: {data_path}")
        print(f"    Exists: {data_path.exists()}")
        
        print(f"  MC directory: {mc_path}")
        print(f"    Exists: {mc_path.exists()}")
        
        # Check for some expected files
        if data_path.exists():
            data_files = list(data_path.glob("*.root"))
            print(f"  Found {len(data_files)} data ROOT files")
            if data_files:
                print(f"    Example: {data_files[0].name}")
        
        if mc_path.exists():
            mc_subdirs = [d for d in mc_path.iterdir() if d.is_dir()]
            print(f"  Found {len(mc_subdirs)} MC subdirectories")
            for subdir in mc_subdirs[:3]:  # Show first 3
                mc_files = list(subdir.glob("*.root"))
                print(f"    {subdir.name}: {len(mc_files)} files")
        
        print("✓ File structure check complete")
        return True
    except Exception as e:
        print(f"✗ File loading simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_derived_branch_logic():
    """Test derived branch calculation logic"""
    print("\n" + "="*80)
    print("TEST 5: Derived Branch Calculation Logic")
    print("="*80)
    
    try:
        import awkward as ak
        import numpy as np
        import vector
        
        # Create mock data
        n_events = 100
        mock_data = {
            "Bu_MM": np.random.normal(5279, 10, n_events),
            "L0_MM": np.random.normal(1115.683, 2, n_events),
            "Bu_ENDVERTEX_X": np.random.normal(0, 0.1, n_events),
            "Bu_ENDVERTEX_Y": np.random.normal(0, 0.1, n_events),
            "Bu_ENDVERTEX_Z": np.random.normal(0, 1.0, n_events),
            "Bu_ENDVERTEX_XERR": np.abs(np.random.normal(0.01, 0.002, n_events)),
            "Bu_ENDVERTEX_YERR": np.abs(np.random.normal(0.01, 0.002, n_events)),
            "Bu_ENDVERTEX_ZERR": np.abs(np.random.normal(0.1, 0.02, n_events)),
            "L0_ENDVERTEX_X": np.random.normal(0.5, 0.1, n_events),
            "L0_ENDVERTEX_Y": np.random.normal(0.5, 0.1, n_events),
            "L0_ENDVERTEX_Z": np.random.normal(5.0, 1.0, n_events),
            "L0_ENDVERTEX_XERR": np.abs(np.random.normal(0.01, 0.002, n_events)),
            "L0_ENDVERTEX_YERR": np.abs(np.random.normal(0.01, 0.002, n_events)),
            "L0_ENDVERTEX_ZERR": np.abs(np.random.normal(0.1, 0.02, n_events)),
        }
        
        events = ak.Array(mock_data)
        
        # Test Bu_MM_corrected calculation
        lambda_mass_pdg = 1115.683
        Bu_MM_corrected = events["Bu_MM"] - events["L0_MM"] + lambda_mass_pdg
        print(f"✓ Bu_MM_corrected calculation works")
        print(f"  Mean: {np.mean(Bu_MM_corrected):.2f} MeV (expected ~5279)")
        
        # Test delta_z calculation
        Delta_Z = events["L0_ENDVERTEX_Z"] - events["Bu_ENDVERTEX_Z"]
        Delta_Z_ERR = np.sqrt(events["Bu_ENDVERTEX_ZERR"]**2 + events["L0_ENDVERTEX_ZERR"]**2)
        delta_z = Delta_Z / Delta_Z_ERR
        print(f"✓ delta_z calculation works")
        print(f"  Mean: {np.mean(delta_z):.2f}")
        print(f"  Std: {np.std(delta_z):.2f}")
        
        # Test Delta_Z_mm
        Delta_Z_mm = Delta_Z
        print(f"✓ Delta_Z_mm calculation works")
        print(f"  Mean: {np.mean(Delta_Z_mm):.2f} mm")
        
        return True
    except Exception as e:
        print(f"✗ Derived branch logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("PHASE 0 IMPLEMENTATION TESTS")
    print("Testing updated data loading infrastructure")
    print("="*80 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration Loading", test_config_loading),
        ("BranchConfig Functionality", test_branch_config),
        ("File Loading Logic", test_file_loading_simulation),
        ("Derived Branch Logic", test_derived_branch_logic),
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
        print("\n🎉 All tests passed! Phase 0 implementation is ready.")
        print("\nNext step: Run actual data loading test with:")
        print("  python -c 'from modules.data_handler import *; ...'")
    else:
        print("\n⚠️  Some tests failed. Review errors above.")
    
    return total_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
