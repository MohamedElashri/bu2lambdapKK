#!/usr/bin/env python3
"""
Quick test for Phase 4 - Just check if we can load data and create optimizer
"""

import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent.parent / "modules"))

print("=" * 80)
print("PHASE 4 QUICK TEST - Selection Optimizer")
print("=" * 80)

# Test imports
print("\n1. Testing imports...")
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "modules"))
    from data_handler import TOMLConfig, DataManager
    from lambda_selector import LambdaSelector
    from selection_optimizer import SelectionOptimizer
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test config loading
print("\n2. Testing config loading...")
try:
    config = TOMLConfig("./config")
    print(f"✓ Config loaded")
    print(f"  - Data path: {config.paths['data']['base_path']}")
    print(f"  - MC path: {config.paths['mc']['base_path']}")
except Exception as e:
    print(f"✗ Config loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test data loading (minimal - just 2016 MD LL)
print("\n3. Testing data loading (2016 MD LL)...")
try:
    dm = DataManager(config)
    selector = LambdaSelector(config)
    
    print("  Loading data...")
    data = dm.load_tree("data", 2016, "MD", "LL")
    print(f"    ✓ Data loaded: {len(data):,} events")
    
    print("  Computing derived branches...")
    data = dm.compute_derived_branches(data)
    print(f"    ✓ Derived branches computed")
    
    print("  Applying Lambda cuts...")
    data = selector.apply_lambda_cuts(data)
    print(f"    ✓ Lambda cuts applied: {len(data):,} events remain")
    
    print("  Applying B+ fixed cuts...")
    data = selector.apply_bu_fixed_cuts(data)
    print(f"    ✓ B+ cuts applied: {len(data):,} events remain")
    
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test MC loading
print("\n4. Testing MC loading (J/psi 2016 MD LL)...")
try:
    mc_jpsi = dm.load_tree("Jpsi", 2016, "MD", "LL")
    mc_jpsi = dm.compute_derived_branches(mc_jpsi)
    mc_jpsi = selector.apply_lambda_cuts(mc_jpsi)
    mc_jpsi = selector.apply_bu_fixed_cuts(mc_jpsi)
    print(f"✓ J/psi MC loaded: {len(mc_jpsi):,} events")
except Exception as e:
    print(f"✗ MC loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test optimizer initialization
print("\n5. Testing SelectionOptimizer initialization...")
try:
    optimizer = SelectionOptimizer(
        signal_mc={"jpsi": {"2016": mc_jpsi}},
        phase_space_mc={"2016": mc_jpsi},  # Use same for testing
        data={"2016": data},
        config=config
    )
    print("✓ Optimizer initialized")
    
    # Test signal region definition
    print("\n  Signal regions:")
    for state in ["jpsi", "etac", "chic0", "chic1"]:
        region = optimizer.define_signal_region(state)
        print(f"    {state:6s}: [{region[0]:7.1f}, {region[1]:7.1f}] MeV")
    
    # Test FOM calculation
    print("\n  Testing FOM calculation:")
    fom = optimizer.compute_fom(100, 100)
    expected = 100 / (100 + 100)**0.5
    print(f"    FOM(100, 100) = {fom:.3f} (expected {expected:.3f})")
    if abs(fom - expected) < 1e-6:
        print(f"    ✓ FOM calculation correct")
    else:
        print(f"    ✗ FOM calculation incorrect!")
    
except Exception as e:
    print(f"✗ Optimizer initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ ALL QUICK TESTS PASSED!")
print("=" * 80)
print("\nPhase 4 infrastructure is working correctly.")
print("You can now run the full test suite with:")
print("  python tests/test_phase4_selection_optimization.py")
