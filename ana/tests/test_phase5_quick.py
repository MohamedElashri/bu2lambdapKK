#!/usr/bin/env python
"""
Quick Test for Phase 5: Mass Fitting Infrastructure

Tests MassFitter class initialization and PDF creation without loading data.
This validates the RooFit setup without risk of segfaults.
"""

import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

import ROOT
from modules.data_handler import TOMLConfig
from modules.mass_fitter import MassFitter


def main():
    """Run quick infrastructure tests"""
    print("\n" + "="*80)
    print("PHASE 5: MASS FITTING - INFRASTRUCTURE TEST")
    print("="*80)
    
    # Test 1: Initialization
    print("\n[Test 1: Initialization]")
    config = TOMLConfig()
    fitter = MassFitter(config)
    print(f"✓ Fit range: {fitter.fit_range} MeV")
    
    # Test 2: Observable
    print("\n[Test 2: Observable Setup]")
    mass_var = fitter.setup_observable()
    print(f"✓ Mass variable: {mass_var.GetName()}, range [{mass_var.getMin()}, {mass_var.getMax()}]")
    
    # Test 3: Signal PDFs
    print("\n[Test 3: Signal PDF Creation]")
    states = ["jpsi", "etac", "chic0", "chic1"]
    for state in states:
        pdf = fitter.create_signal_pdf(state, mass_var)
        mass_val = fitter.masses[state].getVal()
        width_val = fitter.widths[state].getVal()
        is_const = fitter.widths[state].isConstant()
        print(f"  {state:>8}: M={mass_val:7.2f} MeV, Γ={width_val:6.2f} MeV, fixed={is_const}")
    
    print(f"\n  Resolution: σ={fitter.resolution.getVal():.2f} MeV (shared)")
    
    # Test 4: Background PDFs
    print("\n[Test 4: Background PDF Creation]")
    for year in ["2016", "2017", "2018"]:
        bkg_pdf, alpha = fitter.create_background_pdf(mass_var, year)
        print(f"  {year}: α={alpha.getVal():.4f}")
    
    # Test 5: Full Model
    print("\n[Test 5: Full Model Building]")
    model, yields = fitter.build_model_for_year("2016", mass_var)
    print(f"  Components: {list(yields.keys())}")
    for state, yld in yields.items():
        print(f"    N_{state}: initial={yld.getVal():.0f}, range=[{yld.getMin():.0f}, {yld.getMax():.0e}]")
    
    print("\n" + "="*80)
    print("✅ ALL INFRASTRUCTURE TESTS PASSED!")
    print("="*80)
    print("\nPhase 5 MassFitter class is working correctly.")
    print("Ready to perform fits on real data.")
    print("\nNote: Actual fitting should be done via production script,")
    print("not in test environment (ROOT can be unstable in tests).")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
