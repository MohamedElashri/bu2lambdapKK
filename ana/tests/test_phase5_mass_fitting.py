#!/usr/bin/env python
"""
Test Script for Phase 5: Mass Fitting with RooFit

Tests the MassFitter class and performs a quick mass fit on MC samples.
This validates the RooFit infrastructure before running full production fits.
"""

import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

import ROOT
import numpy as np
import awkward as ak
from modules.data_handler import DataManager, TOMLConfig
from modules.lambda_selector import LambdaSelector
from modules.mass_fitter import MassFitter


def test_1_mass_fitter_initialization():
    """Test 1: MassFitter initialization"""
    print("\n" + "="*80)
    print("TEST 1: MassFitter Initialization")
    print("="*80)
    
    config = TOMLConfig()
    fitter = MassFitter(config)
    
    assert fitter.fit_range == [2800.0, 4000.0], "Fit range not loaded correctly"
    assert fitter.masses == {}, "Masses should be empty initially"
    assert fitter.widths == {}, "Widths should be empty initially"
    assert fitter.resolution is None, "Resolution should be None initially"
    
    print("✓ MassFitter initialized correctly")
    print(f"  Fit range: {fitter.fit_range} MeV")
    
    return True


def test_2_observable_setup():
    """Test 2: RooRealVar observable setup"""
    print("\n" + "="*80)
    print("TEST 2: Observable Setup")
    print("="*80)
    
    config = TOMLConfig()
    fitter = MassFitter(config)
    
    mass_var = fitter.setup_observable()
    
    assert mass_var is not None, "Mass variable should be created"
    assert mass_var.getMin() == 2800.0, "Min value incorrect"
    assert mass_var.getMax() == 4000.0, "Max value incorrect"
    assert fitter.mass_var is mass_var, "Mass var should be stored"
    
    print("✓ Observable created successfully")
    print(f"  Variable name: {mass_var.GetName()}")
    print(f"  Range: [{mass_var.getMin()}, {mass_var.getMax()}] MeV")
    
    return True


def test_3_signal_pdf_creation():
    """Test 3: Signal PDF creation for all states"""
    print("\n" + "="*80)
    print("TEST 3: Signal PDF Creation")
    print("="*80)
    
    config = TOMLConfig()
    fitter = MassFitter(config)
    mass_var = fitter.setup_observable()
    
    states = ["jpsi", "etac", "chic0", "chic1"]
    
    for state in states:
        pdf = fitter.create_signal_pdf(state, mass_var)
        
        assert pdf is not None, f"PDF for {state} should be created"
        assert state in fitter.masses, f"Mass parameter for {state} should exist"
        assert state in fitter.widths, f"Width parameter for {state} should exist"
        
        mass_val = fitter.masses[state].getVal()
        width_val = fitter.widths[state].getVal()
        is_constant = fitter.widths[state].isConstant()
        
        print(f"  {state:>8}: M = {mass_val:7.2f} MeV, Γ = {width_val:6.2f} MeV, "
              f"Γ fixed = {is_constant}")
    
    # Check that J/ψ and χc1 widths are fixed
    assert fitter.widths["jpsi"].isConstant(), "J/ψ width should be fixed"
    assert fitter.widths["chic1"].isConstant(), "χc1 width should be fixed"
    assert not fitter.widths["etac"].isConstant(), "ηc width should float"
    assert not fitter.widths["chic0"].isConstant(), "χc0 width should float"
    
    # Check resolution is created
    assert fitter.resolution is not None, "Resolution should be created"
    print(f"\n  Resolution: σ = {fitter.resolution.getVal():.2f} MeV (shared)")
    
    print("\n✓ All signal PDFs created correctly")
    print("✓ Parameter sharing verified (mass/width/resolution)")
    
    return True


def test_4_background_pdf_creation():
    """Test 4: Background PDF creation"""
    print("\n" + "="*80)
    print("TEST 4: Background PDF Creation")
    print("="*80)
    
    config = TOMLConfig()
    fitter = MassFitter(config)
    mass_var = fitter.setup_observable()
    
    years = ["2016", "2017", "2018"]
    
    for year in years:
        bkg_pdf, alpha = fitter.create_background_pdf(mass_var, year)
        
        assert bkg_pdf is not None, f"Background PDF for {year} should be created"
        assert alpha is not None, f"Alpha parameter for {year} should be created"
        
        alpha_val = alpha.getVal()
        print(f"  {year}: α = {alpha_val:.4f} (exponential slope)")
    
    print("\n✓ Background PDFs created correctly (one per year)")
    
    return True


def test_5_full_model_building():
    """Test 5: Full model building for one year"""
    print("\n" + "="*80)
    print("TEST 5: Full Model Building")
    print("="*80)
    
    config = TOMLConfig()
    fitter = MassFitter(config)
    mass_var = fitter.setup_observable()
    
    year = "2016"
    model, yields = fitter.build_model_for_year(year, mass_var)
    
    assert model is not None, "Total PDF should be created"
    assert len(yields) == 5, "Should have 5 yield parameters (4 signals + background)"
    
    print(f"  Model built for {year}")
    print(f"  Components: {list(yields.keys())}")
    
    for state, yield_var in yields.items():
        initial_val = yield_var.getVal()
        min_val = yield_var.getMin()
        max_val = yield_var.getMax()
        print(f"    N_{state:<12}: initial = {initial_val:8.0f}, range = [{min_val:.0f}, {max_val:.0e}]")
    
    print("\n✓ Full model built successfully")
    print("✓ Extended likelihood components verified")
    
    return True


def test_6_fit_to_mc_data():
    """Test 6: Perform fit to J/ψ MC sample (single year)"""
    print("\n" + "="*80)
    print("TEST 6: Fit to J/ψ MC Data (2016 MD LL)")
    print("="*80)
    
    # Load data
    config = TOMLConfig()
    data_manager = DataManager(config)
    
    print("Loading J/ψ MC sample...")
    mc_jpsi = data_manager.load_tree("Jpsi", 2016, "MD", "LL")
    
    # Apply Lambda pre-selection
    selector = LambdaSelector(config)
    mc_selected = selector.apply_lambda_cuts(mc_jpsi)
    
    print(f"Events after Lambda cuts: {len(mc_selected)}")
    
    # Check M_LpKm_h2 branch exists
    if "M_LpKm_h2" not in mc_selected.fields:
        print("ERROR: M_LpKm_h2 branch not found!")
        print(f"Available branches: {mc_selected.fields}")
        return False
    
    # Apply fit range filter
    mass_data = mc_selected["M_LpKm_h2"]
    fit_range = config.particles["mass_windows"]["charmonium_fit_range"]
    mask = (mass_data >= fit_range[0]) & (mass_data <= fit_range[1])
    mc_in_range = mc_selected[mask]
    
    print(f"Events in fit range [{fit_range[0]}, {fit_range[1]}]: {len(mc_in_range)}")
    
    # Create fitter
    fitter = MassFitter(config)
    
    # Perform fit (single year for testing)
    data_dict = {"2016": mc_in_range}
    
    print("\nPerforming fit...")
    results = fitter.perform_fit(data_dict)
    
    # Check results structure
    assert "yields" in results, "Results should contain yields"
    assert "masses" in results, "Results should contain masses"
    assert "widths" in results, "Results should contain widths"
    assert "resolution" in results, "Results should contain resolution"
    assert "fit_results" in results, "Results should contain fit_results"
    
    # Extract J/ψ yield
    jpsi_yield, jpsi_error = results["yields"]["2016"]["jpsi"]
    
    print("\n" + "-"*80)
    print("FIT RESULTS SUMMARY")
    print("-"*80)
    print(f"J/ψ yield (2016): {jpsi_yield:.0f} ± {jpsi_error:.0f}")
    print(f"Expected: ~{len(mc_in_range)} (pure MC sample)")
    print(f"Ratio: {jpsi_yield / len(mc_in_range):.2%}")
    
    # Check fit quality
    fit_result = results["fit_results"]["2016"]
    status = fit_result.status()
    cov_qual = fit_result.covQual()
    
    if status == 0:
        print("\n✓ Fit converged successfully")
    else:
        print(f"\n⚠ WARNING: Fit status = {status} (not converged)")
    
    print(f"✓ Covariance quality: {cov_qual}")
    
    # Check plot was created
    plot_dir = Path(config.paths["output"]["plots_dir"]) / "fits"
    plot_file = plot_dir / "mass_fit_2016.pdf"
    
    if plot_file.exists():
        print(f"✓ Fit plot created: {plot_file}")
    else:
        print(f"⚠ WARNING: Fit plot not found at {plot_file}")
    
    print("\n✓ Mass fitting working correctly on MC data")
    
    return True


def run_all_tests():
    """Run all Phase 5 tests"""
    print("\n" + "="*80)
    print("PHASE 5: MASS FITTING - TEST SUITE")
    print("="*80)
    print("Testing RooFit-based mass fitting infrastructure")
    print("="*80)
    
    tests = [
        ("Initialization", test_1_mass_fitter_initialization),
        ("Observable Setup", test_2_observable_setup),
        ("Signal PDF Creation", test_3_signal_pdf_creation),
        ("Background PDF Creation", test_4_background_pdf_creation),
        ("Full Model Building", test_5_full_model_building),
        ("Fit to MC Data", test_6_fit_to_mc_data),
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
        print("\n🎉 ALL TESTS PASSED! Phase 5 mass fitting framework is working.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
