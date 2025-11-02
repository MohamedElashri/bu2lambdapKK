#!/usr/bin/env python3
"""
Phase 1 Validation: Configuration Setup

Tests that all TOML config files are correctly structured,
contain required fields, and are accessible.
"""

import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent.parent / "modules"))

def test_config_files_exist():
    """Test that all required config files exist"""
    print("="*80)
    print("TEST 1: Configuration Files Exist")
    print("="*80)
    
    config_dir = Path("./config")
    required_files = [
        "paths.toml",
        "particles.toml",
        "selection.toml",
        "triggers.toml",
        "luminosity.toml",
        "efficiency_inputs.toml",
        "branching_fractions.toml"
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = config_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename} exists")
        else:
            print(f"  ✗ {filename} MISSING")
            all_exist = False
    
    return all_exist

def test_paths_config():
    """Test paths.toml structure"""
    print("\n" + "="*80)
    print("TEST 2: paths.toml Structure")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig
        import tomllib
        
        config = TOMLConfig("./config")
        
        # Check data paths exist in config
        print(f"\nData path: {config.paths['data']['base_path']}")
        data_path = Path(config.paths['data']['base_path'])
        assert data_path.exists(), f"Data path does not exist: {data_path}"
        print(f"  ✓ Data path exists")
        
        # Check MC path
        print(f"\nMC path: {config.paths['mc']['base_path']}")
        mc_path = Path(config.paths['mc']['base_path'])
        assert mc_path.exists(), f"MC path does not exist: {mc_path}"
        print(f"  ✓ MC path exists")
        
        # Check years and magnets
        with open("./config/paths.toml", "rb") as f:
            paths_config = tomllib.load(f)
        years = paths_config["data"]["years"]
        magnets = paths_config["data"]["magnets"]
        
        print(f"\nYears: {years}")
        assert years == [2016, 2017, 2018], "Years should be [2016, 2017, 2018]"
        print(f"  ✓ Years correct")
        
        print(f"\nMagnets: {magnets}")
        assert magnets == ["MD", "MU"], "Magnets should be ['MD', 'MU']"
        print(f"  ✓ Magnets correct")
        
        # Check MC states
        states = paths_config["mc"]["states"]
        print(f"\nMC states: {states}")
        assert "Jpsi" in states, "Jpsi must be in MC states"
        assert "etac" in states, "etac must be in MC states"
        assert "chic0" in states, "chic0 must be in MC states"
        assert "chic1" in states, "chic1 must be in MC states"
        assert "KpKm" in states, "KpKm (phase space) must be in MC states"
        print(f"  ✓ All required MC states present")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_particles_config():
    """Test particles.toml structure"""
    print("\n" + "="*80)
    print("TEST 3: particles.toml Structure")
    print("="*80)
    
    try:
        import tomllib
        with open("./config/particles.toml", "rb") as f:
            particles = tomllib.load(f)
        
        # Check PDG masses
        print("\nPDG Masses:")
        required_masses = ["lambda", "proton", "kaon", "jpsi", "etac_1s", "chic0", "chic1"]
        for particle in required_masses:
            mass = particles["pdg_masses"][particle]
            print(f"  {particle:12s}: {mass:8.3f} MeV/c²")
        
        # Verify ηc(1S) mass is correct
        etac_mass = particles["pdg_masses"]["etac_1s"]
        assert 2983.0 < etac_mass < 2984.0, f"ηc(1S) mass should be ~2983.9, got {etac_mass}"
        print(f"  ✓ ηc(1S) mass correct (first charmonium state)")
        
        # Check PDG widths
        print("\nPDG Widths:")
        required_widths = ["jpsi", "etac_1s", "chic0", "chic1"]
        for particle in required_widths:
            width = particles["pdg_widths"][particle]
            print(f"  {particle:12s}: {width:8.3f} MeV/c²")
        
        # Check mass windows
        print("\nMass Windows:")
        lambda_window = particles["mass_windows"]["lambda"]
        print(f"  Lambda: {lambda_window[0]:.1f} - {lambda_window[1]:.1f} MeV")
        assert lambda_window == [1111.0, 1121.0], "Lambda window should be [1111, 1121]"
        
        bu_window = particles["mass_windows"]["bu_corrected"]
        print(f"  B+ (corrected): {bu_window[0]:.1f} - {bu_window[1]:.1f} MeV")
        assert bu_window == [5255.0, 5305.0], "B+ window should be [5255, 5305]"
        
        fit_range = particles["mass_windows"]["charmonium_fit_range"]
        print(f"  Charmonium fit: {fit_range[0]:.1f} - {fit_range[1]:.1f} MeV")
        print(f"  ✓ Mass windows correct")
        
        # Check signal regions
        print("\nSignal Regions (for FOM):")
        for state in ["jpsi", "etac", "chic0", "chic1"]:
            region = particles["signal_regions"][state]
            center = region["center"]
            window = region["window"]
            print(f"  {state:8s}: {center:.2f} ± {window:.1f} MeV")
        print(f"  ✓ All signal regions defined")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_selection_config():
    """Test selection.toml structure"""
    print("\n" + "="*80)
    print("TEST 4: selection.toml Structure")
    print("="*80)
    
    try:
        import tomllib
        with open("./config/selection.toml", "rb") as f:
            selection = tomllib.load(f)
        
        # Check Lambda selection (fixed cuts)
        print("\nLambda Selection (FIXED):")
        lambda_sel = selection["lambda_selection"]
        print(f"  Mass: [{lambda_sel['mass_min']:.1f}, {lambda_sel['mass_max']:.1f}] MeV")
        print(f"  FD χ²: > {lambda_sel['fd_chisq_min']:.0f}")
        print(f"  ΔZ: > {lambda_sel['delta_z_min']:.1f} mm")
        print(f"  Proton PID: > {lambda_sel['proton_probnnp_min']:.2f}")
        print(f"  ✓ Lambda cuts defined")
        
        # Check B+ optimizable selection
        print("\nB+ Selection (OPTIMIZABLE):")
        bu_opt = selection["bu_optimizable_selection"]
        for var in ["pt", "dtf_chi2", "ipchi2", "fdchi2"]:
            var_config = bu_opt[var]
            print(f"  {var_config['description']:40s}: [{var_config['begin']:7.1f}, {var_config['end']:7.1f}] step {var_config['step']:5.1f} ({var_config['cut_type']})")
        print(f"  ✓ B+ variables defined")
        
        # Check bachelor p selection
        print("\nBachelor p̄ Selection (OPTIMIZABLE):")
        p_opt = selection["bachelor_p_optimizable_selection"]
        for var in ["probnnp", "track_chi2ndof", "ipchi2"]:
            var_config = p_opt[var]
            print(f"  {var_config['description']:40s}: [{var_config['begin']:7.2f}, {var_config['end']:7.2f}] step {var_config['step']:5.2f} ({var_config['cut_type']})")
        print(f"  ✓ Bachelor p̄ variables defined")
        
        # Check K+ selection
        print("\nK+ Selection (OPTIMIZABLE):")
        kp_opt = selection["kplus_optimizable_selection"]
        for var in ["probnnk", "track_chi2ndof", "ipchi2"]:
            var_config = kp_opt[var]
            print(f"  {var_config['description']:40s}: [{var_config['begin']:7.2f}, {var_config['end']:7.2f}] step {var_config['step']:5.2f} ({var_config['cut_type']})")
        print(f"  ✓ K+ variables defined")
        
        # Check K- selection
        print("\nK- Selection (OPTIMIZABLE):")
        km_opt = selection["kminus_optimizable_selection"]
        for var in ["probnnk", "track_chi2ndof", "ipchi2"]:
            var_config = km_opt[var]
            print(f"  {var_config['description']:40s}: [{var_config['begin']:7.2f}, {var_config['end']:7.2f}] step {var_config['step']:5.2f} ({var_config['cut_type']})")
        print(f"  ✓ K- variables defined")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_triggers_config():
    """Test triggers.toml structure"""
    print("\n" + "="*80)
    print("TEST 5: triggers.toml Structure")
    print("="*80)
    
    try:
        import tomllib
        with open("./config/triggers.toml", "rb") as f:
            triggers = tomllib.load(f)
        
        # Check L0 triggers
        print("\nL0 Triggers (TIS - Trigger Independent of Signal):")
        l0_lines = triggers["L0_TIS"]["lines"]
        for line in l0_lines:
            print(f"  - {line}")
        assert len(l0_lines) > 0, "Must have L0 trigger lines"
        print(f"  ✓ {len(l0_lines)} L0 lines defined")
        
        # Check HLT1 triggers
        print("\nHLT1 Triggers (TOS - Trigger On Signal):")
        hlt1_lines = triggers["HLT1_TOS"]["lines"]
        for line in hlt1_lines:
            print(f"  - {line}")
        assert len(hlt1_lines) > 0, "Must have HLT1 trigger lines"
        print(f"  ✓ {len(hlt1_lines)} HLT1 lines defined")
        
        # Check HLT2 triggers
        print("\nHLT2 Triggers (TOS):")
        hlt2_lines = triggers["HLT2_TOS"]["lines"]
        for line in hlt2_lines:
            print(f"  - {line}")
        assert len(hlt2_lines) > 0, "Must have HLT2 trigger lines"
        print(f"  ✓ {len(hlt2_lines)} HLT2 lines defined")
        
        # Check combination logic
        combo = triggers["combination"]
        print(f"\nTrigger Combination Logic:")
        print(f"  {combo['logic']}")
        print(f"  {combo['comment']}")
        assert combo['logic'] == "L0_TIS AND HLT1_TOS AND HLT2_TOS"
        print(f"  ✓ Trigger logic correct")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_luminosity_config():
    """Test luminosity.toml structure"""
    print("\n" + "="*80)
    print("TEST 6: luminosity.toml Structure")
    print("="*80)
    
    try:
        import tomllib
        with open("./config/luminosity.toml", "rb") as f:
            lumi = tomllib.load(f)
        
        print("\nIntegrated Luminosity:")
        years = ["2016", "2017", "2018"]
        total = 0.0
        for year in years:
            lumi_val = lumi["integrated_luminosity"][year]
            print(f"  {year}: {lumi_val:.2f} fb⁻¹")
            total += lumi_val
        
        print(f"\nTotal: {total:.2f} fb⁻¹")
        print(f"  ✓ Luminosity values defined")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_integration():
    """Test that TOMLConfig loads all configs correctly"""
    print("\n" + "="*80)
    print("TEST 7: Configuration Integration")
    print("="*80)
    
    try:
        from data_handler import TOMLConfig
        
        print("\nLoading TOMLConfig...")
        config = TOMLConfig("./config")
        print(f"  ✓ TOMLConfig loaded successfully")
        
        # Test accessing values
        print("\nTesting config accessors:")
        
        # PDG mass
        lambda_mass = config.get_pdg_mass("lambda")
        print(f"  Lambda PDG mass: {lambda_mass:.3f} MeV")
        assert lambda_mass == 1115.683
        
        # Mass window
        lambda_window = config.particles["mass_windows"]["lambda"]
        print(f"  Lambda mass window: {lambda_window}")
        
        # Signal region
        jpsi_region = config.particles["signal_regions"]["jpsi"]
        print(f"  J/ψ signal region: {jpsi_region['center']} ± {jpsi_region['window']} MeV")
        
        # Selection cuts
        lambda_cuts = config.selection["lambda_selection"]
        print(f"  Lambda FD χ² cut: > {lambda_cuts['fd_chisq_min']}")
        
        print(f"\n  ✓ All config accessors working")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Phase 1 validation tests"""
    print("\n" + "="*80)
    print("PHASE 1 VALIDATION: Configuration Setup")
    print("="*80 + "\n")
    
    tests = [
        ("Config Files Exist", test_config_files_exist),
        ("paths.toml", test_paths_config),
        ("particles.toml", test_particles_config),
        ("selection.toml", test_selection_config),
        ("triggers.toml", test_triggers_config),
        ("luminosity.toml", test_luminosity_config),
        ("Config Integration", test_config_integration),
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
    print("TEST SUMMARY - PHASE 1")
    print("="*80)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")
    
    total_passed = sum(1 for _, r in results if r)
    total_tests = len(results)
    print(f"\nPassed: {total_passed}/{total_tests}")
    
    if total_passed == total_tests:
        print("\n🎉 Phase 1 COMPLETE: All configuration files validated!")
        print("\nReady to proceed to Phase 2: Data Loading Execution")
    else:
        print("\n⚠️  Some tests failed. Review errors above.")
    
    return total_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
