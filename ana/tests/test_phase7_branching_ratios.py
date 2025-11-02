"""
Phase 7: Branching Fraction Ratios - Test Suite

Tests the BranchingFractionCalculator implementation.

Tests:
1. Initialization with yields and efficiencies
2. Calculate efficiency-corrected yield for single state
3. Calculate ratio to J/ψ
4. Calculate all ratios (ηc, χc0, χc1 to J/ψ)
5. Derived ratio (χc1/χc0)
6. Yield consistency check per year
7. Full pipeline with realistic dummy data
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.branching_fraction_calculator import BranchingFractionCalculator


# Dummy config for testing
class DummyConfig:
    def __init__(self):
        self.paths = {
            "output": {
                "tables_dir": "tables",
                "plots_dir": "plots",
                "results_dir": "results"
            }
        }
        self.luminosity = {
            "integrated_luminosity": {
                "2016": 1.6,  # fb^-1
                "2017": 1.7,
                "2018": 2.2
            }
        }


def test_1_initialization():
    """Test 1: Initialize BranchingFractionCalculator"""
    print("\n" + "="*80)
    print("TEST 1: BranchingFractionCalculator Initialization")
    print("="*80)
    
    # Create dummy yields: {year: {state: (value, error)}}
    yields = {
        "2016": {
            "jpsi": (1000.0, 50.0),
            "etac": (200.0, 20.0),
            "chic0": (300.0, 25.0),
            "chic1": (150.0, 15.0)
        },
        "2017": {
            "jpsi": (1100.0, 55.0),
            "etac": (220.0, 22.0),
            "chic0": (330.0, 27.0),
            "chic1": (165.0, 16.0)
        },
        "2018": {
            "jpsi": (1400.0, 60.0),
            "etac": (280.0, 25.0),
            "chic0": (420.0, 30.0),
            "chic1": (210.0, 18.0)
        }
    }
    
    # Create dummy efficiencies: {state: {year: {"eff": value, "err": error}}}
    efficiencies = {
        "jpsi": {
            "2016": {"eff": 0.85, "err": 0.03},
            "2017": {"eff": 0.86, "err": 0.03},
            "2018": {"eff": 0.84, "err": 0.03}
        },
        "etac": {
            "2016": {"eff": 0.83, "err": 0.03},
            "2017": {"eff": 0.84, "err": 0.03},
            "2018": {"eff": 0.82, "err": 0.03}
        },
        "chic0": {
            "2016": {"eff": 0.87, "err": 0.03},
            "2017": {"eff": 0.88, "err": 0.03},
            "2018": {"eff": 0.86, "err": 0.03}
        },
        "chic1": {
            "2016": {"eff": 0.84, "err": 0.03},
            "2017": {"eff": 0.85, "err": 0.03},
            "2018": {"eff": 0.83, "err": 0.03}
        }
    }
    
    config = DummyConfig()
    
    calculator = BranchingFractionCalculator(yields, efficiencies, config)
    
    print("✓ BranchingFractionCalculator initialized successfully")
    print(f"✓ Yields for {len(yields)} years: {list(yields.keys())}")
    print(f"✓ Efficiencies for {len(efficiencies)} states: {list(efficiencies.keys())}")
    
    return calculator, yields, efficiencies


def test_2_efficiency_corrected_yield(calculator):
    """Test 2: Calculate efficiency-corrected yield for single state"""
    print("\n" + "="*80)
    print("TEST 2: Calculate Efficiency-Corrected Yield")
    print("="*80)
    
    # Calculate for J/ψ
    yield_jpsi, err_jpsi = calculator.calculate_efficiency_corrected_yield("jpsi")
    
    print(f"\nJ/ψ:")
    print(f"  Σ(N/ε) = {yield_jpsi:.1f} ± {err_jpsi:.1f}")
    
    # Manual verification for 2016 only
    n_2016, n_err_2016 = calculator.yields["2016"]["jpsi"]
    eps_2016 = calculator.efficiencies["jpsi"]["2016"]["eff"]
    eps_err_2016 = calculator.efficiencies["jpsi"]["2016"]["err"]
    
    corrected_2016 = n_2016 / eps_2016
    rel_err_n = n_err_2016 / n_2016
    rel_err_eps = eps_err_2016 / eps_2016
    error_2016 = corrected_2016 * np.sqrt(rel_err_n**2 + rel_err_eps**2)
    
    print(f"\n  2016 only (manual check):")
    print(f"    N/ε = {n_2016}/{eps_2016} = {corrected_2016:.1f}")
    print(f"    Error = {error_2016:.1f}")
    print(f"    (Relative errors: N={rel_err_n:.3f}, ε={rel_err_eps:.3f})")
    
    # Check that sum over years is reasonable
    assert yield_jpsi > 0, "Efficiency-corrected yield should be positive"
    assert err_jpsi > 0, "Error should be positive"
    assert yield_jpsi > corrected_2016, "Sum should be larger than single year"
    
    print("\n✓ Efficiency-corrected yield calculation working correctly")
    
    return yield_jpsi, err_jpsi


def test_3_ratio_to_jpsi(calculator):
    """Test 3: Calculate ratio to J/ψ for single state"""
    print("\n" + "="*80)
    print("TEST 3: Calculate Ratio to J/ψ")
    print("="*80)
    
    # Calculate ηc/J/ψ ratio
    ratio_etac, err_etac = calculator.calculate_ratio_to_jpsi("etac")
    
    print(f"\nBr(B⁺ → ηc X) × Br(ηc → Λ̄pK⁻)")
    print(f"───────────────────────────────────")
    print(f"Br(B⁺ → J/ψ X) × Br(J/ψ → Λ̄pK⁻)")
    print(f"\n= {ratio_etac:.4f} ± {err_etac:.4f}")
    
    # Verify ratio is in reasonable range
    assert 0 < ratio_etac < 1, "ηc/J/ψ ratio should be between 0 and 1"
    assert err_etac > 0, "Error should be positive"
    assert err_etac < ratio_etac, "Error should be smaller than value"
    
    print("\n✓ Ratio calculation working correctly")
    
    return ratio_etac, err_etac


def test_4_calculate_all_ratios(calculator):
    """Test 4: Calculate all branching fraction ratios"""
    print("\n" + "="*80)
    print("TEST 4: Calculate All Branching Fraction Ratios")
    print("="*80)
    
    df = calculator.calculate_all_ratios()
    
    print("\nResults DataFrame:")
    print(df.to_string(index=False))
    
    # Verify DataFrame structure
    assert len(df) == 4, "Should have 4 ratios (3 to J/ψ + 1 derived)"
    assert "numerator" in df.columns, "Should have numerator column"
    assert "denominator" in df.columns, "Should have denominator column"
    assert "ratio" in df.columns, "Should have ratio column"
    assert "stat_error" in df.columns, "Should have stat_error column"
    
    # Verify all ratios are positive
    assert all(df["ratio"] > 0), "All ratios should be positive"
    assert all(df["stat_error"] > 0), "All errors should be positive"
    
    # Verify derived ratio (χc1/χc0)
    chic1_jpsi = df[(df["numerator"] == "chic1") & (df["denominator"] == "jpsi")]["ratio"].values[0]
    chic0_jpsi = df[(df["numerator"] == "chic0") & (df["denominator"] == "jpsi")]["ratio"].values[0]
    chic1_chic0 = df[(df["numerator"] == "chic1") & (df["denominator"] == "chic0")]["ratio"].values[0]
    
    expected_chic1_chic0 = chic1_jpsi / chic0_jpsi
    assert abs(chic1_chic0 - expected_chic1_chic0) < 1e-6, "Derived ratio should match calculation"
    
    print("\n✓ All ratios calculated correctly")
    print(f"✓ χc1/χc0 derived ratio verified: {chic1_chic0:.4f}")
    
    return df


def test_5_yield_consistency(calculator):
    """Test 5: Check yield consistency per year"""
    print("\n" + "="*80)
    print("TEST 5: Yield Consistency Check")
    print("="*80)
    
    df = calculator.check_yield_consistency_per_year()
    
    print("\nYield consistency per year (first 12 rows):")
    print(df.head(12).to_string(index=False))
    
    # Verify DataFrame structure
    assert "state" in df.columns, "Should have state column"
    assert "year" in df.columns, "Should have year column"
    assert "N_over_L_eps" in df.columns, "Should have N/(L×ε) column"
    assert "error" in df.columns, "Should have error column"
    
    # Check that we have data for all states and years
    states = df["state"].unique()
    years = df["year"].unique()
    assert len(states) == 4, "Should have 4 states"
    assert len(years) == 3, "Should have 3 years"
    
    # Check consistency: values should not vary wildly across years
    for state in states:
        state_data = df[df["state"] == state]
        values = state_data["N_over_L_eps"].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        rel_std = std_val / mean_val if mean_val > 0 else 0
        
        print(f"\n{state}: mean={mean_val:.1f}, std={std_val:.1f}, rel_std={rel_std:.1%}")
        
        # Relative scatter should be < 20% for good consistency
        if rel_std > 0.20:
            print(f"  ⚠️  Large variation across years (>{20}%)")
        else:
            print(f"  ✓ Good consistency across years (<{20}%)")
    
    print("\n✓ Yield consistency check completed")
    
    return df


def test_6_realistic_scenario():
    """Test 6: Full pipeline with realistic dummy data"""
    print("\n" + "="*80)
    print("TEST 6: Full Pipeline with Realistic Data")
    print("="*80)
    
    # Realistic yields (scaled to match LHCb-like statistics)
    # Assume J/ψ is most abundant, ηc is ~15%, χc0 ~20%, χc1 ~10%
    yields = {
        "2016": {
            "jpsi": (5000.0, 100.0),
            "etac": (750.0, 40.0),
            "chic0": (1000.0, 50.0),
            "chic1": (500.0, 35.0)
        },
        "2017": {
            "jpsi": (5500.0, 110.0),
            "etac": (825.0, 43.0),
            "chic0": (1100.0, 53.0),
            "chic1": (550.0, 37.0)
        },
        "2018": {
            "jpsi": (7000.0, 120.0),
            "etac": (1050.0, 48.0),
            "chic0": (1400.0, 60.0),
            "chic1": (700.0, 42.0)
        }
    }
    
    # Realistic efficiencies (all ~85-90%, similar across states)
    efficiencies = {
        "jpsi": {
            "2016": {"eff": 0.889, "err": 0.004},
            "2017": {"eff": 0.890, "err": 0.004},
            "2018": {"eff": 0.888, "err": 0.004}
        },
        "etac": {
            "2016": {"eff": 0.885, "err": 0.004},
            "2017": {"eff": 0.886, "err": 0.004},
            "2018": {"eff": 0.884, "err": 0.004}
        },
        "chic0": {
            "2016": {"eff": 0.888, "err": 0.003},
            "2017": {"eff": 0.889, "err": 0.003},
            "2018": {"eff": 0.887, "err": 0.003}
        },
        "chic1": {
            "2016": {"eff": 0.881, "err": 0.004},
            "2017": {"eff": 0.882, "err": 0.004},
            "2018": {"eff": 0.880, "err": 0.004}
        }
    }
    
    config = DummyConfig()
    calculator = BranchingFractionCalculator(yields, efficiencies, config)
    
    print("\nCalculating all branching fraction ratios...")
    df = calculator.calculate_all_ratios()
    
    print("\n" + "="*80)
    print("FINAL RESULTS (Realistic Scenario)")
    print("="*80)
    print(df.to_string(index=False))
    
    # Extract key results
    etac_jpsi = df[(df["numerator"] == "etac") & (df["denominator"] == "jpsi")]["ratio"].values[0]
    chic0_jpsi = df[(df["numerator"] == "chic0") & (df["denominator"] == "jpsi")]["ratio"].values[0]
    chic1_jpsi = df[(df["numerator"] == "chic1") & (df["denominator"] == "jpsi")]["ratio"].values[0]
    chic1_chic0 = df[(df["numerator"] == "chic1") & (df["denominator"] == "chic0")]["ratio"].values[0]
    
    print("\n" + "="*80)
    print("Key Physics Results:")
    print("="*80)
    print(f"ηc/J/ψ ratio:    {etac_jpsi:.3f} ± {df[(df['numerator']=='etac')]['stat_error'].values[0]:.3f}")
    print(f"χc0/J/ψ ratio:   {chic0_jpsi:.3f} ± {df[(df['numerator']=='chic0')]['stat_error'].values[0]:.3f}")
    print(f"χc1/J/ψ ratio:   {chic1_jpsi:.3f} ± {df[(df['numerator']=='chic1')]['stat_error'].values[0]:.3f}")
    print(f"χc1/χc0 ratio:   {chic1_chic0:.3f} ± {df[(df['numerator']=='chic1')&(df['denominator']=='chic0')]['stat_error'].values[0]:.3f}")
    
    print("\n" + "="*80)
    print("Physics Interpretation:")
    print("="*80)
    print(f"• ηc/J/ψ ≈ {etac_jpsi:.2f}: ηc production is ~{etac_jpsi*100:.0f}% of J/ψ")
    print(f"• χc0/J/ψ ≈ {chic0_jpsi:.2f}: χc0 production is ~{chic0_jpsi*100:.0f}% of J/ψ")
    print(f"• χc1/J/ψ ≈ {chic1_jpsi:.2f}: χc1 production is ~{chic1_jpsi*100:.0f}% of J/ψ")
    print(f"• χc1/χc0 ≈ {chic1_chic0:.2f}: χc1/χc0 ratio")
    print("\nNote: NRQCD predicts χc1/χc0 ≈ 3 (not observed in data)")
    
    print("\n✓ Full pipeline executed successfully")
    
    return calculator, df


def test_7_error_propagation():
    """Test 7: Verify error propagation is correct"""
    print("\n" + "="*80)
    print("TEST 7: Error Propagation Verification")
    print("="*80)
    
    # Simple case with single year for manual verification
    yields = {
        "2016": {
            "jpsi": (1000.0, 50.0),  # 5% error
            "etac": (200.0, 20.0)     # 10% error
        }
    }
    
    efficiencies = {
        "jpsi": {
            "2016": {"eff": 0.80, "err": 0.04}  # 5% error
        },
        "etac": {
            "2016": {"eff": 0.80, "err": 0.04}  # 5% error
        }
    }
    
    config = DummyConfig()
    calculator = BranchingFractionCalculator(yields, efficiencies, config)
    
    # Manual calculation for J/ψ
    n_jpsi = 1000.0
    n_err_jpsi = 50.0
    eps_jpsi = 0.80
    eps_err_jpsi = 0.04
    
    corrected_jpsi = n_jpsi / eps_jpsi  # = 1250.0
    rel_err_n_jpsi = n_err_jpsi / n_jpsi  # = 0.05
    rel_err_eps_jpsi = eps_err_jpsi / eps_jpsi  # = 0.05
    error_jpsi_manual = corrected_jpsi * np.sqrt(rel_err_n_jpsi**2 + rel_err_eps_jpsi**2)
    
    # Manual calculation for ηc
    n_etac = 200.0
    n_err_etac = 20.0
    eps_etac = 0.80
    eps_err_etac = 0.04
    
    corrected_etac = n_etac / eps_etac  # = 250.0
    rel_err_n_etac = n_err_etac / n_etac  # = 0.10
    rel_err_eps_etac = eps_err_etac / eps_etac  # = 0.05
    error_etac_manual = corrected_etac * np.sqrt(rel_err_n_etac**2 + rel_err_eps_etac**2)
    
    # Ratio manual calculation
    ratio_manual = corrected_etac / corrected_jpsi  # = 250/1250 = 0.2
    rel_err_etac_corrected = error_etac_manual / corrected_etac
    rel_err_jpsi_corrected = error_jpsi_manual / corrected_jpsi
    error_ratio_manual = ratio_manual * np.sqrt(rel_err_etac_corrected**2 + rel_err_jpsi_corrected**2)
    
    print("\nManual Calculation:")
    print(f"  J/ψ: N/ε = {corrected_jpsi:.1f} ± {error_jpsi_manual:.1f}")
    print(f"  ηc:  N/ε = {corrected_etac:.1f} ± {error_etac_manual:.1f}")
    print(f"  Ratio: {ratio_manual:.4f} ± {error_ratio_manual:.4f}")
    
    # Calculator result
    yield_jpsi_calc, err_jpsi_calc = calculator.calculate_efficiency_corrected_yield("jpsi")
    yield_etac_calc, err_etac_calc = calculator.calculate_efficiency_corrected_yield("etac")
    ratio_calc, err_ratio_calc = calculator.calculate_ratio_to_jpsi("etac")
    
    print("\nCalculator Result:")
    print(f"  J/ψ: N/ε = {yield_jpsi_calc:.1f} ± {err_jpsi_calc:.1f}")
    print(f"  ηc:  N/ε = {yield_etac_calc:.1f} ± {err_etac_calc:.1f}")
    print(f"  Ratio: {ratio_calc:.4f} ± {err_ratio_calc:.4f}")
    
    # Verify agreement
    assert abs(yield_jpsi_calc - corrected_jpsi) < 0.1, "J/ψ yield mismatch"
    assert abs(err_jpsi_calc - error_jpsi_manual) < 0.1, "J/ψ error mismatch"
    assert abs(yield_etac_calc - corrected_etac) < 0.1, "ηc yield mismatch"
    assert abs(err_etac_calc - error_etac_manual) < 0.1, "ηc error mismatch"
    assert abs(ratio_calc - ratio_manual) < 1e-6, "Ratio mismatch"
    assert abs(err_ratio_calc - error_ratio_manual) < 1e-4, "Ratio error mismatch"
    
    print("\n✓ Error propagation verified - all values match manual calculation")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PHASE 7: BRANCHING FRACTION RATIOS - TEST SUITE")
    print("="*80)
    print("\nTesting BranchingFractionCalculator implementation")
    
    try:
        # Test 1: Initialization
        calculator, yields, efficiencies = test_1_initialization()
        
        # Test 2: Efficiency-corrected yield
        yield_jpsi, err_jpsi = test_2_efficiency_corrected_yield(calculator)
        
        # Test 3: Ratio to J/ψ
        ratio_etac, err_etac = test_3_ratio_to_jpsi(calculator)
        
        # Test 4: All ratios
        df = test_4_calculate_all_ratios(calculator)
        
        # Test 5: Yield consistency
        consistency_df = test_5_yield_consistency(calculator)
        
        # Test 6: Realistic scenario
        calc_realistic, df_realistic = test_6_realistic_scenario()
        
        # Test 7: Error propagation
        test_7_error_propagation()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\nPhase 7 branching fraction ratio calculation is working correctly.")
        print("\nKey features validated:")
        print("  ✓ Efficiency-corrected yields with proper error propagation")
        print("  ✓ Ratio calculations (ηc, χc0, χc1 to J/ψ)")
        print("  ✓ Derived ratio (χc1/χc0)")
        print("  ✓ Yield consistency checks per year")
        print("  ✓ Full pipeline with realistic data")
        print("  ✓ Error propagation matches manual calculation")
        print("\nReady to proceed with real data from Phase 5 and Phase 6!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
