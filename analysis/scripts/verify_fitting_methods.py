#!/usr/bin/env python3
"""
Quick verification script for binned vs unbinned fitting methods.

This script generates toy data and fits it with both methods to verify
that they actually do what we expect. Run this as a sanity check before
trusting real data fits.

Usage:
    python scripts/verify_fitting_methods.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import awkward as ak
import numpy as np
import ROOT

# Add modules directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))
from mass_fitter import MassFitter

# Suppress RooFit messages
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)


class MockConfig:
    """Mock configuration object for testing MassFitter."""

    def __init__(self, use_binned_fit: bool = True):
        """Initialize mock config with fitting parameters."""
        self.particles = {
            "mass_windows": {
                "charmonium_fit_range": (3000.0, 3200.0),
            },
            "pdg_masses": {
                "jpsi": 3096.9,
                "etac_1s": 2983.9,
                "chic0": 3414.75,
                "chic1": 3510.66,
                "etac_2s": 3637.0,
            },
            "pdg_widths": {
                "jpsi": 0.093,
                "etac_1s": 31.9,
                "chic0": 10.5,
                "chic1": 0.84,
                "etac_2s": 11.3,
            },
            "fitting": {
                "use_binned_fit": use_binned_fit,
                "bin_width": 5.0,
                "argus_endpoint_offset": 200.0,
            },
        }
        self.selection = {
            "bu_fixed_selection": {
                "mass_corrected_min": 5200.0,
                "mass_corrected_max": 5400.0,
            },
        }
        self.paths = {
            "output": {
                "plots_dir": Path(__file__).parent.parent / "analysis_output" / "plots",
            },
        }


def generate_toy_data(
    n_signal: int = 1000,
    n_background: int = 5000,
    signal_mass: float = 3096.9,
    signal_width: float = 0.093,
    resolution: float = 5.0,
    fit_range: tuple[float, float] = (3000.0, 3200.0),
    seed: int = 42,
) -> ak.Array:
    """
    Generate toy Monte Carlo data: Voigtian signal + ARGUS background.

    Returns data in awkward array format compatible with MassFitter.

    Args:
        n_signal: Number of signal events
        n_background: Number of background events
        signal_mass: True signal mass (MeV)
        signal_width: Natural width (MeV)
        resolution: Detector resolution (MeV)
        fit_range: (min, max) fit range in MeV
        seed: Random seed

    Returns:
        Awkward array with M_LpKm_h2 and Bu_MM_corrected branches
    """
    ROOT.RooRandom.randomGenerator().SetSeed(seed)

    # Define observable
    x = ROOT.RooRealVar("x", "x", fit_range[0], fit_range[1])

    # Signal: Voigtian (Breit-Wigner ⊗ Gaussian)
    mean = ROOT.RooRealVar("mean", "mean", signal_mass)
    sigma = ROOT.RooRealVar("sigma", "sigma", resolution)
    gamma = ROOT.RooRealVar("gamma", "gamma", signal_width)
    signal_pdf = ROOT.RooVoigtian("signal", "signal", x, mean, gamma, sigma)

    # Background: ARGUS
    m0 = ROOT.RooRealVar("m0", "m0", fit_range[1] + 200.0)
    c = ROOT.RooRealVar("c", "c", -20.0)
    p = ROOT.RooRealVar("p", "p", 0.5)
    bkg_pdf = ROOT.RooArgusBG("background", "background", x, m0, c, p)

    # Generate data
    signal_data = signal_pdf.generate(ROOT.RooArgSet(x), n_signal)
    bkg_data = bkg_pdf.generate(ROOT.RooArgSet(x), n_background)

    # Combine and shuffle
    all_masses = []
    for i in range(signal_data.numEntries()):
        all_masses.append(signal_data.get(i).getRealValue("x"))
    for i in range(bkg_data.numEntries()):
        all_masses.append(bkg_data.get(i).getRealValue("x"))

    masses = np.array(all_masses)
    np.random.seed(seed)
    np.random.shuffle(masses)

    # Create awkward array with required structure
    # MassFitter expects M_LpKm_h2 and Bu_MM_corrected branches
    data = ak.Array(
        {
            "M_LpKm_h2": masses,
            "Bu_MM_corrected": np.full(len(masses), 5279.0),  # Nominal B+ mass
        }
    )

    return data


def fit_with_mass_fitter(
    data: ak.Array,
    use_binned: bool,
    true_signal: int,
) -> dict:
    """
    Fit data using MassFitter class.

    Args:
        data: Awkward array with M_LpKm_h2 branch
        use_binned: True for binned fit, False for unbinned
        true_signal: True number of signal events (for pull calculation)

    Returns:
        Dictionary with fit results
    """
    # Create mock config
    config = MockConfig(use_binned_fit=use_binned)

    # Create fitter
    fitter = MassFitter(config)

    # Prepare data by year (use "test" as year label)
    data_by_year = {"test": data}

    # Perform fit (disable combined fit since we only have one dataset)
    results = fitter.perform_fit(data_by_year, fit_combined=False)

    # Extract results for "test" year
    test_yields = results["yields"]["test"]
    fit_result = results["fit_results"]["test"]

    # Get J/psi yield (closest to our signal)
    n_sig_fitted, n_sig_error = test_yields["jpsi"]

    # Calculate pull
    pull = (n_sig_fitted - true_signal) / n_sig_error if n_sig_error > 0 else 0

    return {
        "n_signal": n_sig_fitted,
        "n_signal_error": n_sig_error,
        "n_background": test_yields["background"][0],
        "mass": results["masses"]["jpsi"][0],
        "resolution": results["resolution"][0],
        "pull": pull,
        "status": fit_result.status(),
        "nll": fit_result.minNll(),
        "edm": fit_result.edm(),
    }


def main():
    """Run verification tests."""
    print("=" * 80)
    print("BINNED vs UNBINNED FITTING VERIFICATION")
    print("Testing our MassFitter implementation")
    print("=" * 80)

    # Parameters
    true_signal = 1000
    true_background = 5000
    signal_mass = 3096.9  # J/ψ
    fit_range = (3000.0, 3200.0)
    nbins = 40  # 5 MeV bins

    print("\nGenerating toy data:")
    print(f"  True N(signal) = {true_signal}")
    print(f"  True N(background) = {true_background}")
    print(f"  Signal mass = {signal_mass} MeV")
    print(f"  Fit range = {fit_range[0]:.0f} - {fit_range[1]:.0f} MeV")
    print(f"  Binning = {nbins} bins × {(fit_range[1]-fit_range[0])/nbins:.1f} MeV/bin")

    # Generate toy data
    data = generate_toy_data(
        n_signal=true_signal,
        n_background=true_background,
        signal_mass=signal_mass,
        fit_range=fit_range,
        seed=12345,
    )

    print(f"\nGenerated {len(data)} events")

    # === BINNED FIT ===
    print("\n" + "-" * 80)
    print("BINNED MAXIMUM LIKELIHOOD FIT (using MassFitter)")
    print("-" * 80)

    results_binned = fit_with_mass_fitter(
        data=data,
        use_binned=True,
        true_signal=true_signal,
    )

    print("\nFitted parameters:")
    print(
        f"  N(signal) = {results_binned['n_signal']:.0f} ± {results_binned['n_signal_error']:.0f}"
    )
    print(f"  N(background) = {results_binned['n_background']:.0f}")
    print(f"  Mass = {results_binned['mass']:.2f} MeV")
    print(f"  Resolution = {results_binned['resolution']:.2f} MeV")
    print("\nFit quality:")
    print(f"  Status: {results_binned['status']} (0 = success)")
    print(f"  Min NLL: {results_binned['nll']:.1f}")
    print(f"  EDM: {results_binned['edm']:.6f}")
    print(f"  Pull: {results_binned['pull']:.2f}σ")

    # === UNBINNED FIT ===
    print("\n" + "-" * 80)
    print("UNBINNED MAXIMUM LIKELIHOOD FIT (using MassFitter)")
    print("-" * 80)

    results_unbinned = fit_with_mass_fitter(
        data=data,
        use_binned=False,
        true_signal=true_signal,
    )

    print("\nFitted parameters:")
    print(
        f"  N(signal) = {results_unbinned['n_signal']:.0f} ± {results_unbinned['n_signal_error']:.0f}"
    )
    print(f"  N(background) = {results_unbinned['n_background']:.0f}")
    print(f"  Mass = {results_unbinned['mass']:.2f} MeV")
    print(f"  Resolution = {results_unbinned['resolution']:.2f} MeV")
    print("\nFit quality:")
    print(f"  Status: {results_unbinned['status']} (0 = success)")
    print(f"  Min NLL: {results_unbinned['nll']:.1f}")
    print(f"  EDM: {results_unbinned['edm']:.6f}")
    print(f"  Pull: {results_unbinned['pull']:.2f}σ")

    # === COMPARISON TABLE ===
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON TABLE")
    print("=" * 80)

    # Calculate derived quantities
    binned_ok = abs(results_binned["pull"]) < 3
    unbinned_ok = abs(results_unbinned["pull"]) < 3
    error_ratio = results_unbinned["n_signal_error"] / results_binned["n_signal_error"]
    nll_diff = abs(results_binned["nll"] - results_unbinned["nll"])

    # Print comparison table
    print("\n┌─────────────────────────┬──────────────────┬──────────────────┬──────────────────┐")
    print("│ Parameter               │ True Value       │ Binned Fit       │ Unbinned Fit     │")
    print("├─────────────────────────┼──────────────────┼──────────────────┼──────────────────┤")
    print(
        f"│ N(signal)               │ {true_signal:16.0f} │ {results_binned['n_signal']:16.0f} │ {results_unbinned['n_signal']:16.0f} │"
    )
    print(
        f"│ N(signal) error         │ {'—':>16s} │ {results_binned['n_signal_error']:16.1f} │ {results_unbinned['n_signal_error']:16.1f} │"
    )
    print(
        f"│ N(background)           │ {true_background:16.0f} │ {results_binned['n_background']:16.0f} │ {results_unbinned['n_background']:16.0f} │"
    )
    print(
        f"│ Mass [MeV]              │ {signal_mass:16.2f} │ {results_binned['mass']:16.2f} │ {results_unbinned['mass']:16.2f} │"
    )
    print(
        f"│ Resolution [MeV]        │ {'5.00 (input)':>16s} │ {results_binned['resolution']:16.2f} │ {results_unbinned['resolution']:16.2f} │"
    )
    print("├─────────────────────────┼──────────────────┼──────────────────┼──────────────────┤")
    print(
        f"│ Pull [σ]                │ {'—':>16s} │ {results_binned['pull']:16.2f} │ {results_unbinned['pull']:16.2f} │"
    )
    print(
        f"│ Fit status              │ {'—':>16s} │ {results_binned['status']:16d} │ {results_unbinned['status']:16d} │"
    )
    print(
        f"│ Min NLL                 │ {'—':>16s} │ {results_binned['nll']:16.1f} │ {results_unbinned['nll']:16.1f} │"
    )
    print(
        f"│ EDM                     │ {'—':>16s} │ {results_binned['edm']:16.6f} │ {results_unbinned['edm']:16.6f} │"
    )
    print("└─────────────────────────┴──────────────────┴──────────────────┴──────────────────┘")

    # Verification checks
    print("\n" + "=" * 80)
    print("VERIFICATION CHECKS")
    print("=" * 80)

    print("\n1. PARAMETER RECOVERY:")
    print(f"   Both methods should recover true N(signal) = {true_signal}")
    print(
        f"   Binned:   {results_binned['n_signal']:.0f} ± {results_binned['n_signal_error']:.0f} "
        f"(pull = {results_binned['pull']:.2f}σ) {'✓' if binned_ok else '✗'}"
    )
    print(
        f"   Unbinned: {results_unbinned['n_signal']:.0f} ± {results_unbinned['n_signal_error']:.0f} "
        f"(pull = {results_unbinned['pull']:.2f}σ) {'✓' if unbinned_ok else '✗'}"
    )

    print("\n2. ERROR COMPARISON:")
    print("   Unbinned should have equal or smaller errors")
    print(f"   Error(unbinned) / Error(binned) = {error_ratio:.3f}")
    print(f"   Expected: ≤ 1.0  {'✓' if error_ratio <= 1.05 else '✗'}")

    print("\n3. NLL DIFFERENCE:")
    print("   NLL values should be DIFFERENT (not the same method)")
    print(f"   |NLL(binned) - NLL(unbinned)| = {nll_diff:.1f}")
    print(f"   Expected: > 1.0  {'✓' if nll_diff > 1.0 else '✗ WARNING'}")

    if nll_diff < 1.0:
        print("   ⚠ NLL values are suspiciously similar!")
        print("   This might indicate both methods are using the same implementation.")

    print("\n4. FIT CONVERGENCE:")
    print("   Both fits should converge properly")
    print(
        f"   Binned:   Status={results_binned['status']}, EDM={results_binned['edm']:.6f} "
        f"{'✓' if results_binned['status']==0 else '✗'}"
    )
    print(
        f"   Unbinned: Status={results_unbinned['status']}, EDM={results_unbinned['edm']:.6f} "
        f"{'✓' if results_unbinned['status']==0 else '✗'}"
    )

    # Overall assessment
    print("\n" + "=" * 80)
    all_ok = (
        binned_ok
        and unbinned_ok
        and error_ratio <= 1.05
        and nll_diff > 1.0
        and results_binned["status"] == 0
        and results_unbinned["status"] == 0
    )

    if all_ok:
        print("✓ VERIFICATION PASSED")
        print("Both binned and unbinned fits are working correctly and giving different results.")
        print("We can trust our fitting setup (at least partially)!")
    else:
        print("✗ VERIFICATION FAILED")
        print("There may be issues with our fitting implementation.")
        print("Review the comparisons above to identify the problem.")

    print("=" * 80)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
