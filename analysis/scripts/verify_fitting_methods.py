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

import numpy as np
import ROOT

# Suppress RooFit messages
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)


def generate_toy_data(
    n_signal: int = 1000,
    n_background: int = 5000,
    signal_mass: float = 3096.9,
    signal_width: float = 0.093,
    resolution: float = 5.0,
    fit_range: tuple[float, float] = (3000.0, 3200.0),
    seed: int = 42,
) -> np.ndarray:
    """
    Generate toy Monte Carlo data: Voigtian signal + ARGUS background.

    Args:
        n_signal: Number of signal events
        n_background: Number of background events
        signal_mass: True signal mass (MeV)
        signal_width: Natural width (MeV)
        resolution: Detector resolution (MeV)
        fit_range: (min, max) fit range in MeV
        seed: Random seed

    Returns:
        Array of generated masses
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

    return masses


def fit_data(
    masses: np.ndarray,
    fit_range: tuple[float, float],
    nbins: int,
    use_binned: bool,
    true_signal: int,
    true_bkg: int,
) -> dict:
    """
    Fit data with either binned or unbinned maximum likelihood.

    Args:
        masses: Array of mass values
        fit_range: (min, max) fit range
        nbins: Number of bins for binned fit
        use_binned: True for binned fit, False for unbinned
        true_signal: True number of signal events (for pull calculation)
        true_bkg: True number of background events

    Returns:
        Dictionary with fit results
    """
    # Define observable
    x = ROOT.RooRealVar("x", "m(mass) [MeV]", fit_range[0], fit_range[1])
    x.setBins(nbins)

    # Create dataset
    if use_binned:
        # Create unbinned first, then convert to binned
        temp_data = ROOT.RooDataSet("temp_data", "temp_data", ROOT.RooArgSet(x))
        for m in masses:
            x.setVal(m)
            temp_data.add(ROOT.RooArgSet(x))
        dataset = ROOT.RooDataHist("data", "data", ROOT.RooArgSet(x), temp_data)
    else:
        # Unbinned dataset
        dataset = ROOT.RooDataSet("data", "data", ROOT.RooArgSet(x))
        for m in masses:
            x.setVal(m)
            dataset.add(ROOT.RooArgSet(x))

    # Build model
    # Signal: Voigtian
    mean = ROOT.RooRealVar("mean", "mean", 3096.9, 3090.0, 3105.0)
    sigma = ROOT.RooRealVar("sigma", "sigma", 5.0, 1.0, 20.0)
    gamma = ROOT.RooRealVar("gamma", "gamma", 0.093)
    gamma.setConstant(True)
    signal_pdf = ROOT.RooVoigtian("signal", "signal", x, mean, gamma, sigma)

    # Background: ARGUS
    m0 = ROOT.RooRealVar("m0", "m0", fit_range[1] + 200.0)
    m0.setConstant(True)
    c = ROOT.RooRealVar("c", "c", -20.0, -100.0, -0.1)
    p = ROOT.RooRealVar("p", "p", 0.5)
    p.setConstant(True)
    bkg_pdf = ROOT.RooArgusBG("background", "background", x, m0, c, p)

    # Yields
    n_signal = ROOT.RooRealVar("n_signal", "n_signal", len(masses) * 0.2, 0, len(masses))
    n_bkg = ROOT.RooRealVar("n_bkg", "n_bkg", len(masses) * 0.8, 0, len(masses) * 2)

    # Combined model
    model = ROOT.RooAddPdf(
        "model",
        "model",
        ROOT.RooArgList(signal_pdf, bkg_pdf),
        ROOT.RooArgList(n_signal, n_bkg),
    )

    # Fit
    fit_result = model.fitTo(
        dataset,
        ROOT.RooFit.Save(),
        ROOT.RooFit.Extended(True),
        ROOT.RooFit.PrintLevel(-1),
        ROOT.RooFit.Strategy(2),
    )

    # Extract results
    n_sig_fitted = n_signal.getVal()
    n_sig_error = n_signal.getError()
    n_bkg_fitted = n_bkg.getVal()
    mean_fitted = mean.getVal()
    sigma_fitted = sigma.getVal()

    # Calculate pull
    pull = (n_sig_fitted - true_signal) / n_sig_error if n_sig_error > 0 else 0

    # Fit quality
    status = fit_result.status()
    nll = fit_result.minNll()
    edm = fit_result.edm()

    return {
        "n_signal": n_sig_fitted,
        "n_signal_error": n_sig_error,
        "n_background": n_bkg_fitted,
        "mass": mean_fitted,
        "resolution": sigma_fitted,
        "pull": pull,
        "status": status,
        "nll": nll,
        "edm": edm,
        "fit_result": fit_result,
        "model": model,
        "dataset": dataset,
        "x": x,
    }


def main():
    """Run verification tests."""
    print("=" * 80)
    print("BINNED vs UNBINNED FITTING VERIFICATION")
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
    masses = generate_toy_data(
        n_signal=true_signal,
        n_background=true_background,
        signal_mass=signal_mass,
        fit_range=fit_range,
        seed=12345,
    )

    print(f"\nGenerated {len(masses)} events")

    # === BINNED FIT ===
    print("\n" + "-" * 80)
    print("BINNED MAXIMUM LIKELIHOOD FIT")
    print("-" * 80)

    results_binned = fit_data(
        masses=masses,
        fit_range=fit_range,
        nbins=nbins,
        use_binned=True,
        true_signal=true_signal,
        true_bkg=true_background,
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
    print("UNBINNED MAXIMUM LIKELIHOOD FIT")
    print("-" * 80)

    results_unbinned = fit_data(
        masses=masses,
        fit_range=fit_range,
        nbins=nbins,
        use_binned=False,
        true_signal=true_signal,
        true_bkg=true_background,
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

    # === COMPARISON ===
    print("\n" + "=" * 80)
    print("COMPARISON & VERIFICATION")
    print("=" * 80)

    print("\n1. PARAMETER RECOVERY:")
    print(f"   Both methods should recover true N(signal) = {true_signal}")

    binned_ok = abs(results_binned["pull"]) < 3
    unbinned_ok = abs(results_unbinned["pull"]) < 3

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
    error_ratio = results_unbinned["n_signal_error"] / results_binned["n_signal_error"]
    print(f"   Error(unbinned) / Error(binned) = {error_ratio:.3f}")
    print(f"   Expected: ≤ 1.0  {'✓' if error_ratio <= 1.05 else '✗'}")

    print("\n3. NLL DIFFERENCE:")
    print("   NLL values should be DIFFERENT (not the same method)")
    nll_diff = abs(results_binned["nll"] - results_unbinned["nll"])
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
