#!/usr/bin/env python3
"""
Phase 1: Validate M(Lambda p K-) Shape Independence

This script validates that the M(Lambda p K-) distribution shape is
independent of the M(B+) value in sideband regions. This is a critical
assumption for using sidebands to model combinatorial background.

Method:
-------
1. Divide M(B+) sideband into multiple slices
2. Create M(Lambda p K-) projection for each slice
3. Normalize all projections to unit area
4. Compare shapes visually and statistically (KS test, chi2)

If shapes are consistent across M(B+) slices, we can confidently use
sideband data to model the background shape in the signal region.

Usage:
------
    cd analysis/studies/sideband_background
    python validate_shapes.py
    python validate_shapes.py --years 2016,2017,2018
    python validate_shapes.py --no-cuts

Output:
-------
    output/shape_validation_comparison.pdf
    output/shape_validation_ks_tests.txt
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import ROOT

# Add parent directories to path
STUDY_DIR: Path = Path(__file__).parent
ANALYSIS_DIR: Path = STUDY_DIR.parent.parent
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(STUDY_DIR))

from config import CHARMONIUM_CONFIG, MASS_CONFIG
from data_loader import (
    build_cut_string,
    get_config,
    load_mlpk_histogram_in_mbu_region,
    setup_root_style,
)


def create_shape_comparison_plot(
    histograms: list[tuple[ROOT.TH1D, str, int]],
    output_path: Path,
    title: str = "M(#Lambda#bar{p}K^{#minus}) Shape Comparison",
) -> None:
    """
    Create overlay plot comparing normalized M(LpK-) shapes from different M(B+) regions.

    Args:
        histograms: List of (histogram, label, color) tuples
        output_path: Path to save the PDF
        title: Plot title
    """
    keep_alive: list[Any] = []
    canvas: ROOT.TCanvas = ROOT.TCanvas("c_shape", "", 1200, 900)
    ROOT.SetOwnership(canvas, False)
    keep_alive.append(canvas)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.05)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.12)
    # Normalize all histograms to unit area
    normalized_hists: list[ROOT.TH1D] = []
    for hist, label, color in histograms:
        h_norm: ROOT.TH1D = hist.Clone(f"{hist.GetName()}_norm")
        ROOT.SetOwnership(h_norm, False)
        integral: float = h_norm.Integral()
        if integral > 0:
            h_norm.Scale(1.0 / integral)
        h_norm.SetLineColor(color)
        h_norm.SetLineWidth(2)
        h_norm.SetMarkerColor(color)
        h_norm.SetMarkerStyle(20)
        h_norm.SetMarkerSize(0.5)
        normalized_hists.append(h_norm)
        keep_alive.append(h_norm)
    # Find y-axis range
    y_max: float = max(h.GetMaximum() for h in normalized_hists) * 1.3
    # Draw first histogram
    normalized_hists[0].SetTitle(title)
    normalized_hists[0].GetXaxis().SetTitle("#it{M}(#Lambda#it{p}#it{K}^{#minus}) [MeV/#it{c}^{2}]")
    normalized_hists[0].GetYaxis().SetTitle("Normalized entries")
    normalized_hists[0].GetYaxis().SetRangeUser(0, y_max)
    normalized_hists[0].Draw("HIST E")
    # Draw remaining histograms
    for h in normalized_hists[1:]:
        h.Draw("HIST E SAME")
    # Legend
    legend: ROOT.TLegend = ROOT.TLegend(0.55, 0.65, 0.92, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(132)
    legend.SetTextSize(0.035)
    for h_norm, (_, label, _) in zip(normalized_hists, histograms):
        legend.AddEntry(h_norm, label, "l")
    legend.Draw()
    keep_alive.append(legend)
    # LHCb label
    latex: ROOT.TLatex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(132)
    latex.SetTextSize(0.045)
    latex.DrawLatex(0.15, 0.85, "#font[62]{LHCb}")
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.15, 0.80, "Shape validation")
    keep_alive.append(latex)
    # Draw charmonium reference lines
    for mass, label, color in CHARMONIUM_CONFIG.get_all_states():
        if MASS_CONFIG.MLPK_MIN < mass < MASS_CONFIG.MLPK_MAX:
            line: ROOT.TLine = ROOT.TLine(mass, 0, mass, y_max * 0.3)
            line.SetLineColor(ROOT.kGray + 1)
            line.SetLineStyle(2)
            line.SetLineWidth(1)
            line.Draw("same")
            keep_alive.append(line)
    canvas.Update()
    canvas.SaveAs(str(output_path))
    print(f"Saved: {output_path}")


def perform_ks_tests(
    histograms: list[tuple[ROOT.TH1D, str, int]],
) -> list[tuple[str, str, float, float]]:
    """
    Perform Kolmogorov-Smirnov tests between all pairs of histograms.

    Args:
        histograms: List of (histogram, label, color) tuples

    Returns:
        List of (label1, label2, ks_statistic, p_value) tuples
    """
    results: list[tuple[str, str, float, float]] = []
    n_hists: int = len(histograms)
    for i in range(n_hists):
        for j in range(i + 1, n_hists):
            h1, label1, _ = histograms[i]
            h2, label2, _ = histograms[j]
            # ROOT KS test returns p-value
            ks_prob: float = h1.KolmogorovTest(h2)
            # Get KS statistic (maximum distance)
            ks_stat: float = h1.KolmogorovTest(h2, "M")
            results.append((label1, label2, ks_stat, ks_prob))
    return results


def perform_chi2_tests(
    histograms: list[tuple[ROOT.TH1D, str, int]],
) -> list[tuple[str, str, float, int, float]]:
    """
    Perform chi-squared tests between all pairs of histograms.

    Args:
        histograms: List of (histogram, label, color) tuples

    Returns:
        List of (label1, label2, chi2, ndf, p_value) tuples
    """
    results: list[tuple[str, str, float, int, float]] = []
    n_hists: int = len(histograms)
    for i in range(n_hists):
        for j in range(i + 1, n_hists):
            h1, label1, _ = histograms[i]
            h2, label2, _ = histograms[j]
            # Normalize histograms to same integral for chi2 comparison
            h1_norm: ROOT.TH1D = h1.Clone(f"h1_temp_{i}_{j}")
            h2_norm: ROOT.TH1D = h2.Clone(f"h2_temp_{i}_{j}")
            # Let ROOT manage these temporary histograms
            ROOT.SetOwnership(h1_norm, True)
            ROOT.SetOwnership(h2_norm, True)
            if h1_norm.Integral() > 0:
                h1_norm.Scale(1.0 / h1_norm.Integral())
            if h2_norm.Integral() > 0:
                h2_norm.Scale(1.0 / h2_norm.Integral())
            # Chi2 test - get p-value first
            p_value: float = h1_norm.Chi2Test(h2_norm, "WW P")
            chi2: float = h1_norm.Chi2Test(h2_norm, "WW CHI2")
            ndf: int = h1_norm.GetNbinsX()
            results.append((label1, label2, chi2, ndf, p_value))
    return results


def save_test_results(
    ks_results: list[tuple[str, str, float, float]],
    chi2_results: list[tuple[str, str, float, int, float]],
    output_path: Path,
) -> None:
    """
    Save statistical test results to text file.

    Args:
        ks_results: KS test results
        chi2_results: Chi2 test results
        output_path: Path to save results
    """
    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("SHAPE VALIDATION: STATISTICAL TESTS\n")
        f.write("=" * 70 + "\n\n")
        f.write("Kolmogorov-Smirnov Tests\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Region 1':<25} {'Region 2':<25} {'KS Stat':>10} {'p-value':>10}\n")
        f.write("-" * 70 + "\n")
        for label1, label2, ks_stat, p_value in ks_results:
            f.write(f"{label1:<25} {label2:<25} {ks_stat:>10.4f} {p_value:>10.4f}\n")
        f.write("\n")
        f.write("Chi-Squared Tests\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Region 1':<25} {'Region 2':<25} {'chi2/ndf':>12} {'p-value':>10}\n")
        f.write("-" * 70 + "\n")
        for label1, label2, chi2, ndf, p_value in chi2_results:
            chi2_ndf: float = chi2 / ndf if ndf > 0 else 0
            f.write(f"{label1:<25} {label2:<25} {chi2_ndf:>12.2f} {p_value:>10.4f}\n")
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("-" * 70 + "\n")
        f.write("KS Test: p-value > 0.05 suggests shapes are compatible\n")
        f.write("Chi2 Test: chi2/ndf ~ 1 and p-value > 0.05 suggests good agreement\n")
        f.write("=" * 70 + "\n")
    print(f"Saved: {output_path}")


def create_ratio_plot(
    histograms: list[tuple[ROOT.TH1D, str, int]],
    reference_idx: int,
    output_path: Path,
) -> None:
    """
    Create ratio plot comparing all histograms to a reference.

    Args:
        histograms: List of (histogram, label, color) tuples
        reference_idx: Index of reference histogram
        output_path: Path to save the PDF
    """
    keep_alive: list[Any] = []
    canvas: ROOT.TCanvas = ROOT.TCanvas("c_ratio", "", 1200, 1000)
    ROOT.SetOwnership(canvas, False)
    keep_alive.append(canvas)
    # Create two pads: main plot and ratio
    pad1: ROOT.TPad = ROOT.TPad("pad1", "Main", 0, 0.3, 1, 1)
    pad2: ROOT.TPad = ROOT.TPad("pad2", "Ratio", 0, 0, 1, 0.3)
    ROOT.SetOwnership(pad1, False)
    ROOT.SetOwnership(pad2, False)
    keep_alive.extend([pad1, pad2])
    pad1.SetBottomMargin(0.02)
    pad1.SetLeftMargin(0.12)
    pad1.SetRightMargin(0.05)
    pad2.SetTopMargin(0.02)
    pad2.SetBottomMargin(0.35)
    pad2.SetLeftMargin(0.12)
    pad2.SetRightMargin(0.05)
    canvas.cd()
    pad1.Draw()
    pad2.Draw()
    # Normalize histograms
    normalized_hists: list[ROOT.TH1D] = []
    for hist, label, color in histograms:
        h_norm: ROOT.TH1D = hist.Clone(f"{hist.GetName()}_norm_ratio")
        ROOT.SetOwnership(h_norm, False)
        integral: float = h_norm.Integral()
        if integral > 0:
            h_norm.Scale(1.0 / integral)
        h_norm.SetLineColor(color)
        h_norm.SetLineWidth(2)
        normalized_hists.append(h_norm)
        keep_alive.append(h_norm)
    # Main plot
    pad1.cd()
    y_max: float = max(h.GetMaximum() for h in normalized_hists) * 1.3
    normalized_hists[0].SetTitle("")
    normalized_hists[0].GetYaxis().SetTitle("Normalized entries")
    normalized_hists[0].GetYaxis().SetRangeUser(0, y_max)
    normalized_hists[0].GetXaxis().SetLabelSize(0)
    normalized_hists[0].Draw("HIST E")
    for h in normalized_hists[1:]:
        h.Draw("HIST E SAME")
    # Legend
    legend: ROOT.TLegend = ROOT.TLegend(0.55, 0.60, 0.92, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(132)
    legend.SetTextSize(0.045)
    for h_norm, (_, label, _) in zip(normalized_hists, histograms):
        legend.AddEntry(h_norm, label, "l")
    legend.Draw()
    keep_alive.append(legend)
    # LHCb label
    latex: ROOT.TLatex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(132)
    latex.SetTextSize(0.055)
    latex.DrawLatex(0.15, 0.85, "#font[62]{LHCb}")
    keep_alive.append(latex)
    # Ratio plot
    pad2.cd()
    ref_hist: ROOT.TH1D = normalized_hists[reference_idx]
    ratio_hists: list[ROOT.TH1D] = []
    for i, (h_norm, (_, label, color)) in enumerate(zip(normalized_hists, histograms)):
        if i == reference_idx:
            continue
        h_ratio: ROOT.TH1D = h_norm.Clone(f"ratio_{i}")
        ROOT.SetOwnership(h_ratio, False)
        h_ratio.Divide(ref_hist)
        h_ratio.SetLineColor(color)
        h_ratio.SetMarkerColor(color)
        h_ratio.SetMarkerStyle(20)
        h_ratio.SetMarkerSize(0.5)
        ratio_hists.append(h_ratio)
        keep_alive.append(h_ratio)
    if ratio_hists:
        ratio_hists[0].SetTitle("")
        ratio_hists[0].GetXaxis().SetTitle("#it{M}(#Lambda#it{p}#it{K}^{#minus}) [MeV/#it{c}^{2}]")
        ratio_hists[0].GetYaxis().SetTitle("Ratio")
        ratio_hists[0].GetXaxis().SetTitleSize(0.12)
        ratio_hists[0].GetYaxis().SetTitleSize(0.10)
        ratio_hists[0].GetXaxis().SetLabelSize(0.10)
        ratio_hists[0].GetYaxis().SetLabelSize(0.08)
        ratio_hists[0].GetYaxis().SetTitleOffset(0.5)
        ratio_hists[0].GetYaxis().SetRangeUser(0.5, 1.5)
        ratio_hists[0].GetYaxis().SetNdivisions(505)
        ratio_hists[0].Draw("E")
        for h in ratio_hists[1:]:
            h.Draw("E SAME")
        # Unity line
        line: ROOT.TLine = ROOT.TLine(MASS_CONFIG.MLPK_MIN, 1.0, MASS_CONFIG.MLPK_MAX, 1.0)
        line.SetLineColor(ROOT.kBlack)
        line.SetLineStyle(2)
        line.Draw("same")
        keep_alive.append(line)
    canvas.Update()
    canvas.SaveAs(str(output_path))
    print(f"Saved: {output_path}")


def main() -> None:
    """Main function to run shape validation."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Validate M(LpK-) shape independence across M(B+) sidebands"
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2016,2017,2018",
        help="Comma-separated list of years (default: 2016,2017,2018)",
    )
    parser.add_argument(
        "--track-types",
        type=str,
        default="LL,DD",
        help="Comma-separated list of track types (default: LL,DD)",
    )
    parser.add_argument(
        "--no-cuts",
        action="store_true",
        help="Skip selection cuts (for comparison)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: output/)",
    )
    args: argparse.Namespace = parser.parse_args()
    # Parse arguments
    years: list[str] = [y.strip() for y in args.years.split(",")]
    track_types: list[str] = [t.strip() for t in args.track_types.split(",")]
    # Setup
    setup_root_style()
    config = get_config()
    data_path: Path = Path(config.paths["data"]["base_path"])
    # Build cuts
    if args.no_cuts:
        base_cuts: str = ""
        print("\n*** Running WITHOUT selection cuts ***\n")
    else:
        manual_cuts: dict[str, Any] = config.selection.get("manual_cuts", {})
        lambda_selection: dict[str, Any] = config.selection.get("lambda_selection", {})
        base_cuts = build_cut_string(manual_cuts, lambda_selection)
        print("\n*** Running WITH selection cuts ***\n")
    # Output directory
    if args.output_dir:
        output_dir: Path = Path(args.output_dir)
    else:
        output_dir = STUDY_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Years: {years}")
    print(f"Track types: {track_types}\n")
    # Define M(B+) sideband regions to compare
    regions: list[tuple[tuple[float, float], str, int]] = [
        (MASS_CONFIG.LEFT_SIDEBAND_FAR, "Far-left [2800-3500]", ROOT.kBlue),
        (MASS_CONFIG.LEFT_SIDEBAND_MID, "Mid-left [3500-4500]", ROOT.kGreen + 2),
        (MASS_CONFIG.LEFT_SIDEBAND_NEAR, "Near-left [4500-5150]", ROOT.kRed),
        (MASS_CONFIG.RIGHT_SIDEBAND, "Right [5330-5500]", ROOT.kMagenta),
    ]
    # Minimum events required for meaningful shape comparison
    MIN_EVENTS: int = 100
    # Load histograms for each region
    print("=" * 60)
    print("Loading M(LpK-) histograms for each M(B+) region...")
    print(f"(Minimum {MIN_EVENTS} events required for inclusion)")
    print("=" * 60)
    histograms: list[tuple[ROOT.TH1D, str, int]] = []
    for region, label, color in regions:
        print(f"\n  Loading {label}...")
        hist, n_events = load_mlpk_histogram_in_mbu_region(
            data_path=data_path,
            years=years,
            track_types=track_types,
            mbu_region=region,
            hist_name=f"h_mlpk_{label.replace(' ', '_').replace('[', '').replace(']', '').replace('-', '_')}",
            base_cuts=base_cuts,
        )
        print(f"    Events: {n_events:,}")
        if n_events >= MIN_EVENTS:
            histograms.append((hist, label, color))
        elif n_events > 0:
            print(f"    ⚠ Skipping: insufficient statistics (<{MIN_EVENTS} events)")
    if len(histograms) < 2:
        print("\nERROR: Need at least 2 regions with events for comparison!")
        return
    # Create comparison plot
    print("\n" + "=" * 60)
    print("Creating shape comparison plots...")
    print("=" * 60)
    suffix: str = "_no_cuts" if args.no_cuts else ""
    create_shape_comparison_plot(
        histograms=histograms,
        output_path=output_dir / f"shape_validation_comparison{suffix}.pdf",
        title="M(#Lambda#bar{p}K^{#minus}) Shape vs M(B^{+}) Region",
    )
    # Create ratio plot (reference: first histogram with "Near-left" in label, or first one)
    ref_idx: int = 0
    for i, (_, label, _) in enumerate(histograms):
        if "Near-left" in label:
            ref_idx = i
            break
    create_ratio_plot(
        histograms=histograms,
        reference_idx=ref_idx,
        output_path=output_dir / f"shape_validation_ratio{suffix}.pdf",
    )
    # Perform statistical tests
    print("\n" + "=" * 60)
    print("Performing statistical tests...")
    print("=" * 60)
    ks_results = perform_ks_tests(histograms)
    chi2_results = perform_chi2_tests(histograms)
    # Print results
    print("\nKolmogorov-Smirnov Tests:")
    print("-" * 60)
    for label1, label2, ks_stat, p_value in ks_results:
        status: str = "✓ Compatible" if p_value > 0.05 else "✗ Different"
        print(f"  {label1} vs {label2}: KS={ks_stat:.4f}, p={p_value:.4f} {status}")
    print("\nChi-Squared Tests:")
    print("-" * 60)
    for label1, label2, chi2, ndf, p_value in chi2_results:
        chi2_ndf: float = chi2 / ndf if ndf > 0 else 0
        status = "✓ Compatible" if p_value > 0.05 else "✗ Different"
        print(f"  {label1} vs {label2}: chi2/ndf={chi2_ndf:.2f}, p={p_value:.4f} {status}")
    # Save results
    save_test_results(
        ks_results=ks_results,
        chi2_results=chi2_results,
        output_path=output_dir / f"shape_validation_tests{suffix}.txt",
    )
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_compatible_ks: int = sum(1 for _, _, _, p in ks_results if p > 0.05)
    n_compatible_chi2: int = sum(1 for _, _, _, _, p in chi2_results if p > 0.05)
    n_total: int = len(ks_results)
    print(f"KS tests compatible: {n_compatible_ks}/{n_total}")
    print(f"Chi2 tests compatible: {n_compatible_chi2}/{n_total}")
    if n_compatible_ks == n_total and n_compatible_chi2 == n_total:
        print("\n✓ All shapes are statistically compatible!")
        print("  → Safe to use sideband data for background template")
    elif n_compatible_ks >= n_total // 2:
        print("\n⚠ Some shape differences detected")
        print("  → Consider using only near-signal sidebands for template")
    else:
        print("\n✗ Significant shape differences detected!")
        print("  → Sideband background modeling may introduce bias")
    print("=" * 60)


if __name__ == "__main__":
    main()
