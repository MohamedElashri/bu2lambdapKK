#!/usr/bin/env python3
"""
Phase 2: Extract Background Template from Sidebands

This script extracts the M(Lambda p K-) background template from B+ mass
sideband regions. The template can be used as a data-driven background
model in mass fitting.

Method:
-------
1. Select events in M(B+) sideband regions (near-left + right)
2. Create M(Lambda p K-) histogram
3. Optionally smooth the template (KDE or spline)
4. Save template as ROOT histogram and RooDataHist

Usage:
------
    cd analysis/studies/sideband_background
    python extract_template.py
    python extract_template.py --smoothing kde
    python extract_template.py --smoothing spline

Output:
-------
    output/background_template.root
    output/background_template.pdf
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


def smooth_histogram_kde(
    hist: ROOT.TH1D,
    bandwidth: float = 10.0,
) -> ROOT.TH1D:
    """
    Smooth histogram using Gaussian kernel density estimation.

    Args:
        hist: Input histogram
        bandwidth: KDE bandwidth in MeV

    Returns:
        Smoothed histogram
    """
    n_bins: int = hist.GetNbinsX()
    x_min: float = hist.GetXaxis().GetXmin()
    x_max: float = hist.GetXaxis().GetXmax()
    bin_width: float = (x_max - x_min) / n_bins
    # Create smoothed histogram
    h_smooth: ROOT.TH1D = ROOT.TH1D(
        f"{hist.GetName()}_smooth_kde",
        hist.GetTitle(),
        n_bins,
        x_min,
        x_max,
    )
    h_smooth.Sumw2()
    ROOT.SetOwnership(h_smooth, False)
    # KDE smoothing
    for i in range(1, n_bins + 1):
        x_i: float = hist.GetBinCenter(i)
        smoothed_value: float = 0.0
        for j in range(1, n_bins + 1):
            x_j: float = hist.GetBinCenter(j)
            weight: float = hist.GetBinContent(j)
            # Gaussian kernel
            kernel: float = ROOT.TMath.Gaus(x_i, x_j, bandwidth, True)
            smoothed_value += weight * kernel * bin_width
        h_smooth.SetBinContent(i, smoothed_value)
    # Normalize to same integral as original
    if h_smooth.Integral() > 0:
        h_smooth.Scale(hist.Integral() / h_smooth.Integral())
    return h_smooth


def smooth_histogram_root(
    hist: ROOT.TH1D,
    n_smooth: int = 3,
) -> ROOT.TH1D:
    """
    Smooth histogram using ROOT's built-in smoothing (353QH twice).

    Args:
        hist: Input histogram
        n_smooth: Number of smoothing iterations

    Returns:
        Smoothed histogram
    """
    h_smooth: ROOT.TH1D = hist.Clone(f"{hist.GetName()}_smooth_root")
    ROOT.SetOwnership(h_smooth, False)
    h_smooth.Smooth(n_smooth)
    return h_smooth


def create_template_plot(
    hist_raw: ROOT.TH1D,
    hist_smooth: ROOT.TH1D | None,
    output_path: Path,
    sideband_label: str,
) -> None:
    """
    Create plot showing raw and smoothed background templates.

    Args:
        hist_raw: Raw histogram from sideband data
        hist_smooth: Smoothed histogram (or None)
        output_path: Path to save the PDF
        sideband_label: Label describing sideband regions used
    """
    keep_alive: list[Any] = []
    canvas: ROOT.TCanvas = ROOT.TCanvas("c_template", "", 1200, 900)
    ROOT.SetOwnership(canvas, False)
    keep_alive.append(canvas)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.05)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.12)
    # Style raw histogram
    hist_raw.SetLineColor(ROOT.kBlue)
    hist_raw.SetLineWidth(2)
    hist_raw.SetMarkerColor(ROOT.kBlue)
    hist_raw.SetMarkerStyle(20)
    hist_raw.SetMarkerSize(0.5)
    # Find y-axis range
    y_max: float = hist_raw.GetMaximum() * 1.3
    # Draw
    hist_raw.SetTitle("Background Template from M(B^{+}) Sidebands")
    hist_raw.GetXaxis().SetTitle("#it{M}(#Lambda#it{p}#it{K}^{#minus}) [MeV/#it{c}^{2}]")
    hist_raw.GetYaxis().SetTitle(f"Candidates / {MASS_CONFIG.BIN_WIDTH:.0f} MeV")
    hist_raw.GetYaxis().SetRangeUser(0, y_max)
    hist_raw.Draw("E")
    # Draw smoothed if available
    if hist_smooth is not None:
        hist_smooth.SetLineColor(ROOT.kRed)
        hist_smooth.SetLineWidth(2)
        hist_smooth.Draw("HIST SAME")
        keep_alive.append(hist_smooth)
    # Legend
    legend: ROOT.TLegend = ROOT.TLegend(0.55, 0.72, 0.92, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(132)
    legend.SetTextSize(0.035)
    legend.AddEntry(hist_raw, "Raw sideband data", "lep")
    if hist_smooth is not None:
        legend.AddEntry(hist_smooth, "Smoothed template", "l")
    legend.Draw()
    keep_alive.append(legend)
    # LHCb label and info
    latex: ROOT.TLatex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(132)
    latex.SetTextSize(0.045)
    latex.DrawLatex(0.15, 0.85, "#font[62]{LHCb}")
    latex.SetTextSize(0.030)
    latex.DrawLatex(0.15, 0.80, f"Sideband: {sideband_label}")
    latex.DrawLatex(0.15, 0.75, f"Entries: {int(hist_raw.GetEntries()):,}")
    keep_alive.append(latex)
    # Draw charmonium reference lines
    for mass, label, color in CHARMONIUM_CONFIG.get_all_states():
        if MASS_CONFIG.MLPK_MIN < mass < MASS_CONFIG.MLPK_MAX:
            line: ROOT.TLine = ROOT.TLine(mass, 0, mass, y_max * 0.5)
            line.SetLineColor(ROOT.kGray + 1)
            line.SetLineStyle(2)
            line.SetLineWidth(1)
            line.Draw("same")
            keep_alive.append(line)
            # Label
            text: ROOT.TLatex = ROOT.TLatex()
            text.SetTextFont(132)
            text.SetTextSize(0.025)
            text.SetTextAngle(90)
            text.DrawLatex(mass + 10, y_max * 0.35, label)
            keep_alive.append(text)
    canvas.Update()
    canvas.SaveAs(str(output_path))
    print(f"Saved: {output_path}")


def save_template_to_root(
    hist_raw: ROOT.TH1D,
    hist_smooth: ROOT.TH1D | None,
    output_path: Path,
) -> None:
    """
    Save background template to ROOT file.

    Args:
        hist_raw: Raw histogram
        hist_smooth: Smoothed histogram (or None)
        output_path: Path to save ROOT file
    """
    tfile: ROOT.TFile = ROOT.TFile.Open(str(output_path), "RECREATE")
    # Save raw histogram
    hist_raw.SetName("background_template_raw")
    hist_raw.Write()
    # Save smoothed histogram if available
    if hist_smooth is not None:
        hist_smooth.SetName("background_template_smooth")
        hist_smooth.Write()
    # Create normalized versions (for use as PDF)
    hist_raw_norm: ROOT.TH1D = hist_raw.Clone("background_template_raw_normalized")
    if hist_raw_norm.Integral() > 0:
        hist_raw_norm.Scale(1.0 / hist_raw_norm.Integral())
    hist_raw_norm.Write()
    if hist_smooth is not None:
        hist_smooth_norm: ROOT.TH1D = hist_smooth.Clone("background_template_smooth_normalized")
        if hist_smooth_norm.Integral() > 0:
            hist_smooth_norm.Scale(1.0 / hist_smooth_norm.Integral())
        hist_smooth_norm.Write()
    tfile.Close()
    print(f"Saved: {output_path}")


def create_roofit_template(
    hist: ROOT.TH1D,
    mass_var_name: str = "mass",
) -> tuple[ROOT.RooRealVar, ROOT.RooDataHist, ROOT.RooHistPdf]:
    """
    Create RooFit objects from histogram template.

    Args:
        hist: Background template histogram
        mass_var_name: Name for the mass variable

    Returns:
        Tuple of (RooRealVar, RooDataHist, RooHistPdf)
    """
    # Create mass variable
    mass: ROOT.RooRealVar = ROOT.RooRealVar(
        mass_var_name,
        "#it{M}(#Lambda#it{p}#it{K}^{#minus})",
        MASS_CONFIG.MLPK_MIN,
        MASS_CONFIG.MLPK_MAX,
        "MeV/#it{c}^{2}",
    )
    ROOT.SetOwnership(mass, False)
    # Create RooDataHist from histogram
    data_hist: ROOT.RooDataHist = ROOT.RooDataHist(
        "bkg_template_datahist",
        "Background template",
        ROOT.RooArgList(mass),
        hist,
    )
    ROOT.SetOwnership(data_hist, False)
    # Create RooHistPdf
    hist_pdf: ROOT.RooHistPdf = ROOT.RooHistPdf(
        "bkg_template_pdf",
        "Background template PDF",
        ROOT.RooArgSet(mass),
        data_hist,
        2,  # Interpolation order
    )
    ROOT.SetOwnership(hist_pdf, False)
    return mass, data_hist, hist_pdf


def main() -> None:
    """Main function to extract background template."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Extract M(LpK-) background template from B+ sidebands"
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
        "--smoothing",
        type=str,
        choices=["none", "kde", "root"],
        default="none",
        help="Smoothing method: none, kde, or root (default: none)",
    )
    parser.add_argument(
        "--kde-bandwidth",
        type=float,
        default=15.0,
        help="KDE bandwidth in MeV (default: 15.0)",
    )
    parser.add_argument(
        "--no-cuts",
        action="store_true",
        help="Skip selection cuts",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--use-all-sidebands",
        action="store_true",
        help="Use all sideband regions (not just near-signal)",
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
    print(f"Track types: {track_types}")
    print(f"Smoothing: {args.smoothing}\n")
    # Define sideband regions
    if args.use_all_sidebands:
        # Use all sideband regions
        sideband_regions: list[tuple[float, float]] = [
            MASS_CONFIG.LEFT_SIDEBAND_FAR,
            MASS_CONFIG.LEFT_SIDEBAND_MID,
            MASS_CONFIG.LEFT_SIDEBAND_NEAR,
            MASS_CONFIG.RIGHT_SIDEBAND,
        ]
        sideband_label: str = "All sidebands [2800-5150] + [5330-5500]"
    else:
        # Use only near-signal sidebands (recommended)
        sideband_regions = [
            MASS_CONFIG.TEMPLATE_SIDEBAND_LEFT,
            MASS_CONFIG.TEMPLATE_SIDEBAND_RIGHT,
        ]
        sideband_label = "Near sidebands [4500-5150] + [5330-5500]"
    print("=" * 60)
    print(f"Extracting template from: {sideband_label}")
    print("=" * 60)
    # Load and combine histograms from all sideband regions
    combined_hist: ROOT.TH1D | None = None
    total_events: int = 0
    for i, region in enumerate(sideband_regions):
        print(f"\n  Loading region [{region[0]:.0f}-{region[1]:.0f}] MeV...")
        hist, n_events = load_mlpk_histogram_in_mbu_region(
            data_path=data_path,
            years=years,
            track_types=track_types,
            mbu_region=region,
            hist_name=f"h_template_region_{i}",
            base_cuts=base_cuts,
        )
        print(f"    Events: {n_events:,}")
        total_events += n_events
        if combined_hist is None:
            combined_hist = hist.Clone("h_background_template")
            ROOT.SetOwnership(combined_hist, False)
        else:
            combined_hist.Add(hist)
    if combined_hist is None or total_events == 0:
        print("\nERROR: No events found in sideband regions!")
        return
    print(f"\n  Total sideband events: {total_events:,}")
    # Apply smoothing if requested
    hist_smooth: ROOT.TH1D | None = None
    if args.smoothing == "kde":
        print(f"\nApplying KDE smoothing (bandwidth={args.kde_bandwidth} MeV)...")
        hist_smooth = smooth_histogram_kde(combined_hist, args.kde_bandwidth)
    elif args.smoothing == "root":
        print("\nApplying ROOT smoothing...")
        hist_smooth = smooth_histogram_root(combined_hist, n_smooth=3)
    # Create plots
    print("\n" + "=" * 60)
    print("Creating output files...")
    print("=" * 60)
    suffix: str = "_no_cuts" if args.no_cuts else ""
    if args.smoothing != "none":
        suffix += f"_{args.smoothing}"
    create_template_plot(
        hist_raw=combined_hist,
        hist_smooth=hist_smooth,
        output_path=output_dir / f"background_template{suffix}.pdf",
        sideband_label=sideband_label,
    )
    # Save to ROOT file
    save_template_to_root(
        hist_raw=combined_hist,
        hist_smooth=hist_smooth,
        output_path=output_dir / f"background_template{suffix}.root",
    )
    # Create RooFit objects and save
    print("\nCreating RooFit template PDF...")
    template_hist: ROOT.TH1D = hist_smooth if hist_smooth is not None else combined_hist
    mass_var, data_hist, hist_pdf = create_roofit_template(template_hist)
    # Save RooFit workspace
    workspace: ROOT.RooWorkspace = ROOT.RooWorkspace(
        "w_bkg_template", "Background Template Workspace"
    )
    getattr(workspace, "import")(hist_pdf)
    workspace.writeToFile(str(output_dir / f"background_template_workspace{suffix}.root"))
    print(f"Saved: {output_dir / f'background_template_workspace{suffix}.root'}")
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Sideband regions: {sideband_label}")
    print(f"Total events: {total_events:,}")
    print(f"Smoothing: {args.smoothing}")
    print("\nOutput files:")
    print(f"  - background_template{suffix}.pdf (visualization)")
    print(f"  - background_template{suffix}.root (histograms)")
    print(f"  - background_template_workspace{suffix}.root (RooFit workspace)")
    print("=" * 60)


if __name__ == "__main__":
    main()
