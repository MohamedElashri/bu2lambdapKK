#!/usr/bin/env python3
"""
Plot reconstructed data M(LambdaPK) distribution in the B+ signal mass window.

Shows side-by-side comparison of the charmonium invariant mass spectrum:
- Left: Before selection cuts (raw data in B+ signal region)
- Right: After selection cuts (in B+ signal region)

Usage:
    cd analysis/scripts
    python plot_data_mass_window.py
    python plot_data_mass_window.py --years 2016,2017,2018
    python plot_data_mass_window.py --track-types LL,DD
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import ROOT

# Add parent directory (analysis) to path to access modules
ANALYSIS_DIR: Path = Path(__file__).parent.parent
sys.path.insert(0, str(ANALYSIS_DIR))

from modules.data_handler import TOMLConfig  # noqa: E402

# Disable ROOT GUI and info messages
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

# M(LambdaPK) range for charmonium region (MeV)
M_LPKM_MIN: float = 2800.0
M_LPKM_MAX: float = 4000.0
N_BINS: int = 120  # 10 MeV per bin

# Charmonium state masses for reference lines (MeV)
CHARMONIUM_STATES: dict[str, tuple[float, int, str]] = {
    "etac": (2983.9, ROOT.kRed, "#eta_{c}(1S)"),
    "jpsi": (3096.9, ROOT.kBlue, "J/#psi"),
    "chic0": (3414.7, ROOT.kGreen + 2, "#chi_{c0}"),
    "chic1": (3510.7, ROOT.kOrange + 1, "#chi_{c1}"),
    "etac2s": (3637.5, ROOT.kMagenta, "#eta_{c}(2S)"),
}


def setup_lhcb_style() -> None:
    """Set up LHCb-style ROOT plotting options."""
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetPadLeftMargin(0.12)
    ROOT.gStyle.SetPadRightMargin(0.05)
    ROOT.gStyle.SetPadTopMargin(0.08)
    ROOT.gStyle.SetPadBottomMargin(0.12)
    ROOT.gStyle.SetTitleFont(132, "XYZ")
    ROOT.gStyle.SetLabelFont(132, "XYZ")
    ROOT.gStyle.SetTextFont(132)
    ROOT.gStyle.SetTitleSize(0.05, "XYZ")
    ROOT.gStyle.SetLabelSize(0.04, "XYZ")
    ROOT.gStyle.SetTitleOffset(1.0, "X")
    ROOT.gStyle.SetTitleOffset(1.1, "Y")


def build_cut_string(manual_cuts: dict[str, Any] | None) -> str:
    """
    Build ROOT TTree cut string from manual cuts dictionary.

    Args:
        manual_cuts: Dictionary of cuts from config/selection.toml

    Returns:
        Cut string for TTree::Draw
    """
    if not manual_cuts:
        return ""
    cut_parts: list[str] = []
    for branch_name, cut_spec in manual_cuts.items():
        if branch_name == "notes":
            continue
        cut_type = cut_spec.get("cut_type")
        cut_value = cut_spec.get("value")
        if cut_type and cut_value is not None:
            if cut_type == "greater":
                cut_parts.append(f"({branch_name} > {cut_value})")
            elif cut_type == "less":
                cut_parts.append(f"({branch_name} < {cut_value})")
    if cut_parts:
        return " && ".join(cut_parts)
    return ""


def load_data_to_histogram(
    data_path: Path,
    year: str,
    track_types: list[str],
    hist_name: str,
    manual_cuts: dict[str, Any] | None,
    bu_mass_min: float,
    bu_mass_max: float,
    x_min: float = M_LPKM_MIN,
    x_max: float = M_LPKM_MAX,
    nbins: int = N_BINS,
) -> tuple[ROOT.TH1D, int]:
    """
    Load data and fill a ROOT histogram with M(LambdaPK) in B+ mass window.

    Args:
        data_path: Base path to data files
        year: Year string (e.g., "2016")
        track_types: List of track types
        hist_name: Name for the histogram
        manual_cuts: Dictionary of cuts to apply (None for no cuts)
        bu_mass_min: Minimum B+ mass for signal window
        bu_mass_max: Maximum B+ mass for signal window
        x_min: Minimum M(LpK) value
        x_max: Maximum M(LpK) value
        nbins: Number of bins

    Returns:
        Tuple of (histogram, n_events)
    """
    hist: ROOT.TH1D = ROOT.TH1D(hist_name, "", nbins, x_min, x_max)
    hist.Sumw2()
    total_events: int = 0
    year_int: int = int(year)
    # Build cut string from manual cuts
    cut_str: str = build_cut_string(manual_cuts)
    # Add B+ mass window cut (signal region)
    bu_mass_cut: str = f"(Bu_MM > {bu_mass_min}) && (Bu_MM < {bu_mass_max})"
    if cut_str:
        cut_str = cut_str + " && " + bu_mass_cut
    else:
        cut_str = bu_mass_cut
    # M(LambdaPK) formula: M(Lambda + p + K-)
    mass_formula: str = (
        "sqrt((L0_PE + p_PE + h2_PE)*(L0_PE + p_PE + h2_PE) - "
        "(L0_PX + p_PX + h2_PX)*(L0_PX + p_PX + h2_PX) - "
        "(L0_PY + p_PY + h2_PY)*(L0_PY + p_PY + h2_PY) - "
        "(L0_PZ + p_PZ + h2_PZ)*(L0_PZ + p_PZ + h2_PZ))"
    )
    for magnet in ["MD", "MU"]:
        for track_type in track_types:
            filename: str = f"dataBu2L0barPHH_{year_int - 2000}{magnet}.root"
            filepath: Path = data_path / filename
            if not filepath.exists():
                continue
            channel_path: str = f"B2L0barPKpKm_{track_type}"
            tree_path: str = f"{channel_path}/DecayTree"
            try:
                tfile: ROOT.TFile = ROOT.TFile.Open(str(filepath), "READ")
                if not tfile or tfile.IsZombie():
                    continue
                tree: ROOT.TTree = tfile.Get(tree_path)
                if not tree:
                    tfile.Close()
                    continue
                # Draw M(LambdaPK) to temporary histogram
                temp_hist_name: str = f"temp_{hist_name}_{magnet}_{track_type}"
                n_entries: int = tree.Draw(
                    f"{mass_formula}>>{temp_hist_name}({nbins},{x_min},{x_max})",
                    cut_str,
                    "goff",
                )
                if n_entries > 0:
                    temp_hist: ROOT.TH1D = ROOT.gDirectory.Get(temp_hist_name)
                    if temp_hist:
                        hist.Add(temp_hist)
                        total_events += n_entries
                        temp_hist.Delete()
                tfile.Close()
            except Exception as e:
                print(f"    Error loading {year} {magnet} {track_type}: {e}")
    ROOT.SetOwnership(hist, False)
    return hist, total_events


def create_comparison_plot(
    hist_no_cuts: ROOT.TH1D,
    hist_with_cuts: ROOT.TH1D,
    year_label: str,
    output_path: Path,
    n_no_cuts: int,
    n_with_cuts: int,
    bu_mass_min: float,
    bu_mass_max: float,
) -> None:
    """
    Create side-by-side comparison plot of M(LambdaPK) distribution.

    Args:
        hist_no_cuts: Histogram without cuts
        hist_with_cuts: Histogram with cuts applied
        year_label: String like "2016" or "Combined"
        output_path: Path to save the PDF
        n_no_cuts: Number of events without cuts
        n_with_cuts: Number of events with cuts
        bu_mass_min: B+ signal region minimum (for label)
        bu_mass_max: B+ signal region maximum (for label)
    """
    keep_alive: list = []
    # Create canvas
    canvas_name: str = f"c_mlpk_{year_label.replace('-', '_').replace(' ', '_')}"
    canvas: ROOT.TCanvas = ROOT.TCanvas(canvas_name, "", 1600, 700)
    ROOT.SetOwnership(canvas, False)
    keep_alive.append(canvas)
    # Find global maximum
    y_max: float = max(hist_no_cuts.GetMaximum(), hist_with_cuts.GetMaximum()) * 1.4
    if y_max == 0:
        y_max = 100
    # Y-axis title
    bin_width: float = (M_LPKM_MAX - M_LPKM_MIN) / N_BINS
    y_title: str = f"Candidates / ({bin_width:.0f} MeV/#it{{c}}^{{2}})"
    x_title: str = "M(#bar{#Lambda}pK^{#minus}) [MeV/#it{c}^{2}]"
    # Create two pads
    pad1: ROOT.TPad = ROOT.TPad("pad1", "No Cuts", 0.0, 0.0, 0.5, 1.0)
    pad2: ROOT.TPad = ROOT.TPad("pad2", "With Cuts", 0.5, 0.0, 1.0, 1.0)
    ROOT.SetOwnership(pad1, False)
    ROOT.SetOwnership(pad2, False)
    keep_alive.extend([pad1, pad2])
    for pad in [pad1, pad2]:
        pad.SetLeftMargin(0.12)
        pad.SetRightMargin(0.05)
        pad.SetTopMargin(0.08)
        pad.SetBottomMargin(0.12)
    canvas.cd()
    pad1.Draw()
    pad2.Draw()
    # --- Left pad: No cuts ---
    pad1.cd()
    hist_no_cuts.SetLineColor(ROOT.kBlack)
    hist_no_cuts.SetLineWidth(2)
    hist_no_cuts.SetFillColor(ROOT.kAzure - 9)
    hist_no_cuts.SetFillStyle(1001)
    hist_no_cuts.SetMaximum(y_max)
    hist_no_cuts.SetMinimum(0)
    hist_no_cuts.GetXaxis().SetTitle(x_title)
    hist_no_cuts.GetYaxis().SetTitle(y_title)
    hist_no_cuts.GetXaxis().SetTitleFont(132)
    hist_no_cuts.GetYaxis().SetTitleFont(132)
    hist_no_cuts.GetXaxis().SetLabelFont(132)
    hist_no_cuts.GetYaxis().SetLabelFont(132)
    hist_no_cuts.GetXaxis().SetTitleSize(0.05)
    hist_no_cuts.GetYaxis().SetTitleSize(0.045)
    hist_no_cuts.GetXaxis().SetLabelSize(0.04)
    hist_no_cuts.GetYaxis().SetLabelSize(0.04)
    hist_no_cuts.GetYaxis().SetTitleOffset(1.3)
    hist_no_cuts.Draw("HIST")
    # Draw charmonium reference lines
    for state_name, (mass, color, label) in CHARMONIUM_STATES.items():
        if M_LPKM_MIN < mass < M_LPKM_MAX:
            line: ROOT.TLine = ROOT.TLine(mass, 0, mass, y_max * 0.85)
            line.SetLineColor(color)
            line.SetLineStyle(2)
            line.SetLineWidth(2)
            line.Draw("same")
            keep_alive.append(line)
    # Labels for left pad
    lhcb1: ROOT.TLatex = ROOT.TLatex()
    lhcb1.SetNDC()
    lhcb1.SetTextFont(132)
    lhcb1.SetTextSize(0.055)
    lhcb1.DrawLatex(0.15, 0.85, "LHCb Data")
    year1: ROOT.TLatex = ROOT.TLatex()
    year1.SetNDC()
    year1.SetTextFont(132)
    year1.SetTextSize(0.04)
    year1.DrawLatex(0.15, 0.78, year_label)
    title1: ROOT.TLatex = ROOT.TLatex()
    title1.SetNDC()
    title1.SetTextFont(132)
    title1.SetTextSize(0.035)
    title1.DrawLatex(0.15, 0.72, "Before Cuts")
    region1: ROOT.TLatex = ROOT.TLatex()
    region1.SetNDC()
    region1.SetTextFont(132)
    region1.SetTextSize(0.030)
    region1.DrawLatex(0.15, 0.66, f"M(B^{{+}}) #in [{bu_mass_min:.0f}, {bu_mass_max:.0f}] MeV")
    nevt1: ROOT.TLatex = ROOT.TLatex()
    nevt1.SetNDC()
    nevt1.SetTextFont(132)
    nevt1.SetTextSize(0.035)
    nevt1.DrawLatex(0.15, 0.60, f"N = {n_no_cuts:,}")
    keep_alive.extend([lhcb1, year1, title1, region1, nevt1])
    # Legend for left pad (charmonium states)
    legend1: ROOT.TLegend = ROOT.TLegend(0.65, 0.55, 0.92, 0.88)
    ROOT.SetOwnership(legend1, False)
    keep_alive.append(legend1)
    legend1.SetBorderSize(0)
    legend1.SetFillStyle(0)
    legend1.SetTextFont(132)
    legend1.SetTextSize(0.028)
    for state_name, (mass, color, label) in CHARMONIUM_STATES.items():
        dummy_line: ROOT.TLine = ROOT.TLine()
        dummy_line.SetLineColor(color)
        dummy_line.SetLineStyle(2)
        dummy_line.SetLineWidth(2)
        legend1.AddEntry(dummy_line, label, "l")
        keep_alive.append(dummy_line)
    legend1.Draw()
    pad1.Modified()
    pad1.Update()
    # --- Right pad: With cuts ---
    pad2.cd()
    hist_with_cuts.SetLineColor(ROOT.kBlack)
    hist_with_cuts.SetLineWidth(2)
    hist_with_cuts.SetFillColor(ROOT.kAzure - 9)
    hist_with_cuts.SetFillStyle(1001)
    hist_with_cuts.SetMaximum(y_max)
    hist_with_cuts.SetMinimum(0)
    hist_with_cuts.GetXaxis().SetTitle(x_title)
    hist_with_cuts.GetYaxis().SetTitle(y_title)
    hist_with_cuts.GetXaxis().SetTitleFont(132)
    hist_with_cuts.GetYaxis().SetTitleFont(132)
    hist_with_cuts.GetXaxis().SetLabelFont(132)
    hist_with_cuts.GetYaxis().SetLabelFont(132)
    hist_with_cuts.GetXaxis().SetTitleSize(0.05)
    hist_with_cuts.GetYaxis().SetTitleSize(0.045)
    hist_with_cuts.GetXaxis().SetLabelSize(0.04)
    hist_with_cuts.GetYaxis().SetLabelSize(0.04)
    hist_with_cuts.GetYaxis().SetTitleOffset(1.3)
    hist_with_cuts.Draw("HIST")
    # Draw charmonium reference lines
    for state_name, (mass, color, label) in CHARMONIUM_STATES.items():
        if M_LPKM_MIN < mass < M_LPKM_MAX:
            line = ROOT.TLine(mass, 0, mass, y_max * 0.85)
            line.SetLineColor(color)
            line.SetLineStyle(2)
            line.SetLineWidth(2)
            line.Draw("same")
            keep_alive.append(line)
    # Labels for right pad
    lhcb2: ROOT.TLatex = ROOT.TLatex()
    lhcb2.SetNDC()
    lhcb2.SetTextFont(132)
    lhcb2.SetTextSize(0.055)
    lhcb2.DrawLatex(0.15, 0.85, "LHCb Data")
    year2: ROOT.TLatex = ROOT.TLatex()
    year2.SetNDC()
    year2.SetTextFont(132)
    year2.SetTextSize(0.04)
    year2.DrawLatex(0.15, 0.78, year_label)
    title2: ROOT.TLatex = ROOT.TLatex()
    title2.SetNDC()
    title2.SetTextFont(132)
    title2.SetTextSize(0.035)
    title2.DrawLatex(0.15, 0.72, "After Cuts")
    region2: ROOT.TLatex = ROOT.TLatex()
    region2.SetNDC()
    region2.SetTextFont(132)
    region2.SetTextSize(0.030)
    region2.DrawLatex(0.15, 0.66, f"M(B^{{+}}) #in [{bu_mass_min:.0f}, {bu_mass_max:.0f}] MeV")
    nevt2: ROOT.TLatex = ROOT.TLatex()
    nevt2.SetNDC()
    nevt2.SetTextFont(132)
    nevt2.SetTextSize(0.035)
    eff: float = 100.0 * n_with_cuts / n_no_cuts if n_no_cuts > 0 else 0.0
    nevt2.DrawLatex(0.15, 0.60, f"N = {n_with_cuts:,} ({eff:.1f}%)")
    keep_alive.extend([lhcb2, year2, title2, region2, nevt2])
    # Legend for right pad
    legend2: ROOT.TLegend = ROOT.TLegend(0.65, 0.55, 0.92, 0.88)
    ROOT.SetOwnership(legend2, False)
    keep_alive.append(legend2)
    legend2.SetBorderSize(0)
    legend2.SetFillStyle(0)
    legend2.SetTextFont(132)
    legend2.SetTextSize(0.028)
    for state_name, (mass, color, label) in CHARMONIUM_STATES.items():
        dummy_line = ROOT.TLine()
        dummy_line.SetLineColor(color)
        dummy_line.SetLineStyle(2)
        dummy_line.SetLineWidth(2)
        legend2.AddEntry(dummy_line, label, "l")
        keep_alive.append(dummy_line)
    legend2.Draw()
    pad2.Modified()
    pad2.Update()
    # Save
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(output_path))
    png_path: Path = output_path.with_suffix(".png")
    canvas.SaveAs(str(png_path))
    print(f"  Saved: {output_path.name}")


def main() -> None:
    """Main function to run the data mass window plotting script."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Plot M(LambdaPK) data distribution in B+ signal mass window (before/after cuts)"
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2016,2017,2018",
        help="Comma-separated years to plot (default: 2016,2017,2018)",
    )
    parser.add_argument(
        "--track-types",
        type=str,
        default="LL,DD",
        help="Comma-separated track types (default: LL,DD)",
    )
    args: argparse.Namespace = parser.parse_args()
    years: list[str] = [y.strip() for y in args.years.split(",")]
    track_types: list[str] = [t.strip() for t in args.track_types.split(",")]
    print("=" * 80)
    print("M(LambdaPK) DATA DISTRIBUTION IN B+ SIGNAL REGION")
    print("Charmonium Spectrum: Before vs After Cuts")
    print("=" * 80)
    print(f"Years: {years}")
    print(f"Track types: {track_types}")
    print(f"M(LambdaPK) range: [{M_LPKM_MIN:.0f}, {M_LPKM_MAX:.0f}] MeV")
    print("=" * 80)
    print()
    # Set up ROOT style
    setup_lhcb_style()
    # Initialize configuration
    config: TOMLConfig = TOMLConfig(config_dir=str(ANALYSIS_DIR / "config"))
    # Get data path
    data_path: Path = Path(config.paths["data"]["base_path"])
    # Get B+ signal region from config
    bu_fixed: dict = config.selection.get("bu_fixed_selection", {})
    bu_mass_min: float = bu_fixed.get("mass_corrected_min", 5255.0)
    bu_mass_max: float = bu_fixed.get("mass_corrected_max", 5305.0)
    print(f"B+ signal region: [{bu_mass_min:.0f}, {bu_mass_max:.0f}] MeV")
    # Get manual cuts from config
    manual_cuts: dict[str, Any] = config.selection.get("manual_cuts", {})
    if manual_cuts:
        print("\nCuts applied:")
        for branch_name, cut_spec in manual_cuts.items():
            if branch_name == "notes":
                continue
            cut_type = cut_spec.get("cut_type", "")
            cut_value = cut_spec.get("value", "")
            op: str = ">" if cut_type == "greater" else "<"
            print(f"  {branch_name} {op} {cut_value}")
    print()
    # Output directory
    output_dir: Path = ANALYSIS_DIR / "plots" / "data_mass_window"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Storage for combined histograms
    combined_hist_no_cuts: ROOT.TH1D | None = None
    combined_hist_with_cuts: ROOT.TH1D | None = None
    combined_n_no_cuts: int = 0
    combined_n_with_cuts: int = 0
    # Process each year
    for year in years:
        print(f"\n{'=' * 60}")
        print(f"YEAR {year}")
        print("=" * 60)
        print("Loading data...")
        # Load without selection cuts (only B+ mass window)
        hist_no_cuts, n_no_cuts = load_data_to_histogram(
            data_path, year, track_types, f"h_mlpk_{year}_nocuts", None, bu_mass_min, bu_mass_max
        )
        print(f"  Without cuts: {n_no_cuts:,} events")
        # Load with selection cuts (+ B+ mass window)
        hist_with_cuts, n_with_cuts = load_data_to_histogram(
            data_path,
            year,
            track_types,
            f"h_mlpk_{year}_cuts",
            manual_cuts,
            bu_mass_min,
            bu_mass_max,
        )
        print(f"  With cuts:    {n_with_cuts:,} events")
        eff: float = 100.0 * n_with_cuts / n_no_cuts if n_no_cuts > 0 else 0.0
        print(f"  Efficiency:   {eff:.1f}%")
        # Add to combined
        if combined_hist_no_cuts is None:
            combined_hist_no_cuts = hist_no_cuts.Clone("h_mlpk_combined_nocuts")
            combined_hist_with_cuts = hist_with_cuts.Clone("h_mlpk_combined_cuts")
            ROOT.SetOwnership(combined_hist_no_cuts, False)
            ROOT.SetOwnership(combined_hist_with_cuts, False)
        else:
            combined_hist_no_cuts.Add(hist_no_cuts)
            combined_hist_with_cuts.Add(hist_with_cuts)
        combined_n_no_cuts += n_no_cuts
        combined_n_with_cuts += n_with_cuts
        # Create plot for this year
        print(f"\nCreating plot for {year}...")
        year_pdf: Path = output_dir / f"data_mlpk_bu_region_{year}.pdf"
        create_comparison_plot(
            hist_no_cuts,
            hist_with_cuts,
            year,
            year_pdf,
            n_no_cuts,
            n_with_cuts,
            bu_mass_min,
            bu_mass_max,
        )
    # Create combined plot
    print(f"\n{'=' * 60}")
    print("COMBINED (All Years)")
    print("=" * 60)
    print(f"  Without cuts: {combined_n_no_cuts:,} events")
    print(f"  With cuts:    {combined_n_with_cuts:,} events")
    eff = 100.0 * combined_n_with_cuts / combined_n_no_cuts if combined_n_no_cuts > 0 else 0.0
    print(f"  Efficiency:   {eff:.1f}%")
    print("\nCreating combined plot...")
    combined_pdf: Path = output_dir / "data_mlpk_bu_region_combined.pdf"
    create_comparison_plot(
        combined_hist_no_cuts,
        combined_hist_with_cuts,
        "2016-2018",
        combined_pdf,
        combined_n_no_cuts,
        combined_n_with_cuts,
        bu_mass_min,
        bu_mass_max,
    )
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"Generated {len(years) + 1} plots:")
    for year in years:
        print(f"  - data_mlpk_bu_region_{year}.pdf")
    print("  - data_mlpk_bu_region_combined.pdf")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
