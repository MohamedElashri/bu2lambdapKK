#!/usr/bin/env python3
"""
Study 5: chi^2_DTF(B+) Distribution with Restricted Range (0-35)

- Limit chi^2_DTF plot range to 0-35 (instead of 0-100)
- This matches the cut value we apply (chi^2 < 35)

Generates:
- Individual plot of chi^2_DTF with x-axis [0, 35]
- Shows both MC (all states) and Data distributions
- Normalized to unit area for shape comparison
- Vertical dashed line at cut value (35)

Output:
- analysis/studies/feedback_dec2024/output/study5/Bu_DTF_chi2_restricted.pdf

Usage:
    cd analysis/studies/feedback_dec2024
    python study5_chi2_range_fix.py
    python study5_chi2_range_fix.py --years 2016,2017,2018
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import ROOT

# Add analysis directory to path to access modules
SCRIPT_DIR: Path = Path(__file__).parent
ANALYSIS_DIR: Path = SCRIPT_DIR.parent.parent  # Go up two levels to reach analysis/
sys.path.insert(0, str(ANALYSIS_DIR))

from modules.data_handler import TOMLConfig  # noqa: E402

# Disable ROOT GUI and info messages
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

# Signal MC states
SIGNAL_STATES: list[str] = ["Jpsi", "etac", "chic0", "chic1"]

# chi^2_DTF variable with RESTRICTED range
CHI2_VAR: dict[str, Any] = {
    "branch": "Bu_DTF_chi2",
    "cut_type": "less",
    "cut_value": 30.0,
    "x_min": 0.0,
    "x_max": 35.0,  # RESTRICTED from 100.0
    "n_bins": 60,  # Adjusted for new range
    "x_title": "#chi^{2}_{DTF}(B^{+})",
    "log_scale": False,
}


def setup_lhcb_style() -> None:
    """Set up LHCb-style ROOT plotting options."""
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetPadLeftMargin(0.14)
    ROOT.gStyle.SetPadRightMargin(0.05)
    ROOT.gStyle.SetPadTopMargin(0.08)
    ROOT.gStyle.SetPadBottomMargin(0.12)
    ROOT.gStyle.SetTitleFont(132, "XYZ")
    ROOT.gStyle.SetLabelFont(132, "XYZ")
    ROOT.gStyle.SetTextFont(132)
    ROOT.gStyle.SetTitleSize(0.05, "XYZ")
    ROOT.gStyle.SetLabelSize(0.04, "XYZ")


def load_mc_histogram(
    mc_path: Path,
    years: list[str],
    track_types: list[str],
    hist_name: str,
) -> tuple[ROOT.TH1D, int]:
    """
    Load MC data and fill histogram for chi^2_DTF.

    Args:
        mc_path: Base path to MC files
        years: List of years
        track_types: List of track types
        hist_name: Name for histogram

    Returns:
        Tuple of (histogram, n_events)
    """
    branch: str = CHI2_VAR["branch"]
    x_min: float = CHI2_VAR["x_min"]
    x_max: float = CHI2_VAR["x_max"]
    n_bins: int = CHI2_VAR["n_bins"]

    hist: ROOT.TH1D = ROOT.TH1D(hist_name, "", n_bins, x_min, x_max)
    hist.Sumw2()
    total_events: int = 0

    for state in SIGNAL_STATES:
        for year in years:
            year_int: int = int(year)
            for magnet in ["MD", "MU"]:
                for track_type in track_types:
                    filename: str = f"{state}_{year_int - 2000}_{magnet}.root"
                    filepath: Path = mc_path / state / filename
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

                        temp_hist_name: str = (
                            f"temp_{hist_name}_{state}_{year}_{magnet}_{track_type}"
                        )
                        n_entries: int = tree.Draw(
                            f"{branch}>>{temp_hist_name}({n_bins},{x_min},{x_max})",
                            "",
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
                        print(f"    Warning: {filepath.name}: {e}")

    ROOT.SetOwnership(hist, False)
    return hist, total_events


def load_data_histogram(
    data_path: Path,
    years: list[str],
    track_types: list[str],
    hist_name: str,
) -> tuple[ROOT.TH1D, int]:
    """
    Load data and fill histogram for chi^2_DTF.

    Args:
        data_path: Base path to data files
        years: List of years
        track_types: List of track types
        hist_name: Name for histogram

    Returns:
        Tuple of (histogram, n_events)
    """
    branch: str = CHI2_VAR["branch"]
    x_min: float = CHI2_VAR["x_min"]
    x_max: float = CHI2_VAR["x_max"]
    n_bins: int = CHI2_VAR["n_bins"]

    hist: ROOT.TH1D = ROOT.TH1D(hist_name, "", n_bins, x_min, x_max)
    hist.Sumw2()
    total_events: int = 0

    for year in years:
        year_int: int = int(year)
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

                    temp_hist_name: str = f"temp_{hist_name}_{year}_{magnet}_{track_type}"
                    n_entries: int = tree.Draw(
                        f"{branch}>>{temp_hist_name}({n_bins},{x_min},{x_max})",
                        "",
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
                    print(f"    Warning: {filepath.name}: {e}")

    ROOT.SetOwnership(hist, False)
    return hist, total_events


def create_chi2_plot(
    hist_mc: ROOT.TH1D,
    hist_data: ROOT.TH1D,
    output_path: Path,
) -> None:
    """
    Create chi^2_DTF plot with restricted range showing MC and Data with cut line.

    Args:
        hist_mc: MC histogram
        hist_data: Data histogram
        output_path: Path to save PDF
    """
    keep_alive: list = []

    canvas: ROOT.TCanvas = ROOT.TCanvas("c_chi2_restricted", "", 900, 700)
    ROOT.SetOwnership(canvas, False)
    keep_alive.append(canvas)

    canvas.SetLeftMargin(0.14)
    canvas.SetRightMargin(0.05)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.12)

    # Normalize histograms to unit area for shape comparison
    if hist_mc.Integral() > 0:
        hist_mc.Scale(1.0 / hist_mc.Integral())
    if hist_data.Integral() > 0:
        hist_data.Scale(1.0 / hist_data.Integral())

    # Find y maximum
    y_max: float = max(hist_mc.GetMaximum(), hist_data.GetMaximum()) * 1.4
    y_min: float = 0.0

    # Style MC histogram
    hist_mc.SetLineColor(ROOT.kRed)
    hist_mc.SetLineWidth(2)
    hist_mc.SetLineStyle(1)
    hist_mc.SetFillColorAlpha(ROOT.kRed, 0.2)
    hist_mc.SetFillStyle(1001)
    hist_mc.SetMaximum(y_max)
    hist_mc.SetMinimum(y_min)
    hist_mc.GetXaxis().SetTitle(CHI2_VAR["x_title"])
    hist_mc.GetYaxis().SetTitle("Normalized")
    hist_mc.GetXaxis().SetTitleFont(132)
    hist_mc.GetYaxis().SetTitleFont(132)
    hist_mc.GetXaxis().SetLabelFont(132)
    hist_mc.GetYaxis().SetLabelFont(132)
    hist_mc.GetXaxis().SetTitleSize(0.045)
    hist_mc.GetYaxis().SetTitleSize(0.045)
    hist_mc.GetYaxis().SetTitleOffset(1.4)
    hist_mc.Draw("HIST")

    # Style Data histogram
    hist_data.SetLineColor(ROOT.kBlue)
    hist_data.SetLineWidth(2)
    hist_data.SetLineStyle(1)
    hist_data.SetMarkerColor(ROOT.kBlue)
    hist_data.SetMarkerStyle(20)
    hist_data.SetMarkerSize(0.6)
    hist_data.Draw("E SAME")

    # Draw cut line
    cut_value: float = CHI2_VAR["cut_value"]
    cut_line: ROOT.TLine = ROOT.TLine(cut_value, y_min, cut_value, y_max * 0.9)
    ROOT.SetOwnership(cut_line, False)
    keep_alive.append(cut_line)
    cut_line.SetLineColor(ROOT.kBlack)
    cut_line.SetLineWidth(2)
    cut_line.SetLineStyle(2)  # Dashed
    cut_line.Draw()

    # Add arrow indicating kept region (< 35)
    arrow_x1: float = cut_value
    arrow_x2: float = cut_value - (CHI2_VAR["x_max"] - CHI2_VAR["x_min"]) * 0.15
    arrow_y: float = y_max * 0.85
    arrow: ROOT.TArrow = ROOT.TArrow(arrow_x1, arrow_y, arrow_x2, arrow_y, 0.02, ">")
    ROOT.SetOwnership(arrow, False)
    keep_alive.append(arrow)
    arrow.SetLineColor(ROOT.kGreen + 2)
    arrow.SetLineWidth(2)
    arrow.SetFillColor(ROOT.kGreen + 2)
    arrow.Draw()

    # Legend
    legend: ROOT.TLegend = ROOT.TLegend(0.55, 0.65, 0.92, 0.88)
    ROOT.SetOwnership(legend, False)
    keep_alive.append(legend)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(132)
    legend.SetTextSize(0.035)
    legend.AddEntry(hist_mc, "MC (all states)", "f")
    legend.AddEntry(hist_data, "Data", "lep")
    legend.AddEntry(cut_line, f"Cut: {cut_value}", "l")
    legend.Draw()

    # Labels
    lhcb: ROOT.TLatex = ROOT.TLatex()
    lhcb.SetNDC()
    lhcb.SetTextFont(132)
    lhcb.SetTextSize(0.045)
    lhcb.DrawLatex(0.18, 0.85, "LHCb")
    keep_alive.append(lhcb)

    # Cut direction label
    cut_label: ROOT.TLatex = ROOT.TLatex()
    cut_label.SetNDC()
    cut_label.SetTextFont(132)
    cut_label.SetTextSize(0.035)
    cut_label.SetTextColor(ROOT.kGreen + 2)
    cut_label.DrawLatex(0.18, 0.78, f"Keep: < {cut_value}")
    keep_alive.append(cut_label)

    # Study info
    study_label: ROOT.TLatex = ROOT.TLatex()
    study_label.SetNDC()
    study_label.SetTextFont(132)
    study_label.SetTextSize(0.030)
    study_label.SetTextColor(ROOT.kGray + 2)
    keep_alive.append(study_label)

    canvas.Modified()
    canvas.Update()

    canvas.SaveAs(str(output_path))
    canvas.SaveAs(str(output_path.with_suffix(".png")))

    print(f"\n  Saved: {output_path}")
    print(f"  Saved: {output_path.with_suffix('.png')}")


def main() -> None:
    """Main function."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Study 5: chi^2_DTF distribution with restricted range [0, 35]"
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2016,2017,2018",
        help="Comma-separated years (default: 2016,2017,2018)",
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
    print("STUDY 5: chi^2_DTF(B+) Distribution with Restricted Range")
    print("=" * 80)
    print(f"Variable: {CHI2_VAR['branch']}")
    print(f"Range: [{CHI2_VAR['x_min']}, {CHI2_VAR['x_max']}] (restricted from [0, 100])")
    print(f"Cut: < {CHI2_VAR['cut_value']}")
    print(f"Years: {years}")
    print(f"Track types: {track_types}")
    print("=" * 80)

    setup_lhcb_style()

    # Load configuration
    config: TOMLConfig = TOMLConfig(config_dir=str(ANALYSIS_DIR / "config"))
    mc_path: Path = Path(config.paths["mc"]["base_path"])
    data_path: Path = Path(config.paths["data"]["base_path"])

    # Output directory
    output_dir: Path = SCRIPT_DIR / "output" / "study5"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load histograms
    print("\nLoading MC distribution...")
    hist_mc, n_mc = load_mc_histogram(mc_path, years, track_types, "h_mc_chi2")
    print(f"  MC events: {n_mc:,}")

    print("\nLoading Data distribution...")
    hist_data, n_data = load_data_histogram(data_path, years, track_types, "h_data_chi2")
    print(f"  Data events: {n_data:,}")

    # Create plot
    print("\nCreating plot...")
    output_pdf: Path = output_dir / "Bu_DTF_chi2_restricted.pdf"
    create_chi2_plot(hist_mc, hist_data, output_pdf)

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print("Generated chi^2_DTF plot with restricted range [0, 35]")
    print(f"Output directory: {output_dir}")
    print("Files:")
    print("  - Bu_DTF_chi2_restricted.pdf")
    print("  - Bu_DTF_chi2_restricted.png")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
