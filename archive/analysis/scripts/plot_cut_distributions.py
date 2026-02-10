#!/usr/bin/env python3
"""
Plot distributions of cut variables with dashed vertical lines at cut values.

Shows both Data and MC distributions for each cut variable, with a vertical
dashed line indicating where the cut is applied.

Usage:
    cd analysis/scripts
    python plot_cut_distributions.py
    python plot_cut_distributions.py --years 2016,2017,2018
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

# Signal MC states
SIGNAL_STATES: list[str] = ["Jpsi", "etac", "chic0", "chic1"]

# Cut variables with their properties: (branch, cut_type, cut_value, x_min, x_max, n_bins, x_title)
CUT_VARIABLES: list[dict[str, Any]] = [
    {
        "branch": "Bu_DTF_chi2",
        "cut_type": "less",
        "cut_value": 30.0,
        "x_min": 0.0,
        "x_max": 100.0,
        "n_bins": 100,
        "x_title": "#chi^{2}_{DTF}(B^{+})",
        "log_scale": False,
    },
    {
        "branch": "Bu_FDCHI2_OWNPV",
        "cut_type": "greater",
        "cut_value": 100.0,
        "x_min": 0.0,
        "x_max": 1000.0,
        "n_bins": 100,
        "x_title": "FD#chi^{2}(B^{+})",
        "log_scale": True,
    },
    {
        "branch": "Bu_IPCHI2_OWNPV",
        "cut_type": "less",
        "cut_value": 10.0,
        "x_min": 0.0,
        "x_max": 13.0,
        "n_bins": 65,
        "x_title": "IP#chi^{2}(B^{+})",
        "log_scale": False,
    },
    {
        "branch": "Bu_PT",
        "cut_type": "greater",
        "cut_value": 3000.0,
        "x_min": 0.0,
        "x_max": 30000.0,
        "n_bins": 100,
        "x_title": "p_{T}(B^{+}) [MeV/#it{c}]",
        "log_scale": False,
    },
    {
        "branch": "h1_ProbNNk",
        "cut_type": "greater",
        "cut_value": 0.1,
        "x_min": 0.0,
        "x_max": 1.0,
        "n_bins": 100,
        "x_title": "ProbNN_{K}(K^{+})",
        "log_scale": False,
    },
    {
        "branch": "h2_ProbNNk",
        "cut_type": "greater",
        "cut_value": 0.1,
        "x_min": 0.0,
        "x_max": 1.0,
        "n_bins": 100,
        "x_title": "ProbNN_{K}(K^{#minus})",
        "log_scale": False,
    },
    {
        "branch": "p_ProbNNp",
        "cut_type": "greater",
        "cut_value": 0.1,
        "x_min": 0.0,
        "x_max": 1.0,
        "n_bins": 100,
        "x_title": "ProbNN_{p}(p)",
        "log_scale": False,
    },
]


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
    var_info: dict[str, Any],
    hist_name: str,
) -> tuple[ROOT.TH1D, int]:
    """
    Load MC data and fill histogram for a cut variable.

    Args:
        mc_path: Base path to MC files
        years: List of years
        track_types: List of track types
        var_info: Variable info dict
        hist_name: Name for histogram

    Returns:
        Tuple of (histogram, n_events)
    """
    branch: str = var_info["branch"]
    x_min: float = var_info["x_min"]
    x_max: float = var_info["x_max"]
    n_bins: int = var_info["n_bins"]
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
                        print(f"    Error: {e}")
    ROOT.SetOwnership(hist, False)
    return hist, total_events


def load_data_histogram(
    data_path: Path,
    years: list[str],
    track_types: list[str],
    var_info: dict[str, Any],
    hist_name: str,
) -> tuple[ROOT.TH1D, int]:
    """
    Load data and fill histogram for a cut variable.

    Args:
        data_path: Base path to data files
        years: List of years
        track_types: List of track types
        var_info: Variable info dict
        hist_name: Name for histogram

    Returns:
        Tuple of (histogram, n_events)
    """
    branch: str = var_info["branch"]
    x_min: float = var_info["x_min"]
    x_max: float = var_info["x_max"]
    n_bins: int = var_info["n_bins"]
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
                    print(f"    Error: {e}")
    ROOT.SetOwnership(hist, False)
    return hist, total_events


def create_single_variable_plot(
    hist_mc: ROOT.TH1D,
    hist_data: ROOT.TH1D,
    var_info: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Create a single plot for one cut variable showing MC and Data with cut line.

    Args:
        hist_mc: MC histogram
        hist_data: Data histogram
        var_info: Variable info dict
        output_path: Path to save PDF
    """
    keep_alive: list = []
    canvas: ROOT.TCanvas = ROOT.TCanvas(f"c_{var_info['branch']}", "", 900, 700)
    ROOT.SetOwnership(canvas, False)
    keep_alive.append(canvas)
    canvas.SetLeftMargin(0.14)
    canvas.SetRightMargin(0.05)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.12)
    if var_info.get("log_scale", False):
        canvas.SetLogy(True)
    # Normalize histograms to unit area for shape comparison
    if hist_mc.Integral() > 0:
        hist_mc.Scale(1.0 / hist_mc.Integral())
    if hist_data.Integral() > 0:
        hist_data.Scale(1.0 / hist_data.Integral())
    # Find y maximum
    y_max: float = max(hist_mc.GetMaximum(), hist_data.GetMaximum()) * 1.4
    y_min: float = 1e-4 if var_info.get("log_scale", False) else 0
    # Style MC histogram
    hist_mc.SetLineColor(ROOT.kRed)
    hist_mc.SetLineWidth(2)
    hist_mc.SetLineStyle(1)
    hist_mc.SetFillColorAlpha(ROOT.kRed, 0.2)
    hist_mc.SetFillStyle(1001)
    hist_mc.SetMaximum(y_max)
    hist_mc.SetMinimum(y_min)
    hist_mc.GetXaxis().SetTitle(var_info["x_title"])
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
    cut_value: float = var_info["cut_value"]
    cut_line: ROOT.TLine = ROOT.TLine(cut_value, y_min, cut_value, y_max * 0.9)
    ROOT.SetOwnership(cut_line, False)
    keep_alive.append(cut_line)
    cut_line.SetLineColor(ROOT.kBlack)
    cut_line.SetLineWidth(2)
    cut_line.SetLineStyle(2)  # Dashed
    cut_line.Draw()
    # Add arrow indicating kept region
    cut_type: str = var_info["cut_type"]
    arrow_x1: float = cut_value
    arrow_x2: float = cut_value + (var_info["x_max"] - var_info["x_min"]) * 0.1 * (
        1 if cut_type == "greater" else -1
    )
    arrow_y: float = y_max * 0.85
    arrow: ROOT.TArrow = ROOT.TArrow(arrow_x1, arrow_y, arrow_x2, arrow_y, 0.02, ">")
    ROOT.SetOwnership(arrow, False)
    keep_alive.append(arrow)
    arrow.SetLineColor(ROOT.kGreen + 2)
    arrow.SetLineWidth(2)
    arrow.SetFillColor(ROOT.kGreen + 2)
    arrow.Draw()
    # Legend
    legend: ROOT.TLegend = ROOT.TLegend(0.55, 0.70, 0.92, 0.88)
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
    op: str = ">" if cut_type == "greater" else "<"
    cut_label: ROOT.TLatex = ROOT.TLatex()
    cut_label.SetNDC()
    cut_label.SetTextFont(132)
    cut_label.SetTextSize(0.035)
    cut_label.SetTextColor(ROOT.kGreen + 2)
    cut_label.DrawLatex(0.18, 0.78, f"Keep: {op} {cut_value}")
    keep_alive.append(cut_label)
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(output_path))
    canvas.SaveAs(str(output_path.with_suffix(".png")))


def create_multi_panel_plot(
    mc_hists: list[ROOT.TH1D],
    data_hists: list[ROOT.TH1D],
    output_path: Path,
) -> None:
    """
    Create a multi-panel plot with all 7 cut variables.

    Args:
        mc_hists: List of MC histograms
        data_hists: List of Data histograms
        output_path: Path to save PDF
    """
    keep_alive: list = []
    # 4x2 grid for 7 variables (last panel empty or for legend)
    canvas: ROOT.TCanvas = ROOT.TCanvas("c_all_cuts", "", 2000, 1000)
    ROOT.SetOwnership(canvas, False)
    keep_alive.append(canvas)
    canvas.Divide(4, 2, 0.005, 0.005)
    for i, var_info in enumerate(CUT_VARIABLES):
        canvas.cd(i + 1)
        ROOT.gPad.SetLeftMargin(0.15)
        ROOT.gPad.SetRightMargin(0.05)
        ROOT.gPad.SetTopMargin(0.08)
        ROOT.gPad.SetBottomMargin(0.12)
        if var_info.get("log_scale", False):
            ROOT.gPad.SetLogy(True)
        hist_mc: ROOT.TH1D = mc_hists[i].Clone(f"mc_panel_{i}")
        hist_data: ROOT.TH1D = data_hists[i].Clone(f"data_panel_{i}")
        ROOT.SetOwnership(hist_mc, False)
        ROOT.SetOwnership(hist_data, False)
        keep_alive.extend([hist_mc, hist_data])
        # Normalize
        if hist_mc.Integral() > 0:
            hist_mc.Scale(1.0 / hist_mc.Integral())
        if hist_data.Integral() > 0:
            hist_data.Scale(1.0 / hist_data.Integral())
        y_max: float = max(hist_mc.GetMaximum(), hist_data.GetMaximum()) * 1.5
        y_min: float = 1e-4 if var_info.get("log_scale", False) else 0
        # Style MC
        hist_mc.SetLineColor(ROOT.kRed)
        hist_mc.SetLineWidth(2)
        hist_mc.SetFillColorAlpha(ROOT.kRed, 0.2)
        hist_mc.SetFillStyle(1001)
        hist_mc.SetMaximum(y_max)
        hist_mc.SetMinimum(y_min)
        hist_mc.GetXaxis().SetTitle(var_info["x_title"])
        hist_mc.GetYaxis().SetTitle("Normalized")
        hist_mc.GetXaxis().SetTitleFont(132)
        hist_mc.GetYaxis().SetTitleFont(132)
        hist_mc.GetXaxis().SetLabelFont(132)
        hist_mc.GetYaxis().SetLabelFont(132)
        hist_mc.GetXaxis().SetTitleSize(0.055)
        hist_mc.GetYaxis().SetTitleSize(0.055)
        hist_mc.GetXaxis().SetLabelSize(0.045)
        hist_mc.GetYaxis().SetLabelSize(0.045)
        hist_mc.GetYaxis().SetTitleOffset(1.3)
        hist_mc.Draw("HIST")
        # Style Data
        hist_data.SetLineColor(ROOT.kBlue)
        hist_data.SetLineWidth(2)
        hist_data.SetMarkerColor(ROOT.kBlue)
        hist_data.SetMarkerStyle(20)
        hist_data.SetMarkerSize(0.5)
        hist_data.Draw("E SAME")
        # Cut line
        cut_value: float = var_info["cut_value"]
        cut_line: ROOT.TLine = ROOT.TLine(cut_value, y_min, cut_value, y_max * 0.85)
        ROOT.SetOwnership(cut_line, False)
        keep_alive.append(cut_line)
        cut_line.SetLineColor(ROOT.kBlack)
        cut_line.SetLineWidth(2)
        cut_line.SetLineStyle(2)
        cut_line.Draw()
        # Cut label
        cut_type: str = var_info["cut_type"]
        op: str = ">" if cut_type == "greater" else "<"
        cut_label: ROOT.TLatex = ROOT.TLatex()
        cut_label.SetNDC()
        cut_label.SetTextFont(132)
        cut_label.SetTextSize(0.045)
        cut_label.SetTextColor(ROOT.kGreen + 2)
        cut_label.DrawLatex(0.55, 0.82, f"{op} {cut_value}")
        keep_alive.append(cut_label)
        ROOT.gPad.Modified()
        ROOT.gPad.Update()
    # Add legend in last panel
    canvas.cd(8)
    ROOT.gPad.SetLeftMargin(0.1)
    ROOT.gPad.SetRightMargin(0.1)
    legend: ROOT.TLegend = ROOT.TLegend(0.1, 0.3, 0.9, 0.7)
    ROOT.SetOwnership(legend, False)
    keep_alive.append(legend)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(132)
    legend.SetTextSize(0.08)
    # Create dummy histograms for legend
    h_mc_dummy: ROOT.TH1D = ROOT.TH1D("h_mc_dummy", "", 1, 0, 1)
    h_mc_dummy.SetLineColor(ROOT.kRed)
    h_mc_dummy.SetFillColorAlpha(ROOT.kRed, 0.2)
    h_mc_dummy.SetFillStyle(1001)
    ROOT.SetOwnership(h_mc_dummy, False)
    keep_alive.append(h_mc_dummy)
    h_data_dummy: ROOT.TH1D = ROOT.TH1D("h_data_dummy", "", 1, 0, 1)
    h_data_dummy.SetLineColor(ROOT.kBlue)
    h_data_dummy.SetMarkerColor(ROOT.kBlue)
    h_data_dummy.SetMarkerStyle(20)
    ROOT.SetOwnership(h_data_dummy, False)
    keep_alive.append(h_data_dummy)
    cut_line_dummy: ROOT.TLine = ROOT.TLine()
    cut_line_dummy.SetLineColor(ROOT.kBlack)
    cut_line_dummy.SetLineWidth(2)
    cut_line_dummy.SetLineStyle(2)
    ROOT.SetOwnership(cut_line_dummy, False)
    keep_alive.append(cut_line_dummy)
    legend.AddEntry(h_mc_dummy, "MC (all states)", "f")
    legend.AddEntry(h_data_dummy, "Data", "lep")
    legend.AddEntry(cut_line_dummy, "Cut value", "l")
    legend.Draw()
    # LHCb label
    lhcb: ROOT.TLatex = ROOT.TLatex()
    lhcb.SetNDC()
    lhcb.SetTextFont(132)
    lhcb.SetTextSize(0.12)
    lhcb.DrawLatex(0.35, 0.8, "LHCb")
    keep_alive.append(lhcb)
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(output_path))
    canvas.SaveAs(str(output_path.with_suffix(".png")))
    print(f"  Saved: {output_path.name}")


def main() -> None:
    """Main function."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Plot cut variable distributions with cut lines for MC and Data"
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
    print("CUT VARIABLE DISTRIBUTION PLOTTER")
    print("Data and MC distributions with cut lines")
    print("=" * 80)
    print(f"Years: {years}")
    print(f"Track types: {track_types}")
    print("=" * 80)
    print("\nCut variables:")
    for var in CUT_VARIABLES:
        op: str = ">" if var["cut_type"] == "greater" else "<"
        print(f"  - {var['branch']} {op} {var['cut_value']}")
    print()
    setup_lhcb_style()
    config: TOMLConfig = TOMLConfig(config_dir=str(ANALYSIS_DIR / "config"))
    mc_path: Path = Path(config.paths["mc"]["base_path"])
    data_path: Path = Path(config.paths["data"]["base_path"])
    output_dir: Path = ANALYSIS_DIR / "plots" / "cut_distributions"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Load histograms for each variable
    mc_hists: list[ROOT.TH1D] = []
    data_hists: list[ROOT.TH1D] = []
    print("Loading distributions...")
    for i, var_info in enumerate(CUT_VARIABLES):
        print(f"  {var_info['branch']}...")
        hist_mc, n_mc = load_mc_histogram(
            mc_path, years, track_types, var_info, f"h_mc_{var_info['branch']}"
        )
        hist_data, n_data = load_data_histogram(
            data_path, years, track_types, var_info, f"h_data_{var_info['branch']}"
        )
        mc_hists.append(hist_mc)
        data_hists.append(hist_data)
        print(f"    MC: {n_mc:,} events, Data: {n_data:,} events")
    # Create individual plots
    print("\nCreating individual plots...")
    for i, var_info in enumerate(CUT_VARIABLES):
        output_pdf: Path = output_dir / f"{var_info['branch']}.pdf"
        create_single_variable_plot(
            mc_hists[i].Clone(f"mc_{var_info['branch']}_single"),
            data_hists[i].Clone(f"data_{var_info['branch']}_single"),
            var_info,
            output_pdf,
        )
        print(f"  Saved: {output_pdf.name}")
    # Create multi-panel plot
    print("\nCreating multi-panel plot...")
    multi_pdf: Path = output_dir / "all_cut_distributions.pdf"
    create_multi_panel_plot(mc_hists, data_hists, multi_pdf)
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"Generated {len(CUT_VARIABLES)} individual plots + 1 multi-panel plot:")
    for var in CUT_VARIABLES:
        print(f"  - {var['branch']}.pdf")
    print("  - all_cut_distributions.pdf")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
