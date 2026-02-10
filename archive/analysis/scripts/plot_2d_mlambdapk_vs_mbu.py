#!/usr/bin/env python3
"""
2D Plot Study: M(LambdaPK) vs M(B+)

Creates a 2D scatter/histogram plot to study the correlation between:
- X-axis: M(B+) in range [2800, 5000] MeV
- Y-axis: M(Lambda p K-) in range [4800, 6000] MeV

Purpose:
--------
This diagnostic study reveals the correlation structure between the B+ mass
and the Lambda_c p K subsystem mass. Key physics insights:

1. **Identify reflections/backgrounds**: Different physics backgrounds populate
   different regions in this 2D space. A 2D plot reveals structures that 1D
   projections hide.

2. **Understand the Lambda p K mass spectrum**: The [4800-6000] MeV range covers
   regions where we might see:
   - Xi_b^0 -> Lambda_c+ p K- (around 5795 MeV)
   - Other excited states or combinatorial background

3. **B+ mass sidebands**: The [2800-5000] MeV range extends well below and above
   the B+ mass (~5279 MeV), showing how the Lambda p K spectrum changes across
   signal and sideband regions.

4. **Correlations reveal physics**: Diagonal bands or specific structures indicate
   kinematic constraints or specific decay chains.

Usage:
------
    cd analysis/scripts
    python plot_2d_mlambdapk_vs_mbu.py
    python plot_2d_mlambdapk_vs_mbu.py --years 2016,2017,2018
    python plot_2d_mlambdapk_vs_mbu.py --track-types LL,DD
    python plot_2d_mlambdapk_vs_mbu.py --no-cuts  # Skip manual cuts

Output:
-------
    analysis_output/2d_mlambdapk_vs_mbu_<year>.pdf
    analysis_output/2d_mlambdapk_vs_mbu_combined.pdf
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

# Mass ranges for the 2D plot (in MeV)
# These are the DEFAULT ranges - can be overridden via command line
M_BU_MIN: float = 2800.0
M_BU_MAX: float = 5000.0
M_LPKM_MIN: float = 2800.0  # Charmonium region: covers all 5 states
M_LPKM_MAX: float = 4000.0  # eta_c(1S) ~ 2984, eta_c(2S) ~ 3638

# Number of bins for each axis
N_BINS_X: int = 110  # ~20 MeV per bin for M(B+)
N_BINS_Y: int = 60  # ~20 MeV per bin for M(LpK-)


def setup_lhcb_style() -> None:
    """Set up LHCb-style ROOT plotting options."""
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(1)  # Enable title display
    ROOT.gStyle.SetPadLeftMargin(0.14)
    ROOT.gStyle.SetPadRightMargin(0.16)  # Space for color bar
    ROOT.gStyle.SetPadTopMargin(0.08)
    ROOT.gStyle.SetPadBottomMargin(0.12)
    ROOT.gStyle.SetTitleFont(132, "XYZ")
    ROOT.gStyle.SetLabelFont(132, "XYZ")
    ROOT.gStyle.SetTextFont(132)
    ROOT.gStyle.SetTitleSize(0.05, "XYZ")
    ROOT.gStyle.SetLabelSize(0.04, "XYZ")
    ROOT.gStyle.SetTitleOffset(1.1, "X")
    ROOT.gStyle.SetTitleOffset(1.2, "Y")
    ROOT.gStyle.SetTitleOffset(1.0, "Z")
    # Color palette for 2D histogram
    ROOT.gStyle.SetPalette(ROOT.kBird)
    ROOT.gStyle.SetNumberContours(100)


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


def load_data_to_2d_histogram(
    data_path: Path,
    year: str,
    track_types: list[str],
    hist_name: str,
    manual_cuts: dict[str, Any] | None,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n_bins_x: int,
    n_bins_y: int,
) -> tuple[ROOT.TH2D, int]:
    """
    Load data and fill a 2D ROOT histogram with M(B+) vs M(LambdaPK).

    Args:
        data_path: Base path to data files
        year: Year string (e.g., "2016")
        track_types: List of track types (e.g., ["LL", "DD"])
        hist_name: Name for the histogram
        manual_cuts: Dictionary of cuts to apply (None for no cuts)
        x_min: Minimum M(B+) value
        x_max: Maximum M(B+) value
        y_min: Minimum M(LpK-) value
        y_max: Maximum M(LpK-) value
        n_bins_x: Number of bins for X axis
        n_bins_y: Number of bins for Y axis

    Returns:
        Tuple of (2D histogram, total number of events)
    """
    hist: ROOT.TH2D = ROOT.TH2D(hist_name, "", n_bins_x, x_min, x_max, n_bins_y, y_min, y_max)
    hist.Sumw2()
    total_events: int = 0
    year_int: int = int(year)
    # Build cut string from manual cuts
    cut_str: str = build_cut_string(manual_cuts)
    # Add mass range cuts
    mass_range_cuts: list[str] = [
        f"(Bu_MM > {x_min})",
        f"(Bu_MM < {x_max})",
    ]
    if cut_str:
        cut_str = cut_str + " && " + " && ".join(mass_range_cuts)
    else:
        cut_str = " && ".join(mass_range_cuts)
    for magnet in ["MD", "MU"]:
        for track_type in track_types:
            filename: str = f"dataBu2L0barPHH_{year_int - 2000}{magnet}.root"
            filepath: Path = data_path / filename
            if not filepath.exists():
                print(f"  Warning: File not found: {filepath}")
                continue
            channel_path: str = f"B2L0barPKpKm_{track_type}"
            tree_path: str = f"{channel_path}/DecayTree"
            try:
                tfile: ROOT.TFile = ROOT.TFile.Open(str(filepath), "READ")
                if not tfile or tfile.IsZombie():
                    print(f"  Warning: Cannot open file: {filepath}")
                    continue
                tree: ROOT.TTree = tfile.Get(tree_path)
                if not tree:
                    print(f"  Warning: Tree not found: {tree_path}")
                    tfile.Close()
                    continue
                # Calculate M(LpKm) using TTree::Draw with formula
                # M^2 = E^2 - px^2 - py^2 - pz^2
                # M(Lambda + p + K-) where h2 is K-
                mass_formula_lpkm: str = (
                    "sqrt((L0_PE + p_PE + h2_PE)*(L0_PE + p_PE + h2_PE) - "
                    "(L0_PX + p_PX + h2_PX)*(L0_PX + p_PX + h2_PX) - "
                    "(L0_PY + p_PY + h2_PY)*(L0_PY + p_PY + h2_PY) - "
                    "(L0_PZ + p_PZ + h2_PZ)*(L0_PZ + p_PZ + h2_PZ))"
                )
                # 2D draw: Y:X format for TTree::Draw
                draw_expr: str = f"{mass_formula_lpkm}:Bu_MM"
                # Draw to temporary histogram
                temp_hist_name: str = f"temp_{hist_name}_{magnet}_{track_type}"
                temp_hist_binning: str = (
                    f"({n_bins_x},{x_min},{x_max}," f"{n_bins_y},{y_min},{y_max})"
                )
                n_entries: int = tree.Draw(
                    f"{draw_expr}>>{temp_hist_name}{temp_hist_binning}", cut_str, "goff"
                )
                if n_entries > 0:
                    temp_hist: ROOT.TH2D = ROOT.gDirectory.Get(temp_hist_name)
                    if temp_hist:
                        hist.Add(temp_hist)
                        total_events += n_entries
                        print(f"    {year} {magnet} {track_type}: {n_entries:,} events")
                        temp_hist.Delete()
                tfile.Close()
            except Exception as e:
                print(f"    Error loading {year} {magnet} {track_type}: {e}")
    # Keep histogram in memory
    ROOT.SetOwnership(hist, False)
    return hist, total_events


def create_side_by_side_2d_plot(
    hist_no_cuts: ROOT.TH2D,
    hist_with_cuts: ROOT.TH2D,
    year_label: str,
    output_path: Path,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    """
    Create side-by-side 2D plots: without cuts (left) and with cuts (right).

    Args:
        hist_no_cuts: 2D histogram without cuts
        hist_with_cuts: 2D histogram with cuts applied
        year_label: String like "2016" or "Combined"
        output_path: Path to save the PDF
        x_min: Minimum X-axis value
        x_max: Maximum X-axis value
        y_min: Minimum Y-axis value
        y_max: Maximum Y-axis value
    """
    # Keep ROOT objects alive
    keep_alive: list = []
    # Create wide canvas for side-by-side (increased width for less cramped plots)
    canvas_name: str = f"c_2d_{year_label.replace('-', '_').replace(' ', '_')}"
    canvas: ROOT.TCanvas = ROOT.TCanvas(canvas_name, "", 2000, 800)
    ROOT.SetOwnership(canvas, False)
    keep_alive.append(canvas)
    # Charmonium state masses (for horizontal reference lines on Y-axis)
    charmonium_states: dict[str, tuple[float, int, str]] = {
        "etac": (2983.9, ROOT.kRed, "#eta_{c}(1S)"),
        "jpsi": (3096.9, ROOT.kBlue, "J/#psi"),
        "chic0": (3414.7, ROOT.kGreen + 2, "#chi_{c0}"),
        "chic1": (3510.7, ROOT.kOrange + 1, "#chi_{c1}"),
        "etac2s": (3637.5, ROOT.kMagenta, "#eta_{c}(2S)"),
    }
    b_mass: float = 5279.34  # PDG B+ mass
    # Create two pads
    pad1: ROOT.TPad = ROOT.TPad("pad1", "No Cuts", 0.0, 0.0, 0.5, 1.0)
    pad2: ROOT.TPad = ROOT.TPad("pad2", "With Cuts", 0.5, 0.0, 1.0, 1.0)
    ROOT.SetOwnership(pad1, False)
    ROOT.SetOwnership(pad2, False)
    keep_alive.extend([pad1, pad2])
    for pad in [pad1, pad2]:
        pad.SetLeftMargin(0.12)
        pad.SetRightMargin(0.14)
        pad.SetTopMargin(0.10)
        pad.SetBottomMargin(0.10)
        pad.SetLogz(True)
    canvas.cd()
    pad1.Draw()
    pad2.Draw()
    # --- Left pad: No cuts ---
    pad1.cd()
    hist_no_cuts.SetTitle(f"Data {year_label} - No Cuts")
    hist_no_cuts.GetXaxis().SetTitle("#it{M}(#it{B}^{+}) [MeV/#it{c}^{2}]")
    hist_no_cuts.GetYaxis().SetTitle("#it{M}(#Lambda#it{p}#it{K}^{#minus}) [MeV/#it{c}^{2}]")
    # hist_no_cuts.GetZaxis().SetTitle("Candidates")
    hist_no_cuts.GetXaxis().SetTitleSize(0.035)
    hist_no_cuts.GetYaxis().SetTitleSize(0.035)
    hist_no_cuts.GetXaxis().SetLabelSize(0.028)
    hist_no_cuts.GetYaxis().SetLabelSize(0.028)
    hist_no_cuts.GetZaxis().SetLabelSize(0.035)
    hist_no_cuts.GetXaxis().SetTitleOffset(1.1)
    hist_no_cuts.GetYaxis().SetTitleOffset(1.3)
    hist_no_cuts.Draw("COLZ")
    # Draw reference lines for left pad
    for state_name, (mass, color, label) in charmonium_states.items():
        if y_min < mass < y_max:
            line: ROOT.TLine = ROOT.TLine(x_min, mass, x_max, mass)
            line.SetLineColor(color)
            line.SetLineStyle(2)
            line.SetLineWidth(2)
            line.Draw("same")
            keep_alive.append(line)
    if x_min < b_mass < x_max:
        line_b: ROOT.TLine = ROOT.TLine(b_mass, y_min, b_mass, y_max)
        line_b.SetLineColor(ROOT.kRed)
        line_b.SetLineStyle(2)
        line_b.SetLineWidth(2)
        line_b.Draw("same")
        keep_alive.append(line_b)
    # LHCb label on left pad
    latex1: ROOT.TLatex = ROOT.TLatex()
    latex1.SetNDC()
    latex1.SetTextFont(132)
    latex1.SetTextSize(0.035)
    latex1.DrawLatex(0.16, 0.85, "#font[62]{LHCb}")
    keep_alive.append(latex1)
    # --- Right pad: With cuts ---
    pad2.cd()
    hist_with_cuts.SetTitle(f"Data {year_label} - With Cuts")
    hist_with_cuts.GetXaxis().SetTitle("#it{M}(#it{B}^{+}) [MeV/#it{c}^{2}]")
    hist_with_cuts.GetYaxis().SetTitle("#it{M}(#Lambda#it{p}#it{K}^{#minus}) [MeV/#it{c}^{2}]")
    # hist_with_cuts.GetZaxis().SetTitle("Candidates")
    hist_with_cuts.GetXaxis().SetTitleSize(0.035)
    hist_with_cuts.GetYaxis().SetTitleSize(0.035)
    hist_with_cuts.GetXaxis().SetLabelSize(0.028)
    hist_with_cuts.GetYaxis().SetLabelSize(0.028)
    hist_with_cuts.GetZaxis().SetLabelSize(0.035)
    hist_with_cuts.GetXaxis().SetTitleOffset(1.1)
    hist_with_cuts.GetYaxis().SetTitleOffset(1.3)
    hist_with_cuts.Draw("COLZ")
    # Draw reference lines for right pad
    for state_name, (mass, color, label) in charmonium_states.items():
        if y_min < mass < y_max:
            line = ROOT.TLine(x_min, mass, x_max, mass)
            line.SetLineColor(color)
            line.SetLineStyle(2)
            line.SetLineWidth(2)
            line.Draw("same")
            keep_alive.append(line)
    if x_min < b_mass < x_max:
        line_b = ROOT.TLine(b_mass, y_min, b_mass, y_max)
        line_b.SetLineColor(ROOT.kRed)
        line_b.SetLineStyle(2)
        line_b.SetLineWidth(2)
        line_b.Draw("same")
        keep_alive.append(line_b)
    # LHCb label on right pad
    latex2: ROOT.TLatex = ROOT.TLatex()
    latex2.SetNDC()
    latex2.SetTextFont(132)
    latex2.SetTextSize(0.035)
    latex2.DrawLatex(0.16, 0.85, "#font[62]{LHCb}")
    keep_alive.append(latex2)
    # Update and save
    canvas.Update()
    canvas.SaveAs(str(output_path))
    print(f"  Saved: {output_path}")


def main() -> None:
    """Main function to run the 2D plot study."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Create 2D plot of M(LambdaPK) vs M(B+)"
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
        "--mbu-range",
        type=str,
        default="2800,5000",
        help="M(B+) range in MeV as 'min,max' (default: 2800,5000)",
    )
    parser.add_argument(
        "--mlpk-range",
        type=str,
        default="2800,4000",
        help="M(LambdaPK) range in MeV as 'min,max' (default: 2800,4000 for charmonium region)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory (default: analysis_output/)"
    )
    args: argparse.Namespace = parser.parse_args()
    # Parse arguments
    years: list[str] = [y.strip() for y in args.years.split(",")]
    track_types: list[str] = [t.strip() for t in args.track_types.split(",")]
    # Parse mass ranges
    mbu_parts: list[str] = args.mbu_range.split(",")
    m_bu_min: float = float(mbu_parts[0].strip())
    m_bu_max: float = float(mbu_parts[1].strip())
    mlpk_parts: list[str] = args.mlpk_range.split(",")
    m_lpkm_min: float = float(mlpk_parts[0].strip())
    m_lpkm_max: float = float(mlpk_parts[1].strip())
    # Setup
    setup_lhcb_style()
    # Load configuration
    config_dir: Path = ANALYSIS_DIR / "config"
    config: TOMLConfig = TOMLConfig(str(config_dir))
    # Get data path
    data_path: Path = Path(config.paths["data"]["base_path"])
    # Get manual cuts from config
    manual_cuts: dict[str, Any] = config.selection.get("manual_cuts", {})
    if manual_cuts:
        print("\n" + "=" * 60)
        print("MANUAL CUTS (applied to right panel):")
        print("=" * 60)
        for branch_name, cut_spec in manual_cuts.items():
            if branch_name == "notes":
                continue
            cut_type = cut_spec.get("cut_type", "")
            cut_value = cut_spec.get("value", "")
            op: str = ">" if cut_type == "greater" else "<"
            print(f"  {branch_name} {op} {cut_value}")
        print("=" * 60 + "\n")
    # Setup output directory
    if args.output_dir:
        output_dir: Path = Path(args.output_dir)
    else:
        output_dir = ANALYSIS_DIR / "scripts" / "analysis_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print("\nMass ranges:")
    print(f"  M(B+):      [{m_bu_min:.0f}, {m_bu_max:.0f}] MeV")
    print(f"  M(LpK-):    [{m_lpkm_min:.0f}, {m_lpkm_max:.0f}] MeV")
    print(f"\nYears: {years}")
    print(f"Track types: {track_types}\n")
    # Calculate number of bins (~20 MeV per bin)
    n_bins_x: int = int((m_bu_max - m_bu_min) / 20.0)
    n_bins_y: int = int((m_lpkm_max - m_lpkm_min) / 20.0)
    # Create combined histograms (no cuts and with cuts)
    combined_hist_no_cuts: ROOT.TH2D = ROOT.TH2D(
        "h2d_combined_no_cuts", "", n_bins_x, m_bu_min, m_bu_max, n_bins_y, m_lpkm_min, m_lpkm_max
    )
    combined_hist_no_cuts.Sumw2()
    ROOT.SetOwnership(combined_hist_no_cuts, False)
    combined_hist_with_cuts: ROOT.TH2D = ROOT.TH2D(
        "h2d_combined_with_cuts", "", n_bins_x, m_bu_min, m_bu_max, n_bins_y, m_lpkm_min, m_lpkm_max
    )
    combined_hist_with_cuts.Sumw2()
    ROOT.SetOwnership(combined_hist_with_cuts, False)
    total_no_cuts: int = 0
    total_with_cuts: int = 0
    # Process each year
    for year in years:
        print(f"\n{'=' * 40}")
        print(f"Processing {year}")
        print("=" * 40)
        # Load without cuts
        print("  Loading without cuts...")
        hist_no_cuts, n_no_cuts = load_data_to_2d_histogram(
            data_path=data_path,
            year=year,
            track_types=track_types,
            hist_name=f"h2d_{year}_no_cuts",
            manual_cuts=None,
            x_min=m_bu_min,
            x_max=m_bu_max,
            y_min=m_lpkm_min,
            y_max=m_lpkm_max,
            n_bins_x=n_bins_x,
            n_bins_y=n_bins_y,
        )
        # Load with cuts
        print("  Loading with cuts...")
        hist_with_cuts, n_with_cuts = load_data_to_2d_histogram(
            data_path=data_path,
            year=year,
            track_types=track_types,
            hist_name=f"h2d_{year}_with_cuts",
            manual_cuts=manual_cuts,
            x_min=m_bu_min,
            x_max=m_bu_max,
            y_min=m_lpkm_min,
            y_max=m_lpkm_max,
            n_bins_x=n_bins_x,
            n_bins_y=n_bins_y,
        )
        if n_no_cuts > 0 and n_with_cuts > 0:
            # Save individual year plot (side-by-side)
            output_path: Path = output_dir / f"2d_mlambdapk_vs_mbu_{year}.pdf"
            create_side_by_side_2d_plot(
                hist_no_cuts=hist_no_cuts,
                hist_with_cuts=hist_with_cuts,
                year_label=year,
                output_path=output_path,
                x_min=m_bu_min,
                x_max=m_bu_max,
                y_min=m_lpkm_min,
                y_max=m_lpkm_max,
            )
            # Add to combined
            combined_hist_no_cuts.Add(hist_no_cuts)
            combined_hist_with_cuts.Add(hist_with_cuts)
            total_no_cuts += n_no_cuts
            total_with_cuts += n_with_cuts
        else:
            print(f"  Warning: No events loaded for {year}")
    # Create combined plot
    if total_no_cuts > 0 and total_with_cuts > 0:
        print(f"\n{'=' * 40}")
        print("Creating combined plot")
        print("=" * 40)
        output_path = output_dir / "2d_mlambdapk_vs_mbu_combined.pdf"
        create_side_by_side_2d_plot(
            hist_no_cuts=combined_hist_no_cuts,
            hist_with_cuts=combined_hist_with_cuts,
            year_label="2016-2018",
            output_path=output_path,
            x_min=m_bu_min,
            x_max=m_bu_max,
            y_min=m_lpkm_min,
            y_max=m_lpkm_max,
        )
    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Total events (no cuts):   {total_no_cuts:,}")
    print(f"Total events (with cuts): {total_with_cuts:,}")
    print(f"Output files saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
