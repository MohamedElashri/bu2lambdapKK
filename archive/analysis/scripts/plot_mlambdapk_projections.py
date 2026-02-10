#!/usr/bin/env python3
"""
1D Projections Study: M(LambdaPK) in M(B+) Slices

Creates 1D projections of M(Lambda p K-) comparing:
- B+ signal region (around 5279 MeV)
- Low sideband (below signal)
- High sideband (above signal)

Purpose:
--------
This study reveals how the M(Lambda p K-) spectrum changes across different
M(B+) regions. Key physics insights:

1. **Signal region**: Should show clear charmonium peaks (J/psi, eta_c, etc.)
   sitting on top of combinatorial background

2. **Sidebands**: Should show only combinatorial background shape
   - If charmonium peaks appear in sidebands, indicates feed-down or reflections
   - Shape comparison helps understand background under signal

3. **Sideband subtraction**: Comparing signal vs sideband shapes reveals
   the true signal contribution

Usage:
------
    cd analysis/scripts
    python plot_mlambdapk_projections.py
    python plot_mlambdapk_projections.py --years 2016,2017,2018
    python plot_mlambdapk_projections.py --with-cuts

Output:
-------
    analysis_output/mlambdapk_projections_<year>.pdf
    analysis_output/mlambdapk_projections_combined.pdf
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

# M(B+) region definitions (in MeV)
B_MASS_PDG: float = 5279.34
B_SIGNAL_HALF_WIDTH: float = 25.0  # ±25 MeV around B+ mass

# Signal region: [5254, 5304] MeV
B_SIGNAL_MIN: float = B_MASS_PDG - B_SIGNAL_HALF_WIDTH
B_SIGNAL_MAX: float = B_MASS_PDG + B_SIGNAL_HALF_WIDTH

# Low sideband: [5150, 5230] MeV (well below signal)
B_LOW_SB_MIN: float = 5150.0
B_LOW_SB_MAX: float = 5230.0

# High sideband: [5330, 5410] MeV (well above signal)
B_HIGH_SB_MIN: float = 5330.0
B_HIGH_SB_MAX: float = 5410.0

# M(LambdaPK) range for projections
M_LPKM_MIN: float = 2800.0
M_LPKM_MAX: float = 5000.0
N_BINS: int = 60  # ~20 MeV per bin


def setup_lhcb_style() -> None:
    """Set up LHCb-style ROOT plotting options."""
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(1)
    ROOT.gStyle.SetPadLeftMargin(0.14)
    ROOT.gStyle.SetPadRightMargin(0.05)
    ROOT.gStyle.SetPadTopMargin(0.08)
    ROOT.gStyle.SetPadBottomMargin(0.12)
    ROOT.gStyle.SetTitleFont(132, "XYZ")
    ROOT.gStyle.SetLabelFont(132, "XYZ")
    ROOT.gStyle.SetTextFont(132)
    ROOT.gStyle.SetTitleSize(0.045, "XYZ")
    ROOT.gStyle.SetLabelSize(0.035, "XYZ")
    ROOT.gStyle.SetTitleOffset(1.1, "X")
    ROOT.gStyle.SetTitleOffset(1.2, "Y")


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


def load_projection_histogram(
    data_path: Path,
    year: str,
    track_types: list[str],
    hist_name: str,
    bu_mass_min: float,
    bu_mass_max: float,
    manual_cuts: dict[str, Any] | None,
) -> tuple[ROOT.TH1D, int]:
    """
    Load data and fill a 1D histogram with M(LambdaPK) for a specific M(B+) slice.

    Args:
        data_path: Base path to data files
        year: Year string (e.g., "2016")
        track_types: List of track types (e.g., ["LL", "DD"])
        hist_name: Name for the histogram
        bu_mass_min: Minimum M(B+) for this slice
        bu_mass_max: Maximum M(B+) for this slice
        manual_cuts: Dictionary of cuts to apply (None for no cuts)

    Returns:
        Tuple of (1D histogram, total number of events)
    """
    hist: ROOT.TH1D = ROOT.TH1D(hist_name, "", N_BINS, M_LPKM_MIN, M_LPKM_MAX)
    hist.Sumw2()
    total_events: int = 0
    year_int: int = int(year)
    # Build cut string
    cut_str: str = build_cut_string(manual_cuts)
    # Add M(B+) slice cuts
    bu_mass_cuts: str = f"(Bu_MM > {bu_mass_min}) && (Bu_MM < {bu_mass_max})"
    if cut_str:
        cut_str = cut_str + " && " + bu_mass_cuts
    else:
        cut_str = bu_mass_cuts
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
                # M(Lambda + p + K-) formula where h2 is K-
                mass_formula: str = (
                    "sqrt((L0_PE + p_PE + h2_PE)*(L0_PE + p_PE + h2_PE) - "
                    "(L0_PX + p_PX + h2_PX)*(L0_PX + p_PX + h2_PX) - "
                    "(L0_PY + p_PY + h2_PY)*(L0_PY + p_PY + h2_PY) - "
                    "(L0_PZ + p_PZ + h2_PZ)*(L0_PZ + p_PZ + h2_PZ))"
                )
                temp_hist_name: str = f"temp_{hist_name}_{magnet}_{track_type}"
                n_entries: int = tree.Draw(
                    f"{mass_formula}>>{temp_hist_name}({N_BINS},{M_LPKM_MIN},{M_LPKM_MAX})",
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


def draw_single_pad(
    pad: ROOT.TPad,
    hist_signal: ROOT.TH1D,
    hist_low_sb: ROOT.TH1D,
    hist_high_sb: ROOT.TH1D,
    title: str,
    y_max: float,
    n_signal: int,
    keep_alive: list,
) -> None:
    """
    Draw histograms on a single pad.

    Args:
        pad: The TPad to draw on
        hist_signal: Histogram from B+ signal region
        hist_low_sb: Histogram from low sideband
        hist_high_sb: Histogram from high sideband
        title: Title for this pad
        y_max: Maximum Y value for axis
        n_signal: Number of events in signal region
        keep_alive: List to store ROOT objects
    """
    pad.cd()
    # Style histograms
    hist_signal.SetLineColor(ROOT.kBlue)
    hist_signal.SetLineWidth(2)
    hist_signal.SetFillColor(ROOT.kBlue)
    hist_signal.SetFillStyle(3004)
    hist_low_sb.SetLineColor(ROOT.kRed)
    hist_low_sb.SetLineWidth(2)
    hist_low_sb.SetLineStyle(2)
    hist_high_sb.SetLineColor(ROOT.kGreen + 2)
    hist_high_sb.SetLineWidth(2)
    hist_high_sb.SetLineStyle(2)
    # Set titles and draw
    hist_signal.SetTitle(title)
    hist_signal.GetXaxis().SetTitle("#it{M}(#Lambda#it{p}#it{K}^{#minus}) [MeV/#it{c}^{2}]")
    hist_signal.GetYaxis().SetTitle("Candidates / (20 MeV/#it{c}^{2})")
    hist_signal.GetXaxis().SetTitleSize(0.04)
    hist_signal.GetYaxis().SetTitleSize(0.04)
    hist_signal.GetXaxis().SetLabelSize(0.035)
    hist_signal.GetYaxis().SetLabelSize(0.035)
    hist_signal.SetMaximum(y_max)
    hist_signal.SetMinimum(0)
    hist_signal.Draw("HIST")
    hist_low_sb.Draw("HIST SAME")
    hist_high_sb.Draw("HIST SAME")
    # Draw charmonium reference lines
    charmonium_states: dict[str, tuple[float, int, str]] = {
        "etac": (2983.9, ROOT.kRed, "#eta_{c}"),
        "jpsi": (3096.9, ROOT.kBlue, "J/#psi"),
        "chic0": (3414.7, ROOT.kGreen + 2, "#chi_{c0}"),
        "chic1": (3510.7, ROOT.kOrange + 1, "#chi_{c1}"),
        "etac2s": (3637.5, ROOT.kMagenta, "#eta_{c}(2S)"),
    }
    for state_name, (mass, color, label) in charmonium_states.items():
        if M_LPKM_MIN < mass < M_LPKM_MAX:
            line: ROOT.TLine = ROOT.TLine(mass, 0, mass, y_max * 0.9)
            line.SetLineColor(ROOT.kGray + 1)
            line.SetLineStyle(3)
            line.SetLineWidth(1)
            line.Draw("same")
            keep_alive.append(line)
    # Legend
    legend: ROOT.TLegend = ROOT.TLegend(0.50, 0.68, 0.92, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(132)
    legend.SetTextSize(0.025)
    legend.AddEntry(
        hist_signal,
        f"Signal [{B_SIGNAL_MIN:.0f}-{B_SIGNAL_MAX:.0f}] MeV (N={n_signal:,})",
        "f",
    )
    legend.AddEntry(
        hist_low_sb,
        f"Low SB [{B_LOW_SB_MIN:.0f}-{B_LOW_SB_MAX:.0f}] MeV (scaled)",
        "l",
    )
    legend.AddEntry(
        hist_high_sb,
        f"High SB [{B_HIGH_SB_MIN:.0f}-{B_HIGH_SB_MAX:.0f}] MeV (scaled)",
        "l",
    )
    legend.Draw()
    keep_alive.append(legend)
    # LHCb label
    latex: ROOT.TLatex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(132)
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.14, 0.88, "#font[62]{LHCb}")
    keep_alive.append(latex)


def create_side_by_side_projection_plot(
    hists_no_cuts: tuple[ROOT.TH1D, ROOT.TH1D, ROOT.TH1D],
    hists_with_cuts: tuple[ROOT.TH1D, ROOT.TH1D, ROOT.TH1D],
    year_label: str,
    output_path: Path,
    n_no_cuts: tuple[int, int, int],
    n_with_cuts: tuple[int, int, int],
) -> None:
    """
    Create side-by-side projection plots: without cuts (left) and with cuts (right).

    Args:
        hists_no_cuts: Tuple of (signal, low_sb, high_sb) histograms without cuts
        hists_with_cuts: Tuple of (signal, low_sb, high_sb) histograms with cuts
        year_label: String like "2016" or "Combined"
        output_path: Path to save the PDF
        n_no_cuts: Tuple of (n_signal, n_low_sb, n_high_sb) without cuts
        n_with_cuts: Tuple of (n_signal, n_low_sb, n_high_sb) with cuts
    """
    keep_alive: list = []
    # Create wide canvas for side-by-side
    canvas_name: str = f"c_proj_{year_label.replace('-', '_').replace(' ', '_')}"
    canvas: ROOT.TCanvas = ROOT.TCanvas(canvas_name, "", 1800, 700)
    ROOT.SetOwnership(canvas, False)
    keep_alive.append(canvas)
    # Create two pads
    pad1: ROOT.TPad = ROOT.TPad("pad1", "No Cuts", 0.0, 0.0, 0.5, 1.0)
    pad2: ROOT.TPad = ROOT.TPad("pad2", "With Cuts", 0.5, 0.0, 1.0, 1.0)
    ROOT.SetOwnership(pad1, False)
    ROOT.SetOwnership(pad2, False)
    keep_alive.extend([pad1, pad2])
    for pad in [pad1, pad2]:
        pad.SetLeftMargin(0.12)
        pad.SetRightMargin(0.05)
        pad.SetTopMargin(0.10)
        pad.SetBottomMargin(0.12)
    canvas.cd()
    pad1.Draw()
    pad2.Draw()
    # Unpack histograms
    hist_signal_nc, hist_low_sb_nc, hist_high_sb_nc = hists_no_cuts
    hist_signal_wc, hist_low_sb_wc, hist_high_sb_wc = hists_with_cuts
    n_signal_nc, n_low_sb_nc, n_high_sb_nc = n_no_cuts
    n_signal_wc, n_low_sb_wc, n_high_sb_wc = n_with_cuts
    # Normalize sidebands to signal region width
    signal_width: float = B_SIGNAL_MAX - B_SIGNAL_MIN
    low_sb_width: float = B_LOW_SB_MAX - B_LOW_SB_MIN
    high_sb_width: float = B_HIGH_SB_MAX - B_HIGH_SB_MIN
    for hist_low, hist_high in [
        (hist_low_sb_nc, hist_high_sb_nc),
        (hist_low_sb_wc, hist_high_sb_wc),
    ]:
        if hist_low.Integral() > 0:
            hist_low.Scale(signal_width / low_sb_width)
        if hist_high.Integral() > 0:
            hist_high.Scale(signal_width / high_sb_width)
    # Find global maximum for consistent Y-axis
    y_max: float = max(
        hist_signal_nc.GetMaximum(),
        hist_low_sb_nc.GetMaximum(),
        hist_high_sb_nc.GetMaximum(),
        hist_signal_wc.GetMaximum(),
        hist_low_sb_wc.GetMaximum(),
        hist_high_sb_wc.GetMaximum(),
    )
    y_max *= 1.4
    # Draw left pad (no cuts)
    draw_single_pad(
        pad1,
        hist_signal_nc,
        hist_low_sb_nc,
        hist_high_sb_nc,
        f"Data {year_label} - No Cuts",
        y_max,
        n_signal_nc,
        keep_alive,
    )
    # Draw right pad (with cuts)
    draw_single_pad(
        pad2,
        hist_signal_wc,
        hist_low_sb_wc,
        hist_high_sb_wc,
        f"Data {year_label} - With Cuts",
        y_max,
        n_signal_wc,
        keep_alive,
    )
    # Update and save
    canvas.Update()
    canvas.SaveAs(str(output_path))
    print(f"  Saved: {output_path}")


def main() -> None:
    """Main function to run the projection study."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Create M(LambdaPK) projections in M(B+) slices"
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
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: analysis_output/)",
    )
    args: argparse.Namespace = parser.parse_args()
    # Parse arguments
    years: list[str] = [y.strip() for y in args.years.split(",")]
    track_types: list[str] = [t.strip() for t in args.track_types.split(",")]
    # Setup
    setup_lhcb_style()
    # Load configuration
    config_dir: Path = ANALYSIS_DIR / "config"
    config: TOMLConfig = TOMLConfig(str(config_dir))
    data_path: Path = Path(config.paths["data"]["base_path"])
    # Get manual cuts
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
    # Print configuration
    print(f"Output directory: {output_dir}")
    print("\nM(B+) regions:")
    print(f"  Signal:   [{B_SIGNAL_MIN:.0f}, {B_SIGNAL_MAX:.0f}] MeV")
    print(f"  Low SB:   [{B_LOW_SB_MIN:.0f}, {B_LOW_SB_MAX:.0f}] MeV")
    print(f"  High SB:  [{B_HIGH_SB_MIN:.0f}, {B_HIGH_SB_MAX:.0f}] MeV")
    print(f"\nM(LpK-) range: [{M_LPKM_MIN:.0f}, {M_LPKM_MAX:.0f}] MeV")
    print(f"Years: {years}")
    print(f"Track types: {track_types}\n")
    # Combined histograms (no cuts)
    combined_signal_nc: ROOT.TH1D = ROOT.TH1D(
        "h_combined_signal_nc", "", N_BINS, M_LPKM_MIN, M_LPKM_MAX
    )
    combined_low_sb_nc: ROOT.TH1D = ROOT.TH1D(
        "h_combined_low_sb_nc", "", N_BINS, M_LPKM_MIN, M_LPKM_MAX
    )
    combined_high_sb_nc: ROOT.TH1D = ROOT.TH1D(
        "h_combined_high_sb_nc", "", N_BINS, M_LPKM_MIN, M_LPKM_MAX
    )
    # Combined histograms (with cuts)
    combined_signal_wc: ROOT.TH1D = ROOT.TH1D(
        "h_combined_signal_wc", "", N_BINS, M_LPKM_MIN, M_LPKM_MAX
    )
    combined_low_sb_wc: ROOT.TH1D = ROOT.TH1D(
        "h_combined_low_sb_wc", "", N_BINS, M_LPKM_MIN, M_LPKM_MAX
    )
    combined_high_sb_wc: ROOT.TH1D = ROOT.TH1D(
        "h_combined_high_sb_wc", "", N_BINS, M_LPKM_MIN, M_LPKM_MAX
    )
    for h in [
        combined_signal_nc,
        combined_low_sb_nc,
        combined_high_sb_nc,
        combined_signal_wc,
        combined_low_sb_wc,
        combined_high_sb_wc,
    ]:
        h.Sumw2()
        ROOT.SetOwnership(h, False)
    total_signal_nc: int = 0
    total_low_sb_nc: int = 0
    total_high_sb_nc: int = 0
    total_signal_wc: int = 0
    total_low_sb_wc: int = 0
    total_high_sb_wc: int = 0
    # Process each year
    for year in years:
        print(f"\n{'=' * 40}")
        print(f"Processing {year}")
        print("=" * 40)
        # Load WITHOUT cuts
        print("  Loading without cuts...")
        hist_signal_nc, n_signal_nc = load_projection_histogram(
            data_path, year, track_types, f"h_signal_{year}_nc", B_SIGNAL_MIN, B_SIGNAL_MAX, None
        )
        hist_low_sb_nc, n_low_sb_nc = load_projection_histogram(
            data_path, year, track_types, f"h_low_sb_{year}_nc", B_LOW_SB_MIN, B_LOW_SB_MAX, None
        )
        hist_high_sb_nc, n_high_sb_nc = load_projection_histogram(
            data_path, year, track_types, f"h_high_sb_{year}_nc", B_HIGH_SB_MIN, B_HIGH_SB_MAX, None
        )
        print(
            f"    No cuts - Signal: {n_signal_nc:,}, Low SB: {n_low_sb_nc:,}, High SB: {n_high_sb_nc:,}"
        )
        # Load WITH cuts
        print("  Loading with cuts...")
        hist_signal_wc, n_signal_wc = load_projection_histogram(
            data_path,
            year,
            track_types,
            f"h_signal_{year}_wc",
            B_SIGNAL_MIN,
            B_SIGNAL_MAX,
            manual_cuts,
        )
        hist_low_sb_wc, n_low_sb_wc = load_projection_histogram(
            data_path,
            year,
            track_types,
            f"h_low_sb_{year}_wc",
            B_LOW_SB_MIN,
            B_LOW_SB_MAX,
            manual_cuts,
        )
        hist_high_sb_wc, n_high_sb_wc = load_projection_histogram(
            data_path,
            year,
            track_types,
            f"h_high_sb_{year}_wc",
            B_HIGH_SB_MIN,
            B_HIGH_SB_MAX,
            manual_cuts,
        )
        print(
            f"    With cuts - Signal: {n_signal_wc:,}, Low SB: {n_low_sb_wc:,}, High SB: {n_high_sb_wc:,}"
        )
        if n_signal_nc > 0 and n_signal_wc > 0:
            # Create year plot (side-by-side)
            output_path: Path = output_dir / f"mlambdapk_projections_{year}.pdf"
            create_side_by_side_projection_plot(
                (hist_signal_nc, hist_low_sb_nc, hist_high_sb_nc),
                (hist_signal_wc, hist_low_sb_wc, hist_high_sb_wc),
                year,
                output_path,
                (n_signal_nc, n_low_sb_nc, n_high_sb_nc),
                (n_signal_wc, n_low_sb_wc, n_high_sb_wc),
            )
            # Add to combined
            combined_signal_nc.Add(hist_signal_nc)
            combined_low_sb_nc.Add(hist_low_sb_nc)
            combined_high_sb_nc.Add(hist_high_sb_nc)
            combined_signal_wc.Add(hist_signal_wc)
            combined_low_sb_wc.Add(hist_low_sb_wc)
            combined_high_sb_wc.Add(hist_high_sb_wc)
            total_signal_nc += n_signal_nc
            total_low_sb_nc += n_low_sb_nc
            total_high_sb_nc += n_high_sb_nc
            total_signal_wc += n_signal_wc
            total_low_sb_wc += n_low_sb_wc
            total_high_sb_wc += n_high_sb_wc
    # Create combined plot
    if total_signal_nc > 0 and total_signal_wc > 0:
        print(f"\n{'=' * 40}")
        print("Creating combined plot")
        print("=" * 40)
        output_path = output_dir / "mlambdapk_projections_combined.pdf"
        create_side_by_side_projection_plot(
            (combined_signal_nc, combined_low_sb_nc, combined_high_sb_nc),
            (combined_signal_wc, combined_low_sb_wc, combined_high_sb_wc),
            "2016-2018",
            output_path,
            (total_signal_nc, total_low_sb_nc, total_high_sb_nc),
            (total_signal_wc, total_low_sb_wc, total_high_sb_wc),
        )
    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Signal region (no cuts):   {total_signal_nc:,}")
    print(f"Signal region (with cuts): {total_signal_wc:,}")
    print(f"Output files saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
