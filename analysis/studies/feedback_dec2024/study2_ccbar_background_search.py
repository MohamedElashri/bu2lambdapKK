#!/usr/bin/env python3
"""
Study 2: cc̄ Background Search

Search for charm-anticharm (cc̄) pairs in M(Λ̄pK⁻) spectrum to identify:
1. cc̄ from B⁺ decay (B⁺ mass in signal window)
2. cc̄ from other sources - prompt production or non-B background (B⁺ mass outside signal)

Strategy:
- Plot M(Λ̄pK⁻) in full range covering all cc̄ resonances (2800-4000 MeV)
- Separate events by B⁺ mass:
  * B⁺ signal window: [5255, 5305] MeV → cc̄ from B⁺ decay
  * B⁺ sidebands: M(B⁺) < 5200 or > 5350 MeV → cc̄ NOT from B⁺ decay
- Mark known cc̄ resonances: J/ψ (3097), ψ(2S) (3686), χc0 (3415), χc1 (3511), χc2 (3556), ηc (2984)

Output:
- 4 plots showing M(Λ̄pK⁻) distributions:
  1. B⁺ signal window - full spectrum (cc̄ from B⁺ decay)
  2. B⁺ sidebands - full spectrum (cc̄ not from B⁺)
  3. LEFT sideband region (M < 3000 MeV)
  4. RIGHT sideband region (M > 3700 MeV)

Usage:
    cd analysis/studies/feedback_dec2024
    python study2_ccbar_background_search.py
"""

import argparse
import sys
from pathlib import Path

import ROOT

# Add analysis directory to path
SCRIPT_DIR: Path = Path(__file__).parent
ANALYSIS_DIR: Path = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ANALYSIS_DIR))

from modules.data_handler import TOMLConfig  # noqa: E402

# Disable ROOT GUI and warnings
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

# Output directory
OUTPUT_DIR: Path = SCRIPT_DIR / "output" / "study2"

# Known cc̄ resonances (for reference lines on plots)
# Each resonance has (mass, color)
CC_RESONANCES: dict[str, tuple[float, int]] = {
    "J/#psi": (3096.9, ROOT.kRed),
    "#chi_{c0}": (3414.7, ROOT.kBlue),
    "#chi_{c1}": (3510.7, ROOT.kGreen + 2),
    "#chi_{c2}": (3556.2, ROOT.kMagenta),
    "#eta_{c}": (2983.9, ROOT.kOrange + 7),
}


def setup_lhcb_style() -> None:
    """Set up LHCb plotting style."""
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
    ROOT.gStyle.SetLineWidth(2)
    ROOT.gStyle.SetFrameLineWidth(2)


def load_all_trees_data(
    data_path: Path,
    years: list[str],
    track_types: list[str],
) -> tuple[list[ROOT.TTree], list[ROOT.TFile]]:
    """
    Load and cache all Data trees.

    Args:
        data_path: Path to Data directory
        years: List of years
        track_types: List of track types

    Returns:
        Tuple of (list of cached TTree objects, list of TFile objects to keep alive)
    """
    trees: list[ROOT.TTree] = []
    tfiles: list[ROOT.TFile] = []  # Keep files alive
    magnets: list[str] = ["MD", "MU"]

    for year in years:
        for magnet in magnets:
            for track_type in track_types:
                # Data files use format: dataBu2L0barPHH_YYXX.root (e.g., dataBu2L0barPHH_16MD.root)
                year_short: str = year[-2:]  # "2016" -> "16"
                filename: str = f"dataBu2L0barPHH_{year_short}{magnet}.root"
                filepath: Path = data_path / filename
                treename: str = f"B2L0barPKpKm_{track_type}/DecayTree"

                if not filepath.exists():
                    print(f"    Skipping {filename} (not found)")
                    continue

                tfile: ROOT.TFile = ROOT.TFile.Open(str(filepath), "READ")
                if not tfile or tfile.IsZombie():
                    print(f"    Error opening {filename}")
                    continue

                tree: ROOT.TTree = tfile.Get(treename)
                if not tree:
                    print(f"    Tree {treename} not found in {filename}")
                    continue

                entries: int = tree.GetEntries()
                if entries > 0:
                    trees.append(tree)
                    tfiles.append(tfile)  # Keep file alive
                    print(f"    Loaded {filename}:{treename} ({entries:,} events)")
                else:
                    print(f"    Skipping {filename}:{treename} (0 events)")

    return trees, tfiles


def create_histogram_with_cut(
    trees: list[ROOT.TTree],
    hist_name: str,
    mass_formula: str,
    cut_string: str,
    x_min: float = 2800.0,
    x_max: float = 3900.0,
    n_bins: int = 110,
) -> tuple[ROOT.TH1D, int]:
    """
    Create M(Λ̄pK⁻) histogram from trees with cut.

    Args:
        trees: List of TTree objects
        hist_name: Histogram name
        mass_formula: Mass calculation formula
        cut_string: Cut to apply
        x_min: Minimum mass
        x_max: Maximum mass
        n_bins: Number of bins

    Returns:
        Tuple of (histogram, event_count)
    """
    hist: ROOT.TH1D = ROOT.TH1D(hist_name, "", n_bins, x_min, x_max)
    hist.Sumw2()

    for tree in trees:
        tree.Draw(
            f"({mass_formula}) >> +{hist_name}",
            cut_string,
            "goff",
        )

    n_events: int = int(hist.GetEntries())
    return hist, n_events


def create_plot(
    hist: ROOT.TH1D,
    n_events: int,
    bu_category: str,
    output_path: Path,
) -> None:
    """
    Create and save plot with resonance markers.

    Args:
        hist: Histogram to plot
        n_events: Number of events
        bu_category: "signal" or "sideband"
        output_path: Output PDF path
    """
    keep_alive: list = []

    canvas: ROOT.TCanvas = ROOT.TCanvas("c", "", 900, 700)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.05)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.12)

    # Set histogram style
    hist.SetLineColor(ROOT.kBlue + 1)
    hist.SetLineWidth(2)
    hist.SetFillColorAlpha(ROOT.kBlue - 9, 0.3)
    hist.SetFillStyle(1001)
    hist.GetXaxis().SetTitle("M(#bar{#Lambda}pK^{#minus}) [MeV/#it{c}^{2}]")
    hist.GetYaxis().SetTitle("Candidates / 10 MeV")
    hist.GetXaxis().SetTitleFont(132)
    hist.GetYaxis().SetTitleFont(132)
    hist.GetXaxis().SetLabelFont(132)
    hist.GetYaxis().SetLabelFont(132)
    hist.GetXaxis().SetTitleSize(0.045)
    hist.GetYaxis().SetTitleSize(0.045)
    hist.GetYaxis().SetTitleOffset(1.3)
    hist.Draw("HIST")

    # Get y-axis maximum for vertical lines and set minimum to 0
    y_max: float = hist.GetMaximum() * 1.1
    hist.SetMinimum(0)
    hist.SetMaximum(y_max)

    # Draw vertical lines for known cc̄ resonances
    lines: dict[str, ROOT.TLine] = {}
    labels: list[ROOT.TLatex] = []

    x_min: float = hist.GetXaxis().GetXmin()
    x_max: float = hist.GetXaxis().GetXmax()

    for resonance, (mass, color) in CC_RESONANCES.items():
        if x_min < mass < x_max:
            line: ROOT.TLine = ROOT.TLine(mass, 0, mass, y_max * 0.85)
            line.SetLineColor(color)
            line.SetLineStyle(2)
            line.SetLineWidth(2)
            line.Draw("same")  # Draw on same pad to respect axis limits
            lines[resonance] = line
            keep_alive.append(line)

            # Label (rotated at top of line)
            label: ROOT.TLatex = ROOT.TLatex()
            label.SetTextFont(132)
            label.SetTextSize(0.028)
            label.SetTextColor(color)
            label.SetTextAngle(90)
            label.DrawLatex(mass + 15, y_max * 0.5, resonance)
            labels.append(label)
            keep_alive.append(label)

    # Title
    title: ROOT.TLatex = ROOT.TLatex()
    title.SetNDC()
    title.SetTextFont(132)
    title.SetTextSize(0.045)

    if bu_category == "signal":
        title_text = "B^{+} signal window"
    elif bu_category == "sideband":
        title_text = "B^{+} sidebands"
    elif bu_category == "left":
        title_text = "LEFT sideband region"
    else:  # right
        title_text = "RIGHT sideband region"

    title.DrawLatex(0.14, 0.94, f"LHCb Data: {title_text}")
    keep_alive.append(title)

    # Region requirement
    region_req: ROOT.TLatex = ROOT.TLatex()
    region_req.SetNDC()
    region_req.SetTextFont(132)
    region_req.SetTextSize(0.032)
    region_req.SetTextColor(ROOT.kGray + 2)
    if bu_category == "signal":
        region_req.DrawLatex(
            0.14, 0.89, "M(B^{+}) #in [5255, 5305] MeV (c#bar{c} from B^{+} decay)"
        )
    elif bu_category == "sideband":
        region_req.DrawLatex(
            0.14, 0.89, "M(B^{+}) #notin [5200, 5350] MeV (c#bar{c} not from B^{+})"
        )
    elif bu_category == "left":
        region_req.DrawLatex(0.14, 0.89, "M(#bar{#Lambda}pK^{#minus}) < 3000 MeV")
    else:  # right
        region_req.DrawLatex(0.14, 0.89, "M(#bar{#Lambda}pK^{#minus}) > 3700 MeV")
    keep_alive.append(region_req)

    # Event count
    info: ROOT.TLatex = ROOT.TLatex()
    info.SetNDC()
    info.SetTextFont(132)
    info.SetTextSize(0.035)
    info.DrawLatex(0.65, 0.78, f"Events: {n_events:,}")
    keep_alive.append(info)

    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(output_path))
    canvas.SaveAs(str(output_path.with_suffix(".png")))
    canvas.Close()


def main() -> None:
    """Main function."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Study 2: cc̄ background search in sideband regions"
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2016,2017,2018",
        help="Comma-separated years",
    )
    parser.add_argument(
        "--track-types",
        type=str,
        default="LL,DD",
        help="Comma-separated track types",
    )
    args: argparse.Namespace = parser.parse_args()

    years: list[str] = [y.strip() for y in args.years.split(",")]
    track_types: list[str] = [t.strip() for t in args.track_types.split(",")]

    print("=" * 80)
    print("STUDY 2: cc̄ Search in M(Λ̄pK⁻) Spectrum")
    print("=" * 80)
    print(f"Years: {years}")
    print(f"Track types: {track_types}")
    print("=" * 80)

    setup_lhcb_style()

    # Load configuration
    config: TOMLConfig = TOMLConfig(config_dir=str(ANALYSIS_DIR / "config"))
    data_path: Path = Path(config.paths["data"]["base_path"])

    # B⁺ mass windows
    bu_signal_min: float = 5255.0
    bu_signal_max: float = 5305.0
    bu_sideband_low: float = 5200.0
    bu_sideband_high: float = 5350.0

    print("\nB⁺ mass windows:")
    print(f"  B⁺ signal: [{bu_signal_min:.0f}, {bu_signal_max:.0f}] MeV")
    print(f"  B⁺ sidebands: M(B⁺) < {bu_sideband_low:.0f} or M(B⁺) > {bu_sideband_high:.0f} MeV")
    print("\nSearching for cc̄ resonances in M(Λ̄pK⁻) spectrum [2800, 4000] MeV\n")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Mass formula
    mass_formula: str = (
        "sqrt((L0_PE + p_PE + h2_PE)*(L0_PE + p_PE + h2_PE) - "
        "(L0_PX + p_PX + h2_PX)*(L0_PX + p_PX + h2_PX) - "
        "(L0_PY + p_PY + h2_PY)*(L0_PY + p_PY + h2_PY) - "
        "(L0_PZ + p_PZ + h2_PZ)*(L0_PZ + p_PZ + h2_PZ))"
    )

    # B⁺ mass variable (use Bu_MM for Data - the mass-constrained fit variable)
    bu_mass_var: str = "Bu_MM"

    # Load Data trees
    print("Loading Data trees...")
    trees_data, tfiles = load_all_trees_data(data_path, years, track_types)
    print(f"  Total loaded: {len(trees_data)} trees\n")

    # ========================================
    # 1. B⁺ signal window (cc̄ from B⁺ decay)
    # ========================================
    print("=" * 60)
    print("[1/2] B⁺ signal window (cc̄ from B⁺ decay)")
    print("=" * 60)

    cut_bu_signal: str = f"{bu_mass_var} > {bu_signal_min} && {bu_mass_var} < {bu_signal_max}"

    hist_bu_signal, n_bu_signal = create_histogram_with_cut(
        trees_data,
        "h_bu_signal",
        mass_formula,
        cut_bu_signal,
        x_min=2800.0,
        x_max=4000.0,
        n_bins=120,
    )

    print(f"  Events: {n_bu_signal:,}")

    output_bu_signal: Path = OUTPUT_DIR / "mlpk_ccbar_from_bu_decay.pdf"
    create_plot(hist_bu_signal, n_bu_signal, "signal", output_bu_signal)
    print(f"  Saved: {output_bu_signal.name}\n")

    # ========================================
    # 2. B⁺ sidebands (cc̄ NOT from B⁺ decay)
    # ========================================
    print("=" * 60)
    print("[2/2] B⁺ sidebands (cc̄ not from B⁺ decay)")
    print("=" * 60)

    cut_bu_sideband: str = (
        f"({bu_mass_var} < {bu_sideband_low} || {bu_mass_var} > {bu_sideband_high})"
    )

    hist_bu_sideband, n_bu_sideband = create_histogram_with_cut(
        trees_data,
        "h_bu_sideband",
        mass_formula,
        cut_bu_sideband,
        x_min=2800.0,
        x_max=4000.0,
        n_bins=120,
    )

    print(f"  Events: {n_bu_sideband:,}")

    output_bu_sideband: Path = OUTPUT_DIR / "mlpk_ccbar_not_from_bu.pdf"
    create_plot(hist_bu_sideband, n_bu_sideband, "sideband", output_bu_sideband)
    print(f"  Saved: {output_bu_sideband.name}\n")

    # ========================================
    # 3. LEFT sideband region (M < 3000 MeV, B+ sidebands)
    # ========================================
    print("=" * 60)
    print("[3/4] LEFT sideband region (M < 3000 MeV, B+ sidebands)")
    print("=" * 60)

    cut_left: str = (
        f"({mass_formula}) < 3000 && "
        f"({bu_mass_var} < {bu_sideband_low} || {bu_mass_var} > {bu_sideband_high})"
    )

    hist_left, n_left = create_histogram_with_cut(
        trees_data,
        "h_left",
        mass_formula,
        cut_left,
        x_min=2800.0,
        x_max=3000.0,
        n_bins=20,
    )

    print(f"  Events: {n_left:,}")

    output_left: Path = OUTPUT_DIR / "mlpk_left_sideband.pdf"
    create_plot(hist_left, n_left, "left", output_left)
    print(f"  Saved: {output_left.name}\n")

    # ========================================
    # 4. RIGHT sideband region (M > 3700 MeV, B+ sidebands)
    # ========================================
    print("=" * 60)
    print("[4/4] RIGHT sideband region (M > 3700 MeV, B+ sidebands)")
    print("=" * 60)

    cut_right: str = (
        f"({mass_formula}) > 3700 && "
        f"({bu_mass_var} < {bu_sideband_low} || {bu_mass_var} > {bu_sideband_high})"
    )

    hist_right, n_right = create_histogram_with_cut(
        trees_data,
        "h_right",
        mass_formula,
        cut_right,
        x_min=3700.0,
        x_max=4000.0,
        n_bins=30,
    )

    print(f"  Events: {n_right:,}")

    output_right: Path = OUTPUT_DIR / "mlpk_right_sideband.pdf"
    create_plot(hist_right, n_right, "right", output_right)
    print(f"  Saved: {output_right.name}\n")

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Full spectrum:")
    print(f"  B⁺ signal window:  {n_bu_signal:>9,} events (cc̄ from B⁺ decay)")
    print(f"  B⁺ sidebands:      {n_bu_sideband:>9,} events (cc̄ not from B⁺)")
    print("\nSideband regions:")
    print(f"  LEFT (M < 3000):   {n_left:>9,} events")
    print(f"  RIGHT (M > 3700):  {n_right:>9,} events")
    print("\nGenerated 4 plots showing cc̄ search in M(Λ̄pK⁻) spectrum")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
