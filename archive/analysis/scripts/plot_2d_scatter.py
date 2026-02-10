#!/usr/bin/env python3
"""
2D Scatter Plot: M(ΛpK⁻) vs M(B+) with Linear Scale.

Creates a scatter plot (not histogram) showing the correlation between:
- X-axis: M(B+) in range [5100, 5500] MeV
- Y-axis: M(ΛpK⁻) in range [2800, 4800] MeV

Usage:
    cd analysis/scripts
    python plot_2d_scatter.py
    python plot_2d_scatter.py --years 2016,2017,2018
    python plot_2d_scatter.py --max-points 50000
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

# Default mass ranges (from Task 6/7 requirements)
M_BU_MIN: float = 5100.0
M_BU_MAX: float = 5500.0
M_LPKM_MIN: float = 2800.0
M_LPKM_MAX: float = 4800.0

# Regions of interest for background study
REGIONS: dict[str, dict[str, float]] = {
    "full": {
        "x_min": 5100.0,
        "x_max": 5500.0,
        "y_min": 2800.0,
        "y_max": 4800.0,
        "label": "Full region",
    },
    "high_mass": {
        "x_min": 5350.0,
        "x_max": 5500.0,
        "y_min": 3700.0,
        "y_max": 4800.0,
        "label": "High mass (background)",
    },
    "signal_region": {
        "x_min": 5200.0,
        "x_max": 5350.0,
        "y_min": 2800.0,
        "y_max": 3700.0,
        "label": "Signal region",
    },
    "sideband_high_bu": {
        "x_min": 5350.0,
        "x_max": 5500.0,
        "y_min": 2800.0,
        "y_max": 3700.0,
        "label": "High M(B+) sideband",
    },
}

# Mass formula for M(ΛpK⁻) using L0, p, h2 branches
MASS_FORMULA_LPKM: str = (
    "sqrt((L0_PE + p_PE + h2_PE)*(L0_PE + p_PE + h2_PE) - "
    "(L0_PX + p_PX + h2_PX)*(L0_PX + p_PX + h2_PX) - "
    "(L0_PY + p_PY + h2_PY)*(L0_PY + p_PY + h2_PY) - "
    "(L0_PZ + p_PZ + h2_PZ)*(L0_PZ + p_PZ + h2_PZ))"
)


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
        if isinstance(cut_spec, dict):
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


def load_scatter_data(
    data_path: Path,
    years: list[str],
    track_types: list[str],
    manual_cuts: dict[str, Any] | None,
    x_min: float = M_BU_MIN,
    x_max: float = M_BU_MAX,
    y_min: float = M_LPKM_MIN,
    y_max: float = M_LPKM_MAX,
    hist_name: str = "h2d_scatter",
) -> tuple[ROOT.TH2D, int]:
    """
    Load data into a 2D histogram for scatter plot extraction.

    Args:
        data_path: Base path to data files
        years: List of years
        track_types: List of track types
        manual_cuts: Selection cuts dictionary
        x_min: Minimum M(B+) value
        x_max: Maximum M(B+) value
        y_min: Minimum M(ΛpK⁻) value
        y_max: Maximum M(ΛpK⁻) value
        hist_name: Name for histogram

    Returns:
        Tuple of (2D histogram, total_events)
    """
    # Create 2D histogram with fine binning for scatter-like appearance
    n_bins_x: int = max(50, int((x_max - x_min) / 2))
    n_bins_y: int = max(100, int((y_max - y_min) / 4))
    hist: ROOT.TH2D = ROOT.TH2D(hist_name, "", n_bins_x, x_min, x_max, n_bins_y, y_min, y_max)
    hist.Sumw2()
    ROOT.SetOwnership(hist, False)
    total_events: int = 0
    # Build cut string
    cut_str: str = build_cut_string(manual_cuts)
    mass_range_cuts: list[str] = [
        f"(Bu_MM > {x_min})",
        f"(Bu_MM < {x_max})",
    ]
    if cut_str:
        cut_str = cut_str + " && " + " && ".join(mass_range_cuts)
    else:
        cut_str = " && ".join(mass_range_cuts)
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
                    # 2D draw: Y:X format
                    draw_expr: str = f"{MASS_FORMULA_LPKM}:Bu_MM"
                    temp_hist_name: str = f"temp_{hist_name}_{year}_{magnet}_{track_type}"
                    n_entries: int = tree.Draw(
                        f"{draw_expr}>>{temp_hist_name}({n_bins_x},{x_min},{x_max},{n_bins_y},{y_min},{y_max})",
                        cut_str,
                        "goff",
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
                    print(f"    Error: {e}")
    return hist, total_events


def create_scatter_plot(
    hist: ROOT.TH2D,
    n_events: int,
    output_path: Path,
    with_cuts: bool,
    x_min: float = M_BU_MIN,
    x_max: float = M_BU_MAX,
    y_min: float = M_LPKM_MIN,
    y_max: float = M_LPKM_MAX,
    region_label: str = "",
) -> None:
    """
    Create scatter-style plot from 2D histogram.

    Args:
        hist: 2D histogram with data
        n_events: Total number of events
        output_path: Path to save PDF
        with_cuts: Whether cuts were applied
        x_min: Minimum M(B+) value
        x_max: Maximum M(B+) value
        y_min: Minimum M(ΛpK⁻) value
        y_max: Maximum M(ΛpK⁻) value
        region_label: Label for the region
    """
    keep_alive: list = []
    canvas: ROOT.TCanvas = ROOT.TCanvas("c_scatter", "", 1000, 800)
    ROOT.SetOwnership(canvas, False)
    keep_alive.append(canvas)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.05)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.12)
    # Style histogram for scatter-like appearance
    hist.GetXaxis().SetTitle("M(B^{+}) [MeV/#it{c}^{2}]")
    hist.GetYaxis().SetTitle("M(#bar{#Lambda}pK^{#minus}) [MeV/#it{c}^{2}]")
    hist.GetXaxis().SetTitleFont(132)
    hist.GetYaxis().SetTitleFont(132)
    hist.GetXaxis().SetLabelFont(132)
    hist.GetYaxis().SetLabelFont(132)
    hist.GetXaxis().SetTitleSize(0.045)
    hist.GetYaxis().SetTitleSize(0.045)
    hist.GetYaxis().SetTitleOffset(1.3)
    # Draw as scatter plot (SCAT option)
    hist.SetMarkerStyle(6)
    hist.SetMarkerColor(ROOT.kBlue)
    hist.SetMarkerSize(0.3)
    hist.Draw("SCAT")
    # Add B+ mass line if in range
    bu_mass: float = 5279.34
    if x_min < bu_mass < x_max:
        bu_line: ROOT.TLine = ROOT.TLine(bu_mass, y_min, bu_mass, y_max)
        ROOT.SetOwnership(bu_line, False)
        keep_alive.append(bu_line)
        bu_line.SetLineColor(ROOT.kRed)
        bu_line.SetLineWidth(2)
        bu_line.SetLineStyle(2)
        bu_line.Draw()
    # Add charmonium mass lines if in range
    charmonium_masses: dict[str, float] = {
        "J/#psi": 3096.9,
        "#eta_{c}": 2983.9,
        "#chi_{c0}": 3414.7,
        "#chi_{c1}": 3510.7,
    }
    for name, mass in charmonium_masses.items():
        if y_min < mass < y_max:
            line: ROOT.TLine = ROOT.TLine(x_min, mass, x_max, mass)
            ROOT.SetOwnership(line, False)
            keep_alive.append(line)
            line.SetLineColor(ROOT.kGray + 1)
            line.SetLineWidth(1)
            line.SetLineStyle(3)
            line.Draw()
    # Labels
    lhcb: ROOT.TLatex = ROOT.TLatex()
    lhcb.SetNDC()
    lhcb.SetTextFont(132)
    lhcb.SetTextSize(0.05)
    lhcb.DrawLatex(0.15, 0.85, "LHCb Data")
    keep_alive.append(lhcb)
    info: ROOT.TLatex = ROOT.TLatex()
    info.SetNDC()
    info.SetTextFont(132)
    info.SetTextSize(0.035)
    cuts_label: str = "With selection cuts" if with_cuts else "No cuts"
    info.DrawLatex(0.15, 0.78, cuts_label)
    info.DrawLatex(0.15, 0.73, f"N = {n_events:,}")
    if region_label:
        info.DrawLatex(0.15, 0.68, region_label)
    keep_alive.append(info)
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(output_path))
    canvas.SaveAs(str(output_path.with_suffix(".png")))
    print(f"  Saved: {output_path.name}")


def create_2d_histogram_linear(
    hist: ROOT.TH2D,
    n_events: int,
    output_path: Path,
    with_cuts: bool,
    x_min: float = M_BU_MIN,
    x_max: float = M_BU_MAX,
    y_min: float = M_LPKM_MIN,
    y_max: float = M_LPKM_MAX,
    region_label: str = "",
) -> None:
    """
    Create 2D histogram with linear Z scale.

    Args:
        hist: 2D histogram with data
        n_events: Total number of events
        output_path: Path to save PDF
        with_cuts: Whether cuts were applied
        x_min: Minimum M(B+) value
        x_max: Maximum M(B+) value
        y_min: Minimum M(ΛpK⁻) value
        y_max: Maximum M(ΛpK⁻) value
        region_label: Label for the region
    """
    keep_alive: list = []
    canvas: ROOT.TCanvas = ROOT.TCanvas("c_hist2d", "", 1100, 800)
    ROOT.SetOwnership(canvas, False)
    keep_alive.append(canvas)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.15)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.12)
    # Rebin for better visualization (adjust based on region size)
    rebin_x: int = max(1, int(hist.GetNbinsX() / 40))
    rebin_y: int = max(1, int(hist.GetNbinsY() / 50))
    hist_rebinned: ROOT.TH2D = hist.Rebin2D(rebin_x, rebin_y, f"h2d_rebinned_{output_path.stem}")
    ROOT.SetOwnership(hist_rebinned, False)
    keep_alive.append(hist_rebinned)
    # Style
    hist_rebinned.GetXaxis().SetTitle("M(B^{+}) [MeV/#it{c}^{2}]")
    hist_rebinned.GetYaxis().SetTitle("M(#bar{#Lambda}pK^{#minus}) [MeV/#it{c}^{2}]")
    hist_rebinned.GetZaxis().SetTitle("Candidates")
    hist_rebinned.GetXaxis().SetTitleFont(132)
    hist_rebinned.GetYaxis().SetTitleFont(132)
    hist_rebinned.GetZaxis().SetTitleFont(132)
    hist_rebinned.GetXaxis().SetLabelFont(132)
    hist_rebinned.GetYaxis().SetLabelFont(132)
    hist_rebinned.GetZaxis().SetLabelFont(132)
    hist_rebinned.GetXaxis().SetTitleSize(0.045)
    hist_rebinned.GetYaxis().SetTitleSize(0.045)
    hist_rebinned.GetZaxis().SetTitleSize(0.040)
    hist_rebinned.GetYaxis().SetTitleOffset(1.2)
    hist_rebinned.GetZaxis().SetTitleOffset(1.2)
    # Color palette - linear scale
    ROOT.gStyle.SetPalette(ROOT.kBird)
    hist_rebinned.Draw("COLZ")
    # Add B+ mass line if in range
    bu_mass: float = 5279.34
    if x_min < bu_mass < x_max:
        bu_line: ROOT.TLine = ROOT.TLine(bu_mass, y_min, bu_mass, y_max)
        ROOT.SetOwnership(bu_line, False)
        keep_alive.append(bu_line)
        bu_line.SetLineColor(ROOT.kRed)
        bu_line.SetLineWidth(2)
        bu_line.SetLineStyle(2)
        bu_line.Draw()
    # Labels
    lhcb: ROOT.TLatex = ROOT.TLatex()
    lhcb.SetNDC()
    lhcb.SetTextFont(132)
    lhcb.SetTextSize(0.05)
    lhcb.DrawLatex(0.15, 0.85, "LHCb Data")
    keep_alive.append(lhcb)
    info: ROOT.TLatex = ROOT.TLatex()
    info.SetNDC()
    info.SetTextFont(132)
    info.SetTextSize(0.035)
    cuts_label: str = "With selection cuts" if with_cuts else "No cuts"
    info.DrawLatex(0.15, 0.78, cuts_label)
    info.DrawLatex(0.15, 0.73, f"N = {n_events:,}")
    if region_label:
        info.DrawLatex(0.15, 0.68, region_label)
    keep_alive.append(info)
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(output_path))
    canvas.SaveAs(str(output_path.with_suffix(".png")))
    print(f"  Saved: {output_path.name}")


def main() -> None:
    """Main function."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Create 2D scatter plot of M(ΛpK⁻) vs M(B+) with linear scale"
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
    parser.add_argument(
        "--no-cuts",
        action="store_true",
        help="Skip selection cuts",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="all",
        help="Region to plot: 'all', 'full', 'high_mass', 'signal_region', 'sideband_high_bu' (default: all)",
    )
    args: argparse.Namespace = parser.parse_args()
    years: list[str] = [y.strip() for y in args.years.split(",")]
    track_types: list[str] = [t.strip() for t in args.track_types.split(",")]
    print("=" * 80)
    print("2D SCATTER PLOT: M(ΛpK⁻) vs M(B+)")
    print("Linear scale, scatter plot style")
    print("=" * 80)
    print(f"Years: {years}")
    print(f"Track types: {track_types}")
    print("=" * 80)
    setup_lhcb_style()
    config: TOMLConfig = TOMLConfig(config_dir=str(ANALYSIS_DIR / "config"))
    data_path: Path = Path(config.paths["data"]["base_path"])
    # Get manual cuts
    manual_cuts: dict[str, Any] | None = None
    if not args.no_cuts:
        manual_cuts = config.selection.get("manual_cuts", {})
        cut_str: str = build_cut_string(manual_cuts)
        print("\nSelection cuts:")
        for cut in cut_str.split(" && "):
            if cut.strip():
                print(f"  {cut}")
    else:
        print("\nNo selection cuts applied")
    output_dir: Path = ANALYSIS_DIR / "plots" / "2d_scatter"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Determine which regions to process
    if args.region == "all":
        regions_to_process: list[str] = list(REGIONS.keys())
    else:
        regions_to_process = [args.region]
    generated_files: list[str] = []
    # Process each region
    for region_name in regions_to_process:
        if region_name not in REGIONS:
            print(f"Warning: Unknown region '{region_name}', skipping")
            continue
        region: dict[str, Any] = REGIONS[region_name]
        x_min: float = region["x_min"]
        x_max: float = region["x_max"]
        y_min: float = region["y_min"]
        y_max: float = region["y_max"]
        label: str = region["label"]
        print(f"\n{'=' * 60}")
        print(f"Processing region: {label}")
        print(f"  M(B+):   [{x_min:.0f}, {x_max:.0f}] MeV")
        print(f"  M(ΛpK⁻): [{y_min:.0f}, {y_max:.0f}] MeV")
        print("=" * 60)
        # Load data for this region
        print("Loading data...")
        hist, n_events = load_scatter_data(
            data_path,
            years,
            track_types,
            manual_cuts,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            hist_name=f"h2d_{region_name}",
        )
        print(f"Loaded {n_events:,} events")
        if n_events == 0:
            print("  No events in this region, skipping plots")
            continue
        # Create scatter plot
        print("Creating scatter plot...")
        scatter_pdf: Path = output_dir / f"scatter_{region_name}.pdf"
        create_scatter_plot(
            hist.Clone(f"h_scatter_{region_name}"),
            n_events,
            scatter_pdf,
            not args.no_cuts,
            x_min,
            x_max,
            y_min,
            y_max,
            label,
        )
        generated_files.append(scatter_pdf.name)
        # Create 2D histogram with linear scale
        print("Creating 2D histogram (linear scale)...")
        hist_pdf: Path = output_dir / f"hist2d_{region_name}_linear.pdf"
        create_2d_histogram_linear(
            hist.Clone(f"h_linear_{region_name}"),
            n_events,
            hist_pdf,
            not args.no_cuts,
            x_min,
            x_max,
            y_min,
            y_max,
            label,
        )
        generated_files.append(hist_pdf.name)
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"Generated {len(generated_files)} plots:")
    for f in generated_files:
        print(f"  - {f}")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
