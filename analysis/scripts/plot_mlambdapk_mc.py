#!/usr/bin/env python3
"""
Standalone script to plot M(LambdaPK) distributions for MC using PyROOT.

Creates side-by-side comparison showing all 4 signal states:
- Left: MC without cuts (raw distribution)
- Right: MC with cuts applied

All 4 signal states (Jpsi, etac, chic0, chic1) are overlaid with different colors.
Histograms are scaled by relative yields from the mass fit to data, so the
proportions match what is observed in real data (J/ψ dominant, ηc smaller, χc tiny).

The cuts are defined in config/selection.toml under [manual_cuts].

Output: 4 plots total (one per year + one combined).

Usage:
    cd analysis/scripts
    python plot_mlambdapk_mc.py                    # Use default yield estimates
    python plot_mlambdapk_mc.py --run-fit          # Run mass fit to get actual yields
    python plot_mlambdapk_mc.py --years 2016,2017,2018
    python plot_mlambdapk_mc.py --track-types LL,DD
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

# Signal MC states (charmonium resonances)
SIGNAL_STATES: list[str] = ["Jpsi", "etac", "chic0", "chic1"]

# ROOT colors for each state
STATE_COLORS: dict[str, int] = {
    "Jpsi": ROOT.kBlue,
    "etac": ROOT.kRed,
    "chic0": ROOT.kGreen + 2,
    "chic1": ROOT.kOrange + 1,
}

# LaTeX labels for each state (ROOT format)
STATE_LABELS: dict[str, str] = {
    "Jpsi": "J/#psi",
    "etac": "#eta_{c}",
    "chic0": "#chi_{c0}",
    "chic1": "#chi_{c1}",
}


def get_relative_yields_from_fit(config: Any) -> dict[str, float]:
    """
    Get relative yields by running the mass fitter on data.

    This runs the actual fit to data and extracts the yields,
    then normalizes them relative to J/ψ.

    Args:
        config: TOMLConfig object

    Returns:
        Dictionary of relative yields normalized to J/ψ = 1.0
    """
    import awkward as ak
    from modules.data_handler import DataManager
    from modules.mass_fitter import MassFitter

    print("\n" + "=" * 80)
    print("RUNNING MASS FIT TO EXTRACT RELATIVE YIELDS")
    print("=" * 80)
    # Initialize data manager and mass fitter
    data_manager: DataManager = DataManager(config)
    fitter: MassFitter = MassFitter(config)
    # Load data for all years
    years: list[str] = ["2016", "2017", "2018"]
    data_by_year: dict[str, Any] = {}
    for year in years:
        print(f"\nLoading data for {year}...")
        year_data: list = []
        for magnet in ["MD", "MU"]:
            for track_type in ["LL", "DD"]:
                try:
                    events = data_manager.load_tree(
                        particle_type="data",
                        year=year,
                        magnet=magnet,
                        track_type=track_type,
                        apply_derived_branches=True,
                    )
                    if events is not None and len(events) > 0:
                        year_data.append(events)
                        print(f"    {magnet} {track_type}: {len(events):,} events")
                except Exception as e:
                    print(f"  Warning: Could not load {year} {magnet} {track_type}: {e}")
        if year_data:
            data_by_year[year] = ak.concatenate(year_data)
            print(f"  Total {year}: {len(data_by_year[year]):,} events")
    if not data_by_year:
        print("  ERROR: No data loaded! Using default yields.")
        return DEFAULT_RELATIVE_YIELDS.copy()
    # Run the fit
    fit_results: dict[str, Any] = fitter.perform_fit(data_by_year, fit_combined=True)
    # Extract yields from combined fit
    combined_yields: dict[str, tuple[float, float]] = fit_results["yields"].get("combined", {})
    # Get J/ψ yield as reference
    jpsi_yield: float = combined_yields.get("jpsi", (1.0, 0.0))[0]
    if jpsi_yield <= 0:
        jpsi_yield = 1.0
    # Calculate relative yields
    relative_yields: dict[str, float] = {}
    state_map: dict[str, str] = {
        "Jpsi": "jpsi",
        "etac": "etac",
        "chic0": "chic0",
        "chic1": "chic1",
    }
    print("\n" + "-" * 40)
    print("EXTRACTED YIELDS FROM FIT:")
    print("-" * 40)
    for mc_state, fit_state in state_map.items():
        yield_val: float = combined_yields.get(fit_state, (0.0, 0.0))[0]
        yield_err: float = combined_yields.get(fit_state, (0.0, 0.0))[1]
        rel_yield: float = yield_val / jpsi_yield if jpsi_yield > 0 else 0.0
        relative_yields[mc_state] = rel_yield
        print(
            f"  {mc_state:8s}: N = {yield_val:>8.0f} ± {yield_err:>6.0f}  (rel = {rel_yield:.3f})"
        )
    print("-" * 40)
    return relative_yields, combined_yields


# Default relative yields (used if fit is not run)
# These are placeholder values - actual values come from fit
DEFAULT_RELATIVE_YIELDS: dict[str, float] = {
    "Jpsi": 1.0,
    "etac": 0.35,
    "chic0": 0.08,
    "chic1": 0.05,
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


def load_mc_to_histogram(
    mc_path: Path,
    state: str,
    year: str,
    track_types: list[str],
    hist_name: str,
    manual_cuts: dict[str, Any] | None,
    nbins: int = 240,
    xmin: float = 2800.0,
    xmax: float = 4000.0,
) -> tuple[ROOT.TH1D, int]:
    """
    Load MC data and fill a ROOT histogram with M(LambdaPK).

    Args:
        mc_path: Base path to MC files
        state: MC state name (e.g., "Jpsi")
        year: Year string (e.g., "2016")
        track_types: List of track types
        hist_name: Name for the histogram
        manual_cuts: Dictionary of cuts to apply (None for no cuts)
        nbins: Number of bins
        xmin: Minimum x value
        xmax: Maximum x value

    Returns:
        Tuple of (histogram, n_events)
    """
    hist: ROOT.TH1D = ROOT.TH1D(hist_name, "", nbins, xmin, xmax)
    hist.Sumw2()
    total_events: int = 0
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
                # Build cut string
                cut_str: str = ""
                if manual_cuts:
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
                        cut_str = " && ".join(cut_parts)
                # Calculate M(LpKm) using TTree::Draw with formula
                # M^2 = E^2 - px^2 - py^2 - pz^2
                mass_formula: str = (
                    "sqrt((L0_PE + p_PE + h2_PE)*(L0_PE + p_PE + h2_PE) - "
                    "(L0_PX + p_PX + h2_PX)*(L0_PX + p_PX + h2_PX) - "
                    "(L0_PY + p_PY + h2_PY)*(L0_PY + p_PY + h2_PY) - "
                    "(L0_PZ + p_PZ + h2_PZ)*(L0_PZ + p_PZ + h2_PZ))"
                )
                # Draw to temporary histogram
                temp_hist_name: str = f"temp_{hist_name}_{magnet}_{track_type}"
                n_entries: int = tree.Draw(
                    f"{mass_formula}>>{temp_hist_name}({nbins},{xmin},{xmax})", cut_str, "goff"
                )
                if n_entries > 0:
                    temp_hist: ROOT.TH1D = ROOT.gDirectory.Get(temp_hist_name)
                    if temp_hist:
                        hist.Add(temp_hist)
                        total_events += n_entries
                        temp_hist.Delete()
                tfile.Close()
            except Exception as e:
                print(f"    Error loading {state} {year} {magnet} {track_type}: {e}")
    # Keep histogram in memory
    ROOT.SetOwnership(hist, False)
    return hist, total_events


def create_comparison_plot(
    histograms_no_cuts: dict[str, ROOT.TH1D],
    histograms_with_cuts: dict[str, ROOT.TH1D],
    year_label: str,
    output_path: Path,
    absolute_yields: dict[str, float],
    cut_efficiencies: dict[str, float],
) -> None:
    """
    Create side-by-side comparison plot using ROOT.

    Histograms are normalized to match the actual yields from the mass fit,
    so Y-axis shows "Candidates / 20 MeV" like the mass fit plots.

    Args:
        histograms_no_cuts: Dict of {state: TH1D} without cuts
        histograms_with_cuts: Dict of {state: TH1D} with cuts
        year_label: String like "2016" or "Combined"
        output_path: Path to save the PDF
        absolute_yields: Dict of absolute yields from fit {state: N_events}
        cut_efficiencies: Dict of cut efficiencies {state: efficiency}
    """
    # Keep all ROOT objects alive
    keep_alive: list = []
    # Clone histograms, normalize to unit area, then scale to absolute yields
    hists_no_cuts_norm: dict[str, ROOT.TH1D] = {}
    hists_with_cuts_norm: dict[str, ROOT.TH1D] = {}
    for state in SIGNAL_STATES:
        abs_yield: float = absolute_yields.get(state, 0.0)
        eff: float = cut_efficiencies.get(state, 1.0)
        if state in histograms_no_cuts:
            h: ROOT.TH1D = histograms_no_cuts[state].Clone(f"{state}_nocuts_norm")
            ROOT.SetOwnership(h, False)
            if h.Integral() > 0:
                # Normalize to unit area, then scale to absolute yield
                h.Scale(abs_yield / h.Integral())
            hists_no_cuts_norm[state] = h
            keep_alive.append(h)
        if state in histograms_with_cuts:
            h = histograms_with_cuts[state].Clone(f"{state}_cuts_norm")
            ROOT.SetOwnership(h, False)
            if h.Integral() > 0:
                # Scale to yield after cuts (yield * efficiency)
                h.Scale((abs_yield * eff) / h.Integral())
            hists_with_cuts_norm[state] = h
            keep_alive.append(h)
    # Create canvas
    canvas_name: str = f"c_{year_label.replace('-', '_').replace(' ', '_')}"
    canvas: ROOT.TCanvas = ROOT.TCanvas(canvas_name, "", 1600, 700)
    ROOT.SetOwnership(canvas, False)
    keep_alive.append(canvas)
    # Find global maximum for consistent y-axis
    y_max: float = 0.0
    for state in SIGNAL_STATES:
        if state in hists_no_cuts_norm:
            h_max: float = hists_no_cuts_norm[state].GetMaximum()
            y_max = max(y_max, h_max)
        if state in hists_with_cuts_norm:
            h_max = hists_with_cuts_norm[state].GetMaximum()
            y_max = max(y_max, h_max)
    y_max *= 1.4
    if y_max == 0:
        print("  WARNING: All histograms appear empty!")
        y_max = 100
    # Y-axis title - matches mass fit plots (5 MeV bins)
    y_title: str = "Candidates / (5 MeV/#it{c}^{2})"
    # Create two explicit pads
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
    # Draw left pad (no cuts)
    pad1.cd()
    first_drawn: bool = False
    for state in SIGNAL_STATES:
        if state in hists_no_cuts_norm:
            h = hists_no_cuts_norm[state]
            h.SetLineColor(STATE_COLORS[state])
            h.SetLineWidth(2)
            h.SetFillStyle(0)
            h.SetMaximum(y_max)
            h.SetMinimum(0)
            h.GetXaxis().SetTitle("M(#bar{#Lambda}pK^{#minus}) [MeV/#it{c}^{2}]")
            h.GetYaxis().SetTitle(y_title)
            h.GetXaxis().SetTitleFont(132)
            h.GetYaxis().SetTitleFont(132)
            h.GetXaxis().SetLabelFont(132)
            h.GetYaxis().SetLabelFont(132)
            h.GetXaxis().SetTitleSize(0.05)
            h.GetYaxis().SetTitleSize(0.045)
            h.GetXaxis().SetLabelSize(0.04)
            h.GetYaxis().SetLabelSize(0.04)
            h.GetYaxis().SetTitleOffset(1.3)
            if not first_drawn:
                h.Draw("HIST")
                first_drawn = True
            else:
                h.Draw("HIST SAME")
    # Legend for left pad
    legend1: ROOT.TLegend = ROOT.TLegend(0.62, 0.65, 0.92, 0.88)
    ROOT.SetOwnership(legend1, False)
    keep_alive.append(legend1)
    legend1.SetBorderSize(0)
    legend1.SetFillStyle(0)
    legend1.SetTextFont(132)
    legend1.SetTextSize(0.038)
    for state in SIGNAL_STATES:
        if state in hists_no_cuts_norm:
            legend1.AddEntry(hists_no_cuts_norm[state], STATE_LABELS[state], "l")
    legend1.Draw()
    # Labels for left pad
    lhcb1: ROOT.TLatex = ROOT.TLatex()
    lhcb1.SetNDC()
    lhcb1.SetTextFont(132)
    lhcb1.SetTextSize(0.055)
    lhcb1.DrawLatex(0.15, 0.85, "LHCb MC")
    year1: ROOT.TLatex = ROOT.TLatex()
    year1.SetNDC()
    year1.SetTextFont(132)
    year1.SetTextSize(0.045)
    year1.DrawLatex(0.15, 0.78, year_label)
    title1: ROOT.TLatex = ROOT.TLatex()
    title1.SetNDC()
    title1.SetTextFont(132)
    title1.SetTextSize(0.04)
    title1.DrawLatex(0.15, 0.71, "No Cuts")
    keep_alive.extend([lhcb1, year1, title1])
    pad1.Modified()
    pad1.Update()
    # Draw right pad (with cuts)
    pad2.cd()
    first_drawn = False
    for state in SIGNAL_STATES:
        if state in hists_with_cuts_norm:
            h = hists_with_cuts_norm[state]
            h.SetLineColor(STATE_COLORS[state])
            h.SetLineWidth(2)
            h.SetFillStyle(0)
            h.SetMaximum(y_max)
            h.SetMinimum(0)
            h.GetXaxis().SetTitle("M(#bar{#Lambda}pK^{#minus}) [MeV/#it{c}^{2}]")
            h.GetYaxis().SetTitle(y_title)
            h.GetXaxis().SetTitleFont(132)
            h.GetYaxis().SetTitleFont(132)
            h.GetXaxis().SetLabelFont(132)
            h.GetYaxis().SetLabelFont(132)
            h.GetXaxis().SetTitleSize(0.05)
            h.GetYaxis().SetTitleSize(0.045)
            h.GetXaxis().SetLabelSize(0.04)
            h.GetYaxis().SetLabelSize(0.04)
            h.GetYaxis().SetTitleOffset(1.3)
            if not first_drawn:
                h.Draw("HIST")
                first_drawn = True
            else:
                h.Draw("HIST SAME")
    # Legend for right pad
    legend2: ROOT.TLegend = ROOT.TLegend(0.62, 0.65, 0.92, 0.88)
    ROOT.SetOwnership(legend2, False)
    keep_alive.append(legend2)
    legend2.SetBorderSize(0)
    legend2.SetFillStyle(0)
    legend2.SetTextFont(132)
    legend2.SetTextSize(0.038)
    for state in SIGNAL_STATES:
        if state in hists_with_cuts_norm:
            legend2.AddEntry(hists_with_cuts_norm[state], STATE_LABELS[state], "l")
    legend2.Draw()
    # Labels for right pad
    lhcb2: ROOT.TLatex = ROOT.TLatex()
    lhcb2.SetNDC()
    lhcb2.SetTextFont(132)
    lhcb2.SetTextSize(0.055)
    lhcb2.DrawLatex(0.15, 0.85, "LHCb MC")
    year2: ROOT.TLatex = ROOT.TLatex()
    year2.SetNDC()
    year2.SetTextFont(132)
    year2.SetTextSize(0.045)
    year2.DrawLatex(0.15, 0.78, year_label)
    title2: ROOT.TLatex = ROOT.TLatex()
    title2.SetNDC()
    title2.SetTextFont(132)
    title2.SetTextSize(0.04)
    title2.DrawLatex(0.15, 0.71, "With Cuts")
    keep_alive.extend([lhcb2, year2, title2])
    pad2.Modified()
    pad2.Update()
    # Update canvas and save
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(output_path))
    # Also save PNG
    png_path: Path = output_path.with_suffix(".png")
    canvas.SaveAs(str(png_path))
    print(f"  ✓ Saved: {output_path.name}")


def main() -> None:
    """Main function to run the M(LambdaPK) MC plotting script."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Plot M(LambdaPK) MC distributions without and with cuts"
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
    parser.add_argument(
        "--run-fit",
        action="store_true",
        help="Run mass fit on data to extract relative yields (slower but accurate)",
    )
    args: argparse.Namespace = parser.parse_args()
    years: list[str] = [y.strip() for y in args.years.split(",")]
    track_types: list[str] = [t.strip() for t in args.track_types.split(",")]
    print("=" * 80)
    print("M(LambdaPK) MC DISTRIBUTION PLOTTER (PyROOT)")
    print("All Signal States: No Cuts vs With Cuts")
    print("=" * 80)
    print(f"Years: {years}")
    print(f"Signal states: {SIGNAL_STATES}")
    print(f"Track types: {track_types}")
    print(f"Run fit for yields: {args.run_fit}")
    print("=" * 80)
    print()
    # Set up ROOT style
    setup_lhcb_style()
    # Initialize configuration
    config: TOMLConfig = TOMLConfig(config_dir=str(ANALYSIS_DIR / "config"))
    # Get yields - either from fit or use defaults
    absolute_yields: dict[str, float] = {}
    state_map: dict[str, str] = {"Jpsi": "jpsi", "etac": "etac", "chic0": "chic0", "chic1": "chic1"}
    if args.run_fit:
        relative_yields, fit_yields = get_relative_yields_from_fit(config)
        # Extract absolute yields from fit
        for mc_state, fit_state in state_map.items():
            absolute_yields[mc_state] = fit_yields.get(fit_state, (0.0, 0.0))[0]
    else:
        relative_yields = DEFAULT_RELATIVE_YIELDS.copy()
        # Use default absolute yields (approximate)
        absolute_yields = {"Jpsi": 400, "etac": 1300, "chic0": 100, "chic1": 60}
        print("Using default yields (run with --run-fit for actual values from data):")
    print("\nYields from fit (used for MC scaling):")
    for state in SIGNAL_STATES:
        print(
            f"  {STATE_LABELS[state]:12s}: N = {absolute_yields[state]:>6.0f}  (rel = {relative_yields[state]:.3f})"
        )
    print()
    mc_path: Path = Path(config.paths["mc"]["base_path"])
    # Get cuts from config
    manual_cuts: dict[str, Any] = config.selection.get("manual_cuts", {})
    has_manual_cuts: bool = any(k for k in manual_cuts.keys() if k != "notes")
    if not has_manual_cuts:
        print("⚠️  WARNING: No cuts defined in config/selection.toml [manual_cuts]")
        print("    Right panel will show same distribution as left panel.")
    else:
        print("Cuts from config/selection.toml:")
        for branch_name, cut_spec in manual_cuts.items():
            if branch_name == "notes":
                continue
            cut_type = cut_spec.get("cut_type")
            cut_value = cut_spec.get("value")
            if cut_type and cut_value is not None:
                op: str = ">" if cut_type == "greater" else "<"
                print(f"  {branch_name:20s} {op} {cut_value}")
        print()
    # Output directory
    output_dir: Path = ANALYSIS_DIR / "plots" / "mlambdapk_mc"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Storage for combined histograms
    combined_hists_no_cuts: dict[str, ROOT.TH1D] = {}
    combined_hists_with_cuts: dict[str, ROOT.TH1D] = {}
    combined_events_no_cuts: dict[str, int] = {s: 0 for s in SIGNAL_STATES}
    combined_events_with_cuts: dict[str, int] = {s: 0 for s in SIGNAL_STATES}
    # Process each year
    for year in years:
        print(f"\n{'=' * 80}")
        print(f"YEAR {year}")
        print(f"{'=' * 80}")
        print("\nLoading MC data...")
        # Load histograms for each state
        hists_no_cuts: dict[str, ROOT.TH1D] = {}
        hists_with_cuts: dict[str, ROOT.TH1D] = {}
        events_no_cuts: dict[str, int] = {}
        events_with_cuts: dict[str, int] = {}
        for state in SIGNAL_STATES:
            # No cuts
            h_no, n_no = load_mc_to_histogram(
                mc_path, state, year, track_types, f"h_{state}_{year}_nocuts", None
            )
            hists_no_cuts[state] = h_no
            events_no_cuts[state] = n_no
            # With cuts
            h_cut, n_cut = load_mc_to_histogram(
                mc_path, state, year, track_types, f"h_{state}_{year}_cuts", manual_cuts
            )
            hists_with_cuts[state] = h_cut
            events_with_cuts[state] = n_cut
            # Add to combined
            if state not in combined_hists_no_cuts:
                combined_hists_no_cuts[state] = h_no.Clone(f"h_{state}_combined_nocuts")
                combined_hists_with_cuts[state] = h_cut.Clone(f"h_{state}_combined_cuts")
            else:
                combined_hists_no_cuts[state].Add(h_no)
                combined_hists_with_cuts[state].Add(h_cut)
            combined_events_no_cuts[state] += n_no
            combined_events_with_cuts[state] += n_cut
            # Print summary
            eff: float = 100.0 * n_cut / n_no if n_no > 0 else 0.0
            print(f"  {STATE_LABELS[state]:12s}: {n_no:>8,} -> {n_cut:>8,} ({eff:.1f}%)")
        # Calculate cut efficiencies for this year
        cut_effs: dict[str, float] = {}
        for state in SIGNAL_STATES:
            n_no: int = events_no_cuts.get(state, 0)
            n_cut: int = events_with_cuts.get(state, 0)
            cut_effs[state] = n_cut / n_no if n_no > 0 else 1.0
        # Create plot for this year
        print(f"\nCreating plot for {year}...")
        year_pdf: Path = output_dir / f"mlambdapk_mc_all_states_{year}.pdf"
        create_comparison_plot(
            hists_no_cuts, hists_with_cuts, year, year_pdf, absolute_yields, cut_effs
        )
    # Create combined plot (all years)
    print(f"\n{'=' * 80}")
    print("COMBINED (All Years)")
    print(f"{'=' * 80}")
    print("\nCombined events:")
    for state in SIGNAL_STATES:
        n_no: int = combined_events_no_cuts[state]
        n_cut: int = combined_events_with_cuts[state]
        eff: float = 100.0 * n_cut / n_no if n_no > 0 else 0.0
        print(f"  {STATE_LABELS[state]:12s}: {n_no:>8,} -> {n_cut:>8,} ({eff:.1f}%)")
    # Calculate combined cut efficiencies
    combined_cut_effs: dict[str, float] = {}
    for state in SIGNAL_STATES:
        n_no: int = combined_events_no_cuts[state]
        n_cut: int = combined_events_with_cuts[state]
        combined_cut_effs[state] = n_cut / n_no if n_no > 0 else 1.0
    print("\nCreating combined plot...")
    combined_pdf: Path = output_dir / "mlambdapk_mc_all_states_combined.pdf"
    create_comparison_plot(
        combined_hists_no_cuts,
        combined_hists_with_cuts,
        "2016-2018",
        combined_pdf,
        absolute_yields,
        combined_cut_effs,
    )
    # Print cut efficiencies as markdown table
    print(f"\n{'=' * 80}")
    print("CUT EFFICIENCIES (Combined 2016-2018)")
    print(f"{'=' * 80}")
    print("\n```")
    print("| State   | Before Cuts | After Cuts | Efficiency |")
    print("|---------|-------------|------------|------------|")
    for state in SIGNAL_STATES:
        n_before: int = combined_events_no_cuts[state]
        n_after: int = combined_events_with_cuts[state]
        eff: float = 100.0 * n_after / n_before if n_before > 0 else 0.0
        state_name: str = STATE_LABELS[state].replace("#", "").replace("{", "").replace("}", "")
        print(f"| {state_name:7s} | {n_before:>11,} | {n_after:>10,} | {eff:>9.1f}% |")
    print("```")
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    yield_source: str = "from mass fit to data" if args.run_fit else "default estimates"
    print(f"Generated {len(years) + 1} plots (scaled to actual yields {yield_source}):")
    for year in years:
        print(f"  - mlambdapk_mc_all_states_{year}.pdf")
    print("  - mlambdapk_mc_all_states_combined.pdf")
    print("\nAbsolute yields used for scaling:")
    for state in SIGNAL_STATES:
        print(f"  {STATE_LABELS[state]:12s}: N = {absolute_yields[state]:>6.0f}")
    print(f"\nOutput directory: {output_dir}")
    if not args.run_fit:
        print("\nTip: Run with --run-fit to use actual yields from mass fit to data")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
