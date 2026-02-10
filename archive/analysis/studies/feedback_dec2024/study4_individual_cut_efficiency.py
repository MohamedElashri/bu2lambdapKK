#!/usr/bin/env python3
"""
Study 4: Individual Cut Efficiency Analysis

- Show efficiency of EACH cut applied INDIVIDUALLY (not cumulative)
- For each cut, show three categories:
  - All events (baseline)
  - Events passing that specific cut
  - Events failing that specific cut
- Final row/plot: cumulative efficiency (all cuts together)

Key differences from cumulative plots:
- Each cut is applied INDEPENDENTLY to the full dataset
- Shows what fraction passes/fails THAT specific cut alone
- Only the last entry shows cumulative effect of all cuts

Generates:
- Efficiency table with individual + cumulative
- Bar plot showing pass/fail fractions per cut

Output:
- analysis/studies/feedback_dec2024/output/study4/individual_cut_efficiency_table.txt
- analysis/studies/feedback_dec2024/output/study4/individual_cut_efficiency_mc.pdf
- analysis/studies/feedback_dec2024/output/study4/individual_cut_efficiency_data.pdf

Usage:
    cd analysis/studies/feedback_dec2024
    python study4_individual_cut_efficiency.py
"""

import argparse
import sys
from pathlib import Path

import ROOT

# Add analysis directory to path to access modules
SCRIPT_DIR: Path = Path(__file__).parent
ANALYSIS_DIR: Path = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ANALYSIS_DIR))

from modules.data_handler import TOMLConfig  # noqa: E402

# Disable ROOT GUI and info messages
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

# Signal MC states
SIGNAL_STATES: list[str] = ["Jpsi", "etac", "chic0", "chic1"]

# ROOT colors for each state
STATE_COLORS: dict[str, int] = {
    "Jpsi": ROOT.kBlue,
    "etac": ROOT.kRed,
    "chic0": ROOT.kGreen + 2,
    "chic1": ROOT.kOrange + 1,
}

# LaTeX labels for each state
STATE_LABELS: dict[str, str] = {
    "Jpsi": "J/#psi",
    "etac": "#eta_{c}",
    "chic0": "#chi_{c0}",
    "chic1": "#chi_{c1}",
}

# Ordered list of cuts (same as cumulative script)
CUTS_ORDER: list[tuple[str, str, str, float]] = [
    ("Bu_DTF_chi2", "less", "#chi^{2}_{DTF}(B^{+}) < 30", 30.0),
    ("Bu_FDCHI2_OWNPV", "greater", "FD#chi^{2}(B^{+}) > 100", 100.0),
    ("Bu_IPCHI2_OWNPV", "less", "IP#chi^{2}(B^{+}) < 10", 10.0),
    ("Bu_PT", "greater", "p_{T}(B^{+}) > 3 GeV", 3000.0),
    ("h1_ProbNNk", "greater", "ProbNN_{K}(K^{+}) > 0.1", 0.1),
    ("h2_ProbNNk", "greater", "ProbNN_{K}(K^{-}) > 0.1", 0.1),
    ("p_ProbNNp", "greater", "ProbNN_{p}(p) > 0.1", 0.1),
]


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


def build_single_cut_string(cut_index: int) -> tuple[str, str]:
    """
    Build cut string for a SINGLE cut (not cumulative).

    Args:
        cut_index: Index of the cut to apply

    Returns:
        Tuple of (pass_cut_string, fail_cut_string)
    """
    branch, cut_type, label, value = CUTS_ORDER[cut_index]
    if cut_type == "greater":
        pass_cut = f"({branch} > {value})"
        fail_cut = f"({branch} <= {value})"
    else:
        pass_cut = f"({branch} < {value})"
        fail_cut = f"({branch} >= {value})"
    return pass_cut, fail_cut


def build_all_cuts_string() -> str:
    """
    Build cumulative cut string for ALL cuts.

    Returns:
        Cut string with all cuts combined
    """
    cut_parts: list[str] = []
    for i in range(len(CUTS_ORDER)):
        branch, cut_type, label, value = CUTS_ORDER[i]
        if cut_type == "greater":
            cut_parts.append(f"({branch} > {value})")
        else:
            cut_parts.append(f"({branch} < {value})")
    return " && ".join(cut_parts)


def count_events_optimized(
    files_and_trees: list[tuple[ROOT.TFile, ROOT.TTree, str]],
    cut_string: str,
) -> int:
    """
    Count events passing a cut string from cached trees.

    Args:
        files_and_trees: List of (file, tree, identifier) tuples
        cut_string: Cut string to apply (empty for all events)

    Returns:
        Number of events
    """
    total_events: int = 0
    for _, tree, _ in files_and_trees:
        if tree:
            try:
                if cut_string:
                    n_entries: int = tree.GetEntries(cut_string)
                else:
                    n_entries: int = tree.GetEntries()
                total_events += n_entries
            except:
                pass
    return total_events


def load_all_trees_mc(
    mc_path: Path,
    state: str,
    years: list[str],
    track_types: list[str],
) -> list[tuple[ROOT.TFile, ROOT.TTree, str]]:
    """
    Load and cache all MC trees for a state.

    Returns:
        List of (file, tree, identifier) tuples
    """
    trees: list = []
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
                    if tree:
                        identifier: str = f"{state}_{year}_{magnet}_{track_type}"
                        trees.append((tfile, tree, identifier))
                except:
                    pass
    return trees


def load_all_trees_data(
    data_path: Path,
    years: list[str],
    track_types: list[str],
    bu_mass_min: float,
    bu_mass_max: float,
) -> tuple[list[tuple[ROOT.TFile, ROOT.TTree, str]], str]:
    """
    Load and cache all data trees.

    Returns:
        Tuple of (list of (file, tree, identifier), bu_cut_string)
    """
    trees: list = []
    bu_cut: str = f"(Bu_MM > {bu_mass_min}) && (Bu_MM < {bu_mass_max})"

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
                    if tree:
                        identifier: str = f"data_{year}_{magnet}_{track_type}"
                        trees.append((tfile, tree, identifier))
                except:
                    pass
    return trees, bu_cut


def close_all_trees(trees: list[tuple[ROOT.TFile, ROOT.TTree, str]]) -> None:
    """Close all cached files."""
    for tfile, _, _ in trees:
        if tfile:
            try:
                tfile.Close()
            except:
                pass


def create_efficiency_table(
    mc_results: dict[str, dict[str, list[float]]],
    data_results: dict[str, list[float]],
    output_path: Path,
) -> None:
    """
    Create and save efficiency table showing individual cut efficiencies.

    Args:
        mc_results: Dict of {state: {"pass": [...], "fail": [...]}}
        data_results: Dict of {"pass": [...], "fail": [...]}
        output_path: Path to save table
    """
    with open(output_path, "w") as f:
        f.write("=" * 120 + "\n")
        f.write("INDIVIDUAL CUT EFFICIENCY TABLE\n")
        f.write("Study 4: Each cut applied INDEPENDENTLY to full dataset\n")
        f.write("=" * 120 + "\n\n")

        # MC table
        f.write("MC EFFICIENCY (% of all events passing/failing each cut)\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Cut':<45}")
        for state in SIGNAL_STATES:
            f.write(f" {STATE_LABELS[state]:>12}")
        f.write("\n")
        f.write("-" * 120 + "\n")

        # Individual cuts
        for i, (branch, cut_type, label, value) in enumerate(CUTS_ORDER):
            # Pass line
            f.write(f"{label} [PASS]".ljust(45))
            for state in SIGNAL_STATES:
                eff: float = mc_results[state]["pass"][i]
                f.write(f" {eff:>11.1f}%")
            f.write("\n")

            # Fail line
            f.write(f"{label} [FAIL]".ljust(45))
            for state in SIGNAL_STATES:
                eff: float = mc_results[state]["fail"][i]
                f.write(f" {eff:>11.1f}%")
            f.write("\n")
            f.write("\n")

        # Cumulative (all cuts)
        f.write("-" * 120 + "\n")
        f.write(f"{'ALL CUTS COMBINED (cumulative)'.ljust(45)}")
        for state in SIGNAL_STATES:
            eff: float = mc_results[state]["cumulative"]
            f.write(f" {eff:>11.1f}%")
        f.write("\n")
        f.write("=" * 120 + "\n\n")

        # Data table
        f.write("DATA EFFICIENCY (% of all events passing/failing each cut)\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Cut':<45} {'Efficiency':>15}\n")
        f.write("-" * 120 + "\n")

        # Individual cuts
        for i, (branch, cut_type, label, value) in enumerate(CUTS_ORDER):
            f.write(f"{label} [PASS]".ljust(45))
            f.write(f" {data_results['pass'][i]:>14.1f}%\n")

            f.write(f"{label} [FAIL]".ljust(45))
            f.write(f" {data_results['fail'][i]:>14.1f}%\n")
            f.write("\n")

        # Cumulative
        f.write("-" * 120 + "\n")
        f.write(f"{'ALL CUTS COMBINED (cumulative)'.ljust(45)}")
        f.write(f" {data_results['cumulative']:>14.1f}%\n")
        f.write("=" * 120 + "\n")

    print(f"  Saved: {output_path}")


def create_efficiency_plot_mc(
    mc_results: dict[str, dict[str, list[float]]],
    output_path: Path,
) -> None:
    """
    Create bar plot showing pass/fail efficiency for each cut (MC).

    Args:
        mc_results: MC efficiency results
        output_path: Path to save PDF
    """
    keep_alive: list = []
    n_cuts: int = len(CUTS_ORDER)

    canvas: ROOT.TCanvas = ROOT.TCanvas("c_eff_mc", "", 1400, 800)
    ROOT.SetOwnership(canvas, False)
    keep_alive.append(canvas)

    canvas.SetLeftMargin(0.25)
    canvas.SetRightMargin(0.15)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.12)
    canvas.SetGridy(True)

    # Create histogram for each state
    for i_state, state in enumerate(SIGNAL_STATES):
        h_pass: ROOT.TH1D = ROOT.TH1D(f"h_pass_{state}", "", n_cuts + 1, 0, n_cuts + 1)
        h_fail: ROOT.TH1D = ROOT.TH1D(f"h_fail_{state}", "", n_cuts + 1, 0, n_cuts + 1)
        ROOT.SetOwnership(h_pass, False)
        ROOT.SetOwnership(h_fail, False)
        keep_alive.extend([h_pass, h_fail])

        # Fill individual cuts
        for i in range(n_cuts):
            h_pass.SetBinContent(i + 1, mc_results[state]["pass"][i])
            h_fail.SetBinContent(i + 1, mc_results[state]["fail"][i])

        # Fill cumulative
        h_pass.SetBinContent(n_cuts + 1, mc_results[state]["cumulative"])
        h_fail.SetBinContent(n_cuts + 1, 100.0 - mc_results[state]["cumulative"])

        # Set labels
        for i in range(n_cuts):
            short_label: str = f"Cut {i+1}"
            h_pass.GetXaxis().SetBinLabel(i + 1, short_label)
        h_pass.GetXaxis().SetBinLabel(n_cuts + 1, "All Cuts")

        # Style pass
        h_pass.SetFillColorAlpha(STATE_COLORS[state], 0.7)
        h_pass.SetLineColor(STATE_COLORS[state])
        h_pass.SetLineWidth(2)
        h_pass.SetMaximum(105)
        h_pass.SetMinimum(0)
        h_pass.GetYaxis().SetTitle("Efficiency [%]")
        h_pass.GetXaxis().SetTitleFont(132)
        h_pass.GetYaxis().SetTitleFont(132)
        h_pass.GetXaxis().SetLabelFont(132)
        h_pass.GetYaxis().SetLabelFont(132)
        h_pass.GetXaxis().SetLabelSize(0.045)
        h_pass.GetYaxis().SetLabelSize(0.04)
        h_pass.GetYaxis().SetTitleSize(0.045)
        h_pass.GetYaxis().SetTitleOffset(1.5)

        # Create multi-panel plot or overlay
        # For simplicity, create separate canvases for each state
        canvas_state: ROOT.TCanvas = ROOT.TCanvas(f"c_{state}", "", 1400, 700)
        ROOT.SetOwnership(canvas_state, False)
        keep_alive.append(canvas_state)

        canvas_state.SetLeftMargin(0.12)
        canvas_state.SetRightMargin(0.05)
        canvas_state.SetTopMargin(0.08)
        canvas_state.SetBottomMargin(0.15)
        canvas_state.SetGridy(True)

        # Create stacked histogram
        stack: ROOT.THStack = ROOT.THStack("stack", "")
        ROOT.SetOwnership(stack, False)
        keep_alive.append(stack)

        h_fail_plot: ROOT.TH1D = h_fail.Clone(f"h_fail_plot_{state}")
        ROOT.SetOwnership(h_fail_plot, False)
        keep_alive.append(h_fail_plot)
        h_fail_plot.SetFillColorAlpha(ROOT.kGray, 0.3)
        h_fail_plot.SetLineColor(ROOT.kGray + 2)
        h_fail_plot.SetLineWidth(2)

        stack.Add(h_pass)
        stack.Add(h_fail_plot)
        stack.Draw("HIST")
        stack.SetMaximum(105)
        stack.SetMinimum(0)
        stack.GetYaxis().SetTitle("Efficiency [%]")
        stack.GetXaxis().SetTitleFont(132)
        stack.GetYaxis().SetTitleFont(132)
        stack.GetXaxis().SetLabelFont(132)
        stack.GetYaxis().SetLabelFont(132)
        stack.GetXaxis().SetLabelSize(0.045)
        stack.GetYaxis().SetLabelSize(0.04)
        stack.GetYaxis().SetTitleSize(0.045)
        stack.GetYaxis().SetTitleOffset(1.2)

        # Legend
        legend: ROOT.TLegend = ROOT.TLegend(0.75, 0.75, 0.92, 0.88)
        ROOT.SetOwnership(legend, False)
        keep_alive.append(legend)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextFont(132)
        legend.SetTextSize(0.035)
        legend.AddEntry(h_pass, "Pass cut", "f")
        legend.AddEntry(h_fail_plot, "Fail cut", "f")
        legend.Draw()

        # Title
        title: ROOT.TLatex = ROOT.TLatex()
        title.SetNDC()
        title.SetTextFont(132)
        title.SetTextSize(0.045)
        title.DrawLatex(0.15, 0.93, f"Individual Cut Efficiency: {STATE_LABELS[state]}")
        keep_alive.append(title)

        # Labels
        lhcb: ROOT.TLatex = ROOT.TLatex()
        lhcb.SetNDC()
        lhcb.SetTextFont(132)
        lhcb.SetTextSize(0.04)
        lhcb.DrawLatex(0.15, 0.85, "LHCb MC")
        keep_alive.append(lhcb)

        canvas_state.Modified()
        canvas_state.Update()

        # Save individual state plot
        state_pdf: Path = output_path.parent / f"individual_cut_efficiency_mc_{state}.pdf"
        canvas_state.SaveAs(str(state_pdf))
        print(f"  Saved: {state_pdf.name}")


def create_efficiency_plot_data(
    data_results: dict[str, list[float]],
    output_path: Path,
) -> None:
    """
    Create bar plot showing pass/fail efficiency for each cut (Data).

    Args:
        data_results: Data efficiency results
        output_path: Path to save PDF
    """
    keep_alive: list = []
    n_cuts: int = len(CUTS_ORDER)

    canvas: ROOT.TCanvas = ROOT.TCanvas("c_eff_data", "", 1400, 700)
    ROOT.SetOwnership(canvas, False)
    keep_alive.append(canvas)

    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.05)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.15)
    canvas.SetGridy(True)

    h_pass: ROOT.TH1D = ROOT.TH1D("h_pass_data", "", n_cuts + 1, 0, n_cuts + 1)
    h_fail: ROOT.TH1D = ROOT.TH1D("h_fail_data", "", n_cuts + 1, 0, n_cuts + 1)
    ROOT.SetOwnership(h_pass, False)
    ROOT.SetOwnership(h_fail, False)
    keep_alive.extend([h_pass, h_fail])

    # Fill individual cuts
    for i in range(n_cuts):
        h_pass.SetBinContent(i + 1, data_results["pass"][i])
        h_fail.SetBinContent(i + 1, data_results["fail"][i])

    # Fill cumulative
    h_pass.SetBinContent(n_cuts + 1, data_results["cumulative"])
    h_fail.SetBinContent(n_cuts + 1, 100.0 - data_results["cumulative"])

    # Set labels
    for i in range(n_cuts):
        short_label: str = f"Cut {i+1}"
        h_pass.GetXaxis().SetBinLabel(i + 1, short_label)
    h_pass.GetXaxis().SetBinLabel(n_cuts + 1, "All Cuts")

    # Style
    h_pass.SetFillColorAlpha(ROOT.kBlue, 0.7)
    h_pass.SetLineColor(ROOT.kBlue)
    h_pass.SetLineWidth(2)

    h_fail.SetFillColorAlpha(ROOT.kGray, 0.3)
    h_fail.SetLineColor(ROOT.kGray + 2)
    h_fail.SetLineWidth(2)

    # Create stack
    stack: ROOT.THStack = ROOT.THStack("stack_data", "")
    ROOT.SetOwnership(stack, False)
    keep_alive.append(stack)
    stack.Add(h_pass)
    stack.Add(h_fail)
    stack.Draw("HIST")
    stack.SetMaximum(105)
    stack.SetMinimum(0)
    stack.GetYaxis().SetTitle("Efficiency [%]")
    stack.GetXaxis().SetTitleFont(132)
    stack.GetYaxis().SetTitleFont(132)
    stack.GetXaxis().SetLabelFont(132)
    stack.GetYaxis().SetLabelFont(132)
    stack.GetXaxis().SetLabelSize(0.045)
    stack.GetYaxis().SetLabelSize(0.04)
    stack.GetYaxis().SetTitleSize(0.045)
    stack.GetYaxis().SetTitleOffset(1.2)

    # Legend
    legend: ROOT.TLegend = ROOT.TLegend(0.75, 0.75, 0.92, 0.88)
    ROOT.SetOwnership(legend, False)
    keep_alive.append(legend)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(132)
    legend.SetTextSize(0.035)
    legend.AddEntry(h_pass, "Pass cut", "f")
    legend.AddEntry(h_fail, "Fail cut", "f")
    legend.Draw()

    # Title
    title: ROOT.TLatex = ROOT.TLatex()
    title.SetNDC()
    title.SetTextFont(132)
    title.SetTextSize(0.045)
    title.DrawLatex(0.15, 0.93, "Individual Cut Efficiency: Data")
    keep_alive.append(title)

    # Labels
    lhcb: ROOT.TLatex = ROOT.TLatex()
    lhcb.SetNDC()
    lhcb.SetTextFont(132)
    lhcb.SetTextSize(0.04)
    lhcb.DrawLatex(0.15, 0.85, "LHCb Data")
    keep_alive.append(lhcb)

    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(output_path))
    canvas.SaveAs(str(output_path.with_suffix(".png")))

    print(f"  Saved: {output_path.name}")


def main() -> None:
    """Main function."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Study 4: Individual cut efficiency analysis"
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
    print("STUDY 4: Individual Cut Efficiency Analysis")
    print("=" * 80)
    print("Each cut applied INDEPENDENTLY to full dataset")
    print(f"Years: {years}")
    print(f"Track types: {track_types}")
    print(f"MC States: {SIGNAL_STATES}")
    print("=" * 80)
    print("\nCuts:")
    for i, (branch, cut_type, label, value) in enumerate(CUTS_ORDER):
        op: str = ">" if cut_type == "greater" else "<"
        print(f"  {i+1}. {branch} {op} {value}")
    print()

    setup_lhcb_style()

    # Load configuration
    config: TOMLConfig = TOMLConfig(config_dir=str(ANALYSIS_DIR / "config"))
    mc_path: Path = Path(config.paths["mc"]["base_path"])
    data_path: Path = Path(config.paths["data"]["base_path"])

    # Get B+ signal region
    bu_fixed: dict = config.selection.get("bu_fixed_selection", {})
    bu_mass_min: float = bu_fixed.get("mass_corrected_min", 5255.0)
    bu_mass_max: float = bu_fixed.get("mass_corrected_max", 5305.0)
    print(f"B+ signal region for data: [{bu_mass_min:.0f}, {bu_mass_max:.0f}] MeV\n")

    # Output directory
    output_dir: Path = SCRIPT_DIR / "output" / "study4"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Storage for results
    mc_results: dict[str, dict[str, list[float]]] = {}
    for state in SIGNAL_STATES:
        mc_results[state] = {"pass": [], "fail": []}

    data_results: dict[str, list[float]] = {"pass": [], "fail": []}

    # OPTIMIZED: Load and cache all trees once
    print("=" * 60)
    print("LOADING AND CACHING TREES (once)")
    print("=" * 60)

    mc_trees_cache: dict[str, list] = {}
    for state in SIGNAL_STATES:
        print(f"  Loading {STATE_LABELS[state]} trees...")
        mc_trees_cache[state] = load_all_trees_mc(mc_path, state, years, track_types)
        print(f"    Loaded {len(mc_trees_cache[state])} trees")

    print("  Loading Data trees...")
    data_trees_cache, bu_cut = load_all_trees_data(
        data_path, years, track_types, bu_mass_min, bu_mass_max
    )
    print(f"    Loaded {len(data_trees_cache)} trees")

    # Count all events (no cuts)
    print("\n" + "=" * 60)
    print("COUNTING ALL EVENTS (baseline)")
    print("=" * 60)
    mc_all: dict[str, int] = {}
    for state in SIGNAL_STATES:
        n_all: int = count_events_optimized(mc_trees_cache[state], "")
        mc_all[state] = n_all
        print(f"  {STATE_LABELS[state]:12s}: {n_all:>10,} events")

    data_all: int = count_events_optimized(data_trees_cache, bu_cut)
    print(f"  {'Data':12s}: {data_all:>10,} events")

    # Process individual cuts
    print("\n" + "=" * 60)
    print("PROCESSING INDIVIDUAL CUTS")
    print("=" * 60)

    for i, (branch, cut_type, label, value) in enumerate(CUTS_ORDER):
        print(f"\nCut {i+1}: {label}")
        pass_cut, fail_cut = build_single_cut_string(i)

        # MC
        for state in SIGNAL_STATES:
            n_pass: int = count_events_optimized(mc_trees_cache[state], pass_cut)
            n_fail: int = count_events_optimized(mc_trees_cache[state], fail_cut)
            eff_pass: float = 100.0 * n_pass / mc_all[state] if mc_all[state] > 0 else 0.0
            eff_fail: float = 100.0 * n_fail / mc_all[state] if mc_all[state] > 0 else 0.0
            mc_results[state]["pass"].append(eff_pass)
            mc_results[state]["fail"].append(eff_fail)
            print(f"  {STATE_LABELS[state]:12s}: Pass={eff_pass:5.1f}%, Fail={eff_fail:5.1f}%")

        # Data
        pass_cut_full: str = f"{pass_cut} && {bu_cut}"
        fail_cut_full: str = f"{fail_cut} && {bu_cut}"
        n_pass: int = count_events_optimized(data_trees_cache, pass_cut_full)
        n_fail: int = count_events_optimized(data_trees_cache, fail_cut_full)
        eff_pass: float = 100.0 * n_pass / data_all if data_all > 0 else 0.0
        eff_fail: float = 100.0 * n_fail / data_all if data_all > 0 else 0.0
        data_results["pass"].append(eff_pass)
        data_results["fail"].append(eff_fail)
        print(f"  {'Data':12s}: Pass={eff_pass:5.1f}%, Fail={eff_fail:5.1f}%")

    # Cumulative (all cuts together)
    print("\n" + "=" * 60)
    print("CUMULATIVE EFFICIENCY (all cuts)")
    print("=" * 60)
    all_cuts: str = build_all_cuts_string()

    for state in SIGNAL_STATES:
        n_cumulative: int = count_events_optimized(mc_trees_cache[state], all_cuts)
        eff_cumulative: float = 100.0 * n_cumulative / mc_all[state] if mc_all[state] > 0 else 0.0
        mc_results[state]["cumulative"] = eff_cumulative
        print(f"  {STATE_LABELS[state]:12s}: {eff_cumulative:5.1f}%")

    all_cuts_data: str = f"{all_cuts} && {bu_cut}"
    n_cumulative: int = count_events_optimized(data_trees_cache, all_cuts_data)
    eff_cumulative: float = 100.0 * n_cumulative / data_all if data_all > 0 else 0.0
    data_results["cumulative"] = eff_cumulative
    print(f"  {'Data':12s}: {eff_cumulative:5.1f}%")

    # Clean up: close all files
    print("\nCleaning up cached trees...")
    for state in SIGNAL_STATES:
        close_all_trees(mc_trees_cache[state])
    close_all_trees(data_trees_cache)

    # Create outputs
    print("\n" + "=" * 60)
    print("GENERATING OUTPUTS")
    print("=" * 60)

    table_path: Path = output_dir / "individual_cut_efficiency_table.txt"
    create_efficiency_table(mc_results, data_results, table_path)

    mc_plot_path: Path = output_dir / "individual_cut_efficiency_mc.pdf"
    create_efficiency_plot_mc(mc_results, mc_plot_path)

    data_plot_path: Path = output_dir / "individual_cut_efficiency_data.pdf"
    create_efficiency_plot_data(data_results, data_plot_path)

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print("Generated individual cut efficiency analysis:")
    print("  - individual_cut_efficiency_table.txt")
    print("  - individual_cut_efficiency_mc_[state].pdf (per state)")
    print("  - individual_cut_efficiency_data.pdf")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
