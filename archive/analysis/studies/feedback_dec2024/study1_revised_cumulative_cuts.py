#!/usr/bin/env python3
"""
Study 1: Revised Cumulative Cut Efficiency with Three Categories

- Show THREE overlaid distributions for each cumulative cut stage:
  1. All events (baseline, transparent fill)
  2. Events passing cuts (solid color)
  3. Events failing cuts (different color, transparent)
- Remove overlapping text (LHCb and year labels)
- Move "no cuts vs cuts" info to plot title
- Use different colors and transparency for clarity

Key differences from original:
- Shows fail category explicitly (complement of pass)
- Cleaner layout without text overlaps
- Title conveys cut information

Generates:
- Plots for each cumulative cut stage (cut1-cut7)
- Separate plots for MC and Data
- Three categories clearly distinguished

Output:
- analysis/studies/feedback_dec2024/output/study1/cut{i}_three_categories_mc.pdf
- analysis/studies/feedback_dec2024/output/study1/cut{i}_three_categories_data.pdf

Usage:
    cd analysis/studies/feedback_dec2024
    python study1_revised_cumulative_cuts.py
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

# M(LambdaPK) range
M_LPKM_MIN: float = 2800.0
M_LPKM_MAX: float = 4000.0
N_BINS: int = 120

# Ordered list of cuts to apply cumulatively
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
    ROOT.gStyle.SetPadTopMargin(0.10)
    ROOT.gStyle.SetPadBottomMargin(0.12)
    ROOT.gStyle.SetTitleFont(132, "XYZ")
    ROOT.gStyle.SetLabelFont(132, "XYZ")
    ROOT.gStyle.SetTextFont(132)
    ROOT.gStyle.SetTitleSize(0.05, "XYZ")
    ROOT.gStyle.SetLabelSize(0.04, "XYZ")


def build_single_cut_string(cut_index: int) -> str:
    """
    Build cut string for a SINGLE cut (not cumulative).

    Args:
        cut_index: Index of the cut to apply

    Returns:
        Cut string for TTree::Draw
    """
    branch, cut_type, label, value = CUTS_ORDER[cut_index]
    if cut_type == "greater":
        return f"({branch} > {value})"
    else:
        return f"({branch} < {value})"


def build_single_cut_negation(cut_index: int) -> str:
    """
    Build NEGATION of a single cut (events FAILING the cut).

    Args:
        cut_index: Index of the cut

    Returns:
        Negated cut string
    """
    branch, cut_type, label, value = CUTS_ORDER[cut_index]
    if cut_type == "greater":
        # Negation: <= instead of >
        return f"({branch} <= {value})"
    else:
        # Negation: >= instead of <
        return f"({branch} >= {value})"


def load_histogram_with_cut(
    files_trees: list,
    hist_name: str,
    cut_string: str,
    mass_formula: str,
) -> tuple[ROOT.TH1D, int]:
    """
    Load histogram with optional cut from cached trees.

    Args:
        files_trees: List of (file, tree, identifier) tuples
        hist_name: Name for histogram
        cut_string: Cut string to apply
        mass_formula: Formula for M(LambdaPK)

    Returns:
        Tuple of (histogram, n_events)
    """
    hist: ROOT.TH1D = ROOT.TH1D(hist_name, "", N_BINS, M_LPKM_MIN, M_LPKM_MAX)
    hist.Sumw2()
    total_events: int = 0

    for _, tree, identifier in files_trees:
        if tree:
            try:
                temp_hist_name: str = f"temp_{hist_name}_{identifier}"
                n_entries: int = tree.Draw(
                    f"{mass_formula}>>{temp_hist_name}({N_BINS},{M_LPKM_MIN},{M_LPKM_MAX})",
                    cut_string,
                    "goff",
                )
                if n_entries > 0:
                    temp_hist: ROOT.TH1D = ROOT.gDirectory.Get(temp_hist_name)
                    if temp_hist:
                        hist.Add(temp_hist)
                        total_events += n_entries
                        temp_hist.Delete()
            except:
                pass

    ROOT.SetOwnership(hist, False)
    return hist, total_events


def load_all_trees_mc(
    mc_path: Path,
    state: str,
    years: list[str],
    track_types: list[str],
) -> list:
    """Load and cache all MC trees for a state."""
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
) -> tuple[list, str]:
    """Load and cache all data trees."""
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


def close_all_trees(trees: list) -> None:
    """Close all cached files."""
    for tfile, _, _ in trees:
        if tfile:
            try:
                tfile.Close()
            except:
                pass


def create_three_category_plot_mc(
    hist_all: dict[str, ROOT.TH1D],
    hist_pass: dict[str, ROOT.TH1D],
    hist_fail: dict[str, ROOT.TH1D],
    n_all: dict[str, int],
    n_pass: dict[str, int],
    n_fail: dict[str, int],
    cut_idx: int,
    cut_label: str,
    output_path: Path,
) -> None:
    """
    Create MC plot with three categories: all, pass, fail.

    Args:
        hist_all: Dict of {state: histogram} for all events
        hist_pass: Dict of {state: histogram} for events passing cut
        hist_fail: Dict of {state: histogram} for events failing cut
        n_all: Event counts for all
        n_pass: Event counts for pass
        n_fail: Event counts for fail
        cut_idx: Cut index (1-7)
        cut_label: LaTeX label for the cut
        output_path: Path to save PDF
    """
    keep_alive: list = []

    canvas: ROOT.TCanvas = ROOT.TCanvas(f"c_mc_cut{cut_idx}", "", 900, 700)
    ROOT.SetOwnership(canvas, False)
    keep_alive.append(canvas)

    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.05)
    canvas.SetTopMargin(0.12)
    canvas.SetBottomMargin(0.12)

    # Find y maximum across all categories
    y_max: float = 0.0
    for state in SIGNAL_STATES:
        y_max = max(y_max, hist_all[state].GetMaximum())
        y_max = max(y_max, hist_pass[state].GetMaximum())
        y_max = max(y_max, hist_fail[state].GetMaximum())
    y_max *= 1.4

    # Draw order: all (filled, transparent) -> fail (filled, transparent) -> pass (line)
    first_drawn: bool = False

    # 1. Draw ALL events (baseline, gray transparent fill)
    for state in SIGNAL_STATES:
        h: ROOT.TH1D = hist_all[state].Clone(f"all_{state}_cut{cut_idx}")
        ROOT.SetOwnership(h, False)
        keep_alive.append(h)

        h.SetFillColorAlpha(ROOT.kGray, 0.3)
        h.SetFillStyle(1001)
        h.SetLineColor(ROOT.kBlack)
        h.SetLineWidth(2)
        h.SetLineStyle(1)  # Solid
        h.SetMaximum(y_max)
        h.SetMinimum(0)
        h.GetXaxis().SetTitle("M(#bar{#Lambda}pK^{#minus}) [MeV/#it{c}^{2}]")
        h.GetYaxis().SetTitle("Candidates")
        h.GetXaxis().SetTitleFont(132)
        h.GetYaxis().SetTitleFont(132)
        h.GetXaxis().SetLabelFont(132)
        h.GetYaxis().SetLabelFont(132)
        h.GetXaxis().SetTitleSize(0.045)
        h.GetYaxis().SetTitleSize(0.045)
        h.GetYaxis().SetTitleOffset(1.3)

        if not first_drawn:
            h.Draw("HIST")
            first_drawn = True
        else:
            h.Draw("HIST SAME")

    # 2. Draw FAIL events (red transparent fill)
    for state in SIGNAL_STATES:
        h: ROOT.TH1D = hist_fail[state].Clone(f"fail_{state}_cut{cut_idx}")
        ROOT.SetOwnership(h, False)
        keep_alive.append(h)

        h.SetFillColorAlpha(ROOT.kRed, 0.25)
        h.SetFillStyle(3004)  # Hatched pattern
        h.SetLineColor(ROOT.kRed + 1)
        h.SetLineWidth(2)
        h.SetLineStyle(1)
        h.Draw("HIST SAME")

    # 3. Draw PASS events (solid line with state colors, no fill)
    for state in SIGNAL_STATES:
        h: ROOT.TH1D = hist_pass[state].Clone(f"pass_{state}_cut{cut_idx}")
        ROOT.SetOwnership(h, False)
        keep_alive.append(h)

        h.SetLineColor(STATE_COLORS[state])
        h.SetLineWidth(3)
        h.SetLineStyle(1)  # Solid
        h.SetFillStyle(0)  # No fill
        h.Draw("HIST SAME")

    # Legend for category colors (top left)
    legend_cat: ROOT.TLegend = ROOT.TLegend(0.14, 0.68, 0.35, 0.88)
    ROOT.SetOwnership(legend_cat, False)
    keep_alive.append(legend_cat)
    legend_cat.SetBorderSize(0)
    legend_cat.SetFillStyle(0)
    legend_cat.SetTextFont(132)
    legend_cat.SetTextSize(0.032)
    legend_cat.SetHeader("Categories", "L")

    # Create dummy histograms for legend
    h_all_dummy: ROOT.TH1D = ROOT.TH1D("h_all_dummy", "", 1, 0, 1)
    h_all_dummy.SetFillColorAlpha(ROOT.kGray, 0.15)
    h_all_dummy.SetLineColor(ROOT.kGray + 1)
    h_all_dummy.SetLineStyle(2)
    ROOT.SetOwnership(h_all_dummy, False)
    keep_alive.append(h_all_dummy)

    h_pass_dummy: ROOT.TH1D = ROOT.TH1D("h_pass_dummy", "", 1, 0, 1)
    h_pass_dummy.SetLineColor(ROOT.kBlue)
    h_pass_dummy.SetLineWidth(3)
    ROOT.SetOwnership(h_pass_dummy, False)
    keep_alive.append(h_pass_dummy)

    h_fail_dummy: ROOT.TH1D = ROOT.TH1D("h_fail_dummy", "", 1, 0, 1)
    h_fail_dummy.SetFillColorAlpha(ROOT.kRed, 0.25)
    h_fail_dummy.SetFillStyle(3004)
    h_fail_dummy.SetLineColor(ROOT.kRed + 1)
    ROOT.SetOwnership(h_fail_dummy, False)
    keep_alive.append(h_fail_dummy)

    legend_cat.AddEntry(h_all_dummy, "All events", "lf")
    legend_cat.AddEntry(h_pass_dummy, "Pass cut", "l")
    legend_cat.AddEntry(h_fail_dummy, "Fail cut", "lf")
    legend_cat.Draw()

    # Legend for states (top right)
    legend_states: ROOT.TLegend = ROOT.TLegend(0.65, 0.68, 0.92, 0.88)
    ROOT.SetOwnership(legend_states, False)
    keep_alive.append(legend_states)
    legend_states.SetBorderSize(0)
    legend_states.SetFillStyle(0)
    legend_states.SetTextFont(132)
    legend_states.SetTextSize(0.032)
    legend_states.SetHeader("Efficiency", "L")

    for state in SIGNAL_STATES:
        eff: float = 100.0 * n_pass[state] / n_all[state] if n_all[state] > 0 else 0.0
        legend_states.AddEntry(hist_pass[state], f"{STATE_LABELS[state]}: {eff:.1f}%", "l")

    legend_states.Draw()

    # Title with cut name and LHCb MC
    title: ROOT.TLatex = ROOT.TLatex()
    title.SetNDC()
    title.SetTextFont(132)
    title.SetTextSize(0.045)
    title_text: str = f"LHCb MC: {cut_label}"
    title.DrawLatex(0.12, 0.95, title_text)
    keep_alive.append(title)

    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(output_path))
    canvas.SaveAs(str(output_path.with_suffix(".png")))


def create_three_category_plot_data(
    hist_all: ROOT.TH1D,
    hist_pass: ROOT.TH1D,
    hist_fail: ROOT.TH1D,
    n_all: int,
    n_pass: int,
    n_fail: int,
    cut_idx: int,
    cut_label: str,
    output_path: Path,
) -> None:
    """
    Create Data plot with three categories: all, pass, fail.

    Args:
        hist_all: Histogram for all events
        hist_pass: Histogram for events passing cut
        hist_fail: Histogram for events failing cut
        n_all: Event count for all
        n_pass: Event count for pass
        n_fail: Event count for fail
        cut_idx: Cut index (1-7)
        cut_label: LaTeX label for the cut
        output_path: Path to save PDF
    """
    keep_alive: list = []

    canvas: ROOT.TCanvas = ROOT.TCanvas(f"c_data_cut{cut_idx}", "", 900, 700)
    ROOT.SetOwnership(canvas, False)
    keep_alive.append(canvas)

    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.05)
    canvas.SetTopMargin(0.12)
    canvas.SetBottomMargin(0.12)

    # Find y maximum
    y_max: float = max(hist_all.GetMaximum(), hist_pass.GetMaximum(), hist_fail.GetMaximum()) * 1.4

    # 1. ALL events (baseline, gray transparent fill)
    h_all: ROOT.TH1D = hist_all.Clone(f"all_data_cut{cut_idx}")
    ROOT.SetOwnership(h_all, False)
    keep_alive.append(h_all)

    h_all.SetFillColorAlpha(ROOT.kGray, 0.3)
    h_all.SetFillStyle(1001)
    h_all.SetLineColor(ROOT.kBlack)
    h_all.SetLineWidth(2)
    h_all.SetLineStyle(1)
    h_all.SetMaximum(y_max)
    h_all.SetMinimum(0)
    h_all.GetXaxis().SetTitle("M(#bar{#Lambda}pK^{#minus}) [MeV/#it{c}^{2}]")
    h_all.GetYaxis().SetTitle("Candidates")
    h_all.GetXaxis().SetTitleFont(132)
    h_all.GetYaxis().SetTitleFont(132)
    h_all.GetXaxis().SetLabelFont(132)
    h_all.GetYaxis().SetLabelFont(132)
    h_all.GetXaxis().SetTitleSize(0.045)
    h_all.GetYaxis().SetTitleSize(0.045)
    h_all.GetYaxis().SetTitleOffset(1.3)
    h_all.Draw("HIST")

    # 2. FAIL events
    h_fail: ROOT.TH1D = hist_fail.Clone(f"fail_data_cut{cut_idx}")
    ROOT.SetOwnership(h_fail, False)
    keep_alive.append(h_fail)

    h_fail.SetFillColorAlpha(ROOT.kRed, 0.25)
    h_fail.SetFillStyle(3004)
    h_fail.SetLineColor(ROOT.kRed)
    h_fail.SetLineWidth(2)
    h_fail.SetLineStyle(1)
    h_fail.Draw("HIST SAME")

    # 3. PASS events
    h_pass: ROOT.TH1D = hist_pass.Clone(f"pass_data_cut{cut_idx}")
    ROOT.SetOwnership(h_pass, False)
    keep_alive.append(h_pass)

    h_pass.SetLineColor(ROOT.kBlue)
    h_pass.SetLineWidth(3)
    h_pass.SetLineStyle(1)
    h_pass.SetFillStyle(0)
    h_pass.Draw("HIST SAME")

    # Legend
    legend: ROOT.TLegend = ROOT.TLegend(0.55, 0.65, 0.92, 0.88)
    ROOT.SetOwnership(legend, False)
    keep_alive.append(legend)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(132)
    legend.SetTextSize(0.035)

    eff: float = 100.0 * n_pass / n_all if n_all > 0 else 0.0
    legend.AddEntry(h_all, f"All events: {n_all:,}", "lf")
    legend.AddEntry(h_pass, f"Pass cut: {n_pass:,} ({eff:.1f}%)", "l")
    legend.AddEntry(h_fail, f"Fail cut: {n_fail:,}", "lf")
    legend.Draw()

    # Title with cut name and LHCb Data
    title: ROOT.TLatex = ROOT.TLatex()
    title.SetNDC()
    title.SetTextFont(132)
    title.SetTextSize(0.045)
    title_text: str = f"LHCb Data: {cut_label}"
    title.DrawLatex(0.12, 0.95, title_text)
    keep_alive.append(title)

    # B+ region label (smaller, bottom left)
    region: ROOT.TLatex = ROOT.TLatex()
    region.SetNDC()
    region.SetTextFont(132)
    region.SetTextSize(0.028)
    region.DrawLatex(0.14, 0.88, "B^{+} signal region")
    keep_alive.append(region)

    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(output_path))
    canvas.SaveAs(str(output_path.with_suffix(".png")))


def main() -> None:
    """Main function."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Study 1: Three-category cumulative cut efficiency plots"
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
    print("STUDY 1: Three-Category Cumulative Cut Efficiency")
    print("=" * 80)
    print("Shows: All events / Pass cuts / Fail cuts")
    print(f"Years: {years}")
    print(f"Track types: {track_types}")
    print("=" * 80)

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
    output_dir: Path = SCRIPT_DIR / "output" / "study1"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mass formula
    mass_formula: str = (
        "sqrt((L0_PE + p_PE + h2_PE)*(L0_PE + p_PE + h2_PE) - "
        "(L0_PX + p_PX + h2_PX)*(L0_PX + p_PX + h2_PX) - "
        "(L0_PY + p_PY + h2_PY)*(L0_PY + p_PY + h2_PY) - "
        "(L0_PZ + p_PZ + h2_PZ)*(L0_PZ + p_PZ + h2_PZ))"
    )

    # Load and cache trees
    print("=" * 60)
    print("LOADING AND CACHING TREES")
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

    # Process each cut individually (NOT cumulative)
    n_cuts: int = len(CUTS_ORDER)

    for cut_idx in range(n_cuts):
        cut_name: str = CUTS_ORDER[cut_idx][0]  # e.g., "Bu_DTF_chi2"
        cut_label: str = CUTS_ORDER[cut_idx][2]  # e.g., "#chi^{2}_{DTF} < 30"
        print(f"\n{'=' * 60}")
        print(f"CUT {cut_idx + 1}: {cut_name}")
        print("=" * 60)

        # Build SINGLE cut string (NOT cumulative)
        pass_cut: str = build_single_cut_string(cut_idx)
        fail_cut: str = build_single_cut_negation(cut_idx)

        print(f"  Cut applied: {pass_cut}")

        # MC histograms
        print("  Loading MC histograms...")
        mc_hist_all: dict[str, ROOT.TH1D] = {}
        mc_hist_pass: dict[str, ROOT.TH1D] = {}
        mc_hist_fail: dict[str, ROOT.TH1D] = {}
        mc_n_all: dict[str, int] = {}
        mc_n_pass: dict[str, int] = {}
        mc_n_fail: dict[str, int] = {}

        for state in SIGNAL_STATES:
            # All events (no cut)
            h_all, n_all = load_histogram_with_cut(
                mc_trees_cache[state], f"h_mc_all_{state}_cut{cut_idx}", "", mass_formula  # No cut
            )
            mc_hist_all[state] = h_all
            mc_n_all[state] = n_all

            # Pass cut
            h_pass, n_pass = load_histogram_with_cut(
                mc_trees_cache[state], f"h_mc_pass_{state}_cut{cut_idx}", pass_cut, mass_formula
            )
            mc_hist_pass[state] = h_pass
            mc_n_pass[state] = n_pass

            # Fail cut
            h_fail, n_fail = load_histogram_with_cut(
                mc_trees_cache[state], f"h_mc_fail_{state}_cut{cut_idx}", fail_cut, mass_formula
            )
            mc_hist_fail[state] = h_fail
            mc_n_fail[state] = n_fail

            eff: float = 100.0 * n_pass / n_all if n_all > 0 else 0.0
            print(
                f"    {STATE_LABELS[state]:12s}: All={n_all:>8,}, Pass={n_pass:>8,} ({eff:.1f}%), Fail={n_fail:>8,}"
            )

        # Data histograms
        print("  Loading Data histograms...")

        # All events in B+ region
        data_hist_all, data_n_all = load_histogram_with_cut(
            data_trees_cache, f"h_data_all_cut{cut_idx}", bu_cut, mass_formula
        )

        # Pass cut (apply single cut in B+ region)
        pass_cut_data: str = f"{pass_cut} && {bu_cut}"
        data_hist_pass, data_n_pass = load_histogram_with_cut(
            data_trees_cache, f"h_data_pass_cut{cut_idx}", pass_cut_data, mass_formula
        )

        # Fail cut (apply negated single cut in B+ region)
        fail_cut_data: str = f"({fail_cut}) && {bu_cut}"
        data_hist_fail, data_n_fail = load_histogram_with_cut(
            data_trees_cache, f"h_data_fail_cut{cut_idx}", fail_cut_data, mass_formula
        )

        eff_data: float = 100.0 * data_n_pass / data_n_all if data_n_all > 0 else 0.0
        print(
            f"    {'Data':12s}: All={data_n_all:>8,}, Pass={data_n_pass:>8,} ({eff_data:.1f}%), Fail={data_n_fail:>8,}"
        )

        # Create plots
        print("  Creating plots...")

        mc_pdf: Path = output_dir / f"cut{cut_idx + 1}_mc_{cut_name}.pdf"
        create_three_category_plot_mc(
            mc_hist_all,
            mc_hist_pass,
            mc_hist_fail,
            mc_n_all,
            mc_n_pass,
            mc_n_fail,
            cut_idx + 1,
            cut_label,
            mc_pdf,
        )
        print(f"    Saved: {mc_pdf.name}")

        data_pdf: Path = output_dir / f"cut{cut_idx + 1}_data_{cut_name}.pdf"
        create_three_category_plot_data(
            data_hist_all,
            data_hist_pass,
            data_hist_fail,
            data_n_all,
            data_n_pass,
            data_n_fail,
            cut_idx + 1,
            cut_label,
            data_pdf,
        )
        print(f"    Saved: {data_pdf.name}")

    # Clean up
    print("\nCleaning up cached trees...")
    for state in SIGNAL_STATES:
        close_all_trees(mc_trees_cache[state])
    close_all_trees(data_trees_cache)

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"Generated {n_cuts * 2} plots ({n_cuts} MC + {n_cuts} Data)")
    print("Each plot shows three categories: All / Pass / Fail")
    print(f"Output directory: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
