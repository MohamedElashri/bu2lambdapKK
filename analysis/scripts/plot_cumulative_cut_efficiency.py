#!/usr/bin/env python3
"""
Plot M(LambdaPK) distribution after each cumulative cut.

For each cut, shows the distribution OVERLAID on the no-cuts baseline,
making it easy to see what each cut removes.

Creates separate plots for:
- MC (all 4 charmonium states)
- Data (in B+ signal region)

Also saves efficiency table to cuts_eff.txt.

Usage:
    cd analysis/scripts
    python plot_cumulative_cut_efficiency.py
    python plot_cumulative_cut_efficiency.py --years 2016,2017,2018
"""

import argparse
import sys
from pathlib import Path

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
    ROOT.gStyle.SetPadTopMargin(0.08)
    ROOT.gStyle.SetPadBottomMargin(0.12)
    ROOT.gStyle.SetTitleFont(132, "XYZ")
    ROOT.gStyle.SetLabelFont(132, "XYZ")
    ROOT.gStyle.SetTextFont(132)
    ROOT.gStyle.SetTitleSize(0.05, "XYZ")
    ROOT.gStyle.SetLabelSize(0.04, "XYZ")


def build_cumulative_cut_string(cut_index: int) -> str:
    """
    Build cut string for cuts 0 through cut_index (inclusive).

    Args:
        cut_index: Index of the last cut to include (-1 for no cuts)

    Returns:
        Cut string for TTree::Draw
    """
    if cut_index < 0:
        return ""
    cut_parts: list[str] = []
    for i in range(cut_index + 1):
        branch, cut_type, label, value = CUTS_ORDER[i]
        if cut_type == "greater":
            cut_parts.append(f"({branch} > {value})")
        else:
            cut_parts.append(f"({branch} < {value})")
    return " && ".join(cut_parts)


def load_mc_histogram(
    mc_path: Path,
    state: str,
    years: list[str],
    track_types: list[str],
    hist_name: str,
    cut_string: str,
) -> tuple[ROOT.TH1D, int]:
    """
    Load MC data and fill histogram with M(LambdaPK).

    Args:
        mc_path: Base path to MC files
        state: MC state name
        years: List of years
        track_types: List of track types
        hist_name: Name for histogram
        cut_string: Cut string to apply

    Returns:
        Tuple of (histogram, n_events)
    """
    hist: ROOT.TH1D = ROOT.TH1D(hist_name, "", N_BINS, M_LPKM_MIN, M_LPKM_MAX)
    hist.Sumw2()
    total_events: int = 0
    mass_formula: str = (
        "sqrt((L0_PE + p_PE + h2_PE)*(L0_PE + p_PE + h2_PE) - "
        "(L0_PX + p_PX + h2_PX)*(L0_PX + p_PX + h2_PX) - "
        "(L0_PY + p_PY + h2_PY)*(L0_PY + p_PY + h2_PY) - "
        "(L0_PZ + p_PZ + h2_PZ)*(L0_PZ + p_PZ + h2_PZ))"
    )
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
                    temp_hist_name: str = f"temp_{hist_name}_{year}_{magnet}_{track_type}"
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
                    tfile.Close()
                except Exception as e:
                    print(f"    Error: {e}")
    ROOT.SetOwnership(hist, False)
    return hist, total_events


def load_data_histogram(
    data_path: Path,
    years: list[str],
    track_types: list[str],
    hist_name: str,
    cut_string: str,
    bu_mass_min: float,
    bu_mass_max: float,
) -> tuple[ROOT.TH1D, int]:
    """
    Load data and fill histogram with M(LambdaPK) in B+ signal region.

    Args:
        data_path: Base path to data files
        years: List of years
        track_types: List of track types
        hist_name: Name for histogram
        cut_string: Cut string to apply
        bu_mass_min: B+ mass window min
        bu_mass_max: B+ mass window max

    Returns:
        Tuple of (histogram, n_events)
    """
    hist: ROOT.TH1D = ROOT.TH1D(hist_name, "", N_BINS, M_LPKM_MIN, M_LPKM_MAX)
    hist.Sumw2()
    total_events: int = 0
    mass_formula: str = (
        "sqrt((L0_PE + p_PE + h2_PE)*(L0_PE + p_PE + h2_PE) - "
        "(L0_PX + p_PX + h2_PX)*(L0_PX + p_PX + h2_PX) - "
        "(L0_PY + p_PY + h2_PY)*(L0_PY + p_PY + h2_PY) - "
        "(L0_PZ + p_PZ + h2_PZ)*(L0_PZ + p_PZ + h2_PZ))"
    )
    # Add B+ mass window cut
    bu_cut: str = f"(Bu_MM > {bu_mass_min}) && (Bu_MM < {bu_mass_max})"
    full_cut: str = bu_cut if not cut_string else f"{cut_string} && {bu_cut}"
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
                        f"{mass_formula}>>{temp_hist_name}({N_BINS},{M_LPKM_MIN},{M_LPKM_MAX})",
                        full_cut,
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


def create_combined_overlay_plot(
    mc_hist_no_cuts: dict[str, ROOT.TH1D],
    mc_hist_with_cut: dict[str, ROOT.TH1D],
    mc_events_no_cuts: dict[str, int],
    mc_events_with_cut: dict[str, int],
    data_hist_no_cuts: ROOT.TH1D,
    data_hist_with_cut: ROOT.TH1D,
    n_data_no_cuts: int,
    n_data_with_cut: int,
    cut_label: str,
    cut_idx: int,
    output_path: Path,
) -> None:
    """
    Create combined MC+Data plot with MC on left, Data on right.

    Args:
        mc_hist_no_cuts: Dict of {state: histogram} without cuts (MC)
        mc_hist_with_cut: Dict of {state: histogram} after cut (MC)
        mc_events_no_cuts: Dict of {state: n_events} without cuts (MC)
        mc_events_with_cut: Dict of {state: n_events} after cut (MC)
        data_hist_no_cuts: Data histogram without cuts
        data_hist_with_cut: Data histogram after cut
        n_data_no_cuts: Data events without cuts
        n_data_with_cut: Data events after cut
        cut_label: Label for the cut
        cut_idx: Cut index (1-7)
        output_path: Path to save PDF
    """
    keep_alive: list = []
    canvas: ROOT.TCanvas = ROOT.TCanvas(f"c_cut{cut_idx}", "", 1600, 700)
    ROOT.SetOwnership(canvas, False)
    keep_alive.append(canvas)
    # Create two pads
    pad1: ROOT.TPad = ROOT.TPad("pad1", "MC", 0.0, 0.0, 0.5, 1.0)
    pad2: ROOT.TPad = ROOT.TPad("pad2", "Data", 0.5, 0.0, 1.0, 1.0)
    ROOT.SetOwnership(pad1, False)
    ROOT.SetOwnership(pad2, False)
    keep_alive.extend([pad1, pad2])
    for pad in [pad1, pad2]:
        pad.SetLeftMargin(0.14)
        pad.SetRightMargin(0.05)
        pad.SetTopMargin(0.08)
        pad.SetBottomMargin(0.12)
    canvas.cd()
    pad1.Draw()
    pad2.Draw()
    # --- Left pad: MC ---
    pad1.cd()
    y_max_mc: float = 0.0
    for state in SIGNAL_STATES:
        y_max_mc = max(y_max_mc, mc_hist_no_cuts[state].GetMaximum())
    y_max_mc *= 1.4
    first_drawn: bool = False
    for state in SIGNAL_STATES:
        h_base: ROOT.TH1D = mc_hist_no_cuts[state].Clone(f"base_mc_{state}_{cut_idx}")
        ROOT.SetOwnership(h_base, False)
        keep_alive.append(h_base)
        h_base.SetLineColor(STATE_COLORS[state])
        h_base.SetLineWidth(2)
        h_base.SetLineStyle(2)
        h_base.SetFillColorAlpha(STATE_COLORS[state], 0.15)
        h_base.SetFillStyle(1001)
        h_base.SetMaximum(y_max_mc)
        h_base.SetMinimum(0)
        h_base.GetXaxis().SetTitle("M(#bar{#Lambda}pK^{#minus}) [MeV/#it{c}^{2}]")
        h_base.GetYaxis().SetTitle("Candidates")
        h_base.GetXaxis().SetTitleFont(132)
        h_base.GetYaxis().SetTitleFont(132)
        h_base.GetXaxis().SetLabelFont(132)
        h_base.GetYaxis().SetLabelFont(132)
        h_base.GetXaxis().SetTitleSize(0.045)
        h_base.GetYaxis().SetTitleSize(0.045)
        h_base.GetYaxis().SetTitleOffset(1.4)
        if not first_drawn:
            h_base.Draw("HIST")
            first_drawn = True
        else:
            h_base.Draw("HIST SAME")
    for state in SIGNAL_STATES:
        h_cut: ROOT.TH1D = mc_hist_with_cut[state]
        h_cut.SetLineColor(STATE_COLORS[state])
        h_cut.SetLineWidth(3)
        h_cut.SetLineStyle(1)
        h_cut.SetFillStyle(0)
        h_cut.Draw("HIST SAME")
    # MC Legend
    legend_mc: ROOT.TLegend = ROOT.TLegend(0.50, 0.52, 0.92, 0.88)
    ROOT.SetOwnership(legend_mc, False)
    keep_alive.append(legend_mc)
    legend_mc.SetBorderSize(0)
    legend_mc.SetFillStyle(0)
    legend_mc.SetTextFont(132)
    legend_mc.SetTextSize(0.030)
    legend_mc.SetHeader("Dashed=No cuts, Solid=After cut")
    for state in SIGNAL_STATES:
        n_init: int = mc_events_no_cuts[state]
        n_cut: int = mc_events_with_cut[state]
        eff: float = 100.0 * n_cut / n_init if n_init > 0 else 0.0
        legend_mc.AddEntry(mc_hist_with_cut[state], f"{STATE_LABELS[state]}: {eff:.1f}%", "l")
    legend_mc.Draw()
    # MC labels
    lhcb_mc: ROOT.TLatex = ROOT.TLatex()
    lhcb_mc.SetNDC()
    lhcb_mc.SetTextFont(132)
    lhcb_mc.SetTextSize(0.05)
    lhcb_mc.DrawLatex(0.18, 0.85, "LHCb MC")
    keep_alive.append(lhcb_mc)
    pad1.Modified()
    pad1.Update()
    # --- Right pad: Data ---
    pad2.cd()
    y_max_data: float = data_hist_no_cuts.GetMaximum() * 1.4
    h_base_data: ROOT.TH1D = data_hist_no_cuts.Clone(f"base_data_{cut_idx}")
    ROOT.SetOwnership(h_base_data, False)
    keep_alive.append(h_base_data)
    h_base_data.SetLineColor(ROOT.kBlue)
    h_base_data.SetLineWidth(2)
    h_base_data.SetLineStyle(2)
    h_base_data.SetFillColorAlpha(ROOT.kBlue, 0.15)
    h_base_data.SetFillStyle(1001)
    h_base_data.SetMaximum(y_max_data)
    h_base_data.SetMinimum(0)
    h_base_data.GetXaxis().SetTitle("M(#bar{#Lambda}pK^{#minus}) [MeV/#it{c}^{2}]")
    h_base_data.GetYaxis().SetTitle("Candidates")
    h_base_data.GetXaxis().SetTitleFont(132)
    h_base_data.GetYaxis().SetTitleFont(132)
    h_base_data.GetXaxis().SetLabelFont(132)
    h_base_data.GetYaxis().SetLabelFont(132)
    h_base_data.GetXaxis().SetTitleSize(0.045)
    h_base_data.GetYaxis().SetTitleSize(0.045)
    h_base_data.GetYaxis().SetTitleOffset(1.4)
    h_base_data.Draw("HIST")
    data_hist_with_cut.SetLineColor(ROOT.kRed)
    data_hist_with_cut.SetLineWidth(3)
    data_hist_with_cut.SetLineStyle(1)
    data_hist_with_cut.SetFillStyle(0)
    data_hist_with_cut.Draw("HIST SAME")
    # Data Legend
    legend_data: ROOT.TLegend = ROOT.TLegend(0.50, 0.65, 0.92, 0.88)
    ROOT.SetOwnership(legend_data, False)
    keep_alive.append(legend_data)
    legend_data.SetBorderSize(0)
    legend_data.SetFillStyle(0)
    legend_data.SetTextFont(132)
    legend_data.SetTextSize(0.032)
    eff_data: float = 100.0 * n_data_with_cut / n_data_no_cuts if n_data_no_cuts > 0 else 0.0
    legend_data.AddEntry(h_base_data, f"No cuts: {n_data_no_cuts:,}", "lf")
    legend_data.AddEntry(
        data_hist_with_cut, f"After cut: {n_data_with_cut:,} ({eff_data:.1f}%)", "l"
    )
    legend_data.Draw()
    # Data labels
    lhcb_data: ROOT.TLatex = ROOT.TLatex()
    lhcb_data.SetNDC()
    lhcb_data.SetTextFont(132)
    lhcb_data.SetTextSize(0.05)
    lhcb_data.DrawLatex(0.18, 0.85, "LHCb Data")
    keep_alive.append(lhcb_data)
    region: ROOT.TLatex = ROOT.TLatex()
    region.SetNDC()
    region.SetTextFont(132)
    region.SetTextSize(0.030)
    region.DrawLatex(0.18, 0.78, "B^{+} signal region")
    keep_alive.append(region)
    pad2.Modified()
    pad2.Update()
    # Add cut label at top center
    canvas.cd()
    cut_text: ROOT.TLatex = ROOT.TLatex()
    cut_text.SetNDC()
    cut_text.SetTextFont(132)
    cut_text.SetTextSize(0.035)
    cut_text.SetTextAlign(22)  # Center
    cut_text.DrawLatex(0.5, 0.96, f"Cut {cut_idx}: {cut_label}")
    keep_alive.append(cut_text)
    canvas.Modified()
    canvas.Update()
    canvas.SaveAs(str(output_path))
    canvas.SaveAs(str(output_path.with_suffix(".png")))
    print(f"  Saved: {output_path.name}")


def save_efficiency_table(
    mc_events: dict[str, list[int]],
    data_events: list[int],
    output_path: Path,
) -> None:
    """Save efficiency table to text file."""
    with open(output_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("CUMULATIVE CUT EFFICIENCY TABLE\n")
        f.write("=" * 100 + "\n\n")
        # MC table
        f.write("MC EFFICIENCY (Combined 2016-2018)\n")
        f.write("-" * 80 + "\n")
        header: str = f"{'Cut':<40}"
        for state in SIGNAL_STATES:
            header += f" {state:>10}"
        f.write(header + "\n")
        f.write("-" * 80 + "\n")
        row_labels: list[str] = ["No cuts"] + [cut[2] for cut in CUTS_ORDER]
        for i, label in enumerate(row_labels):
            row: str = f"{label:<40}"
            for state in SIGNAL_STATES:
                n_init: int = mc_events[state][0]
                n_curr: int = mc_events[state][i]
                eff: float = 100.0 * n_curr / n_init if n_init > 0 else 0.0
                row += f" {eff:>9.1f}%"
            f.write(row + "\n")
        f.write("\n")
        # Data table
        f.write("DATA EFFICIENCY (Combined 2016-2018, B+ signal region)\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Cut':<40} {'Efficiency':>10}\n")
        f.write("-" * 50 + "\n")
        for i, label in enumerate(row_labels):
            n_init: int = data_events[0]
            n_curr: int = data_events[i]
            eff: float = 100.0 * n_curr / n_init if n_init > 0 else 0.0
            f.write(f"{label:<40} {eff:>9.1f}%\n")
        f.write("=" * 100 + "\n")
    print(f"  Saved: {output_path.name}")


def main() -> None:
    """Main function."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Plot M(LambdaPK) after each cumulative cut for MC and Data"
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
    print("CUMULATIVE CUT EFFICIENCY PLOTTER")
    print("M(LambdaPK) Distribution: No Cuts vs After Each Cut")
    print("=" * 80)
    print(f"Years: {years}")
    print(f"Track types: {track_types}")
    print(f"MC States: {SIGNAL_STATES}")
    print("=" * 80)
    print("\nCuts applied in order:")
    for i, (branch, cut_type, label, value) in enumerate(CUTS_ORDER):
        op: str = ">" if cut_type == "greater" else "<"
        print(f"  {i+1}. {branch} {op} {value}")
    print()
    setup_lhcb_style()
    config: TOMLConfig = TOMLConfig(config_dir=str(ANALYSIS_DIR / "config"))
    mc_path: Path = Path(config.paths["mc"]["base_path"])
    data_path: Path = Path(config.paths["data"]["base_path"])
    # Get B+ signal region
    bu_fixed: dict = config.selection.get("bu_fixed_selection", {})
    bu_mass_min: float = bu_fixed.get("mass_corrected_min", 5255.0)
    bu_mass_max: float = bu_fixed.get("mass_corrected_max", 5305.0)
    print(f"B+ signal region for data: [{bu_mass_min:.0f}, {bu_mass_max:.0f}] MeV")
    output_dir: Path = ANALYSIS_DIR / "plots" / "cumulative_cuts"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Storage for MC histograms and event counts
    mc_histograms: dict[str, list[ROOT.TH1D]] = {s: [] for s in SIGNAL_STATES}
    mc_events: dict[str, list[int]] = {s: [] for s in SIGNAL_STATES}
    # Storage for Data histograms and event counts
    data_histograms: list[ROOT.TH1D] = []
    data_events: list[int] = []
    n_cuts: int = len(CUTS_ORDER)
    # Load MC histograms for each cut level
    print("\n" + "=" * 60)
    print("LOADING MC DATA")
    print("=" * 60)
    for cut_idx in range(-1, n_cuts):
        cut_label: str = "No cuts" if cut_idx < 0 else f"Cut {cut_idx + 1}"
        print(f"\n{cut_label}...")
        cut_string: str = build_cumulative_cut_string(cut_idx)
        for state in SIGNAL_STATES:
            hist, n_evt = load_mc_histogram(
                mc_path, state, years, track_types, f"h_mc_{state}_cut{cut_idx}", cut_string
            )
            mc_histograms[state].append(hist)
            mc_events[state].append(n_evt)
            print(f"  {STATE_LABELS[state]:12s}: {n_evt:>8,} events")
    # Load Data histograms for each cut level
    print("\n" + "=" * 60)
    print("LOADING DATA (B+ signal region)")
    print("=" * 60)
    for cut_idx in range(-1, n_cuts):
        cut_label: str = "No cuts" if cut_idx < 0 else f"Cut {cut_idx + 1}"
        print(f"\n{cut_label}...")
        cut_string: str = build_cumulative_cut_string(cut_idx)
        hist, n_evt = load_data_histogram(
            data_path,
            years,
            track_types,
            f"h_data_cut{cut_idx}",
            cut_string,
            bu_mass_min,
            bu_mass_max,
        )
        data_histograms.append(hist)
        data_events.append(n_evt)
        print(f"  Data: {n_evt:>8,} events")
    # Print efficiency tables
    print(f"\n{'=' * 80}")
    print("MC CUMULATIVE EFFICIENCY TABLE")
    print("=" * 80)
    header: str = f"{'Cut':<35}"
    for state in SIGNAL_STATES:
        header += f" {STATE_LABELS[state]:>10}"
    print(header)
    print("-" * 80)
    row_labels: list[str] = ["No cuts"] + [cut[2] for cut in CUTS_ORDER]
    for i, label in enumerate(row_labels):
        row: str = f"{label:<35}"
        for state in SIGNAL_STATES:
            n_initial: int = mc_events[state][0]
            n_current: int = mc_events[state][i]
            eff: float = 100.0 * n_current / n_initial if n_initial > 0 else 0.0
            row += f" {eff:>9.1f}%"
        print(row)
    print("=" * 80)
    print(f"\n{'=' * 60}")
    print("DATA CUMULATIVE EFFICIENCY TABLE")
    print("=" * 60)
    for i, label in enumerate(row_labels):
        n_initial: int = data_events[0]
        n_current: int = data_events[i]
        eff: float = 100.0 * n_current / n_initial if n_initial > 0 else 0.0
        print(f"{label:<35} {eff:>9.1f}%")
    print("=" * 60)
    # Save efficiency table to file
    print("\nSaving efficiency table...")
    table_path: Path = output_dir / "cuts_eff.txt"
    save_efficiency_table(mc_events, data_events, table_path)
    # Create combined MC+Data overlay plots for each cut
    print("\nCreating combined MC+Data overlay plots...")
    mc_no_cuts_hists: dict[str, ROOT.TH1D] = {s: mc_histograms[s][0] for s in SIGNAL_STATES}
    mc_no_cuts_events: dict[str, int] = {s: mc_events[s][0] for s in SIGNAL_STATES}
    data_no_cuts_hist: ROOT.TH1D = data_histograms[0]
    n_data_no_cuts: int = data_events[0]
    for cut_idx in range(n_cuts):
        cut_label: str = CUTS_ORDER[cut_idx][2]
        mc_cut_hists: dict[str, ROOT.TH1D] = {
            s: mc_histograms[s][cut_idx + 1] for s in SIGNAL_STATES
        }
        mc_cut_events: dict[str, int] = {s: mc_events[s][cut_idx + 1] for s in SIGNAL_STATES}
        data_cut_hist: ROOT.TH1D = data_histograms[cut_idx + 1]
        n_data_cut: int = data_events[cut_idx + 1]
        output_pdf: Path = output_dir / f"cut{cut_idx + 1}_overlay.pdf"
        create_combined_overlay_plot(
            mc_no_cuts_hists,
            mc_cut_hists,
            mc_no_cuts_events,
            mc_cut_events,
            data_no_cuts_hist,
            data_cut_hist,
            n_data_no_cuts,
            n_data_cut,
            cut_label,
            cut_idx + 1,
            output_pdf,
        )
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"Generated {n_cuts} combined overlay plots + efficiency table:")
    for i in range(n_cuts):
        print(f"  - cut{i + 1}_overlay.pdf")
    print("  - cuts_eff.txt")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
