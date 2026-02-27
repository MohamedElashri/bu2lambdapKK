"""
Standalone study: Cumulative Cut Efficiency on M(Λ̄pK⁻) distributions.

Ported from: archive/analysis/studies/feedback_dec2024/study1_revised_cumulative_cuts.py

What it does:
  For each cut stage (cut 1 through cut 7, applied cumulatively):
    - All events: baseline distribution (transparent fill)
    - Events passing all cuts up to this stage (solid colour)
    - Events failing at least one cut up to this stage (hatched fill)
  Separate multi-panel plots for MC (per signal state) and Data.
  Summary CSV with efficiency numbers per cut stage and state.

Improvements over original:
  - matplotlib + mplhep (LHCb style) instead of ROOT TCanvas/TH1
  - Multi-panel layout (one panel per cut stage) instead of individual canvases
  - Summary efficiency table as CSV output
  - Uses cached Step 2 data from the main pipeline (awkward arrays)
  - Proper axis labels with LaTeX

Snakemake injects:
  snakemake.params.config_dir
  snakemake.params.cache_dir
  snakemake.params.output_dir
  snakemake.output.mc_plot
  snakemake.output.data_plot
  snakemake.output.csv
"""

import sys
from pathlib import Path

# Ensure the project root (analysis/) is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from modules.cache_manager import CacheManager
from modules.data_handler import TOMLConfig
from modules.exceptions import AnalysisError

# ---------------------------------------------------------------------------
# Read Snakemake params
# ---------------------------------------------------------------------------
config_dir = snakemake.params.config_dir  # noqa: F821
cache_dir = snakemake.params.cache_dir  # noqa: F821
output_dir = snakemake.params.output_dir  # noqa: F821
mc_plot_path = snakemake.output.mc_plot  # noqa: F821
data_plot_path = snakemake.output.data_plot  # noqa: F821
summary_csv_path = snakemake.output.csv  # noqa: F821

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
plt.style.use(hep.style.LHCb2)

config = TOMLConfig(config_dir)
cache = CacheManager(cache_dir)

# Signal MC states (matching main pipeline convention)
MC_STATES = ["jpsi", "etac", "chic0", "chic1"]

STATE_LABELS = {
    "jpsi": r"$J/\psi$",
    "etac": r"$\eta_c(1S)$",
    "chic0": r"$\chi_{c0}$",
    "chic1": r"$\chi_{c1}$",
}

STATE_COLORS = {
    "jpsi": "#1f77b4",
    "etac": "#d62728",
    "chic0": "#2ca02c",
    "chic1": "#ff7f0e",
}

# M(Λ̄pK⁻) histogram range
M_LPKM_MIN = 2800.0
M_LPKM_MAX = 4000.0
N_BINS = 120

# Ordered list of cuts to apply cumulatively
# (branch_name, cut_type, latex_label, value)
# These match the manual_cuts in config/selection.toml
CUTS_ORDER = [
    ("Bu_DTF_chi2", "less", r"$\chi^{2}_{\mathrm{DTF}}(B^{+}) < 30$", 30.0),
    ("Bu_FDCHI2_OWNPV", "greater", r"$\mathrm{FD}\chi^{2}(B^{+}) > 100$", 100.0),
    ("Bu_IPCHI2_OWNPV", "less", r"$\mathrm{IP}\chi^{2}(B^{+}) < 10$", 10.0),
    ("Bu_PT", "greater", r"$p_{\mathrm{T}}(B^{+}) > 3\,\mathrm{GeV}$", 3000.0),
    ("h1_ProbNNk", "greater", r"$\mathrm{ProbNN}_{K}(K^{+}) > 0.1$", 0.1),
    ("h2_ProbNNk", "greater", r"$\mathrm{ProbNN}_{K}(K^{-}) > 0.1$", 0.1),
    ("p_ProbNNp", "greater", r"$\mathrm{ProbNN}_{p}(p) > 0.1$", 0.1),
]


# ---------------------------------------------------------------------------
# Helper: compute step dependencies (same pattern as fom_comparison)
# ---------------------------------------------------------------------------
def compute_step_dependencies(step, extra_params=None):
    config_files = list(Path(config_dir).glob("*.toml"))
    code_files = []
    if step == "2":
        code_files = [
            project_root / "modules" / "data_handler.py",
            project_root / "modules" / "lambda_selector.py",
        ]
    return cache.compute_dependencies(
        config_files=config_files, code_files=code_files, extra_params=extra_params
    )


# ---------------------------------------------------------------------------
# Helper: build cumulative cut mask
# ---------------------------------------------------------------------------
def _get_flat_branch(events, branch_name):
    """Get a branch as a flat (non-jagged) array, using ak.firsts if needed."""
    br = events[branch_name]
    if "var" in str(ak.type(br)):
        br = ak.firsts(br)
    return br


def build_cumulative_mask(events, cut_index):
    """
    Build a boolean mask for events passing all cuts up to cut_index (inclusive).

    Args:
        events: Awkward array with branch fields
        cut_index: Apply cuts 0..cut_index (inclusive)

    Returns:
        Boolean awkward array mask
    """
    ref_branch = _get_flat_branch(events, "Bu_MM_corrected")
    mask = ak.ones_like(ref_branch, dtype=bool)
    for i in range(cut_index + 1):
        branch, cut_type, _, value = CUTS_ORDER[i]
        if branch not in events.fields:
            print(f"  WARNING: branch '{branch}' not found, skipping cut {i}")
            continue
        br = _get_flat_branch(events, branch)
        if cut_type == "greater":
            mask = mask & (br > value)
        else:
            mask = mask & (br < value)
    return mask


# ---------------------------------------------------------------------------
# Load Step 2 cached data
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("CUMULATIVE CUT EFFICIENCY STUDY")
print("=" * 80)

years = snakemake.config.get("years", ["2016", "2017", "2018"])  # noqa: F821
track_types = snakemake.config.get("track_types", ["LL", "DD"])  # noqa: F821

step2_deps = compute_step_dependencies(
    step="2", extra_params={"years": years, "track_types": track_types}
)

# --- Real data ---
data_dict = cache.load("step2_data_after_lambda", dependencies=step2_deps)
if data_dict is None:
    raise AnalysisError(
        "Step 2 cached data not found! Run Step 2 (load_data) first.\n"
        "  cd analysis/ && uv run snakemake load_data -j1"
    )

# --- Signal MC ---
mc_dict = cache.load("step2_mc_after_lambda", dependencies=step2_deps)
if mc_dict is None:
    raise AnalysisError(
        "Step 2 cached MC not found! Run Step 2 (load_data) first.\n"
        "  cd analysis/ && uv run snakemake load_data -j1"
    )

# Combine data across years and track types
print("\n[Loading real data]")
data_arrays = []
for year in data_dict:
    for track_type in data_dict[year]:
        arr = data_dict[year][track_type]
        if hasattr(arr, "layout"):
            data_arrays.append(arr)
all_data = ak.concatenate(data_arrays, axis=0)
print(f"  Total data (after Lambda cuts): {len(all_data):,}")

# Combine MC per state
print("\n[Loading signal MC]")
mc_combined = {}
for state in MC_STATES:
    if state not in mc_dict:
        print(f"  WARNING: No MC for {state}, skipping")
        continue
    arrays = []
    for year in mc_dict[state]:
        for track_type in mc_dict[state][year]:
            arr = mc_dict[state][year][track_type]
            if hasattr(arr, "layout"):
                arrays.append(arr)
    if arrays:
        mc_combined[state] = ak.concatenate(arrays, axis=0)
        print(f"  MC/{state}: {len(mc_combined[state]):,} events")

# Identify mass branch
mass_branch = "M_LpKm_h2" if "M_LpKm_h2" in all_data.fields else "M_LpKm"
bu_mass_branch = "Bu_MM_corrected" if "Bu_MM_corrected" in all_data.fields else "Bu_M"
print(f"\n  Mass branch: {mass_branch}")
print(f"  B+ mass branch: {bu_mass_branch}")

# B+ signal region for data
bu_fixed = config.selection.get("bu_fixed_selection", {})
bu_mass_min = bu_fixed.get("mass_corrected_min", 5255.0)
bu_mass_max = bu_fixed.get("mass_corrected_max", 5305.0)
print(f"  B+ signal region: [{bu_mass_min:.0f}, {bu_mass_max:.0f}] MeV")

# Apply B+ mass cut to data
bu_mass = all_data[bu_mass_branch]
data_in_signal = all_data[(bu_mass > bu_mass_min) & (bu_mass < bu_mass_max)]
print(f"  Data in B+ signal region: {len(data_in_signal):,}")

# ---------------------------------------------------------------------------
# Compute cumulative cut efficiencies
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("COMPUTING CUMULATIVE CUT EFFICIENCIES")
print("=" * 80)

n_cuts = len(CUTS_ORDER)
bin_edges = np.linspace(M_LPKM_MIN, M_LPKM_MAX, N_BINS + 1)

# Storage for summary table
summary_rows = []

# Storage for histograms: mc_hists[state][cut_idx] = {"all": counts, "pass": counts, "fail": counts}
mc_hists = {state: [] for state in mc_combined}
mc_counts = {state: [] for state in mc_combined}

# Data histograms: data_hists[cut_idx] = {"all": counts, "pass": counts, "fail": counts}
data_hists = []
data_counts = []

# Pre-extract flat mass arrays (avoid repeated jagged→numpy conversion)
mc_mass_flat = {}
for state in mc_combined:
    mc_mass_flat[state] = np.asarray(ak.to_numpy(_get_flat_branch(mc_combined[state], mass_branch)))
data_mass_flat = np.asarray(ak.to_numpy(_get_flat_branch(data_in_signal, mass_branch)))

for cut_idx in range(n_cuts):
    branch_name = CUTS_ORDER[cut_idx][0]
    cut_label = CUTS_ORDER[cut_idx][2]
    print(f"\n  Cut {cut_idx + 1}/{n_cuts}: {branch_name}")

    # Build cumulative mask
    # MC
    for state in mc_combined:
        mc_mass = mc_mass_flat[state]

        h_all, _ = np.histogram(mc_mass, bins=bin_edges)
        n_all = len(mc_mass)

        pass_mask = np.asarray(ak.to_numpy(build_cumulative_mask(mc_combined[state], cut_idx)))
        h_pass, _ = np.histogram(mc_mass[pass_mask], bins=bin_edges)
        n_pass = int(pass_mask.sum())

        fail_mask = ~pass_mask
        h_fail, _ = np.histogram(mc_mass[fail_mask], bins=bin_edges)
        n_fail = int(fail_mask.sum())

        mc_hists[state].append({"all": h_all, "pass": h_pass, "fail": h_fail})
        mc_counts[state].append({"all": n_all, "pass": n_pass, "fail": n_fail})

        eff = 100.0 * n_pass / n_all if n_all > 0 else 0.0
        print(
            f"    MC {state:6s}: All={n_all:>8,}  Pass={n_pass:>8,} ({eff:.1f}%)  Fail={n_fail:>8,}"
        )

        summary_rows.append(
            {
                "cut_index": cut_idx + 1,
                "cut_branch": branch_name,
                "cut_label": CUTS_ORDER[cut_idx][2],
                "sample": f"MC_{state}",
                "n_all": n_all,
                "n_pass": n_pass,
                "n_fail": n_fail,
                "efficiency_pct": round(eff, 2),
            }
        )

    # Data
    h_all, _ = np.histogram(data_mass_flat, bins=bin_edges)
    n_all = len(data_mass_flat)

    pass_mask = np.asarray(ak.to_numpy(build_cumulative_mask(data_in_signal, cut_idx)))
    h_pass, _ = np.histogram(data_mass_flat[pass_mask], bins=bin_edges)
    n_pass = int(pass_mask.sum())

    fail_mask = ~pass_mask
    h_fail, _ = np.histogram(data_mass_flat[fail_mask], bins=bin_edges)
    n_fail = int(fail_mask.sum())

    data_hists.append({"all": h_all, "pass": h_pass, "fail": h_fail})
    data_counts.append({"all": n_all, "pass": n_pass, "fail": n_fail})

    eff = 100.0 * n_pass / n_all if n_all > 0 else 0.0
    print(f"    Data    : All={n_all:>8,}  Pass={n_pass:>8,} ({eff:.1f}%)  Fail={n_fail:>8,}")

    summary_rows.append(
        {
            "cut_index": cut_idx + 1,
            "cut_branch": branch_name,
            "cut_label": CUTS_ORDER[cut_idx][2],
            "sample": "Data",
            "n_all": n_all,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "efficiency_pct": round(eff, 2),
        }
    )

# ---------------------------------------------------------------------------
# Save summary CSV
# ---------------------------------------------------------------------------
output_path = Path(summary_csv_path)
output_path.parent.mkdir(exist_ok=True, parents=True)
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(output_path, index=False)
print(f"\n  Summary CSV saved to {summary_csv_path}")

# ---------------------------------------------------------------------------
# MC Plots: multi-panel layout (one panel per cut stage, all states overlaid)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("GENERATING MC PLOTS")
print("=" * 80)

bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_width = bin_edges[1] - bin_edges[0]

# Layout: 4 rows x 2 cols = 8 panels (7 cuts + 1 summary)
n_rows, n_cols = 4, 2

mc_plot = Path(mc_plot_path)
mc_plot.parent.mkdir(exist_ok=True, parents=True)

with PdfPages(mc_plot) as pdf:
    # --- Page 1: M(Λ̄pK⁻) distributions per cut stage ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 20))
    fig.suptitle(
        r"Cumulative Cut Efficiency — Signal MC — $M(\bar{\Lambda}pK^{-})$",
        fontsize=18,
        y=0.995,
    )

    for cut_idx in range(n_cuts):
        row, col = divmod(cut_idx, n_cols)
        ax = axes[row, col]

        # Plot each state
        for state in mc_combined:
            h_all = mc_hists[state][cut_idx]["all"]
            h_pass = mc_hists[state][cut_idx]["pass"]
            h_fail = mc_hists[state][cut_idx]["fail"]
            n_a = mc_counts[state][cut_idx]["all"]
            n_p = mc_counts[state][cut_idx]["pass"]
            eff = 100.0 * n_p / n_a if n_a > 0 else 0.0

            color = STATE_COLORS[state]

            # All events: filled, transparent
            ax.fill_between(
                bin_centers,
                h_all,
                step="mid",
                alpha=0.15,
                color=color,
            )
            ax.step(
                bin_centers,
                h_all,
                where="mid",
                color=color,
                linewidth=1.0,
                linestyle="--",
                alpha=0.5,
            )

            # Pass events: solid line
            ax.step(
                bin_centers,
                h_pass,
                where="mid",
                color=color,
                linewidth=2.0,
                label=f"{STATE_LABELS[state]}: {eff:.1f}%",
            )

            # Fail events: hatched
            ax.fill_between(
                bin_centers,
                h_fail,
                step="mid",
                alpha=0.08,
                color="red",
                hatch="///",
            )

        ax.set_xlim(M_LPKM_MIN, M_LPKM_MAX)
        ax.set_ylim(0, None)
        ax.set_xlabel(r"$M(\bar{\Lambda}pK^{-})$ [MeV/$c^{2}$]", fontsize=12)
        ax.set_ylabel(f"Candidates / ({bin_width:.0f} MeV/$c^{{2}}$)", fontsize=12)
        ax.set_title(f"Cut {cut_idx + 1}: {CUTS_ORDER[cut_idx][2]}", fontsize=12)
        ax.legend(fontsize=8, loc="upper right", title="Efficiency", title_fontsize=8)

    # Summary panel: efficiency vs cut stage
    ax_summary = axes[n_rows - 1, n_cols - 1]
    cut_labels_short = [f"Cut {i+1}" for i in range(n_cuts)]
    x_pos = np.arange(n_cuts)

    for state in mc_combined:
        effs = []
        for cut_idx in range(n_cuts):
            n_a = mc_counts[state][cut_idx]["all"]
            n_p = mc_counts[state][cut_idx]["pass"]
            effs.append(100.0 * n_p / n_a if n_a > 0 else 0.0)
        ax_summary.plot(
            x_pos,
            effs,
            "o-",
            color=STATE_COLORS[state],
            label=STATE_LABELS[state],
            linewidth=2,
            markersize=5,
        )

    ax_summary.set_xticks(x_pos)
    ax_summary.set_xticklabels(cut_labels_short, fontsize=8, rotation=45, ha="right")
    ax_summary.set_ylabel("Cumulative Efficiency [%]", fontsize=12)
    ax_summary.set_title("Efficiency vs Cut Stage", fontsize=12)
    ax_summary.legend(fontsize=9, loc="lower left")
    ax_summary.set_ylim(0, 105)
    ax_summary.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 2: Legend explanation + per-cut details table ---
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    ax.set_title(
        "Cumulative Cut Efficiency — MC Summary Table",
        fontsize=14,
        pad=20,
    )

    # Build table
    col_labels = ["Cut", "Variable"]
    for state in mc_combined:
        col_labels.append(f"{STATE_LABELS[state]} eff.")
    col_labels.append("Description")

    table_data = []
    for cut_idx in range(n_cuts):
        branch_name, cut_type, label, value = CUTS_ORDER[cut_idx]
        direction = "<" if cut_type == "less" else ">"
        row = [f"{cut_idx + 1}", f"{branch_name} {direction} {value}"]
        for state in mc_combined:
            n_a = mc_counts[state][cut_idx]["all"]
            n_p = mc_counts[state][cut_idx]["pass"]
            eff = 100.0 * n_p / n_a if n_a > 0 else 0.0
            row.append(f"{eff:.1f}%")
        row.append(label)
        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # Colour header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=8)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print(f"  MC plots saved to {mc_plot_path}")

# ---------------------------------------------------------------------------
# Data Plots: multi-panel layout (one panel per cut stage)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("GENERATING DATA PLOTS")
print("=" * 80)

data_plot = Path(data_plot_path)
data_plot.parent.mkdir(exist_ok=True, parents=True)

with PdfPages(data_plot) as pdf:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 20))
    fig.suptitle(
        r"Cumulative Cut Efficiency — Data ($B^{+}$ signal region) — $M(\bar{\Lambda}pK^{-})$",
        fontsize=16,
        y=0.995,
    )

    for cut_idx in range(n_cuts):
        row, col = divmod(cut_idx, n_cols)
        ax = axes[row, col]

        h_all = data_hists[cut_idx]["all"]
        h_pass = data_hists[cut_idx]["pass"]
        h_fail = data_hists[cut_idx]["fail"]
        n_a = data_counts[cut_idx]["all"]
        n_p = data_counts[cut_idx]["pass"]
        n_f = data_counts[cut_idx]["fail"]
        eff = 100.0 * n_p / n_a if n_a > 0 else 0.0

        # All events: gray fill
        ax.fill_between(
            bin_centers,
            h_all,
            step="mid",
            alpha=0.25,
            color="gray",
            label=f"All: {n_a:,}",
        )
        ax.step(
            bin_centers,
            h_all,
            where="mid",
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.5,
        )

        # Pass events: blue solid
        ax.step(
            bin_centers,
            h_pass,
            where="mid",
            color="#1f77b4",
            linewidth=2.0,
            label=f"Pass: {n_p:,} ({eff:.1f}%)",
        )

        # Fail events: red hatched
        ax.fill_between(
            bin_centers,
            h_fail,
            step="mid",
            alpha=0.15,
            color="red",
            hatch="///",
            label=f"Fail: {n_f:,}",
        )

        ax.set_xlim(M_LPKM_MIN, M_LPKM_MAX)
        ax.set_ylim(0, None)
        ax.set_xlabel(r"$M(\bar{\Lambda}pK^{-})$ [MeV/$c^{2}$]", fontsize=12)
        ax.set_ylabel(f"Candidates / ({bin_width:.0f} MeV/$c^{{2}}$)", fontsize=12)
        ax.set_title(f"Cut {cut_idx + 1}: {CUTS_ORDER[cut_idx][2]}", fontsize=12)
        ax.legend(fontsize=9, loc="upper right")

    # Summary panel: efficiency vs cut stage
    ax_summary = axes[n_rows - 1, n_cols - 1]
    effs_data = []
    for cut_idx in range(n_cuts):
        n_a = data_counts[cut_idx]["all"]
        n_p = data_counts[cut_idx]["pass"]
        effs_data.append(100.0 * n_p / n_a if n_a > 0 else 0.0)

    ax_summary.plot(
        x_pos,
        effs_data,
        "o-",
        color="#1f77b4",
        label="Data",
        linewidth=2,
        markersize=6,
    )

    # Also overlay MC average for comparison
    mc_avg_effs = []
    for cut_idx in range(n_cuts):
        state_effs = []
        for state in mc_combined:
            n_a = mc_counts[state][cut_idx]["all"]
            n_p = mc_counts[state][cut_idx]["pass"]
            if n_a > 0:
                state_effs.append(100.0 * n_p / n_a)
        mc_avg_effs.append(np.mean(state_effs) if state_effs else 0.0)

    ax_summary.plot(
        x_pos,
        mc_avg_effs,
        "s--",
        color="gray",
        label="MC (avg)",
        linewidth=1.5,
        markersize=5,
        alpha=0.7,
    )

    ax_summary.set_xticks(x_pos)
    ax_summary.set_xticklabels(cut_labels_short, fontsize=8, rotation=45, ha="right")
    ax_summary.set_ylabel("Cumulative Efficiency [%]", fontsize=12)
    ax_summary.set_title("Efficiency vs Cut Stage", fontsize=12)
    ax_summary.legend(fontsize=10, loc="lower left")
    ax_summary.set_ylim(0, 105)
    ax_summary.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print(f"  Data plots saved to {data_plot_path}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("CUMULATIVE CUT EFFICIENCY STUDY COMPLETE")
print("=" * 80)
print("\n  Outputs:")
print(f"    {mc_plot_path}")
print(f"    {data_plot_path}")
print(f"    {summary_csv_path}")
print(f"\n  {n_cuts} cut stages, {len(mc_combined)} MC states + Data")
print("=" * 80)
