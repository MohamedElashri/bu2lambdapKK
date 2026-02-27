"""
Standalone study: Individual Cut Efficiency Analysis.

Ported from: archive/analysis/studies/feedback_dec2024/study4_individual_cut_efficiency.py

What it does:
  Shows the efficiency of each selection cut applied INDIVIDUALLY (not cumulatively):
    - For each cut variable, applies only that cut to the full dataset
    - Shows pass/fail fractions per cut
    - Final entry shows cumulative effect of all cuts together
    - Bar chart summary comparing individual vs cumulative efficiency

  Performed for both MC (per signal state) and Data.

Improvements over original:
  - ROOT THStack/TCanvas → matplotlib + mplhep (LHCb style)
  - Horizontal bar chart with pass/fail stacked bars
  - Numerical labels on bars
  - Color-coded by cut type (PID, kinematic, vertex quality)
  - Summary table as CSV
  - MC vs Data comparison on same page
  - Uses cached Step 2 data from the main pipeline (awkward arrays)

Snakemake injects:
  snakemake.params.config_dir
  snakemake.params.cache_dir
  snakemake.params.output_dir
  snakemake.output.plot
  snakemake.output.csv
"""

import sys
from pathlib import Path

# Ensure the project root (analysis/) is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import awkward as ak  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from modules.cache_manager import CacheManager  # noqa: E402
from modules.data_handler import TOMLConfig  # noqa: E402
from modules.exceptions import AnalysisError  # noqa: E402

# ---------------------------------------------------------------------------
# Read Snakemake params
# ---------------------------------------------------------------------------
config_dir = snakemake.params.config_dir  # noqa: F821
cache_dir = snakemake.params.cache_dir  # noqa: F821
output_dir = snakemake.params.output_dir  # noqa: F821
plot_path = snakemake.output.plot  # noqa: F821
csv_path = snakemake.output.csv  # noqa: F821

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
plt.style.use(hep.style.LHCb2)

config = TOMLConfig(config_dir)
cache = CacheManager(cache_dir)

# MC states
MC_STATES = ["jpsi", "etac", "chic0", "chic1"]

STATE_LABELS = {
    "jpsi": r"$J/\psi$",
    "etac": r"$\eta_c$",
    "chic0": r"$\chi_{c0}$",
    "chic1": r"$\chi_{c1}$",
}

STATE_COLORS = {
    "jpsi": "#1f77b4",
    "etac": "#d62728",
    "chic0": "#2ca02c",
    "chic1": "#ff7f0e",
}

# Selection cuts with LaTeX labels and cut-type categories
CUTS_ORDER = [
    ("Bu_DTF_chi2", "less", 30.0, r"$\chi^{2}_{\mathrm{DTF}}(B^{+}) < 30$", "vertex"),
    ("Bu_FDCHI2_OWNPV", "greater", 100.0, r"$\mathrm{FD}\chi^{2}(B^{+}) > 100$", "vertex"),
    ("Bu_IPCHI2_OWNPV", "less", 10.0, r"$\mathrm{IP}\chi^{2}(B^{+}) < 10$", "vertex"),
    ("Bu_PT", "greater", 3000.0, r"$p_{\mathrm{T}}(B^{+}) > 3\,\mathrm{GeV}$", "kinematic"),
    ("h1_ProbNNk", "greater", 0.1, r"$\mathrm{ProbNN}_{K}(K^{+}) > 0.1$", "PID"),
    ("h2_ProbNNk", "greater", 0.1, r"$\mathrm{ProbNN}_{K}(K^{-}) > 0.1$", "PID"),
    ("p_ProbNNp", "greater", 0.1, r"$\mathrm{ProbNN}_{p}(p) > 0.1$", "PID"),
]

# Colours by cut category
CATEGORY_COLORS = {
    "vertex": "#4472C4",
    "kinematic": "#ED7D31",
    "PID": "#70AD47",
}


# ---------------------------------------------------------------------------
# Helpers
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


def _get_flat_branch(events, branch_name):
    """Get a branch as a flat (non-jagged) array, using ak.firsts if needed."""
    br = events[branch_name]
    if "var" in str(ak.type(br)):
        br = ak.firsts(br)
    return br


def apply_single_cut(events, branch, cut_type, value):
    """Apply a single cut and return (pass_mask, fail_mask)."""
    if branch not in events.fields:
        print(f"  WARNING: branch '{branch}' not found, returning all-pass")
        ref = _get_flat_branch(events, list(events.fields)[0])
        return ak.ones_like(ref, dtype=bool), ak.zeros_like(ref, dtype=bool)

    br = _get_flat_branch(events, branch)
    if cut_type == "greater":
        pass_mask = br > value
        fail_mask = br <= value
    else:
        pass_mask = br < value
        fail_mask = br >= value
    return pass_mask, fail_mask


def build_all_cuts_mask(events):
    """Build boolean mask for events passing ALL selection cuts."""
    ref = _get_flat_branch(events, list(events.fields)[0])
    mask = ak.ones_like(ref, dtype=bool)
    for branch, cut_type, value, _, _ in CUTS_ORDER:
        if branch not in events.fields:
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
print("INDIVIDUAL CUT EFFICIENCY STUDY")
print("=" * 80)
print("Each cut applied INDEPENDENTLY to full dataset")

years = snakemake.config.get("years", ["2016", "2017", "2018"])  # noqa: F821
track_types = snakemake.config.get("track_types", ["LL", "DD"])  # noqa: F821

step2_deps = compute_step_dependencies(
    step="2", extra_params={"years": years, "track_types": track_types}
)

data_dict = cache.load("step2_data_after_lambda", dependencies=step2_deps)
if data_dict is None:
    raise AnalysisError(
        "Step 2 cached data not found! Run Step 2 (load_data) first.\n"
        "  cd analysis/ && uv run snakemake load_data -j1"
    )

mc_dict = cache.load("step2_mc_after_lambda", dependencies=step2_deps)
if mc_dict is None:
    raise AnalysisError(
        "Step 2 cached MC not found! Run Step 2 (load_data) first.\n"
        "  cd analysis/ && uv run snakemake load_data -j1"
    )

# Combine data
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

# ---------------------------------------------------------------------------
# Compute individual cut efficiencies
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PROCESSING INDIVIDUAL CUTS")
print("=" * 80)

# Structure: rows for CSV, keyed by (sample, cut_label)
summary_rows = []


def process_sample(events, sample_name, n_total):
    """Process all individual cuts for one sample."""
    rows = []

    for branch, cut_type, value, label, category in CUTS_ORDER:
        pass_mask, fail_mask = apply_single_cut(events, branch, cut_type, value)
        n_pass = int(ak.sum(pass_mask))
        n_fail = int(ak.sum(fail_mask))
        eff_pass = 100.0 * n_pass / n_total if n_total > 0 else 0.0
        eff_fail = 100.0 * n_fail / n_total if n_total > 0 else 0.0

        # Strip LaTeX for CSV
        clean_label = (
            label.replace("$", "")
            .replace("\\", "")
            .replace("mathrm{", "")
            .replace("{", "")
            .replace("}", "")
            .replace(",", "")
        )

        print(f"  {clean_label:40s}  Pass={eff_pass:5.1f}%  Fail={eff_fail:5.1f}%")

        rows.append(
            {
                "sample": sample_name,
                "cut_branch": branch,
                "cut_label": clean_label,
                "cut_category": category,
                "n_total": n_total,
                "n_pass": n_pass,
                "n_fail": n_fail,
                "efficiency_pass_pct": round(eff_pass, 2),
                "efficiency_fail_pct": round(eff_fail, 2),
                "is_cumulative": False,
            }
        )

    # Cumulative (all cuts)
    cum_mask = build_all_cuts_mask(events)
    n_cum = int(ak.sum(cum_mask))
    eff_cum = 100.0 * n_cum / n_total if n_total > 0 else 0.0
    print(f"  {'ALL CUTS COMBINED':40s}  Pass={eff_cum:5.1f}%")

    rows.append(
        {
            "sample": sample_name,
            "cut_branch": "ALL",
            "cut_label": "All cuts combined",
            "cut_category": "cumulative",
            "n_total": n_total,
            "n_pass": n_cum,
            "n_fail": n_total - n_cum,
            "efficiency_pass_pct": round(eff_cum, 2),
            "efficiency_fail_pct": round(100.0 - eff_cum, 2),
            "is_cumulative": True,
        }
    )

    return rows


# MC
for state in mc_combined:
    print(f"\n  --- MC: {state} ---")
    n_total = len(mc_combined[state])
    rows = process_sample(mc_combined[state], f"MC_{state}", n_total)
    summary_rows.extend(rows)

# Data
print("\n  --- Data ---")
n_total_data = len(all_data)
rows = process_sample(all_data, "Data", n_total_data)
summary_rows.extend(rows)

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
csv_out = Path(csv_path)
csv_out.parent.mkdir(exist_ok=True, parents=True)
df = pd.DataFrame(summary_rows)
df.to_csv(csv_out, index=False)
print(f"\n  Summary CSV saved to {csv_path}")

# ---------------------------------------------------------------------------
# Generate plots
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("GENERATING PLOTS")
print("=" * 80)

plot_out = Path(plot_path)
plot_out.parent.mkdir(exist_ok=True, parents=True)

# Prepare cut labels (short) for y-axis
cut_labels_short = [label for _, _, _, label, _ in CUTS_ORDER] + ["All cuts"]
cut_categories = [cat for _, _, _, _, cat in CUTS_ORDER] + ["cumulative"]
n_entries = len(cut_labels_short)
y_pos = np.arange(n_entries)

with PdfPages(plot_out) as pdf:
    # --- Page 1: MC per-state horizontal bar charts ---
    n_states = len(mc_combined)
    fig, axes = plt.subplots(1, n_states, figsize=(6 * n_states, 8), sharey=True)
    if n_states == 1:
        axes = [axes]

    for ax, state in zip(axes, mc_combined):
        state_rows = [r for r in summary_rows if r["sample"] == f"MC_{state}"]
        pass_effs = [r["efficiency_pass_pct"] for r in state_rows]
        fail_effs = [r["efficiency_fail_pct"] for r in state_rows]

        # Bar colours by category
        bar_colors = [CATEGORY_COLORS.get(cat, "#888888") for cat in cut_categories]

        bars_pass = ax.barh(
            y_pos, pass_effs, color=bar_colors, alpha=0.8, edgecolor="black", linewidth=0.5
        )
        bars_fail = ax.barh(
            y_pos,
            fail_effs,
            left=pass_effs,
            color="lightgray",
            alpha=0.6,
            edgecolor="black",
            linewidth=0.5,
        )

        # Numerical labels
        for i, (p, f) in enumerate(zip(pass_effs, fail_effs)):
            ax.text(p / 2, i, f"{p:.1f}%", ha="center", va="center", fontsize=7, fontweight="bold")
            if f > 5:
                ax.text(
                    p + f / 2, i, f"{f:.1f}%", ha="center", va="center", fontsize=7, color="gray"
                )

        ax.set_xlim(0, 105)
        ax.set_xlabel("Fraction [%]", fontsize=11)
        ax.set_title(f"MC {STATE_LABELS[state]}", fontsize=13)
        ax.axvline(100, color="black", linewidth=0.5, alpha=0.3)

        if ax == axes[0]:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(cut_labels_short, fontsize=8)
        ax.invert_yaxis()

    # Add legend for categories
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS["vertex"], edgecolor="black", label="Vertex quality"),
        Patch(facecolor=CATEGORY_COLORS["kinematic"], edgecolor="black", label="Kinematic"),
        Patch(facecolor=CATEGORY_COLORS["PID"], edgecolor="black", label="PID"),
        Patch(facecolor="lightgray", edgecolor="black", label="Fail"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10, frameon=False)

    fig.suptitle("Individual Cut Efficiency — MC Signal States", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    pdf.savefig(fig)
    plt.close()

    # --- Page 2: Data horizontal bar chart ---
    fig, ax = plt.subplots(figsize=(10, 8))

    data_rows = [r for r in summary_rows if r["sample"] == "Data"]
    pass_effs = [r["efficiency_pass_pct"] for r in data_rows]
    fail_effs = [r["efficiency_fail_pct"] for r in data_rows]

    bar_colors = [CATEGORY_COLORS.get(cat, "#888888") for cat in cut_categories]

    bars_pass = ax.barh(
        y_pos, pass_effs, color=bar_colors, alpha=0.8, edgecolor="black", linewidth=0.5
    )
    bars_fail = ax.barh(
        y_pos,
        fail_effs,
        left=pass_effs,
        color="lightgray",
        alpha=0.6,
        edgecolor="black",
        linewidth=0.5,
    )

    for i, (p, f) in enumerate(zip(pass_effs, fail_effs)):
        ax.text(p / 2, i, f"{p:.1f}%", ha="center", va="center", fontsize=9, fontweight="bold")
        if f > 5:
            ax.text(p + f / 2, i, f"{f:.1f}%", ha="center", va="center", fontsize=9, color="gray")

    ax.set_xlim(0, 105)
    ax.set_xlabel("Fraction [%]", fontsize=12)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cut_labels_short, fontsize=10)
    ax.set_title("Individual Cut Efficiency — Data", fontsize=14)
    ax.axvline(100, color="black", linewidth=0.5, alpha=0.3)
    ax.invert_yaxis()

    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS["vertex"], edgecolor="black", label="Vertex quality"),
        Patch(facecolor=CATEGORY_COLORS["kinematic"], edgecolor="black", label="Kinematic"),
        Patch(facecolor=CATEGORY_COLORS["PID"], edgecolor="black", label="PID"),
        Patch(facecolor="lightgray", edgecolor="black", label="Fail"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, frameon=True)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 3: MC vs Data comparison (individual efficiencies only, no cumulative) ---
    fig, ax = plt.subplots(figsize=(12, 7))

    n_cuts = len(CUTS_ORDER)
    cut_labels_only = [label for _, _, _, label, _ in CUTS_ORDER]
    x_pos = np.arange(n_cuts)
    bar_width = 0.15

    # Plot MC states
    for i_state, state in enumerate(mc_combined):
        state_rows = [
            r for r in summary_rows if r["sample"] == f"MC_{state}" and not r["is_cumulative"]
        ]
        effs = [r["efficiency_pass_pct"] for r in state_rows]
        offset = (i_state - len(mc_combined) / 2 + 0.5) * bar_width
        ax.bar(
            x_pos + offset,
            effs,
            bar_width,
            label=f"MC {STATE_LABELS[state]}",
            color=STATE_COLORS[state],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

    # Plot Data
    data_rows_ind = [r for r in summary_rows if r["sample"] == "Data" and not r["is_cumulative"]]
    data_effs = [r["efficiency_pass_pct"] for r in data_rows_ind]
    offset_data = (len(mc_combined) - len(mc_combined) / 2 + 0.5) * bar_width
    ax.bar(
        x_pos + offset_data,
        data_effs,
        bar_width,
        label="Data",
        color="#7f7f7f",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(cut_labels_only, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Efficiency [%]", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title("Individual Cut Efficiency: MC vs Data Comparison", fontsize=14)
    ax.legend(fontsize=9, loc="lower left", frameon=True)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 4: Summary table ---
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("off")
    ax.set_title("Individual Cut Efficiency Summary", fontsize=14, pad=20)

    # Build table: rows = cuts, columns = samples
    samples = [f"MC_{s}" for s in mc_combined] + ["Data"]
    col_labels = ["Cut", "Category"] + [STATE_LABELS.get(s.replace("MC_", ""), s) for s in samples]

    table_data = []
    for i, (branch, cut_type, value, label, category) in enumerate(CUTS_ORDER):
        row = [label, category.capitalize()]
        for sample in samples:
            sample_rows = [
                r for r in summary_rows if r["sample"] == sample and r["cut_branch"] == branch
            ]
            if sample_rows:
                row.append(f"{sample_rows[0]['efficiency_pass_pct']:.1f}%")
            else:
                row.append("—")
        table_data.append(row)

    # Cumulative row
    cum_row = ["All cuts combined", "Cumulative"]
    for sample in samples:
        sample_rows = [r for r in summary_rows if r["sample"] == sample and r["is_cumulative"]]
        if sample_rows:
            cum_row.append(f"{sample_rows[0]['efficiency_pass_pct']:.1f}%")
        else:
            cum_row.append("—")
    table_data.append(cum_row)

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)

    # Header styling
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=8)

    # Highlight cumulative row
    last_row = len(table_data)
    for j in range(len(col_labels)):
        table[last_row, j].set_facecolor("#D9E2F3")
        table[last_row, j].set_text_props(fontweight="bold")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print(f"  Plots saved to {plot_path}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("INDIVIDUAL CUT EFFICIENCY STUDY COMPLETE")
print("=" * 80)
print("\n  Outputs:")
print(f"    {plot_path}")
print(f"    {csv_path}")
print(f"\n  {len(mc_combined)} MC states + Data processed")
print(f"  {len(CUTS_ORDER)} individual cuts + 1 cumulative")
print("=" * 80)
