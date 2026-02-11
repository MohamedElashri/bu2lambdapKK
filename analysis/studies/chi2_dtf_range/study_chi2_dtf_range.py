"""
Standalone study: χ²_DTF(B⁺) Distribution with Restricted Range.

Ported from: archive/analysis/studies/feedback_dec2024/study5_chi2_range_fix.py

What it does:
  Plots the χ²_DTF(B⁺) distribution with a restricted x-axis range [0, 35]:
    - The cut value is χ² < 30, so the full range [0, 100] wastes space
    - Shows MC (per-state overlaid + combined) and Data, normalised to unit area
    - Vertical dashed line at the cut value
    - Shape comparison between MC and Data

Improvements over original:
  - matplotlib + mplhep (LHCb style) instead of ROOT TCanvas/TH1
  - Ratio panel (Data/MC) below the main plot
  - Per-state MC distributions overlaid (not just combined)
  - KS test p-value for Data vs MC shape comparison
  - Summary CSV with KS test results
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
from scipy import stats  # noqa: E402

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

# χ²_DTF variable settings
CHI2_BRANCH = "Bu_DTF_chi2"
CUT_VALUE = 30.0
X_MIN = 0.0
X_MAX = 35.0
N_BINS = 70

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


def get_chi2_array(events):
    """Extract χ²_DTF as a flat numpy array, filtered to [X_MIN, X_MAX]."""
    chi2 = np.asarray(ak.to_numpy(_get_flat_branch(events, CHI2_BRANCH)))
    mask = (chi2 >= X_MIN) & (chi2 <= X_MAX)
    return chi2[mask]


# ---------------------------------------------------------------------------
# Load Step 2 cached data
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("χ²_DTF(B⁺) DISTRIBUTION STUDY")
print("=" * 80)
print(f"  Variable: {CHI2_BRANCH}")
print(f"  Range: [{X_MIN}, {X_MAX}] (restricted from [0, 100])")
print(f"  Cut: < {CUT_VALUE}")

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
chi2_data = get_chi2_array(all_data)
print(f"  Data events in [{X_MIN}, {X_MAX}]: {len(chi2_data):,}")

# Combine MC per state and combined
print("\n[Loading signal MC]")
chi2_mc_per_state = {}
mc_all_arrays = []
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
        combined = ak.concatenate(arrays, axis=0)
        chi2_mc_per_state[state] = get_chi2_array(combined)
        mc_all_arrays.append(combined)
        print(f"  MC/{state}: {len(chi2_mc_per_state[state]):,} events in range")

# Combined MC (all states)
all_mc = ak.concatenate(mc_all_arrays, axis=0)
chi2_mc_combined = get_chi2_array(all_mc)
print(f"  MC combined: {len(chi2_mc_combined):,} events in range")

# ---------------------------------------------------------------------------
# KS tests
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("KS TEST: Data vs MC shape comparison")
print("=" * 80)

ks_rows = []

# Data vs combined MC
ks_stat, ks_pval = stats.ks_2samp(chi2_data, chi2_mc_combined)
print(f"  Data vs MC (combined): KS = {ks_stat:.4f}, p-value = {ks_pval:.4e}")
ks_rows.append(
    {
        "comparison": "Data vs MC_combined",
        "ks_statistic": round(ks_stat, 6),
        "p_value": ks_pval,
        "n_sample1": len(chi2_data),
        "n_sample2": len(chi2_mc_combined),
        "compatible": "Yes" if ks_pval > 0.05 else "No",
    }
)

# Data vs each MC state
for state in chi2_mc_per_state:
    ks_stat, ks_pval = stats.ks_2samp(chi2_data, chi2_mc_per_state[state])
    print(f"  Data vs MC_{state}: KS = {ks_stat:.4f}, p-value = {ks_pval:.4e}")
    ks_rows.append(
        {
            "comparison": f"Data vs MC_{state}",
            "ks_statistic": round(ks_stat, 6),
            "p_value": ks_pval,
            "n_sample1": len(chi2_data),
            "n_sample2": len(chi2_mc_per_state[state]),
            "compatible": "Yes" if ks_pval > 0.05 else "No",
        }
    )

# MC state vs MC state
states_list = list(chi2_mc_per_state.keys())
for i in range(len(states_list)):
    for j in range(i + 1, len(states_list)):
        s1, s2 = states_list[i], states_list[j]
        ks_stat, ks_pval = stats.ks_2samp(chi2_mc_per_state[s1], chi2_mc_per_state[s2])
        ks_rows.append(
            {
                "comparison": f"MC_{s1} vs MC_{s2}",
                "ks_statistic": round(ks_stat, 6),
                "p_value": ks_pval,
                "n_sample1": len(chi2_mc_per_state[s1]),
                "n_sample2": len(chi2_mc_per_state[s2]),
                "compatible": "Yes" if ks_pval > 0.05 else "No",
            }
        )

# Fraction passing cut
frac_data = np.sum(chi2_data < CUT_VALUE) / len(chi2_data) * 100
frac_mc = np.sum(chi2_mc_combined < CUT_VALUE) / len(chi2_mc_combined) * 100
print(f"\n  Fraction passing cut (χ² < {CUT_VALUE}):")
print(f"    Data: {frac_data:.1f}%")
print(f"    MC:   {frac_mc:.1f}%")

# Add fraction info to CSV
ks_rows.append(
    {
        "comparison": f"Fraction_passing_cut_{CUT_VALUE}",
        "ks_statistic": frac_data,
        "p_value": frac_mc,
        "n_sample1": len(chi2_data),
        "n_sample2": len(chi2_mc_combined),
        "compatible": f"Data={frac_data:.1f}% MC={frac_mc:.1f}%",
    }
)

# Save CSV
csv_out = Path(csv_path)
csv_out.parent.mkdir(exist_ok=True, parents=True)
df_ks = pd.DataFrame(ks_rows)
df_ks.to_csv(csv_out, index=False)
print(f"\n  KS test CSV saved to {csv_path}")

# ---------------------------------------------------------------------------
# Generate plots
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("GENERATING PLOTS")
print("=" * 80)

bin_edges = np.linspace(X_MIN, X_MAX, N_BINS + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_width = bin_edges[1] - bin_edges[0]

# Histogram all distributions (normalised to unit area)
h_data, _ = np.histogram(chi2_data, bins=bin_edges, density=True)
h_mc_combined, _ = np.histogram(chi2_mc_combined, bins=bin_edges, density=True)

h_mc_states = {}
for state in chi2_mc_per_state:
    h_mc_states[state], _ = np.histogram(chi2_mc_per_state[state], bins=bin_edges, density=True)

# Ratio: Data / MC_combined (with error)
with np.errstate(divide="ignore", invalid="ignore"):
    ratio = np.where(h_mc_combined > 0, h_data / h_mc_combined, np.nan)
    # Approximate error on ratio (Poisson-like)
    h_data_raw, _ = np.histogram(chi2_data, bins=bin_edges)
    h_mc_raw, _ = np.histogram(chi2_mc_combined, bins=bin_edges)
    ratio_err = np.where(
        (h_data_raw > 0) & (h_mc_raw > 0),
        ratio * np.sqrt(1.0 / h_data_raw + 1.0 / h_mc_raw),
        np.nan,
    )

plot_out = Path(plot_path)
plot_out.parent.mkdir(exist_ok=True, parents=True)

with PdfPages(plot_out) as pdf:
    # --- Page 1: Main comparison with ratio panel ---
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)

    ax_main = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

    # Main panel: Data + MC combined + per-state MC
    # Data as points with error bars
    data_err = (
        np.sqrt(h_data_raw) / (len(chi2_data) * bin_width)
        if len(chi2_data) > 0
        else np.zeros_like(h_data)
    )
    ax_main.errorbar(
        bin_centers,
        h_data,
        yerr=data_err,
        fmt="ko",
        markersize=4,
        linewidth=1,
        label="Data",
        zorder=10,
    )

    # MC combined as filled histogram
    ax_main.fill_between(
        bin_centers,
        h_mc_combined,
        alpha=0.2,
        color="red",
        step="mid",
        label="MC (all states)",
    )
    ax_main.step(
        bin_centers,
        h_mc_combined,
        where="mid",
        color="red",
        linewidth=2,
    )

    # Per-state MC as thin lines
    for state in chi2_mc_per_state:
        ax_main.step(
            bin_centers,
            h_mc_states[state],
            where="mid",
            color=STATE_COLORS[state],
            linewidth=1,
            linestyle="--",
            alpha=0.7,
            label=f"MC {STATE_LABELS[state]}",
        )

    # Cut line
    ax_main.axvline(
        CUT_VALUE,
        color="black",
        linewidth=2,
        linestyle="--",
        label=f"Cut: $\\chi^2 < {CUT_VALUE:.0f}$",
    )

    # Arrow indicating kept region
    y_max = max(np.max(h_data), np.max(h_mc_combined)) * 1.1
    ax_main.annotate(
        "Keep",
        xy=(CUT_VALUE, y_max * 0.85),
        xytext=(CUT_VALUE - 5, y_max * 0.85),
        fontsize=12,
        color="green",
        fontweight="bold",
        arrowprops=dict(arrowstyle="<-", color="green", lw=2),
        ha="center",
    )

    # KS test annotation
    ks_combined = ks_rows[0]
    ax_main.text(
        0.98,
        0.55,
        f"KS test (Data vs MC):\n"
        f"  $D = {ks_combined['ks_statistic']:.4f}$\n"
        f"  $p = {ks_combined['p_value']:.2e}$",
        transform=ax_main.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    ax_main.set_xlim(X_MIN, X_MAX)
    ax_main.set_ylim(0, y_max)
    ax_main.set_ylabel("Normalised", fontsize=14)
    ax_main.set_title(
        r"$\chi^{2}_{\mathrm{DTF}}(B^{+})$ distribution (restricted range)",
        fontsize=14,
    )
    ax_main.legend(fontsize=9, loc="upper right", ncol=2)
    ax_main.tick_params(labelbottom=False)

    # Ratio panel
    ax_ratio.errorbar(
        bin_centers,
        ratio,
        yerr=ratio_err,
        fmt="ko",
        markersize=3,
        linewidth=1,
    )
    ax_ratio.axhline(1.0, color="red", linewidth=1, linestyle="-")
    ax_ratio.axhline(1.1, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_ratio.axhline(0.9, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_ratio.axvline(CUT_VALUE, color="black", linewidth=2, linestyle="--")

    ax_ratio.set_xlim(X_MIN, X_MAX)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_xlabel(r"$\chi^{2}_{\mathrm{DTF}}(B^{+})$", fontsize=14)
    ax_ratio.set_ylabel("Data / MC", fontsize=12)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 2: Per-state MC comparison (normalised) ---
    fig, ax = plt.subplots(figsize=(12, 8))

    for state in chi2_mc_per_state:
        ax.step(
            bin_centers,
            h_mc_states[state],
            where="mid",
            color=STATE_COLORS[state],
            linewidth=2,
            label=f"MC {STATE_LABELS[state]}",
        )

    ax.axvline(
        CUT_VALUE,
        color="black",
        linewidth=2,
        linestyle="--",
        label=f"Cut: $\\chi^2 < {CUT_VALUE:.0f}$",
    )

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$\chi^{2}_{\mathrm{DTF}}(B^{+})$", fontsize=14)
    ax.set_ylabel("Normalised", fontsize=14)
    ax.set_title(
        r"$\chi^{2}_{\mathrm{DTF}}(B^{+})$: MC per-state comparison",
        fontsize=14,
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add KS test results between states
    ks_text_lines = ["MC inter-state KS tests:"]
    for row in ks_rows:
        if row["comparison"].startswith("MC_") and "vs MC_" in row["comparison"]:
            ks_text_lines.append(f"  {row['comparison']}: p = {row['p_value']:.2e}")
    ax.text(
        0.98,
        0.50,
        "\n".join(ks_text_lines),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5),
        family="monospace",
    )

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 3: Summary table ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    ax.set_title(
        r"$\chi^{2}_{\mathrm{DTF}}(B^{+})$ KS Test Summary",
        fontsize=14,
        pad=20,
    )

    col_labels = ["Comparison", "KS Statistic", "p-value", "N1", "N2", "Compatible (p>0.05)"]
    table_data = []
    for row in ks_rows:
        if row["comparison"].startswith("Fraction"):
            continue
        table_data.append(
            [
                row["comparison"],
                f"{row['ks_statistic']:.4f}",
                f"{row['p_value']:.2e}",
                f"{row['n_sample1']:,}",
                f"{row['n_sample2']:,}",
                row["compatible"],
            ]
        )

    # Add fraction row
    table_data.append(
        [
            f"Fraction passing χ² < {CUT_VALUE:.0f}",
            f"Data: {frac_data:.1f}%",
            f"MC: {frac_mc:.1f}%",
            f"{len(chi2_data):,}",
            f"{len(chi2_mc_combined):,}",
            f"Δ = {abs(frac_data - frac_mc):.1f}%",
        ]
    )

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=9)

    # Highlight last row
    last_row = len(table_data)
    for j in range(len(col_labels)):
        table[last_row, j].set_facecolor("#D9E2F3")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print(f"  Plots saved to {plot_path}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("χ²_DTF(B⁺) DISTRIBUTION STUDY COMPLETE")
print("=" * 80)
print("\n  Outputs:")
print(f"    {plot_path}")
print(f"    {csv_path}")
print(f"\n  Data: {len(chi2_data):,} events in [{X_MIN}, {X_MAX}]")
print(f"  MC:   {len(chi2_mc_combined):,} events in [{X_MIN}, {X_MAX}]")
print(f"  Fraction passing cut (χ² < {CUT_VALUE}):")
print(f"    Data: {frac_data:.1f}%")
print(f"    MC:   {frac_mc:.1f}%")
print("=" * 80)
