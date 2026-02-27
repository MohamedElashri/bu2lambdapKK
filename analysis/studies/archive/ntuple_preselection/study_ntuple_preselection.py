"""
Standalone study: N-Tuple Pre-Selection Diagnostic

What it does:
  1. Loads raw data directly from ROOT files — NO Lambda cuts applied.
     This reveals what stripping-level pre-selections already exist in the
     n-tuple (as hard cut-offs in the distributions).

  2. Lambda selection variable plots (4 variables):
       - L0_MM           (Lambda mass window cut: 1111 < M < 1121 MeV)
       - L0_FDCHI2_OWNPV (Lambda FD χ² cut: > 250)
       - Delta_Z_mm      (Δz vertex displacement: |Δz| > 5 mm)
       - Lp_ProbNNp      (Proton PID from Lambda: > 0.3)
     Each plot shows the full raw distribution with a vertical line (or pair of
     lines) at our selection cut value. A hard stripping-level cut shows up as
     a sharp distribution onset at a value above/below our cut.

  3. Bu_DTF_chi2 wide-range plot [0, 200]:
     The existing chi2_dtf_range study uses [0, 35]. This study plots [0, 200]
     to check whether the n-tuple already has a hard cut at some larger value
     (e.g. chi2 < 100 at stripping level).

Snakemake injects:
  snakemake.params.config_dir
  snakemake.params.cache_dir
  snakemake.params.output_dir
  snakemake.output.lambda_plot
  snakemake.output.chi2_plot
  snakemake.output.csv
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import awkward as ak  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from modules.data_handler import DataManager, TOMLConfig  # noqa: E402
from modules.exceptions import AnalysisError  # noqa: E402

# ---------------------------------------------------------------------------
# Snakemake params
# ---------------------------------------------------------------------------
config_dir = snakemake.params.config_dir  # noqa: F821
cache_dir = snakemake.params.cache_dir  # noqa: F821
output_dir = snakemake.params.output_dir  # noqa: F821
lambda_plot_path = snakemake.output.lambda_plot  # noqa: F821
chi2_plot_path = snakemake.output.chi2_plot  # noqa: F821
csv_path = snakemake.output.csv  # noqa: F821

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
plt.style.use(hep.style.LHCb2)

config = TOMLConfig(config_dir)
data_manager = DataManager(config)

years = snakemake.config.get("years", ["2016", "2017", "2018"])  # noqa: F821
track_types = snakemake.config.get("track_types", ["LL", "DD"])  # noqa: F821

# Lambda cut values from config
lambda_cuts = config.get_lambda_cuts()
LAMBDA_MASS_MIN = lambda_cuts.get("mass_min", 1111.0)
LAMBDA_MASS_MAX = lambda_cuts.get("mass_max", 1121.0)
LAMBDA_FDCHI2_MIN = lambda_cuts.get("fd_chisq_min", 250.0)
DELTA_Z_MIN = lambda_cuts.get("delta_z_min", 5.0)
PROTON_PROBNNP_MIN = lambda_cuts.get("proton_probnnp_min", 0.3)

# Bu_DTF_chi2 cut value from config
bu_opt = config.selection.get("manual_cuts", {})
CHI2_CUT = bu_opt.get("Bu_DTF_chi2", {}).get("value", 30.0)
CHI2_WIDE_MAX = 200.0

# ---------------------------------------------------------------------------
# Lambda variable plot definitions
# ---------------------------------------------------------------------------
LAMBDA_VARS = [
    {
        "branch": "L0_MM",
        "label": r"$M(\Lambda^0)$ [MeV/$c^2$]",
        "title": r"$\Lambda^0$ mass",
        "x_min": 1100.0,
        "x_max": 1135.0,
        "n_bins": 70,
        "cut_lines": [
            {"value": LAMBDA_MASS_MIN, "label": f"Cut min: {LAMBDA_MASS_MIN:.0f}", "color": "red"},
            {"value": LAMBDA_MASS_MAX, "label": f"Cut max: {LAMBDA_MASS_MAX:.0f}", "color": "blue"},
        ],
        "cut_type": "window",
    },
    {
        "branch": "L0_FDCHI2_OWNPV",
        "label": r"$\Lambda^0$ FD $\chi^2$",
        "title": r"$\Lambda^0$ flight distance $\chi^2$",
        "x_min": 0.0,
        "x_max": 2000.0,
        "n_bins": 100,
        "cut_lines": [
            {
                "value": LAMBDA_FDCHI2_MIN,
                "label": f"Cut: > {LAMBDA_FDCHI2_MIN:.0f}",
                "color": "red",
            },
        ],
        "cut_type": "greater",
    },
    {
        "branch": "Delta_Z_mm",
        "label": r"$\Delta z$ [mm]",
        "title": r"$\Lambda^0$–$B^+$ vertex $\Delta z$ (absolute)",
        "x_min": -50.0,
        "x_max": 200.0,
        "n_bins": 100,
        "cut_lines": [
            {"value": DELTA_Z_MIN, "label": f"|Cut|: > {DELTA_Z_MIN:.0f} mm", "color": "red"},
            {"value": -DELTA_Z_MIN, "label": None, "color": "red"},
        ],
        "cut_type": "abs_greater",
    },
    {
        "branch": "Lp_ProbNNp",
        "label": r"$\Lambda$ proton ProbNNp",
        "title": r"Proton PID from $\Lambda$ decay",
        "x_min": 0.0,
        "x_max": 1.0,
        "n_bins": 50,
        "cut_lines": [
            {
                "value": PROTON_PROBNNP_MIN,
                "label": f"Cut: > {PROTON_PROBNNP_MIN:.2f}",
                "color": "red",
            },
        ],
        "cut_type": "greater",
    },
]

# ---------------------------------------------------------------------------
# Load raw data (NO Lambda cuts)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("N-TUPLE PRE-SELECTION DIAGNOSTIC STUDY")
print("=" * 80)
print("Loading raw data from ROOT files (no Lambda cuts applied)...")
print(f"  Years: {years}")
print(f"  Track types: {track_types}")
print()

raw_arrays = []
n_loaded = 0

for year in years:
    for track_type in track_types:
        try:
            events = data_manager.load_and_process(
                "data",
                year,
                track_type,
                apply_derived_branches=True,  # needed for Delta_Z_mm
                apply_trigger=False,
            )
            if events is not None and len(events) > 0:
                raw_arrays.append(events)
                n_loaded += len(events)
                print(f"  Loaded {year} {track_type}: {len(events):,} events")
            else:
                print(f"  WARNING: No data for {year} {track_type}")
        except Exception as exc:
            print(f"  WARNING: Could not load {year} {track_type}: {exc}")

if not raw_arrays:
    raise AnalysisError(
        "No raw data loaded! Check that data files exist and config/data.toml paths are correct."
    )

all_raw = ak.concatenate(raw_arrays, axis=0)
print(f"\nTotal raw events loaded: {n_loaded:,}")


def get_flat(events, branch):
    """Return branch as flat numpy array, using ak.firsts if jagged."""
    if branch not in events.fields:
        return None
    br = events[branch]
    if "var" in str(ak.type(br)):
        br = ak.firsts(br)
    arr = ak.to_numpy(ak.drop_none(br))
    return arr[np.isfinite(arr)]


# ---------------------------------------------------------------------------
# Collect summary statistics for CSV
# ---------------------------------------------------------------------------
summary_rows = []


def add_summary(var_name, arr, cut_type, cut_values):
    """Record summary stats and pass fraction for a variable."""
    if arr is None or len(arr) == 0:
        return
    row = {
        "variable": var_name,
        "n_events": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }
    if cut_type == "greater":
        thr = cut_values[0]
        row["cut_threshold"] = thr
        row["fraction_passing"] = float(np.mean(arr > thr))
    elif cut_type == "window":
        lo, hi = cut_values
        row["cut_threshold"] = f"[{lo}, {hi}]"
        row["fraction_passing"] = float(np.mean((arr > lo) & (arr < hi)))
    elif cut_type == "abs_greater":
        thr = cut_values[0]
        row["cut_threshold"] = f"|x| > {thr}"
        row["fraction_passing"] = float(np.mean(np.abs(arr) > thr))
    summary_rows.append(row)


# ---------------------------------------------------------------------------
# Generate Lambda variable plots
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("GENERATING LAMBDA VARIABLE PLOTS")
print("=" * 80)

Path(lambda_plot_path).parent.mkdir(parents=True, exist_ok=True)

with PdfPages(lambda_plot_path) as pdf:
    for var_def in LAMBDA_VARS:
        branch = var_def["branch"]
        arr = get_flat(all_raw, branch)

        if arr is None:
            print(f"  WARNING: Branch '{branch}' not found — skipping")
            continue

        # Clip to plot range for display
        x_min, x_max = var_def["x_min"], var_def["x_max"]
        arr_plot = arr[(arr >= x_min) & (arr <= x_max)]

        print(f"  {branch}: {len(arr):,} total, {len(arr_plot):,} in [{x_min}, {x_max}]")

        # Collect summary
        cut_type = var_def["cut_type"]
        cut_vals = [cl["value"] for cl in var_def["cut_lines"] if cl["label"] is not None]
        if cut_type == "window":
            add_summary(branch, arr, "window", [LAMBDA_MASS_MIN, LAMBDA_MASS_MAX])
        elif cut_type == "greater":
            add_summary(branch, arr, "greater", cut_vals[:1])
        elif cut_type == "abs_greater":
            add_summary(branch, arr, "abs_greater", cut_vals[:1])

        bin_edges = np.linspace(x_min, x_max, var_def["n_bins"] + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]

        h, _ = np.histogram(arr_plot, bins=bin_edges)
        h_err = np.sqrt(h)

        fig, ax = plt.subplots(figsize=(10, 7))

        # Data as filled histogram
        ax.fill_between(bin_centers, h, alpha=0.3, color="steelblue", step="mid")
        ax.step(
            bin_centers, h, where="mid", color="steelblue", linewidth=1.5, label="Data (no cuts)"
        )

        # Cut lines
        y_max_plot = np.max(h) * 1.25 if len(h) > 0 else 1.0
        for cl in var_def["cut_lines"]:
            lbl = cl["label"]
            ax.axvline(
                cl["value"],
                color=cl["color"],
                linewidth=2,
                linestyle="--",
                label=lbl if lbl else "_nolegend_",
            )

        # Fraction passing
        if cut_type == "window":
            frac = np.mean((arr > LAMBDA_MASS_MIN) & (arr < LAMBDA_MASS_MAX)) * 100
            ax.text(
                0.97,
                0.92,
                f"Pass window: {frac:.1f}%\n"
                f"N (plot range): {len(arr_plot):,}\n"
                f"N (total): {len(arr):,}",
                transform=ax.transAxes,
                fontsize=10,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.6),
            )
        elif cut_type == "greater":
            thr = cut_vals[0] if cut_vals else 0
            frac = np.mean(arr > thr) * 100
            ax.text(
                0.97,
                0.92,
                f"Pass cut: {frac:.1f}%\n"
                f"N (plot range): {len(arr_plot):,}\n"
                f"N (total): {len(arr):,}",
                transform=ax.transAxes,
                fontsize=10,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.6),
            )
        elif cut_type == "abs_greater":
            thr = cut_vals[0] if cut_vals else 0
            frac = np.mean(np.abs(arr) > thr) * 100
            ax.text(
                0.97,
                0.92,
                f"Pass |cut|: {frac:.1f}%\n"
                f"N (plot range): {len(arr_plot):,}\n"
                f"N (total): {len(arr):,}",
                transform=ax.transAxes,
                fontsize=10,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.6),
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max_plot)
        ax.set_xlabel(var_def["label"], fontsize=14)
        ax.set_ylabel("Events / bin", fontsize=14)
        ax.set_title(
            f"{var_def['title']} — raw data (no cuts)\n" r"$B^+\to\bar{\Lambda}p K^-K^+$",
            fontsize=13,
        )
        ax.legend(fontsize=11, loc="upper left")

        # Annotation: look for hard cut-offs
        ax.text(
            0.03,
            0.60,
            "Look for hard cut-offs\n(sharp onset from below/above)\n" "→ n-tuple stripping cuts",
            transform=ax.transAxes,
            fontsize=9,
            ha="left",
            va="top",
            color="gray",
            style="italic",
        )

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        print(f"    Saved page: {branch}")

    # Summary page: all 4 variables on one canvas
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()

    for idx, var_def in enumerate(LAMBDA_VARS):
        branch = var_def["branch"]
        arr = get_flat(all_raw, branch)
        ax = axes_flat[idx]

        if arr is None:
            ax.text(
                0.5,
                0.5,
                f"Branch\n'{branch}'\nnot found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(var_def["title"])
            continue

        x_min, x_max = var_def["x_min"], var_def["x_max"]
        arr_plot = arr[(arr >= x_min) & (arr <= x_max)]
        bin_edges = np.linspace(x_min, x_max, var_def["n_bins"] // 2 + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        h, _ = np.histogram(arr_plot, bins=bin_edges)

        ax.fill_between(bin_centers, h, alpha=0.3, color="steelblue", step="mid")
        ax.step(bin_centers, h, where="mid", color="steelblue", linewidth=1.5)

        for cl in var_def["cut_lines"]:
            ax.axvline(
                cl["value"],
                color=cl["color"],
                linewidth=2,
                linestyle="--",
                label=cl["label"] if cl["label"] else "_nolegend_",
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, np.max(h) * 1.3 if len(h) > 0 else 1)
        ax.set_xlabel(var_def["label"], fontsize=11)
        ax.set_ylabel("Events / bin", fontsize=10)
        ax.set_title(var_def["title"], fontsize=12)
        ax.legend(fontsize=9, loc="upper right" if idx != 0 else "upper left")

    fig.suptitle(
        r"$\Lambda$ selection variables — raw data (no cuts applied)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    print("  Saved summary page (2×2 grid)")

print(f"\n  Lambda plots saved to: {lambda_plot_path}")

# ---------------------------------------------------------------------------
# Bu_DTF_chi2 wide-range plot
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("GENERATING Bu_DTF_chi2 WIDE-RANGE PLOT")
print("=" * 80)

CHI2_BRANCH = "Bu_DTF_chi2"
chi2_arr = get_flat(all_raw, CHI2_BRANCH)

if chi2_arr is None:
    raise AnalysisError(
        f"Branch '{CHI2_BRANCH}' not found in data! "
        "Check that Bu_DTF_chi2 is present in the n-tuple."
    )

# Wide range: [0, 200]
chi2_wide = chi2_arr[(chi2_arr >= 0) & (chi2_arr <= CHI2_WIDE_MAX)]
print(f"  Events with chi2 in [0, {CHI2_WIDE_MAX}]: {len(chi2_wide):,} / {len(chi2_arr):,}")
frac_above = np.mean(chi2_arr > CHI2_WIDE_MAX) * 100
print(f"  Events above {CHI2_WIDE_MAX}: {frac_above:.2f}%")

# Add to summary
summary_rows.append(
    {
        "variable": "Bu_DTF_chi2",
        "n_events": len(chi2_arr),
        "mean": float(np.mean(chi2_arr)),
        "std": float(np.std(chi2_arr)),
        "min": float(np.min(chi2_arr)),
        "max": float(np.max(chi2_arr)),
        "p5": float(np.percentile(chi2_arr, 5)),
        "p95": float(np.percentile(chi2_arr, 95)),
        "cut_threshold": CHI2_CUT,
        "fraction_passing": float(np.mean(chi2_arr < CHI2_CUT)),
    }
)

Path(chi2_plot_path).parent.mkdir(parents=True, exist_ok=True)

# Adaptive binning: use log scale if distribution is heavy-tailed
n_bins_wide = 100
bin_edges_wide = np.linspace(0, CHI2_WIDE_MAX, n_bins_wide + 1)
bin_centers_wide = 0.5 * (bin_edges_wide[:-1] + bin_edges_wide[1:])
bin_width_wide = bin_edges_wide[1] - bin_edges_wide[0]
h_wide, _ = np.histogram(chi2_wide, bins=bin_edges_wide)

# Also make a zoom on [0, 50] to show the interesting region
chi2_zoom = chi2_arr[(chi2_arr >= 0) & (chi2_arr <= 50.0)]
bin_edges_zoom = np.linspace(0, 50.0, 51)
bin_centers_zoom = 0.5 * (bin_edges_zoom[:-1] + bin_edges_zoom[1:])
h_zoom, _ = np.histogram(chi2_zoom, bins=bin_edges_zoom)

with PdfPages(chi2_plot_path) as pdf:
    # --- Page 1: Wide range [0, 200] ---
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.fill_between(bin_centers_wide, h_wide, alpha=0.3, color="steelblue", step="mid")
    ax.step(
        bin_centers_wide,
        h_wide,
        where="mid",
        color="steelblue",
        linewidth=1.5,
        label="Data (no cuts)",
    )

    # Our analysis cut
    ax.axvline(
        CHI2_CUT,
        color="red",
        linewidth=2,
        linestyle="--",
        label=rf"Analysis cut: $\chi^2 < {CHI2_CUT:.0f}$",
    )

    # Annotation for fraction passing
    frac_pass = np.mean(chi2_arr < CHI2_CUT) * 100
    ax.text(
        0.97,
        0.92,
        rf"Pass cut ($\chi^2 < {CHI2_CUT:.0f}$): {frac_pass:.1f}%"
        f"\nN in [0, {CHI2_WIDE_MAX:.0f}]: {len(chi2_wide):,}"
        f"\nN above {CHI2_WIDE_MAX:.0f}: {frac_above:.1f}%",
        transform=ax.transAxes,
        fontsize=10,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.6),
    )

    # Arrow pointing to "keep" region
    y_max_chi2 = np.max(h_wide) * 1.25
    ax.annotate(
        "Keep",
        xy=(CHI2_CUT, y_max_chi2 * 0.5),
        xytext=(CHI2_CUT - 15, y_max_chi2 * 0.5),
        fontsize=12,
        color="green",
        fontweight="bold",
        arrowprops=dict(arrowstyle="<-", color="green", lw=2),
        ha="center",
    )

    ax.text(
        0.03,
        0.60,
        "If distribution ends abruptly\nbefore the plot range,\n"
        "a stripping-level hard cut\nexists in the n-tuple.",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="top",
        color="gray",
        style="italic",
    )

    ax.set_xlim(0, CHI2_WIDE_MAX)
    ax.set_ylim(0, y_max_chi2)
    ax.set_xlabel(r"$\chi^2_{\mathrm{DTF}}(B^+)$", fontsize=14)
    ax.set_ylabel("Events / bin", fontsize=14)
    ax.set_title(
        r"$\chi^2_{\mathrm{DTF}}(B^+)$ — raw data, wide range [0, 200]"
        "\n(check for n-tuple hard cut)",
        fontsize=13,
    )
    ax.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 2: Zoom [0, 50] ---
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.fill_between(bin_centers_zoom, h_zoom, alpha=0.3, color="steelblue", step="mid")
    ax.step(
        bin_centers_zoom,
        h_zoom,
        where="mid",
        color="steelblue",
        linewidth=1.5,
        label="Data (no cuts)",
    )
    ax.axvline(
        CHI2_CUT,
        color="red",
        linewidth=2,
        linestyle="--",
        label=rf"Analysis cut: $\chi^2 < {CHI2_CUT:.0f}$",
    )

    ax.set_xlim(0, 50.0)
    ax.set_ylim(0, np.max(h_zoom) * 1.3 if len(h_zoom) > 0 else 1)
    ax.set_xlabel(r"$\chi^2_{\mathrm{DTF}}(B^+)$", fontsize=14)
    ax.set_ylabel("Events / bin", fontsize=14)
    ax.set_title(
        r"$\chi^2_{\mathrm{DTF}}(B^+)$ — zoom [0, 50]",
        fontsize=13,
    )
    ax.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 3: Log scale, full range ---
    fig, ax = plt.subplots(figsize=(12, 7))

    # Use log-spaced bins for better view of tail
    max_chi2_obs = float(np.max(chi2_arr)) if len(chi2_arr) > 0 else 1000.0
    log_edges = np.logspace(np.log10(0.5), np.log10(max(max_chi2_obs, 200.0)), 80)
    h_log, log_bin_edges = np.histogram(chi2_arr, bins=log_edges)
    log_centers = np.sqrt(log_bin_edges[:-1] * log_bin_edges[1:])

    ax.fill_between(log_centers, h_log, alpha=0.3, color="steelblue", step="mid")
    ax.step(
        log_centers, h_log, where="mid", color="steelblue", linewidth=1.5, label="Data (no cuts)"
    )
    ax.axvline(
        CHI2_CUT,
        color="red",
        linewidth=2,
        linestyle="--",
        label=rf"Analysis cut: $\chi^2 < {CHI2_CUT:.0f}$",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"$\chi^2_{\mathrm{DTF}}(B^+)$ [log scale]", fontsize=14)
    ax.set_ylabel("Events / bin", fontsize=14)
    ax.set_title(
        r"$\chi^2_{\mathrm{DTF}}(B^+)$ — log scale (full observable range)"
        "\nReveals any hard cut-off in the tail",
        fontsize=13,
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.text(
        0.97,
        0.60,
        f"Max observed: {max_chi2_obs:.0f}\n" f"Median: {np.median(chi2_arr):.1f}",
        transform=ax.transAxes,
        fontsize=10,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.6),
    )
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print(f"  Bu_DTF_chi2 plots saved to: {chi2_plot_path}")

# ---------------------------------------------------------------------------
# Save CSV summary
# ---------------------------------------------------------------------------
Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(csv_path, index=False)
print(f"\n  Summary CSV saved to: {csv_path}")

# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("N-TUPLE PRE-SELECTION DIAGNOSTIC — COMPLETE")
print("=" * 80)
print(f"\n  Total raw events: {n_loaded:,}")
print("\n  Lambda cut pass fractions (before applying any cuts):")
for row in summary_rows:
    if row["variable"] != "Bu_DTF_chi2":
        fp = row.get("fraction_passing", float("nan"))
        print(
            f"    {row['variable']:<30s}: {fp*100:5.1f}% pass  (cut: {row.get('cut_threshold', '?')})"
        )
chi2_row = next((r for r in summary_rows if r["variable"] == "Bu_DTF_chi2"), None)
if chi2_row:
    fp = chi2_row.get("fraction_passing", float("nan"))
    print(f"    {'Bu_DTF_chi2':<30s}: {fp*100:5.1f}% pass  (cut: chi2 < {CHI2_CUT:.0f})")
    print(f"    Max observed chi2: {chi2_row['max']:.0f}")
print("\n  Outputs:")
print(f"    {lambda_plot_path}")
print(f"    {chi2_plot_path}")
print(f"    {csv_path}")
print("=" * 80)
