"""
Sideband Background Study — Phase A: Shape Validation.

Ported from: archive/analysis/studies/sideband_background/validate_shapes.py

Validates that the M(Λ̄pK⁻) distribution shape is independent of M(B⁺)
in sideband regions. This is a critical assumption for using sidebands
to model combinatorial background.

Method:
  1. Divide M(B⁺) sideband into 4 slices (far-left, mid-left, near-left, right)
  2. Create M(Λ̄pK⁻) projection for each slice
  3. Normalise to unit area
  4. Compare shapes via KS and χ² tests

If shapes are consistent → sideband method is valid.

Snakemake injects:
  snakemake.params.config_dir / cache_dir / output_dir
  snakemake.output.plot / csv
"""

import sys
from pathlib import Path

# Ensure the project root (analysis/) is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
study_dir = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(study_dir) not in sys.path:
    sys.path.insert(0, str(study_dir))

import awkward as ak  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from config import CHARMONIUM_LINES, MASS_CONFIG, REGION_COLORS  # noqa: E402
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

# Sideband regions to compare
REGIONS = [
    (MASS_CONFIG.LEFT_SIDEBAND_FAR, "Far-left [2800-3500]", "Far-left"),
    (MASS_CONFIG.LEFT_SIDEBAND_MID, "Mid-left [3500-4500]", "Mid-left"),
    (MASS_CONFIG.LEFT_SIDEBAND_NEAR, "Near-left [4500-5150]", "Near-left"),
    (MASS_CONFIG.RIGHT_SIDEBAND, "Right [5330-5500]", "Right"),
]

MIN_EVENTS = 100


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
    """Get a branch as a flat (non-jagged) array."""
    br = events[branch_name]
    if "var" in str(ak.type(br)):
        br = ak.firsts(br)
    return br


# ---------------------------------------------------------------------------
# Load Step 2 cached data
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SIDEBAND BACKGROUND: SHAPE VALIDATION")
print("=" * 80)

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

# Identify branches
mass_branch = "M_LpKm_h2" if "M_LpKm_h2" in all_data.fields else "M_LpKm"
bu_mass_branch = "Bu_MM_corrected" if "Bu_MM_corrected" in all_data.fields else "Bu_MM"
print(f"  M(Λ̄pK⁻) branch: {mass_branch}")
print(f"  M(B⁺) branch: {bu_mass_branch}")

# Extract flat numpy arrays
mlpk_all = np.asarray(ak.to_numpy(_get_flat_branch(all_data, mass_branch)))
bu_mass_all = np.asarray(ak.to_numpy(_get_flat_branch(all_data, bu_mass_branch)))

# ---------------------------------------------------------------------------
# Select events in each M(B⁺) sideband region
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SELECTING EVENTS IN M(B⁺) SIDEBAND REGIONS")
print("=" * 80)

region_data = {}  # key -> (mlpk_array, label, color_key)
for (mbu_min, mbu_max), label, color_key in REGIONS:
    mask = (bu_mass_all > mbu_min) & (bu_mass_all < mbu_max)
    # Also restrict to M(Λ̄pK⁻) range
    mlpk_in_region = mlpk_all[mask]
    mlpk_in_range = mlpk_in_region[
        (mlpk_in_region >= MASS_CONFIG.MLPK_MIN) & (mlpk_in_region <= MASS_CONFIG.MLPK_MAX)
    ]
    n_events = len(mlpk_in_range)
    print(f"  {label}: {n_events:,} events")

    if n_events >= MIN_EVENTS:
        region_data[color_key] = (mlpk_in_range, label)
    else:
        print(f"    WARNING: insufficient statistics (<{MIN_EVENTS}), skipping")

if len(region_data) < 2:
    raise AnalysisError("Need at least 2 regions with events for comparison!")

# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STATISTICAL TESTS")
print("=" * 80)

bin_edges = np.linspace(MASS_CONFIG.MLPK_MIN, MASS_CONFIG.MLPK_MAX, MASS_CONFIG.N_BINS_MLPK + 1)

test_rows = []
region_keys = list(region_data.keys())

for i in range(len(region_keys)):
    for j in range(i + 1, len(region_keys)):
        k1, k2 = region_keys[i], region_keys[j]
        arr1, label1 = region_data[k1]
        arr2, label2 = region_data[k2]

        # KS test (on raw arrays)
        ks_stat, ks_pval = stats.ks_2samp(arr1, arr2)

        # χ² test (on binned histograms, normalised)
        h1, _ = np.histogram(arr1, bins=bin_edges, density=True)
        h2, _ = np.histogram(arr2, bins=bin_edges, density=True)
        # Avoid division by zero
        nonzero = (h1 + h2) > 0
        if np.sum(nonzero) > 0:
            chi2_val = np.sum((h1[nonzero] - h2[nonzero]) ** 2 / (h1[nonzero] + h2[nonzero]))
            ndf = int(np.sum(nonzero)) - 1
            chi2_pval = 1.0 - stats.chi2.cdf(chi2_val, ndf) if ndf > 0 else 0.0
        else:
            chi2_val, ndf, chi2_pval = 0.0, 0, 0.0

        compat_ks = "Yes" if ks_pval > 0.05 else "No"
        compat_chi2 = "Yes" if chi2_pval > 0.05 else "No"

        print(f"  {label1} vs {label2}:")
        print(f"    KS: D={ks_stat:.4f}, p={ks_pval:.4f} [{compat_ks}]")
        print(f"    χ²: {chi2_val:.1f}/{ndf}, p={chi2_pval:.4f} [{compat_chi2}]")

        test_rows.append(
            {
                "region1": label1,
                "region2": label2,
                "ks_statistic": round(ks_stat, 6),
                "ks_p_value": ks_pval,
                "ks_compatible": compat_ks,
                "chi2": round(chi2_val, 2),
                "chi2_ndf": ndf,
                "chi2_p_value": chi2_pval,
                "chi2_compatible": compat_chi2,
                "n_events_1": len(arr1),
                "n_events_2": len(arr2),
            }
        )

# Save CSV
csv_out = Path(csv_path)
csv_out.parent.mkdir(exist_ok=True, parents=True)
df = pd.DataFrame(test_rows)
df.to_csv(csv_out, index=False)
print(f"\n  Test results saved to {csv_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
n_ks_compat = sum(1 for r in test_rows if r["ks_compatible"] == "Yes")
n_chi2_compat = sum(1 for r in test_rows if r["chi2_compatible"] == "Yes")
n_total = len(test_rows)

print(f"\n  KS compatible: {n_ks_compat}/{n_total}")
print(f"  χ² compatible: {n_chi2_compat}/{n_total}")

# ---------------------------------------------------------------------------
# Generate plots
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("GENERATING PLOTS")
print("=" * 80)

plot_out = Path(plot_path)
plot_out.parent.mkdir(exist_ok=True, parents=True)

bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

with PdfPages(plot_out) as pdf:
    # --- Page 1: Shape comparison (normalised overlay) ---
    fig, ax = plt.subplots(figsize=(12, 8))

    for key in region_data:
        arr, label = region_data[key]
        h, _ = np.histogram(arr, bins=bin_edges, density=True)
        ax.step(
            bin_centers,
            h,
            where="mid",
            color=REGION_COLORS.get(key, "gray"),
            linewidth=2,
            label=f"{label} ({len(arr):,})",
        )

    # Charmonium reference lines
    y_max = ax.get_ylim()[1]
    for mass, clabel, color in CHARMONIUM_LINES:
        if MASS_CONFIG.MLPK_MIN < mass < MASS_CONFIG.MLPK_MAX:
            ax.axvline(mass, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.text(
                mass + 5,
                y_max * 0.85,
                clabel,
                fontsize=7,
                rotation=90,
                va="top",
                color="gray",
            )

    ax.set_xlim(MASS_CONFIG.MLPK_MIN, MASS_CONFIG.MLPK_MAX)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$M(\bar{\Lambda}pK^{-})$ [MeV/$c^2$]", fontsize=14)
    ax.set_ylabel("Normalised", fontsize=14)
    ax.set_title(
        r"$M(\bar{\Lambda}pK^{-})$ shape vs $M(B^{+})$ region",
        fontsize=14,
    )
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 2: Ratio plot (all vs near-left reference) ---
    ref_key = "Near-left" if "Near-left" in region_data else region_keys[0]
    ref_arr, ref_label = region_data[ref_key]
    h_ref, _ = np.histogram(ref_arr, bins=bin_edges, density=True)

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

    for key in region_data:
        arr, label = region_data[key]
        h, _ = np.histogram(arr, bins=bin_edges, density=True)
        ax_main.step(
            bin_centers,
            h,
            where="mid",
            color=REGION_COLORS.get(key, "gray"),
            linewidth=2,
            label=f"{label}",
        )

    ax_main.set_xlim(MASS_CONFIG.MLPK_MIN, MASS_CONFIG.MLPK_MAX)
    ax_main.set_ylim(0, None)
    ax_main.set_ylabel("Normalised", fontsize=14)
    ax_main.set_title(
        r"Shape comparison with ratio to " + ref_label,
        fontsize=14,
    )
    ax_main.legend(fontsize=10, loc="upper right")
    ax_main.tick_params(labelbottom=False)

    # Ratio panel
    for key in region_data:
        if key == ref_key:
            continue
        arr, label = region_data[key]
        h, _ = np.histogram(arr, bins=bin_edges, density=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(h_ref > 0, h / h_ref, np.nan)
        ax_ratio.step(
            bin_centers,
            ratio,
            where="mid",
            color=REGION_COLORS.get(key, "gray"),
            linewidth=1.5,
            label=f"{label} / {ref_label}",
        )

    ax_ratio.axhline(1.0, color="black", linewidth=1, linestyle="-")
    ax_ratio.axhline(1.2, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_ratio.axhline(0.8, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_ratio.set_xlim(MASS_CONFIG.MLPK_MIN, MASS_CONFIG.MLPK_MAX)
    ax_ratio.set_ylim(0.0, 2.0)
    ax_ratio.set_xlabel(r"$M(\bar{\Lambda}pK^{-})$ [MeV/$c^2$]", fontsize=14)
    ax_ratio.set_ylabel(f"Ratio to {ref_label}", fontsize=12)
    ax_ratio.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 3: Summary table ---
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("off")
    ax.set_title("Shape Validation: Statistical Tests", fontsize=14, pad=20)

    col_labels = [
        "Region 1",
        "Region 2",
        "KS stat",
        "KS p-value",
        "KS compat.",
        "chi2/ndf",
        "chi2 p-value",
        "chi2 compat.",
    ]
    table_data = []
    for r in test_rows:
        chi2_ndf_str = f"{r['chi2']:.1f}/{r['chi2_ndf']}" if r["chi2_ndf"] > 0 else "—"
        table_data.append(
            [
                r["region1"],
                r["region2"],
                f"{r['ks_statistic']:.4f}",
                f"{r['ks_p_value']:.4f}",
                r["ks_compatible"],
                chi2_ndf_str,
                f"{r['chi2_p_value']:.4f}",
                r["chi2_compatible"],
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
    table.scale(1.0, 2.2)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=9)

    # Colour-code compatibility cells
    for i, r in enumerate(test_rows):
        row_idx = i + 1
        for col_idx, key in [(4, "ks_compatible"), (7, "chi2_compatible")]:
            if r[key] == "Yes":
                table[row_idx, col_idx].set_facecolor("#C6EFCE")
            else:
                table[row_idx, col_idx].set_facecolor("#FFC7CE")

    # Summary text
    ax.text(
        0.5,
        0.05,
        f"KS compatible: {n_ks_compat}/{n_total}  |  "
        f"chi2 compatible: {n_chi2_compat}/{n_total}",
        transform=ax.transAxes,
        fontsize=12,
        ha="center",
        fontweight="bold",
    )

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print(f"  Plots saved to {plot_path}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SHAPE VALIDATION COMPLETE")
print("=" * 80)
print(f"\n  Outputs: {plot_path}, {csv_path}")
print(f"  {len(region_data)} M(B⁺) regions compared")
print(f"  KS compatible: {n_ks_compat}/{n_total}")
print(f"  χ² compatible: {n_chi2_compat}/{n_total}")
if n_ks_compat == n_total:
    print("  => All shapes statistically compatible — sideband method is valid")
else:
    print("  => Some shape differences — consider using only near-signal sidebands")
print("=" * 80)
