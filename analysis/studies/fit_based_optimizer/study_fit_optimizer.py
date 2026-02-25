"""
Fit-Based N-D Optimizer Study

Context:
  The pid_proxy_comparison study proved that ALL proxy methods for background PID efficiency
  (sideband, Option B ARGUS-tail, Option C approx-sWeight, Option D true sPlot) are
  anti-correlated with the fit-based FOM (Pearson r ≈ -0.63 to -0.93). Every proxy
  concludes "PID cuts make things worse" when in reality PID>0.20 improves FOM1 by +31%.

  This study fixes PID_product > 0.20 as a pre-selection and runs actual mass fits at
  each of 252 grid points (chi2 × FDCHI2 × IPCHI2 × PT) to measure the TRUE FOM.

Grid:
  - Bu_DTF_chi2    < {10, 20, 30}                         (3 pts)
  - Bu_FDCHI2_OWNPV > {100, 150, 200, 250}                (4 pts)
  - Bu_IPCHI2_OWNPV < {4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0} (7 pts)
  - Bu_PT           > {3000, 3200, 3400}                  (3 pts)
  - PID_product     > 0.20  [FIXED, not scanned]
  Total: 3 × 4 × 7 × 3 = 252 grid points

FOM Definitions:
  FOM1 = (N_jpsi + N_etac) / sqrt(N_bkg)          [S/sqrt(B), high-yield states]
  FOM2 = S2 / sqrt(S2 + N_bkg)                      [S/sqrt(S+B), Punzi FOM, low-yield states]
  where S2 = N_chic0 + N_chic1 + N_etac_2s

Output Files:
  output/scan_results.csv          — FOM1, FOM2 at all 252 grid points
  output/fom1_landscape.pdf        — 4-panel FOM1 landscape plot
  output/fom2_landscape.pdf        — 4-panel FOM2 landscape plot
  output/fits/mass_fit_set1_opt.pdf — final fit at Set1 optimum (J/psi, eta_c)
  output/fits/mass_fit_set2_opt.pdf — final fit at Set2 optimum (chi_c0/1, eta_c(2S))
  output/optimal_cuts.csv          — best cuts for Set1 and Set2
"""

import itertools
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import awkward as ak
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")

from modules.cache_manager import CacheManager
from modules.data_handler import TOMLConfig
from modules.exceptions import AnalysisError
from modules.mass_fitter import MassFitter

# ---------------------------------------------------------------------------
# Read Snakemake params
# ---------------------------------------------------------------------------
config_dir = snakemake.params.config_dir
cache_dir = snakemake.params.cache_dir
output_dir = snakemake.params.output_dir

csv_output = snakemake.output.csv
fom1_plot = snakemake.output.fom1
fom2_plot = snakemake.output.fom2
fit_set1_output = snakemake.output.fit_set1
fit_set2_output = snakemake.output.fit_set2
opt_csv_output = snakemake.output.opt_csv

years = snakemake.config.get("years", ["2016", "2017", "2018"])
track_types = snakemake.config.get("track_types", ["LL", "DD"])

Path(output_dir).mkdir(parents=True, exist_ok=True)
Path(fit_set1_output).parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------
PID_CUT = 0.20  # fixed PID_product pre-cut (not scanned)
BU_MIN = 5255.0  # MeV — B+ signal window (also enforced by MassFitter internally)
BU_MAX = 5305.0  # MeV
FIT_MIN = 2800.0  # MeV — M(cc-bar) fit range
FIT_MAX = 4000.0  # MeV

# Grid values
GRID = {
    "chi2": [10.0, 20.0, 30.0],
    "fdchi2": [100.0, 150.0, 200.0, 250.0],
    "ipchi2": [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0],
    "pt": [3000.0, 3200.0, 3400.0],
}
TOTAL_GRID = 1
for v in GRID.values():
    TOTAL_GRID *= len(v)

# Current split_charmonium_opt baseline (for landscape reference lines)
BASELINE_FOM1 = 16.855  # at PID=0 (no PID cut)
# After PID>0.20: from sideband_pid_validity study, FOM1 = 22.13
# (this is the reference we expect to beat or match with the full grid scan)
REFERENCE_FOM1_PID020 = 22.13  # at PID=0.20, Set1 cuts (chi2<30, FD>100, IP<6.5, PT>3000)

# ---------------------------------------------------------------------------
# Configuration and cache
# ---------------------------------------------------------------------------
config = TOMLConfig(config_dir)
config.paths["output"]["plots_dir"] = output_dir

cache = CacheManager(cache_dir)

config_files = list(Path(config_dir).glob("*.toml"))
code_files = [
    project_root / "modules" / "data_handler.py",
    project_root / "modules" / "lambda_selector.py",
]
step2_deps = cache.compute_dependencies(
    config_files=config_files,
    code_files=code_files,
    extra_params={"years": years, "track_types": track_types},
)

print("Loading Step 2 cached data...")
data_dict = cache.load("step2_data_after_lambda", dependencies=step2_deps)

if not data_dict:
    raise AnalysisError("Step 2 cached data not found! Run load_data first.")

# ---------------------------------------------------------------------------
# Build per-year and concatenated arrays
# ---------------------------------------------------------------------------
data_combined = {}
for year in data_dict:
    arrays = []
    for tt in data_dict[year]:
        if hasattr(data_dict[year][tt], "layout"):
            arrays.append(data_dict[year][tt])
    if arrays:
        data_combined[year] = ak.concatenate(arrays, axis=0)

all_data = ak.concatenate(list(data_combined.values()), axis=0)
print(f"Total events loaded: {len(all_data):,}")

# Branch name resolution (match split_charmonium_opt pattern)
mass_branch = "M_LpKm_h2" if "M_LpKm_h2" in all_data.fields else "M_LpKm"
bu_branch = "Bu_MM_corrected" if "Bu_MM_corrected" in all_data.fields else "Bu_M"


def flat(arr, branch):
    """Extract branch as flat 1D numpy array, dereferencing var-length if needed."""
    a = arr[branch]
    if "var" in str(ak.type(a)):
        a = ak.firsts(a)
    return ak.to_numpy(a)


# Pre-extract all branches needed for cuts (flat numpy arrays on concatenated data)
chi2_all = flat(all_data, "Bu_DTF_chi2")
fdchi2_all = flat(all_data, "Bu_FDCHI2_OWNPV")
ipchi2_all = flat(all_data, "Bu_IPCHI2_OWNPV")
pt_all = flat(all_data, "Bu_PT")
pid_all = flat(all_data, "PID_product")
bu_all = flat(all_data, bu_branch)

# Pre-apply PID and B+ window cuts (these are fixed)
pid_mask = pid_all > PID_CUT
bu_mask = (bu_all >= BU_MIN) & (bu_all <= BU_MAX)
base_mask = pid_mask & bu_mask

all_data_base = all_data[base_mask]
chi2_base = chi2_all[base_mask]
fdchi2_base = fdchi2_all[base_mask]
ipchi2_base = ipchi2_all[base_mask]
pt_base = pt_all[base_mask]

n_base = np.sum(base_mask)
print(f"Events after PID>0.20 + B+ window: {n_base:,} (was {len(all_data):,})")

# ---------------------------------------------------------------------------
# Phase 1: Grid Scan
# ---------------------------------------------------------------------------
print(f"\n{'=' * 65}")
print(f"Phase 1: Grid scan — {TOTAL_GRID} fits (PID fixed at {PID_CUT})")
print(f"{'=' * 65}")

records = []
n_failed = 0

with tqdm(total=TOTAL_GRID, desc="Grid scan", unit="fit") as pbar:
    for c, f, i, p in itertools.product(GRID["chi2"], GRID["fdchi2"], GRID["ipchi2"], GRID["pt"]):
        mask = (chi2_base < c) & (fdchi2_base > f) & (ipchi2_base < i) & (pt_base > p)
        n_events = int(np.sum(mask))
        data_for_fit = {"scan": all_data_base[mask]}

        fom1 = fom2 = np.nan
        n_jpsi = n_etac = n_chic0 = n_chic1 = n_etac_2s = n_bkg = np.nan

        if n_events < 50:
            # Too few events to fit reliably
            n_failed += 1
            records.append(
                {
                    "chi2": c,
                    "fdchi2": f,
                    "ipchi2": i,
                    "pt": p,
                    "n_events": n_events,
                    "N_jpsi": np.nan,
                    "N_etac": np.nan,
                    "N_chic0": np.nan,
                    "N_chic1": np.nan,
                    "N_etac_2s": np.nan,
                    "N_bkg": np.nan,
                    "FOM1": np.nan,
                    "FOM2": np.nan,
                }
            )
            pbar.update(1)
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            config.paths["output"]["plots_dir"] = tmpdir
            fitter = MassFitter(config)
            try:
                results = fitter.perform_fit(data_for_fit, fit_combined=True)
                # When passing a single-key dict {"scan": data}, MassFitter does NOT
                # create a combined dataset (requires len > 1). Results key is "scan".
                ylds = results["yields"]["scan"]

                n_jpsi = max(ylds.get("jpsi", (0, 0))[0], 0.0)
                n_etac = max(ylds.get("etac", (0, 0))[0], 0.0)
                n_chic0 = max(ylds.get("chic0", (0, 0))[0], 0.0)
                n_chic1 = max(ylds.get("chic1", (0, 0))[0], 0.0)
                n_etac_2s = max(ylds.get("etac_2s", (0, 0))[0], 0.0)
                n_bkg = max(ylds.get("background", (0, 0))[0], 1.0)

                s1 = n_jpsi + n_etac
                fom1 = s1 / np.sqrt(n_bkg)

                s2 = n_chic0 + n_chic1 + n_etac_2s
                fom2 = s2 / np.sqrt(max(s2 + n_bkg, 1.0))

            except Exception:
                n_failed += 1
                tb = traceback.format_exc()
                tqdm.write(
                    f"  WARN: fit failed at chi2<{c} FD>{f} IP<{i} PT>{p}: "
                    f"{tb.splitlines()[-1]}"
                )

        records.append(
            {
                "chi2": c,
                "fdchi2": f,
                "ipchi2": i,
                "pt": p,
                "n_events": n_events,
                "N_jpsi": n_jpsi,
                "N_etac": n_etac,
                "N_chic0": n_chic0,
                "N_chic1": n_chic1,
                "N_etac_2s": n_etac_2s,
                "N_bkg": n_bkg,
                "FOM1": fom1,
                "FOM2": fom2,
            }
        )
        pbar.update(1)

# Restore output dir
config.paths["output"]["plots_dir"] = output_dir

df_scan = pd.DataFrame(records)
df_scan.to_csv(csv_output, index=False)
print(f"\nScan complete. {n_failed}/{TOTAL_GRID} fits failed.")
print(f"Saved: {csv_output}")

# ---------------------------------------------------------------------------
# Phase 2: Landscape Plots
# ---------------------------------------------------------------------------
print(f"\n{'=' * 65}")
print("Phase 2: Landscape plots")
print(f"{'=' * 65}")

df_valid = df_scan.dropna(subset=["FOM1", "FOM2"])
n_valid = len(df_valid)
print(f"Valid grid points: {n_valid}/{TOTAL_GRID}")


def make_landscape(df, fom_col, fom_label, out_path, ref_line=None):
    """4-panel FOM landscape figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Fit-Based Optimizer — {fom_label} Landscape (PID > {PID_CUT})",
        fontsize=14,
        fontweight="bold",
    )

    cmap = "viridis"

    # --- Panel 1: FDCHI2 vs IPCHI2 (best over chi2 and PT) ---
    ax = axes[0, 0]
    pv1 = df.groupby(["fdchi2", "ipchi2"])[fom_col].max().reset_index()
    if len(pv1) > 0:
        pivot1 = pv1.pivot(index="fdchi2", columns="ipchi2", values=fom_col)
        im1 = ax.imshow(
            pivot1.values,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[
                pivot1.columns.min() - 0.25,
                pivot1.columns.max() + 0.25,
                pivot1.index.min() - 25,
                pivot1.index.max() + 25,
            ],
        )
        plt.colorbar(im1, ax=ax, label=fom_label)
        # Mark best point
        best_idx = pv1[fom_col].idxmax()
        bx = pv1.loc[best_idx, "ipchi2"]
        by = pv1.loc[best_idx, "fdchi2"]
        ax.plot(bx, by, "r*", markersize=12, label=f"Best: {pv1.loc[best_idx, fom_col]:.2f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("IPCHI2 cut (< x)", fontsize=10)
    ax.set_ylabel("FDCHI2 cut (> y)", fontsize=10)
    ax.set_title("Best over chi2, PT", fontsize=10)

    # --- Panel 2: chi2 vs PT (best over FDCHI2 and IPCHI2) ---
    ax = axes[0, 1]
    pv2 = df.groupby(["chi2", "pt"])[fom_col].max().reset_index()
    if len(pv2) > 0:
        pivot2 = pv2.pivot(index="chi2", columns="pt", values=fom_col)
        im2 = ax.imshow(
            pivot2.values,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[
                pivot2.columns.min() - 100,
                pivot2.columns.max() + 100,
                pivot2.index.min() - 5,
                pivot2.index.max() + 5,
            ],
        )
        plt.colorbar(im2, ax=ax, label=fom_label)
        best_idx = pv2[fom_col].idxmax()
        bx = pv2.loc[best_idx, "pt"]
        by = pv2.loc[best_idx, "chi2"]
        ax.plot(bx, by, "r*", markersize=12, label=f"Best: {pv2.loc[best_idx, fom_col]:.2f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("PT cut (> x) [MeV/c]", fontsize=10)
    ax.set_ylabel("DTF chi2 cut (< y)", fontsize=10)
    ax.set_title("Best over FDCHI2, IPCHI2", fontsize=10)

    # --- Panel 3: FOM vs IPCHI2 (line per FDCHI2) ---
    ax = axes[1, 0]
    pv3 = df.groupby(["fdchi2", "ipchi2"])[fom_col].max().reset_index()
    for fd_val, grp in pv3.groupby("fdchi2"):
        grp_s = grp.sort_values("ipchi2")
        ax.plot(grp_s["ipchi2"], grp_s[fom_col], marker="o", label=f"FD>{fd_val:.0f}", markersize=4)
    if ref_line is not None:
        ax.axhline(
            ref_line,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"Ref (PID=0.20): {ref_line:.2f}",
        )
    ax.set_xlabel("IPCHI2 cut (< x)", fontsize=10)
    ax.set_ylabel(fom_label, fontsize=10)
    ax.set_title("FOM vs IPCHI2 (best over chi2, PT)", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: FOM vs FDCHI2 (line per chi2) ---
    ax = axes[1, 1]
    pv4 = df.groupby(["chi2", "fdchi2"])[fom_col].max().reset_index()
    for chi2_val, grp in pv4.groupby("chi2"):
        grp_s = grp.sort_values("fdchi2")
        ax.plot(
            grp_s["fdchi2"], grp_s[fom_col], marker="s", label=f"chi2<{chi2_val:.0f}", markersize=4
        )
    if ref_line is not None:
        ax.axhline(
            ref_line,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"Ref (PID=0.20): {ref_line:.2f}",
        )
    ax.set_xlabel("FDCHI2 cut (> x)", fontsize=10)
    ax.set_ylabel(fom_label, fontsize=10)
    ax.set_title("FOM vs FDCHI2 (best over IPCHI2, PT)", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Find and annotate global best
    if len(df_valid) > 0:
        best_row = df.loc[df[fom_col].idxmax()]
        ann = (
            f"Global best {fom_label} = {best_row[fom_col]:.3f}\n"
            f"  chi2 < {best_row['chi2']:.0f}\n"
            f"  FDCHI2 > {best_row['fdchi2']:.0f}\n"
            f"  IPCHI2 < {best_row['ipchi2']:.1f}\n"
            f"  PT > {best_row['pt']:.0f} MeV/c\n"
            f"  PID_product > {PID_CUT}\n"
            f"  N_events = {int(best_row['n_events']):,}"
        )
        fig.text(
            0.02,
            0.01,
            ann,
            fontsize=8,
            family="monospace",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


make_landscape(
    df_valid,
    "FOM1",
    "FOM1 = (N_J/ψ + N_ηc) / √N_bkg",
    fom1_plot,
    ref_line=REFERENCE_FOM1_PID020,
)
make_landscape(
    df_valid,
    "FOM2",
    "FOM2 = S₂ / √(S₂ + N_bkg)",
    fom2_plot,
)

# ---------------------------------------------------------------------------
# Find optimal cut combinations
# ---------------------------------------------------------------------------
print(f"\n{'=' * 65}")
print("Finding optimal cut combinations...")
print(f"{'=' * 65}")

if df_valid.empty:
    raise AnalysisError("No valid grid points — all fits failed. Cannot determine optimal cuts.")

best1_idx = df_valid["FOM1"].idxmax()
best_cuts1 = df_valid.loc[best1_idx]
print("\nSet 1 (FOM1 = S/sqrt(B)) optimal:")
print(f"  chi2 < {best_cuts1['chi2']:.0f}")
print(f"  FDCHI2 > {best_cuts1['fdchi2']:.0f}")
print(f"  IPCHI2 < {best_cuts1['ipchi2']:.1f}")
print(f"  PT > {best_cuts1['pt']:.0f} MeV/c")
print(f"  PID_product > {PID_CUT}")
print(f"  FOM1 = {best_cuts1['FOM1']:.3f}  (N_events={int(best_cuts1['n_events']):,})")
print(
    f"  N_J/ψ = {best_cuts1['N_jpsi']:.0f},  N_ηc = {best_cuts1['N_etac']:.0f},  N_bkg = {best_cuts1['N_bkg']:.0f}"
)

best2_idx = df_valid["FOM2"].idxmax()
best_cuts2 = df_valid.loc[best2_idx]
print("\nSet 2 (FOM2 = S2/sqrt(S2+B)) optimal:")
print(f"  chi2 < {best_cuts2['chi2']:.0f}")
print(f"  FDCHI2 > {best_cuts2['fdchi2']:.0f}")
print(f"  IPCHI2 < {best_cuts2['ipchi2']:.1f}")
print(f"  PT > {best_cuts2['pt']:.0f} MeV/c")
print(f"  PID_product > {PID_CUT}")
print(f"  FOM2 = {best_cuts2['FOM2']:.3f}  (N_events={int(best_cuts2['n_events']):,})")
print(
    f"  N_χc0 = {best_cuts2['N_chic0']:.0f},  N_χc1 = {best_cuts2['N_chic1']:.0f},  N_bkg = {best_cuts2['N_bkg']:.0f}"
)

# Save optimal cuts
opt_rows = [
    {
        "Set": "Set1_HighYield",
        "FOM_formula": "S/sqrt(B)",
        "FOM_value": best_cuts1["FOM1"],
        "bu_dtf_chi2": best_cuts1["chi2"],
        "bu_fdchi2": best_cuts1["fdchi2"],
        "bu_ipchi2": best_cuts1["ipchi2"],
        "bu_pt": best_cuts1["pt"],
        "pid_product": PID_CUT,
        "n_events": int(best_cuts1["n_events"]),
        "N_jpsi": best_cuts1["N_jpsi"],
        "N_etac": best_cuts1["N_etac"],
        "N_bkg": best_cuts1["N_bkg"],
    },
    {
        "Set": "Set2_LowYield",
        "FOM_formula": "S/sqrt(S+B)",
        "FOM_value": best_cuts2["FOM2"],
        "bu_dtf_chi2": best_cuts2["chi2"],
        "bu_fdchi2": best_cuts2["fdchi2"],
        "bu_ipchi2": best_cuts2["ipchi2"],
        "bu_pt": best_cuts2["pt"],
        "pid_product": PID_CUT,
        "n_events": int(best_cuts2["n_events"]),
        "N_chic0": best_cuts2["N_chic0"],
        "N_chic1": best_cuts2["N_chic1"],
        "N_etac_2s": best_cuts2["N_etac_2s"],
        "N_bkg": best_cuts2["N_bkg"],
    },
]
pd.DataFrame(opt_rows).to_csv(opt_csv_output, index=False)
print(f"\nSaved: {opt_csv_output}")

# ---------------------------------------------------------------------------
# Phase 3: Final Fits at Optimal Cut Points
# ---------------------------------------------------------------------------
print(f"\n{'=' * 65}")
print("Phase 3: Final fits at optimal cut points")
print(f"{'=' * 65}")


def apply_cuts_and_fit(cuts_row, label):
    """Apply cut combination to per-year data and run full MassFitter fit."""
    c_chi2 = cuts_row["chi2"]
    c_fdchi2 = cuts_row["fdchi2"]
    c_ipchi2 = cuts_row["ipchi2"]
    c_pt = cuts_row["pt"]

    data_by_year = {}
    for year, data_year in data_combined.items():
        chi2_y = flat(data_year, "Bu_DTF_chi2")
        fdchi2_y = flat(data_year, "Bu_FDCHI2_OWNPV")
        ipchi2_y = flat(data_year, "Bu_IPCHI2_OWNPV")
        pt_y = flat(data_year, "Bu_PT")
        pid_y = flat(data_year, "PID_product")
        bu_y = flat(data_year, bu_branch)

        sel = (
            (chi2_y < c_chi2)
            & (fdchi2_y > c_fdchi2)
            & (ipchi2_y < c_ipchi2)
            & (pt_y > c_pt)
            & (pid_y > PID_CUT)
            & (bu_y >= BU_MIN)
            & (bu_y <= BU_MAX)
        )
        n_before = len(data_year)
        n_after = int(np.sum(sel))
        print(f"  {label} / {year}: {n_before:,} → {n_after:,} events")
        data_by_year[year] = data_year[sel]
    return data_by_year


# --- Final fit for Set 1 ---
print("\nSet 1 final fit:")
set1_data = apply_cuts_and_fit(best_cuts1, "Set1")

with tempfile.TemporaryDirectory() as tmpdir:
    config.paths["output"]["plots_dir"] = tmpdir
    fitter1 = MassFitter(config)
    results_set1 = fitter1.perform_fit(set1_data, fit_combined=True)
    src1 = Path(tmpdir) / "fits" / "mass_fit_combined.pdf"
    dst1 = Path(fit_set1_output)
    dst1.parent.mkdir(parents=True, exist_ok=True)
    if src1.exists():
        shutil.copy(src1, dst1)
        print(f"  Saved: {dst1}")
    else:
        # Try alternative name
        alt1 = Path(tmpdir) / "fits" / "mass_fit_scan.pdf"
        if alt1.exists():
            shutil.copy(alt1, dst1)
            print(f"  Saved (alt name): {dst1}")
        else:
            fits_dir = Path(tmpdir) / "fits"
            pdfs = list(fits_dir.glob("*.pdf")) if fits_dir.exists() else []
            if pdfs:
                shutil.copy(pdfs[-1], dst1)
                print(f"  Saved (fallback {pdfs[-1].name}): {dst1}")
            else:
                print(f"  WARNING: No mass_fit PDF found for Set1 in {tmpdir}")
                dst1.touch()

config.paths["output"]["plots_dir"] = output_dir

# --- Final fit for Set 2 ---
print("\nSet 2 final fit:")
set2_data = apply_cuts_and_fit(best_cuts2, "Set2")

with tempfile.TemporaryDirectory() as tmpdir:
    config.paths["output"]["plots_dir"] = tmpdir
    fitter2 = MassFitter(config)
    results_set2 = fitter2.perform_fit(set2_data, fit_combined=True)
    src2 = Path(tmpdir) / "fits" / "mass_fit_combined.pdf"
    dst2 = Path(fit_set2_output)
    dst2.parent.mkdir(parents=True, exist_ok=True)
    if src2.exists():
        shutil.copy(src2, dst2)
        print(f"  Saved: {dst2}")
    else:
        alt2 = Path(tmpdir) / "fits" / "mass_fit_scan.pdf"
        if alt2.exists():
            shutil.copy(alt2, dst2)
            print(f"  Saved (alt name): {dst2}")
        else:
            fits_dir = Path(tmpdir) / "fits"
            pdfs = list(fits_dir.glob("*.pdf")) if fits_dir.exists() else []
            if pdfs:
                shutil.copy(pdfs[-1], dst2)
                print(f"  Saved (fallback {pdfs[-1].name}): {dst2}")
            else:
                print(f"  WARNING: No mass_fit PDF found for Set2 in {tmpdir}")
                dst2.touch()

config.paths["output"]["plots_dir"] = output_dir

# ---------------------------------------------------------------------------
# Summary print
# ---------------------------------------------------------------------------
print(f"\n{'=' * 65}")
print("DONE — Fit-Based Optimizer Summary")
print(f"{'=' * 65}")
print(f"Grid: {TOTAL_GRID} points  |  Failed: {n_failed}  |  Valid: {n_valid}")
print("\nSet 1 (FOM1 = S/√B, high-yield J/ψ + ηc):")
print(f"  Optimal FOM1 = {best_cuts1['FOM1']:.3f}")
print(
    f"  chi2 < {best_cuts1['chi2']:.0f}, FDCHI2 > {best_cuts1['fdchi2']:.0f}, "
    f"IPCHI2 < {best_cuts1['ipchi2']:.1f}, PT > {best_cuts1['pt']:.0f}, PID > {PID_CUT}"
)
print("\nSet 2 (FOM2 = S₂/√(S₂+B), low-yield χc0/1 + ηc(2S)):")
print(f"  Optimal FOM2 = {best_cuts2['FOM2']:.3f}")
print(
    f"  chi2 < {best_cuts2['chi2']:.0f}, FDCHI2 > {best_cuts2['fdchi2']:.0f}, "
    f"IPCHI2 < {best_cuts2['ipchi2']:.1f}, PT > {best_cuts2['pt']:.0f}, PID > {PID_CUT}"
)
print("\nOutput files:")
print(f"  {csv_output}")
print(f"  {fom1_plot}")
print(f"  {fom2_plot}")
print(f"  {fit_set1_output}")
print(f"  {fit_set2_output}")
print(f"  {opt_csv_output}")
