"""
Standalone study: Compare Figure of Merit formulas across charmonium states.

Motivation:
  J/psi and eta_c(1S) have high signal yield (B >> S regime)
  chi_c0, chi_c1, eta_c(2S) have low signal yield (S ~ B regime)
  Different FoM formulas may be more appropriate for each regime.

FoM definitions:
  FoM1 = S / sqrt(B)              -- optimal when B >> S (discovery-like)
  FoM2 = S / sqrt(S + B)          -- Punzi FOM for low-yield states

Approach (v3 -- MC-based signal, data-based background):
  1. Estimate N_expected per state from data (loose sideband subtraction, no cuts)
  2. Load signal MC per state (after Lambda cuts from Step 2)
  3. For each cut combination in the N-D grid:
     a. S = epsilon(cuts) * N_expected, where epsilon = N_MC_passing / N_MC_total
     b. B = data sideband interpolation in each state's mass window
  4. Evaluate both FoMs per (state, cut combination)
  This gives each (state, FoM) pair its own independently optimal cuts.

MC states available: jpsi, etac, chic0, chic1
  etac_2s has no MC -- uses chic1 MC as proxy (same as efficiency calculation)

Usage:
  cd analysis/studies/fom_comparison
  uv run snakemake -j1

This is a standalone study, NOT part of the main pipeline DAG.

Snakemake injects:
  snakemake.params.config_dir
  snakemake.params.cache_dir
  snakemake.params.output_dir
  snakemake.output[0]  -- summary CSV
  snakemake.output[1]  -- summary plot
"""

import itertools
import sys
from pathlib import Path

# Ensure the project root (analysis/) is on sys.path
# From studies/fom_comparison/ -> analysis/ is two levels up
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from modules.cache_manager import CacheManager
from modules.data_handler import TOMLConfig
from modules.exceptions import AnalysisError

# ---------------------------------------------------------------------------
# Read Snakemake params
# ---------------------------------------------------------------------------
config_dir = snakemake.params.config_dir  # noqa: F821
cache_dir = snakemake.params.cache_dir  # noqa: F821
output_dir = snakemake.params.output_dir  # noqa: F821
summary_csv = snakemake.output[0]  # noqa: F821
summary_plot = snakemake.output[1]  # noqa: F821

# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------
config = TOMLConfig(config_dir)
cache = CacheManager(cache_dir)

print("\n" + "=" * 80)
print("FoM COMPARISON STUDY (v3 -- MC signal + data background)")
print("=" * 80)
print("Comparing two Figure of Merit formulas across charmonium states:")
print("  FoM1 = S / sqrt(B)              (B >> S regime)")
print("  FoM2 = S / sqrt(S + B)          (Punzi FOM)")
print()
print("Signal estimation: S = epsilon(cuts) * N_expected")
print("  epsilon from MC, N_expected from data sideband subtraction (loose)")
print("Background estimation: B from data sideband interpolation per cut")
print("=" * 80)

# ---------------------------------------------------------------------------
# State classification
# ---------------------------------------------------------------------------
HIGH_YIELD_STATES = ["jpsi", "etac"]
LOW_YIELD_STATES = ["chic0", "chic1", "etac_2s"]
ALL_STATES = HIGH_YIELD_STATES + LOW_YIELD_STATES

# MC states available (etac_2s has no MC, uses chic1 as proxy)
MC_STATES = ["jpsi", "etac", "chic0", "chic1"]
MC_PROXY = {"etac_2s": "chic1"}  # state -> MC proxy mapping

STATE_LABELS = {
    "jpsi": r"$J/\psi$",
    "etac": r"$\eta_c(1S)$",
    "chic0": r"$\chi_{c0}$",
    "chic1": r"$\chi_{c1}$",
    "etac_2s": r"$\eta_c(2S)$",
}

STATE_GROUPS = {
    "jpsi": "high-yield",
    "etac": "high-yield",
    "chic0": "low-yield",
    "chic1": "low-yield",
    "etac_2s": "low-yield",
}


# ---------------------------------------------------------------------------
# FoM functions
# ---------------------------------------------------------------------------
def fom_s_over_sqrt_b(n_sig: float, n_bkg: float) -> float:
    """FoM1 = S / sqrt(B) -- standard for B >> S (measurement precision)."""
    if n_bkg <= 0:
        return 0.0
    return n_sig / np.sqrt(n_bkg)


def fom_s_over_sqrt_s_plus_b(n_sig: float, n_bkg: float) -> float:
    """FoM2 = S / sqrt(S + B) -- Punzi FOM for low-yield states."""
    denom = np.sqrt(max(n_sig + n_bkg, 0.0))
    if denom <= 0:
        return 0.0
    return n_sig / denom


FOM_FUNCTIONS = {
    "S/sqrt(B)": fom_s_over_sqrt_b,
    "S/sqrt(S+B)": fom_s_over_sqrt_s_plus_b,
}


# ---------------------------------------------------------------------------
# Helper: compute step dependencies (same as optimize_selection.py)
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
# Load Step 2 cached data and MC
# ---------------------------------------------------------------------------
years = snakemake.config.get("years", ["2016", "2017", "2018"])  # noqa: F821
track_types = snakemake.config.get("track_types", ["LL", "DD"])  # noqa: F821

step2_deps = compute_step_dependencies(
    step="2", extra_params={"years": years, "track_types": track_types}
)

# --- Real data ---
data_dict = cache.load("step2_data_after_lambda", dependencies=step2_deps)
if data_dict is None:
    raise AnalysisError(
        "Step 2 cached data not found! Run Step 2 (load_data) first.\n" "  snakemake load_data -j1"
    )

# --- Signal MC ---
mc_dict = cache.load("step2_mc_after_lambda", dependencies=step2_deps)
if mc_dict is None:
    raise AnalysisError(
        "Step 2 cached MC not found! Run Step 2 (load_data) first.\n" "  snakemake load_data -j1"
    )

# Combine data: track types and years
print("\n[Loading real data]")
data_combined = {}
for year in data_dict:
    arrays_to_combine = []
    for track_type in data_dict[year]:
        arr = data_dict[year][track_type]
        if hasattr(arr, "layout"):
            arrays_to_combine.append(arr)
    if arrays_to_combine:
        data_combined[year] = ak.concatenate(arrays_to_combine, axis=0)
        print(f"  data/{year}: {len(data_combined[year]):,} events")

all_data = ak.concatenate(list(data_combined.values()), axis=0)
print(f"  Total data (after Lambda cuts): {len(all_data):,}")

# Combine MC: track types and years per state
print("\n[Loading signal MC]")
mc_combined = {}
for state in MC_STATES:
    if state not in mc_dict:
        print(f"  WARNING: No MC for {state}, skipping")
        continue
    arrays_to_combine = []
    for year in mc_dict[state]:
        for track_type in mc_dict[state][year]:
            arr = mc_dict[state][year][track_type]
            if hasattr(arr, "layout"):
                arrays_to_combine.append(arr)
    if arrays_to_combine:
        mc_combined[state] = ak.concatenate(arrays_to_combine, axis=0)
        print(f"  MC/{state}: {len(mc_combined[state]):,} events (after Lambda cuts)")

# Add etac_2s proxy
if "chic1" in mc_combined:
    mc_combined["etac_2s"] = mc_combined["chic1"]
    print(f"  MC/etac_2s: using chic1 MC as proxy ({len(mc_combined['etac_2s']):,} events)")

# ---------------------------------------------------------------------------
# Identify mass branches
# ---------------------------------------------------------------------------
mass_branch = "M_LpKm_h2" if "M_LpKm_h2" in all_data.fields else "M_LpKm"
bu_mass_branch = "Bu_MM_corrected" if "Bu_MM_corrected" in all_data.fields else "Bu_M"

# ---------------------------------------------------------------------------
# Pre-select data in B+ signal region (for background estimation)
# ---------------------------------------------------------------------------
opt_config = config.selection.get("optimization_strategy", {})
b_sig_min = opt_config.get("b_signal_region_min", 5255.0)
b_sig_max = opt_config.get("b_signal_region_max", 5305.0)

bu_mass = all_data[bu_mass_branch]
in_b_signal = (bu_mass > b_sig_min) & (bu_mass < b_sig_max)
data_b_signal = all_data[in_b_signal]

print(
    f"\n  Data in B+ signal region [{b_sig_min:.0f}, {b_sig_max:.0f}]: " f"{len(data_b_signal):,}"
)

# ---------------------------------------------------------------------------
# Build grid scan variables
# ---------------------------------------------------------------------------
nd_config = config.selection.get("nd_optimizable_selection", {})
if not nd_config:
    raise AnalysisError("No 'nd_optimizable_selection' in config/selection.toml!")

all_variables = []
grid_axes = []

for var_name, var_config in nd_config.items():
    if var_name == "notes":
        continue
    begin = var_config["begin"]
    end = var_config["end"]
    step = var_config["step"]
    grid_points = np.arange(begin, end + step / 2, step)
    all_variables.append(
        {
            "var_name": var_name,
            "branch_name": var_config["branch_name"],
            "cut_type": var_config["cut_type"],
            "description": var_config.get("description", ""),
        }
    )
    grid_axes.append(grid_points)

total_combos = int(np.prod([len(ax) for ax in grid_axes]))
print(f"\n  Grid scan: {len(all_variables)} variables, {total_combos:,} combinations")

# ---------------------------------------------------------------------------
# Get signal regions for all states
# ---------------------------------------------------------------------------
signal_regions = config.particles.get("signal_regions", {})
if not signal_regions:
    signal_regions = config.detector.get("signal_regions", {})

# Use sideband multipliers from config (matching SelectionOptimizer)
sb_low_mult = opt_config.get("sideband_low_multiplier", 4.0)
sb_low_end_mult = opt_config.get("sideband_low_end_multiplier", 1.0)
sb_high_start_mult = opt_config.get("sideband_high_start_multiplier", 1.0)
sb_high_mult = opt_config.get("sideband_high_multiplier", 4.0)

# Pre-compute mass windows and sideband regions for each state
state_windows = {}
for state in ALL_STATES:
    sr = signal_regions.get(state, signal_regions.get(state.lower(), {}))
    center = sr.get("center", 0)
    window = sr.get("window", 0)
    sig_lo, sig_hi = center - window, center + window
    low_sb = (center - sb_low_mult * window, center - sb_low_end_mult * window)
    high_sb = (center + sb_high_start_mult * window, center + sb_high_mult * window)
    low_sb_width = low_sb[1] - low_sb[0]
    high_sb_width = high_sb[1] - high_sb[0]
    state_windows[state] = {
        "center": center,
        "window": window,
        "sig_lo": sig_lo,
        "sig_hi": sig_hi,
        "low_sb": low_sb,
        "high_sb": high_sb,
        "sig_width": 2 * window,
        "low_sb_width": low_sb_width,
        "high_sb_width": high_sb_width,
    }
    print(
        f"  {state:10s}: signal [{sig_lo:.0f}, {sig_hi:.0f}] MeV, "
        f"sidebands [{low_sb[0]:.0f},{low_sb[1]:.0f}] + "
        f"[{high_sb[0]:.0f},{high_sb[1]:.0f}]"
    )

# ---------------------------------------------------------------------------
# Step A: Estimate N_expected per state from data (loose sideband subtraction)
# No selection cuts applied -- just Lambda cuts + B+ signal region
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP A: Estimate N_expected per state (loose sideband subtraction)")
print("=" * 80)

data_mass = data_b_signal[mass_branch]
n_expected = {}

for state in ALL_STATES:
    sw = state_windows[state]

    # Count in signal region
    n_sig_region = float(ak.sum((data_mass > sw["sig_lo"]) & (data_mass < sw["sig_hi"])))

    # Count in sidebands
    n_low = float(ak.sum((data_mass > sw["low_sb"][0]) & (data_mass < sw["low_sb"][1])))
    n_high = float(ak.sum((data_mass > sw["high_sb"][0]) & (data_mass < sw["high_sb"][1])))

    # Background estimate via sideband interpolation
    density_low = n_low / sw["low_sb_width"] if sw["low_sb_width"] > 0 else 0
    density_high = n_high / sw["high_sb_width"] if sw["high_sb_width"] > 0 else 0
    avg_density = (density_low + density_high) / 2.0
    bkg_in_signal = avg_density * sw["sig_width"]

    n_exp = max(n_sig_region - bkg_in_signal, 1.0)  # floor at 1 to avoid zero
    n_expected[state] = n_exp

    print(
        f"  {state:10s}: N_signal_region={n_sig_region:.0f}, "
        f"B_estimate={bkg_in_signal:.0f}, "
        f"N_expected={n_exp:.0f}"
    )

# ---------------------------------------------------------------------------
# Step B: Pre-extract MC branch arrays for efficiency calculation
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP B: Prepare MC arrays for efficiency calculation")
print("=" * 80)

mc_branches = {}  # {state: [branch_array_per_variable]}
mc_totals = {}  # {state: total MC events after Lambda cuts}

for state in ALL_STATES:
    if state not in mc_combined:
        print(f"  WARNING: No MC for {state}, will skip in grid scan")
        continue

    mc_events = mc_combined[state]
    mc_totals[state] = len(mc_events)

    branches = []
    for var in all_variables:
        br = mc_events[var["branch_name"]]
        if "var" in str(ak.type(br)):
            br = ak.firsts(br)
        branches.append(br)
    mc_branches[state] = branches

    print(f"  MC/{state}: {mc_totals[state]:,} events, " f"N_expected={n_expected[state]:.0f}")

# ---------------------------------------------------------------------------
# Step C: Pre-extract data branch arrays for background estimation
# ---------------------------------------------------------------------------
data_cut_branches = []
for var in all_variables:
    br = data_b_signal[var["branch_name"]]
    if "var" in str(ak.type(br)):
        br = ak.firsts(br)
    data_cut_branches.append(br)

# Pre-compute data mass region masks (before selection cuts)
state_data_low_sb_masks = {}
state_data_high_sb_masks = {}
for state in ALL_STATES:
    sw = state_windows[state]
    state_data_low_sb_masks[state] = (data_mass > sw["low_sb"][0]) & (data_mass < sw["low_sb"][1])
    state_data_high_sb_masks[state] = (data_mass > sw["high_sb"][0]) & (
        data_mass < sw["high_sb"][1]
    )

# ---------------------------------------------------------------------------
# GRID SCAN: For each cut combination, compute S and B per state
#   S = epsilon(cuts) * N_expected   [epsilon from MC]
#   B = sideband interpolation       [from data]
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Running GRID SCAN: MC-based S, data-based B")
print(f"  {total_combos:,} combinations x {len(ALL_STATES)} states x " f"{len(FOM_FUNCTIONS)} FoMs")
print("=" * 80)

# Track best (cuts, fom, S, B, epsilon) for each (state, fom_name)
best_results = {}
for state in ALL_STATES:
    for fom_name in FOM_FUNCTIONS:
        best_results[(state, fom_name)] = {
            "best_fom": -np.inf,
            "best_cuts": None,
            "best_n_sig": 0.0,
            "best_n_bkg": 0.0,
            "best_epsilon": 0.0,
        }

with tqdm(total=total_combos, desc="  Grid scan", unit="combo", ncols=100) as pbar:
    for cut_combination in itertools.product(*grid_axes):

        # --- Apply cuts to DATA (for B estimation) ---
        data_sel_mask = ak.ones_like(data_cut_branches[0], dtype=bool)
        for j, (cut_val, var) in enumerate(zip(cut_combination, all_variables, strict=False)):
            if var["cut_type"] == "greater":
                data_sel_mask = data_sel_mask & (data_cut_branches[j] > cut_val)
            else:
                data_sel_mask = data_sel_mask & (data_cut_branches[j] < cut_val)

        # --- Apply cuts to MC (for epsilon estimation) ---
        mc_sel_masks = {}
        for state in ALL_STATES:
            if state not in mc_branches:
                continue
            mc_mask = ak.ones_like(mc_branches[state][0], dtype=bool)
            for j, (cut_val, var) in enumerate(zip(cut_combination, all_variables, strict=False)):
                if var["cut_type"] == "greater":
                    mc_mask = mc_mask & (mc_branches[state][j] > cut_val)
                else:
                    mc_mask = mc_mask & (mc_branches[state][j] < cut_val)
            mc_sel_masks[state] = mc_mask

        # --- For each state, compute S and B ---
        for state in ALL_STATES:
            if state not in mc_branches:
                continue

            sw = state_windows[state]

            # Signal: S = epsilon * N_expected
            n_mc_pass = float(ak.sum(mc_sel_masks[state]))
            epsilon = n_mc_pass / mc_totals[state] if mc_totals[state] > 0 else 0.0
            sig_estimate = epsilon * n_expected[state]

            # Background: sideband interpolation from data
            n_low_sb = float(ak.sum(data_sel_mask & state_data_low_sb_masks[state]))
            n_high_sb = float(ak.sum(data_sel_mask & state_data_high_sb_masks[state]))

            density_low = n_low_sb / sw["low_sb_width"] if sw["low_sb_width"] > 0 else 0
            density_high = n_high_sb / sw["high_sb_width"] if sw["high_sb_width"] > 0 else 0
            avg_density = (density_low + density_high) / 2.0
            bkg_estimate = avg_density * sw["sig_width"]

            # Evaluate both FoMs
            for fom_name, fom_func in FOM_FUNCTIONS.items():
                fom_val = fom_func(sig_estimate, bkg_estimate)
                entry = best_results[(state, fom_name)]
                if fom_val > entry["best_fom"]:
                    entry["best_fom"] = fom_val
                    entry["best_cuts"] = cut_combination
                    entry["best_n_sig"] = sig_estimate
                    entry["best_n_bkg"] = bkg_estimate
                    entry["best_epsilon"] = epsilon

        pbar.update(1)

# ---------------------------------------------------------------------------
# Build results table
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("RESULTS: Optimal cuts per (state, FoM)")
print("=" * 80)

result_rows = []
for state in ALL_STATES:
    for fom_name in FOM_FUNCTIONS:
        entry = best_results[(state, fom_name)]
        if entry["best_cuts"] is None:
            continue

        row = {
            "state": state,
            "state_label": STATE_LABELS[state],
            "group": STATE_GROUPS[state],
            "fom_formula": fom_name,
            "best_fom": entry["best_fom"],
            "epsilon": entry["best_epsilon"],
            "n_expected": n_expected[state],
            "sig_estimate": entry["best_n_sig"],
            "bkg_estimate": entry["best_n_bkg"],
            "s_over_b": entry["best_n_sig"] / max(entry["best_n_bkg"], 1),
        }
        for j, var in enumerate(all_variables):
            row[var["var_name"]] = entry["best_cuts"][j]
        result_rows.append(row)

        # Also evaluate the OTHER FoM at this point for cross-comparison
        other_fom_name = [k for k in FOM_FUNCTIONS if k != fom_name][0]
        other_fom_val = FOM_FUNCTIONS[other_fom_name](entry["best_n_sig"], entry["best_n_bkg"])
        row[f"cross_fom_{other_fom_name}"] = other_fom_val

df_results = pd.DataFrame(result_rows)

# Print summary
for state in ALL_STATES:
    group = STATE_GROUPS[state]
    print(f"\n  {STATE_LABELS[state]} ({group}, N_expected={n_expected[state]:.0f}):")
    for fom_name in FOM_FUNCTIONS:
        entry = best_results[(state, fom_name)]
        if entry["best_cuts"] is None:
            print(f"    {fom_name}: no valid cuts found")
            continue
        s, b, eps = entry["best_n_sig"], entry["best_n_bkg"], entry["best_epsilon"]
        sb = s / max(b, 1)
        print(
            f"    {fom_name:25s}: FoM={entry['best_fom']:8.3f}  "
            f"eps={eps:.4f}  S={s:6.1f}  B={b:6.1f}  S/B={sb:.3f}"
        )
        cuts_str = "  ".join(
            f"{var['var_name']}={entry['best_cuts'][j]:.2f}" for j, var in enumerate(all_variables)
        )
        print(f"      cuts: {cuts_str}")

    # Check if the two FoMs give different cuts
    e1 = best_results[(state, "S/sqrt(B)")]
    e2 = best_results[(state, "S/sqrt(S+B)")]
    if e1["best_cuts"] is not None and e2["best_cuts"] is not None:
        cuts_match = all(
            abs(e1["best_cuts"][j] - e2["best_cuts"][j]) < 1e-6 for j in range(len(all_variables))
        )
        if cuts_match:
            print("    --> SAME optimal cuts for both FoMs")
        else:
            diffs = []
            for j, var in enumerate(all_variables):
                v1, v2 = e1["best_cuts"][j], e2["best_cuts"][j]
                if abs(v1 - v2) > 1e-6:
                    diffs.append(f"{var['var_name']}: {v1:.2f} vs {v2:.2f}")
            print(f"    --> DIFFERENT cuts: {', '.join(diffs)}")

# Save CSV
output_path = Path(summary_csv)
output_path.parent.mkdir(exist_ok=True, parents=True)
df_results.to_csv(output_path, index=False)
print(f"\n  Summary saved to {summary_csv}")

# ---------------------------------------------------------------------------
# Generate comparison plots
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Generating comparison plots...")
print("=" * 80)

plot_path = Path(summary_plot)
plot_path.parent.mkdir(exist_ok=True, parents=True)

with PdfPages(plot_path) as pdf:
    # --- Page 1: Per-state S, B, and FoM comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("FoM Comparison: MC-based S, Data-based B", fontsize=14, y=0.98)

    df_fom1 = df_results[df_results["fom_formula"] == "S/sqrt(B)"]
    df_fom2 = df_results[df_results["fom_formula"] == "S/sqrt(S+B)"]

    states_ordered = ALL_STATES
    x = np.arange(len(states_ordered))
    width = 0.35

    # Panel (a): Signal estimate
    ax = axes[0, 0]
    s1 = [df_fom1[df_fom1["state"] == s]["sig_estimate"].values[0] for s in states_ordered]
    s2 = [df_fom2[df_fom2["state"] == s]["sig_estimate"].values[0] for s in states_ordered]
    ax.bar(x - width / 2, s1, width, label=r"Opt. with $S/\sqrt{B}$", color="steelblue")
    ax.bar(x + width / 2, s2, width, label=r"Opt. with $S/\sqrt{S+B}$", color="coral")
    ax.set_ylabel("Estimated signal yield")
    ax.set_title("(a) Signal estimate per state")
    ax.set_xticks(x)
    ax.set_xticklabels([STATE_LABELS[s] for s in states_ordered], fontsize=9)
    ax.legend(fontsize=8)
    ax.set_yscale("symlog", linthresh=1)
    ax.axvspan(1.5, 4.5, alpha=0.08, color="red")

    # Panel (b): Background estimate
    ax = axes[0, 1]
    b1 = [df_fom1[df_fom1["state"] == s]["bkg_estimate"].values[0] for s in states_ordered]
    b2 = [df_fom2[df_fom2["state"] == s]["bkg_estimate"].values[0] for s in states_ordered]
    ax.bar(x - width / 2, b1, width, label=r"Opt. with $S/\sqrt{B}$", color="steelblue")
    ax.bar(x + width / 2, b2, width, label=r"Opt. with $S/\sqrt{S+B}$", color="coral")
    ax.set_ylabel("Estimated background")
    ax.set_title("(b) Background estimate per state")
    ax.set_xticks(x)
    ax.set_xticklabels([STATE_LABELS[s] for s in states_ordered], fontsize=9)
    ax.legend(fontsize=8)
    ax.set_yscale("symlog", linthresh=1)
    ax.axvspan(1.5, 4.5, alpha=0.08, color="red")

    # Panel (c): S/B ratio
    ax = axes[1, 0]
    sb1 = [df_fom1[df_fom1["state"] == s]["s_over_b"].values[0] for s in states_ordered]
    sb2 = [df_fom2[df_fom2["state"] == s]["s_over_b"].values[0] for s in states_ordered]
    ax.bar(x - width / 2, sb1, width, label=r"Opt. with $S/\sqrt{B}$", color="steelblue")
    ax.bar(x + width / 2, sb2, width, label=r"Opt. with $S/\sqrt{S+B}$", color="coral")
    ax.set_ylabel("S/B ratio")
    ax.set_title("(c) S/B ratio per state")
    ax.set_xticks(x)
    ax.set_xticklabels([STATE_LABELS[s] for s in states_ordered], fontsize=9)
    ax.legend(fontsize=8)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.axvspan(1.5, 4.5, alpha=0.08, color="red")

    # Panel (d): Best FoM value (each using its own metric)
    ax = axes[1, 1]
    fom1_vals = [df_fom1[df_fom1["state"] == s]["best_fom"].values[0] for s in states_ordered]
    fom2_vals = [df_fom2[df_fom2["state"] == s]["best_fom"].values[0] for s in states_ordered]
    ax.bar(x - width / 2, fom1_vals, width, label=r"$S/\sqrt{B}$ (own metric)", color="steelblue")
    ax.bar(
        x + width / 2,
        fom2_vals,
        width,
        label=r"$S/\sqrt{S+B}$ (own metric)",
        color="coral",
    )
    ax.set_ylabel("Best FoM value")
    ax.set_title("(d) Best FoM value per state")
    ax.set_xticks(x)
    ax.set_xticklabels([STATE_LABELS[s] for s in states_ordered], fontsize=9)
    ax.legend(fontsize=8)
    ax.axvspan(1.5, 4.5, alpha=0.08, color="red")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 2: Cut comparison per variable ---
    n_vars = len(all_variables)
    fig, axes = plt.subplots((n_vars + 1) // 2, 2, figsize=(14, 3.5 * ((n_vars + 1) // 2)))
    axes = axes.flatten()
    fig.suptitle("Optimal Cut Values per State and FoM", fontsize=14, y=1.0)

    for j, var in enumerate(all_variables):
        ax = axes[j]
        v1 = [df_fom1[df_fom1["state"] == s][var["var_name"]].values[0] for s in states_ordered]
        v2 = [df_fom2[df_fom2["state"] == s][var["var_name"]].values[0] for s in states_ordered]
        ax.bar(x - width / 2, v1, width, label=r"$S/\sqrt{B}$", color="steelblue")
        ax.bar(x + width / 2, v2, width, label=r"$S/\sqrt{S+B}$", color="coral")
        ax.set_ylabel("Cut value")
        direction = ">" if var["cut_type"] == "greater" else "<"
        ax.set_title(f"{var['var_name']} ({direction})")
        ax.set_xticks(x)
        ax.set_xticklabels([STATE_LABELS[s] for s in states_ordered], fontsize=8)
        ax.legend(fontsize=7)
        ax.axvspan(1.5, 4.5, alpha=0.08, color="red")

    # Hide unused axes
    for j in range(n_vars, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 3: Theoretical FoM behaviour ---
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("FoM Behaviour vs S/B Ratio", fontsize=14)

    sb_range = np.linspace(0.001, 2.0, 500)
    B_fixed = 100
    S_range = sb_range * B_fixed
    fom1_curve = S_range / np.sqrt(B_fixed)
    fom2_curve = S_range / (np.sqrt(S_range) + np.sqrt(B_fixed))

    ax.plot(
        sb_range,
        fom1_curve / fom1_curve.max(),
        "b-",
        linewidth=2,
        label=r"$S/\sqrt{B}$ (linear in S)",
    )
    ax.plot(
        sb_range,
        fom2_curve / fom2_curve.max(),
        "r-",
        linewidth=2,
        label=r"$S/\sqrt{S+B}$ (saturates)",
    )
    ax.set_xlabel("S/B ratio")
    ax.set_ylabel("Normalised FoM")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Mark our states' S/B values (from FoM1 optimisation)
    for state in ALL_STATES:
        row = df_fom1[df_fom1["state"] == state]
        if len(row) == 0:
            continue
        sb_val = row["s_over_b"].values[0]
        if 0.001 <= sb_val <= 2.0:
            color = "steelblue" if state in HIGH_YIELD_STATES else "coral"
            ax.axvline(sb_val, color=color, linestyle=":", alpha=0.7)
            ax.text(
                sb_val,
                1.03,
                STATE_LABELS[state],
                rotation=45,
                fontsize=9,
                ha="left",
                va="bottom",
                color=color,
                transform=ax.get_xaxis_transform(),
            )

    ax.set_xlim(0, 2.0)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # --- Page 4: Summary table ---
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("off")
    ax.set_title(
        "Per-State Optimal Cuts: FoM1 vs FoM2\n"
        r"$S = \varepsilon \times N_{\rm expected}$ (MC), "
        r"$B$ from data sidebands",
        fontsize=13,
        pad=20,
    )

    # Build table: one row per state, columns show cuts from each FoM
    col_labels = ["State", "Group", r"$N_{\rm exp}$"]
    for var in all_variables:
        col_labels.append(f"{var['var_name']}\nFoM1")
        col_labels.append(f"{var['var_name']}\nFoM2")
    col_labels.extend(
        [
            r"$\varepsilon_1$",
            r"$\varepsilon_2$",
            "S (FoM1)",
            "S (FoM2)",
            "B (FoM1)",
            "B (FoM2)",
        ]
    )

    table_data = []
    for state in ALL_STATES:
        e1 = best_results[(state, "S/sqrt(B)")]
        e2 = best_results[(state, "S/sqrt(S+B)")]
        if e1["best_cuts"] is None or e2["best_cuts"] is None:
            continue
        row = [STATE_LABELS[state], STATE_GROUPS[state], f"{n_expected[state]:.0f}"]
        for j, var in enumerate(all_variables):
            v1 = e1["best_cuts"][j]
            v2 = e2["best_cuts"][j]
            row.append(f"{v1:.2f}")
            row.append(f"{v2:.2f}")
        row.extend(
            [
                f"{e1['best_epsilon']:.4f}",
                f"{e2['best_epsilon']:.4f}",
                f"{e1['best_n_sig']:.1f}",
                f"{e2['best_n_sig']:.1f}",
                f"{e1['best_n_bkg']:.1f}",
                f"{e2['best_n_bkg']:.1f}",
            ]
        )
        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.6)

    # Colour header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=6)

    # Highlight cells where FoM1 and FoM2 cuts differ
    # Column layout: State, Group, N_exp, then pairs of (var_FoM1, var_FoM2), then eps/S/B
    n_prefix_cols = 3  # State, Group, N_exp
    for i, state in enumerate(ALL_STATES):
        e1 = best_results[(state, "S/sqrt(B)")]
        e2 = best_results[(state, "S/sqrt(S+B)")]
        if e1["best_cuts"] is None or e2["best_cuts"] is None:
            continue
        for j in range(len(all_variables)):
            v1, v2 = e1["best_cuts"][j], e2["best_cuts"][j]
            col_fom1 = n_prefix_cols + 2 * j
            col_fom2 = n_prefix_cols + 2 * j + 1
            if abs(v1 - v2) > 1e-6:
                table[i + 1, col_fom1].set_facecolor("#FFF2CC")
                table[i + 1, col_fom2].set_facecolor("#FFF2CC")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()


# ---------------------------------------------------------------------------
# Print final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("FoM COMPARISON STUDY COMPLETE")
print("=" * 80)
print("\n  Outputs:")
print(f"    {summary_csv}")
print(f"    {summary_plot}")

# Count how many states have different optimal cuts
n_differ = 0
for state in ALL_STATES:
    e1 = best_results[(state, "S/sqrt(B)")]
    e2 = best_results[(state, "S/sqrt(S+B)")]
    if e1["best_cuts"] is not None and e2["best_cuts"] is not None:
        if not all(
            abs(e1["best_cuts"][j] - e2["best_cuts"][j]) < 1e-6 for j in range(len(all_variables))
        ):
            n_differ += 1

print(f"\n  States with DIFFERENT optimal cuts between FoMs: {n_differ}/{len(ALL_STATES)}")
if n_differ > 0:
    print("  --> FoM choice MATTERS for these states")
    print("  --> Consider state-dependent optimisation (Option B)")
else:
    print("  --> FoM choice does not affect optimal cuts")
    print("  --> Either FoM can be used")
print("=" * 80)
