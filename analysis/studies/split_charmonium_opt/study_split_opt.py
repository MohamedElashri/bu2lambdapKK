"""
Standalone study: Split Charmonium Optimization Study

Steps:
1. Estimate N_expected per state.
2. N-D grid scan optimizing sum(S)/sqrt(sum(B)) for Set 1
3. N-D grid scan optimizing sum(S)/(sqrt(sum(S))+sqrt(sum(B))) for Set 2
4. Apply cuts and stitch dataset at 3300 MeV.
5. Mass Fit the stitched dataset.
"""

import itertools
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import awkward as ak
import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

from modules.cache_manager import CacheManager
from modules.data_handler import TOMLConfig
from modules.exceptions import AnalysisError
from modules.mass_fitter import MassFitter

# --- Read Snakemake params ---
config_dir = snakemake.params.config_dir
cache_dir = snakemake.params.cache_dir
output_dir = snakemake.params.output_dir
csv_output = snakemake.output.csv

config = TOMLConfig(config_dir)
config.paths["output"]["plots_dir"] = output_dir

cache = CacheManager(cache_dir)

SET1_STATES = ["jpsi", "etac"]
SET2_STATES = ["chic0", "chic1", "etac_2s"]
ALL_STATES = SET1_STATES + SET2_STATES
MC_STATES = ["jpsi", "etac", "chic0", "chic1"]
MC_PROXY = {"etac_2s": "chic1"}

# Cutoff mass in MeV for data stitching
MASS_CUTOFF = 3300.0


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


years = snakemake.config.get("years", ["2016", "2017", "2018"])
track_types = snakemake.config.get("track_types", ["LL", "DD"])
step2_deps = compute_step_dependencies(
    step="2", extra_params={"years": years, "track_types": track_types}
)

# Load Data and MC
data_dict = cache.load("step2_data_after_lambda", dependencies=step2_deps)
mc_dict = cache.load("step2_mc_after_lambda", dependencies=step2_deps)

if not data_dict or not mc_dict:
    raise AnalysisError("Step 2 cached data/MC not found! Run load_data first.")

data_combined = {}
for year in data_dict:
    arrays = []
    for track_type in data_dict[year]:
        if hasattr(data_dict[year][track_type], "layout"):
            arrays.append(data_dict[year][track_type])
    if arrays:
        data_combined[year] = ak.concatenate(arrays, axis=0)

all_data = ak.concatenate(list(data_combined.values()), axis=0)

mc_combined = {}
for state in MC_STATES:
    if state not in mc_dict:
        continue
    arrays = []
    for year in mc_dict[state]:
        for track_type in mc_dict[state][year]:
            if hasattr(mc_dict[state][year][track_type], "layout"):
                arrays.append(mc_dict[state][year][track_type])
    if arrays:
        mc_combined[state] = ak.concatenate(arrays, axis=0)
if "chic1" in mc_combined:
    mc_combined["etac_2s"] = mc_combined["chic1"]

mass_branch = "M_LpKm_h2" if "M_LpKm_h2" in all_data.fields else "M_LpKm"
bu_mass_branch = "Bu_MM_corrected" if "Bu_MM_corrected" in all_data.fields else "Bu_M"

opt_config = config.selection.get("optimization_strategy", {})
b_sig_min = opt_config.get("b_signal_region_min", 5255.0)
b_sig_max = opt_config.get("b_signal_region_max", 5305.0)

in_b_sig = (all_data[bu_mass_branch] > b_sig_min) & (all_data[bu_mass_branch] < b_sig_max)
data_b_signal = all_data[in_b_sig]

# Grid scanner info
nd_config = config.selection.get("nd_optimizable_selection", {})
all_vars = []
grid_axes = []
cut_types = []
for var_name, var_cf in nd_config.items():
    if var_name == "notes":
        continue
    grid_points = np.arange(var_cf["begin"], var_cf["end"] + var_cf["step"] / 2, var_cf["step"])
    all_vars.append(
        {
            "var_name": var_name,
            "branch_name": var_cf["branch_name"],
            "cut_type": var_cf["cut_type"],
        }
    )
    cut_types.append(1 if var_cf["cut_type"] == "greater" else -1)
    grid_axes.append(grid_points)

cut_types = np.array(cut_types)

# Signal Regions
signal_regions = config.particles.get("signal_regions", {})
if not signal_regions:
    signal_regions = config.detector.get("signal_regions", {})
sb_low_mult = opt_config.get("sideband_low_multiplier", 4.0)
sb_high_mult = opt_config.get("sideband_high_multiplier", 4.0)

state_windows = {}
for state in ALL_STATES:
    sr = signal_regions.get(state, signal_regions.get(state.lower(), {}))
    center = sr.get("center", 0)
    w = sr.get("window", 0)
    state_windows[state] = {
        "sig_lo": center - w,
        "sig_hi": center + w,
        "low_sb": (center - sb_low_mult * w, center - w),
        "high_sb": (center + w, center + sb_high_mult * w),
        "sig_width": 2 * w,
        "low_sb_width": (sb_low_mult - 1) * w,
        "high_sb_width": (sb_high_mult - 1) * w,
    }

print("Estimating N_expected from data...")
data_mass = data_b_signal[mass_branch]
if "var" in str(ak.type(data_mass)):
    data_mass = ak.firsts(data_mass)
data_mass = ak.to_numpy(data_mass)

n_expected = {}
for state in ALL_STATES:
    sw = state_windows[state]
    n_sig = np.sum((data_mass > sw["sig_lo"]) & (data_mass < sw["sig_hi"]))
    n_low = np.sum((data_mass > sw["low_sb"][0]) & (data_mass < sw["low_sb"][1]))
    n_high = np.sum((data_mass > sw["high_sb"][0]) & (data_mass < sw["high_sb"][1]))

    d_low = n_low / sw["low_sb_width"] if sw["low_sb_width"] > 0 else 0
    d_high = n_high / sw["high_sb_width"] if sw["high_sb_width"] > 0 else 0
    b_est = ((d_low + d_high) / 2.0) * sw["sig_width"]
    n_expected[state] = max(n_sig - b_est, 1.0)
    print(f"  {state}: N_exp={n_expected[state]:.0f}")

mc_branches = {}
mc_totals = {}
for state in ALL_STATES:
    if state not in mc_combined:
        continue
    mc_totals[state] = len(mc_combined[state])
    branches = []
    for v in all_vars:
        br = mc_combined[state][v["branch_name"]]
        if "var" in str(ak.type(br)):
            br = ak.firsts(br)
        branches.append(ak.to_numpy(br))
    mc_branches[state] = np.column_stack(branches)

data_cut_branches = []
for v in all_vars:
    br = data_b_signal[v["branch_name"]]
    if "var" in str(ak.type(br)):
        br = ak.firsts(br)
    data_cut_branches.append(ak.to_numpy(br))
data_cut_branches = np.column_stack(data_cut_branches)

data_low_sb = {}
data_high_sb = {}
for state in ALL_STATES:
    sw = state_windows[state]
    data_low_sb[state] = (data_mass > sw["low_sb"][0]) & (data_mass < sw["low_sb"][1])
    data_high_sb[state] = (data_mass > sw["high_sb"][0]) & (data_mass < sw["high_sb"][1])

print("Running Joint Grid Scan (Vectorized)...")

best_fom1 = -np.inf
best_cuts1 = None
best_fom2 = -np.inf
best_cuts2 = None

total_combos = int(np.prod([len(a) for a in grid_axes]))


@njit
def eval_mask(data, cuts, cut_types):
    mask = np.ones(data.shape[0], dtype=np.bool_)
    for j in range(len(cuts)):
        if cut_types[j] == 1:
            mask = mask & (data[:, j] > cuts[j])
        else:
            mask = mask & (data[:, j] < cuts[j])
    return mask


with tqdm(total=total_combos) as pbar:
    for cut_combo in itertools.product(*grid_axes):
        ccombo = np.array(cut_combo)

        # Evaluate Data mask
        data_sel = eval_mask(data_cut_branches, ccombo, cut_types)

        S = {}
        B = {}
        for state in ALL_STATES:
            if state not in mc_branches:
                S[state] = 0.0
                B[state] = 1.0
                continue
            mc_sel = eval_mask(mc_branches[state], ccombo, cut_types)
            eps = np.sum(mc_sel) / mc_totals[state]
            S[state] = eps * n_expected[state]

            n_low = np.sum(data_sel & data_low_sb[state])
            n_high = np.sum(data_sel & data_high_sb[state])
            sw = state_windows[state]
            dl = n_low / sw["low_sb_width"] if sw["low_sb_width"] > 0 else 0
            dh = n_high / sw["high_sb_width"] if sw["high_sb_width"] > 0 else 0
            B[state] = ((dl + dh) / 2.0) * sw["sig_width"]

        s1 = sum(S[st] for st in SET1_STATES)
        b1 = sum(B[st] for st in SET1_STATES)
        fom1 = s1 / np.sqrt(max(b1, 1e-9))
        if fom1 > best_fom1:
            best_fom1 = fom1
            best_cuts1 = cut_combo

        s2 = sum(S[st] for st in SET2_STATES)
        b2 = sum(B[st] for st in SET2_STATES)
        fom2 = s2 / (np.sqrt(max(s2, 0)) + np.sqrt(max(b2, 0)) + 1e-9)
        if fom2 > best_fom2:
            best_fom2 = fom2
            best_cuts2 = cut_combo

        pbar.update(1)

print("\nBest FOM1 (Set 1):", best_fom1, "Cuts:", best_cuts1)
print("Best FOM2 (Set 2):", best_fom2, "Cuts:", best_cuts2)

res_rows = []
res_rows.append(
    {
        "Set": "Set1_HighYield",
        "FOM_formula": "S/sqrt(B)",
        "FOM_value": best_fom1,
        **{v["var_name"]: best_cuts1[i] for i, v in enumerate(all_vars)},
    }
)
res_rows.append(
    {
        "Set": "Set2_LowYield",
        "FOM_formula": "S/(sqrt(S)+sqrt(B))",
        "FOM_value": best_fom2,
        **{v["var_name"]: best_cuts2[i] for i, v in enumerate(all_vars)},
    }
)

df = pd.DataFrame(res_rows)
df.to_csv(csv_output, index=False)
print(f"Saved cut table to {csv_output}")

print("\nStitching data...")
stitched_data_by_year = {}
for year, data_year in data_combined.items():
    mass_arr = data_year[mass_branch]
    if "var" in str(ak.type(mass_arr)):
        mass_arr = ak.firsts(mass_arr)
    mass_arr = ak.to_numpy(mass_arr)

    mask_low = mass_arr < MASS_CUTOFF
    mask_high = mass_arr >= MASS_CUTOFF

    data_brs = []
    for var in all_vars:
        br = data_year[var["branch_name"]]
        if "var" in str(ak.type(br)):
            br = ak.firsts(br)
        data_brs.append(ak.to_numpy(br))
    data_brs = np.column_stack(data_brs)

    set1_mask = eval_mask(data_brs, np.array(best_cuts1), cut_types)
    set2_mask = eval_mask(data_brs, np.array(best_cuts2), cut_types)

    final_mask = (mask_low & set1_mask) | (mask_high & set2_mask)
    stitched_data_by_year[year] = data_year[final_mask]

    print(
        f"  {year}: original {len(data_year)}, after stitched cuts {len(stitched_data_by_year[year])}"
    )

print("\nFitting Stitched Dataset...")
fitter = MassFitter(config)
results = fitter.perform_fit(stitched_data_by_year, fit_combined=True)

import shutil

src_plot = Path(output_dir) / "fits" / "mass_fit_combined.pdf"
dst_plot = Path(output_dir) / "fits" / "mass_fit_combined_stitched.pdf"
if src_plot.exists():
    shutil.copy(src_plot, dst_plot)
    print("Done!")
else:
    print(f"Warning: {src_plot} was not created by MassFitter!")
