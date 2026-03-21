import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tomli

# Ensure the project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.config_loader import StudyConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# States whose MC simulation is in the LHCb production pipeline.
# They are excluded from the main branching-fraction table until real MC arrives.
MC_PENDING_STATES = {"etac_2s"}

if "snakemake" in globals():
    no_cache = snakemake.params.get("no_cache", False)
    config_dir = snakemake.params.config_dir
    cache_dir = snakemake.params.cache_dir
    output_dir = snakemake.params.output_dir
    branch = snakemake.params.branch
    # Phase 1: separate LL and DD inputs
    yields_ll_file = snakemake.input.yields_ll
    yields_dd_file = snakemake.input.yields_dd
    efficiencies_ll_file = snakemake.input.efficiencies_ll
    efficiencies_dd_file = snakemake.input.efficiencies_dd
    ratios_ll_file = snakemake.input.ratios_ll
    ratios_dd_file = snakemake.input.ratios_dd
    br_ratios_file = snakemake.output.br_ratios
    final_results_file = snakemake.output.final_results
else:
    no_cache = False
    config_dir = "config"
    cache_dir = "analysis_output/box/cache"
    output_dir = "analysis_output/box"
    branch = "high_yield"
    yields_ll_file = Path(output_dir) / branch / "LL" / "tables" / "fitted_yields.csv"
    yields_dd_file = Path(output_dir) / branch / "DD" / "tables" / "fitted_yields.csv"
    efficiencies_ll_file = Path(output_dir) / branch / "LL" / "tables" / "efficiencies.csv"
    efficiencies_dd_file = Path(output_dir) / branch / "DD" / "tables" / "efficiencies.csv"
    ratios_ll_file = Path(output_dir) / branch / "LL" / "tables" / "efficiency_ratios.csv"
    ratios_dd_file = Path(output_dir) / branch / "DD" / "tables" / "efficiency_ratios.csv"
    br_ratios_file = Path(output_dir) / branch / "tables" / "branching_fraction_ratios.csv"
    final_results_file = Path(output_dir) / branch / "results" / "final_results.md"

config_path = Path(config_dir) / "selection.toml"
physics_path = Path(config_dir) / "physics.toml"

# Systematic uncertainties are NOT loaded here — they are applied at the export stage
# by export_latex_results.py (which runs after compute_systematics).
# This avoids a circular Snakemake dependency:
#   branching_ratios → branching_fraction_ratios.csv → compute_systematics → systematics.json
#                                                                           → export_latex_results

config = StudyConfig(config_file=str(config_path), output_dir=output_dir)

# Load physics constants
with open(physics_path, "rb") as f:
    physics_data = tomli.load(f)
pdg_bf = physics_data.get("pdg_branching_fractions", {})

br_bu_jpsi_k = pdg_bf.get("bu_to_jpsi_k", {}).get("value", 1.0)
br_bu_jpsi_k_err = pdg_bf.get("bu_to_jpsi_k", {}).get("error", 0.0)
br_jpsi_lpkk = pdg_bf.get("jpsi_to_lpkk", {}).get("value", 1.0)
br_jpsi_lpkk_err = pdg_bf.get("jpsi_to_lpkk", {}).get("error", 0.0)

norm_factor = br_bu_jpsi_k * br_jpsi_lpkk
norm_factor_rel_err = (
    np.sqrt((br_bu_jpsi_k_err / br_bu_jpsi_k) ** 2 + (br_jpsi_lpkk_err / br_jpsi_lpkk) ** 2)
    if norm_factor > 0
    else 0
)

# ---- Phase 1: combine LL and DD yields ----
# Yields from LL and DD fits are summed per state.
# Errors are added in quadrature (fits are statistically independent).
df_ll = pd.read_csv(yields_ll_file)
df_dd = pd.read_csv(yields_dd_file)
df_yields_all = pd.concat([df_ll, df_dd], ignore_index=True)

combined_yields = (
    df_yields_all.groupby("state")[["yield", "yield_err"]]
    .agg(
        yield_sum=("yield", "sum"),
        yield_err_sum=("yield_err", lambda x: np.sqrt((x**2).sum())),
    )
    .rename(columns={"yield_sum": "N", "yield_err_sum": "N_err"})
)
logger.info("Yields combined across LL and DD:")
for state, row in combined_yields.iterrows():
    logger.info(f"  {state}: N={row['N']:.1f} ± {row['N_err']:.1f}")

# ---- Efficiency ratios ----
# Phase 1: average eff ratios from LL and DD (weighted by yield if available;
# simple average otherwise). A more principled approach is to compute the
# luminosity-weighted average efficiency per category, which requires knowing
# ε_LL × N_gen_LL and ε_DD × N_gen_DD. For now, use the yield-weighted average.
df_ratios_ll = pd.read_csv(ratios_ll_file)
df_ratios_dd = pd.read_csv(ratios_dd_file)

# Merge LL and DD efficiency ratios; compute weighted average using total yields as weights
eff_ratios_combined = {}
for state in df_ratios_ll["state"].unique():
    row_ll = df_ratios_ll[df_ratios_ll["state"] == state]
    row_dd = df_ratios_dd[df_ratios_dd["state"] == state]
    if row_ll.empty or row_dd.empty:
        continue
    r_ll = row_ll["ratio_to_ref"].values[0]
    r_dd = row_dd["ratio_to_ref"].values[0]
    e_ll = row_ll["ratio_err"].values[0]
    e_dd = row_dd["ratio_err"].values[0]
    # Yield-weighted average: w_LL * r_LL + w_DD * r_DD
    n_ll = df_ll[df_ll["state"] == state]["yield"].sum() if state in df_ll["state"].values else 1.0
    n_dd = df_dd[df_dd["state"] == state]["yield"].sum() if state in df_dd["state"].values else 1.0
    n_total = n_ll + n_dd
    w_ll = n_ll / n_total if n_total > 0 else 0.5
    w_dd = n_dd / n_total if n_total > 0 else 0.5
    r_avg = w_ll * r_ll + w_dd * r_dd
    e_avg = np.sqrt((w_ll * e_ll) ** 2 + (w_dd * e_dd) ** 2)
    eff_ratios_combined[state] = (r_avg, e_avg)

# Get config labels
plotting_cfg = config.fitting.get("plotting", {})
state_labels = plotting_cfg.get("labels", {})
ref_state = plotting_cfg.get("ref_state", "jpsi")
ref_label = state_labels.get(ref_state, "J/psi")

logger.info(f"Calculating Branching Ratios Relative to {ref_label} (Branch: {branch})")

if ref_state not in combined_yields.index:
    logger.error(f"Reference state {ref_state} ({ref_label}) not found in combined yields!")
    sys.exit(1)

N_ref = combined_yields.loc[ref_state, "N"]
N_ref_err = combined_yields.loc[ref_state, "N_err"]

results = []
placeholder_results = []
for state in combined_yields.index:
    if state == ref_state:
        continue

    N_sig = combined_yields.loc[state, "N"]
    N_sig_err = combined_yields.loc[state, "N_err"]

    eff_ratio, eff_ratio_err = eff_ratios_combined.get(state, (1.0, 0.0))

    ratio = (N_sig / N_ref) / eff_ratio if N_ref > 0 and eff_ratio > 0 else 0

    rel_err_sig = N_sig_err / N_sig if N_sig > 0 else 0
    rel_err_ref = N_ref_err / N_ref if N_ref > 0 else 0
    rel_err_eff = eff_ratio_err / eff_ratio if eff_ratio > 0 else 0

    ratio_stat_err = ratio * np.sqrt(rel_err_sig**2 + rel_err_ref**2)
    ratio_total_err = ratio * np.sqrt(rel_err_sig**2 + rel_err_ref**2 + rel_err_eff**2)

    bf_product = ratio * norm_factor
    bf_product_err = (
        bf_product * np.sqrt((ratio_total_err / ratio) ** 2 + norm_factor_rel_err**2)
        if ratio > 0
        else 0
    )

    row = {
        "state": state,
        "yield_ratio": (N_sig / N_ref) if N_ref > 0 else 0,
        "eff_ratio": eff_ratio,
        "ratio_to_ref": ratio,
        "ratio_stat_err": ratio_stat_err,
        "ratio_total_err": ratio_total_err,
        "bf_product": bf_product,
        "bf_product_err": bf_product_err,
        "syst_err": 0.0,  # populated by export_latex_results.py after compute_systematics
        "mc_pending": state in MC_PENDING_STATES,
    }

    if state in MC_PENDING_STATES:
        placeholder_results.append(row)
        logger.info(f"  {state}: MC pending — excluded from primary results")
    else:
        results.append(row)

df_results = pd.DataFrame(results)
df_all = pd.concat([df_results, pd.DataFrame(placeholder_results)], ignore_index=True)
Path(br_ratios_file).parent.mkdir(parents=True, exist_ok=True)
df_all.to_csv(br_ratios_file, index=False)

# Write final markdown (LL+DD combined)
df_eff_ll = pd.read_csv(efficiencies_ll_file)
df_eff_dd = pd.read_csv(efficiencies_dd_file)

Path(final_results_file).parent.mkdir(parents=True, exist_ok=True)
with open(final_results_file, "w") as f:
    f.write(f"# Final Physics Results ({branch.replace('_', ' ').title()})\n\n")
    f.write("*LL and DD fitted separately; yields summed, efficiency ratios yield-weighted.*\n\n")

    # 1. Yields Table
    f.write("## 1. Fitted Signal Yields (LL + DD combined)\n\n")
    f.write("| State | N (LL) | N (DD) | N (Total) | Stat Err |\n")
    f.write("|-------|--------|--------|-----------|----------|\n")
    for state in combined_yields.index:
        l_tex = state_labels.get(state, state)
        n_ll_s = (
            df_ll[df_ll["state"] == state]["yield"].sum() if state in df_ll["state"].values else 0
        )
        n_dd_s = (
            df_dd[df_dd["state"] == state]["yield"].sum() if state in df_dd["state"].values else 0
        )
        n_tot = combined_yields.loc[state, "N"]
        n_err = combined_yields.loc[state, "N_err"]
        f.write(f"| {l_tex:<7} | {n_ll_s:.1f} | {n_dd_s:.1f} | {n_tot:.1f} | ± {n_err:.1f} |\n")
    f.write("\n")

    # 2. Efficiencies (separate per category)
    f.write("## 2. Efficiencies\n\n")
    f.write("### LL\n\n")
    f.write("| State | ε (LL) | Err |\n|-------|--------|-----|\n")
    for _, row in df_eff_ll.iterrows():
        f.write(
            f"| {state_labels.get(row['state'], row['state']):<7} | {row['efficiency']:.5f} | ± {row['efficiency_err']:.5f} |\n"
        )
    f.write("\n### DD\n\n")
    f.write("| State | ε (DD) | Err |\n|-------|--------|-----|\n")
    for _, row in df_eff_dd.iterrows():
        f.write(
            f"| {state_labels.get(row['state'], row['state']):<7} | {row['efficiency']:.5f} | ± {row['efficiency_err']:.5f} |\n"
        )
    f.write("\n")

    # 3. Branching Fraction Products
    f.write("## 3. Relative Branching Fraction Products\n\n")
    f.write(f"Normalization Reference: {ref_label}\n\n")
    f.write(
        "| State | Yield Ratio | Eff Ratio (ε_X / ε_J/ψ) | Final BR Ratio | Total Ratio Err | BF Product | Total BF Err |\n"
    )
    f.write(
        "|-------|-------------|-------------------------|----------------|-----------------|------------|--------------|\n"
    )
    for _, row in df_results.iterrows():
        l_tex = state_labels.get(row["state"], row["state"])
        f.write(
            f"| {l_tex:<7} | {row['yield_ratio']:.4f} | {row['eff_ratio']:.4f} | "
            f"{row['ratio_to_ref']:.5f} | ± {row['ratio_total_err']:.5f} | "
            f"{row['bf_product']:.2e} | ± {row['bf_product_err']:.2e} |\n"
        )

    # 4. Placeholder states (MC in LHCb production)
    if placeholder_results:
        f.write("\n## 4. Placeholder States (MC in LHCb Production Pipeline)\n\n")
        f.write(
            "> These states are excluded from the primary results above because their MC\n"
            "> simulation is not yet available. Efficiency is set to 1.0 (placeholder).\n"
            "> Results will be updated when MC samples arrive from the LHCb production pipeline.\n\n"
        )
        f.write("| State | N (LL) | N (DD) | N (Total) | Note |\n")
        f.write("|-------|--------|--------|-----------|------|\n")
        for row in placeholder_results:
            st = row["state"]
            l_tex = state_labels.get(st, st)
            n_ll_s = (
                df_ll[df_ll["state"] == st]["yield"].sum() if st in df_ll["state"].values else 0
            )
            n_dd_s = (
                df_dd[df_dd["state"] == st]["yield"].sum() if st in df_dd["state"].values else 0
            )
            n_tot = combined_yields.loc[st, "N"] if st in combined_yields.index else 0
            f.write(f"| {l_tex:<7} | {n_ll_s:.1f} | {n_dd_s:.1f} | {n_tot:.1f} | MC pending |\n")

logger.info(
    f"Branching fraction calculation complete for branch={branch}. "
    f"Saved to {br_ratios_file} and {final_results_file}"
)
