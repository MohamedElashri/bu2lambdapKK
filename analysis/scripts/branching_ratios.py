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

if "snakemake" in globals():
    no_cache = snakemake.params.get("no_cache", False)
    config_dir = snakemake.params.config_dir
    cache_dir = snakemake.params.cache_dir
    output_dir = snakemake.params.output_dir
    branch = snakemake.params.branch
    yields_file = snakemake.input.yields
    efficiencies_file = snakemake.input.efficiencies
    ratios_file = snakemake.input.ratios
    br_ratios_file = snakemake.output.br_ratios
    final_results_file = snakemake.output.final_results
else:
    no_cache = False
    config_dir = "config"
    cache_dir = "analysis_output/box/cache"
    output_dir = "analysis_output/box"
    branch = "high_yield"
    yields_file = Path(output_dir) / branch / "tables" / "fitted_yields.csv"
    efficiencies_file = Path(output_dir) / branch / "tables" / "efficiencies.csv"
    ratios_file = Path(output_dir) / branch / "tables" / "efficiency_ratios.csv"
    br_ratios_file = Path(output_dir) / branch / "tables" / "branching_fraction_ratios.csv"
    final_results_file = Path(output_dir) / branch / "results" / "final_results.md"

config_path = Path(config_dir) / "selection.toml"
physics_path = Path(config_dir) / "physics.toml"

config = StudyConfig(config_file=str(config_path), output_dir=output_dir)

# Load physics constants
with open(physics_path, "rb") as f:
    physics_data = tomli.load(f)
pdg_bf = physics_data.get("pdg_branching_fractions", {})

br_bu_jpsi_k = pdg_bf.get("bu_to_jpsi_k", {}).get("value", 1.0)
br_bu_jpsi_k_err = pdg_bf.get("bu_to_jpsi_k", {}).get("error", 0.0)
br_jpsi_lpkk = pdg_bf.get("jpsi_to_lpkk", {}).get("value", 1.0)
br_jpsi_lpkk_err = pdg_bf.get("jpsi_to_lpkk", {}).get("error", 0.0)

# Normalization factor: Br(B+ -> J/psi K+) * Br(J/psi -> p K- Lambda)
norm_factor = br_bu_jpsi_k * br_jpsi_lpkk
norm_factor_rel_err = (
    np.sqrt((br_bu_jpsi_k_err / br_bu_jpsi_k) ** 2 + (br_jpsi_lpkk_err / br_jpsi_lpkk) ** 2)
    if norm_factor > 0
    else 0
)

# Load yields
df_yields = pd.read_csv(yields_file)
df_eff = pd.read_csv(efficiencies_file)
df_ratios = pd.read_csv(ratios_file)

# We will combine years since branching ratios are a physics quantity.
combined_yields = (
    df_yields.groupby("state")[["yield", "yield_err"]]
    .agg(yield_sum=("yield", "sum"), yield_err_sum=("yield_err", lambda x: np.sqrt((x**2).sum())))
    .rename(columns={"yield_sum": "N", "yield_err_sum": "N_err"})
)

# Get plotting/labels config (Phase 5 refactor)
plotting_cfg = config.fitting.get("plotting", {})
state_labels = plotting_cfg.get("labels", {})
ref_state = plotting_cfg.get("ref_state", "jpsi")
ref_label = state_labels.get(ref_state, "J/psi")

logger.info(f"Calculating Branching Ratios Relative to {ref_label} (Branch: {branch})")

if ref_state not in combined_yields.index:
    logger.error(f"Reference state {ref_state} ({ref_label}) not found in yields!")
    sys.exit(1)

N_ref = combined_yields.loc[ref_state, "N"]
N_ref_err = combined_yields.loc[ref_state, "N_err"]

results = []
for state in combined_yields.index:
    if state == ref_state:
        continue

    N_sig = combined_yields.loc[state, "N"]
    N_sig_err = combined_yields.loc[state, "N_err"]

    # Get efficiency ratio from the pre-calculated ratios table
    eff_ratio_row = df_ratios[df_ratios["state"] == state]
    if not eff_ratio_row.empty:
        eff_ratio = eff_ratio_row["ratio_to_ref"].values[0]
        eff_ratio_err = eff_ratio_row["ratio_err"].values[0]
    else:
        eff_ratio = 1.0
        eff_ratio_err = 0.0

    # R = (N_sig / N_ref) / (eff_sig / eff_ref)
    ratio = (N_sig / N_ref) / eff_ratio if N_ref > 0 and eff_ratio > 0 else 0

    # Error propagation (statistical + efficiency)
    rel_err_sig = N_sig_err / N_sig if N_sig > 0 else 0
    rel_err_ref = N_ref_err / N_ref if N_ref > 0 else 0
    rel_err_eff = eff_ratio_err / eff_ratio if eff_ratio > 0 else 0

    ratio_stat_err = ratio * np.sqrt(rel_err_sig**2 + rel_err_ref**2)
    ratio_total_err = ratio * np.sqrt(rel_err_sig**2 + rel_err_ref**2 + rel_err_eff**2)

    # Branching Fraction (BF) Product: B(B+ -> X K+) * B(X -> p K- Lambda)
    bf_product = ratio * norm_factor
    # Propagate ratio error and normalization error
    bf_product_err = (
        bf_product * np.sqrt((ratio_total_err / ratio) ** 2 + norm_factor_rel_err**2)
        if ratio > 0
        else 0
    )

    results.append(
        {
            "state": state,
            "yield_ratio": (N_sig / N_ref) if N_ref > 0 else 0,
            "eff_ratio": eff_ratio,
            "ratio_to_ref": ratio,
            "ratio_stat_err": ratio_stat_err,
            "ratio_total_err": ratio_total_err,
            "bf_product": bf_product,
            "bf_product_err": bf_product_err,
            "syst_err": 0.0 * ratio,  # Placeholder for future additional systematics
        }
    )

df_results = pd.DataFrame(results)
Path(br_ratios_file).parent.mkdir(parents=True, exist_ok=True)
df_results.to_csv(br_ratios_file, index=False)

# Write final markdown
Path(final_results_file).parent.mkdir(parents=True, exist_ok=True)
with open(final_results_file, "w") as f:
    f.write(f"# Final Physics Results ({branch.replace('_', ' ').title()})\n\n")

    # 1. Yields Table
    f.write("## 1. Fitted Signal Yields\n\n")
    f.write("| State | Yield (N) | Stat Err |\n")
    f.write("|-------|-----------|----------|\n")
    for state in combined_yields.index:
        l_tex = state_labels.get(state, state)
        n_val = combined_yields.loc[state, "N"]
        n_err = combined_yields.loc[state, "N_err"]
        f.write(f"| {l_tex:<7} | {n_val:.1f} | ± {n_err:.1f} |\n")
    f.write("\n")

    # 2. Efficiencies Table
    f.write("## 2. Efficiencies\n\n")
    f.write("| State | Total Efficiency (ε) | Err |\n")
    f.write("|-------|----------------------|-----|\n")
    for _, row in df_eff.iterrows():
        l_tex = state_labels.get(row["state"], row["state"])
        f.write(f"| {l_tex:<7} | {row['efficiency']:.5f} | ± {row['efficiency_err']:.5f} |\n")
    f.write("\n")

    # 3. Branching Fraction Products Table
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
            f"| {l_tex:<7} | {row['yield_ratio']:.4f} | {row['eff_ratio']:.4f} | {row['ratio_to_ref']:.5f} | ± {row['ratio_total_err']:.5f} | {row['bf_product']:.2e} | ± {row['bf_product_err']:.2e} |\n"
        )

logger.info(
    f"Branching fraction calculation complete for branch {branch}. Saved to {br_ratios_file} and {final_results_file}"
)
