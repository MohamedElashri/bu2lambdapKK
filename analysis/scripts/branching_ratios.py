import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    br_ratios_file = snakemake.output.br_ratios
    final_results_file = snakemake.output.final_results
else:
    no_cache = False
    config_dir = "config"
    cache_dir = "cache"
    output_dir = "analysis_output"
    branch = "high_yield"
    opt_type = "box"
    yields_file = Path(output_dir) / opt_type / branch / "tables" / "fitted_yields.csv"
    br_ratios_file = (
        Path(output_dir) / opt_type / branch / "tables" / "branching_fraction_ratios.csv"
    )
    final_results_file = Path(output_dir) / opt_type / branch / "results" / "final_results.md"

config_path = Path(config_dir) / "selection.toml"
config = StudyConfig(config_file=str(config_path), output_dir=output_dir)

# Load yields
df_yields = pd.read_csv(yields_file)

# We will combine years since branching ratios are a physics quantity.
combined_yields = (
    df_yields.groupby("state")[["yield", "yield_err"]]
    .agg(yield_sum=("yield", "sum"), yield_err_sum=("yield_err", lambda x: np.sqrt((x**2).sum())))
    .rename(columns={"yield_sum": "N", "yield_err_sum": "N_err"})
)

# Placeholder for Efficiency (temporarily 1.0 for all states as requested)
efficiencies = {state: 1.0 for state in combined_yields.index}
systematics = {state: 0.0 for state in combined_yields.index}

logger.info(f"Calculating Branching Ratios Relative to J/psi (Branch: {branch})")

ref_state = "jpsi"
if ref_state not in combined_yields.index:
    logger.error(f"Reference state {ref_state} not found in yields!")
    sys.exit(1)

N_ref = combined_yields.loc[ref_state, "N"]
N_ref_err = combined_yields.loc[ref_state, "N_err"]
eff_ref = efficiencies[ref_state]

results = []
for state in combined_yields.index:
    if state == ref_state:
        continue

    N_sig = combined_yields.loc[state, "N"]
    N_sig_err = combined_yields.loc[state, "N_err"]
    eff_sig = efficiencies[state]

    # R = (N_sig / eff_sig) / (N_ref / eff_ref)
    ratio = (N_sig / eff_sig) / (N_ref / eff_ref) if N_ref > 0 else 0

    # Error propagation (statistical only for now)
    rel_err_sig = N_sig_err / N_sig if N_sig > 0 else 0
    rel_err_ref = N_ref_err / N_ref if N_ref > 0 else 0
    stat_err = ratio * np.sqrt(rel_err_sig**2 + rel_err_ref**2)

    results.append(
        {
            "state": state,
            "ratio_to_jpsi": ratio,
            "stat_err": stat_err,
            "syst_err": systematics.get(state, 0.0) * ratio,
        }
    )

df_results = pd.DataFrame(results)
Path(br_ratios_file).parent.mkdir(parents=True, exist_ok=True)
df_results.to_csv(br_ratios_file, index=False)

# Write final markdown
Path(final_results_file).parent.mkdir(parents=True, exist_ok=True)
with open(final_results_file, "w") as f:
    f.write(f"# Final Branching Ratio Results ({branch.replace('_', ' ').title()})\n\n")
    f.write(
        "*Note: Efficiency is currently set to 1.0 (placeholder) and Systematics to 0.0 (placeholder).* \n\n"
    )
    f.write("| State | Ratio to J/psi | Stat Error | Syst Error |\n")
    f.write("|-------|----------------|------------|------------|\n")
    for _, row in df_results.iterrows():
        f.write(
            f"| {row['state']:<5} | {row['ratio_to_jpsi']:.5f} | ± {row['stat_err']:.5f} | ± {row['syst_err']:.5f} |\n"
        )

logger.info(
    f"Branching fraction calculation complete for branch {branch}. Saved to {br_ratios_file} and {final_results_file}"
)
