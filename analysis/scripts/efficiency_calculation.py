import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure the project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.cache_manager import CacheManager
from modules.config_loader import StudyConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# States whose MC simulation is not yet available (in LHCb production pipeline).
# Their efficiency is set to 1.0 as a placeholder and they are excluded from
# the main branching-fraction results table until real MC arrives.
MC_PENDING_STATES = {"etac_2s"}

if "snakemake" in globals():
    no_cache = snakemake.params.no_cache
    config_dir = snakemake.params.config_dir
    cache_dir = snakemake.params.cache_dir
    output_dir = snakemake.params.output_dir
    branch = snakemake.params.branch
    category = snakemake.params.category
    summary_file = snakemake.input.summary
    cuts_file = snakemake.input.cuts
    eff_file = snakemake.output.efficiencies
    ratios_file = snakemake.output.ratios
else:
    no_cache = False
    config_dir = "config"
    cache_dir = "analysis_output/box/cache"
    output_dir = "analysis_output/box"
    branch = "high_yield"
    category = "LL"
    summary_file = Path(output_dir) / branch / category / "tables" / "cut_summary.json"
    cuts_file = Path(output_dir) / branch / category / "models" / "optimized_cuts.json"
    eff_file = Path(output_dir) / branch / category / "tables" / "efficiencies.csv"
    ratios_file = Path(output_dir) / branch / category / "tables" / "efficiency_ratios.csv"

config_path = Path(config_dir) / "selection.toml"
config = StudyConfig(config_file=str(config_path), output_dir=output_dir)

cache = CacheManager(cache_dir=cache_dir)
# Re-compute dependencies
cut_deps = cache.compute_dependencies(
    config_files=list(Path(config_dir).glob("*.toml")),
    code_files=[project_root / "scripts" / "apply_cuts.py"],
)

# Load category-specific cut MC (cache key set by apply_cuts.py)
mc_final = cache.load(f"{branch}_{category}_final_mc", dependencies=cut_deps)

if mc_final is None:
    logger.error(
        f"Cut MC for branch={branch}, category={category} not found in cache. "
        "Run 'snakemake apply_cuts' first."
    )
    sys.exit(1)

import json

eff_json_path = project_root / "studies" / "efficiency_steps" / "output" / "efficiencies.json"
eff_data = {}
if eff_json_path.exists():
    with open(eff_json_path, "r") as f:
        eff_data = json.load(f)
else:
    logger.warning(f"Efficiency file not found at {eff_json_path}. Using 1.0 placeholders.")

# Get states from config (Phase 5 refactor)
plotting_cfg = config.fitting.get("plotting", {})
all_states = plotting_cfg.get("states", ["jpsi", "etac", "chic0", "chic1", "etac_2s"])
ref_state = plotting_cfg.get("ref_state", "jpsi")

# Mapping between config state names and efficiency study state names
state_map = {
    "jpsi": "Jpsi",
    "chic0": "chic0",
    "chic1": "chic1",
    "chic2": "chic2",
    "etac": "etac",
    "etac_2s": "etac_2s",
}

logger.info(
    f"Calculating Efficiencies for branch={branch}, category={category} "
    "(aggregating from efficiency study)"
)

eff_rows = []
for state in all_states:
    study_state = state_map.get(state, state)

    total_eff = 1.0
    total_err = 0.0
    is_placeholder = state in MC_PENDING_STATES

    if is_placeholder:
        note = "MC in LHCb production pipeline — ε = 1.0 placeholder"
    else:
        note = "From efficiency study"

    if not is_placeholder and study_state in eff_data:
        # Phase 1: efficiency is now computed per-category.
        # Average eff_total over years for this specific category only.
        # The efficiency JSON structure is {state: {category: {year: {efficiencies, errors}}}}.
        effs = []
        errs = []
        state_eff = eff_data[study_state]
        # Select the slice for this category; fall back to flat dict if old format
        cat_data = state_eff.get(category, state_eff)
        for year, data in cat_data.items():
            if isinstance(data, dict) and "efficiencies" in data:
                effs.append(data["efficiencies"]["eff_total"])
                errs.append(data["errors"]["err_total"])

        if effs:
            total_eff = sum(effs) / len(effs)
            total_err = sum(e**2 for e in errs) ** 0.5 / len(errs)
            note = f"Averaged over years for category={category}"

    eff_rows.append(
        {
            "state": state,
            "efficiency": total_eff,
            "efficiency_err": total_err,
            "note": note,
            "is_placeholder": is_placeholder,
        }
    )

df_eff = pd.DataFrame(eff_rows)
Path(eff_file).parent.mkdir(parents=True, exist_ok=True)
df_eff.to_csv(eff_file, index=False)

# Ratios relative to ref_state from config
ratios_rows = []
ref_eff = (
    df_eff[df_eff["state"] == ref_state]["efficiency"].values[0]
    if ref_state in df_eff["state"].values
    else 1.0
)
ref_err = (
    df_eff[df_eff["state"] == ref_state]["efficiency_err"].values[0]
    if ref_state in df_eff["state"].values
    else 0.0
)

for _, row in df_eff.iterrows():
    state = row["state"]
    eff = row["efficiency"]
    err = row["efficiency_err"]

    ratio = eff / ref_eff if ref_eff > 0 else 1.0

    # Error propagation
    rel_err_sig = err / eff if eff > 0 else 0
    rel_err_ref = ref_err / ref_eff if ref_eff > 0 else 0
    ratio_err = ratio * (rel_err_sig**2 + rel_err_ref**2) ** 0.5

    ratios_rows.append(
        {"state": state, "ratio_to_ref": ratio, "ratio_err": ratio_err, "note": row["note"]}
    )

df_ratios = pd.DataFrame(ratios_rows)
df_ratios.to_csv(ratios_file, index=False)

logger.info(
    f"Efficiency calculation complete for branch={branch}, category={category}. "
    f"Saved to {eff_file} and {ratios_file}"
)
