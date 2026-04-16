import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure the project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.config_loader import StudyConfig
from modules.generated_paths import pipeline_output_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# States whose MC simulation is not yet available (in LHCb production pipeline).
# Their efficiency is set to 1.0 as a placeholder and they are excluded from
# the main branching-fraction results table until real MC arrives.
MC_PENDING_STATES = {"etac_2s"}

if "snakemake" in globals():
    config_dir = snakemake.params.config_dir
    output_dir = snakemake.params.output_dir
    branch = snakemake.params.branch
    category = snakemake.params.category
    study_eff_file = Path(snakemake.input.study_eff)
    eff_file = snakemake.output.efficiencies
    ratios_file = snakemake.output.ratios
else:
    config_dir = "config"
    output_dir = str(pipeline_output_dir("box", project_root / "generated" / "output"))
    branch = "high_yield"
    category = "LL"
    study_eff_file = (
        project_root
        / "generated"
        / "output"
        / "studies"
        / "efficiency_steps"
        / (f"efficiencies_{branch}.json")
    )
    eff_file = Path(output_dir) / branch / category / "tables" / "efficiencies.csv"
    ratios_file = Path(output_dir) / branch / category / "tables" / "efficiency_ratios.csv"

config = StudyConfig.from_dir(config_dir, output_dir=output_dir)

if not study_eff_file.exists():
    logger.error(
        "Branch-specific efficiency study output not found at %s. "
        "Run the efficiency study for branch=%s first.",
        study_eff_file,
        branch,
    )
    sys.exit(1)

with open(study_eff_file, "r") as f:
    eff_data = json.load(f)

# Get the active plotting states from shared config.
all_states = config.get_plotting_states()
ref_state = config.get_ref_state()

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
    f"(aggregating from {study_eff_file})"
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
        if study_state not in eff_data:
            logger.error(
                "Efficiency study %s does not contain state=%s for branch=%s.",
                study_eff_file,
                study_state,
                branch,
            )
            sys.exit(1)

        # Efficiency is computed per category.
        # Average eff_total over years for this specific category only.
        # The efficiency JSON structure is {state: {category: {year: {efficiencies, errors}}}}.
        effs = []
        errs = []
        state_eff = eff_data[study_state]
        cat_data = state_eff.get(category, state_eff)
        for _, data in cat_data.items():
            if isinstance(data, dict) and "efficiencies" in data:
                effs.append(data["efficiencies"]["eff_total"])
                errs.append(data["errors"]["err_total"])

        if not effs:
            logger.error(
                "Efficiency study %s has no usable efficiencies for state=%s, category=%s.",
                study_eff_file,
                study_state,
                category,
            )
            sys.exit(1)

        total_eff = sum(effs) / len(effs)
        total_err = sum(e**2 for e in errs) ** 0.5 / len(errs)
        note = f"Averaged over years for category={category}, branch={branch}"

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
