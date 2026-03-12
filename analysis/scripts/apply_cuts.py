import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import awkward as ak

from modules.cache_manager import CacheManager
from modules.config_loader import StudyConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

if "snakemake" in globals():
    no_cache = snakemake.params.no_cache
    config_dir = snakemake.params.config_dir
    cache_dir = snakemake.params.cache_dir
    output_dir = snakemake.params.output_dir
    cuts_file = snakemake.input.cuts
    summary_file = snakemake.output.summary
    branch = snakemake.params.branch
    years = snakemake.params.get("years", ["2016", "2017", "2018"])
    track_types = snakemake.params.get("track_types", ["LL", "DD"])
else:
    no_cache = False
    config_dir = "config"
    cache_dir = "analysis_output/box/cache"
    output_dir = "analysis_output/box"
    branch = "high_yield"  # Default for testing
    cuts_file = Path(output_dir) / "tables" / "optimized_cuts.json"
    summary_file = Path(output_dir) / branch / "tables" / "cut_summary.json"
    years = ["2016", "2017", "2018"]
    track_types = ["LL", "DD"]

config_path = Path(config_dir) / "selection.toml"
config = StudyConfig(config_file=str(config_path), output_dir=output_dir)

cache = CacheManager(cache_dir=cache_dir)
preprocessed_deps = cache.compute_dependencies(
    config_files=list(Path(config_dir).glob("*.toml")),
    code_files=[
        project_root / "modules" / "clean_data_loader.py",
        project_root / "scripts" / "load_data.py",
    ],
    extra_params={"years": years, "track_types": track_types},
)
dependencies = cache.compute_dependencies(
    config_files=list(Path(config_dir).glob("*.toml")),
    code_files=[
        project_root / "scripts" / "apply_cuts.py",
    ],
)

# Load preprocessed data
data_dict = cache.load("preprocessed_data", dependencies=preprocessed_deps)
mc_dict = cache.load("preprocessed_mc", dependencies=preprocessed_deps)

if data_dict is None or mc_dict is None:
    logger.error("Step 2 data not found in cache. Run 'snakemake load_data' first.")
    sys.exit(1)

with open(cuts_file, "r") as f:
    optimized_cuts = json.load(f)

opt_type = config.data.get("cut_application", {}).get("optimization_type", "box")

data_final = {}
mc_final = {}

if opt_type == "box":
    logger.info(f"Applying Box Cuts for branch: {branch}")
    # User requested S/sqrt(S+B) for all states in both Box and MVA
    target_fom = "S/sqrt(S+B)"
    target_state = "jpsi" if branch == "high_yield" else "chic1"

    # Collect all cuts for the target state and FoM
    branch_cuts = [
        entry
        for entry in optimized_cuts
        if entry["state"] == target_state and entry["FoM_type"] == target_fom
    ]

    if not branch_cuts:
        logger.error(f"Could not find {target_fom} cuts for state {target_state} in {cuts_file}")
        sys.exit(1)

    logger.info(f"Using cuts from state {target_state} with FoM {target_fom}")

    # Apply to MC
    for state, state_data in mc_dict.items():
        if len(state_data) == 0:
            mc_final[state] = state_data
            continue

        mask = ak.ones_like(state_data[state_data.fields[0]], dtype=bool)
        for cut_entry in branch_cuts:
            var_branch = cut_entry["branch_name"]
            cut_val = cut_entry["optimal_cut"]
            cut_type = cut_entry["cut_type"]

            if var_branch not in state_data.fields:
                continue

            if cut_type == "less":
                mask = mask & (state_data[var_branch] < cut_val)
            else:
                mask = mask & (state_data[var_branch] > cut_val)
        mc_final[state] = state_data[mask]
        logger.info(f"MC {state} passed cuts: {len(mc_final[state])} / {len(state_data)}")

    # Apply to Data
    for year, y_data in data_dict.items():
        if len(y_data) == 0:
            data_final[year] = y_data
            continue

        mask = ak.ones_like(y_data[y_data.fields[0]], dtype=bool)
        for cut_entry in branch_cuts:
            var_branch = cut_entry["branch_name"]
            cut_val = cut_entry["optimal_cut"]
            cut_type = cut_entry["cut_type"]

            if var_branch not in y_data.fields:
                continue

            if cut_type == "less":
                mask = mask & (y_data[var_branch] < cut_val)
            else:
                mask = mask & (y_data[var_branch] > cut_val)
        data_final[year] = y_data[mask]
        logger.info(f"Data {year} passed cuts: {len(data_final[year])} / {len(y_data)}")

elif opt_type == "mva":
    logger.info(f"Applying MVA Cuts (Branch: {branch})")
    if branch == "high_yield":
        threshold = optimized_cuts.get(
            "mva_threshold_high", optimized_cuts.get("mva_threshold", 0.5)
        )
    else:
        threshold = optimized_cuts.get(
            "mva_threshold_low", optimized_cuts.get("mva_threshold", 0.5)
        )

    features = optimized_cuts.get("features", [])

    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(Path(output_dir) / "models" / "mva_model.cbm"))

    # Apply identical MVA threshold to MC and Data
    # 1. Apply to MC
    for state, state_data in mc_dict.items():
        df = ak.to_dataframe(state_data)[features]
        preds = model.predict_proba(df)[:, 1]
        mask = preds > threshold
        mc_final[state] = state_data[mask]
        logger.info(
            f"MC {state} passed MVA > {threshold}: {len(mc_final[state])} / {len(state_data)}"
        )

    # 2. Apply to Data
    for year, y_data in data_dict.items():
        df = ak.to_dataframe(y_data)[features]
        preds = model.predict_proba(df)[:, 1]
        mask = preds > threshold
        data_final[year] = y_data[mask]
        logger.info(
            f"Data {year} passed MVA > {threshold}: {len(data_final[year])} / {len(y_data)}"
        )

# Save final cut results with branch-specific prefix
cache.save(
    f"{branch}_final_data",
    data_final,
    dependencies=dependencies,
    description=f"Data after {branch} cuts",
)
cache.save(
    f"{branch}_final_mc", mc_final, dependencies=dependencies, description=f"MC after {branch} cuts"
)

# Write summary
summary = {
    "optimization_type": opt_type,
    "branch": branch,
    "mc_yields": {k: len(v) for k, v in mc_final.items()},
    "data_yields": {k: len(v) for k, v in data_final.items()},
}

summary_path = Path(summary_file)
summary_path.parent.mkdir(parents=True, exist_ok=True)
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

logger.info(f"Step 4 complete. Summary saved to {summary_file}")
