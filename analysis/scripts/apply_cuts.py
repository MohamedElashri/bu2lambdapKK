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
    cache_dir = "cache"
    output_dir = "analysis_output"
    branch = "high_yield"  # Default for testing
    opt_type = "box"  # Default for testing
    cuts_file = Path(output_dir) / opt_type / "tables" / "optimized_cuts.json"
    summary_file = Path(output_dir) / opt_type / branch / "tables" / "cut_summary.json"
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

    # Identify which state's cuts to use for the whole branch
    # Branch high_yield uses jpsi cuts; branch low_yield uses chic1 cuts.
    target_state = "jpsi" if branch == "high_yield" else "chic1"

    # Find cuts for target_state
    branch_cuts = None
    for entry in optimized_cuts:
        if entry["state"] == target_state:
            branch_cuts = entry["cuts"]
            break

    if branch_cuts is None:
        logger.error(f"Could not find cuts for state {target_state} in {cuts_file}")
        sys.exit(1)

    # Apply identical cuts to MC and Data per state
    # 1. Apply to MC
    for state, state_data in mc_dict.items():
        mask = ak.ones_like(state_data["Bu_MM"], dtype=bool)
        for var, cut_val in branch_cuts.items():
            # Heuristic for cut direction if not explicitly saved:
            if any(x in var for x in ["chi2", "IP", "FD"]):
                mask = mask & (state_data[var] < cut_val)
            else:
                mask = mask & (state_data[var] > cut_val)

        mc_final[state] = state_data[mask]
        logger.info(f"MC {state} passed cuts: {len(mc_final[state])} / {len(state_data)}")

    # 2. Apply to Data (same branch cuts for all data)
    for year, y_data in data_dict.items():
        mask = ak.ones_like(y_data["Bu_MM"], dtype=bool)
        for var, cut_val in branch_cuts.items():
            if any(x in var for x in ["chi2", "IP", "FD"]):
                mask = mask & (y_data[var] < cut_val)
            else:
                mask = mask & (y_data[var] > cut_val)

        data_final[year] = y_data[mask]
        logger.info(f"Data {year} passed cuts: {len(data_final[year])} / {len(y_data)}")

elif opt_type == "mva":
    logger.info(f"Applying MVA Cuts (Branch: {branch})")
    threshold = optimized_cuts.get("mva_threshold", 0.5)
    features = optimized_cuts.get("features", [])

    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(Path(output_dir) / "mva_model.cbm"))

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
