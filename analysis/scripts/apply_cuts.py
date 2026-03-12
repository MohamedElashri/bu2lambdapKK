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
    summary_file = snakemake.output[0]
    years = snakemake.params.get("years", ["2016", "2017", "2018"])
    track_types = snakemake.params.get("track_types", ["LL", "DD"])
else:
    no_cache = False
    config_dir = "config"
    cache_dir = "cache"
    output_dir = "analysis_output"
    cuts_file = Path(output_dir) / "tables" / "optimized_cuts.json"
    summary_file = Path(output_dir) / "tables" / "step4_summary.json"
    years = ["2016", "2017", "2018"]
    track_types = ["LL", "DD"]

config_path = Path(config_dir) / "selection.toml"
config = StudyConfig(config_file=str(config_path), output_dir=output_dir)

cache = CacheManager(cache_dir=cache_dir)
step2_deps = cache.compute_dependencies(
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

# Load step 2 data
data_dict = cache.load("step2_data", dependencies=step2_deps)
mc_dict = cache.load("step2_mc", dependencies=step2_deps)

if data_dict is None or mc_dict is None:
    logger.error("Step 2 data not found in cache. Run 'snakemake load_data' first.")
    sys.exit(1)

with open(cuts_file, "r") as f:
    optimized_cuts = json.load(f)

opt_type = config.data.get("cut_application", {}).get("optimization_type", "box")

data_final = {}
mc_final = {}

if opt_type == "box":
    logger.info("Applying Box Cuts")
    # For option A, the cuts are in optimized_cuts under option A results.
    # Group A: jpsi, etac (High) -> S/sqrt(B)
    # Group B: chic0, chic1 (Low) -> S/sqrt(S+B)
    high_yield_states = ["jpsi", "etac"]
    low_yield_states = ["chic0", "chic1"]

    # We will use chic1 cuts for data by default.
    data_cuts = optimized_cuts["chic1"]["S/sqrt(S+B)"]["cuts"]

    # Apply to MC
    for state, state_data in mc_dict.items():
        if state in high_yield_states:
            cuts = optimized_cuts[state]["S/sqrt(B)"]["cuts"]
        elif state in low_yield_states:
            cuts = optimized_cuts[state]["S/sqrt(S+B)"]["cuts"]
        else:  # etac_2s or fallback
            cuts = optimized_cuts["chic1"]["S/sqrt(S+B)"]["cuts"]

        mask = ak.ones_like(state_data["Bu_MM"], dtype=bool)
        for var, cut_val in cuts.items():
            if var == "Bu_DTF_chi2" or var == "Bu_IPCHI2_OWNPV":
                mask = mask & (state_data[var] < cut_val)
            else:
                mask = mask & (state_data[var] > cut_val)

        mc_final[state] = state_data[mask]
        logger.info(f"MC {state} passed cuts: {len(mc_final[state])} / {len(state_data)}")

    # Apply to Data (same cuts for all data, we use the low yield chic1 S/sqrt(S+B) cuts as baseline)
    for year, y_data in data_dict.items():
        mask = ak.ones_like(y_data["Bu_MM"], dtype=bool)
        for var, cut_val in data_cuts.items():
            if var == "Bu_DTF_chi2" or var == "Bu_IPCHI2_OWNPV":
                mask = mask & (y_data[var] < cut_val)
            else:
                mask = mask & (y_data[var] > cut_val)

        data_final[year] = y_data[mask]
        logger.info(f"Data {year} passed cuts: {len(data_final[year])} / {len(y_data)}")

elif opt_type == "mva":
    logger.info("Applying MVA Cuts")
    threshold = optimized_cuts.get("mva_threshold", 0.5)
    features = optimized_cuts.get("features", [])

    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(Path(output_dir) / "mva_model.cbm"))

    # Apply to MC
    for state, state_data in mc_dict.items():
        df = ak.to_dataframe(state_data)[features]
        preds = model.predict_proba(df)[:, 1]
        mask = preds > threshold
        mc_final[state] = state_data[mask]
        logger.info(
            f"MC {state} passed MVA > {threshold}: {len(mc_final[state])} / {len(state_data)}"
        )

    # Apply to Data
    for year, y_data in data_dict.items():
        df = ak.to_dataframe(y_data)[features]
        preds = model.predict_proba(df)[:, 1]
        mask = preds > threshold
        data_final[year] = y_data[mask]
        logger.info(
            f"Data {year} passed MVA > {threshold}: {len(data_final[year])} / {len(y_data)}"
        )

# Save final step 4 results
cache.save(
    "step4_data_final", data_final, dependencies=dependencies, description="Data after final cuts"
)
cache.save("step4_mc_final", mc_final, dependencies=dependencies, description="MC after final cuts")

# Write summary
summary = {
    "optimization_type": opt_type,
    "mc_yields": {k: len(v) for k, v in mc_final.items()},
    "data_yields": {k: len(v) for k, v in data_final.items()},
}

with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

logger.info(f"Step 4 complete. Summary saved to {summary_file}")
