import json
import logging
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

# Ensure the project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from studies.box_optimization.box_optimizer import SelectionOptimizer as BoxOptimizer

from modules.cache_manager import CacheManager
from modules.config_loader import StudyConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

if "snakemake" in globals():
    no_cache = snakemake.params.no_cache
    config_dir = snakemake.params.config_dir
    cache_dir = snakemake.params.cache_dir
    output_dir = snakemake.params.output_dir
    years = snakemake.params.get("years", ["2016", "2017", "2018"])
    track_types = snakemake.params.get("track_types", ["LL", "DD"])
else:
    no_cache = False
    config_dir = "config"
    cache_dir = "cache"
    output_dir = "analysis_output"
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

# Load step 2 data
data_dict = cache.load("step2_data", dependencies=step2_deps)
mc_dict = cache.load("step2_mc", dependencies=step2_deps)

if data_dict is None or mc_dict is None:
    logger.error("Step 2 data not found in cache. Run 'snakemake load_data' first.")
    sys.exit(1)

opt_type = config.data.get("cut_application", {}).get("optimization_type", "box")
out_path = Path(output_dir) / "tables"
out_path.mkdir(parents=True, exist_ok=True)

if opt_type == "box":
    logger.info("Running Box Optimization (Option A - sequential grouped)")
    optimizer = BoxOptimizer(config_path=str(config_path), output_dir=output_dir)
    results = optimizer.run_optimization(data_dict, mc_dict, option="A")
    # Save optimized cuts
    cuts_file = out_path / "optimized_cuts.json"
    with open(cuts_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Box Optimization complete. Saved cuts to {cuts_file}")

elif opt_type == "mva":
    logger.info("Running MVA Optimization (CatBoost)")

    # Prepare MVA data
    # Background from upper mass sideband
    bkg_dfs = []
    for year, y_data in data_dict.items():
        mask = y_data["Bu_MM_corrected"] > 5330
        bkg_dfs.append(ak.to_dataframe(y_data[mask]))
    bkg_df = pd.concat(bkg_dfs, ignore_index=True) if bkg_dfs else pd.DataFrame()

    # Signal from MC
    sig_dfs = []
    for state, state_data in mc_dict.items():
        if state in ["jpsi", "etac", "chic0", "chic1"]:
            sig_dfs.append(ak.to_dataframe(state_data))
    sig_df = pd.concat(sig_dfs, ignore_index=True) if sig_dfs else pd.DataFrame()

    features = config.data.get("xgboost", {}).get(
        "features",
        [
            "Bu_DTF_chi2",
            "Bu_FDCHI2_OWNPV",
            "Bu_IPCHI2_OWNPV",
            "Bu_PT",
            "p_ProbNNp",
            "h1_ProbNNk",
            "h2_ProbNNk",
        ],
    )

    X_sig = sig_df[features].copy()
    y_sig = np.ones(len(X_sig))

    X_bkg = bkg_df[features].copy()
    y_bkg = np.zeros(len(X_bkg))

    X = pd.concat([X_sig, X_bkg], ignore_index=True)
    y = np.concatenate([y_sig, y_bkg])

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Use standard catboost
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=300, learning_rate=0.1, depth=6, verbose=False, random_state=42
    )
    model.fit(X_train, y_train)

    # Optimize threshold simply (placeholder logic to just return a decent threshold)
    y_pred = model.predict_proba(X_test)[:, 1]
    optimal_threshold = 0.5  # Default

    cuts_file = out_path / "optimized_cuts.json"
    with open(cuts_file, "w") as f:
        json.dump({"mva_threshold": optimal_threshold, "features": features}, f, indent=2)

    # Save the model
    model.save_model(str(Path(output_dir) / "mva_model.cbm"))
    logger.info(f"MVA Optimization complete. Threshold: {optimal_threshold}. Saved to {cuts_file}")

else:
    logger.error(f"Unknown optimization_type: {opt_type}. Must be 'box' or 'mva'.")
    sys.exit(1)
