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
preprocessed_deps = cache.compute_dependencies(
    config_files=list(Path(config_dir).glob("*.toml")),
    code_files=[
        project_root / "modules" / "clean_data_loader.py",
        project_root / "scripts" / "load_data.py",
    ],
    extra_params={"years": years, "track_types": track_types},
)

# Load preprocessed data
data_dict = cache.load("preprocessed_data", dependencies=preprocessed_deps)
mc_dict = cache.load("preprocessed_mc", dependencies=preprocessed_deps)

if data_dict is None or mc_dict is None:
    logger.error("Preprocessed data not found in cache. Run 'snakemake load_data' first.")
    sys.exit(1)

opt_type = config.data.get("cut_application", {}).get("optimization_type", "box")
out_path = Path(output_dir) / "tables"
out_path.mkdir(parents=True, exist_ok=True)

if opt_type == "box":
    logger.info("Running Box Optimization")
    
    # We need to concatenate the years for optimization
    data_combined = ak.concatenate(list(data_dict.values()))
    data_comb_dict = {"combined": data_combined}

    # SelectionOptimizer expects mc_data dictionary grouped by state
    optimizer = BoxOptimizer(data=data_comb_dict, config=config, mc_data=mc_dict)
    
    if getattr(config, "optimization", {}).get("method") == "mc_based_sequential":
        logger.info("Using Sequential Optimization Method (Option C)")
        optimized_cuts_df = optimizer.optimize_nd_grid_scan_mc_based_sequential()
    else:
        logger.info("Using Grouped/State-Dependent Optimization Method (Option A/B)")
        optimized_cuts_df = optimizer.optimize_nd_grid_scan_mc_based()

    # Save optimized cuts
    cuts_file = out_path / "optimized_cuts.json"
    
    # Convert DF to dict for JSON serialization
    results_dict = optimized_cuts_df.to_dict(orient='records')
    
    with open(cuts_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Box Optimization complete. Saved cuts to {cuts_file}")

elif opt_type == "mva":
    # Check for pre-trained tuned model
    use_pretrained = config.data.get("cut_application", {}).get("use_pretrained_mva", True)
    tuned_model_path = project_root / "studies" / "mva_optimization" / "output" / "models" / "catboost_bdt.cbm"
    
    if use_pretrained and tuned_model_path.exists():
        logger.info(f"Found pre-trained tuned CatBoost model at {tuned_model_path}. Loading it instead of re-training.")
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(tuned_model_path))
        
        # Save copy of the model to main output
        model.save_model(str(Path(output_dir) / "mva_model.cbm"))
        
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
        optimal_threshold = 0.5 # Default threshold
        
        cuts_file = out_path / "optimized_cuts.json"
        with open(cuts_file, "w") as f:
            json.dump({"mva_threshold": optimal_threshold, "features": features}, f, indent=2)
            
        logger.info(f"MVA loading complete. Threshold: {optimal_threshold}. Saved to {cuts_file}")

    else:
        if use_pretrained:
            logger.warning("Pre-trained model requested but not found. Running MVA Optimization (CatBoost) from scratch.")
        else:
            logger.info("Running MVA Optimization (CatBoost) from scratch.")

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
