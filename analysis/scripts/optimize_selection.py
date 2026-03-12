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
    cache_dir = "analysis_output/box/cache"
    output_dir = "analysis_output/box"
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

if "snakemake" in globals():
    # Use snakemake output if available to ensure sync with Snakefile
    out_path = Path(snakemake.output[0]).parent
else:
    out_path = Path(output_dir) / opt_type / "tables"

out_path.mkdir(parents=True, exist_ok=True)

# 1. Scientific Signal Estimation (N_expected) shared by both methods
logger.info(
    "Calculating scientific Signal Estimation (N_expected) from data sideband subtraction..."
)

b_sig_min, b_sig_max = config.mass_windows.get("bu_corrected", [5255.0, 5305.0])
b_low_sb_min, b_low_sb_max = 5150.0, 5230.0
b_high_sb_min, b_high_sb_max = 5330.0, 5410.0
b_sig_width = b_sig_max - b_sig_min
b_low_sb_width = b_low_sb_max - b_low_sb_min
b_high_sb_width = b_high_sb_max - b_high_sb_min

data_combined = ak.concatenate(list(data_dict.values()))
bu_mm = data_combined["Bu_MM_corrected"]
cc_mm = (
    data_combined["M_LpKm_h2"] if "M_LpKm_h2" in data_combined.fields else data_combined["M_LpKm"]
)

n_expected = {}
state_windows = {}
for state in ["jpsi", "etac", "chic0", "chic1"]:
    c, w = config.get_signal_region(state)
    in_cc_sig = (cc_mm > c - w) & (cc_mm < c + w)

    sw = {
        "sr": in_cc_sig & (bu_mm > b_sig_min) & (bu_mm < b_sig_max),
        "low_sb": in_cc_sig & (bu_mm > b_low_sb_min) & (bu_mm < b_low_sb_max),
        "high_sb": in_cc_sig & (bu_mm > b_high_sb_min) & (bu_mm < b_high_sb_max),
    }
    state_windows[state] = sw

    n_sr = float(ak.sum(sw["sr"]))
    n_low = float(ak.sum(sw["low_sb"]))
    n_high = float(ak.sum(sw["high_sb"]))
    b_est = ((n_low / b_low_sb_width + n_high / b_high_sb_width) / 2.0) * b_sig_width
    n_expected[state] = max(n_sr - b_est, 1.0)
    logger.info(f"State {state}: N_sr={n_sr}, B_est={b_est:.1f}, N_exp={n_expected[state]:.1f}")

if opt_type == "box":
    logger.info("Running Scientific Box Optimization (Grid Search)")
    data_comb_dict = {"combined": data_combined}
    # SelectionOptimizer already uses this logic internally,
    # but we can ensure it uses the same n_expected if we want to be exact.
    optimizer = BoxOptimizer(data=data_comb_dict, config=config, mc_data=mc_dict)

    if getattr(config, "optimization", {}).get("method") == "mc_based_sequential":
        logger.info("Using Sequential Optimization Method (Option C)")
        optimized_cuts_df = optimizer.optimize_nd_grid_scan_mc_based_sequential()
    else:
        logger.info("Using Grouped Optimization Method (Option A)")
        optimized_cuts_df = optimizer.optimize_nd_grid_scan_mc_based()

    # Save optimized cuts
    cuts_file = out_path / "optimized_cuts.json"
    results_dict = optimized_cuts_df.to_dict(orient="records")
    with open(cuts_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Box Optimization complete. Saved cuts to {cuts_file}")

elif opt_type == "mva":
    # 2. MVA Model Path
    use_pretrained = config.data.get("cut_application", {}).get("use_pretrained_mva", True)
    tuned_model_path = (
        project_root / "studies" / "mva_optimization" / "output" / "models" / "catboost_bdt.cbm"
    )

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

    from catboost import CatBoostClassifier

    if use_pretrained and tuned_model_path.exists():
        logger.info(f"Loading pre-trained model from {tuned_model_path}")
        model = CatBoostClassifier()
        model.load_model(str(tuned_model_path))
    else:
        logger.info("Training MVA model from scratch...")
        # Prepare training data: Signal (MC) vs Background (Upper Sideband)
        bkg_dfs = []
        for year, y_data in data_dict.items():
            mask = (y_data["Bu_MM_corrected"] > 5330) & (y_data["Bu_MM_corrected"] < 5410)
            bkg_dfs.append(ak.to_dataframe(y_data[mask]))
        bkg_train_df = pd.concat(bkg_dfs, ignore_index=True) if bkg_dfs else pd.DataFrame()

        sig_dfs = []
        for state in ["jpsi", "etac", "chic0", "chic1"]:
            if state in mc_dict:
                sig_dfs.append(ak.to_dataframe(mc_dict[state]))
        sig_train_df = pd.concat(sig_dfs, ignore_index=True) if sig_dfs else pd.DataFrame()

        X = pd.concat([sig_train_df[features], bkg_train_df[features]], ignore_index=True)
        y = np.concatenate([np.ones(len(sig_train_df)), np.zeros(len(bkg_train_df))])

        model = CatBoostClassifier(
            iterations=300, learning_rate=0.1, depth=6, verbose=False, random_state=42
        )
        model.fit(X, y)

    # Save the model to method-isolated output
    model_dir = Path(output_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "mva_model.cbm"))

    # 3. Threshold Optimization via Scientific FOM Scan
    logger.info("Starting scientific BDT threshold optimization...")

    # Pre-evaluate BDT on full data
    X_data = ak.to_dataframe(data_combined[features])[features].values
    data_probs = model.predict_proba(X_data)[:, 1]

    # Evaluate BDT on MC and store totals
    mc_probs = {}
    mc_totals = {}
    for state in ["jpsi", "etac", "chic0", "chic1"]:
        if state in mc_dict:
            X_mc = ak.to_dataframe(mc_dict[state][features])[features].values
            mc_probs[state] = model.predict_proba(X_mc)[:, 1]
            mc_totals[state] = len(mc_dict[state])

    # Grid search for groups
    thresholds = np.linspace(0.1, 0.95, 86)
    sig_groups = {"high": ["jpsi", "etac"], "low": ["chic0", "chic1"]}
    best_cuts = {"high": 0.5, "low": 0.5}

    for group, states in sig_groups.items():
        best_fom = -1
        for thr in thresholds:
            s_total = 0
            b_total = 0
            for st in states:
                # Signal: eps(thr) * N_exp
                if st in mc_probs:
                    eps = np.sum(mc_probs[st] > thr) / mc_totals[st]
                    s_total += eps * n_expected[st]

                # Background: sideband counts after cut
                sw = state_windows[st]
                data_mask = data_probs > thr
                n_low_cut = float(ak.sum(data_mask & sw["low_sb"]))
                n_high_cut = float(ak.sum(data_mask & sw["high_sb"]))
                b_total += (
                    (n_low_cut / b_low_sb_width + n_high_cut / b_high_sb_width) / 2.0
                ) * b_sig_width

            # FOM: S/sqrt(B) for high, S/sqrt(S+B) for low
            if group == "high":
                fom = s_total / np.sqrt(b_total) if b_total > 0 else 0
            else:
                fom = s_total / np.sqrt(s_total + b_total) if (s_total + b_total) > 0 else 0

            if fom > best_fom:
                best_fom = fom
                best_cuts[group] = float(thr)

    cuts_file = out_path / "optimized_cuts.json"
    with open(cuts_file, "w") as f:
        json.dump(
            {
                "mva_threshold_high": best_cuts["high"],
                "mva_threshold_low": best_cuts["low"],
                "features": features,
            },
            f,
            indent=2,
        )

    logger.info("MVA Scientific Optimization complete.")
    logger.info(f"Threshold High (S/sqrt(B)): {best_cuts['high']}")
    logger.info(f"Threshold Low (S/sqrt(S+B)): {best_cuts['low']}")

else:
    logger.error(f"Unknown optimization_type: {opt_type}. Must be 'box' or 'mva'.")
    sys.exit(1)
