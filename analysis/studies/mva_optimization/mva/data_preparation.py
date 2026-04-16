"""
Data Preparation for CatBoost BDT Training

Signal:     MC (all charmonium states combined)
Background: Data upper mass sideband (Bu_MM_corrected > 5330 MeV)

Current workflow notes:
- Accepts a `category` parameter ("LL" or "DD") so that separate models can be
  trained per track category, following the LL/DD-separated pipeline.
- Uses the main pipeline's cache (preprocessed_data / preprocessed_mc) rather than
  re-reading ROOT files. This guarantees the training data is identical to what the
  main pipeline will apply the model to.
- PID variables removed from features (sideband proxy is anti-correlated with fit
  FOM for PID variables; see studies/pid_proxy_comparison/).
"""

import logging
import os
import sys
from pathlib import Path

import awkward as ak
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add the main project root to sys.path so we can import the pipeline modules
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.cache_manager import CacheManager
from modules.config_loader import StudyConfig


def load_and_prepare_data(config, category: str = "LL", cache_dir: str | Path | None = None):
    """Load data and MC from the main pipeline cache and prepare training arrays.

    Args:
        config:   StudyConfig loaded from mva_config.toml (for feature list, sideband
                  windows, and train/test split params).
        category: Lambda track category — "LL" or "DD".  A separate BDT is trained
                  for each category.

    Returns:
        dict with keys: X_train, X_test, y_train, y_test, w_train, w_test,
                        features, data_combined, mc_prepared.
    """
    # ---- Locate the main pipeline cache ----
    # project_root resolves to the analysis/ directory (4 parents up from this file).
    # The main pipeline uses generated/cache/pipeline/<opt_method>/ relative to analysis/.
    # We read from whichever cache exists. Prefer "mva" cache, fall back to "box".
    analysis_dir = project_root  # analysis/ dir
    if cache_dir is None:
        env_cache_dir = os.environ.get("ANALYSIS_PIPELINE_CACHE_DIR")
        if env_cache_dir:
            cache_dir = Path(env_cache_dir)

    if cache_dir is None:
        for method in ["mva", "box"]:
            candidate = analysis_dir / "generated" / "cache" / "pipeline" / method
            if candidate.exists():
                cache_dir = candidate
                logger.info(f"Using pipeline cache at {cache_dir}")
                break

    if cache_dir is None:
        raise RuntimeError(
            "No pipeline cache found at analysis/generated/cache/pipeline/[mva|box]/. "
            "Run 'snakemake load_data -j1' from the analysis/ directory first."
        )

    cache_dir = Path(cache_dir)

    # Load the main pipeline config to get the cache dependency hash
    main_config = StudyConfig.from_dir(analysis_dir / "config", output_dir=str(cache_dir.parent))
    cache = CacheManager(cache_dir=str(cache_dir))
    deps = cache.compute_dependencies(
        config_files=main_config.config_paths(),
        code_files=[
            analysis_dir / "modules" / "clean_data_loader.py",
            analysis_dir / "scripts" / "load_data.py",
        ],
        extra_params={"years": ["2016", "2017", "2018"], "track_types": ["LL", "DD"]},
    )

    data_full = cache.load("preprocessed_data", dependencies=deps)
    mc_full = cache.load("preprocessed_mc", dependencies=deps)

    if data_full is None or mc_full is None:
        raise RuntimeError(
            "preprocessed_data / preprocessed_mc not found in cache. "
            "Run 'snakemake load_data -j1' from the analysis/ directory first."
        )

    # ---- Extract the correct category slice ----
    years = list(data_full.keys())
    data_cat = {yr: data_full[yr][category] for yr in years if category in data_full[yr]}
    mc_cat = {st: mc_full[st][category] for st in mc_full if category in mc_full[st]}

    logger.info(
        f"[{category}] Data years: {list(data_cat.keys())}, " f"MC states: {list(mc_cat.keys())}"
    )
    for yr, arr in data_cat.items():
        logger.info(f"  Data {yr}: {len(arr)} events")
    for st, arr in mc_cat.items():
        logger.info(f"  MC {st}: {len(arr)} events")

    # ---- Build training arrays ----
    data_combined = ak.concatenate(list(data_cat.values()))

    features = config.xgboost.get(
        "features", ["Bu_DTF_chi2", "Bu_FDCHI2_OWNPV", "Bu_IPCHI2_OWNPV", "Bu_PT"]
    )
    logger.info(f"[{category}] Training features: {features}")

    b_high_min = config.optimization.get("b_high_sideband", [5330.0, 5410.0])[0]
    b_low_min = config.optimization.get("b_low_sideband", [5150.0, 5230.0])[0]
    b_low_max = config.optimization.get("b_low_sideband", [5150.0, 5230.0])[1]

    bu_mm = data_combined["Bu_MM_corrected"]
    bkg_mask = (bu_mm > b_high_min) | ((bu_mm >= b_low_min) & (bu_mm <= b_low_max))
    bkg_events = data_combined[bkg_mask]
    logger.info(f"[{category}] Background (sideband): {len(bkg_events)} events")

    mc_combined = ak.concatenate(list(mc_cat.values()))
    logger.info(f"[{category}] Signal (MC combined): {len(mc_combined)} events")

    def extract_features(events):
        df_dict = {}
        for feat in features:
            br = events[feat]
            if "var" in str(ak.type(br)):
                br = ak.firsts(br)
            df_dict[feat] = ak.to_numpy(br)
        return pd.DataFrame(df_dict)

    df_bkg = extract_features(bkg_events)
    df_bkg["label"] = 0
    df_bkg["weight"] = 1.0

    df_sig = extract_features(mc_combined)
    df_sig["label"] = 1
    df_sig["weight"] = 1.0

    df_full = pd.concat([df_sig, df_bkg], ignore_index=True).dropna()

    X = df_full[features].values
    y = df_full["label"].values
    w = df_full["weight"].values

    test_size = config.xgboost.get("test_size", 0.3)
    random_state = config.xgboost.get("random_state", 42)

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"[{category}] Train: {X_train.shape}, Test: {X_test.shape}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "w_train": w_train,
        "w_test": w_test,
        "features": features,
        "data_combined": data_combined,
        "data_cat": data_cat,
        "data_prepared": data_cat,  # alias expected by mva_fitter.py
        "mc_prepared": mc_cat,
    }
