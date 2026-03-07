"""
Data Preparation for XGBoost MVA
"""

import logging
from pathlib import Path

import awkward as ak
import pandas as pd
from clean_data_loader import load_all_data, load_all_mc
from config_loader import StudyConfig
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_and_prepare_data(config: StudyConfig):
    """
    Loads Real Data and MC, and prepares them for XGBoost.
    Signal = MC (truth matched)
    Background = Data High Sideband (Bu_MM_corrected > 5330)
    """
    base_data_path = Path(config.paths["data_base_path"])
    base_mc_path = Path(config.paths["mc_base_path"])
    years = config.paths["years"]
    track_types = config.paths.get("track_types", ["LL", "DD"])

    logger.info("Loading Real Data...")
    data_prepared = load_all_data(base_data_path, years, track_types)

    mc_states = config.paths.get("mc_states", ["jpsi", "etac", "chic0", "chic1"])
    # Format state names to match fom_optimization convention (Jpsi is capitalized there)
    formatted_states = [s.capitalize() if s.lower() == "jpsi" else s for s in mc_states]
    logger.info(f"Loading MC Data for {formatted_states}...")
    mc_prepared = load_all_mc(base_mc_path, formatted_states, years, track_types)
    # We rename 'Jpsi' key back to lower for consistency if present
    if "Jpsi" in mc_prepared:
        mc_prepared["jpsi"] = mc_prepared.pop("Jpsi")

    # Combine data across years
    data_combined = ak.concatenate([data_prepared[y] for y in years])

    # Extract background (High mass and Low mass sidebands)
    bu_mass_branch = "Bu_MM_corrected" if "Bu_MM_corrected" in data_combined.fields else "Bu_M"
    b_high_min = getattr(config, "optimization", {}).get("b_high_sideband", [5330.0, 5410.0])[0]
    b_low_min = getattr(config, "optimization", {}).get("b_low_sideband", [5150.0, 5230.0])[0]
    b_low_max = getattr(config, "optimization", {}).get("b_low_sideband", [5150.0, 5230.0])[1]

    bkg_mask = (data_combined[bu_mass_branch] > b_high_min) | (
        (data_combined[bu_mass_branch] >= b_low_min) & (data_combined[bu_mass_branch] <= b_low_max)
    )
    bkg_events = data_combined[bkg_mask]
    logger.info(
        f"Selected {len(bkg_events)} background events from upper (> {b_high_min} MeV) and lower sidebands."
    )

    # Extract signal (combine all MC states)
    mc_combined = ak.concatenate(list(mc_prepared.values()))
    logger.info(f"Selected {len(mc_combined)} signal events from combined MC.")

    features = config.xgboost.get(
        "features", ["Bu_DTF_chi2", "Bu_FDCHI2_OWNPV", "Bu_IPCHI2_OWNPV", "Bu_PT", "PID_product"]
    )

    def extract_features(events):
        df_dict = {}
        for feat in features:
            br = events[feat]
            # Handle jagged arrays gracefully (take first element)
            if "var" in str(ak.type(br)):
                br = ak.firsts(br)
            df_dict[feat] = ak.to_numpy(br)
        return pd.DataFrame(df_dict)

    df_bkg = extract_features(bkg_events)
    df_bkg["label"] = 0
    df_bkg["weight"] = 1.0  # Equal weight for background

    df_sig = extract_features(mc_combined)
    df_sig["label"] = 1
    # We will use equal weight of 1, and let XGBoost scale_pos_weight handle class imbalance later
    df_sig["weight"] = 1.0

    df_full = pd.concat([df_sig, df_bkg], ignore_index=True)

    # Drop NaNs just in case
    df_full = df_full.dropna()

    X = df_full[features].values
    y = df_full["label"].values
    w = df_full["weight"].values

    test_size = config.xgboost.get("test_size", 0.3)
    random_state = config.xgboost.get("random_state", 42)

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train shapes - X: {X_train.shape}, y: {y_train.shape}, w: {w_train.shape}")
    logger.info(f"Test  shapes - X: {X_test.shape}, y: {y_test.shape}, w: {w_test.shape}")

    # Pack needed info into dict for upstream
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "w_train": w_train,
        "w_test": w_test,
        "features": features,
        "data_combined": data_combined,
        "data_prepared": data_prepared,  # dict per year mapping
        "mc_prepared": mc_prepared,
    }
