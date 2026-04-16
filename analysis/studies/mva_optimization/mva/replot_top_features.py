import os
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from config_loader import StudyConfig
from data_preparation import load_and_prepare_data

from utils.presentation_utils import plot_top_features_separation


def study_output_dir() -> Path:
    return Path(
        os.environ.get("ANALYSIS_MVA_OUTPUT_DIR", "../../generated/output/studies/mva_optimization")
    )


def replot(category="LL"):
    config = StudyConfig("mva_config.toml")
    ml_data = load_and_prepare_data(config, category=category)

    model_path = study_output_dir() / "models" / f"catboost_bdt_{category}.cbm"
    model = CatBoostClassifier()
    model.load_model(str(model_path))

    importance = model.get_feature_importance()
    sorted_idx = np.argsort(importance)[::-1]
    top_2_idx = sorted_idx[:2]

    plot_dir = study_output_dir() / "plots"
    plot_top_features_separation(ml_data, top_2_idx, plot_dir, suffix=f"_{category}")
    print(f"Re-plotted top 2 features for {category} in {plot_dir}")


if __name__ == "__main__":
    replot("LL")
    replot("DD")
