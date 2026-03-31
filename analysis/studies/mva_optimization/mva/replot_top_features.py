from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from config_loader import StudyConfig
from data_preparation import load_and_prepare_data

from utils.presentation_utils import plot_top_features_separation


def replot(category="LL"):
    config = StudyConfig("mva_config.toml")
    ml_data = load_and_prepare_data(config, category=category)

    model_path = f"../output/models/catboost_bdt_{category}.cbm"
    model = CatBoostClassifier()
    model.load_model(model_path)

    importance = model.get_feature_importance()
    sorted_idx = np.argsort(importance)[::-1]
    top_2_idx = sorted_idx[:2]

    plot_dir = Path("../output/plots")
    plot_top_features_separation(ml_data, top_2_idx, plot_dir, suffix=f"_{category}")
    print(f"Re-plotted top 2 features for {category} in {plot_dir}")


if __name__ == "__main__":
    replot("LL")
    replot("DD")
