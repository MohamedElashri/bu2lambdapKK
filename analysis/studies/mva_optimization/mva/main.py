"""
Main entry point for MVA Optimization Study
"""

import sys
from pathlib import Path

# Add project root to sys.path to allow correct module resolution
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import logging

from config_loader import StudyConfig
from cut_optimizer import optimize_bdt_cut
from data_preparation import load_and_prepare_data
from mva_fitter import perform_final_fit
from mva_trainer import train_and_evaluate_bdt

from utils.presentation_utils import (
    analyze_feature_importance_and_impact,
    generate_final_comparison_summary,
    plot_top_features_separation,
)

# output_dir inside the study itself
output_dir = Path("../output")
output_dir.mkdir(parents=True, exist_ok=True)
report_file = output_dir / "mva_optimization_report.txt"


class DualOutput:
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __del__(self):
        self.file.close()


sys.stdout = DualOutput(report_file)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.FileHandler(report_file, mode="a"), logging.StreamHandler(sys.stdout.stdout)],
)
logger = logging.getLogger(__name__)


def load_tuned_params():
    best_params_path = Path("../output/models/optuna_catboost_best_params.json")
    catboost_params = {
        "random_seed": 42,
        "learning_rate": 0.05,
        "iterations": 350,
        "depth": 6,
        "verbose": False,
    }
    if best_params_path.exists():
        with open(best_params_path, "r") as f:
            tuned_params = json.load(f)
        for k, v in tuned_params.items():
            catboost_params[k] = v
    return catboost_params


def main():
    logger.info("Starting MVA Optimization Study...")

    config = StudyConfig()

    logger.info("Step 1: Loading and preparing data for ML...")
    ml_data = load_and_prepare_data(config)

    logger.info("Step 2: Training and evaluating XGBoost BDT...")
    model = train_and_evaluate_bdt(config, ml_data)

    logger.info("Step 3: Optimizing BDT cut threshold...")
    optimal_cut = optimize_bdt_cut(config, model, ml_data)

    logger.info(f"Step 4: Performing final mass fits with cut(s) {optimal_cut}...")
    fit_results = perform_final_fit(config, model, optimal_cut, ml_data)

    logger.info("Step 5: Generating Presentation Artifacts (Phase 2, 3, 4)...")

    # Phase 2: Feature impact
    catboost_params = load_tuned_params()
    top_2_idx = analyze_feature_importance_and_impact(
        model, ml_data, catboost_params, Path("../output/tables/feature_impact_table.md")
    )

    # Phase 3: Separation Visualization
    plot_top_features_separation(ml_data, top_2_idx, Path("../output/plots"))

    # Phase 4: Final Comparison Summary
    generate_final_comparison_summary(
        Path("../output/tables/mva_optimization_results.md"),
        Path("../output/tables/final_comparison_summary.md"),
    )

    with open("../mva_optimization_completed.txt", "w") as f:
        f.write("Completed\n")

    logger.info("MVA Study completed successfully.")


if __name__ == "__main__":
    main()
