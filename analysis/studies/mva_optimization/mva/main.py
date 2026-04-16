"""
Main entry point for MVA Optimization Study

Current workflow notes:
- Accepts --category argument ("LL" or "DD") to train separate models per track category.
- Per-category models are saved as catboost_bdt_LL.cbm / catboost_bdt_DD.cbm so that
  the main pipeline's optimize_selection.py can load the correct one.
- Uses the pipeline cache for data loading (via data_preparation.py) rather than
  reading ROOT files directly.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path to allow correct module resolution
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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

output_dir = Path("../../generated/output/studies/mva_optimization")
output_dir.mkdir(parents=True, exist_ok=True)


def load_tuned_params(category: str):
    """Load Optuna-tuned CatBoost hyperparameters, with per-category fallback."""
    # Try per-category params first, then shared params
    for stem in [f"optuna_catboost_best_params_{category}", "optuna_catboost_best_params"]:
        path = output_dir / "models" / f"{stem}.json"
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    return {}


def run_for_category(category: str, cache_dir: str | Path | None = None):
    """Run the full BDT training and threshold optimization for one Lambda category."""
    report_file = output_dir / f"mva_optimization_report_{category}.txt"

    class DualOutput:
        def __init__(self, filename):
            self.file = open(filename, "w")
            self.stdout = sys.__stdout__

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
        handlers=[
            logging.FileHandler(report_file, mode="a"),
            logging.StreamHandler(sys.__stdout__),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"===== MVA Optimization Study — category={category} =====")

    config = StudyConfig()

    logger.info(f"[{category}] Step 1: Loading and preparing data from pipeline cache...")
    ml_data = load_and_prepare_data(config, category=category, cache_dir=cache_dir)

    logger.info(f"[{category}] Step 2: Training CatBoost BDT...")
    # Pass tuned params so that train_and_evaluate_bdt can pick them up if available
    tuned_params = load_tuned_params(category)
    model = train_and_evaluate_bdt(config, ml_data, category=category)

    logger.info(f"[{category}] Step 3: Optimizing BDT threshold (proxy FOM)...")
    optimal_cut = optimize_bdt_cut(config, model, ml_data, category=category)

    logger.info(f"[{category}] Step 4: Performing final mass fits with cut(s) {optimal_cut}...")
    fit_results = perform_final_fit(config, model, optimal_cut, ml_data)

    logger.info(f"[{category}] Step 5: Generating presentation artifacts...")
    top_2_idx = analyze_feature_importance_and_impact(
        model,
        ml_data,
        tuned_params,
        output_dir / "tables" / f"feature_impact_table_{category}.md",
    )
    plot_top_features_separation(ml_data, top_2_idx, output_dir / "plots", suffix=f"_{category}")
    generate_final_comparison_summary(
        output_dir / "tables" / f"mva_optimization_results_{category}.md",
        output_dir / "tables" / f"final_comparison_summary_{category}.md",
    )

    # Restore stdout
    sys.stdout = sys.__stdout__
    logger.info(f"[{category}] MVA study completed. Report: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="MVA Optimization Study")
    parser.add_argument(
        "--category",
        choices=["LL", "DD", "both"],
        default="both",
        help="Lambda track category to train. 'both' trains LL and DD sequentially.",
    )
    parser.add_argument(
        "--output-dir",
        default="../../generated/output/studies/mva_optimization",
        help="Directory where generated MVA study outputs are written.",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Pipeline cache directory to read preprocessed data from.",
    )
    parser.add_argument(
        "--completion-file",
        default="../../generated/output/studies/mva_optimization/mva_optimization_completed.txt",
        help="Completion marker written after the study finishes.",
    )
    args = parser.parse_args()

    global output_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ANALYSIS_MVA_OUTPUT_DIR"] = str(output_dir)
    if args.cache_dir:
        os.environ["ANALYSIS_PIPELINE_CACHE_DIR"] = args.cache_dir

    categories = ["LL", "DD"] if args.category == "both" else [args.category]
    for cat in categories:
        run_for_category(cat, cache_dir=args.cache_dir or None)

    completion_file = Path(args.completion_file)
    completion_file.parent.mkdir(parents=True, exist_ok=True)
    with open(completion_file, "w") as f:
        f.write(f"Completed for categories: {categories}\n")


if __name__ == "__main__":
    main()
