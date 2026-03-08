"""
Main entry point for MVA Optimization Study
"""

import sys
from pathlib import Path

# Add project root to sys.path to allow correct module resolution
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging

from config_loader import StudyConfig
from cut_optimizer import optimize_bdt_cut
from data_preparation import load_and_prepare_data
from mva_fitter import perform_final_fit
from mva_trainer import train_and_evaluate_bdt

# output_dir inside the study itself
output_dir = Path("analysis_output")
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

    with open("../mva_optimization_completed.txt", "w") as f:
        f.write("Completed\n")

    logger.info("MVA Study completed successfully.")


if __name__ == "__main__":
    main()
