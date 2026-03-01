"""
Main entry point for FoM Optimization Study
"""

import sys
from pathlib import Path

# Add project root to sys.path to allow correct module resolution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging

from config_loader import StudyConfig

output_dir = project_root / "analysis_output"
output_dir.mkdir(parents=True, exist_ok=True)
report_file = output_dir / "fom_optimization_report.txt"


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

from data_preparation import load_and_prepare_data
from final_fit import perform_final_fit
from optimization import run_grid_search
from signal_extraction import extract_signal_and_validate


def main():
    logger.info("Starting FoM Optimization Study...")

    config = StudyConfig()

    # ---------------------------------------------------------
    # Steps 1, 2, 3: Data & MC Loading + Pre-selections
    # ---------------------------------------------------------
    data_prepared, mc_prepared = load_and_prepare_data(config)

    # ---------------------------------------------------------
    # Steps 4 & 5a: Signal Extraction & Validation Check
    # ---------------------------------------------------------
    is_valid = extract_signal_and_validate(config, data_prepared, mc_prepared)
    if not is_valid:
        logger.error("Validation failed! Halting before expensive grid search.")
        sys.exit(1)

    # ---------------------------------------------------------
    # Steps 5b & 6: N-dimensional Grid Search
    # ---------------------------------------------------------
    optimal_cuts_df = run_grid_search(config, data_prepared, mc_prepared)
    logger.info(f"Optimal cuts found:\n{optimal_cuts_df}")

    # ---------------------------------------------------------
    # Step 7: Final Simultaneous Fit
    # ---------------------------------------------------------
    fit_results = perform_final_fit(config, data_prepared, optimal_cuts_df)

    # Touch Snakemake completion file
    with open("fom_optimization_completed.txt", "w") as f:
        f.write("Completed\n")

    logger.info("Study completed successfully.")


if __name__ == "__main__":
    main()
