"""
Optimization (Steps 5b & 6)
"""

import logging

import awkward as ak
import pandas as pd
from box_optimizer import SelectionOptimizer
from config_loader import StudyConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_grid_search(
    config: StudyConfig, data_prepared: dict[str, ak.Array], mc_prepared: dict[str, ak.Array]
) -> pd.DataFrame:
    """
    Run the multi-dimensional grid scan to find the best cuts.
    Returns the optimal cut configuration as a DataFrame.
    """
    logger.info("Initializing SelectionOptimizer for grid search...")

    # We need to concatenate the years for optimization
    data_combined = ak.concatenate(list(data_prepared.values()))
    data_dict = {"combined": data_combined}

    # SelectionOptimizer expects mc_data dictionary grouped by state for Option B
    optimizer = SelectionOptimizer(data_dict, config, mc_prepared)

    logger.info("Running N-dimensional grid scan...")
    # Both Option A and Option B use the same method now
    # The optimizer internally checks config.optimization.state_dependent
    # to decide between Grouped (Option A) and Per-State (Option B)
    if getattr(config, "optimization", {}).get("method") == "mc_based_sequential":
        logger.info("Using Sequential Optimization Method (Option C)")
        optimized_cuts_df = optimizer.optimize_nd_grid_scan_mc_based_sequential()
    else:
        optimized_cuts_df = optimizer.optimize_nd_grid_scan_mc_based()

    logger.info("Grid scan optimization completed.")
    return optimized_cuts_df
