"""
Final Fit (Step 7)
"""

import logging
from typing import Dict

import awkward as ak
import pandas as pd
from config_loader import StudyConfig
from fom_fitter import MassFitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def apply_cuts(events: ak.Array, cuts_df: pd.DataFrame, state: str = "jpsi") -> ak.Array:
    """Apply optimized cuts from the DataFrame for a given state to the events."""
    state_cuts = cuts_df[cuts_df["state"] == state]
    if state_cuts.empty:
        # fallback to first available state or no cuts
        state_cuts = cuts_df[cuts_df["state"] == cuts_df["state"].iloc[0]]

    mask = (
        ak.ones_like(events["M_LpKm_h2"], dtype=bool)
        if "M_LpKm_h2" in events.fields
        else ak.ones_like(events["Bu_MM_corrected"], dtype=bool)
    )

    for _, row in state_cuts.iterrows():
        branch = row["branch_name"]
        cut_val = row["optimal_cut"]
        cut_type = str(row["cut_type"]).lower()

        if branch not in events.fields:
            logger.warning(f"Branch {branch} not found in events! Skipping cut.")
            continue

        if cut_type == "min" or cut_type == ">":
            mask = mask & (events[branch] > cut_val)
        elif cut_type == "max" or cut_type == "<":
            mask = mask & (events[branch] < cut_val)

    return events[mask]


def perform_final_fit(
    config: StudyConfig, data_prepared: Dict[str, ak.Array], cuts_df: pd.DataFrame
):
    logger.info("Applying optimal cuts and performing final fit...")

    target_state = config.data.get("cut_application", {}).get("data_cut_state", "jpsi")

    data_cut = {}
    for year, events in data_prepared.items():
        data_cut[year] = apply_cuts(events, cuts_df, state=target_state)
        logger.info(f"Data {year} remaining after optimal cuts: {len(data_cut[year])}")

    fitter = MassFitter(config)

    logger.info("Performing final mass fit on all charmonium states...")
    fit_results = fitter.perform_fit(data_cut, fit_combined=True)

    logger.info("Final fit completed!")
    return fit_results
