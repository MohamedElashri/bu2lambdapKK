"""
Data Preparation (Clean Room Implementation)
- Initialization
- Data & MC Loading via standalone loader
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import awkward as ak
from clean_data_loader import load_all_data, load_all_mc
from config_loader import StudyConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_and_prepare_data(config: StudyConfig) -> Tuple[Dict[str, ak.Array], Dict[str, ak.Array]]:
    """
    Loads Real Data and MC, computes derived branches, applies triggers,
    and applies data reduction and baseline analysis cuts.

    Returns:
        (data_prepared, mc_prepared)
        data_prepared: Dictionary mapping year ("2016", "2017", "2018") to an awkard array of data.
        mc_prepared: Dictionary mapping state ("jpsi") to an awkard array of MC.
    """
    logger.info("Initializing Clean Room Data Preparation...")

    base_data_path = Path(config.paths["data_base_path"])
    base_mc_path = Path(config.paths["mc_base_path"])
    years = config.paths["years"]
    track_types = config.paths.get("track_types", ["LL", "DD"])

    logger.info("Loading Real Data...")
    data_prepared = load_all_data(base_data_path, years, track_types)

    mc_states = ["Jpsi", "etac", "chic0", "chic1"]
    logger.info(f"Loading MC Data for {mc_states}...")
    mc_prepared = load_all_mc(base_mc_path, mc_states, years, track_types)

    return data_prepared, mc_prepared
