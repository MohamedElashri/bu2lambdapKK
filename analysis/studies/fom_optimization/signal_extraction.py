"""
Signal Extraction and Validation (Step 4 & 5a)
"""

import json
import logging
from pathlib import Path
from typing import Dict

import awkward as ak
import numpy as np
from config_loader import StudyConfig
from fom_fitter import MassFitter
from fom_optimizer import SelectionOptimizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_signal_and_validate(
    config: StudyConfig, data_prepared: Dict[str, ak.Array], mc_prepared: Dict[str, ak.Array]
) -> bool:
    """
    Fit data to extract J/psi signal and error.
    Calculate MC scale factor.
    Validate that S/delta_S ~ S / sqrt(S+B).
    Returns True if valid (within 20%), False otherwise.
    """
    logger.info("Initializing MassFitter for signal extraction...")
    fitter = MassFitter(config)

    logger.info("Fitting Data to extract J/psi signal (S_data, delta_S)...")
    try:
        data_combined = ak.concatenate(list(data_prepared.values()))
        data_dict = {"combined": data_combined}
        fit_results = fitter.perform_fit(data_dict, fit_combined=False)
    except Exception as e:
        logger.error(f"Fit failed during signal extraction: {e}")
        return False

    s_data, delta_s = fit_results["yields"]["combined"]["jpsi"]
    logger.info(f"Signal Extracted from Data: S_data = {s_data:.2f} ± {delta_s:.2f}")

    # Obtain S_mc (count MC events in the B+ mass window)
    s_mc_counted = len(mc_prepared.get("jpsi", []))
    logger.info(f"MC events for J/psi (after pre-selection): S_mc = {s_mc_counted}")

    if s_mc_counted == 0:
        logger.error("No MC events found for J/psi!")
        return False

    scale_factor = s_data / s_mc_counted
    logger.info(f"Signal scale between Data and MC: {scale_factor:.4f}")

    # -----------------------------
    # Validation Check
    # -----------------------------
    logger.info("Performing Validation Check before grid search...")

    # 1. S / delta_S
    metric_1 = s_data / delta_s if delta_s > 0 else 0
    logger.info(f"Metric 1: S_data / delta_S = {metric_1:.2f}")

    # 2. S / sqrt(S + B)
    # Estimate background from Data sidebands using the SelectionOptimizer
    logger.info("Estimating Background from data sidebands...")
    optimizer = SelectionOptimizer(data_prepared, config, mc_prepared)

    # The optimizer requires data flat array for sideband estimation
    b_estimated_total = optimizer.estimate_background_in_signal_region(data_combined, "jpsi")

    s_expected = s_data  # Because S_expected = S_mc * scale_factor = S_data
    metric_2 = (
        s_expected / np.sqrt(s_expected + b_estimated_total)
        if (s_expected + b_estimated_total) > 0
        else 0
    )

    logger.info(f"Background estimated under J/psi peak: B = {b_estimated_total:.2f}")
    logger.info(f"Metric 2: S / sqrt(S + B) = {metric_2:.2f}")

    # 3. Compare within 20%
    if metric_2 == 0:
        logger.error("Metric 2 is zero, validation failed.")
        return False

    ratio = metric_1 / metric_2
    diff_percent = abs(1 - ratio) * 100
    logger.info(f"Validation comparison: Ratio = {ratio:.2f} (Difference: {diff_percent:.1f}%)")

    if diff_percent <= 20.0:
        logger.info("✅ Validation PASSES: Metrics agree within 20%.")
        is_valid = True
    else:
        logger.warning("❌ Validation FAILS: Difference > 20%.")
        is_valid = False

    # Save metrics to JSON
    output_dir = Path("analysis_output/results")
    output_dir.mkdir(exist_ok=True, parents=True)
    metrics = {
        "s_data": float(s_data),
        "delta_s": float(delta_s),
        "s_mc_counted": float(s_mc_counted),
        "scale_factor": float(scale_factor),
        "b_estimated_total": float(b_estimated_total),
        "metric_1_s_over_err": float(metric_1),
        "metric_2_s_over_sqrt_sb": float(metric_2),
        "ratio": float(ratio),
        "diff_percent": float(diff_percent),
        "validation_passed": is_valid,
    }
    with open(output_dir / "signal_extraction_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return is_valid
