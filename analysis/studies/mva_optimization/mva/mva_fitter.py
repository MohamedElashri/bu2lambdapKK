"""
Final Fit Execution for MVA
"""

import logging
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
from config_loader import StudyConfig
from mass_fitter import MassFitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OPTIMIZED_STATES = ["jpsi", "etac", "chic0", "chic1"]


def perform_final_fit(config: StudyConfig, model, cut_results: dict, ml_data: dict):
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SIMULTANEOUS FIT (MVA)")
    logger.info("=" * 80)

    features = ml_data["features"]

    # Reconstruct year-split dictionaries for the fitter
    data_combined = ml_data["data_combined"]
    data_prepared = ml_data["data_prepared"]  # year -> array

    # Pre-compute data bdt scores to save time
    # Actually, we need to apply cuts PER YEAR for the fitter
    # We will compute the BDT score for each year's data
    data_bdt_scores = {}
    for year, events in data_prepared.items():
        df_dict = {}
        for feat in features:
            br = events[feat]
            if "var" in str(ak.type(br)):
                br = ak.firsts(br)
            df_dict[feat] = ak.to_numpy(br)
        X_year = pd.DataFrame(df_dict)[features].values
        if len(X_year) > 0:
            data_bdt_scores[year] = model.predict_proba(X_year)[:, 1]
        else:
            data_bdt_scores[year] = np.array([])

    def apply_bdt_cut(year, thr):
        events = data_prepared[year]
        scores = data_bdt_scores[year]
        if len(scores) == 0:
            return events
        mask = scores > thr
        return events[mask]

    # Decide which cuts to loop over based on optimization strategy
    groups_to_run = ["High_Yield", "Low_Yield"]

    all_fit_rows = []

    for group in groups_to_run:
        for fom_type in ["S/sqrt(B)", "S/sqrt(S+B)"]:

            note = "Grouped cuts (MVA)"
            thr = cut_results[group][fom_type]["best_cut"]

            logger.info(f"\n{'='*60}")
            logger.info(f"Fitting with {group!r} BDT cut > {thr:.4f} for {fom_type}")
            logger.info(f"{'='*60}")

            # Apply this state's BDT cut to every year
            data_cut = {}
            for year, events in data_prepared.items():
                data_cut[year] = apply_bdt_cut(year, thr)
                n_before = len(events)
                n_after = len(data_cut[year])
                logger.info(f"  {year}: {n_before} -> {n_after} events after BDT cut")

            fitter = MassFitter(config)
            try:
                safe_fom = fom_type.replace("/", "_").replace("+", "plus")
                plot_tag = f"{group}_{safe_fom}_bdt_cut"
                fit_results = fitter.perform_fit(data_cut, fit_combined=True, plot_tag=plot_tag)

                # Extract total yields
                for fit_state, state_data in fit_results.items():
                    if state_data and "yield" in state_data:
                        val = state_data["yield"]
                        err = state_data["error"]
                        all_fit_rows.append(
                            {
                                "Optimized for": group,
                                "FoM Type": fom_type,
                                "State Fitted": fit_state,
                                "Yield": val,
                                "Error": err,
                                "BDT_Cut": thr,
                            }
                        )

            except Exception as e:
                logger.error(f"Fit failed for {group} ({fom_type}): {e}")

    # Process results
    if all_fit_rows:
        df_fits = pd.DataFrame(all_fit_rows)
        # Pivot to format like previous study
        pivot_df = df_fits.pivot_table(
            index=["Optimized for", "FoM Type", "BDT_Cut"],
            columns="State Fitted",
            values=["Yield", "Error"],
            aggfunc="first",
        )
        # Flatten columns
        pivot_df.columns = [f"{col[1]} ({col[0]})" for col in pivot_df.columns]
        pivot_df = pivot_df.reset_index()

        output_dir = Path("analysis_output/tables")
        output_dir.mkdir(exist_ok=True, parents=True)

        md_file = output_dir / "mva_final_fit_results.md"
        with open(md_file, "w") as f:
            f.write("# Final Fit Results (MVA Cuts)\n\n")
            f.write(
                "Each row shows the simultaneous fit results when the optimal BDT cut is applied to the data.\n\n"
            )
            f.write(pivot_df.to_markdown(index=False))

        logger.info(f"✓ Saved final fit results block to {md_file}")

    logger.info("\n" + "=" * 80)
    logger.info("MVA ANALYSIS COMPLETE")
    logger.info("=" * 80 + "\n")

    return all_fit_rows
