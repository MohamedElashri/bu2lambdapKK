"""
Final Fit (Step 7)

For each state:
 1. Apply that state's optimal cuts to data
 2. Perform the simultaneous fit across all charmonium states in M(LpKm_h2)
 3. Save per-state fit results as a markdown table
"""

import json
import logging
from pathlib import Path
from typing import Dict

import awkward as ak
import pandas as pd
from config_loader import StudyConfig
from fom_fitter import MassFitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# States we fit for in each pass (etac_2s is included as it is present in the mass spectrum)
FIT_STATES = ["jpsi", "etac", "chic0", "chic1", "etac_2s"]
# States with their own optimal cuts (optimized via MC)
OPTIMIZED_STATES = ["jpsi", "etac", "chic0", "chic1"]
# etac_2s inherits chic1 cuts (no MC yet)
ETAC_2S_PROXY = "chic1"


def apply_cuts_for_state(events: ak.Array, cuts_df: pd.DataFrame, state: str) -> ak.Array:
    """Apply optimized cuts for a given state. Falls back to chic1 for etac_2s."""
    lookup_state = state if state != "etac_2s" else ETAC_2S_PROXY
    state_cuts = cuts_df[cuts_df["state"] == lookup_state]
    if state_cuts.empty:
        logger.warning(f"No cuts found for state {lookup_state}, returning uncut events.")
        return events

    ref_branch = "M_LpKm_h2" if "M_LpKm_h2" in events.fields else "Bu_MM_corrected"
    mask = ak.ones_like(events[ref_branch], dtype=bool)

    for _, row in state_cuts.iterrows():
        branch = row["branch_name"]
        cut_val = row["optimal_cut"]
        cut_type = str(row["cut_type"]).lower()

        if branch not in events.fields:
            logger.warning(f"Branch {branch} not in events — skipping.")
            continue

        if cut_type in ("greater", ">", "min"):
            mask = mask & (events[branch] > cut_val)
        elif cut_type in ("less", "<", "max"):
            mask = mask & (events[branch] < cut_val)

    return events[mask]


def perform_final_fit(
    config: StudyConfig, data_prepared: Dict[str, ak.Array], cuts_df: pd.DataFrame
):
    """
    For each optimized (or proxy) state:
      - Apply that state's optimal cuts to data
      - Run the simultaneous fit for all charmonium states
      - Record the extracted yields

    All results are aggregated into a per-state markdown table.
    """
    logger.info("Starting per-state final fits...")

    output_dir = Path("analysis_output/results")
    output_dir.mkdir(exist_ok=True, parents=True)
    tables_dir = Path("analysis_output/tables")
    tables_dir.mkdir(exist_ok=True, parents=True)

    # -----------------------------------------------------------------
    # Baseline fit: no optimization cuts applied (comparison reference)
    # -----------------------------------------------------------------
    logger.info("Running baseline fit (no optimization cuts)...")
    baseline_fitter = MassFitter(config)
    try:
        baseline_fitter.perform_fit(data_prepared, fit_combined=True, plot_tag="baseline_no_cuts")
        logger.info("✓ Baseline fit done — plots saved to plots/fits/baseline_no_cuts/")
    except Exception as e:
        logger.warning(f"Baseline fit failed: {e}")

    # We will collect one row per optimized state
    # (etac_2s uses chic1 cuts, so its "optimization target" is chic1)
    states_to_run = OPTIMIZED_STATES + ["etac_2s"]

    all_fit_rows = []
    all_fit_results_json = {}

    for target_state in states_to_run:
        source_cuts = target_state if target_state != "etac_2s" else ETAC_2S_PROXY
        note = "" if target_state != "etac_2s" else f"cuts from {ETAC_2S_PROXY} (no MC)"
        logger.info(f"\n{'='*60}")
        logger.info(
            f"Fitting with {target_state!r} optimal cuts"
            + (f" (using {ETAC_2S_PROXY} cuts)" if note else "")
        )
        logger.info(f"{'='*60}")

        # Apply this state's cuts to every year
        data_cut = {}
        for year, events in data_prepared.items():
            data_cut[year] = apply_cuts_for_state(events, cuts_df, target_state)
            n_before = len(events)
            n_after = len(data_cut[year])
            logger.info(f"  {year}: {n_before} → {n_after} events after {source_cuts} cuts")

        fitter = MassFitter(config)
        try:
            # plot_tag puts all plots for this state's cuts in a dedicated subdir
            plot_tag = f"{target_state}_cuts"
            fit_results = fitter.perform_fit(data_cut, fit_combined=True, plot_tag=plot_tag)
        except Exception as e:
            logger.error(f"Fit failed for state {target_state}: {e}")
            continue

        # Extract yields for each charmonium state from the combined fit
        combined_yields = fit_results.get("yields", {}).get("combined", {})
        row = {"target_state": target_state, "note": note}
        for fit_state in FIT_STATES:
            if fit_state in combined_yields:
                s_val, s_err = combined_yields[fit_state]
                row[f"{fit_state}_yield"] = f"{s_val:.1f} ± {s_err:.1f}"
            else:
                row[f"{fit_state}_yield"] = "—"
        all_fit_rows.append(row)
        all_fit_results_json[target_state] = {
            fs: {"yield": combined_yields[fs][0], "error": combined_yields[fs][1]}
            for fs in FIT_STATES
            if fs in combined_yields
        }

    if not all_fit_rows:
        logger.error("No fit results collected — something went wrong.")
        return {}

    fit_df = pd.DataFrame(all_fit_rows)

    # ── Save markdown table ──────────────────────────────────────────────────
    md_path = tables_dir / "final_fit_results.md"
    with open(md_path, "w") as mdf:
        mdf.write("# Final Fit Results (per-state optimal cuts)\n\n")
        mdf.write(
            "Each row shows the simultaneous fit results when the **row state's** optimal "
            "cuts are applied to the data.\n\n"
        )
        mdf.write("| Optimized for | Note |")
        for fs in FIT_STATES:
            mdf.write(f" {fs} yield |")
        mdf.write("\n|---|---|")
        for _ in FIT_STATES:
            mdf.write("---|")
        mdf.write("\n")
        for _, r in fit_df.iterrows():
            mdf.write(f"| {r['target_state']} | {r['note']} |")
            for fs in FIT_STATES:
                mdf.write(f" {r.get(f'{fs}_yield', '—')} |")
            mdf.write("\n")
        mdf.write("\n> **Notes:**\n")
        mdf.write("> - `eta_c(2S)` row uses `chi_c1` optimal cuts (no MC available yet).\n")
        mdf.write("> - All fits are simultaneous across the full M(Λ̄pK⁻) spectrum.\n")
    logger.info(f"✓ Saved final fit table: {md_path}")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    json_path = output_dir / "final_fit_results.json"
    with open(json_path, "w") as jf:
        json.dump(all_fit_results_json, jf, indent=4)
    logger.info(f"✓ Saved final fit JSON:  {json_path}")

    logger.info("Final fit completed!")
    return all_fit_results_json
