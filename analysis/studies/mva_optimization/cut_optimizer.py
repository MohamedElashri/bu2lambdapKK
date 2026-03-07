"""
Optimizer for BDT Threshold
"""

import logging
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config_loader import StudyConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def optimize_bdt_cut(config: StudyConfig, model, ml_data: dict):
    logger.info("Starting 1D scan over BDT Output Probability...")

    data_combined = ml_data["data_combined"]
    mc_prepared = ml_data["mc_prepared"]
    features = ml_data["features"]

    # 1) Evaluate BDT on data
    df_data_dict = {}
    for feat in features:
        br = data_combined[feat]
        if "var" in str(ak.type(br)):
            br = ak.firsts(br)
        df_data_dict[feat] = ak.to_numpy(br)
    X_data = pd.DataFrame(df_data_dict)[features].values
    data_bdt_score = model.predict_proba(X_data)[:, 1]

    # 2) Evaluate BDT on MC per state
    mc_bdt_scores = {}
    mc_totals = {}
    for state, evts in mc_prepared.items():
        df_mc_dict = {}
        for feat in features:
            br = evts[feat]
            if "var" in str(ak.type(br)):
                br = ak.firsts(br)
            df_mc_dict[feat] = ak.to_numpy(br)
        X_mc = pd.DataFrame(df_mc_dict)[features].values
        mc_bdt_scores[state] = model.predict_proba(X_mc)[:, 1]
        mc_totals[state] = len(evts)

    # 3) Reconstruct FOM logical boundaries
    bu_mass_branch = "Bu_MM_corrected" if "Bu_MM_corrected" in data_combined.fields else "Bu_M"
    mass_branch = "M_LpKm_h2" if "M_LpKm_h2" in data_combined.fields else "M_LpKm"
    bu_mass = data_combined[bu_mass_branch]
    cc_mass = data_combined[mass_branch]

    opt_config = getattr(config, "optimization", {})
    b_sig_min = opt_config.get("b_signal_region", [5255.0, 5305.0])[0]
    b_sig_max = opt_config.get("b_signal_region", [5255.0, 5305.0])[1]
    b_low_sb_min = opt_config.get("b_low_sideband", [5150.0, 5230.0])[0]
    b_low_sb_max = opt_config.get("b_low_sideband", [5150.0, 5230.0])[1]
    b_high_sb_min = opt_config.get("b_high_sideband", [5330.0, 5410.0])[0]
    b_high_sb_max = opt_config.get("b_high_sideband", [5330.0, 5410.0])[1]

    in_b_sig = (bu_mass > b_sig_min) & (bu_mass < b_sig_max)
    in_b_low_sb = (bu_mass > b_low_sb_min) & (bu_mass < b_low_sb_max)
    in_b_high_sb = (bu_mass > b_high_sb_min) & (bu_mass < b_high_sb_max)

    b_sig_width = b_sig_max - b_sig_min
    b_low_sb_width = b_low_sb_max - b_low_sb_min
    b_high_sb_width = b_high_sb_max - b_high_sb_min

    signal_regions = getattr(config, "data", {}).get(
        "signal_regions", getattr(config, "signal_regions", {})
    )
    ALL_STATES = ["jpsi", "etac", "chic0", "chic1"]
    HIGH_YIELD = ["jpsi", "etac"]
    LOW_YIELD = ["chic0", "chic1"]

    state_windows = {}
    n_expected = {}
    for state in ALL_STATES:
        sr = signal_regions.get(state, signal_regions.get(state.lower(), {}))
        c, w = sr.get("center", 0), sr.get("window", 0)
        in_cc_sig = (cc_mass > c - w) & (cc_mass < c + w)
        sw = {
            "in_sig": in_cc_sig & in_b_sig,
            "in_low_sb": in_cc_sig & in_b_low_sb,
            "in_high_sb": in_cc_sig & in_b_high_sb,
        }
        state_windows[state] = sw

        n_sr = float(ak.sum(sw["in_sig"]))
        n_low = float(ak.sum(sw["in_low_sb"]))
        n_high = float(ak.sum(sw["in_high_sb"]))
        d_low = n_low / b_low_sb_width if b_low_sb_width > 0 else 0
        d_high = n_high / b_high_sb_width if b_high_sb_width > 0 else 0
        b_est = ((d_low + d_high) / 2.0) * b_sig_width
        n_expected[state] = max(n_sr - b_est, 1.0)

    # 4) Scan threshold
    thresholds = np.linspace(0.01, 0.99, 99)
    foms_to_compute = ["S/sqrt(B)", "S/sqrt(S+B)"]
    best_results = {
        state: {
            f: {"best_fom": -np.inf, "best_cut": 0, "eps": 0, "s": 0, "b": 0}
            for f in foms_to_compute
        }
        for state in ALL_STATES
    }

    def fom1(s, b):
        return s / np.sqrt(b) if b > 0 else (s / np.sqrt(0.001) if s > 0 else 0)

    def fom2(s, b):
        return s / np.sqrt(s + b) if (s + b) > 0 else 0

    history = {state: {fom: [] for fom in foms_to_compute} for state in ALL_STATES}

    for thr in thresholds:
        data_mask = data_bdt_score > thr
        state_s_b = {}
        for state in ALL_STATES:
            mc_mask = mc_bdt_scores.get(state, np.array([])) > thr
            eps = np.sum(mc_mask) / mc_totals[state] if mc_totals.get(state, 0) > 0 else 0
            s_est = eps * n_expected[state]

            sw = state_windows[state]
            n_low = float(ak.sum(data_mask & sw["in_low_sb"]))
            n_high = float(ak.sum(data_mask & sw["in_high_sb"]))
            d_l = n_low / b_low_sb_width
            d_h = n_high / b_high_sb_width
            b_est = ((d_l + d_h) / 2.0) * b_sig_width

            state_s_b[state] = (s_est, b_est, eps)

        # High Yield Group
        s_high = sum(state_s_b[st][0] for st in HIGH_YIELD)
        b_high = sum(state_s_b[st][1] for st in HIGH_YIELD)
        val_high1 = fom1(s_high, b_high)
        val_high2 = fom2(s_high, b_high)

        for st in HIGH_YIELD:
            history[st]["S/sqrt(B)"].append(val_high1)
            history[st]["S/sqrt(S+B)"].append(val_high2)
            if val_high1 > best_results[st]["S/sqrt(B)"]["best_fom"]:
                best_results[st]["S/sqrt(B)"]["best_fom"] = val_high1
                best_results[st]["S/sqrt(B)"]["best_cut"] = thr
                best_results[st]["S/sqrt(B)"]["eps"] = state_s_b[st][2]
                best_results[st]["S/sqrt(B)"]["s"] = state_s_b[st][0]
                best_results[st]["S/sqrt(B)"]["b"] = state_s_b[st][1]
            if val_high2 > best_results[st]["S/sqrt(S+B)"]["best_fom"]:
                best_results[st]["S/sqrt(S+B)"]["best_fom"] = val_high2
                best_results[st]["S/sqrt(S+B)"]["best_cut"] = thr
                best_results[st]["S/sqrt(S+B)"]["eps"] = state_s_b[st][2]
                best_results[st]["S/sqrt(S+B)"]["s"] = state_s_b[st][0]
                best_results[st]["S/sqrt(S+B)"]["b"] = state_s_b[st][1]

        # Low Yield Group
        s_low = sum(state_s_b[st][0] for st in LOW_YIELD)
        b_low = sum(state_s_b[st][1] for st in LOW_YIELD)
        val_low1 = fom1(s_low, b_low)
        val_low2 = fom2(s_low, b_low)

        for st in LOW_YIELD:
            history[st]["S/sqrt(B)"].append(val_low1)
            history[st]["S/sqrt(S+B)"].append(val_low2)
            if val_low1 > best_results[st]["S/sqrt(B)"]["best_fom"]:
                best_results[st]["S/sqrt(B)"]["best_fom"] = val_low1
                best_results[st]["S/sqrt(B)"]["best_cut"] = thr
                best_results[st]["S/sqrt(B)"]["eps"] = state_s_b[st][2]
                best_results[st]["S/sqrt(B)"]["s"] = state_s_b[st][0]
                best_results[st]["S/sqrt(B)"]["b"] = state_s_b[st][1]
            if val_low2 > best_results[st]["S/sqrt(S+B)"]["best_fom"]:
                best_results[st]["S/sqrt(S+B)"]["best_fom"] = val_low2
                best_results[st]["S/sqrt(S+B)"]["best_cut"] = thr
                best_results[st]["S/sqrt(S+B)"]["eps"] = state_s_b[st][2]
                best_results[st]["S/sqrt(S+B)"]["s"] = state_s_b[st][0]
                best_results[st]["S/sqrt(S+B)"]["b"] = state_s_b[st][1]

    # Plot FoMs vs Threshold
    plot_dir = Path("analysis_output/plots/mva")
    for group, group_states in [("High_Yield", HIGH_YIELD), ("Low_Yield", LOW_YIELD)]:
        st = group_states[0]  # They share the same FoM calculation
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, history[st]["S/sqrt(B)"], label="S/sqrt(B)")
        plt.plot(thresholds, history[st]["S/sqrt(S+B)"], label="S/sqrt(S+B)")
        plt.axvline(
            best_results[st]["S/sqrt(B)"]["best_cut"],
            color="blue",
            linestyle="--",
            label="Max S/sqrt(B)",
        )
        plt.axvline(
            best_results[st]["S/sqrt(S+B)"]["best_cut"],
            color="orange",
            linestyle="--",
            label="Max S/sqrt(S+B)",
        )
        plt.xlabel("BDT Probability Threshold")
        plt.ylabel("FoM Value")
        plt.title(f"FoM Scan: {group}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"fom_scan_{group}.pdf")
        plt.close()

    # Create summary table
    tables_dir = Path("analysis_output/tables")
    tables_dir.mkdir(exist_ok=True, parents=True)
    rows = []

    # We also need Proxy logic for etac_2s (copying chic1)
    results_ordered = []
    for st in ALL_STATES:
        for f in foms_to_compute:
            results_ordered.append((st, f, best_results[st][f]))
    for f in foms_to_compute:
        results_ordered.append(("etac_2s", f, best_results["chic1"][f]))

    for st, f, r in results_ordered:
        rows.append(
            {
                "state": st,
                "FoM_type": f,
                "optimal_bdt_cut": r["best_cut"],
                "FoM_score": r["best_fom"],
                "S_expected": r["s"],
                "B_estimated": r["b"],
                "efficiency": r["eps"],
            }
        )

    df = pd.DataFrame(rows)
    with open(tables_dir / "mva_optimization_results.md", "w") as f:
        f.write("# BDT Threshold Optimization Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n> **Notes:**\n")
        f.write("> - `eta_c(2S)` inherits `chi_c1` optimal cuts (no MC available).\n")

    logger.info(f"Optimal BDT Cuts:\n{df.to_string()}")

    return best_results
