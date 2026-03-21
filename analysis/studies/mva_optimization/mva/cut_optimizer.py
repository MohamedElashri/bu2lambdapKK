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


def optimize_bdt_cut(config: StudyConfig, model, ml_data: dict, category: str = ""):
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
    mc_truth_expected = {}  # Store true signal count from MC
    mc_truth_scaling = {}  # Store scaling factor for true MC to match data expected
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

        # Calculate MC Truth Signal Yield if branches available
        evts = mc_prepared.get(state, None)
        if evts is not None and "Bu_TRUEID" in evts.fields:
            # Check TRUEID branches:
            # B+ -> 521, p -> 2212, K- -> 321
            true_b = np.abs(evts["Bu_TRUEID"]) == 521
            true_p = np.abs(evts["p_TRUEID"]) == 2212 if "p_TRUEID" in evts.fields else True
            true_k1 = np.abs(evts["h1_TRUEID"]) == 321 if "h1_TRUEID" in evts.fields else True
            true_k2 = np.abs(evts["h2_TRUEID"]) == 321 if "h2_TRUEID" in evts.fields else True

            is_true_sig = true_b & true_p & true_k1 & true_k2
            total_true = float(ak.sum(is_true_sig))
            mc_truth_expected[state] = int(total_true)
            mc_truth_scaling[state] = n_expected[state] / total_true if total_true > 0 else 0.0
        else:
            mc_truth_expected[state] = 0
            mc_truth_scaling[state] = 0.0

    # 4) Scan threshold
    thresholds = np.linspace(0.01, 0.99, 99)

    # Option A Logic: Optimize globally for high yield vs low yield groups
    # Rather than best cut per state, we compute a single optimal cut for HIGH_YIELD
    # and a single optimal cut for LOW_YIELD

    best_results = {
        "High_Yield": {
            "S/sqrt(B)": {
                "best_fom": -np.inf,
                "best_cut": 0,
                "s": 0,
                "b": 0,
                "true_s": 0,
                "true_s_scaled": 0,
            },
            "S/sqrt(S+B)": {
                "best_fom": -np.inf,
                "best_cut": 0,
                "s": 0,
                "b": 0,
                "true_s": 0,
                "true_s_scaled": 0,
            },
        },
        "Low_Yield": {
            "S/sqrt(B)": {
                "best_fom": -np.inf,
                "best_cut": 0,
                "s": 0,
                "b": 0,
                "true_s": 0,
                "true_s_scaled": 0,
            },
            "S/sqrt(S+B)": {
                "best_fom": -np.inf,
                "best_cut": 0,
                "s": 0,
                "b": 0,
                "true_s": 0,
                "true_s_scaled": 0,
            },
        },
    }

    def fom1(s, b):
        return s / np.sqrt(b) if b > 0 else (s / np.sqrt(0.001) if s > 0 else 0)

    def fom2(s, b):
        return s / np.sqrt(s + b) if (s + b) > 0 else 0

    history = {
        group: {fom: [] for fom in ["S/sqrt(B)", "S/sqrt(S+B)"]}
        for group in ["High_Yield", "Low_Yield"]
    }

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

            state_s_b[state] = (s_est, b_est)

        # High Yield Group
        s_high = sum(state_s_b[st][0] for st in HIGH_YIELD)
        b_high = sum(state_s_b[st][1] for st in HIGH_YIELD)
        val_high1 = fom1(s_high, b_high)
        val_high2 = fom2(s_high, b_high)

        history["High_Yield"]["S/sqrt(B)"].append(val_high1)
        history["High_Yield"]["S/sqrt(S+B)"].append(val_high2)

        if val_high1 > best_results["High_Yield"]["S/sqrt(B)"]["best_fom"]:
            best_results["High_Yield"]["S/sqrt(B)"]["best_fom"] = val_high1
            best_results["High_Yield"]["S/sqrt(B)"]["best_cut"] = thr
            best_results["High_Yield"]["S/sqrt(B)"]["s"] = s_high
            best_results["High_Yield"]["S/sqrt(B)"]["b"] = b_high
            best_results["High_Yield"]["S/sqrt(B)"]["true_s"] = sum(
                np.sum(
                    (mc_bdt_scores.get(st, np.array([])) > thr)
                    & (
                        (np.abs(mc_prepared[st]["Bu_TRUEID"]) == 521)
                        if "Bu_TRUEID" in mc_prepared[st].fields
                        else True
                    )
                )
                for st in HIGH_YIELD
            )
            best_results["High_Yield"]["S/sqrt(B)"]["true_s_scaled"] = sum(
                np.sum(
                    (mc_bdt_scores.get(st, np.array([])) > thr)
                    & (
                        (np.abs(mc_prepared[st]["Bu_TRUEID"]) == 521)
                        if "Bu_TRUEID" in mc_prepared[st].fields
                        else True
                    )
                )
                * mc_truth_scaling[st]
                for st in HIGH_YIELD
            )

        if val_high2 > best_results["High_Yield"]["S/sqrt(S+B)"]["best_fom"]:
            best_results["High_Yield"]["S/sqrt(S+B)"]["best_fom"] = val_high2
            best_results["High_Yield"]["S/sqrt(S+B)"]["best_cut"] = thr
            best_results["High_Yield"]["S/sqrt(S+B)"]["s"] = s_high
            best_results["High_Yield"]["S/sqrt(S+B)"]["b"] = b_high
            best_results["High_Yield"]["S/sqrt(S+B)"]["true_s"] = sum(
                np.sum(
                    (mc_bdt_scores.get(st, np.array([])) > thr)
                    & (
                        (np.abs(mc_prepared[st]["Bu_TRUEID"]) == 521)
                        if "Bu_TRUEID" in mc_prepared[st].fields
                        else True
                    )
                )
                for st in HIGH_YIELD
            )
            best_results["High_Yield"]["S/sqrt(S+B)"]["true_s_scaled"] = sum(
                np.sum(
                    (mc_bdt_scores.get(st, np.array([])) > thr)
                    & (
                        (np.abs(mc_prepared[st]["Bu_TRUEID"]) == 521)
                        if "Bu_TRUEID" in mc_prepared[st].fields
                        else True
                    )
                )
                * mc_truth_scaling[st]
                for st in HIGH_YIELD
            )

        # Low Yield Group
        s_low = sum(state_s_b[st][0] for st in LOW_YIELD)
        b_low = sum(state_s_b[st][1] for st in LOW_YIELD)
        val_low1 = fom1(s_low, b_low)
        val_low2 = fom2(s_low, b_low)

        history["Low_Yield"]["S/sqrt(B)"].append(val_low1)
        history["Low_Yield"]["S/sqrt(S+B)"].append(val_low2)

        if val_low1 > best_results["Low_Yield"]["S/sqrt(B)"]["best_fom"]:
            best_results["Low_Yield"]["S/sqrt(B)"]["best_fom"] = val_low1
            best_results["Low_Yield"]["S/sqrt(B)"]["best_cut"] = thr
            best_results["Low_Yield"]["S/sqrt(B)"]["s"] = s_low
            best_results["Low_Yield"]["S/sqrt(B)"]["b"] = b_low
            best_results["Low_Yield"]["S/sqrt(B)"]["true_s"] = sum(
                np.sum(
                    (mc_bdt_scores.get(st, np.array([])) > thr)
                    & (
                        (np.abs(mc_prepared[st]["Bu_TRUEID"]) == 521)
                        if "Bu_TRUEID" in mc_prepared[st].fields
                        else True
                    )
                )
                for st in LOW_YIELD
            )
            best_results["Low_Yield"]["S/sqrt(B)"]["true_s_scaled"] = sum(
                np.sum(
                    (mc_bdt_scores.get(st, np.array([])) > thr)
                    & (
                        (np.abs(mc_prepared[st]["Bu_TRUEID"]) == 521)
                        if "Bu_TRUEID" in mc_prepared[st].fields
                        else True
                    )
                )
                * mc_truth_scaling[st]
                for st in LOW_YIELD
            )

        if val_low2 > best_results["Low_Yield"]["S/sqrt(S+B)"]["best_fom"]:
            best_results["Low_Yield"]["S/sqrt(S+B)"]["best_fom"] = val_low2
            best_results["Low_Yield"]["S/sqrt(S+B)"]["best_cut"] = thr
            best_results["Low_Yield"]["S/sqrt(S+B)"]["s"] = s_low
            best_results["Low_Yield"]["S/sqrt(S+B)"]["b"] = b_low
            best_results["Low_Yield"]["S/sqrt(S+B)"]["true_s"] = sum(
                np.sum(
                    (mc_bdt_scores.get(st, np.array([])) > thr)
                    & (
                        (np.abs(mc_prepared[st]["Bu_TRUEID"]) == 521)
                        if "Bu_TRUEID" in mc_prepared[st].fields
                        else True
                    )
                )
                for st in LOW_YIELD
            )
            best_results["Low_Yield"]["S/sqrt(S+B)"]["true_s_scaled"] = sum(
                np.sum(
                    (mc_bdt_scores.get(st, np.array([])) > thr)
                    & (
                        (np.abs(mc_prepared[st]["Bu_TRUEID"]) == 521)
                        if "Bu_TRUEID" in mc_prepared[st].fields
                        else True
                    )
                )
                * mc_truth_scaling[st]
                for st in LOW_YIELD
            )

    # Plot FoMs vs Threshold
    cat_suffix = f"_{category}" if category else ""
    plot_dir = Path("../output/plots/mva")
    plot_dir.mkdir(parents=True, exist_ok=True)
    for group in ["High_Yield", "Low_Yield"]:
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, history[group]["S/sqrt(B)"], label="S/sqrt(B)")
        plt.plot(thresholds, history[group]["S/sqrt(S+B)"], label="S/sqrt(S+B)")
        plt.axvline(
            best_results[group]["S/sqrt(B)"]["best_cut"],
            color="blue",
            linestyle="--",
            label="Max S/sqrt(B)",
        )
        plt.axvline(
            best_results[group]["S/sqrt(S+B)"]["best_cut"],
            color="orange",
            linestyle="--",
            label="Max S/sqrt(S+B)",
        )
        plt.xlabel("BDT Probability Threshold")
        plt.ylabel("FoM Value")
        plt.title(f"FoM Scan: {group}{' [' + category + ']' if category else ''}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"fom_scan_{group}{cat_suffix}.pdf")
        plt.close()

    # Create summary table
    tables_dir = Path("../output/tables")
    tables_dir.mkdir(exist_ok=True, parents=True)
    rows = []

    foms_to_compute = ["S/sqrt(B)", "S/sqrt(S+B)"]
    for group in ["High_Yield", "Low_Yield"]:
        for f in foms_to_compute:
            r = best_results[group][f]
            rows.append(
                {
                    "Group": group,
                    "FoM_type": f,
                    "optimal_bdt_cut": r["best_cut"],
                    "FoM_score": r["best_fom"],
                    "S_expected": r["s"],
                    "S_true_scaled": r["true_s_scaled"],
                    "S_true_raw": r["true_s"],
                    "B_estimated": r["b"],
                }
            )

    df = pd.DataFrame(rows)
    results_filename = f"mva_optimization_results{cat_suffix}.md"
    with open(tables_dir / results_filename, "w") as f:
        cat_label = f" [{category}]" if category else ""
        f.write(f"# BDT Threshold Optimization Results{cat_label}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n> **Notes:**\n")
        f.write("> - `High_Yield` applies to J/psi and eta_c(1S).\n")
        f.write("> - `Low_Yield` applies to chi_c0, chi_c1, and eta_c(2S).\n")
        f.write("> - `S_expected` is estimated via sideband subtraction on data.\n")
        f.write(
            "> - `S_true_raw` is the exact number of truth-matched signal events passing the cut in MC.\n"
        )
        f.write(
            "> - `S_true_scaled` is `S_true_raw` scaled by the ratio of data yield / total MC yield to be comparable with `S_expected`.\n"
        )

    logger.info(f"Optimal BDT Cuts:\n{df.to_string()}")

    return best_results
