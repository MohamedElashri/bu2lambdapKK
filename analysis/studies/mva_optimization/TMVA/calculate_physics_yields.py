import os
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

script_dir = Path(__file__).resolve().parent
os.chdir(str(script_dir))

project_root = script_dir.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

mva_opt_dir = script_dir.parent / "mva"
if str(mva_opt_dir) not in sys.path:
    sys.path.insert(0, str(mva_opt_dir))

from config_loader import StudyConfig
from data_preparation import load_and_prepare_data


def evaluate_tmva(reader, var_arrays, features, X_df):
    scores = np.zeros(len(X_df))
    X_vals = X_df.values
    for i in range(len(X_vals)):
        for j, f in enumerate(features):
            var_arrays[f][0] = X_vals[i, j]
        scores[i] = reader.EvaluateMVA("BDT")
    return scores


def optimize_tmva_cut(ml_data, tmva_model_path, config):
    import ROOT

    ROOT.gROOT.SetBatch(True)

    features = ml_data["features"]
    reader = ROOT.TMVA.Reader("!Color:!Silent")
    var_arrays = {f: np.array([0.0], dtype=np.float32) for f in features}
    for f in features:
        reader.AddVariable(f, var_arrays[f])
    reader.BookMVA("BDT", str(tmva_model_path))

    data_combined = ml_data["data_combined"]
    mc_prepared = ml_data["mc_prepared"]

    # 1) Evaluate BDT on data
    print("Evaluating BDT on data...")
    df_data_dict = {}
    for feat in features:
        br = data_combined[feat]
        if "var" in str(ak.type(br)):
            br = ak.firsts(br)
        df_data_dict[feat] = ak.to_numpy(br)
    X_data = pd.DataFrame(df_data_dict)[features]
    data_bdt_score = evaluate_tmva(reader, var_arrays, features, X_data)

    # 2) Evaluate BDT on MC per state
    print("Evaluating BDT on MC...")
    mc_bdt_scores = {}
    mc_totals = {}
    for state, evts in mc_prepared.items():
        df_mc_dict = {}
        for feat in features:
            br = evts[feat]
            if "var" in str(ak.type(br)):
                br = ak.firsts(br)
            df_mc_dict[feat] = ak.to_numpy(br)
        X_mc = pd.DataFrame(df_mc_dict)[features]
        mc_bdt_scores[state] = evaluate_tmva(reader, var_arrays, features, X_mc)
        mc_totals[state] = len(evts)

    # 3) Reconstruct FOM logical boundaries exactly as cut_optimizer.py
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
    mc_truth_expected = {}
    mc_truth_scaling = {}
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

        evts = mc_prepared.get(state, None)
        if evts is not None and "Bu_TRUEID" in evts.fields:
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

    # 4) Scan threshold for TMVA (AdaBoost is generally -1 to 1)
    thresholds = np.linspace(-0.6, 0.6, 120)

    best_results = {
        "High_Yield": {
            "best_fom": -np.inf,
            "best_cut": 0,
            "s": 0,
            "b": 0,
            "true_s": 0,
            "true_s_scaled": 0,
        },
        "Low_Yield": {
            "best_fom": -np.inf,
            "best_cut": 0,
            "s": 0,
            "b": 0,
            "true_s": 0,
            "true_s_scaled": 0,
        },
    }

    def fom2(s, b):
        return s / np.sqrt(s + b) if (s + b) > 0 else 0

    print("Scanning thresholds...")
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

        # High Yield
        s_high = sum(state_s_b[st][0] for st in HIGH_YIELD)
        b_high = sum(state_s_b[st][1] for st in HIGH_YIELD)
        val_high = fom2(s_high, b_high)

        if val_high > best_results["High_Yield"]["best_fom"]:
            best_results["High_Yield"]["best_fom"] = float(val_high)
            best_results["High_Yield"]["best_cut"] = float(thr)
            best_results["High_Yield"]["s"] = float(s_high)
            best_results["High_Yield"]["b"] = float(b_high)

            true_s = 0
            for st in HIGH_YIELD:
                st_evts = mc_prepared.get(st, None)
                if st_evts is not None:
                    mc_mask = mc_bdt_scores[st] > thr
                    true_b = (
                        np.abs(st_evts["Bu_TRUEID"]) == 521
                        if "Bu_TRUEID" in st_evts.fields
                        else True
                    )
                    true_s += float(np.sum(mc_mask & true_b))
            best_results["High_Yield"]["true_s"] = float(true_s)

        # Low Yield
        s_low = sum(state_s_b[st][0] for st in LOW_YIELD)
        b_low = sum(state_s_b[st][1] for st in LOW_YIELD)
        val_low = fom2(s_low, b_low)

        if val_low > best_results["Low_Yield"]["best_fom"]:
            best_results["Low_Yield"]["best_fom"] = float(val_low)
            best_results["Low_Yield"]["best_cut"] = float(thr)
            best_results["Low_Yield"]["s"] = float(s_low)
            best_results["Low_Yield"]["b"] = float(b_low)

            true_s = 0
            for st in LOW_YIELD:
                st_evts = mc_prepared.get(st, None)
                if st_evts is not None:
                    mc_mask = mc_bdt_scores[st] > thr
                    true_b = (
                        np.abs(st_evts["Bu_TRUEID"]) == 521
                        if "Bu_TRUEID" in st_evts.fields
                        else True
                    )
                    true_s += float(np.sum(mc_mask & true_b))
            best_results["Low_Yield"]["true_s"] = float(true_s)

    # Format identically to match get_tmva_results() in generate_slides.py
    # "cut", "fom", "s_raw", "b_exp"
    output = {
        "High_Yield": {
            "cut": best_results["High_Yield"]["best_cut"],
            "fom": best_results["High_Yield"]["best_fom"],
            "s_raw": best_results["High_Yield"]["true_s"],
            "b_exp": best_results["High_Yield"]["b"],
        },
        "Low_Yield": {
            "cut": best_results["Low_Yield"]["best_cut"],
            "fom": best_results["Low_Yield"]["best_fom"],
            "s_raw": best_results["Low_Yield"]["true_s"],
            "b_exp": best_results["Low_Yield"]["b"],
        },
    }

    return output


if __name__ == "__main__":
    import os

    os.chdir(str(mva_opt_dir))
    config = StudyConfig()
    ml_data = load_and_prepare_data(config)
    os.chdir(str(script_dir))

    model_path = script_dir / "dataset" / "weights" / "TMVAClassification_BDT.weights.xml"
    if model_path.exists():
        results = optimize_tmva_cut(ml_data, model_path, config)

        import json

        out_path = script_dir.parent / "comparison" / "raw_data" / "tmva_yields.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved TMVA yields to {out_path}")
    else:
        print(f"Model not found at {model_path}")
