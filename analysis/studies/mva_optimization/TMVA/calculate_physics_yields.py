import os
import sys
from pathlib import Path

import numpy as np

# Run from TMVA folder
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


def get_bkg_scaling(data_combined, b_high_min, b_low_min, b_low_max, b_sig_min, b_sig_max):
    bu_mass_branch = "Bu_MM_corrected" if "Bu_MM_corrected" in data_combined.fields else "Bu_M"
    masses = np.array(data_combined[bu_mass_branch])

    n_high = np.sum(masses > b_high_min)
    n_low = np.sum((masses >= b_low_min) & (masses <= b_low_max))
    n_sideband = n_high + n_low

    w_high = 5410.0 - b_high_min
    w_low = b_low_max - b_low_min
    w_sideband = w_high + w_low

    w_sig = b_sig_max - b_sig_min
    scaling_factor = w_sig / w_sideband if w_sideband > 0 else 0
    return scaling_factor


def optimize_tmva_cut(ml_data, tmva_model_path, config):
    import ROOT

    ROOT.gROOT.SetBatch(True)

    features = ml_data["features"]

    reader = ROOT.TMVA.Reader("!Color:!Silent")

    var_arrays = {f: np.array([0.0], dtype=np.float32) for f in features}
    for f in features:
        reader.AddVariable(f, var_arrays[f])

    reader.BookMVA("BDT", str(tmva_model_path))

    b_high_min = config.optimization.get("b_high_sideband", [5330.0, 5410.0])[0]
    b_low_min = config.optimization.get("b_low_sideband", [5150.0, 5230.0])[0]
    b_low_max = config.optimization.get("b_low_sideband", [5150.0, 5230.0])[1]
    b_sig_min = config.optimization.get("b_signal_region", [5255.0, 5305.0])[0]
    b_sig_max = config.optimization.get("b_signal_region", [5255.0, 5305.0])[1]

    bkg_scaling = get_bkg_scaling(
        ml_data["data_combined"], b_high_min, b_low_min, b_low_max, b_sig_min, b_sig_max
    )

    data_combined = ml_data["data_combined"]
    bu_mass_branch = "Bu_MM_corrected" if "Bu_MM_corrected" in data_combined.fields else "Bu_M"
    masses = np.array(data_combined[bu_mass_branch])

    bkg_mask = (masses > b_high_min) | ((masses >= b_low_min) & (masses <= b_low_max))

    high_yield_states = ["jpsi", "etac"]
    low_yield_states = ["chic0", "chic1", "etac_2s"]

    mc_prepared = ml_data["mc_prepared"]

    def evaluate_array(events):
        scores = np.zeros(len(events))
        feat_data = {}
        for f in features:
            br = events[f]
            import awkward as ak

            if "var" in str(ak.type(br)):
                br = ak.firsts(br)
            feat_data[f] = ak.to_numpy(br).astype(np.float32)

        for i in range(len(events)):
            for f in features:
                var_arrays[f][0] = feat_data[f][i]
            scores[i] = reader.EvaluateMVA("BDT")
        return scores

    print("Evaluating background...")
    bkg_scores = evaluate_array(data_combined[bkg_mask])

    print("Evaluating signals...")
    high_yield_scores = []
    for state in high_yield_states:
        if state in mc_prepared:
            high_yield_scores.extend(evaluate_array(mc_prepared[state]))
    high_yield_scores = np.array(high_yield_scores)

    low_yield_scores = []
    for state in low_yield_states:
        if state in mc_prepared:
            low_yield_scores.extend(evaluate_array(mc_prepared[state]))
    low_yield_scores = np.array(low_yield_scores)

    cuts = np.linspace(-0.5, 0.5, 100)

    results = {}

    for group, sig_scores in [("High_Yield", high_yield_scores), ("Low_Yield", low_yield_scores)]:
        best_fom = -1
        best_cut = -1
        best_s = 0
        best_b = 0

        for cut in cuts:
            s = float(np.sum(sig_scores > cut))
            b = float(np.sum(bkg_scores > cut) * bkg_scaling)

            if s > 0:
                fom = float(s / np.sqrt(s + b))
                if fom > best_fom:
                    best_fom = fom
                    best_cut = float(cut)
                    best_s = s
                    best_b = b

        results[group] = {"cut": best_cut, "fom": best_fom, "s_raw": best_s, "b_exp": best_b}
        print(
            f"{group}: Best Cut = {best_cut:.3f}, S = {best_s:.1f}, B = {best_b:.1f}, S/sqrt(S+B) = {best_fom:.3f}"
        )

    return results


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
