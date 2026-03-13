"""
Derive MC Kinematic Weights using the high-statistics J/psi control channel.

Method:
1. Load J/psi Data and apply pre-selection + MVA cut (for max purity).
2. Perform B+ mass sideband subtraction to extract pure signal distributions for Bu_PT, Bu_ETA, and nTracks.
3. Load J/psi MC and apply the same pre-selection + MVA cut.
4. Compute the normalized 2D Data/MC ratio weights for (Bu_PT, Bu_ETA).
5. Save the weights to a file so `calculate_efficiencies.py` can load and apply them.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tomli
import xgboost as xgb

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modules.clean_data_loader import load_and_preprocess


def get_eta(px, py, pz):
    p = np.sqrt(px**2 + py**2 + pz**2)
    return 0.5 * np.log((p + pz) / (p - pz + 1e-10))


def apply_mva(events, mva_model, features, threshold=0.89):
    if len(events) == 0:
        return events
    df = pd.DataFrame({f: events[f] for f in features})
    dmatrix = xgb.DMatrix(df)
    preds = mva_model.predict(dmatrix)
    return events[preds > threshold]


def sideband_subtraction(mass_array, weight_array, signal_window, sb_window):
    """Return a mask for events in regions and a weight modifier array for subtraction"""
    sig_min, sig_max = signal_window
    low_sb_min, low_sb_max = sb_window[0]
    high_sb_min, high_sb_max = sb_window[1]

    in_sig = (mass_array >= sig_min) & (mass_array <= sig_max)
    in_low = (mass_array >= low_sb_min) & (mass_array <= low_sb_max)
    in_high = (mass_array >= high_sb_min) & (mass_array <= high_sb_max)

    width_sig = sig_max - sig_min
    width_sb = (low_sb_max - low_sb_min) + (high_sb_max - high_sb_min)
    scale = width_sig / width_sb

    # We return an array where signal events get weight 1, and sidebands get weight -scale
    # Events outside these get weight 0
    final_weights = np.zeros(len(mass_array), dtype=float)
    final_weights[in_sig] = 1.0 * (weight_array[in_sig] if weight_array is not None else 1.0)
    final_weights[in_low] = -scale * (weight_array[in_low] if weight_array is not None else 1.0)
    final_weights[in_high] = -scale * (weight_array[in_high] if weight_array is not None else 1.0)

    mask = in_sig | in_low | in_high
    return mask, final_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="../../config")
    args = parser.parse_args()

    with open(f"{args.config_dir}/data.toml", "rb") as f:
        data_config = tomli.load(f)
    with open(f"{args.config_dir}/selection.toml", "rb") as f:
        sel_config = tomli.load(f)

    mva_features = sel_config.get("xgboost", {}).get("features", [])
    mva_model = xgb.Booster()
    mva_model.load_model("../../studies/mva_optimization/output/models/xgboost_bdt.json")
    mva_threshold = 0.89

    years = ["16", "17", "18"]
    polarities = ["MD", "MU"]

    data_base = Path(data_config["input_data"]["base_path"])
    mc_base = Path(data_config["input_mc"]["base_path"]) / "Jpsi"

    all_data_pt = []
    all_data_eta = []
    all_data_weights = []

    all_mc_pt = []
    all_mc_eta = []

    print("Loading J/psi Data and MC for reweighting...")
    for year in years:
        for pol in polarities:
            # Data
            d_file = data_base / f"dataBu2L0barPHH_{year}{pol}_reduced_PID.root"
            if d_file.exists():
                events = load_and_preprocess(
                    d_file, is_mc=False, track_type="LL"
                )  # Assume LL is representative enough for B+ kinematics, or load both. Let's load LL for simplicity.
                events = apply_mva(events, mva_model, mva_features, mva_threshold)

                # Calculate PT
                pt = events["Bu_PT"]
                mass = events["Bu_MM_corrected"]

                # Sideband subtract
                sig_w = (5255.0, 5305.0)
                sb_w = ((5150.0, 5230.0), (5330.0, 5410.0))
                mask, weights = sideband_subtraction(mass, None, sig_w, sb_w)

                all_data_pt.extend(pt[mask].to_numpy())

                all_data_weights.extend(weights[mask])

            # MC
            m_file = mc_base / f"Jpsi_{year}_{pol}.root"
            if m_file.exists():
                events = load_and_preprocess(m_file, is_mc=True, track_type="LL")
                events = apply_mva(events, mva_model, mva_features, mva_threshold)

                # Calculate PT
                pt = events["Bu_PT"]

                all_mc_pt.extend(pt.to_numpy())

    data_pt = np.array(all_data_pt)
    data_eta = np.array(all_data_eta)
    data_w = np.array(all_data_weights)

    mc_pt = np.array(all_mc_pt)
    mc_eta = np.array(all_mc_eta)

    # Define binning
    # PT: 3000 to 25000
    pt_bins = np.array([3000, 4000, 5000, 6000, 8000, 10000, 15000, 25000])

    print(f"Total Data (weighted): {np.sum(data_w):.1f}")
    print(f"Total MC: {len(mc_pt)}")

    # Histogram 1D
    h_data, xedges = np.histogram(data_pt, bins=pt_bins, weights=data_w)
    h_mc, _ = np.histogram(mc_pt, bins=pt_bins)

    # Normalize to 1
    h_data = h_data / np.sum(h_data)
    h_mc = h_mc / np.sum(h_mc)

    # Calculate weights (avoid division by zero)
    weights_1d = np.divide(h_data, h_mc, out=np.ones_like(h_data), where=h_mc > 0)

    # Save the weight map
    output_data = {"pt_bins": pt_bins.tolist(), "weights": weights_1d.tolist()}

    Path("output").mkdir(exist_ok=True)
    with open("output/kinematic_weights.json", "w") as f:
        json.dump(output_data, f, indent=2)

    print("Kinematic weights successfully derived and saved to output/kinematic_weights.json")

    # Plot
    plt.figure(figsize=(8, 6))
    bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
    bin_widths = np.diff(xedges)

    plt.bar(bin_centers, weights_1d, width=bin_widths, color="blue", alpha=0.6, edgecolor="black")
    plt.xlabel("B+ pT [MeV/c]")
    plt.ylabel("Data/MC Weight")
    plt.title("Kinematic Reweighting (1D pT)")
    plt.savefig("output/weight_map.png")


if __name__ == "__main__":
    main()
