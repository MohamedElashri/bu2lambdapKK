"""
Derive MC kinematic weights using the high-statistics J/psi control channel.

Method:
1. Load J/psi data (both LL and DD) and apply the current shared pre-selection.
2. Perform B+ mass sideband subtraction to extract pure signal pT distributions.
3. Load J/psi MC and apply the same pre-selection.
4. Compute the normalised 1D Data/MC ratio weights for Bu_pT, separately per category.
5. Save per-category weight maps:
   output/kinematic_weights_LL.json
   output/kinematic_weights_DD.json

Current workflow notes:
- Updated from XGBoost to CatBoost model.
- Fixed the bug where all_data_eta was initialised but never populated (eta array removed
  since we only compute 1D pT weights; extending to 2D remains a future enhancement).
- Added per-category (LL and DD) weight derivation.  Separate weights are needed because
  the B+ pT spectrum can differ between LL and DD due to different geometric acceptances.
- The script now reads the pre-selection from the main pipeline's clean_data_loader.py
  (via direct import) to stay consistent.

Note on a possible 2D extension:
  nTracks is available in the ntuples as "nTracks" or "nSPDHits" depending on the year.
  To extend to 2D (pT × nTracks), replace the 1D histogramming below with:
    h_data_2d, xe, ye = np.histogram2d(data_pt, data_ntracks, bins=[pt_bins, ntracks_bins], weights=data_w)
    h_mc_2d,   _,  _ = np.histogram2d(mc_pt,   mc_ntracks,   bins=[pt_bins, ntracks_bins])
  and save the 2D weight map.  The 2D lookup in calculate_efficiencies.py should then
  use np.digitize on both axes.
"""

import argparse
import json
import sys
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

# Add analysis root to path
analysis_root = Path(__file__).resolve().parent.parent.parent
if str(analysis_root) not in sys.path:
    sys.path.insert(0, str(analysis_root))

from modules.clean_data_loader import load_and_preprocess
from modules.config_loader import StudyConfig
from modules.plot_utils import setup_style

setup_style()


def apply_catboost(events, model_path: Path, features: list, threshold: float):
    """Apply CatBoost BDT and return events passing the threshold."""
    if len(events) == 0 or not model_path.exists():
        return events
    import pandas as pd
    from catboost import CatBoostClassifier

    df_dict = {}
    for feat in features:
        arr = events[feat]
        if "var" in str(ak.type(arr)):
            arr = ak.firsts(arr)
        df_dict[feat] = ak.to_numpy(arr)
    df = pd.DataFrame(df_dict)

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    preds = model.predict_proba(df)[:, 1]
    return events[preds > threshold]


def sideband_weights(
    mass_array, signal_window=(5255.0, 5305.0), sb_lo=(5150.0, 5230.0), sb_hi=(5330.0, 5410.0)
):
    """Return per-event sideband-subtraction weights (+1 in signal, −scale in sidebands)."""
    sig_min, sig_max = signal_window
    lo_min, lo_max = sb_lo
    hi_min, hi_max = sb_hi

    width_sig = sig_max - sig_min
    width_sb = (lo_max - lo_min) + (hi_max - hi_min)
    scale = width_sig / width_sb

    w = np.zeros(len(mass_array), dtype=float)
    w[(mass_array >= sig_min) & (mass_array <= sig_max)] = 1.0
    w[(mass_array >= lo_min) & (mass_array <= lo_max)] = -scale
    w[(mass_array >= hi_min) & (mass_array <= hi_max)] = -scale
    return w


def derive_weights_for_category(
    category: str,
    data_base: Path,
    mc_base: Path,
    years: list,
    polarities: list,
    config: StudyConfig,
    mva_model_path: Path,
    mva_features: list,
    mva_threshold: float,
    delta_z_cuts: dict,
) -> dict:
    """Derive 1D pT kinematic weights for one Lambda category."""
    print(f"\n[{category}] Deriving kinematic weights...")

    data_pt_all, data_w_all = [], []
    mc_pt_all = []

    for year in years:
        for pol in polarities:
            # ---- Data ----
            d_file = data_base / f"dataBu2L0barPHH_{year}{pol}.root"
            if d_file.exists():
                events = load_and_preprocess(
                    d_file,
                    is_mc=False,
                    track_type=category,
                    config=config,
                    delta_z_cut=delta_z_cuts[category],
                )
                events = apply_catboost(events, mva_model_path, mva_features, mva_threshold)
                if len(events) > 0:
                    mass = ak.to_numpy(events["Bu_MM_corrected"])
                    pt = ak.to_numpy(events["Bu_PT"])
                    w = sideband_weights(mass)
                    # only keep events in the three regions
                    in_region = w != 0
                    data_pt_all.extend(pt[in_region])
                    data_w_all.extend(w[in_region])
            else:
                print(f"  [{category}] Data file not found: {d_file}")

            # ---- J/psi MC ----
            m_file = mc_base / f"Jpsi_{year}_{pol}.root"
            if m_file.exists():
                events = load_and_preprocess(
                    m_file,
                    is_mc=True,
                    track_type=category,
                    config=config,
                    delta_z_cut=delta_z_cuts[category],
                )
                events = apply_catboost(events, mva_model_path, mva_features, mva_threshold)
                if len(events) > 0:
                    mc_pt_all.extend(ak.to_numpy(events["Bu_PT"]))
            else:
                print(f"  [{category}] MC file not found: {m_file}")

    data_pt = np.array(data_pt_all)
    data_w = np.array(data_w_all)
    mc_pt = np.array(mc_pt_all)

    print(f"  [{category}] Data (sideband-subtracted): {np.sum(data_w):.0f} signal events")
    print(f"  [{category}] MC: {len(mc_pt)} events")

    if len(data_pt) == 0 or len(mc_pt) == 0:
        print(f"  [{category}] Insufficient data — returning uniform weights.")
        return {"pt_bins": [3000, 25000], "weights": [1.0]}

    pt_bins = np.array([3000, 4000, 5000, 6000, 8000, 10000, 15000, 25000], dtype=float)

    h_data, _ = np.histogram(data_pt, bins=pt_bins, weights=data_w)
    h_mc, _ = np.histogram(mc_pt, bins=pt_bins)

    # Normalise to unity so the weights are pure shape corrections
    h_data_norm = h_data / np.sum(h_data) if np.sum(h_data) != 0 else h_data
    h_mc_norm = h_mc / np.sum(h_mc) if np.sum(h_mc) != 0 else h_mc

    weights_1d = np.divide(
        h_data_norm,
        h_mc_norm,
        out=np.ones_like(h_data_norm),
        where=h_mc_norm > 0,
    )

    print(f"  [{category}] Weight range: [{weights_1d.min():.3f}, {weights_1d.max():.3f}]")
    return {"pt_bins": pt_bins.tolist(), "weights": weights_1d.tolist()}


def derive_weights_varied(
    category: str, nominal_weights: dict, variation: str, rng: np.random.Generator
) -> dict:
    """
    Produce one kinematic weight variation for the systematic study.

    Variations:
      "fine_bins"   : Double the number of pT bins (finer granularity)
      "coarse_bins" : Halve the number of pT bins (coarser granularity)
      "bootstrap"   : Resample weights with Gaussian noise proportional to bin occupancy
                      (conservative 5% relative uncertainty per bin as no bin-stat errors
                      are stored in the 1D map)

    Args:
        category:        "LL" or "DD"
        nominal_weights: dict with "pt_bins" and "weights" from `derive_weights_for_category`
        variation:       one of "fine_bins", "coarse_bins", "bootstrap"
        rng:             numpy Generator

    Returns:
        dict with "pt_bins" and "weights" for the varied map
    """
    pt_bins_nom = np.array(nominal_weights["pt_bins"])
    w_nom = np.array(nominal_weights["weights"])

    if variation == "bootstrap":
        # Smear each weight bin by 5% relative (conservative without bin-stat info)
        sigma = 0.05 * np.abs(w_nom)
        w_var = np.clip(w_nom + rng.normal(0.0, sigma), 0.1, 10.0)
        return {"pt_bins": pt_bins_nom.tolist(), "weights": w_var.tolist()}

    if variation == "fine_bins":
        # Insert midpoints between existing bin edges to double bin count
        extra = 0.5 * (pt_bins_nom[:-1] + pt_bins_nom[1:])
        new_bins = np.sort(np.concatenate([pt_bins_nom, extra]))
        # Each new bin inherits the weight of its parent bin
        new_weights = []
        for i in range(len(new_bins) - 1):
            mid = 0.5 * (new_bins[i] + new_bins[i + 1])
            idx = np.searchsorted(pt_bins_nom, mid, side="right") - 1
            idx = int(np.clip(idx, 0, len(w_nom) - 1))
            new_weights.append(float(w_nom[idx]))
        return {"pt_bins": new_bins.tolist(), "weights": new_weights}

    if variation == "coarse_bins":
        # Merge adjacent bins in pairs (average weights, sum bin edges)
        n = len(w_nom)
        pairs = n // 2
        coarse_bins = [pt_bins_nom[0]]
        coarse_weights = []
        for i in range(pairs):
            coarse_bins.append(pt_bins_nom[2 * i + 2])
            coarse_weights.append(0.5 * (w_nom[2 * i] + w_nom[2 * i + 1]))
        if 2 * pairs < n:  # odd bin — keep last bin as-is
            coarse_bins.append(pt_bins_nom[-1])
            coarse_weights.append(float(w_nom[-1]))
        return {"pt_bins": coarse_bins, "weights": coarse_weights}

    raise ValueError(f"Unknown variation '{variation}'")


def main():
    parser = argparse.ArgumentParser(description="Derive kinematic weights per category")
    parser.add_argument("--config-dir", default="../../config")
    parser.add_argument(
        "--branch", default="high_yield", help="Branch name for loading the per-category MVA model."
    )
    parser.add_argument(
        "--compute-variations",
        action="store_true",
        help="Also compute systematic variations and save to "
        "output/kinematic_weights_{cat}_var_{name}.json",
    )
    args = parser.parse_args()

    config = StudyConfig.from_dir(args.config_dir)

    mva_features = config.xgboost.get("features", [])
    mva_threshold = 0.89  # default; overridden by optimized_cuts.json if available

    data_base = config.get_input_data_base_path()
    mc_base = config.get_input_mc_base_path() / "Jpsi"
    years = [year[-2:] for year in (config.get_input_years() or ["2016", "2017", "2018"])]
    polarities = config.get_input_magnets() or ["MD", "MU"]
    delta_z_cuts = {cat: config.get_category_delta_z_cut(cat) for cat in ["LL", "DD"]}

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    all_weights = {}
    for category in ["LL", "DD"]:
        # Try to load optimized threshold for this category
        cuts_path = Path(
            f"../../analysis_output/mva/{args.branch}/{category}/models/optimized_cuts.json"
        )
        cat_threshold = mva_threshold
        if cuts_path.exists():
            with open(cuts_path, "r") as f:
                cuts_data = json.load(f)
            cat_threshold = cuts_data.get("mva_threshold_high", mva_threshold)
            print(f"[{category}] Loaded MVA threshold = {cat_threshold:.3f} from {cuts_path}")

        # Per-category CatBoost model
        model_path = Path(
            f"../../analysis_output/mva/{args.branch}/{category}/models/mva_model.cbm"
        )

        weights = derive_weights_for_category(
            category=category,
            data_base=data_base,
            mc_base=mc_base,
            years=years,
            polarities=polarities,
            config=config,
            mva_model_path=model_path,
            mva_features=mva_features,
            mva_threshold=cat_threshold,
            delta_z_cuts=delta_z_cuts,
        )

        out_file = output_dir / f"kinematic_weights_{category}.json"
        with open(out_file, "w") as f:
            json.dump(weights, f, indent=2)
        print(f"  [{category}] Saved to {out_file}")
        all_weights[category] = weights

        # Plot
        pt_bins = np.array(weights["pt_bins"])
        w_vals = np.array(weights["weights"])
        centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
        widths = np.diff(pt_bins)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(centers, w_vals, width=widths * 0.8, color="steelblue", alpha=0.7, edgecolor="k")
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("B+ pT [MeV/c]")
        ax.set_ylabel("Data/MC weight")
        ax.set_title(f"Kinematic reweighting — {category}")
        fig.tight_layout()
        fig.savefig(output_dir / f"weight_map_{category}.pdf")
        plt.close(fig)

    # Compute systematic variations if requested.
    if args.compute_variations:
        rng = np.random.default_rng(seed=42)
        for cat, nom_w in all_weights.items():
            for var_name in ("fine_bins", "coarse_bins", "bootstrap"):
                var_w = derive_weights_varied(cat, nom_w, var_name, rng)
                var_file = output_dir / f"kinematic_weights_{cat}_var_{var_name}.json"
                with open(var_file, "w") as f:
                    json.dump(var_w, f, indent=2)
                print(f"  [{cat}] Variation '{var_name}' saved to {var_file}")

    # Also save a combined (average) file for backwards compatibility
    if "LL" in all_weights and "DD" in all_weights:
        pt_bins = all_weights["LL"]["pt_bins"]
        w_avg = 0.5 * (
            np.array(all_weights["LL"]["weights"]) + np.array(all_weights["DD"]["weights"])
        )
        with open(output_dir / "kinematic_weights.json", "w") as f:
            json.dump({"pt_bins": pt_bins, "weights": w_avg.tolist()}, f, indent=2)
        print("\nSaved combined (averaged LL+DD) weights to output/kinematic_weights.json")


if __name__ == "__main__":
    main()
