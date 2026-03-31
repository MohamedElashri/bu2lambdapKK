import sys
from pathlib import Path

import awkward as ak
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from mva.config_loader import StudyConfig
from mva.data_preparation import load_and_prepare_data

from modules.data_handler import DataManager


def get_mc_efficiency_flow(category="LL"):
    print(f"Calculating Efficiency Flow for J/psi {category}...")

    config_path = Path("../../config/data.toml")
    data_manager = DataManager(config_path=config_path)

    years = ["2016", "2017", "2018"]

    total_gen = 0
    total_reco = 0
    total_trig = 0
    total_presel = 0
    total_mva = 0

    # Load the CatBoost model to apply the final cut
    from catboost import CatBoostClassifier

    model_path = Path(f"../output/models/catboost_bdt_{category}.cbm")
    model = CatBoostClassifier()
    model.load_model(str(model_path))

    import uproot

    for year in years:
        for magnet in ["MD", "MU"]:
            filepath = data_manager.mc_path / "Jpsi" / f"Jpsi_{int(year)-2000}_{magnet}.root"
            if not filepath.exists():
                continue

            tree_path = f"B2L0barPKpKm_{category}/DecayTree"
            try:
                with uproot.open(filepath) as f:
                    if tree_path not in f:
                        continue
                    tree = f[tree_path]

                    # Estimate generated
                    run_arr = tree["runNumber"].array()
                    evt_arr = tree["eventNumber"].array()
                    unique_evts = np.unique(
                        np.column_stack([ak.to_numpy(run_arr), ak.to_numpy(evt_arr)]), axis=0
                    )
                    n_gen = len(unique_evts)
                    total_gen += n_gen

                    n_reco = len(run_arr)
                    total_reco += n_reco

                    # Trigger
                    l0 = tree["Bu_L0Global_TIS"].array() | tree["Bu_L0HadronDecision_TOS"].array()
                    hlt1 = (
                        tree["Bu_Hlt1TrackMVADecision_TOS"].array()
                        | tree["Bu_Hlt1TwoTrackMVADecision_TOS"].array()
                    )
                    hlt2 = (
                        tree["Bu_Hlt2Topo2BodyDecision_TOS"].array()
                        | tree["Bu_Hlt2Topo3BodyDecision_TOS"].array()
                        | tree["Bu_Hlt2Topo4BodyDecision_TOS"].array()
                    )

                    trig_mask = l0 & hlt1 & hlt2
                    n_trig = np.sum(trig_mask)
                    total_trig += n_trig

                    # Pre-selection (as in data_preparation.py)
                    l0_m = tree["L0_M"].array()
                    l0_fd = tree["L0_FDCHI2_OWNPV"].array()
                    bu_pt = tree["Bu_PT"].array()
                    dz = tree["L0_Z"].array() - tree["Bu_ENDVERTEX_Z"].array()

                    presel_mask = (
                        trig_mask & (l0_m > 1111) & (l0_m < 1121) & (l0_fd > 50) & (bu_pt > 3000)
                    )
                    if category == "LL":
                        presel_mask = presel_mask & (dz > 20)
                    else:
                        presel_mask = presel_mask & (dz > 5)

                    n_presel = np.sum(presel_mask)
                    total_presel += n_presel

            except Exception as e:
                print(f"Error {filepath}: {e}")

    # To get MVA passes, we should just use the prepared ML data for exactness
    mva_config = StudyConfig("mva_config.toml")
    ml_data = load_and_prepare_data(mva_config, category)
    mc_combined = ml_data["mc_combined"]

    # Filter Jpsi
    is_jpsi = mc_combined["state"] == "jpsi"
    features = ml_data["features"]

    import pandas as pd

    df_dict = {}
    for feat in features:
        df_dict[feat] = ak.to_numpy(mc_combined[feat][is_jpsi])

    X_jpsi = pd.DataFrame(df_dict).values
    scores = model.predict_proba(X_jpsi)[:, 1]

    cut = 0.43 if category == "LL" else 0.42
    n_mva = np.sum(scores > cut)

    exact_presel = len(X_jpsi)

    print("--- Cumulative Yields ---")
    print(f"Generated: {total_gen}")
    print(f"Reconstructed: {total_reco}")
    print(f"Triggered: {total_trig}")
    print(f"Pre-selected (+PID): {exact_presel} (approx {total_presel} w/o full PID)")
    print(f"MVA (+{cut}): {n_mva}")

    # Efficiencies
    eff_reco = total_reco / total_gen * 100 if total_gen else 0
    eff_trig = total_trig / total_reco * 100 if total_reco else 0
    eff_presel = exact_presel / total_trig * 100 if total_trig else 0
    eff_mva = n_mva / exact_presel * 100 if exact_presel else 0
    eff_tot = n_mva / total_gen * 100 if total_gen else 0

    return [eff_reco, eff_trig, eff_presel, eff_mva, eff_tot]


if __name__ == "__main__":
    eff_ll = get_mc_efficiency_flow("LL")
    eff_dd = get_mc_efficiency_flow("DD")

    print("\n\nLaTeX Table:")
    print(r"\begin{table}[htb]")
    print(r"\centering")
    print(
        r"\caption{Sequential average efficiencies for the $J/\psi$ normalization channel evaluate on Run 2 MC.}"
    )
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Selection Level & \LLL & \LDD \\")
    print(r"\midrule")
    print(rf"Reconstruction \& Stripping & {eff_ll[0]:.1f}\\% & {eff_dd[0]:.1f}\\% \\\\")
    print(f"Trigger (L0+HLT1+HLT2)      & {eff_ll[1]:.1f}\\% & {eff_dd[1]:.1f}\\% \\\\")
    print(f"Pre-selection + PID ($>0.2$)& {eff_ll[2]:.1f}\\% & {eff_dd[2]:.1f}\\% \\\\")
    print(f"MVA Classifier              & {eff_ll[3]:.1f}\\% & {eff_dd[3]:.1f}\\% \\\\")
    print(r"\midrule")
    print(
        f"Total                       & \textbf{{{eff_ll[4]:.2f}\\%}} & \textbf{{{eff_dd[4]:.2f}\\%}} \\\\"
    )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
