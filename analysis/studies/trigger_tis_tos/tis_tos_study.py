"""
Data-driven Trigger Efficiency via TIS/TOS overlap method.

Calculates the trigger efficiency for the TOS decision on data and MC,
allowing the derivation of a correction factor.

The method uses:
eps_TOS = N_TOS_and_TIS / N_TIS

For real data, we must subtract background using the B mass sidebands or sPlot.
Here we use sideband subtraction as a simpler, robust method for counting.
"""

import argparse
import json

import awkward as ak
import numpy as np
import uproot


def apply_lambda_selection(tree, category):
    """Apply standard lambda pre-selection (Phase 0 values).

    Changes from original:
    - Lambda FD chi2 lowered: 250 → 50 (allows optimizer to scan the range freely).
    - Delta_Z is now category-aware: LL uses 20 mm, DD uses 5 mm.
    - PID_product > 0.20 added as fixed pre-cut (validated by fit_based_optimizer study).
    """
    l0_mass = tree["L0_M"].array()
    l0_fdchi2 = tree["L0_FDCHI2_OWNPV"].array()
    l0_end_z = tree["L0_ENDVERTEX_Z"].array()
    bu_end_z = tree["Bu_ENDVERTEX_Z"].array()
    lp_probnnp = tree["Lp_ProbNNp"].array()

    # Track types
    p_track = tree["p_TRACK_Type"].array()
    h1_track = tree["h1_TRACK_Type"].array()
    h2_track = tree["h2_TRACK_Type"].array()
    lp_track = tree["Lp_TRACK_Type"].array()
    lpi_track = tree["Lpi_TRACK_Type"].array()

    expected_lambda_track_type = 3 if category == "LL" else 5

    # Category-aware Delta_Z cut (Phase 0)
    delta_z_cut = 20.0 if category == "LL" else 5.0

    # PID variables for fixed pre-cut
    p_probnnp = tree["p_ProbNNp"].array() if "p_ProbNNp" in tree else ak.ones_like(l0_mass)
    h1_probnnk = tree["h1_ProbNNk"].array() if "h1_ProbNNk" in tree else ak.ones_like(l0_mass)
    h2_probnnk = tree["h2_ProbNNk"].array() if "h2_ProbNNk" in tree else ak.ones_like(l0_mass)

    mask = (
        (l0_mass > 1111.0)
        & (l0_mass < 1121.0)
        & (l0_fdchi2 > 50.0)  # Phase 0: was 250
        & ((l0_end_z - bu_end_z) > delta_z_cut)  # Phase 0: category-aware
        & (lp_probnnp > 0.3)
        & ((p_probnnp * h1_probnnk * h2_probnnk) > 0.20)  # Phase 0: fixed PID pre-cut
        & (p_track == 3)
        & (h1_track == 3)
        & (h2_track == 3)
        & (lp_track == expected_lambda_track_type)
        & (lpi_track == expected_lambda_track_type)
    )
    return mask


def get_trigger_masks(tree):
    """Returns TIS and TOS masks with fallback branch names."""

    def get_array(names):
        for n in names:
            if n in tree:
                return tree[n].array()
        print(f"Warning: None of {names} found in tree!")
        return ak.zeros_like(tree["Bu_MM"].array(), dtype=bool)

    l0_global_tis = get_array(["Bu_L0GlobalDecision_TIS", "Bu_L0Global_TIS"])
    l0_hadron_tis = get_array(["Bu_L0HadronDecision_TIS", "Bu_L0Hadron_TIS"])
    l0_muon_tis = get_array(["Bu_L0MuonDecision_TIS", "Bu_L0Muon_TIS"])
    l0_dimuon_tis = get_array(["Bu_L0DiMuonDecision_TIS", "Bu_L0DiMuon_TIS"])
    l0_photon_tis = get_array(["Bu_L0PhotonDecision_TIS", "Bu_L0Photon_TIS"])
    l0_electron_tis = get_array(["Bu_L0ElectronDecision_TIS", "Bu_L0Electron_TIS"])

    l0_tis = (
        l0_global_tis
        | l0_hadron_tis
        | l0_muon_tis
        | l0_dimuon_tis
        | l0_photon_tis
        | l0_electron_tis
    )

    hlt1_track_tos = get_array(
        ["Bu_Hlt1TrackMVADecision_TOS", "Bu_Hlt1TrackMVA_TOS", "Bu_Hlt1TrackAllL0Decision_TOS"]
    )
    hlt1_two_track_tos = get_array(["Bu_Hlt1TwoTrackMVADecision_TOS", "Bu_Hlt1TwoTrackMVA_TOS"])

    hlt1_tos = hlt1_track_tos | hlt1_two_track_tos

    hlt2_topo2_tos = get_array(["Bu_Hlt2Topo2BodyDecision_TOS", "Bu_Hlt2Topo2Body_TOS"])
    hlt2_topo3_tos = get_array(["Bu_Hlt2Topo3BodyDecision_TOS", "Bu_Hlt2Topo3Body_TOS"])
    hlt2_topo4_tos = get_array(["Bu_Hlt2Topo4BodyDecision_TOS", "Bu_Hlt2Topo4Body_TOS"])

    hlt2_tos = hlt2_topo2_tos | hlt2_topo3_tos | hlt2_topo4_tos

    return l0_tis, hlt1_tos, hlt2_tos


def sideband_subtraction(mass_array, mask):
    """
    Perform sideband subtraction to estimate signal yield.
    Assuming J/psi mass window for normalization.
    Returns estimated signal yield.
    """
    masses = mass_array[mask]

    # B mass regions (approximate, should match optimization strategy)
    signal_min, signal_max = 5255.0, 5305.0
    low_sb_min, low_sb_max = 5150.0, 5230.0
    high_sb_min, high_sb_max = 5330.0, 5410.0

    n_sig_region = ak.sum((masses >= signal_min) & (masses <= signal_max))
    n_low_sb = ak.sum((masses >= low_sb_min) & (masses <= low_sb_max))
    n_high_sb = ak.sum((masses >= high_sb_min) & (masses <= high_sb_max))

    # Scale factor for sidebands
    width_sig = signal_max - signal_min
    width_sb = (low_sb_max - low_sb_min) + (high_sb_max - high_sb_min)
    scale = width_sig / width_sb

    n_bkg_est = (n_low_sb + n_high_sb) * scale
    n_sig_est = n_sig_region - n_bkg_est

    # Simple error propagation
    err_sig_region = np.sqrt(n_sig_region)
    err_bkg_est = scale * np.sqrt(n_low_sb + n_high_sb)
    err_sig_est = np.sqrt(err_sig_region**2 + err_bkg_est**2)

    return float(n_sig_est), float(err_sig_est)


def analyze_file(file_path: str, is_data: bool, category: str = "LL"):
    """Analyze a single file for TIS/TOS overlap."""
    tree_name = f"B2L0barPKpKm_{category}/DecayTree"
    try:
        f = uproot.open(file_path)
        if tree_name not in f:
            return None
        tree = f[tree_name]
    except Exception as e:
        print(f"Error opening {file_path}: {e}")
        return None

    if tree.num_entries == 0:
        return None

    base_mask = apply_lambda_selection(tree, category)

    if not is_data:
        # For MC, require truth matching to J/psi
        truth_mask = (
            (np.abs(tree["Bu_TRUEID"].array()) == 521)
            & (np.abs(tree["p_TRUEID"].array()) == 2212)
            & (np.abs(tree["h1_TRUEID"].array()) == 321)
            & (np.abs(tree["h2_TRUEID"].array()) == 321)
            & (np.abs(tree["L0_TRUEID"].array()) == 3122)
        )
        base_mask = base_mask & truth_mask

    # We want to measure the efficiency of HLT (TOS) given L0 (TIS)
    # eps_HLT = N(L0_TIS and HLT_TOS) / N(L0_TIS)

    l0_tis, hlt1_tos, hlt2_tos = get_trigger_masks(tree)
    hlt_tos = hlt1_tos & hlt2_tos

    mask_tis = base_mask & l0_tis
    mask_tis_tos = base_mask & l0_tis & hlt_tos

    if is_data:
        b_mass = tree["Bu_MM"].array()
        n_tis, err_tis = sideband_subtraction(b_mass, mask_tis)
        n_tis_tos, err_tis_tos = sideband_subtraction(b_mass, mask_tis_tos)
    else:
        n_tis = float(ak.sum(mask_tis))
        err_tis = np.sqrt(n_tis)
        n_tis_tos = float(ak.sum(mask_tis_tos))
        err_tis_tos = np.sqrt(n_tis_tos)

    return n_tis, err_tis, n_tis_tos, err_tis_tos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="../../config")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    years = ["16", "17", "18"]
    polarities = ["MD", "MU"]
    categories = ["LL", "DD"]

    mc_base = "/share/lazy/Mohamed/Bu2LambdaPPP/files/mc/Jpsi"
    data_base = "/share/lazy/Mohamed/Bu2LambdaPPP/rootfiles/reduced"

    results = {"data": {}, "mc": {}}

    for category in categories:
        results["data"][category] = {}
        results["mc"][category] = {}

        for year in years:
            # Data
            n_tis_data_tot = 0
            n_tis_tos_data_tot = 0
            err_tis_data_sq = 0
            err_tis_tos_data_sq = 0

            # MC
            n_tis_mc_tot = 0
            n_tis_tos_mc_tot = 0

            for pol in polarities:
                # Data file
                data_file = f"{data_base}/dataBu2L0barPHH_{year}{pol}_reduced_PID.root"
                res_data = analyze_file(data_file, is_data=True, category=category)
                if res_data:
                    n_tis_data_tot += res_data[0]
                    err_tis_data_sq += res_data[1] ** 2
                    n_tis_tos_data_tot += res_data[2]
                    err_tis_tos_data_sq += res_data[3] ** 2

                # MC file
                mc_file = f"{mc_base}/Jpsi_{year}_{pol}.root"
                res_mc = analyze_file(mc_file, is_data=False, category=category)
                if res_mc:
                    n_tis_mc_tot += res_mc[0]
                    n_tis_tos_mc_tot += res_mc[2]

            # Compute efficiencies
            def calc_eff(n_tot, n_pass, err_tot_sq=None, err_pass_sq=None):
                if n_tot <= 0:
                    return 0.0, 0.0
                eff = n_pass / n_tot
                if err_tot_sq is None:
                    # Binomial
                    err = np.sqrt(eff * (1 - eff) / n_tot)
                else:
                    # Error propagation for ratio A/B with sideband subtraction
                    # Var(A/B) approx (A/B)^2 * (Var(A)/A^2 + Var(B)/B^2 - 2Cov(A,B)/(AB))
                    # Simplified: Assume fully correlated since A is subset of B
                    # For sideband sub, it's complex. Let's use binomial approx on the raw yields.
                    err = np.sqrt(eff * (1 - eff) / max(1, n_tot))
                return eff, err

            eff_data, err_data = calc_eff(
                n_tis_data_tot, n_tis_tos_data_tot, err_tis_data_sq, err_tis_tos_data_sq
            )
            eff_mc, err_mc = calc_eff(n_tis_mc_tot, n_tis_tos_mc_tot)

            correction = eff_data / eff_mc if eff_mc > 0 else 0
            err_corr = (
                correction * np.sqrt((err_data / eff_data) ** 2 + (err_mc / eff_mc) ** 2)
                if eff_data > 0 and eff_mc > 0
                else 0
            )

            results["data"][category][year] = {"eff": eff_data, "err": err_data}
            results["mc"][category][year] = {"eff": eff_mc, "err": err_mc}

            results["data"][category][year] = {"eff": eff_data, "err": err_data}
            results["mc"][category][year] = {"eff": eff_mc, "err": err_mc}
            results.setdefault("correction", {}).setdefault(category, {})[year] = {
                "value": correction,
                "err": err_corr,
            }

            print(f"[{category} 20{year}]")
            print(
                f"  Data: {n_tis_tos_data_tot:.1f} / {n_tis_data_tot:.1f}"
                f" = {eff_data*100:.2f} ± {err_data*100:.2f}%"
            )
            print(
                f"  MC  : {n_tis_tos_mc_tot:.1f} / {n_tis_mc_tot:.1f}"
                f" = {eff_mc*100:.2f} ± {err_mc*100:.2f}%"
            )
            print(f"  Correction Factor: {correction:.3f} ± {err_corr:.3f}\n")

    import os

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = f"{args.output_dir}/tis_tos_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
