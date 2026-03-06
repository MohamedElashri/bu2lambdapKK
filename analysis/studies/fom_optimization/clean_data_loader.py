import logging
from pathlib import Path
from typing import Dict

import awkward as ak
import numpy as np
import uproot
import vector

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

M_LAMBDA_PDG = 1115.683


def load_and_preprocess(filepath: Path, is_mc: bool, track_type: str = "LL") -> ak.Array:
    logger.info(f"Loading {filepath.name}...")
    tree_path = f"B2L0barPKpKm_{track_type}/DecayTree"
    with uproot.open(filepath) as f:
        tree = f[tree_path]

        # Determine exact branch names natively in the file
        all_branches = tree.keys()

        branches_to_load = [
            "Bu_MM",
            "L0_MM",
            "L0_ENDVERTEX_Z",
            "Bu_ENDVERTEX_Z",
            "Bu_FDCHI2_OWNPV",
            "Bu_IPCHI2_OWNPV",
            "Bu_PT",
            "L0_PX",
            "L0_PY",
            "L0_PZ",
            "L0_PE",
            "p_PX",
            "p_PY",
            "p_PZ",
            "p_PE",
            "h1_PX",
            "h1_PY",
            "h1_PZ",
            "h1_PE",
            "h2_PX",
            "h2_PY",
            "h2_PZ",
            "h2_PE",
            "L0_FDCHI2_OWNPV",
        ]

        # Add DTF Chi2
        dtf_chi2_k = "Bu_DTF_chi2" if "Bu_DTF_chi2" in all_branches else "Bu_DTF_CHI2"
        branches_to_load.append(dtf_chi2_k)

        # Add PID
        lp_p_k = "Lp_MC12TuneV4_ProbNNp" if is_mc else "Lp_MC15TuneV1_ProbNNp"
        p_p_k = "p_MC12TuneV4_ProbNNp" if is_mc else "p_MC15TuneV1_ProbNNp"
        h1_k_k = "h1_MC12TuneV4_ProbNNk" if is_mc else "h1_MC15TuneV1_ProbNNk"
        h2_k_k = "h2_MC12TuneV4_ProbNNk" if is_mc else "h2_MC15TuneV1_ProbNNk"

        branches_to_load.extend([lp_p_k, p_p_k, h1_k_k, h2_k_k])

        # Add trigger
        if is_mc:
            l0_branches = [
                "Bu_L0Global_TIS",
                "Bu_L0HadronDecision_TIS",
                "Bu_L0MuonDecision_TIS",
                "Bu_L0DiMuonDecision_TIS",
                "Bu_L0PhotonDecision_TIS",
                "Bu_L0ElectronDecision_TIS",
            ]
        else:
            l0_branches = [
                "Bu_L0GlobalDecision_TIS",
                "Bu_L0PhysDecision_TIS",
                "Bu_L0HadronDecision_TIS",
                "Bu_L0MuonDecision_TIS",
                "Bu_L0MuonHighDecision_TIS",
                "Bu_L0DiMuonDecision_TIS",
                "Bu_L0PhotonDecision_TIS",
                "Bu_L0ElectronDecision_TIS",
            ]

        hlt1_branches = ["Bu_Hlt1TrackMVADecision_TOS", "Bu_Hlt1TwoTrackMVADecision_TOS"]
        hlt2_branches = [
            "Bu_Hlt2Topo2BodyDecision_TOS",
            "Bu_Hlt2Topo3BodyDecision_TOS",
            "Bu_Hlt2Topo4BodyDecision_TOS",
        ]

        l0_avail = [b for b in l0_branches if b in all_branches]
        hlt1_avail = [b for b in hlt1_branches if b in all_branches]
        hlt2_avail = [b for b in hlt2_branches if b in all_branches]

        branches_to_load.extend(l0_avail + hlt1_avail + hlt2_avail)

        # Check missing
        missing = [b for b in branches_to_load if b not in all_branches]
        if missing:
            logger.warning(f"Missing branches in {filepath}: {missing}")
            branches_to_load = [b for b in branches_to_load if b in all_branches]

        events = tree.arrays(branches_to_load, library="ak")

    # Standardize branch names for downstream
    if dtf_chi2_k != "Bu_DTF_chi2":
        events["Bu_DTF_chi2"] = events[dtf_chi2_k]

    # In MC, Bu_DTF_chi2 can be a jagged array with multiple fits per event.
    # We take the first fit result (the primary one) to flatten the array.
    if "var" in str(ak.type(events["Bu_DTF_chi2"])):
        events["Bu_DTF_chi2"] = events["Bu_DTF_chi2"][:, 0]

    events["Lp_ProbNNp"] = events[lp_p_k]
    events["p_ProbNNp"] = events[p_p_k]
    events["h1_ProbNNk"] = events[h1_k_k]
    events["h2_ProbNNk"] = events[h2_k_k]

    # Derivations
    events["Bu_MM_corrected"] = events["Bu_MM"] - events["L0_MM"] + M_LAMBDA_PDG
    # Absolute value of Delta_Z is what the pre-selection uses!
    events["Delta_Z_mm"] = np.abs(events["L0_ENDVERTEX_Z"] - events["Bu_ENDVERTEX_Z"])
    events["prodProbKK"] = events["h1_ProbNNk"] * events["h2_ProbNNk"]
    events["PID_product"] = events["p_ProbNNp"] * events["h1_ProbNNk"] * events["h2_ProbNNk"]

    # 4-momentum calculation for M_LpKm_h2
    # Register vector to awkward behavior
    vector.register_awkward()

    L0_vec = vector.zip(
        {"px": events["L0_PX"], "py": events["L0_PY"], "pz": events["L0_PZ"], "E": events["L0_PE"]}
    )
    p_vec = vector.zip(
        {"px": events["p_PX"], "py": events["p_PY"], "pz": events["p_PZ"], "E": events["p_PE"]}
    )
    h2_vec = vector.zip(
        {"px": events["h2_PX"], "py": events["h2_PY"], "pz": events["h2_PZ"], "E": events["h2_PE"]}
    )

    events["M_LpKm_h2"] = (L0_vec + p_vec + h2_vec).mass

    n_total = len(events)

    # 1. Triggers
    mask_l0 = ak.zeros_like(events["Bu_MM"], dtype=bool)
    for b in l0_avail:
        mask_l0 = mask_l0 | (events[b] > 0)

    mask_hlt1 = ak.zeros_like(events["Bu_MM"], dtype=bool)
    for b in hlt1_avail:
        mask_hlt1 = mask_hlt1 | (events[b] > 0)

    mask_hlt2 = ak.zeros_like(events["Bu_MM"], dtype=bool)
    for b in hlt2_avail:
        mask_hlt2 = mask_hlt2 | (events[b] > 0)

    # If no trigger branches found, bypass that level to avoid 100% rejection.
    if len(l0_avail) == 0:
        mask_l0 = ak.ones_like(events["Bu_MM"], dtype=bool)
    if len(hlt1_avail) == 0:
        mask_hlt1 = ak.ones_like(events["Bu_MM"], dtype=bool)
    if len(hlt2_avail) == 0:
        mask_hlt2 = ak.ones_like(events["Bu_MM"], dtype=bool)

    trigger_pass = mask_l0 & mask_hlt1 & mask_hlt2
    events = events[trigger_pass]
    logger.info(
        f"Trigger {filepath.name}: {n_total} -> {len(events)} ({100*len(events)/n_total if n_total>0 else 0:.1f}%)"
    )

    # 2. Data reduction
    n_before = len(events)
    mask_red = (
        (events["Bu_FDCHI2_OWNPV"] > 175)
        & (events["Delta_Z_mm"] > 2.5)
        & (events["Lp_ProbNNp"] > 0.05)
        & (events["p_ProbNNp"] > 0.05)
        & (events["prodProbKK"] > 0.10)
        & (events["Bu_PT"] > 3000)
        & (events["Bu_DTF_chi2"] < 30)
    )
    events = events[mask_red]
    logger.info(
        f"Data Reduction {filepath.name}: {n_before} -> {len(events)} ({100*len(events)/n_before if n_before>0 else 0:.1f}%)"
    )

    # 3. Final Analysis Baseline
    n_before = len(events)

    mask_base = (
        (events["L0_MM"] > 1111)
        & (events["L0_MM"] < 1121)
        & (events["L0_FDCHI2_OWNPV"] > 250)
        & (events["Delta_Z_mm"] > 5)
        & (events["Lp_ProbNNp"] > 0.3)
    )

    events = events[mask_base]
    logger.info(
        f"Analysis Baseline {filepath.name}: {n_before} -> {len(events)} ({100*len(events)/n_before if n_before>0 else 0:.1f}%)"
    )

    return events


def load_all_data(
    base_data_path: Path, years: list, track_types: list = ["LL", "DD"]
) -> Dict[str, ak.Array]:
    data_dict = {}
    for year in years:
        arrs = []
        for track_type in track_types:
            for magnet in ["MD", "MU"]:
                fname = f"dataBu2L0barPHH_{int(year)-2000}{magnet}.root"
                fpath = base_data_path / fname
                if fpath.exists():
                    arr = load_and_preprocess(fpath, is_mc=False, track_type=track_type)
                    arrs.append(arr)
        if arrs:
            data_dict[str(year)] = ak.concatenate(arrs)

    # Apply B+ mass window (including sidebands) to real data
    for year in data_dict:
        n_before = len(data_dict[year])
        mask_bmass = (data_dict[year]["Bu_MM_corrected"] > 5150) & (
            data_dict[year]["Bu_MM_corrected"] < 5410
        )
        data_dict[year] = data_dict[year][mask_bmass]
        logger.info(
            f"Data {year} B+ Mass Window: {n_before} -> {len(data_dict[year])} ({100*len(data_dict[year])/n_before if n_before>0 else 0:.1f}%)"
        )

    return data_dict


def load_all_mc(
    base_mc_path: Path, states: list, years: list, track_types: list = ["LL", "DD"]
) -> Dict[str, ak.Array]:
    mc_dict = {}
    for state in states:
        arrs = []
        for year in years:
            for track_type in track_types:
                for magnet in ["MD", "MU"]:
                    fname = f"{state}_{int(year)-2000}_{magnet}.root"
                    fpath = base_mc_path / state / fname
                    if fpath.exists():
                        arr = load_and_preprocess(fpath, is_mc=True, track_type=track_type)
                        arrs.append(arr)
                    else:
                        # Also try Jpsi instead of jpsi string if user config has different casing
                        case_fname = f"{state.capitalize()}_{int(year)-2000}_{magnet}.root"
                        case_fpath = base_mc_path / state.capitalize() / case_fname
                        if case_fpath.exists():
                            arr = load_and_preprocess(case_fpath, is_mc=True, track_type=track_type)
                            arrs.append(arr)
        if arrs:
            state_data = ak.concatenate(arrs)
            # Apply B+ mass window (including sidebands) to MC
            n_before = len(state_data)
            mask_bmass = (state_data["Bu_MM_corrected"] > 5150) & (
                state_data["Bu_MM_corrected"] < 5410
            )
            state_data = state_data[mask_bmass]
            logger.info(
                f"MC {state} B+ Mass Window: {n_before} -> {len(state_data)} ({100*len(state_data)/n_before if n_before>0 else 0:.1f}%)"
            )
            # Always use lowercase state key format across pipeline
            mc_dict[state.lower()] = state_data
    return mc_dict
