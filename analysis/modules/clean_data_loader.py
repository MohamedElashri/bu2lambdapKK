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


def load_and_preprocess(
    filepath: Path,
    is_mc: bool,
    track_type: str = "LL",
    delta_z_cut: float = 5.0,
    lambda_fdchi2_cut: float = 50.0,
) -> ak.Array:
    """Load and preprocess a single ROOT file for one track type.

    Args:
        filepath: Path to the ROOT file.
        is_mc: True for MC, False for real data (affects PID and trigger branch names).
        track_type: "LL" or "DD", selects the correct tree path in the ROOT file.
        delta_z_cut: Minimum |ΔZ| cut in mm (category-dependent: LL uses 20, DD uses 5).
        lambda_fdchi2_cut: Minimum Lambda FD chi2 (default 50; ND scan finds optimal above this).
    """
    logger.info(f"Loading {filepath.name} [{track_type}]...")
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

        # Add event ID branches for multiple candidate detection (optional)
        for ev_branch in ["runNumber", "eventNumber"]:
            if ev_branch in all_branches:
                branches_to_load.append(ev_branch)

        # Add PID — MC15TuneV1 used for both data and MC.
        lp_p_k = "Lp_MC15TuneV1_ProbNNp"
        p_p_k = "p_MC15TuneV1_ProbNNp"
        h1_k_k = "h1_MC15TuneV1_ProbNNk"
        h2_k_k = "h2_MC15TuneV1_ProbNNk"

        branches_to_load.extend([lp_p_k, p_p_k, h1_k_k, h2_k_k])

        # Add trigger branches (different names in data vs MC ntuples)
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

        # MC truth-matching branches (absent in real data; needed for S_true diagnostics
        # in the MVA study cut_optimizer.py)
        if is_mc:
            trueid_branches = ["Bu_TRUEID", "p_TRUEID", "h1_TRUEID", "h2_TRUEID"]
            branches_to_load.extend([b for b in trueid_branches if b in all_branches])

        # Filter to branches that actually exist
        missing = [b for b in branches_to_load if b not in all_branches]
        if missing:
            logger.warning(f"Missing branches in {filepath}: {missing}")
            branches_to_load = [b for b in branches_to_load if b in all_branches]

        events = tree.arrays(branches_to_load, library="ak")

    # Standardize branch names for downstream
    if dtf_chi2_k != "Bu_DTF_chi2":
        events["Bu_DTF_chi2"] = events[dtf_chi2_k]

    # In MC, Bu_DTF_chi2 can be a jagged array with multiple fits per event.
    # Take the first fit result (the primary one) to flatten.
    if "var" in str(ak.type(events["Bu_DTF_chi2"])):
        events["Bu_DTF_chi2"] = events["Bu_DTF_chi2"][:, 0]

    events["Lp_ProbNNp"] = events[lp_p_k]
    events["p_ProbNNp"] = events[p_p_k]
    events["h1_ProbNNk"] = events[h1_k_k]
    events["h2_ProbNNk"] = events[h2_k_k]

    # Derived branches
    events["Bu_MM_corrected"] = events["Bu_MM"] - events["L0_MM"] + M_LAMBDA_PDG
    events["Delta_Z_mm"] = np.abs(events["L0_ENDVERTEX_Z"] - events["Bu_ENDVERTEX_Z"])
    events["prodProbKK"] = events["h1_ProbNNk"] * events["h2_ProbNNk"]
    events["PID_product"] = events["p_ProbNNp"] * events["h1_ProbNNk"] * events["h2_ProbNNk"]
    # log(IP chi2): more uniform distribution in sensitive region; used in ND scan
    events["log_Bu_IPCHI2"] = np.log(events["Bu_IPCHI2_OWNPV"])

    # 4-momentum: invariant mass of (Lambda + bachelor p + h2) system
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

    # ---- 1. Trigger (L0 TIS AND HLT1 TOS AND HLT2 TOS) ----
    mask_l0 = ak.zeros_like(events["Bu_MM"], dtype=bool)
    for b in l0_avail:
        mask_l0 = mask_l0 | (events[b] > 0)

    mask_hlt1 = ak.zeros_like(events["Bu_MM"], dtype=bool)
    for b in hlt1_avail:
        mask_hlt1 = mask_hlt1 | (events[b] > 0)

    mask_hlt2 = ak.zeros_like(events["Bu_MM"], dtype=bool)
    for b in hlt2_avail:
        mask_hlt2 = mask_hlt2 | (events[b] > 0)

    # If no trigger branches found, bypass that level to avoid 100% rejection
    if len(l0_avail) == 0:
        mask_l0 = ak.ones_like(events["Bu_MM"], dtype=bool)
    if len(hlt1_avail) == 0:
        mask_hlt1 = ak.ones_like(events["Bu_MM"], dtype=bool)
    if len(hlt2_avail) == 0:
        mask_hlt2 = ak.ones_like(events["Bu_MM"], dtype=bool)

    trigger_pass = mask_l0 & mask_hlt1 & mask_hlt2
    events = events[trigger_pass]
    logger.info(
        f"  Trigger [{track_type}] {filepath.name}: {n_total} -> {len(events)} "
        f"({100*len(events)/n_total if n_total>0 else 0:.1f}%)"
    )

    # ---- 2. Data reduction baseline (mirrors stripping-level cuts applied to ntuples) ----
    n_before = len(events)

    if "Bu_IPCHI2_OWNPV" in events.fields:
        ipchi2_mask = events["Bu_IPCHI2_OWNPV"] < 10.0
    else:
        ipchi2_mask = ak.ones_like(events["Bu_MM"], dtype=bool)

    mask_red = (
        (events["Bu_FDCHI2_OWNPV"] > 175)
        & ipchi2_mask
        & (events["Delta_Z_mm"] > 2.5)
        & (events["Lp_ProbNNp"] > 0.05)
        & (events["p_ProbNNp"] > 0.05)
        & (events["prodProbKK"] > 0.05)
        & (events["Bu_PT"] > 3000)
    )
    events = events[mask_red]
    logger.info(
        f"  Data Reduction [{track_type}] {filepath.name}: {n_before} -> {len(events)} "
        f"({100*len(events)/n_before if n_before>0 else 0:.1f}%)"
    )

    # ---- 3. Lambda pre-selection (category-aware) ----
    # - Lambda mass window: same for both LL and DD
    # - Lambda FD chi2: soft cut (50); ND scanner will find the optimal value
    # - Delta_Z: category-dependent (LL uses tight 20 mm cut per reference analysis;
    #   DD relies on FD chi2 and the 5 mm cut is sufficient)
    # - PID fixed pre-cut at 0.20 (validated by fit_based_optimizer study;
    #   proxy-based optimization is unreliable for PID, see studies/pid_proxy_comparison)
    n_before = len(events)

    mask_lambda = (
        (events["L0_MM"] > 1111)
        & (events["L0_MM"] < 1121)
        & (events["L0_FDCHI2_OWNPV"] > lambda_fdchi2_cut)
        & (events["Delta_Z_mm"] > delta_z_cut)
        & (events["Lp_ProbNNp"] > 0.3)
        & (events["PID_product"] > 0.20)  # fixed pre-cut (see [fixed_selection] in selection.toml)
    )

    events = events[mask_lambda]
    logger.info(
        f"  Lambda pre-sel [{track_type}] {filepath.name}: {n_before} -> {len(events)} "
        f"({100*len(events)/n_before if n_before>0 else 0:.1f}%)"
    )

    return events


def load_all_data(
    base_data_path: Path,
    years: list,
    track_types: list = ["LL", "DD"],
) -> Dict[str, Dict[str, ak.Array]]:
    """Load real data, returning a nested dict: {year: {category: array}}.

    LL and DD are kept separate throughout the pipeline so that per-category
    optimization and efficiency calculations can be performed independently.
    They are combined only at the mass fit / branching ratio stage.
    """
    # Category-dependent Delta_Z cuts (config/selection.toml [lambda_selection])
    delta_z_cuts = {"LL": 20.0, "DD": 5.0}

    data_dict: Dict[str, Dict[str, ak.Array]] = {}
    for year in years:
        data_dict[str(year)] = {}
        for track_type in track_types:
            arrs = []
            for magnet in ["MD", "MU"]:
                fname = f"dataBu2L0barPHH_{int(year)-2000}{magnet}.root"
                fpath = base_data_path / fname
                if fpath.exists():
                    arr = load_and_preprocess(
                        fpath,
                        is_mc=False,
                        track_type=track_type,
                        delta_z_cut=delta_z_cuts[track_type],
                    )
                    arrs.append(arr)
            if arrs:
                data_dict[str(year)][track_type] = ak.concatenate(arrs)
            else:
                data_dict[str(year)][track_type] = ak.Array([])

    # Apply B+ mass window (including sidebands for sideband subtraction) per category
    for year in data_dict:
        for cat in data_dict[year]:
            arr = data_dict[year][cat]
            if len(arr) == 0:
                continue
            n_before = len(arr)
            mask_bmass = (arr["Bu_MM_corrected"] > 5150) & (arr["Bu_MM_corrected"] < 5410)
            data_dict[year][cat] = arr[mask_bmass]
            logger.info(
                f"Data {year} [{cat}] B+ mass window: {n_before} -> {len(data_dict[year][cat])} "
                f"({100*len(data_dict[year][cat])/n_before if n_before>0 else 0:.1f}%)"
            )

    return data_dict


def load_all_mc(
    base_mc_path: Path,
    states: list,
    years: list,
    track_types: list = ["LL", "DD"],
) -> Dict[str, Dict[str, ak.Array]]:
    """Load signal MC, returning a nested dict: {state: {category: array}}.

    Years and magnet polarities are concatenated within each (state, category) cell.
    LL and DD are kept separate so that per-category efficiency calculations and
    optimization can be performed independently.
    """
    delta_z_cuts = {"LL": 20.0, "DD": 5.0}

    mc_dict: Dict[str, Dict[str, ak.Array]] = {}
    for state in states:
        mc_dict[state.lower()] = {}
        for track_type in track_types:
            arrs = []
            for year in years:
                for magnet in ["MD", "MU"]:
                    fname = f"{state}_{int(year)-2000}_{magnet}.root"
                    fpath = base_mc_path / state / fname
                    if not fpath.exists():
                        # Try capitalised directory/filename (e.g. Jpsi/Jpsi_16_MD.root)
                        case_fname = f"{state.capitalize()}_{int(year)-2000}_{magnet}.root"
                        fpath = base_mc_path / state.capitalize() / case_fname
                    if fpath.exists():
                        arr = load_and_preprocess(
                            fpath,
                            is_mc=True,
                            track_type=track_type,
                            delta_z_cut=delta_z_cuts[track_type],
                        )
                        arrs.append(arr)
            if arrs:
                state_cat = ak.concatenate(arrs)
                # Apply B+ mass window (including sidebands)
                n_before = len(state_cat)
                mask_bmass = (state_cat["Bu_MM_corrected"] > 5150) & (
                    state_cat["Bu_MM_corrected"] < 5410
                )
                state_cat = state_cat[mask_bmass]
                logger.info(
                    f"MC {state} [{track_type}] B+ mass window: {n_before} -> {len(state_cat)} "
                    f"({100*len(state_cat)/n_before if n_before>0 else 0:.1f}%)"
                )
                mc_dict[state.lower()][track_type] = state_cat
            else:
                mc_dict[state.lower()][track_type] = ak.Array([])

    return mc_dict
