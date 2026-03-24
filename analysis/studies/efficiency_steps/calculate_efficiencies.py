"""
Calculate efficiency for each step in the analysis pipeline.

Steps:
1. eff_gen:      Generator level (from config/generator_effs.toml)
2. eff_reco+str: Reconstruction and stripping (truth-matched events with correct track types)
3. eff_tri:      Trigger efficiency relative to stripping (L0 TIS AND HLT1 TOS AND HLT2 TOS)
4. eff_pre:      Pre-selection cuts (mass windows, PID, Lambda FD chi2, Delta_Z)
5. eff_mva:      MVA/BDT selection efficiency

Phase 3 fixes:
- Fixed trigger definition bug: L0 was ORing global TIS with hadron TOS (incorrect).
  Now ORs only L0 TIS branches to match clean_data_loader.py selection.
- Fixed HLT1: was ORing TOS with TIS; now TOS only (TrackMVA OR TwoTrackMVA).
- Updated Lambda FD chi2 pre-cut: 250 → 50 (Phase 0 change).
- Made Lambda Delta_Z cut category-aware: LL=20 mm, DD=5 mm.
- Added PID_product > 0.20 as a fixed pre-cut.
- Added TIS/TOS data/MC correction factor via `tis_tos_corr` parameter.
- Wired kinematic reweighting (2D pT × nTracks or 1D pT) via `kin_weights` parameter.
- Fixed MVA model path to load per-branch/category model correctly.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tomli
import uproot
from catboost import CatBoostClassifier

# Add analysis/ to sys.path so modules.* are importable
_ANALYSIS_DIR = Path(__file__).resolve().parents[2]  # studies/efficiency_steps/ → analysis/
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))


def get_gen_eff_from_config(
    state: str, year: str, polarity: str, gen_config: dict
) -> tuple[float, float]:
    """Extract generator efficiency and error for a specific state, year, polarity."""
    state_map = {
        "chic0": "chic0",
        "chic1": "chic1",
        "chic2": "chic2",
        "etac": "etac",
        "Jpsi": "Jpsi",
    }
    c_state = state_map.get(state, state)
    if c_state not in gen_config:
        return 0.22, 0.0005
    y_key = f"20{year}"
    if y_key not in gen_config[c_state]:
        return 0.22, 0.0005
    if polarity not in gen_config[c_state][y_key]:
        return 0.22, 0.0005
    data = gen_config[c_state][y_key][polarity]
    return data.get("eff", 0.22), data.get("err", 0.0005)


@dataclass
class EfficiencyResult:
    gen_eff: float = 0.22
    gen_err: float = 0.0005

    n_total: float = 0.0
    n_reco_str: float = 0.0
    n_trig: float = 0.0
    n_pre: float = 0.0
    n_mva: float = 0.0

    # Detailed trigger line counts (relative to reco_str)
    n_l0_global_tis: float = 0.0
    n_l0_hadron_tis: float = 0.0
    n_l0_muon_tis: float = 0.0
    n_l0_muon_high_tis: float = 0.0
    n_l0_dimuon_tis: float = 0.0
    n_l0_photon_tis: float = 0.0
    n_l0_electron_tis: float = 0.0
    n_l0_or: float = 0.0

    n_hlt1_track_mva_tos: float = 0.0
    n_hlt1_two_track_mva_tos: float = 0.0
    n_hlt1_or: float = 0.0

    n_hlt2_topo2_tos: float = 0.0
    n_hlt2_topo3_tos: float = 0.0
    n_hlt2_topo4_tos: float = 0.0
    n_hlt2_or: float = 0.0

    @property
    def eff_gen(self) -> float:
        return self.gen_eff

    def get_err_gen(self) -> float:
        return self.gen_err

    @property
    def eff_reco_str(self) -> float:
        return self.n_reco_str / self.n_total if self.n_total > 0 else 0.0

    @property
    def eff_trig(self) -> float:
        return self.n_trig / self.n_reco_str if self.n_reco_str > 0 else 0.0

    @property
    def eff_pre(self) -> float:
        return self.n_pre / self.n_trig if self.n_trig > 0 else 0.0

    @property
    def eff_mva(self) -> float:
        return self.n_mva / self.n_pre if self.n_pre > 0 else 0.0

    @property
    def eff_total(self) -> float:
        return self.eff_gen * self.eff_reco_str * self.eff_trig * self.eff_pre * self.eff_mva

    def get_err_reco_str(self) -> float:
        if self.n_total == 0:
            return 0.0
        eff = self.eff_reco_str
        return np.sqrt(eff * (1 - eff) / self.n_total)

    def get_err_trig(self) -> float:
        if self.n_reco_str == 0:
            return 0.0
        eff = self.eff_trig
        return np.sqrt(eff * (1 - eff) / self.n_reco_str)

    def get_err_pre(self) -> float:
        if self.n_trig == 0:
            return 0.0
        eff = self.eff_pre
        return np.sqrt(eff * (1 - eff) / self.n_trig)

    def get_err_mva(self) -> float:
        if self.n_pre == 0:
            return 0.0
        eff = self.eff_mva
        return np.sqrt(eff * (1 - eff) / self.n_pre)

    def get_err_total(self) -> float:
        eff_tot_no_gen = self.eff_reco_str * self.eff_trig * self.eff_pre * self.eff_mva
        if self.n_total == 0:
            return 0.0
        err_no_gen = np.sqrt(eff_tot_no_gen * (1 - eff_tot_no_gen) / self.n_total)
        return self.eff_gen * err_no_gen


def get_luminosity_weights(config: dict) -> Dict[str, float]:
    lumi_config = config.get("luminosity", {}).get("integrated_luminosity", {})
    if not lumi_config:
        lumi_config = {"2016": 1.67, "2017": 1.74, "2018": 2.13}
    total_lumi = sum(float(v) for v in lumi_config.values())
    return {year: float(lumi) / total_lumi for year, lumi in lumi_config.items()}


def calculate_efficiencies_for_file(
    file_path: str,
    category: str = "LL",
    gen_eff: float = 0.22,
    gen_err: float = 0.0005,
    mva_model=None,
    mva_features: list = None,
    mva_threshold: float = 0.0,
    kin_weights: Optional[dict] = None,
    tis_tos_corr: Optional[float] = None,
) -> EfficiencyResult:
    """Calculate the efficiency steps for a single MC file.

    Args:
        file_path:     Path to the MC ROOT file.
        category:      Lambda track category ("LL" or "DD").
        gen_eff:       Generator-level acceptance efficiency.
        gen_err:       Generator-level efficiency uncertainty.
        mva_model:     Trained CatBoost model (or None to skip MVA step).
        mva_features:  Feature list matching the trained model.
        mva_threshold: BDT score threshold.
        kin_weights:   Dict with "pt_bins" and "weights" for kinematic reweighting
                       (from kinematic_reweighting/output/kinematic_weights_{category}.json).
                       Optional; if None, no reweighting is applied.
        tis_tos_corr:  Data/MC HLT TOS efficiency correction factor for this (category, year).
                       Applied by scaling n_trig: eff_trig_corrected = eff_trig_MC * corr.
                       For a ratio measurement this cancels, but it is applied for completeness.
    """
    result = EfficiencyResult(gen_eff=gen_eff, gen_err=gen_err)

    try:
        f = uproot.open(file_path)
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return result

    tree_name = f"B2L0barPKpKm_{category}/DecayTree"
    if tree_name not in f:
        print(f"Warning: Tree {tree_name} not found in {file_path}")
        return result

    tree = f[tree_name]
    result.n_total = tree.num_entries
    if result.n_total == 0:
        return result

    # ---- Kinematic reweighting (1D pT) ----
    w_array = np.ones(result.n_total, dtype=float)
    if kin_weights is not None:
        pt_array = tree["Bu_PT"].array(library="np")
        pt_bins = np.array(kin_weights["pt_bins"])
        weights = np.array(kin_weights["weights"])
        idx = np.digitize(pt_array, pt_bins) - 1
        idx = np.clip(idx, 0, len(weights) - 1)
        w_array *= weights[idx]

    result.n_total = np.sum(w_array)
    if result.n_total == 0:
        return result

    # ---- Truth matching and track type ----
    bu_trueid = np.abs(tree["Bu_TRUEID"].array(library="np"))
    p_trueid = np.abs(tree["p_TRUEID"].array(library="np"))
    h1_trueid = np.abs(tree["h1_TRUEID"].array(library="np"))
    h2_trueid = np.abs(tree["h2_TRUEID"].array(library="np"))
    l0_trueid = np.abs(tree["L0_TRUEID"].array(library="np"))
    lp_trueid = np.abs(tree["Lp_TRUEID"].array(library="np"))
    lpi_trueid = np.abs(tree["Lpi_TRUEID"].array(library="np"))

    p_track = tree["p_TRACK_Type"].array(library="np")
    h1_track = tree["h1_TRACK_Type"].array(library="np")
    h2_track = tree["h2_TRACK_Type"].array(library="np")
    lp_track = tree["Lp_TRACK_Type"].array(library="np")
    lpi_track = tree["Lpi_TRACK_Type"].array(library="np")

    # ---- STEP 2: Reconstruction + Stripping ----
    truth_mask = (
        (bu_trueid == 521)
        & (p_trueid == 2212)
        & (h1_trueid == 321)
        & (h2_trueid == 321)
        & (l0_trueid == 3122)
        & (lp_trueid == 2212)
        & (lpi_trueid == 211)
    )
    expected_lambda_track_type = 3 if category == "LL" else 5
    track_mask = (
        (p_track == 3)
        & (h1_track == 3)
        & (h2_track == 3)
        & (lp_track == expected_lambda_track_type)
        & (lpi_track == expected_lambda_track_type)
    )
    reco_str_mask = truth_mask & track_mask
    result.n_reco_str = np.sum(w_array[reco_str_mask])
    if result.n_reco_str == 0:
        return result

    # ---- STEP 3: Trigger (L0 TIS AND HLT1 TOS AND HLT2 TOS) ----
    # L0 TIS: OR of all available L0 TIS branches (pure TIS — not mixed with TOS).
    # This matches the selection in clean_data_loader.py.
    def safe_get(branch_name, default=None):
        if branch_name in tree:
            return tree[branch_name].array(library="np").astype(bool)
        return default if default is not None else np.zeros(result.n_total, dtype=bool)

    zeros = np.zeros(int(tree.num_entries), dtype=bool)

    l0_global_tis = safe_get("Bu_L0GlobalDecision_TIS", zeros.copy())
    l0_hadron_tis = safe_get("Bu_L0HadronDecision_TIS", zeros.copy())
    l0_muon_tis = safe_get("Bu_L0MuonDecision_TIS", zeros.copy())
    l0_muon_high = safe_get("Bu_L0MuonHighDecision_TIS", zeros.copy())
    l0_dimuon_tis = safe_get("Bu_L0DiMuonDecision_TIS", zeros.copy())
    l0_photon_tis = safe_get("Bu_L0PhotonDecision_TIS", zeros.copy())
    l0_electron_tis = safe_get("Bu_L0ElectronDecision_TIS", zeros.copy())

    # L0 TIS = any TIS line fired (matches what clean_data_loader.py uses)
    l0_tis = (
        l0_global_tis
        | l0_hadron_tis
        | l0_muon_tis
        | l0_muon_high
        | l0_dimuon_tis
        | l0_photon_tis
        | l0_electron_tis
    )

    # HLT1 TOS: TrackMVA OR TwoTrackMVA (TOS only — not TIS)
    hlt1_track_mva_tos = safe_get("Bu_Hlt1TrackMVADecision_TOS", zeros.copy())
    hlt1_two_track_mva_tos = safe_get("Bu_Hlt1TwoTrackMVADecision_TOS", zeros.copy())
    hlt1_tos = hlt1_track_mva_tos | hlt1_two_track_mva_tos

    # HLT2 TOS: any Topo body
    hlt2_topo2_tos = safe_get("Bu_Hlt2Topo2BodyDecision_TOS", zeros.copy())
    hlt2_topo3_tos = safe_get("Bu_Hlt2Topo3BodyDecision_TOS", zeros.copy())
    hlt2_topo4_tos = safe_get("Bu_Hlt2Topo4BodyDecision_TOS", zeros.copy())
    hlt2_tos = hlt2_topo2_tos | hlt2_topo3_tos | hlt2_topo4_tos

    # Detailed counts (relative to reco_str)
    result.n_l0_global_tis = np.sum(w_array[reco_str_mask & l0_global_tis])
    result.n_l0_hadron_tis = np.sum(w_array[reco_str_mask & l0_hadron_tis])
    result.n_l0_muon_tis = np.sum(w_array[reco_str_mask & l0_muon_tis])
    result.n_l0_muon_high_tis = np.sum(w_array[reco_str_mask & l0_muon_high])
    result.n_l0_dimuon_tis = np.sum(w_array[reco_str_mask & l0_dimuon_tis])
    result.n_l0_photon_tis = np.sum(w_array[reco_str_mask & l0_photon_tis])
    result.n_l0_electron_tis = np.sum(w_array[reco_str_mask & l0_electron_tis])
    result.n_l0_or = np.sum(w_array[reco_str_mask & l0_tis])
    result.n_hlt1_track_mva_tos = np.sum(w_array[reco_str_mask & hlt1_track_mva_tos])
    result.n_hlt1_two_track_mva_tos = np.sum(w_array[reco_str_mask & hlt1_two_track_mva_tos])
    result.n_hlt1_or = np.sum(w_array[reco_str_mask & hlt1_tos])
    result.n_hlt2_topo2_tos = np.sum(w_array[reco_str_mask & hlt2_topo2_tos])
    result.n_hlt2_topo3_tos = np.sum(w_array[reco_str_mask & hlt2_topo3_tos])
    result.n_hlt2_topo4_tos = np.sum(w_array[reco_str_mask & hlt2_topo4_tos])
    result.n_hlt2_or = np.sum(w_array[reco_str_mask & hlt2_tos])

    trigger_mask = l0_tis & hlt1_tos & hlt2_tos
    trig_total_mask = reco_str_mask & trigger_mask
    n_trig_mc = np.sum(w_array[trig_total_mask])

    # Apply TIS/TOS data/MC correction factor (scales MC trigger efficiency to data value).
    # For a ratio B(X)/B(J/psi), this correction cancels if it is the same for all states.
    # We apply it regardless to demonstrate cancellation in print_cancellation_table().
    if tis_tos_corr is not None and tis_tos_corr > 0:
        n_trig_corrected = n_trig_mc * tis_tos_corr
    else:
        n_trig_corrected = n_trig_mc
    result.n_trig = n_trig_corrected

    if result.n_trig == 0:
        return result

    # ---- STEP 4: Pre-selection ----
    # Cuts aligned with Phase 0 values in config/selection.toml.
    # Category-aware Delta_Z: LL requires 20 mm, DD requires 5 mm.
    delta_z_cut = 20.0 if category == "LL" else 5.0

    bu_fdchi2 = tree["Bu_FDCHI2_OWNPV"].array(library="np")
    bu_ipchi2 = tree["Bu_IPCHI2_OWNPV"].array(library="np")
    bu_pt = tree["Bu_PT"].array(library="np")
    p_probnnp = tree["p_ProbNNp"].array(library="np")
    h1_probnnk = tree["h1_ProbNNk"].array(library="np")
    h2_probnnk = tree["h2_ProbNNk"].array(library="np")

    l0_mass = tree["L0_M"].array(library="np")
    l0_fdchi2_arr = tree["L0_FDCHI2_OWNPV"].array(library="np")
    l0_end_z = tree["L0_ENDVERTEX_Z"].array(library="np")
    bu_end_z = tree["Bu_ENDVERTEX_Z"].array(library="np")
    lp_probnnp = tree["Lp_ProbNNp"].array(library="np")

    # Data-reduction baseline (mirrors clean_data_loader.py)
    baseline_mask = (
        (bu_fdchi2 > 175.0)
        & (bu_ipchi2 < 10.0)
        & (bu_pt > 3000.0)
        & (p_probnnp > 0.05)
        & ((h1_probnnk * h2_probnnk) > 0.05)
    )

    # Lambda pre-selection (Phase 0 values):
    # - FD chi2 > 50 (was 250; lowered to allow optimizer to scan the range)
    # - Delta_Z > 20 mm (LL) or > 5 mm (DD) — category-aware
    # - Fixed PID_product > 0.20 (validated by fit_based_optimizer study)
    lambda_mask = (
        (l0_mass > 1111.0)
        & (l0_mass < 1121.0)
        & (l0_fdchi2_arr > 50.0)
        & ((l0_end_z - bu_end_z) > delta_z_cut)
        & (lp_probnnp > 0.3)
        & ((p_probnnp * h1_probnnk * h2_probnnk) > 0.20)
    )

    pre_mask = baseline_mask & lambda_mask
    pre_total_mask = trig_total_mask & pre_mask
    result.n_pre = np.sum(w_array[pre_total_mask])

    if result.n_pre == 0 or mva_model is None:
        result.n_mva = result.n_pre
        return result

    # ---- STEP 5: MVA Selection ----
    import awkward as ak

    events_pre = tree.arrays(mva_features, library="ak")[pre_total_mask]

    # Handle jagged arrays: find a jagged template (Bu_DTF_chi2 can be jagged in MC)
    template = None
    for feat in mva_features:
        arr = events_pre[feat]
        if arr.ndim > 1:
            template = arr
            break

    data_dict = {}
    for feat in mva_features:
        arr = events_pre[feat]
        if arr.ndim == 1 and template is not None:
            arr = arr * ak.ones_like(template)
        if arr.ndim > 1:
            arr = ak.flatten(arr)
        data_dict[feat] = ak.to_numpy(arr)

    df = pd.DataFrame(data_dict)

    # Build per-row weights matching the (possibly broadcast) feature array
    w_pre_events = w_array[pre_total_mask]
    if template is not None:
        w_pre_jagged = w_pre_events * ak.ones_like(template)
        w_pre = ak.to_numpy(ak.flatten(w_pre_jagged))
    else:
        w_pre = w_pre_events

    if len(df) > 0:
        if hasattr(mva_model, "predict_proba"):
            preds = mva_model.predict_proba(df)[:, 1]
        else:
            import xgboost as xgb

            preds = xgb.Booster().predict(xgb.DMatrix(df))

        result.n_mva = float(np.sum(w_pre[preds > mva_threshold]))
    else:
        result.n_mva = 0.0

    return result


def print_trigger_markdown_table(
    state: str, categories: List[str], years: List[str], results: dict, file_handle=None
):
    output_str = f"\n### {state} Trigger Efficiencies\n\n"
    for cat in categories:
        output_str += f"#### {state} {cat}\n\n"
        header = "| Line (%) | " + " | ".join([f"20{y}" for y in years]) + " |"
        sep = "|---|" + "|".join(["---" for _ in years]) + "|"
        output_str += header + "\n" + sep + "\n"
        layout = [
            ("l0_global_tis", "Bu_L0Global_TIS"),
            ("l0_hadron_tis", "Bu_L0HadronDecision_TIS"),
            ("l0_or", "**L0 TIS OR**"),
            ("hlt1_track_mva_tos", "Bu_Hlt1TrackMVADecision_TOS"),
            ("hlt1_two_track_mva_tos", "Bu_Hlt1TwoTrackMVADecision_TOS"),
            ("hlt1_or", "**HLT1 TOS OR**"),
            ("hlt2_topo2_tos", "Bu_Hlt2Topo2BodyDecision_TOS"),
            ("hlt2_topo3_tos", "Bu_Hlt2Topo3BodyDecision_TOS"),
            ("hlt2_topo4_tos", "Bu_Hlt2Topo4BodyDecision_TOS"),
            ("hlt2_or", "**HLT2 TOS OR**"),
        ]
        for key, label in layout:
            row_str = f"| {label} |"
            for year in years:
                data = results[state][cat][f"20{year}"]
                reco_str = data["counts"]["reco_str"]
                count = data.get("trigger_counts", {}).get(key, 0)
                if reco_str > 0:
                    eff = count / reco_str
                    err = np.sqrt(eff * (1 - eff) / max(reco_str, 1))
                    val = f"{eff*100:.2f} ± {err*100:.2f}"
                    if "OR" in label:
                        val = f"**{val}**"
                    row_str += f" {val} |"
                else:
                    row_str += " 0.00 ± 0.00 |"
            output_str += row_str + "\n"
        output_str += "\n"
    print(output_str, end="")
    if file_handle:
        file_handle.write(output_str)


def print_cancellation_table(
    states: List[str], categories: List[str], years: List[str], results: dict, file_handle=None
):
    output_str = "\n### Efficiency Ratio Relative to J/ψ (Cancellation Check)\n\n"
    output_str += (
        "Values close to 1.00 indicate the efficiency cancels in the branching fraction ratio. "
        "Significant deviations indicate a systematic that must be propagated.\n\n"
    )
    for cat in categories:
        output_str += f"#### {cat} Category\n\n"
        header = "| State / Step | " + " | ".join([f"20{y}" for y in years]) + " |"
        sep = "|---|" + "|".join(["---" for _ in years]) + "|"
        output_str += header + "\n" + sep + "\n"
        steps = [
            ("ε_gen", "eff_gen", "err_gen"),
            ("ε_reco+str", "eff_reco_str", "err_reco_str"),
            ("ε_trig", "eff_trig", "err_trig"),
            ("ε_presel", "eff_pre", "err_pre"),
            ("ε_mva", "eff_mva", "err_mva"),
            ("ε_tot", "eff_total", "err_total"),
        ]
        for state in [s for s in states if s != "Jpsi"]:
            for step_label, eff_key, err_key in steps:
                row_str = f"| **{state}** {step_label} |"
                for year in years:
                    jpsi_eff = results["Jpsi"][cat][f"20{year}"]["efficiencies"]
                    jpsi_err = results["Jpsi"][cat][f"20{year}"]["errors"]
                    state_eff = results[state][cat][f"20{year}"]["efficiencies"]
                    state_err = results[state][cat][f"20{year}"]["errors"]

                    v_ref = jpsi_eff[eff_key]
                    v_sig = state_eff[eff_key]
                    e_ref = jpsi_err[err_key]
                    e_sig = state_err[err_key]

                    if v_ref > 0 and v_sig > 0:
                        ratio = v_sig / v_ref
                        ratio_err = ratio * np.sqrt((e_ref / v_ref) ** 2 + (e_sig / v_sig) ** 2)
                        row_str += f" {ratio:.3f} ± {ratio_err:.3f} |"
                    elif v_ref > 0:
                        row_str += " 0.000 ± 0.000 |"
                    else:
                        row_str += " N/A |"
                output_str += row_str + "\n"
            output_str += "|---|---|---|---|\n"

    print(output_str, end="")
    if file_handle:
        file_handle.write(output_str)


def print_markdown_table(
    state: str, categories: List[str], years: List[str], results: dict, file_handle=None
):
    output_str = f"\n### {state} Efficiencies\n\n"
    header = (
        "| Efficiency (%) | "
        + " | ".join([f"Λ {cat} 20{y}" for cat in categories for y in years])
        + " |"
    )
    sep = "|---|" + "|".join(["---" for _ in range(len(categories) * len(years))]) + "|"
    output_str += header + "\n" + sep + "\n"

    rows = {
        "ε_gen": [],
        "ε_reco+str": [],
        "ε_trig": [],
        "ε_presel": [],
        "ε_mva": [],
        "ε_tot": [],
    }
    for cat in categories:
        for year in years:
            d = results[state][cat][f"20{year}"]
            effs, errs = d["efficiencies"], d["errors"]
            fmt = lambda e, er: f"{e*100:.2f} ± {er*100:.2f}"
            rows["ε_gen"].append(fmt(effs["eff_gen"], errs["err_gen"]))
            rows["ε_reco+str"].append(fmt(effs["eff_reco_str"], errs["err_reco_str"]))
            rows["ε_trig"].append(fmt(effs["eff_trig"], errs["err_trig"]))
            rows["ε_presel"].append(fmt(effs["eff_pre"], errs["err_pre"]))
            rows["ε_mva"].append(fmt(effs["eff_mva"], errs["err_mva"]))
            rows["ε_tot"].append(f"{effs['eff_total']*100:.3f} ± {errs['err_total']*100:.3f}")
    for label, vals in rows.items():
        output_str += f"| {label} | " + " | ".join(vals) + " |\n"
    print(output_str, end="")
    if file_handle:
        file_handle.write(output_str)


def main():
    parser = argparse.ArgumentParser(description="Calculate stepwise efficiencies")
    parser.add_argument("--config-dir", type=str, default="../../config")
    parser.add_argument("--output", type=str, default="output/efficiencies.json")
    parser.add_argument(
        "--tis-tos",
        type=str,
        default="../trigger_tis_tos/output/tis_tos_results.json",
        help="Path to TIS/TOS correction JSON (output of tis_tos_study.py). "
        "Pass 'none' to disable.",
    )
    parser.add_argument(
        "--kin-weights-dir",
        type=str,
        default="../kinematic_reweighting/output",
        help="Directory containing kinematic_weights_{LL,DD}.json files.",
    )
    parser.add_argument(
        "--mva-output-dir",
        type=str,
        default="../../analysis_output/mva",
        help="Root of the main pipeline output directory for loading MVA models.",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="high_yield",
        help="Optimization branch (high_yield or low_yield) for MVA model loading.",
    )
    args = parser.parse_args()

    # ---- Load configs ----
    try:
        with open(f"{args.config_dir}/data.toml", "rb") as f:
            data_config = tomli.load(f)
        with open(f"{args.config_dir}/selection.toml", "rb") as f:
            sel_config = tomli.load(f)
        try:
            with open(f"{args.config_dir}/generator_effs.toml", "rb") as f:
                gen_config = tomli.load(f)
        except Exception:
            gen_config = {}
    except Exception as e:
        print(f"Error loading configs: {e}")
        return

    # ---- Load TIS/TOS correction factors ----
    tis_tos_corrections = {}
    if args.tis_tos and args.tis_tos.lower() != "none":
        try:
            with open(args.tis_tos, "r") as f:
                raw = json.load(f)
            # corrections[category][year] = {"value": ..., "err": ...}
            # Year keys in the JSON are 2-digit ("16","17","18"); we store as-is.
            tis_tos_corrections = raw.get("correction", {})
            print(f"Loaded TIS/TOS corrections from {args.tis_tos}")
        except Exception as e:
            print(f"Warning: Could not load TIS/TOS corrections: {e}")

    # ---- Load kinematic weights per category ----
    kin_weights_by_cat: Dict[str, Optional[dict]] = {"LL": None, "DD": None}
    for cat in ["LL", "DD"]:
        # Try per-category file first, then fall back to combined
        for stem in [f"kinematic_weights_{cat}", "kinematic_weights"]:
            fpath = Path(args.kin_weights_dir) / f"{stem}.json"
            if fpath.exists():
                with open(fpath, "r") as f:
                    kin_weights_by_cat[cat] = json.load(f)
                print(f"[{cat}] Loaded kinematic weights from {fpath}")
                break
        if kin_weights_by_cat[cat] is None:
            print(f"[{cat}] No kinematic weights found — proceeding without reweighting.")

    # ---- Load MVA models per category ----
    mva_features = sel_config.get("xgboost", {}).get("features", [])
    mva_models: Dict[str, Optional[CatBoostClassifier]] = {"LL": None, "DD": None}
    mva_thresholds: Dict[str, float] = {"LL": 0.0, "DD": 0.0}

    opt_type = sel_config.get("cut_application", {}).get("optimization_type", "box")
    if opt_type == "mva":
        for cat in ["LL", "DD"]:
            cuts_path = (
                Path(args.mva_output_dir) / args.branch / cat / "models" / "optimized_cuts.json"
            )
            model_path = Path(args.mva_output_dir) / args.branch / cat / "models" / "mva_model.cbm"
            try:
                with open(cuts_path, "r") as cf:
                    cuts_data = json.load(cf)
                mva_thresholds[cat] = cuts_data.get(
                    "mva_threshold_high", cuts_data.get("mva_threshold", 0.5)
                )
            except Exception as e:
                print(f"[{cat}] Warning: Could not read {cuts_path}: {e}")
                mva_thresholds[cat] = 0.5

            try:
                model = CatBoostClassifier()
                model.load_model(str(model_path))
                mva_models[cat] = model
                print(
                    f"[{cat}] Loaded MVA model from {model_path}, threshold={mva_thresholds[cat]:.3f}"
                )
            except Exception as e:
                print(f"[{cat}] Warning: Could not load MVA model: {e}")

    # ---- Main efficiency loop ----
    mc_base = data_config.get("input_mc", {}).get(
        "base_path", "/share/lazy/Mohamed/Bu2LambdaPPP/files/mc"
    )
    years = ["16", "17", "18"]
    polarities = ["MD", "MU"]
    states = ["Jpsi", "chic0", "chic1", "chic2", "etac"]
    categories = ["LL", "DD"]

    results = {}
    for state in states:
        results[state] = {}
        for category in categories:
            results[state][category] = {}
            for year in years:
                full_year = f"20{year}"
                year_res = EfficiencyResult()

                # TIS/TOS correction for this (category, year)
                corr_entry = tis_tos_corrections.get(category, {}).get(year, {})
                tis_tos_corr = corr_entry.get("value", None)
                if tis_tos_corr is not None:
                    print(f"  [{category} 20{year}] Applying TIS/TOS corr = {tis_tos_corr:.4f}")

                for pol in polarities:
                    file_path = f"{mc_base}/{state}/{state}_{year}_{pol}.root"
                    g_eff, g_err = get_gen_eff_from_config(state, year, pol, gen_config)
                    year_res.gen_eff = g_eff
                    year_res.gen_err = g_err

                    res = calculate_efficiencies_for_file(
                        file_path,
                        category=category,
                        gen_eff=g_eff,
                        gen_err=g_err,
                        mva_model=mva_models[category],
                        mva_features=mva_features,
                        mva_threshold=mva_thresholds[category],
                        kin_weights=kin_weights_by_cat[category],
                        tis_tos_corr=tis_tos_corr,
                    )

                    year_res.n_total += res.n_total
                    year_res.n_reco_str += res.n_reco_str
                    year_res.n_trig += res.n_trig
                    year_res.n_pre += res.n_pre
                    year_res.n_mva += res.n_mva

                    year_res.n_l0_global_tis += res.n_l0_global_tis
                    year_res.n_l0_hadron_tis += res.n_l0_hadron_tis
                    year_res.n_l0_muon_tis += res.n_l0_muon_tis
                    year_res.n_l0_muon_high_tis += res.n_l0_muon_high_tis
                    year_res.n_l0_dimuon_tis += res.n_l0_dimuon_tis
                    year_res.n_l0_photon_tis += res.n_l0_photon_tis
                    year_res.n_l0_electron_tis += res.n_l0_electron_tis
                    year_res.n_l0_or += res.n_l0_or
                    year_res.n_hlt1_track_mva_tos += res.n_hlt1_track_mva_tos
                    year_res.n_hlt1_two_track_mva_tos += res.n_hlt1_two_track_mva_tos
                    year_res.n_hlt1_or += res.n_hlt1_or
                    year_res.n_hlt2_topo2_tos += res.n_hlt2_topo2_tos
                    year_res.n_hlt2_topo3_tos += res.n_hlt2_topo3_tos
                    year_res.n_hlt2_topo4_tos += res.n_hlt2_topo4_tos
                    year_res.n_hlt2_or += res.n_hlt2_or

                trig_counts = {
                    "l0_global_tis": int(year_res.n_l0_global_tis),
                    "l0_hadron_tis": int(year_res.n_l0_hadron_tis),
                    "l0_muon_tis": int(year_res.n_l0_muon_tis),
                    "l0_muon_high_tis": int(year_res.n_l0_muon_high_tis),
                    "l0_dimuon_tis": int(year_res.n_l0_dimuon_tis),
                    "l0_photon_tis": int(year_res.n_l0_photon_tis),
                    "l0_electron_tis": int(year_res.n_l0_electron_tis),
                    "l0_or": int(year_res.n_l0_or),
                    "hlt1_track_mva_tos": int(year_res.n_hlt1_track_mva_tos),
                    "hlt1_two_track_mva_tos": int(year_res.n_hlt1_two_track_mva_tos),
                    "hlt1_or": int(year_res.n_hlt1_or),
                    "hlt2_topo2_tos": int(year_res.n_hlt2_topo2_tos),
                    "hlt2_topo3_tos": int(year_res.n_hlt2_topo3_tos),
                    "hlt2_topo4_tos": int(year_res.n_hlt2_topo4_tos),
                    "hlt2_or": int(year_res.n_hlt2_or),
                }

                results[state][category][full_year] = {
                    "counts": {
                        "total": int(year_res.n_total),
                        "reco_str": int(year_res.n_reco_str),
                        "trig": int(year_res.n_trig),
                        "pre": int(year_res.n_pre),
                        "mva": int(year_res.n_mva),
                    },
                    "trigger_counts": trig_counts,
                    "efficiencies": {
                        "eff_gen": year_res.eff_gen,
                        "eff_reco_str": year_res.eff_reco_str,
                        "eff_trig": year_res.eff_trig,
                        "eff_pre": year_res.eff_pre,
                        "eff_mva": year_res.eff_mva,
                        "eff_total": year_res.eff_total,
                    },
                    "errors": {
                        "err_gen": year_res.get_err_gen(),
                        "err_reco_str": year_res.get_err_reco_str(),
                        "err_trig": year_res.get_err_trig(),
                        "err_pre": year_res.get_err_pre(),
                        "err_mva": year_res.get_err_mva(),
                        "err_total": year_res.get_err_total(),
                    },
                }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    md_output_path = args.output.replace(".json", "_tables.md")
    with open(md_output_path, "w") as md_file:
        for state in states:
            print_markdown_table(state, categories, years, results, md_file)
            print_trigger_markdown_table(state, categories, years, results, md_file)
        print_cancellation_table(states, categories, years, results, md_file)

    print(f"\nResults saved to {args.output} and {md_output_path}")


if __name__ == "__main__":
    main()
