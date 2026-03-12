"""
Calculate efficiency for each step in the analysis pipeline.
Steps:
1. eff_gen: Generator level (placeholder)
2. eff_reco+str: Reconstruction and stripping (truth matched + track types)
3. eff_tri: Trigger efficiency (pass trigger requirements relative to stripping)
4. eff_pre: Pre-selection cuts (mass windows, PID, FD chi2, etc.)

Efficiencies are calculated for each year and combined using luminosity weights.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import awkward as ak
import numpy as np
import tomli
import uproot


def get_gen_eff_from_config(
    state: str, year: str, polarity: str, gen_config: dict
) -> tuple[float, float]:
    """Extract generator efficiency and error for a specific state, year, polarity."""
    # Map state names to config keys if necessary
    state_map = {
        "chic0": "chic0",
        "chic1": "chic1",
        "chic2": "chic2",
        "etac": "etac",
        "Jpsi": "Jpsi",
    }
    c_state = state_map.get(state, state)

    # Handle missing states like chic2 using generic placeholder if not in config
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
    # Generator info
    gen_eff: float = 0.22
    gen_err: float = 0.0005

    # Raw counts
    n_total: int = 0
    n_gen: int = 0  # Placeholder for generated count
    n_reco_str: int = 0
    n_trig: int = 0
    n_pre: int = 0

    # Store individual trigger line counts
    n_l0_global_tis: int = 0
    n_l0_phys_tis: int = 0
    n_l0_hadron_tis: int = 0
    n_l0_muon_tis: int = 0
    n_l0_muon_high_tis: int = 0
    n_l0_dimuon_tis: int = 0
    n_l0_photon_tis: int = 0
    n_l0_electron_tis: int = 0
    n_l0_hadron_tos: int = 0
    n_l0_or: int = 0

    n_hlt1_track_mva_tos: int = 0
    n_hlt1_track_mva_tis: int = 0
    n_hlt1_two_track_mva_tos: int = 0
    n_hlt1_two_track_mva_tis: int = 0
    n_hlt1_or: int = 0

    n_hlt2_topo2_tos: int = 0
    n_hlt2_topo3_tos: int = 0
    n_hlt2_topo4_tos: int = 0
    n_hlt2_or: int = 0

    # Calculate efficiencies relative to previous step
    @property
    def eff_gen(self) -> float:
        # Placeholder: assume 1.0 or implement lookup from MC report
        return 0.22  # Placeholder 22% for gen

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
    def eff_total(self) -> float:
        return self.eff_gen * self.eff_reco_str * self.eff_trig * self.eff_pre

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

    def get_err_total(self) -> float:
        # Simple error propagation assuming uncorrelated binomial steps
        # This is an approximation for eff_total without gen error
        eff_tot_no_gen = self.eff_reco_str * self.eff_trig * self.eff_pre
        if self.n_total == 0:
            return 0.0
        # For a sequence of cuts from n_total to n_pre, it's just a single binomial:
        err_no_gen = np.sqrt(eff_tot_no_gen * (1 - eff_tot_no_gen) / self.n_total)
        return self.eff_gen * err_no_gen


def get_luminosity_weights(config: dict) -> Dict[str, float]:
    """Calculate luminosity weight for each year."""
    lumi_config = config.get("luminosity", {}).get("integrated_luminosity", {})
    if not lumi_config:
        # Default weights if not found
        lumi_config = {"2016": 1.67, "2017": 1.74, "2018": 2.13}

    total_lumi = sum(float(v) for v in lumi_config.values())
    return {year: float(lumi) / total_lumi for year, lumi in lumi_config.items()}


def calculate_efficiencies_for_file(
    file_path: str, category: str = "LL", gen_eff: float = 0.22, gen_err: float = 0.0005
) -> EfficiencyResult:
    """Calculate the efficiency steps for a single MC file for a given Lambda category."""
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

    # Truth IDs
    bu_trueid = np.abs(tree["Bu_TRUEID"].array())
    p_trueid = np.abs(tree["p_TRUEID"].array())
    h1_trueid = np.abs(tree["h1_TRUEID"].array())
    h2_trueid = np.abs(tree["h2_TRUEID"].array())
    l0_trueid = np.abs(tree["L0_TRUEID"].array())
    lp_trueid = np.abs(tree["Lp_TRUEID"].array())
    lpi_trueid = np.abs(tree["Lpi_TRUEID"].array())

    # Track types (3 = Long, 5 = Downstream)
    p_track = tree["p_TRACK_Type"].array()
    h1_track = tree["h1_TRACK_Type"].array()
    h2_track = tree["h2_TRACK_Type"].array()
    lp_track = tree["Lp_TRACK_Type"].array()
    lpi_track = tree["Lpi_TRACK_Type"].array()

    # STEP 2: Reconstruction and Stripping (eff_{reco+str})
    truth_mask = (
        (bu_trueid == 521)
        & (p_trueid == 2212)
        & (h1_trueid == 321)
        & (h2_trueid == 321)
        & (l0_trueid == 3122)
        & (lp_trueid == 2212)
        & (lpi_trueid == 211)
    )

    if category == "LL":
        expected_lambda_track_type = 3
    else:  # DD
        expected_lambda_track_type = 5

    track_mask = (
        (p_track == 3)
        & (h1_track == 3)
        & (h2_track == 3)
        & (lp_track == expected_lambda_track_type)
        & (lpi_track == expected_lambda_track_type)
    )

    reco_str_mask = truth_mask & track_mask
    result.n_reco_str = ak.sum(reco_str_mask)

    if result.n_reco_str == 0:
        return result

    # STEP 3: Trigger (eff_{tri})
    l0_global_tis = tree["Bu_L0GlobalDecision_TIS"].array()
    l0_phys_tis = tree["Bu_L0PhysDecision_TIS"].array()
    l0_hadron_tis = tree["Bu_L0HadronDecision_TIS"].array()
    l0_muon_tis = tree["Bu_L0MuonDecision_TIS"].array()
    l0_muon_high_tis = tree["Bu_L0MuonHighDecision_TIS"].array()
    l0_dimuon_tis = tree["Bu_L0DiMuonDecision_TIS"].array()
    l0_photon_tis = tree["Bu_L0PhotonDecision_TIS"].array()
    l0_electron_tis = tree["Bu_L0ElectronDecision_TIS"].array()
    l0_hadron_tos = tree["Bu_L0HadronDecision_TOS"].array()

    l0_tis = l0_global_tis | l0_hadron_tos

    hlt1_track_mva_tos = tree["Bu_Hlt1TrackMVADecision_TOS"].array()
    hlt1_track_mva_tis = tree["Bu_Hlt1TrackMVADecision_TIS"].array()

    hlt1_tos = hlt1_track_mva_tos | hlt1_track_mva_tis

    hlt2_topo2_tos = tree["Bu_Hlt2Topo2BodyDecision_TOS"].array()
    hlt2_topo3_tos = tree["Bu_Hlt2Topo3BodyDecision_TOS"].array()
    hlt2_topo4_tos = tree["Bu_Hlt2Topo4BodyDecision_TOS"].array()

    hlt2_tos = hlt2_topo2_tos | hlt2_topo3_tos | hlt2_topo4_tos

    # Fill individual trigger line counts relative to stripping
    result.n_l0_global_tis = ak.sum(reco_str_mask & l0_global_tis)
    result.n_l0_phys_tis = ak.sum(reco_str_mask & l0_phys_tis)
    result.n_l0_hadron_tis = ak.sum(reco_str_mask & l0_hadron_tis)
    result.n_l0_muon_tis = ak.sum(reco_str_mask & l0_muon_tis)
    result.n_l0_muon_high_tis = ak.sum(reco_str_mask & l0_muon_high_tis)
    result.n_l0_dimuon_tis = ak.sum(reco_str_mask & l0_dimuon_tis)
    result.n_l0_photon_tis = ak.sum(reco_str_mask & l0_photon_tis)
    result.n_l0_electron_tis = ak.sum(reco_str_mask & l0_electron_tis)
    result.n_l0_hadron_tos = ak.sum(reco_str_mask & l0_hadron_tos)
    result.n_l0_or = ak.sum(reco_str_mask & l0_tis)

    result.n_hlt1_track_mva_tos = ak.sum(reco_str_mask & hlt1_track_mva_tos)
    result.n_hlt1_track_mva_tis = ak.sum(reco_str_mask & hlt1_track_mva_tis)
    result.n_hlt1_or = ak.sum(reco_str_mask & hlt1_tos)

    result.n_hlt2_topo2_tos = ak.sum(reco_str_mask & hlt2_topo2_tos)
    result.n_hlt2_topo3_tos = ak.sum(reco_str_mask & hlt2_topo3_tos)
    result.n_hlt2_topo4_tos = ak.sum(reco_str_mask & hlt2_topo4_tos)
    result.n_hlt2_or = ak.sum(reco_str_mask & hlt2_tos)

    trigger_mask = l0_tis & hlt1_tos & hlt2_tos
    trig_mask_total = reco_str_mask & trigger_mask
    result.n_trig = ak.sum(trig_mask_total)

    if result.n_trig == 0:
        return result

    # STEP 4: Pre-selection (eff_{pre})
    # Data reduction baseline cuts
    bu_fdchi2 = tree["Bu_FDCHI2_OWNPV"].array()
    bu_ipchi2 = tree["Bu_IPCHI2_OWNPV"].array()
    bu_pt = tree["Bu_PT"].array()
    p_probnnp = tree["p_ProbNNp"].array()
    h1_probnnk = tree["h1_ProbNNk"].array()
    h2_probnnk = tree["h2_ProbNNk"].array()

    # Lambda selection cuts
    l0_mass = tree["L0_M"].array()
    l0_fdchi2 = tree["L0_FDCHI2_OWNPV"].array()
    l0_end_z = tree["L0_ENDVERTEX_Z"].array()
    bu_end_z = tree["Bu_ENDVERTEX_Z"].array()
    lp_probnnp = tree["Lp_ProbNNp"].array()

    baseline_mask = (
        (bu_fdchi2 > 175.0)
        & (bu_ipchi2 < 10.0)
        & (bu_pt > 3000.0)
        & (p_probnnp > 0.05)
        & ((h1_probnnk * h2_probnnk) > 0.05)
    )

    lambda_mask = (
        (l0_mass > 1111.0)
        & (l0_mass < 1121.0)
        & (l0_fdchi2 > 250.0)
        & ((l0_end_z - bu_end_z) > 5.0)
        & (lp_probnnp > 0.3)
    )

    pre_mask = baseline_mask & lambda_mask

    pre_mask_total = trig_mask_total & pre_mask
    result.n_pre = ak.sum(pre_mask_total)

    return result


def print_trigger_markdown_table(
    state: str, categories: List[str], years: List[str], results: dict, file_handle=None
):
    output_str = f"\n### {state} Trigger Efficiencies\n\n"

    for cat in categories:
        output_str += f"#### {state} {cat}\n\n"
        header = "| Levels(%) | " + " | ".join([f"20{y}" for y in years]) + " |"
        sep = "|---|" + "|".join(["---" for _ in years]) + "|"
        output_str += header + "\n" + sep + "\n"

        layout = [
            ("l0_global_tis", "Bu_L0Global_TIS"),
            ("l0_hadron_tos", "Bu_L0HadronDecision_TOS"),
            ("l0_or", "**OR**"),
            ("hlt1_track_mva_tos", "Bu_Hlt1TrackMVADecision_TOS"),
            ("hlt1_track_mva_tis", "Bu_Hlt1TrackMVADecision_TIS"),
            ("hlt1_or", "**OR**"),
            ("hlt2_topo2_tos", "Bu_Hlt2Topo2BodyDecision_TOS"),
            ("hlt2_topo3_tos", "Bu_Hlt2Topo3BodyDecision_TOS"),
            ("hlt2_topo4_tos", "Bu_Hlt2Topo4BodyDecision_TOS"),
            ("hlt2_or", "**OR**"),
        ]

        for key, label in layout:
            row_str = f"| {label} |"
            for year in years:
                data = results[state][cat][f"20{year}"]
                reco_str = data["counts"]["reco_str"]
                trig_counts = data.get("trigger_counts", {})
                count = trig_counts.get(key, 0)

                if reco_str > 0:
                    eff = count / reco_str
                    err = np.sqrt(eff * (1 - eff) / reco_str)
                    val_str = f"{eff*100:.2f} ± {err*100:.2f}"
                    if "**OR**" in label:
                        val_str = f"**{val_str}**"
                    row_str += f" {val_str} |"
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
    output_str = "\n### Efficiency Ratio Relative to J/ψ (Cancellation Test)\n\n"
    output_str += "If the ratio is close to 1.00, the efficiency cancels out perfectly in the branching fraction measurement.\n\n"

    # We will compute the ratio of each step's efficiency to J/psi
    for cat in categories:
        output_str += f"#### {cat} Category\n\n"

        # We'll use Run2 (weighted average) if we calculate it, but currently we just have per year.
        # Let's just output 2016, 2017, 2018 for now.
        header = "| State / Step | " + " | ".join([f"20{y}" for y in years]) + " |"
        sep = "|---|" + "|".join(["---" for _ in years]) + "|"
        output_str += header + "\n" + sep + "\n"

        steps = [
            ("ε_gen", "eff_gen"),
            ("ε_strip+reco", "eff_reco_str"),
            ("ε_trig", "eff_trig"),
            ("ε_presel", "eff_pre"),
            ("ε_tot", "eff_total"),
        ]

        for state in [s for s in states if s != "Jpsi"]:
            for step_label, step_key in steps:
                row_str = f"| **{state}** {step_label} |"

                for year in years:
                    jpsi_data = results["Jpsi"][cat][f"20{year}"]["efficiencies"]
                    jpsi_err = results["Jpsi"][cat][f"20{year}"]["errors"]

                    state_data = results[state][cat][f"20{year}"]["efficiencies"]
                    state_err = results[state][cat][f"20{year}"]["errors"]

                    val_jpsi = jpsi_data[step_key]
                    val_state = state_data[step_key]

                    err_jpsi = jpsi_err[step_key.replace("eff_", "err_")]
                    err_state = state_err[step_key.replace("eff_", "err_")]

                    if val_jpsi > 0:
                        ratio = val_state / val_jpsi
                        # simple error propagation for ratio A/B = R -> dR = R * sqrt((dA/A)^2 + (dB/B)^2)
                        rel_err_jpsi = err_jpsi / val_jpsi if val_jpsi > 0 else 0
                        rel_err_state = err_state / val_state if val_state > 0 else 0
                        ratio_err = ratio * np.sqrt(rel_err_jpsi**2 + rel_err_state**2)

                        row_str += f" {ratio:.3f} ± {ratio_err:.3f} |"
                    else:
                        row_str += " N/A |"
                output_str += row_str + "\n"
            output_str += "|---|---|---|---|\n"  # separator between states

    print(output_str, end="")
    if file_handle:
        file_handle.write(output_str)


def print_markdown_table(
    state: str, categories: List[str], years: List[str], results: dict, file_handle=None
):
    output_str = f"\n### {state} Efficiencies\n\n"

    header = (
        "| Efficiency(%) | "
        + " | ".join([f"Λ {cat} 20{y}" for cat in categories for y in years])
        + " |"
    )
    sep = "|---|" + "|".join(["---" for _ in range(len(categories) * len(years))]) + "|"
    output_str += header + "\n" + sep + "\n"

    rows = {"ε_gen": [], "ε_strip+reco": [], "ε_trig": [], "ε_presel": [], "ε_tot": []}

    for cat in categories:
        for year in years:
            data = results[state][cat][f"20{year}"]
            effs = data["efficiencies"]
            errs = data["errors"]

            # Format as percentages
            rows["ε_gen"].append(f"{effs['eff_gen']*100:.2f} ± {errs['err_gen']*100:.2f}")
            rows["ε_strip+reco"].append(
                f"{effs['eff_reco_str']*100:.2f} ± {errs['err_reco_str']*100:.2f}"
            )
            rows["ε_trig"].append(f"{effs['eff_trig']*100:.2f} ± {errs['err_trig']*100:.2f}")
            rows["ε_presel"].append(f"{effs['eff_pre']*100:.2f} ± {errs['err_pre']*100:.2f}")

            # For total we keep precision slightly higher
            tot_val = effs["eff_total"] * 100
            tot_err = errs["err_total"] * 100
            rows["ε_tot"].append(f"{tot_val:.3f} ± {tot_err:.3f}")

    for label, vals in rows.items():
        output_str += f"| {label} | " + " | ".join(vals) + " |\n"

    print(output_str, end="")
    if file_handle:
        file_handle.write(output_str)


def main():
    parser = argparse.ArgumentParser(description="Calculate stepwise efficiencies")
    parser.add_argument(
        "--config-dir", type=str, default="../../config", help="Path to config directory"
    )
    parser.add_argument(
        "--output", type=str, default="output/efficiencies.json", help="Output JSON file"
    )
    args = parser.parse_args()

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

                for pol in polarities:
                    file_path = f"{mc_base}/{state}/{state}_{year}_{pol}.root"

                    g_eff, g_err = get_gen_eff_from_config(state, year, pol, gen_config)
                    year_res.gen_eff = g_eff  # For simplicity, since MD/MU effs are almost identical, just take the last one or average later
                    year_res.gen_err = g_err

                    res = calculate_efficiencies_for_file(
                        file_path, category=category, gen_eff=g_eff, gen_err=g_err
                    )

                    year_res.n_total += res.n_total
                    year_res.n_reco_str += res.n_reco_str
                    year_res.n_trig += res.n_trig
                    year_res.n_pre += res.n_pre

                    year_res.n_l0_global_tis += res.n_l0_global_tis
                    year_res.n_l0_phys_tis += res.n_l0_phys_tis
                    year_res.n_l0_hadron_tis += res.n_l0_hadron_tis
                    year_res.n_l0_muon_tis += res.n_l0_muon_tis
                    year_res.n_l0_muon_high_tis += res.n_l0_muon_high_tis
                    year_res.n_l0_dimuon_tis += res.n_l0_dimuon_tis
                    year_res.n_l0_photon_tis += res.n_l0_photon_tis
                    year_res.n_l0_electron_tis += res.n_l0_electron_tis
                    year_res.n_l0_hadron_tos += res.n_l0_hadron_tos
                    year_res.n_l0_or += res.n_l0_or

                    year_res.n_hlt1_track_mva_tos += res.n_hlt1_track_mva_tos
                    year_res.n_hlt1_track_mva_tis += res.n_hlt1_track_mva_tis

                    year_res.n_hlt1_or += res.n_hlt1_or

                    year_res.n_hlt2_topo2_tos += res.n_hlt2_topo2_tos
                    year_res.n_hlt2_topo3_tos += res.n_hlt2_topo3_tos
                    year_res.n_hlt2_topo4_tos += res.n_hlt2_topo4_tos
                    year_res.n_hlt2_or += res.n_hlt2_or

                trig_counts = {
                    "l0_global_tis": int(year_res.n_l0_global_tis),
                    "l0_phys_tis": int(year_res.n_l0_phys_tis),
                    "l0_hadron_tis": int(year_res.n_l0_hadron_tis),
                    "l0_muon_tis": int(year_res.n_l0_muon_tis),
                    "l0_muon_high_tis": int(year_res.n_l0_muon_high_tis),
                    "l0_dimuon_tis": int(year_res.n_l0_dimuon_tis),
                    "l0_photon_tis": int(year_res.n_l0_photon_tis),
                    "l0_electron_tis": int(year_res.n_l0_electron_tis),
                    "l0_hadron_tos": int(year_res.n_l0_hadron_tos),
                    "l0_or": int(year_res.n_l0_or),
                    "hlt1_track_mva_tos": int(year_res.n_hlt1_track_mva_tos),
                    "hlt1_track_mva_tis": int(year_res.n_hlt1_track_mva_tis),
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
                    },
                    "trigger_counts": trig_counts,
                    "efficiencies": {
                        "eff_gen": year_res.eff_gen,
                        "eff_reco_str": year_res.eff_reco_str,
                        "eff_trig": year_res.eff_trig,
                        "eff_pre": year_res.eff_pre,
                        "eff_total": year_res.eff_total,
                    },
                    "errors": {
                        "err_gen": year_res.get_err_gen(),
                        "err_reco_str": year_res.get_err_reco_str(),
                        "err_trig": year_res.get_err_trig(),
                        "err_pre": year_res.get_err_pre(),
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
