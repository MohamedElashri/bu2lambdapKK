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


@dataclass
class EfficiencyResult:
    n_total: int = 0
    n_gen: int = 0  # Placeholder for generated count
    n_reco_str: int = 0
    n_trig: int = 0
    n_pre: int = 0

    # Calculate efficiencies relative to previous step
    @property
    def eff_gen(self) -> float:
        # Placeholder: assume 1.0 or implement lookup from MC report
        return 0.22  # Placeholder 22% for gen

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


def calculate_efficiencies_for_file(file_path: str, category: str = "LL") -> EfficiencyResult:
    """Calculate the efficiency steps for a single MC file for a given Lambda category."""
    result = EfficiencyResult()

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
    l0_tis = (
        tree["Bu_L0GlobalDecision_TIS"].array()
        | tree["Bu_L0PhysDecision_TIS"].array()
        | tree["Bu_L0HadronDecision_TIS"].array()
        | tree["Bu_L0MuonDecision_TIS"].array()
        | tree["Bu_L0MuonHighDecision_TIS"].array()
        | tree["Bu_L0DiMuonDecision_TIS"].array()
        | tree["Bu_L0PhotonDecision_TIS"].array()
        | tree["Bu_L0ElectronDecision_TIS"].array()
    )

    hlt1_tos = (
        tree["Bu_Hlt1TrackMVADecision_TOS"].array() | tree["Bu_Hlt1TwoTrackMVADecision_TOS"].array()
    )

    hlt2_tos = (
        tree["Bu_Hlt2Topo2BodyDecision_TOS"].array()
        | tree["Bu_Hlt2Topo3BodyDecision_TOS"].array()
        | tree["Bu_Hlt2Topo4BodyDecision_TOS"].array()
    )

    trigger_mask = l0_tis & hlt1_tos & hlt2_tos
    trig_mask_total = reco_str_mask & trigger_mask
    result.n_trig = ak.sum(trig_mask_total)

    if result.n_trig == 0:
        return result

    # STEP 4: Pre-selection (eff_{pre})
    l0_mass = tree["L0_M"].array()
    l0_fdchi2 = tree["L0_FDCHI2_OWNPV"].array()
    l0_end_z = tree["L0_ENDVERTEX_Z"].array()
    bu_end_z = tree["Bu_ENDVERTEX_Z"].array()
    lp_probnnp = tree["Lp_ProbNNp"].array()

    pre_mask = (
        (l0_mass > 1111.0)
        & (l0_mass < 1121.0)
        & (l0_fdchi2 > 250.0)
        & ((l0_end_z - bu_end_z) > 5.0)
        & (lp_probnnp > 0.3)
    )

    pre_mask_total = trig_mask_total & pre_mask
    result.n_pre = ak.sum(pre_mask_total)

    return result


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
        with open(f"{args.config_dir}/luminosity.toml", "rb") as f:
            lumi_config = {"luminosity": tomli.load(f)}
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

    lumi_weights = get_luminosity_weights(lumi_config)

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
                    res = calculate_efficiencies_for_file(file_path, category=category)

                    year_res.n_total += res.n_total
                    year_res.n_reco_str += res.n_reco_str
                    year_res.n_trig += res.n_trig
                    year_res.n_pre += res.n_pre

                results[state][category][full_year] = {
                    "counts": {
                        "total": int(year_res.n_total),
                        "reco_str": int(year_res.n_reco_str),
                        "trig": int(year_res.n_trig),
                        "pre": int(year_res.n_pre),
                    },
                    "efficiencies": {
                        "eff_gen": year_res.eff_gen,
                        "eff_reco_str": year_res.eff_reco_str,
                        "eff_trig": year_res.eff_trig,
                        "eff_pre": year_res.eff_pre,
                        "eff_total": year_res.eff_total,
                    },
                    "errors": {
                        "err_gen": 0.0004,  # Placeholder error
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

    print(f"\nResults saved to {args.output} and {md_output_path}")


if __name__ == "__main__":
    main()
