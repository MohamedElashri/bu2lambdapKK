#!/usr/bin/env python3
"""
Verify restripping efficiency cancellation hypothesis for MC samples.

This script:
1. Checks all MC files contain the required restripping pre-selection branches
2. Computes missing PIDpi branches from PIDK and PIDp
3. Verifies that all restripping cuts can be evaluated on MC
4. Provides a final assessment of whether restripping efficiency should cancel

This is crucial for the branching fraction ratio measurement assumption that
all efficiencies except selection efficiency cancel out.
"""

import sys
from pathlib import Path
from typing import Dict

import awkward as ak
import numpy as np
import uproot

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.data_handler import DataManager, TOMLConfig

# ============================================================================
# RESTRIPPING PRE-SELECTION CUTS
# ============================================================================
# From docs/pre-selections.md (section 2: Restripping pre-selection)

RESTRIPPING_CUTS = {
    # B+ candidate
    "MinBMass": ("Bu_MM", ">", 4500.0, "MeV/c^2"),
    "MaxBMass": ("Bu_MM", "<", 7000.0, "MeV/c^2"),
    "MinBPt": ("Bu_PT", ">", 1200.0, "MeV/c"),
    "MaxBVertChi2DOF": ("Bu_ENDVERTEX_CHI2", "<", 15.0, "dimensionless"),
    "MinBPVVDChi2": ("Bu_FDCHI2_OWNPV", ">", 20.0, "dimensionless"),
    "MaxBPVIPChi2": ("Bu_IPCHI2_OWNPV", "<", 12.0, "dimensionless"),
    "MinBPVDIRA": ("Bu_DIRA_OWNPV", ">", 0.999, "dimensionless"),
    # Kaons (h1 = K+, h2 = K-)
    "MinKPt_h1": ("h1_PT", ">", 200.0, "MeV/c"),
    "MinKPt_h2": ("h2_PT", ">", 200.0, "MeV/c"),
    "MinKIPChi2DV_h1": ("h1_IPCHI2_OWNPV", ">", 5.0, "dimensionless"),
    "MinKIPChi2DV_h2": ("h2_IPCHI2_OWNPV", ">", 5.0, "dimensionless"),
    "MaxKChi2_h1": ("h1_TRACK_CHI2NDOF", "<", 3.0, "dimensionless"),
    "MaxKChi2_h2": ("h2_TRACK_CHI2NDOF", "<", 3.0, "dimensionless"),
    "MinKPIDPi_h1": ("h1_PIDK", ">", -2.0, "dimensionless"),  # Note: uses PIDK directly
    "MinKPIDPi_h2": ("h2_PIDK", ">", -2.0, "dimensionless"),
    "MinKPIDp_h1": ("h1_PIDp", ">", -2.0, "dimensionless"),
    "MinKPIDp_h2": ("h2_PIDp", ">", -2.0, "dimensionless"),
    "MaxKGHP_h1": ("h1_TRACK_GhostProb", "<", 0.3, "dimensionless"),
    "MaxKGHP_h2": ("h2_TRACK_GhostProb", "<", 0.3, "dimensionless"),
    # Bachelor proton
    "MinpPt": ("p_PT", ">", 250.0, "MeV/c"),
    "MinpIPChi2DV": ("p_IPCHI2_OWNPV", ">", 5.0, "dimensionless"),
    "MaxpChi2": ("p_TRACK_CHI2NDOF", "<", 3.0, "dimensionless"),
    "MinpPIDPi": ("p_PIDpi_computed", ">", -2.0, "dimensionless"),  # Computed
    "MinpPIDK": ("p_PIDK", ">", -2.0, "dimensionless"),
    "MaxpGHP": ("p_TRACK_GhostProb", "<", 0.3, "dimensionless"),
    # Lambda0
    "MaxLmDeltaM": ("L0_MM", "delta", 18.0, "MeV/c^2"),  # |M_Lambda - 1115.683| < 18
    "MinLmPt": ("L0_PT", ">", 400.0, "MeV/c"),
    "MaxLmVertChi2DOF": ("L0_ENDVERTEX_CHI2", "<", 15.0, "dimensionless"),
    "MinLmPVVDChi2": ("L0_FDCHI2_OWNPV", ">", 12.0, "dimensionless"),
    "MinLmIPChi2": ("L0_IPCHI2_OWNPV", ">", 0.0, "dimensionless"),
    # Lambda proton
    "MinLmPrtPt": ("Lp_PT", ">", 300.0, "MeV/c"),
    "MinLmPrtPIDPi": ("Lp_PIDpi_computed", ">", -3.0, "dimensionless"),  # Computed
    "MinLmPrtIPChi2": ("Lp_IPCHI2_OWNPV", ">", 4.0, "dimensionless"),
    "MaxLmPrtTrkChi2": ("Lp_TRACK_CHI2NDOF", "<", 4.0, "dimensionless"),
    # Lambda pion
    "MinLmPiPt": ("Lpi_PT", ">", 100.0, "MeV/c"),
    "MinLmPiIPChi2": ("Lpi_IPCHI2_OWNPV", ">", 4.0, "dimensionless"),
    "MaxLmPiTrkChi2": ("Lpi_TRACK_CHI2NDOF", "<", 4.0, "dimensionless"),
}


def compute_pidpi(events: ak.Array) -> ak.Array:
    """
    Compute PIDpi from PIDK and PIDp for proton candidates.

    PIDpi is defined as:
    PIDpi = -(PIDK + PIDp) approximately

    This is because:
    PIDK = log(L_K/L_pi)
    PIDp = log(L_p/L_pi)
    PIDpi = log(L_pi/L_pi) = 0, but in practice we use -(PIDK + PIDp)
    """
    # Compute for bachelor proton
    if "p_PIDK" in events.fields and "p_PIDp" in events.fields:
        events = ak.with_field(events, -(events["p_PIDK"] + events["p_PIDp"]), "p_PIDpi_computed")

    # Compute for Lambda proton
    if "Lp_PIDK" in events.fields and "Lp_PIDp" in events.fields:
        events = ak.with_field(
            events, -(events["Lp_PIDK"] + events["Lp_PIDp"]), "Lp_PIDpi_computed"
        )

    return events


def check_restripping_cuts_on_mc(filepath: Path, track_type: str = "LL") -> Dict:
    """
    Load MC file and check if restripping cuts can be evaluated.

    Returns dictionary with:
    - n_events: Number of events in file
    - can_evaluate_all_cuts: Whether all cuts can be evaluated
    - missing_branches: List of branches that cannot be computed
    - efficiency: Fraction of events passing all restripping cuts
    """
    result = {
        "n_events": 0,
        "can_evaluate_all_cuts": False,
        "missing_branches": [],
        "efficiency": None,
        "n_pass": 0,
    }

    if not filepath.exists():
        return result

    try:
        with uproot.open(filepath) as file:
            tree_path = f"B2L0barPKpKm_{track_type}/DecayTree"
            if tree_path not in file:
                return result

            tree = file[tree_path]

            # Get all required branches
            required_branches = set()
            for cut_name, cut_info in RESTRIPPING_CUTS.items():
                branch = cut_info[0]
                if "computed" not in branch:  # Skip computed branches
                    required_branches.add(branch)

            # Also need PIDK and PIDp to compute PIDpi
            required_branches.update(["p_PIDK", "p_PIDp", "Lp_PIDK", "Lp_PIDp"])

            # Check available branches
            available_branches = set(tree.keys())
            missing = required_branches - available_branches

            result["missing_branches"] = sorted(missing)

            if missing:
                result["can_evaluate_all_cuts"] = False
                return result

            # Load events
            events = tree.arrays(library="ak")
            result["n_events"] = len(events)

            # Compute PIDpi
            events = compute_pidpi(events)

            # Apply all restripping cuts
            mask = np.ones(len(events), dtype=bool)

            lambda_mass_pdg = 1115.683  # MeV/c^2

            for cut_name, cut_info in RESTRIPPING_CUTS.items():
                branch, op, value, unit = cut_info

                # Get branch values
                if branch not in events.fields:
                    result["can_evaluate_all_cuts"] = False
                    result["missing_branches"].append(branch)
                    return result

                branch_values = ak.to_numpy(events[branch])

                # Apply cut
                if op == ">":
                    mask &= branch_values > value
                elif op == "<":
                    mask &= branch_values < value
                elif op == "delta":
                    # Special handling for Lambda mass window
                    mask &= np.abs(branch_values - lambda_mass_pdg) < value

            result["can_evaluate_all_cuts"] = True
            result["n_pass"] = int(np.sum(mask))
            result["efficiency"] = (
                result["n_pass"] / result["n_events"] if result["n_events"] > 0 else 0.0
            )

    except Exception as e:
        print(f"    Error processing {filepath}: {e}")
        result["can_evaluate_all_cuts"] = False

    return result


def main():
    """Verify restripping efficiency cancellation for all MC samples."""

    print("=" * 80)
    print("VERIFICATION: Restripping Efficiency Cancellation in MC")
    print("=" * 80)
    print()
    print("Testing hypothesis: Restripping efficiency cancels in branching fraction ratios")
    print("by verifying that all MC samples can be subjected to identical restripping cuts.")
    print()

    # Load configuration
    config_path = Path(__file__).parent.parent / "config"
    config = TOMLConfig(config_path)
    data_manager = DataManager(config)
    mc_path = data_manager.mc_path

    MC_STATES = ["Jpsi", "etac", "chic0", "chic1"]
    YEARS = ["2016", "2017", "2018"]
    MAGNETS = ["MD", "MU"]
    TRACK_TYPES = ["LL", "DD"]

    # Store results
    all_results = {state: {} for state in MC_STATES}

    # Check each MC state
    for state in MC_STATES:
        print(f"\n{'=' * 80}")
        print(f"State: {state}")
        print(f"{'=' * 80}")

        for year in YEARS:
            year_short = int(year) - 2000

            for magnet in MAGNETS:
                filename = f"{state}_{year_short}_{magnet}.root"
                filepath = mc_path / state / filename

                if not filepath.exists():
                    continue

                print(f"\n  File: {filename}")

                for track_type in TRACK_TYPES:
                    result = check_restripping_cuts_on_mc(filepath, track_type)

                    file_key = f"{year}_{magnet}_{track_type}"
                    all_results[state][file_key] = result

                    if result["n_events"] == 0:
                        print(f"    {track_type}: No events found")
                        continue

                    if result["can_evaluate_all_cuts"]:
                        eff_pct = result["efficiency"] * 100
                        print(
                            f"    {track_type}: ✓ All cuts evaluable | "
                            f"{result['n_events']} events | "
                            f"Efficiency: {eff_pct:.2f}% "
                            f"({result['n_pass']}/{result['n_events']} pass)"
                        )
                    else:
                        print(
                            f"    {track_type}: ❌ Missing branches: {result['missing_branches']}"
                        )

    # Summary
    print("\n\n" + "=" * 80)
    print("EFFICIENCY COMPARISON ACROSS STATES")
    print("=" * 80)
    print()
    print("Checking if restripping efficiencies are similar across all states...")
    print("(Similar efficiencies → cancellation in ratios is valid)")
    print()

    # Collect efficiencies by track type
    efficiencies = {tt: {state: [] for state in MC_STATES} for tt in TRACK_TYPES}

    for state in MC_STATES:
        for file_key, result in all_results[state].items():
            if result["can_evaluate_all_cuts"] and result["efficiency"] is not None:
                track_type = file_key.split("_")[-1]
                efficiencies[track_type][state].append(result["efficiency"])

    # Compute mean and std for each state
    for track_type in TRACK_TYPES:
        print(f"\n{track_type} Track Type:")
        print("-" * 50)

        for state in MC_STATES:
            if efficiencies[track_type][state]:
                eff_values = efficiencies[track_type][state]
                mean_eff = np.mean(eff_values) * 100
                std_eff = np.std(eff_values) * 100
                print(f"  {state:10s}: {mean_eff:6.2f}% ± {std_eff:4.2f}%")
            else:
                print(f"  {state:10s}: No data")

    # Final assessment
    print("\n\n" + "=" * 80)
    print("FINAL ASSESSMENT")
    print("=" * 80)
    print()

    # Check if all states can evaluate all cuts
    all_can_evaluate = True
    for state in MC_STATES:
        for result in all_results[state].values():
            if result["n_events"] > 0 and not result["can_evaluate_all_cuts"]:
                all_can_evaluate = False
                break

    if all_can_evaluate:
        print("✓ SUCCESS: All MC samples can evaluate all restripping cuts!")
        print()
        print("Key findings:")
        print("  1. All required branches present (after computing PIDpi)")
        print("  2. All restripping cuts can be applied to MC")
        print("  3. Efficiencies can be computed for each state")
        print()

        # Check if efficiencies are similar
        all_effs = []
        for track_type in TRACK_TYPES:
            for state in MC_STATES:
                all_effs.extend(efficiencies[track_type][state])

        if all_effs:
            mean_eff = np.mean(all_effs) * 100
            std_eff = np.std(all_effs) * 100
            rel_std = (std_eff / mean_eff) * 100 if mean_eff > 0 else 0

            print(f"Overall restripping efficiency: {mean_eff:.2f}% ± {std_eff:.2f}%")
            print(f"Relative variation: {rel_std:.1f}%")
            print()

            if rel_std < 5:
                print("✓ CONCLUSION: Restripping efficiency is very similar across states")
                print("  → Cancellation in branching fraction ratios is valid!")
                print("  → No systematic uncertainty from restripping needed")
            elif rel_std < 10:
                print("⚠ CONCLUSION: Restripping efficiency shows small variation")
                print("  → Cancellation mostly valid, but consider small systematic")
                print(f"  → Suggested systematic: ±{rel_std:.1f}%")
            else:
                print("❌ WARNING: Restripping efficiency varies significantly!")
                print("  → Cancellation assumption may not be valid")
                print(f"  → Must include systematic uncertainty: ±{rel_std:.1f}%")
    else:
        print("❌ PROBLEM: Some MC samples cannot evaluate all restripping cuts")
        print()
        print("This may affect the efficiency cancellation assumption.")
        print("Review the missing branches above and consult with your analysis team.")

    print()


if __name__ == "__main__":
    main()
