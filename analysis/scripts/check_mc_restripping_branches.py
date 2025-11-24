#!/usr/bin/env python3
"""
Check if MC files contain all branches required by restripping pre-selection.

This script verifies that our MC samples (Jpsi, etac, chic0, chic1) contain
all the branches that were used in the restripping pre-selection cuts.
This is important because we assume all efficiencies except selection efficiency
cancel in the branching fraction ratios.
"""

import sys
from pathlib import Path
from typing import Dict, Set

import uproot

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.data_handler import DataManager, TOMLConfig

# ============================================================================
# RESTRIPPING PRE-SELECTION BRANCHES
# ============================================================================
# These are the branches that appear in the restripping pre-selection cuts
# as documented in docs/pre-selections.md (section 2: Restripping pre-selection)

# Note: We map the documented cut names to actual branch names in ROOT files
RESTRIPPING_BRANCHES = {
    # B+ candidate variables
    "Bu_MM": "MinBMass / MaxBMass - B+ invariant mass",
    "Bu_PT": "MinBPt - B+ transverse momentum",
    "Bu_ENDVERTEX_CHI2": "MaxBVertChi2DOF - B+ vertex chi2/ndf",
    "Bu_FDCHI2_OWNPV": "MinBPVVDChi2 - B+ flight distance chi2",
    "Bu_IPCHI2_OWNPV": "MaxBPVIPChi2 - B+ impact parameter chi2",
    "Bu_DIRA_OWNPV": "MinBPVDIRA - B+ direction angle",
    # Hadron h1 (K+) variables
    "h1_PT": "MinKPt - K+ transverse momentum",
    "h1_IPCHI2_OWNPV": "MinKIPChi2DV - K+ impact parameter chi2",
    "h1_TRACK_CHI2NDOF": "MaxKChi2 - K+ track chi2",
    "h1_PIDK": "MinKPIDPi - K+ PIDK",
    "h1_PIDp": "MinKPIDp - K+ PIDp",
    "h1_TRACK_GhostProb": "MaxKGHP - K+ ghost probability",
    # Hadron h2 (K-) variables
    "h2_PT": "MinKPt - K- transverse momentum",
    "h2_IPCHI2_OWNPV": "MinKIPChi2DV - K- impact parameter chi2",
    "h2_TRACK_CHI2NDOF": "MaxKChi2 - K- track chi2",
    "h2_PIDK": "MinKPIDPi - K- PIDK",
    "h2_PIDp": "MinKPIDp - K- PIDp",
    "h2_TRACK_GhostProb": "MaxKGHP - K- ghost probability",
    # Proton (from B+) variables
    "p_PT": "MinpPt - proton transverse momentum",
    "p_IPCHI2_OWNPV": "MinpIPChi2DV - proton impact parameter chi2",
    "p_TRACK_CHI2NDOF": "MaxpChi2 - proton track chi2",
    "p_PIDpi": "MinpPIDPi - proton PIDpi",
    "p_PIDK": "MinpPIDK - proton PIDK",
    "p_TRACK_GhostProb": "MaxpGHP - proton ghost probability",
    # Lambda0 variables
    "L0_MM": "MaxLmDeltaM - Lambda mass",
    "L0_PT": "MinLmPt - Lambda transverse momentum",
    "L0_ENDVERTEX_CHI2": "MaxLmVertChi2DOF - Lambda vertex chi2/ndf",
    "L0_FDCHI2_OWNPV": "MinLmPVVDChi2 - Lambda flight distance chi2",
    "L0_IPCHI2_OWNPV": "MinLmIPChi2 - Lambda impact parameter chi2",
    # Lambda proton variables
    "Lp_PT": "MinLmPrtPt - Lambda proton transverse momentum",
    "Lp_PIDpi": "MinLmPrtPIDPi - Lambda proton PIDpi",
    "Lp_IPCHI2_OWNPV": "MinLmPrtIPChi2 - Lambda proton impact parameter chi2",
    "Lp_TRACK_CHI2NDOF": "MaxLmPrtTrkChi2 - Lambda proton track chi2",
    # Lambda pion variables
    "Lpi_PT": "MinLmPiPt - Lambda pion transverse momentum",
    "Lpi_IPCHI2_OWNPV": "MinLmPiIPChi2 - Lambda pion impact parameter chi2",
    "Lpi_TRACK_CHI2NDOF": "MaxLmPiTrkChi2 - Lambda pion track chi2",
}

# Additional note: Some branches like TRACK_Type, MaxTrLong might not be directly
# in the trees but computed or have different names


# ============================================================================
# MC STATES TO CHECK
# ============================================================================
MC_STATES = ["Jpsi", "etac", "chic0", "chic1"]
YEARS = ["2016", "2017", "2018"]
MAGNETS = ["MD", "MU"]
TRACK_TYPES = ["LL", "DD"]


def check_branches_in_file(filepath: Path, track_type: str) -> Set[str]:
    """
    Check which restripping branches exist in a ROOT file.

    Args:
        filepath: Path to ROOT file
        track_type: "LL" or "DD"

    Returns:
        Set of branch names that exist in the file
    """
    if not filepath.exists():
        return set()

    try:
        with uproot.open(filepath) as file:
            tree_path = f"B2L0barPKpKm_{track_type}/DecayTree"
            if tree_path not in file:
                return set()

            tree = file[tree_path]
            available_branches = set(tree.keys())

            # Check which restripping branches are present
            found_branches = set()
            for branch in RESTRIPPING_BRANCHES.keys():
                if branch in available_branches:
                    found_branches.add(branch)

            return found_branches
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return set()


def main():
    """Check all MC files for restripping branches."""

    print("=" * 80)
    print("Checking MC files for restripping pre-selection branches")
    print("=" * 80)

    # Load configuration
    config_path = Path(__file__).parent.parent / "config"
    config = TOMLConfig(config_path)
    data_manager = DataManager(config)
    mc_path = data_manager.mc_path

    print(f"\nMC base path: {mc_path}")
    print(f"Checking {len(RESTRIPPING_BRANCHES)} restripping branches")
    print(f"MC states: {', '.join(MC_STATES)}")
    print()

    # Track results across all files
    all_results: Dict[str, Dict[str, Dict[str, Set[str]]]] = {state: {} for state in MC_STATES}

    # Check each MC state
    for state in MC_STATES:
        print(f"\n{'=' * 80}")
        print(f"State: {state}")
        print(f"{'=' * 80}")

        state_has_any_file = False

        # Check all year/magnet/track_type combinations
        for year in YEARS:
            year_short = int(year) - 2000

            for magnet in MAGNETS:
                filename = f"{state}_{year_short}_{magnet}.root"
                filepath = mc_path / state / filename

                if not filepath.exists():
                    continue

                state_has_any_file = True
                print(f"\n  File: {filename}")

                # Check both track types
                for track_type in TRACK_TYPES:
                    found_branches = check_branches_in_file(filepath, track_type)

                    # Store results
                    file_key = f"{year}_{magnet}"
                    if file_key not in all_results[state]:
                        all_results[state][file_key] = {}
                    all_results[state][file_key][track_type] = found_branches

                    # Print summary
                    found_count = len(found_branches)
                    total_count = len(RESTRIPPING_BRANCHES)
                    missing_count = total_count - found_count

                    print(f"    {track_type}: {found_count}/{total_count} branches found", end="")
                    if missing_count > 0:
                        print(f" ({missing_count} missing) ❌")
                    else:
                        print(" ✓")

        if not state_has_any_file:
            print(f"  No files found for {state}")

    # Generate detailed report
    print("\n\n" + "=" * 80)
    print("DETAILED REPORT: Missing Branches per State")
    print("=" * 80)

    for state in MC_STATES:
        if not all_results[state]:
            print(f"\n{state}: No files found")
            continue

        print(f"\n{state}:")

        # Get union of all missing branches across all files for this state
        all_missing = set(RESTRIPPING_BRANCHES.keys())
        for file_data in all_results[state].values():
            for track_data in file_data.values():
                all_missing -= track_data

        if not all_missing:
            print("  ✓ All restripping branches present!")
        else:
            print(f"  ❌ Missing {len(all_missing)} branches:")
            for branch in sorted(all_missing):
                description = RESTRIPPING_BRANCHES[branch]
                print(f"    - {branch}: {description}")

    # Summary statistics
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for state in MC_STATES:
        if not all_results[state]:
            continue

        # Get typical coverage from first available file
        first_file_data = next(iter(all_results[state].values()))
        first_track_data = next(iter(first_file_data.values()))
        coverage = len(first_track_data) / len(RESTRIPPING_BRANCHES) * 100

        status = "✓" if coverage == 100 else "❌"
        print(f"  {state}: {coverage:.1f}% coverage {status}")

    print("\n" + "=" * 80)

    # Check if we can proceed with efficiency cancellation assumption
    all_complete = True
    for state in MC_STATES:
        if all_results[state]:
            first_file_data = next(iter(all_results[state].values()))
            first_track_data = next(iter(first_file_data.values()))
            if len(first_track_data) < len(RESTRIPPING_BRANCHES):
                all_complete = False
                break

    if all_complete:
        print("\n✓ All MC samples contain all restripping branches!")
        print("  → Restripping efficiency should cancel in branching fraction ratios")
    else:
        print("\n❌ Some MC samples are missing restripping branches!")
        print("  → Need to investigate if missing branches affect efficiency cancellation")
        print("  → Consider systematic uncertainty from restripping efficiency")

    print()


if __name__ == "__main__":
    main()
