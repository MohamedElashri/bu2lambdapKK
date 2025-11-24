#!/usr/bin/env python3
"""
Compare pre-stripping MC with post-stripping MC.

This script checks the MC files in the DaVinciTuples directory to see if they
are generator-level (pre-stripping) samples and compares event counts.

Target files of interest for our charmonium analysis:
- MCBu2JpsiK,PL0barK (B+ → J/ψ K+, with pΛ̄K final state)
- MCBu2etacK,PL0barK (B+ → ηc K+, with pΛ̄K final state)
- MCBu2chic0K (B+ → χc0 K+)
- MCBu2chic1K (B+ → χc1 K+)
"""

import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.data_handler import DataManager, TOMLConfig


def analyze_prestrip_file(filepath: Path, check_restripping_cuts: bool = True) -> dict:
    """
    Analyze a pre-stripping MC file.

    Returns:
    - tree_names: List of available trees
    - n_events: Number of events (if accessible)
    - restrip_efficiency: Fraction passing restripping cuts (if check_restripping_cuts=True)
    """
    result = {
        "exists": False,
        "tree_names": [],
        "n_events": {},
        "restrip_efficiency": {},
        "is_prestrip": None,
    }

    if not filepath.exists():
        return result

    result["exists"] = True

    try:
        with uproot.open(filepath) as file:
            result["tree_names"] = [k for k in file.keys() if "DecayTree" in k]

            # Check each tree
            for tree_name in result["tree_names"]:
                tree = file[tree_name]
                n_events = tree.num_entries
                result["n_events"][tree_name] = n_events

                if check_restripping_cuts and n_events > 0:
                    # Check if we have the required branches
                    branches = list(tree.keys())
                    required = ["Bu_MM", "Bu_PT", "Bu_FDCHI2_OWNPV", "Bu_IPCHI2_OWNPV"]

                    # Note: Branch names might be different, try to find them
                    # Common alternatives: B_MM, Bp_MM, etc.
                    b_mass_branches = [
                        b for b in branches if ("B" in b and "MM" in b) or ("M_" in b)
                    ]
                    b_pt_branches = [
                        b for b in branches if ("B" in b and "PT" in b) or ("pT" in b.lower())
                    ]

                    if all(b in branches for b in required):
                        # Load and apply restripping cuts
                        events = tree.arrays(required, library="ak")

                        mask = np.ones(n_events, dtype=bool)
                        mask &= ak.to_numpy(events["Bu_MM"]) > 4500
                        mask &= ak.to_numpy(events["Bu_MM"]) < 7000
                        mask &= ak.to_numpy(events["Bu_PT"]) > 1200
                        mask &= ak.to_numpy(events["Bu_FDCHI2_OWNPV"]) > 20
                        mask &= ak.to_numpy(events["Bu_IPCHI2_OWNPV"]) < 12

                        n_pass = np.sum(mask)
                        efficiency = n_pass / n_events
                        result["restrip_efficiency"][tree_name] = {
                            "n_pass": int(n_pass),
                            "efficiency": float(efficiency),
                        }

                        # If <50% pass, likely pre-stripping
                        if efficiency < 0.5:
                            result["is_prestrip"] = True
                        elif efficiency > 0.9:
                            result["is_prestrip"] = False
                    else:
                        # Can't determine, but log what we found
                        result["restrip_efficiency"][tree_name] = {
                            "note": "Required branches not found",
                            "b_mass_candidates": b_mass_branches[:3] if b_mass_branches else [],
                            "b_pt_candidates": b_pt_branches[:3] if b_pt_branches else [],
                        }

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    """Compare pre-stripping and post-stripping MC."""

    print("=" * 80)
    print("COMPARING PRE-STRIPPING vs POST-STRIPPING MC")
    print("=" * 80)
    print()

    # Paths
    prestrip_dir = Path("/share/lazy/Mohamed/Bu2LambdaPPP/MC/DaVinciTuples/b2fourbodyline")

    config_path = Path(__file__).parent.parent / "config"
    config = TOMLConfig(config_path)
    data_manager = DataManager(config)
    poststrip_dir = data_manager.mc_path

    # Files to check for each charmonium state
    files_to_check = {
        "J/ψ": {
            "prestrip": "MCBu2JpsiK,PL0barK_18MU.root",
            "poststrip": "Jpsi/Jpsi_18_MU.root",
        },
        "ηc(1S)": {
            "prestrip": "MCBu2etacK,PL0barK_18MU.root",
            "poststrip": "etac/etac_18_MU.root",
        },
        "χc0(1P)": {
            "prestrip": "MCBu2chic0K_18MU.root",
            "poststrip": "chic0/chic0_18_MU.root",
        },
        "χc1(1P)": {
            "prestrip": "MCBu2chic1K_18MU.root",
            "poststrip": "chic1/chic1_18_MU.root",
        },
    }

    print("Checking 2018 MU samples for each state...\n")

    comparison_results = {}

    for state, files in files_to_check.items():
        print(f"\n{'=' * 80}")
        print(f"State: {state}")
        print(f"{'=' * 80}")

        # Analyze pre-stripping file
        prestrip_path = prestrip_dir / files["prestrip"]
        print("\nPRE-STRIPPING MC:")
        print(f"  File: {files['prestrip']}")
        print(f"  Path: {prestrip_path}")

        prestrip_result = analyze_prestrip_file(prestrip_path)

        if not prestrip_result["exists"]:
            print("  ⚠️  File not found")
            continue

        print(f"  Trees found: {len(prestrip_result['tree_names'])}")
        for tree_name in prestrip_result["tree_names"][:5]:  # Show first 5
            n = prestrip_result["n_events"].get(tree_name, 0)
            print(f"    - {tree_name}: {n:,} events")

            if tree_name in prestrip_result["restrip_efficiency"]:
                eff_info = prestrip_result["restrip_efficiency"][tree_name]
                if "efficiency" in eff_info:
                    eff = eff_info["efficiency"] * 100
                    n_pass = eff_info["n_pass"]
                    print(f"      → Restripping efficiency: {eff:.1f}% ({n_pass:,} pass)")

        # Analyze post-stripping file
        poststrip_path = poststrip_dir / files["poststrip"]
        print("\nPOST-STRIPPING MC (current analysis):")
        print(f"  File: {files['poststrip']}")
        print(f"  Path: {poststrip_path}")

        try:
            with uproot.open(poststrip_path) as file:
                # Check B2L0barPKpKm trees
                ll_tree = file["B2L0barPKpKm_LL/DecayTree"]
                dd_tree = file["B2L0barPKpKm_DD/DecayTree"]

                n_ll = ll_tree.num_entries
                n_dd = dd_tree.num_entries
                n_total = n_ll + n_dd

                print("  Trees: B2L0barPKpKm_LL, B2L0barPKpKm_DD")
                print(f"  Events: {n_ll:,} (LL) + {n_dd:,} (DD) = {n_total:,} total")

                comparison_results[state] = {
                    "poststrip_total": n_total,
                }

        except Exception as e:
            print(f"  ⚠️  Error: {e}")

        if len(prestrip_result["tree_names"]) > 5:
            print(f"  ... and {len(prestrip_result['tree_names']) - 5} more trees")

    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    if any(
        r.get("is_prestrip")
        for r in [
            analyze_prestrip_file(prestrip_dir / files["prestrip"])
            for files in files_to_check.values()
        ]
    ):
        print("✓ Pre-stripping MC files appear to be generator-level (low restripping efficiency)")
        print()
        print("These files could be used to:")
        print("  1. Calculate TRUE restripping efficiency for each state")
        print("  2. Verify efficiency cancellation directly")
        print("  3. Understand what fraction of events survive restripping")
        print()
        print("However, note that:")
        print("  - These files may contain multiple decay channels (not just B→ΛpKK)")
        print("  - Branch names may differ from post-stripping files")
        print("  - Would need to identify correct tree for our decay mode")
    else:
        print("⚠️  Could not definitively determine if files are pre-stripping")
        print("    (may need to check branch names and structure more carefully)")

    print()


if __name__ == "__main__":
    main()
