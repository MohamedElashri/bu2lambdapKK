#!/usr/bin/env python3
"""
Check MC file decay topology to verify correct signal MC.

This script examines the mother ID distributions in MC files to determine
if the decay proceeds directly (cc̄ → Λ̄⁰ p K⁻) or via intermediate
resonances (cc̄ → Λ̄⁰ Λ*(1520)(→ p K⁻)).

Usage:
    python check_mc_topology.py /path/to/mc/files/

Author: Analysis script for Bu2LambdaPKK
"""

import sys
from pathlib import Path
from collections import Counter

import uproot
import awkward as ak
import numpy as np


# PDG ID lookup
PDG_NAMES: dict[int, str] = {
    0: "unknown",
    521: "B⁺",
    443: "J/ψ",
    441: "ηc",
    10441: "χc0",
    20443: "χc1",
    445: "χc2",
    100443: "ψ(2S)",
    3122: "Λ",
    3124: "Λ(1520)",
    23122: "Λ(1600)",
    33122: "Λ(1670)",
    13122: "Λ(1405)",
    3126: "Λ(1820)",
    13124: "Λ(1690)",
    4: "c quark",
    5: "b quark",
    111: "π⁰",
    211: "π⁺",
    321: "K⁺",
    311: "K⁰",
    2212: "p",
}


def get_pdg_name(pdg_id: int) -> str:
    """Get particle name from PDG ID."""
    return PDG_NAMES.get(abs(pdg_id), f"PDG={pdg_id}")


def check_mc_file(filepath: Path) -> dict:
    """
    Check a single MC file for decay topology.

    Args:
        filepath: Path to ROOT file

    Returns:
        Dictionary with analysis results
    """
    results = {
        "file": filepath.name,
        "trees_found": [],
        "n_events": 0,
        "p_mothers": Counter(),
        "h2_mothers": Counter(),
        "has_lambda_star": False,
        "lambda_star_fraction": 0.0,
    }

    try:
        file = uproot.open(filepath)
    except Exception as e:
        results["error"] = str(e)
        return results

    # Look for DecayTree in any channel
    tree_paths = []
    for key in file.keys():
        if "DecayTree" in key:
            tree_paths.append(key)

    if not tree_paths:
        results["error"] = "No DecayTree found"
        return results

    results["trees_found"] = tree_paths

    # Try to load from first available tree
    for tree_path in tree_paths:
        try:
            tree = file[tree_path]

            # Check for required branches
            available = list(tree.keys())
            required = ["p_MC_MOTHER_ID", "h2_MC_MOTHER_ID"]

            if not all(b in available for b in required):
                # Try alternative branch names
                alt_required = ["p_TRUEID", "h2_TRUEID"]
                if not all(b in available for b in alt_required):
                    continue

            # Load mother ID branches
            branches_to_load = []
            if "p_MC_MOTHER_ID" in available:
                branches_to_load.append("p_MC_MOTHER_ID")
            if "h2_MC_MOTHER_ID" in available:
                branches_to_load.append("h2_MC_MOTHER_ID")

            if not branches_to_load:
                continue

            events = tree.arrays(branches_to_load, library="ak")
            results["n_events"] = len(events)

            # Count mother IDs
            if "p_MC_MOTHER_ID" in events.fields:
                p_mothers = ak.to_numpy(abs(events["p_MC_MOTHER_ID"]))
                results["p_mothers"] = Counter(p_mothers)

            if "h2_MC_MOTHER_ID" in events.fields:
                h2_mothers = ak.to_numpy(abs(events["h2_MC_MOTHER_ID"]))
                results["h2_mothers"] = Counter(h2_mothers)

            # Check for Λ* resonances
            lambda_star_ids = [3124, 23122, 33122, 13122, 3126, 13124, 23124]
            lambda_star_count = sum(results["p_mothers"].get(ls_id, 0) for ls_id in lambda_star_ids)
            if results["n_events"] > 0:
                results["lambda_star_fraction"] = lambda_star_count / results["n_events"]
                results["has_lambda_star"] = results["lambda_star_fraction"] > 0.01

            break  # Successfully processed

        except Exception as e:
            results["error"] = str(e)
            continue

    return results


def print_results(results: dict) -> None:
    """Print analysis results for a file."""
    print(f"\n{'='*60}")
    print(f"File: {results['file']}")
    print(f"{'='*60}")

    if "error" in results:
        print(f"  ERROR: {results['error']}")
        return

    print(f"  Trees found: {len(results['trees_found'])}")
    print(f"  Events: {results['n_events']}")

    if results["n_events"] == 0:
        print("  No events to analyze")
        return

    # Print p mother distribution
    print("\n  p (bachelor proton) mothers (top 5):")
    for mother_id, count in results["p_mothers"].most_common(5):
        pct = count / results["n_events"] * 100
        name = get_pdg_name(mother_id)
        print(f"    {name} ({mother_id}): {count} ({pct:.1f}%)")

    # Print h2 mother distribution
    print("\n  h2 (K⁻) mothers (top 5):")
    for mother_id, count in results["h2_mothers"].most_common(5):
        pct = count / results["n_events"] * 100
        name = get_pdg_name(mother_id)
        print(f"    {name} ({mother_id}): {count} ({pct:.1f}%)")

    # Summary
    print(f"\n  Λ* resonance fraction: {results['lambda_star_fraction']*100:.1f}%")
    if results["has_lambda_star"]:
        print("  ⚠️  Contains Λ(1520) or other Λ* resonances in decay chain")
    else:
        print("  ✓  Direct decay (no Λ* intermediate states)")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python check_mc_topology.py /path/to/mc/files/")
        print("       python check_mc_topology.py file1.root file2.root ...")
        sys.exit(1)

    # Collect files to check
    files_to_check = []

    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.is_dir():
            # Find all relevant MC files (etac, Jpsi, chic0, chic1)
            patterns = ["*etac*", "*Jpsi*", "*chic0*", "*chic1*", "*psi2S*"]
            for pattern in patterns:
                files_to_check.extend(path.glob(pattern))
        elif path.is_file() and path.suffix == ".root":
            files_to_check.append(path)

    if not files_to_check:
        print("No ROOT files found to check")
        sys.exit(1)

    # Sort files by name
    files_to_check = sorted(set(files_to_check))

    print(f"Checking {len(files_to_check)} MC files...")
    print("Looking for decay topology: B⁺ → cc̄(→ Λ̄⁰ p K⁻) K⁺")

    # Analyze each file
    all_results = []
    for filepath in files_to_check:
        results = check_mc_file(filepath)
        all_results.append(results)
        print_results(results)

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'File':<50} {'Events':>10} {'Λ* %':>10} {'Status':<20}")
    print("-" * 90)

    for results in all_results:
        if "error" in results:
            status = "ERROR"
            ls_pct = "-"
        else:
            ls_pct = f"{results['lambda_star_fraction']*100:.1f}%"
            if results["has_lambda_star"]:
                status = "⚠️ Has Λ*"
            else:
                status = "✓ Direct"

        print(f"{results['file']:<50} {results['n_events']:>10} {ls_pct:>10} {status:<20}")


if __name__ == "__main__":
    main()
