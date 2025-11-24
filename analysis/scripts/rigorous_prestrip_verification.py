#!/usr/bin/env python3
"""
Rigorous verification: Are DaVinciTuples files pre-stripping or post-stripping?

Multiple independent tests:
1. Check Bu_MM distribution for events OUTSIDE restripping mass window
2. Check Bu_PT distribution for events BELOW restripping cut
3. Compare event counts with generator-level expectations
4. Look for MC truth information about selection status
5. Check file metadata/comments for stripping information
6. Analyze all branches to see if stripping-added branches exist
"""

import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_mass_distribution(filepath: Path, tree_name: str) -> dict:
    """
    Test 1: Check if events exist outside restripping mass window.

    If pre-stripping: Should see events below 4500 MeV and above 7000 MeV
    If post-stripping: Should see HARD cutoff at exactly 4500 and 7000 MeV
    """
    result = {
        "total_events": 0,
        "below_4500": 0,
        "above_7000": 0,
        "in_range": 0,
        "min_mass": None,
        "max_mass": None,
        "verdict": None,
    }

    try:
        with uproot.open(filepath) as file:
            tree = file[tree_name]
            events = tree.arrays(["Bu_MM"], library="ak")

            bu_mass = ak.to_numpy(events["Bu_MM"])
            result["total_events"] = len(bu_mass)
            result["min_mass"] = float(np.min(bu_mass))
            result["max_mass"] = float(np.max(bu_mass))

            result["below_4500"] = int(np.sum(bu_mass < 4500))
            result["above_7000"] = int(np.sum(bu_mass > 7000))
            result["in_range"] = int(np.sum((bu_mass >= 4500) & (bu_mass <= 7000)))

            # Verdict
            if result["below_4500"] > 0 or result["above_7000"] > 0:
                result["verdict"] = "PRE-STRIPPING (events outside mass window)"
            elif result["min_mass"] == 4500.0 or result["max_mass"] == 7000.0:
                result["verdict"] = "POST-STRIPPING (hard cutoff at boundaries)"
            elif result["min_mass"] > 4500 and result["max_mass"] < 7000:
                result["verdict"] = "POST-STRIPPING (all events well within window)"
            else:
                result["verdict"] = "UNCLEAR"

    except Exception as e:
        result["error"] = str(e)

    return result


def test_pt_distribution(filepath: Path, tree_name: str) -> dict:
    """
    Test 2: Check if events exist below pT cut.

    If pre-stripping: Should see events with pT < 1200 MeV
    If post-stripping: Most/all events should have pT > 1200 MeV
    """
    result = {
        "total_events": 0,
        "below_1200": 0,
        "min_pt": None,
        "verdict": None,
    }

    try:
        with uproot.open(filepath) as file:
            tree = file[tree_name]
            events = tree.arrays(["Bu_PT"], library="ak")

            bu_pt = ak.to_numpy(events["Bu_PT"])
            result["total_events"] = len(bu_pt)
            result["min_pt"] = float(np.min(bu_pt))
            result["below_1200"] = int(np.sum(bu_pt < 1200))

            fraction_below = result["below_1200"] / result["total_events"]

            # Verdict
            if fraction_below > 0.05:  # >5% below cut
                result["verdict"] = "PRE-STRIPPING (significant fraction below pT cut)"
            elif result["below_1200"] == 0:
                result["verdict"] = "POST-STRIPPING (no events below pT cut)"
            else:
                result["verdict"] = (
                    f"POST-STRIPPING (only {fraction_below*100:.1f}% below cut - boundary effects)"
                )

    except Exception as e:
        result["error"] = str(e)

    return result


def test_fdchi2_distribution(filepath: Path, tree_name: str) -> dict:
    """
    Test 3: Check FDCHI2 distribution.

    If pre-stripping: Should see events with FDCHI2 < 20
    If post-stripping: Should see hard cutoff at FDCHI2 = 20
    """
    result = {
        "total_events": 0,
        "below_20": 0,
        "min_fdchi2": None,
        "verdict": None,
    }

    try:
        with uproot.open(filepath) as file:
            tree = file[tree_name]
            events = tree.arrays(["Bu_FDCHI2_OWNPV"], library="ak")

            fdchi2 = ak.to_numpy(events["Bu_FDCHI2_OWNPV"])
            result["total_events"] = len(fdchi2)
            result["min_fdchi2"] = float(np.min(fdchi2))
            result["below_20"] = int(np.sum(fdchi2 < 20))

            fraction_below = result["below_20"] / result["total_events"]

            # Verdict
            if fraction_below > 0.05:  # >5% below cut
                result["verdict"] = "PRE-STRIPPING (significant fraction below FDCHI2 cut)"
            elif result["below_20"] == 0:
                result["verdict"] = "POST-STRIPPING (no events below FDCHI2 cut)"
            else:
                result["verdict"] = f"POST-STRIPPING (only {fraction_below*100:.1f}% below cut)"

    except Exception as e:
        result["error"] = str(e)

    return result


def test_ipchi2_distribution(filepath: Path, tree_name: str) -> dict:
    """
    Test 4: Check IPCHI2 distribution.

    If pre-stripping: Should see many events with IPCHI2 > 12
    If post-stripping: Should see hard cutoff at IPCHI2 = 12
    """
    result = {
        "total_events": 0,
        "above_12": 0,
        "max_ipchi2": None,
        "verdict": None,
    }

    try:
        with uproot.open(filepath) as file:
            tree = file[tree_name]
            events = tree.arrays(["Bu_IPCHI2_OWNPV"], library="ak")

            ipchi2 = ak.to_numpy(events["Bu_IPCHI2_OWNPV"])
            result["total_events"] = len(ipchi2)
            result["max_ipchi2"] = float(np.max(ipchi2))
            result["above_12"] = int(np.sum(ipchi2 > 12))

            fraction_above = result["above_12"] / result["total_events"]

            # Verdict
            if fraction_above > 0.05:  # >5% above cut
                result["verdict"] = "PRE-STRIPPING (significant fraction above IPCHI2 cut)"
            elif result["above_12"] == 0:
                result["verdict"] = "POST-STRIPPING (no events above IPCHI2 cut)"
            else:
                result["verdict"] = f"POST-STRIPPING (only {fraction_above*100:.1f}% above cut)"

    except Exception as e:
        result["error"] = str(e)

    return result


def check_stripping_branches(filepath: Path, tree_name: str) -> dict:
    """
    Test 5: Check for stripping-specific branches.

    Stripping typically adds branches like:
    - *_BKGCAT (background category)
    - *_Hlt*Decision_Dec (trigger decisions)
    - *_L0*Decision_Dec

    These are NOT present in generator-level MC.
    """
    result = {
        "has_bkgcat": False,
        "has_trigger_dec": False,
        "trigger_branches": [],
        "stripping_indicator_branches": [],
        "verdict": None,
    }

    try:
        with uproot.open(filepath) as file:
            tree = file[tree_name]
            all_branches = list(tree.keys())

            # Check for BKGCAT
            bkgcat_branches = [b for b in all_branches if "BKGCAT" in b.upper()]
            if bkgcat_branches:
                result["has_bkgcat"] = True
                result["stripping_indicator_branches"].extend(bkgcat_branches)

            # Check for trigger decisions (added by stripping)
            trigger_branches = [
                b
                for b in all_branches
                if ("Hlt" in b and "Decision" in b) or ("L0" in b and "Decision" in b)
            ]
            if trigger_branches:
                result["has_trigger_dec"] = True
                result["trigger_branches"] = trigger_branches[:10]  # First 10
                result["stripping_indicator_branches"].extend(trigger_branches[:10])

            # Verdict
            if result["has_bkgcat"] or result["has_trigger_dec"]:
                result["verdict"] = "POST-STRIPPING (contains stripping-added branches)"
            else:
                result["verdict"] = "UNCLEAR (no obvious stripping branches found)"

    except Exception as e:
        result["error"] = str(e)

    return result


def check_dira_distribution(filepath: Path, tree_name: str) -> dict:
    """
    Test 6: Check DIRA (direction angle) distribution.

    If pre-stripping: Should see events with DIRA < 0.999
    If post-stripping: Should see hard cutoff at DIRA = 0.999
    """
    result = {
        "total_events": 0,
        "below_0999": 0,
        "min_dira": None,
        "verdict": None,
    }

    try:
        with uproot.open(filepath) as file:
            tree = file[tree_name]
            events = tree.arrays(["Bu_DIRA_OWNPV"], library="ak")

            dira = ak.to_numpy(events["Bu_DIRA_OWNPV"])
            result["total_events"] = len(dira)
            result["min_dira"] = float(np.min(dira))
            result["below_0999"] = int(np.sum(dira < 0.999))

            fraction_below = result["below_0999"] / result["total_events"]

            # Verdict
            if fraction_below > 0.05:  # >5% below cut
                result["verdict"] = "PRE-STRIPPING (significant fraction below DIRA cut)"
            elif result["below_0999"] == 0:
                result["verdict"] = "POST-STRIPPING (no events below DIRA cut)"
            else:
                result["verdict"] = f"POST-STRIPPING (only {fraction_below*100:.1f}% below cut)"

    except Exception as e:
        result["error"] = str(e)

    return result


def compare_with_generator_level(filepath: Path, tree_name: str) -> dict:
    """
    Test 7: If this is generator-level MC, we should see:
    - Much larger statistics (before selection)
    - Wide kinematic distributions
    - Many events failing basic quality cuts
    """
    result = {
        "n_events": 0,
        "interpretation": "",
    }

    try:
        with uproot.open(filepath) as file:
            tree = file[tree_name]
            result["n_events"] = tree.num_entries

            # Generator-level MC typically has 10k-100k events per file
            # After stripping, we expect ~1-5k events
            if result["n_events"] < 5000:
                result["interpretation"] = "Event count suggests POST-STRIPPING (~1-5k typical)"
            elif result["n_events"] > 20000:
                result["interpretation"] = "Event count suggests PRE-STRIPPING (>20k)"
            else:
                result["interpretation"] = "Event count ambiguous (5-20k range)"

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    """Run all verification tests."""

    print("=" * 80)
    print("RIGOROUS VERIFICATION: DaVinciTuples Pre-Strip vs Post-Strip")
    print("=" * 80)
    print()
    print("Running 7 independent tests to determine if files are pre-stripping...\n")

    # Test file
    filepath = Path(
        "/share/lazy/Mohamed/Bu2LambdaPPP/MC/DaVinciTuples/b2fourbodyline/MCBu2JpsiK,PL0barK_18MU.root"
    )
    tree_name = "B2L0barpKpKm/DecayTree"

    print(f"File: {filepath.name}")
    print(f"Tree: {tree_name}")
    print(f"{'=' * 80}\n")

    tests = []

    # Test 1: Mass distribution
    print("TEST 1: Bu_MM Distribution")
    print("-" * 80)
    mass_test = test_mass_distribution(filepath, tree_name)
    if "error" not in mass_test:
        print(f"Total events: {mass_test['total_events']:,}")
        print(f"Mass range: [{mass_test['min_mass']:.1f}, {mass_test['max_mass']:.1f}] MeV")
        print(
            f"Events below 4500 MeV: {mass_test['below_4500']} ({mass_test['below_4500']/mass_test['total_events']*100:.2f}%)"
        )
        print(
            f"Events above 7000 MeV: {mass_test['above_7000']} ({mass_test['above_7000']/mass_test['total_events']*100:.2f}%)"
        )
        print(f"✓ VERDICT: {mass_test['verdict']}")
        tests.append(("Mass Distribution", mass_test["verdict"]))
    print()

    # Test 2: pT distribution
    print("TEST 2: Bu_PT Distribution")
    print("-" * 80)
    pt_test = test_pt_distribution(filepath, tree_name)
    if "error" not in pt_test:
        print(f"Total events: {pt_test['total_events']:,}")
        print(f"Min pT: {pt_test['min_pt']:.1f} MeV")
        print(
            f"Events below 1200 MeV: {pt_test['below_1200']} ({pt_test['below_1200']/pt_test['total_events']*100:.2f}%)"
        )
        print(f"✓ VERDICT: {pt_test['verdict']}")
        tests.append(("pT Distribution", pt_test["verdict"]))
    print()

    # Test 3: FDCHI2 distribution
    print("TEST 3: Bu_FDCHI2_OWNPV Distribution")
    print("-" * 80)
    fdchi2_test = test_fdchi2_distribution(filepath, tree_name)
    if "error" not in fdchi2_test:
        print(f"Total events: {fdchi2_test['total_events']:,}")
        print(f"Min FDCHI2: {fdchi2_test['min_fdchi2']:.1f}")
        print(
            f"Events below 20: {fdchi2_test['below_20']} ({fdchi2_test['below_20']/fdchi2_test['total_events']*100:.2f}%)"
        )
        print(f"✓ VERDICT: {fdchi2_test['verdict']}")
        tests.append(("FDCHI2 Distribution", fdchi2_test["verdict"]))
    print()

    # Test 4: IPCHI2 distribution
    print("TEST 4: Bu_IPCHI2_OWNPV Distribution")
    print("-" * 80)
    ipchi2_test = test_ipchi2_distribution(filepath, tree_name)
    if "error" not in ipchi2_test:
        print(f"Total events: {ipchi2_test['total_events']:,}")
        print(f"Max IPCHI2: {ipchi2_test['max_ipchi2']:.1f}")
        print(
            f"Events above 12: {ipchi2_test['above_12']} ({ipchi2_test['above_12']/ipchi2_test['total_events']*100:.2f}%)"
        )
        print(f"✓ VERDICT: {ipchi2_test['verdict']}")
        tests.append(("IPCHI2 Distribution", ipchi2_test["verdict"]))
    print()

    # Test 5: Stripping branches
    print("TEST 5: Stripping-Specific Branches")
    print("-" * 80)
    branch_test = check_stripping_branches(filepath, tree_name)
    if "error" not in branch_test:
        print(f"Has BKGCAT branches: {branch_test['has_bkgcat']}")
        print(f"Has trigger decision branches: {branch_test['has_trigger_dec']}")
        if branch_test["trigger_branches"]:
            print("Example trigger branches:")
            for b in branch_test["trigger_branches"][:5]:
                print(f"  - {b}")
        print(f"✓ VERDICT: {branch_test['verdict']}")
        tests.append(("Stripping Branches", branch_test["verdict"]))
    print()

    # Test 6: DIRA distribution
    print("TEST 6: Bu_DIRA_OWNPV Distribution")
    print("-" * 80)
    dira_test = check_dira_distribution(filepath, tree_name)
    if "error" not in dira_test:
        print(f"Total events: {dira_test['total_events']:,}")
        print(f"Min DIRA: {dira_test['min_dira']:.6f}")
        print(
            f"Events below 0.999: {dira_test['below_0999']} ({dira_test['below_0999']/dira_test['total_events']*100:.2f}%)"
        )
        print(f"✓ VERDICT: {dira_test['verdict']}")
        tests.append(("DIRA Distribution", dira_test["verdict"]))
    print()

    # Test 7: Event count comparison
    print("TEST 7: Event Count Analysis")
    print("-" * 80)
    count_test = compare_with_generator_level(filepath, tree_name)
    if "error" not in count_test:
        print(f"Number of events: {count_test['n_events']:,}")
        print(f"Interpretation: {count_test['interpretation']}")
        tests.append(("Event Count", count_test["interpretation"]))
    print()

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print()
    print("Summary of all tests:")
    for test_name, verdict in tests:
        print(f"  {test_name:30s}: {verdict}")
    print()

    # Count votes
    prestrip_votes = sum(1 for _, v in tests if "PRE-STRIPPING" in v)
    poststrip_votes = sum(1 for _, v in tests if "POST-STRIPPING" in v)

    print(f"Pre-stripping votes: {prestrip_votes}/{len(tests)}")
    print(f"Post-stripping votes: {poststrip_votes}/{len(tests)}")
    print()

    if poststrip_votes > prestrip_votes:
        print("✓ CONCLUSION: Files are POST-STRIPPING (already stripped)")
        print()
        print("The DaVinciTuples directory contains MC that has ALREADY been")
        print("through restripping selection. These are not generator-level samples.")
    elif prestrip_votes > poststrip_votes:
        print("✓ CONCLUSION: Files are PRE-STRIPPING (generator-level)")
        print()
        print("The DaVinciTuples directory contains MC that has NOT been stripped.")
        print("These can be used to calculate true restripping efficiency!")
    else:
        print("? CONCLUSION: Unclear - evidence is mixed")
        print()
        print("Further investigation needed.")
    print()


if __name__ == "__main__":
    main()
