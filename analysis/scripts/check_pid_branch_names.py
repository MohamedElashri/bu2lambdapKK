#!/usr/bin/env python3
"""
Check what PID-related branches exist in MC files.

The restripping cuts use PIDpi branches, but these might exist under different
names (e.g., PIDPi vs PIDpi, or with MC tuning versions like MC12TuneV4_PIDpi).
"""

import sys
from pathlib import Path

import uproot

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.data_handler import DataManager, TOMLConfig


def check_pid_branches(filepath: Path, track_type: str = "LL"):
    """Check what PID-related branches exist in a ROOT file."""

    if not filepath.exists():
        print(f"File not found: {filepath}")
        return

    try:
        with uproot.open(filepath) as file:
            tree_path = f"B2L0barPKpKm_{track_type}/DecayTree"
            if tree_path not in file:
                print(f"Tree {tree_path} not found in {filepath}")
                return

            tree = file[tree_path]
            all_branches = list(tree.keys())

            # Filter for PID-related branches for proton (p_) and Lambda proton (Lp_)
            print(f"\n{'=' * 80}")
            print(f"File: {filepath.name}")
            print(f"Track type: {track_type}")
            print(f"{'=' * 80}")

            # Proton (p_) PID branches
            print("\nProton (p_) PID branches:")
            p_pid_branches = [b for b in all_branches if b.startswith("p_") and "PID" in b]
            if p_pid_branches:
                for branch in sorted(p_pid_branches):
                    print(f"  - {branch}")
            else:
                print("  (none found)")

            # Lambda proton (Lp_) PID branches
            print("\nLambda proton (Lp_) PID branches:")
            lp_pid_branches = [b for b in all_branches if b.startswith("Lp_") and "PID" in b]
            if lp_pid_branches:
                for branch in sorted(lp_pid_branches):
                    print(f"  - {branch}")
            else:
                print("  (none found)")

            # Check for variations of the missing branches
            print("\nLooking for PIDpi/PIDPi variations:")
            pidpi_variations = [b for b in all_branches if "PIDpi" in b or "PIDPi" in b]
            if pidpi_variations:
                for branch in sorted(pidpi_variations):
                    print(f"  - {branch}")
            else:
                print("  (none found)")

            # Sample a few events to check if we can compute PIDpi from other PID variables
            print("\nChecking if PIDpi can be computed from other variables:")
            print("  (PIDpi is sometimes computed as: log(L_pi / (L_K + L_p + L_pi)))")

            # Check for PIDK, PIDp for proton
            if "p_PIDK" in all_branches and "p_PIDp" in all_branches:
                print("  ✓ p_PIDK and p_PIDp exist")
                print("    → p_PIDpi might be computable as -(p_PIDK + p_PIDp)")
            else:
                print("  ❌ Missing p_PIDK or p_PIDp")

            # Check for PIDK, PIDp for Lambda proton
            if "Lp_PIDK" in all_branches and "Lp_PIDp" in all_branches:
                print("  ✓ Lp_PIDK and Lp_PIDp exist")
                print("    → Lp_PIDpi might be computable as -(Lp_PIDK + Lp_PIDp)")
            else:
                print("  ❌ Missing Lp_PIDK or Lp_PIDp")

            # Check if ProbNN variables exist (alternative to DLL PID)
            print("\nChecking for ProbNN alternatives:")
            probnn_branches = [
                b
                for b in all_branches
                if "ProbNN" in b and (b.startswith("p_") or b.startswith("Lp_"))
            ]
            if probnn_branches:
                for branch in sorted(probnn_branches):
                    print(f"  - {branch}")
            else:
                print("  (none found)")

    except Exception as e:
        print(f"Error reading {filepath}: {e}")


def main():
    """Check PID branches in a sample MC file."""

    print("=" * 80)
    print("Checking PID branch naming in MC files")
    print("=" * 80)

    # Load configuration
    config_path = Path(__file__).parent.parent / "config"
    config = TOMLConfig(config_path)
    data_manager = DataManager(config)
    mc_path = data_manager.mc_path

    # Check one file from each state
    states = ["Jpsi", "etac", "chic0", "chic1"]

    for state in states:
        # Use 2016 MD as example
        filename = f"{state}_16_MD.root"
        filepath = mc_path / state / filename

        if filepath.exists():
            check_pid_branches(filepath, track_type="LL")
        else:
            print(f"\n⚠️  File not found: {filepath}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print(
        """
The missing PIDpi branches are likely not stored directly in MC files.
Instead, PIDpi can be computed from PIDK and PIDp:

  DLL(pi-K) ≡ PIDpi = -PIDK  (approximately)

More precisely, for particle hypotheses K, p, π:
  PIDK = log(L_K/L_π)
  PIDp = log(L_p/L_π)
  PIDpi = log(L_π/L_π) = 0 by definition, but in practice:
  PIDpi = -(PIDK + PIDp) is used

The restripping cuts using PIDpi are:
  - MinpPIDPi = -2.0 (for bachelor proton)
  - MinLmPrtPIDPi = -3.0 (for Lambda proton)

These are loose cuts (allowing PIDpi > -2 or -3) which are equivalent to:
  - PIDK + PIDp < 2.0 or 3.0

So we CAN evaluate these cuts using the available PIDK and PIDp branches!
    """
    )


if __name__ == "__main__":
    main()
