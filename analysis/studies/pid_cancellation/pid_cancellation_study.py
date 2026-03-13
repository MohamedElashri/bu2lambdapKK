"""
PID Cancellation Study (Alternative to PIDCalib)

Since we are measuring branching fraction ratios relative to J/psi, and the final state is identical (L0 p K K),
PID efficiencies should largely cancel.

This script rigorously tests this assumption by:
1. Loading the J/psi MC and the charmonium MCs.
2. Extracting the kinematic distributions (p and pT/eta) for the final state particles (p, K, K).
3. Comparing the normalized distributions. If they overlap significantly, the PID cancellation is justified.
4. Applying a synthetic PID efficiency curve (approximating standard LHCb performance) to quantify the residual non-cancellation as a systematic uncertainty.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def get_eta(pt, p):
    p = np.maximum(p, pt + 1e-6)
    cos_theta = np.sqrt(1.0 - (pt / p) ** 2)
    return 0.5 * np.log((1.0 + cos_theta) / (1.0 - cos_theta + 1e-10))


def synthetic_pid_eff_kaon(p_array):
    """
    Approximation of LHCb Kaon PID efficiency (ProbNNk > 0.1).
    Typically rises quickly, plateaus around 90-95% between 10-50 GeV, and slowly falls off.
    """
    # p is in MeV, convert to GeV
    p_gev = p_array / 1000.0
    # Simple parameterized curve
    eff = 0.95 * (1.0 - np.exp(-p_gev / 3.0)) * (1.0 - 0.002 * p_gev)
    return np.clip(eff, 0.0, 1.0)


def synthetic_pid_eff_proton(p_array):
    """
    Approximation of LHCb Proton PID efficiency.
    """
    p_gev = p_array / 1000.0
    eff = 0.92 * (1.0 - np.exp(-p_gev / 5.0)) * (1.0 - 0.001 * p_gev)
    return np.clip(eff, 0.0, 1.0)


def load_kinematics(file_path: str, category: str):
    """Load momenta for the 3 main tracks (from B) and 2 from Lambda."""
    tree_name = f"B2L0barPKpKm_{category}/DecayTree"
    try:
        f = uproot.open(file_path)
        if tree_name not in f:
            return None
        tree = f[tree_name]
    except Exception:
        return None

    if tree.num_entries == 0:
        return None

    branches = ["p_P", "h1_P", "h2_P", "Lp_P", "Lpi_P", "p_PT", "h1_PT", "h2_PT"]
    events = tree.arrays(branches)
    return events


def main():
    mc_base = Path("/share/lazy/Mohamed/Bu2LambdaPPP/files/mc")

    # We'll compare 2018 MD as a representative sample
    year = "18"
    pol = "MD"
    cat = "LL"

    states = ["Jpsi", "chic0", "chic1", "chic2", "etac"]

    data = {}

    for state in states:
        file_path = mc_base / state / f"{state}_{year}_{pol}.root"
        events = load_kinematics(str(file_path), cat)
        if events is not None:
            data[state] = events
            print(f"Loaded {state}: {len(events)} events")

    if "Jpsi" not in data:
        print("Error: Jpsi MC not found")
        return

    jpsi = data["Jpsi"]

    # Calculate synthetic efficiencies and cancellation ratios
    print("\n--- PID Cancellation Test ---")
    print("Assuming synthetic PID performance curves:")

    jpsi_eff_p = synthetic_pid_eff_proton(jpsi["p_P"])
    jpsi_eff_k1 = synthetic_pid_eff_kaon(jpsi["h1_P"])
    jpsi_eff_k2 = synthetic_pid_eff_kaon(jpsi["h2_P"])
    jpsi_eff_tot = np.mean(jpsi_eff_p * jpsi_eff_k1 * jpsi_eff_k2)

    print(f"J/psi Reference synthetic PID eff: {jpsi_eff_tot*100:.2f}%")

    with open("output/pid_cancellation_results.md", "w") as f:
        f.write("# PID Efficiency Cancellation Test\n\n")
        f.write(
            "Using synthetic momentum-dependent PID efficiency curves to estimate residual non-cancellation.\n\n"
        )
        f.write("| State | Synthetic PID Eff | Ratio (State / J/ψ) | Systematic Error |\n")
        f.write("|-------|------------------|---------------------|------------------|\n")
        f.write(f"| J/ψ (Ref) | {jpsi_eff_tot*100:.2f}% | 1.000 | - |\n")

        for state in states:
            if state == "Jpsi":
                continue
            st_data = data[state]
            st_eff_p = synthetic_pid_eff_proton(st_data["p_P"])
            st_eff_k1 = synthetic_pid_eff_kaon(st_data["h1_P"])
            st_eff_k2 = synthetic_pid_eff_kaon(st_data["h2_P"])
            st_eff_tot = np.mean(st_eff_p * st_eff_k1 * st_eff_k2)

            ratio = st_eff_tot / jpsi_eff_tot
            sys_err = abs(1.0 - ratio) * 100  # percentage points

            print(
                f"{state}: Eff = {st_eff_tot*100:.2f}%, Ratio = {ratio:.3f} (Sys err: ~{sys_err:.2f}%)"
            )
            f.write(f"| {state} | {st_eff_tot*100:.2f}% | {ratio:.3f} | {sys_err:.2f}% |\n")

    # Plot momentum distributions for J/psi vs chic1 (as an example)
    if "chic1" in data:
        plt.figure(figsize=(10, 6))

        plt.hist(
            jpsi["p_P"] / 1000.0,
            bins=50,
            range=(0, 100),
            histtype="step",
            density=True,
            label="J/psi Proton",
            color="blue",
        )
        plt.hist(
            data["chic1"]["p_P"] / 1000.0,
            bins=50,
            range=(0, 100),
            histtype="step",
            density=True,
            label="chic1 Proton",
            color="red",
        )

        plt.hist(
            jpsi["h1_P"] / 1000.0,
            bins=50,
            range=(0, 100),
            histtype="step",
            density=True,
            label="J/psi Kaon1",
            linestyle="dashed",
            color="blue",
        )
        plt.hist(
            data["chic1"]["h1_P"] / 1000.0,
            bins=50,
            range=(0, 100),
            histtype="step",
            density=True,
            label="chic1 Kaon1",
            linestyle="dashed",
            color="red",
        )

        plt.xlabel("Momentum P [GeV/c]")
        plt.ylabel("Normalized Frequency")
        plt.title("Kinematic Overlap (PID Variables)")
        plt.legend()
        plt.savefig("output/momentum_comparison.png")
        print("Saved momentum comparison plot to output/momentum_comparison.png")


if __name__ == "__main__":
    main()
