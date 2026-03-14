"""
Apply real data-driven PID efficiency histograms (PIDCalib2) to the charmonium MC
to calculate accurate PID cancellation ratios.
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Use LHCb style
plt.style.use(hep.style.LHCb2)

# Make sure we can unpickle boost_histograms


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

    # Load both p and pt so we can compute eta
    branches = ["p_P", "h1_P", "h2_P", "Lp_P", "Lpi_P", "p_PT", "h1_PT", "h2_PT"]
    events = tree.arrays(branches)
    return events


def get_eta(pt, p):
    """Calculate pseudorapidity from pT and p."""
    p = np.maximum(p, pt + 1e-6)
    cos_theta = np.sqrt(1.0 - (pt / p) ** 2)
    return 0.5 * np.log((1.0 + cos_theta) / (1.0 - cos_theta + 1e-10))


class PIDCalibMap:
    def __init__(self, pkl_path: Path):
        with open(pkl_path, "rb") as f:
            self.hist = pickle.load(f)

        self.p_axis = self.hist.axes[0].edges
        self.eta_axis = self.hist.axes[1].edges
        self.effs = self.hist.values()

    def get_efficiency(self, p_array, pt_array):
        """Lookup efficiency for arrays of P and PT."""
        eta_array = get_eta(pt_array, p_array)

        # Digitize (1-indexed, subtract 1 for 0-indexed array indexing)
        p_idx = np.digitize(p_array, self.p_axis) - 1
        eta_idx = np.digitize(eta_array, self.eta_axis) - 1

        # Clip to array bounds to handle under/overflow
        p_idx = np.clip(p_idx, 0, len(self.p_axis) - 2)
        eta_idx = np.clip(eta_idx, 0, len(self.eta_axis) - 2)

        # Lookup values
        eff = self.effs[p_idx, eta_idx]
        return eff


def calculate_binned_efficiencies(data, states, p_maps, k_maps, p_branch, pt_branch, bins):
    """Calculate the average event PID efficiency in bins of a specific particle's momentum."""
    # Calculate binned efficiencies
    binned_effs = {}

    for state in states:
        if state not in data:
            continue

        events = data[state]
        p_val = events[p_branch]

        # Average up and down efficiency
        eff_p_up = p_maps["up"].get_efficiency(events["p_P"], events["p_PT"])
        eff_p_down = p_maps["down"].get_efficiency(events["p_P"], events["p_PT"])
        eff_p = (eff_p_up + eff_p_down) / 2.0

        eff_k1_up = k_maps["up"].get_efficiency(events["h1_P"], events["h1_PT"])
        eff_k1_down = k_maps["down"].get_efficiency(events["h1_P"], events["h1_PT"])
        eff_k1 = (eff_k1_up + eff_k1_down) / 2.0

        eff_k2_up = k_maps["up"].get_efficiency(events["h2_P"], events["h2_PT"])
        eff_k2_down = k_maps["down"].get_efficiency(events["h2_P"], events["h2_PT"])
        eff_k2 = (eff_k2_up + eff_k2_down) / 2.0

        tot_eff = eff_p * eff_k1 * eff_k2

        # Profile: mean tot_eff in bins of p_val
        bin_indices = np.digitize(p_val, bins) - 1

        bin_centers = (bins[:-1] + bins[1:]) / 2
        mean_effs = np.zeros(len(bin_centers))
        err_effs = np.zeros(len(bin_centers))

        for i in range(len(bin_centers)):
            mask = bin_indices == i
            if np.sum(mask) > 10:  # Require at least 10 events for a stable mean
                mean_effs[i] = np.mean(tot_eff[mask])
                err_effs[i] = np.std(tot_eff[mask]) / np.sqrt(
                    np.sum(mask)
                )  # Standard error on mean
            else:
                mean_effs[i] = np.nan
                err_effs[i] = np.nan

        binned_effs[state] = {"mean": mean_effs, "err": err_effs}

    return binned_effs


def generate_condensed_plot(data, states, p_maps, k_maps):
    """Generate a 2x2 professional plot grid for the efficiency ratios."""

    # 20 bins from 0 to 100 GeV
    bins = np.linspace(0, 100000, 21)
    bin_centers = (bins[:-1] + bins[1:]) / 2 / 1000.0  # Centers in GeV
    bin_widths = (bins[1:] - bins[:-1]) / 2 / 1000.0

    panels = [
        {"title": "Prompt Proton", "p_branch": "p_P", "pt_branch": "p_PT"},
        {"title": r"Prompt Kaon ($K^+$)", "p_branch": "h1_P", "pt_branch": "h1_PT"},
        {"title": r"Prompt Kaon ($K^-$)", "p_branch": "h2_P", "pt_branch": "h2_PT"},
        {"title": r"$\Lambda^0$ Proton", "p_branch": "Lp_P", "pt_branch": "Lp_PT"},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
    axes = axes.flatten()

    # We want to display visually distinct markers for states
    markers = ["o", "s", "^", "D"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, panel in enumerate(panels):
        ax = axes[idx]
        binned_effs = calculate_binned_efficiencies(
            data, states, p_maps, k_maps, panel["p_branch"], panel["pt_branch"], bins
        )

        if "Jpsi" not in binned_effs:
            continue

        jpsi_mean = binned_effs["Jpsi"]["mean"]
        jpsi_err = binned_effs["Jpsi"]["err"]

        # Plot J/psi unity reference line
        ax.axhline(1.0, color="black", linestyle="-", linewidth=2, zorder=1)
        # 1% band to show how tight the cancellation is
        ax.axhspan(0.99, 1.01, color="gray", alpha=0.2, label=r"$\pm 1\%$ Band", zorder=0)

        m_idx = 0
        for state in states:
            if state == "Jpsi" or state not in binned_effs:
                continue

            state_mean = binned_effs[state]["mean"]
            state_err = binned_effs[state]["err"]

            # Compute Ratio with standard error propagation (ignoring correlation)
            ratio = state_mean / jpsi_mean
            ratio_err = ratio * np.sqrt((state_err / state_mean) ** 2 + (jpsi_err / jpsi_mean) ** 2)

            # Format nicely for legend
            label_map = {
                "chic0": r"$\chi_{c0}$",
                "chic1": r"$\chi_{c1}$",
                "chic2": r"$\chi_{c2}$",
                "etac": r"$\eta_{c}$",
            }
            formatted_label = label_map.get(state, state)

            ax.errorbar(
                bin_centers,
                ratio,
                xerr=bin_widths,
                yerr=ratio_err,
                fmt=markers[m_idx],
                color=colors[m_idx],
                label=formatted_label,
                markersize=6,
                capsize=0,
                zorder=2,
            )
            m_idx += 1

        ax.set_title(panel["title"])
        ax.set_xlabel(r"$P$ [GeV/$c$]")
        if idx % 2 == 0:
            ax.set_ylabel(
                r"Ratio $\left( \varepsilon_{\text{PID}}^{X} / \varepsilon_{\text{PID}}^{J/\psi} \right)$"
            )

        # Very tight y-axis specifically to show it completely cancels
        ax.set_ylim(0.97, 1.03)

        if idx == 0:
            ax.legend(loc="upper right", ncol=2, frameon=True, edgecolor="white")

    plt.tight_layout()
    plt.savefig("output/real_ratio_condensed_panel.pdf", bbox_inches="tight")
    plt.savefig("output/real_ratio_condensed_panel.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved condensed multi-panel plots to output/")


def main():
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    calib_dir = Path("pidcalib_output")

    if not calib_dir.exists():
        print("Error: pidcalib_output directory not found. Please extract the tarball first.")
        return

    # Load the efficiency maps
    print("Loading PIDCalib2 efficiency maps...")
    p_maps = {
        "up": PIDCalibMap(calib_dir / "effhists-Turbo18-up-P-MC15TuneV1_ProbNNp>0.05-P.ETA.pkl"),
        "down": PIDCalibMap(
            calib_dir / "effhists-Turbo18-down-P-MC15TuneV1_ProbNNp>0.05-P.ETA.pkl"
        ),
    }

    k_maps = {
        "up": PIDCalibMap(calib_dir / "effhists-Turbo18-up-K-MC15TuneV1_ProbNNk>0.224-P.ETA.pkl"),
        "down": PIDCalibMap(
            calib_dir / "effhists-Turbo18-down-K-MC15TuneV1_ProbNNk>0.224-P.ETA.pkl"
        ),
    }

    mc_base = Path("/share/lazy/Mohamed/Bu2LambdaPPP/files/mc")
    year = "18"
    pol = "MD"
    cat = "LL"
    states = ["Jpsi", "chic0", "chic1", "chic2", "etac"]
    data = {}

    print("Loading MC events...")
    for state in states:
        file_path = mc_base / state / f"{state}_{year}_{pol}.root"
        events = load_kinematics(str(file_path), cat)
        if events is not None:
            data[state] = events
            print(f"Loaded {state}: {len(events)} events")

    if "Jpsi" not in data:
        print("Error: Jpsi MC not found")
        return

    # Calculate overall data-driven efficiencies
    print("\n--- Data-Driven PID Cancellation Test ---")

    jpsi = data["Jpsi"]
    jpsi_eff_p = (
        p_maps["up"].get_efficiency(jpsi["p_P"], jpsi["p_PT"])
        + p_maps["down"].get_efficiency(jpsi["p_P"], jpsi["p_PT"])
    ) / 2.0
    jpsi_eff_k1 = (
        k_maps["up"].get_efficiency(jpsi["h1_P"], jpsi["h1_PT"])
        + k_maps["down"].get_efficiency(jpsi["h1_P"], jpsi["h1_PT"])
    ) / 2.0
    jpsi_eff_k2 = (
        k_maps["up"].get_efficiency(jpsi["h2_P"], jpsi["h2_PT"])
        + k_maps["down"].get_efficiency(jpsi["h2_P"], jpsi["h2_PT"])
    ) / 2.0
    jpsi_eff_tot = np.mean(jpsi_eff_p * jpsi_eff_k1 * jpsi_eff_k2)

    print(f"J/psi Reference Data-Driven PID eff: {jpsi_eff_tot*100:.2f}%")

    with open(out_dir / "real_pidcalib_results.md", "w") as f:
        f.write("# Data-Driven PID Efficiency Cancellation Test (PIDCalib2)\n\n")
        f.write(
            "Using real calibration data (Turbo18) from CERN EOS to evaluate PID efficiencies.\n\n"
        )
        f.write("| State | Real PID Eff | Ratio (State / J/ψ) | Systematic Error |\n")
        f.write("|-------|-------------|---------------------|------------------|\n")
        f.write(f"| J/ψ (Ref) | {jpsi_eff_tot*100:.2f}% | 1.000 | - |\n")

        for state in states:
            if state == "Jpsi":
                continue
            st_data = data[state]

            st_eff_p = (
                p_maps["up"].get_efficiency(st_data["p_P"], st_data["p_PT"])
                + p_maps["down"].get_efficiency(st_data["p_P"], st_data["p_PT"])
            ) / 2.0
            st_eff_k1 = (
                k_maps["up"].get_efficiency(st_data["h1_P"], st_data["h1_PT"])
                + k_maps["down"].get_efficiency(st_data["h1_P"], st_data["h1_PT"])
            ) / 2.0
            st_eff_k2 = (
                k_maps["up"].get_efficiency(st_data["h2_P"], st_data["h2_PT"])
                + k_maps["down"].get_efficiency(st_data["h2_P"], st_data["h2_PT"])
            ) / 2.0
            st_eff_tot = np.mean(st_eff_p * st_eff_k1 * st_eff_k2)

            ratio = st_eff_tot / jpsi_eff_tot
            sys_err = abs(1.0 - ratio) * 100  # percentage points

            print(
                f"{state}: Eff = {st_eff_tot*100:.2f}%, Ratio = {ratio:.3f} (Sys err: ~{sys_err:.2f}%)"
            )
            f.write(f"| {state} | {st_eff_tot*100:.2f}% | {ratio:.3f} | {sys_err:.2f}% |\n")

    # Generate condensed panel plots
    print("\nGenerating clean condensed panel plots using mplhep...")
    generate_condensed_plot(data, states, p_maps, k_maps)

    print("\nAll done! Results saved in output/")


if __name__ == "__main__":
    main()
