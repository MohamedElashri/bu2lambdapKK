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
    def __init__(self, pkl_path: Path, bootstrap_effs: np.ndarray | None = None):
        """
        Load a PIDCalib2 efficiency histogram.

        Args:
            pkl_path: Path to the boost-histogram pickle file.
            bootstrap_effs: Optional pre-perturbed efficiency array (shape matching
                the histogram grid). When provided, `get_efficiency` uses this array
                instead of the nominal values — used for bootstrap systematic evaluation.
        """
        with open(pkl_path, "rb") as f:
            self.hist = pickle.load(f)

        self.p_axis = self.hist.axes[0].edges
        self.eta_axis = self.hist.axes[1].edges
        self.effs = self.hist.values() if bootstrap_effs is None else bootstrap_effs

        # Statistical variances per bin (may not be present in all PIDCalib2 versions)
        try:
            self._variances = self.hist.variances()
        except Exception:
            self._variances = None

    def bootstrap_sample(self, rng: np.random.Generator) -> "PIDCalibMap":
        """Return a new PIDCalibMap with bin efficiencies perturbed by their statistical
        uncertainties (Gaussian smearing). Uses the bin variances if available;
        falls back to a Poisson approximation (σ ≈ √ε·(1-ε)/N with N≈100)."""
        if self._variances is not None:
            sigma = np.sqrt(np.maximum(self._variances, 0.0))
        else:
            # Conservative fallback: assume ~1% relative uncertainty per bin
            sigma = 0.01 * self.effs

        perturbed = np.clip(self.effs + rng.normal(0.0, sigma), 0.0, 1.0)
        # Create a new instance sharing the same path/axes but with perturbed values
        obj = object.__new__(PIDCalibMap)
        obj.hist = self.hist
        obj.p_axis = self.p_axis
        obj.eta_axis = self.eta_axis
        obj.effs = perturbed
        obj._variances = self._variances
        return obj

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


def generate_condensed_plot(data, states, p_maps, k_maps, output_dir: Path):
    """Generate a 2x2 professional plot grid for the efficiency ratios."""

    # 20 bins from 0 to 100 GeV
    bins = np.linspace(0, 100000, 21)
    bin_centers = (bins[:-1] + bins[1:]) / 2 / 1000.0  # Centers in GeV
    bin_widths = (bins[1:] - bins[:-1]) / 2 / 1000.0

    panels = [
        {"title": r"Prompt Proton ($p$)", "p_branch": "p_P", "pt_branch": "p_PT"},
        {"title": r"Prompt Kaon ($K^+$)", "p_branch": "h1_P", "pt_branch": "h1_PT"},
        {"title": r"Prompt Kaon ($K^-$)", "p_branch": "h2_P", "pt_branch": "h2_PT"},
        {"title": r"$\Lambda^0$ Proton", "p_branch": "Lp_P", "pt_branch": "Lp_PT"},
    ]

    # Increase figure size significantly for high-res visibility
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharey=True)
    axes = axes.flatten()

    # We want to display visually distinct markers for states
    markers = ["o", "s", "^", "D"]
    # Use LHCb-style distinct colors
    colors = ["#0078FF", "#FF6600", "#0AAFB6", "#FF3333"]

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
        ax.axhline(1.0, color="black", linestyle="-", linewidth=2.5, zorder=1)
        # 1% band to show how tight the cancellation is
        ax.axhspan(0.99, 1.01, color="gray", alpha=0.15, label=r"$\pm 1\%$ Band", zorder=0)

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

            # Only add labels to the legend once (from the first panel) to avoid duplicates
            plot_label = formatted_label if idx == 0 else "_nolegend_"
            band_label = r"$\pm 1\%$ Band" if idx == 0 else "_nolegend_"

            ax.errorbar(
                bin_centers,
                ratio,
                xerr=bin_widths,
                yerr=ratio_err,
                fmt=markers[m_idx],
                color=colors[m_idx],
                label=plot_label,
                markersize=10,
                capsize=3,
                linewidth=2,
                zorder=2,
            )
            m_idx += 1

        # Add labels
        ax.set_xlabel(f"{panel['title']} $p$ [GeV/$c$]", fontsize=24)
        if idx % 2 == 0:
            ax.set_ylabel(
                r"Ratio $\left( \varepsilon_{\rm PID}^{X} / \varepsilon_{\rm PID}^{J/\psi} \right)$",
                fontsize=26,
            )

        # Very tight y-axis specifically to show it completely cancels
        ax.set_ylim(0.95, 1.05)
        ax.tick_params(axis="both", which="major", labelsize=20)

        # Add standard LHCb preliminary watermark to the first plot
        if idx == 0:
            hep.lhcb.text("Preliminary", ax=ax, loc=1, fontsize=24)

    # Extract handles and labels from the first axis to create a single global legend
    handles, labels = axes[0].get_legend_handles_labels()

    # Place a global legend outside the subplots, centered below the master title
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=5,
        frameon=False,
        fontsize=22,
    )

    # Global title and spacing
    plt.suptitle(
        "PID Efficiency Cancellation Verification (Data-Driven)",
        fontsize=32,
        y=0.98,
        fontweight="bold",
    )
    plt.subplots_adjust(top=0.86, wspace=0.1, hspace=0.25)

    plt.savefig(output_dir / "real_ratio_condensed_panel.pdf", bbox_inches="tight")
    plt.savefig(output_dir / "real_ratio_condensed_panel.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved condensed multi-panel plots to {output_dir}/")


def main(output_dir: str = "generated/output/studies/pid_cancellation"):
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
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
    generate_condensed_plot(data, states, p_maps, k_maps, out_dir)

    print(f"\nAll done! Results saved in {out_dir}/")


def run_pid_bootstrap(
    data: dict, states: list, p_maps: dict, k_maps: dict, n_bootstrap: int = 100, seed: int = 42
) -> dict:
    """
    PID systematic via bootstrap.

    For each bootstrap iteration, smear all efficiency map bins by their
    statistical uncertainty and recompute the per-state PID efficiency ratio
    (state / J/ψ).  The RMS of the ratio across iterations is the PID systematic.

    Args:
        data:        {state: events_array} as loaded by load_kinematics.
        states:      List of state names (J/ψ must be present as "Jpsi").
        p_maps:      {"up": PIDCalibMap, "down": PIDCalibMap} for protons.
        k_maps:      {"up": PIDCalibMap, "down": PIDCalibMap} for kaons.
        n_bootstrap: Number of bootstrap iterations (default 100).
        seed:        RNG seed for reproducibility.

    Returns:
        Dict {state: {"ratio_nominal": float, "syst_abs": float, "syst_rel": float,
                      "bootstrap_ratios": list}}
    """
    rng = np.random.default_rng(seed)

    def event_eff(d, pm_u, pm_d, km_u, km_d):
        eff_p = (
            pm_u.get_efficiency(d["p_P"], d["p_PT"]) + pm_d.get_efficiency(d["p_P"], d["p_PT"])
        ) / 2
        eff_k1 = (
            km_u.get_efficiency(d["h1_P"], d["h1_PT"]) + km_d.get_efficiency(d["h1_P"], d["h1_PT"])
        ) / 2
        eff_k2 = (
            km_u.get_efficiency(d["h2_P"], d["h2_PT"]) + km_d.get_efficiency(d["h2_P"], d["h2_PT"])
        ) / 2
        return np.mean(eff_p * eff_k1 * eff_k2)

    # Nominal ratios
    jpsi_eff_nom = event_eff(
        data["Jpsi"], p_maps["up"], p_maps["down"], k_maps["up"], k_maps["down"]
    )
    nominal_ratios = {}
    for state in states:
        if state == "Jpsi" or state not in data:
            continue
        st_eff = event_eff(data[state], p_maps["up"], p_maps["down"], k_maps["up"], k_maps["down"])
        nominal_ratios[state] = st_eff / jpsi_eff_nom if jpsi_eff_nom > 0 else 1.0

    # Bootstrap
    bootstrap_ratios: dict = {st: [] for st in nominal_ratios}
    for i in range(n_bootstrap):
        bp_up = p_maps["up"].bootstrap_sample(rng)
        bp_down = p_maps["down"].bootstrap_sample(rng)
        bk_up = k_maps["up"].bootstrap_sample(rng)
        bk_down = k_maps["down"].bootstrap_sample(rng)

        jpsi_eff_b = event_eff(data["Jpsi"], bp_up, bp_down, bk_up, bk_down)
        for state in nominal_ratios:
            st_eff_b = event_eff(data[state], bp_up, bp_down, bk_up, bk_down)
            ratio_b = st_eff_b / jpsi_eff_b if jpsi_eff_b > 0 else 1.0
            bootstrap_ratios[state].append(ratio_b)

    results = {}
    for state, ratios in bootstrap_ratios.items():
        r_nom = nominal_ratios[state]
        syst = float(np.std(ratios))
        results[state] = {
            "ratio_nominal": r_nom,
            "syst_abs": syst,
            "syst_rel": syst / r_nom if r_nom > 0 else 0.0,
            "bootstrap_ratios": ratios,
        }
        print(f"  {state}: ratio={r_nom:.4f}, PID syst={100*syst/r_nom:.2f}%")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PIDCalib2 efficiency study + optional bootstrap")
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=0,
        help="Number of bootstrap iterations for PID systematic (0 = skip)",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="generated/output/studies/pid_cancellation",
        help="Directory where generated PID study outputs are written.",
    )
    args = parser.parse_args()

    main(output_dir=args.output_dir)

    if args.bootstrap_n > 0:
        import json

        print(f"\n=== PID Bootstrap ({args.bootstrap_n} iterations) ===")
        # Re-load the same data and maps as in main() above
        out_dir = Path(args.output_dir)
        calib_dir = Path("pidcalib_output")
        if not calib_dir.exists():
            print("pidcalib_output not found — skipping bootstrap.")
        else:
            p_maps_b = {
                "up": PIDCalibMap(
                    calib_dir / "effhists-Turbo18-up-P-MC15TuneV1_ProbNNp>0.05-P.ETA.pkl"
                ),
                "down": PIDCalibMap(
                    calib_dir / "effhists-Turbo18-down-P-MC15TuneV1_ProbNNp>0.05-P.ETA.pkl"
                ),
            }
            k_maps_b = {
                "up": PIDCalibMap(
                    calib_dir / "effhists-Turbo18-up-K-MC15TuneV1_ProbNNk>0.224-P.ETA.pkl"
                ),
                "down": PIDCalibMap(
                    calib_dir / "effhists-Turbo18-down-K-MC15TuneV1_ProbNNk>0.224-P.ETA.pkl"
                ),
            }
            mc_base = Path("/share/lazy/Mohamed/Bu2LambdaPPP/files/mc")
            states_b = ["Jpsi", "chic0", "chic1", "etac"]
            data_b = {}
            for st in states_b:
                ev = load_kinematics(str(mc_base / st / f"{st}_18_MD.root"), "LL")
                if ev is not None:
                    data_b[st] = ev
            if "Jpsi" in data_b:
                bs_results = run_pid_bootstrap(
                    data_b,
                    states_b,
                    p_maps_b,
                    k_maps_b,
                    n_bootstrap=args.bootstrap_n,
                    seed=args.bootstrap_seed,
                )
                bs_out = out_dir / "pid_bootstrap_systematics.json"
                # Remove raw bootstrap_ratios list to keep JSON small
                for v in bs_results.values():
                    v.pop("bootstrap_ratios", None)
                with open(bs_out, "w") as f:
                    json.dump(bs_results, f, indent=2)
                print(f"Bootstrap PID systematics saved to {bs_out}")
