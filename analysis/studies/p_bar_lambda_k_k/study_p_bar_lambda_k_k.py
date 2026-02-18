"""
Study: B+ -> p_bar Lambda K+ K+ Resonance Check

Reconstructs M(Lambda p_bar K+) from the B2L0PbarKpKp tree
to check for charmonium resonances (J/psi, eta_c) in this
conjugate-like decay mode.

Applies standard selection cuts to clean the signal.
"""

import sys
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import vector

# Add analysis root to path to import modules
current_dir = Path(__file__).parent
analysis_dir = current_dir.parent.parent
sys.path.insert(0, str(analysis_dir))

from modules.data_handler import DataManager, TOMLConfig
from modules.lambda_selector import LambdaSelector
from utils.logging_config import suppress_warnings

# Set styling
plt.style.use(hep.style.LHCb2)
suppress_warnings()


def calculate_derived_vars(events, config):
    """Calculate derived variables needed for cuts"""
    # 1. Delta Z used in Lambda Selection
    if all(b in events.fields for b in ["Bu_ENDVERTEX_Z", "L0_ENDVERTEX_Z"]):
        delta_z = events["L0_ENDVERTEX_Z"] - events["Bu_ENDVERTEX_Z"]
        events = ak.with_field(events, delta_z, "Delta_Z_mm")

    # 2. Corrected Mass for B+ (if not present)
    # Bu_MM_corrected = Bu_MM - L0_MM + M_PDG(Lambda)
    if (
        "Bu_MM_corrected" not in events.fields
        and "Bu_MM" in events.fields
        and "L0_MM" in events.fields
    ):
        lambda_mass = config.get_pdg_mass("lambda")
        corrected = events["Bu_MM"] - events["L0_MM"] + lambda_mass
        events = ak.with_field(events, corrected, "Bu_MM_corrected")

    return events


def apply_manual_cuts(events, config):
    """Apply manual/standard cuts from configuration"""
    cuts = config.selection.get("manual_cuts", {})
    if not cuts:
        print("Warning: No [manual_cuts] found in config. Using defaults.")
        cuts = {
            "Bu_DTF_chi2": {"cut_type": "less", "value": 30.0},
            "Bu_FDCHI2_OWNPV": {"cut_type": "greater", "value": 100.0},
            "Bu_IPCHI2_OWNPV": {"cut_type": "less", "value": 10.0},
            "Bu_PT": {"cut_type": "greater", "value": 3000.0},
            "h1_ProbNNk": {"cut_type": "greater", "value": 0.1},
            "h2_ProbNNk": {"cut_type": "greater", "value": 0.1},
            "p_ProbNNp": {"cut_type": "greater", "value": 0.1},
        }

    mask = ak.ones_like(events["Bu_MM"], dtype=bool)
    n_total = len(events)

    print("Applying standard cuts:")
    for branch, rule in cuts.items():
        if branch not in events.fields:
            print(f"  Skipping {branch} (not found)")
            continue

        val = rule["value"]
        if rule["cut_type"] == "less":
            mask = mask & (events[branch] < val)
            op = "<"
        else:
            mask = mask & (events[branch] > val)
            op = ">"

        print(f"  {branch} {op} {val}")

    n_pass = ak.sum(mask)
    print(f"  Result: {n_total} -> {n_pass} ({100*n_pass/n_total:.1f}%)")

    return events[mask]


def load_data(config):
    """Load B2L0PbarKpKp tree from all data years"""
    data_manager = DataManager(config)
    years = ["2016", "2017", "2018"]
    magnets = ["MD", "MU"]
    track_types = ["LL", "DD"]

    channel_name = "B2L0PbarKpKp"
    all_events = []

    print(f"Loading {channel_name} from data...")

    for year in years:
        for magnet in magnets:
            for track_type in track_types:
                try:
                    events = data_manager.load_tree(
                        particle_type="data",
                        year=year,
                        magnet=magnet,
                        track_type=track_type,
                        channel_name=channel_name,
                        apply_derived_branches=False,
                        apply_trigger=False,
                    )

                    if events is not None and len(events) > 0:
                        # Add metadata
                        events = ak.with_field(events, int(year), "year")
                        events = ak.with_field(events, track_type, "track_type")
                        events = calculate_derived_vars(events, config)
                        all_events.append(events)
                        print(f"  {year} {magnet} {track_type}: {len(events)} events")
                except Exception:
                    pass

    if not all_events:
        print("Warning: No data loaded!")
        return None

    print(f"Total raw data events: {sum(len(e) for e in all_events)}")
    return ak.concatenate(all_events)


def reconstruct_masses(events):
    """Reconstruct M(Lambda p_bar K) for both kaon combinations."""
    p4_L0 = vector.zip(
        {"px": events["L0_PX"], "py": events["L0_PY"], "pz": events["L0_PZ"], "E": events["L0_PE"]}
    )
    p4_p = vector.zip(
        {"px": events["p_PX"], "py": events["p_PY"], "pz": events["p_PZ"], "E": events["p_PE"]}
    )
    p4_h1 = vector.zip(
        {"px": events["h1_PX"], "py": events["h1_PY"], "pz": events["h1_PZ"], "E": events["h1_PE"]}
    )
    p4_h2 = vector.zip(
        {"px": events["h2_PX"], "py": events["h2_PY"], "pz": events["h2_PZ"], "E": events["h2_PE"]}
    )

    mass_1 = (p4_L0 + p4_p + p4_h1).mass
    mass_2 = (p4_L0 + p4_p + p4_h2).mass
    mass_B = (p4_L0 + p4_p + p4_h1 + p4_h2).mass

    return mass_1, mass_2, mass_B


def plot_mass_spectrum(mass_1, mass_2, mass_B, output_dir, n_events):
    """Plot the invariant mass spectrum with improved styling"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    all_masses = np.concatenate([mass_1, mass_2])
    m_min, m_max = 2800, 4000
    bins = 120

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 12), height_ratios=[3, 1], constrained_layout=True
    )

    # Header
    try:
        hep.lhcb.text("LHCb Data", ax=ax1, loc=0)
    except AttributeError:
        hep.label(data=True, label="LHCb Data", ax=ax1, loc=0)
    ax1.text(
        0.02,
        0.95,
        f"$B^+ \\to \\bar{{p}} \\Lambda K^+ K^+$\n(Standard Cuts)\n$N_{{events}}$: {n_events}",
        transform=ax1.transAxes,
        fontsize=14,
        verticalalignment="top",
    )

    # Plot 1: M(Lambda p_bar K)
    counts, edges, _ = ax1.hist(
        all_masses,
        bins=bins,
        range=(m_min, m_max),
        histtype="stepfilled",
        color="royalblue",
        alpha=0.6,
        label="Data Candidates",
    )

    # Resonances
    resonances = {
        r"$\eta_c(1S)$": 2984.1,
        r"$J/\psi(1S)$": 3096.9,
        r"$\chi_{c0}(1P)$": 3414.7,
        r"$\chi_{c1}(1P)$": 3510.7,
        r"$\eta_c(2S)$": 3637.8,
    }

    # Plot resonance lines with text instead of legend to reduce clutter
    colors = ["red", "darkorange", "green", "purple", "brown"]
    ylim = ax1.get_ylim()

    for (name, mass), color in zip(resonances.items(), colors):
        ax1.axvline(mass, color=color, linestyle="--", linewidth=1.5, alpha=0.8)
        # Add label near the top
        ax1.text(
            mass,
            ylim[1] * 0.9,
            name,
            rotation=90,
            color=color,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=12,
            fontweight="bold",
        )

    ax1.set_xlim(m_min, m_max)
    ax1.set_xlabel(r"$M(\Lambda \bar{p} K^+)$ [MeV/$c^2$]")
    ax1.set_ylabel(f"Candidates / {(m_max-m_min)/bins:.0f} MeV/$c^2$")
    ax1.legend(loc="upper right")

    # Plot 2: B+ Mass check
    b_min, b_max = 5150, 5450
    b_bins = 60

    ax2.hist(
        mass_B,
        bins=b_bins,
        range=(b_min, b_max),
        histtype="stepfilled",
        color="gray",
        alpha=0.5,
        label="Reconstructed $M(B^+)$",
    )
    ax2.set_xlabel("$M(B^+)$ [MeV/$c^2$]")
    ax2.set_ylabel("Events")
    ax2.set_xlim(b_min, b_max)

    # Mark nominal B+
    ax2.axvline(5279.34, color="red", linestyle="--", label="$M(B^+)_{PDG}$")
    ax2.legend(loc="upper right")

    plt.savefig(output_path / "mass_spectrum.pdf")
    print(f"Saved plot to {output_path / 'mass_spectrum.pdf'}")

    return all_masses


def count_resonances(masses, output_dir):
    """Count events in resonance windows"""
    windows = {
        "eta_c(1S)": (2984.1 - 50, 2984.1 + 50),
        "J/psi(1S)": (3096.9 - 50, 3096.9 + 50),
        "chi_c0(1P)": (3414.7 - 30, 3414.7 + 30),
        "chi_c1(1P)": (3510.7 - 30, 3510.7 + 30),
        "eta_c(2S)": (3637.8 - 30, 3637.8 + 30),
    }

    results = []
    for name, (low, high) in windows.items():
        count = np.sum((masses >= low) & (masses <= high))
        results.append(
            {"Resonance": name, "Mass Window [MeV]": f"{low:.1f} - {high:.1f}", "Candidates": count}
        )

    df = pd.DataFrame(results)
    output_path = Path(output_dir) / "summary.csv"
    df.to_csv(output_path, index=False)
    print("Resonance counts:")
    print(df)


def main():
    try:
        config_dir = snakemake.params.config_dir
        output_dir = snakemake.params.output_dir
    except NameError:
        config_dir = "../../config"
        output_dir = "output"

    config = TOMLConfig(config_dir)
    selector = LambdaSelector(config)

    try:
        # 1. Load Data
        events = load_data(config)

        # 2. Apply Lambda Selection (Fixed)
        print("\n=== Applying Lambda Selection ===")
        events = selector.apply_lambda_cuts(events)

        # 3. Apply B+ Fixed Selection (Mass window)
        print("\n=== Applying B+ Fixed Selection ===")
        events = selector.apply_bu_fixed_cuts(events)

        # 4. Apply Standard/Manual Cuts (PID, Kinematics)
        print("\n=== Applying Standard (Manual) Cuts ===")
        events = apply_manual_cuts(events, config)

        # 5. Reconstruct and Plot
        m1, m2, mB = reconstruct_masses(events)
        all_masses = plot_mass_spectrum(m1, m2, mB, output_dir, len(events))
        count_resonances(all_masses, output_dir)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
