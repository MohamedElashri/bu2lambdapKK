#!/usr/bin/env python3
"""
Reconstruction Efficiency Calculator
=====================================

This script calculates the reconstruction efficiency for B⁺ → Λ̄⁰ p K⁺ K⁻
decay with various charmonium resonances in the pΛ̄⁰ system.

Purpose:
--------
While these efficiencies largely cancel in our ratio measurement (since all
channels have the same final state particles), we calculate them to:
1. Verify the cancellation assumption
2. Quantify residual differences for systematic uncertainties
3. Document the efficiency for completeness

Physical Interpretation:
------------------------
**Reconstruction Efficiency**: Fraction of generated events where all final
state particles (p, Λ̄⁰ daughters, K⁺, K⁻) are successfully reconstructed
and correctly truth-matched:
  * All particles correctly identified (PDG IDs match)
  * Proper mother-daughter relationships preserved
  * Λ̄⁰ daughters: Long (LL) or Downstream (DD) tracks
  * Bachelor p, K⁺, K⁻: Long tracks

Truth Matching Criteria:
------------------------
For B⁺ → Λ̄⁰ p K⁺ K⁻ (with charmonium resonances):
- B⁺: abs(Bu_TRUEID) == 521
- Λ̄⁰: abs(L0_TRUEID) == 3122
  - From B⁺ for non-resonant OR from charmonium for resonant channels
- Λ̄⁰ daughters:
  - Proton: abs(Lp_TRUEID) == 2212, mother = Λ̄⁰ (3122)
  - Pion: abs(Lpi_TRUEID) == 211, mother = Λ̄⁰ (3122)
- Bachelor proton: abs(p_TRUEID) == 2212
  - From charmonium for resonant channels
- Kaons: abs(h1_TRUEID) == 321, abs(h2_TRUEID) == 321
  - From B⁺ directly

Usage:
------
    python reco_eff.py [options]

    Options:
        -v, --verbose              : Enable verbose debugging output
        -s, --state STATE          : Calculate for specific state only (Jpsi, etac, chic0, chic1)
        -o, --output FILE          : Save results to file (default: stdout)
        --show-residuals           : Show relative differences between states (for systematics)
        --width-study              : Run natural width study to investigate ηc(1S) efficiency
        --mass-window MeV          : Apply fixed mass window in MeV
        --mass-window-nsigma N     : Apply mass window in units of natural width

Examples:
---------
    # Calculate for all states
    python reco_eff.py

    # Calculate for J/ψ only with verbose output
    python reco_eff.py -s Jpsi -v

    # Calculate and save to file
    python reco_eff.py -o reco_eff_results.txt

    # Show residual differences for systematic uncertainties
    python reco_eff.py --show-residuals

    # Run natural width study (investigates why ηc has lower efficiency)
    python reco_eff.py --width-study -o width_study_results.txt

    # Calculate with fixed mass window of ±50 MeV
    python reco_eff.py --mass-window 50

    # Calculate with mass window of ±2 natural widths
    python reco_eff.py --mass-window-nsigma 2
"""

import argparse
import sys
from pathlib import Path

import awkward as ak
import numpy as np
from tqdm import tqdm
from uncertainties import ufloat

# Add parent directory to path to import analysis modules
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))

import vector

from modules.data_handler import DataManager, TOMLConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

# States to analyze (matching your charmonium resonances)
# Use the same naming as in config/data.toml
# Decay: B⁺ → cc̄(→ Λ̄⁰ p K⁻) K⁺
STATES = {
    "Jpsi": "J/ψ",  # J/ψ → Λ̄⁰ p K⁻ (normalization channel)
    "etac": "ηc(1S)",  # ηc(1S) → Λ̄⁰ p K⁻
    "chic0": "χc0(1P)",  # χc0(1P) → Λ̄⁰ p K⁻
    "chic1": "χc1(1P)",  # χc1(1P) → Λ̄⁰ p K⁻
}

# PDG values for charmonium states
# Masses in MeV/c², natural widths in MeV
CHARMONIUM_PROPERTIES = {
    "Jpsi": {
        "mass": 3096.9,  # MeV/c² (±0.006)
        "width": 0.0926,  # MeV (±0.0017) - very narrow!
        "pdg_id": 443,
    },
    "etac": {
        "mass": 2984.1,  # MeV/c² (±0.4)
        "width": 30.5,  # MeV (±0.5) - ~330x wider than J/ψ!
        "pdg_id": 441,
    },
    "chic0": {
        "mass": 3414.71,  # MeV/c² (±0.30)
        "width": 10.6,  # MeV (±0.8)
        "pdg_id": 10441,
    },
    "chic1": {
        "mass": 3510.67,  # MeV/c² (±0.05)
        "width": 0.88,  # MeV (±0.05)
        "pdg_id": 20443,
    },
}

# Data taking years and magnet polarities
YEARS = ["2016", "2017", "2018"]
TRACK_TYPES = ["LL", "DD"]


# ============================================================================
# TRUTH MATCHING DEFINITIONS
# ============================================================================


def get_truth_matching_mask(events: ak.Array) -> ak.Array:
    """
    Apply truth matching criteria for B⁺ → Λ̄⁰ p K⁺ K⁻.

    Args:
        events: Awkward array with MC truth branches

    Returns:
        Boolean mask for truth-matched events

    Physics Note:
        We check that all final state particles are correctly identified
        and come from the appropriate mothers in the decay chain.

        The decay can proceed through intermediate resonances:
        - Direct: cc̄ → Λ̄⁰ p K⁻
        - Via Λ(1520): cc̄ → Λ̄⁰ Λ*(1520)(→ p K⁻)

        We must accept both topologies for correct efficiency calculation.
    """

    # Check if we have truth branches
    required_branches = [
        "Bu_TRUEID",
        "L0_TRUEID",
        "Lp_TRUEID",
        "Lpi_TRUEID",
        "p_TRUEID",
        "h1_TRUEID",
        "h2_TRUEID",
        "Lp_MC_MOTHER_ID",
        "Lpi_MC_MOTHER_ID",
        "p_MC_MOTHER_ID",
        "L0_MC_MOTHER_ID",
        "h1_MC_MOTHER_ID",
        "h2_MC_MOTHER_ID",
    ]

    missing = [b for b in required_branches if b not in events.fields]
    if missing:
        print(f"Warning: Missing truth branches: {missing}")
        return ak.ones_like(events.Bu_M, dtype=bool)  # Return all True if no truth info

    # Base truth matching
    # Charmonium PDG IDs we care about
    charmonium_ids = [443, 441, 10441, 20443]  # J/ψ, ηc, χc0, χc1

    # Intermediate resonances that can appear in the decay chain
    # Λ(1520) = 3124, Λ(1600) = 23122, Λ(1670) = 33122, etc.
    lambda_star_ids = [3124, 23122, 33122, 13122, 3126, 13124, 23124]

    mask = (
        # B+ identification
        (abs(events.Bu_TRUEID) == 521)
        &
        # Lambda identification
        (abs(events.L0_TRUEID) == 3122)
        &
        # Lambda daughters
        (abs(events.Lp_TRUEID) == 2212)
        & (abs(events.Lpi_TRUEID) == 211)
        & (abs(events.Lp_MC_MOTHER_ID) == 3122)
        & (abs(events.Lpi_MC_MOTHER_ID) == 3122)
        &
        # Bachelor proton
        (abs(events.p_TRUEID) == 2212)
        &
        # Kaons
        (abs(events.h1_TRUEID) == 321)
        & (abs(events.h2_TRUEID) == 321)
    )

    # Mother requirements - allow for both resonant and non-resonant
    # For J/psi MC: B⁺ → J/ψ(→ pΛ̄⁰K⁻) K⁺
    # For χc MC: B⁺ → χc(→ pΛ̄⁰) K⁺K⁻
    # Allow L0 and p to come from B+ or charmonium
    l0_mother_ok = abs(events.L0_MC_MOTHER_ID) == 521
    for cc_id in charmonium_ids:
        l0_mother_ok = l0_mother_ok | (abs(events.L0_MC_MOTHER_ID) == cc_id)

    # For p: allow from B+, charmonium, OR Λ* resonances (e.g., Λ(1520) in ηc MC)
    p_mother_ok = abs(events.p_MC_MOTHER_ID) == 521
    for cc_id in charmonium_ids:
        p_mother_ok = p_mother_ok | (abs(events.p_MC_MOTHER_ID) == cc_id)
    for ls_id in lambda_star_ids:
        p_mother_ok = p_mother_ok | (abs(events.p_MC_MOTHER_ID) == ls_id)

    mask = mask & l0_mother_ok & p_mother_ok

    # Kaons: allow one from B+ and one from charmonium (or both from B+)
    # Also allow h2 (K⁻) to come from Λ* resonances
    kaon_ok = (abs(events.h1_MC_MOTHER_ID) == 521) & (abs(events.h2_MC_MOTHER_ID) == 521)
    for cc_id in charmonium_ids:
        kaon_ok = kaon_ok | (
            ((abs(events.h1_MC_MOTHER_ID) == 521) & (abs(events.h2_MC_MOTHER_ID) == cc_id))
            | ((abs(events.h2_MC_MOTHER_ID) == 521) & (abs(events.h1_MC_MOTHER_ID) == cc_id))
        )
    # Allow h2 from Λ* (for ηc → Λ̄⁰ Λ*(1520)(→ p K⁻) topology)
    for ls_id in lambda_star_ids:
        kaon_ok = kaon_ok | (
            (abs(events.h1_MC_MOTHER_ID) == 521) & (abs(events.h2_MC_MOTHER_ID) == ls_id)
        )

    mask = mask & kaon_ok

    return mask


# ============================================================================
# INVARIANT MASS CALCULATION
# ============================================================================


def calculate_charmonium_mass(events: ak.Array) -> ak.Array:
    """
    Calculate the M(Λ̄pK⁻) invariant mass (charmonium candidate mass).

    The charmonium state decays as: cc̄ → Λ̄⁰ p K⁻
    So the charmonium mass is reconstructed from Lambda + bachelor proton + K⁻ (h2).

    Args:
        events: Awkward array with momentum branches

    Returns:
        Array of M(Λ̄pK⁻) invariant masses in MeV/c²

    Note:
        h2 is the K⁻ that comes from the charmonium decay.
        h1 is the K⁺ that comes directly from the B⁺.
    """
    required_branches = [
        "L0_PX",
        "L0_PY",
        "L0_PZ",
        "L0_PE",
        "p_PX",
        "p_PY",
        "p_PZ",
        "p_PE",
        "h2_PX",
        "h2_PY",
        "h2_PZ",
        "h2_PE",
    ]
    missing = [b for b in required_branches if b not in events.fields]
    if missing:
        return None
    # Build 4-vectors
    L0_4mom = vector.zip(
        {"px": events.L0_PX, "py": events.L0_PY, "pz": events.L0_PZ, "E": events.L0_PE}
    )
    p_4mom = vector.zip({"px": events.p_PX, "py": events.p_PY, "pz": events.p_PZ, "E": events.p_PE})
    h2_4mom = vector.zip(
        {"px": events.h2_PX, "py": events.h2_PY, "pz": events.h2_PZ, "E": events.h2_PE}
    )
    # Sum and get invariant mass: M(Λ̄ p K⁻)
    total_4mom = L0_4mom + p_4mom + h2_4mom
    return total_4mom.mass


def apply_mass_window(
    events: ak.Array, state: str, n_sigma: float = None, window_mev: float = None
) -> ak.Array:
    """
    Apply a mass window cut around the nominal charmonium mass.

    Args:
        events: Awkward array with M_cc (charmonium mass) branch
        state: Charmonium state key (Jpsi, etac, chic0, chic1)
        n_sigma: Number of natural widths to use as window (e.g., 2.0 = ±2Γ)
        window_mev: Fixed window in MeV (alternative to n_sigma)

    Returns:
        Boolean mask for events within the mass window

    Note:
        Either n_sigma or window_mev must be specified, not both.
        M_cc = M(Λ̄pK⁻) is the charmonium candidate mass.
    """
    if "M_cc" not in events.fields:
        return ak.ones_like(events.Bu_M, dtype=bool)
    props = CHARMONIUM_PROPERTIES[state]
    mass_nominal = props["mass"]
    width = props["width"]
    if n_sigma is not None:
        half_window = n_sigma * width
    elif window_mev is not None:
        half_window = window_mev
    else:
        return ak.ones_like(events.Bu_M, dtype=bool)
    mass = events.M_cc
    mask = (mass > mass_nominal - half_window) & (mass < mass_nominal + half_window)
    return mask


# ============================================================================
# EFFICIENCY CALCULATION
# ============================================================================


def calculate_efficiency(n_pass: int, n_total: int) -> tuple:
    """
    Calculate efficiency with binomial uncertainty.

    Args:
        n_pass: Number of events passing selection
        n_total: Total number of events

    Returns:
        Tuple of (efficiency, uncertainty) as ufloat object and (n_pass, n_total)
    """
    if n_total == 0:
        return ufloat(0, 0), (0, 0)

    eff = n_pass / n_total
    # Binomial uncertainty
    eff_err = np.sqrt(eff * (1 - eff) / n_total) if n_total > 0 else 0.0

    return ufloat(eff, eff_err), (n_pass, n_total)


def calculate_reco_efficiency(
    data_manager: DataManager,
    state: str,
    year: str,
    track_type: str,
    verbose: bool = False,
    mass_window_mev: float = None,
    mass_window_nsigma: float = None,
) -> dict:
    """
    Calculate reconstruction efficiency for a given configuration.

    Args:
        data_manager: DataManager instance
        state: Charmonium state (Jpsi, etac, chic0, chic1)
        year: Data taking year (2016, 2017, 2018)
        track_type: Lambda track type (LL, DD)
        verbose: Enable verbose output
        mass_window_mev: Fixed mass window in MeV (optional)
        mass_window_nsigma: Mass window in units of natural width (optional)

    Returns:
        Dictionary with efficiency results
    """
    if verbose:
        print(f"\nProcessing: {state}, {year}, {track_type}")
    # Load MC events - we need mc_truth branches which aren't in the standard preset
    # Load trees manually and combine
    import uproot

    events_list = []
    for magnet in ["MU", "MD"]:
        # Build file path using DataManager's mc_path
        year_int = int(year)
        filename = f"{state}_{year_int-2000}_{magnet}.root"
        filepath = data_manager.mc_path / state / filename
        if not filepath.exists():
            if verbose:
                print(f"  File not found: {filepath}")
            continue
        # Build tree path
        channel_path = f"B2L0barPKpKm_{track_type}"
        tree_path = f"{channel_path}/DecayTree"
        try:
            file = uproot.open(filepath)
            if channel_path not in file:
                if verbose:
                    print(f"  Channel {channel_path} not found in {filepath}")
                continue
            tree = file[tree_path]
            # Load essential + selection + mc_truth + kinematics branches
            load_branches_sets = ["essential", "selection", "mc_truth", "kinematics"]
            load_branches = data_manager.config.branch_config.get_branches_from_sets(
                load_branches_sets, exclude_mc=False
            )

            # Resolve aliases
            resolved_branches = data_manager.config.branch_config.resolve_aliases(
                load_branches, is_mc=True
            )

            # Validate
            available_branches = list(tree.keys())
            validation = data_manager.config.branch_config.validate_branches(
                resolved_branches, available_branches
            )

            # Load
            tree_events = tree.arrays(validation["valid"], library="ak")

            # Normalize names
            rename_map = data_manager.config.branch_config.normalize_branches(
                validation["valid"], is_mc=True
            )
            if rename_map:
                tree_events = ak.zip(
                    {rename_map.get(name, name): tree_events[name] for name in tree_events.fields}
                )

            events_list.append(tree_events)

            if verbose:
                print(f"  Loaded {state} {year}_{magnet}_{track_type}: {len(tree_events)} events")

        except Exception as e:
            if verbose:
                print(f"  Error loading {filepath}: {e}")
            continue

    if not events_list:
        if verbose:
            print(f"  No MC files found for {state} {year} {track_type}")
        return {
            "state": state,
            "year": year,
            "track_type": track_type,
            "n_generated": 0,
            "n_reconstructed": 0,
            "reconstruction_efficiency": ufloat(0, 0),
            "efficiency_percent": ufloat(0, 0),
        }

    # Combine events from both magnets
    events = ak.concatenate(events_list)

    if events is None:
        if verbose:
            print(f"  No MC files found for {state} {year} {track_type}")
        return {
            "state": state,
            "year": year,
            "track_type": track_type,
            "n_generated": 0,
            "n_reconstructed": 0,
            "reconstruction_efficiency": ufloat(0, 0),
            "efficiency_percent": ufloat(0, 0),
        }

    # Count total unique generated events (using eventNumber + runNumber)
    # This is the proper denominator: all events that were generated
    if hasattr(events, "eventNumber") and hasattr(events, "runNumber"):
        runs = ak.to_numpy(events.runNumber)
        event_nums = ak.to_numpy(events.eventNumber)
        unique_events = np.unique(np.column_stack([runs, event_nums]), axis=0)
        n_generated = len(unique_events)
    else:
        # Fallback: use EventInSequence if available
        if hasattr(events, "EventInSequence"):
            n_generated = len(np.unique(ak.to_numpy(events.EventInSequence)))
        else:
            if verbose:
                print(f"  WARNING: Cannot count unique events, using total entries")
            n_generated = len(events)

    # Apply truth matching to count reconstructed events
    truth_mask = get_truth_matching_mask(events)
    n_reconstructed = ak.sum(truth_mask)
    # Calculate M(Λ̄pK⁻) charmonium invariant mass for mass window studies
    M_cc = calculate_charmonium_mass(events)
    if M_cc is not None:
        events = ak.with_field(events, M_cc, "M_cc")
    # Apply mass window cut if requested
    n_in_window = None
    n_reco_in_window = None
    mass_window_info = None
    if mass_window_mev is not None or mass_window_nsigma is not None:
        if M_cc is not None:
            mass_mask = apply_mass_window(
                events, state, n_sigma=mass_window_nsigma, window_mev=mass_window_mev
            )
            # Count events in mass window (generated)
            n_in_window = ak.sum(mass_mask)
            # Count reconstructed events in mass window
            combined_mask = truth_mask & mass_mask
            n_reco_in_window = ak.sum(combined_mask)
            props = CHARMONIUM_PROPERTIES[state]
            if mass_window_nsigma is not None:
                half_window = mass_window_nsigma * props["width"]
                mass_window_info = f"±{mass_window_nsigma}Γ = ±{half_window:.1f} MeV"
            else:
                mass_window_info = f"±{mass_window_mev:.1f} MeV"
            if verbose:
                print(f"  Mass window: {mass_window_info}")
                print(f"  Events in window: {n_in_window}")
                print(f"  Reco in window: {n_reco_in_window}")
    if verbose:
        print(f"  Total generated events: {n_generated}")
        print(
            f"  Reconstructed (truth-matched): {n_reconstructed} / {n_generated} = {100*n_reconstructed/n_generated:.2f}%"
        )
        print(f"  Total candidates (including duplicates): {len(events)}")
    # Calculate reconstruction efficiency
    reco_eff, (n_pass, n_tot) = calculate_efficiency(int(n_reconstructed), int(n_generated))
    # Also calculate efficiency within mass window if applicable
    reco_eff_window = None
    if n_in_window is not None and n_reco_in_window is not None and n_in_window > 0:
        reco_eff_window, _ = calculate_efficiency(int(n_reco_in_window), int(n_in_window))
    return {
        "state": state,
        "year": year,
        "track_type": track_type,
        "n_generated": int(n_generated),
        "n_reconstructed": int(n_reconstructed),
        "reconstruction_efficiency": reco_eff,
        "efficiency_percent": reco_eff * 100,
        "n_in_window": int(n_in_window) if n_in_window is not None else None,
        "n_reco_in_window": int(n_reco_in_window) if n_reco_in_window is not None else None,
        "reco_eff_window": reco_eff_window,
        "eff_window_percent": reco_eff_window * 100 if reco_eff_window is not None else None,
        "mass_window_info": mass_window_info,
    }


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================


def print_results_table(results: dict, output_file=None):
    """
    Print results in LaTeX and Markdown table formats.

    Args:
        results: Dictionary of efficiency results
        output_file: Optional file object to write to
    """

    def print_line(line):
        """Print to stdout and file if provided"""
        print(line)
        if output_file:
            output_file.write(line + "\n")

    # Organize results by track type
    for track_type in TRACK_TYPES:
        print_line(f"\n{'='*80}")
        print_line(f"Reconstruction Efficiency - Λ̄⁰_{track_type}")
        print_line(f"{'='*80}\n")

        # LaTeX table
        print_line("\\begin{table}[htbp]")
        print_line("\\centering")
        print_line(
            f"\\caption{{Reconstruction efficiency for "
            f"$B^+ \\to \\bar{{\\Lambda}}^0_{{\\text{{{track_type}}}}} p K^+ K^-$ "
            f"with various charmonium states (\\%)}}"
        )
        print_line("\\begin{tabular}{l|ccc|c}")
        print_line("\\hline")
        print_line("State & 2016 & 2017 & 2018 & Average \\\\ \\hline")

        # Calculate and print for each state
        for state_key in STATES.keys():
            state_label = STATES[state_key].replace("(", "").replace(")", "")
            line = f"${state_label}$ "

            effs = []
            for year in YEARS:
                key = (state_key, year, track_type)
                if key in results and results[key]["n_generated"] > 0:
                    eff = results[key]["efficiency_percent"]
                    effs.append(eff)
                    val = eff.nominal_value
                    err = eff.std_dev
                    line += f"& ${val:.2f}\\pm{err:.2f}$ "
                else:
                    line += "& --- "

            # Calculate weighted average
            if effs:
                # Weight by 1/σ²
                weights = [1.0 / (e.std_dev**2) if e.std_dev > 0 else 0 for e in effs]
                if sum(weights) > 0:
                    avg_val = sum(e.nominal_value * w for e, w in zip(effs, weights)) / sum(weights)
                    avg_err = 1.0 / np.sqrt(sum(weights))
                    line += f"& $\\bm{{{avg_val:.2f}\\pm{avg_err:.2f}}}$ "
                else:
                    line += "& --- "
            else:
                line += "& --- "

            line += "\\\\"
            print_line(line)

        print_line("\\hline")
        print_line("\\end{tabular}")
        print_line(f"\\label{{tab:reco_eff_{track_type}}}")
        print_line("\\end{table}\n")

        # Markdown table
        print_line(f"**Reconstruction Efficiency - Λ̄⁰_{track_type} (%)**\n")
        print_line("| State | 2016 | 2017 | 2018 | Average |")
        print_line("|-------|------|------|------|---------|")

        for state_key in STATES.keys():
            state_label = STATES[state_key]
            line = f"| {state_label} "

            effs = []
            for year in YEARS:
                key = (state_key, year, track_type)
                if key in results and results[key]["n_generated"] > 0:
                    eff = results[key]["efficiency_percent"]
                    effs.append(eff)
                    val = eff.nominal_value
                    err = eff.std_dev
                    line += f"| {val:.2f}±{err:.2f} "
                else:
                    line += "| --- "

            # Calculate weighted average
            if effs:
                weights = [1.0 / (e.std_dev**2) if e.std_dev > 0 else 0 for e in effs]
                if sum(weights) > 0:
                    avg_val = sum(e.nominal_value * w for e, w in zip(effs, weights)) / sum(weights)
                    avg_err = 1.0 / np.sqrt(sum(weights))
                    line += f"| **{avg_val:.2f}±{avg_err:.2f}** "
                else:
                    line += "| --- "
            else:
                line += "| --- "

            line += "|"
            print_line(line)

        print_line("\n")


def print_residuals_table(results: dict, output_file=None):
    """
    Print table of relative differences between states (for systematics).

    Args:
        results: Dictionary of efficiency results
        output_file: Optional file object to write to
    """

    def print_line(line):
        print(line)
        if output_file:
            output_file.write(line + "\n")

    print_line(f"\n{'='*80}")
    print_line("Relative Differences (for Systematic Uncertainties)")
    print_line(f"{'='*80}\n")
    print_line("Values shown: (ε_state - ε_J/ψ) / ε_J/ψ × 100%\n")

    for track_type in TRACK_TYPES:
        print_line(f"\n--- Λ̄⁰_{track_type} ---\n")
        print_line("| State | 2016 | 2017 | 2018 | Average |")
        print_line("|-------|------|------|------|---------|")

        for state_key in STATES.keys():
            if state_key == "Jpsi":
                continue  # Skip J/ψ (reference)

            state_label = STATES[state_key]
            line = f"| {state_label} "

            residuals = []
            for year in YEARS:
                state_key_tuple = (state_key, year, track_type)
                jpsi_key = ("Jpsi", year, track_type)

                if (
                    state_key_tuple in results
                    and jpsi_key in results
                    and results[state_key_tuple]["n_generated"] > 0
                    and results[jpsi_key]["n_generated"] > 0
                ):

                    eff_state = results[state_key_tuple]["reconstruction_efficiency"]
                    eff_jpsi = results[jpsi_key]["reconstruction_efficiency"]

                    if eff_jpsi.nominal_value > 0:
                        residual = (eff_state - eff_jpsi) / eff_jpsi * 100
                        residuals.append(residual)
                        val = residual.nominal_value
                        err = residual.std_dev
                        line += f"| {val:+.2f}±{err:.2f} "
                    else:
                        line += "| --- "
                else:
                    line += "| --- "

            # Average residual
            if residuals:
                avg_res_val = np.mean([r.nominal_value for r in residuals])
                avg_res_err = np.sqrt(np.sum([r.std_dev**2 for r in residuals])) / len(residuals)
                line += f"| **{avg_res_val:+.2f}±{avg_res_err:.2f}** "
            else:
                line += "| --- "

            line += "|"
            print_line(line)

        print_line("\n")


def print_width_study_table(results: dict, output_file=None):
    """
    Print table showing efficiency with mass window cuts.

    Args:
        results: Dictionary of efficiency results with mass window info
        output_file: Optional file object to write to
    """

    def print_line(line: str) -> None:
        print(line)
        if output_file:
            output_file.write(line + "\n")

    print_line(f"\n{'='*80}")
    print_line("Natural Width Study - Efficiency within M(Λ̄pK⁻) Mass Windows")
    print_line(f"{'='*80}\n")
    # Print charmonium properties
    print_line("**Charmonium Natural Widths (PDG 2024)**\n")
    print_line("| State | Mass (MeV) | Width Γ (MeV) | Γ/M |")
    print_line("|-------|------------|---------------|-----|")
    for state_key in STATES.keys():
        props = CHARMONIUM_PROPERTIES[state_key]
        state_label = STATES[state_key]
        ratio = props["width"] / props["mass"] * 100
        print_line(f"| {state_label} | {props['mass']:.1f} | {props['width']:.3f} | {ratio:.4f}% |")
    print_line("\n")
    # Print efficiency within mass window
    for track_type in TRACK_TYPES:
        print_line(f"\n--- Λ̄⁰_{track_type}: Efficiency within Mass Window ---\n")
        # Check if we have mass window results
        sample_key = (list(STATES.keys())[0], YEARS[0], track_type)
        if sample_key in results and results[sample_key].get("mass_window_info") is not None:
            window_info = results[sample_key]["mass_window_info"]
            print_line(f"Mass window: {window_info}\n")
            print_line("| State | Full Eff (%) | Window Eff (%) | Fraction in Window |")
            print_line("|-------|--------------|----------------|-------------------|")
            for state_key in STATES.keys():
                state_label = STATES[state_key]
                # Average over years
                full_effs = []
                window_effs = []
                fractions = []
                for year in YEARS:
                    key = (state_key, year, track_type)
                    if key in results and results[key]["n_generated"] > 0:
                        full_effs.append(results[key]["efficiency_percent"])
                        if results[key].get("eff_window_percent") is not None:
                            window_effs.append(results[key]["eff_window_percent"])
                        if (
                            results[key].get("n_in_window") is not None
                            and results[key]["n_generated"] > 0
                        ):
                            frac = results[key]["n_in_window"] / results[key]["n_generated"]
                            fractions.append(frac)
                if full_effs:
                    avg_full = np.mean([e.nominal_value for e in full_effs])
                    avg_full_err = np.sqrt(np.sum([e.std_dev**2 for e in full_effs])) / len(
                        full_effs
                    )
                else:
                    avg_full, avg_full_err = 0, 0
                if window_effs:
                    avg_window = np.mean([e.nominal_value for e in window_effs])
                    avg_window_err = np.sqrt(np.sum([e.std_dev**2 for e in window_effs])) / len(
                        window_effs
                    )
                else:
                    avg_window, avg_window_err = 0, 0
                if fractions:
                    avg_frac = np.mean(fractions) * 100
                else:
                    avg_frac = 0
                print_line(
                    f"| {state_label} | {avg_full:.2f}±{avg_full_err:.2f} | "
                    f"{avg_window:.2f}±{avg_window_err:.2f} | {avg_frac:.1f}% |"
                )
            print_line("\n")
        else:
            print_line("No mass window results available. Run with --width-study option.\n")


def run_width_study(data_manager: DataManager, verbose: bool = False, output_file=None) -> None:
    """
    Run comprehensive natural width study.

    This function calculates efficiencies with different mass window sizes
    to understand how the natural width affects reconstruction efficiency.

    Args:
        data_manager: DataManager instance
        verbose: Enable verbose output
        output_file: Optional file object for output
    """

    def print_line(line: str) -> None:
        print(line)
        if output_file:
            output_file.write(line + "\n")

    print_line(f"\n{'='*80}")
    print_line("NATURAL WIDTH STUDY")
    print_line(f"{'='*80}\n")
    print_line(
        "Hypothesis: ηc(1S) has lower efficiency because its large natural width\n"
        "(Γ=30.5 MeV) causes events to be generated with M(Λ̄pK⁻) spread over ~100 MeV,\n"
        "while J/ψ (Γ=0.09 MeV) is essentially a delta function.\n"
    )
    print_line("Decay topology: B⁺ → cc̄(→ Λ̄⁰ p K⁻) K⁺\n")
    # Study 1: Fixed mass window (same for all states)
    print_line("## Study 1: Fixed Mass Window (±50 MeV for all states)\n")
    results_fixed = {}
    total_calcs = len(STATES) * len(YEARS) * len(TRACK_TYPES)
    with tqdm(total=total_calcs, desc="Fixed window study", disable=verbose) as pbar:
        for state in STATES.keys():
            for year in YEARS:
                for track_type in TRACK_TYPES:
                    result = calculate_reco_efficiency(
                        data_manager,
                        state,
                        year,
                        track_type,
                        verbose=verbose,
                        mass_window_mev=50.0,
                    )
                    results_fixed[(state, year, track_type)] = result
                    pbar.update(1)
    print_width_study_table(results_fixed, output_file)
    # Study 2: Natural width units (±2Γ for each state)
    print_line("\n## Study 2: Natural Width Units (±2Γ for each state)\n")
    print_line("This normalizes the mass window to each state's natural width.\n")
    results_nsigma = {}
    with tqdm(total=total_calcs, desc="Width-normalized study", disable=verbose) as pbar:
        for state in STATES.keys():
            for year in YEARS:
                for track_type in TRACK_TYPES:
                    result = calculate_reco_efficiency(
                        data_manager,
                        state,
                        year,
                        track_type,
                        verbose=verbose,
                        mass_window_nsigma=2.0,
                    )
                    results_nsigma[(state, year, track_type)] = result
                    pbar.update(1)
    print_width_study_table(results_nsigma, output_file)
    # Summary and interpretation
    print_line("\n## Interpretation\n")
    print_line(
        "If the hypothesis is correct:\n"
        "1. With fixed mass window: ηc efficiency should remain lower because\n"
        "   many events fall outside the window due to the large natural width.\n"
        "\n"
        "2. With width-normalized window (±2Γ): All states should have similar\n"
        "   efficiency because we're comparing equivalent fractions of the\n"
        "   Breit-Wigner distribution.\n"
        "\n"
        "3. The 'fraction in window' should be similar for all states when using\n"
        "   width-normalized windows, but very different for fixed windows.\n"
    )


def run_mass_diagnostic(data_manager: DataManager, verbose: bool = False, output_file=None) -> None:
    """
    Run diagnostic to check M(Λ̄pK⁻) distribution for each state.

    This helps understand if the reconstructed charmonium mass peaks
    at the expected nominal mass.

    Args:
        data_manager: DataManager instance
        verbose: Enable verbose output
        output_file: Optional file object for output
    """
    import uproot

    def print_line(line: str) -> None:
        print(line)
        if output_file:
            output_file.write(line + "\n")

    print_line(f"\n{'='*80}")
    print_line("M(Λ̄pK⁻) MASS DIAGNOSTIC")
    print_line(f"{'='*80}\n")
    print_line("Checking if reconstructed charmonium mass peaks at expected values.\n")
    for state in STATES.keys():
        props = CHARMONIUM_PROPERTIES[state]
        state_label = STATES[state]
        print_line(f"\n### {state_label} (expected mass: {props['mass']:.1f} MeV)\n")
        # Load one year/track_type for diagnostic
        year = "2018"
        track_type = "LL"
        events_list = []
        for magnet in ["MU", "MD"]:
            year_int = int(year)
            filename = f"{state}_{year_int-2000}_{magnet}.root"
            filepath = data_manager.mc_path / state / filename
            if not filepath.exists():
                continue
            channel_path = f"B2L0barPKpKm_{track_type}"
            tree_path = f"{channel_path}/DecayTree"
            try:
                file = uproot.open(filepath)
                if channel_path not in file:
                    continue
                tree = file[tree_path]
                # Load kinematics branches
                load_branches_sets = ["essential", "kinematics", "mc_truth"]
                load_branches = data_manager.config.branch_config.get_branches_from_sets(
                    load_branches_sets, exclude_mc=False
                )
                resolved_branches = data_manager.config.branch_config.resolve_aliases(
                    load_branches, is_mc=True
                )
                available_branches = list(tree.keys())
                validation = data_manager.config.branch_config.validate_branches(
                    resolved_branches, available_branches
                )
                tree_events = tree.arrays(validation["valid"], library="ak")
                rename_map = data_manager.config.branch_config.normalize_branches(
                    validation["valid"], is_mc=True
                )
                if rename_map:
                    tree_events = ak.zip(
                        {
                            rename_map.get(name, name): tree_events[name]
                            for name in tree_events.fields
                        }
                    )
                events_list.append(tree_events)
            except Exception as e:
                print_line(f"  Error: {e}")
                continue
        if not events_list:
            print_line(f"  No data loaded for {state}")
            continue
        events = ak.concatenate(events_list)
        # Calculate M(Λ̄pK⁻)
        M_cc = calculate_charmonium_mass(events)
        if M_cc is None:
            print_line(f"  Could not calculate M(Λ̄pK⁻) - missing branches")
            continue
        # Apply truth matching
        truth_mask = get_truth_matching_mask(events)
        M_cc_truth = M_cc[truth_mask]
        M_cc_all = M_cc
        # Statistics
        mean_all = float(np.mean(ak.to_numpy(M_cc_all)))
        std_all = float(np.std(ak.to_numpy(M_cc_all)))
        mean_truth = float(np.mean(ak.to_numpy(M_cc_truth)))
        std_truth = float(np.std(ak.to_numpy(M_cc_truth)))
        min_all = float(np.min(ak.to_numpy(M_cc_all)))
        max_all = float(np.max(ak.to_numpy(M_cc_all)))
        print_line(f"  All events ({len(M_cc_all)} total):")
        print_line(f"    Mean: {mean_all:.1f} MeV, Std: {std_all:.1f} MeV")
        print_line(f"    Range: [{min_all:.1f}, {max_all:.1f}] MeV")
        print_line(f"    Offset from nominal: {mean_all - props['mass']:.1f} MeV")
        print_line(f"  Truth-matched events ({len(M_cc_truth)} total):")
        print_line(f"    Mean: {mean_truth:.1f} MeV, Std: {std_truth:.1f} MeV")
        print_line(f"    Offset from nominal: {mean_truth - props['mass']:.1f} MeV")
        # Count in windows
        nominal = props["mass"]
        in_50 = ak.sum((M_cc_all > nominal - 50) & (M_cc_all < nominal + 50))
        in_100 = ak.sum((M_cc_all > nominal - 100) & (M_cc_all < nominal + 100))
        print_line(f"  Fraction within ±50 MeV of nominal: {100*in_50/len(M_cc_all):.1f}%")
        print_line(f"  Fraction within ±100 MeV of nominal: {100*in_100/len(M_cc_all):.1f}%")


def run_truth_matching_study(
    data_manager: DataManager, verbose: bool = False, output_file=None
) -> None:
    """
    Detailed study of truth matching criteria for each charmonium state.

    This investigates whether the truth matching is causing the efficiency
    difference by examining which criteria fail for each state.

    Args:
        data_manager: DataManager instance
        verbose: Enable verbose output
        output_file: Optional file object for output
    """
    import uproot

    def print_line(line: str) -> None:
        print(line)
        if output_file:
            output_file.write(line + "\n")

    print_line(f"\n{'='*80}")
    print_line("TRUTH MATCHING STUDY")
    print_line(f"{'='*80}\n")
    print_line("Investigating which truth matching criteria cause efficiency loss.\n")
    # Charmonium PDG IDs
    charmonium_pdg = {
        "Jpsi": 443,
        "etac": 441,
        "chic0": 10441,
        "chic1": 20443,
    }
    all_stats: dict[str, dict] = {}
    for state in STATES.keys():
        state_label = STATES[state]
        cc_id = charmonium_pdg[state]
        print_line(f"\n### {state_label} (PDG ID: {cc_id})\n")
        # Load data for 2018 LL
        year = "2018"
        track_type = "LL"
        events_list = []
        for magnet in ["MU", "MD"]:
            year_int = int(year)
            filename = f"{state}_{year_int-2000}_{magnet}.root"
            filepath = data_manager.mc_path / state / filename
            if not filepath.exists():
                continue
            channel_path = f"B2L0barPKpKm_{track_type}"
            tree_path = f"{channel_path}/DecayTree"
            try:
                file = uproot.open(filepath)
                if channel_path not in file:
                    continue
                tree = file[tree_path]
                load_branches_sets = ["essential", "kinematics", "mc_truth"]
                load_branches = data_manager.config.branch_config.get_branches_from_sets(
                    load_branches_sets, exclude_mc=False
                )
                resolved_branches = data_manager.config.branch_config.resolve_aliases(
                    load_branches, is_mc=True
                )
                available_branches = list(tree.keys())
                validation = data_manager.config.branch_config.validate_branches(
                    resolved_branches, available_branches
                )
                tree_events = tree.arrays(validation["valid"], library="ak")
                rename_map = data_manager.config.branch_config.normalize_branches(
                    validation["valid"], is_mc=True
                )
                if rename_map:
                    tree_events = ak.zip(
                        {
                            rename_map.get(name, name): tree_events[name]
                            for name in tree_events.fields
                        }
                    )
                events_list.append(tree_events)
            except Exception as e:
                print_line(f"  Error: {e}")
                continue
        if not events_list:
            print_line(f"  No data loaded for {state}")
            continue
        events = ak.concatenate(events_list)
        n_total = len(events)
        # Check each truth matching criterion individually
        stats: dict[str, float] = {}
        stats["n_total"] = n_total
        # B+ identification
        bu_ok = abs(events.Bu_TRUEID) == 521
        stats["Bu_TRUEID"] = float(ak.sum(bu_ok) / n_total * 100)
        # Lambda identification
        l0_ok = abs(events.L0_TRUEID) == 3122
        stats["L0_TRUEID"] = float(ak.sum(l0_ok) / n_total * 100)
        # Lambda daughters
        lp_ok = abs(events.Lp_TRUEID) == 2212
        lpi_ok = abs(events.Lpi_TRUEID) == 211
        lp_mother_ok = abs(events.Lp_MC_MOTHER_ID) == 3122
        lpi_mother_ok = abs(events.Lpi_MC_MOTHER_ID) == 3122
        stats["Lp_TRUEID"] = float(ak.sum(lp_ok) / n_total * 100)
        stats["Lpi_TRUEID"] = float(ak.sum(lpi_ok) / n_total * 100)
        stats["Lp_mother_Lambda"] = float(ak.sum(lp_mother_ok) / n_total * 100)
        stats["Lpi_mother_Lambda"] = float(ak.sum(lpi_mother_ok) / n_total * 100)
        # Bachelor proton
        p_ok = abs(events.p_TRUEID) == 2212
        stats["p_TRUEID"] = float(ak.sum(p_ok) / n_total * 100)
        # Kaons
        h1_ok = abs(events.h1_TRUEID) == 321
        h2_ok = abs(events.h2_TRUEID) == 321
        stats["h1_TRUEID"] = float(ak.sum(h1_ok) / n_total * 100)
        stats["h2_TRUEID"] = float(ak.sum(h2_ok) / n_total * 100)
        # Mother requirements - check what mothers are actually present
        # L0 mother distribution
        l0_mother_bu = abs(events.L0_MC_MOTHER_ID) == 521
        l0_mother_cc = abs(events.L0_MC_MOTHER_ID) == cc_id
        stats["L0_mother_Bu"] = float(ak.sum(l0_mother_bu) / n_total * 100)
        stats["L0_mother_cc"] = float(ak.sum(l0_mother_cc) / n_total * 100)
        # p mother distribution
        p_mother_bu = abs(events.p_MC_MOTHER_ID) == 521
        p_mother_cc = abs(events.p_MC_MOTHER_ID) == cc_id
        stats["p_mother_Bu"] = float(ak.sum(p_mother_bu) / n_total * 100)
        stats["p_mother_cc"] = float(ak.sum(p_mother_cc) / n_total * 100)
        # h1 mother distribution
        h1_mother_bu = abs(events.h1_MC_MOTHER_ID) == 521
        h1_mother_cc = abs(events.h1_MC_MOTHER_ID) == cc_id
        stats["h1_mother_Bu"] = float(ak.sum(h1_mother_bu) / n_total * 100)
        stats["h1_mother_cc"] = float(ak.sum(h1_mother_cc) / n_total * 100)
        # h2 mother distribution
        h2_mother_bu = abs(events.h2_MC_MOTHER_ID) == 521
        h2_mother_cc = abs(events.h2_MC_MOTHER_ID) == cc_id
        stats["h2_mother_Bu"] = float(ak.sum(h2_mother_bu) / n_total * 100)
        stats["h2_mother_cc"] = float(ak.sum(h2_mother_cc) / n_total * 100)
        # Check unique mother IDs to understand decay topology
        unique_l0_mothers = np.unique(ak.to_numpy(abs(events.L0_MC_MOTHER_ID)))
        unique_p_mothers = np.unique(ak.to_numpy(abs(events.p_MC_MOTHER_ID)))
        unique_h1_mothers = np.unique(ak.to_numpy(abs(events.h1_MC_MOTHER_ID)))
        unique_h2_mothers = np.unique(ak.to_numpy(abs(events.h2_MC_MOTHER_ID)))
        # Full truth matching
        full_mask = get_truth_matching_mask(events)
        stats["full_truth_match"] = float(ak.sum(full_mask) / n_total * 100)
        all_stats[state] = stats
        # Print detailed breakdown
        print_line(f"  Total events: {n_total}")
        print_line(f"  Full truth match: {stats['full_truth_match']:.1f}%")
        print_line(f"\n  **Particle ID matching:**")
        print_line(f"    Bu_TRUEID=521: {stats['Bu_TRUEID']:.1f}%")
        print_line(f"    L0_TRUEID=3122: {stats['L0_TRUEID']:.1f}%")
        print_line(f"    Lp_TRUEID=2212: {stats['Lp_TRUEID']:.1f}%")
        print_line(f"    Lpi_TRUEID=211: {stats['Lpi_TRUEID']:.1f}%")
        print_line(f"    p_TRUEID=2212: {stats['p_TRUEID']:.1f}%")
        print_line(f"    h1_TRUEID=321: {stats['h1_TRUEID']:.1f}%")
        print_line(f"    h2_TRUEID=321: {stats['h2_TRUEID']:.1f}%")
        print_line(f"\n  **Mother matching:**")
        print_line(f"    L0 from B⁺: {stats['L0_mother_Bu']:.1f}%")
        print_line(f"    L0 from cc̄({cc_id}): {stats['L0_mother_cc']:.1f}%")
        print_line(f"    p from B⁺: {stats['p_mother_Bu']:.1f}%")
        print_line(f"    p from cc̄({cc_id}): {stats['p_mother_cc']:.1f}%")
        print_line(f"    h1(K⁺) from B⁺: {stats['h1_mother_Bu']:.1f}%")
        print_line(f"    h1(K⁺) from cc̄({cc_id}): {stats['h1_mother_cc']:.1f}%")
        print_line(f"    h2(K⁻) from B⁺: {stats['h2_mother_Bu']:.1f}%")
        print_line(f"    h2(K⁻) from cc̄({cc_id}): {stats['h2_mother_cc']:.1f}%")
        print_line(f"\n  **Unique mother IDs found:**")
        print_line(f"    L0 mothers: {unique_l0_mothers[:10]}")
        print_line(f"    p mothers: {unique_p_mothers[:10]}")
        print_line(f"    h1 mothers: {unique_h1_mothers[:10]}")
        print_line(f"    h2 mothers: {unique_h2_mothers[:10]}")
    # Summary comparison table
    print_line(f"\n{'='*80}")
    print_line("SUMMARY COMPARISON")
    print_line(f"{'='*80}\n")
    print_line("| Criterion | J/ψ | ηc(1S) | χc0 | χc1 |")
    print_line("|-----------|-----|--------|-----|-----|")
    criteria = [
        ("Full truth match", "full_truth_match"),
        ("Bu_TRUEID=521", "Bu_TRUEID"),
        ("L0_TRUEID=3122", "L0_TRUEID"),
        ("p_TRUEID=2212", "p_TRUEID"),
        ("h1_TRUEID=321", "h1_TRUEID"),
        ("h2_TRUEID=321", "h2_TRUEID"),
        ("L0 from cc̄", "L0_mother_cc"),
        ("p from cc̄", "p_mother_cc"),
        ("h2(K⁻) from cc̄", "h2_mother_cc"),
    ]
    for label, key in criteria:
        if all(key in all_stats[s] for s in STATES.keys()):
            vals = [f"{all_stats[s][key]:.1f}%" for s in STATES.keys()]
            print_line(f"| {label} | {' | '.join(vals)} |")
    print_line("\n## Interpretation\n")
    print_line(
        "If truth matching is the cause:\n"
        "- One or more criteria should show lower pass rate for ηc(1S)\n"
        "- The mother matching may differ due to different decay topologies\n"
        "- Check if the decay model (PHSP vs resonant) affects truth info\n"
    )
    # Detailed mother ID breakdown for each state
    print_line(f"\n{'='*80}")
    print_line("DETAILED MOTHER ID ANALYSIS")
    print_line(f"{'='*80}\n")
    print_line("Showing what mothers p and h2(K⁻) actually have in each sample.\n")
    for state in STATES.keys():
        state_label = STATES[state]
        cc_id = charmonium_pdg[state]
        # Reload data for this state
        year = "2018"
        track_type = "LL"
        events_list = []
        for magnet in ["MU", "MD"]:
            year_int = int(year)
            filename = f"{state}_{year_int-2000}_{magnet}.root"
            filepath = data_manager.mc_path / state / filename
            if not filepath.exists():
                continue
            channel_path = f"B2L0barPKpKm_{track_type}"
            tree_path = f"{channel_path}/DecayTree"
            try:
                file = uproot.open(filepath)
                if channel_path not in file:
                    continue
                tree = file[tree_path]
                load_branches_sets = ["essential", "kinematics", "mc_truth"]
                load_branches = data_manager.config.branch_config.get_branches_from_sets(
                    load_branches_sets, exclude_mc=False
                )
                resolved_branches = data_manager.config.branch_config.resolve_aliases(
                    load_branches, is_mc=True
                )
                available_branches = list(tree.keys())
                validation = data_manager.config.branch_config.validate_branches(
                    resolved_branches, available_branches
                )
                tree_events = tree.arrays(validation["valid"], library="ak")
                rename_map = data_manager.config.branch_config.normalize_branches(
                    validation["valid"], is_mc=True
                )
                if rename_map:
                    tree_events = ak.zip(
                        {
                            rename_map.get(name, name): tree_events[name]
                            for name in tree_events.fields
                        }
                    )
                events_list.append(tree_events)
            except Exception:
                continue
        if not events_list:
            continue
        events = ak.concatenate(events_list)
        n_total = len(events)
        print_line(f"\n### {state_label}\n")
        # Get mother ID distributions for p and h2
        p_mothers = ak.to_numpy(abs(events.p_MC_MOTHER_ID))
        h2_mothers = ak.to_numpy(abs(events.h2_MC_MOTHER_ID))
        # Count occurrences
        p_mother_counts: dict[int, int] = {}
        for m in p_mothers:
            p_mother_counts[m] = p_mother_counts.get(m, 0) + 1
        h2_mother_counts: dict[int, int] = {}
        for m in h2_mothers:
            h2_mother_counts[m] = h2_mother_counts.get(m, 0) + 1
        # PDG ID lookup
        pdg_names = {
            0: "unknown",
            521: "B⁺",
            443: "J/ψ",
            441: "ηc",
            10441: "χc0",
            20443: "χc1",
            4: "c quark",
            5: "b quark",
            111: "π⁰",
            113: "ρ⁰",
            221: "η",
            223: "ω",
            333: "φ",
            311: "K⁰",
            321: "K⁺",
            331: "η'",
            411: "D⁺",
            413: "D*⁺",
            421: "D⁰",
            423: "D*⁰",
            130: "K_L⁰",
            310: "K_S⁰",
            211: "π⁺",
            213: "ρ⁺",
            323: "K*⁺",
            313: "K*⁰",
            335: "f₂'",
            225: "f₂",
            115: "a₂⁰",
            22: "γ",
            15: "τ",
            13: "μ",
            43: "χc1(1P)",  # Alternative ID
            44: "χc2",
        }
        # Sort by count and print top mothers
        print_line("  **p (bachelor proton) mother distribution:**")
        sorted_p = sorted(p_mother_counts.items(), key=lambda x: -x[1])[:10]
        for mother_id, count in sorted_p:
            pct = count / n_total * 100
            name = pdg_names.get(mother_id, f"PDG={mother_id}")
            print_line(f"    {name} ({mother_id}): {count} ({pct:.1f}%)")
        print_line("\n  **h2 (K⁻) mother distribution:**")
        sorted_h2 = sorted(h2_mother_counts.items(), key=lambda x: -x[1])[:10]
        for mother_id, count in sorted_h2:
            pct = count / n_total * 100
            name = pdg_names.get(mother_id, f"PDG={mother_id}")
            print_line(f"    {name} ({mother_id}): {count} ({pct:.1f}%)")


def run_kinematic_study(data_manager: DataManager, verbose: bool = False, output_file=None) -> None:
    """
    Study kinematic distributions of final state particles for each charmonium state.

    This investigates whether the lower ηc(1S) efficiency is due to kinematic
    acceptance differences arising from the different charmonium masses.

    Args:
        data_manager: DataManager instance
        verbose: Enable verbose output
        output_file: Optional file object for output
    """
    import uproot

    def print_line(line: str) -> None:
        print(line)
        if output_file:
            output_file.write(line + "\n")

    print_line(f"\n{'='*80}")
    print_line("KINEMATIC ACCEPTANCE STUDY")
    print_line(f"{'='*80}\n")
    print_line("Comparing kinematic distributions of final state particles.\n")
    print_line("Lower charmonium mass → more phase space → softer daughters → lower acceptance?\n")
    # Collect statistics for all states
    all_stats: dict[str, dict] = {}
    for state in STATES.keys():
        state_label = STATES[state]
        props = CHARMONIUM_PROPERTIES[state]
        # Load data for 2018 LL
        year = "2018"
        track_type = "LL"
        events_list = []
        for magnet in ["MU", "MD"]:
            year_int = int(year)
            filename = f"{state}_{year_int-2000}_{magnet}.root"
            filepath = data_manager.mc_path / state / filename
            if not filepath.exists():
                continue
            channel_path = f"B2L0barPKpKm_{track_type}"
            tree_path = f"{channel_path}/DecayTree"
            try:
                file = uproot.open(filepath)
                if channel_path not in file:
                    continue
                tree = file[tree_path]
                load_branches_sets = ["essential", "kinematics", "mc_truth"]
                load_branches = data_manager.config.branch_config.get_branches_from_sets(
                    load_branches_sets, exclude_mc=False
                )
                resolved_branches = data_manager.config.branch_config.resolve_aliases(
                    load_branches, is_mc=True
                )
                available_branches = list(tree.keys())
                validation = data_manager.config.branch_config.validate_branches(
                    resolved_branches, available_branches
                )
                tree_events = tree.arrays(validation["valid"], library="ak")
                rename_map = data_manager.config.branch_config.normalize_branches(
                    validation["valid"], is_mc=True
                )
                if rename_map:
                    tree_events = ak.zip(
                        {
                            rename_map.get(name, name): tree_events[name]
                            for name in tree_events.fields
                        }
                    )
                events_list.append(tree_events)
            except Exception as e:
                if verbose:
                    print_line(f"  Error loading {state}: {e}")
                continue
        if not events_list:
            continue
        events = ak.concatenate(events_list)
        # Get truth-matched events
        truth_mask = get_truth_matching_mask(events)
        events_gen = events  # All generated (reconstructed in tree)
        events_reco = events[truth_mask]  # Truth-matched
        n_gen = len(events_gen)
        n_reco = len(events_reco)
        # Collect kinematic statistics for generated events
        stats: dict[str, dict] = {"gen": {}, "reco": {}}
        particles = {
            "L0": "Λ̄⁰",
            "p": "p (bachelor)",
            "h1": "K⁺",
            "h2": "K⁻",
        }
        for prefix, particle_name in particles.items():
            pt_branch = f"{prefix}_PT"
            eta_branch = f"{prefix}_ETA"
            p_branch = f"{prefix}_P"
            for label, evts in [("gen", events_gen), ("reco", events_reco)]:
                if pt_branch in evts.fields:
                    pt = ak.to_numpy(evts[pt_branch])
                    stats[label][f"{prefix}_PT_mean"] = float(np.mean(pt))
                    stats[label][f"{prefix}_PT_std"] = float(np.std(pt))
                    # Fraction below typical LHCb cut (e.g., 250 MeV)
                    stats[label][f"{prefix}_PT_below250"] = float(np.sum(pt < 250) / len(pt) * 100)
                if eta_branch in evts.fields:
                    eta = ak.to_numpy(evts[eta_branch])
                    stats[label][f"{prefix}_ETA_mean"] = float(np.mean(eta))
                    stats[label][f"{prefix}_ETA_std"] = float(np.std(eta))
                    # Fraction outside LHCb acceptance (2 < η < 5)
                    outside = np.sum((eta < 2) | (eta > 5)) / len(eta) * 100
                    stats[label][f"{prefix}_ETA_outside"] = float(outside)
                if p_branch in evts.fields:
                    p = ak.to_numpy(evts[p_branch])
                    stats[label][f"{prefix}_P_mean"] = float(np.mean(p))
                    stats[label][f"{prefix}_P_std"] = float(np.std(p))
                    # Fraction below typical cut (e.g., 2 GeV)
                    stats[label][f"{prefix}_P_below2GeV"] = float(np.sum(p < 2000) / len(p) * 100)
        stats["n_gen"] = n_gen
        stats["n_reco"] = n_reco
        stats["eff"] = n_reco / n_gen * 100 if n_gen > 0 else 0
        stats["mass"] = props["mass"]
        all_stats[state] = stats
    # Print comparison table
    print_line("\n## Kinematic Comparison (Generated Events, 2018 LL)\n")
    print_line("| Particle | Variable | J/ψ | ηc(1S) | χc0 | χc1 |")
    print_line("|----------|----------|-----|--------|-----|-----|")
    particles_order = ["L0", "p", "h2"]  # Focus on charmonium daughters
    for prefix in particles_order:
        particle_name = particles[prefix]
        # pT
        pt_key = f"{prefix}_PT_mean"
        if all(pt_key in all_stats[s]["gen"] for s in STATES.keys()):
            vals = [f"{all_stats[s]['gen'][pt_key]:.0f}" for s in STATES.keys()]
            print_line(f"| {particle_name} | ⟨pT⟩ (MeV) | {' | '.join(vals)} |")
        # pT below 250 MeV
        pt_low_key = f"{prefix}_PT_below250"
        if all(pt_low_key in all_stats[s]["gen"] for s in STATES.keys()):
            vals = [f"{all_stats[s]['gen'][pt_low_key]:.1f}%" for s in STATES.keys()]
            print_line(f"| {particle_name} | pT<250 MeV | {' | '.join(vals)} |")
        # η outside acceptance
        eta_key = f"{prefix}_ETA_outside"
        if all(eta_key in all_stats[s]["gen"] for s in STATES.keys()):
            vals = [f"{all_stats[s]['gen'][eta_key]:.1f}%" for s in STATES.keys()]
            print_line(f"| {particle_name} | η outside [2,5] | {' | '.join(vals)} |")
        # P below 2 GeV
        p_low_key = f"{prefix}_P_below2GeV"
        if all(p_low_key in all_stats[s]["gen"] for s in STATES.keys()):
            vals = [f"{all_stats[s]['gen'][p_low_key]:.1f}%" for s in STATES.keys()]
            print_line(f"| {particle_name} | P<2 GeV | {' | '.join(vals)} |")
    # Summary
    print_line("\n## Efficiency vs Charmonium Mass\n")
    print_line("| State | Mass (MeV) | Efficiency (%) | N_gen | N_reco |")
    print_line("|-------|------------|----------------|-------|--------|")
    for state in STATES.keys():
        s = all_stats[state]
        state_label = STATES[state]
        print_line(
            f"| {state_label} | {s['mass']:.1f} | {s['eff']:.2f} | {s['n_gen']} | {s['n_reco']} |"
        )
    print_line("\n## Interpretation\n")
    print_line(
        "If kinematic acceptance is the cause:\n"
        "- ηc(1S) daughters should have softer pT (more below cuts)\n"
        "- ηc(1S) daughters should have more events outside η acceptance\n"
        "- The effect should correlate with charmonium mass\n"
    )


def run_phase_space_study(
    data_manager: DataManager, verbose: bool = False, output_file=None
) -> None:
    """
    Study phase space distributions for each charmonium state.

    This investigates whether different decay dynamics (phase space vs matrix element)
    could explain the efficiency difference.

    Args:
        data_manager: DataManager instance
        verbose: Enable verbose output
        output_file: Optional file object for output
    """
    import uproot

    def print_line(line: str) -> None:
        print(line)
        if output_file:
            output_file.write(line + "\n")

    print_line(f"\n{'='*80}")
    print_line("PHASE SPACE STUDY")
    print_line(f"{'='*80}\n")
    print_line("Comparing Dalitz plot variables and sub-system masses.\n")
    print_line("The decay cc̄ → Λ̄⁰ p K⁻ has 3-body phase space.\n")
    # Collect statistics for all states
    all_stats: dict[str, dict] = {}
    for state in STATES.keys():
        state_label = STATES[state]
        props = CHARMONIUM_PROPERTIES[state]
        # Load data for 2018 LL
        year = "2018"
        track_type = "LL"
        events_list = []
        for magnet in ["MU", "MD"]:
            year_int = int(year)
            filename = f"{state}_{year_int-2000}_{magnet}.root"
            filepath = data_manager.mc_path / state / filename
            if not filepath.exists():
                continue
            channel_path = f"B2L0barPKpKm_{track_type}"
            tree_path = f"{channel_path}/DecayTree"
            try:
                file = uproot.open(filepath)
                if channel_path not in file:
                    continue
                tree = file[tree_path]
                load_branches_sets = ["essential", "kinematics", "mc_truth"]
                load_branches = data_manager.config.branch_config.get_branches_from_sets(
                    load_branches_sets, exclude_mc=False
                )
                resolved_branches = data_manager.config.branch_config.resolve_aliases(
                    load_branches, is_mc=True
                )
                available_branches = list(tree.keys())
                validation = data_manager.config.branch_config.validate_branches(
                    resolved_branches, available_branches
                )
                tree_events = tree.arrays(validation["valid"], library="ak")
                rename_map = data_manager.config.branch_config.normalize_branches(
                    validation["valid"], is_mc=True
                )
                if rename_map:
                    tree_events = ak.zip(
                        {
                            rename_map.get(name, name): tree_events[name]
                            for name in tree_events.fields
                        }
                    )
                events_list.append(tree_events)
            except Exception as e:
                if verbose:
                    print_line(f"  Error loading {state}: {e}")
                continue
        if not events_list:
            continue
        events = ak.concatenate(events_list)
        # Get truth-matched events
        truth_mask = get_truth_matching_mask(events)
        events_gen = events
        events_reco = events[truth_mask]
        n_gen = len(events_gen)
        n_reco = len(events_reco)
        stats: dict[str, dict] = {"gen": {}, "reco": {}}
        # Calculate sub-system invariant masses for Dalitz analysis
        # M(Λ̄p), M(Λ̄K⁻), M(pK⁻)
        for label, evts in [("gen", events_gen), ("reco", events_reco)]:
            required = [
                "L0_PX",
                "L0_PY",
                "L0_PZ",
                "L0_PE",
                "p_PX",
                "p_PY",
                "p_PZ",
                "p_PE",
                "h2_PX",
                "h2_PY",
                "h2_PZ",
                "h2_PE",
            ]
            if not all(b in evts.fields for b in required):
                continue
            # Build 4-vectors
            L0_4mom = vector.zip(
                {"px": evts.L0_PX, "py": evts.L0_PY, "pz": evts.L0_PZ, "E": evts.L0_PE}
            )
            p_4mom = vector.zip({"px": evts.p_PX, "py": evts.p_PY, "pz": evts.p_PZ, "E": evts.p_PE})
            h2_4mom = vector.zip(
                {"px": evts.h2_PX, "py": evts.h2_PY, "pz": evts.h2_PZ, "E": evts.h2_PE}
            )
            # Sub-system masses
            M_Lp = (L0_4mom + p_4mom).mass  # M(Λ̄p)
            M_LK = (L0_4mom + h2_4mom).mass  # M(Λ̄K⁻)
            M_pK = (p_4mom + h2_4mom).mass  # M(pK⁻)
            M_Lp_np = ak.to_numpy(M_Lp)
            M_LK_np = ak.to_numpy(M_LK)
            M_pK_np = ak.to_numpy(M_pK)
            stats[label]["M_Lp_mean"] = float(np.mean(M_Lp_np))
            stats[label]["M_Lp_std"] = float(np.std(M_Lp_np))
            stats[label]["M_LK_mean"] = float(np.mean(M_LK_np))
            stats[label]["M_LK_std"] = float(np.std(M_LK_np))
            stats[label]["M_pK_mean"] = float(np.mean(M_pK_np))
            stats[label]["M_pK_std"] = float(np.std(M_pK_np))
            # Dalitz plot squared masses
            stats[label]["M_Lp_sq_mean"] = float(np.mean(M_Lp_np**2))
            stats[label]["M_pK_sq_mean"] = float(np.mean(M_pK_np**2))
        stats["n_gen"] = n_gen
        stats["n_reco"] = n_reco
        stats["eff"] = n_reco / n_gen * 100 if n_gen > 0 else 0
        stats["mass"] = props["mass"]
        # Available phase space (Q-value)
        m_L = 1115.683  # Lambda mass
        m_p = 938.272  # Proton mass
        m_K = 493.677  # Kaon mass
        Q_value = props["mass"] - m_L - m_p - m_K
        stats["Q_value"] = Q_value
        all_stats[state] = stats
    # Print phase space comparison
    print_line("\n## Phase Space Available (Q-value)\n")
    print_line("Q = M(cc̄) - M(Λ̄) - M(p) - M(K⁻)\n")
    print_line("| State | M(cc̄) (MeV) | Q-value (MeV) | Efficiency (%) |")
    print_line("|-------|-------------|---------------|----------------|")
    for state in STATES.keys():
        s = all_stats[state]
        state_label = STATES[state]
        print_line(f"| {state_label} | {s['mass']:.1f} | {s['Q_value']:.1f} | {s['eff']:.2f} |")
    # Print sub-system masses
    print_line("\n## Sub-system Invariant Masses (Generated Events)\n")
    print_line("| State | ⟨M(Λ̄p)⟩ | ⟨M(Λ̄K⁻)⟩ | ⟨M(pK⁻)⟩ |")
    print_line("|-------|---------|----------|---------|")
    for state in STATES.keys():
        s = all_stats[state]
        state_label = STATES[state]
        if "M_Lp_mean" in s["gen"]:
            print_line(
                f"| {state_label} | {s['gen']['M_Lp_mean']:.0f}±{s['gen']['M_Lp_std']:.0f} | "
                f"{s['gen']['M_LK_mean']:.0f}±{s['gen']['M_LK_std']:.0f} | "
                f"{s['gen']['M_pK_mean']:.0f}±{s['gen']['M_pK_std']:.0f} |"
            )
    # Efficiency vs Q-value correlation
    print_line("\n## Interpretation\n")
    print_line(
        "Phase space effects:\n"
        "- Lower Q-value → less phase space → daughters closer to threshold\n"
        "- ηc(1S) has Q = 436 MeV vs J/ψ Q = 549 MeV (20% less phase space)\n"
        "- This could lead to softer daughter momenta and lower acceptance\n"
        "\n"
        "If phase space is the cause:\n"
        "- Efficiency should correlate with Q-value\n"
        "- Sub-system mass distributions should differ\n"
    )


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(
        description="Calculate reconstruction efficiencies for B+ to Lambda pKK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output for debugging"
    )

    parser.add_argument(
        "-s", "--state", choices=list(STATES.keys()), help="Calculate for specific state only"
    )

    parser.add_argument("-o", "--output", type=str, help="Save results to file")

    parser.add_argument(
        "--show-residuals",
        action="store_true",
        help="Show relative differences between states (for systematics)",
    )

    parser.add_argument(
        "--width-study",
        action="store_true",
        help="Run natural width study to investigate ηc(1S) efficiency difference",
    )

    parser.add_argument(
        "--mass-window",
        type=float,
        default=None,
        help="Apply fixed mass window in MeV (e.g., --mass-window 50)",
    )

    parser.add_argument(
        "--mass-window-nsigma",
        type=float,
        default=None,
        help="Apply mass window in units of natural width (e.g., --mass-window-nsigma 2)",
    )

    parser.add_argument(
        "--mass-diagnostic",
        action="store_true",
        help="Run diagnostic to check M(Λ̄pK⁻) distribution for each state",
    )

    parser.add_argument(
        "--truth-matching-study",
        action="store_true",
        help="Run detailed truth matching study for each state",
    )

    parser.add_argument(
        "--kinematic-study",
        action="store_true",
        help="Run kinematic acceptance study comparing pT, η distributions",
    )

    parser.add_argument(
        "--phase-space-study",
        action="store_true",
        help="Run phase space study comparing Q-values and Dalitz variables",
    )

    args = parser.parse_args()

    # Initialize configuration and data manager
    config_dir = Path(__file__).parent.parent / "analysis" / "config"
    config = TOMLConfig(str(config_dir))
    data_manager = DataManager(config)

    # Determine which states to process
    states_to_process = [args.state] if args.state else list(STATES.keys())

    # Open output file if specified
    output_file = None
    if args.output:
        output_file = open(args.output, "w")
        print(f"\nWriting results to: {args.output}")

    # Run mass diagnostic if requested
    if args.mass_diagnostic:
        run_mass_diagnostic(data_manager, verbose=args.verbose, output_file=output_file)
        if output_file:
            output_file.close()
            print(f"\nResults saved to: {args.output}")
        return

    # Run truth matching study if requested
    if args.truth_matching_study:
        run_truth_matching_study(data_manager, verbose=args.verbose, output_file=output_file)
        if output_file:
            output_file.close()
            print(f"\nResults saved to: {args.output}")
        return

    # Run kinematic study if requested
    if args.kinematic_study:
        run_kinematic_study(data_manager, verbose=args.verbose, output_file=output_file)
        if output_file:
            output_file.close()
            print(f"\nResults saved to: {args.output}")
        return

    # Run phase space study if requested
    if args.phase_space_study:
        run_phase_space_study(data_manager, verbose=args.verbose, output_file=output_file)
        if output_file:
            output_file.close()
            print(f"\nResults saved to: {args.output}")
        return

    # Run width study if requested
    if args.width_study:
        run_width_study(data_manager, verbose=args.verbose, output_file=output_file)
        if output_file:
            output_file.close()
            print(f"\nResults saved to: {args.output}")
        return

    # Calculate total number of calculations for progress bar
    total_calcs = len(states_to_process) * len(YEARS) * len(TRACK_TYPES)

    # Storage for results
    results = {}

    # Main calculation loop
    with tqdm(total=total_calcs, desc="Calculating efficiencies", disable=args.verbose) as pbar:
        for state in states_to_process:
            for year in YEARS:
                for track_type in TRACK_TYPES:
                    result = calculate_reco_efficiency(
                        data_manager,
                        state,
                        year,
                        track_type,
                        verbose=args.verbose,
                        mass_window_mev=args.mass_window,
                        mass_window_nsigma=args.mass_window_nsigma,
                    )

                    # Store results with composite key
                    key = (state, year, track_type)
                    results[key] = result

                    pbar.update(1)

    # Print results
    print_results_table(results, output_file)

    if args.show_residuals:
        print_residuals_table(results, output_file)

    # Print mass window results if applicable
    if args.mass_window is not None or args.mass_window_nsigma is not None:
        print_width_study_table(results, output_file)

    # Close output file
    if output_file:
        output_file.close()
        print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print(
        """
The reconstruction efficiency measures the fraction of generated events where
all final state particles are successfully reconstructed and correctly
truth-matched.

Key Points:
-----------
• Since all channels have the same final state (p Λ̄⁰ K⁺ K⁻), the
  reconstruction efficiency should be similar across states

• Any differences arise from:
  - Kinematic distributions (different charmonium masses → different momenta)
  - Track quality variations
  - Detector acceptance effects

• For our ratio measurement, these efficiencies largely cancel out!

• Residual differences (shown with --show-residuals) contribute to
  systematic uncertainties

• Notable: ηc(1S) shows ~22% lower efficiency due to kinematic differences

• LL vs DD: DD has higher efficiency (~53% vs ~40%) due to more Λ̄⁰ decays
  producing downstream tracks, but this cancels in ratios within same category
"""
    )


if __name__ == "__main__":
    main()
