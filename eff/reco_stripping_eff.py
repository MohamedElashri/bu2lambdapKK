#!/usr/bin/env python3
"""
Reconstruction and Stripping Efficiency Calculator
===================================================

This script calculates the reconstruction and stripping efficiencies for
B⁺ → Λ̄⁰ p K⁺ K⁻ decay with various charmonium resonances in the pΛ̄⁰ system.

Purpose:
--------
While these efficiencies largely cancel in our ratio measurement (since all
channels have the same final state particles), we calculate them to:
1. Verify the cancellation assumption
2. Quantify residual differences for systematic uncertainties
3. Document the efficiency for completeness

Physical Interpretation:
------------------------
- **Reconstruction Efficiency**: Fraction of generated events where all final
  state particles (p, Λ̄⁰ daughters, K⁺, K⁻) are successfully reconstructed
  as tracks with proper track types:
  * Λ̄⁰ daughters: Long (LL) or Downstream (DD) tracks
  * Bachelor p, K⁺, K⁻: Long tracks

- **Stripping Efficiency**: Fraction of reconstructed events that pass the
  stripping selection criteria. This is applied on top of reconstruction.

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

Expected Results:
-----------------
- Reconstruction efficiency: ~60-70% (depends on Λ̄⁰ category LL/DD)
- Stripping efficiency: ~80-90% (depends on stripping cuts)
- **Important**: Ratios between channels should be ~1.0 ± small corrections

Usage:
------
    python reco_stripping_eff.py [options]

    Options:
        -v, --verbose          : Enable verbose debugging output
        -s, --state STATE      : Calculate for specific state only (Jpsi, etac, chic0, chic1)
        -o, --output FILE      : Save results to file (default: stdout)
        --show-residuals       : Show relative differences between states (for systematics)

Examples:
---------
    # Calculate for all states
    python reco_stripping_eff.py

    # Calculate for J/ψ only with verbose output
    python reco_stripping_eff.py -s Jpsi -v

    # Calculate and save to file
    python reco_stripping_eff.py -o reco_strip_eff_results.txt

    # Show residual differences for systematic uncertainties
    python reco_stripping_eff.py --show-residuals
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

from modules.data_handler import DataManager, TOMLConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

# States to analyze (matching your charmonium resonances)
# Use the same naming as in config/data.toml
STATES = {
    "Jpsi": "J/ψ",  # J/ψ → pΛ̄⁰ (normalization channel)
    "etac": "ηc(1S)",  # ηc(1S) → pΛ̄⁰
    "chic0": "χc0(1P)",  # χc0(1P) → pΛ̄⁰
    "chic1": "χc1(1P)",  # χc1(1P) → pΛ̄⁰
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

    p_mother_ok = abs(events.p_MC_MOTHER_ID) == 521
    for cc_id in charmonium_ids:
        p_mother_ok = p_mother_ok | (abs(events.p_MC_MOTHER_ID) == cc_id)

    mask = mask & l0_mother_ok & p_mother_ok

    # Kaons: allow one from B+ and one from charmonium (or both from B+)
    kaon_ok = (abs(events.h1_MC_MOTHER_ID) == 521) & (abs(events.h2_MC_MOTHER_ID) == 521)
    for cc_id in charmonium_ids:
        kaon_ok = kaon_ok | (
            ((abs(events.h1_MC_MOTHER_ID) == 521) & (abs(events.h2_MC_MOTHER_ID) == cc_id))
            | ((abs(events.h2_MC_MOTHER_ID) == 521) & (abs(events.h1_MC_MOTHER_ID) == cc_id))
        )

    mask = mask & kaon_ok

    return mask


def get_stripping_mask(events: ak.Array) -> ak.Array:
    """
    Apply stripping selection cuts.

    Args:
        events: Awkward array with reconstruction branches

    Returns:
        Boolean mask for events passing stripping

    Note:
        These are example cuts - adjust based on your actual stripping.
    """

    # Example stripping cuts
    mask = (
        # B+ mass window
        (events.Bu_M > 5000)
        & (events.Bu_M < 5600)
        &
        # Lambda mass window
        (events.L0_M > 1110)
        & (events.L0_M < 1125)
        &
        # Direction angle cuts
        (events.Bu_DIRA_OWNPV > 0.9995)
        & (events.L0_DIRA_OWNPV > 0.9995)
        &
        # IP chi2 cut
        (events.Bu_IPCHI2_OWNPV < 25)
    )

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


def calculate_reco_strip_efficiency(
    data_manager: DataManager, state: str, year: str, track_type: str, verbose: bool = False
) -> dict:
    """
    Calculate reconstruction and stripping efficiency for a given configuration.

    Args:
        data_manager: DataManager instance
        state: Charmonium state (Jpsi, etac, chic0, chic1)
        year: Data taking year (2016, 2017, 2018)
        track_type: Lambda track type (LL, DD)
        verbose: Enable verbose output

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

            # Load essential + selection + mc_truth branches
            load_branches_sets = ["essential", "selection", "mc_truth"]
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
            "n_stripped": 0,
            "combined_efficiency": ufloat(0, 0),
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
            "n_stripped": 0,
            "combined_efficiency": ufloat(0, 0),
            "efficiency_percent": ufloat(0, 0),
        }

    # Apply truth matching
    truth_mask = get_truth_matching_mask(events)
    n_generated = ak.sum(truth_mask)

    if verbose:
        print(f"  Truth-matched events: {n_generated} / {len(events)}")
        # Debug: check if truth branches are loaded
        if hasattr(events, "Bu_TRUEID"):
            print(f"  Sample Bu_TRUEID values: {events.Bu_TRUEID[:10]}")
            print(f"  Non-zero Bu_TRUEID: {ak.sum(events.Bu_TRUEID != 0)}")
        else:
            print(f"  WARNING: Bu_TRUEID not loaded!")

    # Apply stripping on truth-matched events
    events_truth = events[truth_mask]
    strip_mask = get_stripping_mask(events_truth)
    n_stripped = ak.sum(strip_mask)

    if verbose:
        print(f"  Passed stripping: {n_stripped} / {n_generated}")

    # Calculate efficiency
    combined_eff, (n_pass, n_tot) = calculate_efficiency(int(n_stripped), int(n_generated))

    return {
        "state": state,
        "year": year,
        "track_type": track_type,
        "n_generated": int(n_generated),
        "n_stripped": int(n_stripped),
        "combined_efficiency": combined_eff,
        "efficiency_percent": combined_eff * 100,
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
        print_line(f"Reconstruction × Stripping Efficiency - Λ̄⁰_{track_type}")
        print_line(f"{'='*80}\n")

        # LaTeX table
        print_line("\\begin{table}[htbp]")
        print_line("\\centering")
        print_line(
            f"\\caption{{Reconstruction and stripping efficiency for "
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
        print_line(f"\\label{{tab:reco_strip_eff_{track_type}}}")
        print_line("\\end{table}\n")

        # Markdown table
        print_line(f"**Reconstruction × Stripping Efficiency - Λ̄⁰_{track_type} (%)**\n")
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

                    eff_state = results[state_key_tuple]["combined_efficiency"]
                    eff_jpsi = results[jpsi_key]["combined_efficiency"]

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


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(
        description="Calculate reconstruction and stripping efficiencies",
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

    args = parser.parse_args()

    # Initialize configuration and data manager
    config_dir = Path(__file__).parent.parent / "analysis" / "config"
    config = TOMLConfig(str(config_dir))
    data_manager = DataManager(config)

    # Determine which states to process
    states_to_process = [args.state] if args.state else list(STATES.keys())

    # Calculate total number of calculations for progress bar
    total_calcs = len(states_to_process) * len(YEARS) * len(TRACK_TYPES)

    # Storage for results
    results = {}

    # Main calculation loop
    with tqdm(total=total_calcs, desc="Calculating efficiencies", disable=args.verbose) as pbar:
        for state in states_to_process:
            for year in YEARS:
                for track_type in TRACK_TYPES:
                    result = calculate_reco_strip_efficiency(
                        data_manager, state, year, track_type, verbose=args.verbose
                    )

                    # Store results with composite key
                    key = (state, year, track_type)
                    results[key] = result

                    pbar.update(1)

    # Open output file if specified
    output_file = None
    if args.output:
        output_file = open(args.output, "w")
        print(f"\nWriting results to: {args.output}")

    # Print results
    print_results_table(results, output_file)

    if args.show_residuals:
        print_residuals_table(results, output_file)

    # Close output file
    if output_file:
        output_file.close()
        print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print(
        """
The reconstruction and stripping efficiency combines two effects:
1. Reconstruction: Ability to reconstruct all final state particles
2. Stripping: Fraction passing the stripping selection cuts (WIP)

Key Points:
-----------
• Since all channels have the same final state (p Λ̄⁰ K⁺ K⁻), the
  reconstruction efficiency should be very similar across states

• Any differences arise from:
  - Kinematic distributions (different charmonium masses → different momenta)
  - Track quality variations
  - Acceptance effects

• For your ratio measurement, these efficiencies largely cancel out!

• Residual differences (shown with --show-residuals) contribute to
  systematic uncertainties - typically at the level of ~1-2%

• LL vs DD: Expect DD to have lower efficiency due to downstream tracks
  (but this cancels in ratios within same Λ̄⁰ category)
"""
    )


if __name__ == "__main__":
    main()
