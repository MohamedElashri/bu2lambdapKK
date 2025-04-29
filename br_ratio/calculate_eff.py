#!/usr/bin/env python3
import yaml
import json
import os
import numpy as np
import argparse
from pathlib import Path
from collections import OrderedDict
import awkward as ak
import sys

# Ensure the script's directory is in the Python path
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

try:
    from loaders import load_data_condensed
    from selections import trigger_mask, create_selection_mask
    from branches import nominal_branches
except ImportError:
    print("Warning: Could not import custom modules. Make sure loaders.py, selections.py, and branches.py are available.")
    exit()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Calculate efficiencies of selection cuts')
    parser.add_argument('--config', default='config.yml', help='Path to configuration file')
    parser.add_argument('--output', default='results/eff.json', help='Output file for efficiency results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--track', choices=['LL', 'DD'], help='Track type (LL or DD)')
    return parser.parse_args()

def log(message, verbose_only=False, args=None):
    """Utility function for controlled logging"""
    if not verbose_only or (args and args.verbose):
        print(message)

def calculate_efficiencies(data, sample):
    """
    Calculates overall trigger and selection efficiency.
    """
    initial_count = len(data)
    if initial_count == 0:
        return {"initial": 0, "trigger": 0, "selection": 0, "total": 0}, {"trigger": 0.0, "selection": 0.0, "total": 0.0}

    # Apply trigger mask
    trig_mask_arr = trigger_mask(data, sample)
    count_after_trigger = ak.sum(trig_mask_arr)
    data_after_trigger = data[trig_mask_arr]

    if count_after_trigger == 0:
        return {"initial": initial_count, "trigger": 0, "selection": 0, "total": 0}, {"trigger": 0.0, "selection": 0.0, "total": 0.0}

    # Apply selection mask
    sel_mask_arr = create_selection_mask(data_after_trigger, sample)
    count_after_selection = ak.sum(sel_mask_arr)

    counts = {
        "initial": initial_count,
        "trigger": count_after_trigger,
        "selection": count_after_selection,  # This is count after *both* trigger and selection
        "total": count_after_selection
    }

    efficiencies = {
        "trigger": count_after_trigger / initial_count,
        "selection": count_after_selection / count_after_trigger,  # Relative efficiency of selection cuts
        "total": count_after_selection / initial_count
    }

    return counts, efficiencies

def calculate_all_efficiencies(config, args):
    """
    Calculate efficiencies for all combinations of years and track types
    """
    # Extract configuration
    years = config.get('years', [])
    tracks = config.get('tracks', [])
    signal_data_dir = config.get('signal_data_dir')
    norm_data_dir = config.get('norm_data_dir')

    # Get decay modes - fall back to values from fit.py if not in config
    signal_mode = config.get('signal_mode', 'L0barPKpKm')
    norm_mode = config.get('norm_mode', 'KSKmKpPip')

    # Log configuration
    log(f"Processing data with the following configuration:", args=args)
    log(f"  Years: {years}", args=args)
    log(f"  Tracks: {tracks}", args=args)
    log(f"  Signal data directory: {signal_data_dir}", args=args)
    log(f"  Norm data directory: {norm_data_dir}", args=args)
    log(f"  Signal mode: {signal_mode}", args=args)
    log(f"  Norm mode: {norm_mode}", args=args)

    # Initialize results dictionary
    all_efficiencies = {
        'sig': {},
        'norm': {},
        'config': {
            'years': years,
            'tracks': tracks,
            'signal_mode': signal_mode,
            'norm_mode': norm_mode
        }
    }

    # Calculate efficiencies for individual year/track combinations
    for year in years:
        for track in tracks:
            category = f"{year}_{track}"
            log(f"\n--- Processing {category} ---", args=args)

            # Calculate signal efficiencies
            try:
                sig_data = load_data_condensed(data_path=signal_data_dir, decay_mode=signal_mode, years=[year], tracks=[track])
                if sig_data is not None and len(sig_data) > 0:
                    sig_counts, sig_efficiencies = calculate_efficiencies(sig_data, 'signal')
                    all_efficiencies['sig'][category] = {
                        'counts': sig_counts,
                        'efficiencies': sig_efficiencies
                    }
                    log(f"Signal efficiency ({category}): {sig_efficiencies['total']:.6f} ({sig_counts['total']}/{sig_counts['initial']})", args=args)
                else:
                    log(f"No signal data found for {category}", args=args)
            except Exception as e:
                log(f"Error calculating signal efficiency for {category}: {e}", args=args)

            # Calculate normalization efficiencies
            try:
                norm_data = load_data_condensed(data_path=norm_data_dir, decay_mode=norm_mode, years=[year], tracks=[track])
                if norm_data is not None and len(norm_data) > 0:
                    norm_counts, norm_efficiencies = calculate_efficiencies(norm_data, 'norm')
                    all_efficiencies['norm'][category] = {
                        'counts': norm_counts,
                        'efficiencies': norm_efficiencies
                    }
                    log(f"Norm efficiency ({category}): {norm_efficiencies['total']:.6f} ({norm_counts['total']}/{norm_counts['initial']})", args=args)
                else:
                    log(f"No normalization data found for {category}", args=args)
            except Exception as e:
                log(f"Error calculating normalization efficiency for {category}: {e}", args=args)

    return all_efficiencies

def print_summary(all_efficiencies, config, args):
    """
    Print summary of all efficiencies
    """
    years = config.get('years', [])
    tracks = config.get('tracks', [])

    log("\n" + "="*80, args=args)
    log(" "*30 + "EFFICIENCY SUMMARY", args=args)
    log("="*80, args=args)

    # Print individual results by channel
    for channel, title in [('sig', 'SIGNAL CHANNEL (B+ → Λ̄0pK+K-)'), ('norm', 'NORMALIZATION CHANNEL (B+ → K0sπ+K+K-)')]:
        log(f"\n{title}", args=args)
        log("-"*80, args=args)
        log(f"{'Dataset':<15} {'Events':<15} {'Passed':<15} {'Efficiency':<15}", args=args)
        log("-"*80, args=args)

        # Individual year/track combinations
        for year in years:
            for track in tracks:
                category = f"{year}_{track}"
                if category in all_efficiencies[channel]:
                    eff_data = all_efficiencies[channel][category]
                    n_tot = eff_data['counts']['initial']
                    n_pass = eff_data['counts']['total']
                    eff = eff_data['efficiencies']['total']

                    log(f"{category:<15} {n_tot:<15} {n_pass:<15} {eff:.6f}", args=args)
                else:
                    log(f"{category:<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}", args=args)

    # Calculate and print efficiency ratios
    log("\n" + "="*80, args=args)
    log(" "*20 + "SIGNAL/NORMALIZATION EFFICIENCY RATIOS", args=args)
    log("="*80, args=args)
    log(f"{'Dataset':<15} {'Signal Eff.':<15} {'Norm Eff.':<15} {'Ratio (N/S)':<15}", args=args)
    log("-"*80, args=args)

    # Ratios for each year/track combination
    for year in years:
        for track in tracks:
            category = f"{year}_{track}"

            sig_data = all_efficiencies['sig'].get(category, {})
            norm_data = all_efficiencies['norm'].get(category, {})

            sig_n_tot = sig_data.get('counts', {}).get('initial', 0)
            sig_n_pass = sig_data.get('counts', {}).get('total', 0)
            norm_n_tot = norm_data.get('counts', {}).get('initial', 0)
            norm_n_pass = norm_data.get('counts', {}).get('total', 0)

            sig_eff = sig_n_pass / sig_n_tot if sig_n_tot > 0 else 0
            norm_eff = norm_n_pass / norm_n_tot if norm_n_tot > 0 else 0

            if sig_eff > 0 and norm_eff > 0:
                ratio = norm_eff / sig_eff
                log(f"{category:<15} {sig_eff:.6f}{' '*9} {norm_eff:.6f}{' '*9} {ratio:.6f}", args=args)
            else:
                log(f"{category:<15} {sig_eff:.6f if sig_eff else 'N/A':<15} {norm_eff:.6f if norm_eff else 'N/A':<15} {'N/A':<15}", args=args)

def main():
    """Main function to calculate efficiencies"""
    args = parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        log(f"Error loading configuration: {e}")
        return 1

    # Calculate efficiencies
    all_efficiencies = calculate_all_efficiencies(config, args)

    # Print summary
    print_summary(all_efficiencies, config, args)

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    os.makedirs(output_path.parent, exist_ok=True)

    # Save results to JSON file
    with open(output_path, 'w') as f:
        json.dump(all_efficiencies, f, indent=2)

    log(f"\nEfficiency results saved to: {output_path}")

    return 0

if __name__ == "__main__":
    exit(main())