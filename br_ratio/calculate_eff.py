#!/usr/bin/env python3
import yaml
import json
import os
import numpy as np
import argparse
from pathlib import Path
from collections import OrderedDict

try:
    from loaders import load_data
    from selections import trigger_mask
    from branches import canonical
except ImportError:
    print("Warning: Could not import custom modules. Make sure loaders.py, selections.py, and branches.py are available.")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Calculate efficiencies of selection cuts')
    parser.add_argument('--config', default='config.yml', help='Path to configuration file')
    parser.add_argument('--output', default='results/eff.json', help='Output file for efficiency results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    return parser.parse_args()

def log(message, verbose_only=False, args=None):
    """Utility function for controlled logging"""
    if not verbose_only or (args and args.verbose):
        print(message)

def create_selection_mask(data, sample, step_by_step=False):
    """
    Create selection mask for data based on sample type, tracking each cut's effect
    
    Parameters:
    -----------
    data : awkward array
        Data to apply selection to
    sample : str
        'signal' or 'norm' to determine which selection to apply
    step_by_step : bool
        If True, return a dictionary of masks for each cut
        
    Returns:
    --------
    if step_by_step:
        dict: Dictionary of boolean masks for each selection step
    else:
        boolean mask: Combined mask to apply to data
    """
    is_sig = sample == "signal"
    initial_mask = np.ones(len(data), dtype=bool)  # Start with all True
    
    # Get correct branch names based on sample
    if is_sig:
        # Signal-specific branch names
        l0_endvertex_z = "L0_ENDVERTEX_Z"
        l0_ownpv_z = "L0_OWNPV_Z"
        l0_fdchi2 = "L0_FDCHI2_OWNPV"
        l0_m = "L0_M"
        l0_proton_nn = "Lp_MC15TuneV1_ProbNNp"
        proton_nn = "p_MC15TuneV1_ProbNNp"
        h1_nn = "h1_MC15TuneV1_ProbNNk"
        h2_nn = "h2_MC15TuneV1_ProbNNk"
        bu_pt = "Bu_PT"
        bu_dtf_chi2 = "Bu_DTF_chi2"
        bu_ipchi2 = "Bu_IPCHI2_OWNPV"
        bu_fdchi2 = "Bu_FDCHI2_OWNPV"
    else:
        # Normalization-specific branch names
        l0_endvertex_z = "Ks_ENDVERTEX_Z"
        l0_ownpv_z = "Ks_OWNPV_Z"
        l0_fdchi2 = "Ks_FDCHI2_OWNPV"
        l0_m = "Ks_M"
        l0_proton_nn = "KsPi_MC15TuneV1_ProbNNpi"  # For Ks pion
        proton_nn = "P1_MC15TuneV1_ProbNNpi"  # For pion in norm channel
        h1_nn = "P0_MC15TuneV1_ProbNNk"
        h2_nn = "P2_MC15TuneV1_ProbNNk"
        bu_pt = "B_PT"
        bu_dtf_chi2 = "B_DTF_chi2"
        bu_ipchi2 = "B_IPCHI2_OWNPV"
        bu_fdchi2 = "B_FDCHI2_OWNPV"
    
    # Dictionary to store individual cut masks
    cut_masks = OrderedDict()
    cut_masks["initial"] = initial_mask.copy()
    
    # Current cumulative mask
    current_mask = initial_mask.copy()
    
    try:
        # Delta Z cut
        if l0_endvertex_z in data.fields and l0_ownpv_z in data.fields:
            delta_z_mask = ((data[l0_endvertex_z] - data[l0_ownpv_z]) > 20)
            cut_masks["delta_z"] = delta_z_mask
            current_mask = current_mask & delta_z_mask
        else:
            cut_masks["delta_z"] = None
        
        # FD chi2 cut
        if l0_fdchi2 in data.fields:
            fd_chi2_mask = (data[l0_fdchi2] > 45)
            cut_masks["l0_fdchi2"] = fd_chi2_mask
            current_mask = current_mask & fd_chi2_mask
        else:
            cut_masks["l0_fdchi2"] = None
        
        # Mass window cut - different for Lambda vs Ks
        if l0_m in data.fields:
            pdg_mass = 1115.6 if is_sig else 497.6  # Lambda vs Ks PDG mass
            window = 6 if is_sig else 15  # Wider window for Ks
            mass_window_mask = (np.abs(data[l0_m] - pdg_mass) < window)
            cut_masks["l0_mass"] = mass_window_mask
            current_mask = current_mask & mass_window_mask
        else:
            cut_masks["l0_mass"] = None
        
        # PID cuts
        if is_sig and proton_nn in data.fields:
            proton_pid_mask = (data[proton_nn] > 0.05)
            cut_masks["proton_pid"] = proton_pid_mask
            current_mask = current_mask & proton_pid_mask
        else:
            cut_masks["proton_pid"] = None if is_sig else np.ones(len(data), dtype=bool)
        
        if l0_proton_nn in data.fields:
            threshold = 0.2 if is_sig else 0.1  # Different threshold for proton vs pion
            l0_pid_mask = (data[l0_proton_nn] > threshold)
            cut_masks["l0_pid"] = l0_pid_mask
            current_mask = current_mask & l0_pid_mask
        else:
            cut_masks["l0_pid"] = None
        
        # Kaon PID product cut (for both channels)
        if h1_nn in data.fields and h2_nn in data.fields:
            kaon_pid_mask = ((data[h1_nn] * data[h2_nn]) > 0.04)
            cut_masks["kaon_pid"] = kaon_pid_mask
            current_mask = current_mask & kaon_pid_mask
        else:
            cut_masks["kaon_pid"] = None
        
        # B PT cut
        if bu_pt in data.fields:
            b_pt_mask = (data[bu_pt] > 3000)
            cut_masks["b_pt"] = b_pt_mask
            current_mask = current_mask & b_pt_mask
        else:
            cut_masks["b_pt"] = None
        
        # DTF chi2 cut
        if bu_dtf_chi2 in data.fields:
            dtf_chi2_mask = (data[bu_dtf_chi2] < 30)
            cut_masks["dtf_chi2"] = dtf_chi2_mask
            current_mask = current_mask & dtf_chi2_mask
        else:
            cut_masks["dtf_chi2"] = None
        
        # IP chi2 cut
        if bu_ipchi2 in data.fields:
            ip_chi2_mask = (data[bu_ipchi2] < 10)
            cut_masks["ip_chi2"] = ip_chi2_mask
            current_mask = current_mask & ip_chi2_mask
        else:
            cut_masks["ip_chi2"] = None
        
        # FD chi2 cut for B
        if bu_fdchi2 in data.fields:
            b_fd_chi2_mask = (data[bu_fdchi2] > 175)
            cut_masks["b_fdchi2"] = b_fd_chi2_mask
            current_mask = current_mask & b_fd_chi2_mask
        else:
            cut_masks["b_fdchi2"] = None
        
    except Exception as e:
        print(f"Error applying selection: {e}")
    
    # Store the final combined mask
    cut_masks["combined"] = current_mask
    
    if step_by_step:
        return cut_masks
    else:
        return current_mask

def calculate_efficiency(data, sample, include_trigger=True):
    """
    Calculate selection efficiency
    
    Parameters:
    -----------
    data : awkward array
        Data to use for efficiency calculation
    sample : str
        'signal' or 'norm' to determine which selection to apply
    include_trigger : bool
        Whether to include trigger efficiency
        
    Returns:
    --------
    dict
        Dictionary of efficiency results
    """
    total_events = len(data)
    
    # Apply trigger selection if requested
    if include_trigger:
        try:
            trigger_selection = trigger_mask(data, sample)
            data_after_trigger = data[trigger_selection]
            trigger_efficiency = len(data_after_trigger) / total_events if total_events > 0 else 0.0
        except Exception as e:
            print(f"Warning: Error applying trigger selection: {e}")
            print("Proceeding without trigger selection")
            data_after_trigger = data
            trigger_efficiency = 1.0
    else:
        data_after_trigger = data
        trigger_efficiency = 1.0
    
    # Get detailed cut masks
    cut_masks = create_selection_mask(data_after_trigger, sample, step_by_step=True)
    
    # Calculate initial counts
    initial_count = len(data_after_trigger)
    
    # Calculate efficiency for each cut
    efficiencies = {}
    efficiencies['total'] = {
        'n_tot': total_events,
        'n_pass': 0,  # Will be updated after all cuts
        'efficiency': 0.0  # Will be updated after all cuts
    }
    
    efficiencies['trigger'] = {
        'n_tot': total_events,
        'n_pass': initial_count,
        'efficiency': trigger_efficiency
    }
    
    # Track events passing each cut sequentially
    cumulative_mask = np.ones(initial_count, dtype=bool)
    
    # Process each cut
    for cut_name, mask in cut_masks.items():
        if cut_name in ['initial', 'combined']:
            continue  # Skip initial and combined masks
            
        if mask is not None:
            # Update cumulative mask
            cumulative_mask = cumulative_mask & mask
            
            # Calculate passing events and efficiency
            passing_events = np.sum(cumulative_mask)
            
            efficiencies[cut_name] = {
                'n_tot': int(initial_count),  # Ensure integer type
                'n_pass': int(passing_events),  # Ensure integer type
                'efficiency': float(passing_events) / initial_count if initial_count > 0 else 0.0
            }
    
    # Update total efficiency
    final_passing = np.sum(cut_masks['combined']) if 'combined' in cut_masks else 0
    efficiencies['total']['n_pass'] = int(final_passing)  # Ensure integer type
    efficiencies['total']['n_tot'] = int(total_events)    # Ensure integer type
    efficiencies['total']['efficiency'] = float(final_passing) / total_events if total_events > 0 else 0.0
    
    # Calculate relative efficiency (efficiency after trigger)
    efficiencies['relative'] = {
        'n_tot': int(initial_count),  # Ensure integer type
        'n_pass': int(final_passing),  # Ensure integer type
        'efficiency': float(final_passing) / initial_count if initial_count > 0 else 0.0
    }
    
    return efficiencies

def calculate_all_efficiencies(config, args):
    """
    Calculate efficiencies for all combinations of years and track types
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    dict
        Dictionary of all efficiency results
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
    
    # Initialize results dictionary - IMPORTANT: Use a different structure here that is 
    # compatible with the branching ratio calculation script
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
                sig_data = load_data(data_path=signal_data_dir, 
                                     decay_mode=signal_mode,
                                     years=[year], 
                                     tracks=[track])
                
                if sig_data is not None and len(sig_data) > 0:
                    sig_eff = calculate_efficiency(sig_data, 'signal')
                    
                    # Store detailed efficiency information
                    all_efficiencies['sig'][f"{category}_detailed"] = sig_eff
                    
                    # Also store in format expected by generate_result.py
                    all_efficiencies['sig'][category] = {
                        'n_tot': int(sig_eff['total']['n_tot']),
                        'n_pass': int(sig_eff['total']['n_pass'])
                    }
                    
                    log(f"Signal efficiency ({category}): {sig_eff['total']['efficiency']:.6f} ({sig_eff['total']['n_pass']}/{sig_eff['total']['n_tot']})", args=args)
                else:
                    log(f"No signal data found for {category}", args=args)
            except Exception as e:
                log(f"Error calculating signal efficiency for {category}: {e}", args=args)
            
            # Calculate normalization efficiencies
            try:
                norm_data = load_data(data_path=norm_data_dir, 
                                      decay_mode=norm_mode,
                                      years=[year], 
                                      tracks=[track])
                
                if norm_data is not None and len(norm_data) > 0:
                    norm_eff = calculate_efficiency(norm_data, 'norm')
                    
                    # Store detailed efficiency information
                    all_efficiencies['norm'][f"{category}_detailed"] = norm_eff
                    
                    # Also store in format expected by generate_result.py
                    all_efficiencies['norm'][category] = {
                        'n_tot': int(norm_eff['total']['n_tot']),
                        'n_pass': int(norm_eff['total']['n_pass'])
                    }
                    
                    log(f"Norm efficiency ({category}): {norm_eff['total']['efficiency']:.6f} ({norm_eff['total']['n_pass']}/{norm_eff['total']['n_tot']})", args=args)
                else:
                    log(f"No normalization data found for {category}", args=args)
            except Exception as e:
                log(f"Error calculating normalization efficiency for {category}: {e}", args=args)
    
    # Calculate combined efficiencies
    sig_total_passed = sig_total_events = 0
    norm_total_passed = norm_total_events = 0
    
    for year in years:
        for track in tracks:
            category = f"{year}_{track}"
            
            # Signal
            if category in all_efficiencies['sig']:
                sig_total_passed += all_efficiencies['sig'][category]['n_pass']
                sig_total_events += all_efficiencies['sig'][category]['n_tot']
                
            # Norm
            if category in all_efficiencies['norm']:
                norm_total_passed += all_efficiencies['norm'][category]['n_pass']
                norm_total_events += all_efficiencies['norm'][category]['n_tot']
    
    # Store combined efficiencies in expected format
    if sig_total_events > 0:
        all_efficiencies['sig']['all_all'] = {
            'n_tot': int(sig_total_events),
            'n_pass': int(sig_total_passed)
        }
        log(f"Signal combined efficiency: {sig_total_passed/sig_total_events:.6f} ({sig_total_passed}/{sig_total_events})", args=args)
    
    if norm_total_events > 0:
        all_efficiencies['norm']['all_all'] = {
            'n_tot': int(norm_total_events),
            'n_pass': int(norm_total_passed)
        }
        log(f"Norm combined efficiency: {norm_total_passed/norm_total_events:.6f} ({norm_total_passed}/{norm_total_events})", args=args)
    
    return all_efficiencies

def print_summary(all_efficiencies, config, args):
    """
    Print summary of all efficiencies
    
    Parameters:
    -----------
    all_efficiencies : dict
        Dictionary of all efficiency results
    config : dict
        Configuration dictionary
    args : argparse.Namespace
        Command line arguments
    """
    years = config.get('years', [])
    tracks = config.get('tracks', [])
    
    log("\n" + "="*80, args=args)
    log(" "*30 + "EFFICIENCY SUMMARY", args=args)
    log("="*80, args=args)
    
    # Print individual results by channel
    for channel, title in [('sig', 'SIGNAL CHANNEL (B+ → Λ̄0pK+K-)'), 
                          ('norm', 'NORMALIZATION CHANNEL (B+ → K0sπ+K+K-)')]:
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
                    n_tot = eff_data['n_tot']
                    n_pass = eff_data['n_pass']
                    eff = n_pass / n_tot if n_tot > 0 else 0.0
                    
                    log(f"{category:<15} {n_tot:<15} {n_pass:<15} {eff:.6f}", args=args)
                else:
                    log(f"{category:<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}", args=args)
        
        # Combined efficiency
        category = "all_all"
        if category in all_efficiencies[channel]:
            eff_data = all_efficiencies[channel][category]
            n_tot = eff_data['n_tot']
            n_pass = eff_data['n_pass']
            eff = n_pass / n_tot if n_tot > 0 else 0.0
            
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
            
            sig_n_tot = sig_data.get('n_tot', 0)
            sig_n_pass = sig_data.get('n_pass', 0)
            norm_n_tot = norm_data.get('n_tot', 0)
            norm_n_pass = norm_data.get('n_pass', 0)
            
            sig_eff = sig_n_pass / sig_n_tot if sig_n_tot > 0 else 0
            norm_eff = norm_n_pass / norm_n_tot if norm_n_tot > 0 else 0
            
            if sig_eff > 0 and norm_eff > 0:
                ratio = norm_eff / sig_eff
                log(f"{category:<15} {sig_eff:.6f}{' '*9} {norm_eff:.6f}{' '*9} {ratio:.6f}", args=args)
            else:
                log(f"{category:<15} {sig_eff:.6f if sig_eff else 'N/A':<15} {norm_eff:.6f if norm_eff else 'N/A':<15} {'N/A':<15}", args=args)
    
    # Combined efficiency ratio
    category = "all_all"
    
    sig_data = all_efficiencies['sig'].get(category, {})
    norm_data = all_efficiencies['norm'].get(category, {})
    
    sig_n_tot = sig_data.get('n_tot', 0)
    sig_n_pass = sig_data.get('n_pass', 0)
    norm_n_tot = norm_data.get('n_tot', 0)
    norm_n_pass = norm_data.get('n_pass', 0)
    
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
    
    # Convert NumPy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        """Convert NumPy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_numpy_types(obj.tolist())
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    # Convert all NumPy types in the results dictionary
    serializable_efficiencies = convert_numpy_types(all_efficiencies)
    
    # Save results to JSON file
    with open(output_path, 'w') as f:
        json.dump(serializable_efficiencies, f, indent=2)
    
    log(f"\nEfficiency results saved to: {output_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())