#!/usr/bin/env python3
"""
Cut analyzer for B+ → Λ̅pK+K- branching ratio analysis.

All cut parameters are defined as constants in the script.
Only the sample (signal or norm) needs to be specified as an argument.

Usage:
  python cut_analyzer.py --sample [signal|norm]
"""

import yaml, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from loaders import load_data
from selections import trigger_mask

# Configure argument parser (only sample is required)
parser = argparse.ArgumentParser(description='Cut analyzer with predefined constants')
parser.add_argument('--sample', type=str, choices=['signal', 'norm'], required=True,
                   help='Sample to analyze (signal or norm)')
args = parser.parse_args()

# Load configuration
CFG = yaml.safe_load(open("config.yml"))

# Create output directory
PLOTS_DIR = "cut_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

#####################################################################
# DEFINE ALL CUT CONSTANTS HERE - MODIFY THESE AS NEEDED
#####################################################################

# pT cut parameters
PT_CUT = {
    'signal_branch': 'Bu_PT',
    'norm_branch': 'B_PT',
    'title': 'B^{+} p_{T}',
    'cut_value': 3000,
    'direction': '>',
    'range': (0, 30000),
    'bins': 60,
    'unit': 'MeV/c'
}

# IP chi2 cut parameters 
IPCHI2_CUT = {
    'signal_branch': 'Bu_IPCHI2_OWNPV',
    'norm_branch': 'B_IPCHI2_OWNPV',
    'title': 'B^{+} IP\\chi^{2}',
    'cut_value': 10,
    'direction': '<',
    'range': (0, 30),
    'bins': 50,
    'unit': ''
}

# FD chi2 cut parameters
FDCHI2_CUT = {
    'signal_branch': 'Bu_FDCHI2_OWNPV',
    'norm_branch': 'B_FDCHI2_OWNPV',
    'title': 'B^{+} FD\\chi^{2}',
    'cut_value': 250,
    'direction': '>',
    'range': (0, 2000),
    'bins': 50,
    'unit': ''
}

# DTF chi2 cut parameters
DTF_CHI2_CUT = {
    'signal_branch': 'Bu_DTF_chi2',
    'norm_branch': 'B_DTF_chi2',
    'title': 'B^{+} DTF \\chi^{2}',
    'cut_value': 25,
    'direction': '<',
    'range': (0, 50),
    'bins': 50,
    'unit': ''
}

# List of all cuts to analyze
# UNCOMMENT/COMMENT THE ONES WE WANT TO RUN
CUTS_TO_ANALYZE = [
    PT_CUT,
    IPCHI2_CUT,
    FDCHI2_CUT,
    DTF_CHI2_CUT
]

#####################################################################
# END OF CONSTANTS - CODE BELOW SHOULD NOT NEED MODIFICATION
#####################################################################

import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = False 
plt.rcParams["font.family"] = "serif"

def load_all_data(sample, branches_to_load):
    """Load data for all years and track types combined."""
    if sample == 'signal':
        data_path = CFG['signal_data_dir']
        decay_mode = "L0barPKpKm"
    else:
        data_path = CFG['norm_data_dir']
        decay_mode = CFG.get('norm_mode_data', "KSKmKpPip")
    
    print(f"Loading branches: {branches_to_load}")
    
    # Load all data (combining years and track types)
    data_chunks = []
    
    # Load data for each combination
    for year in CFG['years']:
        for track in CFG['tracks']:
            print(f"Loading {sample} data for {year} {track}...")
            # Include our branches as additional branches
            data = load_data(data_path=data_path, decay_mode=decay_mode,
                           years=[year], tracks=[track],
                           additional_branches=branches_to_load)
            if data is not None and len(data) > 0:
                print(f"  Loaded {len(data)} events")
                data_chunks.append(data)
    
    # Combine all data
    if data_chunks:
        try:
            combined_data = ak.concatenate(data_chunks)
            print(f"Combined dataset: {len(combined_data)} events")
            return combined_data
        except Exception as e:
            print(f"Error combining data: {e}")
            # Return the first chunk if we can't combine
            if data_chunks:
                return data_chunks[0]
    
    return None

def analyze_cut(data, sample, cut_params):
    """Analyze a specific cut using the provided parameters."""
    # Determine which branch to use based on sample
    branch = cut_params['signal_branch'] if sample == 'signal' else cut_params['norm_branch']
    
    # Check if branch exists
    if branch not in data.fields:
        print(f"Error: {branch} not found in data fields: {data.fields}")
        return
    
    # Get cut parameters
    cut_value = cut_params['cut_value']
    direction = cut_params['direction']
    plot_range = cut_params['range']
    bins = cut_params['bins']
    title = cut_params['title']
    unit = cut_params['unit']
    
    # Calculate bin width for y-axis label
    bin_width = (plot_range[1] - plot_range[0]) / bins
    
    # Apply trigger
    trigger_mask_array = trigger_mask(data, sample)
    n_trigger = np.sum(trigger_mask_array)
    trigger_eff = n_trigger / len(data)
    print(f"Trigger efficiency: {trigger_eff:.2%} ({n_trigger}/{len(data)} events)")
    
    # Get events passing trigger
    triggered_data = data[trigger_mask_array]
    
    # Apply variable cut on triggered events
    values = triggered_data[branch]
    if direction == '>':
        cut_mask = values > cut_value
    else:
        cut_mask = values < cut_value
        
    n_pass_cut = np.sum(cut_mask)
    cut_eff = n_pass_cut / len(triggered_data)
    print(f"Cut efficiency: {cut_eff:.2%} ({n_pass_cut}/{len(triggered_data)} events)")
    
    # Get events passing both trigger and cut
    cut_data = triggered_data[cut_mask]
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    
    # Convert to numpy arrays for plotting
    triggered_values = ak.to_numpy(triggered_data[branch])
    cut_values = ak.to_numpy(cut_data[branch])
    
    # Plot histograms
    plt.hist(triggered_values, bins=bins, range=plot_range, 
             alpha=0.6, label=r'trigger', density=False)
    plt.hist(cut_values, bins=bins, range=plot_range, 
             alpha=0.6, label=r'cut', density=False)
    
    # Add labels
    unit_label = f" [{unit}]" if unit else ""
    plt.xlabel(rf"{title}{unit_label}")
    plt.ylabel(rf"Candidates / {bin_width:.1f} {unit}")
    
    # Add cut line
    plt.axvline(x=cut_value, color='r', linestyle='--', alpha=0.7)
    plt.text(cut_value, plt.ylim()[1]*0.9, rf"Cut: {cut_value} {direction}", 
             rotation=90, verticalalignment='top', horizontalalignment='right')
    
    # Add title
    sample_name = "Signal" if sample == "signal" else "Normalization"
    plt.title(rf"{sample_name}: {title} Cut")
    
    # Add efficiency text box
    plt.figtext(0.7, 0.75, 
                rf"Cut efficiency: {cut_eff:.2%}\n"
                rf"Events after trigger: {len(triggered_data)}\n"
                rf"Events after cut: {n_pass_cut}",
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add legend and grid
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create filename from branch name
    var_name = branch.split('_')[-1].lower()  # Extract the variable name part
    filename_base = f"{sample}_{var_name}"
    
    # Save plot
    plt.savefig(f"{PLOTS_DIR}/{filename_base}_cut.pdf")
    plt.savefig(f"{PLOTS_DIR}/{filename_base}_cut.png")
    print(f"Plot saved to {PLOTS_DIR}/{filename_base}_cut.pdf")
    
    # Save efficiency info to text file
    with open(f"{PLOTS_DIR}/{filename_base}_efficiency.txt", 'w') as f:
        f.write(f"Sample: {sample}\n")
        f.write(f"Branch analyzed: {branch}\n")
        f.write(f"Cut applied: {branch} {direction} {cut_value} {unit}\n")
        f.write(f"Total events: {len(data)}\n")
        f.write(f"Events passing trigger: {n_trigger} ({trigger_eff:.2%})\n")
        f.write(f"Events passing trigger and cut: {n_pass_cut} ({cut_eff:.2%} of triggered events)\n")
        f.write(f"Overall efficiency (trigger and cut): {n_pass_cut/len(data):.2%}\n")
    
    return {
        'branch': branch,
        'cut_value': cut_value,
        'direction': direction,
        'trigger_eff': trigger_eff,
        'cut_eff': cut_eff,
        'overall_eff': n_pass_cut/len(data)
    }

def main():
    # Get the branches we need to load
    branches_to_load = []
    for cut in CUTS_TO_ANALYZE:
        if args.sample == 'signal':
            branches_to_load.append(cut['signal_branch'])
        else:
            branches_to_load.append(cut['norm_branch'])
    
    # Load all data with the required branches
    data = load_all_data(args.sample, branches_to_load)
    if data is None:
        print("No data found. Exiting.")
        return
    
    # Analyze each cut
    print(f"\nAnalyzing {len(CUTS_TO_ANALYZE)} cuts for {args.sample} sample")
    results = []
    
    for cut in CUTS_TO_ANALYZE:
        branch = cut['signal_branch'] if args.sample == 'signal' else cut['norm_branch']
        print(f"\n--- Analyzing {branch} cut ---")
        result = analyze_cut(data, args.sample, cut)
        if result:
            results.append(result)
    
    # Create summary of all cuts
    if results:
        print("\nCut Efficiency Summary:")
        print("-" * 80)
        print(f"{'Branch':<20} {'Cut':<15} {'Trigger Eff':<15} {'Cut Eff':<15} {'Overall Eff':<15}")
        print("-" * 80)
        
        for result in results:
            cut_str = f"{result['direction']} {result['cut_value']}"
            # Fixed format specifiers
            print(f"{result['branch']:<20} {cut_str:<15} {result['trigger_eff']*100:.2f}%{'':<10} {result['cut_eff']*100:.2f}%{'':<10} {result['overall_eff']*100:.2f}%{'':<10}")
        
        print("-" * 80)
        
        # Save summary to file
        with open(f"{PLOTS_DIR}/{args.sample}_cut_summary.txt", 'w') as f:
            f.write("Cut Efficiency Summary\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Branch':<20} {'Cut':<15} {'Trigger Eff':<15} {'Cut Eff':<15} {'Overall Eff':<15}\n")
            f.write("-" * 80 + "\n")
            
            for result in results:
                cut_str = f"{result['direction']} {result['cut_value']}"
                # Fixed format here too
                f.write(f"{result['branch']:<20} {cut_str:<15} {result['trigger_eff']*100:.2f}%{'':<10} {result['cut_eff']*100:.2f}%{'':<10} {result['overall_eff']*100:.2f}%{'':<10}\n")
            
            f.write("-" * 80 + "\n")

if __name__ == "__main__":
    main()