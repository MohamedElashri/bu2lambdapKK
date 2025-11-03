#!/usr/bin/env python3
"""
Standalone script to plot optimization cut variable distributions
Shows MC (left) and Real Data (right) for each year and combined
With vertical lines indicating optimal cut values and shaded accepted regions

Usage:
    cd ana/scripts
    python plot_optimization_variables.py
    python plot_optimization_variables.py --years 2016,2017,2018
    python plot_optimization_variables.py --state jpsi
"""

import sys
from pathlib import Path
import argparse
import pandas as pd

# Add parent directory (ana) to path to access modules
ana_dir = Path(__file__).parent.parent
sys.path.insert(0, str(ana_dir))

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from modules.data_handler import TOMLConfig, DataManager
from modules.lambda_selector import LambdaSelector


# Define the 7 optimization variables with their plotting properties
# Separate ranges for MC and data where needed
OPTIMIZATION_VARIABLES = {
    'Bu_PT': {
        'label': r'$p_T(B^+)$ [MeV/$c$]',
        'range_mc': (0, 10000),
        'range_data': (2900, 10000),  # Start from 2900 for data
        'bins': 100,
        'scale': 1.0  # MeV
    },
    'Bu_FDCHI2_OWNPV': {
        'label': r'$\chi^2_{\mathrm{FD}}(B^+)$',
        'range_mc': (0, 1000),
        'range_data': (100, 1000),  # Start from 100 for data
        'bins': 100,
        'scale': 1.0
    },
    'Bu_IPCHI2_OWNPV': {
        'label': r'$\chi^2_{\mathrm{IP}}(B^+)$',
        'range_mc': (0, 14),  # End at 14 for MC
        'range_data': (0, 12),  # End at 12 for data
        'bins': 100,
        'scale': 1.0
    },
    'Bu_DTF_chi2': {
        'label': r'$\chi^2_{\mathrm{DTF}}(B^+)$',
        'range_mc': (0, 100),
        'range_data': (0, 35),  # End at 35 for data
        'bins': 100,
        'scale': 1.0
    },
    'h1_ProbNNk': {
        'label': r'ProbNN$_K(K^+)$',
        'range_mc': (0, 1.0),
        'range_data': (0, 1.0),
        'bins': 100,
        'scale': 1.0
    },
    'h2_ProbNNk': {
        'label': r'ProbNN$_K(K^-)$',
        'range_mc': (0, 1.0),
        'range_data': (0, 1.0),
        'bins': 100,
        'scale': 1.0
    },
    'p_ProbNNp': {
        'label': r'ProbNN$_p(p)$',
        'range_mc': (0, 1.0),
        'range_data': (0, 1.0),
        'bins': 100,
        'scale': 1.0
    }
}


def plot_variable_comparison(mc_events, data_events, year_label, variable_name, 
                             var_props, optimal_cuts_dict):
    """
    Create vertical plots for MC (top) and Data (bottom) for one variable
    
    Args:
        mc_events: Awkward array of MC events
        data_events: Awkward array of data events
        year_label: String like "2016" or "2016-2018"
        variable_name: Branch name (e.g., "Bu_PT")
        var_props: Dictionary with plotting properties
        optimal_cuts_dict: Dict with {state: (cut_value, cut_type)} for all 4 states
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Use separate ranges for MC and data
    var_range_mc = var_props['range_mc']
    var_range_data = var_props['range_data']
    bins = var_props['bins']
    label = var_props['label']
    
    # Colors for different states
    state_colors = {
        'jpsi': 'red',
        'etac': 'orange',
        'chic0': 'purple',
        'chic1': 'brown'
    }
    state_labels = {
        'jpsi': r'J/$\psi$',
        'etac': r'$\eta_c$',
        'chic0': r'$\chi_{c0}$',
        'chic1': r'$\chi_{c1}$'
    }
    
    # Plot MC (left)
    if mc_events is not None and len(mc_events) > 0 and variable_name in mc_events.fields:
        mc_var = mc_events[variable_name]
        # Flatten if jagged/nested array
        try:
            mc_data = ak.to_numpy(mc_var)
        except ValueError:
            # If it's a jagged array, flatten it completely
            mc_data = ak.to_numpy(ak.flatten(mc_var))
        mc_data = mc_data * var_props['scale']
        
        ax1.hist(mc_data, bins=bins, range=var_range_mc,
                histtype='step', color='blue', linewidth=1.5,
                label='MC', density=True)
        
        # Add vertical lines for all 4 states' optimal cuts
        for i, (state, (cut_value, cut_type)) in enumerate(optimal_cuts_dict.items()):
            linestyle = '--' if i == 0 else ':'
            ax1.axvline(cut_value, color=state_colors[state], linestyle=linestyle, 
                       linewidth=1.5, label=f'{state_labels[state]}: {cut_value:.1f}',
                       alpha=0.8)
        
        # Shade accepted region based on first state (they're all the same anyway)
        first_state = list(optimal_cuts_dict.keys())[0]
        cut_value, cut_type = optimal_cuts_dict[first_state]
        if cut_type == "greater":
            # Keep events > cut value (shade right side)
            ax1.axvspan(cut_value, var_range_mc[1], alpha=0.15, color='green',
                       label='Accepted region')
        else:  # less
            # Keep events < cut value (shade left side)
            ax1.axvspan(var_range_mc[0], cut_value, alpha=0.15, color='green',
                       label='Accepted region')
        
        ax1.set_xlabel(label, fontsize=12)
        ax1.set_ylabel('Normalized Events', fontsize=12)
        # Replace "Combined" with year range if applicable
        title_label = year_label if year_label != "Combined" else "2016-2018"
        ax1.set_title(f'MC - {title_label}', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(var_range_mc)
    else:
        ax1.text(0.5, 0.5, f'No MC data available\nfor {variable_name}',
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_xlabel(label, fontsize=12)
        ax1.set_ylabel('Normalized Events', fontsize=12)
        ax1.set_title(f'MC - {year_label}', fontsize=14, fontweight='bold')
    
    # Plot Data (right)
    if data_events is not None and len(data_events) > 0 and variable_name in data_events.fields:
        data_var = data_events[variable_name]
        # Flatten if jagged/nested array
        try:
            data_data = ak.to_numpy(data_var)
        except ValueError:
            # If it's a jagged array, flatten it completely
            data_data = ak.to_numpy(ak.flatten(data_var))
        data_data = data_data * var_props['scale']
        
        ax2.hist(data_data, bins=bins, range=var_range_data,
                histtype='step', color='black', linewidth=1.5,
                label='$B^+ \\to \\bar{\\Lambda} p K^+ K^-$', density=True)
        
        # Add vertical lines for all 4 states' optimal cuts
        for i, (state, (cut_value, cut_type)) in enumerate(optimal_cuts_dict.items()):
            linestyle = '--' if i == 0 else ':'
            ax2.axvline(cut_value, color=state_colors[state], linestyle=linestyle,
                       linewidth=1.5, label=f'{state_labels[state]}: {cut_value:.1f}',
                       alpha=0.8)
        
        # Shade accepted region based on first state
        first_state = list(optimal_cuts_dict.keys())[0]
        cut_value, cut_type = optimal_cuts_dict[first_state]
        if cut_type == "greater":
            ax2.axvspan(cut_value, var_range_data[1], alpha=0.15, color='green',
                       label='Accepted region')
        else:
            ax2.axvspan(var_range_data[0], cut_value, alpha=0.15, color='green',
                       label='Accepted region')
        
        ax2.set_xlabel(label, fontsize=12)
        ax2.set_ylabel('Normalized Events', fontsize=12)
        # Replace "Combined" with year range if applicable
        title_label = year_label if year_label != "Combined" else "2016-2018"
        ax2.set_title(f'$B^+ \\to \\bar{{\\Lambda}} p K^+ K^-$ - {title_label}', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(var_range_data)
    else:
        ax2.text(0.5, 0.5, f'No data available\nfor {variable_name}',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_xlabel(label, fontsize=12)
        ax2.set_ylabel('Normalized Events', fontsize=12)
        ax2.set_title(f'$B^+ \\to \\bar{{\\Lambda}} p K^+ K^-$ - {year_label}', 
                     fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot optimization variable distributions with optimal cuts'
    )
    parser.add_argument('--years', type=str, default='2016,2017,2018',
                       help='Comma-separated years to plot (default: 2016,2017,2018)')
    parser.add_argument('--mc-state', type=str, default='Jpsi',
                       help='MC state to use for plots (default: Jpsi)')
    parser.add_argument('--state', type=str, default='jpsi',
                       help='State to get optimal cuts from (default: jpsi)')
    args = parser.parse_args()
    
    years = [int(y.strip()) for y in args.years.split(',')]
    mc_state = args.mc_state
    opt_state = args.state.lower()
    
    print("=" * 80)
    print("OPTIMIZATION VARIABLES DISTRIBUTION PLOTTER")
    print("=" * 80)
    print(f"Years: {years}")
    print(f"MC state: {mc_state}")
    print(f"Optimization state: {opt_state}")
    print("=" * 80)
    print()
    
    # Initialize configuration and data manager
    ana_dir = Path(__file__).parent.parent
    config = TOMLConfig(config_dir=str(ana_dir / "config"))
    data_manager = DataManager(config)
    lambda_selector = LambdaSelector(config)
    
    # Load optimal cuts for ALL 4 states
    all_optimal_cuts = {}
    for state in ['jpsi', 'etac', 'chic0', 'chic1']:
        cuts_file = ana_dir / "tables" / f"optimized_cuts_nd_{state}.csv"
        if cuts_file.exists():
            df = pd.read_csv(cuts_file)
            all_optimal_cuts[state] = df
            print(f"✓ Loaded optimal cuts for {state}: {len(df)} variables")
        else:
            print(f"⚠️  Warning: Cuts file not found: {cuts_file}")
    
    if not all_optimal_cuts:
        print("⚠️  No optimal cuts files found!")
        return
    
    # Output directory
    output_dir = ana_dir / "plots" / "optimization_variables"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track types
    track_types = ["LL", "DD"]
    
    # Storage for combined data
    all_mc_events = []
    all_data_events = []
    
    # Process each year
    for year in years:
        print(f"\n{'=' * 80}")
        print(f"YEAR {year}")
        print(f"{'=' * 80}\n")
        
        year_mc_events = []
        year_data_events = []
        
        # Load MC and Data for this year
        for magnet in ["MD", "MU"]:
            for track_type in track_types:
                print(f"\nLoading {year} {magnet} {track_type}...")
                
                # Load MC
                try:
                    mc_events = data_manager.load_tree(
                        mc_state, year, magnet, track_type
                    )
                    if mc_events is not None:
                        mc_events = data_manager.compute_derived_branches(mc_events)
                        # Skip trigger selection for MC
                        year_mc_events.append(mc_events)
                except Exception as e:
                    print(f"  ⚠️  MC loading failed: {e}")
                
                # Load Data
                try:
                    data_events = data_manager.load_tree(
                        "data", year, magnet, track_type
                    )
                    if data_events is not None:
                        data_events = data_manager.compute_derived_branches(data_events)
                        data_events = data_manager.apply_trigger_selection(data_events)
                        year_data_events.append(data_events)
                except Exception as e:
                    print(f"  ⚠️  Data loading failed: {e}")
        
        # Combine for this year
        if year_mc_events:
            year_mc_combined = ak.concatenate(year_mc_events)
            all_mc_events.append(year_mc_combined)
            print(f"\n✓ Combined MC {year}: {len(year_mc_combined)} events")
        else:
            year_mc_combined = None
            print(f"\n⚠️  No MC data for {year}")
        
        if year_data_events:
            year_data_combined = ak.concatenate(year_data_events)
            all_data_events.append(year_data_combined)
            print(f"✓ Combined Data {year}: {len(year_data_combined)} events")
        else:
            year_data_combined = None
            print(f"⚠️  No data for {year}")
        
        # Create plots for this year (one PDF per variable)
        print(f"\nCreating plots for {year}...")
        year_output_dir = output_dir / str(year)
        year_output_dir.mkdir(parents=True, exist_ok=True)
        
        for var_name, var_props in OPTIMIZATION_VARIABLES.items():
            # Get optimal cuts for this variable from ALL 4 states
            optimal_cuts_for_var = {}
            for state, cuts_df in all_optimal_cuts.items():
                var_row = cuts_df[cuts_df['branch_name'] == var_name]
                if not var_row.empty:
                    optimal_cuts_for_var[state] = (
                        var_row.iloc[0]['optimal_cut'],
                        var_row.iloc[0]['cut_type']
                    )
            
            if not optimal_cuts_for_var:
                print(f"  ⚠️  No optimal cuts found for {var_name}, skipping")
                continue
            
            print(f"  - Plotting {var_name}")
            
            fig = plot_variable_comparison(
                year_mc_combined,
                year_data_combined,
                str(year),
                var_name,
                var_props,
                optimal_cuts_for_var
            )
            
            # Save individual PDF for this variable
            var_pdf = year_output_dir / f"{var_name}_{year}.pdf"
            fig.savefig(var_pdf, bbox_inches='tight')
            plt.close(fig)
        
        print(f"✓ Saved plots to: {year_output_dir}")
    
    # Create combined plot
    print(f"\n{'=' * 80}")
    print("COMBINED (All Years)")
    print(f"{'=' * 80}\n")
    
    if all_mc_events:
        combined_mc = ak.concatenate(all_mc_events)
        print(f"✓ Combined MC: {len(combined_mc)} events")
    else:
        combined_mc = None
        print("⚠️  No MC data available")
    
    if all_data_events:
        combined_data = ak.concatenate(all_data_events)
        print(f"✓ Combined Data: {len(combined_data)} events")
    else:
        combined_data = None
        print("⚠️  No data available")
    
    print("\nCreating combined plots...")
    combined_output_dir = output_dir / "combined"
    combined_output_dir.mkdir(parents=True, exist_ok=True)
    
    for var_name, var_props in OPTIMIZATION_VARIABLES.items():
        # Get optimal cuts for this variable from ALL 4 states
        optimal_cuts_for_var = {}
        for state, cuts_df in all_optimal_cuts.items():
            var_row = cuts_df[cuts_df['branch_name'] == var_name]
            if not var_row.empty:
                optimal_cuts_for_var[state] = (
                    var_row.iloc[0]['optimal_cut'],
                    var_row.iloc[0]['cut_type']
                )
        
        if not optimal_cuts_for_var:
            print(f"  ⚠️  No optimal cuts found for {var_name}, skipping")
            continue
        
        print(f"  - Plotting {var_name}")
        
        fig = plot_variable_comparison(
            combined_mc,
            combined_data,
            "Combined",
            var_name,
            var_props,
            optimal_cuts_for_var
        )
        
        # Save individual PDF for this variable
        var_pdf = combined_output_dir / f"{var_name}_combined.pdf"
        fig.savefig(var_pdf, bbox_inches='tight')
        plt.close(fig)
    
    print(f"✓ Saved combined plots to: {combined_output_dir}")
    
    print(f"\n{'=' * 80}")
    print(f"✓ All plots saved to: {output_dir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
