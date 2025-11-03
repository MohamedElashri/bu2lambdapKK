#!/usr/bin/env python3
"""
Standalone script to plot Lambda mass distributions
Plots MC (left) and Real Data (right) for each year and combined

Usage:
    python plot_lambda_mass.py
    python plot_lambda_mass.py --years 2016,2017,2018
"""

import sys
from pathlib import Path
import argparse

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from modules.data_handler import TOMLConfig, DataManager
from modules.lambda_selector import LambdaSelector


def plot_lambda_mass_comparison(mc_events, data_events, year_label, lambda_cuts):
    """
    Create side-by-side plots for MC and Data Lambda mass
    
    Args:
        mc_events: Awkward array of MC events
        data_events: Awkward array of data events
        year_label: String like "2016" or "Combined"
        lambda_cuts: Dict with lambda selection cuts
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Define Lambda mass range
    mass_min = lambda_cuts["mass_min"]
    mass_max = lambda_cuts["mass_max"]
    
    # Extended range for visualization
    plot_min = 1100
    plot_max = 1130
    bins = 60
    
    # Plot MC (left)
    if mc_events is not None and len(mc_events) > 0:
        # Use L0_MM if available, fallback to L0_M
        if "L0_MM" in mc_events.fields:
            mc_mass = ak.to_numpy(mc_events["L0_MM"])
        elif "L0_M" in mc_events.fields:
            mc_mass = ak.to_numpy(mc_events["L0_M"])
        else:
            mc_mass = []
        
        if len(mc_mass) > 0:
            ax1.hist(mc_mass, bins=bins, range=(plot_min, plot_max),
                    histtype='step', color='blue', linewidth=1.5,
                    label='MC')
            
            # Add vertical lines for cuts
            ax1.axvline(mass_min, color='red', linestyle='--', linewidth=1.5, 
                       label='Cut region')
            ax1.axvline(mass_max, color='red', linestyle='--', linewidth=1.5)
            
            # Shade signal region
            ax1.axvspan(mass_min, mass_max, alpha=0.2, color='green')
            
            ax1.set_xlabel(r'$M(\Lambda)$ [MeV/$c^2$]', fontsize=12)
            ax1.set_ylabel('Events / 0.5 MeV', fontsize=12)
            ax1.set_title(f'MC - {year_label}', fontsize=14, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No MC data available', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14)
        ax1.set_xlabel(r'$M(\Lambda)$ [MeV/$c^2$]', fontsize=12)
        ax1.set_ylabel('Events', fontsize=12)
        ax1.set_title(f'MC - {year_label}', fontsize=14, fontweight='bold')
    
    # Plot Data (right)
    if data_events is not None and len(data_events) > 0:
        if "L0_MM" in data_events.fields:
            data_mass = ak.to_numpy(data_events["L0_MM"])
        elif "L0_M" in data_events.fields:
            data_mass = ak.to_numpy(data_events["L0_M"])
        else:
            data_mass = []
        
        if len(data_mass) > 0:
            ax2.hist(data_mass, bins=bins, range=(plot_min, plot_max),
                    histtype='step', color='black', linewidth=1.5,
                    label='$B^+ \\to \\bar{{\\Lambda}} p K^+ K^-$')
            
            # Add vertical lines for cuts
            ax2.axvline(mass_min, color='red', linestyle='--', linewidth=1.5,
                       label='Cut region')
            ax2.axvline(mass_max, color='red', linestyle='--', linewidth=1.5)
            
            # Shade signal region
            ax2.axvspan(mass_min, mass_max, alpha=0.2, color='green')
            
            ax2.set_xlabel(r'$M(\Lambda)$ [MeV/$c^2$]', fontsize=12)
            ax2.set_ylabel('Events / 0.5 MeV', fontsize=12)
            ax2.set_title(f'$B^+ \\to \\bar{{\\Lambda}} p K^+ K^-$ - {year_label}', fontsize=14, fontweight='bold')
            ax2.legend(loc='upper right', fontsize=10)
            ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No data available',
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_xlabel(r'$M(\Lambda)$ [MeV/$c^2$]', fontsize=12)
        ax2.set_ylabel('Events', fontsize=12)
        ax2.set_title(f'$B^+ \\to \\bar{{\\Lambda}} p K^+ K^-$ - {year_label}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot Lambda mass distributions')
    parser.add_argument('--years', type=str, default='2016,2017,2018',
                       help='Comma-separated years to plot (default: 2016,2017,2018)')
    parser.add_argument('--mc-state', type=str, default='Jpsi',
                       help='MC state to use for plots (default: Jpsi)')
    args = parser.parse_args()
    
    years = [int(y.strip()) for y in args.years.split(',')]
    mc_state = args.mc_state
    
    print("=" * 80)
    print("LAMBDA MASS DISTRIBUTION PLOTTER")
    print("=" * 80)
    print(f"Years: {years}")
    print(f"MC state: {mc_state}")
    print("=" * 80)
    print()
    
    # Initialize configuration and data manager
    config = TOMLConfig(config_dir="./config")
    data_manager = DataManager(config)
    lambda_selector = LambdaSelector(config)
    
    # Get Lambda cuts for reference
    lambda_cuts = config.get_lambda_cuts()
    
    # Output directory
    output_dir = Path("./plots/lambda_mass")
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
        
        # Load MC and Data for this year (both magnets, both track types)
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
                        # DON'T apply Lambda mass cuts - show full distribution
                        # (Other Lambda quality cuts like FD_CHI2 could be applied here if needed)
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
                        # DON'T apply Lambda mass cuts - show full distribution
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
        
        # Create plot for this year and save as separate PDF
        print(f"\nCreating plot for {year}...")
        fig = plot_lambda_mass_comparison(
            year_mc_combined, 
            year_data_combined, 
            str(year),
            lambda_cuts
        )
        year_pdf = output_dir / f"lambda_mass_{year}_{mc_state}.pdf"
        fig.savefig(year_pdf, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved plot to: {year_pdf}")
    
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
    
    print("\nCreating combined plot...")
    fig = plot_lambda_mass_comparison(
        combined_mc,
        combined_data,
        "Combined",
        lambda_cuts
    )
    combined_pdf = output_dir / f"lambda_mass_combined_{mc_state}.pdf"
    fig.savefig(combined_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved combined plot to: {combined_pdf}")
    
    print(f"\n{'=' * 80}")
    print(f"✓ All plots saved to: {output_dir}")
    print(f"{'=' * 80}\n")
if __name__ == "__main__":
    main()
