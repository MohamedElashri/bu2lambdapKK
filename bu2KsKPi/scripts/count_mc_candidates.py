#!/usr/bin/env python3
"""
count the number of events/candidates in the MC data for different decay modes and tuple types.
This script generates a LaTeX table summarizing the statistics.

Usage:
  python count_mc_candidates.py [--data-dir DATA_DIR]

Options:
  --data-dir DATA_DIR  Directory containing the processed ROOT files
"""

import os
import sys
import glob
import argparse
from typing import Dict, List, Union
import numpy as np
import uproot
import awkward as ak

# Try to import the load_mc function from the current directory
try:
    from utils.load_mc import load_mc, DECAY_MODES, TUPLE_TYPES, AVAILABLE_YEARS
    print("Successfully imported load_mc functions")
except ImportError:
    print("Warning: Unable to import load_mc module. Make sure it's in the same directory.")
    print("Continuing with fallback method only...")
    
    # Define constants that would normally be imported
    DECAY_MODES = {
        "B2K0s2PipPimKmPipKp": "B+ → (K0_S → π+π-)K-π+K+",
        "B2Jpsi2K0s2PipPimKmPipKp": "B+ → (J/ψ → (K0_S → π+π-)K-π+)K+",
        "B2Etac2K0s2PipPimKmPipKp": "B+ → (ηc → (K0_S → π+π-)K-π+)K+",
        "B2Etac2S2K0s2PipPimKmPipKp": "B+ → (ηc(2S) → (K0_S → π+π-)K-π+)K+",
        "B2Chic12K0s2PipPimKmPipKp": "B+ → (χc1 → (K0_S → π+π-)K-π+)K+"
    }
    
    TUPLE_TYPES = [
        "KSKmKpPip_DD",
        "KSKmKpPip_LL",
        "KSKpKpPim_DD",
        "KSKpKpPim_LL"
    ]
    
    AVAILABLE_YEARS = ["2015", "2016", "2017", "2018"]
    
    # Define a simplified version of load_mc for fallback
    def load_mc(years=None, decay_modes=None, tuple_types=None, data_dir=None, 
                tree_name="DecayTree", branches=None, cut=None, verbose=False):
        raise ImportError("load_mc function is not available")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate MC statistics for B meson decays")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing the processed ROOT files")
    return parser.parse_args()

def collect_statistics(data_dir: str, years: List[str], decay_mode: str, verbose: bool = True) -> Dict:
    """
    Load MC data and collect statistics for each year and tuple type.
    Since year is not stored as a branch, we process each year's file separately.
    
    Args:
        data_dir: Directory containing the processed ROOT files
        years: List of years to process
        decay_mode: Decay mode to analyze
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with structure: {tuple_type: {year: count}}
    """
    results = {}
    
    # Initialize the results structure
    for tuple_type in TUPLE_TYPES:
        if tuple_type.startswith("KSKmKpPip") or tuple_type.startswith("KSKpKpPim"):
            results[tuple_type] = {}
            for year in years:
                results[tuple_type][year] = 0
    
    if verbose:
        print(f"Loading MC data from: {data_dir}")
    
    # Process each year individually
    for year in years:
        if verbose:
            print(f"\nProcessing year: {year}")
        
        # Try loading with load_mc first for this specific year
        try:
            data = load_mc(
                years=year,
                decay_modes=decay_mode,
                tuple_types="all",
                data_dir=data_dir,
                verbose=verbose
            )
            
            # Extract counts from the loaded data
            if decay_mode in data:
                for tuple_type in results.keys():
                    if tuple_type in data[decay_mode]:
                        # Store event count for this year and tuple type
                        results[tuple_type][year] = len(data[decay_mode][tuple_type])
                        if verbose:
                            print(f"  {year} - {tuple_type}: {results[tuple_type][year]} events")
            else:
                if verbose:
                    print(f"  No data found for decay mode: {decay_mode} in year {year}")
                use_fallback = True
                fallback_count(data_dir, year, decay_mode, results, verbose)
            
        except Exception as e:
            if verbose:
                print(f"Error using load_mc for year {year}: {str(e)}")
                print(f"Falling back to direct file reading for year {year}...")
            
            fallback_count(data_dir, year, decay_mode, results, verbose)
    
    return results

def fallback_count(data_dir: str, year: str, decay_mode: str, results: Dict, verbose: bool = True):
    """
    Fallback method to count entries directly from ROOT files when load_mc fails.
    
    Args:
        data_dir: Directory containing the processed ROOT files
        year: Year to process
        decay_mode: Decay mode to analyze
        results: Results dictionary to update
        verbose: Whether to print detailed information
    """
    try:
        # Find the file for this year and decay mode
        file_pattern = os.path.join(data_dir, f"{year}_{decay_mode}.root")
        matching_files = glob.glob(file_pattern)
        
        if not matching_files:
            if verbose:
                print(f"  Warning: No files found for year {year}")
            return
        
        file_path = matching_files[0]
        if verbose:
            print(f"  Processing file: {os.path.basename(file_path)}")
        
        # Open the file and count entries in each tuple type
        with uproot.open(file_path) as file:
            for tuple_type in results.keys():
                # Try different tree paths to handle any inconsistencies
                tree_paths = [
                    f"{tuple_type}/DecayTree",
                    f"{tuple_type}/DecayTree;1",
                    f"{tuple_type}_Tuple/DecayTree",
                    f"{tuple_type}_Tuple/DecayTree;1"
                ]
                
                for tree_path in tree_paths:
                    try:
                        if tree_path in file:
                            num_entries = file[tree_path].num_entries
                            results[tuple_type][year] = num_entries
                            if verbose:
                                print(f"    {tuple_type}: {num_entries} entries")
                            break
                    except Exception:
                        continue
    
    except Exception as e:
        if verbose:
            print(f"Error in fallback counting for year {year}: {str(e)}")

def generate_latex_table(statistics: Dict, years: List[str]) -> str:
    """
    Generate a LaTeX table from the collected statistics.
    
    Args:
        statistics: Dictionary with structure {tuple_type: {year: count}}
        years: List of years to include in the table
        
    Returns:
        LaTeX table as a string
    """
    # Calculate totals for each tuple type
    for tuple_type in statistics:
        statistics[tuple_type]['Total'] = sum(statistics[tuple_type][year] for year in years)
    
    # Group by LL and DD categories
    final_stats = {
        "KSKmKpPip_LL": statistics.get("KSKmKpPip_LL", {year: 0 for year in years + ['Total']}),
        "KSKmKpPip_DD": statistics.get("KSKmKpPip_DD", {year: 0 for year in years + ['Total']}),
        "KSKpKpPim_LL": statistics.get("KSKpKpPim_LL", {year: 0 for year in years + ['Total']}),
        "KSKpKpPim_DD": statistics.get("KSKpKpPim_DD", {year: 0 for year in years + ['Total']})
    }
    
    # Build the LaTeX table
    years_str = " & ".join(years + ["Total"])
    
    latex_table = f"""
\\begin{{table}}
\\centering
\\caption{{Reconstructed MC statistics for the different final states.}}
\\begin{{tabular}}{{l{'c' * (len(years) + 1)}}}
\\hline
Final state & {years_str} \\\\
\\hline
"""
    
    # Map tuple types to more formal physics notation
    type_to_latex = {
        "KSKmKpPip_LL": "B^+ \\to K^0_{S(LL)}K^+K^-\\pi^+",
        "KSKmKpPip_DD": "B^+ \\to K^0_{S(DD)}K^+K^-\\pi^+",
        "KSKpKpPim_LL": "B^+ \\to K^0_{S(LL)}K^+K^+\\pi^-",
        "KSKpKpPim_DD": "B^+ \\to K^0_{S(DD)}K^+K^+\\pi^-"
    }
    
    # Add rows for each final state
    for tuple_type, label in type_to_latex.items():
        if tuple_type in final_stats:
            counts = [final_stats[tuple_type].get(year, 0) for year in years + ['Total']]
            counts_str = " & ".join([f"{count}" for count in counts])
            latex_table += f"{label} & {counts_str} \\\\\n"
    
    latex_table += "\\hline\n\\end{tabular}\n\\end{table}"
    
    return latex_table

def main():
    """Main function to generate the MC statistics table"""
    args = parse_arguments()
    
    # Set default data directory if not provided
    data_dir = args.data_dir or "/share/lazy/Mohamed/bu2kskpik/MC/processed"
    
    # Default years and decay mode
    years = AVAILABLE_YEARS
    decay_mode = "B2K0s2PipPimKmPipKp"
    
    print("MC Statistics Generator for B Meson Decays")
    print("------------------------------------------")
    print(f"Data directory: {data_dir}")
    print(f"Years: {', '.join(years)}")
    print(f"Decay mode: {decay_mode} ({DECAY_MODES.get(decay_mode, 'Unknown')})")
    print("------------------------------------------")
    
    # Collect the statistics
    print("\nCollecting MC statistics...")
    statistics = collect_statistics(data_dir, years, decay_mode)
    
    # Generate the LaTeX table
    print("\nGenerating LaTeX table...")
    latex_table = generate_latex_table(statistics, years)
    
    # Print the table
    print("\nLatex Table:")
    print(latex_table)
    
    # Save the table to a file
    output_file = "mc_statistics_table.tex"
    with open(output_file, "w") as f:
        f.write(latex_table)
    print(f"\nTable saved to {output_file}")

if __name__ == "__main__":
    main()