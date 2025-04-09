#!/usr/bin/env python3
"""
Count the number of events/candidates in the real data for different decay modes and tuple types.
This script generates a LaTeX table summarizing the statistics.

Usage:
  python count_data_candidates.py [--data-dir DATA_DIR]

Options:
  --data-dir DATA_DIR  Directory containing the processed ROOT files
"""

import uproot
from pathlib import Path
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import logging
import re
import argparse
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate data statistics for B meson decays")
    parser.add_argument("--data-dir", type=str, default="/share/lazy/Mohamed/bu2kskpik/RD/merged",
                        help="Directory containing the processed ROOT files")
    return parser.parse_args()

def collect_statistics(data_dir):
    """
    Collect statistics for each year and tuple type from real data.
    
    Args:
        data_dir: Directory containing the processed ROOT files
        
    Returns:
        Dictionary with structure: {tuple_type: {year: count}}
    """
    # Convert to Path object if not already
    data_dir = Path(data_dir)
    
    # Mapping to final state labels (LaTeX-style) - matches MC formatting
    final_state_latex = {
        "KSKmKpPip_LL": r"B^+ \to K^0_{S(LL)}K^+K^-\pi^+",
        "KSKpKpPim_LL": r"B^+ \to K^0_{S(LL)}K^+K^+\pi^-",
        "KSKmKpPip_DD": r"B^+ \to K^0_{S(DD)}K^+K^-\pi^+",
        "KSKpKpPim_DD": r"B^+ \to K^0_{S(DD)}K^+K^+\pi^-",
    }
    
    # Count container: {tuple_type: {year: count}}
    counts = defaultdict(lambda: defaultdict(int))
    
    print(f"\nCollecting statistics from: {data_dir}")
    print("------------------------------------------")
    
    # Iterate all files
    for file_path in tqdm(sorted(data_dir.rglob("*.root"))):
        match = re.match(r"(\d{4})_mag\w+", file_path.parent.name)
        if not match:
            logging.warning(f"Cannot extract year from {file_path}")
            continue
            
        year = match.group(1)
        mode = file_path.stem.replace("_reduced", "")
        
        try:
            file = uproot.open(file_path)
            for track in ["LL", "DD"]:
                tree_name = f"{mode}_{track}"
                if tree_name in file:
                    n = file[tree_name].num_entries
                    tuple_type = f"{mode}_{track}"
                    counts[tuple_type][year] += n
                else:
                    logging.warning(f"{tree_name} missing in {file_path.name}")
        except Exception as e:
            logging.error(f"Failed to open {file_path}: {e}")
    
    # Calculate totals for each tuple type
    for tuple_type in counts:
        counts[tuple_type]['Total'] = sum(counts[tuple_type].values())
    
    return counts, final_state_latex

def generate_latex_table(statistics, final_state_latex, years):
    """
    Generate a LaTeX table from the collected statistics.
    
    Args:
        statistics: Dictionary with structure {tuple_type: {year: count}}
        final_state_latex: Mapping from tuple_type to LaTeX representation
        years: List of years to include in the table
        
    Returns:
        LaTeX table as a string
    """
    # Build the LaTeX table
    years_str = " & ".join(years + ["Total"])
    
    latex_table = f"""\\begin{{table}}
\\centering
\\caption{{Reconstructed data statistics for the different final states.}}
\\begin{{tabular}}{{l{'c' * (len(years) + 1)}}}
\\hline
Final state & {years_str} \\\\
\\hline"""
    
    # Order matches the MC script exactly: LL first, then DD
    tuple_order = [
        "KSKmKpPip_LL",
        "KSKmKpPip_DD",
        "KSKpKpPim_LL",
        "KSKpKpPim_DD"
    ]
    
    # Add rows in the correct order
    for tuple_type in tuple_order:
        if tuple_type in final_state_latex and tuple_type in statistics:
            label = final_state_latex[tuple_type]
            counts = [statistics[tuple_type].get(year, 0) for year in years + ['Total']]
            counts_str = " & ".join([f"{count}" for count in counts])
            latex_table += f"\n{label} & {counts_str} \\\\"
    
    latex_table += "\n\\hline\n\\end{tabular}\n\\end{table}"
    
    return latex_table

def main():
    """Main function to generate the data statistics table"""
    args = parse_arguments()
    
    print("Data Statistics Generator for B Meson Decays")
    print("------------------------------------------")
    print(f"Data directory: {args.data_dir}")
    
    # Collect the statistics
    print("\nCollecting data statistics...")
    statistics, final_state_latex = collect_statistics(args.data_dir)
    
    # Get full set of years
    years = sorted({year for tuple_stats in statistics.values() for year in tuple_stats if year != 'Total'})
    print(f"Years detected: {', '.join(years)}")
    
    # Generate the LaTeX table
    print("\nGenerating LaTeX table...")
    latex_table = generate_latex_table(statistics, final_state_latex, years)
    
    # Print the table to console
    print("\nLatex Table:")
    print(latex_table)
    
    # Generate DataFrame for display
    table = {}
    for tuple_type in statistics:
        label = final_state_latex.get(tuple_type, tuple_type)
        row = {year: statistics[tuple_type].get(year, 0) for year in years}
        row["Total"] = statistics[tuple_type].get('Total', 0)
        table[label] = row
    
    df = pd.DataFrame.from_dict(table, orient="index").fillna(0).astype(int)
    df = df[years + ["Total"]]  # Reorder columns
    
    # Display summary
    print("\n📊 Final Candidate Count Table:")
    print(df.to_string())
    
    # Save the LaTeX table to a file
    output_file = "data_statistics_table.tex"
    with open(output_file, "w") as f:
        f.write(latex_table)
    print(f"\n✅ LaTeX table written to: {output_file}")

if __name__ == "__main__":
    main()