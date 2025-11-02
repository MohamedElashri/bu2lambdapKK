#!/usr/bin/env python3
"""
2D Grid Optimization Module for Selection Study

Performs 2D grid search optimization where we scan combinations of cut values
for multiple variables and calculate S/√B figure of merit for each combination.

Generates tables with:
- Variables as columns
- Cut combinations as rows  
- S/√B as the last column

Tables generated for both MC and Data.

Author: Mohamed Elashri
Date: October 29, 2025
"""

import logging
import numpy as np
import pandas as pd
import awkward as ak
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import product


class GridOptimizer:
    """
    Perform 2D grid search optimization to maximize S/√B
    """
    def __init__(self, config: dict, output_dir: Path, logger=None):
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # Get optimization config
        self.opt_config = config.get('optimization', {})
    
    def perform_2d_grid_search(self, signal_data: ak.Array, background_data: ak.Array,
                               variables: List[Tuple[str, dict]], 
                               dataset_name: str = "MC") -> pd.DataFrame:
        """
        Perform 2D grid search over all variable combinations
        
        Creates a table where:
        - Each row represents a unique combination of cuts
        - Each column represents a variable (with its cut value)
        - Last column is S/√B figure of merit
        
        Parameters:
        - signal_data: Signal events (J/ψ window for MC, or signal region for data)
        - background_data: Background events (sidebands)
        - variables: List of (var_name, var_config) tuples to scan
        - dataset_name: Name of dataset ("MC" or "Data")
        
        Returns:
        - DataFrame with grid search results
        """
        self.logger.info(f"\n=== Performing 2D Grid Search Optimization ({dataset_name}) ===")
        self.logger.info(f"Signal events: {len(signal_data):,}")
        self.logger.info(f"Background events: {len(background_data):,}")
        
        # Prepare scan ranges for each variable
        scan_grids = {}
        for var_name, var_config in variables:
            if 'scan_range' not in var_config:
                self.logger.warning(f"Variable {var_name} has no scan_range, skipping")
                continue
            
            scan_range = var_config['scan_range']
            n_steps = var_config.get('scan_steps', 10)  # Use fewer steps for 2D grid
            scan_values = np.linspace(scan_range[0], scan_range[1], n_steps)
            scan_grids[var_name] = {
                'values': scan_values,
                'branch': var_config['branch'],
                'operator': var_config['operator'],
                'config': var_config
            }
        
        if len(scan_grids) == 0:
            self.logger.error("No variables to scan!")
            return pd.DataFrame()
        
        self.logger.info(f"Scanning {len(scan_grids)} variables:")
        for var_name, grid_info in scan_grids.items():
            self.logger.info(f"  {var_name}: {len(grid_info['values'])} points")
        
        # Generate all combinations of cut values
        var_names = list(scan_grids.keys())
        cut_combinations = list(product(*[scan_grids[var]['values'] for var in var_names]))
        
        total_combinations = len(cut_combinations)
        self.logger.info(f"Total combinations to evaluate: {total_combinations:,}")
        
        # Evaluate each combination
        results = []
        for idx, cut_combo in enumerate(cut_combinations):
            if idx % max(1, total_combinations // 20) == 0:
                self.logger.info(f"  Progress: {idx:,}/{total_combinations:,} ({100*idx/total_combinations:.1f}%)")
            
            # Build cut dictionary for this combination
            cuts_dict = {var_names[i]: cut_combo[i] for i in range(len(var_names))}
            
            # Apply cuts to signal
            S = self._apply_cuts(signal_data, cuts_dict, scan_grids)
            
            # Apply cuts to background
            B = self._apply_cuts(background_data, cuts_dict, scan_grids)
            
            # Calculate figure of merit
            s_over_sqrt_b = S / np.sqrt(B) if B > 0 else 0.0
            
            # Calculate efficiencies
            signal_eff = S / len(signal_data) if len(signal_data) > 0 else 0.0
            bkg_rej = 1.0 - (B / len(background_data)) if len(background_data) > 0 else 0.0
            
            # Store result
            result_row = {**cuts_dict}  # Start with cut values
            result_row.update({
                'Signal': S,
                'Background': B,
                'S_over_sqrtB': s_over_sqrt_b,
                'Signal_Efficiency': signal_eff,
                'Background_Rejection': bkg_rej
            })
            results.append(result_row)
        
        self.logger.info(f"  Completed evaluation of {total_combinations:,} combinations")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Sort by S/√B (descending)
        df = df.sort_values('S_over_sqrtB', ascending=False).reset_index(drop=True)
        
        # Save to CSV
        output_file = self.output_dir / f"grid_optimization_{dataset_name.lower()}.csv"
        df.to_csv(output_file, index=False, float_format='%.6f')
        self.logger.info(f"Saved grid optimization results: {output_file}")
        
        # Create summary of top results
        self._create_top_cuts_summary(df, dataset_name, var_names)
        
        # Create markdown table
        self._create_markdown_table(df, dataset_name, var_names)
        
        return df
    
    def _apply_cuts(self, data: ak.Array, cuts_dict: Dict[str, float], 
                    scan_grids: Dict) -> int:
        """
        Apply all cuts in cuts_dict to data and return number of passing events
        
        Parameters:
        - data: Data to apply cuts to
        - cuts_dict: Dictionary mapping var_name -> cut_value
        - scan_grids: Dictionary with variable metadata
        
        Returns:
        - Number of events passing all cuts
        """
        mask = np.ones(len(data), dtype=bool)
        
        for var_name, cut_value in cuts_dict.items():
            grid_info = scan_grids[var_name]
            branch = grid_info['branch']
            operator = grid_info['operator']
            
            if branch not in data.fields:
                continue
            
            # Extract branch data
            branch_data = data[branch]
            
            # Handle jagged arrays
            try:
                if len(ak.flatten(branch_data)) != len(branch_data):
                    branch_data = branch_data[:, 0]
            except:
                pass
            
            branch_vals = ak.to_numpy(branch_data)
            
            # Apply operator
            if operator == '>':
                cut_mask = branch_vals > cut_value
            elif operator == '<':
                cut_mask = branch_vals < cut_value
            elif operator == '>=':
                cut_mask = branch_vals >= cut_value
            elif operator == '<=':
                cut_mask = branch_vals <= cut_value
            else:
                continue
            
            # Combine with existing mask
            mask = mask & cut_mask
        
        return int(np.sum(mask))
    
    def _create_top_cuts_summary(self, df: pd.DataFrame, dataset_name: str, 
                                  var_names: List[str], n_top: int = 20):
        """
        Create text summary of top N cut combinations
        
        Parameters:
        - df: DataFrame with optimization results (already sorted by S/√B)
        - dataset_name: Name of dataset
        - var_names: List of variable names
        - n_top: Number of top combinations to report
        """
        output_file = self.output_dir / f"top_{n_top}_cuts_{dataset_name.lower()}.txt"
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"TOP {n_top} CUT COMBINATIONS ({dataset_name})\n")
            f.write("Ranked by S/√B Figure of Merit\n")
            f.write("=" * 80 + "\n\n")
            
            # Get top N rows
            top_df = df.head(n_top)
            
            for rank, (idx, row) in enumerate(top_df.iterrows(), 1):
                f.write(f"Rank {rank}:\n")
                f.write(f"  S/√B = {row['S_over_sqrtB']:.4f}\n")
                f.write(f"  Signal: {row['Signal']:.1f} (eff: {row['Signal_Efficiency']:.2%})\n")
                f.write(f"  Background: {row['Background']:.1f} (rej: {row['Background_Rejection']:.2%})\n")
                f.write(f"  Cut values:\n")
                for var_name in var_names:
                    if var_name in row:
                        f.write(f"    {var_name}: {row[var_name]:.4f}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("NOTE: These are optimal combinations considering all variables together.\n")
            f.write("Individual variable optima may differ from single-variable scans.\n")
            f.write("=" * 80 + "\n")
        
        self.logger.info(f"Saved top cuts summary: {output_file}")
    
    def _create_markdown_table(self, df: pd.DataFrame, dataset_name: str,
                               var_names: List[str], n_rows: int = 50):
        """
        Create markdown-formatted table of optimization results
        
        Parameters:
        - df: DataFrame with results
        - dataset_name: Name of dataset
        - var_names: List of variable names
        - n_rows: Number of rows to include in table
        """
        output_file = self.output_dir / f"grid_optimization_{dataset_name.lower()}.md"
        
        with open(output_file, 'w') as f:
            f.write(f"# 2D Grid Search Optimization Results ({dataset_name})\n\n")
            f.write("Maximizing S/√B figure of merit across all variable combinations.\n\n")
            
            f.write("## Summary Statistics\n\n")
            f.write(f"- Total combinations evaluated: {len(df):,}\n")
            f.write(f"- Best S/√B achieved: {df['S_over_sqrtB'].max():.4f}\n")
            f.write(f"- Variables scanned: {', '.join(var_names)}\n")
            f.write("\n")
            
            f.write(f"## Top {n_rows} Cut Combinations\n\n")
            f.write("Sorted by S/√B (highest first).\n\n")
            
            # Create header
            header_parts = ["Rank"] + var_names + ["Signal", "Background", "S/√B", "Sig Eff", "Bkg Rej"]
            f.write("| " + " | ".join(header_parts) + " |\n")
            f.write("|" + "|".join(["---"] * len(header_parts)) + "|\n")
            
            # Write rows
            top_df = df.head(n_rows)
            for rank, (idx, row) in enumerate(top_df.iterrows(), 1):
                row_parts = [str(rank)]
                
                # Add cut values
                for var_name in var_names:
                    row_parts.append(f"{row[var_name]:.3f}")
                
                # Add metrics
                row_parts.append(f"{row['Signal']:.1f}")
                row_parts.append(f"{row['Background']:.1f}")
                row_parts.append(f"{row['S_over_sqrtB']:.4f}")
                row_parts.append(f"{row['Signal_Efficiency']:.2%}")
                row_parts.append(f"{row['Background_Rejection']:.2%}")
                
                f.write("| " + " | ".join(row_parts) + " |\n")
            
            f.write("\n")
            f.write("## Interpretation\n\n")
            f.write("- **Rank**: Position when sorted by S/√B (1 = best)\n")
            f.write("- **Variables**: Cut values for each variable\n")
            f.write("- **Signal**: Number of signal events passing all cuts\n")
            f.write("- **Background**: Number of background events passing all cuts\n")
            f.write("- **S/√B**: Signal over square root of background (figure of merit)\n")
            f.write("- **Sig Eff**: Signal efficiency (fraction of signal retained)\n")
            f.write("- **Bkg Rej**: Background rejection (fraction of background removed)\n")
            f.write("\n")
            
            f.write("## Usage\n\n")
            f.write("To use these results:\n")
            f.write("1. Choose a rank based on your optimization priorities\n")
            f.write("2. Apply the corresponding cut values to your analysis\n")
            f.write("3. Consider trade-offs between signal efficiency and background rejection\n")
            f.write("4. Validate on independent data samples\n")
            f.write("\n")
        
        self.logger.info(f"Saved markdown table: {output_file}")
    
    def create_comparison_plots(self, mc_df: pd.DataFrame, data_df: pd.DataFrame,
                                var_names: List[str]):
        """
        Create comparison plots showing MC vs Data optimization results
        
        Parameters:
        - mc_df: DataFrame with MC optimization results
        - data_df: DataFrame with Data optimization results
        - var_names: List of variable names
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        self.logger.info("\n=== Creating MC vs Data Comparison Plots ===")
        
        # Create output directory
        comparison_dir = self.output_dir / "grid_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. S/√B distribution comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(mc_df['S_over_sqrtB'], bins=50, alpha=0.5, label='MC', color='red')
        ax.hist(data_df['S_over_sqrtB'], bins=50, alpha=0.5, label='Data', color='blue')
        
        ax.set_xlabel('S/√B')
        ax.set_ylabel('Number of Combinations')
        ax.set_title('S/√B Distribution: MC vs Data')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_file = comparison_dir / "s_over_sqrtb_comparison.pdf"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        self.logger.info(f"Saved S/√B comparison: {output_file}")
        
        # 2. Best cut values comparison
        mc_best = mc_df.iloc[0]
        data_best = data_df.iloc[0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(var_names))
        width = 0.35
        
        mc_values = [mc_best[var] for var in var_names]
        data_values = [data_best[var] for var in var_names]
        
        ax.bar(x_pos - width/2, mc_values, width, label='MC Best', color='red', alpha=0.7)
        ax.bar(x_pos + width/2, data_values, width, label='Data Best', color='blue', alpha=0.7)
        
        ax.set_xlabel('Variable')
        ax.set_ylabel('Optimal Cut Value')
        ax.set_title('Best Cut Values: MC vs Data')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(var_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        output_file = comparison_dir / "best_cuts_comparison.pdf"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        self.logger.info(f"Saved best cuts comparison: {output_file}")
        
        # 3. Efficiency vs Rejection scatter plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # MC
        sc1 = ax1.scatter(mc_df['Signal_Efficiency'], mc_df['Background_Rejection'],
                         c=mc_df['S_over_sqrtB'], cmap='viridis', alpha=0.6, s=20)
        ax1.set_xlabel('Signal Efficiency')
        ax1.set_ylabel('Background Rejection')
        ax1.set_title('MC: Signal Eff vs Bkg Rejection')
        plt.colorbar(sc1, ax=ax1, label='S/√B')
        ax1.grid(True, alpha=0.3)
        
        # Data
        sc2 = ax2.scatter(data_df['Signal_Efficiency'], data_df['Background_Rejection'],
                         c=data_df['S_over_sqrtB'], cmap='viridis', alpha=0.6, s=20)
        ax2.set_xlabel('Signal Efficiency')
        ax2.set_ylabel('Background Rejection')
        ax2.set_title('Data: Signal Eff vs Bkg Rejection')
        plt.colorbar(sc2, ax=ax2, label='S/√B')
        ax2.grid(True, alpha=0.3)
        
        output_file = comparison_dir / "efficiency_rejection_scatter.pdf"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        self.logger.info(f"Saved efficiency-rejection scatter: {output_file}")
        
        self.logger.info(f"Saved comparison plots to {comparison_dir}")
