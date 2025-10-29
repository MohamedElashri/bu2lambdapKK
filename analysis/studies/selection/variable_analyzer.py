#!/usr/bin/env python3
"""
Variable Analyzer Module for Selection Study

Analyze individual selection variables with distributions and efficiency scans.

Author: Mohamed Elashri
Date: October 28, 2025
"""

import logging
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
from pathlib import Path
from typing import Dict, List, Tuple

from selection_efficiency import EfficiencyCalculator


class VariableAnalyzer:
    """
    Analyze individual selection variables
    """
    def __init__(self, config: dict, output_dir: Path, logger=None):
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.eff_calc = EfficiencyCalculator(logger)
        
        # Get plot settings
        self.plot_config = config.get('plot_settings', {})
        self.colors = self.plot_config.get('colors', {})
        self.bins_config = self.plot_config.get('bins', {})
        self.labels_config = self.plot_config.get('labels', {})
    
    def plot_distribution(self, jpsi_data: ak.Array, sideband_data: ak.Array, 
                         real_data: ak.Array, var_config: dict,
                         output_name: str):
        """
        Plot variable distribution with cut lines
        
        Parameters:
        - jpsi_data: J/ψ MC signal data
        - sideband_data: Real data sidebands (background)
        - real_data: Full real data
        - var_config: Variable configuration dictionary
        - output_name: Output filename (without extension)
        """
        branch = var_config['branch']
        
        # Extract data (flatten jagged arrays if needed)
        def safe_extract(data, branch):
            if branch not in data.fields:
                return np.array([])
            vals = data[branch]
            try:
                if len(ak.flatten(vals)) != len(vals):
                    vals = vals[:, 0]
            except:
                pass
            return ak.to_numpy(vals)
        
        jpsi_vals = safe_extract(jpsi_data, branch)
        sideband_vals = safe_extract(sideband_data, branch)
        data_vals = safe_extract(real_data, branch)
        
        if len(jpsi_vals) == 0 and len(sideband_vals) == 0 and len(data_vals) == 0:
            self.logger.warning(f"No data for variable {branch}")
            return
        
        # Get plot settings
        plot_range = var_config.get('plot_range', [np.min(data_vals), np.max(data_vals)])
        n_bins = self.bins_config.get(var_config.get('name', branch), 50)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.plot_config.get('figsize', [10, 7]))
        
        # Plot histograms
        if len(jpsi_vals) > 0:
            ax.hist(jpsi_vals, bins=n_bins, range=plot_range, 
                   histtype='step', linewidth=2, label='J/ψ MC Signal',
                   color=self.colors.get('jpsi_signal', '#E41A1C'))
        
        if len(sideband_vals) > 0:
            ax.hist(sideband_vals, bins=n_bins, range=plot_range,
                   histtype='step', linewidth=2, label='Data Sidebands (Background)',
                   color=self.colors.get('data_sideband', '#377EB8'))
        
        if len(data_vals) > 0:
            ax.hist(data_vals, bins=n_bins, range=plot_range,
                   histtype='step', linewidth=2, label='Full Data', linestyle='--',
                   color=self.colors.get('data', '#000000'))
        
        # Add cut lines
        operator = var_config['operator']
        tight_cut = var_config.get('current_tight')
        loose_cut = var_config.get('current_loose')
        
        if tight_cut is not None:
            ax.axvline(tight_cut, color='red', linestyle='--', linewidth=2, 
                      label=f'Tight cut ({operator} {tight_cut})')
        
        if loose_cut is not None and loose_cut != tight_cut:
            ax.axvline(loose_cut, color='orange', linestyle='--', linewidth=2,
                      label=f'Loose cut ({operator} {loose_cut})')
        
        # Labels
        xlabel = self.labels_config.get(var_config.get('name', branch), branch)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Events')
        ax.set_title(var_config.get('description', branch))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save
        output_path = self.output_dir / f"{output_name}.{self.plot_config.get('format', 'pdf')}"
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.plot_config.get('dpi', 300))
        plt.close()
        
        self.logger.info(f"Saved distribution plot: {output_path}")
    
    def plot_efficiency_scan(self, scan_results: List[Tuple],
                            var_config: dict, output_name: str,
                            label: str = ""):
        """
        Plot efficiency vs cut value
        
        Parameters:
        - scan_results: List of (cut_value, efficiency, n_passed, n_total)
        - var_config: Variable configuration
        - output_name: Output filename
        - label: Dataset label
        """
        cut_values = [r[0] for r in scan_results]
        efficiencies = [r[1] * 100 for r in scan_results]  # Convert to percentage
        
        fig, ax = plt.subplots(figsize=self.plot_config.get('figsize', [10, 7]))
        
        ax.plot(cut_values, efficiencies, 'b-', linewidth=2, label=label)
        
        # Mark current cuts
        tight_cut = var_config.get('current_tight')
        loose_cut = var_config.get('current_loose')
        
        if tight_cut is not None:
            tight_idx = np.argmin(np.abs(np.array(cut_values) - tight_cut))
            ax.plot(tight_cut, efficiencies[tight_idx], 'ro', markersize=10,
                   label=f'Tight cut ({efficiencies[tight_idx]:.1f}%)')
        
        if loose_cut is not None and loose_cut != tight_cut:
            loose_idx = np.argmin(np.abs(np.array(cut_values) - loose_cut))
            ax.plot(loose_cut, efficiencies[loose_idx], 'go', markersize=10,
                   label=f'Loose cut ({efficiencies[loose_idx]:.1f}%)')
        
        # Labels
        xlabel = self.labels_config.get(var_config.get('name', ''), var_config['branch'])
        ax.set_xlabel(f"Cut value: {xlabel} {var_config['operator']} X")
        ax.set_ylabel('Selection Efficiency [%]')
        ax.set_title(f"Efficiency Scan: {var_config.get('description', '')}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        # Save
        output_path = self.output_dir / f"{output_name}.{self.plot_config.get('format', 'pdf')}"
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.plot_config.get('dpi', 300))
        plt.close()
        
        self.logger.info(f"Saved efficiency scan: {output_path}")
    
    def plot_2d_efficiency(self, data: ak.Array, var1_config: dict, 
                          var2_config: dict, output_name: str):
        """
        Create 2D efficiency map for two variables
        
        Parameters:
        - data: Dataset to analyze
        - var1_config: Configuration for first variable (x-axis)
        - var2_config: Configuration for second variable (y-axis)
        - output_name: Output filename
        """
        branch1 = var1_config['branch']
        branch2 = var2_config['branch']
        
        # Extract data
        vals1 = ak.to_numpy(data[branch1])
        vals2 = ak.to_numpy(data[branch2])
        
        # Get scan ranges
        scan1 = np.linspace(*var1_config['scan_range'], var1_config['scan_points'])
        scan2 = np.linspace(*var2_config['scan_range'], var2_config['scan_points'])
        
        # Build efficiency grid
        efficiency_grid = np.zeros((len(scan2), len(scan1)))
        
        for i, cut2 in enumerate(scan2):
            for j, cut1 in enumerate(scan1):
                mask1 = self._apply_cut(vals1, cut1, var1_config['operator'])
                mask2 = self._apply_cut(vals2, cut2, var2_config['operator'])
                combined_mask = mask1 & mask2
                efficiency_grid[i, j] = 100 * np.sum(combined_mask) / len(vals1)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 9))
        
        im = ax.imshow(efficiency_grid, extent=[scan1[0], scan1[-1], scan2[0], scan2[-1]],
                      origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=100)
        
        # Contour lines
        contour = ax.contour(scan1, scan2, efficiency_grid, 
                           levels=[50, 60, 70, 80, 90],
                           colors='white', linewidths=1.5, alpha=0.8)
        ax.clabel(contour, inline=True, fontsize=10, fmt='%d%%')
        
        # Mark current cuts
        if var1_config.get('current_tight') and var2_config.get('current_tight'):
            ax.plot(var1_config['current_tight'], var2_config['current_tight'],
                   'r*', markersize=15, label='Current tight cuts')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Efficiency [%]')
        
        # Labels
        xlabel = self.labels_config.get(var1_config.get('name', ''), branch1)
        ylabel = self.labels_config.get(var2_config.get('name', ''), branch2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"2D Efficiency Map: {var1_config.get('name', '')} vs {var2_config.get('name', '')}")
        ax.legend()
        
        # Save
        output_path = self.output_dir / f"{output_name}.{self.plot_config.get('format', 'pdf')}"
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.plot_config.get('dpi', 300))
        plt.close()
        
        self.logger.info(f"Saved 2D efficiency map: {output_path}")
    
    def _apply_cut(self, values: np.ndarray, cut_value: float, operator: str) -> np.ndarray:
        """Apply cut with specified operator"""
        if operator == '>':
            return values > cut_value
        elif operator == '<':
            return values < cut_value
        elif operator == '>=':
            return values >= cut_value
        elif operator == '<=':
            return values <= cut_value
        elif operator == '==':
            return values == cut_value
        elif operator == '!=':
            return values != cut_value
        else:
            raise ValueError(f"Unknown operator: {operator}")
