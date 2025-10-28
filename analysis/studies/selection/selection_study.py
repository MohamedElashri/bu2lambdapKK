#!/usr/bin/env python3
"""
Selection Study for B+ → pK⁻Λ̄ K+ Analysis
Focus: J/ψ Signal vs Background Discrimination

This module implements a comprehensive selection optimization study
with hierarchical approach: good Λ → good PID → good B+ → clean J/ψ

Author: Mohamed Elashri
Date: October 28, 2025
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import awkward as ak
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
import tomli
from typing import Dict, List, Tuple, Optional
import warnings
from collections import OrderedDict

# Suppress font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Add analysis directory to path for imports
# From selection/ go up to studies/, then up to analysis/
analysis_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(analysis_dir))

from data_loader import DataLoader
from mc_loader import MCLoader
from selection import SelectionProcessor
from mass_calculator import MassCalculator
from branch_config import BranchConfig

# Set LHCb style
plt.style.use(hep.style.LHCb2)

# Configure matplotlib for better fonts
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'text.usetex': False,
    'mathtext.fontset': 'dejavusans',
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})


class EfficiencyCalculator:
    """
    Calculate selection efficiencies for single and combined cuts
    """
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_single_cut(self, data: ak.Array, branch: str, 
                            cut_value: float, operator: str) -> Tuple[float, int, int]:
        """
        Calculate efficiency for a single cut
        
        Parameters:
        - data: awkward array with event data
        - branch: branch name to cut on
        - cut_value: cut threshold value
        - operator: comparison operator ('>', '<', '>=', '<=', '==', '!=')
        
        Returns:
        - efficiency, n_passed, n_total
        """
        if branch not in data.fields:
            self.logger.error(f"Branch {branch} not found in data")
            return 0.0, 0, len(data)
        
                # Get branch data (flatten if jagged)
        if branch not in data.fields:
            return np.zeros(len(data), dtype=bool), 0.0, 0
            
        branch_data = data[branch]
        # Check if jagged by comparing flattened length to original
        try:
            flat = ak.flatten(branch_data)
            if len(flat) != len(branch_data):
                # It's jagged, use flattened version
                branch_data = flat
        except:
            pass  # Not flattenable or already flat
        
        branch_data = ak.to_numpy(branch_data)
        n_total = len(branch_data)
        
        # Apply operator
        if operator == '>':
            mask = branch_data > cut_value
        elif operator == '<':
            mask = branch_data < cut_value
        elif operator == '>=':
            mask = branch_data >= cut_value
        elif operator == '<=':
            mask = branch_data <= cut_value
        elif operator == '==':
            mask = branch_data == cut_value
        elif operator == '!=':
            mask = branch_data != cut_value
        else:
            raise ValueError(f"Unknown operator: {operator}")
        
        n_passed = np.sum(mask)
        efficiency = n_passed / n_total if n_total > 0 else 0.0
        
        return efficiency, n_passed, n_total
    
    def scan_efficiency(self, data: ak.Array, branch: str, 
                       cut_values: np.ndarray, operator: str) -> List[Tuple]:
        """
        Scan efficiency over range of cut values
        
        Returns:
        - List of (cut_value, efficiency, n_passed, n_total) tuples
        """
        results = []
        for cut_val in cut_values:
            eff, n_pass, n_tot = self.calculate_single_cut(data, branch, cut_val, operator)
            results.append((cut_val, eff, n_pass, n_tot))
        
        return results
    
    def find_optimal_cut(self, scan_results: List[Tuple], 
                        min_efficiency: float = 0.70) -> float:
        """
        Find optimal cut value based on criteria
        
        Parameters:
        - scan_results: List of (cut_value, efficiency, n_passed, n_total)
        - min_efficiency: Minimum acceptable efficiency
        
        Returns:
        - Optimal cut value
        """
        valid_cuts = [(cut, eff) for cut, eff, _, _ in scan_results if eff >= min_efficiency]
        
        if not valid_cuts:
            self.logger.warning(f"No cuts found with efficiency >= {min_efficiency}")
            # Return cut with highest efficiency
            return max(scan_results, key=lambda x: x[1])[0]
        
        # Return cut with highest efficiency among valid cuts
        return max(valid_cuts, key=lambda x: x[1])[0]
    
    def generate_cutflow(self, data_dict: Dict[str, ak.Array], 
                        cuts: List[Tuple[str, callable]]) -> Dict:
        """
        Generate sequential cutflow table
        
        Parameters:
        - data_dict: Dictionary of datasets {'name': data}
        - cuts: List of (cut_name, cut_function) tuples
        
        Returns:
        - Dictionary with cutflow information
        """
        cutflow = {}
        
        for dataset_name, initial_data in data_dict.items():
            cutflow[dataset_name] = OrderedDict()
            cutflow[dataset_name]['initial'] = {
                'events': len(initial_data),
                'efficiency': 100.0
            }
            
            # Apply cuts sequentially
            current_data = initial_data
            for cut_name, cut_func in cuts:
                current_data = cut_func(current_data)
                n_events = len(current_data)
                efficiency = (n_events / len(initial_data)) * 100.0
                
                cutflow[dataset_name][cut_name] = {
                    'events': n_events,
                    'efficiency': efficiency
                }
        
        return cutflow


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
    
    def plot_distribution(self, jpsi_data: ak.Array, kpkm_data: ak.Array, 
                         real_data: ak.Array, var_config: dict,
                         output_name: str):
        """
        Plot variable distribution with cut lines
        
        Parameters:
        - jpsi_data: J/ψ MC signal data
        - kpkm_data: KpKm MC background data
        - real_data: Real data
        - var_config: Variable configuration dictionary
        - output_name: Output filename (without extension)
        """
        branch = var_config['branch']
        
        # Extract data (flatten jagged arrays if needed)
        def safe_extract(data, branch):
            if branch not in data.fields:
                return np.array([])
            arr = data[branch]
            # Check if array is jagged by comparing flattened length to original length
            try:
                flat_arr = ak.flatten(arr)
                if len(flat_arr) != len(arr):
                    # It's jagged, use flattened version
                    arr = flat_arr
            except:
                pass  # Not flattenable or already flat
            return ak.to_numpy(arr)
        
        jpsi_vals = safe_extract(jpsi_data, branch)
        kpkm_vals = safe_extract(kpkm_data, branch)
        data_vals = safe_extract(real_data, branch)
        
        if len(jpsi_vals) == 0 and len(kpkm_vals) == 0 and len(data_vals) == 0:
            self.logger.warning(f"No data found for branch {branch}")
            return
        
        # Get plot settings
        plot_range = var_config.get('plot_range', [np.min(data_vals), np.max(data_vals)])
        n_bins = self.bins_config.get(var_config.get('name', branch), 50)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.plot_config.get('figsize', [10, 7]))
        
        # Plot histograms
        if len(jpsi_vals) > 0:
            ax.hist(jpsi_vals, bins=n_bins, range=plot_range, 
                   alpha=0.5, label='J/$\\psi$ Signal MC',
                   color=self.colors.get('jpsi_signal', '#E41A1C'), 
                   histtype='stepfilled')
        
        if len(kpkm_vals) > 0:
            ax.hist(kpkm_vals, bins=n_bins, range=plot_range,
                   alpha=0.5, label='KpKm Background MC',
                   color=self.colors.get('kpkm_background', '#377EB8'),
                   histtype='stepfilled')
        
        if len(data_vals) > 0:
            ax.hist(data_vals, bins=n_bins, range=plot_range,
                   histtype='step', linewidth=2, label='Data',
                   color=self.colors.get('data', '#000000'))
        
        # Add cut lines
        operator = var_config['operator']
        tight_cut = var_config.get('current_tight')
        loose_cut = var_config.get('current_loose')
        
        if tight_cut is not None:
            ax.axvline(tight_cut, color=self.colors.get('tight_cut', '#2CA02C'),
                      linestyle='--', linewidth=2, label=f'Tight cut {operator} {tight_cut}')
        
        if loose_cut is not None and loose_cut != tight_cut:
            ax.axvline(loose_cut, color=self.colors.get('loose_cut', '#FF7F00'),
                      linestyle=':', linewidth=2, label=f'Loose cut {operator} {loose_cut}')
        
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
        if self.plot_config.get('save_png', True):
            plt.savefig(output_path.with_suffix('.png'), dpi=150)
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
            # Find efficiency at tight cut
            idx = np.argmin(np.abs(np.array(cut_values) - tight_cut))
            eff_tight = efficiencies[idx]
            ax.plot(tight_cut, eff_tight, 'go', markersize=10, 
                   label=f'Tight: {tight_cut} (ε={eff_tight:.1f}%)')
        
        if loose_cut is not None and loose_cut != tight_cut:
            idx = np.argmin(np.abs(np.array(cut_values) - loose_cut))
            eff_loose = efficiencies[idx]
            ax.plot(loose_cut, eff_loose, 'ro', markersize=10,
                   label=f'Loose: {loose_cut} (ε={eff_loose:.1f}%)')
        
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
        if self.plot_config.get('save_png', True):
            plt.savefig(output_path.with_suffix('.png'), dpi=150)
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
                # Apply both cuts
                mask1 = self._apply_cut(vals1, cut1, var1_config['operator'])
                mask2 = self._apply_cut(vals2, cut2, var2_config['operator'])
                combined_mask = mask1 & mask2
                
                efficiency_grid[i, j] = np.sum(combined_mask) / len(data) * 100
        
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
                   'r*', markersize=20, label='Tight cuts')
        
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
        if self.plot_config.get('save_png', True):
            plt.savefig(output_path.with_suffix('.png'), dpi=150)
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


class StudyPlotter:
    """
    Create comparison and summary plots
    """
    def __init__(self, config: dict, output_dir: Path, logger=None):
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        self.plot_config = config.get('plot_settings', {})
        self.colors = self.plot_config.get('colors', {})
    
    def plot_cutflow(self, cutflow: Dict, output_name: str = "cutflow"):
        """
        Plot sequential cutflow comparison
        
        Parameters:
        - cutflow: Dictionary from EfficiencyCalculator.generate_cutflow()
        - output_name: Output filename
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Get cut names from first dataset
        first_dataset = list(cutflow.keys())[0]
        cut_names = list(cutflow[first_dataset].keys())
        x_pos = np.arange(len(cut_names))
        
        # Plot 1: Absolute event counts
        width = 0.8 / len(cutflow)  # Width of bars
        
        for i, (dataset_name, cuts) in enumerate(cutflow.items()):
            events = [cuts[cut]['events'] for cut in cut_names]
            offset = (i - len(cutflow)/2 + 0.5) * width
            
            color = self.colors.get(dataset_name.lower().replace(' ', '_'), f'C{i}')
            ax1.bar(x_pos + offset, events, width, label=dataset_name, color=color, alpha=0.8)
        
        ax1.set_xlabel('Cut Stage')
        ax1.set_ylabel('Events')
        ax1.set_title('Cutflow: Event Counts')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(cut_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_yscale('log')
        
        # Plot 2: Efficiencies
        for dataset_name, cuts in cutflow.items():
            efficiencies = [cuts[cut]['efficiency'] for cut in cut_names]
            color = self.colors.get(dataset_name.lower().replace(' ', '_'), None)
            ax2.plot(x_pos, efficiencies, 'o-', linewidth=2, markersize=8,
                    label=dataset_name, color=color)
        
        ax2.set_xlabel('Cut Stage')
        ax2.set_ylabel('Cumulative Efficiency [%]')
        ax2.set_title('Cutflow: Efficiencies')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(cut_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 105])
        
        # Save
        output_path = self.output_dir / f"{output_name}.{self.plot_config.get('format', 'pdf')}"
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.plot_config.get('dpi', 300))
        if self.plot_config.get('save_png', True):
            plt.savefig(output_path.with_suffix('.png'), dpi=150)
        plt.close()
        
        self.logger.info(f"Saved cutflow plot: {output_path}")
    
    def plot_signal_to_background(self, signal_counts: List[float], 
                                  background_counts: List[float],
                                  cut_labels: List[str],
                                  output_name: str = "signal_to_background"):
        """
        Plot signal/background ratio evolution
        
        Parameters:
        - signal_counts: Signal event counts at each cut stage
        - background_counts: Background event counts at each cut stage
        - cut_labels: Labels for cut stages
        - output_name: Output filename
        """
        # Calculate S/B ratio
        sb_ratios = []
        for s, b in zip(signal_counts, background_counts):
            if b > 0:
                sb_ratios.append(s / b)
            else:
                sb_ratios.append(0)
        
        # Calculate significance
        significance = []
        for s, b in zip(signal_counts, background_counts):
            if s + b > 0:
                significance.append(s / np.sqrt(s + b))
            else:
                significance.append(0)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        x_pos = np.arange(len(cut_labels))
        
        # S/B ratio
        ax1.plot(x_pos, sb_ratios, 'o-', linewidth=2, markersize=8,
                color=self.colors.get('sb_ratio', '#E41A1C'))
        ax1.set_ylabel('Signal / Background')
        ax1.set_title('Signal to Background Ratio')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(cut_labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Significance
        ax2.plot(x_pos, significance, 's-', linewidth=2, markersize=8,
                color=self.colors.get('significance', '#377EB8'))
        ax2.set_xlabel('Cut Stage')
        ax2.set_ylabel('S / √(S+B)')
        ax2.set_title('Signal Significance')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(cut_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Save
        output_path = self.output_dir / f"{output_name}.{self.plot_config.get('format', 'pdf')}"
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.plot_config.get('dpi', 300))
        if self.plot_config.get('save_png', True):
            plt.savefig(output_path.with_suffix('.png'), dpi=150)
        plt.close()
        
        self.logger.info(f"Saved S/B plot: {output_path}")
    
    def plot_combined_efficiency(self, data: ak.Array, variables_config: dict,
                                 phase_name: str, output_name: str):
        """
        Create combined efficiency plot showing all variables for a phase
        
        Parameters:
        - data: Data array (J/ψ signal MC)
        - variables_config: Dict of variable configurations
        - phase_name: Name of the phase (e.g., "Lambda", "PID", "B+")
        - output_name: Output filename
        """
        # Get enabled variables with scan ranges
        plot_vars = []
        for var_name, var_config in variables_config.items():
            if var_name in ['description', 'enabled'] or not isinstance(var_config, dict):
                continue
            if not var_config.get('enabled', True):
                continue
            if 'scan_range' not in var_config:
                continue
            plot_vars.append((var_name, var_config))
        
        if not plot_vars:
            self.logger.warning(f"No variables with scan ranges found for {phase_name}")
            return
        
        # Create subplots
        n_vars = len(plot_vars)
        n_cols = 2
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        # Calculate efficiency for each variable
        for idx, (var_name, var_config) in enumerate(plot_vars):
            ax = axes[idx]
            branch = var_config['branch']
            operator = var_config['operator']
            
            # Scan efficiency
            scan_vals = np.linspace(
                var_config['scan_range'][0],
                var_config['scan_range'][1],
                var_config.get('scan_points', 50)
            )
            
            efficiencies = []
            for cut_val in scan_vals:
                if branch in data.fields:
                    if operator == '>':
                        mask = data[branch] > cut_val
                    elif operator == '<':
                        mask = data[branch] < cut_val
                    elif operator == '>=':
                        mask = data[branch] >= cut_val
                    elif operator == '<=':
                        mask = data[branch] <= cut_val
                    else:
                        mask = data[branch] == cut_val
                    
                    n_passed = ak.sum(mask)
                    n_total = len(data)
                    eff = (n_passed / n_total * 100) if n_total > 0 else 0
                    efficiencies.append(eff)
                else:
                    efficiencies.append(0)
            
            # Plot
            ax.plot(scan_vals, efficiencies, 'b-', linewidth=2)
            ax.set_xlabel(var_config.get('name', branch))
            ax.set_ylabel('Efficiency [%]')
            ax.set_title(var_config.get('description', var_name))
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 105])
            
            # Mark current cuts if available
            if 'current_tight' in var_config:
                ax.axvline(var_config['current_tight'], color='green', 
                          linestyle='--', linewidth=2, label='Tight', alpha=0.7)
            if 'current_loose' in var_config:
                ax.axvline(var_config['current_loose'], color='orange',
                          linestyle='--', linewidth=2, label='Loose', alpha=0.7)
            if 'current_tight' in var_config or 'current_loose' in var_config:
                ax.legend(fontsize=8)
        
        # Hide unused subplots
        for idx in range(len(plot_vars), len(axes)):
            axes[idx].axis('off')
        
        # Overall title
        fig.suptitle(f'{phase_name} Selection: Combined Efficiency Scans', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Save
        output_path = self.output_dir / f"{output_name}.{self.plot_config.get('format', 'pdf')}"
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(output_path, dpi=self.plot_config.get('dpi', 300))
        if self.plot_config.get('save_png', True):
            plt.savefig(output_path.with_suffix('.png'), dpi=150)
        plt.close()
        
        self.logger.info(f"Saved combined efficiency plot: {output_path}")
    
    def plot_2d_correlation(self, data: ak.Array, var1_config: dict, var2_config: dict,
                           output_name: str, plot_type: str = "efficiency"):
        """
        Create 2D correlation/efficiency heatmap for two variables
        
        Parameters:
        - data: Data array
        - var1_config: First variable configuration (x-axis)
        - var2_config: Second variable configuration (y-axis)
        - output_name: Output filename
        - plot_type: "efficiency" or "density"
        """
        branch1 = var1_config['branch']
        branch2 = var2_config['branch']
        
        if branch1 not in data.fields or branch2 not in data.fields:
            self.logger.warning(f"Branches {branch1} or {branch2} not found in data")
            return
        
        # Extract data
        def safe_extract(arr, branch):
            if branch not in arr.fields:
                return np.array([])
            val = arr[branch]
            try:
                flat = ak.flatten(val)
                if len(flat) != len(val):
                    val = flat
            except:
                pass
            return ak.to_numpy(val)
        
        x_vals = safe_extract(data, branch1)
        y_vals = safe_extract(data, branch2)
        
        if len(x_vals) == 0 or len(y_vals) == 0:
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if plot_type == "efficiency":
            # Create 2D efficiency grid
            x_edges = np.linspace(var1_config['scan_range'][0], 
                                 var1_config['scan_range'][1], 20)
            y_edges = np.linspace(var2_config['scan_range'][0],
                                 var2_config['scan_range'][1], 20)
            
            efficiency_grid = np.zeros((len(y_edges)-1, len(x_edges)-1))
            
            for i in range(len(x_edges)-1):
                for j in range(len(y_edges)-1):
                    x_cut = x_edges[i]
                    y_cut = y_edges[j]
                    
                    # Apply cuts based on operators
                    if var1_config['operator'] == '>':
                        mask1 = data[branch1] > x_cut
                    else:
                        mask1 = data[branch1] < x_cut
                    
                    if var2_config['operator'] == '>':
                        mask2 = data[branch2] > y_cut
                    else:
                        mask2 = data[branch2] < y_cut
                    
                    combined_mask = mask1 & mask2
                    efficiency_grid[j, i] = ak.sum(combined_mask) / len(data) * 100
            
            # Plot heatmap
            im = ax.imshow(efficiency_grid, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                          origin='lower', aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Efficiency [%]', rotation=270, labelpad=20)
            
            # Add contour lines
            X, Y = np.meshgrid(x_edges[:-1], y_edges[:-1])
            contours = ax.contour(X, Y, efficiency_grid, levels=[50, 70, 80, 90],
                                 colors='black', linewidths=1, alpha=0.5)
            ax.clabel(contours, inline=True, fontsize=8)
            
        else:  # density plot
            # 2D histogram
            h, x_edges, y_edges = np.histogram2d(x_vals, y_vals, bins=50,
                                                range=[[var1_config['scan_range'][0], var1_config['scan_range'][1]],
                                                      [var2_config['scan_range'][0], var2_config['scan_range'][1]]])
            
            im = ax.imshow(h.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                          origin='lower', aspect='auto', cmap='viridis', norm=matplotlib.colors.LogNorm())
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Events', rotation=270, labelpad=20)
        
        # Labels
        ax.set_xlabel(var1_config.get('name', branch1))
        ax.set_ylabel(var2_config.get('name', branch2))
        ax.set_title(f"2D {plot_type.capitalize()}: {var1_config.get('description', branch1)} vs {var2_config.get('description', branch2)}")
        ax.grid(True, alpha=0.2, color='white', linewidth=0.5)
        
        # Save
        output_path = self.output_dir / f"{output_name}.{self.plot_config.get('format', 'pdf')}"
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.plot_config.get('dpi', 300))
        if self.plot_config.get('save_png', True):
            plt.savefig(output_path.with_suffix('.png'), dpi=150)
        plt.close()
        
        self.logger.info(f"Saved 2D correlation plot: {output_path}")


class JPsiAnalyzer:
    """
    J/ψ mass spectrum analysis and signal extraction
    """
    def __init__(self, config: dict, output_dir: Path, logger=None):
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # Get J/ψ study regions
        jpsi_config = config.get('study_regions', {})
        self.jpsi_range = jpsi_config.get('jpsi_range', [3000, 3200])
        self.jpsi_window = jpsi_config.get('jpsi_window', [3070, 3120])
        self.left_sideband = jpsi_config.get('sideband_left', [3000, 3050])
        self.right_sideband = jpsi_config.get('sideband_right', [3150, 3200])
        
        self.plot_config = config.get('plot_settings', {})
        self.colors = self.plot_config.get('colors', {})
        self.mass_calc = MassCalculator()
    
    def calculate_mass(self, data: ak.Array) -> ak.Array:
        """Calculate M(pK⁻Λ̄) invariant mass"""
        # Extract four-momenta
        p_px, p_py, p_pz, p_e = data.p_PX, data.p_PY, data.p_PZ, data.p_PE
        
        # Identify K- (negative kaon) - use h1 if negative ID, else h2
        k_minus_mask = data.h1_ID < 0
        k_px = ak.where(k_minus_mask, data.h1_PX, data.h2_PX)
        k_py = ak.where(k_minus_mask, data.h1_PY, data.h2_PY)
        k_pz = ak.where(k_minus_mask, data.h1_PZ, data.h2_PZ)
        k_e = ak.where(k_minus_mask, data.h1_PE, data.h2_PE)
        
        # Lambda four-momentum
        l_px, l_py, l_pz, l_e = data.L0_PX, data.L0_PY, data.L0_PZ, data.L0_PE
        
        # Calculate invariant mass: M² = E² - p²
        px_tot = p_px + k_px + l_px
        py_tot = p_py + k_py + l_py
        pz_tot = p_pz + k_pz + l_pz
        e_tot = p_e + k_e + l_e
        
        m_squared = e_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)
        # Handle numerical precision issues
        m_squared = ak.where(m_squared < 0, 0, m_squared)
        
        return np.sqrt(m_squared)
    
    def apply_jpsi_region(self, data: ak.Array) -> ak.Array:
        """Apply J/ψ region cut"""
        mass = self.calculate_mass(data)
        mask = (mass >= self.jpsi_range[0]) & (mass <= self.jpsi_range[1])
        return data[mask]
    
    def apply_signal_window(self, data: ak.Array) -> ak.Array:
        """Apply J/ψ signal window cut"""
        mass = self.calculate_mass(data)
        mask = (mass >= self.jpsi_window[0]) & (mass <= self.jpsi_window[1])
        return data[mask]
    
    def apply_sidebands(self, data: ak.Array) -> Tuple[ak.Array, ak.Array]:
        """
        Split data into left and right sidebands
        
        Returns:
        - left_sideband_data, right_sideband_data
        """
        mass = self.calculate_mass(data)
        
        left_mask = (mass >= self.left_sideband[0]) & (mass <= self.left_sideband[1])
        right_mask = (mass >= self.right_sideband[0]) & (mass <= self.right_sideband[1])
        
        return data[left_mask], data[right_mask]
    
    def plot_mass_spectrum(self, jpsi_data: ak.Array, kpkm_data: ak.Array,
                          real_data: ak.Array, output_name: str = "jpsi_mass"):
        """
        Plot M(pK⁻Λ̄) mass spectrum with regions marked
        
        Parameters:
        - jpsi_data: J/ψ MC signal
        - kpkm_data: KpKm MC background
        - real_data: Real data
        - output_name: Output filename
        """
        # Calculate masses
        jpsi_mass = ak.to_numpy(self.calculate_mass(jpsi_data))
        kpkm_mass = ak.to_numpy(self.calculate_mass(kpkm_data))
        data_mass = ak.to_numpy(self.calculate_mass(real_data))
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.plot_config.get('figsize', [12, 8]))
        
        # Get binning
        n_bins = self.plot_config.get('bins', {}).get('mass', 100)
        
        # Plot histograms
        if len(jpsi_mass) > 0:
            ax.hist(jpsi_mass, bins=n_bins, range=self.jpsi_range,
                   alpha=0.5, label='J/$\\psi$ Signal MC (SS+OS)',
                   color=self.colors.get('jpsi_signal', '#E41A1C'),
                   histtype='stepfilled')
        
        if len(kpkm_mass) > 0:
            ax.hist(kpkm_mass, bins=n_bins, range=self.jpsi_range,
                   alpha=0.5, label='KpKm Background MC',
                   color=self.colors.get('kpkm_background', '#377EB8'),
                   histtype='stepfilled')
        
        if len(data_mass) > 0:
            ax.hist(data_mass, bins=n_bins, range=self.jpsi_range,
                   histtype='step', linewidth=2, label='Data',
                   color=self.colors.get('data', '#000000'))
        
        # Mark regions
        y_max = ax.get_ylim()[1]
        
        # Signal window
        ax.axvspan(self.jpsi_window[0], self.jpsi_window[1],
                  alpha=0.2, color='green', label='Signal window')
        
        # Sidebands
        ax.axvspan(self.left_sideband[0], self.left_sideband[1],
                  alpha=0.2, color='orange', label='Left sideband')
        ax.axvspan(self.right_sideband[0], self.right_sideband[1],
                  alpha=0.2, color='orange', label='Right sideband')
        
        # Labels
        ax.set_xlabel('$M(pK^-\\bar{\\Lambda})$ [MeV/$c^2$]')
        ax.set_ylabel('Events / bin')
        ax.set_title('J/$\\psi$ Mass Spectrum: $B^+ \\to pK^-\\bar{\\Lambda} K^+$')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add LHCb label
        hep.lhcb.text("Simulation + Data", loc=0)
        
        # Save
        output_path = self.output_dir / f"{output_name}.{self.plot_config.get('format', 'pdf')}"
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.plot_config.get('dpi', 300))
        if self.plot_config.get('save_png', True):
            plt.savefig(output_path.with_suffix('.png'), dpi=150)
        plt.close()
        
        self.logger.info(f"Saved mass spectrum: {output_path}")
    
    def plot_mass_by_cutlevel(self, data_dict: Dict[str, ak.Array], 
                              output_name: str = "jpsi_mass_by_cutlevel"):
        """
        Plot J/ψ mass spectrum at different cut levels
        Shows evolution of mass distribution through cutflow
        
        Parameters:
        - data_dict: Dictionary with keys 'trigger', 'lambda', 'pid', 'bplus'
                    containing data after each cut level
        - output_name: Base name for output file
        """
        self.logger.info("Creating J/ψ mass cutflow comparison...")
        
        # Define cut levels to plot
        cut_levels = [
            ('trigger', 'Trigger Only', self.colors.get('trigger', '#E41A1C')),
            ('lambda', '+ $\\bar{\\Lambda}$ Cuts', self.colors.get('lambda', '#377EB8')),
            ('pid', '+ PID Cuts', self.colors.get('pid', '#4DAF4A')),
            ('bplus', '+ $B^+$ Cuts', self.colors.get('bplus', '#984EA3'))
        ]
        
        # Setup figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Common binning
        n_bins = self.plot_config.get('n_bins', 100)
        mass_range = self.jpsi_range
        
        # Loop through cut levels
        for idx, (key, label, color) in enumerate(cut_levels):
            ax = axes[idx]
            
            # Get data for this cut level
            if key not in data_dict or len(data_dict[key]) == 0:
                self.logger.warning(f"No data for cut level: {key}")
                continue
            
            data = data_dict[key]
            
            # Calculate mass
            mass = self.calculate_mass(data)
            
            if len(mass) == 0:
                self.logger.warning(f"No valid mass values for cut level: {key}")
                continue
            
            # Plot histogram
            counts, bins, patches = ax.hist(mass, bins=n_bins, range=mass_range,
                                           histtype='stepfilled', alpha=0.6,
                                           color=color, edgecolor='black', linewidth=0.5)
            
            # Mark signal window
            y_max = ax.get_ylim()[1]
            ax.axvspan(self.jpsi_window[0], self.jpsi_window[1],
                      alpha=0.15, color='green', zorder=0)
            
            # Mark sidebands
            ax.axvspan(self.left_sideband[0], self.left_sideband[1],
                      alpha=0.1, color='orange', zorder=0)
            ax.axvspan(self.right_sideband[0], self.right_sideband[1],
                      alpha=0.1, color='orange', zorder=0)
            
            # Statistics text
            n_events = len(mass)
            mean_mass = float(np.mean(mass))
            std_mass = float(np.std(mass))
            
            # Count events in signal window
            in_signal = np.sum((mass >= self.jpsi_window[0]) & 
                              (mass <= self.jpsi_window[1]))
            signal_frac = in_signal / n_events * 100 if n_events > 0 else 0
            
            stats_text = (f"Events: {n_events:,}\n"
                         f"Mean: {mean_mass:.1f} MeV\n"
                         f"Std: {std_mass:.1f} MeV\n"
                         f"In window: {signal_frac:.1f}%")
            
            ax.text(0.97, 0.97, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Labels
            ax.set_xlabel('$M(pK^-\\bar{\\Lambda})$ [MeV/$c^2$]')
            ax.set_ylabel('Events / bin')
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add cut level indicator
            ax.text(0.03, 0.97, f"Level {idx+1}",
                   transform=ax.transAxes,
                   verticalalignment='top',
                   fontsize=10,
                   fontweight='bold',
                   color=color)
        
        # Add overall title
        fig.suptitle('J/$\\psi$ Mass Spectrum Evolution Through Cutflow',
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Add LHCb label to first subplot
        plt.sca(axes[0])
        hep.lhcb.text("Simulation", loc=1)
        
        # Save
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        output_path = self.output_dir / f"{output_name}.{self.plot_config.get('format', 'pdf')}"
        plt.savefig(output_path, dpi=self.plot_config.get('dpi', 300))
        if self.plot_config.get('save_png', True):
            plt.savefig(output_path.with_suffix('.png'), dpi=150)
        plt.close()
        
        self.logger.info(f"Saved cutflow comparison: {output_path}")
        
        # Return event counts for reporting
        return {key: len(data_dict[key]) if key in data_dict else 0 
                for key, _, _ in cut_levels}
    
    def calculate_purity(self, signal_data: ak.Array, 
                        background_data: ak.Array) -> Dict:
        """
        Calculate signal purity in different regions
        
        Returns:
        - Dictionary with purity metrics
        """
        # Signal window
        signal_in_window = len(self.apply_signal_window(signal_data))
        background_in_window = len(self.apply_signal_window(background_data))
        total_in_window = signal_in_window + background_in_window
        
        purity_window = signal_in_window / total_in_window if total_in_window > 0 else 0
        
        # Full J/ψ region
        signal_in_region = len(self.apply_jpsi_region(signal_data))
        background_in_region = len(self.apply_jpsi_region(background_data))
        total_in_region = signal_in_region + background_in_region
        
        purity_region = signal_in_region / total_in_region if total_in_region > 0 else 0
        
        return {
            'signal_window': {
                'signal': signal_in_window,
                'background': background_in_window,
                'purity': purity_window * 100,
                'sb_ratio': signal_in_window / background_in_window if background_in_window > 0 else float('inf')
            },
            'jpsi_region': {
                'signal': signal_in_region,
                'background': background_in_region,
                'purity': purity_region * 100,
                'sb_ratio': signal_in_region / background_in_region if background_in_region > 0 else float('inf')
            }
        }


class SelectionStudy:
    """
    Main coordinator for selection optimization study
    """
    def __init__(self, config_path: str):
        """
        Initialize study with configuration file
        
        Parameters:
        - config_path: Path to TOML configuration file
        """
        # Load configuration
        with open(config_path, 'rb') as f:
            self.config = tomli.load(f)
        
        # Setup logging
        self.setup_logging()
        
        # Create output directory
        self.output_dir = Path(self.config['output']['base_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create organized subdirectories
        self._create_output_structure()
        
        # Initialize components
        self.branch_config = BranchConfig()
        self.data_loader = DataLoader(
            data_dir=self.config['data_paths']['data_dir'],
            branch_config=self.branch_config
        )
        self.mc_loader = MCLoader(
            mc_dir=self.config['data_paths']['mc_dir'],
            branch_config=self.branch_config
        )
        self.selection_processor = SelectionProcessor()
        
        self.eff_calc = EfficiencyCalculator(self.logger)
        self.var_analyzer = VariableAnalyzer(self.config, self.output_dir, self.logger)
        self.study_plotter = StudyPlotter(self.config, self.output_dir, self.logger)
        self.jpsi_analyzer = JPsiAnalyzer(self.config, self.output_dir, self.logger)
        
        self.logger.info("="*80)
        self.logger.info("Selection Study Initialized")
        self.logger.info(f"Description: {self.config['metadata']['description']}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info("="*80)
    
    def _create_output_structure(self):
        """Create organized output subdirectories"""
        subdirs = [
            'lambda_quality',      # Lambda selection plots
            'pid_selection',       # PID plots (THE CROWN JEWEL!)
            'bplus_quality',       # B+ quality plots
            'jpsi_region',         # J/ψ analysis plots
            'reports'              # Text reports
        ]
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created output structure with {len(subdirs)} subdirectories")
    
    def setup_logging(self):
        """Setup logging to file and console"""
        log_level = self.config['output'].get('verbose', True) and 'INFO' or 'WARNING'
        log_file = Path(self.config['output']['base_dir']) / 'selection_study.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('SelectionStudy')
        self.logger.setLevel(getattr(logging, log_level))
        
        # File handler
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fh_formatter)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level))
        ch_formatter = logging.Formatter('%(levelname)s - %(message)s')
        ch.setFormatter(ch_formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def load_data(self):
        """Load all datasets (J/ψ MC, KpKm MC, real data)"""
        self.logger.info("Loading datasets...")
        
        # Get configuration
        data_paths = self.config['data_paths']
        mc_config = self.config['mc_selection']
        data_config = self.config['data_selection']
        
        # Load J/ψ signal MC
        self.logger.info("Loading J/ψ signal MC...")
        jpsi_dict = self.mc_loader.load_reconstructed(
            sample_name=mc_config['signal_samples'][0],
            years=data_config['years'],
            polarities=data_config['polarities'],
            track_types=data_config['track_types'],
            channel_name=mc_config['channel_name'],
            preset=mc_config.get('branch_preset', 'standard')
        )
        # Concatenate all year/polarity/track_type combinations
        self.jpsi_signal = ak.concatenate(list(jpsi_dict.values()), axis=0)
        self.logger.info(f"  J/ψ signal: {len(jpsi_dict)} datasets, {len(self.jpsi_signal):,} total events")
        
        # Load KpKm background MC
        self.logger.info("Loading KpKm background MC...")
        kpkm_dict = self.mc_loader.load_reconstructed(
            sample_name=mc_config['background_samples'][0],
            years=data_config['years'],
            polarities=data_config['polarities'],
            track_types=data_config['track_types'],
            channel_name=mc_config['channel_name'],
            preset=mc_config.get('branch_preset', 'standard')
        )
        self.kpkm_background = ak.concatenate(list(kpkm_dict.values()), axis=0)
        self.logger.info(f"  KpKm background: {len(kpkm_dict)} datasets, {len(self.kpkm_background):,} total events")
        
        # Load real data
        self.logger.info("Loading real data...")
        data_dict = self.data_loader.load_data(
            years=data_config['years'],
            polarities=data_config['polarities'],
            track_types=data_config['track_types'],
            channel_name=data_config['channel_name'],
            preset=data_config.get('branch_preset', 'standard')
        )
        self.real_data = ak.concatenate(list(data_dict.values()), axis=0)
        self.logger.info(f"  Real data: {len(data_dict)} datasets, {len(self.real_data):,} total events")
        
        # Compute derived variables
        self.logger.info("\nComputing derived variables...")
        self._compute_derived_variables()
        
        self.logger.info("Data loading complete!\n")
    
    def _compute_derived_variables(self):
        """Compute derived variables that don't exist in the data"""
        # delta_z = L0_ENDVERTEX_Z - Bu_ENDVERTEX_Z
        if 'L0_ENDVERTEX_Z' in self.jpsi_signal.fields and 'Bu_ENDVERTEX_Z' in self.jpsi_signal.fields:
            self.jpsi_signal = ak.with_field(
                self.jpsi_signal, 
                self.jpsi_signal.L0_ENDVERTEX_Z - self.jpsi_signal.Bu_ENDVERTEX_Z,
                "delta_z"
            )
            self.logger.info("  Added delta_z to J/ψ signal")
        
        if 'L0_ENDVERTEX_Z' in self.kpkm_background.fields and 'Bu_ENDVERTEX_Z' in self.kpkm_background.fields:
            self.kpkm_background = ak.with_field(
                self.kpkm_background,
                self.kpkm_background.L0_ENDVERTEX_Z - self.kpkm_background.Bu_ENDVERTEX_Z,
                "delta_z"
            )
            self.logger.info("  Added delta_z to KpKm background")
        
        if 'L0_ENDVERTEX_Z' in self.real_data.fields and 'Bu_ENDVERTEX_Z' in self.real_data.fields:
            self.real_data = ak.with_field(
                self.real_data,
                self.real_data.L0_ENDVERTEX_Z - self.real_data.Bu_ENDVERTEX_Z,
                "delta_z"
            )
            self.logger.info("  Added delta_z to real data")
        
        # kk_product = h1_ProbNNk * h2_ProbNNk
        if 'h1_ProbNNk' in self.jpsi_signal.fields and 'h2_ProbNNk' in self.jpsi_signal.fields:
            self.jpsi_signal = ak.with_field(
                self.jpsi_signal,
                self.jpsi_signal.h1_ProbNNk * self.jpsi_signal.h2_ProbNNk,
                "kk_product"
            )
            self.logger.info("  Added kk_product to J/ψ signal")
        
        if 'h1_ProbNNk' in self.kpkm_background.fields and 'h2_ProbNNk' in self.kpkm_background.fields:
            self.kpkm_background = ak.with_field(
                self.kpkm_background,
                self.kpkm_background.h1_ProbNNk * self.kpkm_background.h2_ProbNNk,
                "kk_product"
            )
            self.logger.info("  Added kk_product to KpKm background")
        
        if 'h1_ProbNNk' in self.real_data.fields and 'h2_ProbNNk' in self.real_data.fields:
            self.real_data = ak.with_field(
                self.real_data,
                self.real_data.h1_ProbNNk * self.real_data.h2_ProbNNk,
                "kk_product"
            )
            self.logger.info("  Added kk_product to real data")
        
        # pid_product = p_ProbNNp * h1_ProbNNk * h2_ProbNNk
        if 'p_ProbNNp' in self.jpsi_signal.fields and 'h1_ProbNNk' in self.jpsi_signal.fields and 'h2_ProbNNk' in self.jpsi_signal.fields:
            self.jpsi_signal = ak.with_field(
                self.jpsi_signal,
                self.jpsi_signal.p_ProbNNp * self.jpsi_signal.h1_ProbNNk * self.jpsi_signal.h2_ProbNNk,
                "pid_product"
            )
            self.logger.info("  Added pid_product to J/ψ signal")
        
        if 'p_ProbNNp' in self.kpkm_background.fields and 'h1_ProbNNk' in self.kpkm_background.fields and 'h2_ProbNNk' in self.kpkm_background.fields:
            self.kpkm_background = ak.with_field(
                self.kpkm_background,
                self.kpkm_background.p_ProbNNp * self.kpkm_background.h1_ProbNNk * self.kpkm_background.h2_ProbNNk,
                "pid_product"
            )
            self.logger.info("  Added pid_product to KpKm background")
        
        if 'p_ProbNNp' in self.real_data.fields and 'h1_ProbNNk' in self.real_data.fields and 'h2_ProbNNk' in self.real_data.fields:
            self.real_data = ak.with_field(
                self.real_data,
                self.real_data.p_ProbNNp * self.real_data.h1_ProbNNk * self.real_data.h2_ProbNNk,
                "pid_product"
            )
            self.logger.info("  Added pid_product to real data")
    
    def _prepare_cutflow_data(self) -> Dict[str, ak.Array]:
        """
        Prepare data at different cut levels for cutflow comparison
        
        Returns dict with keys: 'trigger', 'lambda', 'pid', 'bplus'
        """
        self.logger.info("Preparing cutflow data...")
        
        # Start with J/ψ signal (after trigger)
        trigger_data = self.jpsi_signal
        
        # Apply Lambda cuts
        lambda_cuts = self.config.get('lambda_variables', {})
        lambda_data = trigger_data
        for var_name, var_config in lambda_cuts.items():
            if var_name in ['description', 'enabled'] or not isinstance(var_config, dict):
                continue
            if not var_config.get('enabled', True):
                continue
            
            # Use tight cut (try both 'current_tight' and 'tight_cut' for compatibility)
            cut_value = var_config.get('current_tight') or var_config.get('tight_cut')
            if cut_value is None:
                continue
            
            branch = var_config['branch']
            operator = var_config['operator']
            
            # Apply cut with jagged array handling
            if branch in lambda_data.fields:
                branch_data = lambda_data[branch]
                
                # Create boolean mask
                if operator == '>':
                    mask = branch_data > cut_value
                elif operator == '<':
                    mask = branch_data < cut_value
                else:
                    continue
                
                # Handle jagged arrays - check if mask has more than 1 dimension
                try:
                    # If mask.ndim > 1, it's jagged and needs reduction
                    if mask.ndim > 1:
                        mask = ak.all(mask, axis=-1)
                except AttributeError:
                    # If no ndim attribute, try the operation anyway
                    try:
                        test_mask = ak.all(mask, axis=-1)
                        mask = test_mask
                    except:
                        pass  # Already flat, use as is
                
                lambda_data = lambda_data[mask]
        
        # Apply PID cuts
        pid_cuts = self.config.get('pid_variables', {})
        pid_data = lambda_data
        for var_name, var_config in pid_cuts.items():
            if var_name in ['description', 'enabled'] or not isinstance(var_config, dict):
                continue
            if not var_config.get('enabled', True):
                continue
            
            cut_value = var_config.get('current_tight') or var_config.get('tight_cut')
            if cut_value is None:
                continue
            
            branch = var_config['branch']
            operator = var_config['operator']
            
            if branch in pid_data.fields:
                branch_data = pid_data[branch]
                
                # Create boolean mask
                if operator == '>':
                    mask = branch_data > cut_value
                elif operator == '<':
                    mask = branch_data < cut_value
                else:
                    continue
                
                # Handle jagged arrays
                try:
                    if mask.ndim > 1:
                        mask = ak.all(mask, axis=-1)
                except AttributeError:
                    try:
                        test_mask = ak.all(mask, axis=-1)
                        mask = test_mask
                    except:
                        pass
                
                pid_data = pid_data[mask]
        
        # Apply B+ cuts
        bplus_cuts = self.config.get('bplus_variables', {})
        bplus_data = pid_data
        for var_name, var_config in bplus_cuts.items():
            if var_name in ['description', 'enabled'] or not isinstance(var_config, dict):
                continue
            if not var_config.get('enabled', True):
                continue
            
            cut_value = var_config.get('current_tight') or var_config.get('tight_cut')
            if cut_value is None:
                continue
            
            branch = var_config['branch']
            operator = var_config['operator']
            
            if branch in bplus_data.fields:
                branch_data = bplus_data[branch]
                
                # Create boolean mask
                if operator == '>':
                    mask = branch_data > cut_value
                elif operator == '<':
                    mask = branch_data < cut_value
                else:
                    continue
                
                # Handle jagged arrays
                try:
                    if mask.ndim > 1:
                        mask = ak.all(mask, axis=-1)
                except AttributeError:
                    try:
                        test_mask = ak.all(mask, axis=-1)
                        mask = test_mask
                    except:
                        pass
                
                bplus_data = bplus_data[mask]
        
        cutflow = {
            'trigger': trigger_data,
            'lambda': lambda_data,
            'pid': pid_data,
            'bplus': bplus_data
        }
        
        self.logger.info("Cutflow data prepared:")
        for level, data in cutflow.items():
            self.logger.info(f"  {level}: {len(data):,} events")
        
        return cutflow
    
    def study_lambda_selection(self):
        """
        Phase 1: Study Lambda selection variables
        """
        self.logger.info("="*80)
        self.logger.info("PHASE 1: Lambda Selection Study")
        self.logger.info("="*80)
        
        lambda_vars = self.config['lambda_variables']
        
        for var_name, var_config in lambda_vars.items():
            # Skip meta fields
            if var_name in ['description', 'enabled'] or not isinstance(var_config, dict):
                continue
            
            if not var_config.get('enabled', True):
                continue
            
            self.logger.info(f"\nStudying variable: {var_name}")
            self.logger.info(f"  Branch: {var_config['branch']}")
            self.logger.info(f"  Operator: {var_config['operator']}")
            
            # Plot distributions
            self.var_analyzer.plot_distribution(
                self.jpsi_signal, self.kpkm_background, self.real_data,
                var_config, f"lambda_quality/lambda_{var_name}_dist"
            )
            
            # Efficiency scan for J/ψ signal
            if 'scan_range' in var_config:
                scan_vals = np.linspace(
                    var_config['scan_range'][0],
                    var_config['scan_range'][1],
                    var_config.get('scan_points', 50)
                )
                
                scan_results = self.eff_calc.scan_efficiency(
                    self.jpsi_signal, var_config['branch'],
                    scan_vals, var_config['operator']
                )
                
                self.var_analyzer.plot_efficiency_scan(
                    scan_results, var_config,
                    f"lambda_quality/lambda_{var_name}_effscan",
                    label="J/$\\psi$ Signal MC"
                )
        
        # Create combined efficiency plot
        self.study_plotter.plot_combined_efficiency(
            self.jpsi_signal, lambda_vars, "Lambda",
            "lambda_quality/combined_lambda_efficiency"
        )
        
        # Create 2D optimization plot (FD_CHI2 vs delta_z)
        if 'lambda_fdchi2' in lambda_vars and 'delta_z' in lambda_vars:
            self.study_plotter.plot_2d_correlation(
                self.jpsi_signal,
                lambda_vars['lambda_fdchi2'],
                lambda_vars['delta_z'],
                "lambda_quality/2d_lambda_optimization",
                plot_type="efficiency"
            )
        
        self.logger.info("\nPhase 1 complete!\n")
    
    def study_pid_selection(self):
        """
        Phase 2: Study PID selection variables (THE CROWN JEWEL)
        """
        self.logger.info("="*80)
        self.logger.info("PHASE 2: PID Selection Study - THE CROWN JEWEL")
        self.logger.info("="*80)
        
        pid_vars = self.config['pid_variables']
        
        for var_name, var_config in pid_vars.items():
            # Skip meta fields
            if var_name in ['description', 'enabled'] or not isinstance(var_config, dict):
                continue
            
            if not var_config.get('enabled', True):
                continue
            
            self.logger.info(f"\nStudying variable: {var_name}")
            self.logger.info(f"  Branch: {var_config['branch']}")
            
            # Plot distributions
            self.var_analyzer.plot_distribution(
                self.jpsi_signal, self.kpkm_background, self.real_data,
                var_config, f"pid_selection/pid_{var_name}_dist"
            )
            
            # Efficiency scan
            if 'scan_range' in var_config:
                scan_vals = np.linspace(
                    var_config['scan_range'][0],
                    var_config['scan_range'][1],
                    var_config.get('scan_points', 50)
                )
                
                scan_results = self.eff_calc.scan_efficiency(
                    self.jpsi_signal, var_config['branch'],
                    scan_vals, var_config['operator']
                )
                
                self.var_analyzer.plot_efficiency_scan(
                    scan_results, var_config,
                    f"pid_selection/pid_{var_name}_effscan",
                    label="J/$\\psi$ Signal MC"
                )
        
        # 2D efficiency maps for PID combinations
        self.logger.info("\nCreating 2D PID efficiency maps...")
        
        # p_ProbNNp vs h1_ProbNNk
        if pid_vars.get('p_probnnp', {}).get('enabled') and pid_vars.get('h1_probnnk', {}).get('enabled'):
            self.study_plotter.plot_2d_correlation(
                self.jpsi_signal,
                pid_vars['p_probnnp'],
                pid_vars['h1_probnnk'],
                "pid_selection/2d_p_vs_h1_efficiency",
                plot_type="efficiency"
            )
        
        # h1_ProbNNk vs h2_ProbNNk
        if pid_vars.get('h1_probnnk', {}).get('enabled') and pid_vars.get('h2_probnnk', {}).get('enabled'):
            self.study_plotter.plot_2d_correlation(
                self.jpsi_signal,
                pid_vars['h1_probnnk'],
                pid_vars['h2_probnnk'],
                "pid_selection/2d_h1_vs_h2_efficiency",
                plot_type="efficiency"
            )
        
        # Combined plot for all PID correlations
        self.study_plotter.plot_combined_efficiency(
            self.jpsi_signal, pid_vars, "PID",
            "pid_selection/combined_pid_efficiency"
        )
        
        self.logger.info("\nPhase 2 complete!\n")
    
    def study_bplus_selection(self):
        """
        Phase 3: Study B+ quality variables
        """
        self.logger.info("="*80)
        self.logger.info("PHASE 3: B+ Quality Selection Study")
        self.logger.info("="*80)
        
        bplus_vars = self.config['bplus_variables']
        
        for var_name, var_config in bplus_vars.items():
            # Skip meta fields
            if var_name in ['description', 'enabled'] or not isinstance(var_config, dict):
                continue
            
            if not var_config.get('enabled', True):
                continue
            
            self.logger.info(f"\nStudying variable: {var_name}")
            self.logger.info(f"  Branch: {var_config['branch']}")
            
            # Plot distributions
            self.var_analyzer.plot_distribution(
                self.jpsi_signal, self.kpkm_background, self.real_data,
                var_config, f"bplus_quality/bplus_{var_name}_dist"
            )
            
            # Efficiency scan
            if 'scan_range' in var_config:
                scan_vals = np.linspace(
                    var_config['scan_range'][0],
                    var_config['scan_range'][1],
                    var_config.get('scan_points', 50)
                )
                
                scan_results = self.eff_calc.scan_efficiency(
                    self.jpsi_signal, var_config['branch'],
                    scan_vals, var_config['operator']
                )
                
                self.var_analyzer.plot_efficiency_scan(
                    scan_results, var_config,
                    f"bplus_quality/bplus_{var_name}_effscan",
                    label="J/$\\psi$ Signal MC"
                )
        
        # Create combined efficiency plot
        self.study_plotter.plot_combined_efficiency(
            self.jpsi_signal, bplus_vars, "B+",
            "bplus_quality/combined_bplus_efficiency"
        )
        
        self.logger.info("\nPhase 3 complete!\n")
    
    def study_jpsi_region(self):
        """
        Phase 4: J/ψ region analysis
        """
        self.logger.info("="*80)
        self.logger.info("PHASE 4: J/ψ Region Analysis")
        self.logger.info("="*80)
        
        # Apply J/ψ region cut
        jpsi_signal_region = self.jpsi_analyzer.apply_jpsi_region(self.jpsi_signal)
        kpkm_region = self.jpsi_analyzer.apply_jpsi_region(self.kpkm_background)
        data_region = self.jpsi_analyzer.apply_jpsi_region(self.real_data)
        
        self.logger.info(f"Events in J/ψ region [{self.jpsi_analyzer.jpsi_range[0]}-{self.jpsi_analyzer.jpsi_range[1]} MeV]:")
        self.logger.info(f"  J/ψ signal: {len(jpsi_signal_region)}")
        self.logger.info(f"  KpKm background: {len(kpkm_region)}")
        self.logger.info(f"  Real data: {len(data_region)}")
        
        # Plot mass spectrum
        self.jpsi_analyzer.plot_mass_spectrum(
            jpsi_signal_region, kpkm_region, data_region,
            output_name="jpsi_region/jpsi_mass"
        )
        
        # Create cutflow comparison
        # Show mass evolution through selection levels
        self.logger.info("\nCreating J/ψ mass cutflow comparison...")
        cutflow_data = self._prepare_cutflow_data()
        if cutflow_data:
            cutflow_counts = self.jpsi_analyzer.plot_mass_by_cutlevel(
                cutflow_data,
                output_name="jpsi_region/jpsi_mass_cutflow"
            )
            
            self.logger.info("\nCutflow Event Counts:")
            for level, count in cutflow_counts.items():
                self.logger.info(f"  {level}: {count:,} events")
        
        # Calculate purity
        purity_metrics = self.jpsi_analyzer.calculate_purity(
            jpsi_signal_region, kpkm_region
        )
        
        self.logger.info("\nSignal Purity Metrics:")
        self.logger.info(f"  Signal window [{self.jpsi_analyzer.jpsi_window[0]}-{self.jpsi_analyzer.jpsi_window[1]} MeV]:")
        self.logger.info(f"    Signal events: {purity_metrics['signal_window']['signal']}")
        self.logger.info(f"    Background events: {purity_metrics['signal_window']['background']}")
        self.logger.info(f"    Purity: {purity_metrics['signal_window']['purity']:.2f}%")
        self.logger.info(f"    S/B ratio: {purity_metrics['signal_window']['sb_ratio']:.3f}")
        
        self.logger.info(f"\n  Full J/ψ region:")
        self.logger.info(f"    Signal events: {purity_metrics['jpsi_region']['signal']}")
        self.logger.info(f"    Background events: {purity_metrics['jpsi_region']['background']}")
        self.logger.info(f"    Purity: {purity_metrics['jpsi_region']['purity']:.2f}%")
        self.logger.info(f"    S/B ratio: {purity_metrics['jpsi_region']['sb_ratio']:.3f}")
        
        self.logger.info("\nPhase 4 complete!\n")
    
    def generate_reports(self):
        """Generate comprehensive text reports"""
        self.logger.info("="*80)
        self.logger.info("Generating Text Reports")
        self.logger.info("="*80)
        
        # Prepare cutflow data
        cutflow_data = self._prepare_cutflow_data()
        
        # Generate individual reports
        self._generate_efficiency_summary(cutflow_data)
        self._generate_cutflow_table(cutflow_data)
        self._generate_recommendations()
        
        self.logger.info("Report generation complete!\n")
    
    def _generate_efficiency_summary(self, cutflow_data: Dict[str, ak.Array]):
        """Generate comprehensive efficiency summary report"""
        report_path = self.output_dir / "reports" / "efficiency_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SELECTION EFFICIENCY SUMMARY\n")
            f.write(f"B+ → pK⁻Λ̄ K+ Analysis\n")
            f.write("="*80 + "\n\n")
            
            # Study metadata
            f.write("Study Information:\n")
            f.write(f"  Description: {self.config['metadata']['description']}\n")
            f.write(f"  Author: {self.config['metadata'].get('author', 'Unknown')}\n")
            f.write(f"  Date: {self.config['metadata'].get('date', 'Unknown')}\n")
            f.write("\n")
            
            # Data summary
            f.write("Dataset Summary:\n")
            f.write(f"  J/ψ signal MC: {len(self.jpsi_signal):,} events\n")
            f.write(f"  KpKm background MC: {len(self.kpkm_background):,} events\n")
            f.write(f"  Real data: {len(self.real_data):,} events\n")
            f.write("\n")
            
            # Cutflow efficiency
            f.write("="*80 + "\n")
            f.write("CUTFLOW EFFICIENCY (J/ψ Signal MC)\n")
            f.write("="*80 + "\n\n")
            
            trigger_count = len(cutflow_data['trigger'])
            
            f.write(f"{'Level':<20} {'Events':>12} {'Efficiency':>12} {'Reduction':>12}\n")
            f.write("-"*60 + "\n")
            
            levels = [
                ('Trigger', 'trigger'),
                ('+ Λ cuts', 'lambda'),
                ('+ PID cuts', 'pid'),
                ('+ B+ cuts', 'bplus')
            ]
            
            for label, key in levels:
                count = len(cutflow_data[key])
                eff = count / trigger_count * 100 if trigger_count > 0 else 0
                reduction = (1 - count/trigger_count) * 100 if trigger_count > 0 else 0
                f.write(f"{label:<20} {count:>12,} {eff:>11.2f}% {reduction:>11.2f}%\n")
            
            f.write("\n")
            
            # Phase-by-phase breakdown
            f.write("="*80 + "\n")
            f.write("PHASE-BY-PHASE CUT VALUES\n")
            f.write("="*80 + "\n\n")
            
            # Lambda variables
            f.write("Phase 1: Λ Selection\n")
            f.write("-"*60 + "\n")
            for var_name, var_config in self.config['lambda_variables'].items():
                if var_name in ['description', 'enabled'] or not isinstance(var_config, dict):
                    continue
                if not var_config.get('enabled', True):
                    continue
                
                f.write(f"  {var_name}:\n")
                f.write(f"    Branch: {var_config['branch']}\n")
                f.write(f"    Operator: {var_config['operator']}\n")
                f.write(f"    Loose cut: {var_config.get('current_loose') or var_config.get('loose_cut', 'N/A')}\n")
                f.write(f"    Tight cut: {var_config.get('current_tight') or var_config.get('tight_cut', 'N/A')}\n")
                f.write("\n")
            
            # PID variables
            f.write("Phase 2: PID Selection\n")
            f.write("-"*60 + "\n")
            for var_name, var_config in self.config['pid_variables'].items():
                if var_name in ['description', 'enabled'] or not isinstance(var_config, dict):
                    continue
                if not var_config.get('enabled', True):
                    continue
                
                f.write(f"  {var_name}:\n")
                f.write(f"    Branch: {var_config['branch']}\n")
                f.write(f"    Operator: {var_config['operator']}\n")
                f.write(f"    Loose cut: {var_config.get('current_loose') or var_config.get('loose_cut', 'N/A')}\n")
                f.write(f"    Tight cut: {var_config.get('current_tight') or var_config.get('tight_cut', 'N/A')}\n")
                f.write("\n")
            
            # B+ variables
            f.write("Phase 3: B+ Quality Selection\n")
            f.write("-"*60 + "\n")
            for var_name, var_config in self.config['bplus_variables'].items():
                if var_name in ['description', 'enabled'] or not isinstance(var_config, dict):
                    continue
                if not var_config.get('enabled', True):
                    continue
                
                f.write(f"  {var_name}:\n")
                f.write(f"    Branch: {var_config['branch']}\n")
                f.write(f"    Operator: {var_config['operator']}\n")
                f.write(f"    Loose cut: {var_config.get('current_loose') or var_config.get('loose_cut', 'N/A')}\n")
                f.write(f"    Tight cut: {var_config.get('current_tight') or var_config.get('tight_cut', 'N/A')}\n")
                f.write("\n")
            
            # J/ψ region analysis
            f.write("="*80 + "\n")
            f.write("J/ψ REGION ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            jpsi_signal_region = self.jpsi_analyzer.apply_jpsi_region(self.jpsi_signal)
            kpkm_region = self.jpsi_analyzer.apply_jpsi_region(self.kpkm_background)
            
            purity_metrics = self.jpsi_analyzer.calculate_purity(jpsi_signal_region, kpkm_region)
            
            f.write(f"J/ψ region: {self.jpsi_analyzer.jpsi_range[0]}-{self.jpsi_analyzer.jpsi_range[1]} MeV\n")
            f.write(f"Signal window: {self.jpsi_analyzer.jpsi_window[0]}-{self.jpsi_analyzer.jpsi_window[1]} MeV\n")
            f.write("\n")
            
            f.write("Signal Window Metrics:\n")
            f.write(f"  Signal events: {purity_metrics['signal_window']['signal']:,}\n")
            f.write(f"  Background events: {purity_metrics['signal_window']['background']:,}\n")
            f.write(f"  Purity: {purity_metrics['signal_window']['purity']:.2f}%\n")
            f.write(f"  S/B ratio: {purity_metrics['signal_window']['sb_ratio']:.3f}\n")
            f.write("\n")
            
            f.write("Full J/ψ Region Metrics:\n")
            f.write(f"  Signal events: {purity_metrics['jpsi_region']['signal']:,}\n")
            f.write(f"  Background events: {purity_metrics['jpsi_region']['background']:,}\n")
            f.write(f"  Purity: {purity_metrics['jpsi_region']['purity']:.2f}%\n")
            f.write(f"  S/B ratio: {purity_metrics['jpsi_region']['sb_ratio']:.3f}\n")
            f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        self.logger.info(f"Saved efficiency summary: {report_path}")
    
    def _generate_cutflow_table(self, cutflow_data: Dict[str, ak.Array]):
        """Generate machine-readable cutflow table (CSV)"""
        import csv
        
        report_path = self.output_dir / "reports" / "cutflow_table.csv"
        
        trigger_count = len(cutflow_data['trigger'])
        
        with open(report_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Cut_Level', 'Events', 'Absolute_Efficiency_%', 
                           'Relative_Efficiency_%', 'Reduction_%'])
            
            # Cutflow levels
            levels = [
                ('Trigger', 'trigger'),
                ('Lambda', 'lambda'),
                ('PID', 'pid'),
                ('Bplus', 'bplus')
            ]
            
            prev_count = trigger_count
            for label, key in levels:
                count = len(cutflow_data[key])
                abs_eff = count / trigger_count * 100 if trigger_count > 0 else 0
                rel_eff = count / prev_count * 100 if prev_count > 0 else 0
                reduction = (1 - count/trigger_count) * 100 if trigger_count > 0 else 0
                
                writer.writerow([label, count, f'{abs_eff:.2f}', f'{rel_eff:.2f}', f'{reduction:.2f}'])
                prev_count = count
        
        self.logger.info(f"Saved cutflow table: {report_path}")
    
    def _generate_recommendations(self):
        """Generate recommendations for cut optimization"""
        report_path = self.output_dir / "reports" / "recommendations.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SELECTION RECOMMENDATIONS\n")
            f.write(f"B+ → pK⁻Λ̄ K+ Analysis\n")
            f.write("="*80 + "\n\n")
            
            f.write("Based on the selection optimization study, the following recommendations\n")
            f.write("are provided for the final analysis selection:\n\n")
            
            # Lambda recommendations
            f.write("1. Λ SELECTION (Phase 1)\n")
            f.write("-"*60 + "\n")
            f.write("   Recommended cuts (use TIGHT values for high purity):\n\n")
            
            for var_name, var_config in self.config['lambda_variables'].items():
                if var_name in ['description', 'enabled'] or not isinstance(var_config, dict):
                    continue
                if not var_config.get('enabled', True):
                    continue
                
                tight = var_config.get('current_tight') or var_config.get('tight_cut')
                loose = var_config.get('current_loose') or var_config.get('loose_cut')
                operator = var_config['operator']
                branch = var_config['branch']
                
                if tight is not None:
                    f.write(f"   • {branch} {operator} {tight}")
                    if loose is not None and loose != tight:
                        f.write(f"  (loose: {operator} {loose})")
                    f.write("\n")
            
            f.write("\n")
            
            # PID recommendations
            f.write("2. PID SELECTION (Phase 2)\n")
            f.write("-"*60 + "\n")
            f.write("   Recommended cuts:\n\n")
            
            for var_name, var_config in self.config['pid_variables'].items():
                if var_name in ['description', 'enabled'] or not isinstance(var_config, dict):
                    continue
                if not var_config.get('enabled', True):
                    continue
                
                tight = var_config.get('current_tight') or var_config.get('tight_cut')
                loose = var_config.get('current_loose') or var_config.get('loose_cut')
                operator = var_config['operator']
                branch = var_config['branch']
                
                if tight is not None:
                    f.write(f"   • {branch} {operator} {tight}")
                    if loose is not None and loose != tight:
                        f.write(f"  (loose: {operator} {loose})")
                    f.write("\n")
            
            f.write("\n")
            
            # B+ recommendations
            f.write("3. B+ QUALITY SELECTION (Phase 3)\n")
            f.write("-"*60 + "\n")
            f.write("   Recommended cuts:\n\n")
            
            for var_name, var_config in self.config['bplus_variables'].items():
                if var_name in ['description', 'enabled'] or not isinstance(var_config, dict):
                    continue
                if not var_config.get('enabled', True):
                    continue
                
                tight = var_config.get('current_tight') or var_config.get('tight_cut')
                loose = var_config.get('current_loose') or var_config.get('loose_cut')
                operator = var_config['operator']
                branch = var_config['branch']
                
                if tight is not None:
                    f.write(f"   • {branch} {operator} {tight}")
                    if loose is not None and loose != tight:
                        f.write(f"  (loose: {operator} {loose})")
                    f.write("\n")
            
            f.write("\n")
            
            # General recommendations
            f.write("4. GENERAL RECOMMENDATIONS\n")
            f.write("-"*60 + "\n")
            f.write("   • Review 2D correlation plots for potential interdependencies\n")
            f.write("   • Check combined efficiency plots for optimal cut balance\n")
            f.write("   • Verify J/ψ mass spectrum shows clean signal after cuts\n")
            f.write("   • Consider systematic uncertainties from cut variations\n")
            f.write("   • Validate on independent data samples if available\n")
            f.write("\n")
            
            f.write("5. NEXT STEPS\n")
            f.write("-"*60 + "\n")
            f.write("   1. Implement recommended cuts in main analysis\n")
            f.write("   2. Perform systematic studies with loose/tight variations\n")
            f.write("   3. Cross-check with alternative selection strategies\n")
            f.write("   4. Document final selection in analysis note\n")
            f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("For questions or issues, consult the full study plots in:\n")
            f.write(f"  {self.output_dir}/\n")
            f.write("="*80 + "\n")
        
        self.logger.info(f"Saved recommendations: {report_path}")
    
    def run(self):
        """Execute full study"""
        self.logger.info("Starting Selection Optimization Study")
        self.logger.info(f"Description: {self.config['metadata']['description']}")
        
        # Load data
        self.load_data()
        
        # Run study phases
        self.study_lambda_selection()
        self.study_pid_selection()
        self.study_bplus_selection()
        self.study_jpsi_region()
        
        # Generate reports
        self.generate_reports()
        
        self.logger.info("="*80)
        self.logger.info("SELECTION STUDY COMPLETE")
        self.logger.info(f"All outputs saved to: {self.output_dir}")
        self.logger.info("="*80)


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Selection Optimization Study for B+ → pK⁻Λ̄ K+ Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full study with default config
  python selection_study.py
  
  # Run with custom config
  python selection_study.py -c my_config.toml
  
  # Run specific phase only
  python selection_study.py --phase lambda
  python selection_study.py --phase pid
  python selection_study.py --phase bplus
  python selection_study.py --phase jpsi
        """
    )
    
    parser.add_argument('-c', '--config', 
                       default='selection_study_config.toml',
                       help='Path to configuration file (default: selection_study_config.toml)')
    
    parser.add_argument('--phase',
                       choices=['lambda', 'pid', 'bplus', 'jpsi', 'all'],
                       default='all',
                       help='Study phase to run (default: all)')
    
    parser.add_argument('-v', '--verbose',
                       action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create study
    study = SelectionStudy(args.config)
    
    # Override log level if verbose
    if args.verbose:
        study.logger.setLevel(logging.DEBUG)
    
    # Load data (required for all phases)
    study.load_data()
    
    # Run requested phase(s)
    if args.phase == 'all':
        study.run()
    else:
        if args.phase == 'lambda':
            study.study_lambda_selection()
        elif args.phase == 'pid':
            study.study_pid_selection()
        elif args.phase == 'bplus':
            study.study_bplus_selection()
        elif args.phase == 'jpsi':
            study.study_jpsi_region()


if __name__ == '__main__':
    main()
