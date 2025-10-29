#!/usr/bin/env python3
"""
Plot Module for Selection Study

Wrapper around main plotter.py with additional study-specific plotting functions.

Author: Mohamed Elashri
Date: October 28, 2025
"""

import logging
import sys
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
from pathlib import Path
from typing import Dict, List, Tuple

# Import main plotter from analysis
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from plotter import MassSpectrumPlotter


class StudyPlotter:
    """
    Create comparison and summary plots for selection study
    """
    def __init__(self, config: dict, output_dir: Path, logger=None):
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        self.plot_config = config.get('plot_settings', {})
        self.colors = self.plot_config.get('colors', {})
        
        # Initialize main plotter for mass spectrum plots
        self.mass_plotter = MassSpectrumPlotter(output_dir)
        
        # Set default font configuration for all plots
        self._configure_fonts()
    
    def _configure_fonts(self):
        """Configure matplotlib font settings for consistent styling across all plots"""
        import matplotlib as mpl
        
        # Set font sizes
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 13,
            'figure.titlesize': 20,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
            'mathtext.fontset': 'cm',
            'axes.linewidth': 1.5,
            'grid.linewidth': 1.0,
            'lines.linewidth': 2.5,
            'patch.linewidth': 1.5,
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
            'xtick.minor.width': 1.0,
            'ytick.minor.width': 1.0,
        })
    
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
            counts = [cuts[cut_name]['n_passed'] for cut_name in cut_names]
            color = self.colors.get(dataset_name, f'C{i}')
            ax1.bar(x_pos + i * width, counts, width, label=dataset_name, color=color)
        
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
            efficiencies = [cuts[cut_name]['abs_efficiency'] * 100 for cut_name in cut_names]
            color = self.colors.get(dataset_name, 'blue')
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
        from selection_efficiency import EfficiencyCalculator
        eff_calc = EfficiencyCalculator(self.logger)
        
        # Get enabled variables with scan ranges
        plot_vars = []
        for var_name, var_config in variables_config.items():
            # Skip meta fields and non-dict entries
            if var_name in ['description', 'enabled'] or not isinstance(var_config, dict):
                continue
            if not var_config.get('enabled', True):
                continue
            if 'scan_range' not in var_config or 'scan_points' not in var_config:
                continue
            plot_vars.append((var_name, var_config))
        
        if not plot_vars:
            self.logger.warning(f"No variables to plot for {phase_name}")
            return
        
        # Create subplots
        n_vars = len(plot_vars)
        n_cols = 2
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = np.array(axes).flatten()
        else:
            axes = axes.flatten()
        
        # Calculate efficiency for each variable
        for idx, (var_name, var_config) in enumerate(plot_vars):
            ax = axes[idx]
            branch = var_config['branch']
            
            if branch not in data.fields:
                ax.text(0.5, 0.5, f'Branch {branch}\nnot found',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(var_config.get('description', var_name))
                continue
            
            # Perform efficiency scan
            scan_range = var_config['scan_range']
            scan_points = var_config['scan_points']
            cut_values = np.linspace(scan_range[0], scan_range[1], scan_points)
            
            scan_results = eff_calc.scan_efficiency(data, branch, cut_values,
                                                   var_config['operator'])
            
            efficiencies = [r[1] * 100 for r in scan_results]
            
            # Plot efficiency curve
            ax.plot(cut_values, efficiencies, 'b-', linewidth=2)
            
            # Mark current cuts
            tight_cut = var_config.get('current_tight')
            loose_cut = var_config.get('current_loose')
            
            if tight_cut is not None:
                tight_idx = np.argmin(np.abs(cut_values - tight_cut))
                ax.axvline(tight_cut, color='red', linestyle='--', linewidth=1.5,
                          label=f'Tight: {efficiencies[tight_idx]:.1f}%')
            
            if loose_cut is not None and loose_cut != tight_cut:
                loose_idx = np.argmin(np.abs(cut_values - loose_cut))
                ax.axvline(loose_cut, color='orange', linestyle='--', linewidth=1.5,
                          label=f'Loose: {efficiencies[loose_idx]:.1f}%')
            
            # Labels
            labels_config = self.plot_config.get('labels', {})
            xlabel = labels_config.get(var_name, branch)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Efficiency [%]')
            ax.set_title(var_config.get('description', var_name))
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 105])
            
            if tight_cut is not None or loose_cut is not None:
                ax.legend(fontsize=9)
        
        # Hide unused subplots
        for idx in range(len(plot_vars), len(axes)):
            axes[idx].set_visible(False)
        
        # Overall title
        fig.suptitle(f'{phase_name} Selection: Combined Efficiency Scans', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Save
        output_path = self.output_dir / f"{output_name}.{self.plot_config.get('format', 'pdf')}"
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(output_path, dpi=self.plot_config.get('dpi', 300))
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
            self.logger.warning(f"Branches {branch1} or {branch2} not found")
            return
        
        # Extract data
        def safe_extract(arr, branch):
            vals = arr[branch]
            try:
                if len(ak.flatten(vals)) != len(vals):
                    vals = vals[:, 0]
            except:
                pass
            return ak.to_numpy(vals)
        
        x_vals = safe_extract(data, branch1)
        y_vals = safe_extract(data, branch2)
        
        if len(x_vals) == 0 or len(y_vals) == 0:
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if plot_type == "efficiency":
            # 2D efficiency map
            scan1 = np.linspace(*var1_config.get('scan_range', [x_vals.min(), x_vals.max()]), 
                              var1_config.get('scan_points', 20))
            scan2 = np.linspace(*var2_config.get('scan_range', [y_vals.min(), y_vals.max()]),
                              var2_config.get('scan_points', 20))
            
            efficiency_grid = np.zeros((len(scan2), len(scan1)))
            
            for i, cut2 in enumerate(scan2):
                for j, cut1 in enumerate(scan1):
                    mask1 = self._apply_cut(x_vals, cut1, var1_config['operator'])
                    mask2 = self._apply_cut(y_vals, cut2, var2_config['operator'])
                    combined_mask = mask1 & mask2
                    efficiency_grid[i, j] = 100 * np.sum(combined_mask) / len(x_vals)
            
            im = ax.imshow(efficiency_grid, extent=[scan1[0], scan1[-1], scan2[0], scan2[-1]],
                          origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=100)
            
            contour = ax.contour(scan1, scan2, efficiency_grid,
                               levels=[50, 60, 70, 80, 90],
                               colors='white', linewidths=1.5, alpha=0.8)
            ax.clabel(contour, inline=True, fontsize=10, fmt='%d%%')
            
            plt.colorbar(im, ax=ax, label='Efficiency [%]')
            
        else:  # density plot
            h, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=50)
            im = ax.imshow(h.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                          origin='lower', aspect='auto', cmap='viridis')
            plt.colorbar(im, ax=ax, label='Events')
        
        # Labels
        ax.set_xlabel(var1_config.get('name', branch1))
        ax.set_ylabel(var2_config.get('name', branch2))
        ax.set_title(f"2D {plot_type.capitalize()}: {var1_config.get('description', branch1)} vs {var2_config.get('description', branch2)}")
        ax.grid(True, alpha=0.2, color='white', linewidth=0.5)
        
        # Save
        output_path = self.output_dir / f"{output_name}.{self.plot_config.get('format', 'pdf')}"
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.plot_config.get('dpi', 300))
        plt.close()
        
        self.logger.info(f"Saved 2D correlation plot: {output_path}")
    
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
    
    def plot_cut_visualizations_mc(self, signal_data: ak.Array, background_data: ak.Array, 
                                   optimal_cuts: dict, output_subdir: str = "cut_visualizations"):
        """
        Create cut visualization plots for MC showing optimal cut locations.
        
        For each variable, plot the distribution with:
        - Histogram of signal (red) and background (blue) 
        - Vertical line at optimal cut value
        - Transparent box highlighting the accepted region
        
        Parameters:
        - signal_data: J/ψ signal MC data
        - background_data: Sideband background data
        - optimal_cuts: Dictionary of optimal cuts from optimization
        - output_subdir: Subdirectory name for plots
        """
        self.logger.info("\n=== Creating MC Cut Visualization Plots ===")
        
        # Create output directory
        cut_viz_dir = self.output_dir / output_subdir
        cut_viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set LHCb style
        plt.style.use(hep.style.LHCb2)
        
        for var_name, cut_info in optimal_cuts.items():
            branch = cut_info['branch']
            cut_value = cut_info['cut_value']
            operator = cut_info['operator']
            
            self.logger.info(f"  Plotting {var_name} ({branch} {operator} {cut_value:.4f})")
            
            # Extract data for signal and background
            if branch not in signal_data.fields or branch not in background_data.fields:
                self.logger.warning(f"    Branch {branch} not found, skipping")
                continue
            
            signal_vals = self._extract_branch_data(signal_data, branch)
            bkg_vals = self._extract_branch_data(background_data, branch)
            
            # Get plot range from config
            plot_range = self._get_plot_range(var_name, signal_vals, bkg_vals)
            
            # Get axis label from config
            axis_label = self.plot_config.get('labels', {}).get(var_name, branch)
            
            # Create figure with LHCb style
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # Plot histograms
            n_bins = self.plot_config.get('bins', {}).get(var_name, 50)
            
            # Plot signal (red)
            counts_sig, bins, _ = ax.hist(signal_vals, bins=n_bins, range=tuple(plot_range),
                                         histtype='step', linewidth=2.5, color='#E41A1C',
                                         label='Signal MC')
            
            # Plot background (blue)
            counts_bkg, _, _ = ax.hist(bkg_vals, bins=n_bins, range=tuple(plot_range),
                                      histtype='step', linewidth=2.5, color='#377EB8',
                                      label='Sidebands')
            
            # Determine accepted region and highlight
            self._highlight_accepted_region(ax, cut_value, operator, plot_range)
            
            # Plot vertical line at cut value
            ymax = max(np.max(counts_sig), np.max(counts_bkg))
            ax.axvline(cut_value, color='#9467BD', linestyle='--', linewidth=2.5,
                      label=f'Cut: {operator} {cut_value:.3f}')
            
            # Labels and styling
            ax.set_xlabel(axis_label, fontsize=16)
            ax.set_ylabel('Events', fontsize=16)
            ax.legend(fontsize=13, loc='best', frameon=True, fancybox=True, shadow=True)
            ax.set_ylim(0, ymax * 1.15)
            
            # Check if log scale is requested
            if self._use_log_scale(var_name):
                ax.set_yscale('log')
                ax.set_ylim(0.5, ymax * 5)
            
            # Add S/√B value as text
            self._add_cut_info_text(ax, cut_info)
            
            # Add LHCb label
            hep.lhcb.text("Simulation", loc=0, ax=ax, fontsize=14)
            
            # Save figure
            output_file = cut_viz_dir / f"{var_name}_cut_visualization.pdf"
            fig.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        self.logger.info(f"Saved {len(optimal_cuts)} MC cut visualization plots to {cut_viz_dir}")
    
    def plot_cut_visualizations_data(self, data_before_cuts: ak.Array, optimal_cuts: dict,
                                     output_subdir: str = "cut_visualizations"):
        """
        Create cut visualization plots for Data showing optimal cut locations.
        
        For each variable, plot the distribution with:
        - Histogram of raw data distribution
        - Vertical line at optimal cut value
        - Transparent box highlighting the accepted region
        
        Parameters:
        - data_before_cuts: Real data before cuts applied
        - optimal_cuts: Dictionary of optimal cuts from MC optimization
        - output_subdir: Subdirectory name for plots
        """
        self.logger.info("\n=== Creating Data Cut Visualization Plots ===")
        
        # Create output directory
        cut_viz_dir = self.output_dir / output_subdir
        cut_viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set LHCb style
        plt.style.use(hep.style.LHCb2)
        
        for var_name, cut_info in optimal_cuts.items():
            branch = cut_info['branch']
            cut_value = cut_info['cut_value']
            operator = cut_info['operator']
            
            self.logger.info(f"  Plotting {var_name} ({branch} {operator} {cut_value:.4f})")
            
            # Extract data
            if branch not in data_before_cuts.fields:
                self.logger.warning(f"    Branch {branch} not found, skipping")
                continue
            
            data_vals = self._extract_branch_data(data_before_cuts, branch)
            
            # Get plot range from config
            plot_range = self._get_plot_range(var_name, data_vals)
            
            # Get axis label from config
            axis_label = self.plot_config.get('labels', {}).get(var_name, branch)
            
            # Create figure with LHCb style
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # Plot histogram
            n_bins = self.plot_config.get('bins', {}).get(var_name, 50)
            
            counts, bins, _ = ax.hist(data_vals, bins=n_bins, range=tuple(plot_range),
                                     histtype='step', linewidth=2.5, color='#000000',
                                     label='Data')
            
            # Determine accepted region and highlight
            self._highlight_accepted_region(ax, cut_value, operator, plot_range)
            
            # Plot vertical line at cut value
            ymax = np.max(counts)
            ax.axvline(cut_value, color='#9467BD', linestyle='--', linewidth=2.5,
                      label=f'Cut: {operator} {cut_value:.3f}')
            
            # Labels and styling
            ax.set_xlabel(axis_label, fontsize=16)
            ax.set_ylabel('Candidates', fontsize=16)
            ax.legend(fontsize=13, loc='best', frameon=True, fancybox=True, shadow=True)
            ax.set_ylim(0, ymax * 1.15)
            
            # Check if log scale is requested
            if self._use_log_scale(var_name):
                ax.set_yscale('log')
                ax.set_ylim(0.5, ymax * 5)
            
            # Add efficiency info as text (from MC optimization)
            self._add_cut_info_text(ax, cut_info, is_data=True)
            
            # Add LHCb label
            hep.lhcb.text("", loc=0, ax=ax, fontsize=14)
            
            # Save figure
            output_file = cut_viz_dir / f"{var_name}_cut_visualization.pdf"
            fig.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        self.logger.info(f"Saved {len(optimal_cuts)} data cut visualization plots to {cut_viz_dir}")
    
    def plot_data_mass_spectrum(self, data_after_cuts: ak.Array, sidebands_after_cuts: ak.Array,
                                jpsi_analyzer, output_subdir: str = "jpsi_analysis"):
        """
        Plot J/ψ mass spectrum after applying optimal cuts with proper LHCb styling.
        
        Parameters:
        - data_after_cuts: Data after all cuts applied
        - sidebands_after_cuts: Sideband data after cuts
        - jpsi_analyzer: JPsiAnalyzer instance for mass calculation
        - output_subdir: Output subdirectory name
        """
        self.logger.info("\n=== Creating J/ψ Mass Spectrum (Data After Cuts) ===")
        
        # Create output directory
        output_dir = self.output_dir / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate masses
        data_mass = ak.to_numpy(jpsi_analyzer.calculate_mass(data_after_cuts))
        sideband_mass = ak.to_numpy(jpsi_analyzer.calculate_mass(sidebands_after_cuts))
        
        # Set LHCb style
        plt.style.use(hep.style.LHCb2)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        n_bins = self.plot_config.get('bins', {}).get('jpsi_mass_spectrum', 100)
        
        # Plot full data after cuts
        if len(data_mass) > 0:
            ax.hist(data_mass, bins=n_bins, range=jpsi_analyzer.jpsi_range,
                   histtype='step', linewidth=2.5, label='Data (after optimal cuts)',
                   color='#000000')
        
        # Plot sidebands after cuts
        if len(sideband_mass) > 0:
            ax.hist(sideband_mass, bins=n_bins, range=jpsi_analyzer.jpsi_range,
                   histtype='step', linewidth=2.5, label='Sidebands (background)',
                   color='#377EB8', linestyle='--')
        
        # Mark signal window
        ax.axvspan(jpsi_analyzer.jpsi_window[0], jpsi_analyzer.jpsi_window[1],
                  alpha=0.2, color='green', label='Signal window', zorder=0)
        
        # Labels and styling
        ax.set_xlabel('$M(pK^{-}\\bar{\\Lambda})$ [MeV/$c^{2}$]', fontsize=16)
        ax.set_ylabel('Candidates / (2 MeV/$c^{2}$)', fontsize=16)
        ax.legend(fontsize=13, loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Add LHCb label
        hep.lhcb.text("", loc=0, ax=ax, fontsize=14)
        
        # Save
        output_file = output_dir / "jpsi_mass_data_after_cuts.pdf"
        fig.savefig(output_file, bbox_inches='tight', dpi=self.plot_config.get('dpi', 300))
        plt.close(fig)
        
        self.logger.info(f"Saved data mass spectrum: {output_file}")
    
    def _extract_branch_data(self, data: ak.Array, branch: str) -> np.ndarray:
        """Extract and flatten branch data from awkward array"""
        vals = data[branch]
        
        # Handle jagged arrays
        try:
            if len(ak.flatten(vals)) != len(vals):
                vals = vals[:, 0]
        except:
            pass
        
        return ak.to_numpy(vals)
    
    def _get_plot_range(self, var_name: str, *data_arrays) -> list:
        """Get plot range for variable from config or auto-calculate"""
        # Try to get from config
        for section_name in ['lambda_variables', 'pid_variables', 'bplus_variables']:
            section = self.config.get(section_name, {})
            if var_name in section and 'plot_range' in section[var_name]:
                return section[var_name]['plot_range']
        
        # Auto range from data
        all_vals = np.concatenate([arr for arr in data_arrays])
        return [float(np.percentile(all_vals, 0.1)), float(np.percentile(all_vals, 99.9))]
    
    def _use_log_scale(self, var_name: str) -> bool:
        """Check if variable should use log scale"""
        for section_name in ['lambda_variables', 'pid_variables', 'bplus_variables']:
            section = self.config.get(section_name, {})
            if var_name in section and section[var_name].get('log_y', False):
                return True
        return False
    
    def _highlight_accepted_region(self, ax, cut_value: float, operator: str, plot_range: list):
        """Highlight the accepted region based on cut operator"""
        if operator in ['>', '>=']:
            # Accepted region is RIGHT of cut
            ax.axvspan(cut_value, plot_range[1], alpha=0.15, color='green',
                      label='Accepted Region', zorder=0)
        elif operator in ['<', '<=']:
            # Accepted region is LEFT of cut
            ax.axvspan(plot_range[0], cut_value, alpha=0.15, color='green',
                      label='Accepted Region', zorder=0)
    
    def _add_cut_info_text(self, ax, cut_info: dict, is_data: bool = False):
        """Add text box with cut information"""
        s_over_sqrtb = cut_info['s_over_sqrtb']
        signal_eff = cut_info['signal_eff']
        bkg_rej = cut_info['bkg_rej']
        
        if is_data:
            textstr = f'Optimal cut from MC:\n'
        else:
            textstr = ''
        
        textstr += f'S/√B = {s_over_sqrtb:.2f}\n'
        textstr += f'Signal Eff. = {signal_eff:.1%}\n'
        textstr += f'Bkg. Rej. = {bkg_rej:.1%}'
        
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
               fontsize=12, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
