#!/usr/bin/env python3
"""
Main Orchestrator for Selection Study

Coordinates the selection optimization study for B+ → pK⁻Λ̄ K+ Analysis.
This is the main entry point for running selection studies.

Author: Mohamed Elashri
Date: October 28, 2025
"""

import sys
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict
import tomli
import numpy as np
import pandas as pd
import awkward as ak

# Suppress font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Add analysis directory to path for imports
analysis_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(analysis_dir))

from data_loader import DataLoader
from mc_loader import MCLoader
from mass_calculator import MassCalculator
from branch_config import BranchConfig

# Import study modules
from selection_efficiency import EfficiencyCalculator
from variable_analyzer import VariableAnalyzer
from plot import StudyPlotter
from jpsi_analyzer import JPsiAnalyzer


class SelectionStudy:
    """
    Main coordinator for selection optimization study
    
    Three-phase workflow:
    1. MC Optimization: Optimize cuts on signal MC for efficiency
    2. Data Application: Apply optimized cuts to real data with trigger
    3. Combined Plots: Create normalized MC vs data comparison plots
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
        
        # Create organized subdirectories for three phases
        self._create_output_structure()
        
        # Storage for optimized cuts (determined in Phase 1, used in Phase 2)
        self.optimized_cuts = {}
        
        # Current phase output directory (set by _set_phase_output_dir)
        self.phase_output_dir = self.output_dir
        
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
        
        # Initialize study modules
        self.eff_calc = EfficiencyCalculator(self.logger)
        self.var_analyzer = VariableAnalyzer(self.config, self.output_dir, self.logger)
        self.study_plotter = StudyPlotter(self.config, self.output_dir, self.logger)
        self.jpsi_analyzer = JPsiAnalyzer(self.config, self.output_dir, self.logger)
        
        self.logger.info("="*80)
        self.logger.info("Selection Study Initialized")
        self.logger.info(f"Description: {self.config['metadata']['description']}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info("="*80)
    
    @staticmethod
    def safe_extract(data):
        """Safely extract scalar values from potentially jagged awkward arrays"""
        if isinstance(data, ak.Array):
            # Check if data has nested structure
            if data.ndim > 1:
                # Flatten jagged arrays
                return ak.to_numpy(ak.flatten(data))
            else:
                return ak.to_numpy(data)
        return np.array(data)
    
    def _create_output_structure(self):
        """Create organized output subdirectories for three-phase workflow"""
        # Phase 1: MC optimization
        mc_base = self.output_dir / self.config['output']['mc_dir']
        mc_subdirs = self.config['output'].get('mc_subdirs', {})
        for subdir in mc_subdirs.values():
            if isinstance(subdir, str):  # Only process string values
                (mc_base / subdir).mkdir(parents=True, exist_ok=True)
        
        # Phase 2: Data application
        data_base = self.output_dir / self.config['output']['data_dir']
        data_subdirs = self.config['output'].get('data_subdirs', {})
        for subdir in data_subdirs.values():
            if isinstance(subdir, str):  # Only process string values
                (data_base / subdir).mkdir(parents=True, exist_ok=True)
        
        total_dirs = len(mc_subdirs) + len(data_subdirs)
        self.logger.info(f"Created two-phase output structure with {total_dirs} subdirectories")
    
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
        """Load all datasets (J/ψ MC for signal, real data for background sidebands)"""
        self.logger.info("Loading datasets...")
        
        # Get configuration
        data_paths = self.config['data_paths']
        mc_config = self.config['mc_selection']
        data_config = self.config['data_selection']
        
        # Load J/ψ signal MC (for efficiency studies)
        self.logger.info("Loading J/ψ signal MC...")
        jpsi_dict = self.mc_loader.load_reconstructed(
            sample_name=mc_config['signal_sample'],
            years=data_config['years'],
            polarities=data_config['polarities'],
            track_types=data_config['track_types'],
            channel_name=mc_config['channel_name'],
            preset=mc_config.get('branch_preset', 'standard')
        )
        # Concatenate all year/polarity/track_type combinations
        self.jpsi_signal = ak.concatenate(list(jpsi_dict.values()), axis=0)
        self.logger.info(f"  J/ψ signal MC: {len(jpsi_dict)} datasets, {len(self.jpsi_signal):,} total events")
        
        # Load real data (for background characterization via sidebands)
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
        
        # Normalize MC to data if requested
        if self.config.get('mc_normalization', {}).get('normalize_to_data', True):
            self.logger.info("\nNormalizing MC to data...")
            self._normalize_mc_to_data()
        
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
            self.logger.info("  Added delta_z to J/ψ signal MC")
        
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
            self.logger.info("  Added kk_product to J/ψ signal MC")
        
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
            self.logger.info("  Added pid_product to J/ψ signal MC")
        
        if 'p_ProbNNp' in self.real_data.fields and 'h1_ProbNNk' in self.real_data.fields and 'h2_ProbNNk' in self.real_data.fields:
            self.real_data = ak.with_field(
                self.real_data,
                self.real_data.p_ProbNNp * self.real_data.h1_ProbNNk * self.real_data.h2_ProbNNk,
                "pid_product"
            )
            self.logger.info("  Added pid_product to real data")
    
    def phase1_mc_optimization(self):
        """
        PHASE 1: Optimize selection cuts using J/ψ MC with S/√B metric
        
        Goal: Scan cut values, calculate S/√B for each combination, find optimal cuts
        Output: mc/ directory with:
            - optimization_table.csv: All cut combinations with S/√B values
            - best_cuts_summary.txt: Recommended cuts
            - cutflow_table.csv: Event counts at each cut stage
        """
        self.logger.info("="*80)
        self.logger.info("PHASE 1: MC OPTIMIZATION (S/√B Maximization)")
        self.logger.info("="*80)
        
        # Update output paths for Phase 1
        self._set_phase_output_dir('mc')
        
        # Perform multidimensional cut optimization
        self.optimize_cuts_grid_search()
        
        self.logger.info("\nPhase 1 complete! Optimal cuts identified.")
    
    def phase2_data_application(self):
        """
        PHASE 2: Apply optimal cuts (from Phase 1) to real data
        
        Goal: Apply best cuts + trigger to real data, extract yields
        Output: data/ directory with:
            - jpsi_mass_data.pdf: J/ψ mass spectrum in data
            - data_yields.txt: Signal and background yields
            - variable_distributions/: Data distributions with cuts applied
        """
        self.logger.info("="*80)
        self.logger.info("PHASE 2: DATA APPLICATION (Apply Optimal Cuts)")
        self.logger.info("="*80)
        
        # Update output paths for Phase 2
        self._set_phase_output_dir('data')
        
        # Apply best cuts to data and analyze
        self.apply_cuts_to_data()
        
        self.logger.info("\nPhase 2 complete! Data analysis finished.")
    
    def _set_phase_output_dir(self, phase: str):
        """Set output directory for current phase"""
        if phase == 'mc':
            self.phase_output_dir = self.output_dir / self.config['output']['mc_dir']
        elif phase == 'data':
            self.phase_output_dir = self.output_dir / self.config['output']['data_dir']
        else:
            self.phase_output_dir = self.output_dir
        
        # Update all analyzers' output directories to match phase
        self.var_analyzer.output_dir = self.phase_output_dir
        self.jpsi_analyzer.output_dir = self.phase_output_dir
        self.study_plotter.output_dir = self.phase_output_dir
    
    def _create_normalized_mass_plot(self):
        """Create properly normalized MC vs data J/ψ mass spectrum"""
        self.logger.info("\n--- Creating Normalized J/ψ Mass Spectrum ---")
        
        # Apply J/ψ region cut to MC and data
        jpsi_signal_region = self.jpsi_analyzer.apply_jpsi_region(self.jpsi_signal)
        data_region = self.jpsi_analyzer.apply_jpsi_region(self.real_data)
        
        # Get sidebands from data for background
        data_left_sb, data_right_sb = self.jpsi_analyzer.apply_sidebands(self.real_data)
        data_sidebands = ak.concatenate([data_left_sb, data_right_sb], axis=0)
        
        # Normalize MC to data using sidebands if not already done
        if not hasattr(self, 'mc_scale_factor'):
            self._normalize_mc_to_data()
        
        # Create combined plot directory
        combined_output = self.phase_output_dir / "mass_spectra"
        combined_output.mkdir(parents=True, exist_ok=True)
        
        # Use existing plotting method but with scaled MC
        import matplotlib.pyplot as plt
        import mplhep as hep
        
        # Calculate masses
        jpsi_mass = ak.to_numpy(self.jpsi_analyzer.calculate_mass(jpsi_signal_region))
        sideband_mass = ak.to_numpy(self.jpsi_analyzer.calculate_mass(data_sidebands))
        data_mass = ak.to_numpy(self.jpsi_analyzer.calculate_mass(data_region))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        n_bins = 100
        
        # Plot MC (scaled by mc_scale_factor)
        if len(jpsi_mass) > 0:
            counts_mc, bins_mc = np.histogram(jpsi_mass, bins=n_bins, range=self.jpsi_analyzer.jpsi_range)
            bin_centers = (bins_mc[:-1] + bins_mc[1:]) / 2
            counts_mc_scaled = counts_mc * self.mc_scale_factor
            ax.step(bin_centers, counts_mc_scaled, where='mid', linewidth=2, 
                   label=f'J/ψ MC (scaled ×{self.mc_scale_factor:.2f})',
                   color='#E41A1C')
        
        # Plot data sidebands (background)
        if len(sideband_mass) > 0:
            ax.hist(sideband_mass, bins=n_bins, range=self.jpsi_analyzer.jpsi_range,
                   histtype='step', linewidth=2, label='Data Sidebands (Background)',
                   color='#377EB8')
        
        # Plot full data
        if len(data_mass) > 0:
            ax.hist(data_mass, bins=n_bins, range=self.jpsi_analyzer.jpsi_range,
                   histtype='step', linewidth=2, label='Full Data',
                   color='#000000', linestyle='--')
        
        # Mark regions
        ax.axvspan(self.jpsi_analyzer.jpsi_window[0], self.jpsi_analyzer.jpsi_window[1],
                  alpha=0.2, color='green', label='Signal window')
        ax.axvspan(self.jpsi_analyzer.left_sideband[0], self.jpsi_analyzer.left_sideband[1],
                  alpha=0.2, color='orange', label='Left sideband')
        ax.axvspan(self.jpsi_analyzer.right_sideband[0], self.jpsi_analyzer.right_sideband[1],
                  alpha=0.2, color='orange', label='Right sideband')
        
        # Labels
        ax.set_xlabel('$M(pK^-\\bar{\\Lambda})$ [MeV/$c^2$]')
        ax.set_ylabel('Events / bin')
        ax.set_title('J/$\\psi$ Mass Spectrum: MC vs Data (Normalized)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add LHCb label
        hep.lhcb.text("Preliminary", loc=0)
        
        # Save
        output_path = combined_output / "jpsi_mass_combined.pdf"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        self.logger.info(f"Saved combined mass spectrum: {output_path}")
    
    def _create_combined_distribution_plots(self):
        """Create normalized MC vs data variable distribution plots"""
        self.logger.info("\n--- Creating Combined Distribution Plots ---")
        
        combined_output = self.phase_output_dir / "distributions"
        combined_output.mkdir(parents=True, exist_ok=True)
        
        # Normalize MC to data if not already done
        if not hasattr(self, 'mc_scale_factor'):
            self._normalize_mc_to_data()
        
        # Plot key variables with MC scaled to data
        all_variables = []
        for group_name in ['lambda_variables', 'pid_variables', 'bplus_variables']:
            group = self.config.get(group_name, {})
            all_variables.extend([
                (var_name, var_config) 
                for var_name, var_config in group.items()
                if isinstance(var_config, dict) and var_name not in ['description', 'enabled']
            ])
        
        for var_name, var_config in all_variables:
            if not var_config.get('enabled', True):
                continue
            
            branch = var_config['branch']
            if branch not in self.jpsi_signal.fields or branch not in self.real_data.fields:
                continue
            
            self.logger.info(f"  Creating combined plot for {var_name}")
            
            # Extract data safely
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
            
            mc_vals = safe_extract(self.jpsi_signal, branch)
            data_vals = safe_extract(self.real_data, branch)
            
            if len(mc_vals) == 0 or len(data_vals) == 0:
                continue
            
            # Create plot
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 7))
            
            plot_range = var_config.get('plot_range', [min(np.min(mc_vals), np.min(data_vals)), 
                                                       max(np.max(mc_vals), np.max(data_vals))])
            n_bins = 50
            
            # Plot MC scaled
            counts_mc, bins, _ = ax.hist(mc_vals, bins=n_bins, range=plot_range,
                                         histtype='step', linewidth=2, 
                                         label='J/ψ MC (scaled)', color='#E41A1C',
                                         weights=np.ones_like(mc_vals) * self.mc_scale_factor)
            
            # Plot data
            ax.hist(data_vals, bins=n_bins, range=plot_range,
                   histtype='step', linewidth=2, label='Data',
                   color='#000000')
            
            # Add cut lines
            operator = var_config.get('operator', '>')
            tight_cut = var_config.get('current_tight')
            loose_cut = var_config.get('current_loose')
            
            if tight_cut is not None:
                ax.axvline(tight_cut, color='red', linestyle='--', linewidth=2,
                          label=f'Tight cut ({operator} {tight_cut})')
            
            if loose_cut is not None and loose_cut != tight_cut:
                ax.axvline(loose_cut, color='orange', linestyle='--', linewidth=2,
                          label=f'Loose cut ({operator} {loose_cut})')
            
            # Labels
            ax.set_xlabel(var_config.get('description', branch))
            ax.set_ylabel('Events')
            ax.set_title(f'{var_config.get("description", branch)} (MC vs Data)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save
            output_path = combined_output / f"{var_name}_combined.pdf"
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
        
        self.logger.info(f"Saved combined distribution plots to: {combined_output}")
    
    def _normalize_mc_to_data(self):
        """
        Normalize MC to data scale using signal window method
        
        This calculates a scale factor to make MC and data comparable.
        For J/ψ signal MC, we normalize to the signal yield in data
        (total events in signal window - background from sidebands).
        """
        norm_config = self.config.get('mc_normalization', {})
        method = norm_config.get('normalization_method', 'sideband')
        
        if method == 'manual':
            self.mc_scale_factor = norm_config.get('manual_scale_factor', 1.0)
            self.logger.info(f"Using manual scale factor: {self.mc_scale_factor:.4f}")
            return
        
        if method == 'sideband':
            # For J/ψ signal MC: normalize to signal yield in data
            # Get MC in signal window
            mc_signal_window = self.jpsi_analyzer.apply_signal_window(
                self.jpsi_analyzer.apply_jpsi_region(self.jpsi_signal)
            )
            n_mc_signal = len(mc_signal_window)
            
            # Get data in signal window
            data_jpsi_region = self.jpsi_analyzer.apply_jpsi_region(self.real_data)
            data_signal_window = self.jpsi_analyzer.apply_signal_window(data_jpsi_region)
            n_data_signal_window = len(data_signal_window)
            
            # Estimate background in signal window from sidebands
            left_sb_data, right_sb_data = self.jpsi_analyzer.apply_sidebands(self.real_data)
            n_data_sidebands = len(left_sb_data) + len(right_sb_data)
            
            sideband_width = (self.jpsi_analyzer.left_sideband[1] - self.jpsi_analyzer.left_sideband[0]) + \
                           (self.jpsi_analyzer.right_sideband[1] - self.jpsi_analyzer.right_sideband[0])
            signal_width = self.jpsi_analyzer.jpsi_window[1] - self.jpsi_analyzer.jpsi_window[0]
            n_bkg_in_signal = n_data_sidebands * (signal_width / sideband_width)
            
            # Signal yield = total - background
            n_data_signal = n_data_signal_window - n_bkg_in_signal
            
            if n_mc_signal > 0:
                # Scale factor: signal in data / signal in MC
                self.mc_scale_factor = n_data_signal / n_mc_signal
                self.logger.info(f"Signal window normalization:")
                self.logger.info(f"  MC signal events: {n_mc_signal:,}")
                self.logger.info(f"  Data in signal window: {n_data_signal_window:,}")
                self.logger.info(f"  Background estimate: {n_bkg_in_signal:.1f}")
                self.logger.info(f"  Data signal yield: {n_data_signal:.1f}")
                self.logger.info(f"  Scale factor (data/MC): {self.mc_scale_factor:.4f}")
            else:
                self.logger.warning("No MC events in signal window! Using scale factor = 1.0")
                self.mc_scale_factor = 1.0
        
        elif method == 'luminosity':
            # This would require cross-section, BR, efficiency, luminosity
            # Placeholder for now
            self.logger.warning("Luminosity normalization not yet implemented. Using scale factor = 1.0")
            self.mc_scale_factor = 1.0
        
        else:
            self.logger.warning(f"Unknown normalization method '{method}'. Using scale factor = 1.0")
            self.mc_scale_factor = 1.0
    
    def optimize_cuts_grid_search(self):
        """
        Perform grid search over cut values to maximize S/√B
        
        Scans combinations of cuts, calculates S/√B for each, saves results to table.
        """
        self.logger.info("\n=== Cut Optimization via Grid Search ===")
        self.logger.info("Scanning cut combinations to maximize S/√B...")
        
        # Get signal (J/ψ region) and background (sidebands) from MC
        jpsi_signal_region = self.jpsi_analyzer.apply_jpsi_region(self.jpsi_signal)
        jpsi_signal_window = self.jpsi_analyzer.apply_signal_window(jpsi_signal_region)
        
        left_sb, right_sb = self.jpsi_analyzer.apply_sidebands(self.jpsi_signal)
        jpsi_sidebands = ak.concatenate([left_sb, right_sb], axis=0)
        
        self.logger.info(f"Signal events (J/ψ window): {len(jpsi_signal_window):,}")
        self.logger.info(f"Background events (sidebands): {len(jpsi_sidebands):,}")
        
        # Prepare variables to scan
        scan_variables = []
        
        # Collect all variables with scan ranges
        for section_name in ['lambda_variables', 'pid_variables', 'bplus_variables']:
            section_vars = self.config.get(section_name, {})
            for var_name, var_config in section_vars.items():
                if var_name in ['description', 'enabled'] or not isinstance(var_config, dict):
                    continue
                if not var_config.get('enabled', True):
                    continue
                if 'scan_range' in var_config and not var_config.get('fixed', False):
                    scan_variables.append((var_name, var_config, section_name))
        
        self.logger.info(f"\nScanning {len(scan_variables)} variables:")
        for var_name, var_config, section in scan_variables:
            self.logger.info(f"  {var_name}: {var_config['scan_range']} ({var_config.get('scan_steps', 20)} steps)")
        
        # Perform 1D scans for each variable
        results = []
        
        for var_name, var_config, section in scan_variables:
            branch = var_config['branch']
            operator = var_config['operator']
            scan_range = var_config['scan_range']
            n_steps = var_config.get('scan_steps', 20)
            
            self.logger.info(f"\n  Scanning {var_name} ({branch})...")
            
            # Generate scan values
            scan_values = np.linspace(scan_range[0], scan_range[1], n_steps)
            
            for cut_value in scan_values:
                # Apply cut to signal
                signal_data = jpsi_signal_window
                if branch in signal_data.fields:
                    vals = signal_data[branch]
                    try:
                        if len(ak.flatten(vals)) != len(vals):
                            vals = vals[:, 0]
                    except:
                        pass
                    
                    if operator == '>':
                        mask = ak.to_numpy(vals) > cut_value
                    elif operator == '<':
                        mask = ak.to_numpy(vals) < cut_value
                    elif operator == '>=':
                        mask = ak.to_numpy(vals) >= cut_value
                    elif operator == '<=':
                        mask = ak.to_numpy(vals) <= cut_value
                    else:
                        continue
                    
                    S = np.sum(mask)  # Signal after cut
                else:
                    S = len(signal_data)
                
                # Apply cut to background
                bkg_data = jpsi_sidebands
                if branch in bkg_data.fields:
                    vals = bkg_data[branch]
                    try:
                        if len(ak.flatten(vals)) != len(vals):
                            vals = vals[:, 0]
                    except:
                        pass
                    
                    if operator == '>':
                        mask = ak.to_numpy(vals) > cut_value
                    elif operator == '<':
                        mask = ak.to_numpy(vals) < cut_value
                    elif operator == '>=':
                        mask = ak.to_numpy(vals) >= cut_value
                    elif operator == '<=':
                        mask = ak.to_numpy(vals) <= cut_value
                    else:
                        continue
                    
                    B = np.sum(mask)  # Background after cut
                else:
                    B = len(bkg_data)
                
                # Calculate metrics
                signal_eff = S / len(jpsi_signal_window) if len(jpsi_signal_window) > 0 else 0
                bkg_rej = 1 - (B / len(jpsi_sidebands)) if len(jpsi_sidebands) > 0 else 0
                s_over_sqrt_b = S / np.sqrt(B) if B > 0 else 0
                
                results.append({
                    'variable': var_name,
                    'section': section,
                    'branch': branch,
                    'operator': operator,
                    'cut_value': cut_value,
                    'signal_events': S,
                    'background_events': B,
                    'signal_efficiency': signal_eff,
                    'background_rejection': bkg_rej,
                    'S_over_sqrtB': s_over_sqrt_b
                })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(results)
        
        # Create output directories
        output_dir = self.phase_output_dir
        tables_dir = output_dir / "cut_tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full table
        csv_file = tables_dir / "optimization_scan_full.csv"
        df.to_csv(csv_file, index=False)
        self.logger.info(f"\nSaved full scan results: {csv_file}")
        
        # Find optimal cuts (maximize S/√B for each variable)
        optimal_cuts = {}
        for var_name in df['variable'].unique():
            var_df = df[df['variable'] == var_name]
            best_idx = var_df['S_over_sqrtB'].idxmax()
            best_row = var_df.loc[best_idx]
            optimal_cuts[var_name] = {
                'cut_value': best_row['cut_value'],
                'operator': best_row['operator'],
                's_over_sqrtb': best_row['S_over_sqrtB'],
                'signal_eff': best_row['signal_efficiency'],
                'bkg_rej': best_row['background_rejection'],
                'signal': best_row['signal_events'],
                'background': best_row['background_events'],
                'branch': best_row['branch'],
                'section': best_row['section']
            }
        
        # Save optimal cuts summary
        self._save_optimal_cuts_summary(optimal_cuts, tables_dir)
        
        # Save markdown table
        self._save_optimization_markdown(df, optimal_cuts, tables_dir)
        
        # Store optimal cuts for Phase 2
        self.optimal_cuts = optimal_cuts
        
        # Create cut visualization plots for MC using the plotter
        self.study_plotter.plot_cut_visualizations_mc(jpsi_signal_window, jpsi_sidebands, optimal_cuts)
        
        self.logger.info(f"\nOptimization complete! Found optimal cuts for {len(optimal_cuts)} variables.")
    
    def _save_optimal_cuts_summary(self, optimal_cuts: dict, tables_dir: Path):
        """Save a text summary of the optimal cuts found."""
        summary_file = tables_dir / "optimal_cuts_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("OPTIMAL CUTS FROM GRID SEARCH (Maximizing S/√B)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total variables optimized: {len(optimal_cuts)}\n\n")
            
            for var_name, opt_data in optimal_cuts.items():
                f.write(f"{var_name}:\n")
                f.write(f"  Optimal cut value: {opt_data['cut_value']:.4f}\n")
                f.write(f"  S/√B at optimum: {opt_data['s_over_sqrtb']:.4f}\n")
                f.write(f"  Signal efficiency: {opt_data['signal_eff']:.2%}\n")
                f.write(f"  Background rejection: {opt_data['bkg_rej']:.2%}\n")
                f.write(f"  Signal (S): {opt_data['signal']:.1f}\n")
                f.write(f"  Background (B): {opt_data['background']:.1f}\n")
                f.write("\n")
        
        self.logger.info(f"Saved optimal cuts summary to {summary_file}")
    
    def _save_optimization_markdown(self, df: pd.DataFrame, optimal_cuts: dict, tables_dir: Path):
        """Save markdown-formatted tables of the optimization results."""
        md_file = tables_dir / "optimization_results.md"
        
        with open(md_file, 'w') as f:
            f.write("# Grid Search Optimization Results\n\n")
            f.write("Maximizing S/√B figure of merit for signal/background separation.\n\n")
            
            f.write("## Optimal Cuts Summary\n\n")
            f.write("| Variable | Optimal Cut | S/√B | Signal Eff | Bkg Rejection | Signal (S) | Background (B) |\n")
            f.write("|----------|-------------|------|------------|---------------|------------|----------------|\n")
            
            for var_name, opt_data in optimal_cuts.items():
                f.write(f"| {var_name} | {opt_data['cut_value']:.4f} | "
                       f"{opt_data['s_over_sqrtb']:.4f} | "
                       f"{opt_data['signal_eff']:.2%} | "
                       f"{opt_data['bkg_rej']:.2%} | "
                       f"{opt_data['signal']:.1f} | "
                       f"{opt_data['background']:.1f} |\n")
            
            f.write("\n## Full Scan Results by Variable\n\n")
            
            # Group by variable and create tables
            for var_name in df['variable'].unique():
                var_df = df[df['variable'] == var_name].copy()
                var_df = var_df.sort_values('cut_value')
                
                f.write(f"### {var_name}\n\n")
                f.write("| Cut Value | Signal (S) | Background (B) | S/√B | Signal Eff | Bkg Rejection |\n")
                f.write("|-----------|------------|----------------|------|------------|---------------|\n")
                
                for _, row in var_df.iterrows():
                    f.write(f"| {row['cut_value']:.4f} | "
                           f"{row['signal_events']:.1f} | "
                           f"{row['background_events']:.1f} | "
                           f"{row['S_over_sqrtB']:.4f} | "
                           f"{row['signal_efficiency']:.2%} | "
                           f"{row['background_rejection']:.2%} |\n")
                
                f.write("\n")
        
        self.logger.info(f"Saved optimization markdown tables to {md_file}")
    
    def apply_cuts_to_data(self):
        """
        Apply optimal cuts (from Phase 1) to real data
        
        Uses optimal cuts determined by grid search to filter data, then calculates yields.
        """
        self.logger.info("\n=== Applying Optimal Cuts to Data ===")
        
        if not hasattr(self, 'optimal_cuts') or not self.optimal_cuts:
            self.logger.error("No optimal cuts available! Run Phase 1 first.")
            return
        
        # Start with full data in J/ψ region
        data_region = self.jpsi_analyzer.apply_jpsi_region(self.real_data)
        initial_count = len(data_region)
        self.logger.info(f"Starting with {initial_count:,} events in J/ψ region")
        
        # Apply optimal cuts sequentially
        self.logger.info("\nApplying optimal cuts:")
        data_after_cuts = data_region
        cut_summary = []
        
        for var_name, cut_data in self.optimal_cuts.items():
            before_count = len(data_after_cuts)
            
            branch = cut_data['branch']
            operator = cut_data['operator']
            cut_value = cut_data['cut_value']
            
            # Apply cut
            if branch in data_after_cuts.fields:
                branch_data = data_after_cuts[branch]
                
                # Handle jagged arrays
                if hasattr(branch_data, 'ndim') and branch_data.ndim > 1:
                    branch_data = ak.firsts(branch_data, axis=1)
                
                if operator == '>':
                    mask = branch_data > cut_value
                elif operator == '<':
                    mask = branch_data < cut_value
                else:
                    continue
                
                # Fill None values in mask with False
                mask = ak.fill_none(mask, False)
                data_after_cuts = data_after_cuts[mask]
                after_count = len(data_after_cuts)
                efficiency = after_count / before_count if before_count > 0 else 0
                
                self.logger.info(f"  {var_name}: {branch} {operator} {cut_value:.4f}")
                self.logger.info(f"    Events: {before_count:,} → {after_count:,} (eff: {efficiency:.2%})")
                
                cut_summary.append({
                    'variable': var_name,
                    'cut': f"{branch} {operator} {cut_value:.4f}",
                    'events_before': before_count,
                    'events_after': after_count,
                    'efficiency': efficiency
                })
        
        # Final yield calculation
        final_count = len(data_after_cuts)
        overall_efficiency = final_count / initial_count if initial_count > 0 else 0
        
        self.logger.info(f"\n=== Final Data Yields ===")
        self.logger.info(f"Initial events (J/ψ region): {initial_count:,}")
        self.logger.info(f"After optimal cuts: {final_count:,}")
        self.logger.info(f"Overall efficiency: {overall_efficiency:.2%}")
        
        # Calculate signal and background in final sample
        signal_window_data = self.jpsi_analyzer.apply_signal_window(data_after_cuts)
        
        # Get sidebands for background estimation (also with cuts applied)
        data_left_sb, data_right_sb = self.jpsi_analyzer.apply_sidebands(self.real_data)
        data_sidebands = ak.concatenate([data_left_sb, data_right_sb], axis=0)
        
        # Apply same cuts to sidebands
        for var_name, cut_data in self.optimal_cuts.items():
            branch = cut_data['branch']
            operator = cut_data['operator']
            cut_value = cut_data['cut_value']
            
            if branch in data_sidebands.fields:
                branch_data = data_sidebands[branch]
                
                # Handle jagged arrays
                if hasattr(branch_data, 'ndim') and branch_data.ndim > 1:
                    branch_data = ak.firsts(branch_data, axis=1)
                
                if operator == '>':
                    mask = branch_data > cut_value
                elif operator == '<':
                    mask = branch_data < cut_value
                else:
                    continue
                
                # Fill None values in mask with False
                mask = ak.fill_none(mask, False)
                data_sidebands = data_sidebands[mask]
        
        # Background estimation
        sideband_width = (self.jpsi_analyzer.left_sideband[1] - self.jpsi_analyzer.left_sideband[0]) + \
                        (self.jpsi_analyzer.right_sideband[1] - self.jpsi_analyzer.right_sideband[0])
        signal_width = self.jpsi_analyzer.jpsi_window[1] - self.jpsi_analyzer.jpsi_window[0]
        bkg_in_signal = len(data_sidebands) * (signal_width / sideband_width)
        signal_events = len(signal_window_data) - bkg_in_signal
        signal_purity = (signal_events / len(signal_window_data) * 100) if len(signal_window_data) > 0 else 0
        
        self.logger.info(f"\nSignal Window [{self.jpsi_analyzer.jpsi_window[0]}-{self.jpsi_analyzer.jpsi_window[1]} MeV]:")
        self.logger.info(f"  Events in signal window: {len(signal_window_data):,}")
        self.logger.info(f"  Expected background: {bkg_in_signal:.1f}")
        self.logger.info(f"  Estimated signal: {signal_events:.1f}")
        self.logger.info(f"  Signal purity: {signal_purity:.1f}%")
        
        # Save cut summary table
        self._save_data_cut_summary(cut_summary, final_count, signal_events, bkg_in_signal, signal_purity)
        
        # Create cut visualization plots for data using the plotter
        self.study_plotter.plot_cut_visualizations_data(data_region, self.optimal_cuts)
        
        # Plot mass spectrum after cuts using the plotter
        self.study_plotter.plot_data_mass_spectrum(data_after_cuts, data_sidebands, self.jpsi_analyzer)
        
        # Store for Phase 3
        self.data_after_optimal_cuts = data_after_cuts
        
        self.logger.info("\nData analysis with optimal cuts complete!")
    
    def _save_data_cut_summary(self, cut_summary: list, final_count: int, 
                                signal_events: float, bkg_events: float, signal_purity: float):
        """Save summary of cuts applied to data with yields"""
        tables_dir = self.phase_output_dir / "cut_tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_file = tables_dir / "data_cuts_applied.csv"
        df = pd.DataFrame(cut_summary)
        df.to_csv(csv_file, index=False)
        self.logger.info(f"Saved cut summary to {csv_file}")
        
        # Save text summary
        summary_file = tables_dir / "data_yields_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DATA YIELDS AFTER OPTIMAL CUTS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("CUTS APPLIED:\n")
            for cut_info in cut_summary:
                f.write(f"  {cut_info['variable']}: {cut_info['cut']}\n")
                f.write(f"    Efficiency: {cut_info['efficiency']:.2%}\n")
            
            f.write(f"\nFINAL YIELDS:\n")
            f.write(f"  Total events after cuts: {final_count}\n")
            f.write(f"  Estimated signal events: {signal_events:.1f}\n")
            f.write(f"  Expected background events: {bkg_events:.1f}\n")
            f.write(f"  Signal purity: {signal_purity:.1f}%\n")
        
        self.logger.info(f"Saved yields summary to {summary_file}")
    
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
            
            # Use tight cut
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
                
                if operator == '>':
                    mask = branch_data > cut_value
                elif operator == '<':
                    mask = branch_data < cut_value
                else:
                    continue
                
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
                
                if operator == '>':
                    mask = branch_data > cut_value
                elif operator == '<':
                    mask = branch_data < cut_value
                else:
                    continue
                
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
            f.write(f"  Real data: {len(self.real_data):,} events\n")
            
            # Extract sideband info
            data_left_sb, data_right_sb = self.jpsi_analyzer.apply_sidebands(self.real_data)
            data_sidebands = ak.concatenate([data_left_sb, data_right_sb], axis=0)
            f.write(f"  Data sidebands (background): {len(data_sidebands):,} events\n")
            f.write("\n")
            
            # MC normalization info
            if hasattr(self, 'mc_scale_factor'):
                f.write("MC Normalization:\n")
                f.write(f"  Method: {self.config.get('mc_normalization', {}).get('normalization_method', 'N/A')}\n")
                f.write(f"  Scale factor (data/MC): {self.mc_scale_factor:.4f}\n")
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
            data_region = self.jpsi_analyzer.apply_jpsi_region(self.real_data)
            
            # Extract sideband data for background estimation
            data_left_sb, data_right_sb = self.jpsi_analyzer.apply_sidebands(self.real_data)
            data_sidebands = ak.concatenate([data_left_sb, data_right_sb], axis=0)
            
            # Signal window analysis
            signal_window_mc = self.jpsi_analyzer.apply_signal_window(jpsi_signal_region)
            data_signal_window = self.jpsi_analyzer.apply_signal_window(data_region)
            
            # Estimate background from sidebands
            sideband_width = (self.jpsi_analyzer.left_sideband[1] - self.jpsi_analyzer.left_sideband[0]) + \
                            (self.jpsi_analyzer.right_sideband[1] - self.jpsi_analyzer.right_sideband[0])
            signal_width = self.jpsi_analyzer.jpsi_window[1] - self.jpsi_analyzer.jpsi_window[0]
            bkg_estimate = len(data_sidebands) * (signal_width / sideband_width)
            signal_estimate = len(data_signal_window) - bkg_estimate
            
            f.write(f"J/ψ region: {self.jpsi_analyzer.jpsi_range[0]}-{self.jpsi_analyzer.jpsi_range[1]} MeV\n")
            f.write(f"Signal window: {self.jpsi_analyzer.jpsi_window[0]}-{self.jpsi_analyzer.jpsi_window[1]} MeV\n")
            f.write(f"Left sideband: {self.jpsi_analyzer.left_sideband[0]}-{self.jpsi_analyzer.left_sideband[1]} MeV\n")
            f.write(f"Right sideband: {self.jpsi_analyzer.right_sideband[0]}-{self.jpsi_analyzer.right_sideband[1]} MeV\n")
            f.write("\n")
            
            f.write("Signal Window Metrics:\n")
            f.write(f"  J/ψ MC events: {len(signal_window_mc):,}\n")
            f.write(f"  Data events: {len(data_signal_window):,}\n")
            f.write(f"  Background estimate (from sidebands): {bkg_estimate:.1f}\n")
            f.write(f"  Signal estimate: {signal_estimate:.1f}\n")
            if len(data_signal_window) > 0:
                f.write(f"  Signal purity: {signal_estimate/len(data_signal_window)*100:.1f}%\n")
            f.write("\n")
            
            f.write("Data Sideband Region:\n")
            f.write(f"  Total sideband events: {len(data_sidebands):,}\n")
            f.write(f"  Sideband width: {sideband_width:.0f} MeV\n")
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
        """Execute full three-phase study"""
        self.logger.info("="*80)
        self.logger.info("SELECTION OPTIMIZATION STUDY - THREE-PHASE WORKFLOW")
        self.logger.info("="*80)
        self.logger.info(f"Description: {self.config['metadata']['description']}")
        self.logger.info("")
        
        # Load all data
        self.load_data()
        
        # ====================================================================
        # PHASE 1: MC OPTIMIZATION
        # ====================================================================
        if self.config['study_workflow'].get('run_mc_optimization', True):
            self.logger.info("\n" + "="*80)
            self.logger.info("PHASE 1: MC OPTIMIZATION (Signal Efficiency)")
            self.logger.info("="*80)
            self.logger.info("Goal: Optimize cuts on J/ψ MC to maximize signal efficiency")
            self.logger.info("Output: mc/ directory\n")
            
            self.phase1_mc_optimization()
        
        # ====================================================================
        # PHASE 2: DATA APPLICATION
        # ====================================================================
        if self.config['study_workflow'].get('run_data_application', True):
            self.logger.info("\n" + "="*80)
            self.logger.info("PHASE 2: DATA APPLICATION (Real Data Analysis)")
            self.logger.info("="*80)
            self.logger.info("Goal: Apply optimized cuts + trigger to real data")
            self.logger.info("Output: data/ directory\n")
            
            self.phase2_data_application()
        
        self.logger.info("\n" + "="*80)
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
  python main.py
  
  # Run with custom config
  python main.py -c my_config.toml
  
  # Run specific phase only
  python main.py --phase lambda
  python main.py --phase pid
  python main.py --phase bplus
  python main.py --phase jpsi
        """
    )
    
    parser.add_argument('-c', '--config', 
                       default='config.toml',
                       help='Path to configuration file (default: config.toml)')
    
    parser.add_argument('-v', '--verbose',
                       action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create study
    study = SelectionStudy(args.config)
    
    # Override log level if verbose
    if args.verbose:
        study.logger.setLevel(logging.DEBUG)
    
    # Run two-phase study
    study.run()


if __name__ == '__main__':
    main()
