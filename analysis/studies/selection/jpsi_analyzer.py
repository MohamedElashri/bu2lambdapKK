#!/usr/bin/env python3
"""
J/ψ Analyzer Module for Selection Study

J/ψ mass spectrum analysis and signal extraction.

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

# Add analysis directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from mass_calculator import MassCalculator


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
    
    def plot_mass_spectrum(self, jpsi_data: ak.Array, data_sidebands: ak.Array,
                          real_data: ak.Array, output_name: str = "jpsi_mass"):
        """
        Plot M(pK⁻Λ̄) mass spectrum with regions marked
        
        Parameters:
        - jpsi_data: J/ψ MC signal (scaled to data)
        - data_sidebands: Real data sidebands (for background shape)
        - real_data: Full real data
        - output_name: Output filename
        """
        # Calculate masses
        jpsi_mass = ak.to_numpy(self.calculate_mass(jpsi_data))
        sideband_mass = ak.to_numpy(self.calculate_mass(data_sidebands))
        data_mass = ak.to_numpy(self.calculate_mass(real_data))
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.plot_config.get('figsize', [12, 8]))
        
        # Get binning
        n_bins = self.plot_config.get('bins', {}).get('mass', 100)
        
        # Plot histograms
        if len(jpsi_mass) > 0:
            ax.hist(jpsi_mass, bins=n_bins, range=self.jpsi_range,
                   histtype='step', linewidth=2, label='J/ψ MC (scaled)',
                   color=self.colors.get('jpsi_signal', '#E41A1C'))
        
        if len(sideband_mass) > 0:
            ax.hist(sideband_mass, bins=n_bins, range=self.jpsi_range,
                   histtype='step', linewidth=2, label='Data Sidebands (Background)',
                   color=self.colors.get('data_sideband', '#377EB8'))
        
        if len(data_mass) > 0:
            ax.hist(data_mass, bins=n_bins, range=self.jpsi_range,
                   histtype='step', linewidth=2, label='Full Data',
                   color=self.colors.get('data', '#000000'), linestyle='--')
        
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
            
            if key not in data_dict:
                ax.text(0.5, 0.5, f'No data for {label}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(label)
                continue
            
            data = data_dict[key]
            n_events = len(data)
            
            if n_events == 0:
                ax.text(0.5, 0.5, 'No events',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{label}\n(0 events)')
                continue
            
            # Calculate mass
            mass = ak.to_numpy(self.calculate_mass(data))
            
            # Create histogram
            counts, bin_edges = np.histogram(mass, bins=n_bins, range=mass_range)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = (mass_range[1] - mass_range[0]) / n_bins
            
            # Plot
            ax.hist(mass, bins=n_bins, range=mass_range,
                   histtype='stepfilled', alpha=0.3, color=color,
                   label=f'{n_events} events')
            ax.hist(mass, bins=n_bins, range=mass_range,
                   histtype='step', linewidth=2, color=color)
            
            # Mark signal window
            ax.axvspan(self.jpsi_window[0], self.jpsi_window[1],
                      alpha=0.2, color='green', zorder=0)
            
            # Labels
            ax.set_xlabel('$M(pK^-\\bar{\\Lambda})$ [MeV/$c^2$]')
            ax.set_ylabel(f'Events / ({bin_width:.1f} MeV/$c^2$)')
            ax.set_title(f'{label}\n({n_events} events)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
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
        purity_signal = signal_in_window / total_in_window if total_in_window > 0 else 0.0
        
        # Full J/ψ region
        signal_in_region = len(self.apply_jpsi_region(signal_data))
        background_in_region = len(self.apply_jpsi_region(background_data))
        
        total_in_region = signal_in_region + background_in_region
        purity_region = signal_in_region / total_in_region if total_in_region > 0 else 0.0
        
        return {
            'signal_window': {
                'signal': signal_in_window,
                'background': background_in_window,
                'total': total_in_window,
                'purity': purity_signal
            },
            'jpsi_region': {
                'signal': signal_in_region,
                'background': background_in_region,
                'total': total_in_region,
                'purity': purity_region
            }
        }
