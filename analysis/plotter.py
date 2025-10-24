"""
Module for creating plots for B+ → pK⁻Λ̄ K+ analysis

Expected resonances in M(pK⁻Λ̄):
    - J/ψ (3097 MeV)
    - η_c (2984 MeV)
    - χ_c0 (3415 MeV)
    - χ_c1 (3511 MeV)
    - η_c(2S) (3637 MeV)

Example usage:
    plotter = MassSpectrumPlotter(output_dir="output")
    
    # Identify all peaks in the data
    plotter.identify_peaks(data)
    
    # Plot with specific resonances marked
    resonances = plotter.get_resonances('jpsi', 'eta_c', 'chi_c0', 'chi_c1')
    plotter.plot_mass_spectrum(data, resonances=resonances)
    
    # Plot full spectrum without marking resonances
    plotter.plot_mass_spectrum(data)
    
    # Plot J/ψ region specifically
    plotter.plot_mass_spectrum(data, mass_range=(3000, 3200), bins=50,
                              resonances=plotter.get_resonances('jpsi'))
"""

import logging
import numpy as np
import awkward as ak
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
from pathlib import Path
import warnings

# Suppress all font-related warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Configure matplotlib to use available fonts before setting style
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']

# Set LHCb style for all plots
plt.style.use(hep.style.LHCb2)

# Override any Times New Roman settings from the style
matplotlib.rcParams['font.family'] = 'sans-serif'

class MassSpectrumPlotter:
    """Class for creating mass spectrum plots"""
    
    # Expected resonances in pK⁻Λ̄ system (masses in MeV)
    # Widths available in PDG for precise analysis
    KNOWN_RESONANCES = {
        'jpsi': {'name': r'$J/\psi$', 'mass': 3097, 'color': 'red', 'window': 50},
        'eta_c': {'name': r'$\eta_c$', 'mass': 2984, 'color': 'blue', 'window': 50},
        'chi_c0': {'name': r'$\chi_{c0}$', 'mass': 3415, 'color': 'green', 'window': 50},
        'chi_c1': {'name': r'$\chi_{c1}$', 'mass': 3511, 'color': 'orange', 'window': 50},
        'eta_c2s': {'name': r'$\eta_c(2S)$', 'mass': 3637, 'color': 'purple', 'window': 50},
    }
    
    def __init__(self, output_dir):
        """
        Initialize with output directory
        
        Parameters:
        - output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("Bu2LambdaPKK.MassSpectrumPlotter")
    
    def get_resonances(self, *resonance_names):
        """
        Get resonance definitions by name
        
        Parameters:
        - resonance_names: Names of resonances (e.g., 'jpsi', 'psi2s')
        
        Returns:
        - List of resonance dictionaries
        """
        resonances = []
        for name in resonance_names:
            if name in self.KNOWN_RESONANCES:
                resonances.append(self.KNOWN_RESONANCES[name])
            else:
                self.logger.warning(f"Unknown resonance: {name}")
        return resonances
    
    def plot_mass_spectrum(self, data, mass_range=None, bins=100, resonances=None):
        """
        Plot the pK⁻Λ̄ mass spectrum to identify peaks and resonances
        
        Parameters:
        - data: Dictionary with data arrays containing M_pKLambdabar
        - mass_range: Range for the mass plot in MeV (auto-determined if None)
        - bins: Number of bins for the histogram
        - resonances: List of dicts with resonance info to mark on plot
                     e.g., [{'name': 'J/ψ', 'mass': 3096.9, 'color': 'red'}]
        """
        # Combine all datasets
        all_masses = []
        ll_masses = []
        dd_masses = []
        
        for key, events in data.items():
            if 'M_pKLambdabar' in events.fields:
                masses = ak.to_numpy(events['M_pKLambdabar'])
                # Filter out invalid masses (NaN, inf, negative)
                masses = masses[np.isfinite(masses) & (masses > 0)]
                if len(masses) > 0:
                    all_masses.append(masses)
                    
                    # Separate LL and DD
                    if '_LL' in key:
                        ll_masses.append(masses)
                    elif '_DD' in key:
                        dd_masses.append(masses)
                    
                    # Also create individual plots for each dataset
                    self._create_mass_plot(masses, f"mass_spectrum_{key}", 
                                         mass_range, bins, resonances)
                else:
                    self.logger.warning(f"No valid masses found for {key}")
            else:
                self.logger.warning(f"M_pKLambdabar field not found in {key}")
        
        if not all_masses:
            self.logger.error("No mass data available for plotting! Did you calculate the invariant mass?")
            return
        
        # Create combined plot (all data)
        all_masses = np.concatenate(all_masses)
        self._create_mass_plot(all_masses, "mass_spectrum_combined", 
                             mass_range, bins, resonances)
        
        # Create combined LL plot
        if ll_masses:
            ll_masses_combined = np.concatenate(ll_masses)
            self._create_mass_plot(ll_masses_combined, "mass_spectrum_combined_LL", 
                                 mass_range, bins, resonances)
        
        # Create combined DD plot
        if dd_masses:
            dd_masses_combined = np.concatenate(dd_masses)
            self._create_mass_plot(dd_masses_combined, "mass_spectrum_combined_DD", 
                                 mass_range, bins, resonances)
    
    # Keep backward compatibility
    def plot_jpsi_mass_spectrum(self, data, mass_range=(2900, 3300), bins=50):
        """
        Plot the pK⁻Λ̄ mass spectrum focused on the J/ψ region
        (Legacy method - use plot_mass_spectrum for more flexibility)
        """
        jpsi_resonance = [{'name': r'$J/\psi$', 'mass': 3096.9, 'color': 'red'}]
        self.plot_mass_spectrum(data, mass_range, bins, jpsi_resonance)
    
    def identify_peaks(self, data, min_mass=2000, max_mass=5000):
        """
        Identify and report potential peaks in the mass spectrum
        
        Parameters:
        - data: Dictionary with data arrays containing M_pKLambdabar
        - min_mass: Minimum mass to consider (MeV)
        - max_mass: Maximum mass to consider (MeV)
        """
        # Combine all masses
        all_masses = []
        for key, events in data.items():
            if 'M_pKLambdabar' in events.fields:
                masses = ak.to_numpy(events['M_pKLambdabar'])
                # Filter out invalid masses
                masses = masses[np.isfinite(masses) & (masses > 0)]
                if len(masses) > 0:
                    all_masses.append(masses)
            else:
                self.logger.warning(f"M_pKLambdabar field not found in {key}")
        
        if not all_masses:
            self.logger.error("No mass data available for peak identification! Did you calculate the invariant mass?")
            return
        
        all_masses = np.concatenate(all_masses)
        all_masses = all_masses[(all_masses >= min_mass) & (all_masses <= max_mass)]
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("PEAK IDENTIFICATION REPORT")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total candidates: {len(all_masses)}")
        self.logger.info(f"Mass range: [{np.min(all_masses):.1f}, {np.max(all_masses):.1f}] MeV")
        self.logger.info(f"\nChecking for known resonances:")
        
        # Check for each known resonance
        for res_name, res_info in self.KNOWN_RESONANCES.items():
            mass = res_info['mass']
            window = res_info.get('window', 50)
            
            # Count candidates in window
            in_window = np.sum((all_masses >= mass - window) & 
                             (all_masses <= mass + window))
            
            if in_window > 0:
                percentage = 100 * in_window / len(all_masses)
                self.logger.info(f"  {res_info['name']:20s} ({mass:7.1f} MeV): "
                               f"{in_window:5d} candidates ({percentage:5.2f}%)")
        
        self.logger.info(f"{'='*60}\n")
    
    def _create_mass_plot(self, masses, filename, mass_range=None, bins=100, resonances=None):
        """
        Create and save a mass spectrum plot
        
        Parameters:
        - masses: Array of mass values
        - filename: Base filename for the plot
        - mass_range: Range for the mass plot (auto-determined if None)
        - bins: Number of bins
        - resonances: List of dicts with resonance info to mark on plot
        """
        # Auto-determine mass range if not provided
        if mass_range is None:
            mass_min = np.min(masses)
            mass_max = np.max(masses)
            # Add 5% padding
            padding = (mass_max - mass_min) * 0.05
            mass_range = (mass_min - padding, mass_max + padding)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Create histogram
        hist, bin_edges = np.histogram(masses, bins=bins, range=mass_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = (mass_range[1] - mass_range[0]) / bins
        
        # Calculate errors (Poisson statistics)
        hist_errors = np.sqrt(hist)
        
        # Plot histogram with error bars
        ax.errorbar(bin_centers, hist, yerr=hist_errors, 
                   fmt='o', color='black', markersize=4,
                   capsize=2, capthick=1, elinewidth=1,
                   label=r'$B^+ \rightarrow pK^-\bar{\Lambda}$')
        
        # Mark known resonances if provided
        if resonances:
            for res in resonances:
                ax.axvline(x=res['mass'], 
                          color=res.get('color', 'red'), 
                          linestyle='--', 
                          linewidth=2, 
                          label=f"{res['name']} ({res['mass']:.1f} MeV/$c^2$)")
        
        # Add labels and title
        ax.set_xlabel(r'$M(pK^-\bar{\Lambda})$ [MeV/$c^2$]', fontsize=14)
        ax.set_ylabel(f'Candidates / ({bin_width:.1f} MeV/$c^2$)', fontsize=14)
        ax.set_title(r'$pK^-\bar{\Lambda}$ Invariant Mass Spectrum', fontsize=16, pad=20)
        
        # Add grid and legend
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=12, loc='best')
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot in PDF format only
        plot_path_pdf = self.output_dir / f"{filename}.pdf"
        plt.savefig(plot_path_pdf, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Created plot: {plot_path_pdf}")
        
        # Log statistics about the mass distribution
        self.logger.info(f"Mass distribution: min={np.min(masses):.1f}, max={np.max(masses):.1f}, mean={np.mean(masses):.1f}, std={np.std(masses):.1f} MeV")
        
        # Log statistics for each resonance window if provided
        if resonances:
            for res in resonances:
                window_width = res.get('window', 50)  # Default ±50 MeV window
                window = (res['mass'] - window_width, res['mass'] + window_width)
                in_window = np.sum((masses >= window[0]) & (masses <= window[1]))
                total = len(masses)
                self.logger.info(f"Candidates near {res['name']} [{window[0]:.0f}-{window[1]:.0f} MeV]: {in_window}/{total} ({100*in_window/total:.2f}%)")