"""
Module for creating plots for B+ → pK⁻Λ̄ K+ analysis
"""

import logging
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from pathlib import Path

class MassSpectrumPlotter:
    """Class for creating mass spectrum plots"""
    
    def __init__(self, output_dir):
        """
        Initialize with output directory
        
        Parameters:
        - output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("Bu2LambdaPKK.MassSpectrumPlotter")
    
    def plot_jpsi_mass_spectrum(self, data, mass_range=(2900, 3300), bins=50):
        """
        Plot the pK⁻Λ̄ mass spectrum to visualize the J/ψ peak
        
        Parameters:
        - data: Dictionary with data arrays containing M_pKLambdabar
        - mass_range: Range for the mass plot in MeV
        - bins: Number of bins for the histogram
        """
        # Combine all datasets
        all_masses = []
        for key, events in data.items():
            if 'M_pKLambdabar' in events.fields:
                masses = ak.to_numpy(events['M_pKLambdabar'])
                all_masses.append(masses)
                
                # Also create individual plots for each dataset
                self._create_mass_plot(masses, f"jpsi_mass_{key}", mass_range, bins)
        
        if not all_masses:
            self.logger.warning("No mass data available for plotting")
            return
        
        # Create combined plot
        all_masses = np.concatenate(all_masses)
        self._create_mass_plot(all_masses, "jpsi_mass_combined", mass_range, bins)
    
    def _create_mass_plot(self, masses, filename, mass_range, bins):
        """
        Create and save a mass spectrum plot
        
        Parameters:
        - masses: Array of mass values
        - filename: Base filename for the plot
        - mass_range: Range for the mass plot
        - bins: Number of bins
        """
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        hist, bin_edges = np.histogram(masses, bins=bins, range=mass_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot histogram
        plt.bar(bin_centers, hist, width=(mass_range[1]-mass_range[0])/bins, alpha=0.7)
        
        # Add J/ψ mass line
        plt.axvline(x=3096.9, color='r', linestyle='--', label='J/ψ mass')
        
        # Add labels and title
        plt.xlabel('M(pK⁻Λ̄) [MeV/c²]')
        plt.ylabel('Candidates / ({:.1f} MeV/c²)'.format((mass_range[1]-mass_range[0])/bins))
        plt.title('pK⁻Λ̄ Mass Spectrum')
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Save plot
        plot_path = self.output_dir / f"{filename}.pdf"
        plt.savefig(plot_path)
        plt.close()
        
        self.logger.info(f"Created plot: {plot_path}")