"""
Module for fitting the J/ψ peak in the pK⁻Λ̄ mass spectrum with improved
techniques for low statistics samples
"""

import logging
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

class JpsiPeakFitter:
    """Class for fitting the J/ψ peak in the pK⁻Λ̄ mass spectrum"""
    
    def __init__(self, output_dir="output"):
        """Initialize the peak fitter"""
        self.logger = logging.getLogger("Bu2LambdaPKK.JpsiPeakFitter")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fit_jpsi_peak(self, data, mass_range=(2900, 3300)):
        """
        Fit the J/ψ peak in the pK⁻Λ̄ mass spectrum with techniques
        optimized for low statistics
        
        Parameters:
        - data: Dictionary with data arrays containing M_pKLambdabar
        - mass_range: Range for the mass fit in MeV
        
        Returns:
        - Dictionary with fit results
        """
        # Combine all datasets
        all_masses = []
        for key, events in data.items():
            if 'M_pKLambdabar' in events.fields:
                masses = ak.to_numpy(events['M_pKLambdabar'])
                all_masses.append(masses)
        
        if not all_masses:
            self.logger.error("No mass data available for fitting")
            return None
        
        all_masses = np.concatenate(all_masses)
        mask = (all_masses >= mass_range[0]) & (all_masses <= mass_range[1])
        masses_in_range = all_masses[mask]
        
        self.logger.info(f"Fitting J/ψ peak with {len(masses_in_range)} events in range {mass_range}")
        
        # Try different binning strategies to find optimal bin width
        # For low statistics, we want fewer bins
        bins = max(10, min(30, int(len(masses_in_range)/5)))
        self.logger.info(f"Using {bins} bins for histogram with {len(masses_in_range)} events")
        
        hist, bin_edges = np.histogram(masses_in_range, bins=bins, range=mass_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = (mass_range[1] - mass_range[0]) / bins
        
        # First, save raw histogram to understand the distribution
        self._plot_raw_histogram(masses_in_range, mass_range, bins)
        
        # Try multiple fitting approaches
        fit_results = self._try_multiple_fits(hist, bin_centers, bin_width, masses_in_range)
        
        # Print selection efficiency before fit
        for key, events in data.items():
            if 'M_pKLambdabar' in events.fields:
                self.logger.info(f"Sample {key}: {len(events)} events after selection")
        
        return fit_results
    
    def _plot_raw_histogram(self, masses, mass_range, bins):
        """Plot raw histogram to visualize the data distribution"""
        plt.figure(figsize=(10, 6))
        
        plt.hist(masses, bins=bins, range=mass_range, alpha=0.7, color='skyblue')
        
        # Add J/ψ mass line
        plt.axvline(x=3096.9, color='r', linestyle='--', label='J/ψ mass')
        
        plt.xlabel('M(pK⁻Λ̄) [MeV/c²]', fontsize=12)
        plt.ylabel('Candidates', fontsize=12)
        plt.title('pK⁻Λ̄ Mass Spectrum - Raw Distribution', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        raw_hist_path = self.output_dir / "jpsi_raw_histogram.png"
        plt.savefig(raw_hist_path, dpi=300)
        plt.close()
        
        self.logger.info(f"Raw histogram saved to {raw_hist_path}")
    
    def _try_multiple_fits(self, hist, bin_centers, bin_width, masses):
        """Try multiple fitting strategies and select the best one"""
        fit_strategies = [
            {
                'name': 'gaussian_fixed_mean',
                'function': self._fit_gaussian_fixed_mean,
                'description': 'Gaussian with fixed J/ψ mean'
            },
            {
                'name': 'gaussian_free_mean',
                'function': self._fit_gaussian_free,
                'description': 'Gaussian with free mean'
            },
            {
                'name': 'voigtian',
                'function': self._fit_voigtian,
                'description': 'Voigtian (Gaussian + Lorentzian)'
            }
        ]
        
        best_fit = None
        best_chi2 = float('inf')
        
        for strategy in fit_strategies:
            try:
                self.logger.info(f"Trying fit strategy: {strategy['description']}")
                fit_result = strategy['function'](hist, bin_centers, bin_width, masses)
                
                if fit_result and fit_result.get('chi2_ndf', float('inf')) < best_chi2:
                    best_fit = fit_result
                    best_chi2 = fit_result['chi2_ndf']
                    self.logger.info(f"New best fit: {strategy['name']} with χ²/ndf = {best_chi2:.2f}")
            except Exception as e:
                self.logger.warning(f"Fit strategy {strategy['name']} failed: {e}")
        
        if best_fit:
            self.logger.info(f"Best fit selected: χ²/ndf = {best_fit['chi2_ndf']:.2f}, "
                           f"signal yield = {best_fit['signal_yield']:.2f} ± {best_fit['signal_yield_error']:.2f}")
        else:
            self.logger.error("All fit strategies failed")
            best_fit = {
                'signal_yield': 0,
                'signal_yield_error': 0,
                'background_yield': len(masses),
                'background_yield_error': np.sqrt(len(masses)),
                'mean': 3096.9,
                'mean_error': 0,
                'width': 15.0,
                'width_error': 0,
                'fit_status': 'all_failed'
            }
        
        return best_fit
    
    def _fit_gaussian_fixed_mean(self, hist, bin_centers, bin_width, masses):
        """Fit with Gaussian signal (fixed mean) + linear background"""
        # Initial parameter guesses [signal_yield, background_yield, sigma, a, b]
        p0 = [len(masses) * 0.2, len(masses) * 0.8, 15.0, 1.0, 0.0]
        
        # Function with fixed mean at J/ψ mass
        def fit_func(x, sig_yield, bkg_yield, sigma, a, b):
            # Signal with fixed mean
            mean = 3096.9  # J/ψ mass
            signal = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / sigma) ** 2)
            signal_norm = np.sum(signal)
            signal = signal / signal_norm * sig_yield if signal_norm > 0 else signal * 0
            
            # Background: linear function
            background = a + b * (x - mean)
            background_norm = np.sum(background)
            background = background / background_norm * bkg_yield if background_norm > 0 else background * 0
            
            return signal + background
        
        try:
            # Perform the fit
            popt, pcov = curve_fit(
                fit_func,
                bin_centers, 
                hist,
                p0=p0,
                bounds=([0, 0, 5, 0, -0.01], 
                        [len(masses), len(masses), 50, 10, 0.01])
            )
            
            # Extract parameters and errors
            perr = np.sqrt(np.diag(pcov))
            
            signal_yield = popt[0]
            background_yield = popt[1]
            sigma = popt[2]
            
            # Create plot
            fixed_mean = 3096.9
            self._plot_fit_result(masses, (2900, 3300), popt, fixed_mean, bin_width, 
                                 'gaussian_fixed_mean', fit_func)
            
            # Calculate chi2/ndf
            expected = fit_func(bin_centers, *popt)
            chi2 = np.sum(((hist - expected) ** 2) / (expected + 1e-10))
            ndf = len(bin_centers) - len(popt)
            chi2_ndf = chi2 / max(1, ndf)
            
            return {
                'signal_yield': signal_yield,
                'signal_yield_error': perr[0],
                'background_yield': background_yield,
                'background_yield_error': perr[1],
                'mean': fixed_mean,
                'mean_error': 0.0,  # Fixed parameter
                'width': sigma,
                'width_error': perr[2],
                'fit_status': 'success',
                'chi2_ndf': chi2_ndf,
                'fit_type': 'gaussian_fixed_mean'
            }
            
        except Exception as e:
            self.logger.error(f"Fixed mean Gaussian fit failed: {e}")
            return None
    
    def _fit_gaussian_free(self, hist, bin_centers, bin_width, masses):
        """Fit with Gaussian signal (free mean) + linear background"""
        # Initial parameter guesses [signal_yield, background_yield, mean, sigma, a, b]
        p0 = [len(masses) * 0.2, len(masses) * 0.8, 3096.9, 15.0, 1.0, 0.0]
        
        # Function with free mean
        def fit_func(x, sig_yield, bkg_yield, mean, sigma, a, b):
            # Signal
            signal = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / sigma) ** 2)
            signal_norm = np.sum(signal)
            signal = signal / signal_norm * sig_yield if signal_norm > 0 else signal * 0
            
            # Background: linear function
            background = a + b * (x - mean)
            background_norm = np.sum(background)
            background = background / background_norm * bkg_yield if background_norm > 0 else background * 0
            
            return signal + background
        
        try:
            # Perform the fit
            popt, pcov = curve_fit(
                fit_func,
                bin_centers, 
                hist,
                p0=p0,
                bounds=([0, 0, 3080, 5, 0, -0.01], 
                        [len(masses), len(masses), 3120, 50, 10, 0.01])
            )
            
            # Extract parameters and errors
            perr = np.sqrt(np.diag(pcov))
            
            signal_yield = popt[0]
            background_yield = popt[1]
            mean = popt[2]
            sigma = popt[3]
            
            # Create plot
            self._plot_fit_result(masses, (2900, 3300), popt, mean, bin_width, 
                                 'gaussian_free_mean', fit_func)
            
            # Calculate chi2/ndf
            expected = fit_func(bin_centers, *popt)
            chi2 = np.sum(((hist - expected) ** 2) / (expected + 1e-10))
            ndf = len(bin_centers) - len(popt)
            chi2_ndf = chi2 / max(1, ndf)
            
            return {
                'signal_yield': signal_yield,
                'signal_yield_error': perr[0],
                'background_yield': background_yield,
                'background_yield_error': perr[1],
                'mean': mean,
                'mean_error': perr[2],
                'width': sigma,
                'width_error': perr[3],
                'fit_status': 'success',
                'chi2_ndf': chi2_ndf,
                'fit_type': 'gaussian_free_mean'
            }
            
        except Exception as e:
            self.logger.error(f"Free mean Gaussian fit failed: {e}")
            return None
    
    def _fit_voigtian(self, hist, bin_centers, bin_width, masses):
        """Fit with Voigtian signal + polynomial background"""
        # For low statistics, this might be overkill, but let's try it
        # Initial parameter guesses [signal_yield, background_yield, mean, sigma, gamma, a, b]
        p0 = [len(masses) * 0.2, len(masses) * 0.8, 3096.9, 10.0, 5.0, 1.0, 0.0]
        
        # Function with Voigtian signal (Gaussian + Lorentzian)
        def fit_func(x, sig_yield, bkg_yield, mean, sigma, gamma, a, b):
            from scipy.special import voigt_profile
            
            # Voigtian profile
            signal = voigt_profile(x - mean, sigma, gamma)
            signal_norm = np.sum(signal)
            signal = signal / signal_norm * sig_yield if signal_norm > 0 else signal * 0
            
            # Background: linear function
            background = a + b * (x - mean)
            background_norm = np.sum(background)
            background = background / background_norm * bkg_yield if background_norm > 0 else background * 0
            
            return signal + background
        
        try:
            # Perform the fit
            popt, pcov = curve_fit(
                fit_func,
                bin_centers, 
                hist,
                p0=p0,
                bounds=([0, 0, 3080, 1, 1, 0, -0.01], 
                        [len(masses), len(masses), 3120, 30, 30, 10, 0.01])
            )
            
            # Extract parameters and errors
            perr = np.sqrt(np.diag(pcov))
            
            signal_yield = popt[0]
            background_yield = popt[1]
            mean = popt[2]
            sigma = popt[3]
            gamma = popt[4]
            
            # Create plot
            self._plot_fit_result(masses, (2900, 3300), popt, mean, bin_width, 
                                 'voigtian', fit_func)
            
            # Calculate chi2/ndf
            expected = fit_func(bin_centers, *popt)
            chi2 = np.sum(((hist - expected) ** 2) / (expected + 1e-10))
            ndf = len(bin_centers) - len(popt)
            chi2_ndf = chi2 / max(1, ndf)
            
            return {
                'signal_yield': signal_yield,
                'signal_yield_error': perr[0],
                'background_yield': background_yield,
                'background_yield_error': perr[1],
                'mean': mean,
                'mean_error': perr[2],
                'width': np.sqrt(sigma**2 + gamma**2),  # Effective width
                'sigma': sigma,
                'sigma_error': perr[3],
                'gamma': gamma,
                'gamma_error': perr[4],
                'fit_status': 'success',
                'chi2_ndf': chi2_ndf,
                'fit_type': 'voigtian'
            }
            
        except Exception as e:
            self.logger.error(f"Voigtian fit failed: {e}")
            return None
    
    def _plot_fit_result(self, masses, mass_range, fit_params, mean, bin_width, fit_type, fit_func):
        """
        Create and save a plot of the fit result
        
        Parameters:
        - masses: Array of mass values
        - mass_range: Range for the mass plot
        - fit_params: Fit parameters
        - mean: Mean of the signal peak
        - bin_width: Width of histogram bins
        - fit_type: Type of fit performed
        - fit_func: Function used for fitting
        """
        plt.figure(figsize=(10, 6))
        
        # Create histogram with optimized binning
        bins = max(10, min(30, int(len(masses)/5)))
        hist, bin_edges = np.histogram(masses, bins=bins, range=mass_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot histogram
        plt.bar(bin_centers, hist, width=bin_width, alpha=0.7, label='Data', color='skyblue')
        
        # Generate fit curve points
        x_fit = np.linspace(mass_range[0], mass_range[1], 1000)
        y_fit = fit_func(x_fit, *fit_params)
        
        # Scale to match histogram binning
        bin_scale = bin_width * len(masses) / np.sum(y_fit)
        y_fit_scaled = y_fit * bin_scale
        
        # Generate signal and background components
        # Implementation depends on fit_type
        if fit_type == 'gaussian_fixed_mean' or fit_type == 'gaussian_free_mean':
            # Extract parameters based on fit type
            if fit_type == 'gaussian_fixed_mean':
                # [signal_yield, background_yield, sigma, a, b]
                sig_yield, bkg_yield, sigma = fit_params[:3]
                a, b = fit_params[3:]
            else:
                # [signal_yield, background_yield, mean, sigma, a, b]
                sig_yield, bkg_yield, mean, sigma = fit_params[:4]
                a, b = fit_params[4:]
            
            # Signal component
            signal = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_fit - mean) / sigma) ** 2)
            signal_norm = np.sum(signal)
            signal_scaled = signal * (sig_yield * bin_scale / signal_norm) if signal_norm > 0 else signal * 0
            
            # Background component
            background = a + b * (x_fit - mean)
            background_norm = np.sum(background)
            background_scaled = background * (bkg_yield * bin_scale / background_norm) if background_norm > 0 else background * 0
            
        elif fit_type == 'voigtian':
            # [signal_yield, background_yield, mean, sigma, gamma, a, b]
            sig_yield, bkg_yield, mean, sigma, gamma = fit_params[:5]
            a, b = fit_params[5:]
            
            from scipy.special import voigt_profile
            
            # Signal component
            signal = voigt_profile(x_fit - mean, sigma, gamma)
            signal_norm = np.sum(signal)
            signal_scaled = signal * (sig_yield * bin_scale / signal_norm) if signal_norm > 0 else signal * 0
            
            # Background component
            background = a + b * (x_fit - mean)
            background_norm = np.sum(background)
            background_scaled = background * (bkg_yield * bin_scale / background_norm) if background_norm > 0 else background * 0
            
        else:
            # Default case
            signal_scaled = np.zeros_like(x_fit)
            background_scaled = y_fit_scaled
        
        # Plot fit components
        plt.plot(x_fit, y_fit_scaled, 'r-', linewidth=2, label='Total Fit')
        plt.plot(x_fit, signal_scaled, 'g--', linewidth=2, label='Signal')
        plt.plot(x_fit, background_scaled, 'b--', linewidth=2, label='Background')
        
        # Add J/ψ mass line
        plt.axvline(x=3096.9, color='r', linestyle='--', label='J/ψ mass')
        
        # Add labels and title
        plt.xlabel('M(pK⁻Λ̄) [MeV/c²]', fontsize=12)
        plt.ylabel(f'Candidates / ({bin_width:.1f} MeV/c²)', fontsize=12)
        plt.title(f'pK⁻Λ̄ Mass Spectrum with J/ψ Fit ({fit_type})', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=10, loc='upper right')
        
        # Add text with fit results
        if fit_type == 'gaussian_fixed_mean':
            text = f"Signal yield = {fit_params[0]:.1f} ± {np.sqrt(fit_params[0]) if fit_params[0] > 0 else 0:.1f}\n"
            text += f"Mean = 3096.9 MeV/c² (fixed)\n"
            text += f"Width = {fit_params[2]:.1f} MeV/c²"
        elif fit_type == 'gaussian_free_mean':
            text = f"Signal yield = {fit_params[0]:.1f} ± {np.sqrt(fit_params[0]) if fit_params[0] > 0 else 0:.1f}\n"
            text += f"Mean = {fit_params[2]:.1f} MeV/c²\n"
            text += f"Width = {fit_params[3]:.1f} MeV/c²"
        elif fit_type == 'voigtian':
            text = f"Signal yield = {fit_params[0]:.1f} ± {np.sqrt(fit_params[0]) if fit_params[0] > 0 else 0:.1f}\n"
            text += f"Mean = {fit_params[2]:.1f} MeV/c²\n"
            text += f"Gaussian σ = {fit_params[3]:.1f} MeV/c²\n"
            text += f"Lorentzian γ = {fit_params[4]:.1f} MeV/c²"
        
        plt.text(0.65, 0.7, text, transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
        
        # Save plot
        plt.tight_layout()
        plot_path = self.output_dir / f"jpsi_fit_{fit_type}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        self.logger.info(f"Fit plot ({fit_type}) saved to {plot_path}")