#!/usr/bin/env python3
"""
Study: ηc(2S) vs ψ(2S) Discrimination in M(pK⁻Λ̄) Spectrum

This study investigates whether the observed peak around 3640-3690 MeV
corresponds to ηc(2S) at 3637.8 MeV or ψ(2S) at 3686.1 MeV.

Performs both single-peak and double-peak fits with statistical comparison.
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
from scipy.optimize import curve_fit
from scipy.stats import chi2, f as f_dist
import warnings

# Suppress font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data_loader import DataLoader
from selection import SelectionProcessor
from mass_calculator import MassCalculator

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
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('etac2s_vs_psi2s.log')
    ]
)
logger = logging.getLogger(__name__)

# PDG masses (2024)
ETAC_2S_MASS = 3637.8  # MeV
ETAC_2S_ERROR = 1.1    # MeV
PSI_2S_MASS = 3686.10   # MeV
PSI_2S_ERROR = 0.06     # MeV

# Study region
STUDY_REGION = (3550, 3750)  # MeV

# Default data directory (from main.py)
DEFAULT_DATA_DIR = "/share/lazy/Mohamed/Bu2LambdaPPP/files/data"


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Study ηc(2S) vs ψ(2S) discrimination'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f'Directory containing data ROOT files (default: {DEFAULT_DATA_DIR})'
    )
    
    parser.add_argument(
        '--years',
        type=str,
        nargs='+',
        default=['16', '17', '18'],
        help='Data taking years (default: 16 17 18)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory'
    )
    
    parser.add_argument(
        '--bins',
        type=int,
        default=50,
        help='Number of bins in mass histogram'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


# ============================================================================
# FIT MODELS
# ============================================================================

def gaussian(x, amplitude, mean, sigma):
    """Single Gaussian"""
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma)**2)


def gaussian_with_bkg(x, amplitude, mean, sigma, bkg_c0, bkg_c1):
    """Single Gaussian + linear background"""
    return gaussian(x, amplitude, mean, sigma) + bkg_c0 + bkg_c1 * x


def double_gaussian_with_bkg(x, amp1, mean1, sigma1, amp2, mean2, sigma2, bkg_c0, bkg_c1):
    """Double Gaussian + linear background"""
    gauss1 = gaussian(x, amp1, mean1, sigma1)
    gauss2 = gaussian(x, amp2, mean2, sigma2)
    return gauss1 + gauss2 + bkg_c0 + bkg_c1 * x


# ============================================================================
# FITTING FUNCTIONS
# ============================================================================

def fit_single_peak(bin_centers, counts, errors):
    """Fit single Gaussian + linear background
    
    Significance is calculated using:
    S = N_sig / sqrt(N_sig + N_bkg)
    
    where N_sig is the signal yield (integral of Gaussian)
    and N_bkg is the background in ±2.5σ window
    
    For a 5σ discovery: Need N_sig ~ 25 + sqrt(N_bkg)
    For a 3σ evidence: Need N_sig ~ 9 + sqrt(N_bkg)
    """
    logger.info("Fitting single peak model...")
    
    # Initial guesses
    max_idx = np.argmax(counts)
    max_count = counts[max_idx]
    max_pos = bin_centers[max_idx]
    
    p0 = [
        max_count * 0.5,  # amplitude
        max_pos,          # mean
        10.0,             # sigma (reasonable detector resolution)
        np.min(counts),   # background constant
        0.0               # background slope
    ]
    
    # Bounds
    bounds = (
        [0, STUDY_REGION[0], 1, 0, -1],  # lower
        [np.inf, STUDY_REGION[1], 50, np.inf, 1]  # upper
    )
    
    try:
        popt, pcov = curve_fit(
            gaussian_with_bkg,
            bin_centers,
            counts,
            p0=p0,
            sigma=errors,
            absolute_sigma=True,
            bounds=bounds,
            maxfev=10000
        )
        
        perr = np.sqrt(np.diag(pcov))
        
        # Calculate chi-square
        fitted = gaussian_with_bkg(bin_centers, *popt)
        chi_sq = np.sum(((counts - fitted) / errors)**2)
        ndf = len(counts) - len(popt)
        p_value = 1 - chi2.cdf(chi_sq, ndf)
        
        results = {
            'amplitude': (popt[0], perr[0]),
            'mean': (popt[1], perr[1]),
            'sigma': (popt[2], perr[2]),
            'bkg_c0': (popt[3], perr[3]),
            'bkg_c1': (popt[4], perr[4]),
            'chi_square': chi_sq,
            'ndf': ndf,
            'p_value': p_value,
            'params': popt,
            'errors': perr,
            'covariance': pcov,
            'fit_function': lambda x: gaussian_with_bkg(x, *popt),
            'signal_function': lambda x: gaussian(x, popt[0], popt[1], popt[2]),
            'bkg_function': lambda x: popt[3] + popt[4] * x
        }
        
        # Calculate yield (integral of Gaussian)
        yield_val = popt[0] * popt[2] * np.sqrt(2 * np.pi)
        # Error propagation
        yield_err = yield_val * np.sqrt((perr[0]/popt[0])**2 + (perr[2]/popt[2])**2)
        results['yield'] = (yield_val, yield_err)
        
        # Significance
        background_in_window = (popt[3] + popt[4] * popt[1]) * 2.5 * popt[2]  # ±2.5σ window
        if background_in_window > 0:
            significance = yield_val / np.sqrt(yield_val + background_in_window)
            results['significance'] = significance
        else:
            results['significance'] = 0
        
        logger.info(f"  Mean: {popt[1]:.2f} ± {perr[1]:.2f} MeV")
        logger.info(f"  Sigma: {popt[2]:.2f} ± {perr[2]:.2f} MeV")
        logger.info(f"  Yield: {yield_val:.1f} ± {yield_err:.1f}")
        logger.info(f"  Significance: {results['significance']:.1f}σ")
        logger.info(f"  χ²/ndf: {chi_sq:.2f}/{ndf} = {chi_sq/ndf:.2f} (p = {p_value:.4f})")
        
        return results
        
    except Exception as e:
        logger.error(f"Single peak fit failed: {e}")
        return None


def fit_double_peak(bin_centers, counts, errors):
    """Fit double Gaussian + linear background"""
    logger.info("Fitting double peak model...")
    
    # Initial guesses based on expected positions
    max_count = np.max(counts)
    
    p0 = [
        max_count * 0.3,   # amp1
        ETAC_2S_MASS,      # mean1 (ηc(2S))
        8.0,               # sigma1
        max_count * 0.2,   # amp2
        PSI_2S_MASS,       # mean2 (ψ(2S))
        8.0,               # sigma2
        np.min(counts),    # background constant
        0.0                # background slope
    ]
    
    # Bounds
    bounds = (
        [0, 3600, 2, 0, 3660, 2, 0, -1],  # lower
        [np.inf, 3660, 30, np.inf, 3720, 30, np.inf, 1]  # upper
    )
    
    try:
        popt, pcov = curve_fit(
            double_gaussian_with_bkg,
            bin_centers,
            counts,
            p0=p0,
            sigma=errors,
            absolute_sigma=True,
            bounds=bounds,
            maxfev=20000
        )
        
        perr = np.sqrt(np.diag(pcov))
        
        # Calculate chi-square
        fitted = double_gaussian_with_bkg(bin_centers, *popt)
        chi_sq = np.sum(((counts - fitted) / errors)**2)
        ndf = len(counts) - len(popt)
        p_value = 1 - chi2.cdf(chi_sq, ndf)
        
        results = {
            'amp1': (popt[0], perr[0]),
            'mean1': (popt[1], perr[1]),
            'sigma1': (popt[2], perr[2]),
            'amp2': (popt[3], perr[3]),
            'mean2': (popt[4], perr[4]),
            'sigma2': (popt[5], perr[5]),
            'bkg_c0': (popt[6], perr[6]),
            'bkg_c1': (popt[7], perr[7]),
            'chi_square': chi_sq,
            'ndf': ndf,
            'p_value': p_value,
            'params': popt,
            'errors': perr,
            'covariance': pcov,
            'fit_function': lambda x: double_gaussian_with_bkg(x, *popt),
            'signal1_function': lambda x: gaussian(x, popt[0], popt[1], popt[2]),
            'signal2_function': lambda x: gaussian(x, popt[3], popt[4], popt[5]),
            'bkg_function': lambda x: popt[6] + popt[7] * x
        }
        
        # Calculate yields
        yield1 = popt[0] * popt[2] * np.sqrt(2 * np.pi)
        yield1_err = yield1 * np.sqrt((perr[0]/popt[0])**2 + (perr[2]/popt[2])**2)
        results['yield1'] = (yield1, yield1_err)
        
        yield2 = popt[3] * popt[5] * np.sqrt(2 * np.pi)
        yield2_err = yield2 * np.sqrt((perr[3]/popt[3])**2 + (perr[5]/popt[5])**2)
        results['yield2'] = (yield2, yield2_err)
        
        # Significances
        bkg1 = (popt[6] + popt[7] * popt[1]) * 2.5 * popt[2]
        if bkg1 > 0 and yield1 > 0:
            results['significance1'] = yield1 / np.sqrt(yield1 + bkg1)
        else:
            results['significance1'] = 0
            
        bkg2 = (popt[6] + popt[7] * popt[4]) * 2.5 * popt[5]
        if bkg2 > 0 and yield2 > 0:
            results['significance2'] = yield2 / np.sqrt(yield2 + bkg2)
        else:
            results['significance2'] = 0
        
        logger.info(f"  Peak 1: {popt[1]:.2f} ± {perr[1]:.2f} MeV, σ = {popt[2]:.2f} ± {perr[2]:.2f} MeV")
        logger.info(f"    Yield: {yield1:.1f} ± {yield1_err:.1f}, Significance: {results['significance1']:.1f}σ")
        logger.info(f"  Peak 2: {popt[4]:.2f} ± {perr[4]:.2f} MeV, σ = {popt[5]:.2f} ± {perr[5]:.2f} MeV")
        logger.info(f"    Yield: {yield2:.1f} ± {yield2_err:.1f}, Significance: {results['significance2']:.1f}σ")
        logger.info(f"  χ²/ndf: {chi_sq:.2f}/{ndf} = {chi_sq/ndf:.2f} (p = {p_value:.4f})")
        
        return results
        
    except Exception as e:
        logger.error(f"Double peak fit failed: {e}")
        return None


def compare_models(single_fit, double_fit, bin_centers):
    """Compare single vs double peak models using F-test"""
    if not single_fit or not double_fit:
        return None
    
    chi2_single = single_fit['chi_square']
    ndf_single = single_fit['ndf']
    
    chi2_double = double_fit['chi_square']
    ndf_double = double_fit['ndf']
    
    # F-test for nested models
    delta_chi2 = chi2_single - chi2_double
    delta_ndf = ndf_single - ndf_double
    
    if delta_ndf <= 0 or delta_chi2 < 0:
        logger.warning("Cannot perform F-test: invalid chi-square difference")
        return None
    
    F_stat = (delta_chi2 / delta_ndf) / (chi2_double / ndf_double)
    p_value = 1 - f_dist.cdf(F_stat, delta_ndf, ndf_double)
    
    logger.info("\n" + "="*70)
    logger.info("MODEL COMPARISON (F-test)")
    logger.info("="*70)
    logger.info(f"Single peak:  χ² = {chi2_single:.2f}, ndf = {ndf_single}")
    logger.info(f"Double peak:  χ² = {chi2_double:.2f}, ndf = {ndf_double}")
    logger.info(f"Δχ² = {delta_chi2:.2f}, Δndf = {delta_ndf}")
    logger.info(f"F-statistic = {F_stat:.3f}")
    logger.info(f"p-value = {p_value:.4f}")
    
    if p_value < 0.05:
        logger.info("➤ Double peak model is significantly better (p < 0.05)")
    else:
        logger.info("➤ Single peak model is preferred (simpler model)")
    
    return {
        'F_statistic': F_stat,
        'p_value': p_value,
        'delta_chi2': delta_chi2,
        'delta_ndf': delta_ndf
    }


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_fit_with_residuals(bin_centers, counts, errors, fit_result, title, filename):
    """Create plot with fit and residuals using LHCb style with proper subplots"""
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    
    ax_main = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_main)
    
    # Main plot - Data points with LHCb style
    ax_main.errorbar(bin_centers, counts, yerr=errors, fmt='o', 
                     color='black', markersize=5,
                     markerfacecolor='black', markeredgewidth=0,
                     ecolor='black', elinewidth=1.2, capsize=2.5,
                     capthick=1.2, label='Data', zorder=10)
    
    x_fine = np.linspace(STUDY_REGION[0], STUDY_REGION[1], 500)
    
    # Full fit
    ax_main.plot(x_fine, fit_result['fit_function'](x_fine), 
                'r-', linewidth=2.5, label='Total fit', zorder=8)
    
    # Background
    ax_main.plot(x_fine, fit_result['bkg_function'](x_fine),
                'b--', linewidth=2, label='Background', zorder=7, alpha=0.8)
    
    # Signal components
    if 'signal_function' in fit_result:
        # Single peak
        ax_main.plot(x_fine, fit_result['signal_function'](x_fine),
                    'g--', linewidth=2, label='Signal', zorder=7, alpha=0.8)
        
        mean, mean_err = fit_result['mean']
        sigma, sigma_err = fit_result['sigma']
        yield_val, yield_err = fit_result['yield']
        chi2_ndf = fit_result['chi_square'] / fit_result['ndf']
        
        info_text = (f"Mean: ${mean:.1f} \\pm {mean_err:.1f}$ MeV\n"
                    f"$\\sigma$: ${sigma:.1f} \\pm {sigma_err:.1f}$ MeV\n"
                    f"Yield: ${yield_val:.0f} \\pm {yield_err:.0f}$\n"
                    f"Sig.: ${fit_result['significance']:.1f}\\sigma$\n"
                    f"$\\chi^2$/ndf: ${chi2_ndf:.2f}$")
        
    else:
        # Double peak
        ax_main.plot(x_fine, fit_result['signal1_function'](x_fine),
                    'g--', linewidth=2, label='Peak 1', zorder=7, alpha=0.8)
        ax_main.plot(x_fine, fit_result['signal2_function'](x_fine),
                    'm--', linewidth=2, label='Peak 2', zorder=7, alpha=0.8)
        
        mean1, mean1_err = fit_result['mean1']
        mean2, mean2_err = fit_result['mean2']
        yield1, yield1_err = fit_result['yield1']
        yield2, yield2_err = fit_result['yield2']
        chi2_ndf = fit_result['chi_square'] / fit_result['ndf']
        
        info_text = (f"Peak 1: ${mean1:.1f} \\pm {mean1_err:.1f}$ MeV\n"
                    f"  Yield: ${yield1:.0f} \\pm {yield1_err:.0f}$ ({fit_result['significance1']:.1f}$\\sigma$)\n"
                    f"Peak 2: ${mean2:.1f} \\pm {mean2_err:.1f}$ MeV\n"
                    f"  Yield: ${yield2:.0f} \\pm {yield2_err:.0f}$ ({fit_result['significance2']:.1f}$\\sigma$)\n"
                    f"$\\chi^2$/ndf: ${chi2_ndf:.2f}$")
    
    # PDG lines
    ax_main.axvline(ETAC_2S_MASS, color='orange', linestyle=':', 
                   linewidth=2, alpha=0.7, label=f'$\\eta_c(2S)$ PDG')
    ax_main.axvline(PSI_2S_MASS, color='purple', linestyle=':', 
                   linewidth=2, alpha=0.7, label='$\\psi(2S)$ PDG')
    
    # Add info box
    ax_main.text(0.97, 0.97, info_text,
                transform=ax_main.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black'),
                fontsize=10,
                family='monospace')
    
    # Add LHCb label
    hep.lhcb.text(loc=1, ax=ax_main, fontsize=14)
    
    ax_main.set_ylabel('Counts', fontsize=14, fontweight='bold')
    ax_main.set_title(title, fontsize=16, fontweight='bold', pad=10)
    ax_main.legend(fontsize=10, loc='upper left', frameon=True, fancybox=False, 
                  edgecolor='black', framealpha=0.95)
    ax_main.set_ylim(bottom=0)
    ax_main.tick_params(labelbottom=False, direction='in', which='both')
    ax_main.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    # Residuals plot (Pull)
    fitted_vals = fit_result['fit_function'](bin_centers)
    residuals = (counts - fitted_vals) / errors
    
    ax_res.axhline(0, color='red', linestyle='-', linewidth=1.5, zorder=5)
    ax_res.axhline(2, color='gray', linestyle='--', linewidth=1, alpha=0.6, zorder=3)
    ax_res.axhline(-2, color='gray', linestyle='--', linewidth=1, alpha=0.6, zorder=3)
    ax_res.axhspan(-2, 2, alpha=0.1, color='gray', zorder=1)
    
    ax_res.errorbar(bin_centers, residuals, yerr=1.0, fmt='o', 
                   color='black', markersize=4, capsize=2, elinewidth=1,
                   markerfacecolor='black', markeredgewidth=0)
    
    ax_res.set_xlabel('$M(pK^-\\bar{\\Lambda})$ [MeV]', fontsize=14, fontweight='bold')
    ax_res.set_ylabel('Pull', fontsize=12, fontweight='bold')
    ax_res.set_ylim(-4.5, 4.5)
    ax_res.set_xlim(STUDY_REGION)
    ax_res.tick_params(direction='in', which='both')
    ax_res.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    # Add minor ticks
    ax_main.minorticks_on()
    ax_res.minorticks_on()
    
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot: {filename}")


def create_comparison_plot(bin_centers, counts, errors, single_fit, double_fit, filename):
    """Create side-by-side comparison of single and double peak fits with residuals"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.05, wspace=0.25)
    
    x_fine = np.linspace(STUDY_REGION[0], STUDY_REGION[1], 500)
    
    # Single peak model
    ax1_main = fig.add_subplot(gs[0, 0])
    ax1_res = fig.add_subplot(gs[1, 0], sharex=ax1_main)
    
    # Double peak model
    ax2_main = fig.add_subplot(gs[0, 1])
    ax2_res = fig.add_subplot(gs[1, 1], sharex=ax2_main)
    
    for (ax_main, ax_res, fit_result, model_name) in [
        (ax1_main, ax1_res, single_fit, 'Single Peak Model'),
        (ax2_main, ax2_res, double_fit, 'Double Peak Model')
    ]:
        # Data points
        ax_main.errorbar(bin_centers, counts, yerr=errors, fmt='o', 
                   color='black', markersize=5,
                   markerfacecolor='black', markeredgewidth=0,
                   ecolor='black', elinewidth=1.2, capsize=2.5,
                   capthick=1.2, label='Data', zorder=10)
        
        # Fits
        ax_main.plot(x_fine, fit_result['fit_function'](x_fine), 
               'r-', linewidth=2.5, label='Total fit', zorder=8)
        ax_main.plot(x_fine, fit_result['bkg_function'](x_fine),
               'b--', linewidth=2, label='Background', zorder=7, alpha=0.8)
        
        # Signal components
        if 'signal_function' in fit_result:
            ax_main.plot(x_fine, fit_result['signal_function'](x_fine),
                       'g--', linewidth=2, label='Signal', zorder=7, alpha=0.8)
        else:
            ax_main.plot(x_fine, fit_result['signal1_function'](x_fine),
                       'g--', linewidth=2, label='Peak 1', zorder=7, alpha=0.8)
            ax_main.plot(x_fine, fit_result['signal2_function'](x_fine),
                       'm--', linewidth=2, label='Peak 2', zorder=7, alpha=0.8)
        
        # PDG lines
        ax_main.axvline(ETAC_2S_MASS, color='orange', linestyle=':', 
                  linewidth=2, alpha=0.7, label='$\\eta_c(2S)$ PDG')
        ax_main.axvline(PSI_2S_MASS, color='purple', linestyle=':', 
                  linewidth=2, alpha=0.7, label='$\\psi(2S)$ PDG')
        
        # Add LHCb label
        hep.lhcb.text(loc=1, ax=ax_main, fontsize=12)
        
        ax_main.set_ylabel('Counts', fontsize=14, fontweight='bold')
        ax_main.set_title(model_name, fontsize=16, fontweight='bold', pad=10)
        ax_main.legend(fontsize=9, loc='upper left', frameon=True, 
                      fancybox=False, edgecolor='black', framealpha=0.95)
        ax_main.set_xlim(STUDY_REGION)
        ax_main.set_ylim(bottom=0)
        ax_main.tick_params(labelbottom=False, direction='in', which='both')
        ax_main.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
        ax_main.minorticks_on()
        
        # Residuals
        fitted_vals = fit_result['fit_function'](bin_centers)
        residuals = (counts - fitted_vals) / errors
        
        ax_res.axhline(0, color='red', linestyle='-', linewidth=1.5, zorder=5)
        ax_res.axhline(2, color='gray', linestyle='--', linewidth=1, alpha=0.6, zorder=3)
        ax_res.axhline(-2, color='gray', linestyle='--', linewidth=1, alpha=0.6, zorder=3)
        ax_res.axhspan(-2, 2, alpha=0.1, color='gray', zorder=1)
        
        ax_res.errorbar(bin_centers, residuals, yerr=1.0, fmt='o', 
                       color='black', markersize=4, capsize=2, elinewidth=1,
                       markerfacecolor='black', markeredgewidth=0)
        
        ax_res.set_xlabel('$M(pK^-\\bar{\\Lambda})$ [MeV]', fontsize=14, fontweight='bold')
        ax_res.set_ylabel('Pull', fontsize=12, fontweight='bold')
        ax_res.set_ylim(-4.5, 4.5)
        ax_res.set_xlim(STUDY_REGION)
        ax_res.tick_params(direction='in', which='both')
        ax_res.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
        ax_res.minorticks_on()
    
    # Add chi2 info as text
    chi2_ndf1 = single_fit['chi_square'] / single_fit['ndf']
    chi2_ndf2 = double_fit['chi_square'] / double_fit['ndf']
    
    fig.text(0.5, 0.02, 
             f"Single peak: $\\chi^2$/ndf = {chi2_ndf1:.2f}  |  "
             f"Double peak: $\\chi^2$/ndf = {chi2_ndf2:.2f}",
             ha='center', fontsize=13, fontweight='bold')
    
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comparison plot: {filename}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Main analysis function"""
    args = parse_arguments()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("ηc(2S) vs ψ(2S) DISCRIMINATION STUDY")
    logger.info("="*70)
    logger.info(f"Study region: {STUDY_REGION[0]}-{STUDY_REGION[1]} MeV")
    logger.info(f"ηc(2S) mass (PDG): {ETAC_2S_MASS} ± {ETAC_2S_ERROR} MeV")
    logger.info(f"ψ(2S) mass (PDG):  {PSI_2S_MASS} ± {PSI_2S_ERROR} MeV")
    logger.info(f"Separation:        {PSI_2S_MASS - ETAC_2S_MASS:.1f} MeV")
    logger.info("="*70)
    logger.info("")
    
    # Load data
    logger.info("Loading data...")
    loader = DataLoader(data_dir=args.data_dir)
    
    data = {}
    for year in args.years:
        for polarity in ['MD', 'MU']:
            for track_type in ['LL', 'DD']:
                key = f"{year}_{polarity}_{track_type}"
                file_path = Path(args.data_dir) / f"dataBu2L0barPHH_{year}{polarity}.root"
                
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                    
                try:
                    import uproot
                    with uproot.open(file_path) as file:
                        tree_name = f"B2L0barPKpKm_{track_type}/DecayTree"
                        if tree_name.split('/')[0] in file:
                            tree = file[tree_name]
                            data[key] = tree.arrays()
                            logger.info(f"  Loaded {key}: {len(data[key])} events")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
    
    if not data:
        logger.error("No data loaded! Check data directory and file paths.")
        return
    
    # Apply selection
    logger.info("Applying selection...")
    selector = SelectionProcessor()
    selected_data = selector.apply_basic_selection(data)
    
    total_events = sum(len(events) for events in selected_data.values())
    logger.info(f"After selection: {total_events} events")
    
    if total_events == 0:
        logger.error("No events after selection!")
        return
    
    # Calculate invariant masses
    logger.info("Calculating invariant masses...")
    mass_calc = MassCalculator()
    data_with_masses = mass_calc.calculate_jpsi_candidates(selected_data)
    
    # Extract masses in study region
    all_masses = []
    for key, events in data_with_masses.items():
        if 'M_pKLambdabar' in events.fields:
            masses = ak.to_numpy(events['M_pKLambdabar'])
            mask = (masses >= STUDY_REGION[0]) & (masses <= STUDY_REGION[1])
            all_masses.append(masses[mask])
    
    if len(all_masses) == 0:
        logger.error("No masses in study region!")
        return
    
    all_masses = np.concatenate(all_masses)
    logger.info(f"Events in study region: {len(all_masses)}")
    
    # Create histogram
    counts, bin_edges = np.histogram(all_masses, bins=args.bins, 
                                     range=STUDY_REGION)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    errors = np.sqrt(counts)
    errors[errors == 0] = 1.0  # Avoid division by zero
    
    # Perform fits
    logger.info("\n" + "="*70)
    logger.info("FITTING")
    logger.info("="*70)
    
    single_fit = fit_single_peak(bin_centers, counts, errors)
    double_fit = fit_double_peak(bin_centers, counts, errors)
    
    if not single_fit or not double_fit:
        logger.error("Fitting failed!")
        return
    
    # Compare models
    comparison = compare_models(single_fit, double_fit, bin_centers)
    
    # Create plots
    logger.info("\n" + "="*70)
    logger.info("CREATING PLOTS")
    logger.info("="*70)
    
    plot_fit_with_residuals(
        bin_centers, counts, errors, single_fit,
        "Single Peak Fit: $\\eta_c(2S)$ Hypothesis",
        output_dir / "single_peak_fit.pdf"
    )
    
    plot_fit_with_residuals(
        bin_centers, counts, errors, double_fit,
        "Double Peak Fit: $\\eta_c(2S)$ + $\\psi(2S)$ Hypothesis",
        output_dir / "double_peak_fit.pdf"
    )
    
    create_comparison_plot(
        bin_centers, counts, errors, single_fit, double_fit,
        output_dir / "model_comparison.pdf"
    )
    
    # Write results
    logger.info("\n" + "="*70)
    logger.info("WRITING RESULTS")
    logger.info("="*70)
    
    results_file = output_dir / "fit_results.txt"
    with open(results_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ηc(2S) vs ψ(2S) DISCRIMINATION STUDY - RESULTS\n")
        f.write("="*70 + "\n\n")
        
        # Add explanation of significance
        f.write("SIGNIFICANCE CALCULATION:\n")
        f.write("-" * 70 + "\n")
        f.write("Statistical significance is calculated using:\n")
        f.write("  S = N_sig / sqrt(N_sig + N_bkg)\n\n")
        f.write("where N_sig is the signal yield (integral of Gaussian fit)\n")
        f.write("and N_bkg is the estimated background in ±2.5σ window.\n\n")
        f.write("Discovery thresholds (particle physics convention):\n")
        f.write("  • 5σ (discovery):  < 0.00006% probability of fluctuation\n")
        f.write("  • 3σ (evidence):   < 0.3% probability of fluctuation\n") 
        f.write("  • 2σ (hint):       < 5% probability of fluctuation\n\n")
        f.write("For a 5σ discovery, approximately:\n")
        f.write("  N_sig ≈ 25 + sqrt(N_bkg)  events are needed\n")
        f.write("For a 3σ evidence:\n")
        f.write("  N_sig ≈ 9 + sqrt(N_bkg)   events are needed\n")
        f.write("="*70 + "\n\n")
        
        # Single peak results
        f.write("SINGLE PEAK FIT:\n")
        f.write(f"  Mean:  {single_fit['mean'][0]:.3f} ± {single_fit['mean'][1]:.3f} MeV\n")
        f.write(f"  Sigma: {single_fit['sigma'][0]:.3f} ± {single_fit['sigma'][1]:.3f} MeV\n")
        f.write(f"  Yield: {single_fit['yield'][0]:.1f} ± {single_fit['yield'][1]:.1f}\n")
        f.write(f"  Significance: {single_fit['significance']:.1f}σ\n")
        f.write(f"  χ²/ndf: {single_fit['chi_square']:.3f}/{single_fit['ndf']} = "
               f"{single_fit['chi_square']/single_fit['ndf']:.3f} "
               f"(p = {single_fit['p_value']:.4f})\n\n")
        
        # Comparison with PDG
        f.write("  Comparison with PDG:\n")
        diff_etac = single_fit['mean'][0] - ETAC_2S_MASS
        sigma_diff = abs(diff_etac) / np.sqrt(single_fit['mean'][1]**2 + ETAC_2S_ERROR**2)
        f.write(f"    vs ηc(2S): Δm = {diff_etac:+.1f} MeV ({sigma_diff:.1f}σ)\n")
        
        diff_psi = single_fit['mean'][0] - PSI_2S_MASS
        sigma_diff_psi = abs(diff_psi) / np.sqrt(single_fit['mean'][1]**2 + PSI_2S_ERROR**2)
        f.write(f"    vs ψ(2S):  Δm = {diff_psi:+.1f} MeV ({sigma_diff_psi:.1f}σ)\n\n")
        
        # Double peak results
        f.write("DOUBLE PEAK FIT:\n")
        f.write(f"  Peak 1 Mean:  {double_fit['mean1'][0]:.3f} ± {double_fit['mean1'][1]:.3f} MeV\n")
        f.write(f"  Peak 1 Sigma: {double_fit['sigma1'][0]:.3f} ± {double_fit['sigma1'][1]:.3f} MeV\n")
        f.write(f"  Peak 1 Yield: {double_fit['yield1'][0]:.1f} ± {double_fit['yield1'][1]:.1f}\n")
        f.write(f"  Peak 1 Significance: {double_fit['significance1']:.1f}σ\n\n")
        
        f.write(f"  Peak 2 Mean:  {double_fit['mean2'][0]:.3f} ± {double_fit['mean2'][1]:.3f} MeV\n")
        f.write(f"  Peak 2 Sigma: {double_fit['sigma2'][0]:.3f} ± {double_fit['sigma2'][1]:.3f} MeV\n")
        f.write(f"  Peak 2 Yield: {double_fit['yield2'][0]:.1f} ± {double_fit['yield2'][1]:.1f}\n")
        f.write(f"  Peak 2 Significance: {double_fit['significance2']:.1f}σ\n\n")
        
        f.write(f"  χ²/ndf: {double_fit['chi_square']:.3f}/{double_fit['ndf']} = "
               f"{double_fit['chi_square']/double_fit['ndf']:.3f} "
               f"(p = {double_fit['p_value']:.4f})\n\n")
        
        # Model comparison
        if comparison:
            f.write("MODEL COMPARISON (F-test):\n")
            f.write(f"  F-statistic: {comparison['F_statistic']:.3f}\n")
            f.write(f"  p-value: {comparison['p_value']:.4f}\n")
            f.write(f"  Δχ²: {comparison['delta_chi2']:.2f}\n")
            f.write(f"  Δndf: {comparison['delta_ndf']}\n\n")
            
            if comparison['p_value'] < 0.05:
                f.write("  ➤ Double peak model is statistically preferred (p < 0.05)\n\n")
            else:
                f.write("  ➤ Single peak model is preferred (simpler, adequate fit)\n\n")
        
        # Conclusion
        f.write("="*70 + "\n")
        f.write("CONCLUSION:\n")
        f.write("="*70 + "\n")
        
        mean_val = single_fit['mean'][0]
        if abs(mean_val - ETAC_2S_MASS) < abs(mean_val - PSI_2S_MASS):
            f.write(f"The observed peak at {mean_val:.1f} MeV is more consistent with\n")
            f.write(f"ηc(2S) at {ETAC_2S_MASS} MeV than ψ(2S) at {PSI_2S_MASS} MeV.\n\n")
        else:
            f.write(f"The observed peak at {mean_val:.1f} MeV is more consistent with\n")
            f.write(f"ψ(2S) at {PSI_2S_MASS} MeV than ηc(2S) at {ETAC_2S_MASS} MeV.\n\n")
        
        if double_fit['significance2'] < 3.0:
            f.write(f"The second peak has low significance ({double_fit['significance2']:.1f}σ)\n")
            f.write("and may be a statistical fluctuation.\n")
    
    logger.info(f"Results written to: {results_file}")
    
    logger.info("\n" + "="*70)
    logger.info("STUDY COMPLETE")
    logger.info("="*70)
    logger.info(f"Output directory: {output_dir}")
    logger.info("Generated files:")
    logger.info("  - single_peak_fit.pdf")
    logger.info("  - double_peak_fit.pdf")
    logger.info("  - model_comparison.pdf")
    logger.info("  - fit_results.txt")
    logger.info("="*70)


if __name__ == "__main__":
    main()
