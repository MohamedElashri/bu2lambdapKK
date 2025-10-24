#!/usr/bin/env python3
"""
Main control script for B+ → pK⁻Λ̄ K+ analysis
"""

import argparse
import logging
from pathlib import Path

from data_loader import DataLoader
from selection import SelectionProcessor
from mass_calculator import MassCalculator
from fitter import JpsiPeakFitter
from efficiency import EfficiencyCalculator
from plotter import MassSpectrumPlotter

def setup_logging(verbose=False):
    """Configure logging level"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("Bu2LambdaPKK")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analysis of B+ → pK⁻Λ̄ K+ decay")
    parser.add_argument("--data-dir", default="/share/lazy/Mohamed/Bu2LambdaPPP/RD/restripped.data/reduced", 
                        help="Directory containing data files")
    parser.add_argument("--mc-dir", default=None, help="Directory containing MC files")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--years", default="16,17,18", help="Years to process (comma-separated)")
    parser.add_argument("--polarity", default="MD,MU", help="Magnet polarities to process")
    parser.add_argument("--track-types", default="LL,DD", help="Track types to process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def main():
    """Main analysis function"""
    args = parse_args()
    logger = setup_logging(args.verbose)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse arguments
    years = args.years.split(",")
    polarities = args.polarity.split(",")
    track_types = args.track_types.split(",")
    
    logger.info(f"Starting analysis for B+ → pK⁻Λ̄ K+ decay")
    logger.info(f"Processing years: {years}, polarities: {polarities}, track types: {track_types}")
    
    # Step 1: Load data
    logger.info("Loading data...")
    loader = DataLoader(args.data_dir)
    data = loader.load_data(years, polarities, track_types, "B2L0barPKpKm")
    
    # Print branch information to help with development
    if logger.level == logging.DEBUG:
        first_sample = next(iter(data.values()))
        logger.debug(f"Available branches: {first_sample.fields}")
    
    # Step 2: Apply selection
    logger.info("Applying selection criteria...")
    selector = SelectionProcessor()
    selected_data = selector.apply_basic_selection(data)
    
    # Step 3: Calculate invariant masses
    logger.info("Calculating pK⁻Λ̄ invariant mass...")
    mass_calc = MassCalculator()
    data_with_masses = mass_calc.calculate_jpsi_candidates(selected_data)
    
    # Step 4: Create mass plots
    logger.info("Creating mass spectrum plots...")
    plotter = MassSpectrumPlotter(output_dir)
    plotter.plot_jpsi_mass_spectrum(data_with_masses)
    
    # Step 5: Fit J/ψ peak
    logger.info("Fitting J/ψ peak...")
    fitter = JpsiPeakFitter()
    fit_results = fitter.fit_jpsi_peak(data_with_masses)
    logger.info(f"Fit results: {fit_results}")
    
    # Step 6: Calculate efficiency (if MC available)
    total_yield = None
    if args.mc_dir:
        logger.info("Calculating efficiency from MC...")
        eff_calc = EfficiencyCalculator(args.mc_dir)
        efficiency = eff_calc.calculate_efficiency()
        
        # Step 7: Estimate total B+ → J/ψ K+ yield
        logger.info("Estimating total B+ → J/ψ K+ yield...")
        total_yield = eff_calc.estimate_total_jpsi_yield(years)
        logger.info(f"Estimated total B+ → J/ψ K+ yield: {total_yield:.2e}")
    else:
        logger.warning("MC directory not provided, skipping efficiency calculation")
    
    # Step 8: Calculate B(J/ψ → pK⁻Λ̄) branching ratio
    if total_yield and fit_results:
        br_jpsi_to_pklambda = fit_results['signal_yield'] / total_yield
        logger.info(f"B(J/ψ → pK⁻Λ̄) = {br_jpsi_to_pklambda:.2e}")
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()