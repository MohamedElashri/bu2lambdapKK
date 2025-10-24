#!/usr/bin/env python3
"""
Main control script for B+ → pK⁻Λ̄ K+ analysis

This script performs a complete analysis including:
1. Loading data from all years (2016, 2017, 2018), polarities (MD, MU), and track types (LL, DD)
2. Applying selection criteria
3. Calculating pK⁻Λ̄ invariant masses
4. Identifying and plotting all expected resonances (J/ψ, η_c, χ_c0, χ_c1, η_c(2S))
5. Fitting the J/ψ peak
6. Calculating efficiencies (if MC provided)
7. Estimating branching ratios

By default, runs on ALL available data without requiring configuration.

Usage:
    # Run with all defaults (recommended)
    python main.py
    
    # Analyze specific subset
    python main.py --years 16 17 --track-types LL
    
    # Include MC for efficiency calculation
    python main.py --mc-dir /path/to/mc
    
    # Custom output directory
    python main.py --output-dir results/my_analysis
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
    parser = argparse.ArgumentParser(
        description="Analysis of B+ → pK⁻Λ̄ K+ decay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
By default, this script will analyze ALL available data:
  - Years: 2016, 2017, 2018
  - Polarities: MagDown (MD), MagUp (MU)
  - Track types: Long-Long (LL), Downstream (DD)
  - Channel: B2L0barPKpKm

Examples:
  # Analyze all data with defaults
  python main.py
  
  # Analyze only 2016 data
  python main.py --years 16
  
  # Analyze only LL tracks
  python main.py --track-types LL
  
  # Custom output directory
  python main.py --output-dir results/full_analysis
        """
    )
    
    parser.add_argument(
        "--data-dir", 
        default="/share/lazy/Mohamed/Bu2LambdaPPP/RD/restripped.data/reduced", 
        help="Directory containing data files (default: /share/lazy/Mohamed/Bu2LambdaPPP/RD/restripped.data/reduced)"
    )
    
    parser.add_argument(
        "--mc-dir", 
        default=None, 
        help="Directory containing MC files (optional)"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="output", 
        help="Output directory (default: output)"
    )
    
    parser.add_argument(
        "--years",
        nargs='+',
        default=['16', '17', '18'],
        choices=['16', '17', '18'],
        help="Years to process (default: 16 17 18)"
    )
    
    parser.add_argument(
        "--polarities",
        nargs='+',
        default=['MD', 'MU'],
        choices=['MD', 'MU'],
        help="Magnet polarities (default: MD MU)"
    )
    
    parser.add_argument(
        "--track-types",
        nargs='+',
        default=['LL', 'DD'],
        choices=['LL', 'DD'],
        help="Track types (default: LL DD)"
    )
    
    parser.add_argument(
        "--channel",
        default="B2L0barPKpKm",
        help="Decay channel name (default: B2L0barPKpKm)"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def main():
    """Main analysis function"""
    args = parse_args()
    logger = setup_logging(args.verbose)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("B+ → pK⁻Λ̄ K+ Analysis")
    logger.info("="*70)
    logger.info(f"Configuration:")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  Years: {args.years}")
    logger.info(f"  Polarities: {args.polarities}")
    logger.info(f"  Track types: {args.track_types}")
    logger.info(f"  Channel: {args.channel}")
    logger.info(f"  Output: {output_dir}")
    logger.info("="*70)
    
    # Step 1: Load data
    logger.info("\nStep 1: Loading data...")
    loader = DataLoader(data_dir=args.data_dir)
    data = loader.load_data(
        years=args.years,
        polarities=args.polarities,
        track_types=args.track_types,
        channel_name=args.channel
    )
    
    if not data:
        logger.error("No data loaded! Exiting.")
        return
    
    # Log statistics
    total_events = sum(len(events) for events in data.values())
    logger.info(f"Loaded {len(data)} dataset(s) with {total_events} total events")
    for key, events in data.items():
        logger.info(f"  {key}: {len(events)} events")
    
    # Print branch information in debug mode
    if logger.level == logging.DEBUG:
        first_sample = next(iter(data.values()))
        logger.debug(f"Available branches: {first_sample.fields}")
    
    # Step 2: Apply selection
    logger.info("\nStep 2: Applying selection criteria...")
    selector = SelectionProcessor()
    selected_data = selector.apply_basic_selection(data)
    
    selected_events = sum(len(events) for events in selected_data.values())
    logger.info(f"After selection: {selected_events} events ({100*selected_events/total_events:.2f}%)")
    
    # Step 3: Calculate invariant masses
    logger.info("\nStep 3: Calculating pK⁻Λ̄ invariant mass...")
    mass_calc = MassCalculator()
    data_with_masses = mass_calc.calculate_jpsi_candidates(selected_data)
    
    # Step 4: Identify and plot resonances
    logger.info("\nStep 4: Analyzing resonances...")
    plotter = MassSpectrumPlotter(output_dir=str(output_dir))
    
    # Identify all peaks
    plotter.identify_peaks(data_with_masses)
    
    # Create plots with all expected resonances marked
    resonances = plotter.get_resonances('jpsi', 'eta_c', 'chi_c0', 'chi_c1', 'eta_c2s')
    plotter.plot_mass_spectrum(data_with_masses, resonances=resonances)
    
    # Also create focused plots for individual resonances
    logger.info("\nCreating focused plots for individual resonances...")
    
    # J/ψ region
    jpsi_res = plotter.get_resonances('jpsi')
    plotter.plot_mass_spectrum(data_with_masses, 
                              mass_range=(3000, 3200), 
                              bins=80,
                              resonances=jpsi_res)
    
    # η_c region
    eta_c_res = plotter.get_resonances('eta_c')
    plotter.plot_mass_spectrum(data_with_masses,
                              mass_range=(2900, 3080),
                              bins=80,
                              resonances=eta_c_res)
    
    # χ_c region
    chi_c_res = plotter.get_resonances('chi_c0', 'chi_c1')
    plotter.plot_mass_spectrum(data_with_masses,
                              mass_range=(3350, 3600),
                              bins=80,
                              resonances=chi_c_res)
    
    # Step 5: Fit J/ψ peak
    logger.info("\nStep 5: Fitting J/ψ peak...")
    fitter = JpsiPeakFitter(output_dir=str(output_dir))
    fit_results = fitter.fit_jpsi_peak(data_with_masses)
    logger.info(f"Fit results: {fit_results}")
    
    # Step 6: Calculate efficiency (if MC available)
    total_yield = None
    if args.mc_dir:
        logger.info("\nStep 6: Calculating efficiency from MC...")
        eff_calc = EfficiencyCalculator(args.mc_dir)
        efficiency = eff_calc.calculate_efficiency()
        
        # Step 7: Estimate total B+ → J/ψ K+ yield
        logger.info("\nStep 7: Estimating total B+ → J/ψ K+ yield...")
        total_yield = eff_calc.estimate_total_jpsi_yield(args.years)
        logger.info(f"Estimated total B+ → J/ψ K+ yield: {total_yield:.2e}")
    else:
        logger.info("\nStep 6-7: MC directory not provided, skipping efficiency calculation")
    
    # Step 8: Calculate B(J/ψ → pK⁻Λ̄) branching ratio
    if total_yield and fit_results:
        logger.info("\nStep 8: Calculating branching ratio...")
        br_jpsi_to_pklambda = fit_results['signal_yield'] / total_yield
        logger.info(f"B(J/ψ → pK⁻Λ̄) = {br_jpsi_to_pklambda:.2e}")
    
    logger.info("\n" + "="*70)
    logger.info("Analysis complete! Check outputs in: " + str(output_dir))
    logger.info("="*70)

if __name__ == "__main__":
    main()