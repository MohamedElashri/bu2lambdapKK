#!/usr/bin/env python3
"""
Script to identify and plot resonances in the M(pK⁻Λ̄) invariant mass spectrum

This script:
1. Loads the data from ROOT files
2. Identifies potential resonance peaks
3. Creates plots with marked resonances
4. Generates a summary report

Usage:
    python resonances.py --input <root_file_or_directory> [options]
    
Example:
    python resonances.py --input data/selected_events.root
    python resonances.py --input data/ --mass-range 2800 3800
"""

import argparse
import logging
import sys
from pathlib import Path

from data_loader import DataLoader
from selection import SelectionProcessor
from mass_calculator import MassCalculator
from plotter import MassSpectrumPlotter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('resonances.log')
    ]
)

logger = logging.getLogger("Bu2LambdaPKK.Resonances")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Identify and plot resonances in M(pK⁻Λ̄) spectrum',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expected resonances:
  J/ψ      : 3097 MeV
  η_c      : 2984 MeV
  χ_c0     : 3415 MeV
  χ_c1     : 3511 MeV
  η_c(2S)  : 3637 MeV

Examples:
  # Analyze all default data (all years, polarities, track types)
  python resonances.py
  
  # Analyze specific years
  python resonances.py --years 16 17
  
  # Focus on J/ψ region
  python resonances.py --mass-range 3000 3200 --resonances jpsi
  
  # Full spectrum with all resonances
  python resonances.py --resonances all --identify-peaks
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/share/lazy/Mohamed/Bu2LambdaPPP/RD/restripped.data/reduced',
        help='Directory containing data ROOT files (default: /share/lazy/Mohamed/Bu2LambdaPPP/RD/restripped.data/reduced)'
    )
    
    parser.add_argument(
        '--years',
        type=str,
        nargs='+',
        default=['16', '17', '18'],
        choices=['16', '17', '18'],
        help='Data taking years (default: 16 17 18)'
    )
    
    parser.add_argument(
        '--polarities',
        type=str,
        nargs='+',
        default=['MD', 'MU'],
        choices=['MD', 'MU'],
        help='Magnet polarities (default: MD MU)'
    )
    
    parser.add_argument(
        '--track-types',
        type=str,
        nargs='+',
        default=['LL', 'DD'],
        choices=['LL', 'DD'],
        help='Track types: LL (long-long) or DD (downstream) (default: LL DD)'
    )
    
    parser.add_argument(
        '--channel',
        type=str,
        default='B2L0barPKpKm',
        help='Decay channel name (default: B2L0barPKpKm)'
    )
    
    parser.add_argument(
        '--selection-config',
        type=str,
        default=None,
        help='Path to selection configuration file (default: analysis/selection.toml). '
             'The active cut set (tight/loose) is controlled in the config file.'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory for plots and reports (default: output)'
    )
    
    parser.add_argument(
        '--mass-range', '-m',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        help='Mass range for plotting in MeV (e.g., 2900 3300)'
    )
    
    parser.add_argument(
        '--bins', '-b',
        type=int,
        default=100,
        help='Number of bins for histogram (default: 100)'
    )
    
    parser.add_argument(
        '--resonances', '-r',
        type=str,
        nargs='+',
        choices=['jpsi', 'eta_c', 'chi_c0', 'chi_c1', 'eta_c2s', 'all'],
        default=['all'],
        help='Resonances to mark on plot (default: all)'
    )
    
    parser.add_argument(
        '--identify-peaks',
        action='store_true',
        default=True,
        help='Run automatic peak identification (default: True)'
    )
    
    parser.add_argument(
        '--peak-range',
        type=float,
        nargs=2,
        default=[2000, 5000],
        metavar=('MIN', 'MAX'),
        help='Mass range for peak identification in MeV (default: 2000 5000)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("="*70)
    logger.info("Resonance Analysis for B+ → pK⁻Λ̄ K+")
    logger.info("="*70)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load data using integrated data_loader
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Years: {args.years}")
    logger.info(f"Polarities: {args.polarities}")
    logger.info(f"Track types: {args.track_types}")
    logger.info(f"Channel: {args.channel}")
    
    loader = DataLoader(data_dir=args.data_dir)
    
    data = loader.load_data(
        years=args.years,
        polarities=args.polarities,
        track_types=args.track_types,
        channel_name=args.channel
    )
    
    if not data:
        logger.error("No data loaded!")
        sys.exit(1)
    
    # Log data statistics
    total_events = sum(len(events) for events in data.values())
    logger.info(f"Loaded {len(data)} dataset(s) with {total_events} total events")
    
    # Log per-dataset statistics
    for key, events in data.items():
        logger.info(f"  {key}: {len(events)} events")
    
    # Apply selection criteria
    logger.info("\nApplying selection criteria...")
    
    # Initialize selector with optional custom config
    if args.selection_config:
        logger.info(f"Using custom selection config: {args.selection_config}")
        selector = SelectionProcessor(config_path=args.selection_config)
    else:
        logger.info("Using default selection config (analysis/selection.toml)")
        selector = SelectionProcessor()
    
    # Apply selection with summary
    selected_data, selection_summary = selector.apply_basic_selection(data, return_summary=True)
    
    selected_events = sum(len(events) for events in selected_data.values())
    logger.info(f"After selection: {selected_events} events ({100*selected_events/total_events:.2f}%)")
    
    # Log per-dataset statistics after selection
    for key, events in selected_data.items():
        summary = selection_summary[key]
        efficiency = summary['final_selected']['efficiency']
        logger.info(f"  {key}: {len(events)} events ({efficiency:.2f}% efficiency)")
    
    # Print detailed cut summary if verbose
    if args.verbose:
        logger.info("\nDetailed cut summary:")
        selector.print_cut_summary(selection_summary)
    
    # Calculate invariant masses
    logger.info("\nCalculating pK⁻Λ̄ invariant masses...")
    mass_calc = MassCalculator()
    data_with_masses = mass_calc.calculate_jpsi_candidates(selected_data)
    
    # Initialize plotter
    plotter = MassSpectrumPlotter(output_dir=str(output_dir))
    
    # Automatic peak identification
    if args.identify_peaks:
        logger.info("\nRunning automatic peak identification...")
        plotter.identify_peaks(data_with_masses, 
                             min_mass=args.peak_range[0], 
                             max_mass=args.peak_range[1])
    
    # Prepare resonances to mark
    resonances = None
    if args.resonances:
        if 'all' in args.resonances:
            resonance_names = ['jpsi', 'eta_c', 'chi_c0', 'chi_c1', 'eta_c2s']
        else:
            resonance_names = args.resonances
        
        resonances = plotter.get_resonances(*resonance_names)
        logger.info(f"Marking resonances: {', '.join(resonance_names)}")
    
    # Convert mass range to tuple if provided
    mass_range = tuple(args.mass_range) if args.mass_range else (2800, 4000)
    # 440 bins for full 2800-4000 MeV range = 5 MeV per bin 
    bins = args.bins if args.mass_range else 440
    
    # Create plots
    logger.info("\nCreating mass spectrum plots...")
    plotter.plot_mass_spectrum(
        data=data_with_masses,
        mass_range=mass_range,
        bins=bins,
        resonances=resonances
    )
    
    # Generate summary report
    logger.info("\nGenerating summary report...")
    report_path = output_dir / "resonance_report.txt"
    generate_report(data_with_masses, resonances, report_path, plotter)
    
    logger.info("="*70)
    logger.info(f"Analysis complete! Check outputs in: {output_dir}")
    logger.info("="*70)


def generate_report(data, resonances, report_path, plotter):
    """
    Generate a text report summarizing the resonance analysis
    
    Parameters:
    - data: Dictionary with loaded data
    - resonances: List of resonance dictionaries
    - report_path: Path to save the report
    - plotter: MassSpectrumPlotter instance
    """
    import numpy as np
    import awkward as ak
    
    # Combine all masses
    all_masses = []
    for key, events in data.items():
        if 'M_pKLambdabar' in events.fields:
            masses = ak.to_numpy(events['M_pKLambdabar'])
            all_masses.append(masses)
    
    if not all_masses:
        logger.warning("No mass data available for report")
        return
    
    all_masses = np.concatenate(all_masses)
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("RESONANCE ANALYSIS REPORT\n")
        f.write(f"B+ → pK⁻Λ̄ K+ Analysis\n")
        f.write("="*70 + "\n\n")
        
        # Data summary
        f.write("DATA SUMMARY\n")
        f.write("-"*70 + "\n")
        f.write(f"Total candidates: {len(all_masses)}\n")
        f.write(f"Mass range: [{np.min(all_masses):.1f}, {np.max(all_masses):.1f}] MeV/c²\n")
        f.write(f"Mean mass: {np.mean(all_masses):.1f} ± {np.std(all_masses):.1f} MeV/c²\n")
        f.write(f"Median mass: {np.median(all_masses):.1f} MeV/c²\n")
        f.write("\n")
        
        # Resonance statistics
        f.write("EXPECTED RESONANCES\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Resonance':<20} {'Mass (MeV)':<15} {'Window (MeV)':<15} {'Candidates':<15} {'Fraction':<10}\n")
        f.write("-"*70 + "\n")
        
        for res_name, res_info in plotter.KNOWN_RESONANCES.items():
            mass = res_info['mass']
            window = res_info.get('window', 50)
            name = res_info['name']
            
            # Count candidates in window
            in_window = np.sum((all_masses >= mass - window) & 
                             (all_masses <= mass + window))
            percentage = 100 * in_window / len(all_masses)
            
            f.write(f"{name:<20} {mass:<15.1f} ±{window:<13.0f} {in_window:<15d} {percentage:>6.2f}%\n")
        
        f.write("\n")
        
        # Mass regions
        f.write("MASS REGION BREAKDOWN\n")
        f.write("-"*70 + "\n")
        regions = [
            ("Below η_c", 0, 2900),
            ("η_c region", 2900, 3050),
            ("J/ψ region", 3050, 3150),
            ("χ_c0 region", 3350, 3480),
            ("χ_c1 region", 3450, 3580),
            ("η_c(2S) region", 3580, 3700),
            ("Above η_c(2S)", 3700, 10000),
        ]
        
        for region_name, min_mass, max_mass in regions:
            in_region = np.sum((all_masses >= min_mass) & (all_masses <= max_mass))
            percentage = 100 * in_region / len(all_masses)
            f.write(f"{region_name:<25} [{min_mass:>6.0f}, {max_mass:>6.0f}] MeV: "
                   f"{in_region:>6d} ({percentage:>5.2f}%)\n")
        
        f.write("\n")
        f.write("="*70 + "\n")
        f.write("End of report\n")
        f.write("="*70 + "\n")
    
    logger.info(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
