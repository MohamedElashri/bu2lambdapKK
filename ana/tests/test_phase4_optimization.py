#!/usr/bin/env python3
"""
Phase 4 Execution: Selection Optimization

This script performs the 2D FOM optimization (variables × states).
For testing/validation, it uses a reduced scope (2016 LL only).
For production, modify to include all years/magnets/tracks.
"""

import sys
from pathlib import Path
import time

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from data_handler import TOMLConfig, DataManager
from lambda_selector import LambdaSelector
from selection_optimizer import SelectionOptimizer

def main():
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*20 + "PHASE 4: SELECTION OPTIMIZATION" + " "*27 + "█")
    print("█" + " "*24 + "2D FOM Scan (Variables × States)" + " "*21 + "█")
    print("█" + " "*78 + "█")
    print("█"*80 + "\n")
    
    start_time = time.time()
    
    # Initialize
    print("="*80)
    print("STEP 1: Initialization")
    print("="*80)
    
    config = TOMLConfig("./config")
    dm = DataManager(config)
    selector = LambdaSelector(config)
    
    print("✓ Configuration loaded")
    print(f"  - Data path: {config.paths['data']['base_path']}")
    print(f"  - MC path: {config.paths['mc']['base_path']}")
    print(f"  - Output: {config.paths['output']['tables_dir']}")
    
    # Load data
    print("\n" + "="*80)
    print("STEP 2: Loading Data")
    print("="*80)
    print("\nNOTE: Using 2016 LL only for speed (production should use all)")
    
    # For production, use load_all_data_combined_magnets
    # For testing, use single year/magnet/track
    YEARS = [2016]
    MAGNETS = ["MD", "MU"]
    TRACK_TYPES = ["LL"]  # or ["LL", "DD"] for production
    
    data_by_year = {}
    for track_type in TRACK_TYPES:
        print(f"\nLoading data ({track_type}):")
        for year in YEARS:
            events_year = []
            for magnet in MAGNETS:
                print(f"  {year} {magnet}...")
                events = dm.load_tree("data", year, magnet, track_type)
                events = dm.compute_derived_branches(events)
                events = selector.apply_lambda_cuts(events)
                events = selector.apply_bu_fixed_cuts(events)
                events_year.append(events)
            
            # Combine magnets
            import awkward as ak
            combined = ak.concatenate(events_year)
            data_by_year[str(year)] = combined
            print(f"  → Total {year}: {len(combined):,} events")
    
    # Load MC for all states
    print("\n" + "="*80)
    print("STEP 3: Loading MC Samples")
    print("="*80)
    
    mc_by_state = {}
    states = ["Jpsi", "etac", "chic0", "chic1"]
    
    for state in states:
        print(f"\nLoading {state} MC ({TRACK_TYPES[0]}):")
        mc_by_state[state.lower()] = {}
        
        for year in YEARS:
            events_year = []
            for magnet in MAGNETS:
                print(f"  {year} {magnet}...")
                events = dm.load_tree(state, year, magnet, TRACK_TYPES[0])
                events = dm.compute_derived_branches(events)
                events = selector.apply_lambda_cuts(events)
                events = selector.apply_bu_fixed_cuts(events)
                events_year.append(events)
            
            import awkward as ak
            combined = ak.concatenate(events_year)
            mc_by_state[state.lower()][str(year)] = combined
            print(f"  → Total {year}: {len(combined):,} events")
    
    # Load phase space MC (KpKm)
    print("\n" + "="*80)
    print("STEP 4: Loading Phase Space MC")
    print("="*80)
    
    phase_space_by_year = {}
    print(f"\nLoading KpKm MC ({TRACK_TYPES[0]}):")
    
    for year in YEARS:
        events_year = []
        for magnet in MAGNETS:
            print(f"  {year} {magnet}...")
            events = dm.load_tree("KpKm", year, magnet, TRACK_TYPES[0])
            events = dm.compute_derived_branches(events)
            events = selector.apply_lambda_cuts(events)
            events = selector.apply_bu_fixed_cuts(events)
            events_year.append(events)
        
        import awkward as ak
        combined = ak.concatenate(events_year)
        phase_space_by_year[str(year)] = combined
        print(f"  → Total {year}: {len(combined):,} events")
    
    # Create optimizer
    print("\n" + "="*80)
    print("STEP 5: Creating Optimizer")
    print("="*80)
    
    optimizer = SelectionOptimizer(
        signal_mc=mc_by_state,
        phase_space_mc=phase_space_by_year,
        data=data_by_year,
        config=config
    )
    
    print("✓ Optimizer created with:")
    print(f"  - States: {list(mc_by_state.keys())}")
    print(f"  - Years: {list(data_by_year.keys())}")
    print(f"  - Variables to optimize: ~14 (B+, bachelor p, K+, K-)")
    
    # Run optimization
    print("\n" + "="*80)
    print("STEP 6: Running 2D Optimization")
    print("="*80)
    print("\nThis will scan each variable for each state...")
    print("Estimated time: 10-30 minutes (depending on # scan points)\n")
    
    results_df = optimizer.optimize_2d_all_variables()
    
    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    
    print(f"\nResults:")
    print(f"  - Total combinations: {len(results_df)}")
    print(f"  - Variables optimized: {results_df['variable'].nunique()}")
    print(f"  - States: {results_df['state'].nunique()}")
    
    # Show example results
    print("\nExample optimal cuts (J/ψ):")
    jpsi_results = results_df[results_df['state'] == 'jpsi'].head(6)
    for _, row in jpsi_results.iterrows():
        print(f"  {row['category']:12s}.{row['variable']:15s}: "
              f"{row['optimal_cut']:8.1f} ({row['cut_type']:7s}) - FOM={row['max_fom']:.3f}")
    
    # Output files
    print("\nOutput files created:")
    output_dir = Path(config.paths["output"]["tables_dir"])
    plot_dir = Path(config.paths["output"]["plots_dir"]) / "optimization"
    
    print(f"  ✓ {output_dir / 'optimized_cuts_2d.csv'}")
    print(f"  ✓ {output_dir / 'optimized_cuts_summary.csv'}")
    print(f"  ✓ {output_dir / 'optimized_cuts_summary.md'}")
    
    if plot_dir.exists():
        n_plots = len(list(plot_dir.glob("*.png")))
        print(f"  ✓ {n_plots} FOM scan plots in {plot_dir}/")
    
    print("\n" + "="*80)
    print("✓ PHASE 4 COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  - Review optimization results in tables/")
    print("  - Check FOM scan plots in plots/optimization/")
    print("  - Proceed to Phase 5: Mass Fitting")
    print("\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
