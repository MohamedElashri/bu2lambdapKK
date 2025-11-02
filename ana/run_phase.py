#!/usr/bin/env python3
"""
Individual Phase Runners for B⁺ → Λ̄pK⁻K⁺ Analysis

Run individual analysis phases independently for testing and development.

Usage:
  python run_phase.py <phase_number> [options]
  
Examples:
  python run_phase.py 2 --years 2016 --track-types LL
  python run_phase.py 5 --use-cached
  python run_phase.py 7 --use-cached
"""

import sys
import argparse
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from run_pipeline import PipelineManager


def run_phase2(pipeline, years, track_types, use_cached):
    """Run Phase 2: Data/MC loading + Lambda cuts"""
    data_dict, mc_dict = pipeline.phase2_load_data_and_lambda_cuts(
        years=years,
        track_types=track_types,
        use_cached=use_cached
    )
    
    # Print summary
    print("\n" + "="*80)
    print("PHASE 2 SUMMARY")
    print("="*80)
    print(f"\nData loaded for {len(data_dict)} years:")
    for year in data_dict:
        for track_type in data_dict[year]:
            n = len(data_dict[year][track_type])
            print(f"  {year} {track_type}: {n:,} events")
    
    print(f"\nMC loaded for {len(mc_dict)} states:")
    for state in mc_dict:
        total = sum(len(mc_dict[state][year][tt]) 
                   for year in mc_dict[state] 
                   for tt in mc_dict[state][year])
        print(f"  {state}: {total:,} events")
    
    return data_dict, mc_dict


def run_phase5(pipeline, use_cached):
    """Run Phase 5: Mass fitting"""
    # Load data from cache
    data_dict = pipeline._load_cache("2", "data_after_lambda")
    if data_dict is None:
        print("❌ Error: No cached data found. Run Phase 2 first.")
        return None
    
    # Apply any cuts (Phase 4 - currently just passes through)
    data_final, _ = pipeline.phase4_apply_optimized_cuts(data_dict, {}, None)
    
    # Run fitting
    fit_results = pipeline.phase5_mass_fitting(data_final, use_cached=use_cached)
    
    # Print summary
    print("\n" + "="*80)
    print("PHASE 5 SUMMARY - YIELDS")
    print("="*80)
    for year in fit_results["yields"]:
        print(f"\n{year}:")
        for state in fit_results["yields"][year]:
            n, err = fit_results["yields"][year][state]
            print(f"  {state:8s}: {n:8.1f} ± {err:6.1f}")
    
    return fit_results


def run_phase6(pipeline, use_cached):
    """Run Phase 6: Efficiency calculation"""
    # Load MC from cache
    mc_dict = pipeline._load_cache("2", "mc_after_lambda")
    if mc_dict is None:
        print("❌ Error: No cached MC found. Run Phase 2 first.")
        return None
    
    # Load or create dummy cuts
    import pandas as pd
    cuts_file = Path("tables/optimized_cuts.csv")
    if cuts_file.exists():
        optimized_cuts_df = pd.read_csv(cuts_file)
    else:
        states = ["jpsi", "etac", "chic0", "chic1"]
        optimized_cuts_df = pd.DataFrame({
            "state": states,
            "variable": ["Bu_PT"] * len(states),
            "optimal_value": [2000.0] * len(states),
            "cut_type": [">"] * len(states),
            "first_branch": ["Bu_PT"] * len(states)
        })
    
    # Run efficiency calculation
    efficiencies = pipeline.phase6_efficiency_calculation(
        mc_dict, optimized_cuts_df, use_cached=use_cached
    )
    
    return efficiencies


def run_phase7(pipeline):
    """Run Phase 7: Branching fraction ratios"""
    # Load fit results and efficiencies from cache
    fit_results = pipeline._load_cache("5", "fit_results")
    efficiencies = pipeline._load_cache("6", "efficiencies")
    
    if fit_results is None:
        print("❌ Error: No cached fit results. Run Phase 5 first.")
        return None
    
    if efficiencies is None:
        print("❌ Error: No cached efficiencies. Run Phase 6 first.")
        return None
    
    # Run branching ratio calculation
    ratios_df = pipeline.phase7_branching_ratios(fit_results, efficiencies)
    
    return ratios_df


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run individual analysis phases"
    )
    parser.add_argument(
        "phase",
        type=int,
        choices=[2, 3, 4, 5, 6, 7],
        help="Phase number to run"
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2016,2017,2018",
        help="Comma-separated years (default: all)"
    )
    parser.add_argument(
        "--track-types",
        type=str,
        default="LL,DD",
        help="Comma-separated track types (default: LL,DD)"
    )
    parser.add_argument(
        "--use-cached",
        action="store_true",
        default=True,
        help="Use cached results if available"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force reprocessing"
    )
    
    args = parser.parse_args()
    
    years = args.years.split(",")
    track_types = args.track_types.split(",")
    use_cached = not args.no_cache
    
    # Initialize pipeline
    pipeline = PipelineManager(config_dir="config", cache_dir="cache")
    
    # Run requested phase
    if args.phase == 2:
        run_phase2(pipeline, years, track_types, use_cached)
    elif args.phase == 5:
        run_phase5(pipeline, use_cached)
    elif args.phase == 6:
        run_phase6(pipeline, use_cached)
    elif args.phase == 7:
        run_phase7(pipeline)
    else:
        print(f"❌ Phase {args.phase} runner not yet implemented")
        print("   Available: 2, 5, 6, 7")


if __name__ == "__main__":
    main()
