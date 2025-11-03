#!/usr/bin/env python3
"""
Individual Phase Runners for B⁺ → Λ̄pK⁻K⁺ Analysis

Run individual analysis phases with automatic dependency resolution.
Each phase automatically runs prerequisite phases if needed.

Usage:
  python run_phase.py <phase_number> [options]
  
Examples:
  python run_phase.py 2 --years 2016 --track-types LL
  python run_phase.py 5              # Auto-runs phase 2 if needed
  python run_phase.py 7              # Auto-runs phases 2, 5, 6 if needed

Phase Dependencies:
  Phase 2: None (loads raw data/MC)
  Phase 3: Phase 2 (uses data/MC for optimization)
  Phase 5: Phase 2 (fits data)
  Phase 6: Phase 2 (calculates MC efficiencies)
  Phase 7: Phases 5 and 6 (computes BR ratios)
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


def run_phase3(pipeline, use_cached):
    """Run Phase 3: Selection optimization"""
    # Load data and MC from cache
    data_dict = pipeline._load_cache("2", "data_after_lambda")
    mc_dict = pipeline._load_cache("2", "mc_after_lambda")
    
    if data_dict is None or mc_dict is None:
        print("❌ Error: No cached data/MC found. Run Phase 2 first.")
        return None
    
    # Run optimization
    optimized_cuts_df = pipeline.phase3_selection_optimization(
        data_dict, mc_dict, use_cached=use_cached
    )
    
    # Print summary
    print("\n" + "="*80)
    print("PHASE 3 SUMMARY - OPTIMIZED CUTS")
    print("="*80)
    print(optimized_cuts_df.to_string(index=False))
    
    return optimized_cuts_df


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


def check_dependencies(pipeline, phase, years, track_types):
    """
    Check and auto-run prerequisite phases if needed.
    
    Dependencies:
      Phase 2: None
      Phase 3: Phase 2
      Phase 5: Phase 2
      Phase 6: Phase 2
      Phase 7: Phases 5 and 6
    """
    print("\n" + "="*80)
    print(f"CHECKING DEPENDENCIES FOR PHASE {phase}")
    print("="*80)
    
    if phase == 2:
        print("✓ Phase 2 has no dependencies")
        return True
    
    if phase in [3, 5, 6]:
        # Check Phase 2 cache
        data_cache = pipeline._cache_path("2", "data_after_lambda")
        mc_cache = pipeline._cache_path("2", "mc_after_lambda")
        
        if data_cache.exists() and mc_cache.exists():
            print("✓ Phase 2 cache found")
            return True
        else:
            print("⚠️  Phase 2 cache not found - running Phase 2 first...")
            run_phase2(pipeline, years, track_types, use_cached=False)
            return True
    
    if phase == 7:
        # Check Phase 5 and 6 caches
        phase5_cache = pipeline._cache_path("5", "fit_results")
        phase6_cache = pipeline._cache_path("6", "efficiencies")
        
        needs_phase2 = False
        needs_phase5 = False
        needs_phase6 = False
        
        # Check Phase 5
        if not phase5_cache.exists():
            print("⚠️  Phase 5 cache not found")
            needs_phase5 = True
            # Phase 5 needs Phase 2
            data_cache = pipeline._cache_path("2", "data_after_lambda")
            if not data_cache.exists():
                needs_phase2 = True
        else:
            print("✓ Phase 5 cache found")
        
        # Check Phase 6
        if not phase6_cache.exists():
            print("⚠️  Phase 6 cache not found")
            needs_phase6 = True
            # Phase 6 needs Phase 2
            mc_cache = pipeline._cache_path("2", "mc_after_lambda")
            if not mc_cache.exists():
                needs_phase2 = True
        else:
            print("✓ Phase 6 cache found")
        
        # Run missing phases in order
        if needs_phase2:
            print("\n⚠️  Running Phase 2 first (required by phases 5 and 6)...")
            run_phase2(pipeline, years, track_types, use_cached=False)
        
        if needs_phase5:
            print("\n⚠️  Running Phase 5 (required for Phase 7)...")
            run_phase5(pipeline, use_cached=False)
        
        if needs_phase6:
            print("\n⚠️  Running Phase 6 (required for Phase 7)...")
            run_phase6(pipeline, use_cached=False)
        
        return True
    
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run individual analysis phases with auto-dependency resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_phase.py 2                  # Load data/MC
  python run_phase.py 2 --years 2016     # Load only 2016
  python run_phase.py 5                  # Fit data (auto-runs phase 2 if needed)
  python run_phase.py 7                  # BR ratios (auto-runs phases 2,5,6 if needed)
  python run_phase.py 7 --no-cache       # Force full reprocessing

Phase Dependencies (automatically resolved):
  Phase 2: None (loads raw data/MC)
  Phase 3: Phase 2
  Phase 5: Phase 2 (fits data)
  Phase 6: Phase 2 (MC efficiencies)
  Phase 7: Phases 5 and 6 (BR ratios)
        """
    )
    parser.add_argument(
        "phase",
        type=int,
        choices=[2, 3, 5, 6, 7],
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
        "--no-cache",
        action="store_true",
        help="Force reprocessing (ignore cache)"
    )
    
    args = parser.parse_args()
    
    years = [int(y) for y in args.years.split(",")]
    track_types = args.track_types.split(",")
    use_cached = not args.no_cache
    
    # Initialize pipeline
    pipeline = PipelineManager(config_dir="config", cache_dir="cache")
    
    # Check and resolve dependencies
    check_dependencies(pipeline, args.phase, years, track_types)
    
    # Run requested phase
    print("\n" + "="*80)
    print(f"RUNNING PHASE {args.phase}")
    print("="*80)
    
    if args.phase == 2:
        run_phase2(pipeline, years, track_types, use_cached)
    elif args.phase == 3:
        run_phase3(pipeline, use_cached)
    elif args.phase == 5:
        run_phase5(pipeline, use_cached)
    elif args.phase == 6:
        run_phase6(pipeline, use_cached)
    elif args.phase == 7:
        run_phase7(pipeline)
    else:
        print(f"❌ Phase {args.phase} not implemented")
        print("   Available: 2, 3, 5, 6, 7")


if __name__ == "__main__":
    main()
