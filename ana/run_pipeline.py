#!/usr/bin/env python3
"""
Complete Pipeline Integration for B⁺ → Λ̄pK⁻K⁺ Charmonium Analysis

This script orchestrates all phases (0-7) with real data and MC processing.
Each phase saves intermediate results for reproducibility and debugging.

Phases:
  1. Configuration validation
  2. Data/MC loading with Lambda pre-selection
  3. Save intermediate files (after Lambda cuts)
  4. Selection optimization (if needed)
  5. Apply optimized cuts to data/MC
  6. Mass fitting on data
  7. Efficiency calculation on MC
  8. Branching fraction ratios

Usage:
  python run_pipeline.py [--skip-optimization] [--use-cached] [--years 2016,2017]
"""

import sys
import argparse
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any
import json

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.data_handler import TOMLConfig, DataManager
from modules.lambda_selector import LambdaSelector
from modules.selection_optimizer import SelectionOptimizer
from modules.mass_fitter import MassFitter
from modules.efficiency_calculator import EfficiencyCalculator
from modules.branching_fraction_calculator import BranchingFractionCalculator


class PipelineManager:
    """
    Manages the complete analysis pipeline with checkpointing
    and intermediate result caching.
    """
    
    def __init__(self, config_dir: str = "config", cache_dir: str = "cache"):
        """
        Initialize pipeline manager
        
        Args:
            config_dir: Directory containing TOML configuration files
            cache_dir: Directory for caching intermediate results
        """
        self.config_dir = Path(config_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Load configuration
        print("\n" + "="*80)
        print("PHASE 1: CONFIGURATION VALIDATION")
        print("="*80)
        self.config = TOMLConfig(str(self.config_dir))
        self._validate_config()
        
        # Create output directories
        self._setup_output_dirs()
        
        print("✓ Configuration loaded and validated")
        print(f"✓ Cache directory: {self.cache_dir}")
        
    def _validate_config(self):
        """Validate that all required configuration is present"""
        required = ["paths", "particles", "luminosity", "selection"]
        for req in required:
            if not hasattr(self.config, req):
                raise ValueError(f"Missing required config: {req}")
        
        # Check that data paths exist
        data_root = Path(self.config.paths["data"]["root_dir"])
        if not data_root.exists():
            print(f"⚠️  Warning: Data root directory not found: {data_root}")
            print("   Make sure data files are available before running")
    
    def _setup_output_dirs(self):
        """Create all output directories"""
        for category in ["tables", "plots", "results"]:
            dir_path = Path(self.config.paths["output"][f"{category}_dir"])
            dir_path.mkdir(exist_ok=True, parents=True)
    
    def _cache_path(self, phase: str, name: str) -> Path:
        """Get path for cached intermediate result"""
        return self.cache_dir / f"phase{phase}_{name}.pkl"
    
    def _save_cache(self, phase: str, name: str, data: Any):
        """Save intermediate result to cache"""
        cache_file = self._cache_path(phase, name)
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"  → Cached: {cache_file.name}")
    
    def _load_cache(self, phase: str, name: str) -> Any:
        """Load intermediate result from cache"""
        cache_file = self._cache_path(phase, name)
        if not cache_file.exists():
            return None
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    def phase2_load_data_and_lambda_cuts(self, years: list = None, 
                                         track_types: list = None,
                                         use_cached: bool = False):
        """
        Phase 2: Load data/MC and apply Lambda pre-selection
        
        This is the most time-consuming phase. Results are cached.
        
        Args:
            years: List of years to process (default: ["2016", "2017", "2018"])
            track_types: List of track types (default: ["LL", "DD"])
            use_cached: Use cached results if available
        
        Returns:
            data_dict: {year: {track_type: awkward_array}}
            mc_dict: {state: {year: {track_type: awkward_array}}}
        """
        print("\n" + "="*80)
        print("PHASE 2: DATA/MC LOADING + LAMBDA PRE-SELECTION")
        print("="*80)
        
        if years is None:
            years = ["2016", "2017", "2018"]
        if track_types is None:
            track_types = ["LL", "DD"]
        
        # Check cache
        if use_cached:
            data_dict = self._load_cache("2", "data_after_lambda")
            mc_dict = self._load_cache("2", "mc_after_lambda")
            if data_dict is not None and mc_dict is not None:
                print("✓ Loaded cached data and MC (after Lambda cuts)")
                return data_dict, mc_dict
        
        # Initialize data manager and Lambda selector
        data_manager = DataManager(self.config)
        lambda_selector = LambdaSelector(self.config)
        
        # Load and process DATA
        print("\n[Loading DATA]")
        data_dict = {}
        for year in years:
            data_dict[year] = {}
            for track_type in track_types:
                print(f"  {year} {track_type}...", end="", flush=True)
                
                # Load raw data
                events = data_manager.load_single_file(
                    year=year, 
                    magnet="MD",  # Load MD, will combine later if needed
                    track_type=track_type,
                    data_type="data"
                )
                
                # Apply Lambda cuts
                events_after = lambda_selector.apply_lambda_cuts(events)
                
                n_before = len(events)
                n_after = len(events_after)
                eff = 100 * n_after / n_before if n_before > 0 else 0
                
                data_dict[year][track_type] = events_after
                
                print(f" {n_before:,} → {n_after:,} ({eff:.1f}%)")
        
        # Load and process MC (all 4 states)
        print("\n[Loading MC - Signal States]")
        states = ["jpsi", "etac", "chic0", "chic1"]
        mc_dict = {}
        
        for state in states:
            print(f"\n  {state}:")
            mc_dict[state] = {}
            
            for year in years:
                mc_dict[state][year] = {}
                for track_type in track_types:
                    print(f"    {year} {track_type}...", end="", flush=True)
                    
                    # Load raw MC
                    events = data_manager.load_single_file(
                        year=year,
                        magnet="MD",
                        track_type=track_type,
                        data_type=state
                    )
                    
                    # Apply Lambda cuts
                    events_after = lambda_selector.apply_lambda_cuts(events)
                    
                    n_before = len(events)
                    n_after = len(events_after)
                    eff = 100 * n_after / n_before if n_before > 0 else 0
                    
                    mc_dict[state][year][track_type] = events_after
                    
                    print(f" {n_before:,} → {n_after:,} ({eff:.1f}%)")
        
        # Cache results
        self._save_cache("2", "data_after_lambda", data_dict)
        self._save_cache("2", "mc_after_lambda", mc_dict)
        
        print("\n✓ Phase 2 complete: Data and MC loaded with Lambda cuts applied")
        
        return data_dict, mc_dict
    
    def phase3_selection_optimization(self, data_dict: Dict, mc_dict: Dict,
                                     use_cached: bool = False,
                                     force_rerun: bool = False):
        """
        Phase 3: Optimize selection cuts using 2D FOM scans
        
        Args:
            data_dict: Data after Lambda cuts
            mc_dict: MC after Lambda cuts
            use_cached: Use cached optimization results
            force_rerun: Force re-optimization even if cache exists
        
        Returns:
            optimized_cuts_df: DataFrame with optimal cuts per state
        """
        print("\n" + "="*80)
        print("PHASE 3: SELECTION OPTIMIZATION")
        print("="*80)
        
        cuts_file = Path("tables/optimized_cuts.csv")
        
        # Check for existing results
        if not force_rerun and cuts_file.exists():
            print(f"✓ Loading existing optimized cuts from {cuts_file}")
            return pd.read_csv(cuts_file)
        
        if use_cached and not force_rerun:
            cached = self._load_cache("3", "optimized_cuts")
            if cached is not None:
                print("✓ Loaded cached optimized cuts")
                return cached
        
        print("\n⚠️  Running full 2D optimization (this may take 30-60 minutes)")
        print("    You can skip this and use default cuts if needed\n")
        
        # Initialize optimizer (needs phase space MC - placeholder for now)
        # TODO: Load phase space MC properly
        phase_space_mc = {}  # Placeholder
        
        optimizer = SelectionOptimizer(
            signal_mc=mc_dict,
            phase_space_mc=phase_space_mc,
            data=data_dict,
            config=self.config
        )
        
        # Run optimization
        optimized_cuts_df = optimizer.optimize_2d_all_variables()
        
        # Cache and save
        self._save_cache("3", "optimized_cuts", optimized_cuts_df)
        optimized_cuts_df.to_csv(cuts_file, index=False)
        
        print(f"\n✓ Phase 3 complete: Optimized cuts saved to {cuts_file}")
        
        return optimized_cuts_df
    
    def phase4_apply_optimized_cuts(self, data_dict: Dict, mc_dict: Dict,
                                   optimized_cuts_df: pd.DataFrame):
        """
        Phase 4: Apply optimized cuts to data and MC
        
        This creates the final datasets for fitting and efficiency calculation.
        
        Args:
            data_dict: Data after Lambda cuts
            mc_dict: MC after Lambda cuts
            optimized_cuts_df: Optimized cuts from Phase 3
        
        Returns:
            data_final: Data after all cuts
            mc_final: MC after all cuts
        """
        print("\n" + "="*80)
        print("PHASE 4: APPLYING OPTIMIZED CUTS")
        print("="*80)
        
        # For now, we'll use Lambda cuts only (simplified)
        # Full implementation would apply Bu_PT, IPCHI2, etc.
        
        print("\n⚠️  Using Lambda cuts only (Bu cuts not yet implemented)")
        print("    This is OK for draft analysis - can add later")
        
        # Just return the Lambda-cut data for now
        data_final = data_dict
        mc_final = mc_dict
        
        # Save summary
        summary = {
            "phase": "4_apply_cuts",
            "cuts_applied": "Lambda selection only",
            "note": "Bu-level cuts to be added in full analysis"
        }
        
        with open("tables/phase4_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("✓ Phase 4 complete: Using Lambda-cut data for subsequent phases")
        
        return data_final, mc_final
    
    def phase5_mass_fitting(self, data_final: Dict, use_cached: bool = False):
        """
        Phase 5: Fit charmonium mass spectrum to extract yields
        
        Args:
            data_final: Data after all cuts
            use_cached: Use cached fit results
        
        Returns:
            fit_results: Dictionary with yields, masses, widths per year
        """
        print("\n" + "="*80)
        print("PHASE 5: MASS FITTING")
        print("="*80)
        
        # Check cache
        if use_cached:
            cached = self._load_cache("5", "fit_results")
            if cached is not None:
                print("✓ Loaded cached fit results")
                return cached
        
        # Initialize fitter
        fitter = MassFitter(self.config)
        
        # Perform fits
        print("\nFitting charmonium mass spectrum...")
        fit_results = fitter.perform_fit(data_final)
        
        # Cache results
        self._save_cache("5", "fit_results", fit_results)
        
        # Save yields table
        yields_data = []
        for year in fit_results["yields"]:
            for state in fit_results["yields"][year]:
                n, n_err = fit_results["yields"][year][state]
                yields_data.append({
                    "year": year,
                    "state": state,
                    "yield": n,
                    "error": n_err
                })
        
        yields_df = pd.DataFrame(yields_data)
        yields_df.to_csv("tables/phase5_yields.csv", index=False)
        
        print(f"\n✓ Phase 5 complete: Yields saved to tables/phase5_yields.csv")
        
        return fit_results
    
    def phase6_efficiency_calculation(self, mc_final: Dict, 
                                     optimized_cuts_df: pd.DataFrame,
                                     use_cached: bool = False):
        """
        Phase 6: Calculate selection efficiencies from MC
        
        Args:
            mc_final: MC after Lambda cuts (before Bu cuts)
            optimized_cuts_df: Optimized cuts to apply
            use_cached: Use cached efficiency results
        
        Returns:
            efficiencies: {state: {year: {"eff": value, "err": error}}}
        """
        print("\n" + "="*80)
        print("PHASE 6: EFFICIENCY CALCULATION")
        print("="*80)
        
        # Check cache
        if use_cached:
            cached = self._load_cache("6", "efficiencies")
            if cached is not None:
                print("✓ Loaded cached efficiencies")
                return cached
        
        # Initialize efficiency calculator
        eff_calculator = EfficiencyCalculator(self.config, optimized_cuts_df)
        
        # Calculate efficiencies
        print("\nCalculating selection efficiencies from MC...")
        efficiencies = {}
        
        states = ["jpsi", "etac", "chic0", "chic1"]
        years = list(mc_final["jpsi"].keys())
        
        for state in states:
            print(f"\n  {state}:")
            efficiencies[state] = {}
            
            for year in years:
                # Combine LL and DD if both exist
                if "LL" in mc_final[state][year] and "DD" in mc_final[state][year]:
                    import awkward as ak
                    mc_combined = ak.concatenate([
                        mc_final[state][year]["LL"],
                        mc_final[state][year]["DD"]
                    ])
                else:
                    # Use whichever is available
                    track_type = list(mc_final[state][year].keys())[0]
                    mc_combined = mc_final[state][year][track_type]
                
                # Calculate efficiency
                eff, err = eff_calculator.calculate_selection_efficiency(mc_combined, state)
                
                efficiencies[state][year] = {"eff": eff, "err": err}
                
                print(f"    {year}: ε = {eff:.4f} ± {err:.4f} ({100*eff:.2f}%)")
        
        # Calculate efficiency ratios
        print("\n  Efficiency ratios (ε_J/ψ / ε_state):")
        ratios = eff_calculator.calculate_efficiency_ratios(efficiencies)
        
        for state in ["etac", "chic0", "chic1"]:
            ratio = ratios[state]["ratio"]
            err = ratios[state]["error"]
            print(f"    ε_J/ψ / ε_{state} = {ratio:.4f} ± {err:.4f}")
        
        # Cache results
        self._save_cache("6", "efficiencies", efficiencies)
        
        # Save tables
        eff_calculator.generate_efficiency_table(efficiencies)
        
        print(f"\n✓ Phase 6 complete: Efficiencies saved to tables/")
        
        return efficiencies
    
    def phase7_branching_ratios(self, fit_results: Dict, efficiencies: Dict):
        """
        Phase 7: Calculate branching fraction ratios
        
        Args:
            fit_results: Results from Phase 5 (yields)
            efficiencies: Results from Phase 6
        
        Returns:
            ratios_df: DataFrame with BR ratios
        """
        print("\n" + "="*80)
        print("PHASE 7: BRANCHING FRACTION RATIOS")
        print("="*80)
        
        # Initialize calculator
        bf_calculator = BranchingFractionCalculator(
            yields=fit_results["yields"],
            efficiencies=efficiencies,
            config=self.config
        )
        
        # Calculate all ratios
        print("\nCalculating branching fraction ratios...")
        ratios_df = bf_calculator.calculate_all_ratios()
        
        # Yield consistency check
        print("\nChecking yield consistency across years...")
        consistency_df = bf_calculator.check_yield_consistency_per_year()
        
        # Generate final summary
        bf_calculator.generate_final_summary(ratios_df)
        
        print(f"\n✓ Phase 7 complete: Results saved to tables/ and results/")
        
        return ratios_df
    
    def run_full_pipeline(self, years: list = None, track_types: list = None,
                         skip_optimization: bool = False, use_cached: bool = True):
        """
        Execute complete analysis pipeline
        
        Args:
            years: Years to process (default: all)
            track_types: Track types to process (default: all)
            skip_optimization: Skip selection optimization phase
            use_cached: Use cached intermediate results when available
        """
        print("\n" + "="*80)
        print("B⁺ → Λ̄pK⁻K⁺ CHARMONIUM ANALYSIS - FULL PIPELINE")
        print("="*80)
        print("\nDraft Analysis - Statistical Uncertainties Only")
        print("Phases: 2→3→4→5→6→7")
        print("="*80)
        
        # Phase 2: Load data and apply Lambda cuts
        data_dict, mc_dict = self.phase2_load_data_and_lambda_cuts(
            years=years,
            track_types=track_types,
            use_cached=use_cached
        )
        
        # Phase 3: Selection optimization (optional)
        if not skip_optimization:
            optimized_cuts_df = self.phase3_selection_optimization(
                data_dict, mc_dict, use_cached=use_cached
            )
        else:
            print("\n⚠️  Skipping optimization - using default cuts")
            # Create dummy cuts dataframe
            states = ["jpsi", "etac", "chic0", "chic1"]
            optimized_cuts_df = pd.DataFrame({
                "state": states,
                "variable": ["Bu_PT"] * len(states),
                "optimal_value": [2000.0] * len(states),
                "cut_type": [">"] * len(states)
            })
        
        # Phase 4: Apply optimized cuts
        data_final, mc_final = self.phase4_apply_optimized_cuts(
            data_dict, mc_dict, optimized_cuts_df
        )
        
        # Phase 5: Mass fitting
        fit_results = self.phase5_mass_fitting(data_final, use_cached=use_cached)
        
        # Phase 6: Efficiency calculation
        efficiencies = self.phase6_efficiency_calculation(
            mc_final, optimized_cuts_df, use_cached=use_cached
        )
        
        # Phase 7: Branching fraction ratios
        ratios_df = self.phase7_branching_ratios(fit_results, efficiencies)
        
        # Final summary
        self._print_final_summary(ratios_df)
        
        return {
            "data": data_final,
            "mc": mc_final,
            "cuts": optimized_cuts_df,
            "fit_results": fit_results,
            "efficiencies": efficiencies,
            "ratios": ratios_df
        }
    
    def _print_final_summary(self, ratios_df: pd.DataFrame):
        """Print final analysis summary"""
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        
        print("\n📊 FINAL RESULTS:")
        print("="*80)
        print(ratios_df.to_string(index=False))
        
        print("\n" + "="*80)
        print("OUTPUT FILES:")
        print("="*80)
        print("  Tables:")
        print("    - tables/phase5_yields.csv")
        print("    - tables/efficiencies.csv")
        print("    - tables/efficiency_ratios.csv")
        print("    - tables/branching_fraction_ratios.csv")
        print("    - tables/yield_consistency.csv")
        print("\n  Plots:")
        print("    - plots/fit_*.png (mass fits)")
        print("    - plots/yield_consistency_check.png")
        print("\n  Results:")
        print("    - results/final_results.md")
        
        print("\n" + "="*80)
        print("⚠️  IMPORTANT NOTES:")
        print("="*80)
        print("  1. This is a DRAFT analysis with statistical uncertainties only")
        print("  2. Systematic uncertainties to be added later")
        print("  3. Results are RATIOS (don't need absolute branching fractions)")
        print("  4. Review results/final_results.md for complete summary")
        print("="*80 + "\n")


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="B⁺ → Λ̄pK⁻K⁺ Charmonium Analysis Pipeline"
    )
    parser.add_argument(
        "--skip-optimization", 
        action="store_true",
        help="Skip selection optimization (use default cuts)"
    )
    parser.add_argument(
        "--use-cached",
        action="store_true",
        default=True,
        help="Use cached intermediate results (default: True)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force reprocessing (ignore cache)"
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2016,2017,2018",
        help="Comma-separated list of years (default: 2016,2017,2018)"
    )
    parser.add_argument(
        "--track-types",
        type=str,
        default="LL,DD",
        help="Comma-separated list of track types (default: LL,DD)"
    )
    
    args = parser.parse_args()
    
    # Parse years and track types
    years = args.years.split(",")
    track_types = args.track_types.split(",")
    use_cached = not args.no_cache
    
    # Initialize pipeline
    pipeline = PipelineManager(config_dir="config", cache_dir="cache")
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(
        years=years,
        track_types=track_types,
        skip_optimization=args.skip_optimization,
        use_cached=use_cached
    )
    
    return results


if __name__ == "__main__":
    results = main()
