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

from __future__ import annotations

import sys
import argparse
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import awkward as ak
from typing import Dict, Any, List, Tuple, Optional
import json

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.data_handler import TOMLConfig, DataManager
from modules.lambda_selector import LambdaSelector
from modules.selection_optimizer import SelectionOptimizer
from modules.mass_fitter import MassFitter
from modules.efficiency_calculator import EfficiencyCalculator
from modules.branching_fraction_calculator import BranchingFractionCalculator
from modules.exceptions import ConfigurationError, AnalysisError


class PipelineManager:
    """
    Manages the complete analysis pipeline with checkpointing
    and intermediate result caching.
    """
    
    def __init__(self, config_dir: str = "./config", cache_dir: str = "./cache") -> None:
        """
        Initialize pipeline manager.
        
        Args:
            config_dir: Path to configuration directory
            cache_dir: Path to cache directory for intermediate results
        """
        self.config: TOMLConfig = TOMLConfig(config_dir)
        self.cache_dir: Path = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Load configuration
        print("\n" + "="*80)
        print("PHASE 1: CONFIGURATION VALIDATION")
        print("="*80)
        self._validate_config()
        
        # Create output directories
        self._setup_output_dirs()
        
        print("✓ Configuration loaded and validated")
        print(f"✓ Cache directory: {self.cache_dir}")
        
    def _validate_config(self):
        """Validate that all required configuration is present"""
        # New logical structure validation
        required = ["physics", "detector", "fitting", "selection", "triggers", "data", "efficiencies"]
        for req in required:
            if not hasattr(self.config, req):
                raise ConfigurationError(
                    f"Missing required configuration section: '{req}'\n"
                    f"Expected file: config/{req}.toml\n"
                    f"Please ensure all required configuration files are present."
                )
        
        # Also check backward compatibility attributes are created
        compat_attrs = ["particles", "paths", "luminosity", "branching_fractions", "efficiency_inputs"]
        for attr in compat_attrs:
            if not hasattr(self.config, attr):
                raise ConfigurationError(
                    f"Backward compatibility layer failed to create attribute: '{attr}'\n"
                    f"This is an internal error in TOMLConfig._create_compatibility_layer()"
                )
        
        # Check that data paths exist
        data_root = Path(self.config.paths["data"]["base_path"])
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
    
    def _save_cache(self, phase: str, name: str, data: Any) -> None:
        """
        Save intermediate results to pickle file.
        
        Args:
            phase: Phase number or identifier
            name: Cache name
            data: Data to cache
        """
        cache_file = self._cache_path(phase, name)
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"  → Cached: {cache_file.name}")
    
    def _load_cache(self, phase: str, name: str) -> Optional[Any]:
        """
        Load intermediate results from pickle file.
        
        Args:
            phase: Phase number or identifier
            name: Cache name
            
        Returns:
            Cached data if exists, None otherwise
        """
        cache_file = self._cache_path(phase, name)
        if not cache_file.exists():
            return None
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    def phase2_load_data_and_lambda_cuts(
        self,
        years: List[str],
        track_types: List[str] = ["LL", "DD"],
        use_cached: bool = False
    ) -> Tuple[Dict[str, Dict[str, ak.Array]], Dict[str, Dict[str, Dict[str, ak.Array]]], Dict[str, Dict[str, ak.Array]], Dict[str, Dict[str, int]]]:
        """
        Phase 2: Load data/MC and apply Lambda pre-selection
        
        This is the most time-consuming phase. Results are cached.
        
        Args:
            years: List of year strings (default: ["2016", "2017", "2018"])
            track_types: List of track types (default: ["LL", "DD"])
            use_cached: Use cached results if available
        
        Returns:
            data_dict: {year: {track_type: awkward_array}}
            mc_dict: {state: {year: {track_type: awkward_array}}}
            phase_space_dict: {year: {track_type: awkward_array}}
            mc_generated_counts: {state: {year: {track_type: int}}}
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
            phase_space_dict = self._load_cache("2", "phase_space_after_lambda")
            mc_generated_counts = self._load_cache("2", "mc_generated_counts")
            if data_dict is not None and mc_dict is not None and phase_space_dict is not None and mc_generated_counts is not None:
                print("✓ Loaded cached data, signal MC, and phase-space MC (after Lambda cuts)")
                return data_dict, mc_dict, phase_space_dict, mc_generated_counts
        
        # Initialize data manager and Lambda selector
        data_manager = DataManager(self.config)
        lambda_selector = LambdaSelector(self.config)
        
        # Load and process REAL DATA
        print("[Loading Real Data]")
        data_dict = {}
        for year in years:
            data_dict[year] = {}
            for track_type in track_types:
                print(f"  {year} {track_type}...", end="", flush=True)
                
                # Load data from both magnets using unified method
                events = data_manager.load_and_process(
                    "data", year, track_type,
                    apply_derived_branches=True,
                    apply_trigger=False
                )
                
                if events is None:
                    print(f" ❌ Both polarities missing! Skipping data {year} {track_type}")
                    continue
                
                # Apply Lambda cuts
                n_before = len(events)
                events_after = lambda_selector.apply_lambda_cuts(events)
                n_after = len(events_after)
                eff = 100 * n_after / n_before if n_before > 0 else 0
                
                data_dict[year][track_type] = events_after
                print(f" {n_before:,} → {n_after:,} ({eff:.1f}%)")
        
        # Load and process PHASE-SPACE MC (KpKm - for background estimation)
        print("\n[Loading Phase-Space MC - KpKm for Background]")
        phase_space_dict = {}
        for year in years:
            phase_space_dict[year] = {}
            for track_type in track_types:
                print(f"  {year} {track_type}...", end="", flush=True)
                
                # Load KpKm MC from both magnets using unified method
                events = data_manager.load_and_process(
                    "KpKm", year, track_type,
                    apply_derived_branches=True,
                    apply_trigger=False
                )
                
                if events is None:
                    print(f" ❌ Both polarities missing! Skipping KpKm {year} {track_type}")
                    continue
                
                # Apply Lambda cuts
                n_before = len(events)
                events_after = lambda_selector.apply_lambda_cuts(events)
                n_after = len(events_after)
                eff = 100 * n_after / n_before if n_before > 0 else 0
                
                phase_space_dict[year][track_type] = events_after
                print(f" {n_before:,} → {n_after:,} ({eff:.1f}%)")
        
        # Load and process MC (all 4 signal states)
        print("\n[Loading MC - Signal States]")
        states = ["jpsi", "etac", "chic0", "chic1"]
        mc_dict = {}
        mc_generated_counts = {}  # Track generator-level counts for signal scaling
        
        for state in states:
            print(f"\n  {state}:")
            mc_dict[state] = {}
            mc_generated_counts[state] = {}
            
            for year in years:
                mc_dict[state][year] = {}
                for track_type in track_types:
                    print(f"    {year} {track_type}...", end="", flush=True)
                    
                    # Map state names: jpsi -> Jpsi, others stay lowercase
                    state_name = "Jpsi" if state == "jpsi" else state
                    
                    # Load MC from both magnets using unified method
                    events = data_manager.load_and_process(
                        state_name, year, track_type,
                        apply_derived_branches=True,
                        apply_trigger=False
                    )
                    
                    if events is None:
                        print(f" ❌ Both polarities missing! Skipping {state} {year} {track_type}")
                        continue
                    
                    # Apply Lambda cuts
                    n_before = len(events)
                    events_after = lambda_selector.apply_lambda_cuts(events)
                    n_after = len(events_after)
                    eff = 100 * n_after / n_before if n_before > 0 else 0
                    
                    mc_dict[state][year][track_type] = events_after
                    
                    # Store generator-level count for signal scaling
                    if year not in mc_generated_counts[state]:
                        mc_generated_counts[state][year] = {}
                    mc_generated_counts[state][year][track_type] = n_before
                    
                    print(f" {n_before:,} → {n_after:,} ({eff:.1f}%)")
        
        # Cache results
        self._save_cache("2", "data_after_lambda", data_dict)
        self._save_cache("2", "mc_after_lambda", mc_dict)
        self._save_cache("2", "phase_space_after_lambda", phase_space_dict)
        self._save_cache("2", "mc_generated_counts", mc_generated_counts)
        
        print("\n✓ Phase 2 complete: Data, signal MC, and phase-space MC loaded")
        print("  → Data: used for background estimation in optimization AND final fitting")
        print("  → Signal MC: used for signal efficiency in optimization (scaled to expected events)")
        print("  → Phase-space MC (KpKm): kept for reference (not used for optimization)")
        print("  → MC generated counts: tracked for proper signal scaling")
        
        return data_dict, mc_dict, phase_space_dict, mc_generated_counts
    
    def phase3_selection_optimization(
        self,
        data_dict: Dict[str, Dict[str, ak.Array]],
        mc_dict: Dict[str, Dict[str, Dict[str, ak.Array]]],
        phase_space_dict: Dict[str, Dict[str, ak.Array]],
        mc_generated_counts: Dict[str, Dict[str, int]],
        use_cached: bool = False,
        force_rerun: bool = False
    ) -> pd.DataFrame:
        """
        Phase 3: Optimize selection cuts using signal MC and real data
        
        CORRECT APPROACH:
        - Signal: signal MC (J/ψ, ηc, χc) - scaled to expected observed events
        - Background: real data sidebands - interpolated to signal region
        - Formula: n_sig = ε * L * σ_eff * 10³
          where ε = n_mc_after_cuts / n_mc_generated (total selection efficiency)
        
        Args:
            data_dict: Data after Lambda cuts (USED for background estimation!)
            mc_dict: Signal MC after Lambda cuts (for signal efficiency)
            phase_space_dict: Phase-space MC after Lambda cuts (kept for reference only)
            mc_generated_counts: Generator-level MC counts (before Lambda cuts)
            use_cached: Use cached optimization results
            force_rerun: Force re-optimization even if cache exists
        
        Returns:
            optimized_cuts_df: DataFrame with optimal cuts per state
        """
        print("\n" + "="*80)
        print("PHASE 3: SELECTION OPTIMIZATION")
        print("="*80)
        
        tables_dir = Path(self.config.paths["output"]["tables_dir"])
        cuts_file = tables_dir / "optimized_cuts.csv"
        
        # Check for existing results
        if not force_rerun and cuts_file.exists():
            print(f"✓ Loading existing optimized cuts from {cuts_file}")
            return pd.read_csv(cuts_file)
        
        if use_cached and not force_rerun:
            cached = self._load_cache("3", "optimized_cuts")
            if cached is not None:
                print("✓ Loaded cached optimized cuts")
                return cached
        
        print("\n  Running full 2D optimization (this may take 30-60 minutes)")
        print("    You can skip this and use default cuts if needed\n")
        
        # Combine track types (LL + DD) for optimizer
        # Optimizer expects: {state: {year: awkward_array}} and {year: awkward_array}
        print("Combining LL and DD track types for optimization...")
        
        mc_combined = {}
        for state in mc_dict:
            mc_combined[state] = {}
            for year in mc_dict[state]:
                # Concatenate LL and DD along axis 0 (event axis)
                arrays_to_combine = []
                for track_type in mc_dict[state][year]:
                    arr = mc_dict[state][year][track_type]
                    # Ensure it's a flat record array
                    if hasattr(arr, 'layout'):
                        arrays_to_combine.append(arr)
                if arrays_to_combine:
                    mc_combined[state][year] = ak.concatenate(arrays_to_combine, axis=0)
                    print(f"  {state}/{year}: {len(mc_combined[state][year])} events")
        
        # Combine phase-space MC track types (for reference, not used for background)
        phase_space_combined = {}
        for year in phase_space_dict:
            arrays_to_combine = []
            for track_type in phase_space_dict[year]:
                arr = phase_space_dict[year][track_type]
                if hasattr(arr, 'layout'):
                    arrays_to_combine.append(arr)
            if arrays_to_combine:
                phase_space_combined[year] = ak.concatenate(arrays_to_combine, axis=0)
                print(f"  phase_space/{year}: {len(phase_space_combined[year])} events")
        
        # Combine real data track types for background estimation
        data_combined = {}
        for year in data_dict:
            arrays_to_combine = []
            for track_type in data_dict[year]:
                arr = data_dict[year][track_type]
                if hasattr(arr, 'layout'):
                    arrays_to_combine.append(arr)
            if arrays_to_combine:
                data_combined[year] = ak.concatenate(arrays_to_combine, axis=0)
                print(f"  data/{year}: {len(data_combined[year])} events")
        
        # Sum MC generated counts across track types (LL + DD)
        mc_generated_combined = {}
        for state in mc_generated_counts:
            mc_generated_combined[state] = {}
            for year in mc_generated_counts[state]:
                total_generated = sum(mc_generated_counts[state][year].values())
                mc_generated_combined[state][year] = total_generated
                print(f"  {state}/{year} generated: {total_generated:,} events")
        
        # Initialize optimizer
        print("\n✓ IMPORTANT: Correct optimization approach:")
        print("    Signal: signal MC (J/ψ, ηc, χc) - scaled to expected events")
        print("    Background: real data sidebands - interpolated to signal region")
        print("    Formula: n_sig = ε * L * σ_eff * 10³")
        print("    Where ε = n_mc_after_cuts / n_mc_generated (full selection efficiency)")
        print("    This properly estimates signal efficiency and background level!")
        
        optimizer = SelectionOptimizer(
            signal_mc=mc_combined,
            phase_space_mc=phase_space_combined,  # Kept for reference, not used
            data=data_combined,  # Real data for background estimation!
            mc_generated_counts=mc_generated_combined,  # Generator-level counts
            config=self.config
        )
        
        # Run N-D grid scan optimization
        print("\n  Running N-D GRID SCAN: exhaustive search over 7 variables")
        print("    Testing all combinations in 7D space ")
        print("    Lambda cuts are FIXED (already applied in Phase 2)")
        optimized_cuts_df = optimizer.optimize_nd_grid_scan()
        
        # Cache and save
        self._save_cache("3", "optimized_cuts", optimized_cuts_df)
        optimized_cuts_df.to_csv(cuts_file, index=False)
        
        print(f"\n✓ Phase 3 complete: Optimized cuts saved to {cuts_file}")
        
        return optimized_cuts_df
    
    def phase4_apply_optimized_cuts(
        self,
        data_dict: Dict[str, Dict[str, ak.Array]],
        mc_dict: Dict[str, Dict[str, Dict[str, ak.Array]]],
        optimized_cuts_df: pd.DataFrame,
        apply_cuts_to_data: Optional[bool] = None,
        data_cut_state: Optional[str] = None
    ) -> Tuple[Dict[str, Dict[str, ak.Array]], Dict[str, Dict[str, Dict[str, ak.Array]]]]:
        """
        Phase 4: Apply optimized cuts to MC (and optionally to data)
        
        IMPORTANT PHYSICS LOGIC FOR FITTING:
        - For mass fitting: apply_cuts_to_data=False (default in config)
          * DATA: Contains all states mixed → fit all simultaneously
          * MC: Apply state-specific cuts for efficiency calculation
        
        OPTIONAL: Apply cuts to data for control plots, validation, etc.
        - Controlled via config/selection.toml [cut_application] section
        - Can override via function parameters
        - Useful for: making plots, validating cuts, debugging
        
        Args:
            data_dict: Data after Lambda cuts
            mc_dict: MC after Lambda cuts
            optimized_cuts_df: Optimized cuts from Phase 3
            apply_cuts_to_data: If specified, override config setting
            data_cut_state: If specified, override config setting
        
        Returns:
            data_final: Data with or without cuts applied
            mc_final: MC with state-specific cuts applied
        """
        print("\n" + "="*80)
        print("PHASE 4: APPLYING OPTIMIZED CUTS")
        print("="*80)
        
        # Read from config if not specified in function call
        cut_config = self.config.selection.get("cut_application", {})
        if apply_cuts_to_data is None:
            apply_cuts_to_data = cut_config.get("apply_cuts_to_data", False)
        if data_cut_state is None:
            data_cut_state = cut_config.get("data_cut_state", "jpsi")
        
        # Show configuration
        print(f"\nConfiguration:")
        print(f"  apply_cuts_to_data = {apply_cuts_to_data}")
        if apply_cuts_to_data:
            print(f"  data_cut_state = {data_cut_state}")
            print(f"    WARNING: Cuts will be applied to data!")
        else:
            print(f"  → Data will remain unchanged (correct for mass fitting)")
        
        if optimized_cuts_df is None or len(optimized_cuts_df) == 0:
            print("\n No optimized cuts provided - using Lambda cuts only")
            return data_dict, mc_dict
        
        print(f"\nApplying {len(optimized_cuts_df)} optimized cut values")
        
        # Apply cuts to MC (state-specific cuts for each state)
        mc_final = {}
        states = ["jpsi", "etac", "chic0", "chic1"]
        
        for state in states:
            print(f"\n[{state}]")
            mc_final[state] = {}
            
            # Get cuts for this state
            state_cuts = optimized_cuts_df[optimized_cuts_df["state"] == state]
            
            if len(state_cuts) == 0:
                print(f"  No cuts found for {state}, using Lambda cuts only")
                mc_final[state] = mc_dict[state]
                continue
            
            print(f"  Applying {len(state_cuts)} cuts for {state}")
            
            for year in mc_dict[state]:
                mc_final[state][year] = {}
                
                for track_type in mc_dict[state][year]:
                    events = mc_dict[state][year][track_type]
                    n_before = len(events)
                    
                    # Start with all events passing
                    mask = ak.ones_like(events["Bu_PT"], dtype=bool)
                    
                    # Apply each cut
                    for _, cut_row in state_cuts.iterrows():
                        branch = cut_row["branch_name"]
                        cut_val = cut_row["optimal_cut"]
                        cut_type = cut_row["cut_type"]
                        
                        if branch not in events.fields:
                            print(f"    Branch {branch} not found, skipping")
                            continue
                        
                        branch_data = events[branch]
                        
                        # Flatten jagged arrays if needed
                        if 'var' in str(ak.type(branch_data)):
                            branch_data = ak.firsts(branch_data)
                        
                        if cut_type == "greater":
                            mask = mask & (branch_data > cut_val)
                        elif cut_type == "less":
                            mask = mask & (branch_data < cut_val)
                    
                    events_after = events[mask]
                    n_after = len(events_after)
                    eff = 100 * n_after / n_before if n_before > 0 else 0
                    
                    mc_final[state][year][track_type] = events_after
                    print(f"    {year} {track_type}: {n_before:,} → {n_after:,} ({eff:.1f}%)")
        
        # Data: Apply cuts only if requested (for control plots, validation, etc.)
        # For fitting, we DON'T apply cuts (all states fit to same data)
        if apply_cuts_to_data:
            print(f"\n  APPLYING CUTS TO DATA (using {data_cut_state} cuts)")
            print("    This should ONLY be used for control plots or validation!")
            print("    For mass fitting, use apply_cuts_to_data=False")
            
            data_final = {}
            data_cuts = optimized_cuts_df[optimized_cuts_df["state"] == data_cut_state]
            
            if len(data_cuts) == 0:
                print(f"    No cuts found for {data_cut_state}, data unchanged")
                data_final = data_dict
            else:
                for year in data_dict:
                    data_final[year] = {}
                    for track_type in data_dict[year]:
                        events = data_dict[year][track_type]
                        n_before = len(events)
                        
                        # Apply cuts
                        mask = ak.ones_like(events["Bu_PT"], dtype=bool)
                        for _, cut_row in data_cuts.iterrows():
                            branch = cut_row["branch_name"]
                            cut_val = cut_row["optimal_cut"]
                            cut_type = cut_row["cut_type"]
                            
                            if branch not in events.fields:
                                continue
                            
                            branch_data = events[branch]
                            if 'var' in str(ak.type(branch_data)):
                                branch_data = ak.firsts(branch_data)
                            
                            if cut_type == "greater":
                                mask = mask & (branch_data > cut_val)
                            elif cut_type == "less":
                                mask = mask & (branch_data < cut_val)
                        
                        events_after = events[mask]
                        n_after = len(events_after)
                        eff = 100 * n_after / n_before if n_before > 0 else 0
                        
                        data_final[year][track_type] = events_after
                        print(f"    Data {year} {track_type}: {n_before:,} → {n_after:,} ({eff:.1f}%)")
        else:
            # Default: Do NOT apply cuts to data (for fitting)
            data_final = data_dict
        
        # Save summary
        summary = {
            "phase": "4_apply_cuts",
            "apply_cuts_to_data": apply_cuts_to_data,
            "data_cut_state": data_cut_state if apply_cuts_to_data else None,
            "data_cuts": f"Cuts from {data_cut_state}" if apply_cuts_to_data else "Lambda pre-selection only",
            "mc_cuts": "State-specific optimized cuts from Phase 3",
            "n_cuts_per_state": {state: len(optimized_cuts_df[optimized_cuts_df["state"] == state]) 
                                 for state in states}
        }
        
        tables_dir = Path(self.config.paths["output"]["tables_dir"])
        with open(tables_dir / "phase4_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("\n✓ Phase 4 complete:")
        if apply_cuts_to_data:
            print("  → Data: CUTS APPLIED (using {data_cut_state} cuts)")
            print("       Use this only for control plots/validation, NOT for fitting!")
        else:
            print("  → Data: UNCHANGED (Lambda pre-selection only)")
            print("      All charmonium states will be fit simultaneously to same data")
        print("  → MC: State-specific optimized cuts applied")
        print("      Used to calculate selection efficiencies per state")
        
        return data_final, mc_final
    
    def apply_manual_cuts(
        self,
        events: ak.Array,
        cuts: Dict[str, Dict[str, Any]]
    ) -> ak.Array:
        """
        Apply manual cuts to any event array (data or MC)
        
        Useful for:
        - Making control plots with specific cuts
        - Testing different cut values
        - Creating validation samples
        
        Args:
            events: Awkward array of events
            cuts: Dictionary of cuts, format:
                  {"branch_name": {"cut_type": "greater"|"less", "value": float}}
                  
        Example:
            cuts = {
                "Bu_PT": {"cut_type": "greater", "value": 5000.0},
                "Bu_IPCHI2_OWNPV": {"cut_type": "less", "value": 9.0},
                "h1_ProbNNk": {"cut_type": "greater", "value": 0.2}
            }
            events_cut = pipeline.apply_manual_cuts(events, cuts)
        
        Returns:
            events_after_cuts: Filtered awkward array
        """
        n_before = len(events)
        
        # Start with all events passing
        if "Bu_PT" in events.fields:
            mask = ak.ones_like(events["Bu_PT"], dtype=bool)
        elif "Bu_MM" in events.fields:
            mask = ak.ones_like(events["Bu_MM"], dtype=bool)
        else:
            raise AnalysisError(
                "Cannot create mask for manual cuts - no reference branch found.\n"
                "Expected Bu_PT or Bu_MM in events. Check data loading."
            )
        
        # Apply each cut
        for branch_name, cut_spec in cuts.items():
            if branch_name not in events.fields:
                print(f"    Branch {branch_name} not found, skipping")
                continue
            
            branch_data = events[branch_name]
            
            # Flatten jagged arrays if needed
            if 'var' in str(ak.type(branch_data)):
                branch_data = ak.firsts(branch_data)
            
            cut_type = cut_spec["cut_type"]
            cut_val = cut_spec["value"]
            
            if cut_type == "greater":
                mask = mask & (branch_data > cut_val)
            elif cut_type == "less":
                mask = mask & (branch_data < cut_val)
            else:
                print(f"    Unknown cut_type '{cut_type}', skipping {branch_name}")
                continue
            
            print(f"  Applied: {branch_name} {cut_type} {cut_val}")
        
        events_after = events[mask]
        n_after = len(events_after)
        eff = 100 * n_after / n_before if n_before > 0 else 0
        
        print(f"  Total: {n_before:,} → {n_after:,} ({eff:.1f}%)")
        
        return events_after
    
    def phase5_mass_fitting(
        self,
        data_final: Dict[str, Dict[str, ak.Array]],
        use_cached: bool = False
    ) -> Dict[str, Any]:
        """
        Phase 5: Fit charmonium mass spectrum to extract yields
        
        Fits ALL charmonium states (J/ψ, ηc, χc0, χc1) simultaneously to the
        SAME data sample. Data has only Lambda pre-selection - no state-specific cuts.
        
        This is correct because:
        - Real data contains all states mixed together
        - Different states appear at different M(KK) masses
        - We extract yields by fitting the mass spectrum
        
        Args:
            data_final: Data with Lambda pre-selection only (NOT state-specific cuts)
            use_cached: Use cached fit results
        
        Returns:
            fit_results: Dictionary with yields per state
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
        
        # Combine LL and DD track types for each year
        data_combined = {}
        for year in data_final:
            events_list = []
            for track_type in data_final[year]:
                events_list.append(data_final[year][track_type])
            data_combined[year] = ak.concatenate(events_list)
        
        # Perform fits
        print("\nFitting charmonium mass spectrum...")
        fit_results = fitter.perform_fit(data_combined)
        
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
        tables_dir = Path(self.config.paths["output"]["tables_dir"])
        yields_df.to_csv(tables_dir / "phase5_yields.csv", index=False)
        
        print(f"\n✓ Phase 5 complete: Yields saved to {tables_dir / 'phase5_yields.csv'}")
        
        return fit_results
    
    def phase6_efficiency_calculation(
        self,
        mc_final: Dict[str, Dict[str, Dict[str, ak.Array]]],
        optimized_cuts_df: pd.DataFrame,
        use_cached: bool = False
    ) -> Dict[str, Any]:
        """
        Phase 6: Calculate selection efficiencies from MC
        
        Args:
            mc_final: MC after Lambda cuts (before Bu cuts)
            optimized_cuts_df: Optimized cuts to apply
            use_cached: Use cached efficiency results if available
        
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
                eff_result = eff_calculator.calculate_selection_efficiency(mc_combined, state)
                
                efficiencies[state][year] = eff_result
                
                print(f"    {year}: ε = {eff_result['eff']:.4f} ± {eff_result['err']:.4f} ({100*eff_result['eff']:.2f}%)")
        
        # Calculate efficiency ratios (returns DataFrame)
        ratios_df = eff_calculator.calculate_efficiency_ratios(efficiencies)
        
        # Cache results
        self._save_cache("6", "efficiencies", efficiencies)
        
        # Save tables
        eff_calculator.generate_efficiency_table(efficiencies)
        
        print(f"\n✓ Phase 6 complete: Efficiencies saved to tables/")
        
        return efficiencies
    
    def phase7_branching_ratios(
        self,
        yields: Dict[str, Dict[str, Tuple[float, float]]],
        efficiencies: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> pd.DataFrame:
        """
        Phase 7: Calculate branching fraction ratios
        
        Args:
            yields: Yields from Phase 5 {year: {state: (value, error)}}
            efficiencies: Results from Phase 6 {state: {year: {"eff": value, "err": error}}}
        
        Returns:
            ratios_df: DataFrame with BR ratios
        """
        print("\n" + "="*80)
        print("PHASE 7: BRANCHING FRACTION RATIOS")
        print("="*80)
        
        # Initialize calculator
        bf_calculator = BranchingFractionCalculator(
            yields=yields,
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
    
    def run_full_pipeline(
        self,
        years: Optional[List[str]] = None,
        track_types: Optional[List[str]] = None,
        force_reoptimize: bool = False,
        no_cache: bool = False
    ) -> Dict[str, Any]:
        """
        Execute complete analysis pipeline
        
        Args:
            years: Years to process (default: all)
            track_types: Track types to process (default: all)
            force_reoptimize: Force re-running cut optimization (ignore saved/cached results)
            no_cache: Force reprocessing (ignore all cached results)
        
        Returns:
            results: Dictionary with all results
            use_cached: Use cached intermediate results when available
        """
        # Convert no_cache to use_cached for internal use
        use_cached = not no_cache
        
        print("\n" + "="*80)
        print("B⁺ → Λ̄pK⁻K⁺ CHARMONIUM ANALYSIS - FULL PIPELINE")
        print("="*80)
        print("\nDraft Analysis - Statistical Uncertainties Only")
        print("Phases: 2→3→4→5→6→7")
        print("\nPHASE OVERVIEW:")
        print("  Phase 2: Load data/MC + apply Lambda pre-selection (FIXED cuts)")
        print("  Phase 3: Optimize B+/bachelor/kaon cuts (PER-STATE optimization)")
        print("           → Produces separate optimal cuts for J/ψ, ηc, χc0, χc1")
        print("  Phase 4: Apply optimized cuts to data/MC")
        print("  Phase 5: Mass fitting → extract yields per state")
        print("  Phase 6: Efficiency calculation per state")
        print("  Phase 7: Branching fraction ratios")
        print("="*80)
        
        # Phase 2: Load data and apply Lambda cuts
        data_dict, mc_dict, phase_space_dict, mc_generated_counts = self.phase2_load_data_and_lambda_cuts(
            years=years,
            track_types=track_types,
            use_cached=use_cached
        )
        
        # Phase 3: Selection optimization (ALWAYS RUN - critical for per-state analysis!)
        print("\n" + "="*80)
        print("PHASE 3: SELECTION OPTIMIZATION (Per-State)")
        print("="*80)
        print("This phase optimizes cuts independently for each charmonium state:")
        print("  - J/ψ (highest stats) → can use tighter cuts")
        print("  - ηc (lowest mass) → may need softer cuts")  
        print("  - χc0, χc1 (intermediate) → state-specific optimization")
        print("")
        print("Each variable (Bu_PT, IPCHI2, etc.) is optimized per-state using FOM.")
        print("="*80)
        
        # Run optimization (will use cache unless force_reoptimize is set)
        optimized_cuts_df = self.phase3_selection_optimization(
            data_dict, mc_dict, phase_space_dict, mc_generated_counts,
            use_cached=use_cached,
            force_rerun=force_reoptimize
        )
        
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
        ratios_df = self.phase7_branching_ratios(fit_results["yields"], efficiencies)
        
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
        print("  IMPORTANT NOTES:")
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
        "--force-reoptimize",
        action="store_true",
        help="Force re-running of cut optimization (ignore cached/saved results)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force reprocessing (ignore all cached results)"
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2016",  # Default to 2016 only for faster testing
        help="Comma-separated list of years (default: 2016; use 2016,2017,2018 for full)"
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
    
    # Validate configuration before running pipeline
    print("Validating configuration...")
    from utils.validate_config import ConfigValidator
    validator = ConfigValidator(config_dir="config", verbose=False)
    if not validator.validate_all():
        print("\nConfiguration validation failed. Please fix errors before running pipeline.")
        sys.exit(1)
    print()
    
    # Initialize pipeline
    pipeline = PipelineManager(config_dir="config", cache_dir="cache")
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(
        years=years,
        track_types=track_types,
        force_reoptimize=args.force_reoptimize,
        no_cache=args.no_cache
    )
    
    return results


if __name__ == "__main__":
    results = main()
