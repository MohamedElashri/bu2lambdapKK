from __future__ import annotations

import matplotlib.pyplot as plt
import awkward as ak
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from .exceptions import OptimizationError

class SelectionOptimizer:
    """
    Optimize cuts on B+, bachelor p̄, K+, K- using Figure of Merit.
    
    Perform 2D optimization: (variable * charmonium_state)
    Lambda cuts are already applied (pre-selection).
    
    Attributes:
        signal_mc: Signal MC events by state and year
        phase_space_mc: Phase-space MC events by year
        data: Real data events by year
        mc_generated_counts: Generator-level event counts
        config: Configuration object
    """
    
    def __init__(
        self,
        signal_mc: Dict[str, Dict[str, ak.Array]],
        phase_space_mc: Dict[str, ak.Array],
        data: Dict[str, ak.Array],
        mc_generated_counts: Dict[str, Dict[str, int]],
        config: Any
    ) -> None:
        """
        Initialize selection optimizer.
        
        Args:
            signal_mc: {state: {year: events_after_lambda_cuts}}
            phase_space_mc: {year: events_after_lambda_cuts} (KpKm non-resonant)
            data: {year: events_after_lambda_cuts}
            mc_generated_counts: {state: {year: n_generated}} (generator level, before cuts)
            config: Configuration object
        """
        self.signal_mc: Dict[str, Dict[str, ak.Array]] = signal_mc
        self.phase_space_mc: Dict[str, ak.Array] = phase_space_mc
        self.data: Dict[str, ak.Array] = data
        self.mc_generated_counts: Dict[str, Dict[str, int]] = mc_generated_counts
        self.config: Any = config
        
    def scale_signal_to_expected_events(
        self,
        n_mc_after_cuts: int,
        state: str,
        year: str
    ) -> float:
        """
        Scale MC to data-equivalent for realistic FOM optimization
        
        For FOM to work correctly, n_sig and n_bkg must be on the same scale
        (both representing expected events in data).
        
        We scale MC by: (n_mc_after_cuts / n_mc_generated) * scale_factor
        
        Where scale_factor is chosen to give realistic S/B ratios without
        needing unknown state-specific branching fractions. We use a
        data-driven estimate based on the size of the MC sample relative
        to the data sample.
        
        Args:
            n_mc_after_cuts: Number of MC events passing cuts and in signal region
            state: Charmonium state ("jpsi", "etac", "chic0", "chic1")
            year: Data-taking year ("2016", "2017", "2018")
        
        Returns:
            float: Data-scaled MC count for FOM calculation
        """
        # Get n_mc_generated for this state and year (generator level)
        n_mc_generated = self.mc_generated_counts.get(state, {}).get(year, 1)
        
        if n_mc_generated == 0:
            return 0.0
        
        # Calculate efficiency
        efficiency = n_mc_after_cuts / n_mc_generated
        
        # Scale factor from configuration
        # This is calibrated to match typical signal yields in similar LHCb analyses
        # The exact value doesn't matter for optimization (we're comparing cuts),
        # but realistic S/B ratios help FOM converge properly
        scale_factor = self.config.selection.get("optimization_strategy", {}).get("signal_scale_factor", 10000.0)
        
        # Return scaled signal estimate
        return efficiency * scale_factor
    
    def compute_fom(self, n_sig: float, n_bkg: float) -> float:
        """
        Figure of Merit: FOM = n_sig / sqrt(n_bkg + n_sig)
        
        Maximizing FOM balances:
        - Signal efficiency (want high n_sig)
        - Background rejection (want low n_bkg)
        
        Args:
            n_sig: Number of signal events
            n_bkg: Number of background events
        
        Returns:
            FOM value (higher is better), 0.0 if invalid
        """
        if n_bkg + n_sig <= 0:
            return 0.0
        return n_sig / np.sqrt(n_bkg + n_sig)
    
    def define_signal_region(self, state: str) -> Tuple[float, float]:
        """Get mass window for counting signal events in FOM"""
        return self.config.get_signal_region(state)
    
    def define_sideband_regions(self, state: str) -> List[Tuple[float, float]]:
        """
        Define mass sidebands for background estimation
        
        Returns: [(low_min, low_max), (high_min, high_max)]
        """
        center_val = self.config.particles["signal_regions"][state.lower()]["center"]
        window = self.config.particles["signal_regions"][state.lower()]["window"]
        
        # Get sideband multipliers from config (with defaults for backward compatibility)
        opt_config = self.config.selection.get("optimization_strategy", {})
        sb_low_mult = opt_config.get("sideband_low_multiplier", 4.0)
        sb_low_end_mult = opt_config.get("sideband_low_end_multiplier", 1.0)
        sb_high_start_mult = opt_config.get("sideband_high_start_multiplier", 1.0)
        sb_high_mult = opt_config.get("sideband_high_multiplier", 4.0)
        
        # Low sideband: (center - sb_low_mult*window) to (center - sb_low_end_mult*window)
        low_sb = (center_val - sb_low_mult*window, center_val - sb_low_end_mult*window)
        
        # High sideband: (center + sb_high_start_mult*window) to (center + sb_high_mult*window)
        high_sb = (center_val + sb_high_start_mult*window, center_val + sb_high_mult*window)
        
        return [low_sb, high_sb]
    
    def define_optimization_mass_region(self, state: str) -> Tuple[float, float]:
        """
        Define a broader mass region for optimization
        
        This region includes signal + both sidebands, so we only optimize
        cuts using events that are actually relevant to this state's mass region.
        
        This prevents contamination from other states' mass regions during optimization.
        
        Returns: (low_mass, high_mass) for filtering data
        """
        center_val = self.config.particles["signal_regions"][state.lower()]["center"]
        window = self.config.particles["signal_regions"][state.lower()]["window"]
        
        # Use range from 5 windows below to 5 windows above center
        # This includes both sidebands plus signal region plus some margin
        low_mass = center_val - 5*window
        high_mass = center_val + 5*window
        
        return (low_mass, high_mass)
    
    def count_events_in_region(
        self,
        events: ak.Array,
        region: Tuple[float, float]
    ) -> int:
        """
        Count events in mass region.
        
        Uses M_LpKm_h2 (Lambdabar-p-Kminus system mass).
        
        Args:
            events: Awkward array of events
            region: Tuple of (low_mass, high_mass) in MeV
            
        Returns:
            Number of events in region
        """
        # Use M_LpKm_h2 (h2 = K- from charmonium, h1 = K+)
        mass_branch = "M_LpKm_h2" if "M_LpKm_h2" in events.fields else "M_LpKm"
        mask = (events[mass_branch] > region[0]) & (events[mass_branch] < region[1])
        return ak.sum(mask)
    
    def estimate_background_in_signal_region(
        self,
        data_events: ak.Array,
        state: str
    ) -> float:
        """
        Estimate combinatorial background in signal region from real data sidebands
        
        Uses sideband interpolation to estimate background under the peak.
        This gives the true background level in data for cut optimization.
        
        Method:
        1. Count events in low sideband [center - 4*window, center - window]
        2. Count events in high sideband [center + window, center + 4*window]
        3. Average the two sidebands
        4. Scale by width ratio: background_in_signal = avg_sideband * (signal_width / sideband_width)
        
        Args:
            data_events: Real data events after cuts
            state: Charmonium state ("jpsi", "etac", "chic0", "chic1")
        
        Returns:
            float: Estimated background in signal region
        """
        signal_region = self.define_signal_region(state)
        sidebands = self.define_sideband_regions(state)
        
        # Count events in each sideband
        n_low_sb = self.count_events_in_region(data_events, sidebands[0])
        n_high_sb = self.count_events_in_region(data_events, sidebands[1])
        
        # Calculate sideband widths
        low_sb_width = sidebands[0][1] - sidebands[0][0]
        high_sb_width = sidebands[1][1] - sidebands[1][0]
        signal_width = signal_region[1] - signal_region[0]
        
        # Average density from both sidebands (events per MeV)
        density_low = n_low_sb / low_sb_width if low_sb_width > 0 else 0
        density_high = n_high_sb / high_sb_width if high_sb_width > 0 else 0
        avg_density = (density_low + density_high) / 2.0
        
        # Estimate background in signal region
        n_bkg_estimate = avg_density * signal_width
        
        return n_bkg_estimate
    
    def scan_single_variable(
        self,
        state: str,
        variable_name: str,
        branch_name: str,
        scan_config: dict
    ) -> pd.DataFrame:
        """
        Scan a single variable and compute FOM at each cut value
        
        Args:
            state: "jpsi", "etac", "chic0", "chic1"
            variable_name: Logical name (e.g., "bu_pt")
            branch_name: Actual branch name in tree
            scan_config: {begin, end, step, cut_type, description}
            
        Returns:
            DataFrame: [cut_value, n_sig, n_bkg, fom]
        """
        # Generate scan points
        cut_values = np.arange(
            scan_config["begin"],
            scan_config["end"] + scan_config["step"],
            scan_config["step"]
        )
        
        # Combine all years for this state
        sig_mc_arrays = [self.signal_mc[state][year] for year in self.signal_mc[state].keys()]
        sig_mc_combined = ak.concatenate(sig_mc_arrays, axis=0)
        
        # Filter data to state-specific mass region
        data_arrays = [self.data[year] for year in self.data.keys()]
        data_combined_all = ak.concatenate(data_arrays, axis=0)
        opt_mass_region = self.define_optimization_mass_region(state)
        mass_branch = "M_LpKm_h2" if "M_LpKm_h2" in data_combined_all.fields else "M_LpKm"
        mass_filter = (data_combined_all[mass_branch] > opt_mass_region[0]) & \
                     (data_combined_all[mass_branch] < opt_mass_region[1])
        data_combined = data_combined_all[mass_filter]
        
        # Compute weighted average scale factor across all years
        total_mc_events = len(sig_mc_combined)
        weighted_scale = 0.0
        for year in self.signal_mc[state].keys():
            year_mc_events = len(self.signal_mc[state][year])
            year_weight = year_mc_events / total_mc_events if total_mc_events > 0 else 0
            year_scale = self.scale_signal_to_expected_events(1, state, year)
            weighted_scale += year_weight * year_scale
        
        results = []
        
        # Check if branch is jagged (multiple values per event) and flatten if needed
        sig_branch_data = sig_mc_combined[branch_name]
        data_branch_data = data_combined[branch_name]
        
        # If jagged (nested list structure), take first element
        # Check by looking at the type - if it has 'var' in type string, it's jagged
        sig_is_jagged = 'var' in str(ak.type(sig_branch_data))
        data_is_jagged = 'var' in str(ak.type(data_branch_data))
        
        if sig_is_jagged or data_is_jagged:
            print(f"    Warning: {branch_name} is jagged (sig:{sig_is_jagged}, data:{data_is_jagged}), taking first element per event")
            if sig_is_jagged:
                sig_branch_data = ak.firsts(sig_branch_data)
            if data_is_jagged:
                data_branch_data = ak.firsts(data_branch_data)
        
        for cut_val in cut_values:
            # Apply cut
            if scan_config["cut_type"] == "greater":
                sig_mask = sig_branch_data > cut_val
                data_mask = data_branch_data > cut_val
            else:  # "less"
                sig_mask = sig_branch_data < cut_val
                data_mask = data_branch_data < cut_val
            
            # Filter events
            sig_pass = sig_mc_combined[sig_mask]
            data_pass = data_combined[data_mask]
            
            # Count signal in signal region (raw MC count)
            signal_region = self.define_signal_region(state)
            n_sig_mc = self.count_events_in_region(sig_pass, signal_region)
            
            # Scale signal to expected observed events
            n_sig = n_sig_mc * weighted_scale
            
            # Estimate background from real data
            n_bkg = self.estimate_background_in_signal_region(data_pass, state)
            
            # Compute FOM
            fom = self.compute_fom(n_sig, n_bkg)
            
            results.append({
                "cut_value": cut_val,
                "n_sig": n_sig,
                "n_bkg": n_bkg,
                "fom": fom
            })
        
        return pd.DataFrame(results)
    
    def scan_2d_variable_pair(
        self,
        state: str,
        var1: Dict[str, Any],
        var2: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Perform 2D scan of two variables simultaneously
        
        Args:
            state: Charmonium state ("jpsi", "etac", "chic0", "chic1")
            var1: {category, var_name, branch_name, config}
            var2: {category, var_name, branch_name, config}
            
        Returns:
            DataFrame with columns: [cut1, cut2, n_sig, n_bkg, fom]
        """
        # Generate scan points for both variables
        cut1_values = np.arange(
            var1["config"]["begin"],
            var1["config"]["end"] + var1["config"]["step"],
            var1["config"]["step"]
        )
        
        cut2_values = np.arange(
            var2["config"]["begin"],
            var2["config"]["end"] + var2["config"]["step"],
            var2["config"]["step"]
        )
        
        # Combine all years for this state
        sig_mc_arrays = [self.signal_mc[state][year] for year in self.signal_mc[state].keys()]
        sig_mc_combined = ak.concatenate(sig_mc_arrays, axis=0)
        
        # Filter data to state-specific mass region
        data_arrays = [self.data[year] for year in self.data.keys()]
        data_combined_all = ak.concatenate(data_arrays, axis=0)
        opt_mass_region = self.define_optimization_mass_region(state)
        mass_branch = "M_LpKm_h2" if "M_LpKm_h2" in data_combined_all.fields else "M_LpKm"
        mass_filter = (data_combined_all[mass_branch] > opt_mass_region[0]) & \
                     (data_combined_all[mass_branch] < opt_mass_region[1])
        data_combined = data_combined_all[mass_filter]
        
        # Compute weighted average scale factor across all years
        total_mc_events = len(sig_mc_combined)
        weighted_scale = 0.0
        for year in self.signal_mc[state].keys():
            year_mc_events = len(self.signal_mc[state][year])
            year_weight = year_mc_events / total_mc_events if total_mc_events > 0 else 0
            year_scale = self.scale_signal_to_expected_events(1, state, year)
            weighted_scale += year_weight * year_scale
        
        # Handle jagged arrays for both variables
        sig_branch1 = sig_mc_combined[var1["branch_name"]]
        sig_branch2 = sig_mc_combined[var2["branch_name"]]
        data_branch1 = data_combined[var1["branch_name"]]
        data_branch2 = data_combined[var2["branch_name"]]
        
        # Flatten jagged arrays if needed
        if 'var' in str(ak.type(sig_branch1)):
            sig_branch1 = ak.firsts(sig_branch1)
        if 'var' in str(ak.type(sig_branch2)):
            sig_branch2 = ak.firsts(sig_branch2)
        if 'var' in str(ak.type(data_branch1)):
            data_branch1 = ak.firsts(data_branch1)
        if 'var' in str(ak.type(data_branch2)):
            data_branch2 = ak.firsts(data_branch2)
        
        results = []
        total_scans = len(cut1_values) * len(cut2_values)
        scan_count = 0
        
        # Scan all combinations of (cut1, cut2)
        for cut1 in cut1_values:
            for cut2 in cut2_values:
                scan_count += 1
                if scan_count % 50 == 0:
                    print(f"    Progress: {scan_count}/{total_scans} ({100*scan_count/total_scans:.1f}%)", end='\r')
                
                # Apply both cuts simultaneously
                # Variable 1 mask
                if var1["config"]["cut_type"] == "greater":
                    sig_mask1 = sig_branch1 > cut1
                    data_mask1 = data_branch1 > cut1
                else:
                    sig_mask1 = sig_branch1 < cut1
                    data_mask1 = data_branch1 < cut1
                
                # Variable 2 mask
                if var2["config"]["cut_type"] == "greater":
                    sig_mask2 = sig_branch2 > cut2
                    data_mask2 = data_branch2 > cut2
                else:
                    sig_mask2 = sig_branch2 < cut2
                    data_mask2 = data_branch2 < cut2
                
                # Combine masks (AND operation)
                sig_mask = sig_mask1 & sig_mask2
                data_mask = data_mask1 & data_mask2
                
                # Apply masks
                sig_pass = sig_mc_combined[sig_mask]
                data_pass = data_combined[data_mask]
                
                # Count signal in signal region (raw MC count)
                signal_region = self.define_signal_region(state)
                n_sig_mc = self.count_events_in_region(sig_pass, signal_region)
                
                # Scale signal to expected observed events
                n_sig = n_sig_mc * weighted_scale
                
                # Estimate background from real data
                n_bkg = self.estimate_background_in_signal_region(data_pass, state)
                
                # Compute FOM
                fom = self.compute_fom(n_sig, n_bkg)
                
                results.append({
                    "cut1": cut1,
                    "cut2": cut2,
                    "n_sig": n_sig,
                    "n_bkg": n_bkg,
                    "fom": fom
                })
        
        print()  # New line after progress
        return pd.DataFrame(results)
    
    def _get_branch_name_for_variable(self, category: str, var_name: str) -> str:
        """
        Map (category, variable) to actual branch name in normalized data.
        
        This mapping is based on branch naming conventions after normalization.
        """
        # Mapping for each category
        # NOTE: h1 is K+, h2 is K- (confirmed from PDG ID analysis)
        branch_map = {
            "bu": {
                "pt": "Bu_PT",
                "dtf_chi2": "Bu_DTF_chi2",
                "ipchi2": "Bu_IPCHI2_OWNPV",
                "fdchi2": "Bu_FDCHI2_OWNPV",
            },
            "bachelor_p": {
                "probnnp": "p_ProbNNp",
                "track_chi2ndof": "p_TRACK_CHI2NDOF",
                "ipchi2": "p_IPCHI2_OWNPV",
            },
            "kplus": {
                "probnnk": "h1_ProbNNk",  # h1 is K+ (confirmed)
                "track_chi2ndof": "h1_TRACK_CHI2NDOF",
                "ipchi2": "h1_IPCHI2_OWNPV",
            },
            "kminus": {
                "probnnk": "h2_ProbNNk",  # h2 is K- (confirmed - used for charmonium)
                "track_chi2ndof": "h2_TRACK_CHI2NDOF",
                "ipchi2": "h2_IPCHI2_OWNPV",
            },
        }
        
        if category in branch_map and var_name in branch_map[category]:
            return branch_map[category][var_name]
        else:
            # Fallback: construct from category_var_name
            return f"{category}_{var_name}"
    
    def optimize_nd_grid_scan(self) -> pd.DataFrame:
        """
        Perform N-dimensional GRID scan: exhaustive search over all cut combinations
        
        Uses only 7 variables from nd_optimizable_selection config:
        - h1_ProbNNk, h2_ProbNNk, p_ProbNNp (PID)
        - Bu_PT, Bu_FDCHI2, Bu_IPCHI2, Bu_DTF_chi2 (B+ kinematics)
        
        Lambda cuts are already FIXED and applied in Phase 2.
        
        Grid size: 3×3×3×2×4×6×3 = 3,888 combinations per state
        
        Returns:
            DataFrame with optimal cuts for each state
        """
        import itertools
        import numpy as np
        
        states = ["jpsi", "etac", "chic0", "chic1"]
        
        # Get N-D grid scan variables from config
        nd_config = self.config.selection.get("nd_optimizable_selection", {})
        
        if not nd_config:
            raise OptimizationError(
                "No 'nd_optimizable_selection' section found in config/selection.toml!\n"
                "N-D grid scan requires this section to define variables and ranges to optimize."
            )
        
        # Build variable list and grid points
        all_variables = []
        grid_axes = []  # Each element is a list of values to scan
        
        for var_name, var_config in nd_config.items():
            if var_name == "notes":
                continue
            
            # Generate grid points for this variable
            begin = var_config["begin"]
            end = var_config["end"]
            step = var_config["step"]
            grid_points = np.arange(begin, end + step/2, step)  # Include endpoint
            
            all_variables.append({
                "var_name": var_name,
                "branch_name": var_config["branch_name"],
                "cut_type": var_config["cut_type"],
                "description": var_config.get("description", "")
            })
            grid_axes.append(grid_points)
        
        n_vars = len(all_variables)
        total_combinations = np.prod([len(axis) for axis in grid_axes])
        
        print(f"\n{'='*80}")
        print(f"N-DIMENSIONAL GRID SCAN: {n_vars} variables, {total_combinations:,} combinations")
        print(f"{'='*80}")
        for i, var in enumerate(all_variables):
            n_points = len(grid_axes[i])
            print(f"  {var['var_name']:20s} ({var['cut_type']:>7s}): {n_points} points {list(grid_axes[i])}")
        print(f"{'='*80}\n")
        
        all_results = []
        
        # Optimize for each state
        for state in states:
            print(f"\n{'='*60}")
            print(f"Scanning grid for state: {state}")
            print(f"{'='*60}")
            
            # Prepare MC for this state
            sig_mc_arrays = [self.signal_mc[state][year] for year in self.signal_mc[state].keys()]
            sig_mc_combined = ak.concatenate(sig_mc_arrays, axis=0)
            
            # Prepare real data for background estimation
            # CRITICAL: Filter data to only this state's mass region!
            # This ensures each state optimizes against its own relevant background
            data_arrays = [self.data[year] for year in self.data.keys()]
            data_combined_all = ak.concatenate(data_arrays, axis=0)
            
            # Get state-specific mass region for optimization
            opt_mass_region = self.define_optimization_mass_region(state)
            mass_branch = "M_LpKm_h2" if "M_LpKm_h2" in data_combined_all.fields else "M_LpKm"
            mass_filter = (data_combined_all[mass_branch] > opt_mass_region[0]) & \
                         (data_combined_all[mass_branch] < opt_mass_region[1])
            data_combined = data_combined_all[mass_filter]
            
            print(f"  Data filtered to mass region [{opt_mass_region[0]:.0f}, {opt_mass_region[1]:.0f}] MeV")
            print(f"  Total data events: {len(data_combined_all):,} → {len(data_combined):,} (in mass region)")
            
            # Compute weighted average scale factor across all years
            total_mc_events = len(sig_mc_combined)
            weighted_scale = 0.0
            for year in self.signal_mc[state].keys():
                year_mc_events = len(self.signal_mc[state][year])
                year_weight = year_mc_events / total_mc_events if total_mc_events > 0 else 0
                year_scale = self.scale_signal_to_expected_events(1, state, year)
                weighted_scale += year_weight * year_scale
            
            # Extract branch data (once, before loop)
            sig_branches = []
            data_branches = []  # Real data for background
            
            for var in all_variables:
                sig_branch = sig_mc_combined[var["branch_name"]]
                data_branch = data_combined[var["branch_name"]]
                
                # Flatten jagged arrays
                if 'var' in str(ak.type(sig_branch)):
                    sig_branch = ak.firsts(sig_branch)
                if 'var' in str(ak.type(data_branch)):
                    data_branch = ak.firsts(data_branch)
                
                sig_branches.append(sig_branch)
                data_branches.append(data_branch)
            
            # Grid scan: test all combinations
            best_fom = -np.inf
            best_cuts = None
            best_n_sig = 0
            best_n_bkg = 0
            
            print(f"  Scanning {total_combinations:,} combinations...")
            
            # Use itertools.product to generate all combinations
            for i, cut_combination in enumerate(itertools.product(*grid_axes)):
                # Show progress every 500 combinations
                if (i + 1) % 500 == 0 or i == 0:
                    print(f"    Progress: {i+1:,}/{total_combinations:,} ({100*(i+1)/total_combinations:.1f}%)")
                
                # Apply this combination of cuts
                sig_mask = ak.ones_like(sig_branches[0], dtype=bool)
                data_mask = ak.ones_like(data_branches[0], dtype=bool)
                
                for j, (cut_val, var) in enumerate(zip(cut_combination, all_variables)):
                    if var["cut_type"] == "greater":
                        sig_mask = sig_mask & (sig_branches[j] > cut_val)
                        data_mask = data_mask & (data_branches[j] > cut_val)
                    else:
                        sig_mask = sig_mask & (sig_branches[j] < cut_val)
                        data_mask = data_mask & (data_branches[j] < cut_val)
                
                # Filter events
                sig_pass = sig_mc_combined[sig_mask]
                data_pass = data_combined[data_mask]
                
                # Calculate FOM using scaled signal and real data background
                signal_region = self.define_signal_region(state)
                n_sig_mc = self.count_events_in_region(sig_pass, signal_region)
                n_sig = n_sig_mc * weighted_scale
                n_bkg = self.estimate_background_in_signal_region(data_pass, state)
                fom = self.compute_fom(n_sig, n_bkg)
                
                # Update best if this is better
                if fom > best_fom:
                    best_fom = fom
                    best_cuts = cut_combination
                    best_n_sig = n_sig
                    best_n_bkg = n_bkg
            
            print(f"\n  ✓ Grid scan complete!")
            print(f"  Best FOM: {best_fom:.3f}")
            print(f"  n_sig: {best_n_sig:.0f}, n_bkg: {best_n_bkg:.1f}")
            
            # Store results
            if best_cuts is None:
                print("  WARNING: No valid cuts found!")
                continue
                
            print(f"\n  Optimal cuts:")
            for j, var in enumerate(all_variables):
                cut_val = best_cuts[j]
                print(f"    {var['var_name']:20s} {var['cut_type']:>7s} {cut_val:8.3f}")
                
                all_results.append({
                    "state": state,
                    "variable": var["var_name"],
                    "branch_name": var["branch_name"],
                    "optimal_cut": cut_val,
                    "cut_type": var["cut_type"],
                    "max_fom": best_fom,
                    "n_sig_at_optimal": best_n_sig,
                    "n_bkg_at_optimal": best_n_bkg,
                    "description": var["description"]
                })
        
        results_df = pd.DataFrame(all_results)
        
        # Save results
        output_dir = Path(self.config.paths["output"]["tables_dir"])
        output_dir.mkdir(exist_ok=True, parents=True)
        
        results_df.to_csv(output_dir / "optimized_cuts_nd.csv", index=False)
        
        # Save per-state tables
        for state in states:
            state_df = results_df[results_df["state"] == state].copy()
            state_df.to_csv(output_dir / f"optimized_cuts_nd_{state}.csv", index=False)
            print(f"✓ Saved N-D optimized cuts for {state}")
        
        print(f"\n✓ N-D optimization complete!")
        
        return results_df
    
    def _plot_fom_scan(
        self,
        scan_results: pd.DataFrame,
        category: str,
        var_name: str,
        state: str,
        var_config: dict
    ) -> None:
        """
        Plot FOM scan for a single (variable, state) pair
        Similar to Appendix A figures in reference analysis
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Normalize to max value for better visualization
        max_sig = scan_results["n_sig"].max()
        max_bkg = scan_results["n_bkg"].max()
        max_fom = scan_results["fom"].max()
        
        # Plot three curves
        ax.plot(scan_results["cut_value"], 
                scan_results["n_sig"] / max_sig,
                'b-', label='$n_{sig}$ (normalized)', linewidth=2)
        
        ax.plot(scan_results["cut_value"],
                scan_results["n_bkg"] / max_bkg,
                'r-', label='$n_{bkg}$ (normalized)', linewidth=2)
        
        ax.plot(scan_results["cut_value"],
                scan_results["fom"] / max_fom,
                'g-', label='FOM (normalized)', linewidth=2)
        
        # Mark optimal point
        idx_max = scan_results["fom"].idxmax()
        optimal_cut = scan_results.loc[idx_max, "cut_value"]
        
        ax.axvline(optimal_cut, color='k', linestyle='--', 
                   label=f'Optimal: {optimal_cut:.2f}')
        
        ax.set_xlabel(f'{var_config["description"]} ({var_config["cut_type"]})')
        ax.set_ylabel('Ratio to maximum value')
        ax.set_title(f'{category}.{var_name} optimization for {state}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save
        plot_dir = Path(self.config.paths["output"]["plots_dir"]) / "optimization"
        plot_dir.mkdir(exist_ok=True, parents=True)
        
        filename = f"fom_scan_{category}_{var_name}_{state}.pdf"
        plt.savefig(plot_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_2d_fom_heatmap(
        self,
        scan_2d_results: pd.DataFrame,
        var1: Dict[str, Any],
        var2: Dict[str, Any],
        state: str
    ) -> None:
        """
        Plot 2D heatmap of FOM as function of two variables
        
        Similar to correlation plots in appendices of reference analysis
        Shows optimal point in 2D variable space
        """
        # Pivot to create 2D grid
        fom_grid = scan_2d_results.pivot(
            index='cut2', 
            columns='cut1', 
            values='fom'
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(fom_grid.values, 
                      aspect='auto',
                      origin='lower',
                      cmap='viridis',
                      extent=[fom_grid.columns.min(), fom_grid.columns.max(),
                             fom_grid.index.min(), fom_grid.index.max()])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Figure of Merit (FOM)', rotation=270, labelpad=20)
        
        # Mark optimal point
        idx_max = scan_2d_results["fom"].idxmax()
        optimal = scan_2d_results.loc[idx_max]
        ax.plot(optimal['cut1'], optimal['cut2'], 'r*', 
                markersize=20, markeredgecolor='white', markeredgewidth=2,
                label=f'Optimal: FOM={optimal["fom"]:.2f}')
        
        # Labels
        ax.set_xlabel(f'{var1["category"]}.{var1["var_name"]} ({var1["config"]["cut_type"]})')
        ax.set_ylabel(f'{var2["category"]}.{var2["var_name"]} ({var2["config"]["cut_type"]})')
        ax.set_title(f'2D FOM Optimization for {state}\n'
                    f'{var1["var_name"]} vs {var2["var_name"]}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        
        # Save
        plot_dir = Path(self.config.paths["output"]["plots_dir"]) / "optimization" / "2d_scans"
        plot_dir.mkdir(exist_ok=True, parents=True)
        
        filename = f"fom_2d_{var1['category']}_{var1['var_name']}_vs_{var2['category']}_{var2['var_name']}_{state}.pdf"
        plt.savefig(plot_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_optimization_summary(self, results_df: pd.DataFrame) -> None:
        """
        Generate summary table similar to Table 7 in reference analysis
        
        Shows optimal cuts for each variable across all states
        """
        # Pivot table: variables × states
        pivot = results_df.pivot_table(
            index=['category', 'variable'],
            columns='state',
            values='optimal_cut'
        )
        
        # Add cut type and description
        cut_info = results_df.drop_duplicates(['category', 'variable'])[
            ['category', 'variable', 'cut_type', 'branch_name']
        ].set_index(['category', 'variable'])
        
        summary = pivot.join(cut_info)
        
        # Reorder columns
        col_order = ['branch_name', 'cut_type', 'jpsi', 'etac', 'chic0', 'chic1']
        summary = summary[col_order]
        
        # Save as markdown and CSV
        output_dir = Path(self.config.paths["output"]["tables_dir"])
        
        summary.to_csv(output_dir / "optimized_cuts_summary.csv")
        summary.to_markdown(output_dir / "optimized_cuts_summary.md")
        
        print("\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)
        print(summary.to_string())
        print("\n✓ Saved to:", output_dir)