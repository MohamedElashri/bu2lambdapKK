import uproot
import awkward as ak
import numpy as np
import pandas as pd
import tomli
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Import BranchConfig from local modules (since files are now copied to ana/)
from branch_config import BranchConfig

import vector

# Register vector behavior for 4-momentum calculations
vector.register_awkward()

class TOMLConfig:
    """Load and manage all TOML configuration files"""
    
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.paths = self._load_toml("paths.toml")
        self.particles = self._load_toml("particles.toml")
        self.branching_fractions = self._load_toml("branching_fractions.toml")
        self.luminosity = self._load_toml("luminosity.toml")
        self.triggers = self._load_toml("triggers.toml")
        self.efficiency_inputs = self._load_toml("efficiency_inputs.toml")
        self.selection = self._load_toml("selection.toml")
        
        # Use BranchConfig - look for branches_config.toml in ana/modules/
        modules_dir = Path(__file__).parent
        branches_config_path = modules_dir / "branches_config.toml"
        self.branch_config = BranchConfig(str(branches_config_path))
        
    def _load_toml(self, filename: str) -> dict:
        with open(self.config_dir / filename, 'rb') as f:
            return tomli.load(f)
    
    def get_pdg_mass(self, particle: str) -> float:
        return self.particles["pdg_masses"][particle]
    
    def get_signal_region(self, state: str) -> Tuple[float, float]:
        """Returns (center, window) for signal region"""
        region = self.particles["signal_regions"][state.lower()]
        center = region["center"]
        window = region["window"]
        return (center - window, center + window)
    
    def get_trigger_lines(self, level: str) -> List[str]:
        """Returns list of trigger lines for L0_TIS, HLT1_TOS, or HLT2_TOS"""
        return self.triggers[level]["lines"]
    
    def get_lambda_cuts(self) -> dict:
        """Get fixed Lambda selection cuts"""
        return self.selection["lambda_selection"]
    
    def get_bu_fixed_cuts(self) -> dict:
        """Get fixed B+ selection cuts"""
        return self.selection["bu_fixed_selection"]
    
    def get_optimizable_cuts(self, category: str) -> dict:
        """
        Get optimization ranges for a category
        
        category: 'bu', 'bachelor_p', 'kplus', 'kminus'
        Returns: {variable: {begin, end, step, cut_type}}
        """
        key = f"{category}_optimizable_selection"
        return self.selection[key]

class FourMomentumCalculator:
    """
    Calculate invariant masses from 4-momenta
    Critical for M(Λ̄pK⁻) charmonium mass
    """
    
    @staticmethod
    def calculate_invariant_mass_lpkm(events: ak.Array,
                                      lambda_branches: Dict[str, str],
                                      bachelor_p_branches: Dict[str, str],
                                      kminus_branches: Dict[str, str]) -> ak.Array:
        """
        Calculate M(Λ̄ p̄ K⁻) invariant mass
        
        Args:
            events: Awkward array with all branches
            lambda_branches: {'px': 'L0_PX', 'py': 'L0_PY', ...}
            bachelor_p_branches: {'px': '???', 'py': '???', ...}
            kminus_branches: {'px': '???', 'py': '???', ...}
        
        Returns:
            Array of invariant masses in MeV/c²
        """
        # Build 4-vectors using awkward-vector
        lambda_4mom = vector.zip({
            "px": events[lambda_branches["px"]],
            "py": events[lambda_branches["py"]],
            "pz": events[lambda_branches["pz"]],
            "E": events[lambda_branches["pe"]]
        })
        
        bachelor_p_4mom = vector.zip({
            "px": events[bachelor_p_branches["px"]],
            "py": events[bachelor_p_branches["py"]],
            "pz": events[bachelor_p_branches["pz"]],
            "E": events[bachelor_p_branches["pe"]]
        })
        
        kminus_4mom = vector.zip({
            "px": events[kminus_branches["px"]],
            "py": events[kminus_branches["py"]],
            "pz": events[kminus_branches["pz"]],
            "E": events[kminus_branches["pe"]]
        })
        
        # Sum 4-vectors
        total_4mom = lambda_4mom + bachelor_p_4mom + kminus_4mom
        
        # Return invariant mass
        return total_4mom.mass

class DataManager:
    """Load and manage data/MC ROOT files"""
    
    def __init__(self, config: TOMLConfig):
        self.config = config
        self.data_path = Path(config.paths["data"]["base_path"])
        self.mc_path = Path(config.paths["mc"]["base_path"])
        self.four_momentum_calc = FourMomentumCalculator()
        
    def load_tree(self, 
                  particle_type: str,
                  year: int,
                  magnet: str,
                  track_type: str = "LL",
                  channel_name: str = "B2L0barPKpKm") -> ak.Array:
        """
        Load a single ROOT tree using proper data structure
        
        Args:
            particle_type: "data", "Jpsi", "etac", "chic0", "chic1", "KpKm"
            year: 2016, 2017, or 2018
            magnet: "MD" or "MU"
            track_type: "LL" or "DD" (Lambda reconstruction category)
            channel_name: Name of decay channel in ROOT file
            
        Returns:
            Awkward array with all branches
        """
        is_mc = particle_type != "data"
        
        # Build file path
        if particle_type == "data":
            filename = f"dataBu2L0barPHH_{year-2000}{magnet}.root"
            filepath = self.data_path / filename
        else:
            filename = f"{particle_type}_{year-2000}_{magnet}.root"
            filepath = self.mc_path / particle_type / filename
            
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Build tree path: channel_LL/DecayTree or channel_DD/DecayTree
        channel_path = f"{channel_name}_{track_type}"
        tree_path = f"{channel_path}/DecayTree"
        
        with uproot.open(filepath) as file:
            if channel_path not in file:
                raise ValueError(f"Channel {channel_path} not found in {filepath}")
            
            tree = file[tree_path]
            
            # Get branches we need using BranchConfig
            load_branches = self.config.branch_config.get_branches_from_preset(
                "standard", exclude_mc=not is_mc
            )
            
            # Resolve aliases to actual branch names
            resolved_branches = self.config.branch_config.resolve_aliases(
                load_branches, is_mc=is_mc
            )
            
            # Validate and load
            available_branches = list(tree.keys())
            validation = self.config.branch_config.validate_branches(
                resolved_branches, available_branches
            )
            
            if validation['missing']:
                print(f"⚠️  Missing {len(validation['missing'])} branches in {channel_path}")
            
            # Load valid branches
            events = tree.arrays(validation['valid'], library='ak')
            
            # Normalize branch names back to common names
            rename_map = self.config.branch_config.normalize_branches(
                validation['valid'], is_mc=is_mc
            )
            if rename_map:
                for old_name, new_name in rename_map.items():
                    events = ak.with_field(events, events[old_name], new_name)
                    events = ak.without_field(events, old_name)
            
        print(f"✓ Loaded {particle_type} {year}_{magnet}_{track_type}: {len(events)} events")
        return events
    
    def compute_derived_branches(self, events: ak.Array) -> ak.Array:
        """
        Compute derived quantities not directly in trees
        
        Critical derived branches:
        1. Bu_MM_corrected: Lambda-corrected B+ mass
        2. delta_z: Significance of Lambda-B vertex separation in Z
        3. M_LpKm: Invariant mass of Λ̄pK⁻ (charmonium candidate)
        """
        # 1. Lambda-corrected B+ mass
        lambda_mass_pdg = self.config.get_pdg_mass("lambda")
        
        # Use actual branch names (these are normalized to common names by BranchConfig)
        if "L0_MM" in events.fields and "Bu_MM" in events.fields:
            events = ak.with_field(
                events,
                events["Bu_MM"] - events["L0_MM"] + lambda_mass_pdg,
                "Bu_MM_corrected"
            )
        
        # 2. Delta_z calculation
        # Branch names after normalization
        if all(b in events.fields for b in [
            "Bu_ENDVERTEX_X", "Bu_ENDVERTEX_Y", "Bu_ENDVERTEX_Z",
            "Bu_ENDVERTEX_XERR", "Bu_ENDVERTEX_YERR", "Bu_ENDVERTEX_ZERR",
            "L0_ENDVERTEX_X", "L0_ENDVERTEX_Y", "L0_ENDVERTEX_Z",
            "L0_ENDVERTEX_XERR", "L0_ENDVERTEX_YERR", "L0_ENDVERTEX_ZERR"
        ]):
            Delta_X = events["L0_ENDVERTEX_X"] - events["Bu_ENDVERTEX_X"]
            Delta_Y = events["L0_ENDVERTEX_Y"] - events["Bu_ENDVERTEX_Y"]
            Delta_Z = events["L0_ENDVERTEX_Z"] - events["Bu_ENDVERTEX_Z"]
            
            Delta_X_ERR = np.sqrt(events["Bu_ENDVERTEX_XERR"]**2 + events["L0_ENDVERTEX_XERR"]**2)
            Delta_Y_ERR = np.sqrt(events["Bu_ENDVERTEX_YERR"]**2 + events["L0_ENDVERTEX_YERR"]**2)
            Delta_Z_ERR = np.sqrt(events["Bu_ENDVERTEX_ZERR"]**2 + events["L0_ENDVERTEX_ZERR"]**2)
            
            events = ak.with_field(events, Delta_X / Delta_X_ERR, "delta_x")
            events = ak.with_field(events, Delta_Y / Delta_Y_ERR, "delta_y")
            events = ak.with_field(events, Delta_Z / Delta_Z_ERR, "delta_z")
            
            # IMPORTANT: Also store absolute Delta_Z in mm for cut
            events = ak.with_field(events, Delta_Z, "Delta_Z_mm")
        
        # 3. Charmonium invariant mass M(Λ̄pK⁻)
        # Need to identify which hadron is K- (h1 or h2)
        # For now, assume we need to calculate M(L0 + p + h1) and M(L0 + p + h2)
        # and pick the right combination based on charge or mass
        
        if all(b in events.fields for b in [
            "L0_PX", "L0_PY", "L0_PZ", "L0_PE",
            "p_PX", "p_PY", "p_PZ", "p_PE",
            "h1_PX", "h1_PY", "h1_PZ", "h1_PE",
            "h2_PX", "h2_PY", "h2_PZ", "h2_PE"
        ]):
            # Calculate invariant mass for both combinations
            # M(L0 + p + h1)
            lambda_4mom_branches = {"px": "L0_PX", "py": "L0_PY", "pz": "L0_PZ", "pe": "L0_PE"}
            bachelor_p_4mom_branches = {"px": "p_PX", "py": "p_PY", "pz": "p_PZ", "pe": "p_PE"}
            h1_4mom_branches = {"px": "h1_PX", "py": "h1_PY", "pz": "h1_PZ", "pe": "h1_PE"}
            h2_4mom_branches = {"px": "h2_PX", "py": "h2_PY", "pz": "h2_PZ", "pe": "h2_PE"}
            
            M_Lp_h1 = self.four_momentum_calc.calculate_invariant_mass_lpkm(
                events,
                lambda_4mom_branches,
                bachelor_p_4mom_branches,
                h1_4mom_branches
            )
            
            M_Lp_h2 = self.four_momentum_calc.calculate_invariant_mass_lpkm(
                events,
                lambda_4mom_branches,
                bachelor_p_4mom_branches,
                h2_4mom_branches
            )
            
            # Store both combinations
            events = ak.with_field(events, M_Lp_h1, "M_LpKm_h1")
            events = ak.with_field(events, M_Lp_h2, "M_LpKm_h2")
            
            # Also store the opposite combination M(L0 + h1 + h2) for K+K- mass
            h1_vec = vector.zip({"px": events["h1_PX"], "py": events["h1_PY"], 
                                "pz": events["h1_PZ"], "E": events["h1_PE"]})
            h2_vec = vector.zip({"px": events["h2_PX"], "py": events["h2_PY"],
                                "pz": events["h2_PZ"], "E": events["h2_PE"]})
            M_KK = (h1_vec + h2_vec).mass
            events = ak.with_field(events, M_KK, "M_KK")
        
        return events
    
    def apply_trigger_selection(self, events: ak.Array) -> ak.Array:
        """
        Apply trigger requirements: (L0_TIS) AND (HLT1_TOS) AND (HLT2_TOS)
        Within each level, at least one line must fire (OR)
        """
        # Use a reference branch that always exists to create mask
        if "Bu_M" not in events.fields:
            print("  ⚠️  Cannot apply trigger selection: Bu_M not found")
            return events
        
        # L0 TIS (any line)
        l0_lines = self.config.get_trigger_lines("L0_TIS")
        l0_pass = ak.zeros_like(events["Bu_M"], dtype=bool)
        for line in l0_lines:
            if line in ak.fields(events):  # Check branch exists
                l0_pass = l0_pass | (events[line] > 0)
        
        # HLT1 TOS (any line)
        hlt1_lines = self.config.get_trigger_lines("HLT1_TOS")
        hlt1_pass = ak.zeros_like(events["Bu_M"], dtype=bool)
        for line in hlt1_lines:
            if line in ak.fields(events):
                hlt1_pass = hlt1_pass | (events[line] > 0)
        
        # HLT2 TOS (any line)
        hlt2_lines = self.config.get_trigger_lines("HLT2_TOS")
        hlt2_pass = ak.zeros_like(events["Bu_M"], dtype=bool)
        for line in hlt2_lines:
            if line in ak.fields(events):
                hlt2_pass = hlt2_pass | (events[line] > 0)
        
        # Combine: all three levels must pass
        trigger_pass = l0_pass & hlt1_pass & hlt2_pass
        
        n_before = len(events)
        n_after = ak.sum(trigger_pass)
        print(f"  Trigger: {n_before} → {n_after} ({100*n_after/n_before:.1f}%)")
        
        return events[trigger_pass]
    
    def load_all_data_combined_magnets(self, 
                                       particle_type: str,
                                       track_types: List[str] = ["LL", "DD"]) -> Dict[str, ak.Array]:
        """
        Load data combining MagDown + MagUp and track types for each year
        
        Args:
            particle_type: "data", "Jpsi", "etac", etc.
            track_types: List of Lambda track types (default ["LL", "DD"])
        
        Returns:
            Dictionary: {"2016": events, "2017": events, "2018": events}
        """
        data_by_year = {}
        
        for year in self.config.paths["data"]["years"]:
            year_events = []
            
            for magnet in self.config.paths["data"]["magnets"]:
                for track_type in track_types:
                    try:
                        events = self.load_tree(particle_type, year, magnet, track_type)
                        events = self.compute_derived_branches(events)
                        events = self.apply_trigger_selection(events)
                        year_events.append(events)
                    except (FileNotFoundError, ValueError) as e:
                        print(f"✗ {e}")
            
            if year_events:
                # Concatenate all combinations
                combined = ak.concatenate(year_events)
                data_by_year[str(year)] = combined
                print(f"✓ Combined {particle_type} {year}: {len(combined)} events\n")
        
        return data_by_year