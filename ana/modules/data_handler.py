import uproot
import awkward as ak
import numpy as np
import pandas as pd
import tomli
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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
        
        # Load branch mapping from Phase 0 output
        self.branch_map = self._load_branch_mapping()
        
    def _load_toml(self, filename: str) -> dict:
        with open(self.config_dir / filename, 'rb') as f:
            return tomli.load(f)
    
    def _load_branch_mapping(self) -> dict:
        """Load branch name mapping from Phase 0 discovery"""
        import json
        with open(self.paths["output"]["docs_dir"] + "/branch_structure.json", 'r') as f:
            return json.load(f)
    
    def get_branch_name(self, logical_name: str) -> str:
        """
        Get actual branch name from logical name
        
        Example: 'bu_mass' → 'Bu_MM'
        """
        return self.branch_map["required_branches"][logical_name]
    
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
                  tree_name: str = "DecayTree") -> ak.Array:
        """
        Load a single ROOT tree
        
        Args:
            particle_type: "data", "Jpsi", "etac", "chic0", "chic1", "KpKm"
            year: 2016, 2017, or 2018
            magnet: "MD" or "MU"
            tree_name: Name of tree in ROOT file
            
        Returns:
            Awkward array with all branches
        """
        if particle_type == "data":
            filename = f"dataBu2L0barPHH_{year-2000}{magnet}.root"
            filepath = self.data_path / filename
        else:
            filename = f"{particle_type}_{year-2000}_{magnet}.root"
            filepath = self.mc_path / particle_type / filename
            
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
        with uproot.open(filepath) as file:
            tree = file[tree_name]
            # Load all branches
            events = tree.arrays()
            
        print(f"✓ Loaded {particle_type} {year}_{magnet}: {len(events)} events")
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
        lambda_mass_branch = self.config.get_branch_name("lambda_mass")
        bu_mass_branch = self.config.get_branch_name("bu_mass")
        
        events["Bu_MM_corrected"] = (
            events[bu_mass_branch] - 
            events[lambda_mass_branch] + 
            lambda_mass_pdg
        )
        
        # 2. Delta_z calculation
        # Get branch names from config
        bu_x = self.config.get_branch_name("bu_endvertex_x")
        bu_y = self.config.get_branch_name("bu_endvertex_y")
        bu_z = self.config.get_branch_name("bu_endvertex_z")
        bu_xerr = self.config.get_branch_name("bu_endvertex_xerr")
        bu_yerr = self.config.get_branch_name("bu_endvertex_yerr")
        bu_zerr = self.config.get_branch_name("bu_endvertex_zerr")
        
        l0_x = self.config.get_branch_name("lambda_endvertex_x")
        l0_y = self.config.get_branch_name("lambda_endvertex_y")
        l0_z = self.config.get_branch_name("lambda_endvertex_z")
        l0_xerr = self.config.get_branch_name("lambda_endvertex_xerr")
        l0_yerr = self.config.get_branch_name("lambda_endvertex_yerr")
        l0_zerr = self.config.get_branch_name("lambda_endvertex_zerr")
        
        Delta_X = events[l0_x] - events[bu_x]
        Delta_Y = events[l0_y] - events[bu_y]
        Delta_Z = events[l0_z] - events[bu_z]
        
        Delta_X_ERR = np.sqrt(events[bu_xerr]**2 + events[l0_xerr]**2)
        Delta_Y_ERR = np.sqrt(events[bu_yerr]**2 + events[l0_yerr]**2)
        Delta_Z_ERR = np.sqrt(events[bu_zerr]**2 + events[l0_zerr]**2)
        
        events["delta_x"] = Delta_X / Delta_X_ERR
        events["delta_y"] = Delta_Y / Delta_Y_ERR
        events["delta_z"] = Delta_Z / Delta_Z_ERR
        
        # IMPORTANT: Also store absolute Delta_Z in mm for cut
        events["Delta_Z_mm"] = Delta_Z  # Should be in mm already
        
        # 3. Charmonium invariant mass M(Λ̄pK⁻)
        lambda_4mom_branches = {
            "px": self.config.get_branch_name("lambda_px"),
            "py": self.config.get_branch_name("lambda_py"),
            "pz": self.config.get_branch_name("lambda_pz"),
            "pe": self.config.get_branch_name("lambda_pe")
        }
        
        bachelor_p_4mom_branches = {
            "px": self.config.get_branch_name("bachelor_p_px"),
            "py": self.config.get_branch_name("bachelor_p_py"),
            "pz": self.config.get_branch_name("bachelor_p_pz"),
            "pe": self.config.get_branch_name("bachelor_p_pe")
        }
        
        kminus_4mom_branches = {
            "px": self.config.get_branch_name("kminus_px"),
            "py": self.config.get_branch_name("kminus_py"),
            "pz": self.config.get_branch_name("kminus_pz"),
            "pe": self.config.get_branch_name("kminus_pe")
        }
        
        events["M_LpKm"] = self.four_momentum_calc.calculate_invariant_mass_lpkm(
            events,
            lambda_4mom_branches,
            bachelor_p_4mom_branches,
            kminus_4mom_branches
        )
        
        return events
    
    def apply_trigger_selection(self, events: ak.Array) -> ak.Array:
        """
        Apply trigger requirements: (L0_TIS) AND (HLT1_TOS) AND (HLT2_TOS)
        Within each level, at least one line must fire (OR)
        """
        # L0 TIS (any line)
        l0_lines = self.config.get_trigger_lines("L0_TIS")
        l0_pass = ak.zeros_like(events[self.config.get_branch_name("bu_mass")], dtype=bool)
        for line in l0_lines:
            if line in ak.fields(events):  # Check branch exists
                l0_pass = l0_pass | (events[line] > 0)
        
        # HLT1 TOS (any line)
        hlt1_lines = self.config.get_trigger_lines("HLT1_TOS")
        hlt1_pass = ak.zeros_like(events[self.config.get_branch_name("bu_mass")], dtype=bool)
        for line in hlt1_lines:
            if line in ak.fields(events):
                hlt1_pass = hlt1_pass | (events[line] > 0)
        
        # HLT2 TOS (any line)
        hlt2_lines = self.config.get_trigger_lines("HLT2_TOS")
        hlt2_pass = ak.zeros_like(events[self.config.get_branch_name("bu_mass")], dtype=bool)
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
                                       particle_type: str) -> Dict[str, ak.Array]:
        """
        Load data combining MagDown + MagUp for each year
        
        Returns:
            Dictionary: {"2016": events, "2017": events, "2018": events}
        """
        data_by_year = {}
        
        for year in self.config.paths["data"]["years"]:
            year_events = []
            
            for magnet in self.config.paths["data"]["magnets"]:
                try:
                    events = self.load_tree(particle_type, year, magnet)
                    events = self.compute_derived_branches(events)
                    events = self.apply_trigger_selection(events)
                    year_events.append(events)
                except FileNotFoundError as e:
                    print(f"✗ {e}")
            
            if year_events:
                # Concatenate MD + MU
                combined = ak.concatenate(year_events)
                data_by_year[str(year)] = combined
                print(f"✓ Combined {particle_type} {year}: {len(combined)} events\n")
        
        return data_by_year