"""
Branch configuration manager

Handles loading and parsing branch configurations from branches_config.toml
"""

import tomli
from pathlib import Path
from typing import List, Dict, Set, Optional
import logging


class BranchConfig:
    """Manager for branch configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize branch configuration
        
        Parameters:
        - config_path: Path to branches_config.toml (auto-detected if None)
        """
        self.logger = logging.getLogger("Bu2LambdaPKK.BranchConfig")
        
        # Auto-detect config file
        if config_path is None:
            config_path = Path(__file__).parent / "branches_config.toml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Branch config not found: {config_path}")
        
        # Load configuration
        with open(config_path, 'rb') as f:
            self.config = tomli.load(f)
        
        self.logger.info(f"Loaded branch configuration from {config_path}")
    
    def get_branches_from_sets(self, sets: List[str], 
                               exclude_mc: bool = False) -> List[str]:
        """
        Get all branches from specified sets
        
        Parameters:
        - sets: List of set names (e.g., ['essential', 'kinematics'])
        - exclude_mc: If True, exclude MC-only branches
        
        Returns:
        - Flattened list of unique branch names
        """
        branches = set()
        
        for set_name in sets:
            if set_name not in self.config['branches']:
                self.logger.warning(f"Branch set '{set_name}' not found in config")
                continue
            
            branch_set = self.config['branches'][set_name]
            
            # Check if this is MC-only and we should skip it
            if exclude_mc and branch_set.get('mc_only', False):
                self.logger.info(f"Skipping MC-only set: {set_name}")
                continue
            
            # Add branches from all particles in this set
            for particle, particle_branches in branch_set.items():
                if particle == 'description' or particle == 'mc_only':
                    continue
                if isinstance(particle_branches, list):
                    branches.update(particle_branches)
        
        return sorted(list(branches))
    
    def get_branches_from_preset(self, preset: str, 
                                 exclude_mc: bool = False) -> List[str]:
        """
        Get branches from a preset configuration
        
        Parameters:
        - preset: Preset name (e.g., 'minimal', 'standard', 'mc_reco')
        - exclude_mc: If True, exclude MC-only branches
        
        Returns:
        - List of branch names
        """
        if 'presets' not in self.config:
            raise ValueError("No presets defined in config")
        
        if preset not in self.config['presets']:
            available = list(self.config['presets'].keys())
            raise ValueError(
                f"Preset '{preset}' not found. Available: {available}"
            )
        
        sets = self.config['presets'][preset]
        return self.get_branches_from_sets(sets, exclude_mc=exclude_mc)
    
    def get_truth_branches(self, particle: Optional[str] = None) -> List[str]:
        """
        Get MC truth branches from MCDecayTree
        
        Parameters:
        - particle: Specific particle name (e.g., 'Bplus', 'Kplus')
                   If None, returns all truth branches
        
        Returns:
        - List of truth branch names
        """
        if 'truth_branches' not in self.config:
            return []
        
        if particle is None:
            # Return all truth branches
            branches = []
            for part, part_branches in self.config['truth_branches'].items():
                if part == 'description':
                    continue
                if isinstance(part_branches, list):
                    branches.extend(part_branches)
            return branches
        else:
            # Return branches for specific particle
            return self.config['truth_branches'].get(particle, [])
    
    def get_default_load_sets(self) -> List[str]:
        """
        Get the default sets to load from config
        
        Returns:
        - List of set names
        """
        return self.config['branches'].get('load_sets', [])
    
    def list_available_sets(self) -> List[str]:
        """List all available branch sets"""
        return [k for k in self.config['branches'].keys() 
                if k != 'load_sets']
    
    def list_available_presets(self) -> List[str]:
        """List all available presets"""
        return list(self.config.get('presets', {}).keys())
    
    def get_branches_by_particle(self, particle: str, sets: List[str],
                                exclude_mc: bool = False) -> List[str]:
        """
        Get branches for a specific particle from specified sets
        
        Parameters:
        - particle: Particle name (e.g., 'Bu', 'L0', 'h1')
        - sets: List of set names
        - exclude_mc: If True, exclude MC-only branches
        
        Returns:
        - List of branch names for that particle
        """
        branches = []
        
        for set_name in sets:
            if set_name not in self.config['branches']:
                continue
            
            branch_set = self.config['branches'][set_name]
            
            # Check if this is MC-only and we should skip it
            if exclude_mc and branch_set.get('mc_only', False):
                continue
            
            if particle in branch_set:
                particle_branches = branch_set[particle]
                if isinstance(particle_branches, list):
                    branches.extend(particle_branches)
        
        return branches
    
    def validate_branches(self, branches: List[str], 
                         available_branches: List[str]) -> Dict:
        """
        Validate that requested branches exist in the file
        
        Parameters:
        - branches: List of requested branch names
        - available_branches: List of available branches from ROOT file
        
        Returns:
        - Dictionary with 'valid' and 'missing' lists
        """
        available_set = set(available_branches)
        requested_set = set(branches)
        
        valid = sorted(list(requested_set & available_set))
        missing = sorted(list(requested_set - available_set))
        
        if missing:
            self.logger.warning(
                f"Missing {len(missing)} requested branches: {missing[:5]}..."
            )
        
        return {
            'valid': valid,
            'missing': missing,
            'found': len(valid),
            'total_requested': len(branches)
        }


def get_branch_config(config_path: Optional[str] = None) -> BranchConfig:
    """
    Convenience function to get a BranchConfig instance
    
    Parameters:
    - config_path: Path to configuration file
    
    Returns:
    - BranchConfig instance
    """
    return BranchConfig(config_path)
