"""
Module for loading data from ROOT files using uproot
"""

import logging
import uproot
from pathlib import Path
from typing import List, Optional, Dict

try:
    from .branch_config import BranchConfig
except ImportError:
    from branch_config import BranchConfig

class DataLoader:
    """Class for loading B+ → pK⁻Λ̄ K+ data from ROOT files"""
    
    def __init__(self, data_dir: str, branch_config: Optional[BranchConfig] = None):
        """
        Initialize with data directory path
        
        Parameters:
        - data_dir: Path to directory containing data files
        - branch_config: BranchConfig instance (created automatically if None)
        """
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger("Bu2LambdaPKK.DataLoader")
        
        # Initialize branch configuration
        if branch_config is None:
            self.branch_config = BranchConfig()
        else:
            self.branch_config = branch_config
        
    def _find_data_files(self, years, polarities):
        """Find data files for specified years and polarities"""
        files = {}
        for year in years:
            files[year] = {}
            for polarity in polarities:
                file_path = self.data_dir / f"dataBu2L0barPHH_{year}{polarity}.root"
                if file_path.exists():
                    files[year][polarity] = file_path
                else:
                    self.logger.warning(f"File not found: {file_path}")
        return files
    
    def load_data(self, years: List[str], polarities: List[str], 
                  track_types: List[str], channel_name: str,
                  branches: Optional[List[str]] = None,
                  preset: Optional[str] = None,
                  branch_sets: Optional[List[str]] = None,
                  use_aliases: bool = True) -> Dict:
        """
        Load data for specified years, polarities, and track types
        
        Parameters:
        - years: List of years ['16', '17', '18']
        - polarities: List of polarities ['MD', 'MU']
        - track_types: List of track types ['LL', 'DD']
        - channel_name: Name of the decay channel (e.g., 'B2L0barPKpKm')
        - branches: Explicit list of branches to load (overrides other options)
        - preset: Preset name from config (e.g., 'minimal', 'standard')
        - branch_sets: List of branch sets to load (e.g., ['essential', 'kinematics'])
        - use_aliases: If True, resolve common names to data-specific names and
                       normalize loaded branches back to common names
        
        If none of branches/preset/branch_sets specified, loads default sets from config.
        
        Returns:
        - Dictionary with data arrays
        """
        # Determine which branches to load
        if branches is not None:
            load_branches = branches
            self.logger.info(f"Loading {len(branches)} explicitly specified branches")
        elif preset is not None:
            load_branches = self.branch_config.get_branches_from_preset(
                preset, exclude_mc=True
            )
            self.logger.info(f"Using preset '{preset}': {len(load_branches)} branches")
        elif branch_sets is not None:
            load_branches = self.branch_config.get_branches_from_sets(
                branch_sets, exclude_mc=True
            )
            self.logger.info(
                f"Using branch sets {branch_sets}: {len(load_branches)} branches"
            )
        else:
            # Load default sets from config
            default_sets = self.branch_config.get_default_load_sets()
            load_branches = self.branch_config.get_branches_from_sets(
                default_sets, exclude_mc=True
            )
            self.logger.info(
                f"Using default sets {default_sets}: {len(load_branches)} branches"
            )
        
        # Resolve aliases to actual data branch names
        if use_aliases:
            load_branches = self.branch_config.resolve_aliases(
                load_branches, is_mc=False
            )
        
        data_files = self._find_data_files(years, polarities)
        data = {}
        
        for year, year_files in data_files.items():
            for polarity, file_path in year_files.items():
                self.logger.info(f"Processing {file_path}")
                
                with uproot.open(file_path) as file:
                    for track_type in track_types:
                        channel_path = f"{channel_name}_{track_type}"
                        tree_path = f"{channel_path}/DecayTree"
                        
                        if channel_path in file:
                            try:
                                tree = file[tree_path]
                                
                                # Validate branches
                                available_branches = list(tree.keys())
                                validation = self.branch_config.validate_branches(
                                    load_branches, available_branches
                                )
                                
                                if validation['missing']:
                                    self.logger.warning(
                                        f"Missing {len(validation['missing'])} branches "
                                        f"in {channel_path}"
                                    )
                                
                                # Store with a meaningful key
                                key = f"{year}_{polarity}_{track_type}"
                                self.logger.info(
                                    f"Loading {key} with {validation['found']} branches..."
                                )
                                
                                # Read specified branches into memory
                                if validation['valid']:
                                    events = tree.arrays(
                                        validation['valid'], library='ak'
                                    )
                                    
                                    # Normalize branch names if using aliases
                                    if use_aliases:
                                        rename_map = self.branch_config.normalize_branches(
                                            validation['valid'], is_mc=False
                                        )
                                        if rename_map:
                                            import awkward as ak
                                            # Rename branches from data names to common names
                                            for old_name, new_name in rename_map.items():
                                                events = ak.with_field(
                                                    events, events[old_name], new_name
                                                )
                                                # Remove old field
                                                events = ak.without_field(events, old_name)
                                            self.logger.info(
                                                f"Normalized {len(rename_map)} branch names to common names"
                                            )
                                    
                                    data[key] = events
                                    self.logger.info(
                                        f"Loaded {len(data[key])} events for {key}"
                                    )
                                else:
                                    self.logger.error(
                                        f"No valid branches to load for {key}"
                                    )
                                
                            except Exception as e:
                                self.logger.error(f"Error loading {tree_path}: {e}")
                        else:
                            self.logger.warning(
                                f"Channel {channel_path} not found in {file_path}"
                            )
        
        return data