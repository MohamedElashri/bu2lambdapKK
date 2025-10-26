"""
Module for loading data from ROOT files using uproot
"""

import logging
import uproot
from pathlib import Path

class DataLoader:
    """Class for loading B+ → pK⁻Λ̄ K+ data from ROOT files"""
    
    def __init__(self, data_dir):
        """Initialize with data directory path"""
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger("Bu2LambdaPKK.DataLoader")
        
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
    
    def load_data(self, years, polarities, track_types, channel_name):
        """
        Load data for specified years, polarities, and track types
        
        Parameters:
        - years: List of years ['16', '17', '18']
        - polarities: List of polarities ['MD', 'MU']
        - track_types: List of track types ['LL', 'DD']
        - channel_name: Name of the decay channel (e.g., 'B2L0barPKpKm')
        
        Returns:
        - Dictionary with data arrays
        """
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
                                
                                # Store with a meaningful key
                                key = f"{year}_{polarity}_{track_type}"
                                self.logger.info(f"Loading {key}...")
                                
                                # Read all branches into memory
                                data[key] = tree.arrays()
                                
                                self.logger.info(f"Loaded {len(data[key])} events for {key}")
                                
                            except Exception as e:
                                self.logger.error(f"Error loading {tree_path}: {e}")
                        else:
                            self.logger.warning(f"Channel {channel_path} not found in {file_path}")
        
        return data