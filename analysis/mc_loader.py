"""
Module for loading MC data from ROOT files using uproot

This module handles both:
1. Reconstructed MC data (DecayTree) - similar to real data but with MC truth branches
2. MC Truth data (MCDecayTree) - generator-level information
"""

import logging
import uproot
from pathlib import Path
from typing import Dict, List, Optional

class MCLoader:
    """Class for loading B+ → pK⁻Λ̄ K+ Monte Carlo data from ROOT files"""
    
    # Mapping of MC sample names to their MC truth tree names
    MC_TREE_MAP = {
        'KpKm': 'MC_B2L0barPKpKm',      # Signal: B+ → Λ̄ p K+ K-
        'Jpsi_SS': 'MC_B2JpsiK_SS',     # J/ψ same-sign
        'Jpsi_OS': 'MC_B2JpsiK_OS',     # J/ψ opposite-sign
        'chic0': 'MC_B2chic0K',         # χc0
        'chic1': 'MC_B2chic1K',         # χc1
        'chic2': 'MC_B2chic2K',         # χc2
        'etac': 'MC_B2etacK',           # ηc
    }
    
    def __init__(self, mc_dir: str):
        """
        Initialize with MC directory path
        
        Parameters:
        - mc_dir: Path to directory containing MC files
        """
        self.mc_dir = Path(mc_dir)
        self.logger = logging.getLogger("Bu2LambdaPKK.MCLoader")
        
    def _find_mc_files(self, sample_name: str, years: List[str], 
                      polarities: List[str]) -> Dict:
        """
        Find MC files for specified sample, years, and polarities
        
        Parameters:
        - sample_name: Name of MC sample (e.g., 'KpKm', 'Jpsi', 'etac')
        - years: List of years ['16', '17', '18']
        - polarities: List of polarities ['MD', 'MU']
        
        Returns:
        - Dictionary with file paths organized by year and polarity
        """
        files = {}
        sample_dir = self.mc_dir / sample_name
        
        if not sample_dir.exists():
            self.logger.warning(f"MC sample directory not found: {sample_dir}")
            return files
            
        for year in years:
            files[year] = {}
            for polarity in polarities:
                # MC files use format: sample_year_polarity.root
                file_path = sample_dir / f"{sample_name}_{year}_{polarity}.root"
                if file_path.exists():
                    files[year][polarity] = file_path
                else:
                    self.logger.warning(f"File not found: {file_path}")
        return files
    
    def load_reconstructed(self, sample_name: str, years: List[str], 
                          polarities: List[str], track_types: List[str], 
                          channel_name: str, 
                          branches: Optional[List[str]] = None) -> Dict:
        """
        Load reconstructed MC data (similar to real data but with MC truth info)
        
        Parameters:
        - sample_name: Name of MC sample (e.g., 'KpKm', 'Jpsi')
        - years: List of years ['16', '17', '18']
        - polarities: List of polarities ['MD', 'MU']
        - track_types: List of track types ['LL', 'DD']
        - channel_name: Name of the decay channel (e.g., 'B2L0barPKpKm')
        - branches: Optional list of branches to load (loads all if None)
        
        Returns:
        - Dictionary with reconstructed MC data arrays
        """
        mc_files = self._find_mc_files(sample_name, years, polarities)
        data = {}
        
        for year, year_files in mc_files.items():
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
                                self.logger.info(f"Loading reconstructed MC {key}...")
                                
                                # Read specified branches or all branches
                                if branches:
                                    data[key] = tree.arrays(branches, library='ak')
                                else:
                                    data[key] = tree.arrays(library='ak')
                                
                                self.logger.info(
                                    f"Loaded {len(data[key])} reconstructed events for {key}"
                                )
                                
                            except Exception as e:
                                self.logger.error(f"Error loading {tree_path}: {e}")
                        else:
                            self.logger.warning(
                                f"Channel {channel_path} not found in {file_path}"
                            )
        
        return data
    
    def load_truth(self, sample_name: str, years: List[str], 
                   polarities: List[str], 
                   mc_tree_name: Optional[str] = None,
                   branches: Optional[List[str]] = None) -> Dict:
        """
        Load MC truth data (generator-level information)
        
        Parameters:
        - sample_name: Name of MC sample (e.g., 'KpKm', 'Jpsi')
        - years: List of years ['16', '17', '18']
        - polarities: List of polarities ['MD', 'MU']
        - mc_tree_name: Name of MC truth tree (auto-detected if None)
        - branches: Optional list of branches to load (loads all if None)
        
        Returns:
        - Dictionary with MC truth data arrays
        """
        mc_files = self._find_mc_files(sample_name, years, polarities)
        data = {}
        
        # Auto-detect MC tree name if not provided
        if mc_tree_name is None:
            mc_tree_name = self.MC_TREE_MAP.get(sample_name.split('_')[0])
            if mc_tree_name is None:
                self.logger.error(
                    f"Cannot auto-detect MC tree for sample '{sample_name}'. "
                    f"Please specify mc_tree_name explicitly."
                )
                return data
        
        for year, year_files in mc_files.items():
            for polarity, file_path in year_files.items():
                self.logger.info(f"Processing {file_path}")
                
                with uproot.open(file_path) as file:
                    tree_path = f"{mc_tree_name}/MCDecayTree"
                    
                    if mc_tree_name in file:
                        try:
                            tree = file[tree_path]
                            
                            # Store with a meaningful key
                            key = f"{year}_{polarity}"
                            self.logger.info(f"Loading MC truth {key}...")
                            
                            # Read specified branches or all branches
                            if branches:
                                data[key] = tree.arrays(branches, library='ak')
                            else:
                                data[key] = tree.arrays(library='ak')
                            
                            self.logger.info(
                                f"Loaded {len(data[key])} truth events for {key}"
                            )
                            
                        except Exception as e:
                            self.logger.error(f"Error loading {tree_path}: {e}")
                    else:
                        self.logger.warning(
                            f"MC tree {mc_tree_name} not found in {file_path}"
                        )
        
        return data
    
    def load_both(self, sample_name: str, years: List[str], 
                  polarities: List[str], track_types: List[str],
                  channel_name: str,
                  reco_branches: Optional[List[str]] = None,
                  truth_branches: Optional[List[str]] = None) -> Dict:
        """
        Load both reconstructed and truth MC data
        
        Parameters:
        - sample_name: Name of MC sample (e.g., 'KpKm', 'Jpsi')
        - years: List of years ['16', '17', '18']
        - polarities: List of polarities ['MD', 'MU']
        - track_types: List of track types ['LL', 'DD']
        - channel_name: Name of the decay channel (e.g., 'B2L0barPKpKm')
        - reco_branches: Optional list of branches for reconstructed data
        - truth_branches: Optional list of branches for truth data
        
        Returns:
        - Dictionary with both 'reconstructed' and 'truth' data
        """
        self.logger.info(f"Loading both reconstructed and truth data for {sample_name}")
        
        return {
            'reconstructed': self.load_reconstructed(
                sample_name, years, polarities, track_types, 
                channel_name, reco_branches
            ),
            'truth': self.load_truth(
                sample_name, years, polarities, 
                branches=truth_branches
            )
        }
    
    def list_available_samples(self) -> List[str]:
        """
        List all available MC samples in the MC directory
        
        Returns:
        - List of sample names
        """
        if not self.mc_dir.exists():
            self.logger.warning(f"MC directory not found: {self.mc_dir}")
            return []
        
        samples = [d.name for d in self.mc_dir.iterdir() if d.is_dir()]
        self.logger.info(f"Available MC samples: {samples}")
        return samples
    
    def get_mc_tree_name(self, sample_name: str) -> Optional[str]:
        """
        Get the MC truth tree name for a given sample
        
        Parameters:
        - sample_name: Name of MC sample
        
        Returns:
        - MC tree name or None if not found
        """
        # Handle both 'Jpsi' and 'Jpsi_SS' formats
        base_name = sample_name.split('_')[0]
        return self.MC_TREE_MAP.get(base_name)
