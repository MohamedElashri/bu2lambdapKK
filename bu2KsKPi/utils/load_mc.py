#!/usr/bin/env python3
"""
Load MC data from ROOT files.
"""

import os
import glob
from pathlib import Path
import uproot
import awkward as ak
import numpy as np
from typing import List, Dict, Union, Tuple, Optional, Set

# Constants for available options
DEFAULT_DATA_DIR = "/share/lazy/Mohamed/bu2kskpik/MC/processed/"

# All available decay modes
DECAY_MODES = {
    "B2K0s2PipPimKmPipKp": "B+ → (K0_S → π+π-)K-π+K+",
    "B2Jpsi2K0s2PipPimKmPipKp": "B+ → (J/ψ → (K0_S → π+π-)K-π+)K+",
    "B2Etac2K0s2PipPimKmPipKp": "B+ → (ηc → (K0_S → π+π-)K-π+)K+",
    "B2Etac2S2K0s2PipPimKmPipKp": "B+ → (ηc(2S) → (K0_S → π+π-)K-π+)K+",
    "B2Chic12K0s2PipPimKmPipKp": "B+ → (χc1 → (K0_S → π+π-)K-π+)K+"
}

# All available tuple types
TUPLE_TYPES = [
    "KSKmKpPip_DD",
    "KSKmKpPip_LL",
    "KSKpKpPim_DD",
    "KSKpKpPim_LL"
]

# All available years
AVAILABLE_YEARS = ["2015", "2016", "2017", "2018"]

def load_mc(years: Union[str, List[str]] = "all", 
            decay_modes: Union[str, List[str]] = "all",
            tuple_types: Union[str, List[str]] = "all",
            data_dir: str = None,
            tree_name: str = "DecayTree",
            branches: Optional[List[str]] = None,
            cut: Optional[str] = None,
            verbose: bool = False) -> Dict[str, Dict[str, ak.Array]]:
    """
    Load Monte Carlo data from ROOT files based on specified criteria.
    
    Args:
        years: Year(s) to load. Can be "all", a single year (e.g., "2015"), or a list of years.
        decay_modes: Decay mode(s) to load. Can be "all", a single decay mode, or a list of decay modes.
            Available modes: "B2K0s2PipPimKmPipKp", "B2Jpsi2K0s2PipPimKmPipKp", etc.
        tuple_types: Tuple type(s) to load. Can be "all", a single type, or a list of types.
            Available types: "KSKmKpPip_DD", "KSKmKpPip_LL", "KSKpKpPim_DD", "KSKpKpPim_LL".
        data_dir: Directory containing the merged ROOT files. Required parameter.
        tree_name: Name of the TTree to load in each directory.
        branches: Specific branches to load. If None, all branches are loaded.
        cut: Selection cut to apply. If None, no cut is applied.
        verbose: If True, print detailed information during loading.
    
    Returns:
        Dictionary with structure: {decay_mode: {tuple_type: awkward array}}
        
    Example:
        # Load all 2015 data for a specific decay mode and all tuple types
        data = load_mc(years="2015", decay_modes="B2K0s2PipPimKmPipKp", 
                       data_dir="/path/to/merged/data")
        
        # Access data for a specific tuple type
        events = data["B2K0s2PipPimKmPipKp"]["KSKmKpPip_DD"]
        
        # Load multiple years and decay modes, but only specific tuple types
        data = load_mc(
            years=["2015", "2016"],
            decay_modes=["B2K0s2PipPimKmPipKp", "B2Jpsi2K0s2PipPimKmPipKp"],
            tuple_types=["KSKmKpPip_DD", "KSKmKpPip_LL"],
            data_dir="/path/to/merged/data"
        )
    """
    # Check that data_dir is provided
    if data_dir is None:
        raise ValueError("data_dir parameter is required. Please specify the directory containing the merged ROOT files.")
    # Normalize input parameters
    if years == "all":
        years_to_load = AVAILABLE_YEARS
    elif isinstance(years, str):
        years_to_load = [years]
    else:
        years_to_load = years
    
    if decay_modes == "all":
        decay_modes_to_load = list(DECAY_MODES.keys())
    elif isinstance(decay_modes, str):
        decay_modes_to_load = [decay_modes]
    else:
        decay_modes_to_load = decay_modes
    
    if tuple_types == "all":
        tuple_types_to_load = TUPLE_TYPES
    elif isinstance(tuple_types, str):
        tuple_types_to_load = [tuple_types]
    else:
        tuple_types_to_load = tuple_types
    
    if verbose:
        print(f"Loading Monte Carlo data:")
        print(f"  Years: {years_to_load}")
        print(f"  Decay modes: {decay_modes_to_load}")
        print(f"  Tuple types: {tuple_types_to_load}")
        print(f"  Data directory: {data_dir}")
    
    # Find all matching files
    data_files = {}
    for decay_mode in decay_modes_to_load:
        data_files[decay_mode] = []
        for year in years_to_load:
            pattern = os.path.join(data_dir, f"{year}_{decay_mode}.root")
            matching_files = glob.glob(pattern)
            
            if matching_files:
                data_files[decay_mode].extend(matching_files)
            elif verbose:
                print(f"  Warning: No files found matching pattern {pattern}")
    
    # Load data by decay mode and tuple type
    result = {}
    
    for decay_mode, files in data_files.items():
        if not files:
            if verbose:
                print(f"  No files found for decay mode: {decay_mode}")
            continue
        
        result[decay_mode] = {}
        
        for tuple_type in tuple_types_to_load:
            if verbose:
                print(f"  Loading {decay_mode} - {tuple_type}...")
            
            # Define the path within the ROOT file
            tree_path = f"{tuple_type}/{tree_name}"
            
            # Try to load from each file and concatenate
            arrays_to_concatenate = []
            
            for file_path in files:
                try:
                    with uproot.open(file_path) as file:
                        if tree_path not in file:
                            if verbose:
                                print(f"    Tree path {tree_path} not found in {file_path}")
                            continue
                        
                        # Load the branches with any specified cut
                        array = file[tree_path].arrays(branches, cut, library="ak")
                        
                        if len(array) > 0:
                            arrays_to_concatenate.append(array)
                            if verbose:
                                print(f"    Loaded {len(array)} events from {file_path}")
                        else:
                            if verbose:
                                print(f"    No events passed selection in {file_path}")
                            
                except Exception as e:
                    if verbose:
                        print(f"    Error loading from {file_path}: {str(e)}")
            
            # Concatenate arrays if any were loaded
            if arrays_to_concatenate:
                result[decay_mode][tuple_type] = ak.concatenate(arrays_to_concatenate)
                if verbose:
                    print(f"    Total events loaded for {tuple_type}: {len(result[decay_mode][tuple_type])}")
            else:
                # Create an empty array with the correct structure if requested
                result[decay_mode][tuple_type] = ak.Array([])
                if verbose:
                    print(f"    No events loaded for {tuple_type}")
    
    return result


def get_available_branches(years: Union[str, List[str]] = "2015",
                           decay_modes: str = "B2K0s2PipPimKmPipKp",
                           tuple_types: str = "KSKmKpPip_DD",
                           data_dir: str = None,
                           tree_name: str = "DecayTree") -> Set[str]:
    """
    Get the set of available branches for the specified data.
    
    Args:
        years: Year(s) to check. Can be a single year or a list of years.
        decay_modes: Decay mode to check.
        tuple_types: Tuple type to check.
        data_dir: Directory containing the merged ROOT files. Required parameter.
        tree_name: Name of the TTree to check in each directory.
    
    Returns:
        Set of available branch names.
    """
    # Check that data_dir is provided
    if data_dir is None:
        raise ValueError("data_dir parameter is required. Please specify the directory containing the merged ROOT files.")
        
    if isinstance(years, list):
        years = years[0]  # Just use the first year to check branches
    
    # Find a matching file
    file_pattern = os.path.join(data_dir, f"{years}_{decay_modes}.root")
    matching_files = glob.glob(file_pattern)
    
    if not matching_files:
        print(f"No files found matching pattern: {file_pattern}")
        return set()
    
    file_path = matching_files[0]
    tree_path = f"{tuple_types}/{tree_name}"
    
    try:
        with uproot.open(file_path) as file:
            if tree_path not in file:
                print(f"Tree path {tree_path} not found in {file_path}")
                return set()
            
            return set(file[tree_path].keys())
    except Exception as e:
        print(f"Error reading branches from {file_path}: {str(e)}")
        return set()


def get_decay_mode_description(decay_mode: str) -> str:
    """Get the human-readable description of a decay mode."""
    return DECAY_MODES.get(decay_mode, "Unknown decay mode")


