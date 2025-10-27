#!/usr/bin/env python3
"""
Load real data from ROOT files with specific structure for Ks decay modes.
"""

import os
import glob
from pathlib import Path
import uproot
import awkward as ak
import numpy as np
from typing import List, Dict, Union, Tuple, Optional, Set

# Constants for available options
DEFAULT_DATA_DIR = "/share/lazy/Mohamed/bu2kskpik/RD/compressed/"

# All available tuple types
TUPLE_TYPES = [
    "KSKmKpPip",
    "KSKpKpPim"
]

# All available Ks reconstruction categories
KS_CATEGORIES = [
    "LL",  # Long-Long
    "DD"   # Downstream-Downstream
]

# All available years
AVAILABLE_YEARS = ["2015", "2016", "2017", "2018"]

# All available magnet polarities
AVAILABLE_POLARITIES = ["magup", "magdown"]

def load_data(years: Union[str, List[str]] = "all", 
              tuple_types: Union[str, List[str]] = "all",
              ks_categories: Union[str, List[str]] = "all",
              polarities: Union[str, List[str]] = "all",
              data_dir: str = None,
              branches: Optional[List[str]] = None,
              cut: Optional[str] = None,
              verbose: bool = False) -> Dict[str, Dict[str, Dict[str, ak.Array]]]:
    """
    Load real data from ROOT files based on specified criteria.
    
    Args:
        years: Year(s) to load. Can be "all", a single year (e.g., "2015"), or a list of years.
        tuple_types: Tuple type(s) to load. Can be "all", a single type, or a list of types.
            Available types: "KSKmKpPip", "KSKpKpPim".
        ks_categories: K-short categories to load. Can be "all", a single category, or a list of categories.
            Available categories: "LL" (Long-Long), "DD" (Downstream-Downstream).
        polarities: Magnet polarity(ies) to load. Can be "all", a single polarity, or a list of polarities.
            Available polarities: "magup", "magdown".
        data_dir: Directory containing the data ROOT files. Required parameter.
        branches: Specific branches to load. If None, all branches are loaded.
        cut: Selection cut to apply. If None, no cut is applied.
        verbose: If True, print detailed information during loading.
    
    Returns:
        Dictionary with structure: {year_polarity: {tuple_type: {ks_category: awkward array}}}
        
    Example:
        # Load all 2015 data for both magnet polarities and all tuple types
        data = load_data(years="2015", 
                       data_dir="/path/to/data")
        
        # Access data for a specific year, polarity, tuple type and Ks category
        events = data["2015_magup"]["KSKmKpPip"]["LL"]
        
        # Load multiple years and specific tuple types
        data = load_data(
            years=["2015", "2016"],
            tuple_types=["KSKmKpPip"],
            ks_categories=["LL", "DD"],
            polarities=["magup", "magdown"],
            data_dir="/path/to/data"
        )
    """
    # Check that data_dir is provided
    if data_dir is None:
        raise ValueError("data_dir parameter is required. Please specify the directory containing the data ROOT files.")
    
    # Normalize input parameters
    if years == "all":
        years_to_load = AVAILABLE_YEARS
    elif isinstance(years, str):
        years_to_load = [years]
    else:
        years_to_load = years
    
    if tuple_types == "all":
        tuple_types_to_load = TUPLE_TYPES
    elif isinstance(tuple_types, str):
        tuple_types_to_load = [tuple_types]
    else:
        tuple_types_to_load = tuple_types
    
    if ks_categories == "all":
        ks_categories_to_load = KS_CATEGORIES
    elif isinstance(ks_categories, str):
        ks_categories_to_load = [ks_categories]
    else:
        ks_categories_to_load = ks_categories
    
    if polarities == "all":
        polarities_to_load = AVAILABLE_POLARITIES
    elif isinstance(polarities, str):
        polarities_to_load = [polarities]
    else:
        polarities_to_load = polarities
    
    if verbose:
        print(f"Loading real data:")
        print(f"  Years: {years_to_load}")
        print(f"  Tuple types: {tuple_types_to_load}")
        print(f"  Ks categories: {ks_categories_to_load}")
        print(f"  Polarities: {polarities_to_load}")
        print(f"  Data directory: {data_dir}")
    
    # Find all matching files and organize by year_polarity and tuple_type
    data_files = {}
    
    for year in years_to_load:
        for polarity in polarities_to_load:
            year_polarity = f"{year}_{polarity}"
            data_files[year_polarity] = {}
            
            for tuple_type in tuple_types_to_load:
                file_pattern = os.path.join(data_dir, year_polarity, f"{tuple_type}_reduced.root")
                matching_files = glob.glob(file_pattern)
                
                if matching_files:
                    data_files[year_polarity][tuple_type] = matching_files
                    if verbose:
                        print(f"  Found file: {matching_files[0]}")
                elif verbose:
                    print(f"  Warning: No files found matching pattern {file_pattern}")
    
    # Load data by year_polarity, tuple_type, and ks_category
    result = {}
    
    for year_polarity, tuple_files in data_files.items():
        if not tuple_files:
            if verbose:
                print(f"  No files found for {year_polarity}")
            continue
        
        result[year_polarity] = {}
        
        for tuple_type, files in tuple_files.items():
            if verbose:
                print(f"  Loading {year_polarity} - {tuple_type}...")
            
            result[year_polarity][tuple_type] = {}
            
            for ks_category in ks_categories_to_load:
                if verbose:
                    print(f"    Loading {ks_category} category...")
                
                # The exact tree name in the ROOT file
                tree_name = f"{tuple_type}_{ks_category};1"
                
                # Try to load from each file
                arrays_to_concatenate = []
                
                for file_path in files:
                    try:
                        with uproot.open(file_path) as file:
                            # Check if tree exists
                            if tree_name not in file:
                                if verbose:
                                    print(f"      Tree {tree_name} not found in {file_path}")
                                continue
                            
                            # Load the branches with any specified cut
                            array = file[tree_name].arrays(branches, cut, library="ak")
                            
                            if len(array) > 0:
                                arrays_to_concatenate.append(array)
                                if verbose:
                                    print(f"      Loaded {len(array)} events from {file_path}")
                            else:
                                if verbose:
                                    print(f"      No events passed selection in {file_path}")
                                
                    except Exception as e:
                        if verbose:
                            print(f"      Error loading from {file_path}: {str(e)}")
                
                # Concatenate arrays if any were loaded
                if arrays_to_concatenate:
                    result[year_polarity][tuple_type][ks_category] = ak.concatenate(arrays_to_concatenate)
                    if verbose:
                        print(f"      Total events loaded for {year_polarity} - {tuple_type} - {ks_category}: "
                              f"{len(result[year_polarity][tuple_type][ks_category])}")
                else:
                    # Create an empty array with the correct structure
                    result[year_polarity][tuple_type][ks_category] = ak.Array([])
                    if verbose:
                        print(f"      No events loaded for {year_polarity} - {tuple_type} - {ks_category}")
    
    return result


def get_available_branches(year: str = "2015",
                           polarity: str = "magup",
                           tuple_type: str = "KSKmKpPip",
                           ks_category: str = "LL",
                           data_dir: str = None) -> Set[str]:
    """
    Get the set of available branches for the specified data.
    
    Args:
        year: Year to check.
        polarity: Magnet polarity to check.
        tuple_type: Tuple type to check.
        ks_category: K-short category to check.
        data_dir: Directory containing the data ROOT files. Required parameter.
    
    Returns:
        Set of available branch names.
    """
    # Check that data_dir is provided
    if data_dir is None:
        raise ValueError("data_dir parameter is required. Please specify the directory containing the data ROOT files.")
        
    # Find a matching file
    year_polarity = f"{year}_{polarity}"
    file_pattern = os.path.join(data_dir, year_polarity, f"{tuple_type}_reduced.root")
    matching_files = glob.glob(file_pattern)
    
    if not matching_files:
        print(f"No files found matching pattern: {file_pattern}")
        return set()
    
    file_path = matching_files[0]
    tree_name = f"{tuple_type}_{ks_category};1"
    
    try:
        with uproot.open(file_path) as file:
            if tree_name not in file:
                print(f"Tree {tree_name} not found in {file_path}")
                return set()
            
            return set(file[tree_name].keys())
    except Exception as e:
        print(f"Error reading branches from {file_path}: {str(e)}")
        return set()


def combine_data(data_dict: Dict[str, Dict[str, Dict[str, ak.Array]]],
                years: Union[str, List[str]] = "all",
                tuple_types: Union[str, List[str]] = "all",
                ks_categories: Union[str, List[str]] = "all",
                polarities: Union[str, List[str]] = "all") -> Dict[str, Dict[str, ak.Array]]:
    """
    Combine data from multiple years, polarities, tuple types, and Ks categories.
    
    Args:
        data_dict: Data dictionary returned by load_data.
        years: Year(s) to include. Can be "all", a single year, or a list of years.
        tuple_types: Tuple type(s) to include. Can be "all", a single type, or a list of types.
        ks_categories: K-short categories to include. Can be "all", a single category, or a list.
        polarities: Polarity(ies) to include. Can be "all", a single polarity, or a list of polarities.
    
    Returns:
        Dictionary with structure: {tuple_type: {ks_category: awkward array}}
    """
    # Normalize input parameters
    if years == "all":
        years_to_combine = AVAILABLE_YEARS
    elif isinstance(years, str):
        years_to_combine = [years]
    else:
        years_to_combine = years
    
    if tuple_types == "all":
        tuple_types_to_combine = TUPLE_TYPES
    elif isinstance(tuple_types, str):
        tuple_types_to_combine = [tuple_types]
    else:
        tuple_types_to_combine = tuple_types
    
    if ks_categories == "all":
        ks_categories_to_combine = KS_CATEGORIES
    elif isinstance(ks_categories, str):
        ks_categories_to_combine = [ks_categories]
    else:
        ks_categories_to_combine = ks_categories
    
    if polarities == "all":
        polarities_to_combine = AVAILABLE_POLARITIES
    elif isinstance(polarities, str):
        polarities_to_combine = [polarities]
    else:
        polarities_to_combine = polarities
    
    # Initialize result dictionary
    result = {}
    for tuple_type in tuple_types_to_combine:
        result[tuple_type] = {}
        for ks_category in ks_categories_to_combine:
            result[tuple_type][ks_category] = []
    
    # Collect arrays to combine
    for year in years_to_combine:
        for polarity in polarities_to_combine:
            year_polarity = f"{year}_{polarity}"
            
            if year_polarity not in data_dict:
                continue
            
            for tuple_type in tuple_types_to_combine:
                if tuple_type not in data_dict[year_polarity]:
                    continue
                
                for ks_category in ks_categories_to_combine:
                    if ks_category in data_dict[year_polarity][tuple_type] and \
                       len(data_dict[year_polarity][tuple_type][ks_category]) > 0:
                        result[tuple_type][ks_category].append(
                            data_dict[year_polarity][tuple_type][ks_category]
                        )
    
    # Combine arrays for each tuple type and ks_category
    for tuple_type in tuple_types_to_combine:
        for ks_category in ks_categories_to_combine:
            if result[tuple_type][ks_category]:
                result[tuple_type][ks_category] = ak.concatenate(result[tuple_type][ks_category])
            else:
                result[tuple_type][ks_category] = ak.Array([])
    
    return result


def flatten_categories(combined_data: Dict[str, Dict[str, ak.Array]]) -> Dict[str, ak.Array]:
    """
    Flatten the Ks categories (LL, DD) for each tuple type into a single array.
    
    Args:
        combined_data: The combined data dictionary with structure {tuple_type: {ks_category: array}}
    
    Returns:
        Dictionary with structure {tuple_type: array} where each array contains all categories
    """
    result = {}
    
    for tuple_type, categories in combined_data.items():
        arrays_to_combine = []
        
        for ks_category, array in categories.items():
            if len(array) > 0:
                arrays_to_combine.append(array)
        
        if arrays_to_combine:
            result[tuple_type] = ak.concatenate(arrays_to_combine)
        else:
            result[tuple_type] = ak.Array([])
    
    return result


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and analyze real data from ROOT files.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory containing data ROOT files")
    parser.add_argument("--years", default="all", help="Years to load (comma-separated or 'all')")
    parser.add_argument("--tuple-types", default="all", help="Tuple types to load (comma-separated or 'all')")
    parser.add_argument("--ks-categories", default="all", 
                        help="K-short categories to load (comma-separated or 'all')")
    parser.add_argument("--polarities", default="all", help="Magnet polarities to load (comma-separated or 'all')")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information during loading")
    
    args = parser.parse_args()
    
    # Parse comma-separated lists
    years = args.years.split(",") if args.years != "all" else "all"
    tuple_types = args.tuple_types.split(",") if args.tuple_types != "all" else "all"
    ks_categories = args.ks_categories.split(",") if args.ks_categories != "all" else "all"
    polarities = args.polarities.split(",") if args.polarities != "all" else "all"
    
    # Load data
    data = load_data(
        years=years,
        tuple_types=tuple_types,
        ks_categories=ks_categories,
        polarities=polarities,
        data_dir=args.data_dir,
        verbose=args.verbose
    )
    
    # Print summary of loaded data
    print("\nData Loading Summary:")
    total_events = 0
    
    for year_polarity, tuple_data in data.items():
        print(f"{year_polarity}:")
        year_pol_total = 0
        
        for tuple_type, ks_data in tuple_data.items():
            print(f"  {tuple_type}:")
            tuple_total = 0
            
            for ks_category, array in ks_data.items():
                n_events = len(array)
                print(f"    {ks_category}: {n_events} events")
                tuple_total += n_events
            
            print(f"    Total for {tuple_type}: {tuple_total} events")
            year_pol_total += tuple_total
        
        print(f"  Total for {year_polarity}: {year_pol_total} events")
        total_events += year_pol_total
    
    print(f"\nTotal number of events: {total_events}")
    
    # Combine data across years and polarities
    combined_data = combine_data(
        data, years, tuple_types, ks_categories, polarities
    )
    
    # Print combined data summary
    print("\nCombined Data (separated by K-short category):")
    for tuple_type, ks_data in combined_data.items():
        tuple_total = 0
        print(f"  {tuple_type}:")
        
        for ks_category, array in ks_data.items():
            n_events = len(array)
            print(f"    {ks_category}: {n_events} events")
            tuple_total += n_events
        
        print(f"    Total: {tuple_total} events")
    
    # Flatten the K-short categories for simpler analysis
    flattened_data = flatten_categories(combined_data)
    
    # Print flattened data summary
    print("\nFlattened Data (LL and DD combined):")
    total_flattened = 0
    
    for tuple_type, array in flattened_data.items():
        n_events = len(array)
        print(f"  {tuple_type}: {n_events} events")
        total_flattened += n_events
    
    print(f"  Total: {total_flattened} events")