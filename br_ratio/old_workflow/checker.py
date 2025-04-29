#!/usr/bin/env python3
"""
Script to check the actual file names in the data directories
"""
import os
import re

def list_root_files(directory, max_files=10):
    """List .root files in a directory (recursively) up to max_files."""
    if not os.path.exists(directory):
        return []
    
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.root'):
                files.append(os.path.join(root, filename))
                if len(files) >= max_files:
                    return files
    return files

# Directories from config.yml
directories = {
    "signal_data": "/share/lazy/Mohamed/Bu2LambdaPPP/RD/restripped.data/reduced",
    "signal_mc": "/share/lazy/Mohamed/Bu2LambdaPPP/MC/DaVinciTuples/restripped.MC",
    "norm_data": "/share/lazy/Mohamed/bu2kskpik/RD/new/final",
    "norm_mc": "/share/lazy/Mohamed/bu2kskpik/MC/processed"
}

# Check each directory
for name, directory in directories.items():
    print(f"\nChecking {name} directory: {directory}")
    files = list_root_files(directory)
    if not files:
        print(f"No .root files found in {directory}")
    else:
        print(f"Found {len(files)} .root files. Sample files:")
        for f in files:
            print(f"  {f}")

# Check for common patterns in filenames
patterns = {
    "L0barPKpKm": re.compile(r"L0barPKpKm", re.IGNORECASE),
    "dataBu2L0barPHH": re.compile(r"dataBu2L0barPHH", re.IGNORECASE),
    "B2K0s2PipPimKmPipKp": re.compile(r"B2K0s2PipPimKmPipKp", re.IGNORECASE),
    "KSKmKpPip": re.compile(r"KSKmKpPip", re.IGNORECASE)
}

print("\nSearching for specific patterns in filenames:")
for name, directory in directories.items():
    if not os.path.exists(directory):
        continue
    
    print(f"\nPatterns found in {name} directory:")
    for pattern_name, pattern in patterns.items():
        found = False
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith('.root') and pattern.search(filename):
                    print(f"  Pattern '{pattern_name}' found in {os.path.join(root, filename)}")
                    found = True
                    break
            if found:
                break
        if not found:
            print(f"  Pattern '{pattern_name}' not found in any filenames")