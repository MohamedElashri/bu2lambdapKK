#!/usr/bin/env python3
"""
Script to check the structure of ROOT files to determine the correct tree name.
"""

import os
import glob
import uproot
import argparse

def explore_root_file(file_path):
    """
    Explore the structure of a ROOT file and print out all trees and their branches.
    
    Args:
        file_path: Path to the ROOT file
    """
    print(f"Exploring ROOT file: {file_path}")
    
    try:
        with uproot.open(file_path) as file:
            # Get all keys (top-level objects) in the file
            top_keys = file.keys()
            print(f"Top-level objects: {top_keys}")
            
            # Explore each key to find trees
            for key in top_keys:
                try:
                    obj = file[key]
                    # Check if it's a TTree or TDirectory
                    if isinstance(obj, uproot.TTree):
                        print(f"Found TTree: {key}")
                        print(f"  Number of entries: {len(obj)}")
                        print(f"  Available branches: {obj.keys()}")
                        print()
                    elif isinstance(obj, uproot.TDirectory):
                        print(f"Found TDirectory: {key}")
                        # Explore the directory for TTrees
                        dir_keys = obj.keys()
                        print(f"  Directory contains: {dir_keys}")
                        for dir_key in dir_keys:
                            try:
                                dir_obj = obj[dir_key]
                                if isinstance(dir_obj, uproot.TTree):
                                    print(f"  Found TTree in directory: {key}/{dir_key}")
                                    print(f"    Number of entries: {len(dir_obj)}")
                                    print(f"    Available branches: {dir_obj.keys()}")
                                    print()
                            except Exception as e:
                                print(f"    Error accessing {key}/{dir_key}: {str(e)}")
                except Exception as e:
                    print(f"  Error accessing {key}: {str(e)}")
            
    except Exception as e:
        print(f"Error opening file: {str(e)}")

def find_all_root_files(data_dir, max_files=5):
    """
    Find all ROOT files in the given directory structure and return a sample.
    
    Args:
        data_dir: Base directory to search
        max_files: Maximum number of files to return
        
    Returns:
        List of paths to ROOT files (up to max_files)
    """
    root_files = []
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".root"):
                root_files.append(os.path.join(root, file))
                if len(root_files) >= max_files:
                    return root_files
    
    return root_files

def main():
    parser = argparse.ArgumentParser(description="Explore the structure of ROOT files")
    parser.add_argument("--data-dir", default="/share/lazy/Mohamed/bu2kskpik/RD/compressed/", 
                        help="Directory containing ROOT files")
    parser.add_argument("--max-files", type=int, default=3, 
                        help="Maximum number of files to explore")
    parser.add_argument("--specific-file", default=None,
                        help="Explore a specific ROOT file instead of searching")
    
    args = parser.parse_args()
    
    if args.specific_file:
        # Explore a specific file
        explore_root_file(args.specific_file)
    else:
        # Find and explore a sample of files
        print(f"Searching for ROOT files in {args.data_dir}")
        sample_files = find_all_root_files(args.data_dir, args.max_files)
        
        if not sample_files:
            print("No ROOT files found!")
            return
        
        print(f"Found {len(sample_files)} ROOT files to explore")
        
        for i, file_path in enumerate(sample_files):
            print(f"\nFile {i+1}/{len(sample_files)}")
            explore_root_file(file_path)

if __name__ == "__main__":
    main()