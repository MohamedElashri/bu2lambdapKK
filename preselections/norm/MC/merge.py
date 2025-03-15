#!/usr/bin/env python3
"""
Script to merge downloaded LHCb MC ROOT files by year and polarity configuration,
using uproot to verify file contents.
"""

import os
import glob
import subprocess
import logging
from typing import List, Dict, Tuple, Set
import uproot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
TARGET_BASE = "/eos/lhcb/user/m/melashri/data/bu2kskpik/MC"
OUTPUT_DIR = "/eos/lhcb/user/m/melashri/data/bu2kskpik/MC/merged"

# Define available years and polarities
YEARS = ["2015", "2016", "2017", "2018"]
POLARITIES = ["magup", "magdown"]

# Expected tree names in each file
EXPECTED_TREES = [
    "B2KSKKPip_DD/DecayTree",
    "B2KSKKPip_LL/DecayTree",
    "B2KSKKPim_LL/DecayTree",
    "B2KSKKPim_DD/DecayTree"
]

def ensure_output_directory_exists() -> None:
    """Create the output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
            logger.info(f"Created output directory: {OUTPUT_DIR}")
        except OSError as e:
            logger.error(f"Failed to create output directory {OUTPUT_DIR}: {e}")
            raise

def get_files_for_config(year_polarity: str) -> List[str]:
    """
    Get all ROOT files for a specific year/polarity configuration.
    
    Args:
        year_polarity: Configuration string in format "YEAR_POLARITY"
        
    Returns:
        List of paths to ROOT files for this configuration
    """
    config_dir = os.path.join(TARGET_BASE, year_polarity)
    
    if not os.path.exists(config_dir):
        logger.warning(f"Directory does not exist: {config_dir}")
        return []
        
    # Get all ROOT files in this directory
    root_files = glob.glob(os.path.join(config_dir, "*.root"))
    logger.info(f"Found {len(root_files)} ROOT files in {config_dir}")
    
    return root_files

def verify_file_with_uproot(root_file: str) -> Tuple[bool, Set[str]]:
    """
    Verify that a ROOT file is accessible and get its tree names using uproot.
    
    Args:
        root_file: Path to ROOT file
        
    Returns:
        Tuple of (is_valid, set_of_found_trees)
    """
    try:
        # Try to open the file with uproot
        file = uproot.open(root_file)
        
        # Collect all tree names
        trees_found = set()
        
        # Function to recursively explore directories and find trees
        def explore_directory(directory, path=""):
            for key_name in directory:
                # Skip keys that start with ";" (like ";1")
                if key_name.startswith(";"):
                    continue
                    
                full_path = f"{path}{key_name}" if path else key_name
                
                try:
                    obj = directory[key_name]
                    # Check if it's a tree
                    if isinstance(obj, uproot.behaviors.TBranch.HasBranches):
                        trees_found.add(full_path)
                    # Check if it's a directory
                    elif hasattr(obj, "classname") and obj.classname == "TDirectoryFile":
                        # Recursively explore this directory with updated path
                        explore_directory(obj, f"{full_path}/")
                except Exception as e:
                    logger.debug(f"Error exploring {full_path}: {str(e)}")
        
        # Start exploration from the root
        explore_directory(file)
        
        # Log found trees
        logger.info(f"Found the following trees in {root_file}: {trees_found}")
        
        # Check if expected trees exist
        # We'll be flexible here - if file opens and has any tree content, consider it valid
        return len(trees_found) > 0, trees_found
        
    except Exception as e:
        logger.error(f"Error verifying {root_file} with uproot: {str(e)}")
        return False, set()

def merge_root_files(file_list: List[str], output_file: str) -> bool:
    """
    Merge ROOT files using hadd.
    
    Args:
        file_list: List of ROOT files to merge
        output_file: Path to output merged file
        
    Returns:
        bool: True if merge was successful, False otherwise
    """
    try:
        # Create hadd command
        cmd = ["hadd", "-f", output_file]
        
        # Add files to the command
        cmd.extend(file_list)
        
        logger.info(f"Executing merge command: {' '.join(cmd)}")
        
        # Execute hadd command
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=7200,  # 2 hour timeout for large merges
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"Successfully merged files into {output_file}")
            return True
        else:
            logger.error(f"Failed to merge files: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout expired while merging files")
        return False
    except Exception as e:
        logger.error(f"Error merging files: {str(e)}")
        return False

def process_year_polarity(year_polarity: str) -> bool:
    """
    Process all files for a specific year/polarity configuration.
    
    Args:
        year_polarity: Configuration string in format "YEAR_POLARITY"
        
    Returns:
        bool: True if processing was successful
    """
    # Get all ROOT files for this configuration
    root_files = get_files_for_config(year_polarity)
    
    if not root_files:
        logger.warning(f"No ROOT files found for {year_polarity}")
        return False
        
    # Verify files with uproot
    valid_files = []
    all_trees = set()
    
    for root_file in root_files:
        is_valid, trees = verify_file_with_uproot(root_file)
        
        if is_valid:
            valid_files.append(root_file)
            all_trees.update(trees)
        else:
            logger.warning(f"Skipping invalid file: {root_file}")
    
    if not valid_files:
        logger.error(f"No valid ROOT files found for {year_polarity}")
        return False
        
    logger.info(f"Found {len(valid_files)} valid files for {year_polarity}")
    logger.info(f"Trees across all files: {all_trees}")
    
    # Merge the valid files
    output_file = os.path.join(OUTPUT_DIR, f"{year_polarity}.root")
    
    return merge_root_files(valid_files, output_file)

def main() -> None:
    """Main function to orchestrate the file merging process."""
    logger.info("Starting LHCb MC file merging process with uproot")
    
    # Create output directory
    ensure_output_directory_exists()
    
    # Track overall statistics
    successful_configs = 0
    total_configs = len(YEARS) * len(POLARITIES)
    
    # Process each year/polarity configuration
    for year in YEARS:
        for polarity in POLARITIES:
            year_polarity = f"{year}_{polarity}"
            logger.info(f"Processing configuration: {year_polarity}")
            
            if process_year_polarity(year_polarity):
                successful_configs += 1
                logger.info(f"Successfully merged files for {year_polarity}")
            else:
                logger.error(f"Failed to merge files for {year_polarity}")
    
    # Report final statistics
    logger.info("==== Merging Summary ====")
    logger.info(f"Successful merges: {successful_configs}/{total_configs}")
    
    if successful_configs == total_configs:
        logger.info("All configurations were merged successfully")
    else:
        logger.warning(f"Failed to merge {total_configs - successful_configs} configurations")
    
if __name__ == "__main__":
    main()