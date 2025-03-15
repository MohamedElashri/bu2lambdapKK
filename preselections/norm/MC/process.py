#!/usr/bin/env python3
"""
Script to download MC files from LHCb production location to a user-specific location,
organizing them by year and polarity.
"""

import os
import subprocess
import logging
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PROD_PREFIX = "/eos/lhcb/grid/prod"
TARGET_BASE = "/eos/lhcb/user/m/melashri/data/bu2kskpik/MC"

# Define available years and polarities
YEARS = ["2015", "2016", "2017", "2018"]
POLARITIES = ["magup", "magdown"]

# Dictionary to hold file paths for each configuration (year_polarity)
file_paths: Dict[str, List[str]] = {
    "2018_magup": [
        "/lhcb/MC/2018/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00146282/0000/00146282_00000001_1.bhadron_b2kshhh_dvntuple.root",
        "/lhcb/MC/2018/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00146274/0000/00146274_00000001_1.bhadron_b2kshhh_dvntuple.root",
        "/lhcb/MC/2018/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00146272/0000/00146272_00000001_1.bhadron_b2kshhh_dvntuple.root"   
    ],
    
    # Fill in the lists for each year and polarity configuration
    "2015_magup": [
        "/lhcb/MC/2015/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00154828/0000/00154828_00000001_1.bhadron_b2kshhh_dvntuple.root",
        "/lhcb/MC/2015/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00154824/0000/00154824_00000001_1.bhadron_b2kshhh_dvntuple.root",
        "/lhcb/MC/2015/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00154820/0000/00154820_00000001_1.bhadron_b2kshhh_dvntuple.root"

    ],
    "2015_magdown": [
        "/lhcb/MC/2015/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00154835/0000/00154835_00000001_1.bhadron_b2kshhh_dvntuple.root",
        "/lhcb/MC/2015/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00154833/0000/00154833_00000001_1.bhadron_b2kshhh_dvntuple.root",
        "/lhcb/MC/2015/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00154831/0000/00154831_00000001_1.bhadron_b2kshhh_dvntuple.root"
    ],
    "2016_magup": [
        "/lhcb/MC/2016/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00154805/0000/00154805_00000001_1.bhadron_b2kshhh_dvntuple.root",
        "/lhcb/MC/2016/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00154803/0000/00154803_00000001_1.bhadron_b2kshhh_dvntuple.root",
        "/lhcb/MC/2016/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00154801/0000/00154801_00000001_1.bhadron_b2kshhh_dvntuple.root"
    ],
    "2016_magdown": [
        "/lhcb/MC/2016/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00154817/0000/00154817_00000001_1.bhadron_b2kshhh_dvntuple.root",
        "/lhcb/MC/2016/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00154812/0000/00154812_00000001_1.bhadron_b2kshhh_dvntuple.root",
        "/lhcb/MC/2016/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00154809/0000/00154809_00000001_1.bhadron_b2kshhh_dvntuple.root"
        
    ],
    "2017_magup": [
        "/lhcb/MC/2017/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00146288/0000/00146288_00000001_1.bhadron_b2kshhh_dvntuple.root",
        "/lhcb/MC/2017/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00146286/0000/00146286_00000001_1.bhadron_b2kshhh_dvntuple.root",
        "/lhcb/MC/2017/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00146284/0000/00146284_00000001_1.bhadron_b2kshhh_dvntuple.root"
       
    ],
    "2017_magdown": [
        "/lhcb/MC/2017/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00146294/0000/00146294_00000001_1.bhadron_b2kshhh_dvntuple.root",
        "/lhcb/MC/2017/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00146292/0000/00146292_00000001_1.bhadron_b2kshhh_dvntuple.root",
        "/lhcb/MC/2017/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00146290/0000/00146290_00000001_1.bhadron_b2kshhh_dvntuple.root"
        
    ],
    "2018_magdown": [
        "/lhcb/MC/2018/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00146282/0000/00146282_00000001_1.bhadron_b2kshhh_dvntuple.root",
        "/lhcb/MC/2018/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00146280/0000/00146280_00000001_1.bhadron_b2kshhh_dvntuple.root",
        "/lhcb/MC/2018/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00146278/0000/00146278_00000001_1.bhadron_b2kshhh_dvntuple.root"

    ],
}

def ensure_target_directories_exist() -> None:
    """Create target directories if they don't exist."""
    for year in YEARS:
        for polarity in POLARITIES:
            target_dir = os.path.join(TARGET_BASE, f"{year}_{polarity}")
            
            # Create directory if it doesn't exist
            if not os.path.exists(target_dir):
                try:
                    os.makedirs(target_dir)
                    logger.info(f"Created directory: {target_dir}")
                except OSError as e:
                    logger.error(f"Failed to create directory {target_dir}: {e}")
                    raise

def copy_file(src_path: str, dest_path: str) -> bool:
    """
    Copy a file from source to destination using xrdcp.
    
    Args:
        src_path: Full source path of the file
        dest_path: Full destination path for the file
        
    Returns:
        bool: True if copy was successful, False otherwise
    """
    try:
        # If the file already exists at the destination, skip it
        if os.path.exists(dest_path):
            logger.info(f"File already exists, skipping: {dest_path}")
            return True
            
        # Execute the xrdcp command to copy the file
        # Using subprocess.run with timeout to prevent hanging
        cmd = ["xrdcp", src_path, dest_path]
        logger.debug(f"Executing command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3600,  # 1 hour timeout
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"Successfully copied to {dest_path}")
            return True
        else:
            logger.error(f"Failed to copy {src_path}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout expired while copying {src_path}")
        return False
    except Exception as e:
        logger.error(f"Error copying {src_path}: {str(e)}")
        return False

def process_configuration(year_polarity: str, file_list: List[str]) -> Tuple[int, int]:
    """
    Process all files for a specific year and polarity configuration.
    
    Args:
        year_polarity: Configuration string in format "YEAR_POLARITY"
        file_list: List of file paths for this configuration
        
    Returns:
        Tuple of (total_files, successful_copies)
    """
    target_dir = os.path.join(TARGET_BASE, year_polarity)
    logger.info(f"Processing configuration: {year_polarity}, Files: {len(file_list)}")
    
    total_files = len(file_list)
    successful_copies = 0
    
    for file_path in file_list:
        # Get the filename from the path
        filename = os.path.basename(file_path)
        
        # Construct the full source and destination paths
        src_path = f"{PROD_PREFIX}{file_path}"
        dest_path = os.path.join(target_dir, filename)
        
        # Copy the file
        if copy_file(src_path, dest_path):
            successful_copies += 1
            
    return total_files, successful_copies

def main() -> None:
    """Main function to orchestrate the file copying process."""
    logger.info("Starting LHCb MC file download process")
    
    # Create target directories
    ensure_target_directories_exist()
    
    # Track overall statistics
    total_configurations = 0
    total_files = 0
    total_successful = 0
    
    # Process each configuration
    for year_polarity, file_list in file_paths.items():
        if not file_list:  # Skip empty lists
            logger.warning(f"No files defined for configuration: {year_polarity}")
            continue
            
        total_configurations += 1
        files, successful = process_configuration(year_polarity, file_list)
        total_files += files
        total_successful += successful
        
        logger.info(f"Completed {year_polarity}: {successful}/{files} files copied successfully")
    
    # Report final statistics
    logger.info("==== Download Summary ====")
    logger.info(f"Total configurations processed: {total_configurations}")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Successfully copied: {total_successful}")
    logger.info(f"Failed copies: {total_files - total_successful}")
    
    if total_files == total_successful:
        logger.info("All files were copied successfully")
    
if __name__ == "__main__":
    main()