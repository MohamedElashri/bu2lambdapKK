#!/usr/bin/env python3
import os
import argparse
import subprocess
import glob
from tqdm import tqdm

def process_dataset(dataset_id, dataset_path, output_dir):
    """Process all ROOT files in a dataset"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all ROOT files in the dataset path
    input_files = glob.glob(f"{dataset_path}/{dataset_id}_*.root")
    
    if not input_files:
        print(f"No ROOT files found in {dataset_path} matching pattern {dataset_id}_*.root")
        return
    
    print(f"Found {len(input_files)} ROOT files to process")
    
    # Process each file
    for input_file in tqdm(input_files, desc=f"Processing files"):
        # Extract file number from filename
        file_basename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, file_basename)
        
        # Call the process_file script
        cmd = ["python3", "process_single_file.py", input_file, output_file]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {input_file}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description='Process all ROOT files in a dataset')
    parser.add_argument('dataset_id', help='Dataset ID (e.g., 00085557)')
    parser.add_argument('dataset_path', help='Path to the dataset directory')
    parser.add_argument('output_dir', help='Output directory for processed files')
    args = parser.parse_args()
    
    process_dataset(args.dataset_id, args.dataset_path, args.output_dir)

if __name__ == "__main__":
    main()
