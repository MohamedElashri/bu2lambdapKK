#!/usr/bin/env python3
import os
import argparse
import subprocess
import glob
import tempfile

def merge_dataset(dataset_name, input_dir, output_dir):
    """Merge all processed ROOT files for a dataset"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file
    output_file = os.path.join(output_dir, f"{dataset_name}.root")
    
    # Find all ROOT files in the input directory
    input_files = glob.glob(f"{input_dir}/*.root")
    
    if not input_files:
        print(f"No ROOT files found in {input_dir}")
        return
    
    # Create a temporary file list
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
        for file_path in input_files:
            temp.write(f"{file_path}\n")
        temp_filename = temp.name
    
    print(f"Merging {len(input_files)} files into {output_file}")
    
    # Call hadd to merge the files
    cmd = ["hadd", "-f", output_file, "@" + temp_filename]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully merged files into {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error merging files: {e}")
    
    # Remove the temporary file
    os.unlink(temp_filename)

def main():
    parser = argparse.ArgumentParser(description='Merge processed ROOT files')
    parser.add_argument('dataset_name', help='Dataset name (e.g., 2015_magup)')
    parser.add_argument('input_dir', help='Directory containing processed files')
    parser.add_argument('output_dir', help='Output directory for merged file')
    args = parser.parse_args()
    
    merge_dataset(args.dataset_name, args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
