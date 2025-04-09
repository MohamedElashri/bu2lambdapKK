#!/usr/bin/env python3
"""
This script handles inconsistencies in TTree paths and naming conventions,
ensuring compatibility with uproot for all processed files.
"""

import os
import glob
import tempfile
import subprocess
import argparse
import time
import shutil
from pathlib import Path

# Define decay codes and their corresponding names and descriptions
DECAY_INFO = {
    "12105160": {
        "name": "B2K0s2PipPimKmPipKp",
        "description": "B+ → (K0_S → π+π-)K-π+K+"
    },
    "12135100": {
        "name": "B2Jpsi2K0s2PipPimKmPipKp",
        "description": "B+ → (J/ψ → (K0_S → π+π-)K-π+)K+"
    },
    "12135102": {
        "name": "B2Etac2K0s2PipPimKmPipKp",
        "description": "B+ → (ηc → (K0_S → π+π-)K-π+)K+"
    },
    "12135104": {
        "name": "B2Etac2S2K0s2PipPimKmPipKp",
        "description": "B+ → (ηc(2S) → (K0_S → π+π-)K-π+)K+"
    },
    "12135106": {
        "name": "B2Chic12K0s2PipPimKmPipKp",
        "description": "B+ → (χc1 → (K0_S → π+π-)K-π+)K+"
    }
}

# Directories we want to keep
DIRS_TO_KEEP = [
    "KSKmKpPip_DD",
    "KSKmKpPip_LL", 
    "KSKpKpPim_DD",
    "KSKpKpPim_LL",
    # Also check for versions with _Tuple suffix
    "KSKmKpPip_DD_Tuple",
    "KSKmKpPip_LL_Tuple",
    "KSKpKpPim_DD_Tuple",
    "KSKpKpPim_LL_Tuple"
]

def setup_args():
    """Set up command line arguments"""
    parser = argparse.ArgumentParser(description='Process ROOT files containing Monte Carlo simulation data')
    parser.add_argument('--base-dir', type=str, 
                        default='/share/lazy/Mohamed/bu2kskpik/MC/data/data',
                        help='Base directory where data is located')
    parser.add_argument('--output-dir', type=str, 
                        default=None,
                        help='Output directory for processed files (default: {base_dir}/processed)')
    parser.add_argument('--years', type=str, nargs='+',
                        default=['2015', '2016', '2017', '2018'],
                        help='Years to process')
    parser.add_argument('--decay-codes', type=str, nargs='+',
                        default=list(DECAY_INFO.keys()),
                        help='Decay codes to process')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Do not delete temporary files')
    return parser.parse_args()

def find_files(base_dir, year, decay_code, debug=False):
    """Find files for a specific year and decay code"""
    year_dir = f"mc_{year}"
    
    # Handle 2018 which has a nested structure
    if year == "2018":
        pattern = f"{base_dir}/{year_dir}/**/*{decay_code}*dvntuple.root"
        files = glob.glob(pattern, recursive=True)
    else:
        pattern = f"{base_dir}/{year_dir}/*{decay_code}*dvntuple.root"
        files = glob.glob(pattern)
    
    if debug:
        print(f"Found {len(files)} files matching pattern: {pattern}")
        if files:
            print(f"Sample file: {files[0]}")
    
    return files

def process_file_with_pyroot(input_files, output_file, year, decay_name, debug=False):
    """Process ROOT files using PyROOT to extract relevant data in a format uproot can read"""
    if not input_files:
        print(f"  No input files provided.")
        return False
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create a temporary file with list of input files
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        for f in input_files:
            temp_file.write(f"{f}\n")
        input_list_path = temp_file.name
    
    # Create temporary directory for the PyROOT script
    temp_dir = tempfile.mkdtemp()
    script_path = os.path.join(temp_dir, "process_root.py")
    
    # Write PyROOT script to process the files
    with open(script_path, 'w') as f:
        f.write(f'''#!/usr/bin/env python3
"""
Enhanced ROOT processing script to extract data from B Meson MC files.
This script handles various naming inconsistencies and structure issues.
"""
import sys
import os
import ROOT
from array import array
import time
import fnmatch

# Suppress ROOT's information messages
ROOT.gErrorIgnoreLevel = ROOT.kWarning

print("Starting PyROOT processing...")

# List of directories we want to process
dirs_to_keep = [
    "KSKmKpPip_DD",
    "KSKmKpPip_LL",
    "KSKpKpPim_DD", 
    "KSKpKpPim_LL",
    "KSKmKpPip_DD_Tuple",
    "KSKmKpPip_LL_Tuple",
    "KSKpKpPim_DD_Tuple",
    "KSKpKpPim_LL_Tuple"
]

# Alternative names for DecayTree
tree_names = ["DecayTree", "tuple", "Tuple", "DecayTree/tuple", "tuple/DecayTree"]

# Create output file
output_file = ROOT.TFile.Open("{output_file}", "RECREATE")
if not output_file or output_file.IsZombie():
    print(f"Error: Failed to create output file {output_file}")
    sys.exit(1)

# Read input file list
with open("{input_list_path}", "r") as file_list:
    input_files = [line.strip() for line in file_list if line.strip()]

print(f"Processing {{len(input_files)}} input files...")

# Function to find directories recursively
def find_directories(directory, pattern):
    result = []
    for key in directory.GetListOfKeys():
        name = key.GetName()
        obj = key.ReadObj()
        
        if obj.InheritsFrom("TDirectory"):
            if fnmatch.fnmatch(name, pattern):
                result.append((name, obj))
            # Also search subdirectories
            subdirs = find_directories(obj, pattern)
            for subname, subobj in subdirs:
                result.append((name + "/" + subname, subobj))
    
    return result

# Function to normalize directory name
def normalize_dir_name(name):
    if name.endswith("_Tuple"):
        return name[:-6]
    return name

# Function to find TTree in directory
def find_tree(directory):
    # First try direct names
    for tree_name in tree_names:
        tree = directory.Get(tree_name)
        if tree and tree.InheritsFrom("TTree"):
            return tree, tree_name
    
    # Then try recursive search for any TTree
    for key in directory.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom("TTree"):
            return obj, key.GetName()
        elif obj.InheritsFrom("TDirectory"):
            # Check subdirectory
            subtree, subtree_name = find_tree(obj)
            if subtree:
                return subtree, key.GetName() + "/" + subtree_name
    
    return None, None

# Dictionary to store all branches across all files for each tuple type
all_branches = {{}}

# First pass: collect all directories and branches from all files
print("First pass: Identifying directories and branches...")

for file_path in input_files:
    try:
        input_file = ROOT.TFile.Open(file_path, "READ")
        if not input_file or input_file.IsZombie():
            print(f"Error: Failed to open {{file_path}}")
            continue
        
        # Get all directories in file
        for key in input_file.GetListOfKeys():
            dir_name = key.GetName()
            obj = key.ReadObj()
            
            # If this is a directory, check if it's one we want
            if not obj.InheritsFrom("TDirectory"):
                continue
                
            # Check if this is a directory we want to keep
            normalized_name = normalize_dir_name(dir_name)
            if normalized_name not in ["KSKmKpPip_DD", "KSKmKpPip_LL", "KSKpKpPim_DD", "KSKpKpPim_LL"]:
                continue
            
            print(f"Found directory: {{dir_name}} -> {{normalized_name}}")
            
            # Find the tree in this directory
            tree, tree_name = find_tree(obj)
            if tree:
                print(f"  Found tree: {{dir_name}}/{{tree_name}} with {{tree.GetEntries()}} entries")
                
                # Store the normalized name in our dictionary
                if normalized_name not in all_branches:
                    all_branches[normalized_name] = set()
                
                # Get all branches for this tree
                branches = tree.GetListOfBranches()
                for branch in branches:
                    branch_name = branch.GetName()
                    all_branches[normalized_name].add(branch_name)
            else:
                print(f"  No suitable tree found in {{dir_name}}")
        
        input_file.Close()
    except Exception as e:
        print(f"Error processing {{file_path}}: {{str(e)}}")

# Second pass: Create new trees with consistent branches
print("Second pass: Creating output trees with consistent branches...")

# Create trees in the output file with all branches
output_trees = {{}}
leaf_values = {{}}

for dir_name, branch_names in all_branches.items():
    output_file.cd()
    
    # Create directory in output file
    output_dir = output_file.mkdir(dir_name)
    output_dir.cd()
    
    # Create tree with a fixed name (easier for uproot)
    tree = ROOT.TTree("DecayTree", "DecayTree")
    output_trees[dir_name] = tree
    
    # Create branch variables
    leaf_values[dir_name] = {{}}
    
    print(f"Creating tree for {{dir_name}} with {{len(branch_names)}} branches")
    
    # Create branches for this tree
    for branch_name in sorted(branch_names):
        # For simplicity, we use double type for all variables
        leaf_values[dir_name][branch_name] = array('d', [0.0])
        tree.Branch(branch_name, leaf_values[dir_name][branch_name], f"{{branch_name}}/D")

# Third pass: Fill the trees with data from all files
print("Third pass: Filling trees with data...")

# Function to extract tuple type from directory path
def get_tuple_type(dir_name):
    # Handle case where dir_name is a path like "A/B/C"
    parts = dir_name.split('/')
    for part in parts:
        normalized = normalize_dir_name(part)
        if normalized in ["KSKmKpPip_DD", "KSKmKpPip_LL", "KSKpKpPim_DD", "KSKpKpPim_LL"]:
            return normalized
    return None

# Process each input file
for file_idx, file_path in enumerate(input_files):
    try:
        print(f"Processing file {{file_idx+1}}/{{len(input_files)}}: {{os.path.basename(file_path)}}")
        input_file = ROOT.TFile.Open(file_path, "READ")
        if not input_file or input_file.IsZombie():
            print(f"Error: Failed to open {{file_path}}")
            continue
        
        # Find all relevant directories in the file
        relevant_dirs = []
        for tuple_pattern in ["*KSKmKpPip_DD*", "*KSKmKpPip_LL*", "*KSKpKpPim_DD*", "*KSKpKpPim_LL*"]:
            # First check top level
            for key in input_file.GetListOfKeys():
                name = key.GetName()
                if fnmatch.fnmatch(name, tuple_pattern):
                    obj = key.ReadObj()
                    if obj.InheritsFrom("TDirectory"):
                        relevant_dirs.append((name, obj))
            
            # Then check recursively in case of nested structure
            for key in input_file.GetListOfKeys():
                obj = key.ReadObj()
                if obj.InheritsFrom("TDirectory"):
                    nested_dirs = find_directories(obj, tuple_pattern)
                    for nested_name, nested_obj in nested_dirs:
                        full_path = key.GetName() + "/" + nested_name
                        relevant_dirs.append((full_path, nested_obj))
        
        # Process each relevant directory
        for dir_path, directory in relevant_dirs:
            # Find the tree in this directory
            tree, tree_name = find_tree(directory)
            if not tree:
                print(f"  No suitable tree found in {{dir_path}}")
                continue
            
            # Get the tuple type
            tuple_type = get_tuple_type(dir_path)
            if not tuple_type:
                print(f"  Could not determine tuple type from {{dir_path}}")
                continue
            
            # Check if we have an output tree for this tuple type
            if tuple_type not in output_trees:
                print(f"  No output tree defined for {{tuple_type}}")
                continue
            
            # Get the output tree
            output_tree = output_trees[tuple_type]
            
            # Process entries
            entries = tree.GetEntries()
            print(f"  Processing {{dir_path}}/{{tree_name}} with {{entries}} entries")
            
            # Create a dictionary to map branch names to leaf objects
            input_leaves = {{}}
            for branch in tree.GetListOfBranches():
                branch_name = branch.GetName()
                leaf = branch.GetLeaf(branch_name)
                if leaf:
                    input_leaves[branch_name] = leaf
            
            # Process each entry
            for entry in range(entries):
                if entry % 10000 == 0 and entry > 0:
                    print(f"    Processed {{entry}}/{{entries}} entries...")
                
                tree.GetEntry(entry)
                
                # Reset all branch values to 0
                for branch_name in leaf_values[tuple_type]:
                    leaf_values[tuple_type][branch_name][0] = 0.0
                
                # Set values from input branches when available
                for branch_name, value_array in leaf_values[tuple_type].items():
                    if branch_name in input_leaves:
                        try:
                            # Get the leaf and try to read its value
                            leaf = input_leaves[branch_name]
                            value_array[0] = leaf.GetValue()
                        except Exception as e:
                            # If we can't get the value, leave it as 0
                            pass
                
                # Fill the output tree
                output_tree.Fill()
        
        input_file.Close()
    except Exception as e:
        print(f"Error processing {{file_path}}: {{str(e)}}")

# Write the output file
print("Writing output file...")
output_file.cd()
for dir_name, tree in output_trees.items():
    print(f"Writing tree for {{dir_name}} with {{tree.GetEntries()}} entries")
    output_file.cd(dir_name)
    tree.Write()

output_file.Close()
print(f"Successfully created output file: {output_file}")
print("PyROOT processing completed!")
''')
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    try:
        # Execute the PyROOT script
        print(f"  Running PyROOT processing script... (this may take a while)")
        start_time = time.time()
        
        cmd = ["python3", script_path]
        if debug:
            print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        elapsed_time = time.time() - start_time
        
        if debug:
            print(f"PyROOT script stdout:\n{result.stdout}")
            print(f"PyROOT script stderr:\n{result.stderr}")
        
        success = result.returncode == 0 and os.path.exists(output_file)
        
        if success:
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"  ✓ Successfully processed files into {output_file} ({file_size_mb:.2f} MB) in {elapsed_time:.1f} seconds")
        else:
            print(f"  ✗ Failed to process files. Check output for errors.")
            # Print part of the output to help identify issues
            if result.stdout:
                print("  Last 20 lines of script output:")
                last_lines = result.stdout.strip().split('\n')[-20:]
                for line in last_lines:
                    print(f"    {line}")
        
        return success
    except Exception as e:
        print(f"  ✗ Error during PyROOT processing: {str(e)}")
        return False
    finally:
        # Clean up temporary files
        os.unlink(input_list_path)
        if not debug:
            shutil.rmtree(temp_dir, ignore_errors=True)

def verify_with_uproot(file_path, debug=False):
    """Verify a ROOT file can be read with uproot"""
    try:
        import uproot
        
        print(f"Verifying {os.path.basename(file_path)} with uproot...")
        
        with uproot.open(file_path) as f:
            # Get all directories
            dirs = list(f.keys())
            print(f"  Found directories: {', '.join(dirs)}")
            
            # Check each directory for DecayTree
            total_entries = 0
            for dir_name in dirs:
                try:
                    # Check different tree paths since there might be inconsistencies
                    tree_paths = [
                        f"{dir_name}/DecayTree",
                        f"{dir_name}/DecayTree;1"
                    ]
                    
                    tree_found = False
                    for tree_path in tree_paths:
                        try:
                            tree = f[tree_path]
                            num_entries = tree.num_entries
                            print(f"  ✓ {tree_path}: {num_entries} entries")
                            total_entries += num_entries
                            tree_found = True
                            break
                        except Exception:
                            if debug:
                                print(f"  Tree path {tree_path} not found, trying alternatives...")
                    
                    if not tree_found:
                        print(f"  ✗ No valid tree found in {dir_name}")
                        
                except Exception as e:
                    if debug:
                        print(f"  ✗ Error accessing {dir_name}: {str(e)}")
            
            print(f"  Total entries across all trees: {total_entries}")
            return total_entries > 0
            
    except Exception as e:
        print(f"  ✗ Error during uproot verification: {str(e)}")
        return False

def process_files(args):
    """Process all files according to command line arguments"""
    # Set up output directory
    output_dir = args.output_dir or os.path.join(args.base_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, "processing_log.txt")
    with open(log_file, 'w') as log:
        log.write(f"Starting ROOT file processing\n")
        log.write(f"----------------------------------------\n\n")
    
    # Process each year and decay combination
    total_processed = 0
    total_failures = 0
    
    for year in args.years:
        print(f"\nProcessing year: {year}")
        print("--------------------------------------------")
        
        for decay_code in args.decay_codes:
            decay_info = DECAY_INFO.get(decay_code, {})
            decay_name = decay_info.get("name", f"Unknown_{decay_code}")
            decay_desc = decay_info.get("description", "Unknown decay")
            
            print(f"  Decay code: {decay_code}")
            print(f"  Description: {decay_desc}")
            
            # Find files for this year and decay code
            input_files = find_files(args.base_dir, year, decay_code, args.debug)
            
            if not input_files:
                print(f"  No files found for year {year} with decay code {decay_code}")
                continue
            
            # Process files using PyROOT
            output_file = os.path.join(output_dir, f"{year}_{decay_name}.root")
            process_success = process_file_with_pyroot(input_files, output_file, year, decay_name, args.debug)
            
            if process_success:
                # Verify with uproot
                verify_success = verify_with_uproot(output_file, args.debug)
                
                if verify_success:
                    print(f"  ✓ File {output_file} verified with uproot")
                    total_processed += 1
                    
                    with open(log_file, 'a') as log:
                        log.write(f"Year: {year}, Decay: {decay_desc}\n")
                        log.write(f"  - Processed {len(input_files)} input files\n")
                        log.write(f"  - Created {output_file}\n")
                        log.write(f"  - Successfully verified with uproot\n\n")
                else:
                    print(f"  ✗ File verification failed with uproot")
                    total_failures += 1
                    
                    with open(log_file, 'a') as log:
                        log.write(f"Year: {year}, Decay: {decay_desc}\n")
                        log.write(f"  - Processed {len(input_files)} input files\n")
                        log.write(f"  - Created {output_file}\n")
                        log.write(f"  - FAILED verification with uproot\n\n")
            else:
                print(f"  ✗ Processing failed")
                total_failures += 1
                
                with open(log_file, 'a') as log:
                    log.write(f"Year: {year}, Decay: {decay_desc}\n")
                    log.write(f"  - FAILED to process files\n\n")
    
    # Write summary to log file
    with open(log_file, 'a') as log:
        log.write("\nProcessing summary\n")
        log.write("----------------------------------------\n")
        log.write(f"Total successfully processed: {total_processed}\n")
        log.write(f"Total failures: {total_failures}\n\n")
        
        # Summary of created files
        log.write("Processed File Statistics:\n")
        log.write("==============================================\n")
        
        total_size = 0
        total_files = 0
        
        for file_path in glob.glob(os.path.join(output_dir, "*.root")):
            file_name = os.path.basename(file_path)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            total_size += file_size_mb
            total_files += 1
            
            log.write(f"{file_name}: {file_size_mb:.2f} MB\n")
        
        log.write("--------------------------------------------\n")
        log.write(f"Total processed files: {total_files}\n")
        log.write(f"Total size: {total_size:.2f} MB\n")
        log.write("==============================================\n")
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {total_processed} files")
    print(f"Failed: {total_failures} files")
    print(f"Log file saved to: {log_file}")
    print(f"Processed files are located in: {output_dir}")

if __name__ == "__main__":
    args = setup_args()
    process_files(args)
    
    # Suggest update to load_mc.py
    print("\nNext steps:")
    print("1. Update load_mc.py to use the processed files:")
    print("   Change DEFAULT_DATA_DIR to: " + (args.output_dir or os.path.join(args.base_dir, "processed")))
    print("2. Make sure code accesses the data using the standardized names:")
    print("   - KSKmKpPip_DD")
    print("   - KSKmKpPip_LL") 
    print("   - KSKpKpPim_DD")
    print("   - KSKpKpPim_LL")