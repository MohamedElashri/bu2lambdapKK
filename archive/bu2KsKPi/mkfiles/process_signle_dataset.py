#!/usr/bin/env python3
import uproot
import numpy as np
import os
import time
import argparse
from tqdm import tqdm
import psutil
import gc

def monitor_memory():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")

def process_file(input_file, output_file, branches_to_keep, tuples_to_keep, name_mapping):
    """Process a single ROOT file, extracting specified branches and tuples"""
    
    # Check if output file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists, skipping...")
        return
    
    print(f"Processing {input_file} -> {output_file}")
    
    # Get original file size
    try:
        original_size = os.path.getsize(input_file)
        print(f"Original file size: {original_size / (1024*1024):.2f} MB")
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found!")
        return

    # Open the original file
    try:
        f = uproot.open(input_file)
    except Exception as e:
        print(f"Error opening {input_file}: {e}")
        return

    # Dictionary to store data for each TDirectory
    all_data = {}
    total_entries = 0

    # Process each desired TDirectory
    for tuple_name in tuples_to_keep:
        try:
            full_tuple_name = f"{tuple_name};1" if not tuple_name.endswith(';1') else tuple_name
            tree_path = f"{full_tuple_name}/DecayTree;1"
            
            if tree_path not in f:
                print(f"Warning: {tree_path} not found in {input_file}")
                continue
                
            tree = f[tree_path]
            
            # Get all branches and check which ones from our list actually exist
            available_branches = set(tree.keys())
            branches_to_extract = [b for b in branches_to_keep if b in available_branches]
            
            # Create dictionary to store the data for this TDirectory
            data = {}
            n_entries = tree.num_entries
            total_entries += n_entries
            
            # Extract data from each branch with progress bar
            print(f"Extracting {len(branches_to_extract)} branches from {n_entries} entries...")
            for branch in tqdm(branches_to_extract, desc=f"Reading {tuple_name}"):
                # Manage memory - read in smaller chunks if many entries
                if n_entries > 1000000:  # For very large trees
                    chunk_size = 500000
                    arrays = []
                    for i in range(0, n_entries, chunk_size):
                        end = min(i + chunk_size, n_entries)
                        arrays.append(tree[branch].array(entry_start=i, entry_stop=end))
                        gc.collect()  # Explicit garbage collection
                    data[branch] = np.concatenate(arrays)
                else:
                    data[branch] = tree[branch].array()
            
            # Store data with the new directory name (without _Tuple)
            new_name = name_mapping.get(full_tuple_name, full_tuple_name.split(';')[0])
            all_data[new_name] = data
            
            monitor_memory()
            gc.collect()  # Explicit garbage collection
            
        except KeyError as e:
            print(f"Error accessing tree in {tuple_name}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {tuple_name}: {e}")
            continue

    # Close input file to free resources
    f.close()
    gc.collect()

    # Write to a new ROOT file
    print("\nWriting to new file...")
    try:
        with uproot.recreate(output_file) as fout:
            for dir_name, data in all_data.items():
                # Create a directory for each tuple with the new name
                fout[f"{dir_name}/DecayTree"] = data
                gc.collect()  # Collect after each directory write
    except Exception as e:
        print(f"Error writing output file: {e}")
        # Remove output file if it exists but is incomplete
        if os.path.exists(output_file):
            os.remove(output_file)
        return

    # Get new file size
    new_size = os.path.getsize(output_file)
    print(f"\nNew file size: {new_size / (1024*1024):.2f} MB")
    print(f"Size reduction: {(original_size - new_size) / (1024*1024):.2f} MB ({(1 - new_size/original_size) * 100:.2f}%)")
    print(f"Total entries processed: {total_entries}")

def main():
    parser = argparse.ArgumentParser(description='Process ROOT files to extract specific branches')
    parser.add_argument('input_file', help='Input ROOT file path')
    parser.add_argument('output_file', help='Output ROOT file path')
    parser.add_argument('--config', choices=['default'], default='default', 
                        help='Configuration preset to use')
    args = parser.parse_args()
    
    # Define branches  to keep
    branches_to_keep = [
    'B_ENDVERTEX_X', 'B_ENDVERTEX_Y', 'B_ENDVERTEX_Z', 'B_ENDVERTEX_XERR', 'B_ENDVERTEX_YERR', 
    'B_ENDVERTEX_ZERR', 'B_ENDVERTEX_CHI2', 'B_ENDVERTEX_NDOF', 'B_ENDVERTEX_COV_', 
    'B_OWNPV_X', 'B_OWNPV_Y', 'B_OWNPV_Z', 'B_OWNPV_XERR', 'B_OWNPV_YERR', 
    'B_OWNPV_ZERR', 'B_OWNPV_CHI2', 'B_OWNPV_NDOF', 'B_OWNPV_COV_', 
    'B_IP_OWNPV', 'B_IPCHI2_OWNPV', 'B_FD_OWNPV', 'B_FDCHI2_OWNPV', 'B_DIRA_OWNPV', 
    'B_P', 'B_PT', 'B_PE', 'B_PX', 'B_PY', 'B_PZ', 'B_MM', 'B_MMERR', 'B_M', 'B_ID', 
    'B_TAU', 'B_TAUERR', 'B_TAUCHI2', 'B_L0Global_Dec', 'B_L0Global_TIS', 'B_L0Global_TOS', 
    'B_L0HadronDecision_TOS', 'B_Hlt1TrackMVADecision_TOS', 'B_Hlt1TwoTrackMVADecision_TOS', 
    'B_Hlt1TrackMuonDecision_Dec', 'B_Hlt1TrackMuonDecision_TIS', 'B_Hlt1TrackMuonDecision_TOS', 
    'B_Hlt2Topo2BodyDecision_TOS', 'B_Hlt2Topo3BodyDecision_TOS', 'B_Hlt2Topo4BodyDecision_TOS', 
    'KS_ENDVERTEX_X', 'KS_ENDVERTEX_Y', 'KS_ENDVERTEX_Z', 'KS_ENDVERTEX_XERR', 'KS_ENDVERTEX_YERR', 
    'KS_ENDVERTEX_ZERR', 'KS_ENDVERTEX_CHI2', 'KS_ENDVERTEX_NDOF', 'KS_ENDVERTEX_COV_', 
    'KS_OWNPV_X', 'KS_OWNPV_Y', 'KS_OWNPV_Z', 'KS_OWNPV_XERR', 'KS_OWNPV_YERR', 
    'KS_OWNPV_ZERR', 'KS_OWNPV_CHI2', 'KS_OWNPV_NDOF', 'KS_OWNPV_COV_', 
    'KS_IP_OWNPV', 'KS_IPCHI2_OWNPV', 'KS_FD_OWNPV', 'KS_FDCHI2_OWNPV', 'KS_DIRA_OWNPV', 
    'KS_P', 'KS_PT', 'KS_PE', 'KS_PX', 'KS_PY', 'KS_PZ', 'KS_MM', 'KS_MMERR', 'KS_M', 
    'KS_TAU', 'KS_TAUERR', 'KS_TAUCHI2', 'KS_P0_OWNPV_X', 'KS_P0_OWNPV_Y', 'KS_P0_OWNPV_Z', 
    'KS_P0_OWNPV_XERR', 'KS_P0_OWNPV_YERR', 'KS_P0_OWNPV_ZERR', 'KS_P0_OWNPV_CHI2', 
    'KS_P0_OWNPV_NDOF', 'KS_P0_OWNPV_COV_', 'KS_P0_IP_OWNPV', 'KS_P0_IPCHI2_OWNPV', 
    'KS_P0_P', 'KS_P0_PT', 'KS_P0_PE', 'KS_P0_PX', 'KS_P0_PY', 'KS_P0_PZ', 'KS_P0_M', 
    'KS_P0_ID', 'KS_P0_PIDe', 'KS_P0_PIDmu', 'KS_P0_PIDK', 'KS_P0_PIDp', 'KS_P0_ProbNNe', 
    'KS_P0_ProbNNk', 'KS_P0_ProbNNp', 'KS_P0_ProbNNpi', 'KS_P0_ProbNNmu', 'KS_P0_ProbNNghost', 
    'KS_P0_hasMuon', 'KS_P0_isMuon', 'KS_P0_hasRich', 'KS_P0_TRACK_Type', 'KS_P0_TRACK_Key', 
    'KS_P0_TRACK_CHI2NDOF', 'KS_P0_TRACK_PCHI2', 'KS_P0_TRACK_MatchCHI2', 'KS_P0_TRACK_GhostProb', 
    'KS_P0_TRACK_CloneDist', 'KS_P0_TRACK_Likelihood', 'KS_P1_OWNPV_X', 'KS_P1_OWNPV_Y', 
    'KS_P1_OWNPV_Z', 'KS_P1_OWNPV_XERR', 'KS_P1_OWNPV_YERR', 'KS_P1_OWNPV_ZERR', 
    'KS_P1_OWNPV_CHI2', 'KS_P1_OWNPV_NDOF', 'KS_P1_OWNPV_COV_', 'KS_P1_IP_OWNPV', 
    'KS_P1_IPCHI2_OWNPV', 'KS_P1_P', 'KS_P1_PT', 'KS_P1_PE', 'KS_P1_PX', 'KS_P1_PY', 
    'KS_P1_PZ', 'KS_P1_M', 'KS_P1_ID', 'KS_P1_PIDe', 'KS_P1_PIDmu', 'KS_P1_PIDK', 
    'KS_P1_PIDp', 'KS_P1_ProbNNe', 'KS_P1_ProbNNk', 'KS_P1_ProbNNp', 'KS_P1_ProbNNpi', 
    'KS_P1_ProbNNmu', 'KS_P1_ProbNNghost', 'KS_P1_hasMuon', 'KS_P1_isMuon', 'KS_P1_hasRich', 
    'KS_P1_TRACK_Type', 'KS_P1_TRACK_Key', 'KS_P1_TRACK_CHI2NDOF', 'KS_P1_TRACK_PCHI2', 
    'KS_P1_TRACK_MatchCHI2', 'KS_P1_TRACK_GhostProb', 'KS_P1_TRACK_CloneDist', 
    'KS_P1_TRACK_Likelihood', 'P0_OWNPV_X', 'P0_OWNPV_Y', 'P0_OWNPV_Z', 'P0_OWNPV_XERR', 
    'P0_OWNPV_YERR', 'P0_OWNPV_ZERR', 'P0_OWNPV_CHI2', 'P0_OWNPV_NDOF', 'P0_OWNPV_COV_', 
    'P0_IP_OWNPV', 'P0_IPCHI2_OWNPV', 'P0_P', 'P0_PT', 'P0_PE', 'P0_PX', 'P0_PY', 
    'P0_PZ', 'P0_M', 'P0_ID', 'P0_PIDe', 'P0_PIDmu', 'P0_PIDK', 'P0_PIDp', 'P0_ProbNNe', 
    'P0_ProbNNk', 'P0_ProbNNp', 'P0_ProbNNpi', 'P0_ProbNNmu', 'P0_ProbNNghost', 'P0_hasMuon', 
    'P0_isMuon', 'P0_hasRich', 'P0_TRACK_Type', 'P0_TRACK_Key', 'P0_TRACK_CHI2NDOF', 
    'P0_TRACK_PCHI2', 'P0_TRACK_GhostProb', 'P1_OWNPV_X', 'P1_OWNPV_Y', 'P1_OWNPV_Z', 
    'P1_OWNPV_XERR', 'P1_OWNPV_YERR', 'P1_OWNPV_ZERR', 'P1_OWNPV_CHI2', 'P1_OWNPV_NDOF', 
    'P1_OWNPV_COV_', 'P1_IP_OWNPV', 'P1_IPCHI2_OWNPV', 'P1_P', 'P1_PT', 'P1_PE', 
    'P1_PX', 'P1_PY', 'P1_PZ', 'P1_M', 'P1_ID', 'P1_PIDe', 'P1_PIDmu', 'P1_PIDK', 
    'P1_PIDp', 'P1_ProbNNe', 'P1_ProbNNk', 'P1_ProbNNp', 'P1_ProbNNpi', 'P1_ProbNNmu', 
    'P1_ProbNNghost', 'P1_hasMuon', 'P1_isMuon', 'P1_hasRich', 'P1_TRACK_Type', 'P1_TRACK_Key', 
    'P1_TRACK_CHI2NDOF', 'P1_TRACK_PCHI2', 'P1_TRACK_GhostProb', 'P2_OWNPV_X', 'P2_OWNPV_Y', 
    'P2_OWNPV_Z', 'P2_OWNPV_XERR', 'P2_OWNPV_YERR', 'P2_OWNPV_ZERR', 'P2_OWNPV_CHI2', 
    'P2_OWNPV_NDOF', 'P2_OWNPV_COV_', 'P2_IP_OWNPV', 'P2_IPCHI2_OWNPV', 'P2_P', 
    'P2_PT', 'P2_PE', 'P2_PX', 'P2_PY', 'P2_PZ', 'P2_M', 'P2_ID', 'P2_PIDe', 'P2_PIDmu', 
    'P2_PIDK', 'P2_PIDp', 'P2_ProbNNe', 'P2_ProbNNk', 'P2_ProbNNp', 'P2_ProbNNpi', 
    'P2_ProbNNmu', 'P2_ProbNNghost', 'P2_hasMuon', 'P2_isMuon', 'P2_hasRich', 'P2_TRACK_Type', 
    'P2_TRACK_Key', 'P2_TRACK_CHI2NDOF', 'P2_TRACK_GhostProb', 'nCandidate', 'totCandidates', 
    'EventInSequence', 'runNumber', 'eventNumber', 'OdinTCK', 'L0DUTCK', 'HLT1TCK', 
    'HLT2TCK', 'GpsTime', 'Polarity', 'nPV', 'PVX', 'PVY', 'PVZ', 'PVXERR', 'PVYERR', 
    'PVZERR', 'PVCHI2', 'PVNDOF', 'PVNTRACKS', 'nPVs', 'nTracks', 'nBackTracks', 
    'L0Global', 'Hlt1Global', 'Hlt2Global'
    ]

    
    # Tuples to keep
    tuples_to_keep = [
        'KSKmKpKp_LL_Tuple', 'KSKmKpKp_DD_Tuple',
        'KSKpKpPim_LL_Tuple', 'KSKpKpPim_DD_Tuple'
    ]
    
    # Name mapping
    name_mapping = {
        'KSKmKpKp_LL_Tuple;1': 'KSKmKpKp_LL',
        'KSKmKpKp_DD_Tuple;1': 'KSKmKpKp_DD',
        'KSKpKpPim_LL_Tuple;1': 'KSKpKpPim_LL',
        'KSKpKpPim_DD_Tuple;1': 'KSKpKpPim_DD'
    }
    
    process_file(args.input_file, args.output_file, branches_to_keep, tuples_to_keep, name_mapping)

if __name__ == "__main__":
    main()
