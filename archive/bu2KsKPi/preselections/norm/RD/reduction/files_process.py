#!/usr/bin/env python3
import uproot
import awkward as ak
import numpy as np
import os
import argparse
from tqdm import tqdm
from typing import List, Dict, Any
import glob
import multiprocessing
from functools import partial
import time


BRANCHES_TO_KEEP = [
    "B_BPVCORRM", "B_ENDVERTEX_X", "B_ENDVERTEX_Y", "B_ENDVERTEX_Z", "B_ENDVERTEX_XERR", "B_ENDVERTEX_YERR",
    "B_ENDVERTEX_ZERR", "B_ENDVERTEX_CHI2", "B_ENDVERTEX_NDOF", "B_ENDVERTEX_COV_", "B_OWNPV_X", "B_OWNPV_Y",
    "B_OWNPV_Z", "B_OWNPV_XERR", "B_OWNPV_YERR", "B_OWNPV_ZERR", "B_OWNPV_CHI2", "B_OWNPV_NDOF", "B_OWNPV_COV_",
    "B_IP_OWNPV", "B_IPCHI2_OWNPV", "B_FD_OWNPV", "B_FDCHI2_OWNPV", "B_DIRA_OWNPV", "B_P", "B_PT", "B_PE", "B_PX",
    "B_PY", "B_PZ", "B_MM", "B_MMERR", "B_M", "B_ID", "B_TAU", "B_TAUERR", "B_TAUCHI2", "KS_ENDVERTEX_X", "KS_ENDVERTEX_Y",
    "KS_ENDVERTEX_Z", "KS_ENDVERTEX_XERR", "KS_ENDVERTEX_YERR", "KS_ENDVERTEX_ZERR", "KS_ENDVERTEX_CHI2", "KS_ENDVERTEX_NDOF", "KS_ENDVERTEX_COV_", "KS_OWNPV_X", "KS_OWNPV_Y",
    "KS_OWNPV_Z", "KS_OWNPV_XERR", "KS_OWNPV_YERR", "KS_OWNPV_ZERR", "KS_OWNPV_CHI2", "KS_OWNPV_NDOF", "KS_OWNPV_COV_",
    "KS_IP_OWNPV", "KS_IPCHI2_OWNPV", "KS_FD_OWNPV", "KS_FDCHI2_OWNPV", "KS_DIRA_OWNPV", "KS_ORIVX_X", "KS_ORIVX_Y",
    "KS_ORIVX_Z", "KS_ORIVX_XERR", "KS_ORIVX_YERR", "KS_ORIVX_ZERR", "KS_ORIVX_CHI2", "KS_ORIVX_NDOF", "KS_ORIVX_COV_",
    "KS_FD_ORIVX", "KS_FDCHI2_ORIVX", "KS_DIRA_ORIVX", "KS_P", "KS_PT", "KS_PE", "KS_PX", "KS_PY", "KS_PZ", "KS_MM",
    "KS_MMERR", "KS_M", "KS_TAU", "KS_TAUERR", "KS_TAUCHI2", "KS_P0_OWNPV_X", "KS_P0_OWNPV_Y", "KS_P0_OWNPV_Z",
    "KS_P0_OWNPV_XERR", "KS_P0_OWNPV_YERR", "KS_P0_OWNPV_ZERR", "KS_P0_OWNPV_CHI2", "KS_P0_OWNPV_NDOF", "KS_P0_OWNPV_COV_",
    "KS_P0_IP_OWNPV", "KS_P0_IPCHI2_OWNPV", "KS_P0_ORIVX_X", "KS_P0_ORIVX_Y", "KS_P0_ORIVX_Z", "KS_P0_ORIVX_XERR",
    "KS_P0_ORIVX_YERR", "KS_P0_ORIVX_ZERR", "KS_P0_ORIVX_CHI2", "KS_P0_ORIVX_NDOF", "KS_P0_ORIVX_COV_", "KS_P0_P",
    "KS_P0_PT", "KS_P0_PE", "KS_P0_PX", "KS_P0_PY", "KS_P0_PZ", "KS_P0_M", "KS_P0_ID", "KS_P0_PIDe", "KS_P0_PIDmu",
    "KS_P0_PIDK", "KS_P0_PIDp", "KS_P0_ProbNNe", "KS_P0_ProbNNk", "KS_P0_ProbNNp", "KS_P0_ProbNNpi", "KS_P0_ProbNNmu",
    "KS_P0_ProbNNghost", "KS_P0_hasMuon", "KS_P0_isMuon", "KS_P0_TRACK_Type", "KS_P0_TRACK_Key", "KS_P0_TRACK_CHI2NDOF",
    "KS_P0_TRACK_PCHI2", "KS_P0_TRACK_MatchCHI2", "KS_P0_TRACK_GhostProb", "KS_P0_TRACK_CloneDist", "KS_P0_TRACK_Likelihood",
    "KS_P1_OWNPV_X", "KS_P1_OWNPV_Y", "KS_P1_OWNPV_Z", "KS_P1_OWNPV_XERR", "KS_P1_OWNPV_YERR", "KS_P1_OWNPV_ZERR",
    "KS_P1_OWNPV_CHI2", "KS_P1_OWNPV_NDOF", "KS_P1_OWNPV_COV_", "KS_P1_IP_OWNPV", "KS_P1_IPCHI2_OWNPV", "KS_P1_ORIVX_X",
    "KS_P1_ORIVX_Y", "KS_P1_ORIVX_Z", "KS_P1_ORIVX_XERR", "KS_P1_ORIVX_YERR", "KS_P1_ORIVX_ZERR", "KS_P1_ORIVX_CHI2",
    "KS_P1_ORIVX_NDOF", "KS_P1_ORIVX_COV_", "KS_P1_P", "KS_P1_PT", "KS_P1_PE", "KS_P1_PX", "KS_P1_PY", "KS_P1_PZ",
    "KS_P1_M", "KS_P1_ID", "KS_P1_PIDe", "KS_P1_PIDmu", "KS_P1_PIDK", "KS_P1_PIDp", "KS_P1_ProbNNe", "KS_P1_ProbNNk",
    "KS_P1_ProbNNp", "KS_P1_ProbNNpi", "KS_P1_ProbNNmu", "KS_P1_ProbNNghost", "KS_P1_hasMuon", "KS_P1_isMuon",
    "KS_P1_TRACK_Type", "KS_P1_TRACK_Key", "KS_P1_TRACK_CHI2NDOF", "KS_P1_TRACK_PCHI2", "KS_P1_TRACK_MatchCHI2",
    "KS_P1_TRACK_GhostProb", "KS_P1_TRACK_CloneDist", "KS_P1_TRACK_Likelihood", "P0_OWNPV_X", "P0_OWNPV_Y", "P0_OWNPV_Z",
    "P0_OWNPV_XERR", "P0_OWNPV_YERR", "P0_OWNPV_ZERR", "P0_OWNPV_CHI2", "P0_OWNPV_NDOF", "P0_OWNPV_COV_", "P0_IP_OWNPV",
    "P0_IPCHI2_OWNPV", "P0_ORIVX_X", "P0_ORIVX_Y", "P0_ORIVX_Z", "P0_ORIVX_XERR", "P0_ORIVX_YERR", "P0_ORIVX_ZERR",
    "P0_ORIVX_CHI2", "P0_ORIVX_NDOF", "P0_ORIVX_COV_", "P0_P", "P0_PT", "P0_PE", "P0_PX", "P0_PY", "P0_PZ", "P0_M",
    "P0_ID", "P0_PIDe", "P0_PIDmu", "P0_PIDK", "P0_PIDp", "P0_ProbNNe", "P0_ProbNNk", "P0_ProbNNp", "P0_ProbNNpi",
    "P0_ProbNNmu", "P0_ProbNNghost", "P0_hasMuon", "P0_isMuon", "P0_TRACK_Type", "P0_TRACK_Key", "P0_TRACK_CHI2NDOF",
    "P0_TRACK_PCHI2", "P0_TRACK_MatchCHI2", "P0_TRACK_GhostProb", "P0_TRACK_CloneDist", "P0_TRACK_Likelihood",
    "P1_OWNPV_X", "P1_OWNPV_Y", "P1_OWNPV_Z", "P1_OWNPV_XERR", "P1_OWNPV_YERR", "P1_OWNPV_ZERR", "P1_OWNPV_CHI2",
    "P1_OWNPV_NDOF", "P1_OWNPV_COV_", "P1_IP_OWNPV", "P1_IPCHI2_OWNPV", "P1_ORIVX_X", "P1_ORIVX_Y", "P1_ORIVX_Z",
    "P1_ORIVX_XERR", "P1_ORIVX_YERR", "P1_ORIVX_ZERR", "P1_ORIVX_CHI2", "P1_ORIVX_NDOF", "P1_ORIVX_COV_", "P1_P",
    "P1_PT", "P1_PE", "P1_PX", "P1_PY", "P1_PZ", "P1_M", "P1_ID", "P1_PIDe", "P1_PIDmu", "P1_PIDK", "P1_PIDp",
    "P1_ProbNNe", "P1_ProbNNk", "P1_ProbNNp", "P1_ProbNNpi", "P1_ProbNNmu", "P1_ProbNNghost", "P1_hasMuon", "P1_isMuon",
    "P1_TRACK_Type", "P1_TRACK_Key", "P1_TRACK_CHI2NDOF", "P1_TRACK_PCHI2", "P1_TRACK_MatchCHI2", "P1_TRACK_GhostProb",
    "P1_TRACK_CloneDist", "P1_TRACK_Likelihood", "P2_OWNPV_X", "P2_OWNPV_Y", "P2_OWNPV_Z", "P2_OWNPV_XERR",
    "P2_OWNPV_YERR", "P2_OWNPV_ZERR", "P2_OWNPV_CHI2", "P2_OWNPV_NDOF", "P2_OWNPV_COV_", "P2_IP_OWNPV", "P2_IPCHI2_OWNPV",
    "P2_ORIVX_X", "P2_ORIVX_Y", "P2_ORIVX_Z", "P2_ORIVX_XERR", "P2_ORIVX_YERR", "P2_ORIVX_ZERR", "P2_ORIVX_CHI2",
    "P2_ORIVX_NDOF", "P2_ORIVX_COV_", "P2_P", "P2_PT", "P2_PE", "P2_PX", "P2_PY", "P2_PZ", "P2_M", "P2_ID", "P2_PIDe",
    "P2_PIDmu", "P2_PIDK", "P2_PIDp", "P2_ProbNNe", "P2_ProbNNk", "P2_ProbNNp", "P2_ProbNNpi", "P2_ProbNNmu",
    "P2_ProbNNghost", "P2_hasMuon", "P2_isMuon", "P2_TRACK_Type", "P2_TRACK_Key", "P2_TRACK_CHI2NDOF", "P2_TRACK_PCHI2",
    "P2_TRACK_MatchCHI2", "P2_TRACK_GhostProb", "P2_TRACK_CloneDist", "P2_TRACK_Likelihood", "nCandidate", "totCandidates",
    "EventInSequence", "runNumber", "eventNumber", "BCID", "BCType", "OdinTCK", "L0DUTCK","Polarity",
    "nPV", "PVX", "PVY", "PVZ", "PVXERR", "PVYERR", "PVZERR", "PVCHI2", "PVNDOF", "PVNTRACKS", "nPVs","nTracks"]

def prepare_for_writing(data: ak.Array) -> Dict[str, np.ndarray]:
    """Convert awkward arrays to a format suitable for writing to ROOT while preserving structure."""
    output = {}
    for field in data.fields:
        try:
            arr = ak.to_numpy(data[field])
            output[field] = arr
        except Exception:
            try:
                padded = ak.fill_none(ak.pad_none(data[field], 1, clip=True), -999)
                arr = ak.to_numpy(padded)
                output[field] = arr
            except Exception as e:
                print(f"Warning: Could not convert field {field}: {e}")
                continue
    return output

def process_root_file(input_file: str, output_dir: str) -> None:
    """Process a single ROOT file and save decay trees."""
    # Create output directory based on input filename
    file_basename = os.path.basename(input_file).split('.')[0]
    file_output_dir = os.path.join(output_dir, file_basename)
    os.makedirs(file_output_dir, exist_ok=True)
    
    print(f"Processing file: {input_file}")
    print(f"Output directory: {file_output_dir}")
    
    # Open input ROOT file
    with uproot.open(input_file) as file:
        # Extract unique decay modes
        decay_modes = set()
        for key in file.keys():
            if isinstance(file[key], uproot.ReadOnlyDirectory):
                base_name = key.split(";")[0]
                if base_name:
                    decay_mode = "_".join(base_name.split("_")[:-2])
                    if decay_mode:
                        decay_modes.add(decay_mode)
        
        # Process each decay mode
        for decay_mode in decay_modes:
            output_file = os.path.join(file_output_dir, f"{decay_mode}.root")
            
            # Initialize output file
            with uproot.recreate(output_file) as out_file:
                # Process both LL and DD versions
                for category in ["LL", "DD"]:
                    tree_path = f"{decay_mode}_{category}_Tuple/DecayTree"
                    
                    try:
                        if tree_path not in file:
                            continue
                            
                        tree = file[tree_path]
                        available_branches = tree.keys()
                        branches_to_read = [b for b in BRANCHES_TO_KEEP if b in available_branches]
                        
                        if not branches_to_read:
                            continue
                        
                        # Set up the output tree
                        out_tree_name = f"{decay_mode}_{category}"
                        first_chunk = True
                        
                        # Read and process data in chunks
                        chunk_size = 100000
                        for chunk in tree.iterate(branches_to_read, library="ak", step_size=chunk_size):
                            processed_chunk = prepare_for_writing(chunk)
                            
                            # Write the chunk to the output file
                            if first_chunk:
                                # For the first chunk, create the tree
                                out_file[out_tree_name] = processed_chunk
                                first_chunk = False
                            else:
                                # For subsequent chunks, extend the existing tree
                                out_file[out_tree_name].extend(processed_chunk)
                        
                    except Exception as e:
                        print(f"Error processing {tree_path}: {e}")
                        continue
            
            # Check if the output file was created successfully
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print(f"Created: {output_file}")
            else:
                print(f"Warning: Failed to create or empty file: {output_file}")
                # Remove empty files
                if os.path.exists(output_file):
                    os.remove(output_file)

def process_multiple_files(input_files: List[str], output_dir: str, num_processes: int) -> None:
    """Process multiple ROOT files in parallel using multiprocessing."""
    if num_processes <= 0:
        num_processes = multiprocessing.cpu_count()
    
    print(f"Using {num_processes} processes for parallel processing")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Create a partial function with fixed output_dir parameter
        process_func = partial(process_root_file, output_dir=output_dir)
        
        # Process files in parallel with a progress bar
        list(tqdm(
            pool.imap_unordered(process_func, input_files),
            total=len(input_files),
            desc="Processing files"
        ))

def main():
    parser = argparse.ArgumentParser(description='Process ROOT files containing decay trees.')
    parser.add_argument('--input', required=True, nargs='+', help='Input ROOT file(s) or pattern(s)')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--processes', type=int, default=0, 
                        help='Number of processes to use (default: number of CPU cores)')
    
    args = parser.parse_args()
    
    # Expand any glob patterns in input files
    input_files = []
    for pattern in args.input:
        if '*' in pattern:
            matched_files = glob.glob(pattern)
            input_files.extend(matched_files)
        else:
            input_files.append(pattern)
    
    # Remove duplicates and sort
    input_files = sorted(set(input_files))
    
    if not input_files:
        print("No input files found!")
        return
    
    print(f"Found {len(input_files)} input files")
    os.makedirs(args.output, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    process_multiple_files(input_files, args.output, args.processes)
    
    # Print timing information
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Average processing time per file: {elapsed_time / len(input_files):.2f} seconds")

if __name__ == "__main__":
    # Ensure proper multiprocessing behavior
    multiprocessing.freeze_support()
    main()
