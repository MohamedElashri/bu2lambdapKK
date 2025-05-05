#!/usr/bin/env python3
import uproot
import awkward as ak
import numpy as np
import os
import glob
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import gc
import resource
import psutil
import tempfile
import shutil

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def limit_memory(max_gb=8):
    """Limit maximum memory usage"""
    max_bytes = max_gb * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))

def check_disk_space(path, required_gb=10):
    """Check if there's enough disk space"""
    stats = os.statvfs(path)
    free_gb = (stats.f_bavail * stats.f_frsize) / (1024**3)
    return free_gb >= required_gb

def process_chunk_safely(chunk_arrays: Dict[str, List], out_file: Any, tree_name: str) -> None:
    """Process chunk with error handling and memory management"""
    try:
        # Only process non-empty arrays
        if not any(len(arr) > 0 for arr in chunk_arrays.values()):
            return

        # Process arrays in smaller sub-chunks if needed
        max_chunk_size = 100000  # Adjust based on memory constraints
        total_rows = len(next(iter(chunk_arrays.values()))[0])
        
        for start_idx in range(0, total_rows, max_chunk_size):
            end_idx = min(start_idx + max_chunk_size, total_rows)
            
            sub_arrays = {
                br: [arr[start_idx:end_idx] for arr in arrs]
                for br, arrs in chunk_arrays.items()
            }
            
            processed_arrays = {
                br: np.concatenate(arrs) if arrs else np.array([])
                for br, arrs in sub_arrays.items()
            }
            
            if tree_name not in out_file:
                out_file[tree_name] = processed_arrays
            else:
                out_file[tree_name].extend(processed_arrays)
            
            # Force garbage collection
            del processed_arrays
            gc.collect()
            
    except Exception as e:
        print(f"Warning: Error processing chunk: {e}")

def merge_partial_files(temp_files: List[str], output_file: str, chunk_size: int = 100000) -> None:
    """Merge partial files with controlled memory usage"""
    print(f"Merging {len(temp_files)} partial files...")
    temp_merge_files = []
    current_merge_level = 0
    
    try:
        # Merge files in multiple passes if needed
        files_to_merge = temp_files
        while len(files_to_merge) > 1:
            next_level_files = []
            
            # Merge files in groups of 3
            for i in range(0, len(files_to_merge), 3):
                group = files_to_merge[i:i + 3]
                if len(group) == 1 and i + 1 == len(files_to_merge):
                    # If it's the last single file, just pass it through
                    next_level_files.append(group[0])
                    continue
                    
                temp_output = f"{output_file}.merge{current_merge_level}.{i//3}"
                temp_merge_files.append(temp_output)
                
                print(f"Merging group of {len(group)} files...")
                with uproot.recreate(temp_output) as out_file:
                    # Process each file in the group
                    for input_file in group:
                        with uproot.open(input_file) as f:
                            for key in f.keys():
                                tree_name = key.split(';')[0]
                                tree = f[key]
                                
                                # Process tree in chunks
                                for batch in tree.iterate(step_size=chunk_size):
                                    if tree_name not in out_file:
                                        out_file[tree_name] = batch
                                    else:
                                        out_file[tree_name].extend(batch)
                                    gc.collect()  # Force garbage collection after each chunk
                
                next_level_files.append(temp_output)
            
            # Clean up previous level's files
            if current_merge_level > 0:
                for f in files_to_merge:
                    try:
                        if f not in next_level_files:
                            os.remove(f)
                    except:
                        pass
            
            files_to_merge = next_level_files
            current_merge_level += 1
            
        # Rename final merged file
        if files_to_merge:
            os.rename(files_to_merge[0], output_file)
            
    except Exception as e:
        print(f"Error during merge: {e}")
        raise
    finally:
        # Clean up any remaining temporary merge files
        for f in temp_merge_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass

def merge_root_files(decay_channel: str, input_files: List[str], output_file: str) -> None:
    """Merge ROOT files with improved error handling and memory management"""
    temp_files = []
    try:
        # Set memory limit
        limit_memory(7)  # Leave some headroom
        
        # Check disk space
        output_dir = os.path.dirname(output_file)
        if not check_disk_space(output_dir):
            raise OSError(f"Insufficient disk space in {output_dir}")
            
        # Split output into multiple files if needed
        MAX_FILE_SIZE = 1.8 * 1024 * 1024 * 1024  # 1.8GB limit for EOS
        
        print(f"Analyzing {decay_channel} structure...")
        # First pass: analyze structure and count entries
        total_entries = {}
        tree_structures = {}
        
        for input_file in tqdm(input_files, desc="Analyzing files"):
            try:
                with uproot.open(input_file) as f:
                    for key in f.keys():
                        tree_name = key.split(';')[0]
                        tree = f[key]
                        
                        if tree_name not in total_entries:
                            total_entries[tree_name] = 0
                            tree_structures[tree_name] = list(tree.keys())
                        
                        total_entries[tree_name] += tree.num_entries
            except Exception as e:
                print(f"Warning: Error analyzing file {input_file}: {e}")
                continue

        print(f"Processing {decay_channel}...")
        current_file_index = 0
        current_output = f"{output_file}.part{current_file_index}"
        temp_files.append(current_output)
        
        with uproot.recreate(current_output) as out_file:
            # Process one tree type at a time
            for tree_name in tree_structures:
                print(f"Processing tree: {tree_name}")
                
                # Process chunks of files
                chunk_size = 10000  # Reduced chunk size
                file_chunks = [input_files[i:i + 2] for i in range(0, len(input_files), 2)]
                
                for file_chunk in tqdm(file_chunks, desc=f"{tree_name} file chunks"):
                    chunk_arrays = {branch: [] for branch in tree_structures[tree_name]}
                    current_chunk_size = 0
                    
                    # Process files in current chunk
                    for input_file in file_chunk:
                        try:
                            with uproot.open(input_file) as f:
                                if tree_name not in f:
                                    continue
                                    
                                tree = f[tree_name]
                                
                                # Process tree in smaller chunks
                                for batch in tree.iterate(step_size=chunk_size):
                                    try:
                                        for branch in tree_structures[tree_name]:
                                            data = ak.to_numpy(batch[branch])
                                            chunk_arrays[branch].append(data)
                                            current_chunk_size += data.nbytes
                                        
                                        # Check if current file is getting too large
                                        if os.path.getsize(current_output) > MAX_FILE_SIZE:
                                            # Close current file and start a new one
                                            out_file.close()
                                            current_file_index += 1
                                            current_output = f"{output_file}.part{current_file_index}"
                                            temp_files.append(current_output)
                                            out_file = uproot.recreate(current_output)
                                        
                                        # Check memory usage and write if getting too high
                                        if get_memory_usage() > 6 or current_chunk_size > 500 * 1024 * 1024:  # 500MB
                                            process_chunk_safely(chunk_arrays, out_file, tree_name)
                                            chunk_arrays = {branch: [] for branch in tree_structures[tree_name]}
                                            current_chunk_size = 0
                                            gc.collect()
                                            
                                    except Exception as e:
                                        print(f"Warning: Error processing branch {branch}: {e}")
                                        continue
                                        
                        except Exception as e:
                            print(f"Error processing file {input_file}: {e}")
                            continue
                    
                    # Write remaining data for current chunk
                    if any(len(arr) > 0 for arr in chunk_arrays.values()):
                        process_chunk_safely(chunk_arrays, out_file, tree_name)
                    
                    # Clear memory
                    del chunk_arrays
                    gc.collect()
        
        # If we have multiple parts, merge them using the new merge strategy
        if len(temp_files) > 1:
            merge_partial_files(temp_files, output_file)
        else:
            # Just rename the single temporary file
            os.rename(temp_files[0], output_file)
            
        print(f"Successfully created merged file: {output_file}")
        
    except Exception as e:
        print(f"Error merging files for {decay_channel}: {e}")
        # Clean up any temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        raise
    finally:
        # Final garbage collection
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description='Merge ROOT files by decay channel.')
    parser.add_argument('--input', required=True, help='Input base directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--decay', help='Comma-separated list of decay channels to process (e.g., "KSKmKpPip,KSKpKpPim")')
    parser.add_argument('--max-memory', type=float, default=7.0, 
                       help='Maximum memory usage in GB')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Number of entries to process in each chunk')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Find ROOT files based on decay channel specification
    if args.decay:
        # Process specified decay channels
        decay_channels = [ch.strip() for ch in args.decay.split(',')]
        print(f"Processing specified decay channels: {', '.join(decay_channels)}")
        
        for decay_channel in decay_channels:
            pattern = os.path.join(args.input, "**", f"{decay_channel}.root")
            input_files = sorted(glob.glob(pattern, recursive=True))
            
            if not input_files:
                print(f"No files found for decay channel: {decay_channel}")
                continue
            
            output_file = os.path.join(args.output, f"{decay_channel}.root")
            try:
                merge_root_files(decay_channel, input_files, output_file)
                gc.collect()
            except Exception as e:
                print(f"Failed to process {decay_channel}: {e}")
                try:
                    if os.path.exists(output_file):
                        os.remove(output_file)
                except:
                    pass
    else:
        # Process all decay channels
        pattern = os.path.join(args.input, "**", "*.root")
        all_files = glob.glob(pattern, recursive=True)
        decay_channels = set(os.path.basename(f).split('.')[0] for f in all_files)
        
        if not decay_channels:
            print("No ROOT files found!")
            return
        
        print(f"Found {len(decay_channels)} decay channels to process")
        
        # Process each decay channel sequentially
        for decay_channel in sorted(decay_channels):
            print(f"\nProcessing decay channel: {decay_channel}")
            
            # Find all ROOT files for this decay channel
            pattern = os.path.join(args.input, "**", f"{decay_channel}.root")
            input_files = sorted(glob.glob(pattern, recursive=True))
            
            if not input_files:
                print(f"No files found for {decay_channel}")
                continue
            
            # Create output filename
            output_file = os.path.join(args.output, f"{decay_channel}.root")
            
            try:
                merge_root_files(decay_channel, input_files, output_file)
                gc.collect()
            except Exception as e:
                print(f"Failed to process {decay_channel}: {e}")
                try:
                    if os.path.exists(output_file):
                        os.remove(output_file)
                except:
                    pass
                continue

if __name__ == "__main__":
    main()
