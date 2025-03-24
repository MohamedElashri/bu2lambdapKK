import os
import glob
import uproot
import multiprocessing
import awkward as ak
from typing import List

# 1) List of branches to keep
keep_branches = [
    # B candidate
    "B_BPVCORRM", "B_ENDVERTEX_X", "B_ENDVERTEX_Y", "B_ENDVERTEX_Z",
    "B_ENDVERTEX_CHI2", "B_OWNPV_X", "B_OWNPV_Y", "B_OWNPV_Z",
    "B_IP_OWNPV", "B_IPCHI2_OWNPV", "B_FD_OWNPV", "B_FDCHI2_OWNPV",
    "B_DIRA_OWNPV", "B_P", "B_PT", "B_PE", "B_PX", "B_PY", "B_PZ",
    "B_MM", "B_MMERR", "B_M", "B_ID", "B_TAU", "B_TAUERR", "B_TAUCHI2",
    
    # K_S candidate
    "KS_ENDVERTEX_X", "KS_ENDVERTEX_Y", "KS_ENDVERTEX_Z",
    "KS_ENDVERTEX_CHI2", "KS_OWNPV_X", "KS_OWNPV_Y", "KS_OWNPV_Z",
    "KS_IP_OWNPV", "KS_IPCHI2_OWNPV", "KS_FD_OWNPV", "KS_FDCHI2_OWNPV",
    "KS_DIRA_OWNPV", "KS_P", "KS_PT", "KS_PE", "KS_PX", "KS_PY", "KS_PZ",
    "KS_MM", "KS_MMERR", "KS_M", "KS_TAU", "KS_TAUERR", "KS_TAUCHI2",

    # K_S daughter pions
    "KS_P0_P",  "KS_P0_PT",  "KS_P0_PE",
    "KS_P0_PX", "KS_P0_PY",  "KS_P0_PZ",
    "KS_P0_M",  "KS_P0_ID",
    "KS_P0_IP_OWNPV",     
    "KS_P0_IPCHI2_OWNPV", 
    "KS_P0_PIDK", "KS_P0_PIDp", "KS_P0_PIDe", "KS_P0_PIDmu",
    "KS_P0_ProbNNk", "KS_P0_ProbNNpi", "KS_P0_ProbNNp",

    "KS_P1_P",  "KS_P1_PT",  "KS_P1_PE",
    "KS_P1_PX", "KS_P1_PY",  "KS_P1_PZ",
    "KS_P1_M",  "KS_P1_ID",
    "KS_P1_IP_OWNPV",
    "KS_P1_IPCHI2_OWNPV",
    "KS_P1_PIDK", "KS_P1_PIDp", "KS_P1_PIDe", "KS_P1_PIDmu",
    "KS_P1_ProbNNk", "KS_P1_ProbNNpi", "KS_P1_ProbNNp",

    # Other final-state tracks (the non-KS children)
    # e.g., B -> KS K+ K- pi+ might appear as P0, P1, P2, etc.
    "P0_P", "P0_PT", "P0_PE", "P0_PX", "P0_PY", "P0_PZ", "P0_M", "P0_ID",
    "P0_IP_OWNPV", "P0_IPCHI2_OWNPV",
    "P0_PIDK", "P0_PIDp", "P0_PIDe", "P0_PIDmu",
    "P0_ProbNNk", "P0_ProbNNpi", "P0_ProbNNp",

    "P1_P", "P1_PT", "P1_PE", "P1_PX", "P1_PY", "P1_PZ", "P1_M", "P1_ID",
    "P1_IP_OWNPV", "P1_IPCHI2_OWNPV",
    "P1_PIDK", "P1_PIDp", "P1_PIDe", "P1_PIDmu",
    "P1_ProbNNk", "P1_ProbNNpi", "P1_ProbNNp",

    "P2_P", "P2_PT", "P2_PE", "P2_PX", "P2_PY", "P2_PZ", "P2_M", "P2_ID",
    "P2_IP_OWNPV", "P2_IPCHI2_OWNPV",
    "P2_PIDK", "P2_PIDp", "P2_PIDe", "P2_PIDmu",
    "P2_ProbNNk", "P2_ProbNNpi", "P2_ProbNNp",

    # Minimal event info
    "eventNumber", "runNumber", "Polarity", "nCandidate"
]


def reduce_file(fpath: str, output_dir: str) -> str:
    """
    Reads the ROOT file at fpath, checks each TTree, drops branches not in keep_branches,
    applies PID cuts, and writes a new file with _reduced.root suffix in the specified output directory.
    Returns the path to the reduced file.
    """
    print(f"Processing {fpath}")
    
    # Extract the year and polarity from the file path
    # Example: /share/lazy/Mohamed/bu2kskpik/RD/2015_magdown/KSKmKpPip.root
    file_dir = os.path.dirname(fpath)
    dir_name = os.path.basename(file_dir)  # e.g., "2015_magdown"
    file_name = os.path.basename(fpath)    # e.g., "KSKmKpPip.root"
    
    # Create the output directory structure
    output_subdir = os.path.join(output_dir, dir_name)
    os.makedirs(output_subdir, exist_ok=True)
    
    # Define the new reduced path
    base_name, ext = os.path.splitext(file_name)
    reduced_file_name = f"{base_name}_reduced{ext}"
    reduced_fpath = os.path.join(output_subdir, reduced_file_name)
    
    try:
        with uproot.open(fpath) as infile, uproot.recreate(reduced_fpath) as outfile:
            for key in infile.keys():
                ttree_name = key.split(";")[0]  # e.g. "KSKpKpPim_DD"
                ttree = infile[ttree_name]
                # Get a set of all branch names in this TTree
                present_branches = set(ttree.keys())
                # Filter keep_branches to what's actually present
                existing = [b for b in keep_branches if b in present_branches]
                missing = [b for b in keep_branches if b not in present_branches]
                if missing:
                    print(f"  [Missing branches in {ttree_name}] {missing}")
                
                # Read only the existing branches
                arrays = ttree.arrays(expressions=existing, library="ak", how="zip")
                
                # Apply PID cuts
                mask = ak.ones_like(arrays['eventNumber'], dtype=bool)
                
                # Cut 1: K_S daughters should be pions with ProbNNpi > 0.05
                if 'KS_P0_ProbNNpi' in arrays.fields and 'KS_P1_ProbNNpi' in arrays.fields:
                    ks_pion_cut = (arrays['KS_P0_ProbNNpi'] > 0.05) & (arrays['KS_P1_ProbNNpi'] > 0.05)
                    mask = mask & ks_pion_cut
                    print(f"  Applied K_S pion cut in {ttree_name}, kept {ak.sum(ks_pion_cut)}/{len(ks_pion_cut)} events")
                
                # Cut 2: P0 and P1 should be kaons with ProbNNk > 0.05
                if 'P0_ProbNNk' in arrays.fields and 'P1_ProbNNk' in arrays.fields:
                    kaon_cut = (arrays['P0_ProbNNk'] > 0.05) & (arrays['P1_ProbNNk'] > 0.05)
                    mask = mask & kaon_cut
                    print(f"  Applied P0/P1 kaon cut in {ttree_name}, kept {ak.sum(kaon_cut)}/{len(kaon_cut)} events")
                
                # Cut 3: P2 should be a pion with ProbNNpi > 0.05
                if 'P2_ProbNNpi' in arrays.fields:
                    pion_cut = arrays['P2_ProbNNpi'] > 0.05
                    mask = mask & pion_cut
                    print(f"  Applied P2 pion cut in {ttree_name}, kept {ak.sum(pion_cut)}/{len(pion_cut)} events")
                
                # Apply all cuts
                filtered_arrays = arrays[mask]
                print(f"  After all cuts in {ttree_name}: kept {len(filtered_arrays)}/{len(arrays)} events ({len(filtered_arrays)/len(arrays)*100:.1f}%)")
                
                # Write filtered arrays to the new file under the same TTree name
                outfile[ttree_name] = filtered_arrays
        
        print(f"[DONE] {fpath} -> {reduced_fpath}")
        return reduced_fpath
    except Exception as e:
        print(f"[ERROR] {fpath} raised an exception: {e}")
        return None

def main():
    base_dir = "/share/lazy/Mohamed/bu2kskpik/RD/merged"
    output_dir = "/share/lazy/Mohamed/bu2kskpik/RD/reduced"
    years = ["2015", "2016", "2017", "2018"]
    polarities = ["magdown", "magup"]
    
    # Create the main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Gather all ROOT file paths in one list
    all_files = []
    for year in years:
        for pol in polarities:
            pattern = os.path.join(base_dir, f"{year}_{pol}", "*.root")
            found_files = glob.glob(pattern)
            all_files.extend(found_files)
    
    print(f"Found {len(all_files)} files to process")
    
    # Determine the number of processes to use
    num_processes = 6
    print(f"Using {num_processes} processes")
    
    # Prepare arguments for multiprocessing (file path and output directory)
    args = [(f, output_dir) for f in all_files]
    
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use starmap to pass multiple arguments to the worker function
        results = pool.starmap(reduce_file, args)
    
    # Count successful reductions
    successful = sum(1 for result in results if result is not None)
    print(f"Successfully processed {successful} out of {len(all_files)} files")
    
    # Calculate reduction in size
    total_original_size = sum(os.path.getsize(f) for f in all_files if os.path.exists(f))
    total_reduced_size = sum(os.path.getsize(f) for f in results if f is not None and os.path.exists(f))
    
    if total_original_size > 0:
        reduction_pct = (1 - total_reduced_size/total_original_size) * 100
        print(f"Original size: {total_original_size/1024/1024:.1f} MB")
        print(f"Reduced size: {total_reduced_size/1024/1024:.1f} MB")
        print(f"Reduction: {reduction_pct:.1f}%")
        
    print(f"\nReduced files are stored in: {output_dir}")
    
    # Print directory structure for verification
    print("\nDirectory structure created:")
    for year in years:
        for pol in polarities:
            subdir_path = os.path.join(output_dir, f"{year}_{pol}")
            if os.path.exists(subdir_path):
                files = os.listdir(subdir_path)
                if files:
                    print(f"├── {year}_{pol}")
                    for i, f in enumerate(sorted(files)):
                        prefix = "└──" if i == len(files) - 1 else "├──"
                        print(f"│   {prefix} {f}")

if __name__ == "__main__":
    main()