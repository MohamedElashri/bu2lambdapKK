'''
 This file aims to reduce the data sample size of B2L0barPHH decays.
 Then use preselections by root
'''
import uproot
import numpy as np
import os
import traceback
from tqdm import tqdm
from utils.branches import branches
from utils.constants import trigcut

dataPath = "/eos/lhcb/wg/BnoC/Bu2LambdaPPP/RD/restripped.data/reduced/"
trees = ['B2L0PbarKpKp_DD/DecayTree', 'B2L0barPKpKm_DD/DecayTree',
'B2L0PbarKpKp_LL/DecayTree', 'B2L0barPKpKm_LL/DecayTree']

def safe_reshape(arrays, branch_name):
    """Safely reshape a branch if it's 2D or higher"""
    if branch_name not in arrays:
        return
        
    try:
        shape = arrays[branch_name].shape
        if len(shape) > 1 and shape[1] > 0:
            arrays[branch_name] = arrays[branch_name][:, 0]
    except (AttributeError, IndexError, ValueError) as e:
        print(f"Warning: Could not reshape {branch_name}: {e}")

def process_tree(file_path, tree_name, output_file):
    """Process a single tree from a file"""
    print(f"\nProcessing tree: {tree_name} from file: {os.path.basename(file_path)}")
    
    treename = f"{file_path}:{tree_name}"
    particles = ["h1", "h2", "p"]
    branch_list = branches(particles)
    
    try:
        # First check if the tree exists
        with uproot.open(file_path) as f:
            if tree_name not in f:
                print(f"Tree {tree_name} not found in file {file_path}")
                return False
                
        # Try to read the tree directly without any cuts first
        print("Reading basic tree info...")
        with uproot.open(treename) as tree:
            n_entries = tree.num_entries
            print(f"Tree contains {n_entries} entries")
            
            if n_entries == 0:
                print("Tree is empty, skipping")
                return False
                
            # Get a list of all available branches for debugging
            all_tree_branches = tree.keys()
            print(f"Tree contains {len(all_tree_branches)} branches")
            
            # Check for missing branches in our branch list
            missing_branches = [b for b in branch_list if b not in all_tree_branches]
            if missing_branches:
                print(f"Warning: {len(missing_branches)} requested branches are missing from the tree")
                print(f"First few missing: {missing_branches[:5]}")
                
            # Filter our branch list to only include branches present in the tree
            available_branches = [b for b in branch_list if b in all_tree_branches]
            print(f"Will process {len(available_branches)} available branches")
            
            # Create our selection conditions
            cuts = '(Lp_ProbNNp>0.05) & (Bu_FDCHI2_OWNPV>175) & (Bu_IPCHI2_OWNPV<10) & (Bu_PT>3000) & ((L0_ENDVERTEX_Z-Bu_ENDVERTEX_Z)>2.5)'
            cuts += '& (h1_ProbNNk*h2_ProbNNk>0.05) & (p_ProbNNp>0.05)'
            
            # Process in smaller chunks to avoid memory issues
            chunk_size = 10000
            total_chunks = (n_entries + chunk_size - 1) // chunk_size
            
            print(f"Processing in {total_chunks} chunks of {chunk_size} entries...")
            
            # Get arrays without any cuts, then filter manually
            all_data = {}
            filtered_indices = []
            
            # Process data in chunks
            for i in range(total_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, n_entries)
                print(f"Processing chunk {i+1}/{total_chunks} (entries {start}-{end})")
                
                try:
                    # Read important filter branches first to create a selection mask
                    filter_branches = ["Lp_ProbNNp", "Bu_FDCHI2_OWNPV", "Bu_IPCHI2_OWNPV", 
                                       "Bu_PT", "L0_ENDVERTEX_Z", "Bu_ENDVERTEX_Z", 
                                       "h1_ProbNNk", "h2_ProbNNk", "p_ProbNNp",
                                       "Bu_L0Global_TIS", "Bu_L0HadronDecision_TOS",
                                       "Bu_Hlt1TrackMVADecision_TOS", "Bu_Hlt1TwoTrackMVADecision_TOS",
                                       "Bu_Hlt2Topo2BodyDecision_TOS", "Bu_Hlt2Topo3BodyDecision_TOS",
                                       "Bu_Hlt2Topo4BodyDecision_TOS"]
                    
                    # Read only filter branches for this chunk
                    chunk_data = tree.arrays(filter_branches, 
                                             entry_start=start, 
                                             entry_stop=end,
                                             library="np")
                    
                    # Create trigger selection mask
                    trigger_mask = ((chunk_data["Bu_L0Global_TIS"] | chunk_data["Bu_L0HadronDecision_TOS"]) &
                                   (chunk_data["Bu_Hlt1TrackMVADecision_TOS"] | chunk_data["Bu_Hlt1TwoTrackMVADecision_TOS"]) &
                                   (chunk_data["Bu_Hlt2Topo2BodyDecision_TOS"] | chunk_data["Bu_Hlt2Topo3BodyDecision_TOS"] | 
                                   chunk_data["Bu_Hlt2Topo4BodyDecision_TOS"]))
                    
                    # Create physics selection mask
                    physics_mask = ((chunk_data["Lp_ProbNNp"] > 0.05) & 
                                   (chunk_data["Bu_FDCHI2_OWNPV"] > 175) & 
                                   (chunk_data["Bu_IPCHI2_OWNPV"] < 10) & 
                                   (chunk_data["Bu_PT"] > 3000) & 
                                   ((chunk_data["L0_ENDVERTEX_Z"] - chunk_data["Bu_ENDVERTEX_Z"]) > 2.5) &
                                   (chunk_data["h1_ProbNNk"] * chunk_data["h2_ProbNNk"] > 0.05) & 
                                   (chunk_data["p_ProbNNp"] > 0.05))
                    
                    # Combine masks
                    combined_mask = trigger_mask & physics_mask
                    
                    # Get indices of events that pass the selection
                    chunk_indices = np.where(combined_mask)[0] + start
                    filtered_indices.extend(chunk_indices.tolist())
                    
                    print(f"Found {len(chunk_indices)} events passing cuts in this chunk")
                    
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    traceback_str = traceback.format_exc()
                    print(f"Traceback:\n{traceback_str}")
                    continue
            
            print(f"Total events passing cuts: {len(filtered_indices)}")
            
            if not filtered_indices:
                print("No events passed the cuts, skipping tree")
                return False
            
            # Now read only the selected events for all branches
            print("Reading selected events for all branches...")
            branch_data = {}
            
            # Process filtered events in smaller chunks to avoid memory issues
            filtered_indices.sort()  # Sort indices for more efficient reading
            filtered_chunks = [filtered_indices[i:i+1000] for i in range(0, len(filtered_indices), 1000)]
            
            for i, indices_chunk in enumerate(filtered_chunks):
                print(f"Reading branches for chunk {i+1}/{len(filtered_chunks)} ({len(indices_chunk)} events)")
                
                try:
                    # Read data for these indices
                    chunk_data = tree.arrays(available_branches, entry_start=indices_chunk[0], 
                                             entry_stop=indices_chunk[-1]+1, library="np")
                    
                    # Filter to just the events we want and add to our data dictionary
                    local_indices = [idx - indices_chunk[0] for idx in indices_chunk]
                    
                    for branch in available_branches:
                        if branch not in branch_data:
                            branch_data[branch] = []
                        
                        branch_data[branch].append(chunk_data[branch][local_indices])
                    
                except Exception as e:
                    print(f"Error reading branches for chunk: {e}")
                    traceback_str = traceback.format_exc()
                    print(f"Traceback:\n{traceback_str}")
                    continue
            
            # Combine data from all chunks
            for branch in available_branches:
                if branch in branch_data and branch_data[branch]:
                    try:
                        branch_data[branch] = np.concatenate(branch_data[branch])
                    except Exception as e:
                        print(f"Error concatenating branch {branch}: {e}")
                        branch_data[branch] = np.array([])
            
            # Handle reshaping for special variables
            reshape_vars = [
                "Bu_DTFL0_M", "Bu_DTFL0_MERR", "Bu_DTF_decayLength", "Bu_DTF_decayLengthErr",
                "Bu_DTF_Lambda0_decayLength", "Bu_DTF_Lambda0_decayLengthErr", "Bu_DTF_ctau",
                "Bu_DTF_ctauErr", "Bu_DTF_status", "Bu_DTF_chi2", "Bu_DTF_nDOF", 
                "Bu_DTFL0_chi2", "Bu_DTFL0_nDOF", "Bu_DTFL0_ctau", "Bu_DTFL0_ctauErr"
            ]
            
            for var in reshape_vars:
                safe_reshape(branch_data, var)
            
            # Store the processed data
            print(f"Storing processed data for {len(filtered_indices)} events...")
            output_file[tree_name] = branch_data
            
            return True
            
    except Exception as e:
        print(f"Error processing tree {tree_name}: {e}")
        traceback_str = traceback.format_exc()
        print(f"Traceback:\n{traceback_str}")
        return False

def preselection(file):
    file_name = os.path.basename(file)
    output_path = f"/eos/lhcb/user/m/melashri/data/bu2LpKK/RD/reduced/{file_name[:-5]}_reduced.root"
    
    print(f"\nProcessing file: {file}")
    print(f"Output will be saved to: {output_path}")
    
    with uproot.recreate(output_path) as outFile:
        success_count = 0
        for tree in trees:
            if process_tree(file, tree, outFile):
                success_count += 1
        
        print(f"Completed processing file: {file}")
        print(f"Successfully processed {success_count}/{len(trees)} trees")
    
    return file_name

def main():
    from multiprocessing import Pool
    
    # Get only the specific ROOT files in the main directory
    filelist = []
    for file in os.listdir(dataPath):
        if file.startswith("dataBu2L0barPHH_") and file.endswith("_reduced.root"):
            filelist.append(os.path.join(dataPath, file))
    
    print("FileList to Reduce: ", filelist)
    print(f"Total files found: {len(filelist)}")
    
    if not filelist:
        print("No files found matching the criteria.")
        return
    
    # Process files sequentially for easier debugging
    # Comment this and uncomment the Pool section below for parallel processing
    for file in tqdm(filelist, desc="Processing files"):
        preselection(file)
    
    # Uncomment for parallel processing
    # pool = Pool(processes=4)  # Using fewer processes for more stability
    # r = list(tqdm(pool.imap(preselection, filelist), total=len(filelist), desc="Processing files"))
    # pool.close()
    # pool.join()

if __name__ == "__main__":
    main()