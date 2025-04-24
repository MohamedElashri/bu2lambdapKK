import os, uproot, yaml, awkward as ak
from tqdm.auto import tqdm
from branches import canonical, get_branches
CFG = yaml.safe_load(open("config.yml"))

# ------------------------------------------------------------
# tree‑path resolver (updated with more flexible matching)
# ------------------------------------------------------------
def _tree_path(fpath, mode, track):
    print(f"DEBUG: Looking for tree {mode}_{track} in {fpath}")
    
    # Try the traditional direct TTree paths
    for tpl in (f"B2{mode}_{track}/DecayTree",
               f"{mode}_{track}/DecayTree",
               f"B2{mode}_{track}",
               f"{mode}_{track}"):
        path = f"{fpath}:{tpl}"
        try:
            uproot.open(path)
            print(f"DEBUG: Found direct path {path}")
            return path
        except Exception:
            continue
    
    # Try the TDirectoryFile structure with more flexible matching
    try:
        rf = uproot.open(fpath)
        print(f"DEBUG: Available keys in {fpath}: {rf.keys()}")
        
        # First try exact match
        for key in rf.keys():
            key_name = key.split(";")[0]  # Remove version number
            print(f"DEBUG: Checking key {key_name}")
            
            # Look for exact or partial matches
            if f"{mode}_{track}" in key_name:
                dir_path = f"{fpath}:{key_name}"
                try:
                    directory = rf[key_name]
                    if isinstance(directory, uproot.ReadOnlyDirectory):
                        # Check for DecayTree
                        if any("DecayTree" in k for k in directory.keys()):
                            decay_tree_key = next(k.split(";")[0] for k in directory.keys() if "DecayTree" in k)
                            return f"{dir_path}/{decay_tree_key}"
                        # Return directory if no DecayTree
                        return dir_path
                except Exception as e:
                    print(f"DEBUG: Error examining directory {key_name}: {e}")
                    continue
        
        # Try more flexible matching strategies
        for key in rf.keys():
            key_name = key.split(";")[0]
            # For KS/Ks modes, match directories containing KS and the track type
            if mode.startswith("KS") or mode.startswith("K0s"):
                if ("KS" in key_name or "Ks" in key_name) and f"_{track}" in key_name:
                    dir_path = f"{fpath}:{key_name}"
                    try:
                        directory = rf[key_name]
                        if isinstance(directory, uproot.ReadOnlyDirectory):
                            if any("DecayTree" in k for k in directory.keys()):
                                decay_tree_key = next(k.split(";")[0] for k in directory.keys() if "DecayTree" in k)
                                return f"{dir_path}/{decay_tree_key}"
                            return dir_path
                    except Exception as e:
                        print(f"DEBUG: Error examining directory {key_name}: {e}")
                        continue
            
            # Special handling for L0bar modes
            if "L0bar" in mode:
                if "L0bar" in key_name and f"_{track}" in key_name:
                    dir_path = f"{fpath}:{key_name}"
                    try:
                        directory = rf[key_name]
                        if isinstance(directory, uproot.ReadOnlyDirectory):
                            if any("DecayTree" in k for k in directory.keys()):
                                decay_tree_key = next(k.split(";")[0] for k in directory.keys() if "DecayTree" in k)
                                return f"{dir_path}/{decay_tree_key}"
                            return dir_path
                    except Exception:
                        continue
        
        # If still not found, try any directory that matches track type
        for key in rf.keys():
            key_name = key.split(";")[0]
            if f"_{track}" in key_name:
                print(f"DEBUG: Found potential match {key_name} based on track type")
                dir_path = f"{fpath}:{key_name}"
                try:
                    directory = rf[key_name]
                    if isinstance(directory, uproot.ReadOnlyDirectory):
                        if any("DecayTree" in k for k in directory.keys()):
                            decay_tree_key = next(k.split(";")[0] for k in directory.keys() if "DecayTree" in k)
                            return f"{dir_path}/{decay_tree_key}"
                        return dir_path
                except Exception:
                    continue
    except Exception as e:
        print(f"DEBUG: Error opening file {fpath}: {e}")
    
    print(f"DEBUG: No tree found for {mode}_{track} in {fpath}")
    raise RuntimeError(f"{fpath}: no tree for {mode}_{track}")

# ------------------------------------------------------------
# discover ROOT files (enhanced with better pattern matching)
# ------------------------------------------------------------
def _discover(base, pattern, years):
    print(f"DEBUG: Discovering files in {base} matching '{pattern}' for years {years}")
    tokens = {y for y in years} | {y[-2:] for y in years}
    found_files = []
    
    # Handle patterns with pipe characters (OR)
    patterns = pattern.split("|")
    
    for root, _, files in os.walk(base):
        for f in files:
            full = os.path.join(root, f)
            
            # First check if file is a ROOT file
            if not f.endswith(".root"):
                continue
                
            # Check for year match
            if not any(t in full for t in tokens):
                continue
                
            # Check pattern match
            if pattern == ".root" or pattern == "":
                # Special case: match any ROOT file
                found_files.append(full)
            elif any(p in f for p in patterns):
                # Regular pattern matching
                found_files.append(full)
    
    print(f"DEBUG: Found {len(found_files)} files: {found_files}")
    return found_files

# ------------------------------------------------------------
# core loader
# ------------------------------------------------------------
def _loader(base, pattern, mode, years, tracks, sample, extra, cuts):
    print(f"DEBUG: Loading {sample} {mode} for years {years}, tracks {tracks}")
    files = list(_discover(base, pattern, years))
    if not files:
        print(f"[loader] no files ({pattern}) in {base}")
        return None
    
    trees = []
    for f in files:
        for tr in tracks:
            try:
                trees.append(_tree_path(f, mode, tr))
            except RuntimeError as e:
                print(f"DEBUG: {e}")
                continue
    
    if not trees:
        print(f"[loader] files found but no trees for {mode}")
        return None
    
    branches = canonical(sample, get_branches(sample) + list(extra))
    print(f"DEBUG: Loading branches: {branches}")
    arrays = []
    
    for tp in tqdm(trees, desc=f"{sample}-{mode}", unit="tree"):
        try:
            print(f"DEBUG: Reading tree {tp}")
            array = uproot.concatenate(tp, branches, cut=(cuts or None), how="zip")
            print(f"DEBUG: Loaded {len(array)} events with fields: {array.fields}")
            arrays.append(array)
        except Exception as e:
            print(f"Error processing {tp}: {e}")
    
    if not arrays:
        print(f"DEBUG: No arrays loaded for {mode}")
        return None
        
    result = ak.concatenate(arrays)
    print(f"DEBUG: Loaded total {len(result)} events with fields: {result.fields}")
    return result

# ------------------------------------------------------------
# public wrappers
# ------------------------------------------------------------
def load_mc_data(mc_path, decay_mode, **kw):
    pattern = CFG["patterns"]["signal_mc"] if "L0bar" in decay_mode else CFG["patterns"]["norm_mc"]
    return _loader(base=mc_path, pattern=pattern, mode=decay_mode,
                  years=kw.get("years", ["16", "17", "18"]),
                  tracks=kw.get("tracks", ["LL", "DD"]),
                  sample="signal" if "L0bar" in decay_mode else "norm",
                  extra=kw.get("additional_branches", ()),
                  cuts=kw.get("cuts", ""))

def load_data(data_path, decay_mode, **kw):
    # Determine if this is signal or normalization mode
    is_signal = "L0bar" in decay_mode
    pattern = CFG["patterns"]["signal_data"] if is_signal else CFG["patterns"]["norm_data"]
    
    # For normalization mode, check if we have a specific directory naming in config
    actual_mode = decay_mode
    if not is_signal and "norm_mode_data" in CFG:
        actual_mode = CFG["norm_mode_data"]
        print(f"DEBUG: Using configured norm_mode_data: {actual_mode} instead of {decay_mode}")
    
    # Get the appropriate years format based on the sample type
    years_format = kw.get("years", ["2016", "2017", "2018"])
    
    return _loader(base=data_path, pattern=pattern, mode=actual_mode,
                  years=years_format,
                  tracks=kw.get("tracks", ["LL", "DD"]),
                  sample="signal" if is_signal else "norm",
                  extra=kw.get("additional_branches", ()),
                  cuts=kw.get("cuts", ""))