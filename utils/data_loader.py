def load_mc_data(mc_path, decay_mode, years=None, tracks=None, particles=None, additional_branches=None, cuts=None):
    """
    Load MC data from ROOT files with configurable parameters.
    
    Parameters:
    -----------
    mc_path : str
        Base path to MC data directory
    decay_mode : str
        Decay mode to filter files (e.g., "Bu2L0barPKpKm")
    years : list, optional
        List of years to include (e.g., ["16", "17", "18"])
    tracks : list, optional
        List of track types (default: ["LL", "DD"])
    particles : list, optional
        List of three particles to get branches for (default: ["h1", "h2", "p"])
    additional_branches : list, optional
        Additional branches to load
    cuts : str, optional
        Selection cuts to apply when loading data
        
    Returns:
    --------
    pandas.DataFrame or awkward.Array
        Loaded MC data
    """
    import os
    import uproot
    from utils.branches import get_branches
    
    # Set defaults
    if years is None:
        years = ["16", "17", "18"]
    if tracks is None:
        tracks = ["LL", "DD"]
    if particles is None:
        particles = ["h1", "h2", "p"]
    if additional_branches is None:
        additional_branches = []
    if cuts is None:
        cuts = ""
    
    # Ensure we have exactly three particles as expected by get_branches
    if len(particles) != 3:
        raise ValueError(f"Expected exactly 3 particles, got {len(particles)}: {particles}")
    
    # Construct year prefixes
    year_prefixes = [f"MC{year}" for year in years]
    
    # Find matching files
    target_files = [
        file for file in os.listdir(mc_path)
        if any(file.startswith(prefix) for prefix in year_prefixes) and
           decay_mode in file and
           file.endswith(".root")
    ]
    
    if not target_files:
        print(f"Warning: No files found matching criteria in {mc_path}")
        return None
    
    # Construct full TTree paths
    filelist = [
        f"{mc_path}/{filename}:B2{decay_mode}_{track}/DecayTree"
        for filename in target_files
        for track in tracks
    ]
    
    print(f"MC Files being processed with trees {tracks}:", filelist)
    
    # Get all branches from the branches.py module
    branch_list = get_branches(particles) + additional_branches
    print("MC Branches being read:", branch_list)
    
    # Load data
    return uproot.concatenate(filelist, branch_list, cut=cuts)