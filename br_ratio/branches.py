import yaml
from pathlib import Path
import pprint # For potentially cleaner printing if needed

# Load configuration only once
CONFIG_PATH = Path(__file__).parent / "config.yml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
CFG = yaml.safe_load(open(CONFIG_PATH))

def _get_cfg_section(section_name):
    """Safely get a section from the config, returning empty dict if not found."""
    return CFG.get(section_name, {})

def _resolve_branch_name(key, sample):
    """
    Resolves a configuration key (like 'l0_fdchi2', 'mass', or 'L0TIS')
    to its canonical branch name for the given sample ('signal' or 'norm')
    by looking up the key in relevant config sections ('selection_vars', 'aliases', 'triggers').

    Args:
        key (str): The generic key (e.g., 'bu_pt', 'mass').
        sample (str): 'signal' or 'norm'.

    Returns:
        str or None: The canonical branch name or None if not found/defined for the sample.
    """
    selection_vars_cfg = _get_cfg_section("selection_vars")
    aliases_cfg = _get_cfg_section("aliases")
    triggers_cfg = _get_cfg_section("triggers")

    # Check selection_vars first
    if key in selection_vars_cfg:
        if isinstance(selection_vars_cfg[key], dict):
            return selection_vars_cfg[key].get(sample)
        else:
             # Handle case where it might still be a direct string (backward compatibility?)
             # Or log a warning about unexpected format
             # print(f"Warning: Expected dict for key '{key}' in selection_vars, found string.")
             return selection_vars_cfg[key] # Assuming it's common if not a dict

    # Check aliases (like 'mass')
    if key in aliases_cfg:
        if isinstance(aliases_cfg[key], dict):
            return aliases_cfg[key].get(sample)
        else:
            # print(f"Warning: Expected dict for key '{key}' in aliases, found string.")
            # Simple alias list handling (like original config) might go here if needed
            return None # Or handle legacy list format if necessary

    # Check triggers
    if key in triggers_cfg:
        if isinstance(triggers_cfg[key], dict):
            return triggers_cfg[key].get(sample)
        else:
            # print(f"Warning: Expected dict for key '{key}' in triggers, found string.")
            return None # Or handle legacy list format if necessary

    # If the key wasn't found in specific sections, assume it *might* be
    # a directly provided canonical name (e.g., a constructed kinematic name)
    # This path should ideally be hit less often with the new config structure.
    return key


def get_nominal_branches(sample):
    """
    Returns the list of all required canonical branch names for data loading,
    based on the configuration file and the sample type. Reads sample-specific
    names from dictionaries in config.yml.

    Args:
        sample (str): 'signal' or 'norm'.

    Returns:
        list: A list of unique, canonical branch names.
    """
    is_sig = sample == "signal"
    selection_vars_cfg = _get_cfg_section("selection_vars")
    triggers_cfg = _get_cfg_section("triggers")
    aliases_cfg = _get_cfg_section("aliases")

    required_branches = set()

    # 1. Selection variables (resolve each key for the sample)
    for key in selection_vars_cfg.keys():
        branch = _resolve_branch_name(key, sample)
        if branch:
            required_branches.add(branch)

    # 2. Trigger branches (resolve each key for the sample)
    for key in triggers_cfg.keys():
        branch = _resolve_branch_name(key, sample)
        if branch:
            required_branches.add(branch)

    # 3. Mass branch (resolve alias 'mass')
    mass_branch = _resolve_branch_name('mass', sample)
    if mass_branch:
        required_branches.add(mass_branch)

    # 4. Kinematic branches (construct names)
    #    These are constructed and assumed canonical unless explicitly mapped above.
    particles_sig = ["h1", "h2", "p", "L0"]
    particles_norm = ["P0", "P2", "P1", "KS"]
    particles = particles_sig if is_sig else particles_norm
    particles.append("Bu" if is_sig else "B") # Add B kinematics

    kin_suffixes = ["_PT", "_PX", "_PY", "_PZ"]
    pid_suffixes = ["_ProbNNk", "_ProbNNp", "_ProbNNpi"]

    # Get set of all values already added from selection_vars, triggers, mass
    # to avoid adding duplicates if kinematics/PIDs are also listed there.
    explicitly_defined_branches = set()
    for section in [selection_vars_cfg, triggers_cfg, aliases_cfg]:
        for key, value in section.items():
             if isinstance(value, dict):
                 branch = value.get(sample)
                 if branch:
                     explicitly_defined_branches.add(branch)

    for p in particles:
        # Add Kinematics
        for k in kin_suffixes:
            branch_name = f"{p}{k}"
            # Only add if not already covered by selection_vars/triggers/mass
            if branch_name not in explicitly_defined_branches:
                required_branches.add(branch_name)

        # Add PIDs (carefully, check against config keys)
        for pid_suffix in pid_suffixes:
            branch_name = f"{p}{pid_suffix}"
            # Check if this specific constructed PID branch name is already added
            if branch_name in required_branches or branch_name in explicitly_defined_branches:
                continue

            # Determine if this PID type is relevant for this particle
            # and if a corresponding generic key exists in selection_vars
            is_kaon = 'k' in pid_suffix and p in ["h1", "h2", "P0", "P2"]
            is_proton = 'p' in pid_suffix and p in ["p", "Lambda"]
            is_pion = 'pi' in pid_suffix and p in ["P1", "Ks", "Lambda"] # Lambda has pion daughter

            # Check if a config key likely corresponds to this constructed PID
            config_key_exists = False
            if is_kaon and p in ["h1", "P0"] and "h1_pid" in selection_vars_cfg: config_key_exists = True
            if is_kaon and p in ["h2", "P2"] and "h2_pid" in selection_vars_cfg: config_key_exists = True
            if is_proton and p == "p" and "bachelor_pid" in selection_vars_cfg: config_key_exists = True # Signal bachelor proton
            if is_proton and p == "Lambda" and "l0_daughter1_pid" in selection_vars_cfg: config_key_exists = True # Lambda proton
            if is_pion and p == "P1" and "bachelor_pid" in selection_vars_cfg: config_key_exists = True # Norm bachelor pion
            if is_pion and p == "Lambda" and "l0_daughter2_pid" in selection_vars_cfg: config_key_exists = True # Lambda pion
            if is_pion and p == "Ks" and "l0_daughter1_pid" in selection_vars_cfg: config_key_exists = True # Ks pion+ (assume daughter1)
            if is_pion and p == "Ks" and "l0_daughter2_pid" in selection_vars_cfg: config_key_exists = True # Ks pion- (assume daughter2)

            # Only add the constructed PID branch if it seems relevant AND
            # it wasn't already handled via a specific key in selection_vars
            if (is_kaon or is_proton or is_pion) and not config_key_exists:
                 # print(f"Adding constructed PID: {branch_name}") # Debug print
                 required_branches.add(branch_name)

    # Final cleanup of None values
    required_branches.discard(None)

    return sorted(list(required_branches))

# Example usage:
if __name__ == "__main__":
    print("--- Signal Branches ---")
    pprint.pprint(get_nominal_branches("signal"))
    print("\n--- Normalization Branches ---")
    pprint.pprint(get_nominal_branches("norm"))
