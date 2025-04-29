import yaml
CFG = yaml.safe_load(open("config.yml"))

# ---------- alias handler -----------
def _alias_table(sample):
    amap = {}
    is_sig = sample == "signal"

    # Handle aliases like 'mass'
    for key, lst in CFG["aliases"].items():
        if key == "mass":
            # Special handling for mass: signal uses index 2, norm uses index -1 (or 3)
            amap[key] = lst[2] if is_sig else lst[-1]
        # Add other potential aliases here if needed

    # Handle trigger keys
    for key, lst in CFG["triggers"].items():
        # Trigger keys map to physical branch names using dict keys 'signal' or 'norm'
        amap[key] = lst["signal"] if is_sig else lst["norm"]

    return amap

def canonical(sample, keys):
    amap = _alias_table(sample)
    return [amap.get(k, k) for k in keys] # Use amap.get(k, k) to pass through unaliased keys


# ---------- branch builder ----------
def get_branches(sample, particles_sig=("h1", "h2", "p"), particles_norm=("P0", "P2", "P1")):
    # Choose particle names based on sample
    particles = particles_sig if sample == "signal" else particles_norm
    kin = ["_P", "_PT", "_PX", "_PY", "_PZ"]
    kin_branches = [f"{p}{k}" for p in particles for k in kin]
    trigger_branches = list(CFG["triggers"].keys()) # Add trigger keys
    other_branches = ["mass"] #  other required branches like mass
    return kin_branches + trigger_branches + other_branches
