import yaml, awkward as ak
CFG = yaml.safe_load(open("config.yml"))

def trigger_mask(evt, sample):
    """
    Boolean mask: event passes *all* trigger lines.
    """
    cname = lambda key: CFG["triggers"][key][sample]
    cols = [cname(k) for k in ("L0TIS", "L0H_TOS", "H12_TOS", "Topo2_TOS")]
    # Initialize mask based on the first trigger, ensuring it's boolean
    mask = ak.values_astype(evt[cols[0]], bool)
    # Combine remaining triggers, ensuring each is cast to boolean
    for c in cols[1:]:
        # Use standard '&' and reassign instead of in-place '&='
        mask = mask & ak.values_astype(evt[c], bool)
    return mask
