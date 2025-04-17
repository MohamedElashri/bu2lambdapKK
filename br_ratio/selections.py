import yaml, awkward as ak
CFG = yaml.safe_load(open("config.yml"))

def trigger_mask(evt, sample):
    """
    Return boolean array – true if event passes *all* trigger lines.
    """
    cols = [CFG["triggers"][k][sample] for k in
            ("L0TIS", "L0H_TOS", "H12_TOS", "Topo2_TOS")]
    mask = ak.ones_like(evt[cols[0]], dtype=bool)
    for c in cols:
        mask &= evt[c]
    return mask
