import yaml, functools
CFG = yaml.safe_load(open("config.yml"))

# ------------------------------------------------------------
# canonical → actual mapping
# ------------------------------------------------------------
def _alias_table(sample):
    amap = {}
    for k, v in CFG["aliases"].items():
        amap[k] = v[0] if sample == "signal" else v[-1]
    for k, m in CFG["triggers"].items():
        amap[k] = m["signal" if sample == "signal" else "norm"]
    return amap


def canonical(sample, names):
    amap = _alias_table(sample)
    return [amap.get(n, n) for n in names]


# ------------------------------------------------------------
# analysis‑branch builder (tracks only demo – keep it minimal)
# ------------------------------------------------------------
def get_branches(particles):
    """Return minimal set for three‑track decay."""
    base = ["Bu_P", "Bu_PT", "Bu_CHI2NDOF"]
    kin  = ["_P", "_PT", "_PX", "_PY", "_PZ"]
    br   = [f"{p}{s}" for p in particles for s in kin]
    return base + br
