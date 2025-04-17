import os, uproot, yaml
from branches import canonical, get_branches

CFG = yaml.safe_load(open("config.yml"))


def _tree_path(fpath, mode, track):
    """
    Discover the correct tree path; supports both layouts.
    """
    attempts = [
        f"{fpath}:B2{mode}_{track}/DecayTree",
        f"{fpath}:{mode}_{track}/DecayTree",
        f"{fpath}:B2{mode}_{track}",
        f"{fpath}:{mode}_{track}",
    ]
    for a in attempts:
        try:
            uproot.open(a)
            return a
        except Exception:
            continue
    raise RuntimeError(f"{fpath}: no tree for {mode}_{track}")


def _discover_files(base, substrings, year_tokens):
    """Recursively yield ROOT files with given substrings & year tags."""
    for root, _, files in os.walk(base):
        for f in files:
            full = os.path.join(root, f)
            if (
                f.endswith(".root")
                and all(s in f for s in substrings)
                and any(t in full for t in year_tokens)
            ):
                yield full


def _loader(base, mode, years, tracks, sample, extra, cuts):
    year_tokens = {y for y in years} | {y[-2:] for y in years}
    files = list(_discover_files(base, [mode], year_tokens))
    if not files:
        print(f"[loader] no files for {mode} in {base}")
        return None
    tpaths = []
    for f in files:
        for tr in tracks:
            try:
                tpaths.append(_tree_path(f, mode, tr))
            except RuntimeError:
                continue
    if not tpaths:
        print(f"[loader] found files but no trees for {mode}")
        return None
    branches = canonical(sample, get_branches(["h1", "h2", "p"]) + list(extra))
    return uproot.concatenate(tpaths, branches, cut=cuts, how="awkward")


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------
def load_mc_data(**kw):
    return _loader(sample="signal" if "L0bar" in kw["decay_mode"] else "norm",
                   base=kw["mc_path"],
                   mode=kw["decay_mode"],
                   years=kw.get("years", ["16", "17", "18"]),
                   tracks=kw.get("tracks", ["LL", "DD"]),
                   extra=kw.get("additional_branches", ()),
                   cuts=kw.get("cuts", ""))


def load_data(**kw):
    return _loader(sample="signal" if "L0bar" in kw["decay_mode"] else "norm",
                   base=kw["data_path"],
                   mode=kw["decay_mode"],
                   years=kw.get("years", ["2016", "2017", "2018"]),
                   tracks=kw.get("tracks", ["LL", "DD"]),
                   extra=kw.get("additional_branches", ()),
                   cuts=kw.get("cuts", ""))
