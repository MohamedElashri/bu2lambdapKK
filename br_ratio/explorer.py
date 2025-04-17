#!/usr/bin/env python3
"""
Branch explorer that works with *any* Run‑2 tuple structure.
Compares branch keys for:

    signal_mc   : MC  Λ̅ p K⁺ K⁻
    signal_data : DATA Λ̅ p K⁺ K⁻
    norm_mc     : MC  K_S⁰ π⁺ K⁺ K⁻
    norm_data   : DATA K_S⁰ π⁺ K⁺ K⁻

Outputs
    • console summary
    • branch_overview.csv   (columns = sample classes, rows = branch names)
"""

import os, yaml, uproot, pandas as pd
from collections import defaultdict, OrderedDict

CFG = yaml.safe_load(open("config.yml"))

# ----------------------------------------------------------------------
# sample‑class → (base directory, filename substring to locate a file)
# ----------------------------------------------------------------------
MAP = OrderedDict([
    ("signal_mc",   (CFG["signal_mc_dir"],   "Bu2L0barPKpKm")),
    ("signal_data", (CFG["signal_data_dir"], "dataBu2L0barPHH")),
    ("norm_mc",     (CFG["norm_mc_dir"],     "B2K0s2PipPimKmPipKp")),
    ("norm_data",   (CFG["norm_data_dir"],   "KSKmKpPip")),
])

# accept both “2018” and “18”
YEAR_TOKENS = {y for y in CFG["years"]} | {y[-2:] for y in CFG["years"]}


# ----------------------------------------------------------------------
def first_file(base_dir: str, name_pattern: str) -> str:
    """Return first *.root file matching both the pattern and a year token."""
    for root, _, files in os.walk(base_dir):
        for f in sorted(files):
            full = os.path.join(root, f)
            if (
                f.endswith(".root")                 # file is a ROOT file
                and name_pattern in f               # contains decay‑mode tag
                and any(tok in full for tok in YEAR_TOKENS)   # year token anywhere in path
            ):
                return full

def find_tree_path(fpath: str, tracks) -> str:
    """
    Open *fpath* with uproot and return 'fpath:Dir/DecayTree'
    where Dir ends with _LL or _DD (tracks[0] preferred).
    """
    rf = uproot.open(fpath)
    keys = rf.keys()                      # may be TTrees or TDirectories

    for tr in tracks:                     # honour order in config.yml
        # 1) case: top‑level TTree name ends with _LL or _DD
        for k in keys:
            if k.rstrip(";1").endswith(f"_{tr}") and isinstance(rf[k], uproot.TTree):
                return f"{fpath}:{k.rstrip(';1')}"

        # 2) case: directory → DecayTree
        for k in keys:
            if k.rstrip(";1").endswith(f"_{tr}") and isinstance(rf[k], uproot.ReadOnlyDirectory):
                candidate = f"{fpath}:{k.rstrip(';1')}/DecayTree"
                try:
                    uproot.open(candidate)
                    return candidate
                except KeyError:
                    continue
    # 3) case: no *_LL or *_DD tree found
    raise RuntimeError(f"{fpath}: no *_LL or *_DD tree found")

def branch_set(tree_path: str) -> set:
    """Return the set of branch names for the given tree."""
    return set(uproot.open(tree_path).keys())


# ----------------------------------------------------------------------
# 1) locate one tree per class
# ----------------------------------------------------------------------
tree_paths = OrderedDict()
for cls, (base, patt) in MAP.items():
    fpath = first_file(base, patt)
    tree_paths[cls] = find_tree_path(fpath, CFG["tracks"])

print("\n>>> Representative trees\n")
for cls, path in tree_paths.items():
    print(f"{cls:12s} -> {path}")

# ----------------------------------------------------------------------
# 2) build branch sets & compare
# ----------------------------------------------------------------------
branch_sets = {cls: branch_set(tpath) for cls, tpath in tree_paths.items()}
common      = set.intersection(*branch_sets.values())
exclusive   = {cls: sorted(bset - common) for cls, bset in branch_sets.items()}

print(f"\n>>> {len(common)} branches common to ALL four samples\n")
print(", ".join(sorted(common)))

print("\n>>> Class‑exclusive branches\n")
for cls, lst in exclusive.items():
    print(f"{cls:12s}: {len(lst):3d} exclusive")
    if lst:
        print("   ", ", ".join(lst[:12]), ("..." if len(lst) > 12 else ""))
    print()

# ----------------------------------------------------------------------
# 3) CSV export  ––  presence matrix (no length mismatch)
# ----------------------------------------------------------------------
import numpy as np

all_branches = sorted(set().union(*branch_sets.values()))
df = pd.DataFrame(index=all_branches)

for cls, bset in branch_sets.items():
    df[cls] = np.where(df.index.isin(bset), "*", "")

df.to_csv("branch_overview.csv")
print("CSV written → branch_overview.csv  (* = branch present)")