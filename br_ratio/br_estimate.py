import yaml, json, math, numpy as np
import sys

CFG = yaml.safe_load(open("config.yml"))

# Check if files exist before loading
try:
    ylds = json.load(open("yields.json"))
    effs = json.load(open("eff.json"))
except FileNotFoundError as e:
    print(f"Error: Could not find required JSON file: {e}")
    sys.exit(1)

def sum_y(cls):
    n, e2 = 0, 0
    if cls not in ylds or not ylds[cls]:
        print(f"Warning: No yield data found for '{cls}'")
        return 0, 0
    
    for v, err in ylds[cls].values():
        n += v
        e2 += err**2
    return n, math.sqrt(e2)

def eff_avg(cls):
    num = den = 0
    if cls not in ylds or not ylds[cls]:
        print(f"Warning: No yield data found for '{cls}'")
        return 0
    
    if cls not in effs or not effs[cls]:
        print(f"Warning: No efficiency data found for '{cls}'")
        return 0
    
    for key in ylds[cls]:
        if key not in effs[cls]:
            print(f"Warning: Efficiency data for '{key}' not found, skipping")
            continue
            
        nrec = ylds[cls][key][0]
        num += nrec * effs[cls][key]["eff"]
        den += nrec
    
    if den == 0:
        print(f"Error: Total yield for '{cls}' is zero, cannot calculate efficiency average")
        return 0
    
    return num/den

# Get yields and errors
Nsig, dNsig = sum_y("sig")
Nnorm, dNnorm = sum_y("norm")

# Check if we have valid data
if Nsig == 0 or Nnorm == 0:
    print("\n----------------------------------------------------")
    print(" Error: Missing data for signal or normalization channel")
    if Nsig == 0:
        print(" - Signal yield is zero")
    if Nnorm == 0:
        print(" - Normalization yield is zero")
    print("----------------------------------------------------\n")
    sys.exit(1)

# Calculate efficiencies
eps_sig = eff_avg("sig")
eps_norm = eff_avg("norm")

# Check if efficiencies are valid
if eps_sig == 0 or eps_norm == 0:
    print("\n----------------------------------------------------")
    print(" Error: Missing efficiency data")
    if eps_sig == 0:
        print(" - Signal efficiency is zero")
    if eps_norm == 0:
        print(" - Normalization efficiency is zero")
    print("----------------------------------------------------\n")
    sys.exit(1)

# Calculate branching ratio
br = (Nsig/eps_sig) / (Nnorm/eps_norm) * CFG["br_norm_pdg"]
dbr = br * math.sqrt((dNsig/Nsig)**2 + (dNnorm/Nnorm)**2 +
                    (CFG["br_norm_pdg_unc"]/CFG["br_norm_pdg"])**2)

print("\n----------------------------------------------------")
print(f" Trigger‑only branching ratio: {br:10.3e} ± {dbr: .2e}")
print("----------------------------------------------------\n")