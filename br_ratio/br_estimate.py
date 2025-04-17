import yaml, json, math

CFG  = yaml.safe_load(open("config.yml"))
ylds = json.load(open("yields.json"))
effs = json.load(open("eff.json"))

def sum_y(cls):
    n=0; e2=0
    for v,err in ylds[cls].values():
        n  += v
        e2 += err**2
    return n, math.sqrt(e2)

def eff_avg(cls):
    num=den=0
    for key in ylds[cls]:
        nrec = ylds[cls][key][0]
        num += nrec * effs[cls][key]["eff"]
        den += nrec
    return num/den

Nsig,dNsig  = sum_y("sig")
Nnorm,dNnorm= sum_y("norm")
eps_sig     = eff_avg("sig")
eps_norm    = eff_avg("norm")

br   = (Nsig/eps_sig) / (Nnorm/eps_norm) * CFG["br_norm_pdg"]
dbr  = br * math.sqrt((dNsig/Nsig)**2 + (dNnorm/Nnorm)**2 +
                      (CFG["br_norm_pdg_unc"]/CFG["br_norm_pdg"])**2)

print(f"\nBranching ratio (trigger‑only): {br:8.3e} ± {dbr:8.2e}\n")
