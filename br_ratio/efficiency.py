import yaml, json
from loaders import load_mc_data
from selections import trigger_mask

CFG = yaml.safe_load(open("config.yml"))

eff = {}
for cls, mode in [("sig", "L0barPKpKm"), ("norm", "K0s2PipPimKmPipKp")]:
    eff[cls] = {}
    for y in CFG["years"]:
        for tr in CFG["tracks"]:
            mc = load_mc_data(mc_path=CFG[f"{cls}_mc_dir"], decay_mode=mode,
                              years=[y[-2:]], tracks=[tr])
            if mc is None: continue
            total  = len(mc)
            passed = len(mc[trigger_mask(mc, "signal" if cls=="sig" else "norm")])
            eff[cls][f"{y}_{tr}"] = dict(eff=passed/total, n_pass=passed, n_tot=total)

json.dump(eff, open("eff.json", "w"), indent=2)
print("Saved → eff.json")
