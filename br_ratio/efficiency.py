import yaml, json
from loaders import load_mc_data
from selections import trigger_mask
CFG = yaml.safe_load(open("config.yml"))
eff = {}
for cls, mode in [("sig", "L0barPKpKm"), ("norm", "K0s2PipPimKmPipKp")]:
    eff[cls] = {}
    for y in CFG["years"]:
        yr_tag = y[-2:] # MC file prefix: MC16, MC17, …
        for tr in CFG["tracks"]:
            # Change sig_mc_dir to signal_mc_dir and norm_mc_dir to the proper keys from config
            mc_dir_key = "signal_mc_dir" if cls == "sig" else "norm_mc_dir"
            mc = load_mc_data(mc_path=CFG[mc_dir_key], decay_mode=mode,
                              years=[yr_tag], tracks=[tr])
            if mc is None:
                continue
            total = len(mc)
            passed = len(mc[trigger_mask(mc, "signal" if cls == "sig" else "norm")])
            eff[cls][f"{y}_{tr}"] = dict(eff=passed/total, n_pass=passed, n_tot=total)
json.dump(eff, open("eff.json", "w"), indent=2)
print("Saved → eff.json")