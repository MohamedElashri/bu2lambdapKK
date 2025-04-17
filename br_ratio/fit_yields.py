import yaml, ROOT, json
from loaders import load_mc_data, load_data
from selections import trigger_mask

CFG = yaml.safe_load(open("config.yml"))

def fit(sample_class, year, track):
    is_sig  = sample_class == "sig"
    mode    = "L0barPKpKm" if is_sig else "KSKmKpPip"
    paths   = (CFG["signal_data_dir"], CFG["signal_mc_dir"]) if is_sig \
              else (CFG["norm_data_dir"],   CFG["norm_mc_dir"])

    # choose data loader
    data = load_data(data_path=paths[0], decay_mode=mode,
                     years=[year], tracks=[track])
    if data is None: return None
    data = data[trigger_mask(data, "signal" if is_sig else "norm")]

    mass_col = CFG["aliases"]["mass"][0] if is_sig else CFG["aliases"]["mass"][-1]
    m = ROOT.RooRealVar("m", "mass", 5100, 5800, "MeV")
    rds = ROOT.RooDataSet("rds", "", ROOT.RooArgSet(m),
                          ROOT.RooFit.Import(data[mass_col].to_numpy()))

    mean  = ROOT.RooRealVar("mean", 5280, 5200, 5350)
    sigma = ROOT.RooRealVar("sigma", 20, 5, 60)
    gaus  = ROOT.RooGaussian("gaus", "", m, mean, sigma)
    c     = ROOT.RooRealVar("c", -0.002, -1, 0)
    expo  = ROOT.RooExponential("expo", "", m, c)
    fsig  = ROOT.RooRealVar("fsig", 0.5, 0, 1)
    model = ROOT.RooAddPdf("model", "", gaus, expo, fsig)

    model.fitTo(rds, ROOT.RooFit.PrintLevel(-1))
    nsig = fsig.getVal() * rds.numEntries()
    err  = fsig.getError() * rds.numEntries()
    return nsig, err


results = {}
for cls in ("sig", "norm"):
    results[cls] = {}
    for y in CFG["years"]:
        for tr in CFG["tracks"]:
            out = fit(cls, y, tr)
            if out: results[cls][f"{y}_{tr}"] = out

json.dump(results, open("yields.json", "w"), indent=2)
print("Saved → yields.json")
