import yaml, ROOT, json
from loaders import load_data
from selections import trigger_mask
from branches import canonical

CFG = yaml.safe_load(open("config.yml"))


def fit(sample, year, track):
    """sample = 'sig' | 'norm'"""
    is_sig = sample == "sig"
    mode   = "L0barPKpKm" if is_sig else "KSKmKpPip"
    base   = CFG["signal_data_dir"] if is_sig else CFG["norm_data_dir"]

    # Load data - it will have physical branch names like Bu_MM or B_MM
    data = load_data(data_path=base, decay_mode=mode,
                     years=[year], tracks=[track])
    if data is None:
        return None

    data = data[trigger_mask(data, "signal" if is_sig else "norm")]

    # Determine the correct physical mass column name using canonical
    mass_col = canonical("signal" if is_sig else "norm", ["mass"])[0]
    m = ROOT.RooRealVar("m", "mass", 5100, 5800, "MeV")

    # Create an empty dataset with the mass variable
    rds = ROOT.RooDataSet("rds", "", ROOT.RooArgSet(m))
    mass_data_np = data[mass_col].to_numpy() # Get NumPy array once

    # Loop through the NumPy array and fill the RooDataSet
    print(f"\tFilling RooDataSet for {sample}/{year}/{track} with {len(mass_data_np)} entries...")
    for mass_val in mass_data_np:
        m.setVal(mass_val)
        rds.add(ROOT.RooArgSet(m))
    print("\t...filling complete.")

    mean  = ROOT.RooRealVar("mean", "mean", 5280, 5200, 5350)
    sigma = ROOT.RooRealVar("sigma", "sigma", 20, 5, 60)
    c     = ROOT.RooRealVar("c", "c", -0.002, -1, 0)
    expo  = ROOT.RooExponential("expo", "", m, c)
    fsig  = ROOT.RooRealVar("fsig", "fsig", 0.6, 0, 1)
    gaus  = ROOT.RooGaussian("gaus", "", m, mean, sigma)
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
            if out:
                results[cls][f"{y}_{tr}"] = out

json.dump(results, open("yields.json", "w"), indent=2)
print("Saved → yields.json")
