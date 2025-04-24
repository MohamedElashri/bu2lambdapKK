import yaml, ROOT, json
import os
from loaders import load_data
from selections import trigger_mask
from branches import canonical

CFG = yaml.safe_load(open("config.yml"))

# Prevent plots from popping up
ROOT.gROOT.SetBatch(True)


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

    nentries = rds.numEntries()

    # Define model parameters
    mean  = ROOT.RooRealVar("mean", "mean", 5280, 5200, 5350)
    sigma1 = ROOT.RooRealVar("sigma1", "sigma1", 15, 5, 40) # Renamed from sigma
    sigma2 = ROOT.RooRealVar("sigma2", "sigma2", 40, 10, 80) # Second sigma
    fgaus1 = ROOT.RooRealVar("fgaus1", "fgaus1", 0.7, 0, 1) # Fraction of first Gaussian
    c     = ROOT.RooRealVar("c", "c", -0.001, -0.1, 0) # Exp slope
    nsig  = ROOT.RooRealVar("nsig", "Nsig", nentries*0.8, 0, nentries*1.2)
    nbkg  = ROOT.RooRealVar("nbkg", "Nbkg", nentries*0.2, 0, nentries*1.2)

    # Define PDFs
    gaus1 = ROOT.RooGaussian("gaus1", "Gaussian 1", m, mean, sigma1)
    gaus2 = ROOT.RooGaussian("gaus2", "Gaussian 2", m, mean, sigma2) # Using same mean
    signal_model = ROOT.RooAddPdf("signal_model", "Double Gaussian", gaus1, gaus2, fgaus1)
    background_model = ROOT.RooExponential("background_model", "Exponential background", m, c)

    # Combine signal and background
    model = ROOT.RooAddPdf("model", "Signal+Background", ROOT.RooArgList(signal_model, background_model), ROOT.RooArgList(nsig, nbkg))

    # Perform the fit
    # Use Extended(True) for RooAddPdf with yields
    fit_result = model.fitTo(rds, ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Save(), ROOT.RooFit.Extended(True))

    # Create and save the plot
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    plot_filename = os.path.join(plots_dir, f"{sample}_{year}_{track}.png")

    frame = m.frame(ROOT.RooFit.Title(f"Fit for {sample} {year} {track}"))
    rds.plotOn(frame, ROOT.RooFit.Name("data"))
    model.plotOn(frame, ROOT.RooFit.Name("total_fit"))
    # Plot components
    model.plotOn(frame, ROOT.RooFit.Components("background_model"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kGreen+2), ROOT.RooFit.Name("background"))
    model.plotOn(frame, ROOT.RooFit.Components("signal_model"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.Name("signal"))

    c1 = ROOT.TCanvas("c1", "", 800, 600)
    frame.Draw()
    c1.SaveAs(plot_filename)
    print(f"\tSaved plot -> {plot_filename}")

    # Return signal yield and error
    fitted_nsig = nsig.getVal()
    err  = nsig.getError()
    print(f"\tFitted Nsig = {fitted_nsig:.2f} +/- {err:.2f}")
    return fitted_nsig, err


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
