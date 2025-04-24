import yaml, ROOT, json
from ROOT import RooRealVar, RooGaussian, RooChebychev, RooAddPdf, RooArgList, RooDataSet, RooFit, RooArgSet, RooCBShape, TH1F, TLegend
import os
from loaders import load_data
from selections import trigger_mask
from branches import canonical

# Prevent plots from popping up
ROOT.gROOT.SetBatch(True)

# Load and apply LHCb style from the C macro
ROOT.gROOT.ProcessLine(".L lhcbStyle.C") # Load the macro file
ROOT.gROOT.ProcessLine("CMSStyle()")      # Try common name matching filename
print("Applied LHCb plotting style from lhcbStyle.C")

CFG = yaml.safe_load(open("config.yml"))

def fit(sample, year, track):
    """sample = 'sig' | 'norm'"""
    is_sig = sample == "sig"
    mode   = "L0barPKpKm" if is_sig else "KSKmKpPip"
    base   = CFG["signal_data_dir"] if is_sig else CFG["norm_data_dir"]

    # --- Generate Title and Filename --- (Moved up)
    year_str = str(year) if year != 'all' else "Run 2" # Or "Combined Years"
    track_str = track if track != 'all' else "Combined Tracks"
    sample_str = "Signal" if sample == "sig" else "Normalization"
    plot_title = f"{sample_str} Candidates ({year_str}, {track_str})"
    plot_filename_base = f"{sample}_{year}_{track}" # Use 'all' directly in filename

    print(f"\n--- Fitting: {plot_filename_base} ---")

    # Load data - it will have physical branch names like Bu_MM or B_MM
    data = load_data(data_path=base, decay_mode=mode,
                     years=[year], tracks=[track])
    if data is None:
        return None

    data = data[trigger_mask(data, "signal" if is_sig else "norm")]

    # Determine the correct physical mass column name using canonical
    mass_col = canonical("signal" if is_sig else "norm", ["mass"])[0]
    # Fit range
    fit_min, fit_max = 5200, 5400 # Changed range
    m = ROOT.RooRealVar("m", "mass", fit_min, fit_max) # Define variable without unit

    # --- Set Axis Titles ---
    # Set BEFORE creating the frame
    # Use # to denote LaTeX parts for ROOT
    m.SetTitle("M(B^{+}) [MeV/c^{2}]") # X-axis title with units
    # Y-axis title will be set later based on bin width

    # Create an empty dataset with the mass variable
    rds = ROOT.RooDataSet("rds", "", ROOT.RooArgSet(m))
    mass_data_np = data[mass_col].to_numpy() # Get NumPy array once

    # Ensure RooDataSet is filled only with data in the new range
    print(f"\tFilling RooDataSet for {sample}/{year}/{track} with {len(mass_data_np)} potential entries...")
    rds.reset() # Clear dataset before refilling
    mass_data_in_range = mass_data_np[(mass_data_np >= fit_min) & (mass_data_np <= fit_max)]
    for mass_val in mass_data_in_range:
         m.setVal(mass_val)
         rds.add(ROOT.RooArgSet(m))
    print(f"\t...filling complete with {rds.numEntries()} entries in range [{fit_min}, {fit_max}].")

    nentries = rds.numEntries() # Recalculated for the new range

    # Define model parameters
    # --- Signal: Crystal Ball ---
    mean  = ROOT.RooRealVar("mean", "mean", 5280, 5260, 5290)
    sigma = ROOT.RooRealVar("sigma", "sigma", 15, 5, 30)   
    alpha = ROOT.RooRealVar("alpha", "alpha", 1.5, 0.1, 5.0)  # CB tail parameter
    n     = ROOT.RooRealVar("n", "n", 2.0, 0.1, 10.0)    # CB tail parameter
    # --- Background: Chebychev ---
    c1    = ROOT.RooRealVar("c1", "c1", -0.1, -1, 1) # 1st order coefficient
    c2    = ROOT.RooRealVar("c2", "c2", 0.05, -1, 1) # 2nd order coefficient
    # Yields - Start closer to 50/50 and allow wider range
    # Use nentries (events in range) for initial guess
    nsig_init = nentries * 0.5 if nentries > 0 else 100 # Avoid 0 if no entries
    nbkg_init = nentries * 0.5 if nentries > 0 else 100
    nsig  = ROOT.RooRealVar("nsig", "Nsig", nsig_init, 0, nentries*1.5 if nentries > 0 else 1000)
    nbkg  = ROOT.RooRealVar("nbkg", "Nbkg", nbkg_init, 0, nentries*1.5 if nentries > 0 else 1000)

    # Define PDFs
    # --- Signal: Crystal Ball ---
    signal_model = ROOT.RooCBShape("signal_model", "Crystal Ball", m, mean, sigma, alpha, n)
    # 2nd order Chebychev background
    background_model = ROOT.RooChebychev("background_model", "Chebychev Background", m, ROOT.RooArgList(c1, c2))

    # Combine signal and background
    model = ROOT.RooAddPdf("model", "Signal+Background", ROOT.RooArgList(signal_model, background_model), ROOT.RooArgList(nsig, nbkg))

    # Perform the fit
    # Use Extended(True) for RooAddPdf with yields
    # Add PrintLevel(1) for more fit info, Minos(False) for stability
    fit_result = model.fitTo(rds, ROOT.RooFit.Save(), ROOT.RooFit.Extended(True),
                             ROOT.RooFit.PrintLevel(1), ROOT.RooFit.Minos(False)) # More verbose, disable Minos

    # Check fit status
    if fit_result:
        print(f"\tFit status: {fit_result.status()}, Covariance Matrix Quality: {fit_result.covQual()}")
        if fit_result.status() != 0:
            print("\tWARNING: Fit did not converge properly!")

    # Create and save the plot
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    plot_filename = os.path.join(plots_dir, f"{plot_filename_base}.pdf") # Use dynamic base name

    # Create frame WITH the Title argument
    frame = m.frame(ROOT.RooFit.Title(plot_title))
    # Explicitly set plotting bins
    n_bins_plot = 50 # Adjust bins for the smaller range? Let's try 50.
    rds.plotOn(frame, ROOT.RooFit.Name("data"), ROOT.RooFit.Binning(n_bins_plot))
    model.plotOn(frame, ROOT.RooFit.Name("total_fit"))#, ROOT.RooFit.Range(fit_min, fit_max), ROOT.RooFit.NormRange("fit_range"))
    # Plot components
    model.plotOn(frame, ROOT.RooFit.Components("background_model"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kGreen+2), ROOT.RooFit.Name("background"))#, ROOT.RooFit.Range(fit_min, fit_max), ROOT.RooFit.NormRange("fit_range"))
    model.plotOn(frame, ROOT.RooFit.Components("signal_model"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.Name("signal"))#, ROOT.RooFit.Range(fit_min, fit_max), ROOT.RooFit.NormRange("fit_range"))

    # Set Y-axis title (refined text)
    bin_width = (fit_max - fit_min) / n_bins_plot
    frame.SetYTitle(f"Candidates / ({bin_width:.1f} MeV/c^{{2}})") # Updated units

    canvas = ROOT.TCanvas("canvas", "", 1000, 750)
    frame.Draw()

    # Add Legend (position depends on sample)
    if is_sig:
        legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.85) # Top-right for signal
    else: # Normalization
        legend = ROOT.TLegend(0.7, 0.2, 0.9, 0.35) # Bottom-right (raised) for norm

    legend.SetBorderSize(0)
    legend.SetFillStyle(0) # Transparent background
    legend.SetTextSize(0.032) 

    # Set data legend entry based on sample type
    if is_sig:
        data_legend_text = "B^{+} #rightarrow #Lambda #bar{p} K^{+} K^{+}"
    else:
        data_legend_text = "B^{+} #rightarrow K_{S}^{0} #pi^{-} K^{+} K^{+}"

    legend.AddEntry(frame.findObject("data"), data_legend_text, "PE")
    legend.AddEntry(frame.findObject("total_fit"), "Fit", "L")
    legend.AddEntry(frame.findObject("signal"), "Signal", "L")
    legend.AddEntry(frame.findObject("background"), "Background", "L")
    legend.Draw()

    canvas.SaveAs(plot_filename)
    print(f"\tSaved plot -> {plot_filename}")

    # Return fit results as a dictionary
    final_params = fit_result.floatParsFinal()
    result_params = {}
    for param in [nsig, nbkg, mean, sigma]:
        par_name = param.GetName()
        fit_param = final_params.find(par_name)
        if fit_param:
            result_params[par_name] = {
                'value': fit_param.getVal(),
                'error': fit_param.getError()
            }
        else: # Handle fixed parameters if needed (though param_list should only contain floated ones)
             result_params[par_name] = {'value': param.getVal(), 'error': 0.0} 
    if is_sig:
        for param in [alpha, n]:
            par_name = param.GetName()
            fit_param = final_params.find(par_name)
            if fit_param:
                result_params[par_name] = {
                    'value': fit_param.getVal(),
                    'error': fit_param.getError()
                }
            else: # Handle fixed parameters if needed (though param_list should only contain floated ones)
                 result_params[par_name] = {'value': param.getVal(), 'error': 0.0} 
    else: # norm
        for param in [c1]:
            par_name = param.GetName()
            fit_param = final_params.find(par_name)
            if fit_param:
                result_params[par_name] = {
                    'value': fit_param.getVal(),
                    'error': fit_param.getError()
                }
            else: # Handle fixed parameters if needed (though param_list should only contain floated ones)
                 result_params[par_name] = {'value': param.getVal(), 'error': 0.0} 
    # Optionally add fit status and covQual
    result_params['_fit_status'] = fit_result.status()
    result_params['_cov_qual'] = fit_result.covQual()

    # Clean up
    canvas.Close()
    del frame
    del rds
    return result_params

if __name__ == "__main__":
    years = CFG["years"]
    tracks = CFG["tracks"]      

    all_fit_results = {} # Dictionary to store all results

    # --- Individual Fits ---
    for cls in ["sig", "norm"]:
        for y in years:
            for tr in tracks:
                # Pass sample, year, track to fit function
                fit_id = f"{cls}_{y}_{tr}"
                print(f"\n--- Fitting: {fit_id} ---")
                try:
                    params = fit(cls, y, tr)
                    all_fit_results[fit_id] = params
                except Exception as e:
                    print(f"ERROR fitting {fit_id}: {e}")
                    all_fit_results[fit_id] = {'error': str(e)}

    # --- Combined Fits ---
    print("\n--- Processing Combined Datasets ---")
    for cls in ["sig", "norm"]:
        for tr in tracks:
            fit_id = f"{cls}_all_{tr}"
            print(f"\nFitting combined dataset: {fit_id}")
            try:
                params = fit(cls, 'all', tr)
                all_fit_results[fit_id] = params
            except Exception as e:
                print(f"ERROR fitting {fit_id}: {e}")
                all_fit_results[fit_id] = {'error': str(e)}

        fit_id = f"{cls}_all_all"
        print(f"\nFitting combined dataset: {fit_id}")
        try:
            params = fit(cls, 'all', 'all')
            all_fit_results[fit_id] = params
        except Exception as e:
            print(f"ERROR fitting {fit_id}: {e}")
            all_fit_results[fit_id] = {'error': str(e)}

    # Save results to JSON
    results_filename = "fit_results.json"
    with open(results_filename, 'w') as f:
        json.dump(all_fit_results, f, indent=4)
    print(f"\nSaved all fit results to {results_filename}")
