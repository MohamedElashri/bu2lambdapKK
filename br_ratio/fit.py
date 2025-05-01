import yaml, ROOT, json
from ROOT import RooRealVar, RooGaussian, RooChebychev, RooAddPdf, RooArgList, RooDataSet
from ROOT import RooFit, RooArgSet, RooCBShape, TH1F, TLegend, TLatex
import os
import numpy as np
from loaders import load_data
from selections import trigger_mask, create_selection_mask
from branches import _resolve_branch_name

# Prevent plots from popping up
ROOT.gROOT.SetBatch(True)

# Load and apply LHCb style from the C macro
try:
    ROOT.gROOT.ProcessLine(".L lhcbStyle.C")  # Load the macro file
    ROOT.gROOT.ProcessLine("CMSStyle()")     # Apply the style
    print("Applied LHCb plotting style from lhcbStyle.C")
except:
    print("Could not load LHCb style, using default styles")
    # Apply some basic style settings
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetOptFit(1)

# Suppress RooFit info messages
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

# Load configuration
CFG = yaml.safe_load(open("config.yml"))

# Add a PDF factory function
def create_pdf(pdf_type, pdf_name, observable, params_dict, order=None):
    """
    Factory function to create RooFit PDFs based on the configuration
    
    Parameters:
    -----------
    pdf_type : str
        Type of PDF to create (e.g., 'CrystalBall', 'Chebychev')
    pdf_name : str
        Name to assign to the PDF object
    observable : RooRealVar
        Observable variable (usually mass)
    params_dict : dict
        Dictionary of parameter objects (RooRealVars)
    order : int, optional
        Order for PDFs that need it (e.g., polynomial order)
        
    Returns:
    --------
    RooAbsPdf
        The created PDF object
    """
    pdf_type = pdf_type.strip()
    
    # Signal PDF types
    if pdf_type == "CrystalBall":
        return ROOT.RooCBShape(pdf_name, "Crystal Ball", 
                               observable, 
                               params_dict["mean"], 
                               params_dict["sigma"], 
                               params_dict["alpha"], 
                               params_dict["n"])
    
    elif pdf_type == "Gaussian":
        return ROOT.RooGaussian(pdf_name, "Gaussian", 
                              observable, 
                              params_dict["mean"], 
                              params_dict["sigma"])
    
    elif pdf_type == "DoubleGaussian":
        # Create individual Gaussians
        g1 = ROOT.RooGaussian(f"{pdf_name}_g1", "Gaussian 1", 
                            observable, 
                            params_dict["mean"], 
                            params_dict["sigma1"])
        
        g2 = ROOT.RooGaussian(f"{pdf_name}_g2", "Gaussian 2", 
                            observable, 
                            params_dict["mean"], 
                            params_dict["sigma2"])
        
        # Combine them with a fraction parameter
        return ROOT.RooAddPdf(pdf_name, "Double Gaussian", 
                           ROOT.RooArgList(g1, g2), 
                           ROOT.RooArgList(params_dict["frac"]))
    
    elif pdf_type == "Voigtian":
        return ROOT.RooVoigtian(pdf_name, "Voigtian", 
                              observable, 
                              params_dict["mean"], 
                              params_dict["width"], 
                              params_dict["sigma"])
    
    elif pdf_type == "Breit-Wigner":
        return ROOT.RooBreitWigner(pdf_name, "Breit-Wigner", 
                                 observable, 
                                 params_dict["mean"], 
                                 params_dict["width"])
    
    # Background PDF types
    elif pdf_type == "Chebychev":
        # Get coefficients based on order
        order = order if order is not None else 1
        coef_list = ROOT.RooArgList()
        for i in range(1, order + 1):
            coef_name = f"c{i}"
            if coef_name in params_dict:
                coef_list.add(params_dict[coef_name])
        
        return ROOT.RooChebychev(pdf_name, f"Chebychev Background (order {order})", 
                              observable, coef_list)
    
    elif pdf_type == "Exponential":
        return ROOT.RooExponential(pdf_name, "Exponential Background", 
                                 observable, 
                                 params_dict["c"])
    
    elif pdf_type == "Argus":
        return ROOT.RooArgusBG(pdf_name, "Argus Background", 
                            observable, 
                            params_dict["m0"], 
                            params_dict["c"], 
                            params_dict["p"])
    
    elif pdf_type == "Polynomial":
        # Get coefficients based on order
        order = order if order is not None else 1
        coef_list = ROOT.RooArgList()
        for i in range(1, order + 1):
            coef_name = f"c{i}"
            if coef_name in params_dict:
                coef_list.add(params_dict[coef_name])
        
        return ROOT.RooPolynomial(pdf_name, f"Polynomial Background (order {order})", 
                               observable, coef_list)
    
    else:
        print(f"WARNING: Unknown PDF type '{pdf_type}', defaulting to Gaussian")
        # Default to a simple Gaussian for unknown types
        return ROOT.RooGaussian(pdf_name, "Default Gaussian", 
                               observable, 
                               params_dict["mean"], 
                               params_dict.get("sigma", params_dict.get("width", None)))

def fit(sample, year, track):
    """
    Perform mass fit for signal or normalization channel
    
    Parameters:
    -----------
    sample : str
        'sig' for signal or 'norm' for normalization channel
    year : str or int
        Year of data taking or 'all' for combined
    track : str
        Track type ('LL', 'DD') or 'all' for combined
        
    Returns:
    --------
    dict
        Dictionary of fit parameters and results
    """
    is_sig = sample == "sig"
    mode = "L0barPKpKm" if is_sig else "KSKmKpPip"
    base = CFG["signal_data_dir"] if is_sig else CFG["norm_data_dir"]
    sample_label = "signal" if is_sig else "norm"  # For trigger_mask and canonical

    # --- Generate Title and Filename ---
    year_str = str(year) if year != 'all' else "Run 2"
    track_str = track if track != 'all' else "Combined Tracks"
    sample_str = "Signal" if is_sig else "Normalization"
    decay_str = "#bar{\Lambda}^{0}pK^{+}K^{-}" if is_sig else "K^{0}_{S}#pi^{+}K^{+}K^{-}"
    plot_title = f"B^{{+}} #rightarrow {decay_str} ({year_str}, {track_str})"
    plot_filename_base = f"{sample}_{year}_{track}"

    print(f"\n--- Fitting: {plot_filename_base} ---")

    # Load data - it will have physical branch names like Bu_MM or B_MM
    data = load_data(data_path=base, decay_mode=mode,
                     years=[year], tracks=[track])
    if data is None or len(data) == 0:
        print(f"No data found for {sample}/{year}/{track}")
        return None

    # First apply trigger selection
    data = data[trigger_mask(data, sample_label)]
    if len(data) == 0:
        print(f"No data remaining after trigger selection for {sample}/{year}/{track}")
        return None
    
    # Apply additional selection cuts
    selection_mask = create_selection_mask(data, sample_label)
    data = data[selection_mask]
    if len(data) == 0:
        print(f"No data remaining after selection cuts for {sample}/{year}/{track}")
        return None

    # Determine the correct physical mass column name using canonical
    mass_col = _resolve_branch_name("mass", sample_label) # Resolve 'mass' directly
    
    # Get model configuration from config
    fit_config = CFG["fit_params"]
    model_config = fit_config["models"]["signal"] if is_sig else fit_config["models"]["norm"]
    
    # Set fit range from config
    fit_min, fit_max = fit_config["mass_range"]["signal"] if is_sig else fit_config["mass_range"]["norm"]
    m = ROOT.RooRealVar("m", "mass", fit_min, fit_max)

    # --- Set Axis Titles ---
    m.SetTitle(fit_config["plotting"]["axis_title"])

    # Create an empty dataset with the mass variable
    rds = ROOT.RooDataSet("rds", "", ROOT.RooArgSet(m))
    mass_data_np = data[mass_col].to_numpy()

    # Ensure RooDataSet is filled only with data in the fit range
    print(f"\tFilling RooDataSet for {sample}/{year}/{track} with {len(mass_data_np)} potential entries...")
    rds.reset()  # Clear dataset before refilling
    mass_data_in_range = mass_data_np[(mass_data_np >= fit_min) & (mass_data_np <= fit_max)]
    for mass_val in mass_data_in_range:
        m.setVal(mass_val)
        rds.add(ROOT.RooArgSet(m))
    print(f"\t...filling complete with {rds.numEntries()} entries in range [{fit_min}, {fit_max}].")

    nentries = rds.numEntries()  # Recalculated for the new range
    if nentries == 0:
        print(f"No entries in fit range for {sample}/{year}/{track}")
        return None

    # --- Create RooRealVars for all parameters in the configuration ---
    params = model_config["parameters"]
    param_vars = {}  # Dictionary to store parameter variables
    
    # Loop through all parameters in the config and create corresponding RooRealVars
    for param_name, param_values in params.items():
        if len(param_values) >= 3:  # We need at least [initial, min, max]
            init_val, min_val, max_val = param_values[:3]
            param_vars[param_name] = ROOT.RooRealVar(param_name, param_name, init_val, min_val, max_val)

    # --- Yields ---
    # Use nentries (events in range) for initial guess
    nsig_init = nentries * 0.5 if nentries > 0 else 100  # Avoid 0 if no entries
    nbkg_init = nentries * 0.5 if nentries > 0 else 100
    nsig = ROOT.RooRealVar("nsig", "Nsig", nsig_init, 0, nentries*1.5 if nentries > 0 else 1000)
    nbkg = ROOT.RooRealVar("nbkg", "Nbkg", nbkg_init, 0, nentries*1.5 if nentries > 0 else 1000)

    # --- Create signal and background PDFs based on configuration ---
    signal_pdf_type = model_config["signal_pdf"]
    background_pdf_type = model_config["background_pdf"]
    background_order = model_config.get("background_order", 1)
    
    print(f"\tCreating signal PDF: {signal_pdf_type}")
    signal_model = create_pdf(signal_pdf_type, "signal_model", m, param_vars)
    
    print(f"\tCreating background PDF: {background_pdf_type} (order: {background_order})")
    background_model = create_pdf(background_pdf_type, "background_model", m, param_vars, background_order)

    # Combine signal and background
    model_comp_list = ROOT.RooArgList(signal_model, background_model)
    model_yield_list = ROOT.RooArgList(nsig, nbkg)
    model = ROOT.RooAddPdf("model", "Signal+Background", model_comp_list, model_yield_list)

    # Get fit options from config
    fit_options = fit_config["fit_options"]
    
    # Perform the fit with settings from config
    fit_result = model.fitTo(
        rds, 
        ROOT.RooFit.Save(),
        ROOT.RooFit.Extended(fit_options["use_extended"]),
        ROOT.RooFit.PrintLevel(fit_options["print_level"]),
        ROOT.RooFit.Minos(fit_options["use_minos"]),  
        ROOT.RooFit.InitialHesse(fit_options["use_initial_hesse"])
    )

    # Check fit status
    if fit_result:
        print(f"\tFit status: {fit_result.status()}, Covariance Matrix Quality: {fit_result.covQual()}")
        if fit_result.status() != 0 or fit_result.covQual() < 2:
            print("\tWARNING: Fit did not converge properly!")

    # Create and save the plot
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    plot_filename = os.path.join(plots_dir, f"{plot_filename_base}.pdf")

    # Create frame WITH the Title argument
    frame = m.frame(ROOT.RooFit.Title(plot_title))
    
    # Explicitly set plotting bins from config
    n_bins_plot = fit_config["plotting"]["nbins"]
    rds.plotOn(frame, ROOT.RooFit.Name("data"), ROOT.RooFit.Binning(n_bins_plot))
    model.plotOn(frame, ROOT.RooFit.Name("total_fit"))
    
    # Plot components - only background, not signal
    model.plotOn(frame, ROOT.RooFit.Components("background_model"), 
                ROOT.RooFit.LineStyle(ROOT.kDashed), 
                ROOT.RooFit.LineColor(ROOT.kGreen+2), 
                ROOT.RooFit.Name("background"))

    # Set Y-axis title (refined text)
    bin_width = (fit_max - fit_min) / n_bins_plot
    frame.SetYTitle(f"Candidates / ({bin_width:.1f} MeV/c^{{2}})")

    canvas = ROOT.TCanvas("canvas", "", 1000, 750)
    frame.Draw()

    # Add Legend (position depends on sample)
    if is_sig:
        legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.85)  # Top-right for signal
    else:
        legend = ROOT.TLegend(0.7, 0.2, 0.9, 0.35)  # Bottom-right for norm

    legend.SetBorderSize(0)
    legend.SetFillStyle(0)  # Transparent background
    legend.SetTextSize(0.032)

    # Set data legend entry based on sample type
    if is_sig:
        data_legend_text = "B^{+} #rightarrow #bar{\Lambda} #bar{p} K^{+} K^{-}"
    else:
        data_legend_text = "B^{+} #rightarrow K_{S}^{0} #pi^{+} K^{+} K^{-}"

    legend.AddEntry(frame.findObject("data"), data_legend_text, "PE")
    legend.AddEntry(frame.findObject("total_fit"), "Fit", "L")
    legend.AddEntry(frame.findObject("background"), "Background", "L")
    legend.Draw()

    # Calculate chi2 without adding to plot
    try:
        chi2 = frame.chiSquare("total_fit", "data", fit_result.floatParsFinal().getSize())
        print(f"\tChi2/ndf = {chi2:.3f}")
    except:
        print("\tWarning: Could not calculate chi2")
    
    # Store fit status without adding to plot
    fit_status = fit_result.status()
    cov_qual = fit_result.covQual()
    print(f"\tFit status: {fit_status}, Covariance quality: {cov_qual}")

    canvas.SaveAs(plot_filename)
    print(f"\tSaved plot -> {plot_filename}")

    # Return fit results as a dictionary
    final_params = fit_result.floatParsFinal()
    result_params = {}
    
    # Basic parameters for both channels
    for param in [nsig, nbkg, param_vars["mean"], param_vars["sigma"]]:
        par_name = param.GetName()
        fit_param = final_params.find(par_name)
        if fit_param:
            result_params[par_name] = {
                'value': fit_param.getVal(),
                'error': fit_param.getError()
            }
        else:
            # Handle fixed parameters if needed
            result_params[par_name] = {'value': param.getVal(), 'error': 0.0}
    
    # Crystal Ball parameters for both channels
    for param in [param_vars["alpha"], param_vars["n"]]:
        par_name = param.GetName()
        fit_param = final_params.find(par_name)
        if fit_param:
            result_params[par_name] = {
                'value': fit_param.getVal(),
                'error': fit_param.getError()
            }
        else:
            result_params[par_name] = {'value': param.getVal(), 'error': 0.0}
    
    # Background parameters for both channels
    param = param_vars["c1"]
    par_name = param.GetName()
    fit_param = final_params.find(par_name)
    if fit_param:
        result_params[par_name] = {
            'value': fit_param.getVal(),
            'error': fit_param.getError()
        }
    else:
        result_params[par_name] = {'value': param.getVal(), 'error': 0.0}
    
    # Add c2 parameter for signal channel only
    if is_sig and "c2" in param_vars:
        param = param_vars["c2"]
        par_name = param.GetName()
        fit_param = final_params.find(par_name)
        if fit_param:
            result_params[par_name] = {
                'value': fit_param.getVal(),
                'error': fit_param.getError()
            }
        else:
            result_params[par_name] = {'value': param.getVal(), 'error': 0.0}
    
    # Add fit status information
    result_params['_fit_status'] = fit_result.status()
    result_params['_cov_qual'] = fit_result.covQual()
    if 'chi2' in locals():
        result_params['_chi2_ndf'] = chi2
    
    # Signal significance
    if nsig.getVal() > 0 and nsig.getError() > 0:
        significance = nsig.getVal() / nsig.getError()
        result_params['significance'] = significance
        print(f"\tSignal significance: {significance:.2f} sigma")

    # Clean up
    canvas.Close()
    del frame
    del rds
    
    return result_params

def compare_channels(results_sig, results_norm, year='all', track='all'):
    """
    Compare fit results between signal and normalization channels
    
    Parameters:
    -----------
    results_sig : dict
        Fit results for signal channel
    results_norm : dict
        Fit results for normalization channel
    year : str
        Year identifier for the comparison
    track : str
        Track type identifier
        
    Returns:
    --------
    dict
        Dictionary with comparison results
    """
    if results_sig is None or results_norm is None:
        print("Cannot compare channels: missing results")
        return None
    
    comparison = {}
    track_label = "all tracks" if track == 'all' else f"track type {track}"
    year_label = "Run 2" if year == 'all' else f"year {year}"
    
    print(f"\n=== Detailed Channel Comparison ({year_label}, {track_label}) ===")
    print(f"{'Parameter':<15} {'Signal':<22} {'Normalization':<22} {'Ratio':<15}")
    print("-" * 75)
    
    # Compare yields
    sig_yield_val = results_sig['nsig']['value']
    sig_yield_err = results_sig['nsig']['error'] 
    norm_yield_val = results_norm['nsig']['value']
    norm_yield_err = results_norm['nsig']['error']
    
    sig_yield = f"{sig_yield_val:.1f} ± {sig_yield_err:.1f}"
    norm_yield = f"{norm_yield_val:.1f} ± {norm_yield_err:.1f}"
    
    # Store in comparison dict
    comparison['yields'] = {
        'signal': {'value': sig_yield_val, 'error': sig_yield_err},
        'norm': {'value': norm_yield_val, 'error': norm_yield_err}
    }
    
    print(f"{'Yield':<15} {sig_yield:<22} {norm_yield:<22}")
    
    # Compare masses
    sig_mass_val = results_sig['mean']['value']
    sig_mass_err = results_sig['mean']['error']
    norm_mass_val = results_norm['mean']['value']
    norm_mass_err = results_norm['mean']['error']
    
    sig_mass = f"{sig_mass_val:.1f} ± {sig_mass_err:.1f}"
    norm_mass = f"{norm_mass_val:.1f} ± {norm_mass_err:.1f}"
    
    # Mass difference (could be important)
    mass_diff = sig_mass_val - norm_mass_val
    mass_diff_err = np.sqrt(sig_mass_err**2 + norm_mass_err**2)
    
    comparison['masses'] = {
        'signal': {'value': sig_mass_val, 'error': sig_mass_err},
        'norm': {'value': norm_mass_val, 'error': norm_mass_err},
        'difference': {'value': mass_diff, 'error': mass_diff_err}
    }
    
    print(f"{'Mass (MeV)':<15} {sig_mass:<22} {norm_mass:<22} {mass_diff:.1f} ± {mass_diff_err:.1f}")
    
    # Compare widths
    sig_width_val = results_sig['sigma']['value']
    sig_width_err = results_sig['sigma']['error']
    norm_width_val = results_norm['sigma']['value'] 
    norm_width_err = results_norm['sigma']['error']
    
    sig_width = f"{sig_width_val:.1f} ± {sig_width_err:.1f}"
    norm_width = f"{norm_width_val:.1f} ± {norm_width_err:.1f}"
    
    # Width ratio
    try:
        width_ratio = sig_width_val / norm_width_val
        width_ratio_err = width_ratio * np.sqrt(
            (sig_width_err/sig_width_val)**2 + 
            (norm_width_err/norm_width_val)**2
        )
        width_ratio_str = f"{width_ratio:.2f} ± {width_ratio_err:.2f}"
    except:
        width_ratio_str = "N/A"
        width_ratio = None
        width_ratio_err = None
    
    comparison['widths'] = {
        'signal': {'value': sig_width_val, 'error': sig_width_err},
        'norm': {'value': norm_width_val, 'error': norm_width_err}
    }
    
    if width_ratio is not None:
        comparison['widths']['ratio'] = {'value': width_ratio, 'error': width_ratio_err}
    
    print(f"{'Width (MeV)':<15} {sig_width:<22} {norm_width:<22} {width_ratio_str}")
    
    # Compare significances if available
    if 'significance' in results_sig or 'significance' in results_norm:
        sig_signif_val = results_sig.get('significance', 0)
        norm_signif_val = results_norm.get('significance', 0)
        
        sig_signif = f"{sig_signif_val:.2f}σ"
        norm_signif = f"{norm_signif_val:.2f}σ"
        
        comparison['significance'] = {
            'signal': sig_signif_val,
            'norm': norm_signif_val
        }
        
        print(f"{'Significance':<15} {sig_signif:<22} {norm_signif:<22}")
    
    # Calculate signal ratio (S/N ratio is key for branching fraction)
    try:
        ratio = sig_yield_val / norm_yield_val
        ratio_err = ratio * np.sqrt(
            (sig_yield_err / sig_yield_val)**2 +
            (norm_yield_err / norm_yield_val)**2
        )
        
        comparison['yield_ratio'] = {'value': ratio, 'error': ratio_err}
        
        print(f"\nSignal / Norm ratio = {ratio:.4f} ± {ratio_err:.4f}")
        
        # Calculate branching fraction if we have the normalization BR
        if 'br_norm_pdg' in CFG and 'br_norm_pdg_unc' in CFG:
            br_norm = CFG['br_norm_pdg']
            br_norm_unc = CFG['br_norm_pdg_unc']
            
            # BR(sig) = BR(norm) * N_sig/N_norm * efficiency factors
            # Efficiency factors would come from MC - not including here
            br_sig_est = br_norm * ratio
            br_sig_est_err = br_sig_est * np.sqrt(
                (ratio_err/ratio)**2 + 
                (br_norm_unc/br_norm)**2
            )
            
            comparison['br_estimate'] = {
                'value': br_sig_est,
                'error': br_sig_est_err
            }
            
            print(f"Est. branching fraction = ({br_sig_est:.2e} ± {br_sig_est_err:.2e})")
            print("(Note: Efficiency corrections not applied)")
    except:
        print("\nCould not calculate signal ratio")
    
    return comparison

def run_all_fits(run_individual=False):
    """
    Run mass fits for specific combinations or all combinations
    
    Parameters:
    -----------
    run_individual : bool, optional
        If True, run fits for each individual year/track combination
        Default is False (only run the combined fits)
    
    Returns:
    --------
    dict
        Dictionary with all fit results
    """
    # Get years and tracks from config file
    years = CFG.get("years", ["2016", "2017", "2018"])
    tracks = CFG.get("tracks", ["LL", "DD"])
    
    all_fit_results = {}  # Dictionary to store all results
    
    # --- Individual Fits (optional) ---
    if run_individual:
        print("\n=== Fitting Individual Year/Track Combinations ===")
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
    
    # --- Default Fits: All years combined, by track type ---
    print("\n\n=== Fitting Default Combined Datasets ===")
    
    # 1. All data (combined years and tracks)
    print("\n--- Fitting all years and all tracks combined ---")
    for cls in ["sig", "norm"]:
        fit_id = f"{cls}_all_all"
        print(f"\nFitting combined dataset: {fit_id}")
        try:
            params = fit(cls, 'all', 'all')
            all_fit_results[fit_id] = params
        except Exception as e:
            print(f"ERROR fitting {fit_id}: {e}")
            all_fit_results[fit_id] = {'error': str(e)}
    
    # 2. All LL tracks (combined years)
    print("\n--- Fitting all years with LL tracks ---")
    for cls in ["sig", "norm"]:
        fit_id = f"{cls}_all_LL"
        print(f"\nFitting combined dataset: {fit_id}")
        try:
            params = fit(cls, 'all', 'LL')
            all_fit_results[fit_id] = params
        except Exception as e:
            print(f"ERROR fitting {fit_id}: {e}")
            all_fit_results[fit_id] = {'error': str(e)}
    
    # 3. All DD tracks (combined years)
    print("\n--- Fitting all years with DD tracks ---")
    for cls in ["sig", "norm"]:
        fit_id = f"{cls}_all_DD"
        print(f"\nFitting combined dataset: {fit_id}")
        try:
            params = fit(cls, 'all', 'DD')
            all_fit_results[fit_id] = params
        except Exception as e:
            print(f"ERROR fitting {fit_id}: {e}")
            all_fit_results[fit_id] = {'error': str(e)}
    
    # Save results to JSON
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_filename = os.path.join(results_dir, "fit_results.json")
    with open(results_filename, 'w') as f:
        json.dump(all_fit_results, f, indent=4)
    print(f"\nSaved all fit results to {results_filename}")
    
    # Print comprehensive summary table
    print("\n" + "="*80)
    print(" "*30 + "FITTING SUMMARY")
    print("="*80)
    
    # Display summary for signal channel
    print(f"\nSIGNAL CHANNEL (B+ → Λ̅0pK+K-)")
    print("-"*80)
    print(f"{'Dataset':<15} {'Yield':<20} {'Mass (MeV)':<20} {'Width (MeV)':<15} {'Significance':<12}")
    print("-"*80)
    
    # Print results for the 3 default signal fits
    for fit_id in ["sig_all_all", "sig_all_LL", "sig_all_DD"]:
        if fit_id in all_fit_results and all_fit_results[fit_id] and 'error' not in all_fit_results[fit_id]:
            res = all_fit_results[fit_id]
            # Display in format: "all/all", "all/LL", "all/DD"
            dataset = "/".join(fit_id.split("_")[1:])
            yield_val = f"{res['nsig']['value']:.1f} ± {res['nsig']['error']:.1f}"
            mass = f"{res['mean']['value']:.1f} ± {res['mean']['error']:.1f}"
            width = f"{res['sigma']['value']:.1f} ± {res['sigma']['error']:.1f}"
            signif = f"{res.get('significance', 0):.2f}σ"
            print(f"{dataset:<15} {yield_val:<20} {mass:<20} {width:<15} {signif:<12}")
    
    # Display summary for normalization channel
    print(f"\nNORMALIZATION CHANNEL (B+ → K0sπ+K+K-)")
    print("-"*80)
    print(f"{'Dataset':<15} {'Yield':<20} {'Mass (MeV)':<20} {'Width (MeV)':<15} {'Significance':<12}")
    print("-"*80)
    
    # Print results for the 3 default normalization fits
    for fit_id in ["norm_all_all", "norm_all_LL", "norm_all_DD"]:
        if fit_id in all_fit_results and all_fit_results[fit_id] and 'error' not in all_fit_results[fit_id]:
            res = all_fit_results[fit_id]
            dataset = "/".join(fit_id.split("_")[1:])
            yield_val = f"{res['nsig']['value']:.1f} ± {res['nsig']['error']:.1f}"
            mass = f"{res['mean']['value']:.1f} ± {res['mean']['error']:.1f}"
            width = f"{res['sigma']['value']:.1f} ± {res['sigma']['error']:.1f}"
            signif = f"{res.get('significance', 0):.2f}σ"
            print(f"{dataset:<15} {yield_val:<20} {mass:<20} {width:<15} {signif:<12}")
    
    # Channel comparison section
    print("\n" + "="*80)
    print(" "*25 + "SIGNAL/NORMALIZATION RATIO")
    print("="*80)
    print(f"{'Dataset':<15} {'Signal Yield':<20} {'Norm Yield':<20} {'Ratio S/N':<20}")
    print("-"*80)
    
    # Compare ratios for the default fits
    for track in ["all", "LL", "DD"]:
        sig_results = all_fit_results.get(f"sig_all_{track}")
        norm_results = all_fit_results.get(f"norm_all_{track}")
        
        if sig_results and norm_results and 'error' not in sig_results and 'error' not in norm_results:
            dataset = f"all/{track}"
                
            sig_yield = f"{sig_results['nsig']['value']:.1f} ± {sig_results['nsig']['error']:.1f}"
            norm_yield = f"{norm_results['nsig']['value']:.1f} ± {norm_results['nsig']['error']:.1f}"
            
            # Calculate signal ratio
            try:
                ratio = sig_results['nsig']['value'] / norm_results['nsig']['value']
                ratio_err = ratio * np.sqrt(
                    (sig_results['nsig']['error'] / sig_results['nsig']['value'])**2 +
                    (norm_results['nsig']['error'] / norm_results['nsig']['value'])**2
                )
                ratio_str = f"{ratio:.4f} ± {ratio_err:.4f}"
            except:
                ratio_str = "N/A"
                
            print(f"{dataset:<15} {sig_yield:<20} {norm_yield:<20} {ratio_str:<20}")
    
    print("\nAll fit results details saved to:", results_filename)
    
    # Run detailed channel comparisons for default fits
    comparisons = {}
    
    # Compare channels for default combinations
    for track in ["all", "LL", "DD"]:
        sig_results = all_fit_results.get(f"sig_all_{track}")
        norm_results = all_fit_results.get(f"norm_all_{track}")
        if sig_results and norm_results and 'error' not in sig_results and 'error' not in norm_results:
            comp = compare_channels(sig_results, norm_results, 'all', track)
            if comp:
                comparisons[f"all_{track}"] = comp
    
    # Save comparisons to JSON
    comparison_filename = os.path.join(results_dir, "channel_comparisons.json")
    with open(comparison_filename, 'w') as f:
        json.dump(comparisons, f, indent=4)
    print(f"Channel comparisons saved to: {comparison_filename}")
    
    return all_fit_results

if __name__ == "__main__":
    # Add command line argument parsing
    import argparse
    parser = argparse.ArgumentParser(description="Run B→ΛpKK fitting procedures")
    parser.add_argument("--all", action="store_true", 
                        help="Run fits for all individual year/track combinations")
    args = parser.parse_args()
    
    # Run the fits with the specified options
    results = run_all_fits(run_individual=args.all)