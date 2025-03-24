import numpy as np
import ROOT
from ROOT import TH1F, TCanvas, TF1, RooRealVar, RooGaussian, RooDataHist, RooCBShape
from ROOT import RooFit, RooPlot, RooArgList, RooArgSet, TLatex, gStyle
from ROOT import RooAddPdf, RooFFTConvPdf, RooDataSet, RooFormulaVar
from array import array
import os
import json
import pickle

# Define a custom Double Crystal Ball function since RooDoubleCB isn't available
def create_double_crystal_ball(name, title, x, mean, sigma, alpha1, n1, alpha2, n2):
    """
    Create a Double Crystal Ball PDF from scratch (using RooGenericPdf)
    
    Parameters:
    -----------
    name, title : str
        Name and title for the PDF
    x : RooRealVar
        Observable (mass)
    mean, sigma : RooRealVar
        Mean and width of the Gaussian core
    alpha1, n1 : RooRealVar
        Parameters for the left tail
    alpha2, n2 : RooRealVar
        Parameters for the right tail
        
    Returns:
    --------
    RooGenericPdf
    """
    # CB function formula - transition to power law when (x-mean)/sigma < -alpha
    formula = (
        f"(("
        f"((x-{mean.GetName()})<(-{alpha1.GetName()}*{sigma.GetName()})) * "  # Left tail
        f"exp({alpha1.GetName()}*{alpha1.GetName()}/2) * "
        f"pow({mean.GetName()}-x+{alpha1.GetName()}*{sigma.GetName()},{n1.GetName()}) / "
        f"pow({alpha1.GetName()},{n1.GetName()}) * {sigma.GetName()} * "
        f"pow({n1.GetName()}/{alpha1.GetName()},{n1.GetName()}) / {n1.GetName()}"
        f") + "
        f"(((x-{mean.GetName()})>=(-{alpha1.GetName()}*{sigma.GetName()})) && "  # Gaussian core (left)
        f"((x-{mean.GetName()})<=({alpha2.GetName()}*{sigma.GetName()}))) * "
        f"exp(-(x-{mean.GetName()})*(x-{mean.GetName()})/(2*{sigma.GetName()}*{sigma.GetName()}))"
        f" + "
        f"((x-{mean.GetName()})>({alpha2.GetName()}*{sigma.GetName()})) * "  # Right tail
        f"exp({alpha2.GetName()}*{alpha2.GetName()}/2) * "
        f"pow(x-{mean.GetName()}+{alpha2.GetName()}*{sigma.GetName()},{n2.GetName()}) / "
        f"pow({alpha2.GetName()},{n2.GetName()}) * {sigma.GetName()} * "
        f"pow({n2.GetName()}/{alpha2.GetName()},{n2.GetName()}) / {n2.GetName()}"
        f")"
    )
    
    # Create the RooGenericPdf
    dcb = ROOT.RooGenericPdf(name, title, formula, ROOT.RooArgList(x, mean, sigma, alpha1, n1, alpha2, n2))
    
    return dcb

# Configure ROOT settings
ROOT.gROOT.SetBatch(True)  # Run in batch mode (no graphics)
gStyle.SetOptStat(0)       # Don't display stat box
gStyle.SetOptFit(1)        # Display fit parameters
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)  # Reduce verbosity

def create_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

def create_root_histogram(data, bins, x_min, x_max, name="h_data", title="Data"):
    """Convert numpy array to ROOT histogram"""
    hist = TH1F(name, title, len(bins)-1, array('d', bins))
    for val in data:
        if x_min <= val <= x_max:
            hist.Fill(val)
    return hist

def extract_signal_shape_crystal_ball(mc_data, true_id_branch='Bu_TRUEID', mass_branch='Bu_MM', 
                         track_type=None, track_branch=None, output_dir="./"):
    """
    Extract signal shape from MC using single Crystal Ball
    
    Parameters:
    -----------
    mc_data : awkward array or dict
        MC data containing mass and truth information
    true_id_branch : str
        Name of the branch containing truth matching information
    mass_branch : str
        Name of the branch containing the B mass
    track_type : str, optional
        If specified, only use this track type ('LL' or 'DD')
    track_branch : str, optional
        Name of the branch containing track type information
    output_dir : str
        Directory to save output plots
        
    Returns:
    --------
    dict of shape parameters
    """
    # Create output directory
    create_output_dir(output_dir)
    
    # Get mass data and apply truth matching
    mass_data = np.array(mc_data[mass_branch])
    truth_match = np.array(mc_data[true_id_branch])
    
    # Apply truth matching cut (typically Bu_TRUEID > 0)
    truth_matched_mask = truth_match > 0
    mass_data_truth_matched = mass_data[truth_matched_mask]
    
    # Apply track type selection if specified
    if track_type is not None and track_branch is not None:
        track_data = np.array(mc_data[track_branch])
        track_mask = track_data == track_type
        mass_data_truth_matched = mass_data_truth_matched[track_mask[truth_matched_mask]]
        sample_name = f"MC_{track_type}"
    else:
        sample_name = "MC_All"
    
    print(f"Extracting shape for {sample_name}")
    print(f"Number of truth-matched events: {len(mass_data_truth_matched)}")
    
    # Define fit range and bins
    mass_min, mass_max = 5200, 5400  # MeV/c^2
    n_bins = 80
    mass_bins = np.linspace(mass_min, mass_max, n_bins+1)
    
    # Create ROOT canvas
    canvas = TCanvas(f"canvas_{sample_name}", f"B+ Mass Fit ({sample_name})", 900, 700)
    canvas.SetLeftMargin(0.12)
    canvas.SetBottomMargin(0.12)
    
    # Create and fill histogram
    hist_name = f"h_mass_{sample_name}"
    hist = create_root_histogram(
        mass_data_truth_matched, mass_bins, mass_min, mass_max, 
        name=hist_name, 
        title="B^{+} #rightarrow #bar{#Lambda}^{0}pK^{+}K^{-}"
    )
    
    # Decorate histogram
    hist.SetMarkerStyle(20)
    hist.SetMarkerSize(0.8)
    hist.SetLineColor(ROOT.kBlack)
    hist.GetXaxis().SetTitle("m(#bar{#Lambda}^{0}pK^{+}K^{-}) [MeV/c^{2}]")
    hist.GetYaxis().SetTitle("Candidates / (%.1f MeV/c^{2})" % ((mass_max - mass_min)/n_bins))
    hist.GetYaxis().SetTitleOffset(1.5)
    hist.Draw("E")
    
    # Fit with Crystal Ball function
    nominal_mass = 5279.0  # B+ PDG mass in MeV/c^2
    initial_sigma = 15.0   # Initial guess for resolution
    
    # RooFit variables
    mass = RooRealVar("mass", "B^{+} mass [MeV/c^{2}]", mass_min, mass_max)
    
    # Create RooDataHist
    data_hist = RooDataHist("data_hist", "B+ Mass Data", RooArgList(mass), hist)
    
    # Crystal Ball parameters
    mean = RooRealVar("mean", "Mean", nominal_mass, nominal_mass-30, nominal_mass+30)
    sigma = RooRealVar("sigma", "Sigma", initial_sigma, 5, 30)
    alpha = RooRealVar("alpha", "Alpha", 1.5, 0.5, 5.0)  # Tail parameter
    n = RooRealVar("n", "n", 2, 0.5, 10.0)  # Power law parameter
    
    # Create Crystal Ball PDF
    cb = RooCBShape("cb", "Crystal Ball", mass, mean, sigma, alpha, n)
    
    # Perform the fit
    fit_result = cb.fitTo(
        data_hist, 
        RooFit.Save(), 
        RooFit.PrintLevel(1),  # Minimal output
        RooFit.Range(mass_min, mass_max)
    )
    
    # Plot the result
    frame = mass.frame(RooFit.Title(f"B+ Mass Shape - {sample_name}"))
    data_hist.plotOn(frame, RooFit.Name("data"))
    cb.plotOn(frame, RooFit.LineColor(ROOT.kBlue), RooFit.Name("cb"))
    
    # Add fit parameters to plot
    cb.paramOn(frame, RooFit.Layout(0.65, 0.90, 0.90))
    frame.Draw()
    
    # Calculate chi-squared
    chi2 = frame.chiSquare("cb", "data", fit_result.floatParsFinal().getSize())
    
    # Add text with fit results
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.15, 0.85, f"Mean = {mean.getVal():.2f} #pm {mean.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.15, 0.80, f"#sigma = {sigma.getVal():.2f} #pm {sigma.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.15, 0.75, f"#alpha = {alpha.getVal():.2f} #pm {alpha.getError():.2f}")
    latex.DrawLatex(0.15, 0.70, f"n = {n.getVal():.2f} #pm {n.getError():.2f}")
    latex.DrawLatex(0.15, 0.65, f"#chi^{{2}}/ndf = {chi2:.3f}")
    
    # Save the plot
    canvas.SaveAs(f"{output_dir}/b_mass_shape_{sample_name}.png")
    canvas.SaveAs(f"{output_dir}/b_mass_shape_{sample_name}.pdf")
    
    # Store results in a dictionary
    shape_params = {
        "model": "crystal_ball",
        "mean": {"value": mean.getVal(), "error": mean.getError()},
        "sigma": {"value": sigma.getVal(), "error": sigma.getError()},
        "alpha": {"value": alpha.getVal(), "error": alpha.getError()},
        "n": {"value": n.getVal(), "error": n.getError()},
        "chi2_ndf": chi2,
        "events": len(mass_data_truth_matched)
    }
    
    # Save parameters to JSON file
    with open(f"{output_dir}/b_mass_shape_params_{sample_name}.json", 'w') as f:
        json.dump(shape_params, f, indent=4)
    
    print(f"Shape parameters for {sample_name}:")
    print(f"  Mean = {mean.getVal():.2f} ± {mean.getError():.2f} MeV/c²")
    print(f"  Sigma = {sigma.getVal():.2f} ± {sigma.getError():.2f} MeV/c²")
    print(f"  Alpha = {alpha.getVal():.2f} ± {alpha.getError():.2f}")
    print(f"  n = {n.getVal():.2f} ± {n.getError():.2f}")
    print(f"  χ²/ndf = {chi2:.3f}")
    
    return shape_params

def extract_signal_shape_double_cb(mc_data, true_id_branch='Bu_TRUEID', mass_branch='Bu_MM', 
                     track_type=None, track_branch=None, output_dir="./"):
    """
    Extract signal shape from MC using Double Crystal Ball (better for asymmetric tails)
    
    Parameters similar to extract_signal_shape_crystal_ball
    """
    # Create output directory
    create_output_dir(output_dir)
    
    # Get mass data and apply truth matching
    mass_data = np.array(mc_data[mass_branch])
    truth_match = np.array(mc_data[true_id_branch])
    
    # Apply truth matching cut
    truth_matched_mask = truth_match > 0
    mass_data_truth_matched = mass_data[truth_matched_mask]
    
    # Apply track type selection if specified
    if track_type is not None and track_branch is not None:
        track_data = np.array(mc_data[track_branch])
        track_mask = track_data == track_type
        mass_data_truth_matched = mass_data_truth_matched[track_mask[truth_matched_mask]]
        sample_name = f"MC_{track_type}_DCB"
    else:
        sample_name = "MC_All_DCB"
    
    print(f"Extracting shape for {sample_name} using Double Crystal Ball")
    print(f"Number of truth-matched events: {len(mass_data_truth_matched)}")
    
    # Define fit range and bins
    mass_min, mass_max = 5200, 5400  # MeV/c^2
    n_bins = 80
    mass_bins = np.linspace(mass_min, mass_max, n_bins+1)
    
    # Create ROOT canvas
    canvas = TCanvas(f"canvas_{sample_name}", f"B+ Mass Fit ({sample_name})", 900, 700)
    canvas.SetLeftMargin(0.12)
    canvas.SetBottomMargin(0.12)
    
    # Create and fill histogram
    hist_name = f"h_mass_{sample_name}"
    hist = create_root_histogram(
        mass_data_truth_matched, mass_bins, mass_min, mass_max, 
        name=hist_name, 
        title="B^{+} #rightarrow #bar{#Lambda}^{0}pK^{+}K^{-}"
    )
    
    # Decorate histogram
    hist.SetMarkerStyle(20)
    hist.SetMarkerSize(0.8)
    hist.SetLineColor(ROOT.kBlack)
    hist.GetXaxis().SetTitle("m(#bar{#Lambda}^{0}pK^{+}K^{-}) [MeV/c^{2}]")
    hist.GetYaxis().SetTitle("Candidates / (%.1f MeV/c^{2})" % ((mass_max - mass_min)/n_bins))
    hist.GetYaxis().SetTitleOffset(1.5)
    hist.Draw("E")
    
    # RooFit variables
    mass = RooRealVar("mass", "B^{+} mass [MeV/c^{2}]", mass_min, mass_max)
    
    # Create RooDataHist
    data_hist = RooDataHist("data_hist", "B+ Mass Data", RooArgList(mass), hist)
    
    # Double Crystal Ball parameters
    mean = RooRealVar("mean", "Mean", 5279, 5260, 5290)
    sigma = RooRealVar("sigma", "Sigma", 15, 5, 30)
    
    # Left tail (low mass side)
    alpha1 = RooRealVar("alpha1", "Alpha1", 1.5, 0.5, 5.0)
    n1 = RooRealVar("n1", "n1", 2, 0.5, 10.0)
    
    # Right tail (high mass side)
    alpha2 = RooRealVar("alpha2", "Alpha2", 1.5, 0.5, 5.0)
    n2 = RooRealVar("n2", "n2", 2, 0.5, 10.0)
    
    # Create Double Crystal Ball PDF using our custom function
    dcb = create_double_crystal_ball("dcb", "Double Crystal Ball", mass, mean, sigma, alpha1, n1, alpha2, n2)
    
    # Perform the fit
    fit_result = dcb.fitTo(
        data_hist, 
        RooFit.Save(), 
        RooFit.PrintLevel(1),
        RooFit.Range(mass_min, mass_max)
    )
    
    # Plot the result
    frame = mass.frame(RooFit.Title(f"B+ Mass Shape - {sample_name}"))
    data_hist.plotOn(frame, RooFit.Name("data"))
    dcb.plotOn(frame, RooFit.LineColor(ROOT.kBlue), RooFit.Name("dcb"))
    
    # Draw the frame
    frame.Draw()
    
    # Calculate chi-squared
    chi2 = frame.chiSquare("dcb", "data", fit_result.floatParsFinal().getSize())
    
    # Add text with fit results
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.15, 0.85, f"Mean = {mean.getVal():.2f} #pm {mean.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.15, 0.80, f"#sigma = {sigma.getVal():.2f} #pm {sigma.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.15, 0.75, f"#alpha_{{1}} = {alpha1.getVal():.2f} #pm {alpha1.getError():.2f}")
    latex.DrawLatex(0.15, 0.70, f"n_{{1}} = {n1.getVal():.2f} #pm {n1.getError():.2f}")
    latex.DrawLatex(0.15, 0.65, f"#alpha_{{2}} = {alpha2.getVal():.2f} #pm {alpha2.getError():.2f}")
    latex.DrawLatex(0.15, 0.60, f"n_{{2}} = {n2.getVal():.2f} #pm {n2.getError():.2f}")
    latex.DrawLatex(0.15, 0.55, f"#chi^{{2}}/ndf = {chi2:.3f}")
    
    # Save the plot
    canvas.SaveAs(f"{output_dir}/b_mass_shape_{sample_name}.png")
    canvas.SaveAs(f"{output_dir}/b_mass_shape_{sample_name}.pdf")
    
    # Store results in a dictionary
    shape_params = {
        "model": "double_crystal_ball",
        "mean": {"value": mean.getVal(), "error": mean.getError()},
        "sigma": {"value": sigma.getVal(), "error": sigma.getError()},
        "alpha1": {"value": alpha1.getVal(), "error": alpha1.getError()},
        "n1": {"value": n1.getVal(), "error": n1.getError()},
        "alpha2": {"value": alpha2.getVal(), "error": alpha2.getError()},
        "n2": {"value": n2.getVal(), "error": n2.getError()},
        "chi2_ndf": chi2,
        "events": len(mass_data_truth_matched)
    }
    
    # Save parameters to JSON file
    with open(f"{output_dir}/b_mass_shape_params_{sample_name}.json", 'w') as f:
        json.dump(shape_params, f, indent=4)
    
    print(f"Double Crystal Ball shape parameters for {sample_name}:")
    print(f"  Mean = {mean.getVal():.2f} ± {mean.getError():.2f} MeV/c²")
    print(f"  Sigma = {sigma.getVal():.2f} ± {sigma.getError():.2f} MeV/c²")
    print(f"  Alpha1 = {alpha1.getVal():.2f} ± {alpha1.getError():.2f}")
    print(f"  n1 = {n1.getVal():.2f} ± {n1.getError():.2f}")
    print(f"  Alpha2 = {alpha2.getVal():.2f} ± {alpha2.getError():.2f}")
    print(f"  n2 = {n2.getVal():.2f} ± {n2.getError():.2f}")
    print(f"  χ²/ndf = {chi2:.3f}")
    
    return shape_params

def extract_signal_shape_and_resolution(mc_data, true_id_branch='Bu_TRUEID', mass_branch='Bu_MM', 
                          track_types=None, track_branch=None, output_dir="./"):
    """
    Extract signal shapes for different track types and create a summary
    
    Parameters:
    -----------
    mc_data : awkward array or dict
        MC data containing mass and truth information
    true_id_branch : str
        Name of the branch containing truth matching information
    mass_branch : str
        Name of the branch containing the B mass
    track_types : list of str, optional
        Track types to process separately (e.g., ['LL', 'DD'])
    track_branch : str, optional
        Name of the branch containing track type information
    output_dir : str
        Directory to save output plots
        
    Returns:
    --------
    dict of shape parameters for all processed categories
    """
    # Create output directory
    create_output_dir(output_dir)
    
    # Dictionary to store all results
    all_results = {}
    
    # First, fit all events combined
    print("\n=== Fitting all events combined ===")
    cb_results_all = extract_signal_shape_crystal_ball(
        mc_data, 
        true_id_branch=true_id_branch,
        mass_branch=mass_branch,
        output_dir=output_dir
    )
    
    dcb_results_all = extract_signal_shape_double_cb(
        mc_data, 
        true_id_branch=true_id_branch,
        mass_branch=mass_branch,
        output_dir=output_dir
    )
    
    all_results["All"] = {
        "crystal_ball": cb_results_all,
        "double_crystal_ball": dcb_results_all
    }
    
    # Then fit each track type separately if requested
    if track_types is not None and track_branch is not None:
        for track_type in track_types:
            print(f"\n=== Fitting {track_type} events ===")
            
            cb_results = extract_signal_shape_crystal_ball(
                mc_data,
                true_id_branch=true_id_branch,
                mass_branch=mass_branch,
                track_type=track_type,
                track_branch=track_branch,
                output_dir=output_dir
            )
            
            dcb_results = extract_signal_shape_double_cb(
                mc_data,
                true_id_branch=true_id_branch,
                mass_branch=mass_branch,
                track_type=track_type,
                track_branch=track_branch,
                output_dir=output_dir
            )
            
            all_results[track_type] = {
                "crystal_ball": cb_results,
                "double_crystal_ball": dcb_results
            }
    
    # Create summary of results
    summary = []
    
    for category, models in all_results.items():
        for model_name, params in models.items():
            summary_entry = {
                "Category": category,
                "Model": model_name,
                "Events": params["events"],
                "Mean (MeV/c²)": f"{params['mean']['value']:.2f} ± {params['mean']['error']:.2f}",
                "Resolution (MeV/c²)": f"{params['sigma']['value']:.2f} ± {params['sigma']['error']:.2f}",
                "χ²/ndf": params["chi2_ndf"]
            }
            summary.append(summary_entry)
    
    # Print summary table
    print("\n=== Summary of Signal Shape Parameters ===")
    print(f"{'Category':<10} {'Model':<20} {'Events':<10} {'Mean (MeV/c²)':<20} {'Resolution (MeV/c²)':<20} {'χ²/ndf':<10}")
    print("-" * 90)
    
    for entry in summary:
        print(f"{entry['Category']:<10} {entry['Model']:<20} {entry['Events']:<10} {entry['Mean (MeV/c²)']:<20} {entry['Resolution (MeV/c²)']:<20} {entry['χ²/ndf']:<10.3f}")
    
    # Save all results to a pickle file for later use
    with open(f"{output_dir}/all_shape_parameters.pkl", 'wb') as f:
        pickle.dump(all_results, f)
    
    # Also save as JSON for better human readability
    # Convert numpy values to Python native types
    json_results = {}
    for category, models in all_results.items():
        json_results[category] = {}
        for model_name, params in models.items():
            json_results[category][model_name] = {}
            for param_name, param_value in params.items():
                if isinstance(param_value, dict):
                    json_results[category][model_name][param_name] = {
                        "value": float(param_value["value"]),
                        "error": float(param_value["error"])
                    }
                else:
                    json_results[category][model_name][param_name] = float(param_value) if isinstance(param_value, (np.number, float, int)) else param_value
    
    with open(f"{output_dir}/all_shape_parameters.json", 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print(f"\nAll results saved to {output_dir}/all_shape_parameters.json and .pkl")
    
    return all_results

def create_data_fit_model_with_mc_shapes(shape_params, mass_min=5200, mass_max=5400):
    """
    Create a RooFit model for fitting data using the MC shape parameters
    
    Parameters:
    -----------
    shape_params : dict
        Dictionary of shape parameters from extract_signal_shape_*
    mass_min, mass_max : float
        Mass range for the fit
        
    Returns:
    --------
    tuple of (mass, signal_pdf, background_pdf, nsig, nbkg, model)
    """
    # Create mass variable
    mass = RooRealVar("mass", "B^{+} mass [MeV/c^{2}]", mass_min, mass_max)
    
    # Create signal PDF based on MC shape
    model_type = shape_params["model"]
    
    if model_type == "crystal_ball":
        # Get parameters from MC fit
        mean_val = shape_params["mean"]["value"]
        sigma_val = shape_params["sigma"]["value"]
        alpha_val = shape_params["alpha"]["value"]
        n_val = shape_params["n"]["value"]
        
        # Create RooFit variables with MC values as starting points
        # Allow mean and sigma to float within constraints
        # Fix tail parameters (alpha, n) to MC values
        mean = RooRealVar("mean", "Mean", mean_val, mean_val - 5, mean_val + 5)
        sigma = RooRealVar("sigma", "Sigma", sigma_val, sigma_val * 0.5, sigma_val * 1.5)
        alpha = RooRealVar("alpha", "Alpha", alpha_val)
        n = RooRealVar("n", "n", n_val)
        
        # Fix tail parameters to MC values
        alpha.setConstant(True)
        n.setConstant(True)
        
        # Create Crystal Ball signal PDF
        signal_pdf = RooCBShape("signal", "Crystal Ball Signal", mass, mean, sigma, alpha, n)
        
    elif model_type == "double_crystal_ball":
        # Get parameters from MC fit
        mean_val = shape_params["mean"]["value"]
        sigma_val = shape_params["sigma"]["value"]
        alpha1_val = shape_params["alpha1"]["value"]
        n1_val = shape_params["n1"]["value"]
        alpha2_val = shape_params["alpha2"]["value"]
        n2_val = shape_params["n2"]["value"]
        
        # Create RooFit variables
        mean = RooRealVar("mean", "Mean", mean_val, mean_val - 5, mean_val + 5)
        sigma = RooRealVar("sigma", "Sigma", sigma_val, sigma_val * 0.5, sigma_val * 1.5)
        alpha1 = RooRealVar("alpha1", "Alpha1", alpha1_val)
        n1 = RooRealVar("n1", "n1", n1_val)
        alpha2 = RooRealVar("alpha2", "Alpha2", alpha2_val)
        n2 = RooRealVar("n2", "n2", n2_val)
        
        # Fix tail parameters to MC values
        alpha1.setConstant(True)
        n1.setConstant(True)
        alpha2.setConstant(True)
        n2.setConstant(True)
        
        # Create Double Crystal Ball signal PDF using our custom function
        signal_pdf = create_double_crystal_ball("signal", "Double Crystal Ball Signal", mass, mean, sigma, 
                                             alpha1, n1, alpha2, n2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create background PDF - exponential is a good default for B physics
    slope = RooRealVar("slope", "Background Slope", -0.001, -0.1, 0.1)
    background_pdf = ROOT.RooExponential("background", "Background", mass, slope)
    
    # Create yields
    nsig = RooRealVar("nsig", "Signal Yield", 1000, 0, 1e6)
    nbkg = RooRealVar("nbkg", "Background Yield", 1000, 0, 1e6)
    
    # Create the combined model
    model = RooAddPdf("model", "Signal + Background", 
                     RooArgList(signal_pdf, background_pdf), 
                     RooArgList(nsig, nbkg))
    
    return (mass, signal_pdf, background_pdf, nsig, nbkg, model)