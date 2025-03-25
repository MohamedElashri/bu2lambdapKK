import numpy as np
import ROOT
from ROOT import TH1F, TCanvas, TF1, RooRealVar, RooGaussian, RooDataHist, RooCBShape
from ROOT import RooFit, RooPlot, RooArgList, RooArgSet, TLatex, gStyle
from ROOT import RooAddPdf, RooFFTConvPdf, RooDataSet, RooFormulaVar, RooChebychev
from array import array
import os
import json
import pickle

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

def extract_signal_shape_gaussian(mc_data, true_id_branch='Bu_TRUEID', mass_branch='Bu_MM', output_dir="./"):
    """
    Extract signal shape from MC using a Gaussian
    
    Parameters:
    -----------
    mc_data : awkward array or dict
        MC data containing mass and truth information
    true_id_branch : str
        Name of the branch containing truth matching information
    mass_branch : str
        Name of the branch containing the B mass
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
    sample_name = "MC_Gaussian"
    
    print(f"Extracting shape using Gaussian model")
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
    
    # Gaussian parameters
    nominal_mass = 5279.0  # B+ PDG mass in MeV/c^2
    initial_sigma = 15.0   # Initial guess for resolution
    
    mean = RooRealVar("mean", "Mean", nominal_mass, nominal_mass-30, nominal_mass+30)
    sigma = RooRealVar("sigma", "Sigma", initial_sigma, 5, 30)
    
    # Create Gaussian PDF
    gauss = RooGaussian("gauss", "Gaussian", mass, mean, sigma)
    
    # Perform the fit
    fit_result = gauss.fitTo(
        data_hist, 
        RooFit.Save(), 
        RooFit.PrintLevel(1),  # Minimal output
        RooFit.Range(mass_min, mass_max)
    )
    
    # Plot the result
    frame = mass.frame(RooFit.Title("B+ Mass Shape - Gaussian Model"))
    data_hist.plotOn(frame, RooFit.Name("data"))
    gauss.plotOn(frame, RooFit.LineColor(ROOT.kBlue), RooFit.Name("gauss"))
    
    # Add fit parameters to plot
    gauss.paramOn(frame, RooFit.Layout(0.65, 0.90, 0.90))
    frame.Draw()
    
    # Calculate chi-squared
    chi2 = frame.chiSquare("gauss", "data", fit_result.floatParsFinal().getSize())
    
    # Add text with fit results
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.15, 0.85, f"Mean = {mean.getVal():.2f} #pm {mean.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.15, 0.80, f"#sigma = {sigma.getVal():.2f} #pm {sigma.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.15, 0.75, f"#chi^{{2}}/ndf = {chi2:.3f}")
    
    # Save the plot
    canvas.SaveAs(f"{output_dir}/b_mass_shape_{sample_name}.pdf")
    
    # Store results in a dictionary
    shape_params = {
        "model": "gaussian",
        "mean": {"value": mean.getVal(), "error": mean.getError()},
        "sigma": {"value": sigma.getVal(), "error": sigma.getError()},
        "chi2_ndf": chi2,
        "events": len(mass_data_truth_matched)
    }
    
    # Save parameters to JSON file
    with open(f"{output_dir}/b_mass_shape_params_{sample_name}.json", 'w') as f:
        json.dump(shape_params, f, indent=4)
    
    print(f"Gaussian shape parameters:")
    print(f"  Mean = {mean.getVal():.2f} ± {mean.getError():.2f} MeV/c²")
    print(f"  Sigma = {sigma.getVal():.2f} ± {sigma.getError():.2f} MeV/c²")
    print(f"  χ²/ndf = {chi2:.3f}")
    
    return shape_params

def extract_signal_shape_double_gaussian(mc_data, true_id_branch='Bu_TRUEID', mass_branch='Bu_MM', output_dir="./"):
    """
    Extract signal shape from MC using a Double Gaussian
    
    Parameters:
    -----------
    mc_data : awkward array or dict
        MC data containing mass and truth information
    true_id_branch : str
        Name of the branch containing truth matching information
    mass_branch : str
        Name of the branch containing the B mass
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
    sample_name = "MC_DoubleGaussian"
    
    print(f"Extracting shape using Double Gaussian model")
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
    
    # Double Gaussian parameters
    nominal_mass = 5279.0  # B+ PDG mass in MeV/c^2
    
    # Parameters for narrower Gaussian (core)
    mean = RooRealVar("mean", "Mean", nominal_mass, nominal_mass-30, nominal_mass+30)
    sigma1 = RooRealVar("sigma1", "Sigma1", 12.0, 5.0, 25.0)
    gauss1 = RooGaussian("gauss1", "Core Gaussian", mass, mean, sigma1)
    
    # Parameters for wider Gaussian (tails)
    sigma2 = RooRealVar("sigma2", "Sigma2", 20.0, 15.0, 40.0)
    gauss2 = RooGaussian("gauss2", "Tail Gaussian", mass, mean, sigma2)
    
    # Fraction of events in the core Gaussian
    frac = RooRealVar("frac", "Fraction", 0.7, 0.0, 1.0)
    
    # Combine the two Gaussians
    double_gauss = RooAddPdf("double_gauss", "Double Gaussian", RooArgList(gauss1, gauss2), RooArgList(frac))
    
    # Perform the fit
    fit_result = double_gauss.fitTo(
        data_hist, 
        RooFit.Save(), 
        RooFit.PrintLevel(1),
        RooFit.Range(mass_min, mass_max)
    )
    
    # Plot the result
    frame = mass.frame(RooFit.Title("B+ Mass Shape - Double Gaussian Model"))
    data_hist.plotOn(frame, RooFit.Name("data"))
    double_gauss.plotOn(frame, RooFit.LineColor(ROOT.kBlue), RooFit.Name("double_gauss"))
    
    # Plot individual components
    double_gauss.plotOn(
        frame, 
        RooFit.Components("gauss1"), 
        RooFit.LineColor(ROOT.kGreen+2), 
        RooFit.LineStyle(ROOT.kDashed),
        RooFit.Name("gauss1")
    )
    double_gauss.plotOn(
        frame, 
        RooFit.Components("gauss2"), 
        RooFit.LineColor(ROOT.kRed), 
        RooFit.LineStyle(ROOT.kDashed),
        RooFit.Name("gauss2")
    )
    
    # Draw the frame
    frame.Draw()
    
    # Calculate chi-squared
    chi2 = frame.chiSquare("double_gauss", "data", fit_result.floatParsFinal().getSize())
    
    # Calculate effective sigma (weighted average of the two sigmas)
    effective_sigma = np.sqrt(frac.getVal() * sigma1.getVal()**2 + (1-frac.getVal()) * sigma2.getVal()**2)
    effective_sigma_err = 0.5 * np.sqrt(
        (frac.getVal() * sigma1.getVal() * sigma1.getError())**2 + 
        ((1-frac.getVal()) * sigma2.getVal() * sigma2.getError())**2 +
        ((sigma1.getVal()**2 - sigma2.getVal()**2) * frac.getError())**2
    ) / effective_sigma
    
    # Add text with fit results
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.15, 0.85, f"Mean = {mean.getVal():.2f} #pm {mean.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.15, 0.80, f"#sigma_{{1}} = {sigma1.getVal():.2f} #pm {sigma1.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.15, 0.75, f"#sigma_{{2}} = {sigma2.getVal():.2f} #pm {sigma2.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.15, 0.70, f"Fraction = {frac.getVal():.2f} #pm {frac.getError():.2f}")
    latex.DrawLatex(0.15, 0.65, f"#sigma_{{eff}} = {effective_sigma:.2f} #pm {effective_sigma_err:.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.15, 0.60, f"#chi^{{2}}/ndf = {chi2:.3f}")
    
    # Add legend
    legend = ROOT.TLegend(0.65, 0.70, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.AddEntry(frame.findObject("data"), "Data", "P")
    legend.AddEntry(frame.findObject("double_gauss"), "Double Gaussian", "L")
    legend.AddEntry(frame.findObject("gauss1"), "Core Gaussian", "L")
    legend.AddEntry(frame.findObject("gauss2"), "Tail Gaussian", "L")
    legend.Draw()
    
    # Save the plot
    canvas.SaveAs(f"{output_dir}/b_mass_shape_{sample_name}.pdf")
    
    # Store results in a dictionary
    shape_params = {
        "model": "double_gaussian",
        "mean": {"value": mean.getVal(), "error": mean.getError()},
        "sigma1": {"value": sigma1.getVal(), "error": sigma1.getError()},
        "sigma2": {"value": sigma2.getVal(), "error": sigma2.getError()},
        "fraction": {"value": frac.getVal(), "error": frac.getError()},
        "effective_sigma": {"value": effective_sigma, "error": effective_sigma_err},
        "chi2_ndf": chi2,
        "events": len(mass_data_truth_matched)
    }
    
    # Save parameters to JSON file
    with open(f"{output_dir}/b_mass_shape_params_{sample_name}.json", 'w') as f:
        json.dump(shape_params, f, indent=4)
    
    print(f"Double Gaussian shape parameters:")
    print(f"  Mean = {mean.getVal():.2f} ± {mean.getError():.2f} MeV/c²")
    print(f"  Sigma1 = {sigma1.getVal():.2f} ± {sigma1.getError():.2f} MeV/c²")
    print(f"  Sigma2 = {sigma2.getVal():.2f} ± {sigma2.getError():.2f} MeV/c²")
    print(f"  Fraction = {frac.getVal():.2f} ± {frac.getError():.2f}")
    print(f"  Effective Sigma = {effective_sigma:.2f} ± {effective_sigma_err:.2f} MeV/c²")
    print(f"  χ²/ndf = {chi2:.3f}")
    
    return shape_params

def extract_signal_shape_crystal_ball(mc_data, true_id_branch='Bu_TRUEID', mass_branch='Bu_MM', output_dir="./"):
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
    sample_name = "MC_CrystalBall"
    
    print(f"Extracting shape using Crystal Ball model")
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
    
    # Crystal Ball parameters
    nominal_mass = 5279.0  # B+ PDG mass in MeV/c^2
    initial_sigma = 15.0   # Initial guess for resolution
    
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
    frame = mass.frame(RooFit.Title("B+ Mass Shape - Crystal Ball Model"))
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
    
    print(f"Crystal Ball shape parameters:")
    print(f"  Mean = {mean.getVal():.2f} ± {mean.getError():.2f} MeV/c²")
    print(f"  Sigma = {sigma.getVal():.2f} ± {sigma.getError():.2f} MeV/c²")
    print(f"  Alpha = {alpha.getVal():.2f} ± {alpha.getError():.2f}")
    print(f"  n = {n.getVal():.2f} ± {n.getError():.2f}")
    print(f"  χ²/ndf = {chi2:.3f}")
    
    return shape_params

def extract_signal_shapes(mc_data, true_id_branch='Bu_TRUEID', mass_branch='Bu_MM', output_dir="./"):
    """
    Extract signal shapes using three different models and compare them
    
    Parameters:
    -----------
    mc_data : awkward array or dict
        MC data containing mass and truth information
    true_id_branch : str
        Name of the branch containing truth matching information
    mass_branch : str
        Name of the branch containing the B mass
    output_dir : str
        Directory to save output plots
        
    Returns:
    --------
    dict of shape parameters for all models
    """
    # Create output directory
    create_output_dir(output_dir)
    
    # Dictionary to store all results
    all_results = {}
    
    # Extract shapes using different models
    print("\n=== Fitting with Gaussian model ===")
    gauss_results = extract_signal_shape_gaussian(
        mc_data, 
        true_id_branch=true_id_branch,
        mass_branch=mass_branch,
        output_dir=output_dir
    )
    
    print("\n=== Fitting with Double Gaussian model ===")
    double_gauss_results = extract_signal_shape_double_gaussian(
        mc_data, 
        true_id_branch=true_id_branch,
        mass_branch=mass_branch,
        output_dir=output_dir
    )
    
    print("\n=== Fitting with Crystal Ball model ===")
    cb_results = extract_signal_shape_crystal_ball(
        mc_data, 
        true_id_branch=true_id_branch,
        mass_branch=mass_branch,
        output_dir=output_dir
    )
    
    # Store all results
    all_results = {
        "gaussian": gauss_results,
        "double_gaussian": double_gauss_results,
        "crystal_ball": cb_results
    }
    
    # Create summary of results
    summary = []
    
    for model_name, params in all_results.items():
        if model_name == "gaussian":
            resolution = params["sigma"]["value"]
            resolution_err = params["sigma"]["error"]
        elif model_name == "double_gaussian":
            resolution = params["effective_sigma"]["value"]
            resolution_err = params["effective_sigma"]["error"]
        elif model_name == "crystal_ball":
            resolution = params["sigma"]["value"]
            resolution_err = params["sigma"]["error"]
        
        summary_entry = {
            "Model": model_name,
            "Events": params["events"],
            "Mean (MeV/c²)": f"{params['mean']['value']:.2f} ± {params['mean']['error']:.2f}",
            "Resolution (MeV/c²)": f"{resolution:.2f} ± {resolution_err:.2f}",
            "χ²/ndf": params["chi2_ndf"]
        }
        summary.append(summary_entry)
    
    # Print summary table
    print("\n=== Summary of Signal Shape Parameters ===")
    print(f"{'Model':<20} {'Events':<10} {'Mean (MeV/c²)':<20} {'Resolution (MeV/c²)':<20} {'χ²/ndf':<10}")
    print("-" * 80)
    
    for entry in summary:
        print(f"{entry['Model']:<20} {entry['Events']:<10} {entry['Mean (MeV/c²)']:<20} {entry['Resolution (MeV/c²)']:<20} {entry['χ²/ndf']:<10.3f}")
    
    # Save all results to a pickle file for later use
    with open(f"{output_dir}/all_shape_parameters.pkl", 'wb') as f:
        pickle.dump(all_results, f)
    
    # Also save as JSON for better human readability
    # Convert numpy values to Python native types
    json_results = {}
    for model_name, params in all_results.items():
        json_results[model_name] = {}
        for param_name, param_value in params.items():
            if isinstance(param_value, dict):
                json_results[model_name][param_name] = {
                    "value": float(param_value["value"]),
                    "error": float(param_value["error"])
                }
            else:
                json_results[model_name][param_name] = float(param_value) if isinstance(param_value, (np.number, float, int)) else param_value
    
    with open(f"{output_dir}/all_shape_parameters.json", 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print(f"\nAll results saved to {output_dir}/all_shape_parameters.json and .pkl")
    
    # Create a comparison plot of all models
    create_model_comparison_plot(mc_data, true_id_branch, mass_branch, all_results, output_dir)
    
    return all_results

def create_model_comparison_plot(mc_data, true_id_branch, mass_branch, model_results, output_dir):
    """
    Create a plot comparing all three signal shape models on the same data
    
    Parameters:
    -----------
    mc_data : awkward array or dict
        MC data containing mass and truth information
    true_id_branch : str
        Name of the branch containing truth matching information
    mass_branch : str
        Name of the branch containing the B mass
    model_results : dict
        Dictionary containing the fit results for each model
    output_dir : str
        Directory to save output plots
    """
    # Create output directory
    create_output_dir(output_dir)
    
    # Get mass data and apply truth matching
    mass_data = np.array(mc_data[mass_branch])
    truth_match = np.array(mc_data[true_id_branch])
    
    # Apply truth matching cut (typically Bu_TRUEID > 0)
    truth_matched_mask = truth_match > 0
    mass_data_truth_matched = mass_data[truth_matched_mask]
    
    # Define fit range and bins
    mass_min, mass_max = 5200, 5400  # MeV/c^2
    n_bins = 80
    mass_bins = np.linspace(mass_min, mass_max, n_bins+1)
    
    # Create ROOT canvas
    canvas = TCanvas("canvas_comparison", "B+ Mass Model Comparison", 900, 700)
    canvas.SetLeftMargin(0.12)
    canvas.SetBottomMargin(0.12)
    
    # Create and fill histogram
    hist_name = "h_mass_comparison"
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
    
    # Create frame for plotting
    frame = mass.frame(RooFit.Title("B+ Mass Shape Models Comparison"))
    data_hist.plotOn(frame, RooFit.Name("data"))
    
    # Plot each model
    # 1. Gaussian
    mean_g = RooRealVar("mean_g", "Mean", model_results["gaussian"]["mean"]["value"])
    sigma_g = RooRealVar("sigma_g", "Sigma", model_results["gaussian"]["sigma"]["value"])
    gauss = RooGaussian("gauss", "Gaussian", mass, mean_g, sigma_g)
    
    # 2. Double Gaussian
    mean_dg = RooRealVar("mean_dg", "Mean", model_results["double_gaussian"]["mean"]["value"])
    sigma1_dg = RooRealVar("sigma1_dg", "Sigma1", model_results["double_gaussian"]["sigma1"]["value"])
    sigma2_dg = RooRealVar("sigma2_dg", "Sigma2", model_results["double_gaussian"]["sigma2"]["value"])
    frac_dg = RooRealVar("frac_dg", "Fraction", model_results["double_gaussian"]["fraction"]["value"])
    
    gauss1_dg = RooGaussian("gauss1_dg", "Core Gaussian", mass, mean_dg, sigma1_dg)
    gauss2_dg = RooGaussian("gauss2_dg", "Tail Gaussian", mass, mean_dg, sigma2_dg)
    double_gauss = RooAddPdf("double_gauss", "Double Gaussian", RooArgList(gauss1_dg, gauss2_dg), RooArgList(frac_dg))
    
    # 3. Crystal Ball
    mean_cb = RooRealVar("mean_cb", "Mean", model_results["crystal_ball"]["mean"]["value"])
    sigma_cb = RooRealVar("sigma_cb", "Sigma", model_results["crystal_ball"]["sigma"]["value"])
    alpha_cb = RooRealVar("alpha_cb", "Alpha", model_results["crystal_ball"]["alpha"]["value"])
    n_cb = RooRealVar("n_cb", "n", model_results["crystal_ball"]["n"]["value"])
    cb = RooCBShape("cb", "Crystal Ball", mass, mean_cb, sigma_cb, alpha_cb, n_cb)
    
    # Normalize all PDFs to same area
    norm_factor = hist.Integral() * (mass_max - mass_min) / n_bins
    
    # Plot each model
    gauss.plotOn(frame, RooFit.LineColor(ROOT.kGreen+2), RooFit.Name("gaussian"), 
                 RooFit.Normalization(norm_factor, ROOT.RooAbsReal.NumEvent))
    
    double_gauss.plotOn(frame, RooFit.LineColor(ROOT.kBlue), RooFit.Name("double_gaussian"),
                        RooFit.Normalization(norm_factor, ROOT.RooAbsReal.NumEvent))
    
    cb.plotOn(frame, RooFit.LineColor(ROOT.kRed), RooFit.Name("crystal_ball"),
              RooFit.Normalization(norm_factor, ROOT.RooAbsReal.NumEvent))
    
    # Draw the frame
    frame.Draw()
    
    # Add legend
    legend = ROOT.TLegend(0.65, 0.70, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.AddEntry(frame.findObject("data"), "MC Data", "P")
    legend.AddEntry(frame.findObject("gaussian"), "Gaussian", "L")
    legend.AddEntry(frame.findObject("double_gaussian"), "Double Gaussian", "L")
    legend.AddEntry(frame.findObject("crystal_ball"), "Crystal Ball", "L")
    legend.Draw()
    
    # Add text with fit quality comparison
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.15, 0.85, "Model Comparison (by #chi^{2}/ndf):")
    latex.DrawLatex(0.15, 0.80, f"Gaussian: {model_results['gaussian']['chi2_ndf']:.3f}")
    latex.DrawLatex(0.15, 0.75, f"Double Gaussian: {model_results['double_gaussian']['chi2_ndf']:.3f}")
    latex.DrawLatex(0.15, 0.70, f"Crystal Ball: {model_results['crystal_ball']['chi2_ndf']:.3f}")
    
    # Save the plot
    canvas.SaveAs(f"{output_dir}/b_mass_shape_comparison.pdf")
    
    print(f"Model comparison plot saved to {output_dir}/b_mass_shape_comparison.pdf")

def create_model_comparison_plot(mc_data, true_id_branch, mass_branch, model_results, output_dir):
    """
    Create a plot comparing all three signal shape models on the same data
    
    Parameters:
    -----------
    mc_data : awkward array or dict
        MC data containing mass and truth information
    true_id_branch : str
        Name of the branch containing truth matching information
    mass_branch : str
        Name of the branch containing the B mass
    model_results : dict
        Dictionary containing the fit results for each model
    output_dir : str
        Directory to save output plots
    """
    # Create output directory
    create_output_dir(output_dir)
    
    # Get mass data and apply truth matching
    mass_data = np.array(mc_data[mass_branch])
    truth_match = np.array(mc_data[true_id_branch])
    
    # Apply truth matching cut (typically Bu_TRUEID > 0)
    truth_matched_mask = truth_match > 0
    mass_data_truth_matched = mass_data[truth_matched_mask]
    
    # Define fit range and bins
    mass_min, mass_max = 5200, 5400  # MeV/c^2
    n_bins = 80
    mass_bins = np.linspace(mass_min, mass_max, n_bins+1)
    
    # Create ROOT canvas
    canvas = TCanvas("canvas_comparison", "B+ Mass Model Comparison", 900, 700)
    canvas.SetLeftMargin(0.12)
    canvas.SetBottomMargin(0.12)
    
    # Create and fill histogram
    hist_name = "h_mass_comparison"
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
    
    # Create frame for plotting
    frame = mass.frame(RooFit.Title("B+ Mass Shape Models Comparison"))
    data_hist.plotOn(frame, RooFit.Name("data"))
    
    # Plot each model
    # 1. Gaussian
    mean_g = RooRealVar("mean_g", "Mean", model_results["gaussian"]["mean"]["value"])
    sigma_g = RooRealVar("sigma_g", "Sigma", model_results["gaussian"]["sigma"]["value"])
    gauss = RooGaussian("gauss", "Gaussian", mass, mean_g, sigma_g)
    
    # 2. Double Gaussian
    mean_dg = RooRealVar("mean_dg", "Mean", model_results["double_gaussian"]["mean"]["value"])
    sigma1_dg = RooRealVar("sigma1_dg", "Sigma1", model_results["double_gaussian"]["sigma1"]["value"])
    sigma2_dg = RooRealVar("sigma2_dg", "Sigma2", model_results["double_gaussian"]["sigma2"]["value"])
    frac_dg = RooRealVar("frac_dg", "Fraction", model_results["double_gaussian"]["fraction"]["value"])
    
    gauss1_dg = RooGaussian("gauss1_dg", "Core Gaussian", mass, mean_dg, sigma1_dg)
    gauss2_dg = RooGaussian("gauss2_dg", "Tail Gaussian", mass, mean_dg, sigma2_dg)
    double_gauss = RooAddPdf("double_gauss", "Double Gaussian", RooArgList(gauss1_dg, gauss2_dg), RooArgList(frac_dg))
    
    # 3. Crystal Ball
    mean_cb = RooRealVar("mean_cb", "Mean", model_results["crystal_ball"]["mean"]["value"])
    sigma_cb = RooRealVar("sigma_cb", "Sigma", model_results["crystal_ball"]["sigma"]["value"])
    alpha_cb = RooRealVar("alpha_cb", "Alpha", model_results["crystal_ball"]["alpha"]["value"])
    n_cb = RooRealVar("n_cb", "n", model_results["crystal_ball"]["n"]["value"])
    cb = RooCBShape("cb", "Crystal Ball", mass, mean_cb, sigma_cb, alpha_cb, n_cb)
    
    # Normalize all PDFs to same area
    norm_factor = hist.Integral() * (mass_max - mass_min) / n_bins
    
    # Plot each model
    gauss.plotOn(frame, RooFit.LineColor(ROOT.kGreen+2), RooFit.Name("gaussian"), 
                 RooFit.Normalization(norm_factor, ROOT.RooAbsReal.NumEvent))
    
    double_gauss.plotOn(frame, RooFit.LineColor(ROOT.kBlue), RooFit.Name("double_gaussian"),
                        RooFit.Normalization(norm_factor, ROOT.RooAbsReal.NumEvent))
    
    cb.plotOn(frame, RooFit.LineColor(ROOT.kRed), RooFit.Name("crystal_ball"),
              RooFit.Normalization(norm_factor, ROOT.RooAbsReal.NumEvent))
    
    # Draw the frame
    frame.Draw()
    
    # Add legend
    legend = ROOT.TLegend(0.65, 0.70, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.AddEntry(frame.findObject("data"), "MC Data", "P")
    legend.AddEntry(frame.findObject("gaussian"), "Gaussian", "L")
    legend.AddEntry(frame.findObject("double_gaussian"), "Double Gaussian", "L")
    legend.AddEntry(frame.findObject("crystal_ball"), "Crystal Ball", "L")
    legend.Draw()
    
    # Add text with fit quality comparison
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.15, 0.85, "Model Comparison (by #chi^{2}/ndf):")
    latex.DrawLatex(0.15, 0.80, f"Gaussian: {model_results['gaussian']['chi2_ndf']:.3f}")
    latex.DrawLatex(0.15, 0.75, f"Double Gaussian: {model_results['double_gaussian']['chi2_ndf']:.3f}")
    latex.DrawLatex(0.15, 0.70, f"Crystal Ball: {model_results['crystal_ball']['chi2_ndf']:.3f}")
    
    # Save the plot
    canvas.SaveAs(f"{output_dir}/b_mass_shape_comparison.pdf")
    
    print(f"Model comparison plot saved to {output_dir}/b_mass_shape_comparison.pdf")
    
def extract_signal_shape_with_background(mc_data, true_id_branch='Bu_TRUEID', mass_branch='Bu_MM', output_dir="./"):
    """
    Extract signal shape from MC using a Gaussian+polynomial model like the TF1 fit
    
    Parameters:
    -----------
    mc_data : awkward array or dict
        MC data containing mass and truth information
    true_id_branch : str
        Name of the branch containing truth matching information
    mass_branch : str
        Name of the branch containing the B mass
    output_dir : str
        Directory to save output plots
        
    Returns:
    --------
    dict of shape parameters
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get mass data - but DON'T filter on truth matched events
    mass_data = np.array(mc_data[mass_branch])
    sample_name = "MC_Gauss_Pol2"
    
    print(f"Extracting shape using Gaussian+Poly2 model")
    print(f"Number of events: {len(mass_data)}")
    
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
        mass_data, mass_bins, mass_min, mass_max, 
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
    
    # Gaussian signal parameters
    nominal_mass = 5279.0  # B+ PDG mass in MeV/c^2
    initial_sigma = 15.0   # Initial guess for resolution
    
    mean = RooRealVar("mean", "Mean", nominal_mass, nominal_mass-30, nominal_mass+30)
    sigma = RooRealVar("sigma", "Sigma", initial_sigma, 5, 30)
    
    # Create Gaussian PDF for signal
    gauss = RooGaussian("gauss", "Gaussian", mass, mean, sigma)
    
    # Create polynomial PDF for background - similar to pol2 in TF1
    c0 = RooRealVar("c0", "c0", 0.1, -1, 1)
    c1 = RooRealVar("c1", "c1", 0.1, -1, 1)
    c2 = RooRealVar("c2", "c2", 0.1, -1, 1)
    poly = RooChebychev("poly", "Polynomial", mass, RooArgList(c0, c1, c2))
    
    # Create yields
    nsig = RooRealVar("nsig", "signal yield", hist.Integral()/2, 0, hist.Integral()*2)
    nbkg = RooRealVar("nbkg", "background yield", hist.Integral()/2, 0, hist.Integral()*2)
    
    # Create the combined model
    model = RooAddPdf("model", "Signal + Background", RooArgList(gauss, poly), RooArgList(nsig, nbkg))
    
    # Perform the fit
    fit_result = model.fitTo(
        data_hist, 
        RooFit.Save(), 
        RooFit.PrintLevel(1),
        RooFit.Range(mass_min, mass_max)
    )
    
    # Plot the result
    frame = mass.frame(RooFit.Title("B+ Mass Shape - Gaussian+Poly2 Model"))
    data_hist.plotOn(frame, RooFit.Name("data"))
    model.plotOn(frame, RooFit.LineColor(ROOT.kBlue), RooFit.Name("model"))
    
    # Plot individual components
    model.plotOn(
        frame, 
        RooFit.Components("gauss"), 
        RooFit.LineColor(ROOT.kGreen+2), 
        RooFit.LineStyle(ROOT.kDashed),
        RooFit.Name("gauss")
    )
    model.plotOn(
        frame, 
        RooFit.Components("poly"), 
        RooFit.LineColor(ROOT.kRed), 
        RooFit.LineStyle(ROOT.kDashed),
        RooFit.Name("poly")
    )
    
    # Draw the frame
    frame.Draw()
    
    # Calculate chi-squared
    chi2 = frame.chiSquare("model", "data", fit_result.floatParsFinal().getSize())
    
    # Add text with fit results
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.025) 
    latex.DrawLatex(0.15, 0.85, f"Mean = {mean.getVal():.2f} #pm {mean.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.15, 0.80, f"#sigma = {sigma.getVal():.2f} #pm {sigma.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.15, 0.75, f"Signal yield = {nsig.getVal():.0f} #pm {nsig.getError():.0f}")
    latex.DrawLatex(0.15, 0.70, f"#chi^{{2}}/ndf = {chi2:.3f}")
    
    # Add legend
    legend = ROOT.TLegend(0.65, 0.70, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.AddEntry(frame.findObject("data"), "MC Data", "P")
    legend.AddEntry(frame.findObject("model"), "Total Fit", "L")
    legend.AddEntry(frame.findObject("gauss"), "Signal (Gaussian)", "L")
    legend.AddEntry(frame.findObject("poly"), "Background (Poly)", "L")
    legend.Draw()
    
    # Save the plot
    canvas.SaveAs(f"{output_dir}/b_mass_shape_{sample_name}.pdf")
    
    # Store results in a dictionary
    shape_params = {
        "model": "gaussian_poly2",
        "mean": {"value": mean.getVal(), "error": mean.getError()},
        "sigma": {"value": sigma.getVal(), "error": sigma.getError()},
        "signal_yield": {"value": nsig.getVal(), "error": nsig.getError()},
        "background_yield": {"value": nbkg.getVal(), "error": nbkg.getError()},
        "chi2_ndf": chi2,
        "events": len(mass_data)
    }
    
    # Save parameters to JSON file
    with open(f"{output_dir}/b_mass_shape_params_{sample_name}.json", 'w') as f:
        json.dump(shape_params, f, indent=4)
    
    print(f"Gaussian+Poly2 shape parameters:")
    print(f"  Mean = {mean.getVal():.2f} ± {mean.getError():.2f} MeV/c²")
    print(f"  Sigma = {sigma.getVal():.2f} ± {sigma.getError():.2f} MeV/c²")
    print(f"  Signal yield = {nsig.getVal():.0f} ± {nsig.getError():.0f}")
    print(f"  Background yield = {nbkg.getVal():.0f} ± {nbkg.getError():.0f}")
    print(f"  Signal fraction = {nsig.getVal()/(nsig.getVal()+nbkg.getVal())*100:.1f}%")
    print(f"  χ²/ndf = {chi2:.3f}")
    
    return shape_params    

def extract_signal_shape_cb_with_background(mc_data, true_id_branch='Bu_TRUEID', mass_branch='Bu_MM', output_dir="./"):
    """
    Extract signal shape from MC using a Crystal Ball + polynomial model
    
    Parameters:
    -----------
    mc_data : awkward array or dict
        MC data containing mass and truth information
    true_id_branch : str
        Name of the branch containing truth matching information
    mass_branch : str
        Name of the branch containing the B mass
    output_dir : str
        Directory to save output plots
        
    Returns:
    --------
    dict of shape parameters
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get mass data - don't filter on truth matched events
    mass_data = np.array(mc_data[mass_branch])
    sample_name = "MC_CB_Pol2"
    
    print(f"Extracting shape using Crystal Ball + Poly2 model")
    print(f"Number of events: {len(mass_data)}")
    
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
        mass_data, mass_bins, mass_min, mass_max, 
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
    
    # Crystal Ball signal parameters
    nominal_mass = 5279.0  # B+ PDG mass in MeV/c^2
    initial_sigma = 15.0   # Initial guess for resolution
    
    mean = RooRealVar("mean", "Mean", nominal_mass, nominal_mass-30, nominal_mass+30)
    sigma = RooRealVar("sigma", "Sigma", initial_sigma, 5, 30)
    alpha = RooRealVar("alpha", "Alpha", 1.5, 0.5, 5.0)  # Tail parameter
    n = RooRealVar("n", "n", 2, 0.5, 10.0)  # Power law parameter
    
    # Create Crystal Ball PDF for signal
    cb = RooCBShape("cb", "Crystal Ball", mass, mean, sigma, alpha, n)
    
    # Create polynomial PDF for background 
    c0 = RooRealVar("c0", "c0", 0.1, -1, 1)
    c1 = RooRealVar("c1", "c1", 0.1, -1, 1)
    c2 = RooRealVar("c2", "c2", 0.1, -1, 1)
    
    # Use the ROOT accessor to avoid import issues
    poly = ROOT.RooChebychev("poly", "Polynomial", mass, RooArgList(c0, c1, c2))
    
    # Create yields
    nsig = RooRealVar("nsig", "signal yield", hist.Integral()/2, 0, hist.Integral()*2)
    nbkg = RooRealVar("nbkg", "background yield", hist.Integral()/2, 0, hist.Integral()*2)
    
    # Create the combined model
    model = RooAddPdf("model", "Signal + Background", RooArgList(cb, poly), RooArgList(nsig, nbkg))
    
    # Perform the fit
    fit_result = model.fitTo(
        data_hist, 
        RooFit.Save(), 
        RooFit.PrintLevel(1),
        RooFit.Range(mass_min, mass_max)
    )
    
    # Plot the result
    frame = mass.frame(RooFit.Title("B+ Mass Shape - Crystal Ball + Poly2 Model"))
    data_hist.plotOn(frame, RooFit.Name("data"))
    model.plotOn(frame, RooFit.LineColor(ROOT.kBlue), RooFit.Name("model"))
    
    # Plot individual components
    model.plotOn(
        frame, 
        RooFit.Components("cb"), 
        RooFit.LineColor(ROOT.kGreen+2), 
        RooFit.LineStyle(ROOT.kDashed),
        RooFit.Name("cb")
    )
    model.plotOn(
        frame, 
        RooFit.Components("poly"), 
        RooFit.LineColor(ROOT.kRed), 
        RooFit.LineStyle(ROOT.kDashed),
        RooFit.Name("poly")
    )
    
    # Draw the frame
    frame.Draw()
    
    # Calculate chi-squared
    chi2 = frame.chiSquare("model", "data", fit_result.floatParsFinal().getSize())
    
    # Add text with fit results - positioned at top right to avoid overlap
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.030)
    latex.DrawLatex(0.65, 0.85, f"Mean = {mean.getVal():.2f} #pm {mean.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.65, 0.81, f"#sigma = {sigma.getVal():.2f} #pm {sigma.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.65, 0.77, f"#alpha = {alpha.getVal():.2f} #pm {alpha.getError():.2f}")
    latex.DrawLatex(0.65, 0.73, f"n = {n.getVal():.2f} #pm {n.getError():.2f}")
    latex.DrawLatex(0.65, 0.69, f"Signal yield = {nsig.getVal():.0f} #pm {nsig.getError():.0f}")
    latex.DrawLatex(0.65, 0.65, f"#chi^{{2}}/ndf = {chi2:.3f}")
    
    # Add legend
    legend = ROOT.TLegend(0.65, 0.50, 0.88, 0.64)
    legend.SetBorderSize(0)
    legend.AddEntry(frame.findObject("data"), "MC Data", "P")
    legend.AddEntry(frame.findObject("model"), "Total Fit", "L")
    legend.AddEntry(frame.findObject("cb"), "Signal (Crystal Ball)", "L")
    legend.AddEntry(frame.findObject("poly"), "Background (Poly)", "L")
    legend.Draw()
    
    # Save the plot
    canvas.SaveAs(f"{output_dir}/b_mass_shape_{sample_name}.png")
    canvas.SaveAs(f"{output_dir}/b_mass_shape_{sample_name}.pdf")
    
    # Store results in a dictionary
    shape_params = {
        "model": "crystal_ball_poly2",
        "mean": {"value": mean.getVal(), "error": mean.getError()},
        "sigma": {"value": sigma.getVal(), "error": sigma.getError()},
        "alpha": {"value": alpha.getVal(), "error": alpha.getError()},
        "n": {"value": n.getVal(), "error": n.getError()},
        "signal_yield": {"value": nsig.getVal(), "error": nsig.getError()},
        "background_yield": {"value": nbkg.getVal(), "error": nbkg.getError()},
        "chi2_ndf": chi2,
        "events": len(mass_data)
    }
    
    # Save parameters to JSON file
    with open(f"{output_dir}/b_mass_shape_params_{sample_name}.json", 'w') as f:
        json.dump(shape_params, f, indent=4)
    
    print(f"Crystal Ball + Poly2 shape parameters:")
    print(f"  Mean = {mean.getVal():.2f} ± {mean.getError():.2f} MeV/c²")
    print(f"  Sigma = {sigma.getVal():.2f} ± {sigma.getError():.2f} MeV/c²")
    print(f"  Alpha = {alpha.getVal():.2f} ± {alpha.getError():.2f}")
    print(f"  n = {n.getVal():.2f} ± {n.getError():.2f}")
    print(f"  Signal yield = {nsig.getVal():.0f} ± {nsig.getError():.0f}")
    print(f"  Background yield = {nbkg.getVal():.0f} ± {nbkg.getError():.0f}")
    print(f"  Signal fraction = {nsig.getVal()/(nsig.getVal()+nbkg.getVal())*100:.1f}%")
    print(f"  χ²/ndf = {chi2:.3f}")
    
    return shape_params

def extract_signal_shape_double_gaussian_with_background(mc_data, true_id_branch='Bu_TRUEID', mass_branch='Bu_MM', output_dir="./"):
    """
    Extract signal shape from MC using a Double Gaussian + polynomial model
    
    Parameters:
    -----------
    mc_data : awkward array or dict
        MC data containing mass and truth information
    true_id_branch : str
        Name of the branch containing truth matching information
    mass_branch : str
        Name of the branch containing the B mass
    output_dir : str
        Directory to save output plots
        
    Returns:
    --------
    dict of shape parameters
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get mass data - don't filter on truth matched events
    mass_data = np.array(mc_data[mass_branch])
    sample_name = "MC_DoubleGauss_Pol2"
    
    print(f"Extracting shape using Double Gaussian + Poly2 model")
    print(f"Number of events: {len(mass_data)}")
    
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
        mass_data, mass_bins, mass_min, mass_max, 
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
    
    # Double Gaussian parameters
    nominal_mass = 5279.0  # B+ PDG mass in MeV/c^2
    
    # Parameters for both Gaussians (shared mean)
    mean = RooRealVar("mean", "Mean", nominal_mass, nominal_mass-30, nominal_mass+30)
    
    # Core Gaussian (narrower)
    sigma1 = RooRealVar("sigma1", "Sigma1", 12.0, 5.0, 20.0)
    gauss1 = RooGaussian("gauss1", "Core Gaussian", mass, mean, sigma1)
    
    # Tail Gaussian (wider)
    sigma2 = RooRealVar("sigma2", "Sigma2", 20.0, 15.0, 40.0)
    gauss2 = RooGaussian("gauss2", "Tail Gaussian", mass, mean, sigma2)
    
    # Fraction parameter
    frac = RooRealVar("frac", "Fraction", 0.7, 0.0, 1.0)
    
    # Combine the two Gaussians
    double_gauss = RooAddPdf("double_gauss", "Double Gaussian", RooArgList(gauss1, gauss2), RooArgList(frac))
    
    # Create polynomial background
    c0 = RooRealVar("c0", "c0", 0.1, -1, 1)
    c1 = RooRealVar("c1", "c1", 0.1, -1, 1)
    c2 = RooRealVar("c2", "c2", 0.1, -1, 1)
    
    # Use the ROOT accessor to avoid import issues
    poly = ROOT.RooChebychev("poly", "Polynomial", mass, RooArgList(c0, c1, c2))
    
    # Create yields
    nsig = RooRealVar("nsig", "signal yield", hist.Integral()/2, 0, hist.Integral()*2)
    nbkg = RooRealVar("nbkg", "background yield", hist.Integral()/2, 0, hist.Integral()*2)
    
    # Create the combined model
    model = RooAddPdf("model", "Signal + Background", RooArgList(double_gauss, poly), RooArgList(nsig, nbkg))
    
    # Perform the fit
    fit_result = model.fitTo(
        data_hist, 
        RooFit.Save(), 
        RooFit.PrintLevel(1),
        RooFit.Range(mass_min, mass_max)
    )
    
    # Plot the result
    frame = mass.frame(RooFit.Title("B+ Mass Shape - Double Gaussian + Poly2 Model"))
    data_hist.plotOn(frame, RooFit.Name("data"))
    model.plotOn(frame, RooFit.LineColor(ROOT.kBlue), RooFit.Name("model"))
    
    # Plot individual components
    model.plotOn(
        frame, 
        RooFit.Components("double_gauss"), 
        RooFit.LineColor(ROOT.kGreen+2), 
        RooFit.LineStyle(ROOT.kDashed),
        RooFit.Name("double_gauss")
    )
    model.plotOn(
        frame, 
        RooFit.Components("poly"), 
        RooFit.LineColor(ROOT.kRed), 
        RooFit.LineStyle(ROOT.kDashed),
        RooFit.Name("poly")
    )
    
    # Draw the frame
    frame.Draw()
    
    # Calculate chi-squared
    chi2 = frame.chiSquare("model", "data", fit_result.floatParsFinal().getSize())
    
    # Calculate effective sigma (weighted average of the two sigmas)
    effective_sigma = np.sqrt(frac.getVal() * sigma1.getVal()**2 + (1-frac.getVal()) * sigma2.getVal()**2)
    effective_sigma_err = 0.5 * np.sqrt(
        (frac.getVal() * sigma1.getVal() * sigma1.getError())**2 + 
        ((1-frac.getVal()) * sigma2.getVal() * sigma2.getError())**2 +
        ((sigma1.getVal()**2 - sigma2.getVal()**2) * frac.getError())**2
    ) / effective_sigma
    
    # Add text with fit results - positioned at top right with smaller font
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.025)  # Reduced text size as requested
    latex.DrawLatex(0.65, 0.85, f"Mean = {mean.getVal():.2f} #pm {mean.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.65, 0.82, f"#sigma_{{1}} = {sigma1.getVal():.2f} #pm {sigma1.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.65, 0.79, f"#sigma_{{2}} = {sigma2.getVal():.2f} #pm {sigma2.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.65, 0.76, f"Fraction = {frac.getVal():.2f} #pm {frac.getError():.2f}")
    latex.DrawLatex(0.65, 0.73, f"#sigma_{{eff}} = {effective_sigma:.2f} #pm {effective_sigma_err:.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.65, 0.70, f"Signal yield = {nsig.getVal():.0f} #pm {nsig.getError():.0f}")
    latex.DrawLatex(0.65, 0.67, f"#chi^{{2}}/ndf = {chi2:.3f}")
    
    # Add legend
    legend = ROOT.TLegend(0.65, 0.52, 0.88, 0.64)
    legend.SetBorderSize(0)
    legend.AddEntry(frame.findObject("data"), "MC Data", "P")
    legend.AddEntry(frame.findObject("model"), "Total Fit", "L")
    legend.AddEntry(frame.findObject("double_gauss"), "Signal (Double Gauss)", "L")
    legend.AddEntry(frame.findObject("poly"), "Background (Poly)", "L")
    legend.Draw()
    
    # Save the plot
    canvas.SaveAs(f"{output_dir}/b_mass_shape_{sample_name}.pdf")
    
    # Store results in a dictionary
    shape_params = {
        "model": "double_gaussian_poly2",
        "mean": {"value": mean.getVal(), "error": mean.getError()},
        "sigma1": {"value": sigma1.getVal(), "error": sigma1.getError()},
        "sigma2": {"value": sigma2.getVal(), "error": sigma2.getError()},
        "fraction": {"value": frac.getVal(), "error": frac.getError()},
        "effective_sigma": {"value": effective_sigma, "error": effective_sigma_err},
        "signal_yield": {"value": nsig.getVal(), "error": nsig.getError()},
        "background_yield": {"value": nbkg.getVal(), "error": nbkg.getError()},
        "chi2_ndf": chi2,
        "events": len(mass_data)
    }
    
    # Save parameters to JSON file
    with open(f"{output_dir}/b_mass_shape_params_{sample_name}.json", 'w') as f:
        json.dump(shape_params, f, indent=4)
    
    print(f"Double Gaussian + Poly2 shape parameters:")
    print(f"  Mean = {mean.getVal():.2f} ± {mean.getError():.2f} MeV/c²")
    print(f"  Sigma1 = {sigma1.getVal():.2f} ± {sigma1.getError():.2f} MeV/c²")
    print(f"  Sigma2 = {sigma2.getVal():.2f} ± {sigma2.getError():.2f} MeV/c²")
    print(f"  Fraction = {frac.getVal():.2f} ± {frac.getError():.2f}")
    print(f"  Effective Sigma = {effective_sigma:.2f} ± {effective_sigma_err:.2f} MeV/c²")
    print(f"  Signal yield = {nsig.getVal():.0f} ± {nsig.getError():.0f}")
    print(f"  Background yield = {nbkg.getVal():.0f} ± {nbkg.getError():.0f}")
    print(f"  Signal fraction = {nsig.getVal()/(nsig.getVal()+nbkg.getVal())*100:.1f}%")
    print(f"  χ²/ndf = {chi2:.3f}")
    
    return shape_params

def extract_signal_shape_double_cb_with_background(mc_data, true_id_branch='Bu_TRUEID', mass_branch='Bu_MM', output_dir="./"):
    """
    Extract signal shape from MC using a Double Crystal Ball + polynomial model
    
    Parameters:
    -----------
    mc_data : awkward array or dict
        MC data containing mass and truth information
    true_id_branch : str
        Name of the branch containing truth matching information
    mass_branch : str
        Name of the branch containing the B mass
    output_dir : str
        Directory to save output plots
        
    Returns:
    --------
    dict of shape parameters
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get mass data - don't filter on truth matched events
    mass_data = np.array(mc_data[mass_branch])
    sample_name = "MC_DoubleCB_Pol2"
    
    print(f"Extracting shape using Double Crystal Ball + Poly2 model")
    print(f"Number of events: {len(mass_data)}")
    
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
        mass_data, mass_bins, mass_min, mass_max, 
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
    
    # Double Crystal Ball parameters (shared mean and sigma)
    nominal_mass = 5279.0  # B+ PDG mass in MeV/c^2
    
    mean = RooRealVar("mean", "Mean", nominal_mass, nominal_mass-30, nominal_mass+30)
    sigma = RooRealVar("sigma", "Sigma", 12.0, 5.0, 30.0)
    
    # First Crystal Ball (right tail)
    alpha1 = RooRealVar("alpha1", "Alpha1", 1.5, 0.5, 5.0)  # Right tail
    n1 = RooRealVar("n1", "n1", 5.0, 0.5, 20.0)
    cb1 = RooCBShape("cb1", "Crystal Ball Right", mass, mean, sigma, alpha1, n1)
    
    # Second Crystal Ball (left tail) - negative alpha for left tail
    alpha2 = RooRealVar("alpha2", "Alpha2", -1.5, -5.0, -0.5)  # Left tail (negative alpha)
    n2 = RooRealVar("n2", "n2", 5.0, 0.5, 20.0)
    cb2 = RooCBShape("cb2", "Crystal Ball Left", mass, mean, sigma, alpha2, n2)
    
    # Fraction parameter
    frac = RooRealVar("frac", "Fraction", 0.7, 0.0, 1.0)
    
    # Combine the two Crystal Balls
    double_cb = RooAddPdf("double_cb", "Double Crystal Ball", RooArgList(cb1, cb2), RooArgList(frac))
    
    # Create polynomial background
    c0 = RooRealVar("c0", "c0", 0.1, -1, 1)
    c1 = RooRealVar("c1", "c1", 0.1, -1, 1)
    c2 = RooRealVar("c2", "c2", 0.1, -1, 1)
    
    # Use the ROOT accessor to avoid import issues
    poly = ROOT.RooChebychev("poly", "Polynomial", mass, RooArgList(c0, c1, c2))
    
    # Create yields
    nsig = RooRealVar("nsig", "signal yield", hist.Integral()/2, 0, hist.Integral()*2)
    nbkg = RooRealVar("nbkg", "background yield", hist.Integral()/2, 0, hist.Integral()*2)
    
    # Create the combined model
    model = RooAddPdf("model", "Signal + Background", RooArgList(double_cb, poly), RooArgList(nsig, nbkg))
    
    # Perform the fit
    fit_result = model.fitTo(
        data_hist, 
        RooFit.Save(), 
        RooFit.PrintLevel(1),
        RooFit.Range(mass_min, mass_max)
    )
    
    # Plot the result
    frame = mass.frame(RooFit.Title("B+ Mass Shape - Double Crystal Ball + Poly2 Model"))
    data_hist.plotOn(frame, RooFit.Name("data"))
    model.plotOn(frame, RooFit.LineColor(ROOT.kBlue), RooFit.Name("model"))
    
    # Plot individual components
    model.plotOn(
        frame, 
        RooFit.Components("double_cb"), 
        RooFit.LineColor(ROOT.kGreen+2), 
        RooFit.LineStyle(ROOT.kDashed),
        RooFit.Name("double_cb")
    )
    model.plotOn(
        frame, 
        RooFit.Components("poly"), 
        RooFit.LineColor(ROOT.kRed), 
        RooFit.LineStyle(ROOT.kDashed),
        RooFit.Name("poly")
    )
    
    # Draw the frame
    frame.Draw()
    
    # Calculate chi-squared
    chi2 = frame.chiSquare("model", "data", fit_result.floatParsFinal().getSize())
    
    # Add text with fit results - positioned at top right with smaller font
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.025)  # Reduced text size as requested
    latex.DrawLatex(0.65, 0.85, f"Mean = {mean.getVal():.2f} #pm {mean.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.65, 0.82, f"#sigma = {sigma.getVal():.2f} #pm {sigma.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.65, 0.79, f"#alpha_{{1}} = {alpha1.getVal():.2f} #pm {alpha1.getError():.2f}")
    latex.DrawLatex(0.65, 0.76, f"n_{{1}} = {n1.getVal():.2f} #pm {n1.getError():.2f}")
    latex.DrawLatex(0.65, 0.73, f"#alpha_{{2}} = {alpha2.getVal():.2f} #pm {alpha2.getError():.2f}")
    latex.DrawLatex(0.65, 0.70, f"n_{{2}} = {n2.getVal():.2f} #pm {n2.getError():.2f}")
    latex.DrawLatex(0.65, 0.67, f"Fraction = {frac.getVal():.2f} #pm {frac.getError():.2f}")
    latex.DrawLatex(0.65, 0.64, f"Signal yield = {nsig.getVal():.0f} #pm {nsig.getError():.0f}")
    latex.DrawLatex(0.65, 0.61, f"#chi^{{2}}/ndf = {chi2:.3f}")
    
    # Add legend
    legend = ROOT.TLegend(0.65, 0.46, 0.88, 0.58)
    legend.SetBorderSize(0)
    legend.AddEntry(frame.findObject("data"), "MC Data", "P")
    legend.AddEntry(frame.findObject("model"), "Total Fit", "L")
    legend.AddEntry(frame.findObject("double_cb"), "Signal (Double CB)", "L")
    legend.AddEntry(frame.findObject("poly"), "Background (Poly)", "L")
    legend.Draw()
    
    # Save the plot
    canvas.SaveAs(f"{output_dir}/b_mass_shape_{sample_name}.pdf")
    
    # Store results in a dictionary
    shape_params = {
        "model": "double_crystal_ball_poly2",
        "mean": {"value": mean.getVal(), "error": mean.getError()},
        "sigma": {"value": sigma.getVal(), "error": sigma.getError()},
        "alpha1": {"value": alpha1.getVal(), "error": alpha1.getError()},
        "n1": {"value": n1.getVal(), "error": n1.getError()},
        "alpha2": {"value": alpha2.getVal(), "error": alpha2.getError()},
        "n2": {"value": n2.getVal(), "error": n2.getError()},
        "fraction": {"value": frac.getVal(), "error": frac.getError()},
        "signal_yield": {"value": nsig.getVal(), "error": nsig.getError()},
        "background_yield": {"value": nbkg.getVal(), "error": nbkg.getError()},
        "chi2_ndf": chi2,
        "events": len(mass_data)
    }
    
    # Save parameters to JSON file
    with open(f"{output_dir}/b_mass_shape_params_{sample_name}.json", 'w') as f:
        json.dump(shape_params, f, indent=4)
    
    print(f"Double Crystal Ball + Poly2 shape parameters:")
    print(f"  Mean = {mean.getVal():.2f} ± {mean.getError():.2f} MeV/c²")
    print(f"  Sigma = {sigma.getVal():.2f} ± {sigma.getError():.2f} MeV/c²")
    print(f"  Alpha1 = {alpha1.getVal():.2f} ± {alpha1.getError():.2f}")
    print(f"  n1 = {n1.getVal():.2f} ± {n1.getError():.2f}")
    print(f"  Alpha2 = {alpha2.getVal():.2f} ± {alpha2.getError():.2f}")
    print(f"  n2 = {n2.getVal():.2f} ± {n2.getError():.2f}")
    print(f"  Fraction = {frac.getVal():.2f} ± {frac.getError():.2f}")
    print(f"  Signal yield = {nsig.getVal():.0f} ± {nsig.getError():.0f}")
    print(f"  Background yield = {nbkg.getVal():.0f} ± {nbkg.getError():.0f}")
    print(f"  Signal fraction = {nsig.getVal()/(nsig.getVal()+nbkg.getVal())*100:.1f}%")
    print(f"  χ²/ndf = {chi2:.3f}")
    
    return shape_params