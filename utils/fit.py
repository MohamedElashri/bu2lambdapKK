import numpy as np
import ROOT
from ROOT import TH1F, TCanvas, TF1, RooRealVar, RooGaussian, RooDataHist, RooCBShape
from ROOT import RooFit, RooPlot, RooArgList, RooArgSet, TLatex, gStyle
from ROOT import RooAddPdf, RooFFTConvPdf, RooDataSet, RooFormulaVar, RooChebychev
from array import array
import os
import json
from utils.mc_kpkm import create_root_histogram

def load_mc_fit_params(json_file_path):
    """
    Load MC fit parameters from a JSON file
    
    Parameters:
    -----------
    json_file_path : str
        Path to the JSON file containing MC fit parameters
        
    Returns:
    --------
    dict
        The MC parameters loaded from the JSON file
    """
    try:
        with open(json_file_path, 'r') as f:
            mc_params = json.load(f)
        print(f"Successfully loaded MC parameters from {json_file_path}")
        return mc_params
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: File {json_file_path} is not valid JSON")
        return None
        
    

def fit_data_with_mc_constraints(data, mass_branch='Bu_MM', mc_params=None, output_dir="./"):
    """
    Fit real data using constraints from MC
    
    Parameters:
    -----------
    data : awkward array or dict
        Real data containing mass information
    mass_branch : str
        Name of the branch containing the B mass
    mc_params : dict
        Parameters extracted from MC fits to use as constraints
    output_dir : str
        Directory to save output plots
        
    Returns:
    --------
    dict of fit parameters
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get mass data
    mass_data = np.array(data[mass_branch])
    sample_name = "RealData_DoubleGauss_Pol2"
    
    print(f"Fitting real data using Double Gaussian + Poly2 model with MC constraints")
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
    
    # Double Gaussian parameters - use MC values as starting points
    # The mean is allowed to float but with a constraint from MC
    mc_mean = mc_params.get("mean", {}).get("value", 5280.0)
    mc_mean_err = mc_params.get("mean", {}).get("error", 0.05)
    
    mean = RooRealVar("mean", "Mean", mc_mean, mc_mean-5, mc_mean+5)
    
    # For signal width, we'll apply a Gaussian constraint from MC
    # This assumes detector resolution is similar between MC and data
    if "sigma1" in mc_params:
        # If we have double gaussian parameters
        mc_sigma1 = mc_params.get("sigma1", {}).get("value", 10.0)
        mc_sigma1_err = mc_params.get("sigma1", {}).get("error", 0.1)
        mc_sigma2 = mc_params.get("sigma2", {}).get("value", 20.0)
        mc_sigma2_err = mc_params.get("sigma2", {}).get("error", 0.2)
        mc_frac = mc_params.get("fraction", {}).get("value", 0.7)
        mc_frac_err = mc_params.get("fraction", {}).get("error", 0.05)
    else:
        # If we only have single gaussian parameters, estimate double gaussian
        mc_sigma = mc_params.get("sigma", {}).get("value", 12.0)
        mc_sigma_err = mc_params.get("sigma", {}).get("error", 0.05)
        mc_sigma1 = mc_sigma * 0.8  # Narrower component
        mc_sigma1_err = mc_sigma_err * 0.8
        mc_sigma2 = mc_sigma * 1.5  # Wider component
        mc_sigma2_err = mc_sigma_err * 1.5
        mc_frac = 0.7  # Typical value
        mc_frac_err = 0.1
    
    # Core Gaussian (narrower)
    sigma1 = RooRealVar("sigma1", "Sigma1", mc_sigma1, mc_sigma1*0.5, mc_sigma1*1.5)
    sigma1_constraint = RooGaussian("sigma1_constraint", "Sigma1 Constraint", 
                                     sigma1, RooFit.RooConst(mc_sigma1), RooFit.RooConst(mc_sigma1_err*2))
    
    # Tail Gaussian (wider)
    sigma2 = RooRealVar("sigma2", "Sigma2", mc_sigma2, mc_sigma2*0.5, mc_sigma2*1.5)
    sigma2_constraint = RooGaussian("sigma2_constraint", "Sigma2 Constraint", 
                                     sigma2, RooFit.RooConst(mc_sigma2), RooFit.RooConst(mc_sigma2_err*2))
    
    # Fraction parameter
    frac = RooRealVar("frac", "Fraction", mc_frac, mc_frac*0.5, min(mc_frac*1.5, 0.99))
    frac_constraint = RooGaussian("frac_constraint", "Fraction Constraint", 
                                   frac, RooFit.RooConst(mc_frac), RooFit.RooConst(mc_frac_err*2))
    
    # Create Gaussian components
    gauss1 = RooGaussian("gauss1", "Core Gaussian", mass, mean, sigma1)
    gauss2 = RooGaussian("gauss2", "Tail Gaussian", mass, mean, sigma2)
    
    # Combine the two Gaussians
    double_gauss = RooAddPdf("double_gauss", "Double Gaussian", 
                              RooArgList(gauss1, gauss2), RooArgList(frac))
    
    # Create polynomial background
    c0 = RooRealVar("c0", "c0", 0.1, -1, 1)
    c1 = RooRealVar("c1", "c1", 0.1, -1, 1)
    c2 = RooRealVar("c2", "c2", 0.1, -1, 1)
    
    # Use the ROOT accessor to avoid import issues
    poly = ROOT.RooChebychev("poly", "Polynomial", mass, RooArgList(c0, c1, c2))
    
    # Create yields - these should float freely based on data
    nsig = RooRealVar("nsig", "signal yield", hist.Integral()/2, 0, hist.Integral()*2)
    nbkg = RooRealVar("nbkg", "background yield", hist.Integral()/2, 0, hist.Integral()*2)
    
    # Create the combined model
    model = RooAddPdf("model", "Signal + Background", 
                       RooArgList(double_gauss, poly), RooArgList(nsig, nbkg))
    
    # Create a set of constraints to use in the fit
    constraints = RooArgSet(sigma1_constraint, sigma2_constraint, frac_constraint)
    
    # Perform the fit with constraints
    fit_result = model.fitTo(
        data_hist, 
        RooFit.Save(), 
        RooFit.PrintLevel(1),
        RooFit.Range(mass_min, mass_max),
        RooFit.ExternalConstraints(constraints)
    )
    
    # Plot the result
    frame = mass.frame(RooFit.Title("B+ Mass Fit - Double Gaussian + Poly2 Model"))
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
    
    # Calculate signal significance
    signal_significance = nsig.getVal() / nsig.getError()
    
    # Add text with fit results - positioned at top right with smaller font
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.025)  # Small text size
    latex.DrawLatex(0.65, 0.85, f"Mean = {mean.getVal():.2f} #pm {mean.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.65, 0.82, f"#sigma_{{1}} = {sigma1.getVal():.2f} #pm {sigma1.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.65, 0.79, f"#sigma_{{2}} = {sigma2.getVal():.2f} #pm {sigma2.getError():.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.65, 0.76, f"Fraction = {frac.getVal():.2f} #pm {frac.getError():.2f}")
    latex.DrawLatex(0.65, 0.73, f"#sigma_{{eff}} = {effective_sigma:.2f} #pm {effective_sigma_err:.2f} MeV/c^{{2}}")
    latex.DrawLatex(0.65, 0.70, f"Signal yield = {nsig.getVal():.0f} #pm {nsig.getError():.0f}")
    latex.DrawLatex(0.65, 0.67, f"Bkg yield = {nbkg.getVal():.0f} #pm {nbkg.getError():.0f}")
    latex.DrawLatex(0.65, 0.64, f"S/sqrt(S+B) = {nsig.getVal()/np.sqrt(nsig.getVal()+nbkg.getVal()):.1f}")
    latex.DrawLatex(0.65, 0.61, f"Significance = {signal_significance:.1f}#sigma")
    latex.DrawLatex(0.65, 0.58, f"#chi^{{2}}/ndf = {chi2:.3f}")
    
    # Add legend
    legend = ROOT.TLegend(0.65, 0.43, 0.88, 0.55)
    legend.SetBorderSize(0)
    legend.AddEntry(frame.findObject("data"), "Data", "P")
    legend.AddEntry(frame.findObject("model"), "Total Fit", "L")
    legend.AddEntry(frame.findObject("double_gauss"), "Signal", "L")
    legend.AddEntry(frame.findObject("poly"), "Background", "L")
    legend.Draw()
    
    # Save the plot
    canvas.SaveAs(f"{output_dir}/b_mass_fit_{sample_name}.png")
    canvas.SaveAs(f"{output_dir}/b_mass_fit_{sample_name}.pdf")
    
    # Store results in a dictionary
    fit_params = {
        "model": "double_gaussian_poly2",
        "mean": {"value": mean.getVal(), "error": mean.getError()},
        "sigma1": {"value": sigma1.getVal(), "error": sigma1.getError()},
        "sigma2": {"value": sigma2.getVal(), "error": sigma2.getError()},
        "fraction": {"value": frac.getVal(), "error": frac.getError()},
        "effective_sigma": {"value": effective_sigma, "error": effective_sigma_err},
        "signal_yield": {"value": nsig.getVal(), "error": nsig.getError()},
        "background_yield": {"value": nbkg.getVal(), "error": nbkg.getError()},
        "signal_significance": signal_significance,
        "s_over_sqrt_s_plus_b": nsig.getVal()/np.sqrt(nsig.getVal()+nbkg.getVal()),
        "chi2_ndf": chi2,
        "events": len(mass_data)
    }
    
    # Save parameters to JSON file
    with open(f"{output_dir}/b_mass_fit_params_{sample_name}.json", 'w') as f:
        json.dump(fit_params, f, indent=4)
    
    print(f"Double Gaussian + Poly2 fit results:")
    print(f"  Mean = {mean.getVal():.2f} ± {mean.getError():.2f} MeV/c²")
    print(f"  Sigma1 = {sigma1.getVal():.2f} ± {sigma1.getError():.2f} MeV/c²")
    print(f"  Sigma2 = {sigma2.getVal():.2f} ± {sigma2.getError():.2f} MeV/c²")
    print(f"  Fraction = {frac.getVal():.2f} ± {frac.getError():.2f}")
    print(f"  Effective Sigma = {effective_sigma:.2f} ± {effective_sigma_err:.2f} MeV/c²")
    print(f"  Signal yield = {nsig.getVal():.0f} ± {nsig.getError():.0f}")
    print(f"  Background yield = {nbkg.getVal():.0f} ± {nbkg.getError():.0f}")
    print(f"  Signal significance = {signal_significance:.1f}σ")
    print(f"  S/sqrt(S+B) = {nsig.getVal()/np.sqrt(nsig.getVal()+nbkg.getVal()):.1f}")
    print(f"  χ²/ndf = {chi2:.3f}")
    
    return fit_params