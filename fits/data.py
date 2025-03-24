import numpy as np
import uproot
import ROOT
from ROOT import TH1F, TCanvas, TF1, RooRealVar, RooGaussian, RooChebychev
from ROOT import RooAddPdf, RooArgList, RooDataHist, RooCBShape, RooPolynomial
from ROOT import RooFit, RooPlot, RooArgSet, TLatex, gStyle
from array import array

# Configure ROOT settings
ROOT.gROOT.SetBatch(True)  # Run in batch mode (no graphics)
gStyle.SetOptStat(0)       # Don't display stat box
gStyle.SetOptFit(1)        # Display fit parameters
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)  # Reduce verbosity

def create_root_histogram(data, bins, x_min, x_max, name="h_data", title="Data"):
    """Convert numpy array to ROOT histogram"""
    hist = TH1F(name, title, len(bins)-1, array('d', bins))
    for val in data:
        if x_min <= val <= x_max:
            hist.Fill(val)
    return hist

def fit_bu_mass_simple(data, bins, output_filename="bu_mass_fit.pdf"):
    """Perform a simple unbinned fit to the B+ mass using Gaussian + polynomial"""
    
    # Create canvas
    canvas = TCanvas("canvas", "B+ mass fit", 900, 700)
    canvas.SetLeftMargin(0.12)
    canvas.SetBottomMargin(0.12)
    
    # Create ROOT histogram
    h_mass = create_root_histogram(
        data, bins, min(bins), max(bins), "h_mass", 
        "B^{+} #rightarrow #bar{#Lambda}^{0}pK^{+}K^{-}"
    )
    h_mass.SetMarkerStyle(20)
    h_mass.SetMarkerSize(0.8)
    h_mass.SetLineColor(ROOT.kBlack)
    h_mass.GetXaxis().SetTitle("m(#bar{#Lambda}^{0}pK^{+}K^{-}) [MeV/c^{2}]")
    h_mass.GetYaxis().SetTitle("Candidates / (%.1f MeV/c^{2})" % ((max(bins) - min(bins))/(len(bins)-1)))
    h_mass.GetYaxis().SetTitleOffset(1.5)
    h_mass.Draw("E")
    
    # Define fit function: Gaussian (signal) + polynomial (background)
    fit_func = TF1("fit_func", "gaus(0) + pol2(3)", min(bins), max(bins))
    
    # Initial parameters: [0]=amplitude, [1]=mean, [2]=sigma, [3,4,5]=polynomial
    nominal_mass = 5279.0  # B+ PDG mass in MeV/c^2
    initial_sigma = 15.0   # Initial guess for resolution
    
    fit_func.SetParameters(
        h_mass.GetMaximum() * 0.7,  # Amplitude
        nominal_mass,                # Mean
        initial_sigma,               # Sigma
        h_mass.GetMaximum() * 0.5,   # p0
        -0.1,                        # p1
        0.0                          # p2
    )
    
    # Set parameter names for better readability
    fit_func.SetParNames("Amp", "Mean", "Sigma", "BG_p0", "BG_p1", "BG_p2")
    
    # Set parameter limits
    fit_func.SetParLimits(1, nominal_mass - 30, nominal_mass + 30)  # Constrain mean
    fit_func.SetParLimits(2, 5.0, 50.0)  # Constrain sigma
    
    # Perform fit
    fit_result = h_mass.Fit("fit_func", "SREM")  # S=Save, R=Range, E=Extended output, M=Improved fit
    
    # Get fit parameters
    signal_amp = fit_func.GetParameter(0)
    signal_mean = fit_func.GetParameter(1)
    signal_sigma = fit_func.GetParameter(2)
    
    # Draw individual components
    signal_func = TF1("signal", "gaus", min(bins), max(bins))
    signal_func.SetParameters(signal_amp, signal_mean, signal_sigma)
    signal_func.SetLineColor(ROOT.kGreen+2)
    signal_func.SetLineStyle(2)
    signal_func.Draw("same")
    
    bg_func = TF1("bg", "pol2", min(bins), max(bins))
    bg_func.SetParameters(fit_func.GetParameter(3), fit_func.GetParameter(4), fit_func.GetParameter(5))
    bg_func.SetLineColor(ROOT.kRed)
    bg_func.SetLineStyle(2)
    bg_func.Draw("same")
    
    # Calculate signal yield
    bin_width = (max(bins) - min(bins)) / (len(bins) - 1)
    signal_yield = signal_amp * signal_sigma * np.sqrt(2 * np.pi) / bin_width
    signal_yield_err = signal_yield * fit_func.GetParError(0) / signal_amp
    
    # Add legend
    legend = ROOT.TLegend(0.65, 0.65, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.AddEntry(h_mass, "Data", "PE")
    legend.AddEntry(fit_func, "Fit (Gaus + Pol2)", "L")
    legend.AddEntry(signal_func, "Signal", "L")
    legend.AddEntry(bg_func, "Background", "L")
    legend.Draw()
    
    # Add text with fit results
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.15, 0.85, f"Signal yield = {signal_yield:.1f} #pm {signal_yield_err:.1f}")
    latex.DrawLatex(0.15, 0.80, f"Mass = {signal_mean:.1f} #pm {fit_func.GetParError(1):.1f} MeV/c^{{2}}")
    latex.DrawLatex(0.15, 0.75, f"#sigma = {signal_sigma:.1f} #pm {fit_func.GetParError(2):.1f} MeV/c^{{2}}")
    latex.DrawLatex(0.15, 0.70, f"#chi^{{2}}/ndf = {fit_func.GetChisquare():.1f}/{fit_func.GetNDF()}")
    
    # Save the plot
    canvas.SaveAs(output_filename)
    print(f"Fit saved to {output_filename}")
    
    return {
        "signal_yield": signal_yield,
        "signal_yield_err": signal_yield_err,
        "mass": signal_mean,
        "mass_err": fit_func.GetParError(1),
        "sigma": signal_sigma,
        "sigma_err": fit_func.GetParError(2),
        "chi2_ndf": fit_func.GetChisquare() / fit_func.GetNDF() if fit_func.GetNDF() > 0 else 0
    }

def fit_bu_mass_roofit(data, bins, output_filename="bu_mass_roofit.pdf"):
    """
    Perform a more sophisticated fit using RooFit with Crystal Ball for signal
    and Chebychev polynomial for background
    """
    # Create canvas
    canvas = TCanvas("canvas_roofit", "B+ mass RooFit", 900, 700)
    canvas.SetLeftMargin(0.12)
    canvas.SetBottomMargin(0.12)
    
    # Filter data to be within the bin range
    mass_min, mass_max = min(bins), max(bins)
    mass_data = np.array([m for m in data if mass_min <= m <= mass_max])
    
    # Check if we have data
    if len(mass_data) == 0:
        print("Error: No data points within the specified range")
        return {
            "signal_yield": 0, "signal_yield_err": 0,
            "mass": 0, "mass_err": 0,
            "sigma": 0, "sigma_err": 0,
            "chi2_ndf": 0
        }
    
    # Create mass variable
    mass = RooRealVar("mass", "m(#bar{#Lambda}^{0}pK^{+}K^{-}) [MeV/c^{2}]", mass_min, mass_max)
    
    # Create a temporary ROOT histogram
    hist_name = "temp_hist"
    hist = ROOT.TH1F(hist_name, "B+ Mass", len(bins)-1, array('d', bins))
    for val in mass_data:
        hist.Fill(val)
    
    # Create RooDataHist from the ROOT histogram
    data_hist = ROOT.RooDataHist("data_hist", "B mass data", ROOT.RooArgList(mass), hist)
    
    # Signal model - Crystal Ball function (accounts for radiative tail)
    mean = RooRealVar("mean", "mean", 5279, 5260, 5290)
    sigma = RooRealVar("sigma", "sigma", 15, 5, 30)
    alpha = RooRealVar("alpha", "alpha", 1.5, 0.5, 5.0)
    n = RooRealVar("n", "n", 2, 0.5, 10.0)
    signal = RooCBShape("signal", "Crystal Ball", mass, mean, sigma, alpha, n)
    
    # Background model - First try exponential which is more stable than Chebychev
    slope = RooRealVar("slope", "slope", -0.001, -0.1, 0.1)
    background = ROOT.RooExponential("background", "Background", mass, slope)
    
    # Combined model
    nsig = RooRealVar("nsig", "signal yield", len(mass_data)/2, 0, len(mass_data)*2)
    nbkg = RooRealVar("nbkg", "background yield", len(mass_data)/2, 0, len(mass_data)*2)
    model = RooAddPdf("model", "Signal + Background", RooArgList(signal, background), RooArgList(nsig, nbkg))
    
    # Perform the fit with more robust settings
    result = model.fitTo(
        data_hist, 
        RooFit.Save(), 
        RooFit.PrintLevel(0),  # Suppress output
        RooFit.Range(mass_min, mass_max),
        RooFit.InitialHesse(True),  # More stable error estimation
        RooFit.Minos(False)  # Disable Minos which can be unstable
    )
    
    # Plot the result
    frame = mass.frame(ROOT.RooFit.Title("B^{+} #rightarrow #bar{#Lambda}^{0}pK^{+}K^{-}"))
    data_hist.plotOn(frame, RooFit.Name("data"))
    
    model.plotOn(frame, RooFit.Name("fit"))
    model.plotOn(frame, RooFit.Components("signal"), RooFit.LineStyle(ROOT.kDashed), 
                RooFit.LineColor(ROOT.kGreen+2), RooFit.Name("signal"))
    model.plotOn(frame, RooFit.Components("background"), RooFit.LineStyle(ROOT.kDashed), 
                RooFit.LineColor(ROOT.kRed), RooFit.Name("background"))
    
    frame.Draw()
    frame.GetYaxis().SetTitleOffset(1.5)
    frame.SetYTitle("Candidates / (%.1f MeV/c^{2})" % ((max(bins) - min(bins))/(len(bins)-1)))
    
    # Add legend
    legend = ROOT.TLegend(0.65, 0.65, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.AddEntry(frame.findObject("data"), "Data", "P")
    legend.AddEntry(frame.findObject("fit"), "Fit", "L")
    legend.AddEntry(frame.findObject("signal"), "Signal (Crystal Ball)", "L")
    legend.AddEntry(frame.findObject("background"), "Background (Chebychev)", "L")
    legend.Draw()
    
    # Add text box with results
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.15, 0.85, f"Signal yield = {nsig.getVal():.1f} #pm {nsig.getError():.1f}")
    latex.DrawLatex(0.15, 0.80, f"Mass = {mean.getVal():.1f} #pm {mean.getError():.1f} MeV/c^{{2}}")
    latex.DrawLatex(0.15, 0.75, f"#sigma = {sigma.getVal():.1f} #pm {sigma.getError():.1f} MeV/c^{{2}}")
    
    # Calculate chi2 more safely
    try:
        chi2 = frame.chiSquare("fit", "data", result.floatParsFinal().getSize())
    except:
        chi2 = 0
        print("Warning: Could not calculate chi2")
    
    latex.DrawLatex(0.15, 0.70, f"#chi^{{2}}/ndf = {chi2:.3f}")
    
    # Add fit convergence status
    fit_status = result.status()
    covQual = result.covQual()
    status_text = f"Fit status: {fit_status}, Cov. quality: {covQual}"
    latex.DrawLatex(0.15, 0.65, status_text)
    
    # Save the plot
    canvas.SaveAs(output_filename)
    print(f"RooFit saved to {output_filename}")
    
    # Check if fit converged properly
    if fit_status != 0 or covQual < 2:
        print("Warning: RooFit may not have converged properly!")
    
    return {
        "signal_yield": nsig.getVal(),
        "signal_yield_err": nsig.getError(),
        "mass": mean.getVal(),
        "mass_err": mean.getError(),
        "sigma": sigma.getVal(),
        "sigma_err": sigma.getError(),
        "chi2_ndf": chi2,
        "fit_status": fit_status,
        "cov_quality": covQual
    }

# Function to run both fits
def run_mass_fits(data, n_bins=100, output_dir="./", mass_range=(5200, 5400)):
    """Run both fit methods on the data
    
    Parameters:
    -----------
    data : array-like
        Array containing the mass values to fit
    n_bins : int
        Number of bins to use in the histogram
    output_dir : str
        Directory where to save the output plots
    mass_range : tuple
        Mass range to fit (min, max)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Make sure we have data to fit
    data_in_range = [m for m in data if mass_range[0] <= m <= mass_range[1]]
    if len(data_in_range) == 0:
        print("Error: No data points in the specified mass range")
        return None, None
    
    print(f"Data points in fit range: {len(data_in_range)}")
    
    # Create bins
    mass_bins = np.linspace(mass_range[0], mass_range[1], n_bins+1)
    
    # Run both fits with proper error handling
    print("Running simple TF1 fit...")
    try:
        simple_results = fit_bu_mass_simple(data, mass_bins, f"{output_dir}/bu_mass_fit_tf1.pdf")
        simple_success = True
    except Exception as e:
        print(f"Error in TF1 fit: {e}")
        simple_results = {"signal_yield": 0, "signal_yield_err": 0, "mass": 0, "mass_err": 0, 
                         "sigma": 0, "sigma_err": 0, "chi2_ndf": 0}
        simple_success = False
    
    print("Running RooFit fit...")
    try:
        roofit_results = fit_bu_mass_roofit(data, mass_bins, f"{output_dir}/bu_mass_fit_roofit.pdf")
        roofit_success = True
    except Exception as e:
        print(f"Error in RooFit: {e}")
        roofit_results = {"signal_yield": 0, "signal_yield_err": 0, "mass": 0, "mass_err": 0, 
                         "sigma": 0, "sigma_err": 0, "chi2_ndf": 0}
        roofit_success = False
    
    # Print comparison of results
    print("\nFit Results Comparison:")
    print("-" * 60)
    print(f"{'Parameter':<20} {'TF1 Fit':<20} {'RooFit':<20}")
    print("-" * 60)
    
    for param in ["signal_yield", "mass", "sigma", "chi2_ndf"]:
        if param != "chi2_ndf" and simple_success:
            tf1_val = f"{simple_results[param]:.2f} ± {simple_results[param+'_err']:.2f}"
        else:
            tf1_val = f"{simple_results[param]:.3f}" if simple_success else "Failed"
            
        if param != "chi2_ndf" and roofit_success:
            roofit_val = f"{roofit_results[param]:.2f} ± {roofit_results[param+'_err']:.2f}"
        else:
            roofit_val = f"{roofit_results[param]:.3f}" if roofit_success else "Failed"
            
        print(f"{param:<20} {tf1_val:<20} {roofit_val:<20}")
    
    # Print RooFit fit status if available
    if roofit_success and "fit_status" in roofit_results:
        print(f"\nRooFit status: {roofit_results['fit_status']} (0=good)")
        print(f"Covariance quality: {roofit_results['cov_quality']} (3=good)")
    
    return simple_results, roofit_results

# Here's an example of how to use the functions for fitting
def demo_fit():
    """Demo using simple generated data"""
    # Generate some sample data to demonstrate the fit functions
    np.random.seed(42)
    
    # Parameters for our toy model
    n_signal = 10000
    n_background = 5000
    signal_mean = 5279.0
    signal_sigma = 15.0
    
    # Generate signal (Gaussian)
    signal_data = np.random.normal(signal_mean, signal_sigma, n_signal)
    
    # Generate background (exponential)
    background_data = 5200 + np.random.exponential(100, n_background)
    background_data = background_data[background_data < 5400]  # Truncate to our fit range
    
    # Combine datasets
    combined_data = np.concatenate([signal_data, background_data])
    
    # Run fits
    print("Running fits on sample data...")
    simple_results, roofit_results = run_mass_fits(
        combined_data,
        n_bins=80,
        output_dir="./",
        mass_range=(5200, 5400)
    )
    
    return simple_results, roofit_results
