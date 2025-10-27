"""
LHCb style plotting utility.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
# Turn off interactive plotting to avoid displaying plots over SSH
plt.ioff()
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
import awkward as ak
import ROOT
from ROOT import TCanvas, TH1F, TF1, gStyle, kRed, kBlue, kGreen, TLegend
from ROOT import RooRealVar, RooDataHist, RooDataSet, RooArgList, RooArgSet, RooFit
from ROOT import RooGaussian, RooPolynomial, RooAddPdf, RooHistPdf, kDashed

# Disable ROOT GUI to avoid display issues over SSH
ROOT.gROOT.SetBatch(True)

# Set up LHCb plot style
def set_lhcb_style():
    """Set the standard LHCb plot style."""
    # General appearance
    plt.style.use('classic')
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman"]
    mpl.rcParams["font.size"] = 14
    mpl.rcParams["axes.labelsize"] = 16
    mpl.rcParams["axes.titlesize"] = 16
    mpl.rcParams["xtick.labelsize"] = 14
    mpl.rcParams["ytick.labelsize"] = 14
    mpl.rcParams["legend.fontsize"] = 12
    mpl.rcParams["figure.figsize"] = (10, 8)
    mpl.rcParams["figure.dpi"] = 100
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.alpha"] = 0.3
    mpl.rcParams["lines.linewidth"] = 2
    mpl.rcParams["errorbar.capsize"] = 3
    
    # ROOT style for RooFit and PyROOT plots
    gStyle.SetOptStat(0)
    gStyle.SetOptFit(0)
    gStyle.SetPadTopMargin(0.05)
    gStyle.SetPadRightMargin(0.05)
    gStyle.SetPadBottomMargin(0.15)
    gStyle.SetPadLeftMargin(0.15)
    gStyle.SetTitleOffset(1.2, "x")
    gStyle.SetTitleOffset(1.4, "y")
    gStyle.SetTitleFont(132, "xyz")
    gStyle.SetTitleFont(132, "")
    gStyle.SetTitleSize(0.05, "xyz")
    gStyle.SetTitleSize(0.05, "")
    gStyle.SetLabelFont(132, "xyz")
    gStyle.SetLabelSize(0.045, "xyz")
    gStyle.SetHistLineWidth(2)
    gStyle.SetGridStyle(3)
    gStyle.SetGridColor(ROOT.kGray+1)
    gStyle.SetGridWidth(1)
    gStyle.SetFrameBorderMode(0)
    gStyle.SetCanvasBorderMode(0)
    gStyle.SetPadBorderMode(0)
    gStyle.SetLegendBorderSize(0)
    gStyle.SetLegendTextSize(0.04)
    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)
    gStyle.SetEndErrorSize(5)

def add_lhcb_label(ax, simulation=False, status="", data_label="", pos="left"):
    """
    Add the LHCb label to the plot.
    
    Args:
        ax: Matplotlib axis
        simulation: Whether to add "Simulation" to the label
        status: Status text ("Preliminary", "Work in Progress", etc.)
        data_label: Additional data label (e.g., "5 fb⁻¹, 13 TeV")
        pos: Position of the label ("left", "right")
    """
    if pos == "left":
        x_pos = 0.05
        align = "left"
    else:
        x_pos = 0.95
        align = "right"
    
    # Add LHCb label
    text = "LHCb"
    if simulation:
        text += " Simulation"
    if status:
        text += f" {status}"
    
    ax.text(x_pos, 0.93, text, transform=ax.transAxes, fontsize=20, 
            horizontalalignment=align, verticalalignment="top")
    
    # Add data label if provided
    if data_label:
        ax.text(x_pos, 0.86, data_label, transform=ax.transAxes, fontsize=16,
                horizontalalignment=align, verticalalignment="top", style="italic")

def create_lhcb_figure(figsize=(10, 8), nrows=1, ncols=1):
    """Create a figure with LHCb style and return fig, ax."""
    set_lhcb_style()
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    return fig, ax

def finalize_lhcb_figure(ax, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, 
                         simulation=False, status="", data_label="", legend=True,
                         legend_loc="best", grid=True, tight_layout=True, minor_ticks=True,
                         pos="left"):
    """
    Finalize a LHCb style figure by adding labels, legends, etc.
    
    Args:
        ax: Matplotlib axis
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        xlim: X-axis limits
        ylim: Y-axis limits
        simulation: Whether to add "Simulation" to the LHCb label
        status: Status text ("Preliminary", "Work in Progress", etc.)
        data_label: Additional data label (e.g., "5 fb⁻¹, 13 TeV")
        legend: Whether to show legend
        legend_loc: Legend location
        grid: Whether to show grid
        tight_layout: Whether to use tight_layout
        minor_ticks: Whether to show minor ticks
    """
    if title:
        ax.set_title(title, fontsize=18)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=16)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    
    # Show legend if requested
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc=legend_loc, frameon=False, fontsize=14)
    
    # Grid and minor ticks
    ax.grid(grid, alpha=0.3)
    if minor_ticks:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='both', which='minor', length=2)
        ax.tick_params(axis='both', which='major', length=4)
    
    if tight_layout:
        plt.tight_layout()

def plot_histogram(data, bins=50, range=None, label=None, ax=None, histtype="step", density=False, 
                  color=None, alpha=1.0, weights=None, log_y=False, fillstyle=None):
    """
    Plot a histogram with LHCb style.
    
    Args:
        data: Data to plot
        bins: Number of bins or bin edges
        range: Data range (min, max)
        label: Data label for legend
        ax: Matplotlib axis (created if None)
        histtype: Histogram type ("bar", "step", "stepfilled")
        density: Whether to normalize the histogram
        color: Histogram color
        alpha: Histogram transparency
        weights: Weights for histogram entries
        log_y: Whether to use log scale for y-axis
        fillstyle: Fill style for histogram
        
    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = create_lhcb_figure()
    
    # Convert awkward array to numpy if needed
    if isinstance(data, (ak.Array, ak.Record)):
        data = ak.to_numpy(data)
    
    # Plot histogram
    hist_kwargs = {
        'bins': bins,
        'range': range,
        'histtype': histtype,
        'density': density,
        'alpha': alpha,
        'label': label,
    }
    
    if color:
        hist_kwargs['color'] = color
    if weights is not None:
        hist_kwargs['weights'] = weights
    if fillstyle:
        hist_kwargs['fillstyle'] = fillstyle
    
    ax.hist(data, **hist_kwargs)
    
    if log_y:
        ax.set_yscale('log')
    
    return ax

def plot_2d_histogram(x_data, y_data, bins=50, range=None, ax=None, colorbar=True, 
                     cbar_label=None, cmap='viridis', log_scale=False):
    """
    Plot a 2D histogram with LHCb style.
    
    Args:
        x_data: X-axis data
        y_data: Y-axis data
        bins: Number of bins or (x_bins, y_bins)
        range: Data range ((x_min, x_max), (y_min, y_max))
        ax: Matplotlib axis (created if None)
        colorbar: Whether to show colorbar
        cbar_label: Colorbar label
        cmap: Colormap
        log_scale: Whether to use log scale for color
        
    Returns:
        Matplotlib axis and hist (for colorbar)
    """
    if ax is None:
        fig, ax = create_lhcb_figure()
    
    # Convert awkward arrays to numpy if needed
    if isinstance(x_data, (ak.Array, ak.Record)):
        x_data = ak.to_numpy(x_data)
    if isinstance(y_data, (ak.Array, ak.Record)):
        y_data = ak.to_numpy(y_data)
    
    hist_kwargs = {
        'bins': bins,
        'range': range,
        'cmap': cmap,
    }
    
    if log_scale:
        norm = mpl.colors.LogNorm()
        hist_kwargs['norm'] = norm
    
    h = ax.hist2d(x_data, y_data, **hist_kwargs)
    
    if colorbar:
        cbar = plt.colorbar(h[3], ax=ax)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=14)
    
    return ax, h

def plot_multiple_histograms(data_list, labels, bins=50, range=None, ax=None, colors=None, 
                           alpha=0.7, histtype="step", density=False, stacked=False):
    """
    Plot multiple histograms on the same axis.
    
    Args:
        data_list: List of data arrays
        labels: List of labels for legend
        bins: Number of bins or bin edges
        range: Data range (min, max)
        ax: Matplotlib axis (created if None)
        colors: List of colors for histograms
        alpha: Histogram transparency
        histtype: Histogram type ("bar", "step", "stepfilled")
        density: Whether to normalize the histograms
        stacked: Whether to stack the histograms
        
    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = create_lhcb_figure()
    
    # Convert awkward arrays to numpy if needed
    data_arrays = []
    for data in data_list:
        if isinstance(data, (ak.Array, ak.Record)):
            data_arrays.append(ak.to_numpy(data))
        else:
            data_arrays.append(data)
    
    hist_kwargs = {
        'bins': bins,
        'range': range,
        'histtype': histtype,
        'density': density,
        'alpha': alpha,
        'stacked': stacked,
        'label': labels,
    }
    
    if colors:
        hist_kwargs['color'] = colors
    
    ax.hist(data_arrays, **hist_kwargs)
    
    return ax

def fit_histogram_with_pyroot(data, bins=100, range_min=None, range_max=None, 
                              title="", fit_function="gaus(0) + pol1(3)", 
                              initial_params=None, parameter_names=None,
                              save_path=None, show_components=True):
    """
    Fit a histogram with PyROOT.
    
    Args:
        data: Data to fit (awkward array or numpy array)
        bins: Number of bins
        range_min: Lower fit range
        range_max: Upper fit range
        title: Plot title
        fit_function: Function to fit (ROOT syntax)
        initial_params: Initial parameters for fit
        parameter_names: Names of parameters for display
        save_path: Path to save the plot
        show_components: Whether to show individual fit components
        
    Returns:
        histogram, fit function, and canvas
    """
    # Set LHCb ROOT style
    set_lhcb_style()
    
    # Extract data
    if isinstance(data, (ak.Array, ak.Record)):
        values = ak.to_numpy(data)
    else:
        values = data
    
    # Determine range if not provided
    if range_min is None:
        range_min = np.min(values)
    if range_max is None:
        range_max = np.max(values)
    
    # Create and fill histogram
    hist_name = "h_" + title.replace(" ", "_").replace("-", "_")
    hist = TH1F(hist_name, title, bins, range_min, range_max)
    
    for val in values:
        if range_min <= val <= range_max:
            hist.Fill(val)
    
    # Set histogram style
    hist.SetLineWidth(2)
    hist.SetLineColor(kBlue)
    hist.SetMarkerStyle(20)
    hist.SetMarkerSize(0.7)
    
    # Create fit function
    fit_func = TF1("fit_" + hist_name, fit_function, range_min, range_max)
    
    # Set initial parameters
    if initial_params is None:
        # Default parameters for Gaussian + linear background
        mean_guess = np.mean(values)
        sigma_guess = np.std(values)
        max_bin_content = hist.GetMaximum()
        
        fit_func.SetParameters(
            max_bin_content * 0.8,  # Gaussian amplitude
            mean_guess,             # Gaussian mean
            sigma_guess,            # Gaussian sigma
            0,                      # Linear slope
            max_bin_content * 0.2   # Linear intercept
        )
    else:
        for i, param in enumerate(initial_params):
            fit_func.SetParameter(i, param)
    
    # Set parameter names
    if parameter_names:
        for i, name in enumerate(parameter_names):
            fit_func.SetParName(i, name)
    
    # Perform the fit
    fit_result = hist.Fit(fit_func, "SRQ", "")  # Q for quiet, S for save result, R for fit range
    
    # Create canvas
    canvas = TCanvas("c_" + hist_name, title, 800, 600)
    
    # Draw the histogram and fit
    hist.Draw("E")
    
    # Add LHCb label
    lhcb_label = ROOT.TPaveText(0.18, 0.82, 0.55, 0.92, "NDC")
    lhcb_label.SetFillStyle(0)
    lhcb_label.SetBorderSize(0)
    lhcb_label.SetTextFont(132)
    lhcb_label.SetTextSize(0.05)
    lhcb_label.SetTextAlign(12)
    lhcb_label.AddText("LHCb")
    lhcb_label.Draw()
    
    # Extract and draw fit components if requested
    if show_components and "gaus" in fit_function and "pol" in fit_function:
        # For Gaussian + polynomial background
        if "gaus(0)" in fit_function:
            signal_idx = 0
            if "pol1" in fit_function:
                bkg_idx = 3
            elif "pol2" in fit_function:
                bkg_idx = 4
            else:  # Default to pol1
                bkg_idx = 3
        else:
            # Try to determine component indices from function string
            signal_idx = int(fit_function.split("gaus(")[1].split(")")[0])
            if "pol1" in fit_function:
                bkg_idx = int(fit_function.split("pol1(")[1].split(")")[0])
            elif "pol2" in fit_function:
                bkg_idx = int(fit_function.split("pol2(")[1].split(")")[0])
            else:
                bkg_idx = signal_idx + 3  # Guess
        
        # Signal component
        signal_func = TF1("signal_" + hist_name, "gaus", range_min, range_max)
        signal_func.SetParameters(
            fit_func.GetParameter(signal_idx),
            fit_func.GetParameter(signal_idx + 1),
            fit_func.GetParameter(signal_idx + 2)
        )
        signal_func.SetLineColor(kRed)
        signal_func.SetLineStyle(kDashed)
        signal_func.Draw("same")
        
        # Background component
        if "pol1" in fit_function:
            bkg_func = TF1("bkg_" + hist_name, "pol1", range_min, range_max)
            bkg_func.SetParameters(
                fit_func.GetParameter(bkg_idx),
                fit_func.GetParameter(bkg_idx + 1)
            )
        elif "pol2" in fit_function:
            bkg_func = TF1("bkg_" + hist_name, "pol2", range_min, range_max)
            bkg_func.SetParameters(
                fit_func.GetParameter(bkg_idx),
                fit_func.GetParameter(bkg_idx + 1),
                fit_func.GetParameter(bkg_idx + 2)
            )
        else:
            bkg_func = TF1("bkg_" + hist_name, "pol1", range_min, range_max)
            bkg_func.SetParameters(
                fit_func.GetParameter(bkg_idx),
                fit_func.GetParameter(bkg_idx + 1)
            )
        
        bkg_func.SetLineColor(kGreen+2)
        bkg_func.SetLineStyle(kDashed)
        bkg_func.Draw("same")
        
        # Add legend
        legend = TLegend(0.65, 0.65, 0.89, 0.82)
        legend.AddEntry(hist, "MC Data", "lep")
        legend.AddEntry(fit_func, "Fit", "l")
        legend.AddEntry(signal_func, "Signal", "l")
        legend.AddEntry(bkg_func, "Background", "l")
        legend.SetBorderSize(0)
        legend.Draw()
    
    canvas.Update()
    
    # Save the plot if requested
    if save_path:
        canvas.SaveAs(save_path)
    
    # Print fit results
    print(f"Fit results for {title}:")
    for i in range(fit_func.GetNpar()):
        name = fit_func.GetParName(i)
        value = fit_func.GetParameter(i)
        error = fit_func.GetParError(i)
        print(f"  {name}: {value:.4f} ± {error:.4f}")
    print(f"  Chi2/NDF: {fit_func.GetChisquare():.2f}/{fit_func.GetNDF()} = {fit_func.GetChisquare()/fit_func.GetNDF():.2f}")
    
    # Calculate signal yield if this is a signal+background fit
    if "gaus" in fit_function and show_components:
        bin_width = (range_max - range_min) / bins
        signal_integral = signal_func.Integral(range_min, range_max) / bin_width
        signal_error = signal_integral * (signal_func.GetParError(0) / signal_func.GetParameter(0))
        print(f"  Signal yield: {signal_integral:.1f} ± {signal_error:.1f}")
    
    return hist, fit_func, canvas

def fit_with_roofit(data, bins=100, range_min=None, range_max=None, 
                   variable_name="variable", variable_title="Variable",
                   fit_model="gauss+pol1", title="RooFit Result",
                   save_path=None, pull_plot=True):
    """
    Fit a distribution using RooFit.
    
    Args:
        data: Data to fit (awkward array or numpy array)
        bins: Number of bins
        range_min: Lower fit range
        range_max: Upper fit range
        variable_name: RooFit variable name
        variable_title: Variable title for axis label
        fit_model: Model type ("gauss+pol1", "gauss+pol2", "gauss+exp", "double_gauss+pol1")
        title: Plot title
        save_path: Path to save the plot
        pull_plot: Whether to include pull plot
        
    Returns:
        fit_result, canvas, model
    """
    # Set LHCb ROOT style
    set_lhcb_style()
    
    # Extract data
    if isinstance(data, (ak.Array, ak.Record)):
        values = ak.to_numpy(data)
    else:
        values = data
    
    # Determine range if not provided
    if range_min is None:
        range_min = np.min(values)
    if range_max is None:
        range_max = np.max(values)
    
    # Create RooFit variable
    x = RooRealVar(variable_name, variable_title, range_min, range_max)
    
    # Create dataset
    data_list = []
    for val in values:
        if range_min <= val <= range_max:
            x.setVal(val)
            data_list.append(x.Clone())
    
    dataset = RooDataSet("dataset", f"{variable_title} Dataset", RooArgSet(x))
    for point in data_list:
        dataset.add(RooArgSet(point))
    
    # Create fit model based on specified type
    if fit_model == "gauss+pol1":
        # Gaussian signal + linear background
        mean = RooRealVar("mean", "Mean", np.mean(values), range_min, range_max)
        sigma = RooRealVar("sigma", "Sigma", np.std(values), 0.1, 100)
        signal = RooGaussian("signal", "Signal", x, mean, sigma)
        
        # Use more stable parameterization for the background polynomial
        # For a linear polynomial: f(x) = 1 + c1*(x - x_mid)/(x_max - x_min)
        x_mid = (range_max + range_min)/2
        x_range = (range_max - range_min)
        poly_var = RooFormulaVar("poly_var", f"(x-{x_mid})/{x_range}", RooArgList(x))
        c1 = RooRealVar("c1", "c1", 0.1, -1, 1)  # Slope parameter with tighter bounds
        bkg = ROOT.RooPolynomial("bkg", "Background", poly_var, RooArgList(c1))
        
        sig_yield = RooRealVar("sig_yield", "Signal Yield", len(values)*0.8, 0, len(values)*2)
        bkg_yield = RooRealVar("bkg_yield", "Background Yield", len(values)*0.2, 0, len(values)*2)
        
        model = RooAddPdf("model", "Signal + Background", 
                          RooArgList(signal, bkg), 
                          RooArgList(sig_yield, bkg_yield))
    
    elif fit_model == "gauss+exp":
        # Gaussian signal + exponential background (more stable for B mass)
        mean = RooRealVar("mean", "Mean", np.mean(values), range_min, range_max)
        sigma = RooRealVar("sigma", "Sigma", np.std(values), 0.1, 100)
        signal = RooGaussian("signal", "Signal", x, mean, sigma)
        
        # Exponential background
        alpha = RooRealVar("alpha", "Decay Constant", -0.001, -0.1, 0.0)
        bkg = ROOT.RooExponential("bkg", "Background", x, alpha)
        
        sig_yield = RooRealVar("sig_yield", "Signal Yield", len(values)*0.8, 0, len(values)*2)
        bkg_yield = RooRealVar("bkg_yield", "Background Yield", len(values)*0.2, 0, len(values)*2)
        
        model = RooAddPdf("model", "Signal + Background", 
                          RooArgList(signal, bkg), 
                          RooArgList(sig_yield, bkg_yield))
    
    elif fit_model == "double_gauss+pol1":
        # Double Gaussian signal + linear background
        mean1 = RooRealVar("mean1", "Mean 1", np.mean(values), range_min, range_max)
        sigma1 = RooRealVar("sigma1", "Sigma 1", np.std(values)*0.7, 0.1, 100)
        gauss1 = RooGaussian("gauss1", "Gaussian 1", x, mean1, sigma1)
        
        mean2 = RooRealVar("mean2", "Mean 2", np.mean(values), range_min, range_max)
        sigma2 = RooRealVar("sigma2", "Sigma 2", np.std(values)*1.5, 0.1, 100)
        gauss2 = RooGaussian("gauss2", "Gaussian 2", x, mean2, sigma2)
        
        frac = RooRealVar("frac", "Fraction", 0.7, 0.0, 1.0)
        signal = ROOT.RooAddPdf("signal", "Double Gaussian Signal", 
                               RooArgList(gauss1, gauss2), RooArgList(frac))
        
        # Use more stable parameterization for the background polynomial
        x_mid = (range_max + range_min)/2
        x_range = (range_max - range_min)
        poly_var = RooFormulaVar("poly_var", f"(x-{x_mid})/{x_range}", RooArgList(x))
        c1 = RooRealVar("c1", "c1", 0.1, -1, 1)  # Slope parameter
        bkg = ROOT.RooPolynomial("bkg", "Background", poly_var, RooArgList(c1))
        
        sig_yield = RooRealVar("sig_yield", "Signal Yield", len(values)*0.8, 0, len(values)*2)
        bkg_yield = RooRealVar("bkg_yield", "Background Yield", len(values)*0.2, 0, len(values)*2)
        
        model = RooAddPdf("model", "Signal + Background", 
                          RooArgList(signal, bkg), 
                          RooArgList(sig_yield, bkg_yield))
    
    else:
        # Default to Gaussian + exponential background which is more stable than polynomial for B mass
        mean = RooRealVar("mean", "Mean", np.mean(values), range_min, range_max)
        sigma = RooRealVar("sigma", "Sigma", np.std(values), 0.1, 100)
        signal = RooGaussian("signal", "Signal", x, mean, sigma)
        
        # Use exponential instead of polynomial for better stability
        alpha = RooRealVar("alpha", "alpha", -0.001, -0.1, 0.0)
        bkg = ROOT.RooExponential("bkg", "Background", x, alpha)
        
        sig_yield = RooRealVar("sig_yield", "Signal Yield", len(values)*0.8, 0, len(values)*2)
        bkg_yield = RooRealVar("bkg_yield", "Background Yield", len(values)*0.2, 0, len(values)*2)
        
        model = RooAddPdf("model", "Signal + Background", 
                         RooArgList(signal, bkg), 
                         RooArgList(sig_yield, bkg_yield))
    
    # Perform the fit
    try:
        # Try extended likelihood fit first
        fit_result = model.fitTo(dataset, RooFit.Save(), RooFit.PrintLevel(-1), RooFit.Extended(True))
    except Exception as e:
        print(f"Extended likelihood fit failed: {e}")
        try:
            # If that fails, try standard likelihood fit
            fit_result = model.fitTo(dataset, RooFit.Save(), RooFit.PrintLevel(-1), RooFit.Extended(False))
        except Exception as e:
            print(f"Standard fit also failed: {e}")
            print("Proceeding with plot creation using initial parameter values")
            fit_result = None
    
    
    # Create canvas for plotting
    if pull_plot:
        canvas = TCanvas("c_roofit", title, 900, 700)
        canvas.Divide(1, 2)
        pad1 = canvas.cd(1)
        pad1.SetPad(0, 0.3, 1, 1.0)
        pad1.SetBottomMargin(0.01)
    else:
        canvas = TCanvas("c_roofit", title, 800, 600)
    
    # Create frame for plotting
    frame = x.frame(RooFit.Title(title))
    
    # Plot data and fit on the frame
    dataset.plotOn(frame, RooFit.Name("data"))
    
    # Only plot the model if fit was successful
    if fit_result is not None:
        model.plotOn(frame, RooFit.Name("model"))
        
        # Plot fit components if they exist
        if "bkg" in locals():
            model.plotOn(frame, RooFit.Components("bkg"), RooFit.LineStyle(kDashed), 
                        RooFit.LineColor(kGreen+2), RooFit.Name("bkg"))
        
        if fit_model == "double_gauss+pol1":
            model.plotOn(frame, RooFit.Components("gauss1"), RooFit.LineStyle(kDashed), 
                        RooFit.LineColor(kRed), RooFit.Name("gauss1"))
            model.plotOn(frame, RooFit.Components("gauss2"), RooFit.LineStyle(kDashed), 
                        RooFit.LineColor(kRed+2), RooFit.Name("gauss2"))
        else:
            model.plotOn(frame, RooFit.Components("signal"), RooFit.LineStyle(kDashed), 
                        RooFit.LineColor(kRed), RooFit.Name("signal"))
        
        # Add parameter info
        model.paramOn(frame, RooFit.Layout(0.60, 0.9, 0.89))
    
    # Draw the frame
    frame.Draw()
    
    # Add LHCb label
    lhcb_label = ROOT.TPaveText(0.18, 0.82, 0.55, 0.92, "NDC")
    lhcb_label.SetFillStyle(0)
    lhcb_label.SetBorderSize(0)
    lhcb_label.SetTextFont(132)
    lhcb_label.SetTextSize(0.05)
    lhcb_label.SetTextAlign(12)
    lhcb_label.AddText("LHCb")
    lhcb_label.Draw()
    
    # Add legend
    legend = TLegend(0.65, 0.55, 0.89, 0.65)
    legend.AddEntry(frame.findObject("data"), "MC Data", "p")
    if fit_result is not None:
        legend.AddEntry(frame.findObject("model"), "Fit Model", "l")
        if "bkg" in locals():
            legend.AddEntry(frame.findObject("bkg"), "Background", "l")
        if fit_model == "double_gauss+pol1":
            legend.AddEntry(frame.findObject("gauss1"), "Signal (core)", "l")
            legend.AddEntry(frame.findObject("gauss2"), "Signal (tail)", "l")
        else:
            legend.AddEntry(frame.findObject("signal"), "Signal", "l")
    legend.SetBorderSize(0)
    legend.Draw()
    
    # Add pull plot if requested and fit was successful
    if pull_plot and fit_result is not None:
        canvas.cd(2)
        pad2 = canvas.cd(2)
        pad2.SetPad(0, 0.0, 1, 0.3)
        pad2.SetTopMargin(0.01)
        pad2.SetBottomMargin(0.3)
        
        pull_frame = x.frame(RooFit.Title("Pull Distribution"))
        pull_hist = frame.pullHist()
        pull_frame.addPlotable(pull_hist, "P")
        pull_frame.SetYTitle("Pull")
        pull_frame.SetLabelSize(0.1, "Y")
        pull_frame.SetTitleSize(0.1, "Y")
        pull_frame.SetLabelSize(0.1, "X")
        pull_frame.SetTitleSize(0.1, "X")
        pull_frame.SetTitleOffset(0.4, "Y")
        pull_frame.SetTitleOffset(1.0, "X")
        pull_frame.Draw()
        
        # Add horizontal line at pull=0
        zero_line = ROOT.TLine(range_min, 0, range_max, 0)
        zero_line.SetLineColor(ROOT.kRed)
        zero_line.SetLineStyle(ROOT.kDashed)
        zero_line.Draw()
    
    canvas.Update()
    
    # Save the plot if requested
    if save_path:
        canvas.SaveAs(save_path)
    
    # Print fit results
    print(f"RooFit results for {title}:")
    
    if fit_result is not None:
        # Print specific parameters based on the model type
        if fit_model == "double_gauss+pol1":
            print(f"  Core Gaussian Mean: {mean1.getVal():.2f} ± {mean1.getError():.2f}")
            print(f"  Core Gaussian Sigma: {sigma1.getVal():.2f} ± {sigma1.getError():.2f}")
            print(f"  Tail Gaussian Mean: {mean2.getVal():.2f} ± {mean2.getError():.2f}")
            print(f"  Tail Gaussian Sigma: {sigma2.getVal():.2f} ± {sigma2.getError():.2f}")
            print(f"  Core Fraction: {frac.getVal():.2f} ± {frac.getError():.2f}")
        else:
            print(f"  Mean: {mean.getVal():.2f} ± {mean.getError():.2f}")
            print(f"  Sigma: {sigma.getVal():.2f} ± {sigma.getError():.2f}")
        
        print(f"  Signal Yield: {sig_yield.getVal():.1f} ± {sig_yield.getError():.1f}")
        print(f"  Background Yield: {bkg_yield.getVal():.1f} ± {bkg_yield.getError():.1f}")
    else:
        print("  Fit did not converge successfully, no reliable parameters available")
    
    return fit_result, canvas, model
    
def create_summary_table(data_dict, quantities, labels=None, title="Summary Statistics"):
    """
    Create a summary table from a dictionary of datasets.
    
    Args:
        data_dict: Dictionary of datasets (e.g., {"DD": dd_data, "LL": ll_data})
        quantities: List of quantities to compute (e.g., ["mean", "std"])
        labels: Labels for each quantity (optional)
        title: Table title
        
    Returns:
        matplotlib figure with table
    """
    if labels is None:
        labels = quantities
    
    # Create a figure for the table
    fig, ax = plt.subplots(figsize=(10, len(data_dict) + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    columns = ["Data Sample"] + list(data_dict.keys())
    rows = []
    
    for i, quantity in enumerate(quantities):
        row = [labels[i]]
        for name, data in data_dict.items():
            if callable(quantity):
                value = quantity(data)
            elif hasattr(np, quantity):
                value = getattr(np, quantity)(data)
            else:
                value = f"N/A"
            
            # Format output based on value type
            if isinstance(value, (int, np.integer)):
                row.append(f"{value}")
            elif isinstance(value, (float, np.floating)):
                row.append(f"{value:.2f}")
            else:
                row.append(str(value))
        
        rows.append(row)
    
    # Create table
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Add title
    ax.set_title(title, fontsize=16, pad=20)
    
    plt.tight_layout()
    
    return fig

def plot_pid_distributions(data, particle_names, pid_variables, bins=50, figsize=(15, 10)):
    """
    Plot PID variable distributions for multiple particles.
    
    Args:
        data: Data containing PID variables
        particle_names: List of particle names (e.g., ["K", "Pi"])
        pid_variables: List of PID variables to plot (e.g., ["PIDK", "PIDp"])
        bins: Number of bins for histograms
        figsize: Figure size
        
    Returns:
        Figure with PID variable plots
    """
    n_particles = len(particle_names)
    n_variables = len(pid_variables)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_variables, n_particles, figsize=figsize)
    
    # Make axes accessible for single row or column
    if n_variables == 1 and n_particles == 1:
        axes = np.array([[axes]])
    elif n_variables == 1:
        axes = axes.reshape(1, -1)
    elif n_particles == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each PID variable for each particle
    for i, var in enumerate(pid_variables):
        for j, particle in enumerate(particle_names):
            ax = axes[i, j]
            
            # Construct variable name (e.g., "K_PIDK", "Pi_PIDp")
            full_var = f"{particle}_{var}"
            
            if full_var in data.fields:
                # Plot histogram
                plot_histogram(data[full_var], bins=bins, ax=ax)
                
                # Add labels and title
                ax.set_xlabel(f"{var}")
                ax.set_ylabel("Events")
                ax.set_title(f"{full_var}")
                
            else:
                ax.text(0.5, 0.5, f"Variable {full_var} not found", 
                        ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    return fig

def plot_cc_regions(data, x_var, y_data=None, z_var=None, bins=50, x_range=None, y_range=None,
                   figsize=(12, 10), cc_regions=None, log_scale=False):
    """
    Plot correlated-channel (CC) regions using 2D histograms.
    
    Args:
        data: Data containing variables
        x_var: X-axis variable name or array
        y_data: Y-axis data as a numpy array (if provided, overrides y_var)
        z_var: Variable for color (optional)
        bins: Number of bins
        x_range: X-axis range
        y_range: Y-axis range
        figsize: Figure size
        cc_regions: List of tuples defining CC regions [(x_min, x_max, y_min, y_max, label), ...]
        log_scale: Whether to use log scale for color
        
    Returns:
        Figure with CC regions plot
    """
    # Create figure
    fig, ax = create_lhcb_figure(figsize=figsize)
    
    # Get x data
    if isinstance(x_var, str):
        # x_var is a field name
        x_data = ak.to_numpy(data[x_var]) if isinstance(data[x_var], (ak.Array, ak.Record)) else data[x_var]
    else:
        # x_var is already data
        x_data = x_var
    
    # Plot 2D histogram
    if z_var is None:
        hist_kwargs = {
            'bins': bins,
            'cmap': 'viridis',
        }
        
        if x_range is not None and y_range is not None:
            hist_kwargs['range'] = (x_range, y_range)
        
        if log_scale:
            hist_kwargs['norm'] = mpl.colors.LogNorm()
        
        h = ax.hist2d(x_data, y_data, **hist_kwargs)
        plt.colorbar(h[3], ax=ax, label="Events")
    else:
        z_data = ak.to_numpy(data[z_var]) if isinstance(data[z_var], (ak.Array, ak.Record)) else data[z_var]
        sc = ax.scatter(x_data, y_data, c=z_data, cmap='viridis', alpha=0.7, 
                       edgecolors='none', s=5)
        plt.colorbar(sc, ax=ax, label=z_var)
    
    # Add CC regions if provided
    if cc_regions:
        for region in cc_regions:
            x_min, x_max, y_min, y_max, label = region
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               fill=False, edgecolor='red', linestyle='--', linewidth=2)
            ax.add_patch(rect)
            ax.text(x_min + (x_max - x_min) / 2, y_max + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                   label, ha='center', va='bottom', color='red', fontsize=12)
    
    return fig, ax

def apply_cuts_and_plot(data, cuts, variable, bins=50, range=None, title=None, 
                       xlabel=None, ylabel="Events", figsize=(10, 8),
                       compare_before=True, cut_labels=None):
    """
    Apply cuts to data and plot the resulting distribution.
    
    Args:
        data: Data to apply cuts to
        cuts: List of cut strings (e.g., ["B_PT > 1000", "B_IPCHI2_OWNPV < 25"])
        variable: Variable to plot
        bins: Number of bins
        range: Plot range
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        compare_before: Whether to plot the distribution before cuts
        cut_labels: Labels for each cut (for legend)
        
    Returns:
        Figure with plot and cut information
    """
    # Create figure
    fig, ax = create_lhcb_figure(figsize=figsize)
    
    # Plot before cuts if requested
    if compare_before:
        plot_histogram(data[variable], bins=bins, range=range, 
                      label="Before cuts", ax=ax, histtype="step", 
                      color="blue", alpha=0.7)
    
    # Apply cuts sequentially
    current_data = data
    cut_results = []
    
    for i, cut in enumerate(cuts):
        # Parse and apply cut
        # This is a simple implementation - in a real analysis you'd use a more robust approach
        field, op, value = cut.split(' ', 2)
        
        if op == '>':
            mask = current_data[field] > float(value)
        elif op == '>=':
            mask = current_data[field] >= float(value)
        elif op == '<':
            mask = current_data[field] < float(value)
        elif op == '<=':
            mask = current_data[field] <= float(value)
        elif op == '==':
            mask = current_data[field] == float(value)
        elif op == '!=':
            mask = current_data[field] != float(value)
        else:
            raise ValueError(f"Unsupported operator: {op}")
        
        current_data = current_data[mask]
        cut_results.append(current_data)
        
        # Plot intermediate result if multiple cuts
        if len(cuts) > 1 and i < len(cuts) - 1:
            label = f"After cut {i+1}" if cut_labels is None else cut_labels[i]
            color = plt.cm.tab10(i / 10)
            plot_histogram(current_data[variable], bins=bins, range=range,
                          label=label, ax=ax, histtype="step",
                          color=color, alpha=0.7)
    
    # Plot final result
    if len(cuts) > 1:
        label = "After all cuts" if cut_labels is None else cut_labels[-1]
        # Use plot_histogram and properly pass parameters that it accepts
        plot_histogram(current_data[variable], bins=bins, range=range,
                      label=label, ax=ax, histtype="step",
                      color="red", alpha=1.0)
    else:
        label = "After cut" if cut_labels is None else cut_labels[0]
        # Use plot_histogram and properly pass parameters that it accepts
        plot_histogram(current_data[variable], bins=bins, range=range,
                      label=label, ax=ax, histtype="step",
                      color="red", alpha=1.0)
    
    # Calculate efficiency
    initial_count = len(data)
    final_count = len(current_data)
    efficiency = final_count / initial_count if initial_count > 0 else 0
    
    # Add efficiency text
    efficiency_text = f"Efficiency: {efficiency:.2%} ({final_count}/{initial_count})"
    ax.text(0.95, 0.05, efficiency_text, transform=ax.transAxes,
           horizontalalignment='right', verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add cut information
    cut_text = "Applied cuts:\n" + "\n".join(cuts)
    ax.text(0.05, 0.95, cut_text, transform=ax.transAxes,
           horizontalalignment='left', verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           fontsize=10)
    
    # Finalize plot
    if title is None:
        title = f"{variable} Distribution with Cuts"
    if xlabel is None:
        xlabel = variable
    
    finalize_lhcb_figure(ax, title=title, xlabel=xlabel, ylabel=ylabel,
                       simulation=False, status="Preliminary")
    
    return fig, current_data