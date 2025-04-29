import yaml, ROOT, json
from ROOT import RooRealVar, RooGaussian, RooChebychev, RooAddPdf, RooArgList, RooDataSet
from ROOT import RooFit, RooArgSet, RooCBShape, TH1F, TLegend, TLatex
import os
import numpy as np
from loaders import load_data
from selections import trigger_mask
from branches import canonical

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

def create_selection_mask(data, sample):
    """
    Create selection mask for data based on sample type
    
    Parameters:
    -----------
    data : awkward array
        Data to apply selection to
    sample : str
        'signal' or 'norm' to determine which selection to apply
    
    Returns:
    --------
    boolean mask
        Mask to apply to data
    """
    is_sig = sample == "signal"
    mask = np.ones(len(data), dtype=bool)  # Start with all True
    
    # Get correct branch names based on sample
    if is_sig:
        # Signal-specific branch names
        l0_endvertex_z = "L0_ENDVERTEX_Z"
        l0_ownpv_z = "L0_OWNPV_Z"
        l0_fdchi2 = "L0_FDCHI2_OWNPV"
        l0_m = "L0_M"
        l0_proton_nn = "Lp_MC15TuneV1_ProbNNp"
        proton_nn = "p_MC15TuneV1_ProbNNp"
        h1_nn = "h1_MC15TuneV1_ProbNNk"
        h2_nn = "h2_MC15TuneV1_ProbNNk"
        bu_pt = "Bu_PT"
        bu_dtf_chi2 = "Bu_DTF_chi2"
        bu_ipchi2 = "Bu_IPCHI2_OWNPV"
        bu_fdchi2 = "Bu_FDCHI2_OWNPV"
    else:
        # Normalization-specific branch names
        l0_endvertex_z = "Ks_ENDVERTEX_Z"
        l0_ownpv_z = "Ks_OWNPV_Z"
        l0_fdchi2 = "Ks_FDCHI2_OWNPV"
        l0_m = "Ks_M"
        l0_proton_nn = "KsPi_MC15TuneV1_ProbNNpi"  # For Ks pion
        proton_nn = "P1_MC15TuneV1_ProbNNpi"  # For pion in norm channel
        h1_nn = "P0_MC15TuneV1_ProbNNk"
        h2_nn = "P2_MC15TuneV1_ProbNNk"
        bu_pt = "B_PT"
        bu_dtf_chi2 = "B_DTF_chi2"
        bu_ipchi2 = "B_IPCHI2_OWNPV"
        bu_fdchi2 = "B_FDCHI2_OWNPV"
    
    # Common cuts for both channels with branch name adaptation
    try:
        # Delta Z cut
        if l0_endvertex_z in data.fields and l0_ownpv_z in data.fields:
            mask = mask & ((data[l0_endvertex_z] - data[l0_ownpv_z]) > 20)
            print(f"\tApplied delta Z cut: {mask.sum()}/{len(mask)} remaining")
        
        # FD chi2 cut
        if l0_fdchi2 in data.fields:
            mask = mask & (data[l0_fdchi2] > 45)
            print(f"\tApplied FD chi2 cut: {mask.sum()}/{len(mask)} remaining")
        
        # Mass window cut - different for Lambda vs Ks
        if l0_m in data.fields:
            pdg_mass = 1115.6 if is_sig else 497.6  # Lambda vs Ks PDG mass
            window = 6 if is_sig else 15  # Wider window for Ks
            mask = mask & (np.abs(data[l0_m] - pdg_mass) < window)
            print(f"\tApplied {('Lambda' if is_sig else 'Ks')} mass window cut: {mask.sum()}/{len(mask)} remaining")
        
        # PID cuts
        if is_sig and proton_nn in data.fields:
            mask = mask & (data[proton_nn] > 0.05)
            print(f"\tApplied proton PID cut: {mask.sum()}/{len(mask)} remaining")
        
        if l0_proton_nn in data.fields:
            threshold = 0.2 if is_sig else 0.1  # Different threshold for proton vs pion
            mask = mask & (data[l0_proton_nn] > threshold)
            print(f"\tApplied {'Lambda proton' if is_sig else 'Ks pion'} PID cut: {mask.sum()}/{len(mask)} remaining")
        
        # Kaon PID product cut (for both channels)
        if h1_nn in data.fields and h2_nn in data.fields:
            mask = mask & ((data[h1_nn] * data[h2_nn]) > 0.04)
            print(f"\tApplied KK product cut: {mask.sum()}/{len(mask)} remaining")
        
        # B PT cut
        if bu_pt in data.fields:
            mask = mask & (data[bu_pt] > 3000)
            print(f"\tApplied B PT cut: {mask.sum()}/{len(mask)} remaining")
        
        # DTF chi2 cut
        if bu_dtf_chi2 in data.fields:
            mask = mask & (data[bu_dtf_chi2] < 30)
            print(f"\tApplied DTF chi2 cut: {mask.sum()}/{len(mask)} remaining")
        
        # IP chi2 cut
        if bu_ipchi2 in data.fields:
            mask = mask & (data[bu_ipchi2] < 10)
            print(f"\tApplied IP chi2 cut: {mask.sum()}/{len(mask)} remaining")
        
        # FD chi2 cut for B
        if bu_fdchi2 in data.fields:
            mask = mask & (data[bu_fdchi2] > 175)
            print(f"\tApplied B FD chi2 cut: {mask.sum()}/{len(mask)} remaining")
        
    except Exception as e:
        print(f"Error applying selection: {e}")
    
    return mask

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
    mass_col = canonical(sample_label, ["mass"])[0]
    # Set fit range - potentially different for signal vs norm
    fit_min, fit_max = (5200, 5400)  # Same range for both by default
    m = ROOT.RooRealVar("m", "mass", fit_min, fit_max)

    # --- Set Axis Titles ---
    m.SetTitle("M(B^{+}) [MeV/c^{2}]")

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

    # --- Define Model Parameters (potentially sample-dependent) ---
    # Signal mean - try to use PDG mass as center
    mean = ROOT.RooRealVar("mean", "mean", 5280, 5260, 5290)
    
    # Sigma might differ between channels
    sigma_init = 15 if is_sig else 12  # Slightly smaller initial sigma for norm
    sigma_min = 5 if is_sig else 3     # Tighter range for norm
    sigma_max = 30 if is_sig else 25
    sigma = ROOT.RooRealVar("sigma", "sigma", sigma_init, sigma_min, sigma_max)
    
    # Crystal Ball tail parameters
    alpha = ROOT.RooRealVar("alpha", "alpha", 1.5, 0.1, 5.0)
    n = ROOT.RooRealVar("n", "n", 2.0, 0.1, 10.0)

    # --- Background Shape Parameters ---
    c1 = ROOT.RooRealVar("c1", "c1", -0.1, -1, 1)  # 1st order coefficient (used by both)
    if is_sig:
        # Define c2 only for signal fit (2nd order Chebychev)
        c2 = ROOT.RooRealVar("c2", "c2", 0.05, -1, 1)

    # --- Yields ---
    # Use nentries (events in range) for initial guess
    nsig_init = nentries * 0.5 if nentries > 0 else 100  # Avoid 0 if no entries
    nbkg_init = nentries * 0.5 if nentries > 0 else 100
    nsig = ROOT.RooRealVar("nsig", "Nsig", nsig_init, 0, nentries*1.5 if nentries > 0 else 1000)
    nbkg = ROOT.RooRealVar("nbkg", "Nbkg", nbkg_init, 0, nentries*1.5 if nentries > 0 else 1000)

    # --- Define PDFs --- 
    # Signal model is Crystal Ball for both for now
    signal_model = ROOT.RooCBShape("signal_model", "Crystal Ball", m, mean, sigma, alpha, n)

    # Background model depends on sample
    if is_sig:
        # 2nd order Chebychev for signal
        background_model = ROOT.RooChebychev("background_model", "Chebychev Background (2nd order)", 
                                          m, ROOT.RooArgList(c1, c2))
    else:
        # 1st order Chebychev for normalization
        background_model = ROOT.RooChebychev("background_model", "Chebychev Background (1st order)", 
                                          m, ROOT.RooArgList(c1))

    # Combine signal and background
    model_comp_list = ROOT.RooArgList(signal_model, background_model)
    model_yield_list = ROOT.RooArgList(nsig, nbkg)
    model = ROOT.RooAddPdf("model", "Signal+Background", model_comp_list, model_yield_list)

    # Perform the fit with more robust settings
    fit_result = model.fitTo(
        rds, 
        ROOT.RooFit.Save(),
        ROOT.RooFit.Extended(True),
        ROOT.RooFit.PrintLevel(1),
        ROOT.RooFit.Minos(False),   # Disable Minos for stability
        ROOT.RooFit.InitialHesse(True)  # Better error estimates
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
    
    # Explicitly set plotting bins
    n_bins_plot = 50  # Let's try 50 bins for the smaller range
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
    for param in [nsig, nbkg, mean, sigma]:
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
    for param in [alpha, n]:
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
    param = c1
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
    if is_sig:
        param = c2
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

def run_all_fits():
    """Run mass fits for all combinations and compare results"""
    years = CFG.get("years", ["2016", "2017", "2018"])
    tracks = CFG.get("tracks", ["LL", "DD"])
    
    all_fit_results = {}  # Dictionary to store all results
    
    # --- Individual Fits ---
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
    
    # --- Combined Fits ---
    print("\n\n=== Fitting Combined Datasets ===")
    
    # All years for each track type
    print("\n--- Combining all years for each track type ---")
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
    
    # All tracks for each year 
    print("\n--- Combining all tracks for each year ---")
    for cls in ["sig", "norm"]:
        for y in years:
            fit_id = f"{cls}_{y}_all"
            print(f"\nFitting combined dataset: {fit_id}")
            try:
                params = fit(cls, y, 'all')
                all_fit_results[fit_id] = params
            except Exception as e:
                print(f"ERROR fitting {fit_id}: {e}")
                all_fit_results[fit_id] = {'error': str(e)}
    
    # All years and all tracks (fully combined)
    print("\n--- Combining all years and all tracks ---")
    for cls in ["sig", "norm"]:
        fit_id = f"{cls}_all_all"
        print(f"\nFitting combined dataset: {fit_id}")
        try:
            params = fit(cls, 'all', 'all')
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
    
    # First display individual results by channel
    for cls, title in [("sig", "SIGNAL CHANNEL (B+ → Λ̅0pK+K-)"), 
                       ("norm", "NORMALIZATION CHANNEL (B+ → K0sπ+K+K-)")]:
        print(f"\n{title}")
        print("-"*80)
        print(f"{'Dataset':<15} {'Yield':<20} {'Mass (MeV)':<20} {'Width (MeV)':<15} {'Significance':<12}")
        print("-"*80)
        
        # Individual year/track combinations
        for y in years:
            for tr in tracks:
                fit_id = f"{cls}_{y}_{tr}"
                if fit_id in all_fit_results and all_fit_results[fit_id] and 'error' not in all_fit_results[fit_id]:
                    res = all_fit_results[fit_id]
                    dataset = f"{y}/{tr}"
                    yield_val = f"{res['nsig']['value']:.1f} ± {res['nsig']['error']:.1f}"
                    mass = f"{res['mean']['value']:.1f} ± {res['mean']['error']:.1f}"
                    width = f"{res['sigma']['value']:.1f} ± {res['sigma']['error']:.1f}"
                    signif = f"{res.get('significance', 0):.2f}σ"
                    print(f"{dataset:<15} {yield_val:<20} {mass:<20} {width:<15} {signif:<12}")
        
        # Combined by year (all tracks)
        print("-"*80 + "\nCombined by year (all tracks):")
        for y in years:
            fit_id = f"{cls}_{y}_all"
            if fit_id in all_fit_results and all_fit_results[fit_id] and 'error' not in all_fit_results[fit_id]:
                res = all_fit_results[fit_id]
                dataset = f"{y}/all"
                yield_val = f"{res['nsig']['value']:.1f} ± {res['nsig']['error']:.1f}"
                mass = f"{res['mean']['value']:.1f} ± {res['mean']['error']:.1f}"
                width = f"{res['sigma']['value']:.1f} ± {res['sigma']['error']:.1f}"
                signif = f"{res.get('significance', 0):.2f}σ"
                print(f"{dataset:<15} {yield_val:<20} {mass:<20} {width:<15} {signif:<12}")
        
        # Combined by track type (all years)
        print("-"*80 + "\nCombined by track type (all years):")
        for tr in tracks:
            fit_id = f"{cls}_all_{tr}"
            if fit_id in all_fit_results and all_fit_results[fit_id] and 'error' not in all_fit_results[fit_id]:
                res = all_fit_results[fit_id]
                dataset = f"All/{tr}"
                yield_val = f"{res['nsig']['value']:.1f} ± {res['nsig']['error']:.1f}"
                mass = f"{res['mean']['value']:.1f} ± {res['mean']['error']:.1f}"
                width = f"{res['sigma']['value']:.1f} ± {res['sigma']['error']:.1f}"
                signif = f"{res.get('significance', 0):.2f}σ"
                print(f"{dataset:<15} {yield_val:<20} {mass:<20} {width:<15} {signif:<12}")
        
        # Fully combined
        fit_id = f"{cls}_all_all"
        if fit_id in all_fit_results and all_fit_results[fit_id] and 'error' not in all_fit_results[fit_id]:
            res = all_fit_results[fit_id]
            dataset = "All/All"
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
    
    # Compare ratios for each year (combined tracks)
    for y in years:
        sig_results = all_fit_results.get(f"sig_{y}_all")
        norm_results = all_fit_results.get(f"norm_{y}_all")
        
        if sig_results and norm_results and 'error' not in sig_results and 'error' not in norm_results:
            dataset = f"{y}/All"
                
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
    
    # Compare ratios for each track type (combined years)
    for tr in tracks:
        sig_results = all_fit_results.get(f"sig_all_{tr}")
        norm_results = all_fit_results.get(f"norm_all_{tr}")
        
        if sig_results and norm_results and 'error' not in sig_results and 'error' not in norm_results:
            dataset = f"All/{tr}"
                
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
    
    # Compare ratio for fully combined dataset
    sig_results = all_fit_results.get(f"sig_all_all")
    norm_results = all_fit_results.get(f"norm_all_all")
    
    if sig_results and norm_results and 'error' not in sig_results and 'error' not in norm_results:
        dataset = "All/All"
            
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
    
    # Run detailed channel comparisons and save those results too
    comparisons = {}
    
    # Compare channels for individual years (combined tracks)
    for y in years:
        sig_results = all_fit_results.get(f"sig_{y}_all")
        norm_results = all_fit_results.get(f"norm_{y}_all")
        if sig_results and norm_results and 'error' not in sig_results and 'error' not in norm_results:
            comp = compare_channels(sig_results, norm_results, y, 'all')
            if comp:
                comparisons[f"{y}_all"] = comp
    
    # Compare channels for track types (combined years)
    for tr in tracks:
        sig_results = all_fit_results.get(f"sig_all_{tr}")
        norm_results = all_fit_results.get(f"norm_all_{tr}")
        if sig_results and norm_results and 'error' not in sig_results and 'error' not in norm_results:
            comp = compare_channels(sig_results, norm_results, 'all', tr)
            if comp:
                comparisons[f"all_{tr}"] = comp
    
    # Compare for fully combined dataset
    sig_results = all_fit_results.get(f"sig_all_all")
    norm_results = all_fit_results.get(f"norm_all_all")
    if sig_results and norm_results and 'error' not in sig_results and 'error' not in norm_results:
        comp = compare_channels(sig_results, norm_results, 'all', 'all')
        if comp:
            comparisons["all_all"] = comp
    
    # Save comparisons to JSON
    comparison_filename = os.path.join(results_dir, "channel_comparisons.json")
    with open(comparison_filename, 'w') as f:
        json.dump(comparisons, f, indent=4)
    print(f"Channel comparisons saved to: {comparison_filename}")
    
    return all_fit_results

if __name__ == "__main__":
    results = run_all_fits()