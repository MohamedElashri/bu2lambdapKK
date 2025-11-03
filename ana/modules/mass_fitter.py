"""
Mass Fitting Module for B+ -> Lambda pK-K+ Analysis

Implements RooFit-based simultaneous mass fitting for charmonium states.
Following plan.md Phase 5 specification.
"""

import ROOT
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Any
import os

# Enable RooFit batch mode for better performance
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)


class MassFitter:
    """
    RooFit-based Mass Fitting for B+ -> Lambda pK-K+ Analysis
    
    Fits M(Λ̄pK⁻) invariant mass distribution to extract charmonium yields.
    
    Strategy (following plan.md Phase 5):
    - Fit each YEAR separately (MagDown + MagUp already combined)
    - Share physical parameters (masses, widths, resolution) across years
    - Extract separate yields per year (needed for efficiency correction)
    - Use RooVoigtian (RBW ⊗ Gaussian) for signal shapes
    - Exponential background
    
    Signal PDFs per state:
    - J/ψ: RBW ⊗ Gaussian, Γ_J/ψ fixed to PDG (0.093 MeV)
    - ηc: RBW ⊗ Gaussian, Γ_ηc floating (broad state)
    - χc0: RBW ⊗ Gaussian, Γ_χc0 floating (broad state)
    - χc1: RBW ⊗ Gaussian, Γ_χc1 fixed to PDG (0.84 MeV)
    
    Background: Exponential exp(-α × M)
    """
    
    def __init__(self, config: Any):
        """
        Initialize mass fitter with configuration
        
        Args:
            config: TOMLConfig object with particles, paths configuration
        """
        self.config = config
        self.fit_range = config.particles["mass_windows"]["charmonium_fit_range"]
        
        # B+ mass window for pre-selection (applied BEFORE fitting)
        self.bu_mass_min = config.selection.get("bu_fixed_selection", {}).get("mass_corrected_min", 5255.0)
        self.bu_mass_max = config.selection.get("bu_fixed_selection", {}).get("mass_corrected_max", 5305.0)
        
        # Shared parameters across years (will be created on first use)
        self.masses = {}      # M_J/ψ, M_ηc, M_χc0, M_χc1
        self.widths = {}      # Γ states (some fixed, some floating)
        self.resolution = None  # Single resolution parameter (shared)
        
        # Observable (shared across all fits)
        self.mass_var = None
        
        # Store all PDFs and variables to prevent garbage collection
        self.signal_pdfs = {}  # {state: pdf}
        self.bkg_pdfs = {}     # {year: pdf}
        self.alpha_bkgs = {}   # {year: alpha}
        self.models = {}       # {year: model}
        self.yields = {}       # {year: {state: yield_var}}
        
    def setup_observable(self) -> ROOT.RooRealVar:
        """
        Define RooRealVar for M(Λ̄pK⁻) invariant mass
        
        Returns:
            RooRealVar for mass observable
        """
        if self.mass_var is None:
            self.mass_var = ROOT.RooRealVar(
                "M_LpKm",
                "M(#bar{#Lambda}pK^{-}) [MeV/c^{2}]",
                self.fit_range[0],
                self.fit_range[1]
            )
        return self.mass_var
    
    def create_signal_pdf(self, state: str, mass_var: ROOT.RooRealVar) -> ROOT.RooVoigtian:
        """
        Create signal PDF for one charmonium state
        
        Uses RooVoigtian (Relativistic Breit-Wigner ⊗ Gaussian)
        
        Args:
            state: State name ("jpsi", "etac", "chic0", "chic1")
            mass_var: Observable mass variable
            
        Returns:
            RooVoigtian PDF for this state
        """
        state_lower = state.lower()
        
        # Return cached PDF if it exists
        if state_lower in self.signal_pdfs:
            return self.signal_pdfs[state_lower]
        
        # Map state names to config keys
        config_key_map = {
            "jpsi": "jpsi",
            "etac": "etac_1s",
            "chic0": "chic0",
            "chic1": "chic1"
        }
        config_key = config_key_map.get(state_lower, state_lower)
        
        # Mass parameter (shared across years)
        if state_lower not in self.masses:
            pdg_mass = self.config.particles["pdg_masses"][config_key]
            
            self.masses[state_lower] = ROOT.RooRealVar(
                f"M_{state}",
                f"M_{state} [MeV/c^{{2}}]",
                pdg_mass,
                pdg_mass - 50,  # Allow ±50 MeV variation
                pdg_mass + 50
            )
        
        # Width parameter
        if state_lower not in self.widths:
            pdg_width = self.config.particles["pdg_widths"][config_key]
            
            # Fix narrow states (J/ψ, χc1) to PDG values
            if state_lower in ["jpsi", "chic1"]:
                self.widths[state_lower] = ROOT.RooRealVar(
                    f"Gamma_{state}",
                    f"#Gamma_{state} [MeV/c^{{2}}]",
                    pdg_width
                )
                self.widths[state_lower].setConstant(True)
            else:
                # Float for ηc and χc0 (broader states with larger uncertainties)
                self.widths[state_lower] = ROOT.RooRealVar(
                    f"Gamma_{state}",
                    f"#Gamma_{state} [MeV/c^{{2}}]",
                    pdg_width,
                    0.1,    # Minimum 0.1 MeV
                    100.0   # Maximum 100 MeV
                )
        
        # Resolution (shared Gaussian width - same detector for all states)
        if self.resolution is None:
            self.resolution = ROOT.RooRealVar(
                "sigma_resolution",
                "#sigma_{resolution} [MeV/c^{2}]",
                5.0,   # Initial guess ~5 MeV
                1.0,   # Minimum
                20.0   # Maximum
            )
        
        # Create Voigtian PDF (RBW ⊗ Gaussian)
        signal_pdf = ROOT.RooVoigtian(
            f"pdf_signal_{state}",
            f"Signal PDF for {state}",
            mass_var,
            self.masses[state_lower],
            self.widths[state_lower],
            self.resolution
        )
        
        # Cache the PDF
        self.signal_pdfs[state_lower] = signal_pdf
        
        return signal_pdf
    
    def create_background_pdf(self, mass_var: ROOT.RooRealVar, year: str) -> Tuple[ROOT.RooExponential, ROOT.RooRealVar]:
        """
        Create exponential background PDF
        
        Simple exponential: exp(-α × M)
        α parameter is PER YEAR (different backgrounds per year)
        
        Args:
            mass_var: Observable mass variable
            year: Year string ("2016", "2017", "2018")
            
        Returns:
            (background_pdf, alpha_parameter)
        """
        # Return cached PDF if it exists
        if year in self.bkg_pdfs:
            return self.bkg_pdfs[year], self.alpha_bkgs[year]
        
        alpha = ROOT.RooRealVar(
            f"alpha_bkg_{year}",
            f"#alpha_{{bkg}} {year}",
            -0.001,  # Initial guess
            -0.01,   # Minimum (steeper slope)
            0.0      # Maximum (flat)
        )
        
        bkg_pdf = ROOT.RooExponential(
            f"pdf_bkg_{year}",
            f"Background PDF {year}",
            mass_var,
            alpha
        )
        
        # Cache
        self.bkg_pdfs[year] = bkg_pdf
        self.alpha_bkgs[year] = alpha
        
        return bkg_pdf, alpha

    
    def build_model_for_year(self, year: str, mass_var: ROOT.RooRealVar) -> Tuple[ROOT.RooAddPdf, Dict[str, ROOT.RooRealVar]]:
        """
        Build full extended likelihood model for one year
        
        PDF = Σ[N_state × Signal_state] + N_bkg × Background
        
        States: J/ψ, ηc, χc0, χc1
        
        Args:
            year: Year string
            mass_var: Observable mass variable
            
        Returns:
            (total_pdf, yields_dict)
        """
        pdf_list = ROOT.RooArgList()
        coef_list = ROOT.RooArgList()
        year_yields = {}
        
        # Signal components (4 charmonium states)
        for state in ["jpsi", "etac", "chic0", "chic1"]:
            # Create signal PDF (shares mass/width/resolution across years)
            sig_pdf = self.create_signal_pdf(state, mass_var)
            
            # Create yield parameter (separate per year)
            yield_var = ROOT.RooRealVar(
                f"N_{state}_{year}",
                f"Yield {state} {year}",
                1000,  # Initial guess
                0,     # Minimum (non-negative)
                1e6    # Maximum
            )
            year_yields[state] = yield_var
            
            # Add to lists
            pdf_list.add(sig_pdf)
            coef_list.add(yield_var)
        
        # Background component
        bkg_pdf, alpha_bkg = self.create_background_pdf(mass_var, year)
        
        bkg_yield = ROOT.RooRealVar(
            f"N_bkg_{year}",
            f"Background yield {year}",
            10000,  # Initial guess
            0,      # Minimum
            1e7     # Maximum
        )
        year_yields["background"] = bkg_yield
        
        pdf_list.add(bkg_pdf)
        coef_list.add(bkg_yield)
        
        # Build extended sum PDF
        total_pdf = ROOT.RooAddPdf(
            f"model_{year}",
            f"Total PDF {year}",
            pdf_list,
            coef_list
        )
        
        # Store model and yields to prevent garbage collection
        self.models[year] = total_pdf
        self.yields[year] = year_yields
        
        return total_pdf, year_yields
    
    def perform_fit(self, data_by_year: Dict[str, ak.Array], fit_combined: bool = True) -> Dict[str, Any]:
        """
        Perform mass fits to all years (per-year AND combined)
        
        Strategy:
        1. Apply B+ mass window cut [5255, 5305] MeV
        2. Fit each year separately with shared physics parameters
        3. Fit combined dataset (all years together)
        4. Model all 4 charmonium states simultaneously: J/ψ, ηc, χc0, χc1
        
        Args:
            data_by_year: {year: awkward_array} with M_LpKm_h2 and Bu_M_DTF_PV branches
            fit_combined: If True, also fit combined dataset
            
        Returns:
            {
                "yields": {year: {state: (value, error)}},  # includes "combined"
                "masses": {state: (value, error)},
                "widths": {state: (value, error)},
                "resolution": (value, error),
                "fit_results": {year: RooFitResult}  # includes "combined"
            }
        """
        mass_var = self.setup_observable()
        
        all_yields = {}
        all_fit_results = {}
        
        print("\n" + "="*80)
        print("MASS FITTING WITH ROOFIT - ALL 4 CHARMONIUM STATES")
        print("="*80)
        print(f"Charmonium fit range: {self.fit_range[0]} - {self.fit_range[1]} MeV")
        print(f"B+ mass window: {self.bu_mass_min} - {self.bu_mass_max} MeV")
        print(f"Using M_LpKm_h2 (M(Λ̄pK⁻), h2=K⁻ correct for charmonium)")
        print(f"Modeling: J/ψ, ηc, χc0, χc1 simultaneously")
        print("="*80)
        
        # Prepare all datasets (per-year + combined)
        datasets_to_fit = {}
        
        for year in sorted(data_by_year.keys()):
            print(f"\n[Year {year}]")
            
            events = data_by_year[year]
            
            # Apply B+ mass window cut
            bu_mass = events["Bu_MM_corrected"]  # Lambda-corrected B+ mass
            bu_mask = (bu_mass >= self.bu_mass_min) & (bu_mass <= self.bu_mass_max)
            events_bu_cut = events[bu_mask]
            
            print(f"  Events after B+ mass cut [{self.bu_mass_min}, {self.bu_mass_max}]: {len(events_bu_cut)}")
            
            # Get charmonium mass data (M(Λ̄pK⁻), use h2=K⁻)
            mass_array = events_bu_cut["M_LpKm_h2"]
            
            # Apply charmonium fit range filter
            mask = (mass_array >= self.fit_range[0]) & (mass_array <= self.fit_range[1])
            mass_filtered = mass_array[mask]
            
            print(f"  Events in charmonium fit range [{self.fit_range[0]}, {self.fit_range[1]}]: {len(mass_filtered)}")
            
            datasets_to_fit[year] = mass_filtered
        
        # Create combined dataset
        if fit_combined and len(datasets_to_fit) > 1:
            combined_mass = ak.concatenate([datasets_to_fit[y] for y in sorted(datasets_to_fit.keys())])
            datasets_to_fit["combined"] = combined_mass
            print(f"\n[Combined All Years]")
            print(f"  Total events: {len(combined_mass)}")
        
        # Fit each dataset
        for dataset_name in sorted(datasets_to_fit.keys()):
            print(f"\n{'='*60}")
            print(f"Fitting: {dataset_name}")
            print(f"{'='*60}")
            
            mass_filtered = datasets_to_fit[dataset_name]
            print(f"  Events to fit: {len(mass_filtered)}")
            
            # Convert to numpy for RooDataSet
            mass_np = ak.to_numpy(mass_filtered)
            
            # Create RooDataSet directly from numpy array
            dataset = ROOT.RooDataSet(
                f"data_{dataset_name}",
                f"Data {dataset_name}",
                ROOT.RooArgSet(mass_var)
            )
            
            # Fill dataset with events
            for m in mass_np:
                mass_var.setVal(m)
                dataset.add(ROOT.RooArgSet(mass_var))
            
            print(f"  RooDataSet entries: {dataset.numEntries()}")
            
            # Build model for this dataset
            model, yields = self.build_model_for_year(dataset_name, mass_var)
            
            # Perform fit
            print(f"  Fitting...")
            fit_result = model.fitTo(
                dataset,
                ROOT.RooFit.Save(),
                ROOT.RooFit.Extended(True),
                ROOT.RooFit.PrintLevel(-1),
                ROOT.RooFit.NumCPU(4),
                ROOT.RooFit.Strategy(2)  # More robust
            )
            
            # Check convergence
            status = fit_result.status()
            cov_qual = fit_result.covQual()
            edm = fit_result.edm()
            
            print(f"  Fit status: {status} (0 = success)")
            print(f"  Covariance quality: {cov_qual} (3 = full accurate)")
            print(f"  EDM: {edm:.2e}")
            
            if status != 0:
                print(f"  WARNING: Fit did not converge properly!")
            
            # Extract yields with errors
            dataset_yields = {}
            print(f"\n  Yields for {dataset_name}:")
            for state, yield_var in yields.items():
                value = yield_var.getVal()
                error = yield_var.getError()
                dataset_yields[state] = (value, error)
                
                print(f"    N_{state:<12} = {value:8.0f} ± {error:6.0f}")
            
            all_yields[dataset_name] = dataset_yields
            all_fit_results[dataset_name] = fit_result
            
            # Plot fit result
            self.plot_fit_result(dataset_name, mass_var, dataset, model, yields)
        
        # Extract shared parameters (from last fit)
        print("\n" + "="*80)
        print("FITTED PARAMETERS (shared across years)")
        print("="*80)
        
        masses_result = {}
        widths_result = {}
        
        for state in ["jpsi", "etac", "chic0", "chic1"]:
            mass_val = self.masses[state].getVal()
            mass_err = self.masses[state].getError()
            masses_result[state] = (mass_val, mass_err)
            
            width_val = self.widths[state].getVal()
            width_err = self.widths[state].getError()
            widths_result[state] = (width_val, width_err)
            
            print(f"{state:>8}: M = {mass_val:7.2f} ± {mass_err:5.2f} MeV,  "
                  f"Γ = {width_val:6.2f} ± {width_err:5.2f} MeV")
        
        res_val = self.resolution.getVal()
        res_err = self.resolution.getError()
        print(f"\nResolution: σ = {res_val:.2f} ± {res_err:.2f} MeV")
        
        return {
            "yields": all_yields,
            "masses": masses_result,
            "widths": widths_result,
            "resolution": (res_val, res_err),
            "fit_results": all_fit_results
        }
    
    def plot_fit_result(self, year: str, mass_var: ROOT.RooRealVar, 
                       dataset: ROOT.RooDataSet, model: ROOT.RooAddPdf,
                       yields: Dict[str, ROOT.RooRealVar]):
        """
        Plot fit result with LHCb publication quality
        
        Creates official LHCb-style plot with:
        - Data points with Poisson errors
        - Total fit curve (solid blue)
        - Signal components (dotted, high contrast colors)
        - Background component (dashed gray)
        - Proper LaTeX labels
        - No pull distribution (removed for cleaner presentation)
        
        Args:
            year: Year string (or "combined")
            mass_var: Observable mass variable
            dataset: RooDataSet with data
            model: Total PDF
            yields: Dictionary of yield parameters
        """
        # Create RooPlot with year as title
        title = "2016-2018" if year == "combined" else str(year)
        frame = mass_var.frame(ROOT.RooFit.Title(title))
        
        # Plot data (black points)
        dataset.plotOn(frame, ROOT.RooFit.Name("data"), 
                      ROOT.RooFit.MarkerStyle(20), 
                      ROOT.RooFit.MarkerSize(0.9),
                      ROOT.RooFit.MarkerColor(ROOT.kBlack))
        
        # Plot total PDF (solid blue)
        model.plotOn(frame, ROOT.RooFit.Name("total"), 
                    ROOT.RooFit.LineColor(ROOT.kBlue + 2), 
                    ROOT.RooFit.LineWidth(3),
                    ROOT.RooFit.LineStyle(ROOT.kSolid))
        
        # High-contrast colors for charmonium states (avoiding black/dark colors)
        colors = {
            "jpsi": ROOT.kRed + 1,        # Bright red
            "etac": ROOT.kGreen + 2,      # Bright green
            "chic0": ROOT.kMagenta + 1,   # Bright magenta
            "chic1": ROOT.kOrange + 1,    # Bright orange
            "background": ROOT.kGray + 1  # Gray for background
        }
        
        # Plot signal components with DOTTED lines (high contrast)
        for state in ["jpsi", "etac", "chic0", "chic1"]:
            component_name = f"pdf_signal_{state}"
            model.plotOn(frame, ROOT.RooFit.Components(component_name),
                        ROOT.RooFit.Name(state),
                        ROOT.RooFit.LineColor(colors[state]),
                        ROOT.RooFit.LineStyle(ROOT.kDotted),  # Dotted for signals
                        ROOT.RooFit.LineWidth(3))
        
        # Plot background with DASHED line
        bkg_component_name = f"pdf_bkg_{year}"
        model.plotOn(frame, ROOT.RooFit.Components(bkg_component_name),
                    ROOT.RooFit.Name("background"),
                    ROOT.RooFit.LineColor(colors["background"]),
                    ROOT.RooFit.LineStyle(ROOT.kDashed),  # Dashed for background
                    ROOT.RooFit.LineWidth(3))
        
        # Create canvas WITHOUT pull distribution
        canvas = ROOT.TCanvas(f"c_{year}", f"Fit {year}", 900, 700)
        canvas.SetLeftMargin(0.12)
        canvas.SetRightMargin(0.05)
        canvas.SetTopMargin(0.08)
        canvas.SetBottomMargin(0.12)
        
        # Adjust axis labels for single plot
        frame.GetYaxis().SetTitle("Candidates / (5 MeV/#it{c}^{2})")
        frame.GetYaxis().SetTitleSize(0.045)
        frame.GetYaxis().SetLabelSize(0.040)
        frame.GetYaxis().SetTitleOffset(1.3)
        frame.GetXaxis().SetTitle("m(#bar{#Lambda}pK^{#minus}) [MeV/#it{c}^{2}]")
        frame.GetXaxis().SetTitleSize(0.045)
        frame.GetXaxis().SetLabelSize(0.040)
        frame.GetXaxis().SetTitleOffset(1.1)
        frame.Draw()
        
        # Add compact legend in top right (inside plot area)
        legend = ROOT.TLegend(0.60, 0.55, 0.92, 0.89)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.035)
        legend.SetTextFont(42)
        legend.SetMargin(0.15)  # Reduce space between symbol and text
        
        # Add entries with proper formatting
        legend.AddEntry("data", "Data: B^{+} #rightarrow #bar{#Lambda}pK^{#minus}K^{+}", "lep")
        legend.AddEntry("total", "Total fit", "l")
        
        # Add component labels with proper LaTeX (no extra spacer)
        state_labels = {
            "jpsi": "J/#psi",
            "etac": "#eta_{c}(1S)",
            "chic0": "#chi_{c0}(1P)",
            "chic1": "#chi_{c1}(1P)",
            "background": "Combinatorial bkg."
        }
        
        for state in ["jpsi", "etac", "chic0", "chic1", "background"]:
            legend.AddEntry(state, state_labels[state], "l")
        
        legend.Draw()
        
        # Save plot
        plot_dir = Path(self.config.paths["output"]["plots_dir"]) / "fits"
        plot_dir.mkdir(exist_ok=True, parents=True)
        
        output_file = plot_dir / f"mass_fit_{year}.pdf"
        canvas.SaveAs(str(output_file))
        
        print(f"  ✓ Saved fit plot: {output_file}")
        
        return str(output_file)
