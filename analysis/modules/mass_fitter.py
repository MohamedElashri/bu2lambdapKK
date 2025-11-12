"""
Mass Fitting Module for B+ -> Lambda pK-K+ Analysis

Implements RooFit-based simultaneous mass fitting for charmonium states.
Following plan.md Phase 5 specification.
"""

from __future__ import annotations

import ROOT
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import os

from .exceptions import FittingError

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
    - ARGUS function for background (standard for B meson analyses)
    
    Signal PDFs per state (all 5 charmonium states):
    - ηc(1S): RBW ⊗ Gaussian, M and Γ fixed to PDG
    - J/ψ: RBW ⊗ Gaussian, M and Γ fixed to PDG
    - χc0: RBW ⊗ Gaussian, M and Γ fixed to PDG
    - χc1: RBW ⊗ Gaussian, M and Γ fixed to PDG
    - ηc(2S): RBW ⊗ Gaussian, M and Γ fixed to PDG
    
    Background: ARGUS function (standard for B meson combinatorial background)
    
    Attributes:
        config: Configuration object
        fit_range: Mass fit range tuple
        mass_var: RooRealVar for mass observable
        signal_pdfs: Dictionary of signal PDFs per state
        bkg_pdfs: Dictionary of background PDFs per year
        models: Dictionary of combined models per year
        yields: Nested dictionary of yield variables
    """
    
    def __init__(self, config: Any) -> None:
        """
        Initialize mass fitter with configuration.
        
        Args:
            config: TOMLConfig object with particles, paths configuration
        """
        self.config: Any = config
        self.fit_range: Tuple[float, float] = config.particles["mass_windows"]["charmonium_fit_range"]
        
        # B+ mass window for pre-selection (applied BEFORE fitting)
        self.bu_mass_min: float = config.selection.get("bu_fixed_selection", {}).get("mass_corrected_min", 5255.0)
        self.bu_mass_max: float = config.selection.get("bu_fixed_selection", {}).get("mass_corrected_max", 5305.0)
        
        # Fitting configuration
        fitting_config: Dict[str, Any] = config.particles.get("fitting", {})
        self.use_binned_fit: bool = fitting_config.get("use_binned_fit", True)
        self.bin_width: float = fitting_config.get("bin_width", 5.0)
        
        # Calculate number of bins automatically: always maintain bin_width MeV/bin
        fit_range_width: float = self.fit_range[1] - self.fit_range[0]
        self.nbins: int = int(fit_range_width / self.bin_width)
        
        # Shared parameters across years (will be created on first use)
        self.masses: Dict[str, ROOT.RooRealVar] = {}      # M_J/ψ, M_ηc, M_χc0, M_χc1
        self.widths: Dict[str, ROOT.RooRealVar] = {}      # Γ states (some fixed, some floating)
        self.resolution: Optional[ROOT.RooRealVar] = None  # Single resolution parameter (shared)
        
        # Observable (shared across all fits)
        self.mass_var: Optional[ROOT.RooRealVar] = None
        
        # Store all PDFs and variables to prevent garbage collection
        self.signal_pdfs: Dict[str, ROOT.RooAbsPdf] = {}  # {state: pdf}
        self.bkg_pdfs: Dict[str, ROOT.RooAbsPdf] = {}     # {year: pdf}

        self.models: Dict[str, ROOT.RooAbsPdf] = {}       # {year: model}
        self.yields: Dict[str, Dict[str, ROOT.RooRealVar]] = {}       # {year: {state: yield_var}}
        
    def setup_observable(self) -> ROOT.RooRealVar:
        """
        Define RooRealVar for M(Λ̄pK⁻) invariant mass.
        
        Returns:
            ROOT.RooRealVar for mass observable
        """
        if self.mass_var is None:
            self.mass_var = ROOT.RooRealVar(
                "M_LpKm",
                "M(#bar{#Lambda}pK^{-}) [MeV/c^{2}]",
                self.fit_range[0],
                self.fit_range[1]
            )
            # Set binning for plotting and binned fits
            self.mass_var.setBins(self.nbins)
        return self.mass_var
    
    def create_signal_pdf(self, state: str, mass_var: ROOT.RooRealVar) -> ROOT.RooAbsPdf:
        """
        Create signal PDF for one charmonium state
        
        Uses RooVoigtian (Relativistic Breit-Wigner ⊗ Gaussian)
        
        Args:
            state: State name ("jpsi", "etac", "chic0", "chic1", "etac_2s")
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
            "chic1": "chic1",
            "etac_2s": "etac_2s"
        }
        config_key = config_key_map.get(state_lower, state_lower)
        
        # Mass parameter (shared across years) - FIXED TO PDG VALUE
        if state_lower not in self.masses:
            pdg_mass = self.config.particles["pdg_masses"][config_key]
            
            self.masses[state_lower] = ROOT.RooRealVar(
                f"M_{state}",
                f"M_{state} [MeV/c^{{2}}]",
                pdg_mass
            )
            # Fix mass to PDG value
            self.masses[state_lower].setConstant(True)
        
        # Width parameter - FIXED TO PDG VALUE FOR ALL STATES
        if state_lower not in self.widths:
            pdg_width = self.config.particles["pdg_widths"][config_key]
            
            self.widths[state_lower] = ROOT.RooRealVar(
                f"Gamma_{state}",
                f"#Gamma_{state} [MeV/c^{{2}}]",
                pdg_width
            )
            # Fix width to PDG value
            self.widths[state_lower].setConstant(True)
        
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
    
    def create_background_pdf(self, mass_var: ROOT.RooRealVar, year: str) -> Tuple[ROOT.RooAbsPdf, ROOT.RooRealVar]:
        """
        Create ARGUS background PDF
        
        ARGUS function: f(m) = m * sqrt(1 - (m/m0)^2) * exp(c * (1 - (m/m0)^2))
        
        Args:
            mass_var: Observable mass variable
            year: Year string ("2016", "2017", "2018")
            
        Returns:
            (background_pdf, c_parameter)
        """
        # Return cached PDF if it exists
        if year in self.bkg_pdfs:
            return self.bkg_pdfs[year], self.argus_params[year]["c"]
        
        # ARGUS endpoint - set BEYOND the fit range to avoid sharp cutoff
        # Read offset from config (default 200 MeV for backward compatibility)
        endpoint_offset = self.config.particles.get("fitting", {}).get("argus_endpoint_offset", 200.0)
        
        m0 = ROOT.RooRealVar(
            f"m0_argus_{year}",
            "ARGUS endpoint [MeV/c^{2}]",
            self.fit_range[1] + endpoint_offset  # Extended beyond fit range
        )
        m0.setConstant(True)
        
        # ARGUS shape parameter (fitted per year)
        c = ROOT.RooRealVar(
            f"c_argus_{year}",
            f"c_{{ARGUS}} {year}",
            -20.0,  # Initial guess
            -100.0, # Minimum
            -0.1    # Maximum (must be negative)
        )
        
        # Power parameter (typically fixed to 0.5)
        p = ROOT.RooRealVar(
            f"p_argus_{year}",
            "ARGUS power",
            0.5
        )
        p.setConstant(True)
        
        # Create ARGUS PDF
        bkg_pdf = ROOT.RooArgusBG(
            f"pdf_bkg_{year}",
            f"Background PDF {year}",
            mass_var,
            m0,
            c,
            p
        )
        
        # Initialize storage if needed
        if not hasattr(self, 'argus_params'):
            self.argus_params = {}
        
        # Cache ALL parameters to prevent garbage collection
        self.bkg_pdfs[year] = bkg_pdf
        self.argus_params[year] = {"m0": m0, "c": c, "p": p}
        
        return bkg_pdf, c
    
    def build_model_for_year(self, year: str, mass_var: ROOT.RooRealVar) -> Tuple[ROOT.RooAbsPdf, Dict[str, ROOT.RooRealVar]]:
        """
        Build full extended likelihood model for one year
        
        PDF = Σ[N_state * Signal_state] + N_bkg * Background
        
        States: ηc(1S), J/ψ, χc0, χc1, ηc(2S), ψ(3770)
        
        Args:
            year: Year string
            mass_var: Observable mass variable
            
        Returns:
            (total_pdf, yields_dict)
        """
        pdf_list = ROOT.RooArgList()
        coef_list = ROOT.RooArgList()
        year_yields = {}
        
        # Signal components (5 charmonium states)
        for state in ["jpsi", "etac", "chic0", "chic1", "etac_2s"]:
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
    
    def perform_fit(
        self,
        data_by_year: Dict[str, ak.Array],
        fit_combined: bool = True
    ) -> Dict[str, Any]:
        """
        Perform mass fits to all years (per-year AND combined).
        
        Following Phase 5:
        - Fit each year individually (MagDown + MagUp already combined)
        - Share physical parameters (masses, widths, resolution) across years
        - Extract per-year yields for efficiency correction
        - Also fit combined dataset for overall yield check
        
        Args:
            data_by_year: {year: awkward_array} with M_LpKm_h2 branch
            fit_combined: If True, also fit combined dataset
            
        Returns:
            Dictionary with fit results:
            - "yields": {year: {state: (value, error)}}
            - "masses": {state: (value, error)}
            - "widths": {state: (value, error)}
            - "resolution": (value, error)
            - "fit_quality": {year: {"status": int, "minNll": float}}
            
        Raises:
            FittingError: If fit fails for any year
        """
        mass_var = self.setup_observable()
        
        all_yields = {}
        all_fit_results = {}
        
        print("\n" + "="*80)
        print("MASS FITTING WITH ROOFIT - ALL 6 CHARMONIUM STATES")
        print("="*80)
        print(f"Charmonium fit range: {self.fit_range[0]} - {self.fit_range[1]} MeV")
        print(f"B+ mass window: {self.bu_mass_min} - {self.bu_mass_max} MeV")
        print(f"Using M_LpKm_h2 (M(Λ̄pK⁻), h2=K⁻ correct for charmonium)")
        print(f"Modeling: ηc(1S), J/ψ, χc0, χc1, ηc(2S), ψ(3770) simultaneously")
        print(f"All masses and widths FIXED to PDG values")
        print(f"Fit type: {'BINNED' if self.use_binned_fit else 'UNBINNED'} maximum likelihood")
        print(f"Binning: {self.nbins} bins × {self.bin_width} MeV/bin = {self.fit_range[1] - self.fit_range[0]} MeV range")
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
            
            # Convert to numpy for RooDataSet/RooDataHist
            mass_np = ak.to_numpy(mass_filtered)
            
            # Create dataset (binned or unbinned based on configuration)
            if self.use_binned_fit:
                # Create unbinned dataset first
                temp_dataset = ROOT.RooDataSet(
                    f"temp_data_{dataset_name}",
                    f"Temp Data {dataset_name}",
                    ROOT.RooArgSet(mass_var)
                )
                for m in mass_np:
                    mass_var.setVal(m)
                    temp_dataset.add(ROOT.RooArgSet(mass_var))
                
                # Convert to binned dataset (RooDataHist)
                dataset = ROOT.RooDataHist(
                    f"data_{dataset_name}",
                    f"Data {dataset_name}",
                    ROOT.RooArgSet(mass_var),
                    temp_dataset
                )
                print(f"  RooDataHist entries: {dataset.numEntries()} (binned in {self.nbins} bins)")
            else:
                # Create unbinned dataset (RooDataSet)
                dataset = ROOT.RooDataSet(
                    f"data_{dataset_name}",
                    f"Data {dataset_name}",
                    ROOT.RooArgSet(mass_var)
                )
                for m in mass_np:
                    mass_var.setVal(m)
                    dataset.add(ROOT.RooArgSet(mass_var))
                print(f"  RooDataSet entries: {dataset.numEntries()} (unbinned)")
            
            # Build model for this dataset
            model, yields = self.build_model_for_year(dataset_name, mass_var)
            
            # Perform fit
            print(f"  Fitting...")
            if self.use_binned_fit:
                # Binned maximum likelihood fit
                fit_result = model.fitTo(
                    dataset,
                    ROOT.RooFit.Save(),
                    ROOT.RooFit.Extended(True),
                    ROOT.RooFit.PrintLevel(-1),
                    ROOT.RooFit.NumCPU(4),
                    ROOT.RooFit.Strategy(2)  # More robust
                )
            else:
                # Unbinned maximum likelihood fit
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
        
        for state in ["jpsi", "etac", "chic0", "chic1", "etac_2s"]:
            mass_val = self.masses[state].getVal()
            mass_err = self.masses[state].getError()
            masses_result[state] = (mass_val, mass_err)
            
            width_val = self.widths[state].getVal()
            width_err = self.widths[state].getError()
            widths_result[state] = (width_val, width_err)
            
            print(f"{state:>10}: M = {mass_val:7.2f} ± {mass_err:5.2f} MeV,  "
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
    
    def plot_fit_result(
        self,
        year: str,
        mass_var: Any,
        dataset: Any,
        model: Any,
        yields: Dict[str, Any]
    ) -> None:
        """
        Plot fit result with LHCb publication quality
        
        Creates official LHCb-style plot with:
        - Data points with Poisson errors
        - Total fit curve (solid blue)
        - Signal components (dotted, high contrast colors)
        - Background component (dashed gray)
        - Pull distribution below fit
        - Proper LaTeX labels
        
        Args:
            year: Year string (or "combined")
            mass_var: Observable mass variable
            dataset: RooDataSet or RooDataHist with data
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
            "etac_2s": ROOT.kCyan + 1,    # Bright cyan
            "background": ROOT.kGray + 1  # Gray for background
        }
        
        # Plot signal components with DOTTED lines (high contrast)
        for state in ["jpsi", "etac", "chic0", "chic1", "etac_2s"]:
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
        
        # Create canvas with two pads for fit and pull distribution
        canvas = ROOT.TCanvas(f"c_{year}", f"Fit {year}", 900, 800)
        
        # Upper pad for fit
        pad1 = ROOT.TPad("pad1", "Fit", 0.0, 0.25, 1.0, 1.0)
        pad1.SetBottomMargin(0.02)
        pad1.SetLeftMargin(0.12)
        pad1.SetRightMargin(0.05)
        pad1.SetTopMargin(0.08)
        pad1.Draw()
        
        # Lower pad for pulls
        pad2 = ROOT.TPad("pad2", "Pulls", 0.0, 0.0, 1.0, 0.25)
        pad2.SetTopMargin(0.02)
        pad2.SetBottomMargin(0.35)
        pad2.SetLeftMargin(0.12)
        pad2.SetRightMargin(0.05)
        pad2.SetGridy()
        pad2.Draw()
        
        # Draw fit in upper pad
        pad1.cd()
        frame.GetYaxis().SetTitle(f"Candidates / ({self.bin_width:.0f} MeV/#it{{c}}^{{2}})")
        frame.GetYaxis().SetTitleSize(0.055)
        frame.GetYaxis().SetLabelSize(0.050)
        frame.GetYaxis().SetTitleOffset(1.0)
        frame.GetXaxis().SetLabelSize(0.0)
        frame.GetXaxis().SetTitleSize(0.0)
        frame.Draw()
        
        # Add compact legend in top right (inside plot area)
        legend = ROOT.TLegend(0.58, 0.42, 0.92, 0.89)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.035)
        legend.SetTextFont(42)
        legend.SetMargin(0.15)
        legend.AddEntry("data", "Data: B^{+} #rightarrow #bar{#Lambda}pK^{#minus}K^{+}", "lep")
        legend.AddEntry("total", "Total fit", "l")
        state_labels = {
            "jpsi": "J/#psi",
            "etac": "#eta_{c}(1S)",
            "chic0": "#chi_{c0}(1P)",
            "chic1": "#chi_{c1}(1P)",
            "etac_2s": "#eta_{c}(2S)",
            "background": "Combinatorial bkg."
        }
        for state in ["jpsi", "etac", "chic0", "chic1", "etac_2s", "background"]:
            legend.AddEntry(state, state_labels[state], "l")
        legend.Draw()
        
        # Create and draw pull distribution in lower pad
        pad2.cd()
        pull_frame = mass_var.frame(ROOT.RooFit.Title(""))
        pull_hist = frame.pullHist("data", "total")
        pull_frame.addPlotable(pull_hist, "P")
        pull_frame.GetYaxis().SetTitle("Pull")
        pull_frame.GetYaxis().SetTitleSize(0.15)
        pull_frame.GetYaxis().SetLabelSize(0.12)
        pull_frame.GetYaxis().SetTitleOffset(0.35)
        pull_frame.GetYaxis().SetNdivisions(505)
        pull_frame.GetYaxis().SetRangeUser(-5.0, 5.0)
        pull_frame.GetXaxis().SetTitle("m(#bar{#Lambda}pK^{#minus}) [MeV/#it{c}^{2}]")
        pull_frame.GetXaxis().SetTitleSize(0.15)
        pull_frame.GetXaxis().SetLabelSize(0.12)
        pull_frame.GetXaxis().SetTitleOffset(1.0)
        pull_frame.Draw()
        
        # Add horizontal line at zero
        line = ROOT.TLine(self.fit_range[0], 0.0, self.fit_range[1], 0.0)
        line.SetLineColor(ROOT.kRed)
        line.SetLineStyle(2)
        line.SetLineWidth(2)
        line.Draw()
        
        # Save plot
        canvas.cd()
        plot_dir = Path(self.config.paths["output"]["plots_dir"]) / "fits"
        plot_dir.mkdir(exist_ok=True, parents=True)
        output_file = plot_dir / f"mass_fit_{year}.pdf"
        canvas.SaveAs(str(output_file))
        print(f"  ✓ Saved fit plot: {output_file}")
        return str(output_file)
