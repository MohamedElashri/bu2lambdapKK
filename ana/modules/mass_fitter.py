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
        
        # Shared parameters across years (will be created on first use)
        self.masses = {}      # M_J/ψ, M_ηc, M_χc0, M_χc1
        self.widths = {}      # Γ states (some fixed, some floating)
        self.resolution = None  # Single resolution parameter (shared)
        
        # Observable (shared across all fits)
        self.mass_var = None
        
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
        yields = {}
        
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
            yields[state] = yield_var
            
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
        yields["background"] = bkg_yield
        
        pdf_list.add(bkg_pdf)
        coef_list.add(bkg_yield)
        
        # Build extended sum PDF
        total_pdf = ROOT.RooAddPdf(
            f"model_{year}",
            f"Total PDF {year}",
            pdf_list,
            coef_list
        )
        
        return total_pdf, yields
    
    def perform_fit(self, data_by_year: Dict[str, ak.Array]) -> Dict[str, Any]:
        """
        Perform mass fits to all years
        
        Strategy:
        1. Fit each year separately
        2. Share physical parameters (masses, widths, resolution) across years
        3. Extract separate yields per year
        
        Args:
            data_by_year: {year: awkward_array} with M_LpKm_h2 branch
            
        Returns:
            {
                "yields": {year: {state: (value, error)}},
                "masses": {state: (value, error)},
                "widths": {state: (value, error)},
                "resolution": (value, error),
                "fit_results": {year: RooFitResult}
            }
        """
        mass_var = self.setup_observable()
        
        all_yields = {}
        all_fit_results = {}
        
        print("\n" + "="*80)
        print("MASS FITTING WITH ROOFIT")
        print("="*80)
        print(f"Fit range: {self.fit_range[0]} - {self.fit_range[1]} MeV")
        print(f"Using M_LpKm_h2 branch (h2 = K-, correct for charmonium)")
        print("="*80)
        
        for year in sorted(data_by_year.keys()):
            print(f"\n[Year {year}]")
            
            # Get mass data (use M_LpKm_h2 - correct branch for charmonium)
            mass_array = data_by_year[year]["M_LpKm_h2"]
            
            # Apply fit range filter
            mask = (mass_array >= self.fit_range[0]) & (mass_array <= self.fit_range[1])
            mass_filtered = mass_array[mask]
            
            print(f"  Events in fit range: {len(mass_filtered)}")
            
            # Convert to numpy for RooDataSet
            mass_np = ak.to_numpy(mass_filtered)
            
            # Create RooDataSet using TTree-based approach
            temp_filename = f"temp_fit_{year}.root"
            temp_file = ROOT.TFile(temp_filename, "RECREATE")
            temp_tree = ROOT.TTree("tree", "Mass data")
            
            # Create branch
            mass_value = np.zeros(1, dtype=float)
            temp_tree.Branch("M_LpKm", mass_value, "M_LpKm/D")
            
            # Fill tree
            for m in mass_np:
                mass_value[0] = m
                temp_tree.Fill()
            
            temp_tree.Write()
            temp_file.Close()
            
            # Load back and create RooDataSet
            temp_file = ROOT.TFile(temp_filename, "READ")
            temp_tree = temp_file.Get("tree")
            
            dataset = ROOT.RooDataSet(
                f"data_{year}",
                f"Data {year}",
                temp_tree,
                ROOT.RooArgSet(mass_var)
            )
            
            print(f"  RooDataSet entries: {dataset.numEntries()}")
            
            # Build model for this year
            model, yields = self.build_model_for_year(year, mass_var)
            
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
            year_yields = {}
            print(f"\n  Yields for {year}:")
            for state, yield_var in yields.items():
                value = yield_var.getVal()
                error = yield_var.getError()
                year_yields[state] = (value, error)
                
                print(f"    N_{state:<12} = {value:8.0f} ± {error:6.0f}")
            
            all_yields[year] = year_yields
            all_fit_results[year] = fit_result
            
            # Generate fit plot
            self.plot_fit_result(year, mass_var, dataset, model, yields)
            
            # Cleanup
            temp_file.Close()
            os.remove(temp_filename)
        
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
        Plot fit result with data, total fit, and components
        
        Creates LHCb-style plot with:
        - Data points with Poisson errors
        - Total fit curve
        - Individual signal components
        - Background component
        - Pull distribution
        
        Args:
            year: Year string
            mass_var: Observable mass variable
            dataset: RooDataSet with data
            model: Total PDF
            yields: Dictionary of yield parameters
        """
        # Create RooPlot
        frame = mass_var.frame(ROOT.RooFit.Title(f"Mass Fit - Year {year}"))
        
        # Plot data
        dataset.plotOn(frame, ROOT.RooFit.Name("data"), ROOT.RooFit.MarkerSize(0.5))
        
        # Plot total PDF
        model.plotOn(frame, ROOT.RooFit.Name("total"), 
                    ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.LineWidth(2))
        
        # Plot individual components
        colors = {
            "jpsi": ROOT.kRed,
            "etac": ROOT.kGreen + 2,
            "chic0": ROOT.kMagenta,
            "chic1": ROOT.kOrange + 7,
            "background": ROOT.kGray + 1
        }
        
        for state in ["jpsi", "etac", "chic0", "chic1", "background"]:
            component_name = f"pdf_signal_{state}" if state != "background" else f"pdf_bkg_{year}"
            model.plotOn(frame, ROOT.RooFit.Components(component_name),
                        ROOT.RooFit.LineColor(colors[state]),
                        ROOT.RooFit.LineStyle(ROOT.kDashed),
                        ROOT.RooFit.LineWidth(2))
        
        # Calculate chi2
        chi2 = frame.chiSquare()
        
        # Create canvas
        canvas = ROOT.TCanvas(f"c_{year}", f"Fit {year}", 800, 600)
        canvas.Divide(1, 2)
        
        # Top pad: data + fit
        pad1 = canvas.cd(1)
        pad1.SetPad(0, 0.3, 1, 1)
        pad1.SetBottomMargin(0.02)
        
        frame.GetYaxis().SetTitle("Events / 5 MeV")
        frame.GetYaxis().SetTitleSize(0.05)
        frame.GetYaxis().SetLabelSize(0.04)
        frame.Draw()
        
        # Add legend
        legend = ROOT.TLegend(0.65, 0.5, 0.88, 0.88)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.AddEntry("data", "Data", "lep")
        legend.AddEntry("total", "Total fit", "l")
        legend.AddEntry(0, f"#chi^{{2}}/ndf = {chi2:.2f}", "")
        legend.AddEntry(0, "", "")
        
        # Add yields to legend
        for state in ["jpsi", "etac", "chic0", "chic1"]:
            n_val = yields[state].getVal()
            n_err = yields[state].getError()
            label = f"N_{state} = {n_val:.0f} #pm {n_err:.0f}"
            legend.AddEntry(0, label, "")
        
        legend.Draw()
        
        # Bottom pad: pulls
        pad2 = canvas.cd(2)
        pad2.SetPad(0, 0, 1, 0.3)
        pad2.SetTopMargin(0.02)
        pad2.SetBottomMargin(0.3)
        
        # Create pull distribution
        pull_frame = mass_var.frame(ROOT.RooFit.Title(""))
        pull_hist = frame.pullHist("data", "total")
        pull_frame.addPlotable(pull_hist, "P")
        
        pull_frame.GetYaxis().SetTitle("Pull [#sigma]")
        pull_frame.GetYaxis().SetTitleSize(0.12)
        pull_frame.GetYaxis().SetLabelSize(0.10)
        pull_frame.GetYaxis().SetNdivisions(505)
        pull_frame.GetYaxis().SetTitleOffset(0.4)
        pull_frame.GetXaxis().SetTitle("M(#bar{#Lambda}pK^{-}) [MeV/c^{2}]")
        pull_frame.GetXaxis().SetTitleSize(0.12)
        pull_frame.GetXaxis().SetLabelSize(0.10)
        pull_frame.SetMinimum(-5)
        pull_frame.SetMaximum(5)
        
        pull_frame.Draw()
        
        # Add zero line
        line = ROOT.TLine(self.fit_range[0], 0, self.fit_range[1], 0)
        line.SetLineColor(ROOT.kBlue)
        line.SetLineStyle(2)
        line.Draw()
        
        # Save plot
        plot_dir = Path(self.config.paths["output"]["plots_dir"]) / "fits"
        plot_dir.mkdir(exist_ok=True, parents=True)
        
        output_file = plot_dir / f"mass_fit_{year}.pdf"
        canvas.SaveAs(str(output_file))
        
        print(f"  ✓ Saved fit plot: {output_file}")
