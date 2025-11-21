"""
Mass Fitting Module for B+ -> Lambda pK-K+ Analysis

Implements RooFit-based simultaneous mass fitting for charmonium states.
MODIFIED: plot_fit_result method updated to match official LHCb publication style
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import awkward as ak
import ROOT  # type: ignore
from tqdm import tqdm

# Enable RooFit batch mode for better performance
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)  # type: ignore

# Set ROOT to batch mode to prevent display/GUI issues during testing
ROOT.gROOT.SetBatch(True)  # type: ignore


class MassFitter:
    """
    RooFit-based Mass Fitting for B+ -> Lambda pK-K+ Analysis

    Fits M(Λ̄pK⁻) invariant mass distribution to extract charmonium yields.

    Strategy:
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
        self.fit_range: tuple[float, float] = config.particles["mass_windows"][
            "charmonium_fit_range"
        ]

        # B+ mass window for pre-selection (applied BEFORE fitting)
        self.bu_mass_min: float = config.selection.get("bu_fixed_selection", {}).get(
            "mass_corrected_min", 5255.0
        )
        self.bu_mass_max: float = config.selection.get("bu_fixed_selection", {}).get(
            "mass_corrected_max", 5305.0
        )

        # Fitting configuration
        fitting_config: dict[str, Any] = config.particles.get("fitting", {})
        self.use_binned_fit: bool = fitting_config.get("use_binned_fit", True)
        self.bin_width: float = fitting_config.get("bin_width", 5.0)

        # Calculate number of bins automatically: always maintain bin_width MeV/bin
        fit_range_width: float = self.fit_range[1] - self.fit_range[0]
        self.nbins: int = int(fit_range_width / self.bin_width)

        # Shared parameters across years (will be created on first use)
        self.masses: dict[str, ROOT.RooRealVar] = {}  # M_J/ψ, M_ηc, M_χc0, M_χc1 # type: ignore
        self.widths: dict[str, ROOT.RooRealVar] = (
            {}
        )  # Γ states (some fixed, some floating) # type: ignore
        self.resolution: ROOT.RooRealVar | None = (
            None  # Single resolution parameter (shared) # type: ignore
        )

        # Observable (shared across all fits)
        self.mass_var: ROOT.RooRealVar | None = None  # type: ignore

        # Store all PDFs and variables to prevent garbage collection
        self.signal_pdfs: dict[str, ROOT.RooAbsPdf] = {}  # {state: pdf} # type: ignore
        self.bkg_pdfs: dict[str, ROOT.RooAbsPdf] = {}  # {year: pdf} # type: ignore
        self.argus_params: dict[str, dict[str, Any]] = {}  # {year: {param: value}}

        self.models: dict[str, ROOT.RooAbsPdf] = {}  # {year: model} # type: ignore
        self.yields: dict[str, dict[str, ROOT.RooRealVar]] = (
            {}
        )  # {year: {state: yield_var}} # type: ignore

    def setup_observable(self) -> ROOT.RooRealVar:  # type: ignore
        """
        Define RooRealVar for M(Λ̄pK⁻) invariant mass.

        Returns:
            ROOT.RooRealVar for mass observable
        """
        if self.mass_var is None:
            self.mass_var = ROOT.RooRealVar(  # type: ignore
                "M_LpKm", "M(#bar{#Lambda}pK^{-}) [MeV/c^{2}]", self.fit_range[0], self.fit_range[1]
            )
            # Set binning for plotting and binned fits
            self.mass_var.setBins(self.nbins)  # type: ignore
        return self.mass_var

    def create_signal_pdf(self, state: str, mass_var: ROOT.RooRealVar) -> ROOT.RooAbsPdf:  # type: ignore
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
            "etac_2s": "etac_2s",
        }
        config_key = config_key_map.get(state_lower, state_lower)

        # Mass parameter (shared across years) - FIXED TO PDG VALUE
        if state_lower not in self.masses:
            pdg_mass = self.config.particles["pdg_masses"][config_key]

            self.masses[state_lower] = ROOT.RooRealVar(  # type: ignore
                f"M_{state}", f"M_{state} [MeV/c^{{2}}]", pdg_mass
            )
            # Fix mass to PDG value
            self.masses[state_lower].setConstant(True)

        # Width parameter - FIXED TO PDG VALUE FOR ALL STATES
        if state_lower not in self.widths:
            pdg_width = self.config.particles["pdg_widths"][config_key]

            self.widths[state_lower] = ROOT.RooRealVar(  # type: ignore
                f"Gamma_{state}", f"#Gamma_{state} [MeV/c^{{2}}]", pdg_width
            )
            # Fix width to PDG value
            self.widths[state_lower].setConstant(True)

        # Resolution (shared Gaussian width - same detector for all states)
        if self.resolution is None:
            self.resolution = ROOT.RooRealVar(  # type: ignore
                "sigma_resolution",
                "#sigma_{resolution} [MeV/c^{2}]",
                5.0,  # Initial guess ~5 MeV
                1.0,  # Minimum
                20.0,  # Maximum
            )

        # Create Voigtian PDF (RBW ⊗ Gaussian)
        signal_pdf = ROOT.RooVoigtian(  # type: ignore
            f"pdf_signal_{state}",
            f"Signal PDF for {state}",
            mass_var,
            self.masses[state_lower],
            self.widths[state_lower],
            self.resolution,
        )

        # Cache the PDF
        self.signal_pdfs[state_lower] = signal_pdf

        return signal_pdf

    def create_background_pdf(
        self, mass_var: ROOT.RooRealVar, year: str  # type: ignore
    ) -> tuple[ROOT.RooAbsPdf, ROOT.RooRealVar]:  # type: ignore
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
        endpoint_offset = self.config.particles.get("fitting", {}).get(
            "argus_endpoint_offset", 200.0
        )

        m0 = ROOT.RooRealVar(  # type: ignore
            f"m0_argus_{year}",
            "ARGUS endpoint [MeV/c^{2}]",
            self.fit_range[1] + endpoint_offset,  # Extended beyond fit range
        )
        m0.setConstant(True)

        # ARGUS shape parameter (fitted per year)
        c = ROOT.RooRealVar(  # type: ignore
            f"c_argus_{year}",
            f"c_{{ARGUS}} {year}",
            -20.0,  # Initial guess
            -100.0,  # Minimum
            -0.1,  # Maximum (must be negative)
        )

        # Power parameter (typically fixed to 0.5)
        p = ROOT.RooRealVar(f"p_argus_{year}", "ARGUS power", 0.5)  # type: ignore
        p.setConstant(True)

        # Create ARGUS PDF
        bkg_pdf = ROOT.RooArgusBG(f"pdf_bkg_{year}", f"Background PDF {year}", mass_var, m0, c, p)  # type: ignore

        # Initialize storage if needed
        if not hasattr(self, "argus_params"):
            self.argus_params = {}

        # Cache ALL parameters to prevent garbage collection
        self.bkg_pdfs[year] = bkg_pdf
        self.argus_params[year] = {"m0": m0, "c": c, "p": p}

        return bkg_pdf, c

    def build_model_for_year(
        self, year: str, mass_var: ROOT.RooRealVar  # type: ignore
    ) -> tuple[ROOT.RooAbsPdf, dict[str, ROOT.RooRealVar]]:  # type: ignore
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
        pdf_list = ROOT.RooArgList()  # type: ignore
        coef_list = ROOT.RooArgList()  # type: ignore
        year_yields = {}

        # Signal components (5 charmonium states)
        for state in ["jpsi", "etac", "chic0", "chic1", "etac_2s"]:
            # Create signal PDF (shares mass/width/resolution across years)
            sig_pdf = self.create_signal_pdf(state, mass_var)

            # Create yield parameter (separate per year)
            yield_var = ROOT.RooRealVar(  # type: ignore
                f"N_{state}_{year}",
                f"Yield {state} {year}",
                1000,  # Initial guess
                0,  # Minimum (non-negative)
                1e6,  # Maximum
            )
            year_yields[state] = yield_var

            # Add to lists
            pdf_list.add(sig_pdf)
            coef_list.add(yield_var)

        # Background component
        bkg_pdf, alpha_bkg = self.create_background_pdf(mass_var, year)

        bkg_yield = ROOT.RooRealVar(  # type: ignore
            f"N_bkg_{year}",
            f"Background yield {year}",
            10000,  # Initial guess
            0,  # Minimum
            1e7,  # Maximum
        )
        year_yields["background"] = bkg_yield

        pdf_list.add(bkg_pdf)
        coef_list.add(bkg_yield)

        # Build extended sum PDF
        total_pdf = ROOT.RooAddPdf(f"model_{year}", f"Total PDF {year}", pdf_list, coef_list)  # type: ignore

        # Store model and yields to prevent garbage collection
        self.models[year] = total_pdf
        self.yields[year] = year_yields

        return total_pdf, year_yields

    def perform_fit(
        self, data_by_year: dict[str, ak.Array], fit_combined: bool = True
    ) -> dict[str, Any]:
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

        print("\n" + "=" * 80)
        print("MASS FITTING WITH ROOFIT - ALL 5 CHARMONIUM STATES")
        print("=" * 80)
        print(f"Charmonium fit range: {self.fit_range[0]} - {self.fit_range[1]} MeV")
        print(f"B+ mass window: {self.bu_mass_min} - {self.bu_mass_max} MeV")
        print("Using M_LpKm_h2 (M(Λ̄pK⁻), h2=K⁻ correct for charmonium)")
        print("Modeling: ηc(1S), J/ψ, χc0, χc1, ηc(2S), ψ(3770) simultaneously")
        print("All masses and widths FIXED to PDG values")
        print(f"Fit type: {'BINNED' if self.use_binned_fit else 'UNBINNED'} maximum likelihood")
        print(
            f"Binning: {self.nbins} bins × {self.bin_width} MeV/bin = {self.fit_range[1] - self.fit_range[0]} MeV range"
        )
        print("=" * 80)

        # Prepare all datasets (per-year + combined)
        datasets_to_fit = {}

        print("Preparing datasets...")
        with tqdm(total=len(data_by_year), desc="Preparing data", unit="year") as pbar:
            for year in sorted(data_by_year.keys()):
                pbar.set_postfix_str(f"Year {year}")

                events = data_by_year[year]

                # Apply B+ mass window cut
                bu_mass = events["Bu_MM_corrected"]  # Lambda-corrected B+ mass
                bu_mask = (bu_mass >= self.bu_mass_min) & (bu_mass <= self.bu_mass_max)
                events_bu_cut = events[bu_mask]

                print(
                    f"  Events after B+ mass cut [{self.bu_mass_min}, {self.bu_mass_max}]: {len(events_bu_cut)}"
                )

                # Get charmonium mass data (M(Λ̄pK⁻), use h2=K⁻)
                mass_array = events_bu_cut["M_LpKm_h2"]

                # Apply charmonium fit range filter
                mask = (mass_array >= self.fit_range[0]) & (mass_array <= self.fit_range[1])
                mass_filtered = mass_array[mask]

                print(
                    f"  Events in charmonium fit range [{self.fit_range[0]}, {self.fit_range[1]}]: {len(mass_filtered)}"
                )

                datasets_to_fit[year] = mass_filtered
                pbar.update(1)

        # Create combined dataset
        if fit_combined and len(datasets_to_fit) > 1:
            combined_mass = ak.concatenate(
                [datasets_to_fit[y] for y in sorted(datasets_to_fit.keys())]
            )
            datasets_to_fit["combined"] = combined_mass
            print("\n[Combined All Years]")
            print(f"  Total events: {len(combined_mass)}")

        # Fit each dataset
        print("\nFitting datasets...")
        with tqdm(total=len(datasets_to_fit), desc="Fitting", unit="dataset") as pbar:
            for dataset_name in sorted(datasets_to_fit.keys()):
                pbar.set_postfix_str(f"{dataset_name}")

                mass_filtered = datasets_to_fit[dataset_name]

                # Convert to numpy for RooDataSet/RooDataHist
                mass_np = ak.to_numpy(mass_filtered)

                # Create dataset (binned or unbinned based on configuration)
                if self.use_binned_fit:
                    # Create unbinned dataset first
                    temp_dataset = ROOT.RooDataSet(  # type: ignore
                        f"temp_data_{dataset_name}",
                        f"Temp Data {dataset_name}",
                        ROOT.RooArgSet(mass_var),  # type: ignore
                    )
                    for m in mass_np:
                        mass_var.setVal(m)
                        temp_dataset.add(ROOT.RooArgSet(mass_var))  # type: ignore

                    # Convert to binned dataset (RooDataHist)
                    dataset = ROOT.RooDataHist(  # type: ignore
                        f"data_{dataset_name}",
                        f"Data {dataset_name}",
                        ROOT.RooArgSet(mass_var),  # type: ignore
                        temp_dataset,
                    )
                else:
                    # Create unbinned dataset (RooDataSet)
                    dataset = ROOT.RooDataSet(  # type: ignore
                        f"data_{dataset_name}", f"Data {dataset_name}", ROOT.RooArgSet(mass_var)  # type: ignore
                    )
                    for m in mass_np:
                        mass_var.setVal(m)
                        dataset.add(ROOT.RooArgSet(mass_var))  # type: ignore

                # Build model for this dataset
                model, yields = self.build_model_for_year(dataset_name, mass_var)

                # Perform fit
                if self.use_binned_fit:
                    # Binned maximum likelihood fit
                    fit_result = model.fitTo(
                        dataset,
                        ROOT.RooFit.Save(),  # type: ignore
                        ROOT.RooFit.Extended(True),  # type: ignore
                        ROOT.RooFit.PrintLevel(-1),  # type: ignore
                        ROOT.RooFit.NumCPU(4),  # type: ignore
                        ROOT.RooFit.Strategy(2),  # More robust # type: ignore
                    )
                else:
                    # Unbinned maximum likelihood fit
                    fit_result = model.fitTo(
                        dataset,
                        ROOT.RooFit.Save(),  # type: ignore
                        ROOT.RooFit.Extended(True),  # type: ignore
                        ROOT.RooFit.PrintLevel(-1),  # type: ignore
                        ROOT.RooFit.NumCPU(4),  # type: ignore
                        ROOT.RooFit.Strategy(2),  # More robust # type: ignore
                    )

                # Check convergence
                status = fit_result.status()

                if status != 0:
                    pbar.write(f"  WARNING: Fit for {dataset_name} did not converge properly!")

                # Extract yields with errors
                dataset_yields = {}
                for state, yield_var in yields.items():
                    value = yield_var.getVal()
                    error = yield_var.getError()
                    dataset_yields[state] = (value, error)

                all_yields[dataset_name] = dataset_yields
                all_fit_results[dataset_name] = fit_result

                # Plot results (pass fit_result for quality metrics)
                self.plot_fit_result(dataset_name, mass_var, dataset, model, yields, fit_result)

                pbar.update(1)

        # Extract shared parameters (from last fit)
        print("\n" + "=" * 80)
        print("FITTED PARAMETERS (shared across years)")
        print("=" * 80)

        masses_result = {}
        widths_result = {}

        for state in ["jpsi", "etac", "chic0", "chic1", "etac_2s"]:
            mass_val = self.masses[state].getVal()
            mass_err = self.masses[state].getError()
            masses_result[state] = (mass_val, mass_err)

            width_val = self.widths[state].getVal()
            width_err = self.widths[state].getError()
            widths_result[state] = (width_val, width_err)

            print(
                f"{state:>10}: M = {mass_val:7.2f} ± {mass_err:5.2f} MeV,  "
                f"Γ = {width_val:6.2f} ± {width_err:5.2f} MeV"
            )

        res_val = self.resolution.getVal()  # type: ignore
        res_err = self.resolution.getError()  # type: ignore
        print(f"\nResolution: σ = {res_val:.2f} ± {res_err:.2f} MeV")

        return {
            "yields": all_yields,
            "masses": masses_result,
            "widths": widths_result,
            "resolution": (res_val, res_err),
            "fit_results": all_fit_results,
        }

    def plot_fit_result(
        self,
        year: str,
        mass_var: Any,
        dataset: Any,
        model: Any,
        yields: dict[str, Any],
        fit_result: Any = None,
    ) -> None:
        """
        Plot fit result with official LHCb publication style

        Creates LHCb-style two-panel plot with:
        - Upper panel: fit with data, model components
        - Lower panel: pull distribution
        - Data points with error bars
        - Total fit curve (solid blue)
        - Signal components (solid red lines)
        - Background component (dashed gray)
        - Compact legend in upper right
        - Year label on plot
        - Dynamic particle labels positioned above peaks

        Args:
            year: Year string (or "combined")
            mass_var: Observable mass variable
            dataset: RooDataSet or RooDataHist with data
            model: Total PDF
            yields: Dictionary of yield parameters
        """
        # Create RooPlot (no title, we'll add it as text)
        frame = mass_var.frame(ROOT.RooFit.Title(""))  # type: ignore

        # Plot data with error bars (black points, small size)
        dataset.plotOn(
            frame,
            ROOT.RooFit.Name("data"),  # type: ignore
            ROOT.RooFit.MarkerStyle(20),  # type: ignore
            ROOT.RooFit.MarkerSize(0.6),  # type: ignore
            ROOT.RooFit.MarkerColor(ROOT.kBlack),  # type: ignore
        )

        # Plot total PDF (solid blue, matching LHCb style)
        model.plotOn(
            frame,
            ROOT.RooFit.Name("total"),  # type: ignore
            ROOT.RooFit.LineColor(ROOT.kBlue),  # type: ignore
            ROOT.RooFit.LineWidth(2),  # type: ignore
            ROOT.RooFit.LineStyle(ROOT.kSolid),  # type: ignore
        )

        # Plot all signal components in red (solid lines, matching official style)
        for state in ["jpsi", "etac", "chic0", "chic1", "etac_2s"]:
            component_name = f"pdf_signal_{state}"
            model.plotOn(
                frame,
                ROOT.RooFit.Components(component_name),  # type: ignore
                ROOT.RooFit.Name(state),  # type: ignore
                ROOT.RooFit.LineColor(ROOT.kRed),  # type: ignore
                ROOT.RooFit.LineStyle(ROOT.kSolid),  # type: ignore
                ROOT.RooFit.LineWidth(2),  # type: ignore
            )

        # Plot background with dashed gray line
        bkg_component_name = f"pdf_bkg_{year}"
        model.plotOn(
            frame,
            ROOT.RooFit.Components(bkg_component_name),  # type: ignore
            ROOT.RooFit.Name("background"),  # type: ignore
            ROOT.RooFit.LineColor(ROOT.kGray + 1),  # type: ignore
            ROOT.RooFit.LineStyle(ROOT.kDashed),  # type: ignore
            ROOT.RooFit.LineWidth(2),  # type: ignore
        )

        # Create canvas with two pads (fit on top, pulls on bottom)
        canvas = ROOT.TCanvas(f"c_{year}", f"Fit {year}", 1200, 800)  # type: ignore
        # Prevent Python from managing canvas memory (let ROOT handle it)
        ROOT.SetOwnership(canvas, False)  # type: ignore

        # Upper pad for fit (70% of canvas)
        pad1 = ROOT.TPad("pad1", "Fit", 0.0, 0.30, 1.0, 1.0)  # type: ignore
        pad1.SetBottomMargin(0.015)
        pad1.SetLeftMargin(0.07)
        pad1.SetRightMargin(0.05)
        pad1.SetTopMargin(0.07)
        pad1.Draw()

        # Lower pad for pulls (30% of canvas)
        pad2 = ROOT.TPad("pad2", "Pulls", 0.0, 0.0, 1.0, 0.30)  # type: ignore
        pad2.SetTopMargin(0.015)
        pad2.SetBottomMargin(0.35)
        pad2.SetLeftMargin(0.07)
        pad2.SetRightMargin(0.05)
        pad2.SetGridy(1)
        pad2.Draw()

        # Draw fit in upper pad
        pad1.cd()

        # Style the frame for upper pad
        frame.GetYaxis().SetTitle(f"Candidates / ({self.bin_width:.0f} MeV/#it{{c}}^{{2}})")
        frame.GetYaxis().SetTitleSize(0.045)
        frame.GetYaxis().SetLabelSize(0.0375)
        frame.GetYaxis().SetTitleOffset(0.7)
        frame.GetYaxis().SetTitleFont(42)
        frame.GetYaxis().SetLabelFont(42)
        frame.GetXaxis().SetLabelSize(0.0)  # Hide x-axis labels on upper pad
        frame.GetXaxis().SetTitleSize(0.0)

        frame.SetTitle("")

        # Set Y-axis range with margin at top for labels
        y_max = frame.GetMaximum()
        frame.SetMaximum(y_max * 1.50)  # More headroom for higher labels
        frame.SetMinimum(0.0)

        frame.Draw()

        # Add LHCb label in top left corner
        lhcb_label = ROOT.TLatex()  # type: ignore
        lhcb_label.SetNDC()
        lhcb_label.SetTextFont(42)
        lhcb_label.SetTextSize(0.06)
        lhcb_label.DrawLatex(0.12, 0.87, "LHCb")

        # Add year label below LHCb
        year_label = ROOT.TLatex()  # type: ignore
        year_label.SetNDC()
        year_label.SetTextFont(42)
        year_label.SetTextSize(0.05)
        year_text = "2016-2018" if year == "combined" else str(year)
        year_label.DrawLatex(0.12, 0.81, year_text)

        # Calculate pull statistics early (before drawing info boxes)
        pull_hist = frame.pullHist("data", "total")
        pull_mean = 0.0
        pull_rms = 0.0
        n_pulls = 0

        # Get pull values from histogram
        for i in range(pull_hist.GetN()):
            x = pull_hist.GetPointX(i)
            y = pull_hist.GetPointY(i)
            if abs(y) < 10:  # Exclude outliers
                pull_mean += y
                pull_rms += y * y
                n_pulls += 1

        if n_pulls > 0:
            pull_mean /= n_pulls
            pull_rms = (pull_rms / n_pulls - pull_mean * pull_mean) ** 0.5

        # Add fit info box to the left of legend - fancy style with two columns
        fit_info_left = ROOT.TPaveText(0.40, 0.60, 0.58, 0.90, "NDC")  # type: ignore
        fit_info_left.SetBorderSize(2)
        fit_info_left.SetLineColor(ROOT.kBlue)  # type: ignore
        fit_info_left.SetFillColor(ROOT.kWhite)  # type: ignore
        fit_info_left.SetFillStyle(1001)
        fit_info_left.SetTextAlign(12)  # Left-aligned
        fit_info_left.SetTextFont(42)
        fit_info_left.SetTextSize(0.032)
        fit_info_left.SetTextColor(ROOT.kBlack)  # type: ignore

        fit_info_right = ROOT.TPaveText(0.58, 0.60, 0.76, 0.90, "NDC")  # type: ignore
        fit_info_right.SetBorderSize(2)
        fit_info_right.SetLineColor(ROOT.kBlue)  # type: ignore
        fit_info_right.SetFillColor(ROOT.kWhite)  # type: ignore
        fit_info_right.SetFillStyle(1001)
        fit_info_right.SetTextAlign(12)  # Left-aligned
        fit_info_right.SetTextFont(42)
        fit_info_right.SetTextSize(0.032)
        fit_info_right.SetTextColor(ROOT.kBlack)  # type: ignore

        # Get fit statistics (top 10 most important quantities)
        total_events = sum(y.getVal() for y in yields.values())
        n_jpsi = yields["jpsi"].getVal()
        n_jpsi_err = yields["jpsi"].getError()
        n_etac = yields["etac"].getVal()
        n_etac_err = yields["etac"].getError()
        n_chic0 = yields["chic0"].getVal()
        n_chic0_err = yields["chic0"].getError()
        n_chic1 = yields["chic1"].getVal()
        n_chic1_err = yields["chic1"].getError()
        n_etac2s = yields["etac_2s"].getVal()
        n_etac2s_err = yields["etac_2s"].getError()
        n_bkg = yields["background"].getVal()
        n_bkg_err = yields["background"].getError()
        sigma_res = self.resolution.getVal()  # type: ignore
        sigma_res_err = self.resolution.getError()  # type: ignore

        # Left column - yields
        fit_info_left.AddText("#bf{Yields}")
        fit_info_left.AddText(f"N_{{J/#psi}} = {n_jpsi:.0f} #pm {n_jpsi_err:.0f}")
        fit_info_left.AddText(f"N_{{#eta_{{c}}}} = {n_etac:.0f} #pm {n_etac_err:.0f}")
        fit_info_left.AddText(f"N_{{#chi_{{c0}}}} = {n_chic0:.0f} #pm {n_chic0_err:.0f}")
        fit_info_left.AddText(f"N_{{#chi_{{c1}}}} = {n_chic1:.0f} #pm {n_chic1_err:.0f}")
        fit_info_left.AddText(f"N_{{#eta_{{c}}(2S)}} = {n_etac2s:.0f} #pm {n_etac2s_err:.0f}")

        # Right column - fit info
        # Calculate fit quality metrics
        n_signal = n_jpsi + n_etac + n_chic0 + n_chic1 + n_etac2s

        # Get fit quality from fit result if available
        fit_status = "N/A"
        edm = -1
        if fit_result is not None:
            fit_status = "OK" if fit_result.status() == 0 else "FAILED"
            edm = fit_result.edm()

        fit_info_right.AddText("#bf{Fit Info}")
        fit_info_right.AddText(f"N_{{bkg}} = {n_bkg:.0f} #pm {n_bkg_err:.0f}")
        fit_info_right.AddText(f"N_{{sig}} = {n_signal:.0f}")
        fit_info_right.AddText(f"N_{{tot}} = {total_events:.0f}")
        fit_info_right.AddText(f"#sigma_{{res}} = {sigma_res:.1f} #pm {sigma_res_err:.1f} MeV")
        fit_info_right.AddText(f"Pull: #mu = {pull_mean:.2f}, #sigma = {pull_rms:.2f}")

        fit_info_left.Draw()
        fit_info_right.Draw()

        # Add compact legend in upper right (official LHCb style)
        legend = ROOT.TLegend(0.77, 0.60, 0.93, 0.90)  # type: ignore
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)  # Transparent
        legend.SetFillColor(0)
        legend.SetTextSize(0.04)
        legend.SetTextFont(42)
        legend.SetMargin(0.15)

        # Add legend entries (simplified, matching official style)
        legend.AddEntry("data", "Data", "l")  # Use line instead of lep for smaller indicator
        legend.AddEntry("total", "Full model", "l")
        legend.AddEntry("jpsi", "Signal", "l")  # All signals shown as one entry
        legend.AddEntry("background", "Background", "l")

        legend.Draw()

        # Dynamically position state labels ABOVE peaks
        state_text = ROOT.TLatex()  # type: ignore
        state_text.SetTextFont(42)
        state_text.SetTextSize(0.04)
        state_text.SetTextAlign(21)  # Center-aligned, bottom-aligned

        # Define states to label with their LaTeX names
        states_to_label = {
            "etac": ("#eta_{c}", 2983.9),  # ηc(1S)
            "jpsi": ("J/#psi", 3096.9),  # J/ψ
            "chic0": ("#chi_{c0}", 3414.1),  # χc0
            "chic1": ("#chi_{c1}", 3510.7),  # χc1
            "etac_2s": ("#eta_{c}(2S)", 3637.6),  # ηc(2S)
        }

        # Evaluate model at each state's mass to get peak height
        for state, (label, mass_pdg) in states_to_label.items():
            # Set mass variable to this state's mass
            mass_var.setVal(mass_pdg)

            # Get the total PDF value at this mass (normalized to bin content)
            pdf_value = model.getVal(ROOT.RooArgSet(mass_var))  # type: ignore

            # Convert PDF value to plot coordinates (multiply by number of events and bin width)
            total_events = sum(y.getVal() for y in yields.values())
            bin_content = pdf_value * total_events * self.bin_width

            # Position label well above the peak to avoid error bars overlap
            # Individual settings for each particle and year
            # Format: {state: {"combined": mult, "2016": mult, "2017": mult, "2018": mult}}
            label_multipliers = {
                "etac": {
                    "combined": 1.30,  # ηc(1S) combined
                    "2016": 1.55,  # ηc(1S) 2016
                    "2017": 1.30,  # ηc(1S) 2017
                    "2018": 1.40,  # ηc(1S) 2018
                },
                "jpsi": {
                    "combined": 1.15,  # J/ψ combined
                    "2016": 1.30,  # J/ψ 2016
                    "2017": 1.30,  # J/ψ 2017
                    "2018": 1.10,  # J/ψ 2018
                },
                "chic0": {
                    "combined": 1.65,  # χc0 combined
                    "2016": 1.90,  # χc0 2016
                    "2017": 1.90,  # χc0 2017
                    "2018": 1.90,  # χc0 2018
                },
                "chic1": {
                    "combined": 1.40,  # χc1 combined
                    "2016": 1.90,  # χc1 2016
                    "2017": 1.90,  # χc1 2017
                    "2018": 1.90,  # χc1 2018
                },
                "etac_2s": {
                    "combined": 1.55,  # ηc(2S) combined
                    "2016": 2.00,  # ηc(2S) 2016
                    "2017": 1.90,  # ηc(2S) 2017
                    "2018": 1.90,  # ηc(2S) 2018
                },
            }

            # Get the appropriate multiplier for this state and year
            # Use default multiplier for unknown years (e.g., test data)
            if year in label_multipliers[state]:
                label_y = bin_content * label_multipliers[state][year]
            else:
                # Default to combined multiplier for unknown years
                label_y = bin_content * label_multipliers[state].get("combined", 1.30)

            # Don't place label if peak is too small (less than 5% of max)
            if bin_content > y_max * 0.05:
                state_text.DrawLatex(mass_pdg, label_y, label)

        # Create and draw pull distribution in lower pad
        pad2.cd()
        pull_frame = mass_var.frame(ROOT.RooFit.Title(""))  # No title # type: ignore
        pull_frame.addPlotable(pull_hist, "P")

        # Style pull plot axes
        pull_frame.GetYaxis().SetTitle("Pull")
        pull_frame.GetYaxis().SetTitleSize(0.0975)
        pull_frame.GetYaxis().SetLabelSize(0.0825)
        pull_frame.GetYaxis().SetTitleOffset(0.3)
        pull_frame.GetYaxis().SetTitleFont(42)
        pull_frame.GetYaxis().SetLabelFont(42)
        pull_frame.GetYaxis().SetNdivisions(505)
        pull_frame.GetYaxis().SetRangeUser(-4.5, 4.5)
        pull_frame.GetYaxis().CenterTitle()

        pull_frame.GetXaxis().SetTitle("M(#bar{#Lambda}pK^{#minus}) [MeV/#it{c}^{2}]")
        pull_frame.GetXaxis().SetTitleSize(0.13)
        pull_frame.GetXaxis().SetLabelSize(0.11)
        pull_frame.GetXaxis().SetTitleOffset(1.1)
        pull_frame.GetXaxis().SetTitleFont(42)
        pull_frame.GetXaxis().SetLabelFont(42)
        pull_frame.Draw()

        # Remove any title that might appear
        pull_frame.SetTitle("")

        # Add horizontal reference lines at 0, ±3σ
        line_zero = ROOT.TLine(self.fit_range[0], 0.0, self.fit_range[1], 0.0)  # type: ignore
        line_zero.SetLineColor(ROOT.kBlack)  # type: ignore
        line_zero.SetLineStyle(1)
        line_zero.SetLineWidth(1)
        line_zero.Draw()

        # Add ±3σ reference lines (dashed gray)
        line_plus3 = ROOT.TLine(self.fit_range[0], 3.0, self.fit_range[1], 3.0)  # type: ignore
        line_plus3.SetLineColor(ROOT.kGray + 1)  # type: ignore
        line_plus3.SetLineStyle(2)
        line_plus3.SetLineWidth(1)
        line_plus3.Draw()

        line_minus3 = ROOT.TLine(self.fit_range[0], -3.0, self.fit_range[1], -3.0)  # type: ignore
        line_minus3.SetLineColor(ROOT.kGray + 1)  # type: ignore
        line_minus3.SetLineStyle(2)
        line_minus3.SetLineWidth(1)
        line_minus3.Draw()

        # Save plot
        canvas.cd()
        plot_dir = Path(self.config.paths["output"]["plots_dir"]) / "fits"
        plot_dir.mkdir(exist_ok=True, parents=True)
        output_file = plot_dir / f"mass_fit_{year}.pdf"
        canvas.SaveAs(str(output_file))
        print(f"  ✓ Saved fit plot: {output_file}")

        # Clean up canvas properly to prevent ROOT segfault during garbage collection
        canvas.Clear()
        canvas.Close()
