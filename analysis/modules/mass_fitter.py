"""
Mass Fitting Module for B+ -> Lambda pK-K+ Analysis

Implements RooFit-based simultaneous mass fitting for charmonium states.
MODIFIED: plot_fit_result method updated to match official LHCb publication style
"""

from __future__ import annotations

import math
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

    def __init__(self, config: Any, systematic_params: dict | None = None) -> None:
        """
        Initialize mass fitter with configuration.

        Args:
            config: TOMLConfig object with particles, paths configuration
            systematic_params: Optional overrides for systematic variation fits.
                Recognised keys:
                  "bkg_model"              : "argus" (default) | "poly2" (Chebyshev)
                  "argus_endpoint_offset"  : float override (default from config, usually 200 MeV)
                  "signal_resolution_fixed": float — fix resolution to this value (MeV) instead
                                             of floating; used for ±1σ resolution systematics
        """
        self.config: Any = config
        self.systematic_params: dict = systematic_params or {}
        self.fit_range: tuple[float, float] = config.mass_windows["charmonium_fit_range"]

        # B+ mass window for pre-selection (applied BEFORE fitting)
        self.bu_mass_min: float = config.mass_windows.get("bu_corrected", [5255.0, 5305.0])[0]
        self.bu_mass_max: float = config.mass_windows.get("bu_corrected", [5255.0, 5305.0])[1]

        # Fitting configuration
        self.use_binned_fit: bool = config.fitting.get("use_binned_fit", True)
        self.bin_width: float = config.fitting.get("bin_width", 5.0)

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
            # Use labels from config if available.
            labels_cfg = getattr(self.config, "labels", {})
            obs_name = labels_cfg.get("mass_observable_name", "M_LpKm")
            obs_label = labels_cfg.get(
                "mass_observable_label", "M(#bar{#Lambda}pK^{-}) [MeV/c^{2}]"
            )

            self.mass_var = ROOT.RooRealVar(  # type: ignore
                obs_name, obs_label, self.fit_range[0], self.fit_range[1]
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
            pdg_mass = self.config.pdg_masses[config_key]

            self.masses[state_lower] = ROOT.RooRealVar(  # type: ignore
                f"M_{state}", f"M_{state} [MeV/c^{{2}}]", pdg_mass
            )
            # Fix mass to PDG value
            self.masses[state_lower].setConstant(True)

        # Width parameter - FIXED TO PDG VALUE FOR ALL STATES
        if state_lower not in self.widths:
            pdg_width = self.config.pdg_widths[config_key]

            self.widths[state_lower] = ROOT.RooRealVar(  # type: ignore
                f"Gamma_{state}", f"#Gamma_{state} [MeV/c^{{2}}]", pdg_width
            )
            # Fix width to PDG value
            self.widths[state_lower].setConstant(True)

        # Resolution (shared Gaussian width - same detector for all states)
        if self.resolution is None:
            fixed_res = self.systematic_params.get("signal_resolution_fixed")
            if fixed_res is not None:
                self.resolution = ROOT.RooRealVar(  # type: ignore
                    "sigma_resolution",
                    "#sigma_{resolution} [MeV/c^{2}]",
                    float(fixed_res),
                )
                self.resolution.setConstant(True)
            else:
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

        # Initialize storage if needed
        if not hasattr(self, "argus_params"):
            self.argus_params = {}

        bkg_model = self.systematic_params.get("bkg_model", "argus")

        if bkg_model == "poly2":
            # 2nd-order Chebyshev polynomial background (systematic alternative)
            c0 = ROOT.RooRealVar(f"c0_poly_{year}", f"poly c0 {year}", 0.0, -10.0, 10.0)  # type: ignore
            c1 = ROOT.RooRealVar(f"c1_poly_{year}", f"poly c1 {year}", 0.0, -10.0, 10.0)  # type: ignore
            coef_list = ROOT.RooArgList(c0, c1)  # type: ignore
            bkg_pdf = ROOT.RooChebychev(  # type: ignore
                f"pdf_bkg_{year}", f"Background PDF {year}", mass_var, coef_list
            )
            # Store a dummy "c" for interface compatibility; consumers only use the PDF
            self.bkg_pdfs[year] = bkg_pdf
            self.argus_params[year] = {"c": c0, "c0": c0, "c1": c1}
            return bkg_pdf, c0

        # Default: ARGUS background
        # Endpoint — set BEYOND the fit range to avoid a sharp cutoff.
        # Allow systematic_params to override the endpoint offset (±50 MeV variations).
        endpoint_offset = self.systematic_params.get(
            "argus_endpoint_offset",
            self.config.fitting.get("argus_endpoint_offset", 200.0),
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

        States: ηc(1S), J/ψ, χc0, χc1, ηc(2S)

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

    @staticmethod
    def _iter_roorealvars(argset: Any):
        """Yield RooRealVar-like objects from a RooArgSet."""
        for obj in argset:
            if obj.InheritsFrom("RooRealVar"):
                yield obj

    def _snapshot_parameters(self, argset: Any) -> dict[str, dict[str, Any]]:
        """Capture current parameter values and const-ness for later restoration."""
        snapshot = {}
        for var in self._iter_roorealvars(argset):
            snapshot[var.GetName()] = {
                "var": var,
                "value": float(var.getVal()),
                "constant": bool(var.isConstant()),
            }
        return snapshot

    @staticmethod
    def _restore_parameters(snapshot: dict[str, dict[str, Any]]) -> None:
        """Restore a previously captured parameter snapshot."""
        for entry in snapshot.values():
            entry["var"].setVal(entry["value"])
            entry["var"].setConstant(entry["constant"])

    def _compute_profile_significances(
        self,
        dataset_name: str,
        dataset: Any,
        model: Any,
        yields: dict[str, Any],
        fit_result: Any,
        states: tuple[str, ...],
    ) -> dict[str, dict[str, float | int]]:
        """
        Compute one-sided profile-likelihood significances for selected states.

        For each target state X, the null hypothesis is N_X = 0 with all other
        nuisance parameters profiled. The test statistic is
        q0 = 2 * (NLL_null - NLL_best), and Z = sqrt(q0).
        """
        best_nll = float(fit_result.minNll())
        best_status = int(fit_result.status())
        params = model.getParameters(dataset)
        snapshot = self._snapshot_parameters(params)
        results: dict[str, dict[str, float | int]] = {}

        for state in states:
            yield_var = yields.get(state)
            if yield_var is None:
                continue

            # Start each state from the nominal best-fit point.
            self._restore_parameters(snapshot)
            best_fit_yield = float(yield_var.getVal())
            best_fit_error = float(yield_var.getError())

            yield_var.setVal(0.0)
            yield_var.setConstant(True)

            null_result = model.fitTo(
                dataset,
                ROOT.RooFit.Save(),  # type: ignore
                ROOT.RooFit.Extended(True),  # type: ignore
                ROOT.RooFit.PrintLevel(-1),  # type: ignore
                ROOT.RooFit.NumCPU(4),  # type: ignore
                ROOT.RooFit.Strategy(2),  # type: ignore
            )

            null_nll = float(null_result.minNll())
            null_status = int(null_result.status())
            q0 = max(0.0, 2.0 * (null_nll - best_nll))
            z_value = math.sqrt(q0)

            results[state] = {
                "best_fit_yield": best_fit_yield,
                "best_fit_error": best_fit_error,
                "best_nll": best_nll,
                "best_status": best_status,
                "null_nll": null_nll,
                "null_status": null_status,
                "q0": q0,
                "z": z_value,
            }

            if null_status != 0:
                print(
                    f"  WARNING: Null-hypothesis fit for {state} in {dataset_name} "
                    f"returned status={null_status}"
                )

        # Leave the caller with the nominal best-fit parameters restored.
        self._restore_parameters(snapshot)
        return results

    def perform_fit(
        self,
        data_by_year: dict[str, ak.Array],
        fit_combined: bool = True,
        plot_dir: "Path | None" = None,
        fit_label: str = "",
        profile_significance_states: tuple[str, ...] | None = None,
        profile_significance_datasets: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        Perform mass fits to all years (per-year AND combined).

        Current workflow behavior:
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
        all_profile_significances = {}

        print("\n" + "=" * 80)
        print("MASS FITTING WITH ROOFIT - ALL 5 CHARMONIUM STATES")
        print("=" * 80)
        print(f"Charmonium fit range: {self.fit_range[0]} - {self.fit_range[1]} MeV")
        print(f"B+ mass window: {self.bu_mass_min} - {self.bu_mass_max} MeV")
        print("Using M_LpKm_h2 (M(Λ̄pK⁻), h2=K⁻ correct for charmonium)")
        print("Modeling: ηc(1S), J/ψ, χc0, χc1, ηc(2S) simultaneously")
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

                should_compute_profile = bool(profile_significance_states) and (
                    profile_significance_datasets is None
                    or dataset_name in profile_significance_datasets
                )
                if should_compute_profile:
                    all_profile_significances[dataset_name] = self._compute_profile_significances(
                        dataset_name=dataset_name,
                        dataset=dataset,
                        model=model,
                        yields=yields,
                        fit_result=fit_result,
                        states=tuple(profile_significance_states or ()),
                    )

                # Plot results only when a target directory is explicitly provided
                if plot_dir is not None:
                    self.plot_fit_result(
                        dataset_name,
                        mass_var,
                        dataset,
                        model,
                        yields,
                        fit_result,
                        plot_dir=plot_dir,
                        fit_label=fit_label,
                    )

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
            "profile_significances": all_profile_significances,
        }

    def plot_fit_result(
        self,
        year: str,
        mass_var: Any,
        dataset: Any,
        model: Any,
        yields: dict[str, Any],
        fit_result: Any = None,
        plot_dir: "Path | None" = None,
        fit_label: str = "",
    ) -> None:
        """
        Plot fit result using matplotlib + LHCb2 style (via plot_utils).

        The RooFit objects are used only to extract data arrays and PDF
        evaluations; all rendering is done with matplotlib.
        """
        import numpy as np

        try:
            from modules.plot_utils import (
                STATE_COLORS,
                STATE_LABELS,
                make_mass_fit_figure,
                save_figure,
                setup_style,
            )
        except ImportError:
            from plot_utils import (  # type: ignore[no-redef]
                STATE_COLORS,
                STATE_LABELS,
                make_mass_fit_figure,
                save_figure,
                setup_style,
            )

        setup_style()

        fitting_cfg = getattr(self.config, "fitting", {})
        plotting_cfg = fitting_cfg.get("plotting", {})
        plot_states = plotting_cfg.get("states", ["jpsi", "etac", "chic0", "chic1", "etac_2s"])

        # ── Extract histogram data from ROOT dataset ───────────────────────────
        h1 = dataset.createHistogram(f"h_data_plt_{year}", mass_var)
        nbins = h1.GetNbinsX()
        bin_centers = np.array([h1.GetBinCenter(i + 1) for i in range(nbins)])
        bin_contents = np.array([h1.GetBinContent(i + 1) for i in range(nbins)])
        bin_errors = np.array([h1.GetBinError(i + 1) for i in range(nbins)])
        h1.Delete()

        # ── Evaluate PDFs at fine grid ─────────────────────────────────────────
        mass_points = np.linspace(self.fit_range[0], self.fit_range[1], 500)
        argset = ROOT.RooArgSet(mass_var)  # type: ignore
        total_events = sum(v.getVal() for v in yields.values())

        def _eval_pdf(pdf: Any, n_events: float) -> np.ndarray:
            curve = np.empty(len(mass_points))
            for k, m in enumerate(mass_points):
                mass_var.setVal(m)
                curve[k] = pdf.getVal(argset) * n_events * self.bin_width
            return curve

        total_curve = _eval_pdf(model, total_events)

        bkg_yield = yields["background"].getVal()
        bkg_pdf = self.bkg_pdfs.get(year)
        background_curve = (
            _eval_pdf(bkg_pdf, bkg_yield) if bkg_pdf is not None else np.zeros(len(mass_points))
        )

        signal_curves = []
        for state in plot_states:
            if state not in self.signal_pdfs or state not in yields:
                continue
            comp_yield = yields[state].getVal()
            if comp_yield <= 0:
                continue
            comp_curve = _eval_pdf(self.signal_pdfs[state], comp_yield)
            signal_curves.append(
                {
                    "x": mass_points,
                    "y": comp_curve,
                    "label": STATE_LABELS.get(state, state),
                    "color": STATE_COLORS.get(state, "crimson"),
                }
            )

        # ── Pull distribution ──────────────────────────────────────────────────
        pulls = np.zeros(len(bin_centers))
        for i, m in enumerate(bin_centers):
            if bin_errors[i] > 0:
                mass_var.setVal(m)
                expected = model.getVal(argset) * total_events * self.bin_width
                pulls[i] = (bin_contents[i] - expected) / bin_errors[i]

        # ── Info text ──────────────────────────────────────────────────────────
        n_bkg = yields["background"].getVal()
        n_bkg_err = yields["background"].getError()
        n_signal = sum(yields[s].getVal() for s in plot_states if s in yields)
        sigma_res = self.resolution.getVal()  # type: ignore
        sigma_res_err = self.resolution.getError()  # type: ignore

        # χ²/ndof from pulls (only populated bins)
        valid_mask = bin_errors > 0
        chi2 = float(np.sum(pulls[valid_mask] ** 2))
        n_params = fit_result.floatParsFinal().getSize() if fit_result is not None else 0
        ndof = max(1, int(valid_mask.sum()) - n_params)

        yield_lines = [
            f"{STATE_LABELS.get(s, s)} : {yields[s].getVal():.0f} ± {yields[s].getError():.0f}"
            for s in plot_states
            if s in yields
        ]
        stat_lines = [
            f"N_bkg = {n_bkg:.0f} ± {n_bkg_err:.0f}",
            f"N_sig = {n_signal:.0f}",
            f"σ_res = {sigma_res:.1f} ± {sigma_res_err:.1f} MeV",
            f"χ²/ndof = {chi2:.1f}/{ndof} = {chi2/ndof:.2f}",
        ]
        info_lines = yield_lines + ["─" * 26] + stat_lines

        year_str = "2016–2018" if year == "combined" else str(year)

        # ── Build and save figure ──────────────────────────────────────────────
        fig, _ = make_mass_fit_figure(
            bin_centers=bin_centers,
            bin_contents=bin_contents,
            bin_errors=bin_errors,
            mass_points=mass_points,
            total_curve=total_curve,
            signal_curves=signal_curves,
            background_curve=background_curve,
            pulls=pulls,
            fit_range=self.fit_range,
            bin_width=self.bin_width,
            year=year_str,
            context_label=fit_label,
            info_lines=info_lines,
        )

        if plot_dir is None:
            import matplotlib.pyplot as plt

            plt.close(fig)
            return

        plot_dir = Path(plot_dir)
        output_file = plot_dir / f"mass_fit_{year}.pdf"
        save_figure(fig, output_file)
        print(f"  ✓ Saved fit plot: {output_file}")
