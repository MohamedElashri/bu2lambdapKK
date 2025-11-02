import zfit
from zfit import z
import tensorflow as tf

class InvariantMassFitter:
    """
    Fit M(Λ̄pK⁻) invariant mass distribution
    Simultaneous fit to all charmonium states
    
    Strategy (following reference analysis Section 5):
    - Fit each YEAR separately (combining MagDown + MagUp)
    - Constrain masses/widths to be same across years
    - Extract yields per year for efficiency correction
    """
    
    def __init__(self, config: TOMLConfig):
        self.config = config
        self.fit_range = config.particles["mass_windows"]["charmonium_fit_range"]
        
        # Model parameters (shared across years)
        self.masses = {}      # M_J/ψ, M_ηc, M_χc0, M_χc1
        self.widths = {}      # Γ_J/ψ (fixed), Γ_ηc, Γ_χc0, Γ_χc1 (fixed)
        self.resolution = {}  # σ_ηc (floating), ratios fixed from MC
        
    def build_signal_model(self, state: str, mass_obs: zfit.Space) -> zfit.pdf.BasePDF:
        """
        Build signal PDF for one charmonium state
        
        Model: RBW ⊗ Gaussian (simplified, no double Gaussian for draft)
        
        RBW with Blatt-Weisskopf form factors:
        RBW(m) = m × Γ_f / [(m² - M²)² + M²Γ_f²]
        
        where Γ_f includes:
        - Phase space factor (momentum K)
        - Blatt-Weisskopf form factor F(L)
        - Threshold effects
        
        Orbital angular momenta (from quantum numbers):
        - J/ψ (1⁻⁻): Need to verify if allowed! Assuming L=1 for now
        - ηc(1S) (0⁻⁺): L = 1
        - χc0 (0⁺⁺): L = 0
        - χc1 (1⁺⁺): L = 2
        
        NOTE: For draft, use simplified RBW. Full form factors later.
        """
        state_lower = state.lower()
        
        # Mass parameter (shared across years, but fit separately per state)
        if state_lower not in self.masses:
            pdg_mass = self.config.particles["pdg_masses"][state_lower.replace("chic", "chic")]
            self.masses[state_lower] = zfit.Parameter(
                f"M_{state}",
                pdg_mass,
                pdg_mass - 50,  # Allow ±50 MeV variation
                pdg_mass + 50
            )
        
        # Width parameter
        if state_lower not in self.widths:
            pdg_width = self.config.particles["pdg_widths"][state_lower.replace("chic", "chic")]
            
            # Fix narrow states (J/ψ, χc1) to PDG values
            if state_lower in ["jpsi", "chic1"]:
                self.widths[state_lower] = zfit.Parameter(
                    f"Gamma_{state}",
                    pdg_width,
                    floating=False  # Fixed
                )
            else:
                # Float for ηc and χc0 (broader states)
                self.widths[state_lower] = zfit.Parameter(
                    f"Gamma_{state}",
                    pdg_width,
                    0.1,  # Minimum 0.1 MeV
                    100.0  # Maximum 100 MeV
                )
        
        # Resolution (Gaussian width)
        # For draft: single Gaussian per state
        # Full analysis: double Gaussian with ratios fixed from MC
        if "sigma_etac" not in self.resolution:
            self.resolution["sigma_etac"] = zfit.Parameter(
                "sigma_etac",
                5.0,  # Initial guess ~5 MeV
                1.0,
                15.0
            )
        
        # Resolution ratios (fixed from MC - TO BE DETERMINED)
        # For now, assume all states have similar resolution
        sigma = self.resolution["sigma_etac"]
        
        # Build Relativistic Breit-Wigner
        # For draft: use zfit's built-in Voigt (RBW ⊗ Gaussian)
        signal_pdf = zfit.pdf.Voigt(
            m=self.masses[state_lower],
            gamma=self.widths[state_lower],
            sigma=sigma,
            obs=mass_obs
        )
        
        return signal_pdf
    
    def build_background_model(self, mass_obs: zfit.Space) -> zfit.pdf.BasePDF:
        """
        Build combinatorial background model
        
        Simplified for draft: Single exponential
        BGR(m) = exp(-α × m)
        
        Full analysis would include:
        BGR(m) = sqrt(m - m_thresh) × exp(-α×m) × (1 + A×m)
        """
        alpha = zfit.Parameter("alpha_bkg", -0.001, -0.01, 0.0)
        
        # Simple exponential
        bkg_pdf = zfit.pdf.Exponential(alpha, obs=mass_obs)
        
        return bkg_pdf
    
    def build_full_model(self, 
                        year: str,
                        mass_obs: zfit.Space) -> Tuple[zfit.pdf.BasePDF, Dict]:
        """
        Build full PDF for one year
        
        PDF = Σ[N_state × Signal_state] + N_bkg × Background
        
        States: J/ψ, ηc, χc0, χc1
        
        Returns:
            (full_pdf, yield_parameters_dict)
        """
        # Yield parameters (separate per year)
        yields = {}
        pdfs = []
        
        for state in ["jpsi", "etac", "chic0", "chic1"]:
            # Signal PDF (shared model, but separate yields per year)
            sig_pdf = self.build_signal_model(state, mass_obs)
            
            # Yield parameter
            yield_param = zfit.Parameter(
                f"N_{state}_{year}",
                1000,  # Initial guess
                0,
                1e6
            )
            yields[state] = yield_param
            
            # Extended PDF
            ext_pdf = sig_pdf.create_extended(yield_param)
            pdfs.append(ext_pdf)
        
        # Background
        bkg_pdf = self.build_background_model(mass_obs)
        bkg_yield = zfit.Parameter(f"N_bkg_{year}", 10000, 0, 1e7)
        yields["background"] = bkg_yield
        
        ext_bkg = bkg_pdf.create_extended(bkg_yield)
        pdfs.append(ext_bkg)
        
        # Sum all components
        full_pdf = zfit.pdf.SumPDF(pdfs)
        
        return full_pdf, yields
    
    def perform_fit(self, 
                   data_by_year: Dict[str, ak.Array]) -> Dict[str, Dict]:
        """
        Perform binned χ² fit to M(Λ̄pK⁻) distribution
        
        Strategy:
        1. Fit each year separately
        2. Mass/width parameters shared across years
        3. Yields separate per year
        4. Use binned data (5 MeV bins)
        
        Args:
            data_by_year: {"2016": events, "2017": events, "2018": events}
            
        Returns:
            {
                "yields": {year: {state: (value, error)}},
                "masses": {state: (value, error)},
                "widths": {state: (value, error)},
                "fit_results": {year: FitResult object}
            }
        """
        # Define observable
        mass_obs = zfit.Space("M_LpKm", self.fit_range)
        
        all_yields = {}
        all_fits = {}
        
        print("\n" + "="*80)
        print("MASS FITTING")
        print("="*80)
        
        for year in sorted(data_by_year.keys()):
            print(f"\n[Year {year}]")
            
            data_array = data_by_year[year]["M_LpKm"]
            
            # Create binned data (5 MeV bins following reference analysis)
            n_bins = int((self.fit_range[1] - self.fit_range[0]) / 5.0)
            
            hist_data, bin_edges = np.histogram(
                ak.to_numpy(data_array),
                bins=n_bins,
                range=self.fit_range
            )
            
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            bin_widths = bin_edges[1:] - bin_edges[:-1]
            
            # Create zfit data
            data_zfit = zfit.Data.from_numpy(
                obs=mass_obs,
                array=bin_centers,
                weights=hist_data
            )
            
            # Build model for this year
            pdf, yields = self.build_full_model(year, mass_obs)
            
            # Create loss (chi-squared for binned data)
            # For draft, use unbinned NLL (easier with zfit)
            # Full analysis: proper binned chi-squared
            loss = zfit.loss.ExtendedUnbinnedNLL(pdf, data_zfit)
            
            # Minimize
            minimizer = zfit.minimize.Minuit()
            result = minimizer.minimize(loss)
            
            print(f"  Fit converged: {result.converged}")
            print(f"  FCN minimum: {result.fmin:.2f}")
            
            # Extract yields with errors
            year_yields = {}
            for state, param in yields.items():
                value = param.value().numpy()
                error = result.hesse()[param]["error"]
                year_yields[state] = (value, error)
                
                print(f"    N_{state} = {value:.0f} ± {error:.0f}")
            
            all_yields[year] = year_yields
            all_fits[year] = result
            
            # Plot fit result
            self._plot_fit_result(year, data_array, pdf, yields, bin_centers, hist_data)
        
        # Extract masses and widths (shared across years)
        masses_result = {}
        widths_result = {}
        
        # Use last fit result for parameter extraction
        last_result = all_fits[list(all_fits.keys())[-1]]
        
        for state in ["jpsi", "etac", "chic0", "chic1"]:
            mass_param = self.masses[state]
            width_param = self.widths[state]
            
            masses_result[state] = (
                mass_param.value().numpy(),
                last_result.hesse()[mass_param]["error"]
            )
            
            widths_result[state] = (
                width_param.value().numpy(),
                last_result.hesse()[width_param]["error"]
            )
        
        print("\n" + "="*80)
        print("FITTED PARAMETERS (shared across years)")
        print("="*80)
        
        for state in ["jpsi", "etac", "chic0", "chic1"]:
            m_val, m_err = masses_result[state]
            g_val, g_err = widths_result[state]
            print(f"{state:>10}: M = {m_val:.2f} ± {m_err:.2f} MeV")
            print(f"{'':>10}  Γ = {g_val:.2f} ± {g_err:.2f} MeV")
        
        return {
            "yields": all_yields,
            "masses": masses_result,
            "widths": widths_result,
            "fit_results": all_fits
        }
    
    def _plot_fit_result(self,
                        year: str,
                        data_array: ak.Array,
                        pdf: zfit.pdf.BasePDF,
                        yields: Dict,
                        bin_centers: np.ndarray,
                        hist_data: np.ndarray):
        """
        Plot fit result similar to Figure 8 in reference analysis
        
        Shows:
        - Data points with error bars
        - Total fit
        - Individual signal components
        - Background component
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                        gridspec_kw={'height_ratios': [3, 1]},
                                        sharex=True)
        
        # Top: Data + fit
        x_plot = np.linspace(self.fit_range[0], self.fit_range[1], 500)
        
        # Data
        ax1.errorbar(bin_centers, hist_data, yerr=np.sqrt(hist_data),
                     fmt='ko', label='Data', markersize=3)
        
        # Total PDF (scale to data)
        total_yield = sum(y.value().numpy() for y in yields.values())
        bin_width = bin_centers[1] - bin_centers[0]
        
        pdf_vals = pdf.pdf(x_plot).numpy() * total_yield * bin_width
        ax1.plot(x_plot, pdf_vals, 'b-', linewidth=2, label='Total fit')
        
        # Individual components (if accessible - zfit makes this tricky)
        # For draft, just show total + background
        
        ax1.set_ylabel('Events / 5 MeV', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.set_title(f'M(Λ̄pK⁻) fit - Year {year}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Bottom: Pulls
        # Pull = (data - fit) / error
        pdf_at_bins = pdf.pdf(bin_centers).numpy() * total_yield * bin_width
        pulls = (hist_data - pdf_at_bins) / np.sqrt(hist_data + 1e-10)
        
        ax2.axhline(0, color='b', linestyle='--', linewidth=1)
        ax2.axhline(3, color='r', linestyle=':', linewidth=1, alpha=0.5)
        ax2.axhline(-3, color='r', linestyle=':', linewidth=1, alpha=0.5)
        ax2.plot(bin_centers, pulls, 'ko', markersize=3)
        
        ax2.set_xlabel('M(Λ̄pK⁻) [MeV/c²]', fontsize=12)
        ax2.set_ylabel('Pull [σ]', fontsize=12)
        ax2.set_ylim(-5, 5)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        plot_dir = Path(self.config.paths["output"]["plots_dir"]) / "fits"
        plot_dir.mkdir(exist_ok=True, parents=True)
        
        plt.savefig(plot_dir / f"mass_fit_{year}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved fit plot: {plot_dir / f'mass_fit_{year}.png'}")