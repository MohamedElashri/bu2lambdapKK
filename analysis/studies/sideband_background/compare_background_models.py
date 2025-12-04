#!/usr/bin/env python3
"""
Compare Background Models: ARGUS vs Sideband Template

This script performs M(Lambda p K-) fits using both:
1. ARGUS parametric background (standard approach)
2. Sideband-derived template background (data-driven)

Both fits use the SAME data (B+ signal region) and SAME signal model
(Voigtian for each charmonium state), differing only in background treatment.

The comparison provides:
- Yield differences between methods
- Systematic uncertainty estimate from background modeling

Usage:
------
    cd analysis/studies/sideband_background
    python compare_background_models.py
    python compare_background_models.py --years 2016,2017,2018

Output:
-------
    output/background_comparison.pdf
    output/background_comparison_results.txt
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import awkward as ak
import ROOT

# Add parent directories to path
STUDY_DIR: Path = Path(__file__).parent
ANALYSIS_DIR: Path = STUDY_DIR.parent.parent
sys.path.insert(0, str(STUDY_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

from config import MASS_CONFIG
from data_loader import get_config, setup_root_style
from modules.data_handler import DataManager


class BackgroundModelComparison:
    """
    Compare ARGUS vs Template background models for M(LpK-) fitting.

    Uses the same signal model (Voigtian) for both, only background differs.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize comparison with configuration.

        Args:
            config: TOMLConfig object
        """
        self.config: Any = config
        self.fit_range: tuple[float, float] = (MASS_CONFIG.MLPK_MIN, MASS_CONFIG.MLPK_MAX)
        self.bin_width: float = MASS_CONFIG.BIN_WIDTH
        self.nbins: int = MASS_CONFIG.N_BINS_MLPK
        # Storage for ROOT objects
        self.mass_var: ROOT.RooRealVar | None = None
        self.signal_pdfs: dict[str, ROOT.RooAbsPdf] = {}
        self.masses: dict[str, ROOT.RooRealVar] = {}
        self.widths: dict[str, ROOT.RooRealVar] = {}
        self.resolution: ROOT.RooRealVar | None = None
        self._keep_alive: list[Any] = []

    def setup_observable(self) -> ROOT.RooRealVar:
        """Create mass observable."""
        if self.mass_var is None:
            self.mass_var = ROOT.RooRealVar(
                "M_LpKm",
                "#it{M}(#Lambda#it{p}#it{K}^{#minus}) [MeV/#it{c}^{2}]",
                self.fit_range[0],
                self.fit_range[1],
            )
            self.mass_var.setBins(self.nbins)
            ROOT.SetOwnership(self.mass_var, False)
        return self.mass_var

    def create_signal_pdf(
        self,
        state: str,
        mass_var: ROOT.RooRealVar,
    ) -> ROOT.RooAbsPdf:
        """
        Create Voigtian signal PDF for a charmonium state.

        Args:
            state: State name (jpsi, etac, chic0, chic1, etac_2s)
            mass_var: Mass observable

        Returns:
            RooVoigtian PDF
        """
        if state in self.signal_pdfs:
            return self.signal_pdfs[state]
        # Config key mapping
        config_key_map: dict[str, str] = {
            "jpsi": "jpsi",
            "etac": "etac_1s",
            "chic0": "chic0",
            "chic1": "chic1",
            "etac_2s": "etac_2s",
        }
        config_key: str = config_key_map.get(state, state)
        # Get PDG values
        pdg_mass: float = self.config.particles["pdg_masses"].get(config_key, 3000.0)
        pdg_width: float = self.config.particles["pdg_widths"].get(config_key, 10.0)
        # Mass parameter (fixed to PDG)
        mean: ROOT.RooRealVar = ROOT.RooRealVar(
            f"mean_{state}",
            f"Mean {state}",
            pdg_mass,
        )
        mean.setConstant(True)
        ROOT.SetOwnership(mean, False)
        self.masses[state] = mean
        # Width parameter (fixed to PDG)
        width: ROOT.RooRealVar = ROOT.RooRealVar(
            f"width_{state}",
            f"Width {state}",
            pdg_width,
        )
        width.setConstant(True)
        ROOT.SetOwnership(width, False)
        self.widths[state] = width
        # Resolution (shared, floating)
        if self.resolution is None:
            self.resolution = ROOT.RooRealVar(
                "sigma_resolution",
                "Detector resolution",
                8.0,
                1.0,
                30.0,
            )
            ROOT.SetOwnership(self.resolution, False)
        # Voigtian PDF
        pdf: ROOT.RooVoigtian = ROOT.RooVoigtian(
            f"sig_{state}",
            f"Signal {state}",
            mass_var,
            mean,
            width,
            self.resolution,
        )
        ROOT.SetOwnership(pdf, False)
        self.signal_pdfs[state] = pdf
        self._keep_alive.extend([mean, width, pdf])
        return pdf

    def create_argus_background(
        self,
        mass_var: ROOT.RooRealVar,
    ) -> tuple[ROOT.RooArgusBG, ROOT.RooRealVar]:
        """
        Create ARGUS background PDF.

        Args:
            mass_var: Mass observable

        Returns:
            Tuple of (ARGUS PDF, slope parameter)
        """
        # Endpoint (fixed, beyond fit range)
        m0: ROOT.RooRealVar = ROOT.RooRealVar(
            "m0_argus",
            "ARGUS endpoint",
            self.fit_range[1] + 200.0,
        )
        m0.setConstant(True)
        ROOT.SetOwnership(m0, False)
        # Slope parameter (floating)
        c: ROOT.RooRealVar = ROOT.RooRealVar(
            "c_argus",
            "ARGUS slope",
            -20.0,
            -100.0,
            -0.1,
        )
        ROOT.SetOwnership(c, False)
        # Power (fixed)
        p: ROOT.RooRealVar = ROOT.RooRealVar("p_argus", "ARGUS power", 0.5)
        p.setConstant(True)
        ROOT.SetOwnership(p, False)
        # ARGUS PDF
        argus: ROOT.RooArgusBG = ROOT.RooArgusBG(
            "bkg_argus",
            "ARGUS background",
            mass_var,
            m0,
            c,
            p,
        )
        ROOT.SetOwnership(argus, False)
        self._keep_alive.extend([m0, c, p, argus])
        return argus, c

    def load_template_background(
        self,
        mass_var: ROOT.RooRealVar,
        template_file: Path,
    ) -> ROOT.RooHistPdf:
        """
        Load sideband template background PDF.

        Args:
            mass_var: Mass observable
            template_file: Path to template ROOT file

        Returns:
            RooHistPdf for background
        """
        tfile: ROOT.TFile = ROOT.TFile.Open(str(template_file), "READ")
        if not tfile or tfile.IsZombie():
            raise FileNotFoundError(f"Cannot open template file: {template_file}")
        # Get histogram
        hist: ROOT.TH1D = tfile.Get("background_template_raw")
        if not hist:
            tfile.Close()
            raise ValueError(f"Template histogram not found in {template_file}")
        # Clone and detach from file
        hist_clone: ROOT.TH1D = hist.Clone("bkg_template_hist")
        hist_clone.SetDirectory(0)
        ROOT.SetOwnership(hist_clone, False)
        tfile.Close()
        # Normalize
        integral: float = hist_clone.Integral(1, hist_clone.GetNbinsX())
        if integral > 0:
            hist_clone.Scale(1.0 / integral)
        # Create RooDataHist
        data_hist: ROOT.RooDataHist = ROOT.RooDataHist(
            "bkg_template_datahist",
            "Background template",
            ROOT.RooArgList(mass_var),
            hist_clone,
        )
        ROOT.SetOwnership(data_hist, False)
        # Create RooHistPdf
        pdf: ROOT.RooHistPdf = ROOT.RooHistPdf(
            "bkg_template",
            "Template background",
            ROOT.RooArgSet(mass_var),
            data_hist,
            2,
        )
        ROOT.SetOwnership(pdf, False)
        self._keep_alive.extend([hist_clone, data_hist, pdf])
        return pdf

    def build_model(
        self,
        mass_var: ROOT.RooRealVar,
        bkg_pdf: ROOT.RooAbsPdf,
        model_name: str,
    ) -> tuple[ROOT.RooAddPdf, dict[str, ROOT.RooRealVar]]:
        """
        Build full model with signal + background.

        Args:
            mass_var: Mass observable
            bkg_pdf: Background PDF (ARGUS or template)
            model_name: Name prefix for the model

        Returns:
            Tuple of (total PDF, yields dictionary)
        """
        pdf_list: ROOT.RooArgList = ROOT.RooArgList()
        coef_list: ROOT.RooArgList = ROOT.RooArgList()
        yields: dict[str, ROOT.RooRealVar] = {}
        # Signal components
        states: list[str] = ["jpsi", "etac", "chic0", "chic1", "etac_2s"]
        for state in states:
            sig_pdf = self.create_signal_pdf(state, mass_var)
            yield_var: ROOT.RooRealVar = ROOT.RooRealVar(
                f"N_{state}_{model_name}",
                f"Yield {state}",
                1000,
                0,
                1e6,
            )
            ROOT.SetOwnership(yield_var, False)
            yields[state] = yield_var
            pdf_list.add(sig_pdf)
            coef_list.add(yield_var)
            self._keep_alive.append(yield_var)
        # Background
        bkg_yield: ROOT.RooRealVar = ROOT.RooRealVar(
            f"N_bkg_{model_name}",
            "Background yield",
            10000,
            0,
            1e7,
        )
        ROOT.SetOwnership(bkg_yield, False)
        yields["background"] = bkg_yield
        pdf_list.add(bkg_pdf)
        coef_list.add(bkg_yield)
        self._keep_alive.append(bkg_yield)
        # Total model
        total_pdf: ROOT.RooAddPdf = ROOT.RooAddPdf(
            f"model_{model_name}",
            f"Total PDF ({model_name})",
            pdf_list,
            coef_list,
        )
        ROOT.SetOwnership(total_pdf, False)
        self._keep_alive.append(total_pdf)
        return total_pdf, yields

    def perform_fit(
        self,
        model: ROOT.RooAddPdf,
        data: ROOT.RooDataHist,
    ) -> ROOT.RooFitResult:
        """
        Perform extended maximum likelihood fit.

        Args:
            model: Total PDF
            data: Data histogram

        Returns:
            RooFitResult
        """
        result: ROOT.RooFitResult = model.fitTo(
            data,
            ROOT.RooFit.Save(True),
            ROOT.RooFit.PrintLevel(-1),
            ROOT.RooFit.Strategy(2),
            ROOT.RooFit.Extended(True),
        )
        ROOT.SetOwnership(result, False)
        return result

    def create_comparison_plot(
        self,
        mass_var: ROOT.RooRealVar,
        data: ROOT.RooDataHist,
        model_argus: ROOT.RooAddPdf,
        model_template: ROOT.RooAddPdf,
        yields_argus: dict[str, ROOT.RooRealVar],
        yields_template: dict[str, ROOT.RooRealVar],
        output_path: Path,
    ) -> None:
        """
        Create side-by-side comparison plot.

        Args:
            mass_var: Mass observable
            data: Data histogram
            model_argus: ARGUS model
            model_template: Template model
            yields_argus: ARGUS yields
            yields_template: Template yields
            output_path: Output PDF path
        """
        canvas: ROOT.TCanvas = ROOT.TCanvas("c_compare", "", 2000, 900)
        ROOT.SetOwnership(canvas, False)
        self._keep_alive.append(canvas)
        # Two pads
        pad1: ROOT.TPad = ROOT.TPad("pad1", "ARGUS", 0.0, 0.0, 0.5, 1.0)
        pad2: ROOT.TPad = ROOT.TPad("pad2", "Template", 0.5, 0.0, 1.0, 1.0)
        ROOT.SetOwnership(pad1, False)
        ROOT.SetOwnership(pad2, False)
        self._keep_alive.extend([pad1, pad2])
        for pad in [pad1, pad2]:
            pad.SetLeftMargin(0.14)
            pad.SetRightMargin(0.05)
            pad.SetTopMargin(0.08)
            pad.SetBottomMargin(0.12)
        canvas.cd()
        pad1.Draw()
        pad2.Draw()
        # ARGUS fit plot
        pad1.cd()
        frame1: ROOT.RooPlot = mass_var.frame(ROOT.RooFit.Title("ARGUS Background"))
        ROOT.SetOwnership(frame1, False)
        data.plotOn(frame1, ROOT.RooFit.Name("data"))
        model_argus.plotOn(frame1, ROOT.RooFit.Name("total"), ROOT.RooFit.LineColor(ROOT.kBlue))
        model_argus.plotOn(
            frame1,
            ROOT.RooFit.Components("bkg_argus"),
            ROOT.RooFit.Name("bkg"),
            ROOT.RooFit.LineColor(ROOT.kRed),
            ROOT.RooFit.LineStyle(ROOT.kDashed),
        )
        frame1.Draw()
        self._keep_alive.append(frame1)
        # ARGUS legend and info
        latex1: ROOT.TLatex = ROOT.TLatex()
        latex1.SetNDC()
        latex1.SetTextFont(132)
        latex1.SetTextSize(0.040)
        latex1.DrawLatex(0.18, 0.85, "#font[62]{LHCb}")
        latex1.SetTextSize(0.030)
        latex1.DrawLatex(0.18, 0.80, "ARGUS background")
        y_pos: float = 0.75
        for state in ["jpsi", "etac", "chic0", "chic1", "etac_2s"]:
            n_val: float = yields_argus[state].getVal()
            n_err: float = yields_argus[state].getError()
            latex1.DrawLatex(0.18, y_pos, f"N({state}) = {n_val:.0f} #pm {n_err:.0f}")
            y_pos -= 0.045
        n_bkg: float = yields_argus["background"].getVal()
        n_bkg_err: float = yields_argus["background"].getError()
        latex1.DrawLatex(0.18, y_pos, f"N(bkg) = {n_bkg:.0f} #pm {n_bkg_err:.0f}")
        self._keep_alive.append(latex1)
        # Template fit plot
        pad2.cd()
        frame2: ROOT.RooPlot = mass_var.frame(ROOT.RooFit.Title("Template Background"))
        ROOT.SetOwnership(frame2, False)
        data.plotOn(frame2, ROOT.RooFit.Name("data"))
        model_template.plotOn(frame2, ROOT.RooFit.Name("total"), ROOT.RooFit.LineColor(ROOT.kBlue))
        model_template.plotOn(
            frame2,
            ROOT.RooFit.Components("bkg_template"),
            ROOT.RooFit.Name("bkg"),
            ROOT.RooFit.LineColor(ROOT.kRed),
            ROOT.RooFit.LineStyle(ROOT.kDashed),
        )
        frame2.Draw()
        self._keep_alive.append(frame2)
        # Template legend and info
        latex2: ROOT.TLatex = ROOT.TLatex()
        latex2.SetNDC()
        latex2.SetTextFont(132)
        latex2.SetTextSize(0.040)
        latex2.DrawLatex(0.18, 0.85, "#font[62]{LHCb}")
        latex2.SetTextSize(0.030)
        latex2.DrawLatex(0.18, 0.80, "Sideband template background")
        y_pos = 0.75
        for state in ["jpsi", "etac", "chic0", "chic1", "etac_2s"]:
            n_val = yields_template[state].getVal()
            n_err = yields_template[state].getError()
            latex2.DrawLatex(0.18, y_pos, f"N({state}) = {n_val:.0f} #pm {n_err:.0f}")
            y_pos -= 0.045
        n_bkg = yields_template["background"].getVal()
        n_bkg_err = yields_template["background"].getError()
        latex2.DrawLatex(0.18, y_pos, f"N(bkg) = {n_bkg:.0f} #pm {n_bkg_err:.0f}")
        self._keep_alive.append(latex2)
        canvas.Update()
        canvas.SaveAs(str(output_path))
        print(f"Saved: {output_path}")


def save_comparison_results(
    yields_argus: dict[str, ROOT.RooRealVar],
    yields_template: dict[str, ROOT.RooRealVar],
    result_argus: ROOT.RooFitResult,
    result_template: ROOT.RooFitResult,
    output_path: Path,
) -> None:
    """
    Save comparison results to text file.

    Args:
        yields_argus: ARGUS fit yields
        yields_template: Template fit yields
        result_argus: ARGUS fit result
        result_template: Template fit result
        output_path: Output file path
    """
    states: list[str] = ["jpsi", "etac", "chic0", "chic1", "etac_2s", "background"]
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("BACKGROUND MODEL COMPARISON: ARGUS vs SIDEBAND TEMPLATE\n")
        f.write("=" * 80 + "\n\n")
        f.write("Fit Status\n")
        f.write("-" * 80 + "\n")
        f.write(f"ARGUS:    status = {result_argus.status()}, EDM = {result_argus.edm():.2e}\n")
        f.write(
            f"Template: status = {result_template.status()}, EDM = {result_template.edm():.2e}\n\n"
        )
        f.write("Yield Comparison\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'State':<12} {'ARGUS':>18} {'Template':>18} {'Diff':>12} {'Rel.Diff':>10}\n")
        f.write("-" * 80 + "\n")
        for state in states:
            n_argus: float = yields_argus[state].getVal()
            e_argus: float = yields_argus[state].getError()
            n_template: float = yields_template[state].getVal()
            e_template: float = yields_template[state].getError()
            diff: float = n_template - n_argus
            rel_diff: float = diff / n_argus * 100 if n_argus != 0 else 0
            f.write(
                f"{state:<12} {n_argus:>8.0f} ± {e_argus:<6.0f} "
                f"{n_template:>8.0f} ± {e_template:<6.0f} "
                f"{diff:>+10.0f} {rel_diff:>+8.1f}%\n"
            )
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("SYSTEMATIC UNCERTAINTY FROM BACKGROUND MODELING\n")
        f.write("-" * 80 + "\n")
        f.write("The relative difference between ARGUS and Template yields\n")
        f.write("can be used as a systematic uncertainty estimate.\n\n")
        for state in ["jpsi", "etac", "chic0", "chic1", "etac_2s"]:
            n_argus = yields_argus[state].getVal()
            n_template = yields_template[state].getVal()
            diff = abs(n_template - n_argus)
            rel_diff = diff / n_argus * 100 if n_argus != 0 else 0
            f.write(f"  {state}: σ_bkg = {rel_diff:.1f}%\n")
        f.write("=" * 80 + "\n")
    print(f"Saved: {output_path}")


def main() -> None:
    """Main function to run background model comparison."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Compare ARGUS vs Template background models"
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2016,2017,2018",
        help="Comma-separated list of years (default: 2016,2017,2018)",
    )
    parser.add_argument(
        "--track-types",
        type=str,
        default="LL,DD",
        help="Comma-separated list of track types (default: LL,DD)",
    )
    parser.add_argument(
        "--template-file",
        type=str,
        default=None,
        help="Path to template ROOT file (default: output/background_template.root)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: output/)",
    )
    args: argparse.Namespace = parser.parse_args()
    # Parse arguments
    years: list[str] = [y.strip() for y in args.years.split(",")]
    track_types: list[str] = [t.strip() for t in args.track_types.split(",")]
    # Setup
    setup_root_style()
    config = get_config()
    # Output directory
    if args.output_dir:
        output_dir: Path = Path(args.output_dir)
    else:
        output_dir = STUDY_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Template file
    if args.template_file:
        template_file: Path = Path(args.template_file)
    else:
        template_file = output_dir / "background_template.root"
    if not template_file.exists():
        print(f"\nERROR: Template file not found: {template_file}")
        print("Run extract_template.py first to create the background template.")
        return
    print("\n" + "=" * 70)
    print("BACKGROUND MODEL COMPARISON: ARGUS vs SIDEBAND TEMPLATE")
    print("=" * 70)
    print(f"Years: {years}")
    print(f"Track types: {track_types}")
    print(f"Template file: {template_file}")
    print(f"Output directory: {output_dir}\n")
    # Load data using DataManager
    print("Loading data...")
    data_manager: DataManager = DataManager(config)
    all_data: list[ak.Array] = []
    for year in years:
        for magnet in ["MD", "MU"]:
            for track_type in track_types:
                try:
                    events = data_manager.load_tree(
                        particle_type="data",
                        year=year,
                        magnet=magnet,
                        track_type=track_type,
                        apply_derived_branches=True,
                    )
                    if events is not None and len(events) > 0:
                        all_data.append(events)
                except Exception as e:
                    print(f"  Warning: Could not load {year} {magnet} {track_type}: {e}")
    if not all_data:
        print("ERROR: No data loaded!")
        return
    combined_data: ak.Array = ak.concatenate(all_data, axis=0)
    print(f"Total events loaded: {len(combined_data):,}")
    # Apply B+ mass selection (signal region)
    bu_mass_min: float = config.selection.get("bu_fixed_selection", {}).get(
        "mass_corrected_min", 5255.0
    )
    bu_mass_max: float = config.selection.get("bu_fixed_selection", {}).get(
        "mass_corrected_max", 5305.0
    )
    # Check which branch exists
    if "Bu_MM_corrected" in combined_data.fields:
        bu_mass_branch: str = "Bu_MM_corrected"
    elif "Bu_M" in combined_data.fields:
        bu_mass_branch = "Bu_M"
    else:
        bu_mass_branch = "Bu_MM"
    bu_mask = (combined_data[bu_mass_branch] > bu_mass_min) & (
        combined_data[bu_mass_branch] < bu_mass_max
    )
    signal_data: ak.Array = combined_data[bu_mask]
    print(f"Events in B+ signal region [{bu_mass_min:.0f}-{bu_mass_max:.0f}]: {len(signal_data):,}")
    # Get M(LpK-) values
    if "M_LpKm_h2" in signal_data.fields:
        mass_values = ak.to_numpy(signal_data["M_LpKm_h2"])
    elif "M_LpKm" in signal_data.fields:
        mass_values = ak.to_numpy(signal_data["M_LpKm"])
    else:
        print("ERROR: Cannot find M(LpK-) branch in data!")
        return
    # Create comparison object
    comparison: BackgroundModelComparison = BackgroundModelComparison(config)
    mass_var: ROOT.RooRealVar = comparison.setup_observable()
    # Create RooDataSet from data
    print("\nCreating RooDataSet...")
    data_set: ROOT.RooDataSet = ROOT.RooDataSet("data", "Data", ROOT.RooArgSet(mass_var))
    ROOT.SetOwnership(data_set, False)
    for m in mass_values:
        if MASS_CONFIG.MLPK_MIN < m < MASS_CONFIG.MLPK_MAX:
            mass_var.setVal(m)
            data_set.add(ROOT.RooArgSet(mass_var))
    print(f"RooDataSet entries: {data_set.numEntries():,}")
    # Create binned data for fitting
    data_hist: ROOT.RooDataHist = ROOT.RooDataHist(
        "data_hist",
        "Binned data",
        ROOT.RooArgSet(mass_var),
        data_set,
    )
    ROOT.SetOwnership(data_hist, False)
    # Build ARGUS model
    print("\n" + "=" * 70)
    print("Building ARGUS background model...")
    print("=" * 70)
    bkg_argus, _ = comparison.create_argus_background(mass_var)
    model_argus, yields_argus = comparison.build_model(mass_var, bkg_argus, "argus")
    # Build Template model
    print("\n" + "=" * 70)
    print("Building Template background model...")
    print("=" * 70)
    bkg_template: ROOT.RooHistPdf = comparison.load_template_background(mass_var, template_file)
    model_template, yields_template = comparison.build_model(mass_var, bkg_template, "template")
    # Perform ARGUS fit
    print("\n" + "=" * 70)
    print("Performing ARGUS fit...")
    print("=" * 70)
    result_argus: ROOT.RooFitResult = comparison.perform_fit(model_argus, data_hist)
    print(f"ARGUS fit status: {result_argus.status()}")
    # Reset signal parameters for template fit (to get independent fit)
    # Note: We need to reset yields but keep shared signal shape
    for state in ["jpsi", "etac", "chic0", "chic1", "etac_2s"]:
        yields_template[state].setVal(1000)
    yields_template["background"].setVal(10000)
    if comparison.resolution is not None:
        comparison.resolution.setVal(8.0)
    # Perform Template fit
    print("\n" + "=" * 70)
    print("Performing Template fit...")
    print("=" * 70)
    result_template: ROOT.RooFitResult = comparison.perform_fit(model_template, data_hist)
    print(f"Template fit status: {result_template.status()}")
    # Create comparison plot
    print("\n" + "=" * 70)
    print("Creating comparison plot...")
    print("=" * 70)
    comparison.create_comparison_plot(
        mass_var=mass_var,
        data=data_hist,
        model_argus=model_argus,
        model_template=model_template,
        yields_argus=yields_argus,
        yields_template=yields_template,
        output_path=output_dir / "background_comparison.pdf",
    )
    # Save results
    save_comparison_results(
        yields_argus=yields_argus,
        yields_template=yields_template,
        result_argus=result_argus,
        result_template=result_template,
        output_path=output_dir / "background_comparison_results.txt",
    )
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'State':<12} {'ARGUS':>15} {'Template':>15} {'Rel.Diff':>12}")
    print("-" * 55)
    for state in ["jpsi", "etac", "chic0", "chic1", "etac_2s"]:
        n_argus: float = yields_argus[state].getVal()
        n_template: float = yields_template[state].getVal()
        rel_diff: float = (n_template - n_argus) / n_argus * 100 if n_argus != 0 else 0
        print(f"{state:<12} {n_argus:>15.0f} {n_template:>15.0f} {rel_diff:>+10.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
