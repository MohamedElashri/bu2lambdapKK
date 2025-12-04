#!/usr/bin/env python3
"""
Phase 3: Template-Based Mass Fitting

This script performs M(Lambda p K-) fits using the sideband-derived
background template. Compares results with the standard ARGUS background model.

Method:
-------
1. Load background template from extract_template.py output
2. Load signal region data
3. Fit with: Signal (DCB for each charmonium) + Background (template PDF)
4. Compare with ARGUS background fit
5. Evaluate systematic differences

Usage:
------
    cd analysis/studies/sideband_background
    python template_fitter.py
    python template_fitter.py --compare-argus
    python template_fitter.py --template-file output/background_template.root

Output:
-------
    output/template_fit_result.pdf
    output/template_fit_comparison.pdf (if --compare-argus)
    output/fit_results.txt
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import ROOT

# Add parent directories to path
STUDY_DIR: Path = Path(__file__).parent
ANALYSIS_DIR: Path = STUDY_DIR.parent.parent
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(STUDY_DIR))

from config import MASS_CONFIG
from data_loader import (
    build_cut_string,
    get_config,
    load_mlpk_histogram_in_mbu_region,
    setup_root_style,
)


def load_template_pdf(
    template_file: Path,
    mass_var: ROOT.RooRealVar,
    use_smoothed: bool = False,
) -> ROOT.RooHistPdf:
    """
    Load background template PDF from ROOT file.

    Args:
        template_file: Path to template ROOT file
        mass_var: RooRealVar for mass
        use_smoothed: Whether to use smoothed template

    Returns:
        RooHistPdf for background
    """
    tfile: ROOT.TFile = ROOT.TFile.Open(str(template_file), "READ")
    if not tfile or tfile.IsZombie():
        raise FileNotFoundError(f"Cannot open template file: {template_file}")
    # Get histogram
    hist_name: str = (
        "background_template_smooth_normalized"
        if use_smoothed
        else "background_template_raw_normalized"
    )
    hist: ROOT.TH1D = tfile.Get(hist_name)
    if not hist:
        # Try non-normalized version
        hist_name = "background_template_smooth" if use_smoothed else "background_template_raw"
        hist = tfile.Get(hist_name)
    if not hist:
        tfile.Close()
        raise ValueError(f"Template histogram not found in {template_file}")
    # Clone to keep in memory - must set directory to None before closing file
    hist_clone: ROOT.TH1D = hist.Clone("bkg_template_hist")
    hist_clone.SetDirectory(0)  # Detach from file before closing
    ROOT.SetOwnership(hist_clone, False)
    tfile.Close()
    # Normalize
    integral: float = hist_clone.Integral(1, hist_clone.GetNbinsX())
    if integral > 0:
        hist_clone.Scale(1.0 / integral)
    # Create RooDataHist
    data_hist: ROOT.RooDataHist = ROOT.RooDataHist(
        "bkg_datahist",
        "Background template",
        ROOT.RooArgList(mass_var),
        hist_clone,
    )
    ROOT.SetOwnership(data_hist, False)
    # Create RooHistPdf
    bkg_pdf: ROOT.RooHistPdf = ROOT.RooHistPdf(
        "bkg_template",
        "Background template PDF",
        ROOT.RooArgSet(mass_var),
        data_hist,
        2,  # Interpolation order
    )
    ROOT.SetOwnership(bkg_pdf, False)
    return bkg_pdf


def create_signal_pdf(
    mass_var: ROOT.RooRealVar,
    state_name: str,
    state_mass: float,
    state_width: float,
) -> tuple[ROOT.RooGaussian, ROOT.RooRealVar, ROOT.RooRealVar]:
    """
    Create Gaussian signal PDF for a charmonium state.

    For simplicity, using Gaussian instead of DCB. Can be upgraded later.

    Args:
        mass_var: RooRealVar for mass
        state_name: Name of the state (e.g., "jpsi")
        state_mass: Central mass value
        state_width: Expected width (resolution)

    Returns:
        Tuple of (PDF, mean, sigma)
    """
    # Mean (fixed to PDG value)
    mean: ROOT.RooRealVar = ROOT.RooRealVar(
        f"mean_{state_name}",
        f"Mean {state_name}",
        state_mass,
    )
    mean.setConstant(True)
    ROOT.SetOwnership(mean, False)
    # Sigma (floating, initialized to expected resolution)
    sigma: ROOT.RooRealVar = ROOT.RooRealVar(
        f"sigma_{state_name}",
        f"Sigma {state_name}",
        state_width,
        state_width * 0.5,
        state_width * 2.0,
    )
    ROOT.SetOwnership(sigma, False)
    # Gaussian PDF
    gauss: ROOT.RooGaussian = ROOT.RooGaussian(
        f"sig_{state_name}",
        f"Signal {state_name}",
        mass_var,
        mean,
        sigma,
    )
    ROOT.SetOwnership(gauss, False)
    return gauss, mean, sigma


def create_argus_pdf(
    mass_var: ROOT.RooRealVar,
    endpoint: float,
) -> tuple[ROOT.RooArgusBG, ROOT.RooRealVar, ROOT.RooRealVar]:
    """
    Create ARGUS background PDF.

    Args:
        mass_var: RooRealVar for mass
        endpoint: ARGUS endpoint (threshold)

    Returns:
        Tuple of (PDF, endpoint_var, slope_var)
    """
    # Endpoint (fixed)
    m0: ROOT.RooRealVar = ROOT.RooRealVar("argus_m0", "ARGUS endpoint", endpoint)
    m0.setConstant(True)
    ROOT.SetOwnership(m0, False)
    # Slope parameter (floating)
    c: ROOT.RooRealVar = ROOT.RooRealVar("argus_c", "ARGUS slope", -20.0, -100.0, 0.0)
    ROOT.SetOwnership(c, False)
    # ARGUS PDF
    argus: ROOT.RooArgusBG = ROOT.RooArgusBG("bkg_argus", "ARGUS background", mass_var, m0, c)
    ROOT.SetOwnership(argus, False)
    return argus, m0, c


def perform_template_fit(
    data_hist: ROOT.RooDataHist,
    mass_var: ROOT.RooRealVar,
    bkg_pdf: ROOT.RooAbsPdf,
    output_dir: Path,
    suffix: str = "",
) -> dict[str, Any]:
    """
    Perform fit with signal + template background.

    Args:
        data_hist: Data histogram
        mass_var: Mass variable
        bkg_pdf: Background PDF (template or ARGUS)
        output_dir: Output directory
        suffix: Suffix for output files

    Returns:
        Dictionary of fit results
    """
    keep_alive: list[Any] = []
    results: dict[str, Any] = {}
    # Define charmonium states with expected resolutions
    states: list[tuple[str, float, float]] = [
        ("etac", 2983.9, 30.0),  # eta_c(1S) - broad
        ("jpsi", 3096.9, 10.0),  # J/psi - narrow
        ("chic0", 3414.7, 15.0),  # chi_c0
        ("chic1", 3510.7, 12.0),  # chi_c1
        ("etac2s", 3637.5, 15.0),  # eta_c(2S)
    ]
    # Create signal PDFs
    signal_pdfs: list[ROOT.RooGaussian] = []
    signal_yields: list[ROOT.RooRealVar] = []
    pdf_list: ROOT.RooArgList = ROOT.RooArgList()
    yield_list: ROOT.RooArgList = ROOT.RooArgList()
    for state_name, state_mass, state_width in states:
        sig_pdf, mean, sigma = create_signal_pdf(mass_var, state_name, state_mass, state_width)
        signal_pdfs.append(sig_pdf)
        keep_alive.extend([sig_pdf, mean, sigma])
        # Yield
        n_sig: ROOT.RooRealVar = ROOT.RooRealVar(
            f"n_{state_name}",
            f"N({state_name})",
            1000,
            0,
            100000,
        )
        ROOT.SetOwnership(n_sig, False)
        signal_yields.append(n_sig)
        keep_alive.append(n_sig)
        pdf_list.add(sig_pdf)
        yield_list.add(n_sig)
    # Background yield
    n_bkg: ROOT.RooRealVar = ROOT.RooRealVar("n_bkg", "N(background)", 10000, 0, 1000000)
    ROOT.SetOwnership(n_bkg, False)
    keep_alive.append(n_bkg)
    pdf_list.add(bkg_pdf)
    yield_list.add(n_bkg)
    # Total PDF
    total_pdf: ROOT.RooAddPdf = ROOT.RooAddPdf("total_pdf", "Total PDF", pdf_list, yield_list)
    ROOT.SetOwnership(total_pdf, False)
    keep_alive.append(total_pdf)
    # Fit
    print("\nPerforming fit...")
    fit_result: ROOT.RooFitResult = total_pdf.fitTo(
        data_hist,
        ROOT.RooFit.Save(True),
        ROOT.RooFit.PrintLevel(-1),
        ROOT.RooFit.Strategy(2),
        ROOT.RooFit.Extended(True),
    )
    ROOT.SetOwnership(fit_result, False)
    keep_alive.append(fit_result)
    # Store results
    results["fit_status"] = fit_result.status()
    results["edm"] = fit_result.edm()
    results["n_bkg"] = (n_bkg.getVal(), n_bkg.getError())
    for i, (state_name, _, _) in enumerate(states):
        results[f"n_{state_name}"] = (signal_yields[i].getVal(), signal_yields[i].getError())
    # Create plot
    canvas: ROOT.TCanvas = ROOT.TCanvas("c_fit", "", 1200, 900)
    ROOT.SetOwnership(canvas, False)
    keep_alive.append(canvas)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.05)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.12)
    # Create frame
    frame: ROOT.RooPlot = mass_var.frame(ROOT.RooFit.Title("Template Fit Result"))
    ROOT.SetOwnership(frame, False)
    keep_alive.append(frame)
    # Plot data
    data_hist.plotOn(frame, ROOT.RooFit.Name("data"))
    # Plot total PDF
    total_pdf.plotOn(frame, ROOT.RooFit.Name("total"), ROOT.RooFit.LineColor(ROOT.kBlue))
    # Plot background component
    total_pdf.plotOn(
        frame,
        ROOT.RooFit.Components("bkg_template,bkg_argus"),
        ROOT.RooFit.Name("bkg"),
        ROOT.RooFit.LineColor(ROOT.kRed),
        ROOT.RooFit.LineStyle(ROOT.kDashed),
    )
    # Plot individual signal components
    colors: list[int] = [
        ROOT.kGreen + 2,
        ROOT.kMagenta,
        ROOT.kCyan + 1,
        ROOT.kOrange + 1,
        ROOT.kYellow + 2,
    ]
    for i, (state_name, _, _) in enumerate(states):
        total_pdf.plotOn(
            frame,
            ROOT.RooFit.Components(f"sig_{state_name}"),
            ROOT.RooFit.Name(f"sig_{state_name}"),
            ROOT.RooFit.LineColor(colors[i % len(colors)]),
            ROOT.RooFit.LineStyle(ROOT.kDotted),
        )
    frame.Draw()
    # Legend
    legend: ROOT.TLegend = ROOT.TLegend(0.60, 0.55, 0.92, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(132)
    legend.SetTextSize(0.030)
    legend.AddEntry("data", "Data", "lep")
    legend.AddEntry("total", "Total fit", "l")
    legend.AddEntry("bkg", "Background", "l")
    legend.Draw()
    keep_alive.append(legend)
    # LHCb label
    latex: ROOT.TLatex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(132)
    latex.SetTextSize(0.045)
    latex.DrawLatex(0.15, 0.85, "#font[62]{LHCb}")
    latex.SetTextSize(0.030)
    latex.DrawLatex(0.15, 0.80, f"Fit status: {fit_result.status()}")
    keep_alive.append(latex)
    # Fit results box
    results_text: ROOT.TPaveText = ROOT.TPaveText(0.15, 0.45, 0.45, 0.75, "NDC")
    results_text.SetBorderSize(0)
    results_text.SetFillStyle(0)
    results_text.SetTextFont(132)
    results_text.SetTextSize(0.025)
    results_text.SetTextAlign(12)
    for state_name, _, _ in states:
        n_val, n_err = results[f"n_{state_name}"]
        results_text.AddText(f"N({state_name}) = {n_val:.0f} #pm {n_err:.0f}")
    n_bkg_val, n_bkg_err = results["n_bkg"]
    results_text.AddText(f"N(bkg) = {n_bkg_val:.0f} #pm {n_bkg_err:.0f}")
    results_text.Draw()
    keep_alive.append(results_text)
    canvas.Update()
    canvas.SaveAs(str(output_dir / f"template_fit_result{suffix}.pdf"))
    print(f"Saved: {output_dir / f'template_fit_result{suffix}.pdf'}")
    return results


def save_fit_results(
    results_template: dict[str, Any],
    results_argus: dict[str, Any] | None,
    output_path: Path,
) -> None:
    """
    Save fit results to text file.

    Args:
        results_template: Template fit results
        results_argus: ARGUS fit results (or None)
        output_path: Output file path
    """
    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("FIT RESULTS: SIDEBAND TEMPLATE BACKGROUND\n")
        f.write("=" * 70 + "\n\n")
        f.write("Template Fit Results\n")
        f.write("-" * 70 + "\n")
        f.write(f"Fit status: {results_template['fit_status']}\n")
        f.write(f"EDM: {results_template['edm']:.2e}\n\n")
        f.write(f"{'State':<15} {'Yield':>15} {'Error':>15}\n")
        f.write("-" * 45 + "\n")
        for key, value in results_template.items():
            if key.startswith("n_") and key != "n_bkg":
                state = key.replace("n_", "")
                f.write(f"{state:<15} {value[0]:>15.0f} {value[1]:>15.0f}\n")
        f.write(
            f"{'background':<15} {results_template['n_bkg'][0]:>15.0f} {results_template['n_bkg'][1]:>15.0f}\n"
        )
        if results_argus is not None:
            f.write("\n\nARGUS Fit Results (for comparison)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Fit status: {results_argus['fit_status']}\n")
            f.write(f"EDM: {results_argus['edm']:.2e}\n\n")
            f.write(f"{'State':<15} {'Yield':>15} {'Error':>15}\n")
            f.write("-" * 45 + "\n")
            for key, value in results_argus.items():
                if key.startswith("n_") and key != "n_bkg":
                    state = key.replace("n_", "")
                    f.write(f"{state:<15} {value[0]:>15.0f} {value[1]:>15.0f}\n")
            f.write(
                f"{'background':<15} {results_argus['n_bkg'][0]:>15.0f} {results_argus['n_bkg'][1]:>15.0f}\n"
            )
            # Comparison
            f.write("\n\nComparison (Template - ARGUS)\n")
            f.write("-" * 70 + "\n")
            for key in results_template:
                if key.startswith("n_"):
                    state = key.replace("n_", "") if key != "n_bkg" else "background"
                    diff = results_template[key][0] - results_argus[key][0]
                    rel_diff = (
                        diff / results_argus[key][0] * 100 if results_argus[key][0] != 0 else 0
                    )
                    f.write(f"{state:<15} Δ = {diff:>+10.0f} ({rel_diff:>+6.1f}%)\n")
        f.write("\n" + "=" * 70 + "\n")
    print(f"Saved: {output_path}")


def main() -> None:
    """Main function to perform template-based fitting."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Perform M(LpK-) fit with sideband template background"
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
        "--use-smoothed",
        action="store_true",
        help="Use smoothed template instead of raw",
    )
    parser.add_argument(
        "--compare-argus",
        action="store_true",
        help="Also fit with ARGUS background for comparison",
    )
    parser.add_argument(
        "--no-cuts",
        action="store_true",
        help="Skip selection cuts",
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
    data_path: Path = Path(config.paths["data"]["base_path"])
    # Build cuts
    if args.no_cuts:
        base_cuts: str = ""
        print("\n*** Running WITHOUT selection cuts ***\n")
    else:
        manual_cuts: dict[str, Any] = config.selection.get("manual_cuts", {})
        lambda_selection: dict[str, Any] = config.selection.get("lambda_selection", {})
        base_cuts = build_cut_string(manual_cuts, lambda_selection)
        print("\n*** Running WITH selection cuts ***\n")
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
    print(f"Output directory: {output_dir}")
    print(f"Template file: {template_file}")
    print(f"Years: {years}")
    print(f"Track types: {track_types}\n")
    # Load signal region data
    print("=" * 60)
    print("Loading signal region data...")
    print("=" * 60)
    signal_region: tuple[float, float] = (MASS_CONFIG.SIGNAL_MIN, MASS_CONFIG.SIGNAL_MAX)
    print(f"Signal region: M(B+) in [{signal_region[0]:.0f}, {signal_region[1]:.0f}] MeV")
    hist_signal, n_signal = load_mlpk_histogram_in_mbu_region(
        data_path=data_path,
        years=years,
        track_types=track_types,
        mbu_region=signal_region,
        hist_name="h_signal_region",
        base_cuts=base_cuts,
    )
    print(f"Signal region events: {n_signal:,}")
    if n_signal == 0:
        print("\nERROR: No events in signal region!")
        return
    # Create mass variable
    mass_var: ROOT.RooRealVar = ROOT.RooRealVar(
        "mass",
        "#it{M}(#Lambda#it{p}#it{K}^{#minus})",
        MASS_CONFIG.MLPK_MIN,
        MASS_CONFIG.MLPK_MAX,
        "MeV/#it{c}^{2}",
    )
    ROOT.SetOwnership(mass_var, False)
    # Create RooDataHist from signal data
    data_hist: ROOT.RooDataHist = ROOT.RooDataHist(
        "data",
        "Signal region data",
        ROOT.RooArgList(mass_var),
        hist_signal,
    )
    ROOT.SetOwnership(data_hist, False)
    # Load template PDF
    print("\n" + "=" * 60)
    print("Loading background template...")
    print("=" * 60)
    bkg_template: ROOT.RooHistPdf = load_template_pdf(
        template_file=template_file,
        mass_var=mass_var,
        use_smoothed=args.use_smoothed,
    )
    print("Template PDF loaded successfully")
    # Perform template fit
    print("\n" + "=" * 60)
    print("Performing template fit...")
    print("=" * 60)
    suffix: str = "_no_cuts" if args.no_cuts else ""
    results_template = perform_template_fit(
        data_hist=data_hist,
        mass_var=mass_var,
        bkg_pdf=bkg_template,
        output_dir=output_dir,
        suffix=suffix,
    )
    # Optionally compare with ARGUS
    results_argus: dict[str, Any] | None = None
    if args.compare_argus:
        print("\n" + "=" * 60)
        print("Performing ARGUS fit for comparison...")
        print("=" * 60)
        argus_endpoint: float = MASS_CONFIG.MLPK_MAX + 200.0
        bkg_argus, _, _ = create_argus_pdf(mass_var, argus_endpoint)
        results_argus = perform_template_fit(
            data_hist=data_hist,
            mass_var=mass_var,
            bkg_pdf=bkg_argus,
            output_dir=output_dir,
            suffix=f"{suffix}_argus",
        )
    # Save results
    save_fit_results(
        results_template=results_template,
        results_argus=results_argus,
        output_path=output_dir / f"fit_results{suffix}.txt",
    )
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Template fit status: {results_template['fit_status']}")
    print(
        f"Background yield: {results_template['n_bkg'][0]:.0f} ± {results_template['n_bkg'][1]:.0f}"
    )
    if results_argus:
        print(f"\nARGUS fit status: {results_argus['fit_status']}")
        print(
            f"Background yield: {results_argus['n_bkg'][0]:.0f} ± {results_argus['n_bkg'][1]:.0f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
