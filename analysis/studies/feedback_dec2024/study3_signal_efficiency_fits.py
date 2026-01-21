#!/usr/bin/env python3
"""
Study 3: Signal Efficiency from M(B⁺) Fits

Fit M(B⁺) mass distribution for two categories:
1. All events
2. Events passing cuts

Extract signal yields from fits to calculate:
    Signal Efficiency = (Signal yield passing cuts) / (Total signal yield)

This is more robust than simple counting because it accounts for background.

Signal model: Crystal Ball (Gaussian core + power-law tail) - standard for B mesons
Background model: ARGUS function (kinematic endpoint at B+ mass)

Generates:
- 2 fit plots (all/pass) for MC
- 2 fit plots (all/pass) for Data
- Efficiency summary table

Output:
- analysis/studies/feedback_dec2024/output/study3/fit_*.pdf
- analysis/studies/feedback_dec2024/output/study3/efficiency_table.txt

Usage:
    cd analysis/studies/feedback_dec2024
    python study3_signal_efficiency_fits.py
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import ROOT

# Add analysis directory to path to access modules
SCRIPT_DIR: Path = Path(__file__).parent
ANALYSIS_DIR: Path = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ANALYSIS_DIR))

from modules.data_handler import TOMLConfig  # noqa: E402

# Disable ROOT GUI and suppress info messages
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

# Suppress RooFit messages except warnings
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

# Signal MC states
SIGNAL_STATES: list[str] = ["Jpsi", "etac", "chic0", "chic1"]

# Output directory
OUTPUT_DIR: Path = SCRIPT_DIR / "output" / "study3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Same cuts as Study 1 and Study 4
CUTS_ORDER: list[tuple[str, str, float]] = [
    ("Bu_DTF_chi2", "less", 30.0),
    ("Bu_FDCHI2_OWNPV", "greater", 100.0),
    ("Bu_IPCHI2_OWNPV", "less", 10.0),
    ("Bu_PT", "greater", 3000.0),
    ("h1_ProbNNk", "greater", 0.1),
    ("h2_ProbNNk", "greater", 0.1),
    ("p_ProbNNp", "greater", 0.1),
]


def load_trees(
    data_type: str,
    config: TOMLConfig,
    years: list[str] = ["2016", "2017", "2018"],
    track_types: list[str] = ["LL", "DD"],
) -> tuple[list[Any], list[Any]]:
    """
    Load ROOT trees for specified data type.

    Args:
        data_type: "mc" or "data"
        config: TOMLConfig object
        years: List of year strings
        track_types: List of track types

    Returns:
        (trees, tfiles) tuple to keep files alive
    """
    trees: list[Any] = []
    tfiles: list[Any] = []

    if data_type == "mc":
        # Load MC signal states
        mc_dir: Path = Path("/share/lazy/Mohamed/Bu2LambdaPPP/files/mc")
        magnets: list[str] = ["MD", "MU"]

        for state in SIGNAL_STATES:
            state_dir: Path = mc_dir / state

            for year in years:
                yy: str = year[2:]  # "2016" -> "16"
                for magnet in magnets:
                    for track_type in track_types:
                        filename: str = (
                            f"{state}_{yy}_{magnet}.root"  # Use original state name (Jpsi, not jpsi)
                        )
                        file_path: Path = state_dir / filename
                        tree_path: str = f"B2L0barPKpKm_{track_type}/DecayTree"

                        if not file_path.exists():
                            continue

                        tfile: Any = ROOT.TFile.Open(str(file_path), "READ")
                        if not tfile or tfile.IsZombie():
                            continue

                        tree: Any = tfile.Get(tree_path)
                        if not tree:
                            continue

                        trees.append(tree)
                        tfiles.append(tfile)  # Keep file alive

    elif data_type == "data":
        # Load Data files
        data_dir: Path = Path("/share/lazy/Mohamed/Bu2LambdaPPP/files/data")
        magnets: list[str] = ["MD", "MU"]

        for year in years:
            yy: str = year[2:]  # "2016" -> "16"
            for magnet in magnets:
                for track_type in track_types:
                    filename: str = f"dataBu2L0barPHH_{yy}{magnet}.root"
                    file_path: Path = data_dir / filename
                    tree_path: str = f"B2L0barPKpKm_{track_type}/DecayTree"

                    if not file_path.exists():
                        continue

                    tfile: Any = ROOT.TFile.Open(str(file_path), "READ")
                    if not tfile or tfile.IsZombie():
                        continue

                    tree: Any = tfile.Get(tree_path)
                    if not tree:
                        continue

                    trees.append(tree)
                    tfiles.append(tfile)

    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    return trees, tfiles


def create_cuts_string() -> str:
    """
    Create TCut string from all selection cuts.

    Returns:
        Cut string for ROOT.TCut
    """
    cuts: list[str] = []

    for branch, cut_type, value in CUTS_ORDER:
        if cut_type == "greater":
            cuts.append(f"({branch} > {value})")
        else:  # cut_type == "less"
            cuts.append(f"({branch} < {value})")

    return " && ".join(cuts) if cuts else "1"


def fit_bu_mass(
    trees: list[Any],
    category: str,
    cuts_pass: str,
    data_type: str,
    bu_mass_var: str = "Bu_MM",
    fit_range: tuple[float, float] = (5150, 5450),
    signal_window: tuple[float, float] = (5255, 5305),
) -> dict[str, float]:
    """
    Fit M(B⁺) distribution with signal + background model.

    Signal: Double Gaussian (narrow core + wide tail)
    Background: Exponential

    Args:
        trees: List of ROOT trees
        category: "all", "pass", or "fail"
        cuts_pass: TCut string for passing cuts
        data_type: "mc" or "data"
        bu_mass_var: Branch name for B+ mass
        fit_range: Fit range (min, max) in MeV
        signal_window: Signal region for initial yield estimate

    Returns:
        Dictionary with fit results:
            - n_signal: Signal yield
            - n_signal_error: Signal yield error
            - n_background: Background yield
            - n_background_error: Background yield error
            - mean: Signal mean
            - sigma: Signal width (core Gaussian)
            - chi2_ndf: Chi-squared/ndf
            - status: Fit status (0 = good)
    """
    print(f"\n{'='*60}")
    print(f"Fitting {data_type.upper()} - {category.upper()} category")
    print(f"{'='*60}")

    # Create temporary histogram to collect events efficiently
    temp_hist = ROOT.TH1D("temp_bu_mass", "temp", 300, fit_range[0], fit_range[1])
    temp_hist.Sumw2()

    # Determine which events to include
    if category == "all":
        cut_formula = ""  # No cuts
    elif category == "pass":
        cut_formula = cuts_pass
    elif category == "fail":
        cut_formula = f"!({cuts_pass})"
    else:
        raise ValueError(f"Unknown category: {category}")

    # Fill histogram from trees using ROOT's Draw (much faster)
    print(f"Loading events from {len(trees)} trees...")

    for tree in trees:
        if cut_formula:
            tree.Draw(f"{bu_mass_var}>>+temp_bu_mass", cut_formula, "goff")
        else:
            tree.Draw(f"{bu_mass_var}>>+temp_bu_mass", "", "goff")

    n_events = int(temp_hist.Integral())
    print(f"  Events in dataset: {n_events}")

    if n_events < 100:
        print(f"  WARNING: Too few events ({n_events}), skipping fit")
        temp_hist.Delete()
        return {
            "n_signal": 0.0,
            "n_signal_error": 0.0,
            "n_background": 0.0,
            "n_background_error": 0.0,
            "mean": 5279.0,
            "sigma": 10.0,
            "chi2_ndf": 0.0,
            "status": -1,
        }

    # Create mass observable
    mass = ROOT.RooRealVar("Bu_mass", "M(B^{+}) [MeV/c^{2}]", fit_range[0], fit_range[1])
    mass.setRange("signal", signal_window[0], signal_window[1])
    mass.setBins(300)

    # Create RooDataHist from histogram (binned fit - much faster)
    dataset = ROOT.RooDataHist("data", "Data", ROOT.RooArgList(mass), temp_hist)

    # Clean up temporary histogram
    temp_hist.Delete()

    # Build signal PDF: Crystal Ball function (standard for B meson peaks)
    # Crystal Ball = Gaussian core + power-law tail
    mean = ROOT.RooRealVar("mean", "mean", 5279.0, 5270.0, 5290.0)
    sigma = ROOT.RooRealVar("sigma", "sigma", 10.0, 3.0, 25.0)
    alpha = ROOT.RooRealVar("alpha", "alpha", 1.5, 0.5, 5.0)  # Tail parameter
    n = ROOT.RooRealVar("n", "n", 2.0, 0.5, 10.0)  # Tail parameter

    signal_pdf = ROOT.RooCBShape("signal", "Crystal Ball", mass, mean, sigma, alpha, n)

    # Background PDF: ARGUS function (standard for B meson combinatorial background)
    # ARGUS: m * sqrt(1 - (m/m0)^2) * exp(c * (1 - (m/m0)^2))
    m0 = ROOT.RooRealVar("m0", "ARGUS endpoint", 5290.0)  # Fixed near B+ mass
    m0.setConstant(True)
    c_argus = ROOT.RooRealVar("c_argus", "ARGUS shape", -20.0, -100.0, -0.1)
    bkg_pdf = ROOT.RooArgusBG("background", "ARGUS background", mass, m0, c_argus)

    # Signal and background yields
    n_sig_init = n_events * 0.1  # Initial guess: 10% signal
    n_bkg_init = n_events * 0.9  # Initial guess: 90% background

    n_signal = ROOT.RooRealVar("n_signal", "Signal yield", n_sig_init, 0, n_events)
    n_background = ROOT.RooRealVar("n_background", "Background yield", n_bkg_init, 0, n_events * 2)

    # Total PDF
    model = ROOT.RooAddPdf(
        "model",
        "Signal + Background",
        ROOT.RooArgList(signal_pdf, bkg_pdf),
        ROOT.RooArgList(n_signal, n_background),
    )

    # Perform fit
    print("  Performing fit...")
    fit_result = model.fitTo(
        dataset,
        ROOT.RooFit.Save(),
        ROOT.RooFit.PrintLevel(-1),
        ROOT.RooFit.Strategy(2),
        ROOT.RooFit.NumCPU(4),
    )

    fit_status = fit_result.status()
    print(f"  Fit status: {fit_status}")

    if fit_status != 0:
        print("  WARNING: Fit did not converge properly!")

    # Extract results
    n_sig_val = n_signal.getVal()
    n_sig_err = n_signal.getError()
    n_bkg_val = n_background.getVal()
    n_bkg_err = n_background.getError()
    mean_val = mean.getVal()
    sigma_val = sigma.getVal()

    print(f"  Signal yield: {n_sig_val:.0f} ± {n_sig_err:.0f}")
    print(f"  Background yield: {n_bkg_val:.0f} ± {n_bkg_err:.0f}")
    print(f"  Signal mean: {mean_val:.2f} MeV")
    print(f"  Signal sigma: {sigma_val:.2f} MeV")

    # Plot result
    plot_fit(mass, dataset, model, signal_pdf, bkg_pdf, category, data_type, n_sig_val, n_bkg_val)

    # Calculate chi2/ndf
    frame = mass.frame()
    dataset.plotOn(frame)
    model.plotOn(frame)
    chi2 = frame.chiSquare()

    return {
        "n_signal": n_sig_val,
        "n_signal_error": n_sig_err,
        "n_background": n_bkg_val,
        "n_background_error": n_bkg_err,
        "mean": mean_val,
        "sigma": sigma_val,
        "chi2_ndf": chi2,
        "status": fit_status,
    }


def plot_fit(
    mass: Any,
    dataset: Any,
    model: Any,
    signal_pdf: Any,
    bkg_pdf: Any,
    category: str,
    data_type: str,
    n_signal: float,
    n_background: float,
) -> None:
    """
    Plot fit result with data and model components.

    Args:
        mass: RooRealVar for mass
        dataset: RooDataSet
        model: Total PDF
        signal_pdf: Signal component
        bkg_pdf: Background component
        category: "all", "pass", or "fail"
        data_type: "mc" or "data"
        n_signal: Signal yield
        n_background: Background yield
    """
    canvas = ROOT.TCanvas("canvas", "canvas", 800, 600)
    canvas.SetLeftMargin(0.14)
    canvas.SetRightMargin(0.05)

    # Create frame
    frame = mass.frame()
    frame.SetTitle("")

    # Plot data
    dataset.plotOn(frame, ROOT.RooFit.MarkerSize(0.8))

    # Plot total model
    model.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.LineWidth(2))

    # Plot signal component
    model.plotOn(
        frame,
        ROOT.RooFit.Components("signal"),
        ROOT.RooFit.LineColor(ROOT.kRed),
        ROOT.RooFit.LineStyle(2),
        ROOT.RooFit.LineWidth(2),
    )

    # Plot background component
    model.plotOn(
        frame,
        ROOT.RooFit.Components("background"),
        ROOT.RooFit.LineColor(ROOT.kGreen + 2),
        ROOT.RooFit.LineStyle(2),
        ROOT.RooFit.LineWidth(2),
    )

    # Draw frame
    frame.GetXaxis().SetTitle("M(B^{+}) [MeV/c^{2}]")
    frame.GetYaxis().SetTitle(f"Candidates / {(mass.getMax() - mass.getMin()) / 60:.1f} MeV")
    frame.GetXaxis().SetTitleFont(132)
    frame.GetYaxis().SetTitleFont(132)
    frame.GetXaxis().SetLabelFont(132)
    frame.GetYaxis().SetLabelFont(132)
    frame.GetXaxis().SetTitleSize(0.045)
    frame.GetYaxis().SetTitleSize(0.045)
    frame.GetYaxis().SetTitleOffset(1.4)
    frame.Draw()

    # Add legend
    legend = ROOT.TLegend(0.65, 0.65, 0.93, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(132)
    legend.SetTextSize(0.035)
    legend.AddEntry(frame.findObject(frame.nameOf(0)), "Data", "lep")
    legend.AddEntry(frame.findObject(frame.nameOf(1)), "Total fit", "l")
    legend.AddEntry(frame.findObject(frame.nameOf(2)), "Signal", "l")
    legend.AddEntry(frame.findObject(frame.nameOf(3)), "Background", "l")
    legend.Draw()

    # Add title text
    title_map = {"all": "All events", "pass": "Events passing cuts", "fail": "Events failing cuts"}
    title_text = ROOT.TLatex()
    title_text.SetNDC()
    title_text.SetTextFont(132)
    title_text.SetTextSize(0.045)
    title_text.DrawLatex(0.16, 0.92, f"{title_map[category]} ({data_type.upper()})")

    # Add yield text
    yield_text = ROOT.TLatex()
    yield_text.SetNDC()
    yield_text.SetTextFont(132)
    yield_text.SetTextSize(0.032)
    yield_text.DrawLatex(0.16, 0.78, f"N_{{signal}} = {n_signal:.0f}")
    yield_text.DrawLatex(0.16, 0.73, f"N_{{bkg}} = {n_background:.0f}")

    # Save
    output_file = OUTPUT_DIR / f"fit_{data_type}_{category}.pdf"
    canvas.SaveAs(str(output_file))
    canvas.SaveAs(str(output_file.with_suffix(".png")))
    print(f"  Saved: {output_file.name}")

    canvas.Close()


def calculate_efficiency(
    n_total: float,
    n_total_err: float,
    n_pass: float,
    n_pass_err: float,
) -> tuple[float, float]:
    """
    Calculate efficiency with error propagation.

    Efficiency = N_pass / N_total
    Error = sqrt((dN_pass/N_total)^2 + (N_pass*dN_total/N_total^2)^2)

    Args:
        n_total: Total signal yield
        n_total_err: Total signal yield error
        n_pass: Passing signal yield
        n_pass_err: Passing signal yield error

    Returns:
        (efficiency, efficiency_error) tuple
    """
    if n_total == 0:
        return 0.0, 0.0

    eff = n_pass / n_total

    # Error propagation
    import math

    term1 = (n_pass_err / n_total) ** 2
    term2 = ((n_pass * n_total_err) / (n_total**2)) ** 2
    eff_err = math.sqrt(term1 + term2)

    return eff, eff_err


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Study 3: Signal efficiency from M(B+) fits")
    parser.add_argument("--config", default="config", help="Config directory")
    args = parser.parse_args()

    print("=" * 80)
    print("STUDY 3: Signal Efficiency from M(B⁺) Fits")
    print("=" * 80)

    # Load configuration
    config_dir: Path = ANALYSIS_DIR / args.config
    config: TOMLConfig = TOMLConfig(config_dir)

    # Get cuts string
    cuts_pass: str = create_cuts_string()
    print(f"\nSelection cuts:\n  {cuts_pass}")

    # Results storage
    results: dict[str, dict[str, dict]] = {"mc": {}, "data": {}}

    # Process MC and Data
    for data_type in ["mc", "data"]:
        print(f"\n{'='*80}")
        print(f"Processing {data_type.upper()}")
        print(f"{'='*80}")

        # Use Bu_MM for both MC and Data
        bu_mass_var = "Bu_MM"

        # Load trees
        trees, tfiles = load_trees(data_type, config)
        print(f"Loaded {len(trees)} trees")

        # Fit two categories: all and pass
        # We don't need to fit "fail" - efficiency is just Signal_pass / Signal_all
        for category in ["all", "pass"]:
            fit_results = fit_bu_mass(
                trees,
                category,
                cuts_pass,
                data_type,
                bu_mass_var=bu_mass_var,
            )
            results[data_type][category] = fit_results

        # Keep files alive during fitting
        del trees, tfiles

    # Calculate efficiencies
    print("\n" + "=" * 80)
    print("SIGNAL EFFICIENCY RESULTS")
    print("=" * 80)

    efficiency_lines: list[str] = []
    efficiency_lines.append("\nSignal Efficiency from M(B+) Fits")
    efficiency_lines.append("=" * 60)

    for data_type in ["mc", "data"]:
        efficiency_lines.append(f"\n{data_type.upper()}:")
        efficiency_lines.append("-" * 60)

        n_total = results[data_type]["all"]["n_signal"]
        n_total_err = results[data_type]["all"]["n_signal_error"]
        n_pass = results[data_type]["pass"]["n_signal"]
        n_pass_err = results[data_type]["pass"]["n_signal_error"]

        eff, eff_err = calculate_efficiency(n_total, n_total_err, n_pass, n_pass_err)

        efficiency_lines.append(f"  Total signal yield:   {n_total:8.0f} ± {n_total_err:6.0f}")
        efficiency_lines.append(f"  Signal passing cuts:  {n_pass:8.0f} ± {n_pass_err:6.0f}")
        efficiency_lines.append(
            f"  Signal efficiency:    {eff:8.4f} ± {eff_err:6.4f} ({eff*100:.2f}%)"
        )

    # Print and save
    for line in efficiency_lines:
        print(line)

    efficiency_file = OUTPUT_DIR / "efficiency_table.txt"
    with open(efficiency_file, "w") as f:
        f.write("\n".join(efficiency_lines))

    print(f"\nSaved efficiency table: {efficiency_file}")
    print(f"Generated {len(results['mc'])} MC + {len(results['data'])} Data fit plots")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
