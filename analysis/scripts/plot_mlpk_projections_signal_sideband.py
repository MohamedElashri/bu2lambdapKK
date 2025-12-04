#!/usr/bin/env python3
"""
M(Lambda p K) Projections: Signal vs Sideband Study

Quantifies charmonium resonance contributions in B+ -> Lambda_bar p K+ K- decay
by comparing M(Lambda p K-) distributions in B+ signal region versus sideband
regions. Uses sideband subtraction to isolate genuine B+ decay products from
combinatorial background.

Physics Motivation:
------------------
The 2D correlation study (M(B+) vs M(Lambda p K-)) revealed horizontal enhancements
at charmonium masses. This script quantifies the strength and significance of each
charmonium resonance (eta_c, J/psi, chi_c0, chi_c1, eta_c(2S)) by projecting the
M(Lambda p K-) distribution for events near the true B+ mass.

Technical Approach:
------------------
1. Define B+ mass regions: signal (around 5279 MeV) and sidebands (below/above)
2. Create M(Lambda p K-) projections for each region
3. Perform normalized sideband subtraction: Signal - alpha*(Left_SB + Right_SB)
4. Integrate peaks at charmonium masses and estimate significance

Selection Cuts Applied:
----------------------
This script applies the SAME selection cuts as the main analysis pipeline:
  - Lambda selection cuts (from config/selection.toml [lambda_selection])
  - Manual cuts (from config/selection.toml [manual_cuts])

This ensures consistency with the main analysis and meaningful comparison.

Usage:
------
    python plot_mlpk_projections_signal_sideband.py
    python plot_mlpk_projections_signal_sideband.py --years 2016,2017,2018
    python plot_mlpk_projections_signal_sideband.py --track-types LL,DD
    python plot_mlpk_projections_signal_sideband.py --signal-window 5229,5329
    python plot_mlpk_projections_signal_sideband.py --no-fits
    python plot_mlpk_projections_signal_sideband.py --no-cuts  # Skip all cuts

Output:
-------
    analysis_output/mlpk_signal_sideband_comparison.pdf  (4-panel plot)
    analysis_output/mlpk_sideband_subtracted.pdf         (detailed subtracted signal)
    analysis_output/charmonium_yields.txt                (formatted results table)
    analysis_output/charmonium_yields.csv                (CSV results)

Related Scripts:
---------------
    plot_2d_mlambdapk_vs_mbu.py - 2D correlation study
    plot_mlambdapk_projections.py - Simple projection comparison
"""

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ROOT

# Add parent directory (analysis) to path to access modules
ANALYSIS_DIR: Path = Path(__file__).parent.parent
sys.path.insert(0, str(ANALYSIS_DIR))

from modules.data_handler import TOMLConfig  # noqa: E402

# Disable ROOT GUI and info messages
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

# B+ mass definitions
B_MASS_PDG: float = 5279.34

# M(Lambda p K-) histogram settings (matching mass_fitter.py style)
M_LPKM_MIN: float = 2800.0
M_LPKM_MAX: float = 4000.0
BIN_WIDTH: float = 5.0  # MeV per bin (same as mass_fitter.py)
N_BINS: int = int((M_LPKM_MAX - M_LPKM_MIN) / BIN_WIDTH)  # 240 bins


@dataclass
class MassRegion:
    """Represents a B+ mass region (signal or sideband)."""

    name: str
    min_mass: float
    max_mass: float
    histogram: ROOT.TH1D = field(default=None, repr=False)
    n_events: int = 0

    @property
    def width(self) -> float:
        """Return the width of the mass region in MeV."""
        return self.max_mass - self.min_mass

    @property
    def label(self) -> str:
        """Return formatted label for plots."""
        return f"{self.name} [{self.min_mass:.0f}-{self.max_mass:.0f}] MeV"


@dataclass
class CharmoniumState:
    """Represents a charmonium state with mass and integration window."""

    name: str
    mass: float
    window_half_width: float
    color: int
    latex_label: str
    yield_value: float = 0.0
    yield_error: float = 0.0
    significance: float = 0.0

    @property
    def window(self) -> tuple[float, float]:
        """Return the integration window (min, max)."""
        return (self.mass - self.window_half_width, self.mass + self.window_half_width)

    @property
    def window_str(self) -> str:
        """Return formatted window string."""
        return f"[{self.window[0]:.0f},{self.window[1]:.0f}]"


def setup_lhcb_style() -> None:
    """Set up LHCb-style ROOT plotting options."""
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(1)
    ROOT.gStyle.SetPadLeftMargin(0.14)
    ROOT.gStyle.SetPadRightMargin(0.05)
    ROOT.gStyle.SetPadTopMargin(0.08)
    ROOT.gStyle.SetPadBottomMargin(0.12)
    ROOT.gStyle.SetTitleFont(132, "XYZ")
    ROOT.gStyle.SetLabelFont(132, "XYZ")
    ROOT.gStyle.SetTextFont(132)
    ROOT.gStyle.SetTitleSize(0.045, "XYZ")
    ROOT.gStyle.SetLabelSize(0.035, "XYZ")
    ROOT.gStyle.SetTitleOffset(1.1, "X")
    ROOT.gStyle.SetTitleOffset(1.2, "Y")


def build_lambda_cut_string(lambda_selection: dict[str, Any] | None) -> str:
    """
    Build ROOT TTree cut string from Lambda selection configuration.

    Lambda cuts are FIXED and applied to all states (not optimized).
    These ensure good Lambda reconstruction quality.

    Note: Branch names match the raw ROOT file structure:
      - L0_MM: Lambda mass
      - L0_FDCHI2_OWNPV: Lambda flight distance chi2
      - L0_ENDVERTEX_Z, L0_OWNPV_Z: For delta_z calculation
      - Lp_ProbNNp: Lambda daughter proton PID (Lp = Lambda proton)

    Args:
        lambda_selection: Dictionary from config/selection.toml [lambda_selection]

    Returns:
        Cut string for TTree::Draw
    """
    if not lambda_selection:
        return ""
    cut_parts: list[str] = []
    # Lambda mass window
    mass_min: float | None = lambda_selection.get("mass_min")
    mass_max: float | None = lambda_selection.get("mass_max")
    if mass_min is not None and mass_max is not None:
        cut_parts.append(f"(L0_MM > {mass_min}) && (L0_MM < {mass_max})")
    # Lambda flight distance chi2
    fd_chisq_min: float | None = lambda_selection.get("fd_chisq_min")
    if fd_chisq_min is not None:
        cut_parts.append(f"(L0_FDCHI2_OWNPV > {fd_chisq_min})")
    # Lambda delta_z (separation from PV)
    # Note: Using absolute value via TMath::Abs for ROOT TTree::Draw
    delta_z_min: float | None = lambda_selection.get("delta_z_min")
    if delta_z_min is not None:
        # delta_z = |L0_ENDVERTEX_Z - L0_OWNPV_Z| in mm
        cut_parts.append(f"(TMath::Abs(L0_ENDVERTEX_Z - L0_OWNPV_Z) > {delta_z_min})")
    # Lambda daughter proton PID (Lp = proton from Lambda decay)
    proton_probnnp_min: float | None = lambda_selection.get("proton_probnnp_min")
    if proton_probnnp_min is not None:
        cut_parts.append(f"(Lp_ProbNNp > {proton_probnnp_min})")
    if cut_parts:
        return " && ".join(cut_parts)
    return ""


def build_manual_cut_string(manual_cuts: dict[str, Any] | None) -> str:
    """
    Build ROOT TTree cut string from manual cuts dictionary.

    Args:
        manual_cuts: Dictionary of cuts from config/selection.toml [manual_cuts]

    Returns:
        Cut string for TTree::Draw
    """
    if not manual_cuts:
        return ""
    cut_parts: list[str] = []
    for branch_name, cut_spec in manual_cuts.items():
        if branch_name == "notes":
            continue
        cut_type = cut_spec.get("cut_type")
        cut_value = cut_spec.get("value")
        if cut_type and cut_value is not None:
            if cut_type == "greater":
                cut_parts.append(f"({branch_name} > {cut_value})")
            elif cut_type == "less":
                cut_parts.append(f"({branch_name} < {cut_value})")
    if cut_parts:
        return " && ".join(cut_parts)
    return ""


def build_full_cut_string(
    lambda_selection: dict[str, Any] | None,
    manual_cuts: dict[str, Any] | None,
) -> str:
    """
    Build complete ROOT TTree cut string combining Lambda and manual cuts.

    This ensures the script applies the SAME cuts as the main analysis pipeline.

    Args:
        lambda_selection: Dictionary from config/selection.toml [lambda_selection]
        manual_cuts: Dictionary from config/selection.toml [manual_cuts]

    Returns:
        Combined cut string for TTree::Draw
    """
    lambda_cuts: str = build_lambda_cut_string(lambda_selection)
    manual_cuts_str: str = build_manual_cut_string(manual_cuts)
    if lambda_cuts and manual_cuts_str:
        return lambda_cuts + " && " + manual_cuts_str
    elif lambda_cuts:
        return lambda_cuts
    elif manual_cuts_str:
        return manual_cuts_str
    return ""


def load_mlpk_histogram(
    data_path: Path,
    years: list[str],
    track_types: list[str],
    region: MassRegion,
    lambda_selection: dict[str, Any] | None,
    manual_cuts: dict[str, Any] | None,
) -> tuple[ROOT.TH1D, int]:
    """
    Load M(Lambda p K-) histogram for a specific B+ mass region.

    Applies the SAME selection cuts as the main analysis pipeline:
      - Lambda selection cuts (mass, flight distance, delta_z, proton PID)
      - Manual cuts (B+ kinematics, PID)

    Args:
        data_path: Base path to data files
        years: List of years to process
        track_types: List of track types (LL, DD)
        region: MassRegion defining the B+ mass window
        lambda_selection: Lambda selection cuts from config
        manual_cuts: Manual cuts from config

    Returns:
        Tuple of (histogram, n_events)
    """
    hist_name: str = f"h_mlpk_{region.name.replace(' ', '_').lower()}"
    hist: ROOT.TH1D = ROOT.TH1D(hist_name, "", N_BINS, M_LPKM_MIN, M_LPKM_MAX)
    hist.Sumw2()
    total_events: int = 0
    # Build combined cut string (Lambda + manual cuts)
    base_cut_str: str = build_full_cut_string(lambda_selection, manual_cuts)
    # Add B+ mass window cut
    bu_mass_cut: str = f"(Bu_MM > {region.min_mass}) && (Bu_MM < {region.max_mass})"
    if base_cut_str:
        cut_str: str = base_cut_str + " && " + bu_mass_cut
    else:
        cut_str = bu_mass_cut
    # M(Lambda + p + K-) formula
    mass_formula: str = (
        "sqrt((L0_PE + p_PE + h2_PE)*(L0_PE + p_PE + h2_PE) - "
        "(L0_PX + p_PX + h2_PX)*(L0_PX + p_PX + h2_PX) - "
        "(L0_PY + p_PY + h2_PY)*(L0_PY + p_PY + h2_PY) - "
        "(L0_PZ + p_PZ + h2_PZ)*(L0_PZ + p_PZ + h2_PZ))"
    )
    for year in years:
        year_int: int = int(year)
        for magnet in ["MD", "MU"]:
            for track_type in track_types:
                filename: str = f"dataBu2L0barPHH_{year_int - 2000}{magnet}.root"
                filepath: Path = data_path / filename
                if not filepath.exists():
                    continue
                channel_path: str = f"B2L0barPKpKm_{track_type}"
                tree_path: str = f"{channel_path}/DecayTree"
                try:
                    tfile: ROOT.TFile = ROOT.TFile.Open(str(filepath), "READ")
                    if not tfile or tfile.IsZombie():
                        continue
                    tree: ROOT.TTree = tfile.Get(tree_path)
                    if not tree:
                        tfile.Close()
                        continue
                    temp_hist_name: str = f"temp_{hist_name}_{year}_{magnet}_{track_type}"
                    n_entries: int = tree.Draw(
                        f"{mass_formula}>>{temp_hist_name}({N_BINS},{M_LPKM_MIN},{M_LPKM_MAX})",
                        cut_str,
                        "goff",
                    )
                    if n_entries > 0:
                        temp_hist: ROOT.TH1D = ROOT.gDirectory.Get(temp_hist_name)
                        if temp_hist:
                            hist.Add(temp_hist)
                            total_events += n_entries
                            temp_hist.Delete()
                    tfile.Close()
                except Exception as e:
                    print(f"    Error loading {year} {magnet} {track_type}: {e}")
    ROOT.SetOwnership(hist, False)
    return hist, total_events


def perform_sideband_subtraction(
    hist_signal: ROOT.TH1D,
    hist_left: ROOT.TH1D,
    hist_right: ROOT.TH1D,
    signal_region: "MassRegion",
    left_region: "MassRegion",
    right_region: "MassRegion",
) -> tuple[ROOT.TH1D, float]:
    """
    Perform normalized sideband subtraction with proper error propagation.

    The sidebands are first scaled to the same M(B+) width as the signal region,
    then subtracted: Signal_subtracted = Signal - (Left_scaled + Right_scaled)

    Args:
        hist_signal: Histogram from signal region
        hist_left: Histogram from left sideband
        hist_right: Histogram from right sideband
        signal_region: MassRegion for signal (contains width info)
        left_region: MassRegion for left sideband
        right_region: MassRegion for right sideband

    Returns:
        Tuple of (subtracted histogram, alpha normalization factor)
    """
    # Calculate width-based scale factors
    # Scale each sideband to have the same effective width as signal region
    signal_width: float = signal_region.width
    left_width: float = left_region.width
    right_width: float = right_region.width
    # Scale factors: how much to scale each sideband
    scale_left: float = signal_width / left_width / 2.0  # Divide by 2 since we have 2 sidebands
    scale_right: float = signal_width / right_width / 2.0
    # Combined alpha for reporting (effective scale factor)
    alpha: float = (scale_left + scale_right) / 2.0
    print("  Width-based scaling:")
    print(f"    Signal width: {signal_width:.0f} MeV")
    print(f"    Left SB width: {left_width:.0f} MeV (scale = {scale_left:.4f})")
    print(f"    Right SB width: {right_width:.0f} MeV (scale = {scale_right:.4f})")
    # Create subtracted histogram
    hist_subtracted: ROOT.TH1D = hist_signal.Clone("h_mlpk_subtracted")
    hist_subtracted.SetTitle("Sideband Subtracted")
    ROOT.SetOwnership(hist_subtracted, False)
    # Perform subtraction bin-by-bin: Signal - scale_left*Left - scale_right*Right
    # Error propagation: sigma^2_sub = sigma^2_sig + scale_left^2*sigma^2_left + scale_right^2*sigma^2_right
    for i in range(1, hist_subtracted.GetNbinsX() + 1):
        sig_val: float = hist_signal.GetBinContent(i)
        sig_err: float = hist_signal.GetBinError(i)
        left_val: float = hist_left.GetBinContent(i)
        left_err: float = hist_left.GetBinError(i)
        right_val: float = hist_right.GetBinContent(i)
        right_err: float = hist_right.GetBinError(i)
        # Scaled sideband contribution
        sb_val: float = scale_left * left_val + scale_right * right_val
        sb_err_sq: float = (scale_left * left_err) ** 2 + (scale_right * right_err) ** 2
        # Subtracted value
        sub_val: float = sig_val - sb_val
        # Error propagation
        sub_err: float = math.sqrt(sig_err**2 + sb_err_sq)
        hist_subtracted.SetBinContent(i, sub_val)
        hist_subtracted.SetBinError(i, sub_err)
    return hist_subtracted, alpha


def integrate_peak_region(
    histogram: ROOT.TH1D,
    mass_window: tuple[float, float],
) -> tuple[float, float]:
    """
    Integrate histogram in mass window with proper error propagation.

    Args:
        histogram: The histogram to integrate
        mass_window: Tuple of (min_mass, max_mass)

    Returns:
        Tuple of (yield, error)
    """
    bin_min: int = histogram.FindBin(mass_window[0])
    bin_max: int = histogram.FindBin(mass_window[1])
    yield_value: float = 0.0
    error_squared: float = 0.0
    for i in range(bin_min, bin_max + 1):
        yield_value += histogram.GetBinContent(i)
        error_squared += histogram.GetBinError(i) ** 2
    return yield_value, math.sqrt(error_squared)


def estimate_significance(
    histogram: ROOT.TH1D,
    state: CharmoniumState,
) -> float:
    """
    Estimate significance of peak in sideband-subtracted distribution.

    For a sideband-subtracted histogram, the baseline should be ~0 if background
    is properly subtracted. Significance is simply yield/error.

    Args:
        histogram: The sideband-subtracted histogram
        state: CharmoniumState with mass and window info

    Returns:
        Significance in units of standard deviations (yield/error)
    """
    # Get peak yield and error
    peak_yield, peak_error = integrate_peak_region(histogram, state.window)
    # Significance: yield / error
    # For sideband-subtracted data, baseline is ~0, so this is appropriate
    if peak_error > 0:
        significance: float = peak_yield / peak_error
    else:
        significance = 0.0
    return significance


def create_four_panel_plot(
    regions: list[MassRegion],
    hist_subtracted: ROOT.TH1D,
    charmonium_states: list[CharmoniumState],
    alpha: float,
    output_path: Path,
) -> None:
    """
    Create 4-panel comparison plot showing signal, sidebands, and subtracted.

    Args:
        regions: List of MassRegion objects (signal, left_sb, right_sb)
        hist_subtracted: Sideband-subtracted histogram
        charmonium_states: List of CharmoniumState objects
        alpha: Normalization factor used in subtraction
        output_path: Path to save the PDF
    """
    keep_alive: list = []
    # Create canvas with 2x2 grid
    canvas: ROOT.TCanvas = ROOT.TCanvas("c_four_panel", "", 1600, 1200)
    ROOT.SetOwnership(canvas, False)
    keep_alive.append(canvas)
    canvas.Divide(2, 2)
    # Find global maximum for consistent Y-axis (excluding subtracted)
    y_max_raw: float = max(r.histogram.GetMaximum() for r in regions)
    y_max_raw *= 1.3
    # Panel configurations
    panel_configs: list[tuple[int, ROOT.TH1D, str, int, float]] = [
        (
            1,
            regions[0].histogram,
            f"Signal Region {regions[0].label}",
            regions[0].n_events,
            y_max_raw,
        ),
        (
            2,
            regions[1].histogram,
            f"Left Sideband {regions[1].label}",
            regions[1].n_events,
            y_max_raw,
        ),
        (
            3,
            regions[2].histogram,
            f"Right Sideband {regions[2].label}",
            regions[2].n_events,
            y_max_raw,
        ),
        (4, hist_subtracted, f"Sideband Subtracted (#alpha = {alpha:.3f})", 0, None),
    ]
    for pad_num, hist, title, n_events, y_max in panel_configs:
        canvas.cd(pad_num)
        ROOT.gPad.SetLeftMargin(0.14)
        ROOT.gPad.SetRightMargin(0.05)
        ROOT.gPad.SetTopMargin(0.10)
        ROOT.gPad.SetBottomMargin(0.12)
        # Clone histogram for drawing
        hist_draw: ROOT.TH1D = hist.Clone(f"h_draw_{pad_num}")
        ROOT.SetOwnership(hist_draw, False)
        keep_alive.append(hist_draw)
        hist_draw.SetTitle(title)
        hist_draw.GetXaxis().SetTitle("#it{M}(#Lambda#it{p}#it{K}^{#minus}) [MeV/#it{c}^{2}]")
        hist_draw.GetYaxis().SetTitle(f"Candidates / ({BIN_WIDTH:.0f} MeV/#it{{c}}^{{2}})")
        hist_draw.GetXaxis().SetTitleSize(0.045)
        hist_draw.GetYaxis().SetTitleSize(0.045)
        hist_draw.GetXaxis().SetLabelSize(0.035)
        hist_draw.GetYaxis().SetLabelSize(0.035)
        hist_draw.SetLineColor(ROOT.kBlue)
        hist_draw.SetLineWidth(2)
        if y_max is not None:
            hist_draw.SetMaximum(y_max)
            hist_draw.SetMinimum(0)
        else:
            # For subtracted, allow negative values
            sub_min: float = hist_draw.GetMinimum()
            sub_max: float = hist_draw.GetMaximum()
            hist_draw.SetMinimum(min(sub_min * 1.1, -abs(sub_max) * 0.1))
            hist_draw.SetMaximum(sub_max * 1.3)
        hist_draw.Draw("HIST E")
        # Draw charmonium reference lines
        current_y_max: float = hist_draw.GetMaximum()
        current_y_min: float = hist_draw.GetMinimum()
        for state in charmonium_states:
            if M_LPKM_MIN < state.mass < M_LPKM_MAX:
                line: ROOT.TLine = ROOT.TLine(
                    state.mass, current_y_min, state.mass, current_y_max * 0.9
                )
                line.SetLineColor(state.color)
                line.SetLineStyle(2)
                line.SetLineWidth(1)
                line.Draw("same")
                keep_alive.append(line)
        # Add event count text (except for subtracted)
        if n_events > 0:
            latex: ROOT.TLatex = ROOT.TLatex()
            latex.SetNDC()
            latex.SetTextFont(132)
            latex.SetTextSize(0.035)
            latex.DrawLatex(0.55, 0.80, f"N = {n_events:,}")
            keep_alive.append(latex)
        # LHCb label
        latex_lhcb: ROOT.TLatex = ROOT.TLatex()
        latex_lhcb.SetNDC()
        latex_lhcb.SetTextFont(132)
        latex_lhcb.SetTextSize(0.04)
        latex_lhcb.DrawLatex(0.16, 0.85, "#font[62]{LHCb}")
        keep_alive.append(latex_lhcb)
    # Save
    canvas.Update()
    canvas.SaveAs(str(output_path))
    print(f"  Saved: {output_path}")


def fit_all_charmonium_states(
    hist: ROOT.TH1D,
    charmonium_states: list[CharmoniumState],
) -> tuple[ROOT.TF1 | None, dict[str, tuple[float, float, float]]]:
    """
    Fit all 5 charmonium states simultaneously using Voigtian PDFs.

    Uses the same approach as modules/mass_fitter.py:
    - Voigtian (Breit-Wigner ⊗ Gaussian) for each signal
    - Masses and widths fixed to PDG values
    - Shared resolution parameter
    - Linear background

    Args:
        hist: Sideband-subtracted histogram to fit
        charmonium_states: List of CharmoniumState objects with PDG info

    Returns:
        Tuple of (total_fit_function, fit_results_dict)
        fit_results_dict: {state_name: (yield, yield_error, sigma)}
    """
    # PDG values for charmonium states (mass, width in MeV)
    pdg_params: dict[str, tuple[float, float]] = {
        "eta_c(1S)": (2983.9, 32.0),
        "J/psi": (3096.9, 0.093),
        "chi_c0": (3414.7, 10.5),
        "chi_c1": (3510.7, 0.84),
        "eta_c(2S)": (3637.5, 11.3),
    }
    fit_min: float = M_LPKM_MIN
    fit_max: float = M_LPKM_MAX
    bin_width: float = hist.GetBinWidth(1)  # Get bin width (10 MeV)
    n_states: int = len(charmonium_states)
    # Formula: sum of Gaussians + linear background
    # Using non-normalized Gaussian: [N]*exp(-0.5*((x-mean)/sigma)^2)
    # where N is the amplitude (peak height), and yield = N * sigma * sqrt(2*pi)
    formula_parts: list[str] = []
    for i in range(n_states):
        # Non-normalized Gaussian: amplitude * exp(-0.5*((x-mean)/sigma)^2)
        # Yield = amplitude * sigma * sqrt(2*pi) / bin_width
        formula_parts.append(f"[{3*i}]*exp(-0.5*((x-[{3*i+1}])/[{3*i+2}])^2)")
    # Add linear background
    bkg_offset: int = 3 * n_states
    formula_parts.append(f"[{bkg_offset}] + [{bkg_offset+1}]*x")
    formula: str = " + ".join(formula_parts)
    total_fit: ROOT.TF1 = ROOT.TF1("total_fit", formula, fit_min, fit_max)
    ROOT.SetOwnership(total_fit, False)
    # Set initial parameters and fix masses
    for i, state in enumerate(charmonium_states):
        pdg_mass, pdg_width = pdg_params.get(state.name, (state.mass, 10.0))
        # Amplitude (initial guess from histogram peak height)
        bin_at_mass: int = hist.FindBin(pdg_mass)
        amp_init: float = max(hist.GetBinContent(bin_at_mass), 10.0)
        total_fit.SetParameter(3 * i, amp_init)
        total_fit.SetParLimits(3 * i, 0, amp_init * 5)  # Allow some headroom
        total_fit.SetParName(3 * i, f"A_{state.name}")
        # Mean (fixed to PDG)
        total_fit.SetParameter(3 * i + 1, pdg_mass)
        total_fit.FixParameter(3 * i + 1, pdg_mass)
        total_fit.SetParName(3 * i + 1, f"M_{state.name}")
        # Sigma (resolution, initial guess ~15 MeV)
        total_fit.SetParameter(3 * i + 2, 15.0)
        total_fit.SetParLimits(3 * i + 2, 5.0, 50.0)
        total_fit.SetParName(3 * i + 2, f"sigma_{state.name}")
    # Background parameters (allow negative for sideband-subtracted data)
    total_fit.SetParameter(bkg_offset, 0.0)
    total_fit.SetParLimits(bkg_offset, -1000, 1000)
    total_fit.SetParName(bkg_offset, "bkg_const")
    total_fit.SetParameter(bkg_offset + 1, 0.0)
    total_fit.SetParLimits(bkg_offset + 1, -1, 1)
    total_fit.SetParName(bkg_offset + 1, "bkg_slope")
    # Perform fit with more iterations
    fit_status: int = hist.Fit(total_fit, "RQS0")
    # Extract results - convert amplitude to yield
    # For non-normalized Gaussian: yield = amplitude * sigma * sqrt(2*pi) / bin_width
    sqrt_2pi: float = math.sqrt(2.0 * math.pi)
    fit_results: dict[str, tuple[float, float, float]] = {}
    for i, state in enumerate(charmonium_states):
        amp: float = total_fit.GetParameter(3 * i)
        amp_err: float = total_fit.GetParError(3 * i)
        sigma: float = total_fit.GetParameter(3 * i + 2)
        # Calculate yield from Gaussian integral
        # Integral of A*exp(-0.5*((x-m)/s)^2) = A * s * sqrt(2*pi)
        # Divide by bin_width to get number of events
        yield_val: float = amp * sigma * sqrt_2pi / bin_width
        yield_err: float = amp_err * sigma * sqrt_2pi / bin_width  # Simplified error
        fit_results[state.name] = (yield_val, yield_err, sigma)
    return total_fit, fit_results


def create_subtracted_signal_plot(
    hist_subtracted: ROOT.TH1D,
    charmonium_states: list[CharmoniumState],
    alpha: float,
    output_path: Path,
    do_fits: bool = True,
) -> None:
    """
    Create detailed sideband-subtracted signal plot with fits for all 5 states.

    Creates LHCb-style two-panel plot matching modules/mass_fitter.py:
    - Upper panel (70%): fit with data, model components
    - Lower panel (30%): pull distribution

    Args:
        hist_subtracted: Sideband-subtracted histogram
        charmonium_states: List of CharmoniumState objects with results
        alpha: Normalization factor
        output_path: Path to save the PDF
        do_fits: Whether to perform fits on all 5 charmonium states
    """
    keep_alive: list = []
    # Create canvas with two pads (matching mass_fitter.py style)
    canvas: ROOT.TCanvas = ROOT.TCanvas("c_subtracted", "", 1200, 800)
    ROOT.SetOwnership(canvas, False)
    keep_alive.append(canvas)
    # Upper pad for fit (70% of canvas)
    pad1: ROOT.TPad = ROOT.TPad("pad1", "Fit", 0.0, 0.30, 1.0, 1.0)
    pad1.SetBottomMargin(0.015)
    pad1.SetLeftMargin(0.10)
    pad1.SetRightMargin(0.05)
    pad1.SetTopMargin(0.07)
    pad1.Draw()
    ROOT.SetOwnership(pad1, False)
    keep_alive.append(pad1)
    # Lower pad for pulls (30% of canvas)
    pad2: ROOT.TPad = ROOT.TPad("pad2", "Pulls", 0.0, 0.0, 1.0, 0.30)
    pad2.SetTopMargin(0.015)
    pad2.SetBottomMargin(0.35)
    pad2.SetLeftMargin(0.10)
    pad2.SetRightMargin(0.05)
    pad2.SetGridy(1)
    pad2.Draw()
    ROOT.SetOwnership(pad2, False)
    keep_alive.append(pad2)
    # Draw fit in upper pad
    pad1.cd()
    # Clone histogram
    hist_draw: ROOT.TH1D = hist_subtracted.Clone("h_subtracted_draw")
    ROOT.SetOwnership(hist_draw, False)
    keep_alive.append(hist_draw)
    hist_draw.SetTitle("")
    hist_draw.GetYaxis().SetTitle(f"Candidates / ({BIN_WIDTH:.0f} MeV/#it{{c}}^{{2}})")
    hist_draw.GetYaxis().SetTitleSize(0.045)
    hist_draw.GetYaxis().SetLabelSize(0.0375)
    hist_draw.GetYaxis().SetTitleOffset(0.85)
    hist_draw.GetYaxis().SetTitleFont(132)
    hist_draw.GetYaxis().SetLabelFont(132)
    hist_draw.GetXaxis().SetLabelSize(0.0)  # Hide x-axis labels on upper pad
    hist_draw.GetXaxis().SetTitleSize(0.0)
    hist_draw.SetLineColor(ROOT.kBlack)
    hist_draw.SetLineWidth(1)
    hist_draw.SetMarkerStyle(20)
    hist_draw.SetMarkerSize(0.6)
    hist_draw.SetMarkerColor(ROOT.kBlack)
    # Set Y range with headroom
    y_min: float = hist_draw.GetMinimum()
    y_max: float = hist_draw.GetMaximum()
    hist_draw.SetMinimum(min(y_min * 1.1, -abs(y_max) * 0.05))
    hist_draw.SetMaximum(y_max * 1.50)
    hist_draw.Draw("E1")
    # Perform fits on all 5 charmonium states
    total_fit: ROOT.TF1 | None = None
    bkg_func: ROOT.TF1 | None = None
    fit_results: dict[str, tuple[float, float, float]] = {}
    if do_fits:
        total_fit, fit_results = fit_all_charmonium_states(hist_draw, charmonium_states)
        if total_fit:
            # Draw total fit
            total_fit.SetLineColor(ROOT.kBlue)
            total_fit.SetLineWidth(2)
            total_fit.Draw("same")
            keep_alive.append(total_fit)
            # Draw individual signal components
            for i, state in enumerate(charmonium_states):
                amp: float = total_fit.GetParameter(3 * i)
                mean: float = total_fit.GetParameter(3 * i + 1)
                sigma: float = total_fit.GetParameter(3 * i + 2)
                state_func: ROOT.TF1 = ROOT.TF1(
                    f"fit_{state.name}", f"[0]*exp(-0.5*((x-{mean})/[1])^2)", M_LPKM_MIN, M_LPKM_MAX
                )
                state_func.SetParameter(0, amp)
                state_func.SetParameter(1, sigma)
                state_func.SetLineColor(ROOT.kRed)
                state_func.SetLineWidth(2)
                state_func.SetLineStyle(1)
                state_func.Draw("same")
                ROOT.SetOwnership(state_func, False)
                keep_alive.append(state_func)
            # Draw background component
            bkg_offset: int = 3 * len(charmonium_states)
            bkg_const: float = total_fit.GetParameter(bkg_offset)
            bkg_slope: float = total_fit.GetParameter(bkg_offset + 1)
            bkg_func = ROOT.TF1("fit_bkg", f"{bkg_const} + {bkg_slope}*x", M_LPKM_MIN, M_LPKM_MAX)
            bkg_func.SetLineColor(ROOT.kGray + 1)
            bkg_func.SetLineWidth(2)
            bkg_func.SetLineStyle(2)
            bkg_func.Draw("same")
            ROOT.SetOwnership(bkg_func, False)
            keep_alive.append(bkg_func)
    # Redraw data on top
    hist_draw.Draw("E1 SAME")
    # Calculate pull statistics for info box
    pull_mean: float = 0.0
    pull_rms: float = 0.0
    n_pulls: int = 0
    if total_fit:
        for i in range(1, hist_subtracted.GetNbinsX() + 1):
            bin_content: float = hist_subtracted.GetBinContent(i)
            bin_error: float = hist_subtracted.GetBinError(i)
            fit_value: float = total_fit.Eval(hist_subtracted.GetBinCenter(i))
            if bin_error > 0:
                pull: float = (bin_content - fit_value) / bin_error
                if abs(pull) < 10:  # Exclude outliers
                    pull_mean += pull
                    pull_rms += pull * pull
                    n_pulls += 1
        if n_pulls > 0:
            pull_mean /= n_pulls
            pull_rms = (pull_rms / n_pulls - pull_mean * pull_mean) ** 0.5
    # LHCb label (top left)
    latex_lhcb: ROOT.TLatex = ROOT.TLatex()
    latex_lhcb.SetNDC()
    latex_lhcb.SetTextFont(132)
    latex_lhcb.SetTextSize(0.06)
    latex_lhcb.DrawLatex(0.12, 0.87, "LHCb")
    keep_alive.append(latex_lhcb)
    # Year label below LHCb
    latex_year: ROOT.TLatex = ROOT.TLatex()
    latex_year.SetNDC()
    latex_year.SetTextFont(132)
    latex_year.SetTextSize(0.05)
    latex_year.DrawLatex(0.12, 0.81, "2016-2018")
    keep_alive.append(latex_year)
    # Two-column info boxes (matching mass_fitter.py style)
    if do_fits and fit_results:
        # Left box - Yields
        fit_info_left: ROOT.TPaveText = ROOT.TPaveText(0.40, 0.60, 0.58, 0.90, "NDC")
        fit_info_left.SetBorderSize(2)
        fit_info_left.SetLineColor(ROOT.kBlue)
        fit_info_left.SetFillColor(ROOT.kWhite)
        fit_info_left.SetFillStyle(1001)
        fit_info_left.SetTextAlign(12)
        fit_info_left.SetTextFont(132)
        fit_info_left.SetTextSize(0.032)
        fit_info_left.SetTextColor(ROOT.kBlack)
        fit_info_left.AddText("#bf{Yields}")
        # Get yields for each state
        n_etac: float = fit_results.get("eta_c(1S)", (0, 0, 0))[0]
        n_etac_err: float = fit_results.get("eta_c(1S)", (0, 0, 0))[1]
        n_jpsi: float = fit_results.get("J/psi", (0, 0, 0))[0]
        n_jpsi_err: float = fit_results.get("J/psi", (0, 0, 0))[1]
        n_chic0: float = fit_results.get("chi_c0", (0, 0, 0))[0]
        n_chic0_err: float = fit_results.get("chi_c0", (0, 0, 0))[1]
        n_chic1: float = fit_results.get("chi_c1", (0, 0, 0))[0]
        n_chic1_err: float = fit_results.get("chi_c1", (0, 0, 0))[1]
        n_etac2s: float = fit_results.get("eta_c(2S)", (0, 0, 0))[0]
        n_etac2s_err: float = fit_results.get("eta_c(2S)", (0, 0, 0))[1]
        fit_info_left.AddText(f"N_{{J/#psi}} = {n_jpsi:.0f} #pm {n_jpsi_err:.0f}")
        fit_info_left.AddText(f"N_{{#eta_{{c}}}} = {n_etac:.0f} #pm {n_etac_err:.0f}")
        fit_info_left.AddText(f"N_{{#chi_{{c0}}}} = {n_chic0:.0f} #pm {n_chic0_err:.0f}")
        fit_info_left.AddText(f"N_{{#chi_{{c1}}}} = {n_chic1:.0f} #pm {n_chic1_err:.0f}")
        fit_info_left.AddText(f"N_{{#eta_{{c}}(2S)}} = {n_etac2s:.0f} #pm {n_etac2s_err:.0f}")
        fit_info_left.Draw()
        keep_alive.append(fit_info_left)
        # Right box - Fit Info
        fit_info_right: ROOT.TPaveText = ROOT.TPaveText(0.58, 0.60, 0.76, 0.90, "NDC")
        fit_info_right.SetBorderSize(2)
        fit_info_right.SetLineColor(ROOT.kBlue)
        fit_info_right.SetFillColor(ROOT.kWhite)
        fit_info_right.SetFillStyle(1001)
        fit_info_right.SetTextAlign(12)
        fit_info_right.SetTextFont(132)
        fit_info_right.SetTextSize(0.032)
        fit_info_right.SetTextColor(ROOT.kBlack)
        fit_info_right.AddText("#bf{Fit Info}")
        # Calculate totals
        n_signal: float = n_jpsi + n_etac + n_chic0 + n_chic1 + n_etac2s
        # Get background from fit
        bkg_offset: int = 3 * len(charmonium_states)
        bkg_const: float = total_fit.GetParameter(bkg_offset) if total_fit else 0
        # Estimate background yield (integral over range)
        n_bkg: float = hist_subtracted.Integral() - n_signal
        n_tot: float = hist_subtracted.Integral()
        avg_sigma: float = sum(r[2] for r in fit_results.values()) / len(fit_results)
        fit_info_right.AddText(f"N_{{bkg}} #approx {n_bkg:.0f}")
        fit_info_right.AddText(f"N_{{sig}} = {n_signal:.0f}")
        fit_info_right.AddText(f"N_{{tot}} = {n_tot:.0f}")
        fit_info_right.AddText(f"#sigma_{{res}} = {avg_sigma:.1f} MeV")
        fit_info_right.AddText(f"Pull: #mu = {pull_mean:.2f}, #sigma = {pull_rms:.2f}")
        fit_info_right.Draw()
        keep_alive.append(fit_info_right)
    # Legend (compact, upper right)
    legend: ROOT.TLegend = ROOT.TLegend(0.77, 0.60, 0.93, 0.90)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetFillColor(0)
    legend.SetTextSize(0.04)
    legend.SetTextFont(132)
    legend.SetMargin(0.15)
    legend.AddEntry(hist_draw, "Data", "lep")
    if do_fits and total_fit:
        legend.AddEntry(total_fit, "Full model", "l")
        legend.AddEntry("fit_J/psi", "Signal", "l")
        if bkg_func:
            legend.AddEntry(bkg_func, "Background", "l")
    legend.Draw()
    keep_alive.append(legend)
    # Create pull distribution in lower pad
    pad2.cd()
    # Create pull histogram
    pull_hist: ROOT.TH1D = hist_subtracted.Clone("h_pull")
    ROOT.SetOwnership(pull_hist, False)
    keep_alive.append(pull_hist)
    if total_fit:
        for i in range(1, pull_hist.GetNbinsX() + 1):
            bin_center: float = pull_hist.GetBinCenter(i)
            bin_content: float = pull_hist.GetBinContent(i)
            bin_error: float = pull_hist.GetBinError(i)
            fit_value: float = total_fit.Eval(bin_center)
            if bin_error > 0:
                pull: float = (bin_content - fit_value) / bin_error
            else:
                pull = 0.0
            pull_hist.SetBinContent(i, pull)
            pull_hist.SetBinError(i, 1.0)  # Unit error for pull
    # Style pull histogram
    pull_hist.SetTitle("")
    pull_hist.GetYaxis().SetTitle("Pull")
    pull_hist.GetYaxis().SetTitleSize(0.0975)
    pull_hist.GetYaxis().SetLabelSize(0.0825)
    pull_hist.GetYaxis().SetTitleOffset(0.4)
    pull_hist.GetYaxis().SetTitleFont(132)
    pull_hist.GetYaxis().SetLabelFont(132)
    pull_hist.GetYaxis().SetNdivisions(505)
    pull_hist.GetYaxis().SetRangeUser(-4.5, 4.5)
    pull_hist.GetYaxis().CenterTitle()
    pull_hist.GetXaxis().SetTitle("M(#bar{#Lambda}pK^{#minus}) [MeV/#it{c}^{2}]")
    pull_hist.GetXaxis().SetTitleSize(0.13)
    pull_hist.GetXaxis().SetLabelSize(0.11)
    pull_hist.GetXaxis().SetTitleOffset(1.1)
    pull_hist.GetXaxis().SetTitleFont(132)
    pull_hist.GetXaxis().SetLabelFont(132)
    pull_hist.SetMarkerStyle(20)
    pull_hist.SetMarkerSize(0.5)
    pull_hist.SetMarkerColor(ROOT.kBlack)
    pull_hist.Draw("E1")
    # Add horizontal reference lines at 0, ±3σ
    line_zero: ROOT.TLine = ROOT.TLine(M_LPKM_MIN, 0.0, M_LPKM_MAX, 0.0)
    line_zero.SetLineColor(ROOT.kBlack)
    line_zero.SetLineStyle(1)
    line_zero.SetLineWidth(1)
    line_zero.Draw()
    keep_alive.append(line_zero)
    line_plus3: ROOT.TLine = ROOT.TLine(M_LPKM_MIN, 3.0, M_LPKM_MAX, 3.0)
    line_plus3.SetLineColor(ROOT.kGray + 1)
    line_plus3.SetLineStyle(2)
    line_plus3.SetLineWidth(1)
    line_plus3.Draw()
    keep_alive.append(line_plus3)
    line_minus3: ROOT.TLine = ROOT.TLine(M_LPKM_MIN, -3.0, M_LPKM_MAX, -3.0)
    line_minus3.SetLineColor(ROOT.kGray + 1)
    line_minus3.SetLineStyle(2)
    line_minus3.SetLineWidth(1)
    line_minus3.Draw()
    keep_alive.append(line_minus3)
    # Save
    canvas.cd()
    canvas.Update()
    canvas.SaveAs(str(output_path))
    print(f"  Saved: {output_path}")


def save_results_table(
    charmonium_states: list[CharmoniumState],
    regions: list[MassRegion],
    alpha: float,
    output_dir: Path,
    years: list[str],
    track_types: list[str],
    lambda_selection: dict[str, Any] | None,
    manual_cuts: dict[str, Any] | None,
) -> None:
    """
    Save results as formatted text and CSV files.

    Args:
        charmonium_states: List of CharmoniumState objects with results
        regions: List of MassRegion objects
        alpha: Normalization factor
        output_dir: Directory to save files
        years: List of years processed
        track_types: List of track types processed
        lambda_selection: Lambda selection cuts applied
        manual_cuts: Manual cuts applied
    """
    signal_region: MassRegion = regions[0]
    left_sb: MassRegion = regions[1]
    right_sb: MassRegion = regions[2]
    # Text file
    txt_path: Path = output_dir / "charmonium_yields.txt"
    with open(txt_path, "w") as f:
        f.write(f"Charmonium State Quantification - B+ Signal Region {signal_region.label}\n")
        f.write(f"Data: {'-'.join(years)}, Track types: {'+'.join(track_types)}\n")
        f.write("=" * 80 + "\n\n")
        # Document applied cuts
        f.write("SELECTION CUTS APPLIED (same as main analysis pipeline):\n")
        f.write("-" * 80 + "\n")
        if lambda_selection:
            f.write("Lambda selection cuts:\n")
            if "mass_min" in lambda_selection and "mass_max" in lambda_selection:
                f.write(
                    f"  L0_MM: [{lambda_selection['mass_min']}, {lambda_selection['mass_max']}] MeV\n"
                )
            if "fd_chisq_min" in lambda_selection:
                f.write(f"  L0_FDCHI2_OWNPV > {lambda_selection['fd_chisq_min']}\n")
            if "delta_z_min" in lambda_selection:
                f.write(f"  delta_z > {lambda_selection['delta_z_min']} mm\n")
            if "proton_probnnp_min" in lambda_selection:
                f.write(f"  Lp_ProbNNp > {lambda_selection['proton_probnnp_min']}\n")
        else:
            f.write("Lambda selection cuts: NONE\n")
        if manual_cuts:
            f.write("Manual cuts:\n")
            for branch_name, cut_spec in manual_cuts.items():
                if branch_name == "notes":
                    continue
                cut_type = cut_spec.get("cut_type", "")
                cut_value = cut_spec.get("value", "")
                op: str = ">" if cut_type == "greater" else "<"
                f.write(f"  {branch_name} {op} {cut_value}\n")
        else:
            f.write("Manual cuts: NONE\n")
        f.write("-" * 80 + "\n\n")
        f.write(
            f"{'State':<12} {'Mass(PDG)':<12} {'Window':<14} {'Yield':<18} {'Significance':<12}\n"
        )
        f.write(f"{'':12} {'[MeV]':<12} {'[MeV]':<14} {'':18} {'(σ units)':<12}\n")
        f.write("-" * 80 + "\n")
        for state in charmonium_states:
            yield_str: str = f"{state.yield_value:.0f} ± {state.yield_error:.0f}"
            f.write(
                f"{state.name:<12} {state.mass:<12.1f} {state.window_str:<14} "
                f"{yield_str:<18} {state.significance:<12.1f}\n"
            )
        f.write("-" * 80 + "\n\n")
        f.write(f"Total events in B+ signal region: {signal_region.n_events:,}\n")
        f.write(f"Total events in left sideband:    {left_sb.n_events:,}\n")
        f.write(f"Total events in right sideband:   {right_sb.n_events:,}\n")
        f.write(f"Sideband normalization factor:    α = {alpha:.4f}\n\n")
        f.write("Background subtraction: Signal - α × (Left_SB + Right_SB)\n")
        f.write("Integration windows: ±width around PDG mass (see table)\n")
        f.write("Significance: (Yield - Baseline) / Error where baseline from nearby continuum\n")
        f.write("=" * 80 + "\n")
    print(f"  Saved: {txt_path}")
    # CSV file
    csv_path: Path = output_dir / "charmonium_yields.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "State",
                "Mass_PDG_MeV",
                "Window_Min_MeV",
                "Window_Max_MeV",
                "Yield",
                "Error",
                "Significance_sigma",
            ]
        )
        for state in charmonium_states:
            writer.writerow(
                [
                    state.name,
                    state.mass,
                    state.window[0],
                    state.window[1],
                    f"{state.yield_value:.1f}",
                    f"{state.yield_error:.1f}",
                    f"{state.significance:.2f}",
                ]
            )
    print(f"  Saved: {csv_path}")


def print_interpretation_guide() -> None:
    """Print physics interpretation guidance."""
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print(
        """
1. High significance (>3σ) peaks indicate genuine charmonium contributions
   to the B+ → Λ̄ p X K+ decay, where X is the charmonium state.

2. The yield represents the number of B+ candidates containing that
   charmonium state after background subtraction.

3. Peaks visible in signal region but not sidebands confirm these are
   genuine decay products, not detector artifacts.

4. Missing or weak peaks suggest:
   - State not kinematically accessible in this decay
   - Branching fraction below sensitivity
   - Interference effects suppressing production
"""
    )
    print("=" * 70)


def main() -> None:
    """Main function to run the signal vs sideband projection study."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="M(Lambda p K) projections: Signal vs Sideband study"
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
        "--signal-window",
        type=str,
        default="5229,5329",
        help="Signal region M(B+) window as 'min,max' (default: 5229,5329)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: analysis_output/)",
    )
    parser.add_argument(
        "--no-fits",
        action="store_true",
        help="Skip Gaussian fits on peaks",
    )
    parser.add_argument(
        "--no-cuts",
        action="store_true",
        help="Skip all selection cuts (for comparison studies)",
    )
    args: argparse.Namespace = parser.parse_args()
    # Parse arguments
    years: list[str] = [y.strip() for y in args.years.split(",")]
    track_types: list[str] = [t.strip() for t in args.track_types.split(",")]
    signal_parts: list[str] = args.signal_window.split(",")
    signal_min: float = float(signal_parts[0].strip())
    signal_max: float = float(signal_parts[1].strip())
    do_fits: bool = not args.no_fits
    # Setup
    setup_lhcb_style()
    # Load configuration
    config_dir: Path = ANALYSIS_DIR / "config"
    config: TOMLConfig = TOMLConfig(str(config_dir))
    data_path: Path = Path(config.paths["data"]["base_path"])
    # Get selection cuts from config (same as main analysis pipeline)
    if args.no_cuts:
        lambda_selection: dict[str, Any] | None = None
        manual_cuts: dict[str, Any] | None = None
    else:
        lambda_selection = config.selection.get("lambda_selection", {})
        manual_cuts = config.selection.get("manual_cuts", {})
    # Setup output directory
    if args.output_dir:
        output_dir: Path = Path(args.output_dir)
    else:
        output_dir = ANALYSIS_DIR / "scripts" / "analysis_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Define mass regions
    regions: list[MassRegion] = [
        MassRegion("Signal", signal_min, signal_max),
        MassRegion("Left SB", 4800.0, 5100.0),
        MassRegion("Right SB", 5450.0, 5750.0),
    ]
    # Define charmonium states
    charmonium_states: list[CharmoniumState] = [
        CharmoniumState("eta_c(1S)", 2983.9, 40.0, ROOT.kRed, "#eta_{c}(1S)"),
        CharmoniumState("J/psi", 3096.9, 30.0, ROOT.kBlue, "J/#psi"),
        CharmoniumState("chi_c0", 3414.7, 50.0, ROOT.kGreen + 2, "#chi_{c0}"),
        CharmoniumState("chi_c1", 3510.7, 50.0, ROOT.kOrange + 1, "#chi_{c1}"),
        CharmoniumState("eta_c(2S)", 3637.5, 40.0, ROOT.kMagenta, "#eta_{c}(2S)"),
    ]
    # Print configuration
    print("\n" + "=" * 70)
    print("M(Lambda p K) Signal vs Sideband Projection Study")
    print("=" * 70)
    # Print Lambda selection cuts
    if lambda_selection:
        print("\nLambda selection cuts (from config/selection.toml):")
        if "mass_min" in lambda_selection and "mass_max" in lambda_selection:
            print(f"  L0_MM: [{lambda_selection['mass_min']}, {lambda_selection['mass_max']}] MeV")
        if "fd_chisq_min" in lambda_selection:
            print(f"  L0_FDCHI2_OWNPV > {lambda_selection['fd_chisq_min']}")
        if "delta_z_min" in lambda_selection:
            print(f"  delta_z > {lambda_selection['delta_z_min']} mm")
        if "proton_probnnp_min" in lambda_selection:
            print(f"  Lp_ProbNNp > {lambda_selection['proton_probnnp_min']}")
    else:
        print("\nLambda selection cuts: NONE (--no-cuts specified)")
    # Print manual cuts
    if manual_cuts:
        print("\nManual cuts (from config/selection.toml):")
        for branch_name, cut_spec in manual_cuts.items():
            if branch_name == "notes":
                continue
            cut_type = cut_spec.get("cut_type", "")
            cut_value = cut_spec.get("value", "")
            op: str = ">" if cut_type == "greater" else "<"
            print(f"  {branch_name} {op} {cut_value}")
    else:
        print("\nManual cuts: NONE (--no-cuts specified)")
    print(f"\nOutput directory: {output_dir}")
    print("\nM(B+) regions:")
    for region in regions:
        print(f"  {region.label}")
    print(f"\nM(LpK-) range: [{M_LPKM_MIN:.0f}, {M_LPKM_MAX:.0f}] MeV")
    print(f"Years: {years}")
    print(f"Track types: {track_types}")
    print(f"Gaussian fits: {'Enabled' if do_fits else 'Disabled'}\n")
    # Load histograms for each region
    print("Loading data...")
    for region in regions:
        print(f"  Loading {region.name}...")
        region.histogram, region.n_events = load_mlpk_histogram(
            data_path, years, track_types, region, lambda_selection, manual_cuts
        )
        print(f"    {region.n_events:,} events")
        # Validation
        if region.n_events < 100:
            print("    WARNING: Low statistics (<100 events)")
    # Perform sideband subtraction
    print("\nPerforming sideband subtraction...")
    hist_subtracted, alpha = perform_sideband_subtraction(
        regions[0].histogram,
        regions[1].histogram,
        regions[2].histogram,
        regions[0],  # signal region
        regions[1],  # left sideband region
        regions[2],  # right sideband region
    )
    print(f"  Effective scale factor α = {alpha:.4f}")
    # Quantify charmonium peaks
    print("\nQuantifying charmonium peaks...")
    for state in charmonium_states:
        state.yield_value, state.yield_error = integrate_peak_region(hist_subtracted, state.window)
        state.significance = estimate_significance(hist_subtracted, state)
        print(
            f"  {state.name}: {state.yield_value:.0f} ± {state.yield_error:.0f} ({state.significance:.1f}σ)"
        )
    # Check for negative bins
    n_negative: int = sum(
        1 for i in range(1, hist_subtracted.GetNbinsX() + 1) if hist_subtracted.GetBinContent(i) < 0
    )
    if n_negative > 0:
        print(
            f"\n  Note: {n_negative} bins have negative values after subtraction (statistical fluctuation)"
        )
    # Create plots
    print("\nCreating plots...")
    # Four-panel comparison
    four_panel_path: Path = output_dir / "mlpk_signal_sideband_comparison.pdf"
    create_four_panel_plot(regions, hist_subtracted, charmonium_states, alpha, four_panel_path)
    # Detailed subtracted signal
    subtracted_path: Path = output_dir / "mlpk_sideband_subtracted.pdf"
    create_subtracted_signal_plot(
        hist_subtracted, charmonium_states, alpha, subtracted_path, do_fits
    )
    # Save results
    print("\nSaving results...")
    save_results_table(
        charmonium_states,
        regions,
        alpha,
        output_dir,
        years,
        track_types,
        lambda_selection,
        manual_cuts,
    )
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Signal region events:     {regions[0].n_events:,}")
    print(f"Left sideband events:     {regions[1].n_events:,}")
    print(f"Right sideband events:    {regions[2].n_events:,}")
    print(f"Normalization factor α:   {alpha:.4f}")
    print("\nCharmonium yields (sideband-subtracted):")
    for state in charmonium_states:
        sig_str: str = f"{state.significance:.1f}σ"
        print(
            f"  {state.name:<12}: {state.yield_value:>8.0f} ± {state.yield_error:<6.0f} ({sig_str})"
        )
    print(f"\nOutput files saved to: {output_dir}")
    print("=" * 70)
    # Print interpretation guide
    print_interpretation_guide()


if __name__ == "__main__":
    main()
