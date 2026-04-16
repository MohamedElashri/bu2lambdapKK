"""
Signal MC mass fit plots for the B+ → Λ̄pK⁻K⁺ analysis note.

Produces individual PDFs:
  figs/LambdaLL/fit/fit_ToJpsi_Run2MDU_MC.pdf
  figs/LambdaDD/fit/fit_ToJpsi_Run2MDU_MC.pdf
  figs/LambdaLL/fit/fit_ToEtac_Run2MDU_MC.pdf
  figs/LambdaDD/fit/fit_ToEtac_Run2MDU_MC.pdf
  figs/LambdaLL/fit/fit_ToChic0_Run2MDU_MC.pdf
  figs/LambdaDD/fit/fit_ToChic0_Run2MDU_MC.pdf
  figs/LambdaLL/fit/fit_ToChic1_Run2MDU_MC.pdf
  figs/LambdaDD/fit/fit_ToChic1_Run2MDU_MC.pdf

Uses Gaussian fit to B+ corrected mass from signal MC.
Plotting matches reference style: MC as error bars, fit as smooth line.

Run from analysis/ directory:
    uv run python presentation/ana_note_plots/scripts/plot_mcfits.py
"""

import logging
import sys
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot
from scipy.optimize import curve_fit

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPTS_DIR.resolve().parents[3]  # analysis/
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.resolve().parents[2]))  # analysis/ for modules.*

from modules.plot_utils import COLORS, figs_path, make_formatter, save_fig, setup_style
from modules.presentation_config import (
    HLT1_TOS_KEYS,
    HLT2_TOS_KEYS,
    MC_L0_TIS_KEYS,
    get_presentation_config,
)

PRESENTATION = get_presentation_config()
MC_BASE = PRESENTATION.mc_base
M_LAMBDA_PDG = PRESENTATION.lambda_mass_pdg
YEARS = PRESENTATION.year_suffixes
MAGNETS = PRESENTATION.magnets
LAMBDA_MIN = PRESENTATION.lambda_mass_min
LAMBDA_MAX = PRESENTATION.lambda_mass_max

# State name → MC directory name + plot label
STATES = {
    "Jpsi": {"dir": "Jpsi", "label": r"$B^+\to J/\psi(\to \bar{\Lambda}pK^-)K^+$"},
    "Etac": {"dir": "etac", "label": r"$B^+\to \eta_c(\to \bar{\Lambda}pK^-)K^+$"},
    "Chic0": {"dir": "chic0", "label": r"$B^+\to \chi_{c0}(\to \bar{\Lambda}pK^-)K^+$"},
    "Chic1": {"dir": "chic1", "label": r"$B^+\to \chi_{c1}(\to \bar{\Lambda}pK^-)K^+$"},
}

# B+ corrected mass fit range
FIT_RANGE = (5150, 5450)
FIT_BINS = 60

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

setup_style()
_fmt = make_formatter()


# ════════════════════════════════════════════════════════════════════════════════
# GAUSSIAN FIT
# ════════════════════════════════════════════════════════════════════════════════


def _gaussian(x, N, mu, sigma):
    return N * np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _fit_gaussian(bin_centers, bin_contents, bin_errors):
    """Fit Gaussian to histogram. Returns (popt, pcov) or None on failure."""
    # Initial guess: peak at B+ mass
    N0 = np.sum(bin_contents) * (bin_centers[1] - bin_centers[0])
    mu0 = 5279.3
    sigma0 = 20.0

    # Only fit bins with content > 0
    mask = bin_contents > 0
    if mask.sum() < 4:
        return None, None

    sigma_err = np.where(bin_errors > 0, bin_errors, 1.0)
    try:
        popt, pcov = curve_fit(
            _gaussian,
            bin_centers[mask],
            bin_contents[mask],
            p0=[N0, mu0, sigma0],
            sigma=sigma_err[mask],
            absolute_sigma=True,
            bounds=([0, 5200, 5], [1e8, 5350, 80]),
        )
        return popt, pcov
    except Exception as e:
        log.warning(f"  Gaussian fit failed: {e}")
        return None, None


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ════════════════════════════════════════════════════════════════════════════════


def _load_mc(path: Path, cat: str) -> np.ndarray:
    """
    Load MC and return Bu_MM_corrected after trigger + Lambda mass cuts.
    """
    want = [
        "Bu_MM",
        "L0_MM",
        "Bu_DTFL0_M",
        # Trigger
        "Bu_L0Global_TIS",
        "Bu_L0HadronDecision_TIS",
        "Bu_Hlt1TrackMVADecision_TOS",
        "Bu_Hlt1TwoTrackMVADecision_TOS",
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
    ]
    tree = uproot.open(path)[f"B2L0barPKpKm_{cat}/DecayTree"]
    avail = [b for b in want if b in tree.keys()]
    ev = tree.arrays(avail, library="ak")

    # Trigger
    l0_keys = MC_L0_TIS_KEYS
    hlt1_keys = HLT1_TOS_KEYS
    hlt2_keys = HLT2_TOS_KEYS
    n = len(ev["Bu_MM"])

    def _or_mask(keys):
        m = ak.zeros_like(ev["Bu_MM"], dtype=bool)
        for k in keys:
            if k in avail:
                m = m | (ev[k] > 0)
        return m

    ml0 = _or_mask(l0_keys)
    mhlt1 = _or_mask(hlt1_keys)
    mhlt2 = _or_mask(hlt2_keys)
    if not any(k in avail for k in l0_keys):
        ml0 = ak.ones_like(ev["Bu_MM"], dtype=bool)
    if not any(k in avail for k in hlt1_keys):
        mhlt1 = ak.ones_like(ev["Bu_MM"], dtype=bool)
    if not any(k in avail for k in hlt2_keys):
        mhlt2 = ak.ones_like(ev["Bu_MM"], dtype=bool)

    mask = ml0 & mhlt1 & mhlt2

    # Lambda mass window
    mask = mask & (ev["L0_MM"] > LAMBDA_MIN) & (ev["L0_MM"] < LAMBDA_MAX)

    ev = ev[mask]

    # Corrected B+ mass
    Bu_corr = ak.to_numpy(ev["Bu_MM"]) - ak.to_numpy(ev["L0_MM"]) + M_LAMBDA_PDG
    return Bu_corr


def _collect_mc(state_dir: str, cat: str) -> np.ndarray:
    """Concatenate MC from all years and magnets for a given state."""
    arrs = []
    for yr in YEARS:
        for mag in MAGNETS:
            p = MC_BASE / state_dir / f"{state_dir}_{yr}_{mag}.root"
            if not p.exists():
                # Try capitalized
                p = MC_BASE / state_dir.capitalize() / f"{state_dir.capitalize()}_{yr}_{mag}.root"
            if not p.exists():
                continue
            try:
                arr = _load_mc(p, cat)
                arrs.append(arr)
                log.info(f"  {p.name} [{cat}]: {len(arr)} events")
            except Exception as e:
                log.warning(f"  Skip {p}: {e}")
    return np.concatenate(arrs) if arrs else np.array([])


# ════════════════════════════════════════════════════════════════════════════════
# PLOT
# ════════════════════════════════════════════════════════════════════════════════


def plot_mc_fit(cat: str, state_key: str, state_info: dict):
    """
    Produce signal MC fit PDF for one state × category.
    """
    state_dir = state_info["dir"]
    label_dec = state_info["label"]

    log.info(f"  === {state_key} {cat} ===")
    mc = _collect_mc(state_dir, cat)
    if len(mc) == 0:
        log.warning(f"  No MC events for {state_key} {cat}, skipping")
        return

    # Histogram
    lo, hi = FIT_RANGE
    bins = FIT_BINS
    width = (hi - lo) / bins
    histstyle = {"range": FIT_RANGE, "bins": bins, "density": False}

    h_mc, edges = np.histogram(mc, **histstyle)
    bin_centers = (edges[1:] + edges[:-1]) / 2
    bin_errors = np.sqrt(np.abs(h_mc) + 1)

    # Fit
    popt, pcov = _fit_gaussian(bin_centers, h_mc, bin_errors)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # MC as error bars (reference style: data points with error bars)
    ax.errorbar(
        x=bin_centers,
        y=h_mc,
        xerr=width / 2,
        yerr=bin_errors,
        label=r"Signal MC",
        ecolor="black",
        mfc="black",
        color="black",
        elinewidth=2,
        markersize=6,
        marker="o",
        fmt=" ",
    )

    # Fit curve
    x_smooth = np.linspace(lo, hi, 500)
    if popt is not None:
        N_fit, mu_fit, sigma_fit = popt
        y_smooth = _gaussian(x_smooth, N_fit, mu_fit, sigma_fit) * width
        ax.plot(
            x_smooth,
            y_smooth,
            color=COLORS[0],
            linewidth=3,
            label=rf"Gaussian fit: $\mu={mu_fit:.1f}$, $\sigma={sigma_fit:.1f}$ MeV$/c^2$",
        )
        log.info(f"  Fit: mu={mu_fit:.2f} sigma={sigma_fit:.2f} N={N_fit:.1f}")

    # Axis labels
    ax.set_xlabel(r"$M(\bar{\Lambda}pKK)_{\rm corr}$ [MeV/$c^2$]")
    ax.set_ylabel(rf"Candidates / ({width:.0f} MeV/$c^2$)")
    ax.set_xlim(FIT_RANGE)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(_fmt)
    ax.legend(loc="upper left", frameon=False)

    # Category label
    ax.text(
        0.97,
        0.97,
        rf"$\Lambda_{{{cat}}}$, Run 2",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=10,
    )

    filename = f"fit_To{state_key}_Run2MDU_MC.pdf"
    out = figs_path(cat, "fit", filename)
    save_fig(fig, out)
    log.info(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════


def main():
    for cat in ("LL", "DD"):
        log.info(f"=== Category: Lambda{cat} ===")
        for state_key, state_info in STATES.items():
            plot_mc_fit(cat, state_key, state_info)
    log.info("=== Done. ===")


if __name__ == "__main__":
    main()
