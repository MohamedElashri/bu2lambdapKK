"""
Normalization channel plots for the B+ → Λ̄pK⁻K⁺ analysis note.

Produces:
  figs/LambdaLL/fit/fit_to_sideband.pdf             — exponential fit to B+ mass sidebands
  figs/LambdaDD/fit/fit_to_sideband.pdf
  figs/LambdaLL/fit/fit_ToNormal_Run2MDU_MC_mB.pdf  — B+ mass Gaussian fit (J/ψ MC)
  figs/LambdaDD/fit/fit_ToNormal_Run2MDU_MC_mB.pdf

Run from analysis/ directory:
    uv run python studies/ana_note_plots/scripts/plot_normchannel.py
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

from modules.plot_utils import BINNING, COLORS, figs_path, save_fig, setup_style

DATA_BASE = Path("/share/lazy/Mohamed/Bu2LambdaPPP/files/data")
MC_BASE = Path("/share/lazy/Mohamed/Bu2LambdaPPP/files/mc")
JPSI_MC = MC_BASE / "Jpsi"

# Sideband definitions (MeV)
SB_LO = (5150, 5230)
SB_HI = (5330, 5410)

M_BPLUS = 5279.6
YEARS = ["16", "17", "18"]
MAGNETS = ["MD", "MU"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

setup_style()


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ════════════════════════════════════════════════════════════════════════════════


def _open_tree(path: Path, cat: str):
    return uproot.open(path)[f"B2L0barPKpKm_{cat}/DecayTree"]


def _load_data(path: Path, cat: str) -> np.ndarray:
    """Load Bu_DTFL0_M after trigger + Lambda mass window."""
    want = [
        "L0_MM",
        "Bu_L0GlobalDecision_TIS",
        "Bu_L0PhysDecision_TIS",
        "Bu_L0HadronDecision_TIS",
        "Bu_Hlt1TrackMVADecision_TOS",
        "Bu_Hlt1TwoTrackMVADecision_TOS",
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
        "Bu_DTFL0_M",
    ]
    tree = _open_tree(path, cat)
    avail = [b for b in want if b in tree.keys()]
    ev = tree.arrays(avail, library="np")
    n = len(ev["L0_MM"])

    l0_keys = ["Bu_L0GlobalDecision_TIS", "Bu_L0PhysDecision_TIS", "Bu_L0HadronDecision_TIS"]
    hlt1_keys = ["Bu_Hlt1TrackMVADecision_TOS", "Bu_Hlt1TwoTrackMVADecision_TOS"]
    hlt2_keys = [
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
    ]

    def _or(keys):
        m = np.zeros(n, dtype=bool)
        for k in keys:
            if k in avail:
                m = m | (ev[k] > 0)
        return m if any(k in avail for k in keys) else np.ones(n, dtype=bool)

    mask = _or(l0_keys) & _or(hlt1_keys) & _or(hlt2_keys)
    mask = mask & (ev["L0_MM"] > 1108) & (ev["L0_MM"] < 1126)
    return ev["Bu_DTFL0_M"][mask]


def _collect_data(cat: str) -> np.ndarray:
    """Concatenate Bu_DTFL0_M from all years and magnets."""
    arrs = []
    for yr in YEARS:
        for mag in MAGNETS:
            p = DATA_BASE / f"dataBu2L0barPHH_{yr}{mag}.root"
            if not p.exists():
                continue
            try:
                a = _load_data(p, cat)
                arrs.append(a)
                log.info(f"  Loaded {p.name} [{cat}]: {len(a)} events")
            except Exception as e:
                log.warning(f"  Skip {p.name} [{cat}]: {e}")
    return np.concatenate(arrs) if arrs else np.array([])


def _load_mc_jpsi(path: Path, cat: str) -> np.ndarray:
    """
    Load J/ψ MC Bu_DTFL0_M after trigger + Lambda mass window.
    Bu_DTFL0_M is jagged in MC (multiple DTF hypotheses) — use ak.firsts.
    """
    want = [
        "L0_MM",
        "Bu_L0GlobalDecision_TIS",
        "Bu_L0PhysDecision_TIS",
        "Bu_L0HadronDecision_TIS",
        "Bu_Hlt1TrackMVADecision_TOS",
        "Bu_Hlt1TwoTrackMVADecision_TOS",
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
    ]
    tree = _open_tree(path, cat)
    avail = [b for b in want if b in tree.keys()]
    ev = tree.arrays(avail, library="np")
    n = len(ev["L0_MM"])

    raw = tree.arrays(["Bu_DTFL0_M"], library="ak")["Bu_DTFL0_M"]
    dtf_mass = ak.to_numpy(ak.firsts(raw))

    l0_keys = ["Bu_L0GlobalDecision_TIS", "Bu_L0PhysDecision_TIS", "Bu_L0HadronDecision_TIS"]
    hlt1_keys = ["Bu_Hlt1TrackMVADecision_TOS", "Bu_Hlt1TwoTrackMVADecision_TOS"]
    hlt2_keys = [
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
    ]

    def _or(keys):
        m = np.zeros(n, dtype=bool)
        for k in keys:
            if k in avail:
                m = m | (ev[k] > 0)
        return m if any(k in avail for k in keys) else np.ones(n, dtype=bool)

    mask = _or(l0_keys) & _or(hlt1_keys) & _or(hlt2_keys)
    mask = mask & (ev["L0_MM"] > 1108) & (ev["L0_MM"] < 1126)
    mask = mask & np.isfinite(dtf_mass) & (dtf_mass > 4000)
    return dtf_mass[mask]


def _collect_mc_jpsi(cat: str) -> np.ndarray:
    """Concatenate J/ψ MC from all years and magnets (Run 2 combined)."""
    arrs = []
    for yr in YEARS:
        for mag in MAGNETS:
            p = JPSI_MC / f"Jpsi_{yr}_{mag}.root"
            if not p.exists():
                continue
            try:
                a = _load_mc_jpsi(p, cat)
                arrs.append(a)
                log.info(f"  MC {p.name} [{cat}]: {len(a)} events")
            except Exception as e:
                log.warning(f"  Skip MC {p.name} [{cat}]: {e}")
    return np.concatenate(arrs) if arrs else np.array([])


# ════════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════


def _plot_fit_errorbars(ax, h, edges, label, color="black"):
    """Plot mass histogram as error bars, skipping empty bins."""
    centers = (edges[1:] + edges[:-1]) / 2
    err = np.sqrt(h + 1)
    mask = h > 0
    ax.errorbar(
        centers[mask],
        h[mask],
        yerr=err[mask],
        label=label,
        ecolor=color,
        mfc=color,
        color=color,
        elinewidth=1.5,
        markersize=3,
        marker="o",
        fmt=" ",
    )


def plot_fit_to_sideband(cat: str, bu: np.ndarray):
    """
    fit/fit_to_sideband.pdf — exponential fit to B+ mass sidebands.

    Only plots data in the actual sideband regions (no empty signal-region bins).
    Fits with an exponential to validate the background shape model.
    """
    log.info(f"  fit_to_sideband [{cat}]")

    sel_lo = (bu >= SB_LO[0]) & (bu <= SB_LO[1])
    sel_hi = (bu >= SB_HI[0]) & (bu <= SB_HI[1])
    bu_lo = bu[sel_lo]
    bu_hi = bu[sel_hi]
    log.info(f"    Sideband events: lo={sel_lo.sum()}, hi={sel_hi.sum()}")

    if sel_lo.sum() + sel_hi.sum() < 10:
        log.warning("    Too few events, skipping")
        return

    bw = 5.0
    bins_lo = int(round((SB_LO[1] - SB_LO[0]) / bw))
    bins_hi = int(round((SB_HI[1] - SB_HI[0]) / bw))

    h_lo, edges_lo = np.histogram(bu_lo, range=SB_LO, bins=bins_lo)
    h_hi, edges_hi = np.histogram(bu_hi, range=SB_HI, bins=bins_hi)

    h_all = np.concatenate([h_lo, h_hi])
    centers_lo = (edges_lo[1:] + edges_lo[:-1]) / 2
    centers_hi = (edges_hi[1:] + edges_hi[:-1]) / 2
    centers_all = np.concatenate([centers_lo, centers_hi])
    err_all = np.maximum(np.sqrt(h_all + 1), 1.0)

    def _exp_bg(x, a, lam):
        return a * np.exp(-lam * (x - M_BPLUS))

    fit_ok = False
    try:
        popt, _ = curve_fit(
            _exp_bg,
            centers_all,
            h_all,
            p0=[h_all.mean(), 0.001],
            sigma=err_all,
            bounds=([0, 0], [h_all.max() * 10, 0.05]),
            maxfev=3000,
        )
        fit_ok = True
        log.info(f"    Exp fit: a={popt[0]:.2f}, λ={popt[1]:.5f}")
    except Exception as e:
        log.warning(f"    Exponential fit failed: {e}")

    fig, ax = plt.subplots()
    _plot_fit_errorbars(ax, h_lo, edges_lo, label="Sideband data")
    _plot_fit_errorbars(ax, h_hi, edges_hi, label=None)

    if fit_ok:
        x_fine = np.linspace(SB_LO[0], SB_HI[1], 600)
        ax.plot(
            x_fine,
            _exp_bg(x_fine, *popt),
            color=COLORS[0],
            linewidth=2,
            label=rf"Exp. fit ($\lambda={popt[1]*1e3:.2f}\times10^{{-3}}$/MeV)",
        )

    ax.axvspan(SB_LO[1], SB_HI[0], color="lightgray", alpha=0.5, label="Signal region (blinded)")
    ax.axvline(M_BPLUS, color="gray", linestyle=":", linewidth=1.5)
    ax.set_xlabel(r"$m(\bar{\Lambda}pKK)_{\rm DTF}$ [MeV/$c^2$]")
    ax.set_ylabel(rf"Candidates / ({int(bw)} MeV/$c^2$)")
    ax.set_xlim(SB_LO[0] - 10, SB_HI[1] + 10)
    ax.legend(frameon=False, fontsize=12)

    out = figs_path(cat, "fit", "fit_to_sideband.pdf")
    save_fig(fig, out)
    log.info(f"    Saved: {out}")


def plot_fit_normal_mc(cat: str, bu: np.ndarray):
    """
    fit/fit_ToNormal_Run2MDU_MC_mB.pdf — B+ mass fit to J/ψ normalization MC.

    Gaussian fit to the B+ DTF mass peak in Run 2 J/ψ MC.
    Fit parameters shown as text annotation (not in legend) to avoid clutter.
    """
    log.info(f"  fit_ToNormal_Run2MDU_MC_mB [{cat}]")

    sel = (bu > 5100) & (bu < 5500)
    bu_sel = bu[sel]
    log.info(f"    MC events in [5100,5500]: {sel.sum()}")

    if len(bu_sel) < 20:
        log.warning("    Too few MC events, skipping")
        return

    _bfit = BINNING["bu_mass_fit"]
    x_range = list(_bfit["range"])
    bins = _bfit["bins"]
    width = (x_range[1] - x_range[0]) / bins

    h, edges = np.histogram(bu_sel, range=x_range, bins=bins)
    centers = (edges[1:] + edges[:-1]) / 2
    err = np.maximum(np.sqrt(h + 1), 1.0)

    fig, ax = plt.subplots()
    ax.hist(
        bu_sel,
        range=x_range,
        bins=bins,
        histtype="step",
        linewidth=2,
        color=COLORS[2],
        label=rf"J/$\psi$K MC (Run 2, $\Lambda_{{{cat}}}$)",
    )

    peak_mask = (centers > 5200) & (centers < 5380)
    x_pk, y_pk, e_pk = centers[peak_mask], h[peak_mask], err[peak_mask]

    try:

        def _gauss(x, n_sig, mu, sigma):
            return (
                width
                * n_sig
                / (sigma * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            )

        popt, pcov = curve_fit(
            _gauss,
            x_pk,
            y_pk,
            sigma=e_pk,
            p0=[h.max() * width, M_BPLUS, 20.0],
            bounds=([0, M_BPLUS - 50, 5], [h.max() * width * 5, M_BPLUS + 50, 80]),
            maxfev=3000,
        )
        perr = np.sqrt(np.diag(pcov))
        n_sig, mu, sigma = popt

        x_fine = np.linspace(x_range[0], x_range[1], 600)
        ax.plot(x_fine, _gauss(x_fine, *popt), color=COLORS[3], linewidth=2, label="Gaussian fit")

        ax.text(
            0.97,
            0.97,
            rf"$\mu = {mu:.1f} \pm {perr[1]:.1f}$ MeV" + "\n"
            rf"$\sigma = {sigma:.1f} \pm {perr[2]:.1f}$ MeV",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )
        log.info(f"    MC fit: μ={mu:.2f}±{perr[1]:.2f}, σ={sigma:.2f}±{perr[2]:.2f}")
    except Exception as e:
        log.warning(f"    MC Gaussian fit failed: {e}")

    ax.axvline(M_BPLUS, color="gray", linestyle=":", linewidth=1.5)
    ax.set_xlabel(r"$m(\bar{\Lambda}pKK)_{\rm DTF}$ [MeV/$c^2$]")
    ax.set_ylabel(rf"Candidates / ({int(width)} MeV/$c^2$)")
    ax.set_xlim(*x_range)
    ax.legend(frameon=False, fontsize=12, loc="upper left")

    out = figs_path(cat, "fit", "fit_ToNormal_Run2MDU_MC_mB.pdf")
    save_fig(fig, out)
    log.info(f"    Saved: {out}")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════


def main():
    for cat in ("LL", "DD"):
        log.info(f"=== Category: Lambda{cat} ===")

        bu = _collect_data(cat)
        if len(bu) == 0:
            log.warning(f"  No data found for {cat}, skipping sideband fit")
        else:
            plot_fit_to_sideband(cat, bu)

        mc = _collect_mc_jpsi(cat)
        if len(mc) == 0:
            log.warning(f"  No J/ψ MC found for {cat}, skipping MC fit")
        else:
            plot_fit_normal_mc(cat, mc)

    log.info("=== Done. ===")


if __name__ == "__main__":
    main()
