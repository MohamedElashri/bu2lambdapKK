"""
Background studies for the B+ → Λ̄pK⁻K⁺ analysis note.

Produces three sets of plots documenting the actual background studies:

1. K_S^0 contamination check (plot_ks0_veto):
   Recompute the Λ daughter mass under the π⁺π⁻ hypothesis (assign pion mass to the
   proton track).  No peak at the K_S^0 mass (497.6 MeV) demonstrates negligible
   contamination.  Uses real data passing trigger + Λ mass window.

2. Non-resonant B+ → Λ̄pK⁻K⁺ shape (plot_nonresonant_shape):
   Show M(Λ̄pK⁻) and B+ corrected mass for phase-space KpKm MC after applying the
   B+ corrected mass signal window [5255, 5305] MeV.  Demonstrates the smooth
   phase-space shape absorbed by the ARGUS background.

3. Partially reconstructed background (plot_partial_reco):
   Show the B+ corrected mass for real data in a wide window [4900, 5500] MeV.
   The smooth tail below the signal window [5255, 5305] MeV visualises where
   partially reconstructed events accumulate and confirms the signal window
   efficiently isolates the B+ peak.

Outputs (per category LL/DD):
   figs/LambdaLL/backgrounds/ks0_veto.pdf
   figs/LambdaDD/backgrounds/ks0_veto.pdf
   figs/LambdaLL/backgrounds/nonresonant_mLpK.pdf
   figs/LambdaDD/backgrounds/nonresonant_mLpK.pdf
   figs/LambdaLL/backgrounds/nonresonant_Bcorr.pdf
   figs/LambdaDD/backgrounds/nonresonant_Bcorr.pdf
   figs/LambdaLL/backgrounds/partial_reco.pdf
   figs/LambdaDD/backgrounds/partial_reco.pdf

Run from analysis/ directory:
    uv run python studies/background_studies/plot_background_studies.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot

STUDY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY_DIR.parents[1]))  # analysis/ for modules.*

from modules.plot_utils import COLORS, figs_path, save_fig, setup_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

setup_style()

# ── Physics constants ─────────────────────────────────────────────────────────
M_LAMBDA_PDG = 1115.683  # MeV/c²
M_PI = 139.57018  # MeV/c²
M_KS0_PDG = 497.611  # MeV/c²  (for labelling only)
BU_CORR_MIN = 5255.0  # MeV/c²  B+ signal window
BU_CORR_MAX = 5305.0
LAMBDA_MIN = 1108.0  # MeV/c²  offline Λ mass window
LAMBDA_MAX = 1126.0

# ── File paths ────────────────────────────────────────────────────────────────
DATA_BASE = Path("/share/lazy/Mohamed/Bu2LambdaPPP/files/data")
MC_BASE = Path("/share/lazy/Mohamed/Bu2LambdaPPP/files/mc")

YEARS = ["16", "17", "18"]
MAGNETS = ["MD", "MU"]

TREE_NAMES = {"LL": "B2L0barPKpPim_LL/DecayTree", "DD": "B2L0barPKpPim_DD/DecayTree"}
KPKM_TREE = {"LL": "B2L0barPKpKm_LL/DecayTree", "DD": "B2L0barPKpKm_DD/DecayTree"}


# ════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════


def _trigger_mask(ev: dict, n: int) -> np.ndarray:
    """Standard trigger selection: L0 TIS or TOS, HLT1 TOS, HLT2 TOS."""
    l0_keys = ["Bu_L0Global_TIS", "Bu_L0HadronDecision_TOS"]
    hlt1_keys = ["Bu_Hlt1TrackMVADecision_TOS", "Bu_Hlt1TwoTrackMVADecision_TOS"]
    hlt2_keys = [
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
    ]
    ml0 = np.zeros(n, dtype=bool)
    for k in l0_keys:
        if k in ev:
            ml0 = ml0 | (np.asarray(ev[k]) > 0)
    if not any(k in ev for k in l0_keys):
        ml0[:] = True
    mhlt1 = np.zeros(n, dtype=bool)
    for k in hlt1_keys:
        if k in ev:
            mhlt1 = mhlt1 | (np.asarray(ev[k]) > 0)
    if not any(k in ev for k in hlt1_keys):
        mhlt1[:] = True
    mhlt2 = np.zeros(n, dtype=bool)
    for k in hlt2_keys:
        if k in ev:
            mhlt2 = mhlt2 | (np.asarray(ev[k]) > 0)
    if not any(k in ev for k in hlt2_keys):
        mhlt2[:] = True
    return ml0 & mhlt1 & mhlt2


def _bu_corrected_mass(ev: dict) -> np.ndarray:
    """Bu_DTFL0_M - L0_MM + M_LAMBDA_PDG.
    Handles jagged (object) arrays by taking the first DTF solution per event."""
    dtf_m = np.asarray(ev["Bu_DTFL0_M"])
    if dtf_m.dtype == object:
        dtf_m = np.array(
            [float(x[0]) if hasattr(x, "__len__") and len(x) > 0 else float(x) for x in dtf_m]
        )
    return dtf_m - np.asarray(ev["L0_MM"]) + M_LAMBDA_PDG


# ════════════════════════════════════════════════════════════════════════════════
# 1. K_S^0 CONTAMINATION STUDY
# ════════════════════════════════════════════════════════════════════════════════


def _ks0_mass_from_event(ev: dict) -> np.ndarray:
    """
    Compute the Λ daughter invariant mass under the π⁺π⁻ hypothesis.

    The proton track (Lp) is assigned the pion mass; the pion track (Lpi) keeps
    the pion mass.  The invariant mass of the resulting two-pion system is computed
    from the track momenta:
        E_p→π = sqrt(|p_Lp|² + m_π²)
        E_π   = sqrt(|p_Lpi|² + m_π²)   [= Lpi energy under π hypothesis]
        m(ππ) = sqrt((E_p→π + E_π)² - |p_Lp + p_Lpi|²)
    A K_S^0 peak at 497.6 MeV/c² in this distribution would indicate contamination.
    """
    # Proton track momenta
    px_p = np.asarray(ev["Lp_PX"])
    py_p = np.asarray(ev["Lp_PY"])
    pz_p = np.asarray(ev["Lp_PZ"])
    p2_p = px_p**2 + py_p**2 + pz_p**2
    # Pion track momenta
    px_pi = np.asarray(ev["Lpi_PX"])
    py_pi = np.asarray(ev["Lpi_PY"])
    pz_pi = np.asarray(ev["Lpi_PZ"])
    p2_pi = px_pi**2 + py_pi**2 + pz_pi**2
    # Energies under π hypothesis for both tracks
    E_p_as_pi = np.sqrt(p2_p + M_PI**2)
    E_pi = np.sqrt(p2_pi + M_PI**2)
    # Combined 4-vector
    E_tot = E_p_as_pi + E_pi
    px_tot = px_p + px_pi
    py_tot = py_p + py_pi
    pz_tot = pz_p + pz_pi
    m2 = E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)
    m2 = np.where(m2 > 0, m2, 0.0)
    return np.sqrt(m2)


def _load_data_for_ks0(cat: str) -> np.ndarray:
    """Load data events for K_S^0 study: trigger + Λ mass window."""
    need = [
        "Bu_DTFL0_M",
        "L0_MM",
        "Lp_PX",
        "Lp_PY",
        "Lp_PZ",
        "Lpi_PX",
        "Lpi_PY",
        "Lpi_PZ",
        "Bu_L0Global_TIS",
        "Bu_L0HadronDecision_TOS",
        "Bu_Hlt1TrackMVADecision_TOS",
        "Bu_Hlt1TwoTrackMVADecision_TOS",
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
    ]
    tree_name = TREE_NAMES[cat]
    arrs = []
    for yr in YEARS:
        for mag in MAGNETS:
            p = DATA_BASE / f"dataBu2L0barPHH_{yr}{mag}.root"
            if not p.exists():
                continue
            try:
                t = uproot.open(p)[tree_name]
                avail = [b for b in need if b in t.keys()]
                ev = t.arrays(avail, library="np")
                n = len(ev["L0_MM"])
                mask = _trigger_mask(ev, n)
                mask &= (ev["L0_MM"] > LAMBDA_MIN) & (ev["L0_MM"] < LAMBDA_MAX)
                # Compute and store Λ→ππ mass for passing events
                ev_masked = {k: v[mask] for k, v in ev.items()}
                m_pipi = _ks0_mass_from_event(ev_masked)
                arrs.append(m_pipi)
            except Exception as e:
                log.warning(f"  Skip {p} [{cat}]: {e}")
    return np.concatenate(arrs) if arrs else np.array([])


def plot_ks0_veto(cat: str) -> None:
    """
    Plot the Λ→ππ invariant mass to demonstrate absence of K_S^0 contamination.
    """
    log.info(f"  K_S^0 veto study [{cat}]")
    m_pipi = _load_data_for_ks0(cat)
    if len(m_pipi) == 0:
        log.warning(f"  No data for K_S^0 study [{cat}]")
        return
    log.info(f"  Events: {len(m_pipi)}")

    fig, ax = plt.subplots()
    bins = np.linspace(300, 900, 61)  # 10 MeV/bin, covers K_S^0 at 498 MeV

    counts, edges = np.histogram(m_pipi, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    errs = np.sqrt(counts)
    nonzero = counts > 0
    ax.errorbar(
        centres[nonzero],
        counts[nonzero],
        yerr=errs[nonzero],
        fmt="o",
        color="black",
        markersize=4,
        elinewidth=1.5,
        label=r"Data ($\Lambda^0 \to p\pi^-$ candidates, $p \to \pi$ mass hypothesis)",
    )

    # Mark K_S^0 mass
    ax.axvline(
        M_KS0_PDG,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=r"$K_S^0$ mass ($497.6\,\mathrm{MeV}/c^2$)",
    )

    ax.set_xlabel(r"$m(\pi^+\pi^-)$ under $p\to\pi$ hypothesis [MeV/$c^2$]")
    ax.set_ylabel("Candidates / (10 MeV/$c^2$)")
    ax.legend(fontsize=11, frameon=False)

    lcat = "LL" if cat == "LL" else "DD"
    ax.text(
        0.97,
        0.97,
        rf"$\Lambda_{{\rm {lcat}}}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )

    out = figs_path(cat, "backgrounds", "ks0_veto.pdf")
    save_fig(fig, out)
    log.info(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════════════
# 2. NON-RESONANT B+ → Λ̄pK⁻K⁺ SHAPE
# ════════════════════════════════════════════════════════════════════════════════


def _load_kpkm_mc(cat: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load KpKm phase-space MC: apply trigger + Λ window + B+ corrected mass window.
    Returns (M_LpKm, Bu_corrected_mass).
    """
    need = [
        "Bu_DTFL0_M",
        "L0_MM",
        "L0_PE",
        "L0_PX",
        "L0_PY",
        "L0_PZ",
        "p_PE",
        "p_PX",
        "p_PY",
        "p_PZ",
        "h2_PE",
        "h2_PX",
        "h2_PY",
        "h2_PZ",
        "Bu_L0Global_TIS",
        "Bu_L0HadronDecision_TOS",
        "Bu_Hlt1TrackMVADecision_TOS",
        "Bu_Hlt1TwoTrackMVADecision_TOS",
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
    ]
    tree_name = KPKM_TREE[cat]
    mlpk_all = []
    bcorr_all = []

    for yr in YEARS:
        for mag in MAGNETS:
            p = MC_BASE / "KpKm" / f"KpKm_{yr}_{mag}.root"
            if not p.exists():
                continue
            try:
                t = uproot.open(p)[tree_name]
                avail = [b for b in need if b in t.keys()]
                ev = t.arrays(avail, library="np")
                n = len(ev["L0_MM"])
                mask = _trigger_mask(ev, n)
                mask &= (ev["L0_MM"] > LAMBDA_MIN) & (ev["L0_MM"] < LAMBDA_MAX)
                # B+ corrected mass
                bcorr = _bu_corrected_mass(ev)
                mask &= (bcorr > BU_CORR_MIN) & (bcorr < BU_CORR_MAX)

                ev_m = {k: v[mask] for k, v in ev.items()}
                bc_m = bcorr[mask]

                # M(Λ̄pK⁻): 4-vector sum of L0 + bachelor p + h2
                E_tot = ev_m["L0_PE"] + ev_m["p_PE"] + ev_m["h2_PE"]
                PX = ev_m["L0_PX"] + ev_m["p_PX"] + ev_m["h2_PX"]
                PY = ev_m["L0_PY"] + ev_m["p_PY"] + ev_m["h2_PY"]
                PZ = ev_m["L0_PZ"] + ev_m["p_PZ"] + ev_m["h2_PZ"]
                m2 = E_tot**2 - (PX**2 + PY**2 + PZ**2)
                m2 = np.where(m2 > 0, m2, 0.0)
                mlpk = np.sqrt(m2)

                mlpk_all.append(mlpk)
                bcorr_all.append(bc_m)
            except Exception as e:
                log.warning(f"  Skip {p} [{cat}]: {e}")

    m = np.concatenate(mlpk_all) if mlpk_all else np.array([])
    b = np.concatenate(bcorr_all) if bcorr_all else np.array([])
    return m, b


def plot_nonresonant_shape(cat: str) -> None:
    """
    Show M(Λ̄pK⁻) and B+ corrected mass for non-resonant phase-space MC.
    Demonstrates the smooth ARGUS-like background shape.
    """
    log.info(f"  Non-resonant shape [{cat}]")
    m_lpk, b_corr = _load_kpkm_mc(cat)
    if len(m_lpk) == 0:
        log.warning(f"  No KpKm MC found [{cat}]")
        return
    log.info(f"  Events in B+ window: {len(m_lpk)}")

    # ── Plot 1: M(Λ̄pK⁻) ──────────────────────────────────────────────────────
    fig, ax = plt.subplots()
    bins_lpk = np.linspace(2800, 4000, 61)  # 20 MeV/bin
    ax.hist(
        m_lpk,
        bins=bins_lpk,
        histtype="step",
        color=COLORS[2],
        linewidth=2,
        label=r"$B^+\to\bar{\Lambda}pK^+K^-$ phase-space MC",
    )
    # Mark charmonium masses
    for name, mass, color in [
        (r"$J/\psi$", 3096.9, "C3"),
        (r"$\eta_c$", 2983.9, "C4"),
        (r"$\chi_{c0}$", 3414.7, "C5"),
        (r"$\chi_{c1}$", 3510.7, "C6"),
        (r"$\eta_c(2S)$", 3637.8, "C7"),
    ]:
        ax.axvline(mass, linestyle=":", color=color, linewidth=1, alpha=0.7)
    ax.set_xlabel(r"$m(\bar{\Lambda}pK^-)$ [MeV/$c^2$]")
    ax.set_ylabel("Candidates / (20 MeV/$c^2$)")
    ax.legend(fontsize=11, frameon=False, loc="upper left")
    lcat = "LL" if cat == "LL" else "DD"
    ax.text(
        0.97,
        0.97,
        rf"$\Lambda_{{\rm {lcat}}}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )
    out1 = figs_path(cat, "backgrounds", "nonresonant_mLpK.pdf")
    save_fig(fig, out1)
    log.info(f"  Saved: {out1}")

    # ── Plot 2: B+ corrected mass ──────────────────────────────────────────────
    fig2, ax2 = plt.subplots()
    bins_bu = np.linspace(BU_CORR_MIN - 5, BU_CORR_MAX + 5, 25)
    ax2.hist(
        b_corr,
        bins=bins_bu,
        histtype="step",
        color=COLORS[2],
        linewidth=2,
        label=r"$B^+\to\bar{\Lambda}pK^+K^-$ phase-space MC",
    )
    ax2.axvspan(
        BU_CORR_MIN,
        BU_CORR_MAX,
        alpha=0.12,
        color="grey",
        label=rf"Signal window $[{BU_CORR_MIN:.0f},{BU_CORR_MAX:.0f}]\,\mathrm{{MeV}}/c^2$",
    )
    ax2.set_xlabel(r"$m_{\rm corr}(\bar{\Lambda}pK^+K^-)$ [MeV/$c^2$]")
    ax2.set_ylabel("Candidates / bin")
    ax2.legend(fontsize=11, frameon=False)
    ax2.text(
        0.97,
        0.97,
        rf"$\Lambda_{{\rm {lcat}}}$",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )
    out2 = figs_path(cat, "backgrounds", "nonresonant_Bcorr.pdf")
    save_fig(fig2, out2)
    log.info(f"  Saved: {out2}")


# ════════════════════════════════════════════════════════════════════════════════
# 3. PARTIALLY RECONSTRUCTED BACKGROUND
# ════════════════════════════════════════════════════════════════════════════════


def _load_data_bu_corr(cat: str, bu_min: float = 4900.0, bu_max: float = 5500.0) -> np.ndarray:
    """
    Load B+ corrected mass from data in a wide window [bu_min, bu_max] MeV,
    after trigger + Λ mass window.  Includes both below-signal and signal regions.
    """
    need = [
        "Bu_DTFL0_M",
        "L0_MM",
        "Bu_L0Global_TIS",
        "Bu_L0HadronDecision_TOS",
        "Bu_Hlt1TrackMVADecision_TOS",
        "Bu_Hlt1TwoTrackMVADecision_TOS",
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
    ]
    tree_name = TREE_NAMES[cat]
    arrs = []
    for yr in YEARS:
        for mag in MAGNETS:
            p = DATA_BASE / f"dataBu2L0barPHH_{yr}{mag}.root"
            if not p.exists():
                continue
            try:
                t = uproot.open(p)[tree_name]
                avail = [b for b in need if b in t.keys()]
                ev = t.arrays(avail, library="np")
                n = len(ev["L0_MM"])
                mask = _trigger_mask(ev, n)
                mask &= (ev["L0_MM"] > LAMBDA_MIN) & (ev["L0_MM"] < LAMBDA_MAX)
                bcorr = _bu_corrected_mass(ev)
                mask &= (bcorr > bu_min) & (bcorr < bu_max)
                arrs.append(bcorr[mask])
            except Exception as e:
                log.warning(f"  Skip {p} [{cat}]: {e}")
    return np.concatenate(arrs) if arrs else np.array([])


def plot_partial_reco(cat: str) -> None:
    """
    Show the B+ corrected mass in a wide window to visualise the partially
    reconstructed background accumulating below the signal window.
    """
    log.info(f"  Partially reconstructed background [{cat}]")
    bu_min, bu_max = 4900.0, 5500.0
    bcorr = _load_data_bu_corr(cat, bu_min, bu_max)
    if len(bcorr) == 0:
        log.warning(f"  No data [{cat}]")
        return
    log.info(f"  Events in wide B+ window: {len(bcorr)}")

    fig, ax = plt.subplots()
    bins = np.linspace(bu_min, bu_max, 61)  # 10 MeV/bin

    counts, edges = np.histogram(bcorr, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    errs = np.sqrt(counts)
    nonzero = counts > 0
    ax.errorbar(
        centres[nonzero],
        counts[nonzero],
        yerr=errs[nonzero],
        fmt="o",
        color="black",
        markersize=4,
        elinewidth=1.5,
        label="Data",
    )

    # Shade signal window
    ax.axvspan(
        BU_CORR_MIN,
        BU_CORR_MAX,
        alpha=0.15,
        color="#003366",
        label=rf"Signal window $[{BU_CORR_MIN:.0f},{BU_CORR_MAX:.0f}]\,\mathrm{{MeV}}/c^2$",
    )
    # Mark boundary
    ax.axvline(BU_CORR_MIN, color="#003366", linestyle="--", linewidth=1)
    ax.axvline(BU_CORR_MAX, color="#003366", linestyle="--", linewidth=1)

    ax.set_xlabel(r"$m_{\rm corr}(\bar{\Lambda}pK^+K^-)$ [MeV/$c^2$]")
    ax.set_ylabel("Candidates / (10 MeV/$c^2$)")
    ax.legend(fontsize=11, frameon=False)
    lcat = "LL" if cat == "LL" else "DD"
    ax.text(
        0.97,
        0.97,
        rf"$\Lambda_{{\rm {lcat}}}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )

    # Annotate the below-signal region
    ax.annotate(
        "Partially reconstructed\n(below signal window)",
        xy=(5150, ax.get_ylim()[1] * 0.85),
        fontsize=11,
        ha="center",
        color="dimgray",
        arrowprops=dict(arrowstyle="->", color="dimgray"),
        xytext=(5150, ax.get_ylim()[1] * 0.85),
    )

    out = figs_path(cat, "backgrounds", "partial_reco.pdf")
    save_fig(fig, out)
    log.info(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════


def main() -> None:
    for cat in ("LL", "DD"):
        log.info(f"\n=== Lambda{cat} ===")
        log.info("--- K_S^0 contamination study ---")
        plot_ks0_veto(cat)
        log.info("--- Non-resonant shape ---")
        plot_nonresonant_shape(cat)
        log.info("--- Partially reconstructed background ---")
        plot_partial_reco(cat)
    log.info("\n=== All background studies done ===")


if __name__ == "__main__":
    main()
