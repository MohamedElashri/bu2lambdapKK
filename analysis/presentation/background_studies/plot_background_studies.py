"""
Background studies for the B+ → Λ̄pK⁻K⁺ analysis note.

Produces three sets of plots documenting the actual background studies:

1. K_S^0 contamination check (plot_ks0_veto):
   Recompute the Λ daughter mass under the π⁺π⁻ hypothesis (assign pion mass to the
   proton track).  No peak at the K_S^0 mass (497.6 MeV) demonstrates negligible
   contamination.  Uses real data passing trigger + Λ mass window.

2. Non-resonant B+ → Λ̄pK⁻K⁺ shape (plot_nonresonant_shape):
   Show M(Λ̄pK⁻) and B+ corrected mass for phase-space KpKm MC after applying the
   configured B+ corrected mass signal window.  Demonstrates the smooth
   phase-space shape absorbed by the ARGUS background.

3. Partial-reconstruction study (plot_partial_reco):
   Fit the wide B+ corrected-mass spectrum after the deployed high-yield MVA,
   but before the nominal B+ mass window, using a narrow B+ peak, an ARGUS-like
   low-mass partial-reco component, and a smooth combinatorial term.  This gives
   a data-driven estimate of how much low-mass background leaks into the nominal
   signal window.

Outputs (per category LL/DD):
   figs/LambdaLL/backgrounds/ks0_veto.pdf
   figs/LambdaDD/backgrounds/ks0_veto.pdf
   figs/LambdaLL/backgrounds/ks0_veto_bcorr_before_after.pdf
   figs/LambdaDD/backgrounds/ks0_veto_bcorr_before_after.pdf
   figs/LambdaLL/backgrounds/nonresonant_mLpK.pdf
   figs/LambdaDD/backgrounds/nonresonant_mLpK.pdf
   figs/LambdaLL/backgrounds/nonresonant_Bcorr.pdf
   figs/LambdaDD/backgrounds/nonresonant_Bcorr.pdf
   figs/LambdaLL/backgrounds/partial_reco.pdf
   figs/LambdaDD/backgrounds/partial_reco.pdf

Run from analysis/ directory:
    uv run python presentation/background_studies/plot_background_studies.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from catboost import CatBoostClassifier
from scipy.optimize import curve_fit
from scipy.stats import norm

STUDY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY_DIR.parents[1]))  # analysis/ for modules.*

from modules.plot_utils import COLORS, figs_path, save_fig, setup_style
from modules.presentation_config import (
    DATA_L0_TIS_KEYS,
    HLT1_TOS_KEYS,
    HLT2_TOS_KEYS,
    MC_L0_TIS_KEYS,
    get_presentation_config,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

setup_style()

# ── Physics constants ─────────────────────────────────────────────────────────
M_PI = 139.57018  # MeV/c²
M_KS0_PDG = 497.611  # MeV/c²  (for labelling only)
KS0_VETO_HALF_WIDTH = 20.0  # MeV/c²
BU_PDG = 5279.34
PARTIAL_RECO_FIT_MIN = 5100.0
PARTIAL_RECO_FIT_MAX = 5450.0

# ── File paths ────────────────────────────────────────────────────────────────
PRESENTATION = get_presentation_config()
DATA_BASE = PRESENTATION.data_base
MC_BASE = PRESENTATION.mc_base
YEARS = PRESENTATION.year_suffixes
MAGNETS = PRESENTATION.magnets
M_LAMBDA_PDG = PRESENTATION.lambda_mass_pdg
LAMBDA_MIN = PRESENTATION.lambda_mass_min
LAMBDA_MAX = PRESENTATION.lambda_mass_max
BU_CORR_MIN, BU_CORR_MAX = PRESENTATION.bu_signal_window()
PARTIAL_RECO_WINDOW_SUMMARY = (BU_CORR_MIN, BU_CORR_MAX)

TREE_NAMES = {"LL": "B2L0barPKpPim_LL/DecayTree", "DD": "B2L0barPKpPim_DD/DecayTree"}
KPKM_TREE = {"LL": "B2L0barPKpKm_LL/DecayTree", "DD": "B2L0barPKpKm_DD/DecayTree"}


# ════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════


def _trigger_mask(ev: dict, n: int) -> np.ndarray:
    """Standard trigger selection: L0 TIS, HLT1 TOS, HLT2 TOS."""
    l0_keys = MC_L0_TIS_KEYS
    hlt1_keys = HLT1_TOS_KEYS
    hlt2_keys = HLT2_TOS_KEYS
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


def _load_data_for_ks0(cat: str) -> tuple[np.ndarray, np.ndarray]:
    """Load data for the K_S^0 study: return (m_pi_pi, B+ corrected mass)."""
    need = [
        "Bu_DTFL0_M",
        "L0_MM",
        "Lp_PX",
        "Lp_PY",
        "Lp_PZ",
        "Lpi_PX",
        "Lpi_PY",
        "Lpi_PZ",
        *DATA_L0_TIS_KEYS,
        *HLT1_TOS_KEYS,
        *HLT2_TOS_KEYS,
    ]
    tree_name = TREE_NAMES[cat]
    mpipi_arrs = []
    bcorr_arrs = []
    for yr in YEARS:
        for mag in MAGNETS:
            p = DATA_BASE / f"dataBu2L0barPHH_{yr}{mag}.root"
            if not p.exists():
                continue
            try:
                t = uproot.open(p)[tree_name]
                avail = [b for b in need if b in t.keys()]
                for ev in t.iterate(avail, library="np", step_size="100 MB"):
                    n = len(ev["L0_MM"])
                    mask = _trigger_mask(ev, n)
                    mask &= (ev["L0_MM"] > LAMBDA_MIN) & (ev["L0_MM"] < LAMBDA_MAX)
                    if not np.any(mask):
                        continue
                    ev_masked = {k: v[mask] for k, v in ev.items()}
                    m_pipi = _ks0_mass_from_event(ev_masked)
                    bcorr = _bu_corrected_mass(ev_masked)
                    mpipi_arrs.append(m_pipi)
                    bcorr_arrs.append(bcorr)
            except Exception as e:
                log.warning(f"  Skip {p} [{cat}]: {e}")
    m_pipi = np.concatenate(mpipi_arrs) if mpipi_arrs else np.array([])
    bcorr = np.concatenate(bcorr_arrs) if bcorr_arrs else np.array([])
    return m_pipi, bcorr


def plot_ks0_veto(cat: str) -> None:
    """
    Plot the Λ→ππ invariant mass to demonstrate absence of K_S^0 contamination.
    """
    log.info(f"  K_S^0 veto study [{cat}]")
    m_pipi, _ = _load_data_for_ks0(cat)
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
        label="Data",
    )

    # Mark K_S^0 mass
    ax.axvline(
        M_KS0_PDG,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=r"$K_S^0$ mass",
    )
    ax.axvspan(
        M_KS0_PDG - KS0_VETO_HALF_WIDTH,
        M_KS0_PDG + KS0_VETO_HALF_WIDTH,
        color="red",
        alpha=0.08,
        label="Veto window",
    )

    ax.set_xlabel(r"$m(\pi^+\pi^-)$ under $p\to\pi$ hypothesis [MeV/$c^2$]")
    ax.set_ylabel("Candidates / (10 MeV/$c^2$)")
    ax.legend(
        fontsize=10.5,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.5,
        borderaxespad=0.2,
    )

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


def plot_ks0_veto_before_after(cat: str) -> None:
    """Overlay B+ corrected mass before/after the K_S^0 veto."""
    log.info(f"  K_S^0 veto before/after [{cat}]")
    m_pipi, bcorr = _load_data_for_ks0(cat)
    if len(m_pipi) == 0 or len(bcorr) == 0:
        log.warning(f"  No data for K_S^0 before/after study [{cat}]")
        return

    keep_mask = np.abs(m_pipi - M_KS0_PDG) >= KS0_VETO_HALF_WIDTH
    kept_frac = 100.0 * np.count_nonzero(keep_mask) / len(keep_mask)

    fig, ax = plt.subplots()
    bins = np.linspace(4900, 5500, 61)  # 10 MeV/bin

    ax.hist(
        bcorr,
        bins=bins,
        histtype="step",
        linewidth=2,
        color="black",
        label="Before veto",
    )
    ax.hist(
        bcorr[keep_mask],
        bins=bins,
        histtype="step",
        linewidth=2,
        color=COLORS[3],
        label=rf"After veto ({kept_frac:.2f}\% retained)",
    )

    ax.axvspan(
        BU_CORR_MIN,
        BU_CORR_MAX,
        alpha=0.12,
        color="#003366",
        label=rf"Signal window $[{BU_CORR_MIN:.0f},{BU_CORR_MAX:.0f}]\,\mathrm{{MeV}}/c^2$",
    )
    ax.set_xlabel(r"$m_{\rm corr}(\bar{\Lambda}pK^+K^-)$ [MeV/$c^2$]")
    ax.set_ylabel("Candidates / (10 MeV/$c^2$)")
    ax.legend(
        fontsize=9.5,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.5,
        borderaxespad=0.2,
    )

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

    out = figs_path(cat, "backgrounds", "ks0_veto_bcorr_before_after.pdf")
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
        *MC_L0_TIS_KEYS,
        *HLT1_TOS_KEYS,
        *HLT2_TOS_KEYS,
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


def _flatten_numeric(arr: np.ndarray) -> np.ndarray:
    """Return a flat float array, taking the first entry for jagged/object branches."""
    arr = np.asarray(arr)
    if arr.dtype != object:
        return arr.astype(float, copy=False)
    return np.array(
        [float(x[0]) if hasattr(x, "__len__") and len(x) > 0 else float(x) for x in arr],
        dtype=float,
    )


def _load_mva_working_point(cat: str) -> tuple[CatBoostClassifier, float, list[str]]:
    """Load the deployed high-yield CatBoost model and threshold for one category."""
    model_dir = PRESENTATION.pipeline_output_dir / "high_yield" / cat / "models"
    model = CatBoostClassifier()
    model.load_model(str(model_dir / "mva_model.cbm"))
    with open(model_dir / "optimized_cuts.json", "r") as f:
        cfg = json.load(f)
    return model, float(cfg["mva_threshold_high"]), list(cfg["features"])


def _deduplicate_candidates(
    masses: np.ndarray, run_numbers: np.ndarray | None, event_numbers: np.ndarray | None
) -> np.ndarray:
    """Mirror the pipeline multiple-candidate handling with a deterministic choice."""
    if run_numbers is None or event_numbers is None or len(masses) == 0:
        return masses
    keys = run_numbers.astype(np.int64) * (10**10) + event_numbers.astype(np.int64)
    _, first_idx = np.unique(keys, return_index=True)
    return masses[np.sort(first_idx)]


def _load_post_mva_bcorr(cat: str) -> np.ndarray:
    """
    Load the wide B+ corrected-mass spectrum after the deployed high-yield MVA.

    This mirrors the current real-data pipeline up to the MVA threshold, but
    deliberately does not impose the nominal B+ mass window so the low-mass
    partial-reconstruction component can be studied directly.
    """
    model, threshold, features = _load_mva_working_point(cat)
    tree_name = KPKM_TREE[cat]
    baseline_cfg = PRESENTATION.config.get_baseline_reduction()
    lambda_cfg = PRESENTATION.config.get_lambda_preselection(cat)
    delta_z_cut = float(lambda_cfg["delta_z_min"])

    branches = [
        "Bu_MM",
        "L0_MM",
        "L0_ENDVERTEX_Z",
        "Bu_ENDVERTEX_Z",
        "Bu_FDCHI2_OWNPV",
        "Bu_IPCHI2_OWNPV",
        "Bu_PT",
        "L0_FDCHI2_OWNPV",
        "Lp_MC15TuneV1_ProbNNp",
        "p_MC15TuneV1_ProbNNp",
        "h1_MC15TuneV1_ProbNNk",
        "h2_MC15TuneV1_ProbNNk",
        "Bu_DTF_chi2",
        "Bu_DTF_CHI2",
        "runNumber",
        "eventNumber",
        "Bu_L0GlobalDecision_TIS",
        "Bu_L0PhysDecision_TIS",
        "Bu_L0HadronDecision_TIS",
        "Bu_L0MuonDecision_TIS",
        "Bu_L0MuonHighDecision_TIS",
        "Bu_L0DiMuonDecision_TIS",
        "Bu_L0PhotonDecision_TIS",
        "Bu_L0ElectronDecision_TIS",
        "Bu_Hlt1TrackMVADecision_TOS",
        "Bu_Hlt1TwoTrackMVADecision_TOS",
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
    ]

    l0_candidates = [
        "Bu_L0GlobalDecision_TIS",
        "Bu_L0PhysDecision_TIS",
        "Bu_L0HadronDecision_TIS",
        "Bu_L0MuonDecision_TIS",
        "Bu_L0MuonHighDecision_TIS",
        "Bu_L0DiMuonDecision_TIS",
        "Bu_L0PhotonDecision_TIS",
        "Bu_L0ElectronDecision_TIS",
    ]
    hlt1_candidates = ["Bu_Hlt1TrackMVADecision_TOS", "Bu_Hlt1TwoTrackMVADecision_TOS"]
    hlt2_candidates = [
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
    ]

    mass_chunks: list[np.ndarray] = []
    run_chunks: list[np.ndarray] = []
    event_chunks: list[np.ndarray] = []

    for yr in YEARS:
        for mag in MAGNETS:
            path = DATA_BASE / f"dataBu2L0barPHH_{yr}{mag}.root"
            if not path.exists():
                continue

            try:
                tree = uproot.open(path)[tree_name]
                avail = [b for b in branches if b in tree.keys()]
                dtf_branch = "Bu_DTF_chi2" if "Bu_DTF_chi2" in avail else "Bu_DTF_CHI2"
                l0_keys = [b for b in l0_candidates if b in avail]
                hlt1_keys = [b for b in hlt1_candidates if b in avail]
                hlt2_keys = [b for b in hlt2_candidates if b in avail]

                for chunk in tree.iterate(avail, library="np", step_size="100 MB"):
                    n = len(chunk["Bu_MM"])
                    mask_l0 = np.zeros(n, dtype=bool)
                    for key in l0_keys:
                        mask_l0 |= np.asarray(chunk[key]) > 0
                    if not l0_keys:
                        mask_l0[:] = True

                    mask_hlt1 = np.zeros(n, dtype=bool)
                    for key in hlt1_keys:
                        mask_hlt1 |= np.asarray(chunk[key]) > 0
                    if not hlt1_keys:
                        mask_hlt1[:] = True

                    mask_hlt2 = np.zeros(n, dtype=bool)
                    for key in hlt2_keys:
                        mask_hlt2 |= np.asarray(chunk[key]) > 0
                    if not hlt2_keys:
                        mask_hlt2[:] = True

                    delta_z = np.abs(
                        np.asarray(chunk["L0_ENDVERTEX_Z"]) - np.asarray(chunk["Bu_ENDVERTEX_Z"])
                    )
                    prod_prob_kk = np.asarray(chunk["h1_MC15TuneV1_ProbNNk"]) * np.asarray(
                        chunk["h2_MC15TuneV1_ProbNNk"]
                    )
                    pid_product = np.asarray(chunk["p_MC15TuneV1_ProbNNp"]) * prod_prob_kk

                    mask = mask_l0 & mask_hlt1 & mask_hlt2
                    mask &= np.asarray(chunk["Bu_FDCHI2_OWNPV"]) > baseline_cfg["bu_fdchi2_min"]
                    mask &= np.asarray(chunk["Bu_IPCHI2_OWNPV"]) < baseline_cfg["bu_ipchi2_max"]
                    mask &= delta_z > baseline_cfg["delta_z_min"]
                    mask &= (
                        np.asarray(chunk["Lp_MC15TuneV1_ProbNNp"]) > baseline_cfg["lp_probnnp_min"]
                    )
                    mask &= (
                        np.asarray(chunk["p_MC15TuneV1_ProbNNp"]) > baseline_cfg["p_probnnp_min"]
                    )
                    mask &= prod_prob_kk > baseline_cfg["hh_probnnk_prod_min"]
                    mask &= np.asarray(chunk["Bu_PT"]) > baseline_cfg["bu_pt_min"]

                    mask &= np.asarray(chunk["L0_MM"]) > lambda_cfg["mass_min"]
                    mask &= np.asarray(chunk["L0_MM"]) < lambda_cfg["mass_max"]
                    mask &= np.asarray(chunk["L0_FDCHI2_OWNPV"]) > lambda_cfg["fd_chisq_min"]
                    mask &= delta_z > delta_z_cut
                    mask &= (
                        np.asarray(chunk["Lp_MC15TuneV1_ProbNNp"])
                        > lambda_cfg["proton_probnnp_min"]
                    )
                    mask &= pid_product > lambda_cfg["pid_product_min"]

                    if not np.any(mask):
                        continue

                    feature_data: dict[str, np.ndarray] = {}
                    for feat in features:
                        if feat == "Bu_DTF_chi2":
                            feature_data[feat] = _flatten_numeric(chunk[dtf_branch])[mask]
                        else:
                            feature_data[feat] = np.asarray(chunk[feat])[mask]

                    scores = model.predict_proba(pd.DataFrame(feature_data)[features])[:, 1]
                    keep_idx = np.flatnonzero(mask)[scores > threshold]
                    if len(keep_idx) == 0:
                        continue

                    bu_corr = np.asarray(chunk["Bu_MM"]) - np.asarray(chunk["L0_MM"]) + M_LAMBDA_PDG
                    mass_chunks.append(bu_corr[keep_idx])

                    if "runNumber" in chunk and "eventNumber" in chunk:
                        run_chunks.append(np.asarray(chunk["runNumber"])[keep_idx])
                        event_chunks.append(np.asarray(chunk["eventNumber"])[keep_idx])
            except Exception as exc:
                log.warning(f"  Skip {path} [{cat}]: {exc}")

    if not mass_chunks:
        return np.array([])

    masses = np.concatenate(mass_chunks)
    runs = np.concatenate(run_chunks) if run_chunks else None
    events = np.concatenate(event_chunks) if event_chunks else None
    return _deduplicate_candidates(masses, runs, events)


def _argus_raw(x: np.ndarray, shape: float, endpoint: float = BU_PDG) -> np.ndarray:
    """ARGUS-like threshold shape for low-mass partially reconstructed backgrounds."""
    x = np.asarray(x, dtype=float)
    term = 1.0 - (x / endpoint) ** 2
    vals = np.zeros_like(x)
    mask = term > 0.0
    vals[mask] = x[mask] * np.sqrt(term[mask]) * np.exp(shape * term[mask])
    return vals


def _partial_reco_model(
    x: np.ndarray,
    n_sig: float,
    mu: float,
    sigma: float,
    n_part: float,
    shape_part: float,
    n_comb: float,
    slope_comb: float,
) -> np.ndarray:
    """Signal Gaussian + ARGUS-like partial-reco + exponential combinatorial density."""
    x = np.asarray(x, dtype=float)

    sig = n_sig * norm.pdf(x, mu, sigma)

    part_grid = np.linspace(PARTIAL_RECO_FIT_MIN, min(PARTIAL_RECO_FIT_MAX, BU_PDG), 2000)
    part_norm = np.trapezoid(_argus_raw(part_grid, shape_part), part_grid)
    part = n_part * _argus_raw(x, shape_part) / max(part_norm, 1e-12)

    slope = max(float(slope_comb), 1e-8)
    exp_norm = 1.0 - np.exp(-slope * (PARTIAL_RECO_FIT_MAX - PARTIAL_RECO_FIT_MIN))
    comb = n_comb * slope * np.exp(-slope * (x - PARTIAL_RECO_FIT_MIN)) / exp_norm

    return sig + part + comb


def _fit_partial_reco_components(mass: np.ndarray) -> dict[str, float | np.ndarray]:
    """Fit the wide post-MVA corrected-mass spectrum and summarise window leakage."""
    mass = np.asarray(mass)
    fit_mass = mass[(mass > PARTIAL_RECO_FIT_MIN) & (mass < PARTIAL_RECO_FIT_MAX)]
    if len(fit_mass) == 0:
        raise RuntimeError("No events in the partial-reco fit range")

    bins = np.linspace(PARTIAL_RECO_FIT_MIN, PARTIAL_RECO_FIT_MAX, 71)  # 5 MeV/bin
    counts, edges = np.histogram(fit_mass, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]
    errors = np.sqrt(np.maximum(counts, 1.0))

    n_sig0 = max(np.count_nonzero((fit_mass > BU_CORR_MIN) & (fit_mass < BU_CORR_MAX)) * 0.65, 50.0)
    n_part0 = max(
        np.count_nonzero((fit_mass > PARTIAL_RECO_FIT_MIN) & (fit_mass < BU_CORR_MIN)) * 0.60, 50.0
    )
    n_comb0 = max(
        np.count_nonzero((fit_mass > BU_CORR_MAX) & (fit_mass < PARTIAL_RECO_FIT_MAX)) * 0.80, 50.0
    )

    popt, pcov = curve_fit(
        lambda x, *p: _partial_reco_model(x, *p) * bin_width,
        centres,
        counts,
        p0=[n_sig0, BU_PDG, 11.0, n_part0, -18.0, n_comb0, 0.012],
        sigma=errors,
        absolute_sigma=True,
        bounds=(
            [0.0, 5270.0, 6.0, 0.0, -80.0, 0.0, 1e-5],
            [
                5.0 * len(fit_mass),
                5288.0,
                22.0,
                5.0 * len(fit_mass),
                25.0,
                5.0 * len(fit_mass),
                0.08,
            ],
        ),
        maxfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))

    grid = np.linspace(PARTIAL_RECO_FIT_MIN, PARTIAL_RECO_FIT_MAX, 3000)
    total = _partial_reco_model(grid, *popt)
    signal = popt[0] * norm.pdf(grid, popt[1], popt[2])
    part_grid = np.linspace(PARTIAL_RECO_FIT_MIN, min(PARTIAL_RECO_FIT_MAX, BU_PDG), 2000)
    part_norm = np.trapezoid(_argus_raw(part_grid, popt[4]), part_grid)
    partial = popt[3] * _argus_raw(grid, popt[4]) / max(part_norm, 1e-12)
    slope = max(float(popt[6]), 1e-8)
    exp_norm = 1.0 - np.exp(-slope * (PARTIAL_RECO_FIT_MAX - PARTIAL_RECO_FIT_MIN))
    combinatorial = popt[5] * slope * np.exp(-slope * (grid - PARTIAL_RECO_FIT_MIN)) / exp_norm

    wmin, wmax = PARTIAL_RECO_WINDOW_SUMMARY
    window_mask = (grid >= wmin) & (grid <= wmax)
    signal_win = float(np.trapezoid(signal[window_mask], grid[window_mask]))
    partial_win = float(np.trapezoid(partial[window_mask], grid[window_mask]))
    comb_win = float(np.trapezoid(combinatorial[window_mask], grid[window_mask]))
    total_win = signal_win + partial_win + comb_win
    observed_win = int(np.count_nonzero((fit_mass >= wmin) & (fit_mass <= wmax)))

    expected = _partial_reco_model(centres, *popt) * bin_width
    chi2 = float(np.sum(((counts - expected) / errors) ** 2))
    ndf = max(len(counts) - len(popt), 1)

    return {
        "fit_mass": fit_mass,
        "counts": counts,
        "edges": edges,
        "centres": centres,
        "errors": errors,
        "bin_width": bin_width,
        "grid": grid,
        "total_curve": total,
        "signal_curve": signal,
        "partial_curve": partial,
        "comb_curve": combinatorial,
        "params": popt,
        "param_errors": perr,
        "observed_window": observed_win,
        "signal_window": signal_win,
        "partial_window": partial_win,
        "comb_window": comb_win,
        "total_window": total_win,
        "partial_fraction_window": partial_win / total_win if total_win > 0 else 0.0,
        "chi2_ndf": chi2 / ndf,
    }


def plot_partial_reco(cat: str) -> None:
    """
    Data-driven post-MVA partial-reconstruction study in the wide corrected-mass spectrum.
    """
    log.info(f"  Partial-reconstruction fit study [{cat}]")
    bcorr = _load_post_mva_bcorr(cat)
    if len(bcorr) == 0:
        log.warning(f"  No post-MVA data [{cat}]")
        return
    log.info(f"  Post-MVA candidates before B+ window [{cat}]: {len(bcorr)}")

    fit = _fit_partial_reco_components(bcorr)
    log.info(
        "  Window summary [%s]: observed=%d, signal=%.1f, partial=%.1f, combinatorial=%.1f",
        cat,
        fit["observed_window"],
        fit["signal_window"],
        fit["partial_window"],
        fit["comb_window"],
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.2), layout="constrained")
    ax.errorbar(
        fit["centres"],
        fit["counts"],
        yerr=fit["errors"],
        fmt="o",
        color="#333333",
        ecolor="#333333",
        markersize=3.0,
        elinewidth=1.0,
        alpha=0.85,
        label="Data",
        zorder=5,
    )
    ax.plot(
        fit["grid"],
        fit["total_curve"] * fit["bin_width"],
        color="#0044AA",
        lw=2.0,
        label="Total fit",
    )
    ax.plot(
        fit["grid"],
        fit["signal_curve"] * fit["bin_width"],
        color="#117733",
        lw=1.8,
        label=r"$B^+$ peak",
    )
    ax.plot(
        fit["grid"],
        fit["partial_curve"] * fit["bin_width"],
        color="#D35400",
        lw=1.8,
        linestyle="--",
        label="Partial reco",
    )
    ax.plot(
        fit["grid"],
        fit["comb_curve"] * fit["bin_width"],
        color="0.35",
        lw=1.5,
        linestyle=":",
        label="Comb. BKG",
    )

    ax.axvspan(BU_CORR_MIN, BU_CORR_MAX, alpha=0.10, color="#003366")
    ax.axvline(BU_CORR_MIN, color="#003366", linestyle="--", linewidth=1.0)
    ax.axvline(BU_CORR_MAX, color="#003366", linestyle="--", linewidth=1.0)
    ax.set_xlim(PARTIAL_RECO_FIT_MIN, PARTIAL_RECO_FIT_MAX)
    ax.set_xlabel(r"$m_{\rm corr}(\bar{\Lambda}pK^+K^-)$ [MeV/$c^2$]")
    ax.set_ylabel("Candidates / (5 MeV/$c^2$)")

    lcat = "LL" if cat == "LL" else "DD"
    ax.text(
        0.03,
        0.97,
        rf"$\Lambda_{{\rm {lcat}}}$, high-yield MVA",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
    )
    ax.text(
        0.03,
        0.87,
        "Wide spectrum before nominal $B^+$ window",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="dimgray",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.80),
    )
    ax.text(
        0.97,
        0.97,
        "Nominal $B^+$ window",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color="#003366",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
    )
    ax.text(
        0.97,
        0.70,
        (
            f"Observed in window: {fit['observed_window']:.0f}\n"
            f"Peak fit: {fit['signal_window']:.0f}\n"
            f"Partial reco: {fit['partial_window']:.0f}\n"
            f"Partial fraction: {100.0 * fit['partial_fraction_window']:.1f}%"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.92),
    )
    ax.legend(fontsize=9, frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(0.98, 0.56))

    out = figs_path(cat, "backgrounds", "partial_reco.pdf")
    save_fig(fig, out)
    log.info(f"  Saved: {out}")

    summary_path = figs_path(cat, "backgrounds", "partial_reco_summary.json")
    summary = {
        "category": cat,
        "fit_range": [PARTIAL_RECO_FIT_MIN, PARTIAL_RECO_FIT_MAX],
        "window": [BU_CORR_MIN, BU_CORR_MAX],
        "observed_window": int(fit["observed_window"]),
        "signal_window": float(fit["signal_window"]),
        "partial_window": float(fit["partial_window"]),
        "comb_window": float(fit["comb_window"]),
        "partial_fraction_window": float(fit["partial_fraction_window"]),
        "chi2_ndf": float(fit["chi2_ndf"]),
        "params": {
            "n_sig": float(fit["params"][0]),
            "mu": float(fit["params"][1]),
            "sigma": float(fit["params"][2]),
            "n_part": float(fit["params"][3]),
            "shape_part": float(fit["params"][4]),
            "n_comb": float(fit["params"][5]),
            "slope_comb": float(fit["params"][6]),
        },
        "param_errors": {
            "n_sig": float(fit["param_errors"][0]),
            "mu": float(fit["param_errors"][1]),
            "sigma": float(fit["param_errors"][2]),
            "n_part": float(fit["param_errors"][3]),
            "shape_part": float(fit["param_errors"][4]),
            "n_comb": float(fit["param_errors"][5]),
            "slope_comb": float(fit["param_errors"][6]),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"  Saved: {summary_path}")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════


def main() -> None:
    for cat in ("LL", "DD"):
        log.info(f"\n=== Lambda{cat} ===")
        log.info("--- K_S^0 contamination study ---")
        plot_ks0_veto(cat)
        plot_ks0_veto_before_after(cat)
        log.info("--- Non-resonant shape ---")
        plot_nonresonant_shape(cat)
        log.info("--- Partially reconstructed background ---")
        plot_partial_reco(cat)
    log.info("\n=== All background studies done ===")


if __name__ == "__main__":
    main()
