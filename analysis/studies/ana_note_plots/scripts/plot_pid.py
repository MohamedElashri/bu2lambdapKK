"""
PID comparison plots for the B+ → Λ̄pK⁻K⁺ analysis note.

Produces individual PDFs:
  figs/LambdaLL/pidcmp.pdf    — product PID (all tracks)
  figs/LambdaDD/pidcmp.pdf
  figs/LambdaLL/pidcmp_p.pdf  — bachelor proton ProbNNp
  figs/LambdaDD/pidcmp_p.pdf
  figs/LambdaLL/pidcmp_k1.pdf — h1 kaon ProbNNk
  figs/LambdaDD/pidcmp_k1.pdf
  figs/LambdaLL/pidcmp_k2.pdf — h2 kaon ProbNNk
  figs/LambdaDD/pidcmp_k2.pdf

Strategy: compare data in J/ψ signal window (|m(ΛpK)−3096.9|<30, B+ signal window)
against J/ψ MC (same Lambda mass window). Both normalised to unit area.
No sideband subtraction — the J/ψ window is signal-dominated.

Run from analysis/ directory:
    uv run python studies/ana_note_plots/scripts/plot_pid.py
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot

SCRIPTS_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPTS_DIR.resolve().parents[3]
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.resolve().parents[2]))  # analysis/ for modules.*

from modules.plot_utils import BINNING, COLORS, figs_path, save_fig, setup_style

DATA_BASE = Path("/share/lazy/Mohamed/Bu2LambdaPPP/files/data")
MC_BASE = Path("/share/lazy/Mohamed/Bu2LambdaPPP/files/mc")

M_LAMBDA_PDG = 1115.683
M_BPLUS = 5279.6
M_JPSI = 3096.9
YEARS = ["16", "17", "18"]
MAGNETS = ["MD", "MU"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

setup_style()


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADER — J/ψ window + B+ signal window
# ════════════════════════════════════════════════════════════════════════════════


def _load_data_pid(path: Path, cat: str) -> dict:
    """
    Load data PID variables selected by:
      - Trigger (L0 TIS, HLT1 TOS, HLT2 TOS)
      - Lambda mass window [1108, 1126] MeV
      - B+ corrected mass signal window [5255, 5305] MeV
      - J/ψ mass window |m(Λ̄pK⁻) − 3096.9| < 30 MeV

    The charmonium mass is m(L0 + p + h2), consistent with M_LpKm_h2 in the
    pipeline (h2 = K⁻). The B+ mass window is required: without it only ~6% of
    J/ψ-window events are genuine B+ decays; the rest are combinatorial background
    with random (near-zero) PID that creates a spurious peak at zero.
    """
    pid_data = [
        "p_MC15TuneV1_ProbNNp",
        "h1_MC15TuneV1_ProbNNk",
        "h2_MC15TuneV1_ProbNNk",
    ]
    mom_branches = [
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
    ]
    trig_branches = [
        "Bu_L0GlobalDecision_TIS",
        "Bu_L0PhysDecision_TIS",
        "Bu_L0HadronDecision_TIS",
        "Bu_Hlt1TrackMVADecision_TOS",
        "Bu_Hlt1TwoTrackMVADecision_TOS",
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
    ]
    want = pid_data + mom_branches + trig_branches + ["Bu_MM", "L0_MM"]

    tree = uproot.open(path)[f"B2L0barPKpKm_{cat}/DecayTree"]
    avail = [b for b in want if b in tree.keys()]
    ev = tree.arrays(avail, library="np")
    n = len(ev["L0_MM"])

    # Trigger mask
    def _or(keys, default):
        m = np.zeros(n, dtype=bool)
        found = False
        for k in keys:
            if k in avail:
                m |= ev[k] > 0
                found = True
        return m if found else np.ones(n, dtype=bool)

    mask = (
        _or(["Bu_L0GlobalDecision_TIS", "Bu_L0PhysDecision_TIS", "Bu_L0HadronDecision_TIS"], True)
        & _or(["Bu_Hlt1TrackMVADecision_TOS", "Bu_Hlt1TwoTrackMVADecision_TOS"], True)
        & _or(
            [
                "Bu_Hlt2Topo2BodyDecision_TOS",
                "Bu_Hlt2Topo3BodyDecision_TOS",
                "Bu_Hlt2Topo4BodyDecision_TOS",
            ],
            True,
        )
    )

    # Lambda mass window
    mask &= (ev["L0_MM"] > 1108) & (ev["L0_MM"] < 1126)

    # B+ corrected mass signal window
    bu_corr = ev["Bu_MM"] - ev["L0_MM"] + M_LAMBDA_PDG
    mask &= (bu_corr >= 5255) & (bu_corr <= 5305)

    ev = {k: v[mask] for k, v in ev.items()}
    n = len(ev["L0_MM"])

    # Charmonium mass m(Λ̄pK⁻) = m(L0 + p + h2)
    mom_ok = all(k in avail for k in mom_branches)
    if not mom_ok or n == 0:
        return {}

    E = ev["L0_PE"] + ev["p_PE"] + ev["h2_PE"]
    px = ev["L0_PX"] + ev["p_PX"] + ev["h2_PX"]
    py = ev["L0_PY"] + ev["p_PY"] + ev["h2_PY"]
    pz = ev["L0_PZ"] + ev["p_PZ"] + ev["h2_PZ"]
    ccbar_M = np.sqrt(np.maximum(E**2 - px**2 - py**2 - pz**2, 0.0))

    # J/ψ mass window
    jpsi_mask = np.abs(ccbar_M - M_JPSI) < 30.0

    result = {}
    for b in pid_data:
        if b in avail:
            result[b] = ev[b][jpsi_mask]
    return result


def _collect_data_pid(cat: str) -> dict:
    combined = {}
    for yr in YEARS:
        for mag in MAGNETS:
            p = DATA_BASE / f"dataBu2L0barPHH_{yr}{mag}.root"
            if not p.exists():
                continue
            try:
                d = _load_data_pid(p, cat)
                for branch, arr in d.items():
                    combined.setdefault(branch, []).append(arr)
            except Exception as e:
                log.warning(f"  Skip data {p}: {e}")
    return {b: np.concatenate(v) for b, v in combined.items()}


# ════════════════════════════════════════════════════════════════════════════════
# MC LOADER — J/ψ MC, Lambda window only
# ════════════════════════════════════════════════════════════════════════════════


def _load_mc_pid(path: Path, cat: str) -> dict:
    """Load MC PID arrays with the same selection as data:
    trigger + Lambda mass + B+ corrected mass signal window + J/ψ window on m(L0+p+h2).
    MC is pure J/ψ signal so the J/ψ window retains almost all events, but applying
    it keeps the comparison apples-to-apples with the data selection.
    """
    pid_mc = [
        "p_MC15TuneV1_ProbNNp",
        "h1_MC15TuneV1_ProbNNk",
        "h2_MC15TuneV1_ProbNNk",
    ]
    mom_branches = [
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
    ]
    trig = [
        "Bu_L0Global_TIS",
        "Bu_L0HadronDecision_TIS",
        "Bu_Hlt1TrackMVADecision_TOS",
        "Bu_Hlt1TwoTrackMVADecision_TOS",
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
    ]
    want = pid_mc + mom_branches + trig + ["Bu_MM", "L0_MM"]
    tree = uproot.open(path)[f"B2L0barPKpKm_{cat}/DecayTree"]
    avail = [b for b in want if b in tree.keys()]
    ev = tree.arrays(avail, library="np")
    n = len(ev["L0_MM"])

    def _or(keys):
        m = np.zeros(n, dtype=bool)
        found = False
        for k in keys:
            if k in avail:
                m |= ev[k] > 0
                found = True
        return m if found else np.ones(n, dtype=bool)

    mask = (
        _or(["Bu_L0Global_TIS", "Bu_L0HadronDecision_TIS"])
        & _or(["Bu_Hlt1TrackMVADecision_TOS", "Bu_Hlt1TwoTrackMVADecision_TOS"])
        & _or(
            [
                "Bu_Hlt2Topo2BodyDecision_TOS",
                "Bu_Hlt2Topo3BodyDecision_TOS",
                "Bu_Hlt2Topo4BodyDecision_TOS",
            ]
        )
    )
    mask &= (ev["L0_MM"] > 1108) & (ev["L0_MM"] < 1126)

    # B+ corrected mass signal window — same as data
    if "Bu_MM" in ev:
        bu_corr = ev["Bu_MM"] - ev["L0_MM"] + M_LAMBDA_PDG
        mask &= (bu_corr >= 5255) & (bu_corr <= 5305)

    # J/ψ window on m(L0 + p + h2) — consistent with data and pipeline M_LpKm_h2
    mom_ok = all(k in avail for k in mom_branches)
    if mom_ok:
        ev_m = {k: v[mask] for k, v in ev.items()}
        E = ev_m["L0_PE"] + ev_m["p_PE"] + ev_m["h2_PE"]
        px = ev_m["L0_PX"] + ev_m["p_PX"] + ev_m["h2_PX"]
        py = ev_m["L0_PY"] + ev_m["p_PY"] + ev_m["h2_PY"]
        pz = ev_m["L0_PZ"] + ev_m["p_PZ"] + ev_m["h2_PZ"]
        ccbar_M = np.sqrt(np.maximum(E**2 - px**2 - py**2 - pz**2, 0.0))
        jpsi_mask = np.abs(ccbar_M - M_JPSI) < 30.0
        result = {}
        for b in pid_mc:
            if b in avail:
                result[b] = ev_m[b][jpsi_mask]
        return result

    # Fallback: no momentum branches — return trigger+Lambda-filtered events
    result = {}
    for b in pid_mc:
        if b in avail:
            result[b] = ev[b][mask]
    return result


def _collect_mc_pid(cat: str) -> dict:
    combined = {}
    for yr in YEARS:
        for mag in MAGNETS:
            p = MC_BASE / "Jpsi" / f"Jpsi_{yr}_{mag}.root"
            if not p.exists():
                continue
            try:
                d = _load_mc_pid(p, cat)
                for b, arr in d.items():
                    combined.setdefault(b, []).append(arr)
            except Exception as e:
                log.warning(f"  Skip MC {p}: {e}")
    return {b: np.concatenate(v) for b, v in combined.items()}


# ════════════════════════════════════════════════════════════════════════════════
# PLOT
# ════════════════════════════════════════════════════════════════════════════════


def _pid_plot(
    cat: str,
    data_arr: np.ndarray,
    mc_arr: np.ndarray,
    xlabel: str,
    bins: int,
    x_range: list,
    log_y: bool,
    outfile: str,
):
    """One PID comparison PDF: normalised data (error bars) vs MC (step histogram)."""
    if len(data_arr) == 0 or len(mc_arr) == 0:
        log.warning(f"  Empty arrays for {outfile} [{cat}], skipping")
        return

    in_range_d = data_arr[(data_arr >= x_range[0]) & (data_arr <= x_range[1])]
    in_range_m = mc_arr[(mc_arr >= x_range[0]) & (mc_arr <= x_range[1])]

    bw = (x_range[1] - x_range[0]) / bins
    h_d, edges = np.histogram(in_range_d, range=x_range, bins=bins)
    h_m, _ = np.histogram(in_range_m, range=x_range, bins=bins)

    total_d = h_d.sum() * bw
    total_m = h_m.sum() * bw
    if total_d <= 0 or total_m <= 0:
        log.warning(f"  Empty histogram for {outfile} [{cat}], skipping")
        return

    h_d_n = h_d / total_d
    h_m_n = h_m / total_m
    # Standard Poisson error: σ = √N / (N_total × bin_width).
    # Using √(N+1) was wrong — it inflates errors by 41% for N=1 bins.
    err_d_n = np.sqrt(h_d) / total_d
    centers = (edges[1:] + edges[:-1]) / 2
    mask = h_d > 0

    fig, ax = plt.subplots()

    # Data — error bars, non-empty bins only
    ax.errorbar(
        centers[mask],
        h_d_n[mask],
        yerr=err_d_n[mask],
        label=r"$J/\psi$ window data",
        ecolor="black",
        mfc="black",
        color="black",
        elinewidth=1.5,
        markersize=4,
        marker="o",
        fmt=" ",
    )

    # MC — step dashed histogram
    ax.hist(
        in_range_m,
        range=x_range,
        bins=bins,
        weights=np.full(len(in_range_m), 1.0 / total_m),
        label=r"$J/\psi$ MC",
        color=COLORS[0],
        histtype="step",
        linestyle="--",
        linewidth=3,
    )

    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(rf"Normalised / ({bw:.2f})")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(*x_range)

    out = figs_path(cat, outfile)
    save_fig(fig, out)
    log.info(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════


def main():
    for cat in ("LL", "DD"):
        log.info(f"=== Category: Lambda{cat} ===")

        data = _collect_data_pid(cat)
        mc = _collect_mc_pid(cat)

        if not data or not mc:
            log.warning(f"  Missing data or MC for {cat}, skipping")
            continue

        log.info(f"  Data (J/ψ signal window): " f"{len(next(iter(data.values())))} events")
        log.info(f"  MC: {len(next(iter(mc.values())))} events")

        # Compute PID product arrays
        d_p = "p_MC15TuneV1_ProbNNp"
        d_h1 = "h1_MC15TuneV1_ProbNNk"
        d_h2 = "h2_MC15TuneV1_ProbNNk"
        m_p = "p_MC15TuneV1_ProbNNp"
        m_h1 = "h1_MC15TuneV1_ProbNNk"
        m_h2 = "h2_MC15TuneV1_ProbNNk"

        if all(b in data for b in [d_p, d_h1, d_h2]) and all(b in mc for b in [m_p, m_h1, m_h2]):
            data_prod = data[d_p] * data[d_h1] * data[d_h2]
            mc_prod = mc[m_p] * mc[m_h1] * mc[m_h2]
            _pid_plot(
                cat,
                data_prod,
                mc_prod,
                xlabel=r"$p \cdot \mathrm{ProbNNp} \times h_1 \cdot \mathrm{ProbNNk}"
                r"\times h_2 \cdot \mathrm{ProbNNk}$",
                bins=10,
                x_range=list(BINNING["pid"]["range"]),
                log_y=False,
                outfile="pidcmp.pdf",
            )

        # Individual tracks — use 10 bins (not the default 20) so that each bin
        # has ~9 events for LL (91 events) and ~18 for DD (180 events).
        # 20 bins gives only ~4.5 events/bin for LL → errors ≥38% in most bins.
        _nb = 10
        _xr = list(BINNING["pid"]["range"])
        configs = [
            (d_p, m_p, r"ProbNNp (bachelor $p$)", _nb, _xr, True, "pidcmp_p.pdf"),
            (d_h1, m_h1, r"ProbNNk ($h_1$ kaon)", _nb, _xr, True, "pidcmp_k1.pdf"),
            (d_h2, m_h2, r"ProbNNk ($h_2$ kaon)", _nb, _xr, True, "pidcmp_k2.pdf"),
        ]
        for db, mb, xlabel, bins, xr, logy, fname in configs:
            if db in data and mb in mc:
                _pid_plot(cat, data[db], mc[mb], xlabel, bins, xr, logy, fname)

    log.info("=== Done. ===")


if __name__ == "__main__":
    main()
