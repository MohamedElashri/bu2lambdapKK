"""
Charmonium mass spectrum — the money plot for the B+ → Λ̄pK⁻K⁺ analysis note.

Produces:
  figs/LambdaLL/charmonium_spectrum.pdf
  figs/LambdaDD/charmonium_spectrum.pdf

Shows m(Λ̄pK⁻) for events in the B+ signal window [5255, 5305] MeV after all
selection cuts (trigger + Λ mass + Set 1 optimal presel).  A scaled sideband
estimate of the combinatorial background is overlaid as a dashed histogram.
Vertical lines label each charmonium state.

Run from analysis/ directory:
    uv run python studies/ana_note_plots/scripts/plot_charmonium_spectrum.py
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

from modules.plot_utils import COLORS, figs_path, save_fig, setup_style

DATA_BASE = Path("/share/lazy/Mohamed/Bu2LambdaPPP/files/data")

M_LAMBDA_PDG = 1115.683
M_BPLUS = 5279.6
YEARS = ["16", "17", "18"]
MAGNETS = ["MD", "MU"]

# B+ windows (MeV, using corrected mass)
SIG_LO, SIG_HI = 5255, 5305  # signal window (50 MeV wide)
SB1_LO, SB1_HI = 5150, 5230  # lower sideband (80 MeV)
SB2_LO, SB2_HI = 5330, 5410  # upper sideband (80 MeV)
SB_TOTAL_WIDTH = (SB1_HI - SB1_LO) + (SB2_HI - SB2_LO)  # 160 MeV
SIG_WIDTH = SIG_HI - SIG_LO  # 50 MeV
SB_SCALE = SIG_WIDTH / SB_TOTAL_WIDTH  # 50/160

# Charmonium spectrum range and binning
CC_LO, CC_HI = 2700, 4100
CC_BINS = 70  # 20 MeV/bin

# Charmonium masses and labels (MeV/c²)
CC_STATES = [
    (2980.3, r"$\eta_c$", "bottom"),
    (3096.9, r"$J/\psi$", "bottom"),
    (3414.7, r"$\chi_{c0}$", "bottom"),
    (3510.7, r"$\chi_{c1}$", "top"),
    (3638.5, r"$\eta_c(2S)$", "bottom"),
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

setup_style()


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ════════════════════════════════════════════════════════════════════════════════


def _load_events(path: Path, cat: str) -> dict:
    """
    Load m(Λ̄pK⁻) for events passing trigger + Λ mass + Set 1 preselection.
    Returns arrays split by B+ corrected mass region (signal / sideband).
    """
    want = [
        # 4-momenta for charmonium mass reconstruction
        "L0_PE",
        "L0_PX",
        "L0_PY",
        "L0_PZ",
        "p_PE",
        "p_PX",
        "p_PY",
        "p_PZ",
        "h1_PE",
        "h1_PX",
        "h1_PY",
        "h1_PZ",
        # Lambda + B+ mass for selections
        "L0_MM",
        "Bu_MM",
        # Trigger
        "Bu_L0GlobalDecision_TIS",
        "Bu_L0PhysDecision_TIS",
        "Bu_L0HadronDecision_TIS",
        "Bu_Hlt1TrackMVADecision_TOS",
        "Bu_Hlt1TwoTrackMVADecision_TOS",
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
        # Set 1 preselection variables
        "p_MC15TuneV1_ProbNNp",
        "h1_MC15TuneV1_ProbNNk",
        "h2_MC15TuneV1_ProbNNk",
        "Bu_FDCHI2_OWNPV",
        "Bu_PT",
        "Bu_IPCHI2_OWNPV",
    ]

    tree = uproot.open(path)[f"B2L0barPKpKm_{cat}/DecayTree"]
    avail = [b for b in want if b in tree.keys()]
    ev = tree.arrays(avail, library="np")
    n = len(ev["L0_MM"])

    # ── Trigger mask ──────────────────────────────────────────────────────────
    l0_keys = ["Bu_L0GlobalDecision_TIS", "Bu_L0PhysDecision_TIS", "Bu_L0HadronDecision_TIS"]
    hlt1_keys = ["Bu_Hlt1TrackMVADecision_TOS", "Bu_Hlt1TwoTrackMVADecision_TOS"]
    hlt2_keys = [
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
    ]

    def _or(keys):
        m = np.zeros(n, dtype=bool)
        found = False
        for k in keys:
            if k in avail:
                m |= ev[k] > 0
                found = True
        return m if found else np.ones(n, dtype=bool)

    mask = _or(l0_keys) & _or(hlt1_keys) & _or(hlt2_keys)

    # ── Lambda mass window ────────────────────────────────────────────────────
    mask &= (ev["L0_MM"] > 1108) & (ev["L0_MM"] < 1126)

    # ── Set 1 preselection (optimal cuts from fit_based_optimizer) ────────────
    pid_p = ev.get("p_MC15TuneV1_ProbNNp", np.ones(n))
    pid_h1 = ev.get("h1_MC15TuneV1_ProbNNk", np.ones(n))
    pid_h2 = ev.get("h2_MC15TuneV1_ProbNNk", np.ones(n))
    mask &= pid_p * pid_h1 * pid_h2 > 0.20
    if "Bu_FDCHI2_OWNPV" in avail:
        mask &= ev["Bu_FDCHI2_OWNPV"] > 250
    if "Bu_PT" in avail:
        mask &= ev["Bu_PT"] > 3400
    if "Bu_IPCHI2_OWNPV" in avail:
        mask &= ev["Bu_IPCHI2_OWNPV"] < 7.0

    ev = {k: v[mask] for k, v in ev.items()}

    # ── B+ corrected mass ─────────────────────────────────────────────────────
    bu_corr = ev["Bu_MM"] - ev["L0_MM"] + M_LAMBDA_PDG

    # ── Charmonium mass m(Λ̄pK⁻) = m(L0 + p + h1) ────────────────────────────
    E = ev["L0_PE"] + ev["p_PE"] + ev["h1_PE"]
    px = ev["L0_PX"] + ev["p_PX"] + ev["h1_PX"]
    py = ev["L0_PY"] + ev["p_PY"] + ev["h1_PY"]
    pz = ev["L0_PZ"] + ev["p_PZ"] + ev["h1_PZ"]
    cc_M = np.sqrt(np.maximum(E**2 - px**2 - py**2 - pz**2, 0.0))

    sig_mask = (bu_corr >= SIG_LO) & (bu_corr <= SIG_HI)
    sb_mask = ((bu_corr >= SB1_LO) & (bu_corr <= SB1_HI)) | (
        (bu_corr >= SB2_LO) & (bu_corr <= SB2_HI)
    )

    return {"signal": cc_M[sig_mask], "sideband": cc_M[sb_mask]}


def _collect(cat: str) -> dict:
    sig_all, sb_all = [], []
    for yr in YEARS:
        for mag in MAGNETS:
            p = DATA_BASE / f"dataBu2L0barPHH_{yr}{mag}.root"
            if not p.exists():
                continue
            try:
                d = _load_events(p, cat)
                sig_all.append(d["signal"])
                sb_all.append(d["sideband"])
                log.info(f"  {p.name} [{cat}]: sig={len(d['signal'])}, sb={len(d['sideband'])}")
            except Exception as e:
                log.warning(f"  Skip {p.name} [{cat}]: {e}")
    return {
        "signal": np.concatenate(sig_all) if sig_all else np.array([]),
        "sideband": np.concatenate(sb_all) if sb_all else np.array([]),
    }


# ════════════════════════════════════════════════════════════════════════════════
# PLOT
# ════════════════════════════════════════════════════════════════════════════════


def plot_spectrum(cat: str, data: dict):
    sig = data["signal"]
    sb = data["sideband"]

    if len(sig) == 0:
        log.warning(f"  No signal events for {cat}, skipping")
        return

    log.info(f"  [{cat}] signal window: {len(sig)} events, sideband: {len(sb)} events")

    x_range = (CC_LO, CC_HI)
    bins = CC_BINS
    bw = (CC_HI - CC_LO) / CC_BINS

    h_sig, edges = np.histogram(sig, range=x_range, bins=bins)

    h_sb = np.zeros(bins)
    if len(sb) > 0:
        h_sb, _ = np.histogram(sb, range=x_range, bins=bins)
        h_sb = h_sb * SB_SCALE  # scale to signal window area

    centers = (edges[1:] + edges[:-1]) / 2
    err_sig = np.sqrt(h_sig + 1)

    fig, ax = plt.subplots(figsize=(8, 5.2))

    # ── Signal window data (error bars) ──────────────────────────────────────
    mask = h_sig > 0
    ax.errorbar(
        centers[mask],
        h_sig[mask],
        yerr=err_sig[mask],
        label=r"$B^+$ signal window data",
        ecolor="black",
        mfc="black",
        color="black",
        elinewidth=1.5,
        markersize=4,
        marker="o",
        fmt=" ",
    )

    # ── Scaled sideband (background estimate) ────────────────────────────────
    if len(sb) > 0:
        ax.hist(
            sb,
            range=x_range,
            bins=bins,
            weights=np.full(len(sb), SB_SCALE),
            label=rf"Sideband estimate ($\times {SB_SCALE:.3f}$)",
            histtype="step",
            linestyle="--",
            linewidth=2,
            color=COLORS[0],
        )

    ax.set_xlabel(r"$m(\bar{\Lambda}pK^-)$ [MeV/$c^2$]")
    ax.set_ylabel(rf"Candidates / ({int(bw)} MeV/$c^2$)")
    ax.set_xlim(*x_range)
    ax.set_ylim(bottom=0)
    # Legend above the axes so it never overlaps state labels or data
    ax.legend(frameon=False, fontsize=12, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.13))

    # ── Charmonium state markers ──────────────────────────────────────────────
    # Use axes-fraction y so labels are always just inside the top edge.
    # χc0 (3415) and χc1 (3511) are close: alternate at 0.95 / 0.85 fraction.
    label_frac = {3414.7: 0.95, 3510.7: 0.83}
    for mass, label, _ in CC_STATES:
        if CC_LO < mass < CC_HI:
            ax.axvline(mass, color="gray", linestyle=":", linewidth=1.2, alpha=0.8)
            frac = label_frac.get(mass, 0.95)
            ax.text(
                mass,
                frac,
                label,
                transform=ax.get_xaxis_transform(),  # x=data, y=axes fraction
                ha="center",
                va="top",
                fontsize=10,
                color="dimgray",
                rotation=90,
            )

    out = figs_path(cat, "charmonium_spectrum.pdf")
    save_fig(fig, out)
    log.info(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════


def main():
    for cat in ("LL", "DD"):
        log.info(f"=== Category: Lambda{cat} ===")
        data = _collect(cat)
        plot_spectrum(cat, data)
    log.info("=== Done. ===")


if __name__ == "__main__":
    main()
