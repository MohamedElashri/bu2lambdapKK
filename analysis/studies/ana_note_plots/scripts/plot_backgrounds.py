"""
Background study plots for the B+ → Λ̄pK⁻K⁺ analysis note.

Produces:
  figs/LambdaLL/misID_LL_beforePID.pdf   — B+ mass of background MC before PID
  figs/LambdaDD/misID_DD_beforePID.pdf
  figs/LambdaLL/misID_LL_afterPID.pdf    — B+ mass of background MC after PID
  figs/LambdaDD/misID_DD_afterPID.pdf

Background MC available:
  KpKm — B+ → Λ̄pK⁻K⁺ non-resonant (phase space)
  KpKp — B+ → Λ̄pK⁺K⁺ (same-sign kaons, mis-ID background)

Run from analysis/ directory:
    uv run python studies/ana_note_plots/scripts/plot_backgrounds.py
"""

import logging
import sys
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPTS_DIR.resolve().parents[3]  # bu2lambdapKK root
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

from plot_helpers import COLORS, figs_path, save_fig, setup_style

DATA_BASE = Path("/share/lazy/Mohamed/Bu2LambdaPPP/files/data")
MC_BASE = Path("/share/lazy/Mohamed/Bu2LambdaPPP/files/mc")

M_LAMBDA_PDG = 1115.683
YEARS = ["16", "17", "18"]
MAGNETS = ["MD", "MU"]

# PID product cut for "afterPID" variants
PID_CUT = 0.20

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

setup_style()


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ════════════════════════════════════════════════════════════════════════════════


def _trigger_mask_mc(ev: dict, n: int) -> np.ndarray:
    """Return trigger mask for MC events."""
    l0_keys = ["Bu_L0Global_TIS", "Bu_L0HadronDecision_TIS"]
    hlt1_keys = ["Bu_Hlt1TrackMVADecision_TOS", "Bu_Hlt1TwoTrackMVADecision_TOS"]
    hlt2_keys = [
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
    ]
    ml0, mhlt1, mhlt2 = (np.zeros(n, dtype=bool) for _ in range(3))
    for k in l0_keys:
        if k in ev:
            ml0 = ml0 | (ev[k] > 0)
    if not any(k in ev for k in l0_keys):
        ml0[:] = True
    for k in hlt1_keys:
        if k in ev:
            mhlt1 = mhlt1 | (ev[k] > 0)
    if not any(k in ev for k in hlt1_keys):
        mhlt1[:] = True
    for k in hlt2_keys:
        if k in ev:
            mhlt2 = mhlt2 | (ev[k] > 0)
    if not any(k in ev for k in hlt2_keys):
        mhlt2[:] = True
    return ml0 & mhlt1 & mhlt2


def _load_mc_misid(path: Path, cat: str, after_pid: bool = False) -> np.ndarray:
    """Load background MC Bu_DTFL0_M after trigger + Lambda cuts + optional PID."""
    want_scalar = [
        "L0_MM",
        "p_MC12TuneV4_ProbNNp",
        "h1_MC12TuneV4_ProbNNk",
        "h2_MC12TuneV4_ProbNNk",
        "Bu_L0Global_TIS",
        "Bu_L0HadronDecision_TIS",
        "Bu_Hlt1TrackMVADecision_TOS",
        "Bu_Hlt1TwoTrackMVADecision_TOS",
        "Bu_Hlt2Topo2BodyDecision_TOS",
        "Bu_Hlt2Topo3BodyDecision_TOS",
        "Bu_Hlt2Topo4BodyDecision_TOS",
    ]
    tree = uproot.open(path)[f"B2L0barPKpKm_{cat}/DecayTree"]
    avail_sc = [b for b in want_scalar if b in tree.keys()]
    ev = tree.arrays(avail_sc, library="np")

    # Load Bu_DTFL0_M separately (jagged) via awkward
    bu_dtf_raw = tree.arrays(["Bu_DTFL0_M"], library="ak")["Bu_DTFL0_M"]
    bu_dtf = ak.to_numpy(ak.firsts(bu_dtf_raw))

    mask = _trigger_mask_mc(ev, len(ev["L0_MM"]))
    mask = mask & (ev["L0_MM"] > 1108) & (ev["L0_MM"] < 1126)

    if after_pid:
        pid_prod = np.ones(len(ev["L0_MM"]))
        for b in ["p_MC12TuneV4_ProbNNp", "h1_MC12TuneV4_ProbNNk", "h2_MC12TuneV4_ProbNNk"]:
            if b in avail_sc:
                pid_prod = pid_prod * ev[b]
        mask = mask & (pid_prod > PID_CUT)

    return bu_dtf[mask]


def _collect_mc_misid(mode_dir: str, cat: str, after_pid: bool) -> np.ndarray:
    """Collect background MC from all years/magnets."""
    arrs = []
    d = MC_BASE / mode_dir
    for yr in YEARS:
        for mag in MAGNETS:
            p = d / f"{mode_dir}_{yr}_{mag}.root"
            if not p.exists():
                continue
            try:
                arrs.append(_load_mc_misid(p, cat, after_pid))
            except Exception as e:
                log.warning(f"  Skip {p}: {e}")
    return np.concatenate(arrs) if arrs else np.array([])


# ════════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════


def plot_misid(cat: str, after_pid: bool):
    """
    Mis-ID background: B+ DTF mass for phase-space MC (KpKm + KpKp) on log scale.
    Matches reference background.ipynb cell 6/9.
    """
    tag = "afterPID" if after_pid else "beforePID"
    outfile = f"misID_{cat}_{tag}.pdf"
    log.info(f"  misID [{cat}] {tag}")

    mc_kpkm = _collect_mc_misid("KpKm", cat, after_pid)
    mc_kpkp = _collect_mc_misid("KpKp", cat, after_pid)

    if len(mc_kpkm) == 0 and len(mc_kpkp) == 0:
        log.warning(f"  No MC found for {outfile}")
        return

    log.info(f"  KpKm MC: {len(mc_kpkm)}  KpKp MC: {len(mc_kpkp)}")

    x_range = [4700, 7000]
    bins = 150

    fig, ax = plt.subplots()
    ax.set_yscale("log")

    histstyle = {"bins": bins, "range": x_range, "histtype": "step"}
    if len(mc_kpkm) > 0:
        ax.hist(
            mc_kpkm,
            color=COLORS[2],
            label=r"$B^+\to \bar{\Lambda}pK^+K^-$ (phase-space MC)",
            linewidth=2,
            **histstyle,
        )
    if len(mc_kpkp) > 0:
        ax.hist(
            mc_kpkp,
            color=COLORS[3],
            label=r"$B^+\to \bar{\Lambda}pK^+K^+$ (same-sign MC)",
            linewidth=2,
            linestyle="--",
            **histstyle,
        )

    ax.set_xlabel(r"$m(\bar{\Lambda}pK^+K^-)_{\rm DTF}$ [MeV/$c^2$]")
    ax.set_ylabel("Events")
    ax.set_ylim(bottom=0.1)
    ax.set_title(rf"$\Lambda_{{{cat}}}$ sample, {tag}")
    ax.legend(fontsize=8, frameon=False)

    out = figs_path(cat, outfile)
    save_fig(fig, out)
    log.info(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════


def main():
    for cat in ("LL", "DD"):
        log.info(f"=== Category: Lambda{cat} ===")
        plot_misid(cat, after_pid=False)
        plot_misid(cat, after_pid=True)
    log.info("=== Done. ===")


if __name__ == "__main__":
    main()
