"""
Charmonium-spectrum plots for the B+ → Λ̄pK⁻K⁺ analysis.

Produces:
  figs/LambdaLL/charmonium_spectrum.pdf
  figs/LambdaDD/charmonium_spectrum.pdf
  figs/LambdaLL/backgrounds/charmonium_sideband.pdf
  figs/LambdaDD/backgrounds/charmonium_sideband.pdf

The main note plot shows m(Λ̄pK⁻) for events in the B+ signal window
[5255, 5305] MeV after the shared current preselection, with the scaled B+ sideband
estimate overlaid.

The background diagnostic plot shows only the B+ sideband data in m(Λ̄pK⁻).
It is used to verify that the combinatorial B+ sidebands do not contain
peaking charmonium structure.

Run from analysis/ directory:
    uv run python studies/ana_note_plots/scripts/plot_charmonium_spectrum.py
"""

import logging
import sys
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPTS_DIR.resolve().parents[3]
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.resolve().parents[2]))  # analysis/ for modules.*

from modules.clean_data_loader import load_all_data
from modules.plot_utils import COLORS, figs_path, save_fig, setup_style

DATA_BASE = Path("/share/lazy/Mohamed/Bu2LambdaPPP/files/data")

M_LAMBDA_PDG = 1115.683
YEARS = [2016, 2017, 2018]

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
def _collect(cat: str) -> dict:
    data_dict = load_all_data(DATA_BASE, years=YEARS, track_types=[cat])
    arrays = [data_dict[str(year)][cat] for year in YEARS if cat in data_dict.get(str(year), {})]
    combined = ak.concatenate(arrays) if arrays else ak.Array([])
    if len(combined) == 0:
        return {"signal": np.array([]), "sideband": np.array([])}

    bu_corr = ak.to_numpy(combined["Bu_MM_corrected"])
    cc_mass = ak.to_numpy(combined["M_LpKm_h2"])
    sig_mask = (bu_corr >= SIG_LO) & (bu_corr <= SIG_HI)
    sb_mask = ((bu_corr >= SB1_LO) & (bu_corr <= SB1_HI)) | (
        (bu_corr >= SB2_LO) & (bu_corr <= SB2_HI)
    )

    log.info(
        f"  [{cat}] current preselection sample: total={len(combined)}, "
        f"signal-window={int(np.sum(sig_mask))}, sideband={int(np.sum(sb_mask))}"
    )
    return {"signal": cc_mass[sig_mask], "sideband": cc_mass[sb_mask]}


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


def plot_sideband_spectrum(cat: str, data: dict):
    """Background check: m(Λ̄pK⁻) in B+ sideband data only."""
    sb = data["sideband"]

    if len(sb) == 0:
        log.warning(f"  No sideband events for {cat}, skipping sideband diagnostic")
        return

    log.info(f"  [{cat}] sideband-only diagnostic: {len(sb)} events")

    x_range = (CC_LO, CC_HI)
    bins = CC_BINS
    bw = (CC_HI - CC_LO) / CC_BINS

    h_sb, edges = np.histogram(sb, range=x_range, bins=bins)
    centers = (edges[1:] + edges[:-1]) / 2
    err_sb = np.sqrt(h_sb + 1)

    fig, ax = plt.subplots(figsize=(8, 5.2))

    mask = h_sb > 0
    ax.errorbar(
        centers[mask],
        h_sb[mask],
        yerr=err_sb[mask],
        label=r"$B^+$ sideband data",
        ecolor="black",
        mfc="black",
        color="black",
        elinewidth=1.5,
        markersize=4,
        marker="o",
        fmt=" ",
    )

    ax.set_xlabel(r"$m(\bar{\Lambda}pK^-)$ [MeV/$c^2$]")
    ax.set_ylabel(rf"Candidates / ({int(bw)} MeV/$c^2$)")
    ax.set_xlim(*x_range)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=12, loc="upper left")

    label_frac = {3414.7: 0.95, 3510.7: 0.83}
    for mass, label, _ in CC_STATES:
        if CC_LO < mass < CC_HI:
            ax.axvline(mass, color="gray", linestyle=":", linewidth=1.2, alpha=0.8)
            frac = label_frac.get(mass, 0.95)
            ax.text(
                mass,
                frac,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=10,
                color="dimgray",
                rotation=90,
            )

    out = figs_path(cat, "backgrounds", "charmonium_sideband.pdf")
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
        plot_sideband_spectrum(cat, data)
    log.info("=== Done. ===")


if __name__ == "__main__":
    main()
