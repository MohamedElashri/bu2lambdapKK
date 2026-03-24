"""
Shared plotting helpers for the B+ → Λ̄pK⁻K⁺ analysis note.

Usage:
    from plot_helpers import setup_style, plot_data, COLORS, HISTSTYLE, FIGS_DIR, BINNING
    setup_style()
    b = BINNING["lambda_mass_full"]
    plot_data(ax, data_array, r"Data", {"range": b["range"], "bins": b["bins"]})
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams


# ── Style (exact copy of reference mass_spectrums.py) ─────────────────────────
# NOTE: reference has plt.style.use(hep.style.LHCb1) COMMENTED OUT
def setup_style():
    """Apply reference style: STIX math + serif. Call once before any plotting."""
    rcParams.update({"mathtext.fontset": "stix"})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]


# ── Scalar formatter (from reference reweight_plot.py) ────────────────────────
def make_formatter():
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    return fmt


# ── Color palette (exact copy of reference utils/colors.py) ───────────────────
COLORS = ["darkgreen", "#6F4F59", "#003366", "#D35400", "black"]


# ── Centralized binning ────────────────────────────────────────────────────────
# Bin widths chosen to match the relevant resolution scale and give ≥20 events/bin.
# Rule of thumb: bin_width ≈ detector resolution or a round multiple thereof.
# Helper: bw(b) = (b["range"][1] - b["range"][0]) / b["bins"]
BINNING = {
    # ── Lambda mass ───────────────────────────────────────────────────────────
    # Detector resolution ~2 MeV → 1 MeV/bin over full stripping range
    "lambda_mass_full": {"range": (1090, 1155), "bins": 65},  # 1.0 MeV/bin
    # 1 MeV/bin tight around the selection window — same width, fewer bins
    "lambda_mass_tight": {"range": (1108, 1126), "bins": 18},  # 1.0 MeV/bin
    # ── B+ mass ───────────────────────────────────────────────────────────────
    # MC resolution σ ≈ 12–13 MeV → 5 MeV/bin gives ~2.5 bins per sigma
    "bu_mass_fit": {"range": (5100, 5500), "bins": 80},  # 5 MeV/bin (fit plots)
    # 10 MeV/bin for display plots where fit detail isn't needed
    "bu_mass_display": {"range": (5100, 5500), "bins": 40},  # 10 MeV/bin
    # ── Charmonium mass m(ΛpK) ───────────────────────────────────────────────
    # Detector resolution ~15 MeV → 20 MeV/bin; covers J/ψ, ηc, χc, ηc(2S)
    "charmonium_mass": {"range": (2800, 3800), "bins": 50},  # 20 MeV/bin
    # ── PID ──────────────────────────────────────────────────────────────────
    # Shape has broad features → 0.05/bin is sufficient resolution
    "pid": {"range": (0.0, 1.0), "bins": 20},  # 0.05/bin
    # ── Geometric & kinematic selection variables ─────────────────────────────
    "log_ipchi2": {"range": (-2.0, 4.0), "bins": 30},  # 0.2/bin
    "bu_pt": {"range": (0, 25000), "bins": 50},  # 500 MeV/bin
    "delta_z_ll": {"range": (0, 150), "bins": 30},  # 5 mm/bin
    "delta_z_dd": {"range": (0, 150), "bins": 30},  # 5 mm/bin
    "dtf_chi2": {"range": (0, 50), "bins": 25},  # 2/bin
    "bu_fdchi2": {"range": (0, 5000), "bins": 25},  # 200/bin
    "l0_fdchi2": {"range": (0, 3000), "bins": 30},  # 100/bin
    "probnnp": {"range": (0, 1), "bins": 20},  # 0.05/bin
}

# ── MC histogram style (from reference reweight_plot.py) ──────────────────────
HISTSTYLE = {"histtype": "step", "linestyle": "--", "linewidth": 4, "density": True}


# ── plot_data (exact copy of reference utils/plot.py) ─────────────────────────
def plot_data(ax, data, label, histstyle, weights=None, color="black", errorbar=True, mkstyle="o"):
    """
    Plot binned data as error bars.

    Exact replica of reference bu2lambdappp/utils/plot.py::plot_data().

    Parameters
    ----------
    ax        : matplotlib Axes
    data      : array-like of values
    label     : legend label
    histstyle : dict with keys 'range' and 'bins' (and optionally 'density')
    weights   : per-event weights (optional)
    color     : marker and error-bar color
    errorbar  : if True use errorbar, else plain errorbar without y-errors
    mkstyle   : marker style string

    Returns
    -------
    ax, data_hist, data_hist_errors
    """
    data_hist, bins = np.histogram(data, weights=weights, **histstyle)
    data_hist_errors = np.sqrt(np.abs(data_hist) + 1)
    try:
        if histstyle.get("density"):
            data_hist_errors = np.sqrt(data_hist / len(data) * np.sum(data_hist))
    except Exception:
        pass
    bin_center = (bins[1:] + bins[:-1]) / 2
    bin_width = (bins[1:] - bins[:-1]) / 2

    # Mask out empty/negative bins so zero-count bins are not plotted
    nonzero = data_hist > 0

    if errorbar:
        ax.errorbar(
            x=bin_center[nonzero],
            y=data_hist[nonzero],
            xerr=bin_width[nonzero],
            yerr=data_hist_errors[nonzero],
            label=label,
            ecolor=color,
            mfc=color,
            color=color,
            elinewidth=1.5,
            markersize=4,
            marker=mkstyle,
            fmt=" ",
        )
    else:
        ax.errorbar(
            x=bin_center[nonzero],
            y=data_hist[nonzero],
            xerr=bin_width[nonzero],
            fmt=mkstyle,
            label=label,
            ecolor=color,
            mfc=color,
            color=color,
            elinewidth=3,
        )
    return ax, data_hist, data_hist_errors


# ── Figure directory ───────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
FIGS_DIR = SCRIPTS_DIR.parent / "figs"


def figs_path(cat: str, *parts: str) -> Path:
    """Return a path under figs/LambdaLL/ or figs/LambdaDD/.

    Example:
        figs_path("LL", "Preselections", "Delta_Z.pdf")
        → .../figs/LambdaLL/Preselections/Delta_Z.pdf
    """
    return FIGS_DIR / f"Lambda{cat}" / Path(*parts)


def save_fig(fig, path: Path):
    """Save figure, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
