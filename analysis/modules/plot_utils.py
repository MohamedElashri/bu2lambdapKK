"""
Centralized plotting utilities for B+ → Λ̄pK⁻K⁺ analysis.

All plots use the LHCb2 style from mplhep.  Import this module and call
``setup_style()`` once at the start of each script (or let individual
high-level helpers call it automatically).

Key functions
─────────────
setup_style()           Apply LHCb2 style globally (call once per script).
make_figure()           Standard figure, optionally with a pull sub-panel.
plot_data_points()      Black data points with error bars.
plot_curve()            Smooth curve (total fit, component, background).
plot_pulls()            Pull distribution sub-panel.
plot_histogram()        Styled 1-D histogram.
plot_2d_histogram()     Styled 2-D histogram heatmap.
save_figure()           Save to PDF/PNG with sensible defaults.
make_mass_fit_figure()  High-level mass-fit plot (data + curves + pulls).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Force non-interactive Agg backend. Called here so importing plot_utils
# switches the backend before any figure is created, even when the calling
# script has already done `import matplotlib.pyplot as plt` with TkAgg.
plt.switch_backend("Agg")
import numpy as np

# ── Reference colour palette (matches reference analysis) ────────────────────
LHCB_COLORS = ["darkgreen", "#6F4F59", "#003366", "#D35400", "black"]
# Alias used by ana_note_plots scripts
COLORS = LHCB_COLORS

# ── Particle label map (matplotlib LaTeX) ───────────────────────────────────
# Used by mass_fitter and any script that needs consistent state labels.
STATE_LABELS = {
    "jpsi": r"$J/\psi$",
    "etac": r"$\eta_c(1S)$",
    "chic0": r"$\chi_{c0}$",
    "chic1": r"$\chi_{c1}$",
    "etac_2s": r"$\eta_c(2S)$",
}

# Per-state colors — distinct, well-separated palette
STATE_COLORS = {
    "jpsi": "#117733",  # forest green
    "etac": "#D35400",  # burnt orange
    "chic0": "#0099BB",  # cyan-blue (distinct from total fit navy)
    "chic1": "#882255",  # deep purple
    "etac_2s": "#AA3377",  # pink-purple (was black — now clearly distinct)
}

# Default curve colors
COLOR_TOTAL = "#0044AA"  # medium blue
COLOR_BACKGROUND = "#CC4411"  # vivid orange-red (was black dashed — now clearly visible)
COLOR_DATA = "#333333"  # dark gray with transparency (not solid black)

# ── MC histogram style (matches reference reweight_plot.py) ──────────────────
HISTSTYLE = {"histtype": "step", "linestyle": "--", "linewidth": 4, "density": True}

# ── Centralized binning ───────────────────────────────────────────────────────
BINNING = {
    "lambda_mass_full": {"range": (1090, 1155), "bins": 65},  # 1.0 MeV/bin
    "lambda_mass_tight": {"range": (1108, 1126), "bins": 18},  # 1.0 MeV/bin
    "bu_mass_fit": {"range": (5100, 5500), "bins": 80},  # 5 MeV/bin
    "bu_mass_display": {"range": (5100, 5500), "bins": 40},  # 10 MeV/bin
    "charmonium_mass": {"range": (2800, 3800), "bins": 50},  # 20 MeV/bin
    "pid": {"range": (0.0, 1.0), "bins": 20},  # 0.05/bin
    "log_ipchi2": {"range": (-2.0, 4.0), "bins": 30},  # 0.2/bin
    "bu_pt": {"range": (0, 25000), "bins": 50},  # 500 MeV/bin
    "delta_z_ll": {"range": (0, 150), "bins": 30},  # 5 mm/bin
    "delta_z_dd": {"range": (0, 150), "bins": 30},  # 5 mm/bin
    "dtf_chi2": {"range": (0, 50), "bins": 25},  # 2/bin
    "bu_fdchi2": {"range": (0, 5000), "bins": 25},  # 200/bin
    "l0_fdchi2": {"range": (0, 3000), "bins": 30},  # 100/bin
    "probnnp": {"range": (0, 1), "bins": 20},  # 0.05/bin
}

# ── Figure directory for ana_note_plots outputs ───────────────────────────────
ANA_NOTE_FIGS_DIR = Path(__file__).resolve().parents[1] / "studies" / "ana_note_plots" / "figs"
FIGS_DIR = ANA_NOTE_FIGS_DIR  # alias


# ── Style ────────────────────────────────────────────────────────────────────


def setup_style() -> None:
    """Apply reference style (mass_spectrums.py equivalent).  Call once per script.

    The reference has LHCb1 commented out for mass spectra plots — uses only
    STIX math + serif font, which gives clean publication-quality text sizes.
    """
    mpl.rcParams.update({"mathtext.fontset": "stix"})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12


# ── Figure construction ───────────────────────────────────────────────────────


def make_figure(
    with_pulls: bool = False,
    figsize: tuple[float, float] | None = None,
    pull_fraction: float = 0.25,
    **subplot_kw: Any,
) -> tuple[plt.Figure, Any]:
    """
    Create a publication-quality figure.

    Parameters
    ----------
    with_pulls :
        If True, add a pull sub-panel below the main axes.
    figsize :
        Figure size (width, height) in inches.
        Defaults to (8, 6) or (8, 9) when ``with_pulls=True``.
    pull_fraction :
        Fraction of total figure height devoted to the pull panel.

    Returns
    -------
    fig, ax                  when *with_pulls* is False
    fig, (ax_main, ax_pull)  when *with_pulls* is True
    """
    setup_style()
    if figsize is None:
        figsize = (8, 9) if with_pulls else (8, 6)

    if not with_pulls:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained", **subplot_kw)
        return fig, ax

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        layout="constrained",
        gridspec_kw={
            "height_ratios": [1 - pull_fraction, pull_fraction],
            "hspace": 0.04,
        },
        **subplot_kw,
    )
    return fig, axes  # (ax_main, ax_pull)


# ── Data points ──────────────────────────────────────────────────────────────


def plot_data_points(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | None = None,
    xerr: np.ndarray | None = None,
    color: str = COLOR_DATA,
    marker: str = "o",
    markersize: float = 4.0,
    label: str = "Data",
    zorder: int = 5,
    **kwargs: Any,
) -> None:
    """
    Plot binned data as points with error bars (matches reference style).

    Parameters
    ----------
    x :    Bin centres.
    y :    Bin contents.
    yerr : Bin uncertainties (Poisson: ``sqrt(|N|+1)``).
    xerr : Half-bin widths for horizontal error bars.
    """
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        xerr=xerr,
        fmt=" ",
        marker=marker,
        ecolor=color,
        mfc=color,
        color=color,
        markersize=markersize,
        capsize=0,
        elinewidth=1.0,
        alpha=0.75,
        label=label,
        zorder=zorder,
        **kwargs,
    )


# ── Smooth curves ─────────────────────────────────────────────────────────────


def plot_curve(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    label: str = "",
    color: str = COLOR_TOTAL,
    linestyle: str = "-",
    lw: float = 2.0,
    zorder: int = 4,
    **kwargs: Any,
) -> None:
    """Plot a smooth curve (total fit, signal component, or background)."""
    ax.plot(x, y, color=color, linestyle=linestyle, lw=lw, label=label, zorder=zorder, **kwargs)


# ── Pull distribution ─────────────────────────────────────────────────────────


def plot_pulls(
    ax: plt.Axes,
    x: np.ndarray,
    pulls: np.ndarray,
    x_range: tuple[float, float] | None = None,
    color: str = COLOR_DATA,
    markersize: float = 3.5,
    ylim: tuple[float, float] = (-4.5, 4.5),
) -> None:
    """
    Plot pull distribution on a sub-panel.

    Parameters
    ----------
    x :      Bin centres.
    pulls :  (data − model) / σ values.
    x_range: Horizontal extent for reference lines.
    """
    # Scatter plot
    ax.scatter(x, pulls, s=markersize**2, color=color, zorder=3)

    if x_range is None and len(x) > 0:
        x_range = (float(x[0]), float(x[-1]))

    # Reference lines: ±3σ (dashed gray) and 0 (solid black)
    for y_val, c, ls, lw in [
        (-3.0, "gray", "--", 0.8),
        (0.0, "black", "-", 0.8),
        (3.0, "gray", "--", 0.8),
    ]:
        ax.axhline(y_val, color=c, linestyle=ls, linewidth=lw, zorder=2)

    ax.set_ylim(ylim)
    ax.set_ylabel("Pull")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2.0))
    if x_range is not None:
        ax.set_xlim(x_range)


# ── Histogram ─────────────────────────────────────────────────────────────────


def plot_histogram(
    ax: plt.Axes,
    data: np.ndarray,
    bins: int | np.ndarray = 50,
    range: tuple[float, float] | None = None,
    density: bool = False,
    color: str = LHCB_COLORS[0],
    histtype: str = "step",
    linestyle: str = "--",
    lw: float = 2.0,
    label: str = "",
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw a styled step histogram (MC style matching reference).

    Returns
    -------
    counts, bin_edges
    """
    counts, edges, _ = ax.hist(
        data,
        bins=bins,
        range=range,
        density=density,
        color=color,
        histtype=histtype,
        linestyle=linestyle,
        linewidth=lw,
        label=label,
        **kwargs,
    )
    return counts, edges


def make_scalar_formatter() -> ticker.ScalarFormatter:
    """Return a ScalarFormatter with scientific notation (matches reference style)."""
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    return fmt


# Alias used by ana_note_plots scripts
make_formatter = make_scalar_formatter


# ── 2-D histogram ─────────────────────────────────────────────────────────────


def plot_2d_histogram(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    bins: int | tuple[int, int] = 50,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    cmap: str = "viridis",
    label_x: str = "",
    label_y: str = "",
    **kwargs: Any,
) -> mpl.image.AxesImage:
    """
    Draw a 2-D histogram heatmap.

    Returns
    -------
    The QuadMesh returned by ``ax.hist2d``.
    """
    range_2d = None
    if x_range is not None and y_range is not None:
        range_2d = [x_range, y_range]

    _, _, _, mesh = ax.hist2d(x, y, bins=bins, range=range_2d, cmap=cmap, **kwargs)
    if label_x:
        ax.set_xlabel(label_x)
    if label_y:
        ax.set_ylabel(label_y)
    return mesh


# ── Save ──────────────────────────────────────────────────────────────────────


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    dpi: int = 150,
    bbox_inches: str = "tight",
    close: bool = True,
    **kwargs: Any,
) -> None:
    """
    Save *fig* to *path*, creating parent directories as needed.

    The figure is closed after saving unless *close* is False.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    if close:
        plt.close(fig)


# ── Legacy-style data plot (used by ana_note_plots scripts) ──────────────────


def plot_data(
    ax: plt.Axes,
    data: np.ndarray,
    label: str,
    histstyle: dict,
    weights=None,
    color: str = "black",
    errorbar: bool = True,
    mkstyle: str = "o",
):
    """
    Plot binned data as error bars (legacy API matching reference bu2lambdappp/utils/plot.py).

    Parameters
    ----------
    histstyle : dict with 'range', 'bins', and optionally 'density'

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


# ── Ana-note figure path helpers ──────────────────────────────────────────────


def figs_path(cat: str, *parts: str) -> Path:
    """Return a path under the ana_note_plots figs directory.

    Example:
        figs_path("LL", "backgrounds", "ks0_veto.pdf")
        → .../studies/ana_note_plots/figs/LambdaLL/backgrounds/ks0_veto.pdf
    """
    return ANA_NOTE_FIGS_DIR / f"Lambda{cat}" / Path(*parts)


def save_fig(fig: plt.Figure, path: Path) -> None:
    """Save figure, creating parent directories as needed (alias for save_figure)."""
    save_figure(fig, path)


# ── High-level mass-fit figure ────────────────────────────────────────────────


def make_mass_fit_figure(
    *,
    bin_centers: np.ndarray,
    bin_contents: np.ndarray,
    bin_errors: np.ndarray,
    mass_points: np.ndarray,
    total_curve: np.ndarray,
    signal_curves: list[dict],
    background_curve: np.ndarray,
    pulls: np.ndarray | None = None,
    x_label: str = r"$M(\bar{\Lambda}pK^{-})\ [\mathrm{MeV}/c^{2}]$",
    y_label: str | None = None,
    bin_width: float = 5.0,
    fit_range: tuple[float, float] = (2800.0, 3700.0),
    year: str = "",
    context_label: str = "",
    info_lines: Sequence[str] | None = None,
    figsize: tuple[float, float] = (7, 6.3),
    total_color: str = COLOR_TOTAL,
    background_color: str = COLOR_BACKGROUND,
) -> tuple[plt.Figure, Any]:
    """
    High-level mass-fit plot: data points + fit curves + optional pull panel.

    Parameters
    ----------
    bin_centers, bin_contents, bin_errors :
        Binned data arrays (length = *nbins*).
    mass_points :
        Fine grid for smooth PDF curves (length ≥ 200).
    total_curve :
        Total model evaluated at *mass_points* (expected counts / bin_width).
    signal_curves :
        List of dicts with keys:
          ``"x"``        – mass points (np.ndarray)
          ``"y"``        – counts / bin_width (np.ndarray)
          ``"label"``    – matplotlib LaTeX label (str)
          ``"color"``    – line colour (str, optional)
          ``"linestyle"``– line style (str, optional, default ``"-"``)
    background_curve :
        Background-only curve evaluated at *mass_points*.
    pulls :
        Pull values at *bin_centers*; if None no pull panel is drawn.
    info_lines :
        Extra text lines displayed as a right-aligned annotation box.
    year :
        Year string placed in the upper-left corner (e.g. ``"2016"``).
    context_label :
        Additional label placed below *year* (e.g. ``"LL / high yield"``).

    Returns
    -------
    fig, axes
        ``axes`` is a single ``Axes`` when *pulls* is None, otherwise a
        length-2 tuple ``(ax_main, ax_pull)``.
    """
    with_pulls = pulls is not None
    fig, axes = make_figure(with_pulls=with_pulls, figsize=figsize)
    ax: plt.Axes = axes[0] if with_pulls else axes
    ax_pull: plt.Axes | None = axes[1] if with_pulls else None

    if y_label is None:
        y_label = f"Candidates / ({bin_width:.0f}" + r" $\mathrm{MeV}/c^{2}$)"

    # ── Data ──────────────────────────────────────────────────────────────────
    plot_data_points(ax, bin_centers, bin_contents, yerr=bin_errors, label="Data")

    # ── Fit curves ────────────────────────────────────────────────────────────
    plot_curve(ax, mass_points, total_curve, label="Total fit", color=total_color)

    for sig in signal_curves:
        plot_curve(
            ax,
            np.asarray(sig["x"]),
            np.asarray(sig["y"]),
            label=sig.get("label", "Signal"),
            color=sig.get("color", "crimson"),
            linestyle=sig.get("linestyle", "-"),
        )

    plot_curve(
        ax,
        mass_points,
        background_curve,
        label="Background",
        color=background_color,
        linestyle="--",
    )

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(fit_range)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(y_label)
    if not with_pulls:
        ax.set_xlabel(x_label)
    else:
        ax.tick_params(labelbottom=False)

    # ── Corner labels ─────────────────────────────────────────────────────────
    corner_parts = [p for p in (year, context_label) if p]
    if corner_parts:
        ax.text(
            0.05,
            0.97,
            "\n".join(corner_parts),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
        )

    # ── Info annotation (right side) ──────────────────────────────────────────
    if info_lines:
        ax.text(
            0.97,
            0.97,
            "\n".join(info_lines),
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.8),
        )

    ax.legend(loc="upper center", frameon=False)

    # ── Pull panel ────────────────────────────────────────────────────────────
    if with_pulls and ax_pull is not None and pulls is not None:
        plot_pulls(ax_pull, bin_centers, pulls, x_range=fit_range)
        ax_pull.set_xlabel(x_label)

    return fig, axes
