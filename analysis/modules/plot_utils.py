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
import mplhep
import numpy as np

# ── Particle label map (matplotlib LaTeX) ───────────────────────────────────
# Used by mass_fitter and any script that needs consistent state labels.
STATE_LABELS = {
    "jpsi": r"$J/\psi$",
    "etac": r"$\eta_c(1S)$",
    "chic0": r"$\chi_{c0}$",
    "chic1": r"$\chi_{c1}$",
    "etac_2s": r"$\eta_c(2S)$",
}

# Per-state colors for overlaid signal components
STATE_COLORS = {
    "jpsi": "crimson",
    "etac": "darkorange",
    "chic0": "mediumseagreen",
    "chic1": "mediumpurple",
    "etac_2s": "dodgerblue",
}

# Default curve colors
COLOR_TOTAL = "royalblue"
COLOR_BACKGROUND = "dimgray"
COLOR_DATA = "black"


# ── Style ────────────────────────────────────────────────────────────────────


def setup_style() -> None:
    """Apply LHCb2 style globally.  Call once at the start of each script."""
    mplhep.style.use("LHCb2")


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
    color: str = COLOR_DATA,
    marker: str = "o",
    markersize: float = 4.0,
    label: str = "Data",
    zorder: int = 5,
    **kwargs: Any,
) -> None:
    """
    Plot binned data as points with error bars.

    Parameters
    ----------
    x :    Bin centres.
    y :    Bin contents.
    yerr : Bin uncertainties (Poisson or ``sqrt(N)``).
    """
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt=marker,
        color=color,
        markersize=markersize,
        capsize=0,
        elinewidth=1.0,
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
    lw: float = 1.8,
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
    color: str = "steelblue",
    alpha: float = 0.75,
    edgecolor: str = "white",
    lw: float = 0.5,
    label: str = "",
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw a styled histogram.

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
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=lw,
        label=label,
        **kwargs,
    )
    return counts, edges


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
    figsize: tuple[float, float] = (9, 8),
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
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.8),
        )

    ax.legend(loc="upper center", frameon=False, fontsize=9)

    # ── Pull panel ────────────────────────────────────────────────────────────
    if with_pulls and ax_pull is not None and pulls is not None:
        plot_pulls(ax_pull, bin_centers, pulls, x_range=fit_range)
        ax_pull.set_xlabel(x_label)

    return fig, axes
