"""
BDT input-variable comparison plots for the B+ → Λ̄pK⁻K⁺ analysis note.

Produces:
  figs/LambdaLL/bdt_variables/BDT_DTFchi2_Sig.pdf
  figs/LambdaLL/bdt_variables/BDT_FDCHI2_Sig.pdf
  figs/LambdaLL/bdt_variables/BDT_IPCHI2_Sig.pdf
  figs/LambdaLL/bdt_variables/BDT_PT_Sig.pdf
  figs/LambdaDD/bdt_variables/BDT_DTFchi2_Sig.pdf
  figs/LambdaDD/bdt_variables/BDT_FDCHI2_Sig.pdf
  figs/LambdaDD/bdt_variables/BDT_IPCHI2_Sig.pdf
  figs/LambdaDD/bdt_variables/BDT_PT_Sig.pdf

Strategy:
  - Read the cached post-BDT-selected samples from analysis_output/mva/cache.
  - For data, require the B+ signal window and the J/psi charmonium window.
  - Compare to post-selection J/psi MC, normalized to unit area.

Run from analysis/ directory:
    ../.venv/bin/python studies/ana_note_plots/scripts/plot_bdt_variables.py
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPTS_DIR.resolve().parents[2]
sys.path.insert(0, str(ANALYSIS_DIR))

from modules.plot_utils import COLORS, figs_path, save_fig, setup_style

M_B_LOW = 5255.0
M_B_HIGH = 5305.0
M_JPSI = 3096.9
JPSI_WINDOW = 30.0

CACHE_META_DIR = ANALYSIS_DIR / "analysis_output" / "mva" / "cache" / "metadata"
CACHE_DATA_DIR = ANALYSIS_DIR / "analysis_output" / "mva" / "cache" / "data"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

setup_style()


def _load_cached(description: str):
    for meta_path in sorted(CACHE_META_DIR.glob("*.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("description") != description:
            continue
        data_path = CACHE_DATA_DIR / f"{meta_path.stem}.pkl"
        with open(data_path, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"Cache entry not found: {description}")


def _charmonium_mass(events: ak.Array) -> np.ndarray:
    e = events["L0_PE"] + events["p_PE"] + events["h1_PE"]
    px = events["L0_PX"] + events["p_PX"] + events["h1_PX"]
    py = events["L0_PY"] + events["p_PY"] + events["h1_PY"]
    pz = events["L0_PZ"] + events["p_PZ"] + events["h1_PZ"]
    return np.sqrt(np.maximum(ak.to_numpy(e**2 - px**2 - py**2 - pz**2), 0.0))


def _load_data(cat: str) -> ak.Array:
    cached = _load_cached(f"Data after high_yield/{cat} cuts")
    events = ak.concatenate([cached[year] for year in sorted(cached)])
    b_mass = ak.to_numpy(events["Bu_MM_corrected"])
    ccbar_mass = _charmonium_mass(events)
    mask = (b_mass >= M_B_LOW) & (b_mass <= M_B_HIGH) & (np.abs(ccbar_mass - M_JPSI) < JPSI_WINDOW)
    return events[mask]


def _load_mc(cat: str) -> ak.Array:
    cached = _load_cached(f"MC after high_yield/{cat} cuts")
    return cached["jpsi"]


def _density_errors(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, _ = np.histogram(values, bins=bins)
    widths = np.diff(bins)
    total = np.sum(counts)
    if total <= 0:
        return np.zeros_like(widths), np.zeros_like(widths)
    density = counts / (total * widths)
    errors = np.sqrt(counts) / (total * widths)
    return density, errors


def _plot_variable(
    cat: str,
    data_vals: np.ndarray,
    mc_vals: np.ndarray,
    *,
    bins: np.ndarray,
    xlabel: str,
    outfile: str,
    logx: bool = False,
    logy: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    centers = 0.5 * (bins[:-1] + bins[1:])
    half_widths = 0.5 * np.diff(bins)
    data_hist, data_err = _density_errors(data_vals, bins)

    ax.errorbar(
        centers[data_hist > 0],
        data_hist[data_hist > 0],
        xerr=half_widths[data_hist > 0],
        yerr=data_err[data_hist > 0],
        fmt="o",
        color="black",
        markersize=4,
        linewidth=1.2,
        label=r"Data ($J/\psi$ window)",
    )

    ax.hist(
        mc_vals,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.0,
        color=COLORS[0],
        label=r"$J/\psi$ MC",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized candidates")
    ax.legend(frameon=False, loc="best")
    ax.text(0.03, 0.95, f"Lambda{cat}", transform=ax.transAxes, ha="left", va="top")

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(bottom=5e-5)

    save_fig(fig, figs_path(cat, "bdt_variables", outfile))


def main() -> None:
    plot_specs = [
        {
            "field": "Bu_DTF_chi2",
            "outfile": "BDT_DTFchi2_Sig.pdf",
            "xlabel": r"$B^+$ DTF $\chi^2$",
            "bins": np.linspace(0.0, 30.0, 31),
        },
        {
            "field": "Bu_FDCHI2_OWNPV",
            "outfile": "BDT_FDCHI2_Sig.pdf",
            "xlabel": r"$B^+$ FD$\chi^2$",
            "bins": np.logspace(np.log10(175.0), np.log10(1.2e5), 31),
            "logx": True,
            "logy": True,
        },
        {
            "field": "Bu_IPCHI2_OWNPV",
            "outfile": "BDT_IPCHI2_Sig.pdf",
            "xlabel": r"$B^+$ IP$\chi^2$",
            "bins": np.linspace(0.0, 10.0, 26),
        },
        {
            "field": "Bu_PT",
            "outfile": "BDT_PT_Sig.pdf",
            "xlabel": r"$B^+$ $p_T$ [MeV/$c$]",
            "bins": np.linspace(3000.0, 30000.0, 28),
        },
    ]

    for cat in ("LL", "DD"):
        data = _load_data(cat)
        mc = _load_mc(cat)
        log.info(f"[Lambda{cat}] data candidates in J/psi window: {len(data)}")
        log.info(f"[Lambda{cat}] J/psi MC candidates after selection: {len(mc)}")

        for spec in plot_specs:
            data_vals = ak.to_numpy(data[spec["field"]])
            mc_vals = ak.to_numpy(mc[spec["field"]])
            _plot_variable(
                cat,
                data_vals,
                mc_vals,
                bins=spec["bins"],
                xlabel=spec["xlabel"],
                outfile=spec["outfile"],
                logx=spec.get("logx", False),
                logy=spec.get("logy", False),
            )
            log.info(f"  wrote {spec['outfile']}")


if __name__ == "__main__":
    main()
