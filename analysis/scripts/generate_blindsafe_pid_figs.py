from __future__ import annotations

import logging
import sys
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _paths import ANALYSIS_DIR, SLIDES_DIR

if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from modules.clean_data_loader import load_all_data, load_all_mc
from modules.config_loader import StudyConfig
from modules.plot_utils import BINNING
from modules.presentation_config import MC15_PID_BRANCHES, get_presentation_config

PRESENTATION = get_presentation_config()
(SB1_LO, SB1_HI), (SB2_LO, SB2_HI) = PRESENTATION.bu_sideband_windows()
SIG_LO, SIG_HI = PRESENTATION.bu_signal_window()
FIGS_DIR = SLIDES_DIR / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    }
)


def _load_prepid_samples():
    """Load data and MC directly from ntuples with PID product cut disabled.

    The preprocessed pipeline cache already has pid_product_min applied, so it
    cannot be used for pre-PID distribution plots.  We override the cut to 0.0
    here and load fresh from the ntuples.
    """
    config = StudyConfig.from_dir(ANALYSIS_DIR / "config")
    config.fixed_selection["pid_product_min"] = 0.0

    years = config.get_input_years() or ["2016", "2017", "2018"]
    magnets = config.get_input_magnets() or ["MD", "MU"]
    states = config.get_input_mc_states() or ["Jpsi", "etac", "chic0", "chic1"]

    log.info("Loading data pre-PID (pid_product_min overridden to 0.0) …")
    data_full = load_all_data(
        config.get_input_data_base_path(),
        years,
        magnets=magnets,
        config=config,
    )
    log.info("Loading MC pre-PID …")
    mc_full = load_all_mc(
        config.get_input_mc_base_path(),
        states,
        years,
        magnets=magnets,
        config=config,
    )
    return data_full, mc_full


def _collect_category_samples(
    data_full: dict,
    mc_full: dict,
    cat: str,
) -> dict[str, ak.Array]:
    data_cat = ak.concatenate(
        [data_full[year][cat] for year in sorted(data_full) if cat in data_full[year]]
    )
    mc_cat = ak.concatenate([mc_full[state][cat] for state in mc_full if cat in mc_full[state]])

    data_mass = ak.to_numpy(data_cat["Bu_MM_corrected"])
    mc_mass = ak.to_numpy(mc_cat["Bu_MM_corrected"])
    sb_mask = ((data_mass >= SB1_LO) & (data_mass <= SB1_HI)) | (
        (data_mass >= SB2_LO) & (data_mass <= SB2_HI)
    )
    sig_mask = (mc_mass >= SIG_LO) & (mc_mass <= SIG_HI)

    log.info(
        f"[Lambda{cat}] pre-PID samples: "
        f"sideband data={int(np.sum(sb_mask))}, signal MC={int(np.sum(sig_mask))}"
    )
    return {"bkg": data_cat[sb_mask], "sig": mc_cat[sig_mask]}


def _draw_overlay(
    ax: plt.Axes,
    bkg: np.ndarray,
    sig: np.ndarray,
    title: str,
    xlabel: str,
    log_y: bool = False,
    bins: int = 12,
    x_range: tuple[float, float] = (0.0, 1.0),
) -> None:
    bkg = bkg[(bkg >= x_range[0]) & (bkg <= x_range[1])]
    sig = sig[(sig >= x_range[0]) & (sig <= x_range[1])]
    if len(bkg) == 0 or len(sig) == 0:
        ax.text(0.5, 0.5, "No events", ha="center", va="center", transform=ax.transAxes)
        return

    width = (x_range[1] - x_range[0]) / bins
    hist_bkg, edges = np.histogram(bkg, bins=bins, range=x_range)
    hist_sig, _ = np.histogram(sig, bins=bins, range=x_range)
    norm_bkg = hist_bkg.sum() * width
    norm_sig = hist_sig.sum() * width
    centers = 0.5 * (edges[1:] + edges[:-1])
    mask = hist_bkg > 0

    ax.errorbar(
        centers[mask],
        hist_bkg[mask] / norm_bkg,
        yerr=np.sqrt(hist_bkg[mask]) / norm_bkg,
        fmt="o",
        color="black",
        markersize=3.5,
        label=r"$B^+$ sideband data",
    )
    ax.hist(
        sig,
        bins=bins,
        range=x_range,
        weights=np.full(len(sig), 1.0 / norm_sig),
        histtype="step",
        linestyle="--",
        linewidth=2.4,
        color="#1f77b4",
        label="Signal MC",
    )

    if log_y:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(rf"Normalised / ({width:.2f})")
    ax.set_xlim(*x_range)


def make_product_figure(samples: dict[str, dict[str, ak.Array]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        r"PID product before PID pre-selection",
        fontsize=12,
    )
    for ax, cat in zip(axes, ("LL", "DD")):
        bkg = samples[cat]["bkg"]
        sig = samples[cat]["sig"]
        _draw_overlay(
            ax,
            ak.to_numpy(bkg["PID_product"]),
            ak.to_numpy(sig["PID_product"]),
            title=rf"$\Lambda$ {cat}",
            xlabel=r"$\mathrm{ProbNN}_p \times \mathrm{ProbNN}_{K^+} \times \mathrm{ProbNN}_{K^-}$",
            log_y=True,
            bins=10,
            x_range=(0.0, 1.0),
        )
        ax.legend(framealpha=0.9, loc="upper left")
    plt.tight_layout()
    out = FIGS_DIR / "blindsafe_pid_product.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_proton_figure(samples: dict[str, dict[str, ak.Array]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        r"Proton PID before PID pre-selection",
        fontsize=12,
    )
    for ax, cat in zip(axes, ("LL", "DD")):
        _draw_overlay(
            ax,
            ak.to_numpy(samples[cat]["bkg"][MC15_PID_BRANCHES["p"]]),
            ak.to_numpy(samples[cat]["sig"][MC15_PID_BRANCHES["p"]]),
            title=rf"$\Lambda$ {cat}",
            xlabel=r"ProbNNp (bachelor $p$)",
            log_y=True,
            bins=10,
            x_range=tuple(BINNING["pid"]["range"]),
        )
        ax.legend(framealpha=0.9, loc="upper left")
    plt.tight_layout()
    out = FIGS_DIR / "blindsafe_pid_proton.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_kaon_figure(samples: dict[str, dict[str, ak.Array]]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(
        r"Kaon PID before PID pre-selection",
        fontsize=12,
    )
    configs = [
        ("LL", MC15_PID_BRANCHES["h1"], r"$\Lambda$ LL: ProbNNk ($h_1 = K^+$)"),
        ("DD", MC15_PID_BRANCHES["h1"], r"$\Lambda$ DD: ProbNNk ($h_1 = K^+$)"),
        ("LL", MC15_PID_BRANCHES["h2"], r"$\Lambda$ LL: ProbNNk ($h_2 = K^-$)"),
        ("DD", MC15_PID_BRANCHES["h2"], r"$\Lambda$ DD: ProbNNk ($h_2 = K^-$)"),
    ]
    for ax, (cat, branch, title) in zip(axes.flat, configs):
        _draw_overlay(
            ax,
            ak.to_numpy(samples[cat]["bkg"][branch]),
            ak.to_numpy(samples[cat]["sig"][branch]),
            title=title,
            xlabel="ProbNNk",
            log_y=True,
            bins=10,
            x_range=tuple(BINNING["pid"]["range"]),
        )
        ax.legend(framealpha=0.9, loc="upper left")
    plt.tight_layout()
    out = FIGS_DIR / "blindsafe_pid_kaons.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    data_full, mc_full = _load_prepid_samples()
    samples = {cat: _collect_category_samples(data_full, mc_full, cat) for cat in ("LL", "DD")}

    outputs = [
        make_product_figure(samples),
        make_proton_figure(samples),
        make_kaon_figure(samples),
    ]
    for output in outputs:
        log.info(f"Saved {output}")


if __name__ == "__main__":
    main()
