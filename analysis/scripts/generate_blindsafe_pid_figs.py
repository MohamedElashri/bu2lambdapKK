from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable

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

ALL_MC_STATES = {"jpsi", "etac", "chic0", "chic1"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 10,
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


def _load_samples(skip_lambda: bool = False):
    """Load data and MC from ntuples with PID product cut disabled.

    If skip_lambda=True, all Lambda selection cuts are also zeroed out so the
    returned sample reflects only the stripping-level baseline reduction.
    """
    config = StudyConfig.from_dir(ANALYSIS_DIR / "config")
    config.fixed_selection["pid_product_min"] = 0.0

    if skip_lambda:
        config.lambda_selection["mass_min"] = 0.0
        config.lambda_selection["mass_max"] = 99999.0
        config.lambda_selection["fd_chisq_min"] = 0.0
        config.lambda_selection["proton_probnnp_min"] = 0.0
        config.lambda_selection["delta_z_min_ll"] = 0.0
        config.lambda_selection["delta_z_min_dd"] = 0.0
        log.info("Loading data — no PID cut, no Lambda selections …")
    else:
        log.info("Loading data — no PID cut, Lambda selections applied …")

    years = config.get_input_years() or ["2016", "2017", "2018"]
    magnets = config.get_input_magnets() or ["MD", "MU"]
    states = config.get_input_mc_states() or ["jpsi", "etac", "chic0", "chic1"]

    data_full = load_all_data(
        config.get_input_data_base_path(),
        years,
        magnets=magnets,
        config=config,
    )
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
    """Return sideband background and MC split into high-yield / low-yield state groups."""
    data_cat = ak.concatenate(
        [data_full[year][cat] for year in sorted(data_full) if cat in data_full[year]]
    )
    mc_cat = ak.concatenate(
        [mc_full[s][cat] for s in ALL_MC_STATES if s in mc_full and cat in mc_full[s]]
    )

    data_mass = ak.to_numpy(data_cat["Bu_MM_corrected"])
    mc_mass = ak.to_numpy(mc_cat["Bu_MM_corrected"])

    # Split sideband into lower (SB1, closer to signal peak = "high signals" region)
    # and upper (SB2, far from signal peak = "low signals" region)
    sb1_mask = (data_mass >= SB1_LO) & (data_mass <= SB1_HI)
    sb2_mask = (data_mass >= SB2_LO) & (data_mass <= SB2_HI)
    sig_mask = (mc_mass >= SIG_LO) & (mc_mass <= SIG_HI)

    log.info(
        f"[Lambda{cat}] pre-PID samples: "
        f"SB1(lower)={int(np.sum(sb1_mask))}, SB2(upper)={int(np.sum(sb2_mask))}, "
        f"signal MC={int(np.sum(sig_mask))}"
    )
    return {
        "bkg_sb1": data_cat[sb1_mask],
        "bkg_sb2": data_cat[sb2_mask],
        "sig": mc_cat[sig_mask],
    }


def _draw_raw(
    ax: plt.Axes,
    bkg: np.ndarray,
    sig: np.ndarray,
    title: str,
    xlabel: str,
    log_y: bool,
    bins: int,
    x_range: tuple[float, float],
) -> None:
    """Plot raw (un-normalised) event counts for signal and background."""
    bkg = bkg[(bkg >= x_range[0]) & (bkg <= x_range[1])]
    sig = sig[(sig >= x_range[0]) & (sig <= x_range[1])]

    hist_bkg, edges = np.histogram(bkg, bins=bins, range=x_range)
    hist_sig, _ = np.histogram(sig, bins=bins, range=x_range)
    centers = 0.5 * (edges[1:] + edges[:-1])
    bkg_mask = hist_bkg > 0

    ax.errorbar(
        centers[bkg_mask],
        hist_bkg[bkg_mask],
        yerr=np.sqrt(hist_bkg[bkg_mask]),
        fmt="o",
        color="black",
        markersize=3.5,
        label=r"$B^+$ sideband data",
    )
    ax.stairs(
        hist_sig,
        edges,
        color="#1f77b4",
        linestyle="--",
        linewidth=2.4,
        label="Signal MC",
    )

    if log_y:
        ax.set_yscale("log")
        # prevent log(0) clipping from hiding low-count bins
        ax.set_ylim(bottom=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events / bin")
    ax.set_xlim(*x_range)
    ax.legend(framealpha=0.9, loc="upper left")


def _make_2x2_figure(
    samples: dict[str, dict[str, ak.Array]],
    get_arr: Callable[[ak.Array], np.ndarray],
    xlabel: str,
    suptitle: str,
    log_y: bool,
    bins: int,
    x_range: tuple[float, float],
    out_path: Path,
) -> Path:
    """2×2 figure: rows = lower/upper B+ sideband, cols = LL/DD."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    scale_tag = "log scale" if log_y else "linear scale"
    fig.suptitle(f"{suptitle} ({scale_tag})", fontsize=12)

    row_keys = [
        ("bkg_sb1", f"Lower sideband ({SB1_LO:.0f}–{SB1_HI:.0f} MeV)"),
        ("bkg_sb2", f"Upper sideband ({SB2_LO:.0f}–{SB2_HI:.0f} MeV)"),
    ]
    for row_idx, (bkg_key, sideband_label) in enumerate(row_keys):
        for col_idx, cat in enumerate(("LL", "DD")):
            _draw_raw(
                axes[row_idx, col_idx],
                get_arr(samples[cat][bkg_key]),
                get_arr(samples[cat]["sig"]),
                title=rf"$\Lambda$ {cat} — {sideband_label}",
                xlabel=xlabel,
                log_y=log_y,
                bins=bins,
                x_range=x_range,
            )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_product_figures(samples: dict, suptitle: str, out_suffix: str = "") -> list[Path]:
    common = dict(
        samples=samples,
        get_arr=lambda ev: ak.to_numpy(ev["PID_product"]),
        xlabel=r"$\mathrm{ProbNN}_p \times \mathrm{ProbNN}_{K^+} \times \mathrm{ProbNN}_{K^-}$",
        suptitle=suptitle,
        bins=10,
        x_range=(0.0, 1.0),
    )
    return [
        _make_2x2_figure(
            **common,
            log_y=False,
            out_path=FIGS_DIR / f"blindsafe_pid_product{out_suffix}_linear.pdf",
        ),
        _make_2x2_figure(
            **common, log_y=True, out_path=FIGS_DIR / f"blindsafe_pid_product{out_suffix}_log.pdf"
        ),
    ]


def make_proton_figures(samples: dict) -> list[Path]:
    branch = MC15_PID_BRANCHES["p"]
    common = dict(
        samples=samples,
        get_arr=lambda ev: ak.to_numpy(ev[branch]),
        xlabel=r"ProbNNp (bachelor $p$)",
        suptitle="Proton PID (Lambda selections applied)",
        bins=10,
        x_range=tuple(BINNING["pid"]["range"]),
    )
    return [
        _make_2x2_figure(
            **common, log_y=False, out_path=FIGS_DIR / "blindsafe_pid_proton_linear.pdf"
        ),
        _make_2x2_figure(**common, log_y=True, out_path=FIGS_DIR / "blindsafe_pid_proton_log.pdf"),
    ]


def make_kaon_figures(samples: dict) -> list[Path]:
    outputs = []
    for tag, branch, label in [
        ("h1", MC15_PID_BRANCHES["h1"], r"ProbNNk ($h_1 = K^+$)"),
        ("h2", MC15_PID_BRANCHES["h2"], r"ProbNNk ($h_2 = K^-$)"),
    ]:
        common = dict(
            samples=samples,
            get_arr=lambda ev, b=branch: ak.to_numpy(ev[b]),
            xlabel=label,
            suptitle=f"Kaon PID ({label}) (Lambda selections applied)",
            bins=10,
            x_range=tuple(BINNING["pid"]["range"]),
        )
        outputs += [
            _make_2x2_figure(
                **common, log_y=False, out_path=FIGS_DIR / f"blindsafe_pid_kaon_{tag}_linear.pdf"
            ),
            _make_2x2_figure(
                **common, log_y=True, out_path=FIGS_DIR / f"blindsafe_pid_kaon_{tag}_log.pdf"
            ),
        ]
    return outputs


def main() -> None:
    # With Lambda selections
    data_full, mc_full = _load_samples(skip_lambda=False)
    samples = {cat: _collect_category_samples(data_full, mc_full, cat) for cat in ("LL", "DD")}

    # Without Lambda selections
    data_full_nl, mc_full_nl = _load_samples(skip_lambda=True)
    samples_nl = {
        cat: _collect_category_samples(data_full_nl, mc_full_nl, cat) for cat in ("LL", "DD")
    }

    outputs = (
        make_product_figures(
            samples, suptitle="PID product (Lambda selections applied)", out_suffix=""
        )
        + make_product_figures(
            samples_nl, suptitle="PID product (no Lambda selections)", out_suffix="_no_lambda"
        )
        + make_proton_figures(samples)
        + make_kaon_figures(samples)
    )
    for out in outputs:
        log.info(f"Saved {out}")


if __name__ == "__main__":
    main()
