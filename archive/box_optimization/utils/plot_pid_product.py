"""
PID Distribution Plots: Data vs. Signal MC vs. Phase-Space MC

Generates two output PDFs:
  1. pid_product_distribution.pdf   — PID product overlay (no cut line)
  2. pid_individual_distribution.pdf — 3 subplots: PNN(p-bar), PNN(K+), PNN(K-)

All distributions at pre-selection level (trigger + Λ⁰ quality cuts), before any
optimisation cuts.  All histograms area-normalised to unity.
"""

import logging
import sys
from pathlib import Path

# Add the parent directory (box_optimization) and its parent to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from clean_data_loader import load_all_data, load_all_mc, load_and_preprocess
from config_loader import StudyConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Style ────────────────────────────────────────────────────────────────────
mpl.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    }
)

SIGNAL_STYLE = {
    "jpsi": {"color": "#E63946", "label": r"$J/\psi$ MC", "ls": "-", "lw": 1.8},
    "etac": {"color": "#457B9D", "label": r"$\eta_c(1S)$ MC", "ls": "--", "lw": 1.8},
    "chic0": {"color": "#2A9D8F", "label": r"$\chi_{c0}$ MC", "ls": "-.", "lw": 1.8},
    "chic1": {"color": "#E9C46A", "label": r"$\chi_{c1}$ MC", "ls": ":", "lw": 2.0},
}

# ── Helpers ──────────────────────────────────────────────────────────────────


def flat_arr(events: ak.Array, branch: str) -> np.ndarray:
    arr = events[branch]
    if "var" in str(ak.type(arr)):
        arr = ak.firsts(arr)
    arr = arr[~ak.is_none(arr)]
    return ak.to_numpy(arr).astype(float)


def norm_hist(values: np.ndarray, bins: np.ndarray):
    h, _ = np.histogram(values, bins=bins)
    bw = bins[1] - bins[0]
    total = h.sum()
    if total == 0:
        return h.astype(float), np.zeros_like(h, dtype=float)
    return h / (total * bw), np.sqrt(h) / (total * bw)


def draw_overlay(ax, data_vals, mc_by_state, phsp_vals, bins, xlabel, title_suffix=""):
    """Draw data + signal MC + phase-space MC on a single Axes."""
    bin_centres = 0.5 * (bins[:-1] + bins[1:])

    # Phase-space MC (filled, background)
    if phsp_vals.size > 0:
        h, _ = norm_hist(phsp_vals, bins)
        ax.stairs(
            h, bins, fill=True, color="silver", alpha=0.55, label="Phase-Space MC", linewidth=0
        )
        ax.stairs(h, bins, color="grey", linewidth=0.9)

    # Signal MC
    for state, vals in mc_by_state.items():
        style = SIGNAL_STYLE.get(state, {"color": "gray", "label": state, "ls": "-", "lw": 1.5})
        h, _ = norm_hist(vals, bins)
        ax.stairs(
            h,
            bins,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=style["lw"],
            label=style["label"],
        )

    # Data
    h_d, h_d_err = norm_hist(data_vals, bins)
    ax.errorbar(
        bin_centres,
        h_d,
        yerr=h_d_err,
        fmt="o",
        color="black",
        markersize=3.5,
        linewidth=1.1,
        capsize=2,
        label="Data",
        zorder=10,
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Normalised entries / bin", fontsize=11)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bottom=0)
    if title_suffix:
        ax.set_title(title_suffix, fontsize=11)
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(which="major", ls=":", lw=0.5, alpha=0.45)


# ── Plot 1: PID product ──────────────────────────────────────────────────────


def plot_pid_product(data_vals, mc_vals_by_state, phsp_vals, out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bins = np.linspace(0, 1, 51)

    draw_overlay(
        ax,
        data_vals,
        mc_vals_by_state,
        phsp_vals,
        bins,
        xlabel=r"PID product  $P_\mathrm{NN}(\bar{p})\cdot P_\mathrm{NN}(K^+)\cdot P_\mathrm{NN}(K^-)$",
    )

    ax.set_title(
        r"PID product — pre-selection level"
        "\n"
        r"(trigger + $\Lambda^0$ cuts applied; no optimisation cuts)",
        fontsize=11,
        pad=6,
    )
    ax.text(
        0.97,
        0.97,
        "LHCb Run 2\n2016\u20132018",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="dimgray",
    )
    ax.legend(framealpha=0.88, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.99))

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✓ Saved: {out}")


# ── Plot 2: three individual PID variables ───────────────────────────────────


def plot_individual_pids(data_by_branch, mc_by_branch, phsp_by_branch, out: Path):
    """
    data_by_branch / mc_by_branch / phsp_by_branch:
      dict keyed by branch label, values are numpy arrays or dict{state: array}
    """
    branch_info = [
        ("p_ProbNNp", r"$P_\mathrm{NN}(p)$", r"Bachelor proton PNN"),
        ("h1_ProbNNk", r"$P_\mathrm{NN}(K^+)$", r"$K^+$ PNN"),
        ("h2_ProbNNk", r"$P_\mathrm{NN}(K^-)$", r"$K^-$ PNN"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    bins = np.linspace(0, 1, 41)  # 40 bins

    for ax, (br, xlabel, title) in zip(axes, branch_info):
        draw_overlay(
            ax,
            data_vals=data_by_branch[br],
            mc_by_state={s: mc_by_branch[br][s] for s in mc_by_branch[br]},
            phsp_vals=phsp_by_branch[br],
            bins=bins,
            xlabel=xlabel,
            title_suffix=title,
        )

    # Shared legend on first subplot only, others hidden
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        framealpha=0.88,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(
        "Individual PID variables — pre-selection level",
        fontsize=12,
        y=1.07,
    )
    fig.text(
        0.98,
        0.98,
        "LHCb Run 2 · 2016\u20132018",
        ha="right",
        va="top",
        fontsize=8.5,
        color="dimgray",
        transform=fig.transFigure,
    )

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✓ Saved: {out}")


# ── Data loading helpers ─────────────────────────────────────────────────────


def load_phsp(mc_base: Path, years, track_types) -> ak.Array:
    arrs = []
    for year in years:
        for tt in track_types:
            for mag in ["MD", "MU"]:
                fp = mc_base / "KpKm" / f"KpKm_{int(year)-2000}_{mag}.root"
                if fp.exists():
                    arrs.append(load_and_preprocess(fp, is_mc=True, track_type=tt))
    return ak.concatenate(arrs) if arrs else ak.Array([])


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    config = StudyConfig("box_config.toml")
    years = config.paths["years"]
    tt = config.paths.get("track_types", ["LL", "DD"])
    data_base = Path(config.paths["data_base_path"])
    mc_base = Path(config.paths["mc_base_path"])
    states = config.paths["mc_states"]

    BRANCHES = ["PID_product", "p_ProbNNp", "h1_ProbNNk", "h2_ProbNNk"]

    logger.info("Loading Data...")
    data_by_year = load_all_data(data_base, years, tt)
    data_comb = ak.concatenate(list(data_by_year.values()))
    data_vecs = {br: flat_arr(data_comb, br) for br in BRANCHES}
    logger.info(f"  Data: {data_vecs['PID_product'].size:,} events")

    logger.info("Loading Signal MC...")
    mc_by_state = load_all_mc(mc_base, states, years, tt)
    # {branch: {state: array}}
    mc_vecs = {br: {s: flat_arr(ev, br) for s, ev in mc_by_state.items()} for br in BRANCHES}

    logger.info("Loading Phase-Space MC (KpKm)...")
    phsp_events = load_phsp(mc_base, years, tt)
    phsp_vecs = {br: flat_arr(phsp_events, br) for br in BRANCHES}
    logger.info(f"  Phase-space MC: {phsp_vecs['PID_product'].size:,} events")

    # ── Plot 1: PID product ──
    plot_pid_product(
        data_vals=data_vecs["PID_product"],
        mc_vals_by_state={s: mc_vecs["PID_product"][s] for s in mc_by_state},
        phsp_vals=phsp_vecs["PID_product"],
        out=config.output_dir / "plots" / "pid_product_distribution.pdf",
    )

    # ── Plot 2: individual PIDs ──
    ind_data = {br: data_vecs[br] for br in ["p_ProbNNp", "h1_ProbNNk", "h2_ProbNNk"]}
    ind_mc = {br: mc_vecs[br] for br in ["p_ProbNNp", "h1_ProbNNk", "h2_ProbNNk"]}
    ind_phsp = {br: phsp_vecs[br] for br in ["p_ProbNNp", "h1_ProbNNk", "h2_ProbNNk"]}
    plot_individual_pids(
        data_by_branch=ind_data,
        mc_by_branch=ind_mc,
        phsp_by_branch=ind_phsp,
        out=config.output_dir / "plots" / "pid_individual_distribution.pdf",
    )

    logger.info("All done.")


if __name__ == "__main__":
    main()
