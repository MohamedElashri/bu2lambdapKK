"""
Compare PID_product between MC15TuneV1 and the default (MC12TuneV4) tune.

PID_product = ProbNNp(p) * ProbNNk(h1) * ProbNNk(h2)

Selection: trigger only.
Both data and J/psi MC are compared.

Output:
  output/pid_tune_comparison_data_LL.pdf
  output/pid_tune_comparison_data_DD.pdf
  output/pid_tune_comparison_mc_LL.pdf
  output/pid_tune_comparison_mc_DD.pdf
  output/pid_tune_comparison_overlay_LL.pdf  (data + mc, both tunes)
  output/pid_tune_comparison_overlay_DD.pdf

Run from analysis/ directory:
    uv run python studies/standalone/pid_tune_comparison/compare_pid_tunes.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parents[2]  # analyses/bu2lambdapKK/analysis/
sys.path.insert(0, str(ANALYSIS_DIR))

from modules.presentation_config import (
    DATA_L0_TIS_KEYS,
    HLT1_TOS_KEYS,
    HLT2_TOS_KEYS,
    MC_L0_TIS_KEYS,
    get_presentation_config,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = SCRIPT_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

PRES = get_presentation_config()

# Tune definitions: (label, p_branch, h1_branch, h2_branch, color, linestyle)
TUNES = [
    (
        "MC15TuneV1 (correct)",
        "p_MC15TuneV1_ProbNNp",
        "h1_MC15TuneV1_ProbNNk",
        "h2_MC15TuneV1_ProbNNk",
        "#1f77b4",
        "-",
    ),
    (
        "Default (MC12TuneV4)",
        "p_ProbNNp",
        "h1_ProbNNk",
        "h2_ProbNNk",
        "#d62728",
        "--",
    ),
]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.direction": "in",
        "xtick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
        "axes.linewidth": 1.2,
        "figure.dpi": 150,
    }
)

BINS = 20
X_RANGE = (0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────


def _trigger_mask(ev: dict, n: int, tis_keys, hlt1_keys, hlt2_keys) -> np.ndarray:
    def _or(keys):
        m = np.zeros(n, dtype=bool)
        found = False
        for k in keys:
            if k in ev:
                m |= ev[k] > 0
                found = True
        return m if found else np.ones(n, dtype=bool)

    return _or(tis_keys) & _or(hlt1_keys) & _or(hlt2_keys)


def _load_events(path: Path, cat: str, is_mc: bool) -> dict | None:
    """Load all PID branches for both tunes — trigger only, no other selection."""
    pid_branches = []
    for _, p_b, h1_b, h2_b, _, _ in TUNES:
        pid_branches += [p_b, h1_b, h2_b]

    tis_keys = MC_L0_TIS_KEYS if is_mc else DATA_L0_TIS_KEYS
    all_trig = tis_keys + HLT1_TOS_KEYS + HLT2_TOS_KEYS
    want = pid_branches + all_trig

    try:
        tree = uproot.open(path)[f"B2L0barPKpKm_{cat}/DecayTree"]
    except Exception as e:
        log.warning(f"Cannot open {path}: {e}")
        return None

    avail = set(tree.keys())
    load = [b for b in want if b in avail]
    ev = tree.arrays(load, library="np")
    n = len(ev[load[0]])

    mask = _trigger_mask(ev, n, tis_keys, HLT1_TOS_KEYS, HLT2_TOS_KEYS)
    return {k: v[mask] for k, v in ev.items() if k in avail}


def _collect(cat: str, is_mc: bool) -> dict:
    combined: dict[str, list] = {}
    if is_mc:
        for yr in PRES.year_suffixes:
            for mag in PRES.magnets:
                p = PRES.mc_base / "Jpsi" / f"Jpsi_{yr}_{mag}.root"
                if not p.exists():
                    continue
                ev = _load_events(p, cat, is_mc=True)
                if ev:
                    for k, v in ev.items():
                        combined.setdefault(k, []).append(v)
    else:
        for yr in PRES.year_suffixes:
            for mag in PRES.magnets:
                p = PRES.data_base / f"dataBu2L0barPHH_{yr}{mag}.root"
                if not p.exists():
                    continue
                ev = _load_events(p, cat, is_mc=False)
                if ev:
                    for k, v in ev.items():
                        combined.setdefault(k, []).append(v)

    return {k: np.concatenate(v) for k, v in combined.items()} if combined else {}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────


def _pid_product(ev: dict, p_b: str, h1_b: str, h2_b: str) -> np.ndarray | None:
    if all(b in ev for b in (p_b, h1_b, h2_b)):
        return ev[p_b] * ev[h1_b] * ev[h2_b]
    return None


def _norm_hist(arr: np.ndarray, bins: int, x_range: tuple) -> tuple:
    h, edges = np.histogram(arr, bins=bins, range=x_range)
    bw = (x_range[1] - x_range[0]) / bins
    total = h.sum() * bw
    if total <= 0:
        return h / 1.0, edges, np.zeros_like(h, dtype=float)
    return h / total, edges, np.sqrt(h) / total


def _plot_single(ev: dict, cat: str, source_label: str, out_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    any_drawn = False
    for label, p_b, h1_b, h2_b, color, ls in TUNES:
        prod = _pid_product(ev, p_b, h1_b, h2_b)
        if prod is None or len(prod) == 0:
            log.warning(f"  Tune '{label}' unavailable for {cat} {source_label}")
            continue
        h_n, edges, err_n = _norm_hist(prod, BINS, X_RANGE)
        centers = 0.5 * (edges[1:] + edges[:-1])
        mask = h_n > 0
        ax.errorbar(
            centers[mask],
            h_n[mask],
            yerr=err_n[mask],
            fmt="o",
            color=color,
            markersize=4,
            label=label,
            linestyle=ls,
            linewidth=1.5,
        )
        any_drawn = True

    if not any_drawn:
        plt.close(fig)
        return

    ax.set_xlabel(
        r"$\mathrm{ProbNNp}(p) \times \mathrm{ProbNNk}(h_1) \times \mathrm{ProbNNk}(h_2)$"
    )
    ax.set_ylabel(rf"Normalised / {(X_RANGE[1]-X_RANGE[0])/BINS:.2f}")
    ax.set_title(rf"PID product tune comparison — $\Lambda${cat}, {source_label} (trigger only)")
    ax.legend(frameon=False)
    ax.set_xlim(*X_RANGE)

    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {out}")


def _plot_overlay(data_ev: dict, mc_ev: dict, cat: str, out_name: str) -> None:
    """Four-panel: rows = tune, cols = data/MC."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(rf"PID product: tune comparison — $\Lambda${cat}, trigger only", fontsize=13)

    for row, (label, p_b, h1_b, h2_b, color, ls) in enumerate(TUNES):
        for col, (ev, src_label) in enumerate([(data_ev, "Data"), (mc_ev, r"$J/\psi$ MC")]):
            ax = axes[row, col]
            prod = _pid_product(ev, p_b, h1_b, h2_b)
            if prod is None or len(prod) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue
            h_n, edges, err_n = _norm_hist(prod, BINS, X_RANGE)
            centers = 0.5 * (edges[1:] + edges[:-1])
            mask_bins = h_n > 0
            ax.errorbar(
                centers[mask_bins],
                h_n[mask_bins],
                yerr=err_n[mask_bins],
                fmt="o",
                color=color,
                markersize=4,
                linestyle=ls,
                linewidth=1.5,
            )
            bw = (X_RANGE[1] - X_RANGE[0]) / BINS
            ax.set_xlabel(r"$\mathrm{PID}_{\mathrm{product}}$")
            ax.set_ylabel(rf"Normalised / {bw:.2f}")
            ax.set_title(f"{label}\n{src_label} (N={len(prod)})")
            ax.set_xlim(*X_RANGE)

    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {out}")


def _plot_overlay_same_panel(data_ev: dict, mc_ev: dict, cat: str, out_name: str) -> None:
    """Two panels side-by-side: left = data, right = MC. Both tunes overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(rf"PID product tune comparison — $\Lambda${cat}, trigger only", fontsize=13)

    for col, (ev, src_label) in enumerate([(data_ev, "Data"), (mc_ev, "MC")]):
        ax = axes[col]
        drawn = False
        for label, p_b, h1_b, h2_b, color, ls in TUNES:
            prod = _pid_product(ev, p_b, h1_b, h2_b)
            if prod is None or len(prod) == 0:
                continue
            h_n, edges, err_n = _norm_hist(prod, BINS, X_RANGE)
            centers = 0.5 * (edges[1:] + edges[:-1])
            mask_bins = h_n > 0
            ax.errorbar(
                centers[mask_bins],
                h_n[mask_bins],
                yerr=err_n[mask_bins],
                fmt="o",
                color=color,
                markersize=4,
                label=f"{label} (N={len(prod)})",
                linestyle=ls,
                linewidth=1.5,
            )
            drawn = True

        if not drawn:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        bw = (X_RANGE[1] - X_RANGE[0]) / BINS
        ax.set_xlabel(
            r"$p \cdot \mathrm{ProbNNp} \times h_1 \cdot \mathrm{ProbNNk} \times h_2 \cdot \mathrm{ProbNNk}$"
        )
        ax.set_ylabel(rf"Normalised / {bw:.2f}")
        ax.set_title(rf"$\Lambda${cat} — {src_label}")
        ax.legend(frameon=False, fontsize=9)
        ax.set_xlim(*X_RANGE)

    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    for cat in ("LL", "DD"):
        log.info(f"=== Lambda{cat} ===")

        log.info("  Loading data …")
        data_ev = _collect(cat, is_mc=False)
        log.info("  Loading MC …")
        mc_ev = _collect(cat, is_mc=True)

        if not data_ev:
            log.warning(f"  No data events for {cat}, skipping")
            continue
        if not mc_ev:
            log.warning(f"  No MC events for {cat}, skipping")
            continue

        n_data = len(next(iter(data_ev.values())))
        n_mc = len(next(iter(mc_ev.values())))
        log.info(f"  Data: {n_data} events, MC: {n_mc} events")

        # Separate data/MC single-source plots
        _plot_single(data_ev, cat, "Data", f"pid_tune_comparison_data_{cat}.pdf")
        _plot_single(mc_ev, cat, r"$J/\psi$ MC", f"pid_tune_comparison_mc_{cat}.pdf")

        # Side-by-side overlay (main result for advisor)
        _plot_overlay_same_panel(data_ev, mc_ev, cat, f"pid_tune_comparison_overlay_{cat}.pdf")

        # 2×2 grid (tune × source) for completeness
        _plot_overlay(data_ev, mc_ev, cat, f"pid_tune_comparison_grid_{cat}.pdf")

    log.info("=== Done ===")


if __name__ == "__main__":
    main()
