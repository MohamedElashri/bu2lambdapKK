"""
2D FOM scan: PID_product threshold × MVA (BDT) threshold.

For each (pid_cut, bdt_cut) pair computes:
  S = ε_MC(PID_product > pid_cut AND BDT_score > bdt_cut) × N_signal_expected
  B = sideband-scaled background passing both cuts
  FOM = S / sqrt(S + B)

One heatmap per (category, group), annotated with FOM values, optimal cell
circled — matching the style of the existing 2D scan plots in the analysis.

Usage:
  uv run python plot_2d_fom_pid_mva.py [--category LL|DD|both]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import awkward as ak
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config_loader import StudyConfig
from data_preparation import load_and_prepare_data

_DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "generated"
    / "output"
    / "studies"
    / "mva_optimization"
)
OUTPUT_DIR = Path(os.environ.get("ANALYSIS_MVA_OUTPUT_DIR", str(_DEFAULT_OUTPUT_DIR)))

# ── Grid axes ────────────────────────────────────────────────────────────────
# PID grid starts at 0.25 (hard pre-cut floor in the cache)
PID_GRID = np.round(np.arange(0.25, 0.90, 0.05), 2)
BDT_GRID = np.round(np.arange(0.05, 1.00, 0.10), 2)

ALL_STATES = ["jpsi", "etac", "chic0", "chic1"]
HIGH_YIELD = ["jpsi", "etac"]
LOW_YIELD = ["chic0", "chic1"]
GROUPS = {"High_Yield": HIGH_YIELD, "Low_Yield": LOW_YIELD}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _extract_df(events, features: list[str]) -> pd.DataFrame:
    d = {}
    for f in features:
        br = events[f]
        if "var" in str(ak.type(br)):
            br = ak.firsts(br)
        d[f] = ak.to_numpy(br)
    return pd.DataFrame(d)


def _bdt_scores(model: CatBoostClassifier, events, features: list[str]) -> np.ndarray:
    return model.predict_proba(_extract_df(events, features)[features].values)[:, 1]


def _state_windows(config: StudyConfig, data_combined):
    """Return masks dict per state: {state: {in_sig, in_low_sb, in_high_sb}}."""
    opt = getattr(config, "optimization", {})
    b_sig_lo, b_sig_hi = opt.get("b_signal_region", [5255.0, 5305.0])
    b_low_lo, b_low_hi = opt.get("b_low_sideband", [5150.0, 5230.0])
    b_high_lo, b_high_hi = opt.get("b_high_sideband", [5330.0, 5410.0])

    bu_mm = data_combined["Bu_MM_corrected"]
    m_cc = (
        data_combined["M_LpKm_h2"]
        if "M_LpKm_h2" in data_combined.fields
        else data_combined["M_LpKm"]
    )

    signal_regions = getattr(config, "data", {}).get(
        "signal_regions", getattr(config, "signal_regions", {})
    )

    windows = {}
    for state in ALL_STATES:
        sr = signal_regions.get(state, {})
        c, w = sr.get("center", 0), sr.get("window", 0)
        in_cc = (m_cc > c - w) & (m_cc < c + w)
        windows[state] = {
            "in_sig": in_cc & ((bu_mm > b_sig_lo) & (bu_mm < b_sig_hi)),
            "in_low_sb": in_cc & ((bu_mm > b_low_lo) & (bu_mm < b_low_hi)),
            "in_high_sb": in_cc & ((bu_mm > b_high_lo) & (bu_mm < b_high_hi)),
            "b_sig_width": b_sig_hi - b_sig_lo,
            "b_low_width": b_low_hi - b_low_lo,
            "b_high_width": b_high_hi - b_high_lo,
        }
    return windows


def _n_expected_baseline(data_combined, windows: dict) -> dict[str, float]:
    """Sideband-subtracted signal yield at no extra cuts (pre-cut baseline)."""
    n_exp = {}
    for state in ALL_STATES:
        sw = windows[state]
        n_sr = float(ak.sum(sw["in_sig"]))
        n_lo = float(ak.sum(sw["in_low_sb"]))
        n_hi = float(ak.sum(sw["in_high_sb"]))
        d_lo = n_lo / sw["b_low_width"]
        d_hi = n_hi / sw["b_high_width"]
        b_est = (d_lo + d_hi) / 2.0 * sw["b_sig_width"]
        n_exp[state] = max(n_sr - b_est, 1.0)
    return n_exp


def _scan_2d(
    data_combined,
    data_bdt: np.ndarray,
    mc_prepared: dict,
    mc_bdt: dict[str, np.ndarray],
    windows: dict,
    n_exp: dict[str, float],
    pid_grid: np.ndarray,
    bdt_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return FOM grids shaped (len(bdt_grid), len(pid_grid)) for each group."""
    pid_arr = ak.to_numpy(data_combined["PID_product"])
    mc_pid = {st: ak.to_numpy(mc_prepared[st]["PID_product"]) for st in ALL_STATES}
    mc_totals = {st: len(mc_prepared[st]) for st in ALL_STATES}

    fom_grids = {g: np.zeros((len(bdt_grid), len(pid_grid))) for g in GROUPS}

    for j, bdt_cut in enumerate(bdt_grid):
        for i, pid_cut in enumerate(pid_grid):
            data_mask = (data_bdt > bdt_cut) & (pid_arr > pid_cut)
            state_sb = {}
            for state in ALL_STATES:
                mc_mask = (mc_bdt[state] > bdt_cut) & (mc_pid[state] > pid_cut)
                eps = np.sum(mc_mask) / mc_totals[state] if mc_totals[state] > 0 else 0.0
                s_est = eps * n_exp[state]

                sw = windows[state]
                n_lo = float(ak.sum(data_mask & sw["in_low_sb"]))
                n_hi = float(ak.sum(data_mask & sw["in_high_sb"]))
                b_est = (
                    ((n_lo / sw["b_low_width"]) + (n_hi / sw["b_high_width"]))
                    / 2.0
                    * sw["b_sig_width"]
                )
                state_sb[state] = (s_est, b_est)

            for group, states in GROUPS.items():
                s = sum(state_sb[st][0] for st in states)
                b = sum(state_sb[st][1] for st in states)
                fom_grids[group][j, i] = s / np.sqrt(s + b) if (s + b) > 0 else 0.0

    return fom_grids


def _plot_heatmap(
    fom_grid: np.ndarray,
    pid_grid: np.ndarray,
    bdt_grid: np.ndarray,
    group: str,
    category: str,
    out_dir: Path,
) -> None:
    """Annotated 2D FOM heatmap matching the analysis style."""
    fig, ax = plt.subplots(figsize=(len(pid_grid) * 0.85 + 1.5, len(bdt_grid) * 0.60 + 1.5))

    vmin, vmax = fom_grid.min(), fom_grid.max()
    im = ax.imshow(
        fom_grid,
        origin="lower",
        aspect="auto",
        cmap="YlGn",
        vmin=vmin,
        vmax=vmax,
        extent=(
            pid_grid[0] - (pid_grid[1] - pid_grid[0]) / 2,
            pid_grid[-1] + (pid_grid[1] - pid_grid[0]) / 2,
            bdt_grid[0] - (bdt_grid[1] - bdt_grid[0]) / 2,
            bdt_grid[-1] + (bdt_grid[1] - bdt_grid[0]) / 2,
        ),
    )

    # Annotate cells
    opt_idx = np.unravel_index(np.argmax(fom_grid), fom_grid.shape)
    for j in range(len(bdt_grid)):
        for i in range(len(pid_grid)):
            val = fom_grid[j, i]
            norm = (val - vmin) / (vmax - vmin + 1e-12)
            txt_color = "black" if norm > 0.5 else "white"
            ax.text(
                pid_grid[i],
                bdt_grid[j],
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=txt_color,
                fontweight="bold" if (j, i) == opt_idx else "normal",
            )
            if (j, i) == opt_idx:
                ax.add_patch(
                    plt.Circle(
                        (pid_grid[i], bdt_grid[j]),
                        radius=min(
                            (pid_grid[1] - pid_grid[0]) * 0.42,
                            (bdt_grid[1] - bdt_grid[0]) * 0.42,
                        ),
                        fill=False,
                        edgecolor="blue",
                        linewidth=1.5,
                    )
                )

    plt.colorbar(im, ax=ax, label="FOM  $S/\\sqrt{S+B}$", fraction=0.03, pad=0.02)

    group_label = "J/ψ + η$_c$" if group == "High_Yield" else "χ$_{c0}$ + χ$_{c1}$"
    ax.set_xlabel("PID product threshold", fontsize=10)
    ax.set_ylabel("MVA (BDT) threshold", fontsize=10)
    ax.set_title(
        f"$\\Lambda^0_{{{category}}}$ samples  —  {group_label}\n" f"$S/\\sqrt{{S+B}}$  (2D scan)",
        fontsize=10,
    )
    ax.set_xticks(pid_grid)
    ax.set_yticks(bdt_grid)
    ax.tick_params(direction="in", top=True, right=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"fom_2d_pid_mva_{group}_{category}.pdf"
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {fname}")


# ── Main ─────────────────────────────────────────────────────────────────────


def run_for_category(category: str) -> None:
    config = StudyConfig()
    ml_data = load_and_prepare_data(config, category=category)
    features: list[str] = ml_data["features"]
    data_combined = ml_data["data_combined"]
    mc_prepared: dict = ml_data["mc_prepared"]

    # Load trained model
    model_dir = OUTPUT_DIR / "models"
    model_path = model_dir / f"catboost_bdt_{category}.cbm"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}\n"
            "Run the MVA study first: uv run python main.py --category both"
        )
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    logger.info(f"[{category}] Loaded model: {model_path}")

    # Score all events
    data_bdt = _bdt_scores(model, data_combined, features)
    mc_bdt = {
        st: _bdt_scores(model, mc_prepared[st], features) for st in ALL_STATES if st in mc_prepared
    }

    # Precompute windows and baseline expected yields
    windows = _state_windows(config, data_combined)
    n_exp = _n_expected_baseline(data_combined, windows)
    logger.info(
        f"[{category}] Baseline signal yields: { {k: f'{v:.1f}' for k, v in n_exp.items()} }"
    )

    # 2D scan
    logger.info(
        f"[{category}] Scanning {len(PID_GRID)} × {len(BDT_GRID)} = {len(PID_GRID)*len(BDT_GRID)} points..."
    )
    fom_grids = _scan_2d(
        data_combined,
        data_bdt,
        mc_prepared,
        mc_bdt,
        windows,
        n_exp,
        PID_GRID,
        BDT_GRID,
    )

    # Plot
    plot_dir = OUTPUT_DIR / "plots" / "mva"
    for group, grid in fom_grids.items():
        opt_j, opt_i = np.unravel_index(np.argmax(grid), grid.shape)
        logger.info(
            f"[{category}] {group} optimal: PID>{PID_GRID[opt_i]:.2f}, "
            f"BDT>{BDT_GRID[opt_j]:.2f}, FOM={grid[opt_j, opt_i]:.3f}"
        )
        _plot_heatmap(grid, PID_GRID, BDT_GRID, group, category, plot_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="2D FOM scan: PID × MVA threshold")
    parser.add_argument("--category", choices=["LL", "DD", "both"], default="both")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        os.environ["ANALYSIS_MVA_OUTPUT_DIR"] = str(OUTPUT_DIR)

    cats = ["LL", "DD"] if args.category == "both" else [args.category]
    for cat in cats:
        run_for_category(cat)


if __name__ == "__main__":
    main()
