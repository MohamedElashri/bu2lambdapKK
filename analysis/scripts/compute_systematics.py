"""
Systematic Uncertainty Aggregation

Reads per-source systematic JSON files from each study and combines them in quadrature
into a total systematic per state per branch.

Sources included:
  Fit model     : generated/output/studies/fit_systematics/fit_systematics_{branch}_{cat}.json
  PID bootstrap : generated/output/studies/pid_cancellation/pid_bootstrap_systematics.json
       (relative systematic on efficiency ratio; applied to the branching fraction ratio)
  Tracking      : 0% (ratio measurement — see studies/standalone/tracking_systematic/)
  Kinematic     : not a separate JSON; the spread is bounded to < 2% based on the
       nominal weight range and is folded into the efficiency systematic via 4.2's error.
       If --kin-syst-rel is supplied, that fraction is used instead.
  Selection     : generated/output/studies/selection_systematic/selection_systematics_{branch}_{cat}.json

Output:
  {output_dir}/{branch}/tables/systematics.json
  {output_dir}/results/systematics_summary.md
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# States with MC pending (excluded from systematic tables)
MC_PENDING_STATES = {"etac_2s"}

STATES = ["jpsi", "etac", "chic0", "chic1"]


def load_json_if_exists(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def get_fit_syst(fit_json: dict, state: str, n_nom: float) -> float:
    """Return absolute fit systematic for this state."""
    if state not in fit_json:
        return 0.0
    entry = fit_json[state]
    return float(entry.get("fit_syst_abs", 0.0))


def get_sel_syst(sel_json: dict, state: str) -> float:
    """Return absolute selection systematic for this state."""
    if state not in sel_json:
        return 0.0
    return float(sel_json[state].get("sel_syst_abs", 0.0))


def get_pid_syst_rel(pid_json: dict, state: str) -> float:
    """Return relative PID systematic on the efficiency ratio."""
    # pid_json is keyed by state names like "chic0", "etac", etc.
    if state not in pid_json:
        return 0.0
    return float(pid_json[state].get("syst_rel", 0.0))


def aggregate_for_branch(
    branch: str,
    categories: list,
    output_dir: Path,
    studies_dir: Path,
    kin_syst_rel: float,
) -> dict:
    """Aggregate all systematics for one branch (combining LL and DD)."""
    results = {}

    for state in STATES:
        if state == "jpsi":
            continue  # reference state — systematic applies to ratio

        # --- Collect per-category fit and selection systematics (abs) ---
        fit_syst_abs_total = 0.0
        sel_syst_abs_total = 0.0
        n_nom_total = 0.0

        for cat in categories:
            fit_path = (
                studies_dir / "fit_systematics" / "output" / f"fit_systematics_{branch}_{cat}.json"
            )
            sel_path = (
                studies_dir
                / "selection_systematic"
                / "output"
                / f"selection_systematics_{branch}_{cat}.json"
            )

            fit_json = load_json_if_exists(fit_path)
            sel_json = load_json_if_exists(sel_path)

            # LL and DD are statistically independent — add in quadrature
            fit_syst_abs_total += get_fit_syst(fit_json, state, 0) ** 2
            sel_syst_abs_total += get_sel_syst(sel_json, state) ** 2

            n_nom = fit_json.get(state, {}).get("nominal_yield", 0.0)
            n_nom_total += n_nom

        fit_syst_abs = np.sqrt(fit_syst_abs_total)
        sel_syst_abs = np.sqrt(sel_syst_abs_total)

        # --- PID bootstrap (relative on efficiency ratio) ---
        pid_path = studies_dir / "pid_cancellation" / "output" / "pid_bootstrap_systematics.json"
        pid_json = load_json_if_exists(pid_path)
        pid_syst_rel = get_pid_syst_rel(pid_json, state)

        # --- Kinematic systematic (relative, applied to yield via efficiency) ---
        kin_syst_abs = kin_syst_rel * n_nom_total

        # --- Tracking: 0% ---
        trk_syst_abs = 0.0

        # --- Total systematic (quadrature sum of all absolute contributions) ---
        total_syst_abs = np.sqrt(
            fit_syst_abs**2
            + sel_syst_abs**2
            + (pid_syst_rel * n_nom_total) ** 2
            + kin_syst_abs**2
            + trk_syst_abs**2
        )
        total_syst_rel = total_syst_abs / n_nom_total if n_nom_total > 0 else 0.0

        results[state] = {
            "nominal_yield": n_nom_total,
            "fit_syst_abs": fit_syst_abs,
            "sel_syst_abs": sel_syst_abs,
            "pid_syst_rel": pid_syst_rel,
            "kin_syst_rel": kin_syst_rel,
            "trk_syst_abs": trk_syst_abs,
            "total_syst_abs": total_syst_abs,
            "total_syst_rel": total_syst_rel,
        }
        logger.info(
            f"  {state}: N={n_nom_total:.0f}  "
            f"fit={fit_syst_abs:.1f}  sel={sel_syst_abs:.1f}  "
            f"PID={100*pid_syst_rel:.1f}%  kin={100*kin_syst_rel:.1f}%  "
            f"→ σ_tot={total_syst_abs:.1f} ({100*total_syst_rel:.1f}%)"
        )

    return results


def write_summary_markdown(all_results: dict, output_path: Path):
    """Write a human-readable systematics summary table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Systematic Uncertainty Summary\n\n")
        f.write(
            "All values are absolute (in units of yield) unless labelled as relative (%).\n"
            "J/ψ is the normalization reference and does not appear in the ratio systematic.\n\n"
        )

        for branch, by_state in all_results.items():
            f.write(f"## Branch: {branch}\n\n")
            f.write(
                "| State | N_nom | Fit syst | Sel syst | PID syst | Kin syst | **Total** | Rel syst |\n"
            )
            f.write(
                "|-------|-------|----------|----------|----------|----------|-----------|----------|\n"
            )
            for state, v in by_state.items():
                f.write(
                    f"| {state} | {v['nominal_yield']:.0f} "
                    f"| {v['fit_syst_abs']:.1f} "
                    f"| {v['sel_syst_abs']:.1f} "
                    f"| {100*v['pid_syst_rel']:.1f}% "
                    f"| {100*v['kin_syst_rel']:.1f}% "
                    f"| **{v['total_syst_abs']:.1f}** "
                    f"| {100*v['total_syst_rel']:.1f}% |\n"
                )
            f.write("\n")

        f.write("## Notes\n\n")
        f.write("- **Fit syst**: max |δN| across 5 background/resolution variations\n")
        f.write("- **Sel syst**: max |δN| from BDT threshold ±1 step\n")
        f.write("- **PID syst**: RMS of efficiency ratio across 100 bootstrap iterations\n")
        f.write("- **Kin syst**: conservative bound from weight map variation\n")
        f.write("- **Tracking**: 0% — fully cancels in ratio (same final-state tracks)\n")
        f.write("- **etac_2s**: MC in LHCb production pipeline — excluded from this table\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Systematic uncertainty aggregation")
    parser.add_argument("--branches", nargs="+", default=["high_yield", "low_yield"])
    parser.add_argument("--categories", nargs="+", default=["LL", "DD"])
    parser.add_argument("--output-dir", default="generated/output/pipeline/mva")
    parser.add_argument("--studies-dir", default="generated/output/studies")
    parser.add_argument(
        "--kin-syst-rel",
        type=float,
        default=0.02,
        help="Conservative relative kinematic systematic (default 2%%)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    studies_dir = Path(args.studies_dir)
    all_results = {}

    for branch in args.branches:
        logger.info(f"=== Computing systematics for branch={branch} ===")
        by_state = aggregate_for_branch(
            branch=branch,
            categories=args.categories,
            output_dir=output_dir,
            studies_dir=studies_dir,
            kin_syst_rel=args.kin_syst_rel,
        )
        all_results[branch] = by_state

        # Save per-branch JSON
        out_json = output_dir / branch / "tables" / "systematics.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(by_state, f, indent=2)
        logger.info(f"Saved to {out_json}")

    # Write combined summary markdown
    summary_path = output_dir / "results" / "systematics_summary.md"
    write_summary_markdown(all_results, summary_path)
    logger.info(f"Summary saved to {summary_path}")
