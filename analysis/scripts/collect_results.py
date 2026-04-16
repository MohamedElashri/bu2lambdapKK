"""
Collect all analysis outputs into a single organised reports directory.

Run via:   snakemake collect_results
Or:        uv run python scripts/collect_results.py --output-dir generated/output/reports/collected

Layout created:
  generated/output/reports/collected/
    final/
      bf_products.tex               ← LaTeX BF table
      final_results_high_yield.md   ← per-branch summary
      final_results_low_yield.md
      branch_comparison.md          ← high vs low comparison
    tables/
      branching_fraction_ratios_high_yield.csv
      branching_fraction_ratios_low_yield.csv
      systematics_high_yield.json
      systematics_low_yield.json
      systematics_summary.md
    plots/
      mass_fits/
        high_yield_LL/  mass_fit_{year}.pdf ...
        high_yield_DD/
        low_yield_LL/
        low_yield_DD/
    studies/
      mva/           ← MVA model files and reports
      efficiency/    ← efficiencies + trigger
      reweighting/   ← kinematic weights
      pid/           ← PID cancellation + bootstrap
      fit_syst/      ← fit systematic JSONs
      sel_syst/      ← selection systematic JSONs
"""

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def copy_if_exists(src: Path, dst: Path) -> bool:
    """Copy src → dst, creating parent dirs. Returns True if copied."""
    if not src.exists():
        logger.debug(f"  skip (missing): {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    logger.info(f"  {src.name} → {dst.relative_to(dst.parents[max(0, len(dst.parts)-4)])}")
    return True


def copy_dir_contents(src_dir: Path, dst_dir: Path, pattern: str = "*") -> int:
    """Copy all files matching pattern from src_dir into dst_dir. Returns count."""
    if not src_dir.is_dir():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(src_dir.glob(pattern)):
        if f.is_file():
            shutil.copy2(f, dst_dir / f.name)
            count += 1
    return count


def first_existing(*paths: Path) -> Path | None:
    """Return the first existing path from a list of legacy/current candidates."""
    for path in paths:
        if path.exists():
            return path
    return None


def collect(pipeline_dir: Path, studies_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Collecting results into {out_dir}/")

    # ------------------------------------------------------------------ #
    # final/                                                               #
    # ------------------------------------------------------------------ #
    final = out_dir / "final"
    copy_if_exists(pipeline_dir / "results" / "bf_products.tex", final / "bf_products.tex")
    copy_if_exists(
        pipeline_dir / "results" / "systematics_summary.md", final / "systematics_summary.md"
    )
    copy_if_exists(
        pipeline_dir / "comparison" / "branch_comparison.md", final / "branch_comparison.md"
    )
    for branch in ("high_yield", "low_yield"):
        copy_if_exists(
            pipeline_dir / branch / "results" / "final_results.md",
            final / f"final_results_{branch}.md",
        )

    # ------------------------------------------------------------------ #
    # tables/                                                              #
    # ------------------------------------------------------------------ #
    tables = out_dir / "tables"
    for branch in ("high_yield", "low_yield"):
        copy_if_exists(
            pipeline_dir / branch / "tables" / "branching_fraction_ratios.csv",
            tables / f"branching_fraction_ratios_{branch}.csv",
        )
        copy_if_exists(
            pipeline_dir / branch / "tables" / "systematics.json",
            tables / f"systematics_{branch}.json",
        )
    copy_if_exists(
        pipeline_dir / "results" / "systematics_summary.md", tables / "systematics_summary.md"
    )

    # ------------------------------------------------------------------ #
    # plots/mass_fits/                                                     #
    # ------------------------------------------------------------------ #
    for branch in ("high_yield", "low_yield"):
        for cat in ("LL", "DD", "LL_DD"):
            src = pipeline_dir / branch / cat / "plots" / "fits"
            dst = out_dir / "plots" / "mass_fits" / f"{branch}_{cat}"
            n = copy_dir_contents(src, dst, "*.pdf")
            if n:
                logger.info(f"  {n} fit plots → plots/mass_fits/{branch}_{cat}/")

    # ------------------------------------------------------------------ #
    # studies/                                                             #
    # ------------------------------------------------------------------ #

    # MVA
    mva_src = first_existing(
        studies_dir / "mva_optimization", studies_dir / "mva_optimization" / "output"
    )
    mva_dst = out_dir / "studies" / "mva"
    if mva_src is not None:
        for cat in ("LL", "DD"):
            copy_if_exists(
                mva_src / "models" / f"catboost_bdt_{cat}.cbm", mva_dst / f"catboost_bdt_{cat}.cbm"
            )
            copy_if_exists(
                mva_src / f"mva_optimization_report_{cat}.txt",
                mva_dst / f"mva_optimization_report_{cat}.txt",
            )
        for md in (mva_src / "tables").glob("*.md"):
            copy_if_exists(md, mva_dst / md.name)

    # Efficiency steps
    eff_dst = out_dir / "studies" / "efficiency"
    efficiency_dir = first_existing(
        studies_dir / "efficiency_steps",
        studies_dir / "efficiency_steps" / "output",
    )
    if efficiency_dir is not None:
        for branch in ("high_yield", "low_yield"):
            copy_if_exists(
                efficiency_dir / f"efficiencies_{branch}.json",
                eff_dst / f"efficiencies_{branch}.json",
            )
            copy_if_exists(
                efficiency_dir / f"efficiencies_tables_{branch}.md",
                eff_dst / f"efficiencies_tables_{branch}.md",
            )
        copy_if_exists(efficiency_dir / "efficiencies.json", eff_dst / "efficiencies.json")
        copy_if_exists(
            efficiency_dir / "efficiencies_tables.md", eff_dst / "efficiencies_tables.md"
        )
    trigger_dir = first_existing(
        studies_dir / "trigger_tis_tos",
        studies_dir / "trigger_tis_tos" / "output",
    )
    if trigger_dir is not None:
        copy_if_exists(trigger_dir / "tis_tos_results.json", eff_dst / "tis_tos_results.json")

    # Kinematic reweighting
    rew_src = first_existing(
        studies_dir / "kinematic_reweighting",
        studies_dir / "kinematic_reweighting" / "output",
    )
    rew_dst = out_dir / "studies" / "reweighting"
    if rew_src is not None:
        for f in (
            "kinematic_weights_LL.json",
            "kinematic_weights_DD.json",
            "kinematic_weights.json",
        ):
            copy_if_exists(rew_src / f, rew_dst / f)
        copy_dir_contents(rew_src, rew_dst, "*.pdf")

    # PID cancellation
    pid_src = first_existing(
        studies_dir / "pid_cancellation",
        studies_dir / "pid_cancellation" / "output",
    )
    pid_dst = out_dir / "studies" / "pid"
    if pid_src is not None:
        copy_if_exists(
            pid_src / "pid_bootstrap_systematics.json", pid_dst / "pid_bootstrap_systematics.json"
        )
        copy_if_exists(pid_src / "real_pidcalib_results.md", pid_dst / "real_pidcalib_results.md")
        copy_dir_contents(pid_src, pid_dst, "*.pdf")
        copy_dir_contents(pid_src, pid_dst, "*.png")

    # Fit systematics
    fit_syst_src = first_existing(
        studies_dir / "fit_systematics",
        studies_dir / "fit_systematics" / "output",
    )
    fit_syst_dst = out_dir / "studies" / "fit_syst"
    if fit_syst_src is not None:
        copy_dir_contents(fit_syst_src, fit_syst_dst, "*.json")

    # Selection systematics
    sel_syst_src = first_existing(
        studies_dir / "selection_systematic",
        studies_dir / "selection_systematic" / "output",
    )
    sel_syst_dst = out_dir / "studies" / "sel_syst"
    if sel_syst_src is not None:
        copy_dir_contents(sel_syst_src, sel_syst_dst, "*.json")

    logger.info(f"\nDone. All results collected in {out_dir}/")
    _print_tree(out_dir)


def _print_tree(root: Path, prefix: str = "", max_depth: int = 3, depth: int = 0):
    if depth > max_depth:
        return
    entries = sorted(root.iterdir()) if root.is_dir() else []
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(f"{prefix}{connector}{entry.name}")
        if entry.is_dir() and depth < max_depth:
            extension = "    " if i == len(entries) - 1 else "│   "
            _print_tree(entry, prefix + extension, max_depth, depth + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect all analysis results into one folder")
    parser.add_argument(
        "--pipeline-dir",
        default="generated/output/pipeline/mva",
        help="Main pipeline output dir",
    )
    parser.add_argument(
        "--studies-dir", default="generated/output/studies", help="Generated studies directory"
    )
    parser.add_argument(
        "--output-dir",
        default="generated/output/reports/collected",
        help="Destination folder",
    )
    args = parser.parse_args()

    collect(
        pipeline_dir=Path(args.pipeline_dir),
        studies_dir=Path(args.studies_dir),
        out_dir=Path(args.output_dir),
    )
