"""
Generate current box-optimization reference cuts from the modern preselection cache.

Why this exists:
- The production box branch in the Snakefile still routes through the legacy
  sequential Option-C path.
- The selection config has since evolved (e.g. log_bu_ipchi2, fixed PID), so
  that path no longer runs unchanged.
- For supervisor/comparison material we still want a code-backed "box" baseline.

This helper uses the maintained grouped N-D grid scan in modules.box_optimizer,
then writes the resulting cut table into the same JSON format expected by
apply_cuts.py.  The same JSON is written for both high_yield and low_yield
branches; apply_cuts.py selects the relevant state/FoM rows itself.

Usage:
    uv run python scripts/generate_box_reference.py --category LL
    uv run python scripts/generate_box_reference.py --category DD
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.box_optimizer import SelectionOptimizer
from modules.cache_manager import CacheManager
from modules.config_loader import StudyConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_box_reference(category: str) -> None:
    config_dir = project_root / "config"
    output_root = project_root / "analysis_output" / "box"
    cache_dir = output_root / "cache"

    config = StudyConfig.from_dir(config_dir, output_dir=output_root)
    # Force the maintained grouped scan rather than the stale sequential path.
    config.optimization["method"] = "mc_based_grouped_reference"

    cache = CacheManager(cache_dir=cache_dir)
    deps = cache.compute_dependencies(
        config_files=config.config_paths(),
        code_files=[
            project_root / "modules" / "clean_data_loader.py",
            project_root / "scripts" / "load_data.py",
        ],
        extra_params={"years": ["2016", "2017", "2018"], "track_types": ["LL", "DD"]},
    )

    data_full = cache.load("preprocessed_data", dependencies=deps)
    mc_full = cache.load("preprocessed_mc", dependencies=deps)
    if data_full is None or mc_full is None:
        raise RuntimeError(
            "Missing box preprocessed cache. Run the box branch load step first "
            "(e.g. snakemake --config opt_method=box analysis_output/box/.data_loaded)."
        )

    data = {yr: data_full[yr][category] for yr in data_full if category in data_full[yr]}
    mc = {st: mc_full[st][category] for st in mc_full if category in mc_full[st]}

    logger.info("Running grouped box reference scan for category=%s", category)
    optimizer = SelectionOptimizer(data=data, config=config, mc_data=mc)
    results_df = optimizer.optimize_nd_grid_scan_mc_based()
    results_payload = results_df.to_dict(orient="records")

    for branch in ("high_yield", "low_yield"):
        out_dir = output_root / branch / category / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "optimized_cuts.json"
        with open(out_file, "w") as f:
            json.dump(results_payload, f, indent=2)
        logger.info("Wrote %s", out_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=["LL", "DD"], required=True)
    args = parser.parse_args()
    build_box_reference(args.category)


if __name__ == "__main__":
    main()
