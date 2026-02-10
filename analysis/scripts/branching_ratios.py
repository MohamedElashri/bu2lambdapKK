"""
Step 7 Snakemake wrapper: Branching Fraction Ratios

Reproduces the logic from:
  - PipelineManager.phase7_branching_ratios() in run_pipeline.py

Key physics:
  - Calculates BR ratios relative to J/ψ (self-normalization)
  - Uses yields from Step 5 and efficiencies from Step 6
  - Only ε_sel enters (other efficiencies CANCEL in ratios)
  - Statistical uncertainties only (draft analysis)

Snakemake injects the `snakemake` object with:
  snakemake.input.yields        — path to step5_yields.csv
  snakemake.input.efficiencies  — path to efficiencies.csv
  snakemake.input.ratios        — path to efficiency_ratios.csv
  snakemake.output.br_ratios    — path to branching_fraction_ratios.csv
  snakemake.output.final_results — path to final_results.md
  snakemake.params.config_dir
  snakemake.params.cache_dir
  snakemake.params.output_dir
"""

import sys
from pathlib import Path
from typing import Any

# Ensure the project root (analysis_make/) is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd

from modules.branching_fraction_calculator import BranchingFractionCalculator
from modules.cache_manager import CacheManager
from modules.data_handler import TOMLConfig
from modules.exceptions import AnalysisError

# ---------------------------------------------------------------------------
# Read Snakemake params
# ---------------------------------------------------------------------------
config_dir = snakemake.params.config_dir  # noqa: F821
cache_dir = snakemake.params.cache_dir  # noqa: F821
output_dir = snakemake.params.output_dir  # noqa: F821
yields_input_file = snakemake.input.yields  # noqa: F821

# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------
config = TOMLConfig(config_dir)
cache = CacheManager(cache_dir)

years = snakemake.config.get("years", ["2016", "2017", "2018"])  # noqa: F821
track_types = snakemake.config.get("track_types", ["LL", "DD"])  # noqa: F821
states = snakemake.config.get("states", ["jpsi", "etac", "chic0", "chic1"])  # noqa: F821


# ---------------------------------------------------------------------------
# Helper: compute step dependencies
# ---------------------------------------------------------------------------
def compute_step_dependencies(
    step: str, extra_params: dict[str, Any] | None = None
) -> dict[str, str]:
    """Compute dependencies for cache validation, matching run_pipeline.py logic."""
    config_files = list(Path(config_dir).glob("*.toml"))

    code_files: list[Path] = []
    if step == "5":
        code_files = [
            project_root / "modules" / "mass_fitter.py",
        ]
    elif step == "6":
        code_files = [
            project_root / "modules" / "efficiency_calculator.py",
        ]

    return cache.compute_dependencies(
        config_files=config_files, code_files=code_files, extra_params=extra_params
    )


print("\n" + "=" * 80)
print("STEP 7: BRANCHING FRACTION RATIOS")
print("=" * 80)

# ---------------------------------------------------------------------------
# Load yields from Step 5
# ---------------------------------------------------------------------------
# Try cache first (has the native dict format)
step5_deps = compute_step_dependencies(step="5", extra_params={"years": years})
fit_results = cache.load("step5_fit_results", dependencies=step5_deps)

if fit_results is not None and "yields" in fit_results:
    yields = fit_results["yields"]
    print("✓ Loaded yields from Step 5 cache")
else:
    # Reconstruct from CSV: {year: {state: (value, error)}}
    print(f"  Loading yields from {yields_input_file}")
    yields_df = pd.read_csv(yields_input_file)
    yields = {}
    for _, row in yields_df.iterrows():
        year = str(row["year"])
        state = row["state"]
        n = row["yield"]
        n_err = row["error"]
        if year not in yields:
            yields[year] = {}
        yields[year][state] = (n, n_err)
    print(f"✓ Loaded yields for {len(yields)} year(s) from CSV")

# ---------------------------------------------------------------------------
# Load efficiencies from Step 6
# ---------------------------------------------------------------------------
step6_deps = compute_step_dependencies(step="6", extra_params={"states": states, "years": years})
efficiencies = cache.load("step6_efficiencies", dependencies=step6_deps)

if efficiencies is None:
    raise AnalysisError(
        "Step 6 cached efficiencies not found! Run Step 6 (efficiency_calculation) first.\n"
        "This should not happen if the Snakemake DAG is correct."
    )

print(f"✓ Loaded efficiencies for {len(efficiencies)} state(s) from cache")

# ---------------------------------------------------------------------------
# Initialize calculator
# ---------------------------------------------------------------------------
bf_calculator = BranchingFractionCalculator(yields=yields, efficiencies=efficiencies, config=config)

# ---------------------------------------------------------------------------
# Calculate all ratios
# ---------------------------------------------------------------------------
print("\nCalculating branching fraction ratios...")
ratios_df = bf_calculator.calculate_all_ratios()

# ---------------------------------------------------------------------------
# Yield consistency check
# ---------------------------------------------------------------------------
print("\nChecking yield consistency across years...")
bf_calculator.check_yield_consistency_per_year()

# ---------------------------------------------------------------------------
# Generate final summary
# ---------------------------------------------------------------------------
bf_calculator.generate_final_summary(ratios_df)

print("\n✓ Step 7 complete: Results saved to tables/ and results/")
