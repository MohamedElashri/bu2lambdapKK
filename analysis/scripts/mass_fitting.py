"""
Step 5 Snakemake wrapper: Mass Fitting with RooFit

Reproduces the logic from:
  - PipelineManager.phase5_mass_fitting() in run_pipeline.py

Key physics:
  - Fits ALL charmonium states simultaneously to the SAME data sample
  - Data has only Lambda pre-selection (no state-specific cuts)
  - Uses RooVoigtian signals + ARGUS background
  - Fits per-year AND combined (if multiple years)
  - Extracts yields per state per year

IMPORTANT: This step uses ROOT/PyROOT which is globally installed,
NOT managed by uv.

Snakemake injects the `snakemake` object with:
  snakemake.input[0]            — path to step4_summary.json (dependency)
  snakemake.output[0]           — path to step5_yields.csv
  snakemake.params.no_cache     — bool: force reprocessing
  snakemake.params.config_dir   — path to TOML config directory
  snakemake.params.cache_dir    — path to cache directory
  snakemake.params.output_dir   — path to output directory root
"""

import sys
from pathlib import Path
from typing import Any

# Ensure the project root (analysis_make/) is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import awkward as ak
import pandas as pd

from modules.cache_manager import CacheManager
from modules.data_handler import TOMLConfig
from modules.exceptions import AnalysisError
from modules.mass_fitter import MassFitter
from utils.logging_config import suppress_warnings

# Suppress warnings by default
suppress_warnings()

# ---------------------------------------------------------------------------
# Read Snakemake params
# ---------------------------------------------------------------------------
no_cache = snakemake.params.no_cache  # noqa: F821
config_dir = snakemake.params.config_dir  # noqa: F821
cache_dir = snakemake.params.cache_dir  # noqa: F821
output_dir = snakemake.params.output_dir  # noqa: F821
yields_output_file = snakemake.output[0]  # noqa: F821

# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------
config = TOMLConfig(config_dir)
cache = CacheManager(cache_dir)

print("\n" + "=" * 80)
print("STEP 5: MASS FITTING")
print("=" * 80)


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

    return cache.compute_dependencies(
        config_files=config_files, code_files=code_files, extra_params=extra_params
    )


# ---------------------------------------------------------------------------
# Load Step 4 cached data (data_final)
# ---------------------------------------------------------------------------
data_final = cache.load("step4_data_final")

if data_final is None:
    raise AnalysisError(
        "Step 4 cached data_final not found! Run Step 4 (apply_cuts) first.\n"
        "This should not happen if the Snakemake DAG is correct."
    )

# ---------------------------------------------------------------------------
# Check cache for Step 5 results
# ---------------------------------------------------------------------------
use_cached = not no_cache
years_list = list(data_final.keys())

if use_cached:
    dependencies = compute_step_dependencies(step="5", extra_params={"years": years_list})
    cached = cache.load("step5_fit_results", dependencies=dependencies)
    if cached is not None:
        print("✓ Loaded cached fit results")

        # Save yields CSV from cached results
        yields_data = []
        for year in cached["yields"]:
            for state in cached["yields"][year]:
                n, n_err = cached["yields"][year][state]
                yields_data.append({"year": year, "state": state, "yield": n, "error": n_err})

        yields_df = pd.DataFrame(yields_data)
        yields_df.to_csv(yields_output_file, index=False)
        print(f"\n✓ Step 5 complete: Yields saved to {yields_output_file}")
        sys.exit(0)

# ---------------------------------------------------------------------------
# Initialize fitter
# ---------------------------------------------------------------------------
fitter = MassFitter(config)

# ---------------------------------------------------------------------------
# Combine LL and DD track types for each year
# ---------------------------------------------------------------------------
data_combined = {}
for year in data_final:
    events_list = []
    for track_type in data_final[year]:
        events_list.append(data_final[year][track_type])
    data_combined[year] = ak.concatenate(events_list)

# ---------------------------------------------------------------------------
# Perform fits
# ---------------------------------------------------------------------------
print("\nFitting charmonium mass spectrum...")
fit_results = fitter.perform_fit(data_combined)

# ---------------------------------------------------------------------------
# Cache results
# ---------------------------------------------------------------------------
dependencies = compute_step_dependencies(step="5", extra_params={"years": years_list})
cache.save(
    "step5_fit_results",
    fit_results,
    dependencies=dependencies,
    description="Mass fit results",
)

# ---------------------------------------------------------------------------
# Save yields table
# ---------------------------------------------------------------------------
yields_data = []
for year in fit_results["yields"]:
    for state in fit_results["yields"][year]:
        n, n_err = fit_results["yields"][year][state]
        yields_data.append({"year": year, "state": state, "yield": n, "error": n_err})

yields_df = pd.DataFrame(yields_data)
tables_dir = Path(config.paths["output"]["tables_dir"])
tables_dir.mkdir(exist_ok=True, parents=True)
yields_df.to_csv(yields_output_file, index=False)

print(f"\n✓ Step 5 complete: Yields saved to {yields_output_file}")
