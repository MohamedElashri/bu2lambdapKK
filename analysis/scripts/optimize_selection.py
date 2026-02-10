"""
Step 3 Snakemake wrapper: Selection Optimization

Reproduces the logic from:
  - PipelineManager.phase3_selection_optimization() in run_pipeline.py
  - PipelineManager._create_manual_cuts_dataframe() in run_pipeline.py

Snakemake injects the `snakemake` object with:
  snakemake.params.use_manual_cuts  — bool: use manual cuts from config
  snakemake.params.no_cache         — bool: force reprocessing
  snakemake.params.config_dir       — path to TOML config directory
  snakemake.params.cache_dir        — path to cache directory
  snakemake.params.output_dir       — path to output directory root
  snakemake.output[0]               — path to optimized_cuts.csv
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
from modules.selection_optimizer import SelectionOptimizer

# ---------------------------------------------------------------------------
# Read Snakemake params
# ---------------------------------------------------------------------------
use_manual_cuts = snakemake.params.use_manual_cuts  # noqa: F821
no_cache = snakemake.params.no_cache  # noqa: F821
config_dir = snakemake.params.config_dir  # noqa: F821
cache_dir = snakemake.params.cache_dir  # noqa: F821
output_dir = snakemake.params.output_dir  # noqa: F821
cuts_output_file = snakemake.output[0]  # noqa: F821

# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------
config = TOMLConfig(config_dir)
cache = CacheManager(cache_dir)

print("\n" + "=" * 80)
print("STEP 3: SELECTION OPTIMIZATION")
print("=" * 80)

tables_dir = Path(config.paths["output"]["tables_dir"])
cuts_file = tables_dir / "optimized_cuts.csv"


# ---------------------------------------------------------------------------
# Helper: compute step dependencies
# ---------------------------------------------------------------------------
def compute_step_dependencies(
    step: str, extra_params: dict[str, Any] | None = None
) -> dict[str, str]:
    """Compute dependencies for cache validation, matching run_pipeline.py logic."""
    config_files = list(Path(config_dir).glob("*.toml"))

    code_files: list[Path] = []
    if step == "2":
        code_files = [
            project_root / "modules" / "data_handler.py",
            project_root / "modules" / "lambda_selector.py",
        ]
    elif step == "3":
        code_files = [
            project_root / "modules" / "selection_optimizer.py",
        ]

    return cache.compute_dependencies(
        config_files=config_files, code_files=code_files, extra_params=extra_params
    )


# ---------------------------------------------------------------------------
# Helper: create manual cuts DataFrame
# ---------------------------------------------------------------------------
def create_manual_cuts_dataframe(manual_cuts_config: dict[str, Any]) -> pd.DataFrame:
    """Convert manual cuts from config to optimized_cuts_df format."""
    nd_config = config.selection.get("nd_optimizable_selection", {})

    all_results = []
    states = ["jpsi", "etac", "chic0", "chic1", "etac_2s"]

    print("\nManual cuts specified:")
    for branch_name, cut_spec in manual_cuts_config.items():
        if branch_name == "notes":
            continue

        cut_type = cut_spec.get("cut_type")
        cut_value = cut_spec.get("value")

        if cut_type is None or cut_value is None:
            print(f"  ⚠️  Skipping {branch_name}: missing cut_type or value")
            continue

        # Find variable name from nd_config
        var_name = None
        description = f"Manual cut: {branch_name}"
        for nd_var, nd_spec in nd_config.items():
            if nd_var != "notes" and nd_spec.get("branch_name") == branch_name:
                var_name = nd_var
                description = nd_spec.get("description", description)
                break

        if var_name is None:
            var_name = branch_name.lower()

        print(f"  {branch_name:20s} {cut_type:>7s} {cut_value:8.3f}")

        # Apply same cuts to all states
        for state in states:
            all_results.append(
                {
                    "state": state,
                    "variable": var_name,
                    "branch_name": branch_name,
                    "optimal_cut": cut_value,
                    "cut_type": cut_type,
                    "max_fom": 0.0,
                    "n_sig_at_optimal": 0.0,
                    "n_bkg_at_optimal": 0.0,
                    "description": description,
                }
            )

    if not all_results:
        raise AnalysisError(
            "No valid manual cuts found in config!\n"
            "Check [manual_cuts] section format in config/selection.toml"
        )

    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# Manual cuts path
# ---------------------------------------------------------------------------
if use_manual_cuts:
    manual_cuts_config = config.selection.get("manual_cuts", {})
    has_manual_cuts = any(k for k in manual_cuts_config.keys() if k != "notes")

    if not has_manual_cuts:
        raise AnalysisError(
            "--use-manual-cuts flag set but no manual cuts defined in config/selection.toml!\n"
            "Please add cuts to [manual_cuts] section or remove the flag."
        )

    print("✓ Using MANUAL CUTS from config (skipping optimization)")
    print("=" * 80)
    cuts_df = create_manual_cuts_dataframe(manual_cuts_config)

    # Save to output file
    tables_dir.mkdir(exist_ok=True, parents=True)
    cuts_df.to_csv(cuts_output_file, index=False)
    print(f"\n✓ Manual cuts saved to {cuts_output_file}")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Grid scan optimization path
# ---------------------------------------------------------------------------

# Load Step 2 cached data
# We need: data_dict, mc_dict, phsp_dict, mc_generated_counts
# Use the same dependency computation as Step 2 to find the right cache entries
years = snakemake.config.get("years", ["2016", "2017", "2018"])  # noqa: F821
track_types = snakemake.config.get("track_types", ["LL", "DD"])  # noqa: F821

step2_deps = compute_step_dependencies(
    step="2", extra_params={"years": years, "track_types": track_types}
)

data_dict = cache.load("step2_data_after_lambda", dependencies=step2_deps)
mc_dict = cache.load("step2_mc_after_lambda", dependencies=step2_deps)
phase_space_dict = cache.load("step2_phase_space_after_lambda", dependencies=step2_deps)
mc_generated_counts = cache.load("step2_mc_generated_counts", dependencies=step2_deps)

if data_dict is None or mc_dict is None:
    raise AnalysisError(
        "Step 2 cached data not found! Run Step 2 (load_data) first.\n"
        "This should not happen if the Snakemake DAG is correct."
    )

# Check cache for Step 3 results
step3_deps = compute_step_dependencies(step="3", extra_params={"states": list(mc_dict.keys())})

use_cached = not no_cache
if use_cached:
    cached = cache.load("step3_optimized_cuts", dependencies=step3_deps)
    if cached is not None:
        print("✓ Loaded cached optimized cuts")
        cached.to_csv(cuts_output_file, index=False)
        print(f"\n✓ Step 3 complete: Optimized cuts saved to {cuts_output_file}")
        sys.exit(0)

print("\n  Running full N-D optimization (this may take 30-60 minutes)")
print("    We can skip this and use default cuts if needed\n")

# Combine track types (LL + DD) for optimizer
print("Combining LL and DD track types for optimization...")

mc_combined = {}
for state in mc_dict:
    mc_combined[state] = {}
    for year in mc_dict[state]:
        arrays_to_combine = []
        for track_type in mc_dict[state][year]:
            arr = mc_dict[state][year][track_type]
            if hasattr(arr, "layout"):
                arrays_to_combine.append(arr)
        if arrays_to_combine:
            mc_combined[state][year] = ak.concatenate(arrays_to_combine, axis=0)
            print(f"  {state}/{year}: {len(mc_combined[state][year])} events")

# Combine phase-space MC track types
phase_space_combined = {}
if phase_space_dict is not None:
    for year in phase_space_dict:
        arrays_to_combine = []
        for track_type in phase_space_dict[year]:
            arr = phase_space_dict[year][track_type]
            if hasattr(arr, "layout"):
                arrays_to_combine.append(arr)
        if arrays_to_combine:
            phase_space_combined[year] = ak.concatenate(arrays_to_combine, axis=0)
            print(f"  phase_space/{year}: {len(phase_space_combined[year])} events")

# Combine real data track types for background estimation
data_combined = {}
for year in data_dict:
    arrays_to_combine = []
    for track_type in data_dict[year]:
        arr = data_dict[year][track_type]
        if hasattr(arr, "layout"):
            arrays_to_combine.append(arr)
    if arrays_to_combine:
        data_combined[year] = ak.concatenate(arrays_to_combine, axis=0)
        print(f"  data/{year}: {len(data_combined[year])} events")

# Sum MC generated counts across track types (LL + DD)
if mc_generated_counts is not None:
    mc_generated_combined = {}
    for state in mc_generated_counts:
        mc_generated_combined[state] = {}
        for year in mc_generated_counts[state]:
            total_generated = sum(mc_generated_counts[state][year].values())
            mc_generated_combined[state][year] = total_generated
            print(f"  {state}/{year} generated: {total_generated:,} events")

# Initialize optimizer
print("\n✓ IMPORTANT: Correct optimization approach:")
print("    Signal: signal MC (J/ψ, ηc, χc) - scaled to expected events")
print("    Background: real data sidebands - interpolated to signal region")
print("    Formula: n_sig = ε * L * σ_eff * 10³")
print("    Where ε = n_mc_after_cuts / n_mc_generated (full selection efficiency)")
print("    This properly estimates signal efficiency and background level!")

# NEW: Unbiased optimization using data proxies only
optimizer = SelectionOptimizer(
    data=data_combined,
    config=config,
)

# Validate data regions before optimization
optimizer.validate_data_regions()

# Run N-D grid scan optimization with UNBIASED method
print("\n  Running UNBIASED N-D GRID SCAN")
print("    Signal proxy: no-charmonium data (M(Λ̄pK⁻) > 4 GeV)")
print("    Background proxy: B⁺ mass sidebands")
print("    Lambda cuts are FIXED (already applied in Step 2)")
optimized_cuts_df = optimizer.optimize_nd_grid_scan()

# Cache and save
cache.save(
    "step3_optimized_cuts",
    optimized_cuts_df,
    dependencies=step3_deps,
    description="Optimized selection cuts (unbiased method)",
)
optimized_cuts_df.to_csv(cuts_output_file, index=False)

print(f"\n✓ Step 3 complete: Optimized cuts saved to {cuts_output_file}")
