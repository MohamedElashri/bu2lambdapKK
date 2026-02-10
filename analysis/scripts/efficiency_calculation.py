"""
Step 6 Snakemake wrapper: Efficiency Calculation

Reproduces the logic from:
  - PipelineManager.phase6_efficiency_calculation() in run_pipeline.py

Key physics:
  - ε_sel = N_pass_all_cuts / N_generated (total selection efficiency)
  - Efficiencies calculated per (state, year) from MC
  - ε_reco, ε_strip, ε_trig CANCEL in ratios (identical final state)
  - ηc(2S) uses χc1 efficiency as proxy (no MC available)
  - Efficiency ratios ε_J/ψ / ε_state enter BR formula

Snakemake injects the `snakemake` object with:
  snakemake.input.summary       — path to step4_summary.json (dependency)
  snakemake.input.cuts          — path to optimized_cuts.csv
  snakemake.output.efficiencies — path to efficiencies.csv
  snakemake.output.ratios       — path to efficiency_ratios.csv
  snakemake.params.no_cache     — bool: force reprocessing
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

import awkward as ak
import pandas as pd

from modules.cache_manager import CacheManager
from modules.data_handler import TOMLConfig
from modules.efficiency_calculator import EfficiencyCalculator
from modules.exceptions import AnalysisError

# ---------------------------------------------------------------------------
# Read Snakemake params
# ---------------------------------------------------------------------------
no_cache = snakemake.params.no_cache  # noqa: F821
config_dir = snakemake.params.config_dir  # noqa: F821
cache_dir = snakemake.params.cache_dir  # noqa: F821
output_dir = snakemake.params.output_dir  # noqa: F821
cuts_input_file = snakemake.input.cuts  # noqa: F821

# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------
config = TOMLConfig(config_dir)
cache = CacheManager(cache_dir)

print("\n" + "=" * 80)
print("STEP 6: EFFICIENCY CALCULATION")
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
    if step == "2":
        code_files = [
            project_root / "modules" / "data_handler.py",
            project_root / "modules" / "lambda_selector.py",
        ]
    elif step == "6":
        code_files = [
            project_root / "modules" / "efficiency_calculator.py",
        ]

    return cache.compute_dependencies(
        config_files=config_files, code_files=code_files, extra_params=extra_params
    )


# ---------------------------------------------------------------------------
# Load Step 4 cached MC (mc_final) and Step 2 generated counts
# ---------------------------------------------------------------------------
mc_final = cache.load("step4_mc_final")

if mc_final is None:
    raise AnalysisError(
        "Step 4 cached mc_final not found! Run Step 4 (apply_cuts) first.\n"
        "This should not happen if the Snakemake DAG is correct."
    )

# Load mc_generated_counts from Step 2 cache
years = snakemake.config.get("years", ["2016", "2017", "2018"])  # noqa: F821
track_types = snakemake.config.get("track_types", ["LL", "DD"])  # noqa: F821

step2_deps = compute_step_dependencies(
    step="2", extra_params={"years": years, "track_types": track_types}
)
mc_generated_counts = cache.load("step2_mc_generated_counts", dependencies=step2_deps)

if mc_generated_counts is None:
    raise AnalysisError(
        "Step 2 cached mc_generated_counts not found! Run Step 2 (load_data) first.\n"
        "This should not happen if the Snakemake DAG is correct."
    )

# ---------------------------------------------------------------------------
# Load optimized cuts from Step 3
# ---------------------------------------------------------------------------
optimized_cuts_df = pd.read_csv(cuts_input_file)

# ---------------------------------------------------------------------------
# Check cache for Step 6 results
# ---------------------------------------------------------------------------
use_cached = not no_cache
states = ["jpsi", "etac", "chic0", "chic1"]
years_list = list(mc_final["jpsi"].keys()) if "jpsi" in mc_final and mc_final["jpsi"] else []

if use_cached:
    dependencies = compute_step_dependencies(
        step="6", extra_params={"states": list(mc_final.keys()), "years": years_list}
    )
    cached = cache.load("step6_efficiencies", dependencies=dependencies)
    if cached is not None:
        print("✓ Loaded cached efficiencies")

        # Still need to regenerate output files from cached data
        eff_calculator = EfficiencyCalculator(config, optimized_cuts_df, {})
        eff_calculator.generate_efficiency_table(cached)
        eff_calculator.calculate_efficiency_ratios(cached)

        print("\n✓ Step 6 complete: Efficiencies saved to tables/")
        sys.exit(0)

# ---------------------------------------------------------------------------
# Combine generated counts across track types (LL + DD)
# ---------------------------------------------------------------------------
mc_gen_combined = {}
for state in mc_generated_counts:
    mc_gen_combined[state] = {}
    for year in mc_generated_counts[state]:
        # Sum LL and DD counts
        total = sum(mc_generated_counts[state][year].values())
        mc_gen_combined[state][year] = total

# ---------------------------------------------------------------------------
# Initialize efficiency calculator with generated counts
# ---------------------------------------------------------------------------
eff_calculator = EfficiencyCalculator(config, optimized_cuts_df, mc_gen_combined)

# ---------------------------------------------------------------------------
# Calculate efficiencies
# ---------------------------------------------------------------------------
print("\nCalculating selection efficiencies from MC...")
efficiencies = {}

for state in states:
    print(f"\n  {state}:")
    efficiencies[state] = {}

    for year in years_list:
        # Combine LL and DD if both exist
        if "LL" in mc_final[state][year] and "DD" in mc_final[state][year]:
            mc_combined = ak.concatenate([mc_final[state][year]["LL"], mc_final[state][year]["DD"]])
        else:
            # Use whichever is available
            track_type = list(mc_final[state][year].keys())[0]
            mc_combined = mc_final[state][year][track_type]

        # Calculate efficiency
        eff_result = eff_calculator.calculate_selection_efficiency(mc_combined, state, year)

        efficiencies[state][year] = eff_result

        print(
            f"    {year}: ε = {eff_result['eff']:.4f} ± {eff_result['err']:.4f} ({100*eff_result['eff']:.2f}%)"
        )

# ---------------------------------------------------------------------------
# Add etac_2s efficiency by copying chi_c1 (no MC available)
# ---------------------------------------------------------------------------
if "chic1" in efficiencies:
    print("\n  etac_2s:")
    print("    Note: No MC available for eta_c(2S)")
    print("    Using chi_c1 efficiency as proxy (similar mass, width, cuts)")
    efficiencies["etac_2s"] = {}
    for year in efficiencies["chic1"].keys():
        efficiencies["etac_2s"][year] = efficiencies["chic1"][year].copy()
        eff = efficiencies["etac_2s"][year]["eff"]
        err = efficiencies["etac_2s"][year]["err"]
        print(f"    {year}: ε = {eff:.4f} ± {err:.4f} ({100*eff:.2f}%) [from chi_c1]")

# ---------------------------------------------------------------------------
# Calculate efficiency ratios (saves efficiency_ratios.csv internally)
# ---------------------------------------------------------------------------
eff_calculator.calculate_efficiency_ratios(efficiencies)

# ---------------------------------------------------------------------------
# Cache results
# ---------------------------------------------------------------------------
dependencies = compute_step_dependencies(
    step="6", extra_params={"states": list(mc_final.keys()), "years": years_list}
)
cache.save(
    "step6_efficiencies",
    efficiencies,
    dependencies=dependencies,
    description="Efficiency calculations",
)

# ---------------------------------------------------------------------------
# Save efficiency table (saves efficiencies.csv internally)
# ---------------------------------------------------------------------------
eff_calculator.generate_efficiency_table(efficiencies)

print("\n✓ Step 6 complete: Efficiencies saved to tables/")
