"""
Step 4 Snakemake wrapper: Apply Optimized Cuts

Reproduces the logic from:
  - PipelineManager.phase4_apply_optimized_cuts() in run_pipeline.py

Key physics logic:
  - MC: Apply state-specific cuts (for efficiency calculation)
  - Data: By default, do NOT apply cuts (for simultaneous mass fitting)
  - Controlled via config/selection.toml [cut_application] section

IMPORTANT: Unlike the original pipeline (which passes data in-memory),
the Snakemake version must cache data_final and mc_final via CacheManager
so that Step 5 (mass fitting) and Step 6 (efficiency) can load them.

Snakemake injects the `snakemake` object with:
  snakemake.input.step2     — sentinel for Step 2 completion
  snakemake.input.cuts      — path to optimized_cuts.csv
  snakemake.output[0]       — path to step4_summary.json
  snakemake.params.no_cache — bool: force reprocessing
  snakemake.params.config_dir
  snakemake.params.cache_dir
  snakemake.params.output_dir
"""

import json
import sys
from pathlib import Path
from typing import Any

# Ensure the project root (analysis_make/) is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import awkward as ak
import pandas as pd
from tqdm import tqdm

from modules.cache_manager import CacheManager
from modules.data_handler import TOMLConfig
from modules.exceptions import AnalysisError

# ---------------------------------------------------------------------------
# Read Snakemake params
# ---------------------------------------------------------------------------
no_cache = snakemake.params.no_cache  # noqa: F821
config_dir = snakemake.params.config_dir  # noqa: F821
cache_dir = snakemake.params.cache_dir  # noqa: F821
output_dir = snakemake.params.output_dir  # noqa: F821
cuts_input_file = snakemake.input.cuts  # noqa: F821
summary_output_file = snakemake.output[0]  # noqa: F821

# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------
config = TOMLConfig(config_dir)
cache = CacheManager(cache_dir)

print("\n" + "=" * 80)
print("STEP 4: APPLYING OPTIMIZED CUTS")
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
    elif step == "3":
        code_files = [
            project_root / "modules" / "selection_optimizer.py",
        ]

    return cache.compute_dependencies(
        config_files=config_files, code_files=code_files, extra_params=extra_params
    )


# ---------------------------------------------------------------------------
# Load Step 2 cached data
# ---------------------------------------------------------------------------
years = snakemake.config.get("years", ["2016", "2017", "2018"])  # noqa: F821
track_types = snakemake.config.get("track_types", ["LL", "DD"])  # noqa: F821

step2_deps = compute_step_dependencies(
    step="2", extra_params={"years": years, "track_types": track_types}
)

data_dict = cache.load("step2_data_after_lambda", dependencies=step2_deps)
mc_dict = cache.load("step2_mc_after_lambda", dependencies=step2_deps)

if data_dict is None or mc_dict is None:
    raise AnalysisError(
        "Step 2 cached data not found! Run Step 2 (load_data) first.\n"
        "This should not happen if the Snakemake DAG is correct."
    )

# ---------------------------------------------------------------------------
# Load optimized cuts from Step 3
# ---------------------------------------------------------------------------
optimized_cuts_df = pd.read_csv(cuts_input_file)

# ---------------------------------------------------------------------------
# Read cut_application config
# ---------------------------------------------------------------------------
cut_config = config.selection.get("cut_application", {})
apply_cuts_to_data = cut_config.get("apply_cuts_to_data", False)
data_cut_state = cut_config.get("data_cut_state", "jpsi")

# Show configuration
print("\nConfiguration:")
print(f"  apply_cuts_to_data = {apply_cuts_to_data}")
if apply_cuts_to_data:
    print(f"  data_cut_state = {data_cut_state}")
    print("    WARNING: Cuts will be applied to data!")
else:
    print("  → Data will remain unchanged (correct for mass fitting)")

if optimized_cuts_df is None or len(optimized_cuts_df) == 0:
    print("\n No optimized cuts provided - using Lambda cuts only")
    # Save summary and exit — data/MC unchanged
    summary = {
        "step": "4_apply_cuts",
        "apply_cuts_to_data": False,
        "data_cut_state": None,
        "data_cuts": "Lambda pre-selection only (no optimized cuts)",
        "mc_cuts": "Lambda pre-selection only (no optimized cuts)",
        "n_cuts_per_state": {},
    }
    tables_dir = Path(config.paths["output"]["tables_dir"])
    tables_dir.mkdir(exist_ok=True, parents=True)
    with open(summary_output_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Cache unchanged data/MC for later steps
    cache.save("step4_data_final", data_dict, description="Data after Step 4")
    cache.save("step4_mc_final", mc_dict, description="MC after Step 4")
    sys.exit(0)

print(f"\nApplying {len(optimized_cuts_df)} optimized cut values")

# ---------------------------------------------------------------------------
# Apply cuts to MC (state-specific cuts for each state)
# ---------------------------------------------------------------------------
mc_final = {}
states = ["jpsi", "etac", "chic0", "chic1"]

print("\nApplying cuts to MC...")
total_mc_cuts = sum(
    len(mc_dict[state][year]) for state in states if state in mc_dict for year in mc_dict[state]
)

with tqdm(total=total_mc_cuts, desc="Applying MC cuts", unit="dataset") as pbar:
    for state in states:
        mc_final[state] = {}

        if state not in mc_dict:
            continue

        # Get cuts for this state
        state_cuts = optimized_cuts_df[optimized_cuts_df["state"] == state]

        if len(state_cuts) == 0:
            mc_final[state] = mc_dict[state]
            pbar.update(sum(len(mc_dict[state][year]) for year in mc_dict[state]))
            continue

        for year in mc_dict[state]:
            mc_final[state][year] = {}

            for track_type in mc_dict[state][year]:
                pbar.set_postfix_str(f"{state} {year} {track_type}")
                events = mc_dict[state][year][track_type]

                # Start with all events passing
                mask = ak.ones_like(events["Bu_PT"], dtype=bool)

                # Apply each cut
                for _, cut_row in state_cuts.iterrows():
                    branch = cut_row["branch_name"]
                    cut_val = cut_row["optimal_cut"]
                    cut_type = cut_row["cut_type"]

                    if branch not in events.fields:
                        print(f"    Branch {branch} not found, skipping")
                        continue

                    branch_data = events[branch]

                    # Flatten jagged arrays if needed
                    if "var" in str(ak.type(branch_data)):
                        branch_data = ak.firsts(branch_data)

                    if cut_type == "greater":
                        mask = mask & (branch_data > cut_val)
                    elif cut_type == "less":
                        mask = mask & (branch_data < cut_val)

                events_after = events[mask]

                mc_final[state][year][track_type] = events_after
                pbar.update(1)

# ---------------------------------------------------------------------------
# Apply cuts to data (only if requested)
# ---------------------------------------------------------------------------
if apply_cuts_to_data:
    print(f"\nApplying cuts to data (using {data_cut_state} cuts)...")

    data_final = {}
    data_cuts = optimized_cuts_df[optimized_cuts_df["state"] == data_cut_state]

    if len(data_cuts) == 0:
        data_final = data_dict
    else:
        total_data = sum(len(data_dict[year]) for year in data_dict)
        with tqdm(total=total_data, desc="Applying data cuts", unit="dataset") as pbar:
            for year in data_dict:
                data_final[year] = {}
                for track_type in data_dict[year]:
                    pbar.set_postfix_str(f"{year} {track_type}")
                    events = data_dict[year][track_type]

                    # Apply cuts
                    mask = ak.ones_like(events["Bu_PT"], dtype=bool)
                    for _, cut_row in data_cuts.iterrows():
                        branch = cut_row["branch_name"]
                        cut_val = cut_row["optimal_cut"]
                        cut_type = cut_row["cut_type"]

                        if branch not in events.fields:
                            continue

                        branch_data = events[branch]
                        if "var" in str(ak.type(branch_data)):
                            branch_data = ak.firsts(branch_data)

                        if cut_type == "greater":
                            mask = mask & (branch_data > cut_val)
                        elif cut_type == "less":
                            mask = mask & (branch_data < cut_val)

                    events_after = events[mask]

                    data_final[year][track_type] = events_after
                    pbar.update(1)
else:
    # Default: Do NOT apply cuts to data (for fitting)
    data_final = data_dict

# ---------------------------------------------------------------------------
# Cache data_final and mc_final for Step 5 and Step 6
# (Not done in original pipeline since it passes in-memory; required here
#  because Snakemake runs each step as a separate process)
# ---------------------------------------------------------------------------
cache.save(
    "step4_data_final",
    data_final,
    description="Data after Step 4 cuts (or unchanged if apply_cuts_to_data=False)",
)
cache.save(
    "step4_mc_final",
    mc_final,
    description="MC after Step 4 state-specific cuts",
)

# ---------------------------------------------------------------------------
# Save summary JSON
# ---------------------------------------------------------------------------
summary = {
    "step": "4_apply_cuts",
    "apply_cuts_to_data": apply_cuts_to_data,
    "data_cut_state": data_cut_state if apply_cuts_to_data else None,
    "data_cuts": (
        f"Cuts from {data_cut_state}" if apply_cuts_to_data else "Lambda pre-selection only"
    ),
    "mc_cuts": "State-specific optimized cuts from Step 3",
    "n_cuts_per_state": {
        state: len(optimized_cuts_df[optimized_cuts_df["state"] == state]) for state in states
    },
}

tables_dir = Path(config.paths["output"]["tables_dir"])
tables_dir.mkdir(exist_ok=True, parents=True)
with open(summary_output_file, "w") as f:
    json.dump(summary, f, indent=2)

print("\n✓ Step 4 complete:")
if apply_cuts_to_data:
    print(f"  → Data: CUTS APPLIED (using {data_cut_state} cuts)")
    print("       Use this only for control plots/validation, NOT for fitting!")
else:
    print("  → Data: UNCHANGED (Lambda pre-selection only)")
    print("      All charmonium states will be fit simultaneously to same data")
print("  → MC: State-specific optimized cuts applied")
print("      Used to calculate selection efficiencies per state")
print(f"  → Summary saved to {summary_output_file}")
