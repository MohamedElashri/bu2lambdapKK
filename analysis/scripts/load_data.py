"""
Step 2 Snakemake wrapper: Data/MC Loading + Lambda Pre-Selection

Reproduces the logic from:
  - PipelineManager.phase2_load_data_and_lambda_cuts() in run_pipeline.py

This is the most I/O-intensive step. It:
  1. Loads real data for all (year, track_type) from both magnets
  2. Loads phase-space MC (KpKm) with same flow
  3. Loads signal MC for all 4 states with same flow
  4. Applies Lambda pre-selection cuts to all datasets
  5. Tracks mc_generated_counts (events before Lambda cuts)
  6. Caches all 4 outputs via CacheManager

Snakemake injects the `snakemake` object with:
  snakemake.params.years        — list of year strings
  snakemake.params.track_types  — list of track types
  snakemake.params.magnets      — list of magnet polarities
  snakemake.params.states       — list of signal MC states
  snakemake.params.no_cache     — bool: force reprocessing
  snakemake.params.config_dir   — path to TOML config directory
  snakemake.params.cache_dir    — path to cache directory
"""

import sys
from pathlib import Path
from typing import Any

# Ensure the project root (analysis_make/) is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tqdm import tqdm

from modules.cache_manager import CacheManager
from modules.data_handler import DataManager, TOMLConfig
from modules.lambda_selector import LambdaSelector
from utils.logging_config import suppress_warnings

# Suppress warnings by default (matches original pipeline behavior)
suppress_warnings()

# ---------------------------------------------------------------------------
# Read Snakemake params
# ---------------------------------------------------------------------------
years = snakemake.params.years  # noqa: F821
track_types = snakemake.params.track_types  # noqa: F821
states = snakemake.params.states  # noqa: F821
no_cache = snakemake.params.no_cache  # noqa: F821
config_dir = snakemake.params.config_dir  # noqa: F821
cache_dir = snakemake.params.cache_dir  # noqa: F821

# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------
config = TOMLConfig(config_dir)
cache = CacheManager(cache_dir)

print("\n" + "=" * 80)
print("STEP 2: DATA/MC LOADING + LAMBDA PRE-SELECTION")
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

    return cache.compute_dependencies(
        config_files=config_files, code_files=code_files, extra_params=extra_params
    )


# ---------------------------------------------------------------------------
# Compute dependencies for cache validation
# ---------------------------------------------------------------------------
dependencies = compute_step_dependencies(
    step="2", extra_params={"years": years, "track_types": track_types}
)

# ---------------------------------------------------------------------------
# Check cache (unless no_cache is set)
# ---------------------------------------------------------------------------
use_cached = not no_cache
if use_cached:
    data_dict = cache.load("step2_data_after_lambda", dependencies=dependencies)
    mc_dict = cache.load("step2_mc_after_lambda", dependencies=dependencies)
    phase_space_dict = cache.load("step2_phase_space_after_lambda", dependencies=dependencies)
    mc_generated_counts = cache.load("step2_mc_generated_counts", dependencies=dependencies)

    if (
        data_dict is not None
        and mc_dict is not None
        and phase_space_dict is not None
        and mc_generated_counts is not None
    ):
        print("✓ Loaded cached data, signal MC, and phase-space MC (after Lambda cuts)")
        # Display cache stats
        stats = cache.get_cache_stats()
        print(f"✓ Cache: {stats['num_entries']} entries, {stats['total_size_mb']:.1f} MB")
        sys.exit(0)

    print("  Cache miss or invalidated - will recompute")

# ---------------------------------------------------------------------------
# Initialize data manager and Lambda selector
# ---------------------------------------------------------------------------
data_manager = DataManager(config)
lambda_selector = LambdaSelector(config)

# ---------------------------------------------------------------------------
# Load and process REAL DATA
# ---------------------------------------------------------------------------
print("[Loading Real Data]")
data_dict = {}
with tqdm(total=len(years) * len(track_types), desc="Loading data", unit="dataset") as pbar:
    for year in years:
        data_dict[year] = {}
        for track_type in track_types:
            pbar.set_postfix_str(f"{year} {track_type}")

            # Load data from both magnets using unified method
            events = data_manager.load_and_process(
                "data", year, track_type, apply_derived_branches=True, apply_trigger=False
            )

            if events is None:
                pbar.set_postfix_str(f"❌ {year} {track_type} missing")
                pbar.update(1)
                continue

            # Apply Lambda cuts
            n_before = len(events)
            events_after = lambda_selector.apply_lambda_cuts(events)
            n_after = len(events_after)
            eff = 100 * n_after / n_before if n_before > 0 else 0

            data_dict[year][track_type] = events_after
            pbar.set_postfix_str(f"{year} {track_type}: {n_before:,}→{n_after:,} ({eff:.1f}%)")
            pbar.update(1)

# ---------------------------------------------------------------------------
# Load and process PHASE-SPACE MC (KpKm - for background estimation)
# ---------------------------------------------------------------------------
print("\n[Loading Phase-Space MC - KpKm for Background]")
phase_space_dict = {}
with tqdm(total=len(years) * len(track_types), desc="Loading KpKm MC", unit="dataset") as pbar:
    for year in years:
        phase_space_dict[year] = {}
        for track_type in track_types:
            pbar.set_postfix_str(f"{year} {track_type}")

            # Load KpKm MC from both magnets using unified method
            events = data_manager.load_and_process(
                "KpKm", year, track_type, apply_derived_branches=True, apply_trigger=False
            )

            if events is None:
                pbar.set_postfix_str(f"❌ KpKm {year} {track_type} missing")
                pbar.update(1)
                continue

            # Apply Lambda cuts
            n_before = len(events)
            events_after = lambda_selector.apply_lambda_cuts(events)
            n_after = len(events_after)
            eff = 100 * n_after / n_before if n_before > 0 else 0

            phase_space_dict[year][track_type] = events_after
            pbar.set_postfix_str(f"{year} {track_type}: {n_before:,}→{n_after:,} ({eff:.1f}%)")
            pbar.update(1)

# ---------------------------------------------------------------------------
# Load and process MC (all 4 signal states)
# ---------------------------------------------------------------------------
print("\n[Loading MC - Signal States]")
mc_dict = {}
mc_generated_counts = {}  # Track generator-level counts for signal scaling

total_mc_datasets = len(states) * len(years) * len(track_types)
with tqdm(total=total_mc_datasets, desc="Loading signal MC", unit="dataset") as pbar:
    for state in states:
        mc_dict[state] = {}
        mc_generated_counts[state] = {}

        for year in years:
            mc_dict[state][year] = {}
            for track_type in track_types:
                pbar.set_postfix_str(f"{state} {year} {track_type}")

                # Map state names: jpsi -> Jpsi, others stay lowercase
                state_name = "Jpsi" if state == "jpsi" else state

                # Load MC from both magnets using unified method
                events = data_manager.load_and_process(
                    state_name,
                    year,
                    track_type,
                    apply_derived_branches=True,
                    apply_trigger=False,
                )

                if events is None:
                    pbar.set_postfix_str(f"❌ {state} {year} {track_type} missing")
                    pbar.update(1)
                    continue

                # Apply Lambda cuts
                n_before = len(events)
                events_after = lambda_selector.apply_lambda_cuts(events)
                n_after = len(events_after)
                eff = 100 * n_after / n_before if n_before > 0 else 0

                mc_dict[state][year][track_type] = events_after

                # Store generator-level count for signal scaling
                if year not in mc_generated_counts[state]:
                    mc_generated_counts[state][year] = {}
                mc_generated_counts[state][year][track_type] = n_before

                pbar.set_postfix_str(
                    f"{state} {year} {track_type}: {n_before:,}→{n_after:,} ({eff:.1f}%)"
                )
                pbar.update(1)

# ---------------------------------------------------------------------------
# Cache results with dependencies
# ---------------------------------------------------------------------------
cache.save(
    "step2_data_after_lambda",
    data_dict,
    dependencies=dependencies,
    description="Data after Lambda cuts",
)
cache.save(
    "step2_mc_after_lambda",
    mc_dict,
    dependencies=dependencies,
    description="Signal MC after Lambda cuts",
)
cache.save(
    "step2_phase_space_after_lambda",
    phase_space_dict,
    dependencies=dependencies,
    description="Phase-space MC after Lambda cuts",
)
cache.save(
    "step2_mc_generated_counts",
    mc_generated_counts,
    dependencies=dependencies,
    description="Generator-level MC counts",
)

print("\n✓ Step 2 complete: Data, signal MC, and phase-space MC loaded")
print("  → Data: used for background estimation in optimization AND final fitting")
print("  → Signal MC: used for signal efficiency in optimization (scaled to expected events)")
print("  → Phase-space MC (KpKm): kept for reference (not used for optimization)")
print("  → MC generated counts: tracked for proper signal scaling")

# Display cache stats
stats = cache.get_cache_stats()
print(f"✓ Cache: {stats['num_entries']} entries, {stats['total_size_mb']:.1f} MB")
