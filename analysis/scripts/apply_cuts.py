import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import awkward as ak

from modules.cache_manager import CacheManager
from modules.config_loader import StudyConfig
from modules.generated_paths import pipeline_cache_dir, pipeline_output_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

if "snakemake" in globals():
    no_cache = snakemake.params.no_cache
    config_dir = snakemake.params.config_dir
    cache_dir = snakemake.params.cache_dir
    output_dir = snakemake.params.output_dir
    cuts_file = snakemake.input.cuts
    summary_file = snakemake.output.summary
    branch = snakemake.params.branch
    category = snakemake.params.category
    years = snakemake.params.get("years", ["2016", "2017", "2018"])
else:
    no_cache = False
    config_dir = "config"
    cache_dir = str(pipeline_cache_dir("box", project_root / "generated" / "cache"))
    output_dir = str(pipeline_output_dir("box", project_root / "generated" / "output"))
    branch = "high_yield"
    category = "LL"
    cuts_file = Path(output_dir) / branch / category / "models" / "optimized_cuts.json"
    summary_file = Path(output_dir) / branch / category / "tables" / "cut_summary.json"
    years = ["2016", "2017", "2018"]

config = StudyConfig.from_dir(config_dir, output_dir=output_dir)

cache = CacheManager(cache_dir=cache_dir)
preprocessed_deps = cache.compute_dependencies(
    config_files=config.config_paths(),
    code_files=[
        project_root / "modules" / "clean_data_loader.py",
        project_root / "scripts" / "load_data.py",
    ],
    extra_params={"years": years, "track_types": ["LL", "DD"]},
)
cut_deps = cache.compute_dependencies(
    config_files=config.config_paths(),
    code_files=[project_root / "scripts" / "apply_cuts.py"],
)

# Load nested dicts from cache; slice to this category
data_dict_full = cache.load("preprocessed_data", dependencies=preprocessed_deps)
mc_dict_full = cache.load("preprocessed_mc", dependencies=preprocessed_deps)

if data_dict_full is None or mc_dict_full is None:
    logger.error("Step 2 data not found in cache. Run 'snakemake load_data' first.")
    sys.exit(1)

# Extract flat {year: array} and {state: array} for this category
data_dict = {
    yr: data_dict_full[yr][category] for yr in data_dict_full if category in data_dict_full[yr]
}
mc_dict = {st: mc_dict_full[st][category] for st in mc_dict_full if category in mc_dict_full[st]}

logger.info(f"Applying cuts for branch={branch}, category={category}")

with open(cuts_file, "r") as f:
    optimized_cuts = json.load(f)

opt_type = config.data.get("cut_application", {}).get("optimization_type", "box")

# Passthrough override: if the cuts file was written by passthrough_selection.py
# the opt_type stored inside it takes priority over selection.toml.
if isinstance(optimized_cuts, dict) and optimized_cuts.get("opt_type") == "passthrough":
    opt_type = "passthrough"
elif isinstance(optimized_cuts, list):
    # Box optimisation outputs are stored as a list of cut rows, whereas the
    # MVA and passthrough modes write dictionaries.  This allows the box
    # reference workflow to coexist with a selection.toml whose nominal method
    # is currently set to "mva".
    opt_type = "box"

data_final = {}
mc_final = {}

if opt_type == "passthrough":
    logger.info(
        f"Passthrough mode [{category}]: no additional cuts applied "
        "beyond trigger + stripping + Lambda pre-selection."
    )
    data_final = dict(data_dict)
    mc_final = dict(mc_dict)

elif opt_type == "box":
    logger.info(f"Applying Box Cuts for branch={branch}, category={category}")
    target_fom = "S/sqrt(S+B)"
    target_state = "jpsi" if branch == "high_yield" else "chic1"

    branch_cuts = [
        entry
        for entry in optimized_cuts
        if entry["state"] == target_state and entry["FoM_type"] == target_fom
    ]

    if not branch_cuts:
        logger.error(f"Could not find {target_fom} cuts for state {target_state} in {cuts_file}")
        sys.exit(1)

    logger.info(f"Using cuts from state {target_state} with FoM {target_fom}")

    for state, state_data in mc_dict.items():
        if len(state_data) == 0:
            mc_final[state] = state_data
            continue
        mask = ak.ones_like(state_data[state_data.fields[0]], dtype=bool)
        for cut_entry in branch_cuts:
            var_branch = cut_entry["branch_name"]
            cut_val = cut_entry["optimal_cut"]
            cut_type = cut_entry["cut_type"]
            if var_branch not in state_data.fields:
                continue
            if cut_type == "less":
                mask = mask & (state_data[var_branch] < cut_val)
            else:
                mask = mask & (state_data[var_branch] > cut_val)
        mc_final[state] = state_data[mask]
        logger.info(
            f"MC {state} [{category}] passed cuts: {len(mc_final[state])} / {len(state_data)}"
        )

    for year, y_data in data_dict.items():
        if len(y_data) == 0:
            data_final[year] = y_data
            continue
        mask = ak.ones_like(y_data[y_data.fields[0]], dtype=bool)
        for cut_entry in branch_cuts:
            var_branch = cut_entry["branch_name"]
            cut_val = cut_entry["optimal_cut"]
            cut_type = cut_entry["cut_type"]
            if var_branch not in y_data.fields:
                continue
            if cut_type == "less":
                mask = mask & (y_data[var_branch] < cut_val)
            else:
                mask = mask & (y_data[var_branch] > cut_val)
        data_final[year] = y_data[mask]
        logger.info(
            f"Data {year} [{category}] passed cuts: {len(data_final[year])} / {len(y_data)}"
        )

elif opt_type == "mva":
    logger.info(f"Applying MVA Cuts (branch={branch}, category={category})")
    if branch == "high_yield":
        threshold = optimized_cuts.get(
            "mva_threshold_high", optimized_cuts.get("mva_threshold", 0.5)
        )
    else:
        threshold = optimized_cuts.get(
            "mva_threshold_low", optimized_cuts.get("mva_threshold", 0.5)
        )

    features = optimized_cuts.get("features", [])

    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model_path = Path(output_dir) / branch / category / "models" / "mva_model.cbm"
    model.load_model(str(model_path))

    for state, state_data in mc_dict.items():
        if len(state_data) == 0:
            mc_final[state] = state_data
            continue

        df = ak.to_dataframe(state_data)[features]
        preds = model.predict_proba(df)[:, 1]
        mask = preds > threshold
        mc_final[state] = state_data[mask]
        logger.info(
            f"MC {state} [{category}] passed MVA > {threshold}: "
            f"{len(mc_final[state])} / {len(state_data)}"
        )

    for year, y_data in data_dict.items():
        if len(y_data) == 0:
            data_final[year] = y_data
            continue

        df = ak.to_dataframe(y_data)[features]
        preds = model.predict_proba(df)[:, 1]
        mask = preds > threshold
        data_final[year] = y_data[mask]
        logger.info(
            f"Data {year} [{category}] passed MVA > {threshold}: "
            f"{len(data_final[year])} / {len(y_data)}"
        )

# ---- Multiple candidate handling ----
# Events with the same (runNumber, eventNumber) represent multiple reconstructed
# candidates from one physical event. Only one should enter the mass fit to maintain
# statistical independence. We apply random candidate selection (fixed seed = 42).
# If runNumber/eventNumber are not available, log a warning and skip.
import numpy as np


def _deduplicate(arr, label=""):
    """Randomly select one candidate per (run, event) pair. Returns the deduplicated array."""
    if "runNumber" not in arr.fields or "eventNumber" not in arr.fields:
        return arr
    run = ak.to_numpy(arr["runNumber"]).astype(int)
    evt = ak.to_numpy(arr["eventNumber"]).astype(int)
    keys = run.astype(np.int64) * (10**10) + evt.astype(np.int64)
    unique_keys, first_idx = np.unique(keys, return_index=True)
    n_in = len(arr)
    n_unique = len(unique_keys)
    n_multi = n_in - n_unique
    frac = n_multi / n_in if n_in > 0 else 0.0

    if frac > 0:
        logger.info(
            f"  Multiple candidates [{label}]: {n_multi}/{n_in} ({frac*100:.1f}%) "
            f"are duplicates — applying random selection (seed=42)"
        )
        rng = np.random.default_rng(seed=42)
        # For each unique event, pick a random candidate (not necessarily the first)
        selected = []
        for uk in unique_keys:
            cand_indices = np.where(keys == uk)[0]
            chosen = rng.choice(cand_indices)
            selected.append(chosen)
        selected = np.sort(np.array(selected))
        return arr[selected]
    return arr


for year in list(data_final.keys()):
    arr = data_final[year]
    if len(arr) > 0:
        data_final[year] = _deduplicate(arr, label=f"data {year} [{category}]")

# Cache keys now include branch AND category to avoid collisions between LL/DD runs
cache_key_data = f"{branch}_{category}_final_data"
cache_key_mc = f"{branch}_{category}_final_mc"
cache.save(
    cache_key_data,
    data_final,
    dependencies=cut_deps,
    description=f"Data after {branch}/{category} cuts",
)
cache.save(
    cache_key_mc, mc_final, dependencies=cut_deps, description=f"MC after {branch}/{category} cuts"
)

# Write summary
multi_cand_info = {}
for year, arr in data_final.items():
    multi_cand_info[year] = len(arr)

summary = {
    "optimization_type": opt_type,
    "branch": branch,
    "category": category,
    "mc_yields": {k: len(v) for k, v in mc_final.items()},
    "data_yields": {k: len(v) for k, v in data_final.items()},
}

summary_path = Path(summary_file)
summary_path.parent.mkdir(parents=True, exist_ok=True)
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

logger.info(f"Step 4 complete [{category}]. Summary saved to {summary_file}")
