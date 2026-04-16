import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.cache_manager import CacheManager
from modules.config_loader import StudyConfig
from modules.mass_fitter import MassFitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

if "snakemake" in globals():
    no_cache = snakemake.params.no_cache
    config_dir = snakemake.params.config_dir
    cache_dir = snakemake.params.cache_dir
    output_dir = snakemake.params.output_dir
    summary_file = snakemake.input[0]
    yields_file = snakemake.output[0]
    branch = snakemake.params.branch
    category = snakemake.params.category
    opt_method = snakemake.params.get("opt_method", "box")
else:
    no_cache = False
    config_dir = "config"
    opt_method = "box"
    cache_dir = f"analysis_output/{opt_method}/cache"
    output_dir = f"analysis_output/{opt_method}"
    branch = "high_yield"
    category = "LL"
    summary_file = Path(output_dir) / branch / category / "tables" / "cut_summary.json"
    yields_file = Path(output_dir) / branch / category / "tables" / "fitted_yields.csv"

config = StudyConfig.from_dir(config_dir, output_dir=output_dir)

cache = CacheManager(cache_dir=cache_dir)
# Re-compute dependencies matching apply cuts
cut_deps = cache.compute_dependencies(
    config_files=config.config_paths(),
    code_files=[project_root / "scripts" / "apply_cuts.py"],
)

# Load cut data: cache key includes branch AND category (set by apply_cuts.py)
data_dict = cache.load(f"{branch}_{category}_final_data", dependencies=cut_deps)

if data_dict is None:
    logger.error(
        f"Cut data for branch={branch}, category={category} not found in cache. "
        "Run 'snakemake apply_cuts' first."
    )
    sys.exit(1)

logger.info(f"Initializing RooFit Mass Fitter for branch={branch}, category={category}")
fitter = MassFitter(config=config)

# Plot directory: LL and DD plots are kept in separate subdirectories
out_path = Path(output_dir) / branch / category / "plots" / "fits"
fit_result = fitter.perform_fit(
    data_dict,
    fit_combined=True,
    plot_dir=out_path,
    fit_label=f"Lambda {category}",
    profile_significance_states=("chic0", "chic1"),
    profile_significance_datasets={"combined"},
)

import pandas as pd

# Extract yields to save to fitted_yields.csv
rows = []
if fit_result and "combined" in fit_result.get("yields", {}):
    combined_yields = fit_result["yields"]["combined"]
    # Get the active plotting states from shared config.
    all_states = config.get_plotting_states()

    for state in all_states:
        if state in combined_yields:
            val, err = combined_yields[state]
            rows.append({"year": "combined", "state": state, "yield": val, "yield_err": err})
        else:
            # Placeholder for states not in yields (like etac_2s)
            rows.append({"year": "combined", "state": state, "yield": 0.0, "yield_err": 0.0})

df_yields = pd.DataFrame(rows)
Path(yields_file).parent.mkdir(parents=True, exist_ok=True)
df_yields.to_csv(yields_file, index=False)

profile_sig_file = Path(yields_file).with_name("profile_significances.json")
with open(profile_sig_file, "w") as f:
    json.dump(fit_result.get("profile_significances", {}), f, indent=2, sort_keys=True)

logger.info(
    f"Mass fitting complete for branch={branch}, category={category}. "
    f"Yields saved to {yields_file}"
)
logger.info(f"Profile-likelihood significances saved to {profile_sig_file}")
