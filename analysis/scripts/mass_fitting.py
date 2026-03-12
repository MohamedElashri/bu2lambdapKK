import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.cache_manager import CacheManager
from modules.config_loader import StudyConfig

# Import the actual Box Fitter
sys.path.insert(0, str(project_root / "studies" / "box_optimization"))
from box_fitter import MassFitter

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
    years = snakemake.params.get("years", ["2016", "2017", "2018"])
    track_types = snakemake.params.get("track_types", ["LL", "DD"])
else:
    no_cache = False
    config_dir = "config"
    cache_dir = "analysis_output/box/cache"
    output_dir = "analysis_output/box"
    branch = "high_yield"
    summary_file = Path(output_dir) / branch / "tables" / "cut_summary.json"
    yields_file = Path(output_dir) / branch / "tables" / "fitted_yields.csv"
    years = ["2016", "2017", "2018"]
    track_types = ["LL", "DD"]

config_path = Path(config_dir) / "selection.toml"
config = StudyConfig(config_file=str(config_path), output_dir=output_dir)

cache = CacheManager(cache_dir=cache_dir)
# Re-compute dependencies matching apply cuts
cut_deps = cache.compute_dependencies(
    config_files=list(Path(config_dir).glob("*.toml")),
    code_files=[
        project_root / "scripts" / "apply_cuts.py",
    ],
)

# Load cut data (Final data after optimal cuts are applied for this branch)
data_dict = cache.load(f"{branch}_final_data", dependencies=cut_deps)

if data_dict is None:
    logger.error(
        f"Cut data for branch {branch} not found in cache. Run 'snakemake apply_cuts' first."
    )
    sys.exit(1)

# Branch-specific plot directory (Hierarchical: {output_dir}/{branch}/plots/fits)
out_path = Path(output_dir) / branch / "plots" / "fits"
out_path.mkdir(parents=True, exist_ok=True)

logger.info(f"Initializing RooFit Mass Fitter for branch: {branch}")
fitter = MassFitter(config=config)

# We pass the absolute path for plots to the fitter via plot_tag
fit_result = fitter.perform_fit(data_dict, fit_combined=True, plot_tag=str(out_path.absolute()))

# The plots are likely saved in a default location by the Fitter class
# If needed, we would move them here.

import pandas as pd

# Extract yields to save to fitted_yields.csv
rows = []
if fit_result and "combined" in fit_result.get("yields", {}):
    combined_yields = fit_result["yields"]["combined"]
    # Include etac_2s as placeholder
    for state in ["jpsi", "etac", "chic0", "chic1", "etac_2s"]:
        if state in combined_yields:
            val, err = combined_yields[state]
            rows.append({"year": "combined", "state": state, "yield": val, "yield_err": err})
        else:
            # Placeholder for etac_2s if not in yields
            rows.append({"year": "combined", "state": state, "yield": 0.0, "yield_err": 0.0})

df_yields = pd.DataFrame(rows)
Path(yields_file).parent.mkdir(parents=True, exist_ok=True)
df_yields.to_csv(yields_file, index=False)

logger.info(f"Mass fitting complete for branch {branch}. Yields saved to {yields_file}")
