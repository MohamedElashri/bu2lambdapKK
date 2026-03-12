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
    years = snakemake.params.get("years", ["2016", "2017", "2018"])
    track_types = snakemake.params.get("track_types", ["LL", "DD"])
else:
    no_cache = False
    config_dir = "config"
    cache_dir = "cache"
    output_dir = "analysis_output"
    summary_file = Path(output_dir) / "tables" / "cut_summary.json"
    yields_file = Path(output_dir) / "tables" / "fitted_yields.csv"
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

# Load cut data (Final data after optimal cuts are applied)
data_dict = cache.load("final_data", dependencies=cut_deps)

if data_dict is None:
    logger.error("Cut data not found in cache. Run 'snakemake apply_cuts' first.")
    sys.exit(1)

out_path = Path(output_dir) / "plots" / "fits"
out_path.mkdir(parents=True, exist_ok=True)

logger.info("Initializing RooFit Mass Fitter")
fitter = MassFitter(config=config)

fit_result = fitter.perform_fit(data_dict, fit_combined=True, plot_tag="final_cut")

import pandas as pd

# Extract yields to save to fitted_yields.csv
rows = []
if fit_result and "combined" in fit_result.get("yields", {}):
    combined_yields = fit_result["yields"]["combined"]
    for state in ["jpsi", "etac", "chic0", "chic1", "etac_2s"]:
        if state in combined_yields:
            val, err = combined_yields[state]
            rows.append({"year": "combined", "state": state, "yield": val, "yield_err": err})

df_yields = pd.DataFrame(rows)
Path(yields_file).parent.mkdir(parents=True, exist_ok=True)
df_yields.to_csv(yields_file, index=False)

logger.info(f"Mass fitting complete. Yields saved to {yields_file}")
