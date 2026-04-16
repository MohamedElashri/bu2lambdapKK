"""
mass_fitting_combined.py — Combined LL+DD mass fit.

Loads the post-cut cached data for both Lambda track categories (LL and DD),
concatenates them per year, and performs a single mass fit on the merged sample.

Purpose
-------
  • Publication-quality combined mass spectrum plot (standard in LHCb notes).
  • Cross-check that combined yields ≈ N_LL + N_DD from the separate fits.

The primary yield extraction still uses the separate LL/DD fits (handled by
mass_fitting.py), which correctly account for the different mass resolutions
and background shapes of the two categories.  This combined fit shares a
single (average) resolution parameter — it is a valid approximation for
display purposes; the combined dataset is not used in branching_ratios.py.

Outputs
-------
  {OUTPUT_DIR}/{branch}/LL_DD/tables/fitted_yields_combined.csv
  {OUTPUT_DIR}/{branch}/LL_DD/plots/fits/mass_fit_{year}.pdf
  {OUTPUT_DIR}/{branch}/LL_DD/plots/fits/mass_fit_combined.pdf
"""

import logging
import sys
from pathlib import Path

import awkward as ak
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.cache_manager import CacheManager
from modules.config_loader import StudyConfig
from modules.generated_paths import pipeline_cache_dir, pipeline_output_dir
from modules.mass_fitter import MassFitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter wiring (Snakemake or standalone)
# ---------------------------------------------------------------------------
if "snakemake" in globals():
    no_cache = snakemake.params.no_cache
    config_dir = snakemake.params.config_dir
    cache_dir = snakemake.params.cache_dir
    output_dir = snakemake.params.output_dir
    branch = snakemake.params.branch
    opt_method = snakemake.params.get("opt_method", "mva")
    yields_file = snakemake.output[0]
else:
    no_cache = False
    config_dir = "config"
    opt_method = "mva"
    cache_dir = str(pipeline_cache_dir(opt_method, project_root / "generated" / "cache"))
    output_dir = str(pipeline_output_dir(opt_method, project_root / "generated" / "output"))
    branch = "high_yield"
    yields_file = Path(output_dir) / branch / "LL_DD" / "tables" / "fitted_yields_combined.csv"

config = StudyConfig.from_dir(config_dir, output_dir=output_dir)
cache = CacheManager(cache_dir=cache_dir)

cut_deps = cache.compute_dependencies(
    config_files=config.config_paths(),
    code_files=[project_root / "scripts" / "apply_cuts.py"],
)

# ---------------------------------------------------------------------------
# Load LL and DD cached datasets
# ---------------------------------------------------------------------------
data_ll = cache.load(f"{branch}_LL_final_data", dependencies=cut_deps)
data_dd = cache.load(f"{branch}_DD_final_data", dependencies=cut_deps)

if data_ll is None or data_dd is None:
    logger.error(
        f"Could not load cached data for branch={branch}. "
        "Run 'snakemake apply_cuts' for both LL and DD first."
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Merge LL + DD per year
# ---------------------------------------------------------------------------
all_years = sorted(set(data_ll.keys()) | set(data_dd.keys()))
data_combined: dict[str, ak.Array] = {}

for year in all_years:
    parts = []
    if year in data_ll:
        parts.append(data_ll[year])
    if year in data_dd:
        parts.append(data_dd[year])
    data_combined[year] = ak.concatenate(parts)
    n_ll = len(data_ll.get(year, []))
    n_dd = len(data_dd.get(year, []))
    logger.info(f"  {year}: LL={n_ll:,}  DD={n_dd:,}  combined={len(data_combined[year]):,}")

# ---------------------------------------------------------------------------
# Run the combined fit
# ---------------------------------------------------------------------------
logger.info(f"Running combined LL+DD mass fit for branch={branch}")
fitter = MassFitter(config=config)

plot_dir = Path(output_dir) / branch / "LL_DD" / "plots" / "fits"
fit_result = fitter.perform_fit(
    data_combined,
    fit_combined=True,
    plot_dir=plot_dir,
    fit_label="Lambda LL+DD",
)

# ---------------------------------------------------------------------------
# Save yields
# ---------------------------------------------------------------------------
all_states = config.get_plotting_states()

rows = []
combined_yields = fit_result.get("yields", {}).get("combined", {})
for state in all_states:
    if state in combined_yields:
        val, err = combined_yields[state]
    else:
        val, err = 0.0, 0.0
    rows.append({"year": "combined", "state": state, "yield": val, "yield_err": err})

df = pd.DataFrame(rows)
Path(yields_file).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(yields_file, index=False)

logger.info(f"Combined LL+DD fit complete for branch={branch}. " f"Yields saved to {yields_file}")
