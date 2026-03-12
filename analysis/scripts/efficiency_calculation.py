import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure the project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.cache_manager import CacheManager
from modules.config_loader import StudyConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

if "snakemake" in globals():
    no_cache = snakemake.params.no_cache
    config_dir = snakemake.params.config_dir
    cache_dir = snakemake.params.cache_dir
    output_dir = snakemake.params.output_dir
    branch = snakemake.params.branch
    summary_file = snakemake.input.summary
    cuts_file = snakemake.input.cuts
    eff_file = snakemake.output.efficiencies
    ratios_file = snakemake.output.ratios
else:
    no_cache = False
    config_dir = "config"
    cache_dir = "cache"
    output_dir = "analysis_output"
    branch = "high_yield"
    opt_type = "box"
    summary_file = Path(output_dir) / opt_type / branch / "tables" / "cut_summary.json"
    cuts_file = Path(output_dir) / opt_type / "tables" / "optimized_cuts.json"
    eff_file = Path(output_dir) / opt_type / branch / "tables" / "efficiencies.csv"
    ratios_file = Path(output_dir) / opt_type / branch / "tables" / "efficiency_ratios.csv"

config_path = Path(config_dir) / "selection.toml"
config = StudyConfig(config_file=str(config_path), output_dir=output_dir)

cache = CacheManager(cache_dir=cache_dir)
# Re-compute dependencies
cut_deps = cache.compute_dependencies(
    config_files=list(Path(config_dir).glob("*.toml")),
    code_files=[
        project_root / "scripts" / "apply_cuts.py",
    ],
)

# Load branch-specific cut data
mc_final = cache.load(f"{branch}_final_mc", dependencies=cut_deps)

if mc_final is None:
    logger.error(
        f"Cut data for branch {branch} not found in cache. Run 'snakemake apply_cuts' first."
    )
    sys.exit(1)

# As requested, we maintain a placeholder efficiency of 1.0 for now.
logger.info(f"Calculating Efficiencies for branch {branch} (using placeholder 1.0 as requested)")

eff_rows = []
for state in ["jpsi", "etac", "chic0", "chic1", "etac_2s"]:
    eff_rows.append(
        {"state": state, "efficiency": 1.0, "efficiency_err": 0.0, "note": "Placeholder"}
    )

df_eff = pd.DataFrame(eff_rows)
Path(eff_file).parent.mkdir(parents=True, exist_ok=True)
df_eff.to_csv(eff_file, index=False)

# Ratios relative to J/psi
ratios_rows = []
for state in ["jpsi", "etac", "chic0", "chic1", "etac_2s"]:
    ratios_rows.append(
        {"state": state, "ratio_to_jpsi": 1.0, "ratio_err": 0.0, "note": "Placeholder"}
    )

df_ratios = pd.DataFrame(ratios_rows)
df_ratios.to_csv(ratios_file, index=False)

logger.info(
    f"Efficiency calculation complete for branch {branch}. Saved to {eff_file} and {ratios_file}"
)
