import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.cache_manager import CacheManager
from modules.clean_data_loader import load_all_data, load_all_mc
from modules.config_loader import StudyConfig
from modules.generated_paths import pipeline_cache_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

if "snakemake" in globals():
    years = snakemake.params.years
    track_types = snakemake.params.track_types
    magnets = snakemake.params.magnets
    states = snakemake.params.states
    no_cache = snakemake.params.no_cache
    config_dir = snakemake.params.config_dir
    cache_dir = snakemake.params.cache_dir
else:
    # Fallback for manual running
    years = None
    track_types = ["LL", "DD"]
    magnets = None
    states = None
    no_cache = False
    config_dir = "config"
    cache_dir = str(pipeline_cache_dir("mva", project_root / "generated" / "cache"))

config = StudyConfig.from_dir(config_dir)

if years is None:
    years = config.get_input_years() or ["2016", "2017", "2018"]
if magnets is None:
    magnets = config.get_input_magnets() or ["MD", "MU"]
if states is None:
    states = config.get_input_mc_states() or ["Jpsi", "etac", "chic0", "chic1"]

# Initialize CacheManager
cache = CacheManager(cache_dir=cache_dir)
dependencies = cache.compute_dependencies(
    config_files=config.config_paths(),
    code_files=[
        project_root / "modules" / "clean_data_loader.py",
        project_root / "scripts" / "load_data.py",
    ],
    extra_params={
        "years": years,
        "track_types": track_types,
        "magnets": magnets,
        "states": states,
    },
)

if not no_cache:
    data_dict = cache.load("preprocessed_data", dependencies=dependencies)
    mc_dict = cache.load("preprocessed_mc", dependencies=dependencies)

    if data_dict is not None and mc_dict is not None:
        logger.info("✓ Loaded cached data and signal MC")
        sys.exit(0)
    logger.info("Cache miss or invalidated - will compute")

# Paths and fixed selections are owned by the shared config layer.
data_base_path = config.get_input_data_base_path()
mc_base_path = config.get_input_mc_base_path()

logger.info("[Loading Real Data]")
data_dict = load_all_data(
    data_base_path, years, magnets=magnets, track_types=track_types, config=config
)

logger.info("\n[Loading MC - Signal States]")
mc_dict = load_all_mc(
    mc_base_path, states, years, magnets=magnets, track_types=track_types, config=config
)

cache.save(
    "preprocessed_data", data_dict, dependencies=dependencies, description="Data Pre-processed"
)
cache.save("preprocessed_mc", mc_dict, dependencies=dependencies, description="MC Pre-processed")

logger.info("\n✓ Step 2 complete: Data and signal MC loaded successfully")
