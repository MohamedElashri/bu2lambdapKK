"""
Step 1 Snakemake wrapper: Configuration Validation

Reproduces the logic from:
  - PipelineManager.__init__() / _validate_config() in run_pipeline.py
  - PipelineManager._setup_output_dirs() in run_pipeline.py
  - ConfigValidator.validate_all() in utils/validate_config.py

Snakemake injects the `snakemake` object with:
  snakemake.params.config_dir   — path to TOML config directory
  snakemake.params.output_dir   — path to output directory root
"""

import sys
from pathlib import Path

# Ensure the project root (analysis_make/) is on sys.path so that
# `modules` and `utils` packages are importable.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.data_handler import TOMLConfig
from modules.exceptions import ConfigurationError
from utils.validate_config import ConfigValidator

# ---------------------------------------------------------------------------
# Read Snakemake params
# ---------------------------------------------------------------------------
config_dir = snakemake.params.config_dir  # noqa: F821 — injected by Snakemake
output_dir = snakemake.params.output_dir  # noqa: F821

# ---------------------------------------------------------------------------
# Step 1: Run ConfigValidator.validate_all()  (thorough checks)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 1: CONFIGURATION VALIDATION")
print("=" * 80)

validator = ConfigValidator(config_dir=config_dir, verbose=False)
success = validator.validate_all()

if not success:
    print("\n[FATAL] Configuration validation failed. Aborting pipeline.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 2: Run PipelineManager._validate_config() equivalent checks
#
# These verify that TOMLConfig created all required attributes and the
# backward-compatibility layer is intact.
# ---------------------------------------------------------------------------
config = TOMLConfig(config_dir)

# Required primary config sections
required = [
    "physics",
    "detector",
    "fitting",
    "selection",
    "triggers",
    "data",
    "efficiencies",
]
for req in required:
    if not hasattr(config, req):
        raise ConfigurationError(
            f"Missing required configuration section: '{req}'\n"
            f"Expected file: config/{req}.toml\n"
            f"Please ensure all required configuration files are present."
        )

# Backward compatibility attributes
compat_attrs = [
    "particles",
    "paths",
    "luminosity",
    "branching_fractions",
]
for attr in compat_attrs:
    if not hasattr(config, attr):
        raise ConfigurationError(
            f"Backward compatibility layer failed to create attribute: '{attr}'\n"
            f"This is an internal error in TOMLConfig._create_compatibility_layer()"
        )

# Warn (but don't fail) if data root doesn't exist
data_root = Path(config.paths["data"]["base_path"])
if not data_root.exists():
    print(f"⚠️  Warning: Data root directory not found: {data_root}")
    print("   Make sure data files are available before running")

# ---------------------------------------------------------------------------
# Step 3: Create output directories (PipelineManager._setup_output_dirs)
# ---------------------------------------------------------------------------
for category in ["tables", "plots", "results"]:
    dir_path = Path(config.paths["output"][f"{category}_dir"])
    dir_path.mkdir(exist_ok=True, parents=True)

# Also create plot subdirectories if defined
if "subdirs" in config.data.get("output", {}):
    for subdir_name, subdir_path in config.data["output"]["subdirs"].items():
        Path(subdir_path).mkdir(exist_ok=True, parents=True)

print("\n✓ Configuration loaded and validated")
print(f"✓ Output directories created under: {output_dir}")
