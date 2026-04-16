import sys
from pathlib import Path

# Ensure the project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.config_loader import StudyConfig

if "snakemake" in globals():
    config_dir = snakemake.params.config_dir
    output_dir = snakemake.params.output_dir
else:
    config_dir = "config"
    output_dir = "analysis_output"

config_path = Path(config_dir).resolve()

print(f"Validating configuration under {config_path}")
try:
    config = StudyConfig.from_dir(config_dir, output_dir=output_dir)
    print("Configuration loaded successfully!")

    # Create root output directory if it doesn't exist
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

except Exception as e:
    print(f"Configuration validation failed: {e}")
    sys.exit(1)
