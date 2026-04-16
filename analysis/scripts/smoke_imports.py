import importlib
import sys
from pathlib import Path

# Ensure the analysis package root is importable when run from analysis/
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

SHARED_MODULES = [
    "modules.config_loader",
    "modules.presentation_config",
    "modules.clean_data_loader",
    "modules.cache_manager",
    "modules.plot_utils",
    "modules.mass_fitter",
]


def main() -> None:
    print("Running shared-module import smoke check")
    for module_name in SHARED_MODULES:
        importlib.import_module(module_name)
        print(f"  imported {module_name}")

    from modules.config_loader import StudyConfig
    from modules.presentation_config import get_presentation_config

    config = StudyConfig.from_dir(project_root / "config")
    presentation = get_presentation_config()

    lambda_window = presentation.lambda_mass_min, presentation.lambda_mass_max
    signal_window = presentation.bu_signal_window()

    if len(config.config_paths()) == 0:
        raise RuntimeError("No config files were discovered by StudyConfig")
    if lambda_window[0] >= lambda_window[1]:
        raise RuntimeError(f"Invalid Lambda window: {lambda_window}")
    if signal_window[0] >= signal_window[1]:
        raise RuntimeError(f"Invalid B signal window: {signal_window}")

    print(f"  config root: {config.config_dir}")
    print(f"  presentation data base: {presentation.data_base}")
    print(f"  presentation mc base: {presentation.mc_base}")
    print(f"  Lambda window: {lambda_window[0]} - {lambda_window[1]}")
    print(f"  B signal window: {signal_window[0]} - {signal_window[1]}")
    print("Shared-module import smoke check passed")


if __name__ == "__main__":
    main()
