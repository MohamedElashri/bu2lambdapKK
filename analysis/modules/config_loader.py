from pathlib import Path

import tomli


class StudyConfig:
    def __init__(self, config_file="box_config.toml", output_dir=None):
        # Allow passing an absolute path
        config_path_obj = Path(config_file)
        if config_path_obj.is_absolute():
            self.config_path = config_path_obj
        else:
            self.config_path = Path.cwd() / config_file

        with open(self.config_path, "rb") as f:
            self.data = tomli.load(f)

        self.output_dir = Path(output_dir) if output_dir else Path("output")

        self.paths = self.data.get("paths", {})
        self.mass_windows = self.data.get("mass_windows", {})
        self.pdg_masses = self.data.get("pdg", {}).get("masses", {})
        self.pdg_widths = self.data.get("pdg", {}).get("widths", {})
        self.fitting = self.data.get("fitting", {})
        self.optimization = self.data.get("optimization_strategy", {})
        self.selection = self.data.get("nd_optimizable_selection", {})

    def get_signal_region(self, state: str) -> tuple[float, float]:
        region = self.data["signal_regions"][state.lower()]
        center = region["center"]
        window = region["window"]
        return (center - window, center + window)
