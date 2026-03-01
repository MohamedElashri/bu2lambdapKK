from pathlib import Path

import tomli


class StudyConfig:
    def __init__(self, config_file="fom_config.toml"):
        self.config_path = Path(__file__).resolve().parent / config_file
        with open(self.config_path, "rb") as f:
            self.data = tomli.load(f)

        self.paths = self.data.get("paths", {})
        self.mass_windows = self.data.get("mass_windows", {})
        self.pdg_masses = self.data.get("pdg", {}).get("masses", {})
        self.pdg_widths = self.data.get("pdg", {}).get("widths", {})
        self.fitting = self.data.get("fitting", {})
        self.optimization = self.data.get("optimization", {})
        self.selection = self.data.get("nd_optimizable_selection", {})

    def get_signal_region(self, state: str) -> tuple[float, float]:
        region = self.data["signal_regions"][state.lower()]
        center = region["center"]
        window = region["window"]
        return (center - window, center + window)
