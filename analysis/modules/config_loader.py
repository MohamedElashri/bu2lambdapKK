from __future__ import annotations

from pathlib import Path
from typing import Any

import tomli


class StudyConfig:
    """Shared configuration root for the active analysis workflow.

    The legacy interface exposed only ``selection.toml`` via ``config.data``.
    The current interface extends that ownership so the active pipeline and
    active studies can read the full config directory through one object while
    keeping backwards compatibility for scripts that still expect
    ``config.data`` to be the selection config.
    """

    CONFIG_FILES = [
        "selection.toml",
        "physics.toml",
        "data.toml",
        "generator_effs.toml",
        "fitting.toml",
        "detector.toml",
        "triggers.toml",
        "efficiencies.toml",
        "paths.toml",
        "luminosity.toml",
        "branching_fractions.toml",
    ]

    def __init__(self, config_file: str = "selection.toml", output_dir: str | Path | None = None):
        config_path_obj = Path(config_file)
        if config_path_obj.suffix == ".toml":
            self.config_path = (
                config_path_obj.resolve()
                if config_path_obj.is_absolute()
                else (Path.cwd() / config_path_obj).resolve()
            )
            self.config_dir = self.config_path.parent
        else:
            self.config_dir = (
                config_path_obj.resolve()
                if config_path_obj.is_absolute()
                else (Path.cwd() / config_path_obj).resolve()
            )
            self.config_path = self.config_dir / "selection.toml"

        self.output_dir = Path(output_dir) if output_dir else Path("output")

        self.configs: dict[str, dict[str, Any]] = {}
        for filename in self.CONFIG_FILES:
            path = self.config_dir / filename
            if path.exists():
                self.configs[path.stem] = self._load_toml(path)

        if "selection" not in self.configs:
            raise FileNotFoundError(f"Could not find selection.toml under {self.config_dir}")

        # Backwards-compatible alias: active scripts still expect .data to be
        # the selection configuration.
        self.data = self.configs["selection"]

        self.selection_data = self.data
        self.physics_data = self.configs.get("physics", {})
        self.data_config = self.configs.get("data", {})
        self.generator_effs = self.configs.get("generator_effs", {})
        self.detector = self.configs.get("detector", {})
        self.triggers = self.configs.get("triggers", {})
        self.efficiencies = self.configs.get("efficiencies", {})
        self.paths_config = self.configs.get("paths", {})
        self.luminosity = self.configs.get("luminosity", {})
        self.branching_fractions = self.configs.get("branching_fractions", {})

        selection_pdg = self.selection_data.get("pdg", {})
        self.pdg_masses = self.physics_data.get("pdg_masses", selection_pdg.get("masses", {}))
        self.pdg_widths = self.physics_data.get("pdg_widths", selection_pdg.get("widths", {}))

        self.mass_windows = dict(self.selection_data.get("mass_windows", {}))
        self.paths = dict(self.selection_data.get("paths", {}))
        self.optimization = dict(self.selection_data.get("optimization_strategy", {}))
        self.selection = dict(self.selection_data.get("nd_optimizable_selection", {}))
        self.cut_application = dict(self.selection_data.get("cut_application", {}))
        self.baseline_selection = dict(self.selection_data.get("baseline_selection", {}))
        self.fixed_selection = dict(self.selection_data.get("fixed_selection", {}))
        self.lambda_selection = dict(self.selection_data.get("lambda_selection", {}))
        self.xgboost = dict(self.selection_data.get("xgboost", {}))

        self.fitting = self._build_fitting_config()

    @classmethod
    def from_dir(cls, config_dir: str | Path = "config", output_dir: str | Path | None = None):
        return cls(config_file=str(config_dir), output_dir=output_dir)

    def _load_toml(self, path: Path) -> dict[str, Any]:
        with open(path, "rb") as f:
            return tomli.load(f)

    def _build_fitting_config(self) -> dict[str, Any]:
        """Compose the active fitting config from reference + operational sources."""
        fitting_ref = self.configs.get("fitting", {})
        merged: dict[str, Any] = {}

        # Reference plotting metadata from fitting.toml stays useful, while the
        # operational fitter parameters still come from selection.toml today.
        merged.update(fitting_ref.get("fit_method", {}))
        merged.update(fitting_ref.get("background_model", {}))
        merged.update(self.selection_data.get("fitting", {}))

        plotting = {}
        plotting.update(fitting_ref.get("plotting", {}))
        plotting.update(self.selection_data.get("plotting", {}))
        if plotting:
            merged["plotting"] = plotting

        labels = {}
        labels.update(fitting_ref.get("labels", {}))
        if labels:
            merged["labels"] = labels

        return merged

    def get(self, config_name: str, default: Any = None) -> Any:
        stem = Path(config_name).stem
        return self.configs.get(stem, default)

    def config_paths(self) -> list[Path]:
        return [
            self.config_dir / filename
            for filename in self.CONFIG_FILES
            if (self.config_dir / filename).exists()
        ]

    def get_signal_region(self, state: str) -> tuple[float, float]:
        region = self.selection_data["signal_regions"][state.lower()]
        center = region["center"]
        window = region["window"]
        return (center - window, center + window)

    def get_optimization_type(self, default: str = "box") -> str:
        return self.cut_application.get("optimization_type", default)

    def get_input_data_base_path(self) -> Path:
        return Path(self.data_config.get("input_data", {}).get("base_path", ""))

    def get_input_mc_base_path(self) -> Path:
        return Path(self.data_config.get("input_mc", {}).get("base_path", ""))

    def get_input_years(self) -> list[str]:
        years = self.data_config.get("input_data", {}).get("years", [])
        return [str(year) for year in years]

    def get_input_magnets(self) -> list[str]:
        return [str(mag) for mag in self.data_config.get("input_data", {}).get("magnets", [])]

    def get_input_mc_states(self) -> list[str]:
        return [str(state) for state in self.data_config.get("input_mc", {}).get("states", [])]

    def get_plotting_states(self, default: list[str] | None = None) -> list[str]:
        if default is None:
            default = ["jpsi", "etac", "chic0", "chic1", "etac_2s"]
        return list(self.fitting.get("plotting", {}).get("states", default))

    def get_ref_state(self, default: str = "jpsi") -> str:
        return self.fitting.get("plotting", {}).get("ref_state", default)

    def get_state_labels(self) -> dict[str, str]:
        return dict(self.fitting.get("plotting", {}).get("labels", {}))

    def get_baseline_reduction(self) -> dict[str, float]:
        return {
            "bu_fdchi2_min": float(self.baseline_selection.get("bu_fdchi2_min", 175.0)),
            "bu_ipchi2_max": float(self.baseline_selection.get("bu_ipchi2_max", 10.0)),
            "bu_pt_min": float(self.baseline_selection.get("bu_pt_min", 3000.0)),
            "delta_z_min": float(self.baseline_selection.get("delta_z_min", 2.5)),
            "lp_probnnp_min": float(self.baseline_selection.get("p_probnnp_min", 0.05)),
            "p_probnnp_min": float(self.baseline_selection.get("p_probnnp_min", 0.05)),
            "hh_probnnk_prod_min": float(
                self.baseline_selection.get("h1_h2_probnnk_prod_min", 0.05)
            ),
        }

    def get_category_delta_z_cut(self, category: str) -> float:
        category = category.upper()
        key = "delta_z_min_ll" if category == "LL" else "delta_z_min_dd"
        default = 20.0 if category == "LL" else 5.0
        return float(self.lambda_selection.get(key, default))

    def get_lambda_preselection(self, category: str) -> dict[str, float]:
        return {
            "mass_min": float(self.lambda_selection.get("mass_min", 1111.0)),
            "mass_max": float(self.lambda_selection.get("mass_max", 1121.0)),
            "fd_chisq_min": float(self.lambda_selection.get("fd_chisq_min", 50.0)),
            "delta_z_min": self.get_category_delta_z_cut(category),
            "proton_probnnp_min": float(self.lambda_selection.get("proton_probnnp_min", 0.3)),
            "pid_product_min": float(self.fixed_selection.get("pid_product_min", 0.25)),
        }

    def get_b_mass_window_with_sidebands(self) -> tuple[float, float]:
        low = min(
            float(self.optimization.get("b_low_sideband_min", 5150.0)),
            float(self.optimization.get("b_signal_region_min", 5255.0)),
        )
        high = max(
            float(self.optimization.get("b_high_sideband_max", 5410.0)),
            float(self.optimization.get("b_signal_region_max", 5305.0)),
        )
        return (low, high)

    def get_norm_branching_fractions(self) -> dict[str, dict[str, float]]:
        return dict(self.physics_data.get("pdg_branching_fractions", {}))
