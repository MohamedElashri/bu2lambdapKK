from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from modules.config_loader import StudyConfig

MC15_PID_BRANCHES = {
    "lp": "Lp_MC15TuneV1_ProbNNp",
    "p": "p_MC15TuneV1_ProbNNp",
    "h1": "h1_MC15TuneV1_ProbNNk",
    "h2": "h2_MC15TuneV1_ProbNNk",
}

DATA_L0_TIS_KEYS = [
    "Bu_L0GlobalDecision_TIS",
    "Bu_L0PhysDecision_TIS",
    "Bu_L0HadronDecision_TIS",
]
MC_L0_TIS_KEYS = [
    "Bu_L0Global_TIS",
    "Bu_L0HadronDecision_TIS",
]
HLT1_TOS_KEYS = [
    "Bu_Hlt1TrackMVADecision_TOS",
    "Bu_Hlt1TwoTrackMVADecision_TOS",
]
HLT2_TOS_KEYS = [
    "Bu_Hlt2Topo2BodyDecision_TOS",
    "Bu_Hlt2Topo3BodyDecision_TOS",
    "Bu_Hlt2Topo4BodyDecision_TOS",
]


@dataclass(frozen=True)
class PresentationConfig:
    analysis_dir: Path
    config: StudyConfig
    data_base: Path
    mc_base: Path
    years: tuple[str, ...]
    year_suffixes: tuple[str, ...]
    year_ints: tuple[int, ...]
    magnets: tuple[str, ...]
    lambda_mass_min: float
    lambda_mass_max: float
    lambda_mass_pdg: float
    pid_product_min: float
    bu_signal_min: float
    bu_signal_max: float
    bu_sideband_low_min: float
    bu_sideband_low_max: float
    bu_sideband_high_min: float
    bu_sideband_high_max: float

    @property
    def kinematic_weight_dir(self) -> Path:
        return self.analysis_dir / "studies" / "kinematic_reweighting" / "output"

    @property
    def pipeline_output_dir(self) -> Path:
        return self.analysis_dir / "analysis_output" / "mva"

    def sideband_scale(self) -> float:
        signal_width = self.bu_signal_max - self.bu_signal_min
        sideband_width = (self.bu_sideband_low_max - self.bu_sideband_low_min) + (
            self.bu_sideband_high_max - self.bu_sideband_high_min
        )
        return signal_width / sideband_width

    def bu_signal_window(self) -> tuple[float, float]:
        return (self.bu_signal_min, self.bu_signal_max)

    def bu_sideband_windows(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (self.bu_sideband_low_min, self.bu_sideband_low_max),
            (self.bu_sideband_high_min, self.bu_sideband_high_max),
        )

    def mc_state_dir(self, state: str) -> str:
        return "Jpsi" if state.lower() == "jpsi" else state


def get_presentation_config() -> PresentationConfig:
    analysis_dir = Path(__file__).resolve().parent.parent
    config = StudyConfig.from_dir(analysis_dir / "config")
    lambda_cfg = config.get_lambda_preselection("LL")

    years = tuple(config.get_input_years())
    year_suffixes = tuple(year[-2:] for year in years)
    year_ints = tuple(int(year) for year in years)

    return PresentationConfig(
        analysis_dir=analysis_dir,
        config=config,
        data_base=config.get_input_data_base_path(),
        mc_base=config.get_input_mc_base_path(),
        years=years,
        year_suffixes=year_suffixes,
        year_ints=year_ints,
        magnets=tuple(config.get_input_magnets()),
        lambda_mass_min=float(lambda_cfg["mass_min"]),
        lambda_mass_max=float(lambda_cfg["mass_max"]),
        lambda_mass_pdg=float(config.pdg_masses.get("lambda", 1115.683)),
        pid_product_min=float(lambda_cfg["pid_product_min"]),
        bu_signal_min=float(config.optimization.get("b_signal_region_min", 5255.0)),
        bu_signal_max=float(config.optimization.get("b_signal_region_max", 5305.0)),
        bu_sideband_low_min=float(config.optimization.get("b_low_sideband_min", 5150.0)),
        bu_sideband_low_max=float(config.optimization.get("b_low_sideband_max", 5230.0)),
        bu_sideband_high_min=float(config.optimization.get("b_high_sideband_min", 5330.0)),
        bu_sideband_high_max=float(config.optimization.get("b_high_sideband_max", 5410.0)),
    )
