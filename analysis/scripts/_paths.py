from __future__ import annotations

from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent  # analysis/scripts/
ANALYSIS_DIR = SCRIPTS_DIR.parent  # analysis/
ROOT_DIR = ANALYSIS_DIR.parent  # bu2lambdapKK/

# Scripts that used to write to slides/figs/ now write here when run standalone.
SLIDES_DIR = ANALYSIS_DIR / "generated" / "output" / "pid_plots"


def resolve_pid_study_dir() -> Path:
    candidates = [
        ANALYSIS_DIR / "studies" / "standalone" / "pid_optimization_study",
        ANALYSIS_DIR / "studies" / "pid_optimization_study",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate pid_optimization_study under analysis/studies/.")


def resolve_pipeline_cache_dir() -> Path:
    candidates = [
        ANALYSIS_DIR / "generated" / "cache" / "pipeline" / "mva",
        ANALYSIS_DIR / "generated" / "cache" / "pipeline" / "box",
        ANALYSIS_DIR / "analysis_output" / "mva" / "cache",
        ANALYSIS_DIR / "analysis_output" / "box" / "cache",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate a pipeline cache directory.")
