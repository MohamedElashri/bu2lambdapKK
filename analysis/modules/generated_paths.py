from __future__ import annotations

from pathlib import Path

DEFAULT_OUTPUT_ROOT = Path("generated/output")
DEFAULT_CACHE_ROOT = Path("generated/cache")


def analysis_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def output_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    return analysis_dir() / DEFAULT_OUTPUT_ROOT


def cache_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    return analysis_dir() / DEFAULT_CACHE_ROOT


def pipeline_output_dir(opt_method: str = "mva", root: str | Path | None = None) -> Path:
    return output_root(root) / "pipeline" / opt_method


def pipeline_cache_dir(opt_method: str = "mva", root: str | Path | None = None) -> Path:
    return cache_root(root) / "pipeline" / opt_method


def studies_output_dir(root: str | Path | None = None) -> Path:
    return output_root(root) / "studies"


def presentation_output_dir(root: str | Path | None = None) -> Path:
    return output_root(root) / "presentation"


def reports_output_dir(root: str | Path | None = None) -> Path:
    return output_root(root) / "reports"
