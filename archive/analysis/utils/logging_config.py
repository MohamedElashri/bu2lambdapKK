"""
Logging and Warning Configuration Utilities

This module provides centralized control over warning messages and logging
throughout the analysis pipeline.

Usage:
    # At the start of the file:
    from utils.logging_config import suppress_warnings
    suppress_warnings()  # Suppress all warnings by default

    # Or with more control:
    suppress_warnings(level='error')  # Only show errors
    suppress_warnings(level='default')  # Show all warnings

    # Via environment variable:
    export ANALYSIS_WARNINGS=on  # Show warnings
    export ANALYSIS_WARNINGS=off  # Suppress warnings (default)

    # Control data loading messages:
    export ANALYSIS_DATA_LOADING=off  # Hide loading messages
    export ANALYSIS_DATA_LOADING=on   # Show loading messages (default)
"""

import os
import warnings
from typing import Literal

# ROOT-specific suppression
try:
    import ROOT  # type: ignore

    ROOT.gErrorIgnoreLevel = ROOT.kError  # Suppress ROOT Info messages # type: ignore
except ImportError:
    pass


def suppress_warnings(level: Literal["off", "error", "default", "all"] = "off") -> None:
    """
    Configure warning levels for the analysis.

    Args:
        level: Warning level to set
            - 'off': Suppress all warnings (default for pipeline)
            - 'error': Show only errors and exceptions
            - 'default': Show important warnings but filter common noise
            - 'all': Show everything (useful for debugging)

    Environment variable ANALYSIS_WARNINGS overrides the level parameter.
    """
    # Check environment variable override
    env_level = os.environ.get("ANALYSIS_WARNINGS", "").lower()
    if env_level in ["on", "yes", "true", "1"]:
        level = "all"
    elif env_level in ["off", "no", "false", "0"]:
        level = "off"
    elif env_level in ["error", "default"]:
        level = env_level

    if level == "off":
        # Suppress all warnings
        warnings.filterwarnings("ignore")

        # Suppress specific common warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        # NumPy warnings
        import numpy as np

        np.seterr(all="ignore")

    elif level == "error":
        # Only show errors, suppress warnings
        warnings.filterwarnings("error")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

    elif level == "default":
        # Show warnings but filter common noise
        warnings.filterwarnings("default")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*uproot.*")
        warnings.filterwarnings("ignore", message=".*awkward.*")

    elif level == "all":
        # Show everything
        warnings.filterwarnings("default")
        import numpy as np

        np.seterr(all="warn")

    # Suppress specific library warnings
    _suppress_library_warnings(level)


def _suppress_library_warnings(level: str) -> None:
    """Suppress known noisy warnings from specific libraries."""
    if level in ["off", "error", "default"]:
        # Awkward array warnings
        warnings.filterwarnings("ignore", module="awkward.*")

        # ROOT warnings
        try:
            import ROOT  # type: ignore

            if level == "off":
                ROOT.gErrorIgnoreLevel = ROOT.kError  # type: ignore
            elif level in ["error", "default"]:
                ROOT.gErrorIgnoreLevel = ROOT.kWarning  # type: ignore
        except ImportError:
            pass

        # Matplotlib backend warnings
        warnings.filterwarnings("ignore", message=".*Matplotlib.*")

        # iminuit warnings
        warnings.filterwarnings("ignore", module="iminuit.*")


def enable_progress_bars() -> bool:
    """
    Check if progress bars should be enabled.

    Returns:
        True if progress bars should be shown, False otherwise.

    Can be controlled via ANALYSIS_PROGRESS environment variable.
    """
    env_progress = os.environ.get("ANALYSIS_PROGRESS", "on").lower()
    return env_progress in ["on", "yes", "true", "1"]


def show_data_loading_messages() -> bool:
    """
    Check if data loading messages should be shown.

    Returns:
        True if loading messages should be shown, False otherwise.

    Reads from config/data.toml [verbosity] section by default.
    Can be overridden via ANALYSIS_DATA_LOADING environment variable.

    Config setting: verbosity.show_data_loading_messages = true/false
    Environment override: ANALYSIS_DATA_LOADING=on/off

    Suppresses loading messages like:
    - "Missing N branches in X"
    - "Loaded data X: N events"
    - "MC file not found (will skip)"
    """
    # Environment variable takes precedence
    env_loading = os.environ.get("ANALYSIS_DATA_LOADING", "").lower()
    if env_loading in ["on", "yes", "true", "1"]:
        return True
    elif env_loading in ["off", "no", "false", "0"]:
        return False

    # Otherwise read from config file
    try:
        from pathlib import Path

        import tomli

        config_path = Path(__file__).parent.parent / "config" / "data.toml"
        if config_path.exists():
            with open(config_path, "rb") as f:
                config = tomli.load(f)
                return config.get("verbosity", {}).get("show_data_loading_messages", True)
    except Exception:
        pass

    # Default: show messages
    return True


def get_tqdm_kwargs(desc: str = "", **kwargs) -> dict:
    """
    Get standard kwargs for tqdm progress bars with consistent styling.

    Args:
        desc: Description for the progress bar
        **kwargs: Additional tqdm parameters

    Returns:
        Dictionary of tqdm parameters
    """
    default_kwargs = {
        "desc": desc,
        "unit": "it",
        "ncols": 80,
        "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        "disable": not enable_progress_bars(),
    }
    default_kwargs.update(kwargs)
    return default_kwargs
