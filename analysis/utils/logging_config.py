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
"""

import os
import warnings
from typing import Literal

# ROOT-specific suppression
try:
    import ROOT

    ROOT.gErrorIgnoreLevel = ROOT.kError  # Suppress ROOT Info messages
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
            import ROOT

            if level == "off":
                ROOT.gErrorIgnoreLevel = ROOT.kError
            elif level in ["error", "default"]:
                ROOT.gErrorIgnoreLevel = ROOT.kWarning
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
