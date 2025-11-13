"""
Global pytest fixtures and configuration for the test suite.

Provides reusable fixtures for testing pipeline components without
duplicating setup code across test modules.
"""

from __future__ import annotations

import pytest
import tempfile
import shutil
import tomli
from pathlib import Path
from typing import Dict, Any, Generator
import numpy as np


@pytest.fixture
def tmp_test_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test operations.
    
    Automatically cleaned up after test completion.
    
    Yields:
        Path to temporary directory
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="bu2lambda_test_"))
    try:
        yield tmp_dir
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


@pytest.fixture
def tmp_cache_dir(tmp_test_dir: Path) -> Path:
    """
    Create a temporary cache directory.
    
    Args:
        tmp_test_dir: Temporary test directory fixture
    
    Returns:
        Path to cache directory
    """
    cache_dir = tmp_test_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def tmp_output_dir(tmp_test_dir: Path) -> Path:
    """
    Create a temporary output directory.
    
    Args:
        tmp_test_dir: Temporary test directory fixture
    
    Returns:
        Path to output directory
    """
    output_dir = tmp_test_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """
    Provide a minimal valid configuration dictionary.
    
    Returns:
        Dictionary with sample configuration
    """
    return {
        "branches": {
            "essential": {
                "Bu": ["Bu_PT", "Bu_ETA"],
                "L0": ["L0_MM", "L0_PT"]
            }
        },
        "presets": {
            "minimal": ["essential"]
        }
    }


@pytest.fixture
def sample_physics_config() -> Dict[str, Any]:
    """
    Provide sample physics configuration.
    
    Returns:
        Dictionary with physics parameters
    """
    return {
        "pdg_masses": {
            "jpsi": 3096.900,
            "etac": 2983.900,
            "lambda": 1115.683
        },
        "pdg_widths": {
            "jpsi": 0.093,
            "etac": 31.900
        }
    }


@pytest.fixture
def sample_detector_config() -> Dict[str, Any]:
    """
    Provide sample detector configuration.
    
    Returns:
        Dictionary with detector parameters
    """
    return {
        "signal_regions": {
            "jpsi": {
                "center": 3096.900,
                "width": 0.050
            }
        },
        "mass_windows": {
            "lambda": {
                "min": 1110.0,
                "max": 1120.0
            }
        }
    }


@pytest.fixture
def sample_branching_fractions() -> Dict[str, Any]:
    """
    Provide sample branching fraction data.
    
    Returns:
        Dictionary with branching fractions
    """
    return {
        "jpsi": {
            "to_lambda_p_km": {
                "value": 1.38e-5,
                "uncertainty": 0.08e-5
            }
        }
    }


@pytest.fixture
def mock_data_array() -> np.ndarray:
    """
    Generate mock physics data array.
    
    Returns:
        Numpy structured array with physics variables
    """
    n_events = 1000
    rng = np.random.default_rng(42)
    
    # Create structured array mimicking ROOT branches
    dtype = [
        ('Bu_PT', 'f8'),
        ('Bu_ETA', 'f8'),
        ('Bu_M', 'f8'),
        ('L0_MM', 'f8'),
        ('L0_PT', 'f8')
    ]
    
    data = np.zeros(n_events, dtype=dtype)
    data['Bu_PT'] = rng.normal(5000.0, 1000.0, n_events)
    data['Bu_ETA'] = rng.normal(3.0, 0.5, n_events)
    data['Bu_M'] = rng.normal(5279.0, 50.0, n_events)
    data['L0_MM'] = rng.normal(1115.683, 2.0, n_events)
    data['L0_PT'] = rng.normal(2000.0, 500.0, n_events)
    
    return data


@pytest.fixture
def config_dir_fixture(tmp_test_dir: Path, sample_config_dict: Dict[str, Any]) -> Path:
    """
    Create a temporary config directory with sample TOML files.
    
    Args:
        tmp_test_dir: Temporary test directory
        sample_config_dict: Sample configuration dictionary
    
    Returns:
        Path to config directory
    """
    config_dir = tmp_test_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Create minimal required config files
    configs = {
        "physics.toml": {
            "pdg_masses": {"jpsi": 3096.900, "etac": 2983.900},
            "pdg_widths": {"jpsi": 0.093},
            "pdg_branching_fractions": {},
            "analysis_method": {}
        },
        "detector.toml": {
            "signal_regions": {"jpsi": {"center": 3096.900, "width": 0.050, "window": 0.025}},
            "mass_windows": {"lambda": {"min": 1110.0, "max": 1120.0}},
            "integrated_luminosity": {"2016": 1.0, "2017": 1.5, "2018": 2.0}
        },
        "fitting.toml": {
            "fit_method": {"method": "extended_unbinned"},
            "background_model": {"type": "exponential"}
        },
        "selection.toml": {
            "cuts": {"Bu_PT": {"min": 0.0}}
        },
        "triggers.toml": {
            "lines": ["L0Hadron"]
        },
        "data.toml": {
            "input_data": {"base_path": "/tmp"},
            "input_mc": {"base_path": "/tmp"},
            "output": {"base_path": str(tmp_test_dir / "output")}
        },
        "efficiencies.toml": {
            "trigger": {"2016": {"jpsi": 0.85}}
        }
    }
    
    # Write TOML files
    for filename, content in configs.items():
        config_file = config_dir / filename
        with open(config_file, 'wb') as f:
            import tomli_w
            tomli_w.dump(content, f)
    
    return config_dir


@pytest.fixture
def branches_config_dict() -> Dict[str, Any]:
    """
    Provide sample branch configuration dictionary.
    
    Returns:
        Dictionary with branch configuration
    """
    return {
        "branches": {
            "essential": {
                "description": "Essential branches",
                "Bu": ["Bu_PT", "Bu_ETA", "Bu_M"],
                "L0": ["L0_MM", "L0_PT", "L0_ETA"]
            },
            "kinematics": {
                "description": "Kinematic variables",
                "Bu": ["Bu_P", "Bu_PHI"],
                "L0": ["L0_P", "L0_PHI"]
            }
        },
        "presets": {
            "minimal": ["essential"],
            "standard": ["essential", "kinematics"]
        },
        "aliases": {
            "pid": {
                "lambda_prob": {
                    "data": "L0_ProbNNp",
                    "mc": "L0_MC15TuneV1_ProbNNp"
                }
            }
        }
    }


@pytest.fixture
def branches_config_file(tmp_test_dir: Path, branches_config_dict: Dict[str, Any]) -> Path:
    """
    Create a temporary branches_config.toml file.
    
    Args:
        tmp_test_dir: Temporary test directory
        branches_config_dict: Branch configuration dictionary
    
    Returns:
        Path to branches_config.toml
    """
    config_file = tmp_test_dir / "branches_config.toml"
    with open(config_file, 'wb') as f:
        import tomli_w
        tomli_w.dump(branches_config_dict, f)
    return config_file


# Marker for tests that need tomli_w
def pytest_configure(config: pytest.Config) -> None:
    """
    Configure pytest with custom markers and settings.
    
    Args:
        config: pytest configuration object
    """
    config.addinivalue_line(
        "markers", "requires_tomli_w: Test requires tomli_w for writing TOML files"
    )
