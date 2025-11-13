"""
Mock data generators for testing pipeline components.

Provides utilities to create synthetic ROOT files and physics data
for reproducible testing without requiring real data files.
"""

from __future__ import annotations

import numpy as np
import uproot
import awkward as ak
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


def generate_mock_physics_data(
    n_events: int = 1000,
    branches: Optional[List[str]] = None,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Generate mock physics data with realistic distributions.
    
    Args:
        n_events: Number of events to generate
        branches: List of branch names to generate (uses defaults if None)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping branch names to numpy arrays
    """
    rng = np.random.default_rng(seed)
    
    # Default branches if not specified
    if branches is None:
        branches = [
            "Bu_PT", "Bu_ETA", "Bu_PHI", "Bu_M", "Bu_P",
            "L0_MM", "L0_PT", "L0_ETA", "L0_P",
            "h1_PT", "h2_PT", "h1_ProbNNk", "h2_ProbNNk"
        ]
    
    data: Dict[str, np.ndarray] = {}
    
    for branch in branches:
        if "PT" in branch:
            # Transverse momentum (exponential-like)
            data[branch] = rng.exponential(3000.0, n_events) + 500.0
        elif "ETA" in branch:
            # Pseudorapidity (gaussian in detector acceptance)
            data[branch] = rng.normal(3.0, 0.8, n_events)
        elif "PHI" in branch:
            # Azimuthal angle (uniform)
            data[branch] = rng.uniform(-np.pi, np.pi, n_events)
        elif "Bu_M" in branch:
            # B+ mass (peaked around PDG value)
            data[branch] = rng.normal(5279.0, 50.0, n_events)
        elif "L0_MM" in branch or "Lambda_MM" in branch:
            # Lambda mass (peaked around PDG value)
            data[branch] = rng.normal(1115.683, 2.0, n_events)
        elif "_M" in branch or "_MM" in branch:
            # Generic mass
            data[branch] = rng.normal(3000.0, 100.0, n_events)
        elif "_P" in branch and not ("PT" in branch or "PHI" in branch):
            # Momentum magnitude
            data[branch] = rng.exponential(10000.0, n_events) + 1000.0
        elif "ProbNN" in branch:
            # PID probabilities (uniform)
            data[branch] = rng.uniform(0.0, 1.0, n_events)
        elif "IPCHI2" in branch:
            # IP chi2 (exponential)
            data[branch] = rng.exponential(10.0, n_events)
        elif "FDCHI2" in branch:
            # Flight distance chi2 (large values)
            data[branch] = rng.exponential(100.0, n_events) + 10.0
        elif "DIRA" in branch:
            # Direction angle (close to 1)
            data[branch] = 1.0 - rng.exponential(0.01, n_events)
        else:
            # Default: gaussian
            data[branch] = rng.normal(0.0, 1.0, n_events)
    
    return data


def create_mock_root_file(
    output_path: Union[str, Path],
    tree_name: str = "DecayTree",
    n_events: int = 1000,
    branches: Optional[List[str]] = None,
    seed: int = 42
) -> Path:
    """
    Create a mock ROOT file with physics data.
    
    Args:
        output_path: Path where ROOT file will be created
        tree_name: Name of the TTree
        n_events: Number of events
        branches: List of branch names (uses defaults if None)
        seed: Random seed for reproducibility
    
    Returns:
        Path to created ROOT file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate mock data
    data = generate_mock_physics_data(n_events, branches, seed)
    
    # Write to ROOT file using uproot
    with uproot.recreate(output_path) as file:
        file[tree_name] = data
    
    return output_path


def create_mock_mc_data(
    n_events: int = 1000,
    branches: Optional[List[str]] = None,
    seed: int = 42,
    include_truth: bool = True
) -> Dict[str, np.ndarray]:
    """
    Generate mock MC data with truth information.
    
    Args:
        n_events: Number of events
        branches: Reconstructed branch names
        seed: Random seed
        include_truth: Whether to include MC truth branches
    
    Returns:
        Dictionary with MC data including truth branches
    """
    rng = np.random.default_rng(seed)
    
    # Generate reconstructed data
    data = generate_mock_physics_data(n_events, branches, seed)
    
    if include_truth:
        # Add truth branches
        truth_branches = {
            "Bu_TRUEID": np.full(n_events, 521, dtype=np.int32),  # B+ PDG ID
            "L0_TRUEID": np.full(n_events, -3122, dtype=np.int32),  # Lambda_bar PDG ID
            "Bu_BKGCAT": rng.integers(0, 60, n_events, dtype=np.int32),
            "L0_BKGCAT": rng.integers(0, 60, n_events, dtype=np.int32),
        }
        data.update(truth_branches)
    
    return data


def create_mock_config_toml(
    output_path: Union[str, Path],
    config_type: str = "physics"
) -> Path:
    """
    Create a mock TOML configuration file.
    
    Args:
        output_path: Path where TOML file will be created
        config_type: Type of config ('physics', 'detector', 'fitting', etc.)
    
    Returns:
        Path to created TOML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define sample configurations
    configs = {
        "physics": {
            "pdg_masses": {
                "jpsi": 3096.900,
                "etac": 2983.900,
                "chic0": 3414.75,
                "chic1": 3510.66,
                "lambda": 1115.683
            },
            "pdg_widths": {
                "jpsi": 0.093,
                "etac": 31.900
            }
        },
        "detector": {
            "signal_regions": {
                "jpsi": {"center": 3096.900, "width": 0.050},
                "etac": {"center": 2983.900, "width": 0.020}
            },
            "mass_windows": {
                "lambda": {"min": 1110.0, "max": 1120.0}
            }
        },
        "fitting": {
            "fit_method": {
                "method": "extended_unbinned",
                "strategy": 1
            },
            "background_model": {
                "type": "exponential",
                "initial_slope": -0.001
            }
        },
        "selection": {
            "cuts": {
                "Bu_PT": {"min": 0.0, "max": 1e6},
                "L0_PT": {"min": 500.0}
            }
        }
    }
    
    content = configs.get(config_type, {})
    
    # Write TOML file using binary mode
    import tomli_w
    with open(output_path, 'wb') as f:
        tomli_w.dump(content, f)
    
    return output_path


def generate_signal_background_mix(
    n_signal: int = 500,
    n_background: int = 1500,
    signal_mass: float = 3096.900,
    signal_width: float = 0.020,
    bkg_slope: float = -0.001,
    mass_range: tuple = (2800.0, 3400.0),
    seed: int = 42
) -> np.ndarray:
    """
    Generate mixed signal and background mass distribution.
    
    Args:
        n_signal: Number of signal events
        n_background: Number of background events
        signal_mass: Signal peak position
        signal_width: Signal width
        bkg_slope: Exponential background slope
        mass_range: (min, max) mass range
        seed: Random seed
    
    Returns:
        Array of mass values
    """
    rng = np.random.default_rng(seed)
    
    # Signal (Gaussian)
    signal = rng.normal(signal_mass, signal_width, n_signal)
    
    # Background (Exponential)
    # Generate from exponential and transform to mass range
    background = rng.exponential(-1.0/bkg_slope, n_background) + mass_range[0]
    
    # Clip to mass range
    background = np.clip(background, mass_range[0], mass_range[1])
    
    # Combine and shuffle
    combined = np.concatenate([signal, background])
    rng.shuffle(combined)
    
    return combined


def generate_efficiency_data(
    states: List[str] = None,
    years: List[str] = None,
    seed: int = 42
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Generate mock efficiency data.
    
    Args:
        states: List of physics states
        years: List of data-taking years
        seed: Random seed
    
    Returns:
        Nested dictionary: {state: {year: {"eff": value, "err": error}}}
    """
    rng = np.random.default_rng(seed)
    
    if states is None:
        states = ["jpsi", "etac", "chic0", "chic1"]
    if years is None:
        years = ["2016", "2017", "2018"]
    
    efficiency_data: Dict[str, Dict[str, Dict[str, float]]] = {}
    
    for state in states:
        efficiency_data[state] = {}
        for year in years:
            # Efficiencies between 0.7 and 0.95
            eff = rng.uniform(0.7, 0.95)
            err = eff * rng.uniform(0.02, 0.05)  # 2-5% relative error
            
            efficiency_data[state][year] = {
                "eff": float(eff),
                "err": float(err)
            }
    
    return efficiency_data


def generate_yield_data(
    states: List[str] = None,
    years: List[str] = None,
    seed: int = 42
) -> Dict[str, Dict[str, tuple]]:
    """
    Generate mock yield data with uncertainties.
    
    Args:
        states: List of physics states
        years: List of data-taking years
        seed: Random seed
    
    Returns:
        Nested dictionary: {year: {state: (yield, error)}}
    """
    rng = np.random.default_rng(seed)
    
    if states is None:
        states = ["jpsi", "etac", "chic0", "chic1"]
    if years is None:
        years = ["2016", "2017", "2018"]
    
    yield_data: Dict[str, Dict[str, tuple]] = {}
    
    for year in years:
        yield_data[year] = {}
        for state in states:
            # Different yields for different states
            if state == "jpsi":
                mean_yield = 10000.0
            elif state == "etac":
                mean_yield = 2000.0
            else:
                mean_yield = 1000.0
            
            yield_val = rng.normal(mean_yield, mean_yield * 0.1)
            error = np.sqrt(yield_val)  # Poisson statistics
            
            yield_data[year][state] = (float(yield_val), float(error))
    
    return yield_data
