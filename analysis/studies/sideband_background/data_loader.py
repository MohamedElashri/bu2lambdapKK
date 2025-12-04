"""
Data Loading Utilities for Sideband Background Study

Provides functions to load ROOT data and create histograms for
M(B+) vs M(Lambda p K-) analysis.
"""

import sys
from pathlib import Path
from typing import Any

import ROOT

# Add directories to path
STUDY_DIR: Path = Path(__file__).parent
ANALYSIS_DIR: Path = STUDY_DIR.parent.parent
sys.path.insert(0, str(STUDY_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

from config import MASS_CONFIG  # noqa: E402
from modules.data_handler import TOMLConfig  # noqa: E402


def setup_root_style() -> None:
    """Set up LHCb-style ROOT plotting options."""
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(1)
    ROOT.gStyle.SetPadLeftMargin(0.14)
    ROOT.gStyle.SetPadRightMargin(0.05)
    ROOT.gStyle.SetPadTopMargin(0.08)
    ROOT.gStyle.SetPadBottomMargin(0.12)
    ROOT.gStyle.SetTitleFont(132, "XYZ")
    ROOT.gStyle.SetLabelFont(132, "XYZ")
    ROOT.gStyle.SetTextFont(132)
    ROOT.gStyle.SetTitleSize(0.045, "XYZ")
    ROOT.gStyle.SetLabelSize(0.035, "XYZ")
    ROOT.gStyle.SetTitleOffset(1.1, "X")
    ROOT.gStyle.SetTitleOffset(1.2, "Y")
    ROOT.gStyle.SetPalette(ROOT.kBird)
    ROOT.gStyle.SetNumberContours(100)


def build_cut_string(
    manual_cuts: dict[str, Any] | None,
    lambda_selection: dict[str, Any] | None = None,
) -> str:
    """
    Build ROOT TTree cut string from configuration dictionaries.

    Args:
        manual_cuts: Dictionary of manual cuts from config/selection.toml
        lambda_selection: Dictionary of Lambda selection cuts

    Returns:
        Combined cut string for TTree::Draw
    """
    cut_parts: list[str] = []
    # Lambda selection cuts
    if lambda_selection:
        mass_window = lambda_selection.get("mass_window", {})
        if mass_window:
            min_mass = mass_window.get("min", 1111.0)
            max_mass = mass_window.get("max", 1121.0)
            cut_parts.append(f"(L0_MM > {min_mass} && L0_MM < {max_mass})")
        fdchi2 = lambda_selection.get("fdchi2_ownpv", {})
        if fdchi2:
            cut_parts.append(f"(L0_FDCHI2_OWNPV > {fdchi2.get('value', 250)})")
        delta_z = lambda_selection.get("delta_z", {})
        if delta_z:
            cut_parts.append(f"(L0_ENDVERTEX_Z - Bu_ENDVERTEX_Z > {delta_z.get('value', 5)})")
        lp_probnnp = lambda_selection.get("lp_probnnp", {})
        if lp_probnnp:
            cut_parts.append(f"(Lp_ProbNNp > {lp_probnnp.get('value', 0.3)})")
    # Manual cuts
    if manual_cuts:
        for branch_name, cut_spec in manual_cuts.items():
            if branch_name == "notes":
                continue
            cut_type = cut_spec.get("cut_type")
            cut_value = cut_spec.get("value")
            if cut_type and cut_value is not None:
                if cut_type == "greater":
                    cut_parts.append(f"({branch_name} > {cut_value})")
                elif cut_type == "less":
                    cut_parts.append(f"({branch_name} < {cut_value})")
    if cut_parts:
        return " && ".join(cut_parts)
    return ""


def get_mlpk_formula() -> str:
    """
    Return the ROOT formula for M(Lambda p K-) calculation.

    M(Lambda + p + K-) where h2 is K-.
    M^2 = E^2 - px^2 - py^2 - pz^2
    """
    return (
        "sqrt((L0_PE + p_PE + h2_PE)*(L0_PE + p_PE + h2_PE) - "
        "(L0_PX + p_PX + h2_PX)*(L0_PX + p_PX + h2_PX) - "
        "(L0_PY + p_PY + h2_PY)*(L0_PY + p_PY + h2_PY) - "
        "(L0_PZ + p_PZ + h2_PZ)*(L0_PZ + p_PZ + h2_PZ))"
    )


def load_mlpk_histogram_in_mbu_region(
    data_path: Path,
    years: list[str],
    track_types: list[str],
    mbu_region: tuple[float, float],
    hist_name: str,
    base_cuts: str = "",
    n_bins: int | None = None,
    mlpk_min: float | None = None,
    mlpk_max: float | None = None,
) -> tuple[ROOT.TH1D, int]:
    """
    Load M(Lambda p K-) histogram for events in a specific M(B+) region.

    Args:
        data_path: Base path to data files
        years: List of years to process (e.g., ["2016", "2017", "2018"])
        track_types: List of track types (e.g., ["LL", "DD"])
        mbu_region: Tuple of (min, max) M(B+) values in MeV
        hist_name: Name for the histogram
        base_cuts: Additional cuts to apply (from selection config)
        n_bins: Number of bins (default from config)
        mlpk_min: Minimum M(LpK-) value (default from config)
        mlpk_max: Maximum M(LpK-) value (default from config)

    Returns:
        Tuple of (histogram, total_events)
    """
    # Use defaults from config if not specified
    if n_bins is None:
        n_bins = MASS_CONFIG.N_BINS_MLPK
    if mlpk_min is None:
        mlpk_min = MASS_CONFIG.MLPK_MIN
    if mlpk_max is None:
        mlpk_max = MASS_CONFIG.MLPK_MAX
    # Create histogram
    hist: ROOT.TH1D = ROOT.TH1D(hist_name, "", n_bins, mlpk_min, mlpk_max)
    hist.Sumw2()
    total_events: int = 0
    # Build M(B+) region cut
    mbu_cut: str = f"(Bu_MM > {mbu_region[0]} && Bu_MM < {mbu_region[1]})"
    # Combine with base cuts
    if base_cuts:
        full_cut: str = f"{base_cuts} && {mbu_cut}"
    else:
        full_cut = mbu_cut
    # M(LpK-) formula
    mlpk_formula: str = get_mlpk_formula()
    # Process each year, magnet polarity, and track type
    for year in years:
        year_int: int = int(year)
        for magnet in ["MD", "MU"]:
            for track_type in track_types:
                filename: str = f"dataBu2L0barPHH_{year_int - 2000}{magnet}.root"
                filepath: Path = data_path / filename
                if not filepath.exists():
                    continue
                channel_path: str = f"B2L0barPKpKm_{track_type}"
                tree_path: str = f"{channel_path}/DecayTree"
                try:
                    tfile: ROOT.TFile = ROOT.TFile.Open(str(filepath), "READ")
                    if not tfile or tfile.IsZombie():
                        continue
                    tree: ROOT.TTree = tfile.Get(tree_path)
                    if not tree:
                        tfile.Close()
                        continue
                    # Draw to temporary histogram
                    temp_name: str = f"temp_{hist_name}_{year}_{magnet}_{track_type}"
                    n_entries: int = tree.Draw(
                        f"{mlpk_formula}>>{temp_name}({n_bins},{mlpk_min},{mlpk_max})",
                        full_cut,
                        "goff",
                    )
                    if n_entries > 0:
                        temp_hist: ROOT.TH1D = ROOT.gDirectory.Get(temp_name)
                        if temp_hist:
                            hist.Add(temp_hist)
                            total_events += n_entries
                            temp_hist.Delete()
                    tfile.Close()
                except Exception as e:
                    print(f"Error loading {year} {magnet} {track_type}: {e}")
    # Keep histogram in memory
    ROOT.SetOwnership(hist, False)
    return hist, total_events


def load_2d_histogram(
    data_path: Path,
    years: list[str],
    track_types: list[str],
    hist_name: str,
    base_cuts: str = "",
    n_bins_x: int = 110,
    n_bins_y: int | None = None,
    mbu_min: float | None = None,
    mbu_max: float | None = None,
    mlpk_min: float | None = None,
    mlpk_max: float | None = None,
) -> tuple[ROOT.TH2D, int]:
    """
    Load 2D histogram of M(B+) vs M(Lambda p K-).

    Args:
        data_path: Base path to data files
        years: List of years to process
        track_types: List of track types
        hist_name: Name for the histogram
        base_cuts: Additional cuts to apply
        n_bins_x: Number of bins for M(B+) axis
        n_bins_y: Number of bins for M(LpK-) axis
        mbu_min: Minimum M(B+) value
        mbu_max: Maximum M(B+) value
        mlpk_min: Minimum M(LpK-) value
        mlpk_max: Maximum M(LpK-) value

    Returns:
        Tuple of (2D histogram, total_events)
    """
    # Use defaults from config if not specified
    if mbu_min is None:
        mbu_min = MASS_CONFIG.MBU_MIN
    if mbu_max is None:
        mbu_max = MASS_CONFIG.MBU_MAX
    if mlpk_min is None:
        mlpk_min = MASS_CONFIG.MLPK_MIN
    if mlpk_max is None:
        mlpk_max = MASS_CONFIG.MLPK_MAX
    if n_bins_y is None:
        n_bins_y = MASS_CONFIG.N_BINS_MLPK
    # Create 2D histogram
    hist: ROOT.TH2D = ROOT.TH2D(
        hist_name, "", n_bins_x, mbu_min, mbu_max, n_bins_y, mlpk_min, mlpk_max
    )
    hist.Sumw2()
    total_events: int = 0
    # M(LpK-) formula
    mlpk_formula: str = get_mlpk_formula()
    # 2D draw expression: Y:X format
    draw_expr: str = f"{mlpk_formula}:Bu_MM"
    # Process each year, magnet polarity, and track type
    for year in years:
        year_int: int = int(year)
        for magnet in ["MD", "MU"]:
            for track_type in track_types:
                filename: str = f"dataBu2L0barPHH_{year_int - 2000}{magnet}.root"
                filepath: Path = data_path / filename
                if not filepath.exists():
                    continue
                channel_path: str = f"B2L0barPKpKm_{track_type}"
                tree_path: str = f"{channel_path}/DecayTree"
                try:
                    tfile: ROOT.TFile = ROOT.TFile.Open(str(filepath), "READ")
                    if not tfile or tfile.IsZombie():
                        continue
                    tree: ROOT.TTree = tfile.Get(tree_path)
                    if not tree:
                        tfile.Close()
                        continue
                    # Draw to temporary histogram
                    temp_name: str = f"temp_{hist_name}_{year}_{magnet}_{track_type}"
                    binning: str = (
                        f"({n_bins_x},{mbu_min},{mbu_max},{n_bins_y},{mlpk_min},{mlpk_max})"
                    )
                    n_entries: int = tree.Draw(
                        f"{draw_expr}>>{temp_name}{binning}",
                        base_cuts if base_cuts else "",
                        "goff",
                    )
                    if n_entries > 0:
                        temp_hist: ROOT.TH2D = ROOT.gDirectory.Get(temp_name)
                        if temp_hist:
                            hist.Add(temp_hist)
                            total_events += n_entries
                            temp_hist.Delete()
                    tfile.Close()
                except Exception as e:
                    print(f"Error loading {year} {magnet} {track_type}: {e}")
    # Keep histogram in memory
    ROOT.SetOwnership(hist, False)
    return hist, total_events


def get_config() -> TOMLConfig:
    """Load and return the analysis configuration."""
    config_dir: Path = ANALYSIS_DIR / "config"
    return TOMLConfig(str(config_dir))
