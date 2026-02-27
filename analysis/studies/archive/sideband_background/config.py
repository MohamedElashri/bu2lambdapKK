"""
Configuration for Sideband Background Modeling Study.

Defines mass regions, sideband boundaries, and charmonium reference lines.
Ported from: archive/analysis/studies/sideband_background/config.py
"""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class MassRegionConfig:
    """M(B⁺) and M(Λ̄pK⁻) region definitions. All masses in MeV/c²."""

    # B+ mass (PDG)
    B_MASS_PDG: ClassVar[float] = 5279.34

    # M(B+) full range
    MBU_MIN: ClassVar[float] = 2800.0
    MBU_MAX: ClassVar[float] = 5500.0

    # M(Λ̄pK⁻) range (charmonium region)
    MLPK_MIN: ClassVar[float] = 2800.0
    MLPK_MAX: ClassVar[float] = 4000.0

    # B+ signal region (±50 MeV around PDG mass)
    SIGNAL_HALF_WIDTH: ClassVar[float] = 50.0
    SIGNAL_MIN: ClassVar[float] = B_MASS_PDG - SIGNAL_HALF_WIDTH
    SIGNAL_MAX: ClassVar[float] = B_MASS_PDG + SIGNAL_HALF_WIDTH

    # Left sideband slices for shape validation
    LEFT_SIDEBAND_FAR: ClassVar[tuple[float, float]] = (2800.0, 3500.0)
    LEFT_SIDEBAND_MID: ClassVar[tuple[float, float]] = (3500.0, 4500.0)
    LEFT_SIDEBAND_NEAR: ClassVar[tuple[float, float]] = (4500.0, 5150.0)

    # Right sideband (above B+ signal)
    RIGHT_SIDEBAND: ClassVar[tuple[float, float]] = (5330.0, 5500.0)

    # Combined sideband for template extraction (near-signal only)
    TEMPLATE_SIDEBAND_LEFT: ClassVar[tuple[float, float]] = (4500.0, 5150.0)
    TEMPLATE_SIDEBAND_RIGHT: ClassVar[tuple[float, float]] = (5330.0, 5500.0)

    # Binning
    BIN_WIDTH: ClassVar[float] = 5.0  # MeV per bin
    N_BINS_MLPK: ClassVar[int] = int((MLPK_MAX - MLPK_MIN) / BIN_WIDTH)


# Charmonium reference lines: (mass_MeV, latex_label, color)
CHARMONIUM_LINES = [
    (2983.9, r"$\eta_c(1S)$", "#d62728"),
    (3096.9, r"$J/\psi$", "#1f77b4"),
    (3414.7, r"$\chi_{c0}$", "#2ca02c"),
    (3510.7, r"$\chi_{c1}$", "#ff7f0e"),
    (3637.5, r"$\eta_c(2S)$", "#9467bd"),
]

# Sideband region colours for plotting
REGION_COLORS = {
    "Far-left": "#1f77b4",
    "Mid-left": "#2ca02c",
    "Near-left": "#d62728",
    "Right": "#9467bd",
}

# Default config instance
MASS_CONFIG = MassRegionConfig()
