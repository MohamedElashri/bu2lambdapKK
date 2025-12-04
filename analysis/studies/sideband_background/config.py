"""
Configuration for Sideband Background Modeling

Defines mass regions, sideband boundaries, and analysis parameters.
"""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class MassRegionConfig:
    """
    Configuration for B+ mass regions used in sideband background modeling.

    All masses in MeV/c².
    """

    # B+ mass (PDG)
    B_MASS_PDG: ClassVar[float] = 5279.34

    # M(B+) full range for 2D study
    MBU_MIN: ClassVar[float] = 2800.0
    MBU_MAX: ClassVar[float] = 5500.0

    # M(Lambda p K-) range (charmonium region)
    MLPK_MIN: ClassVar[float] = 2800.0
    MLPK_MAX: ClassVar[float] = 4000.0

    # B+ signal region (±50 MeV around PDG mass)
    SIGNAL_HALF_WIDTH: ClassVar[float] = 50.0
    SIGNAL_MIN: ClassVar[float] = B_MASS_PDG - SIGNAL_HALF_WIDTH  # 5229.34
    SIGNAL_MAX: ClassVar[float] = B_MASS_PDG + SIGNAL_HALF_WIDTH  # 5329.34

    # Left sideband slices for shape validation
    # These are used to check if M(LpK-) shape depends on M(B+)
    LEFT_SIDEBAND_FAR: ClassVar[tuple[float, float]] = (2800.0, 3500.0)
    LEFT_SIDEBAND_MID: ClassVar[tuple[float, float]] = (3500.0, 4500.0)
    LEFT_SIDEBAND_NEAR: ClassVar[tuple[float, float]] = (4500.0, 5150.0)

    # Right sideband (above B+ signal)
    RIGHT_SIDEBAND: ClassVar[tuple[float, float]] = (5330.0, 5500.0)

    # Combined sideband for template extraction
    # Uses near-left + right for best statistics near signal
    TEMPLATE_SIDEBAND_LEFT: ClassVar[tuple[float, float]] = (4500.0, 5150.0)
    TEMPLATE_SIDEBAND_RIGHT: ClassVar[tuple[float, float]] = (5330.0, 5500.0)

    # Binning
    BIN_WIDTH: ClassVar[float] = 5.0  # MeV per bin
    N_BINS_MLPK: ClassVar[int] = int((MLPK_MAX - MLPK_MIN) / BIN_WIDTH)  # 240 bins


@dataclass(frozen=True)
class CharmoniumConfig:
    """
    Charmonium state masses and windows for reference lines.

    All masses in MeV/c².
    """

    ETAC_1S: ClassVar[tuple[float, str, int]] = (2983.9, "#eta_{c}(1S)", 1)  # kRed
    JPSI: ClassVar[tuple[float, str, int]] = (3096.9, "J/#psi", 4)  # kBlue
    CHIC0: ClassVar[tuple[float, str, int]] = (3414.7, "#chi_{c0}", 8)  # kGreen+2
    CHIC1: ClassVar[tuple[float, str, int]] = (3510.7, "#chi_{c1}", 46)  # kOrange+1
    ETAC_2S: ClassVar[tuple[float, str, int]] = (3637.5, "#eta_{c}(2S)", 6)  # kMagenta

    @classmethod
    def get_all_states(cls) -> list[tuple[float, str, int]]:
        """Return list of all charmonium states (mass, label, color)."""
        return [cls.ETAC_1S, cls.JPSI, cls.CHIC0, cls.CHIC1, cls.ETAC_2S]


# Default configuration instances
MASS_CONFIG: MassRegionConfig = MassRegionConfig()
CHARMONIUM_CONFIG: CharmoniumConfig = CharmoniumConfig()
