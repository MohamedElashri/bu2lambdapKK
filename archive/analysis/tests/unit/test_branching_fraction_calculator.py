"""
Unit tests for BranchingFractionCalculator module.

Tests branching fraction ratio calculations, yield corrections, and
error propagation using mock data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.modules.branching_fraction_calculator import BranchingFractionCalculator


@pytest.fixture
def mock_yields() -> dict[str, dict[str, tuple[float, float]]]:
    """Provide mock yield data."""
    return {
        "2016": {
            "jpsi": (10000.0, 100.0),
            "etac": (2000.0, 50.0),
            "chic0": (1500.0, 40.0),
            "chic1": (1800.0, 45.0),
            "etac_2s": (500.0, 25.0),
        },
        "2017": {
            "jpsi": (12000.0, 110.0),
            "etac": (2400.0, 55.0),
            "chic0": (1800.0, 45.0),
            "chic1": (2100.0, 50.0),
            "etac_2s": (600.0, 27.0),
        },
    }


@pytest.fixture
def mock_efficiencies() -> dict[str, dict[str, dict[str, float]]]:
    """Provide mock efficiency data."""
    return {
        "jpsi": {"2016": {"eff": 0.85, "err": 0.03}, "2017": {"eff": 0.87, "err": 0.028}},
        "etac": {"2016": {"eff": 0.75, "err": 0.04}, "2017": {"eff": 0.77, "err": 0.038}},
        "chic0": {"2016": {"eff": 0.80, "err": 0.035}, "2017": {"eff": 0.82, "err": 0.033}},
        "chic1": {"2016": {"eff": 0.78, "err": 0.036}, "2017": {"eff": 0.80, "err": 0.034}},
        "etac_2s": {"2016": {"eff": 0.78, "err": 0.036}, "2017": {"eff": 0.80, "err": 0.034}},
    }


@pytest.fixture
def mock_config(tmp_test_dir):
    """Create mock config object."""

    class MockConfig:
        def __init__(self):
            self.paths = {
                "output": {
                    "tables_dir": str(tmp_test_dir / "tables"),
                    "plots_dir": str(tmp_test_dir / "plots"),
                    "results_dir": str(tmp_test_dir / "results"),
                }
            }
            self.luminosity = {"integrated_luminosity": {"2016": 1.0, "2017": 1.5}}

    return MockConfig()


@pytest.mark.unit
class TestBranchingFractionCalculatorInitialization:
    """Test BranchingFractionCalculator initialization."""

    def test_init_with_yields_and_efficiencies(
        self, mock_yields, mock_efficiencies, mock_config
    ) -> None:
        """Test initialization with valid inputs."""
        calc = BranchingFractionCalculator(mock_yields, mock_efficiencies, mock_config)

        assert calc.yields is not None
        assert calc.efficiencies is not None
        assert calc.config is not None


@pytest.mark.unit
class TestEfficiencyCorrectedYield:
    """Test efficiency-corrected yield calculation."""

    def test_calculate_corrected_yield_single_state(
        self, mock_yields, mock_efficiencies, mock_config
    ) -> None:
        """Test corrected yield for single state."""
        calc = BranchingFractionCalculator(mock_yields, mock_efficiencies, mock_config)

        corrected_yield, error = calc.calculate_efficiency_corrected_yield("jpsi")

        # Should sum over years: N/ε for each year
        # 2016: 10000/0.85 = 11764.7
        # 2017: 12000/0.87 = 13793.1
        # Total: ~25557.8
        assert corrected_yield > 0
        assert error > 0
        assert corrected_yield > mock_yields["2016"]["jpsi"][0]  # Should be larger than raw yield

    def test_corrected_yield_includes_all_years(
        self, mock_yields, mock_efficiencies, mock_config
    ) -> None:
        """Test that corrected yield combines all years."""
        calc = BranchingFractionCalculator(mock_yields, mock_efficiencies, mock_config)

        # Calculate for state with data in multiple years
        corrected_yield, error = calc.calculate_efficiency_corrected_yield("etac")

        # Manual calculation for verification
        year_2016_corrected = (
            mock_yields["2016"]["etac"][0] / mock_efficiencies["etac"]["2016"]["eff"]
        )
        year_2017_corrected = (
            mock_yields["2017"]["etac"][0] / mock_efficiencies["etac"]["2017"]["eff"]
        )
        expected = year_2016_corrected + year_2017_corrected

        assert corrected_yield == pytest.approx(expected, rel=1e-4)

    def test_error_propagation_in_corrected_yield(
        self, mock_yields, mock_efficiencies, mock_config
    ) -> None:
        """Test that errors are propagated correctly."""
        calc = BranchingFractionCalculator(mock_yields, mock_efficiencies, mock_config)

        corrected_yield, error = calc.calculate_efficiency_corrected_yield("jpsi")

        # Error should be positive and reasonable
        assert error > 0
        # Relative error should be sensible (less than the corrected yield itself)
        assert error < corrected_yield


@pytest.mark.unit
class TestRatioToJpsi:
    """Test branching fraction ratio calculations."""

    def test_calculate_ratio_basic(self, mock_yields, mock_efficiencies, mock_config) -> None:
        """Test basic ratio calculation."""
        calc = BranchingFractionCalculator(mock_yields, mock_efficiencies, mock_config)

        ratio, error = calc.calculate_ratio_to_jpsi("etac")

        # Ratio should be positive
        assert ratio > 0
        assert error > 0

        # Etac has fewer events than jpsi, so ratio should be < 1
        assert ratio < 1.0

    def test_ratio_formula_correctness(self, mock_yields, mock_efficiencies, mock_config) -> None:
        """Test that ratio formula is correct."""
        calc = BranchingFractionCalculator(mock_yields, mock_efficiencies, mock_config)

        # Calculate manually
        yield_etac, _ = calc.calculate_efficiency_corrected_yield("etac")
        yield_jpsi, _ = calc.calculate_efficiency_corrected_yield("jpsi")
        expected_ratio = yield_etac / yield_jpsi

        # Calculate using method
        ratio, _ = calc.calculate_ratio_to_jpsi("etac")

        assert ratio == pytest.approx(expected_ratio, rel=1e-6)

    def test_ratio_error_propagation(self, mock_yields, mock_efficiencies, mock_config) -> None:
        """Test error propagation in ratio calculation."""
        calc = BranchingFractionCalculator(mock_yields, mock_efficiencies, mock_config)

        ratio, error = calc.calculate_ratio_to_jpsi("chic0")

        # Relative error should be reasonable
        rel_error = error / ratio
        assert 0 < rel_error < 1.0  # Should be less than 100%


@pytest.mark.unit
class TestAllRatiosCalculation:
    """Test calculating all branching fraction ratios."""

    def test_calculate_all_ratios_structure(
        self, mock_yields, mock_efficiencies, mock_config, tmp_test_dir
    ) -> None:
        """Test structure of all ratios calculation."""
        # Create output directories
        (tmp_test_dir / "tables").mkdir(parents=True, exist_ok=True)

        calc = BranchingFractionCalculator(mock_yields, mock_efficiencies, mock_config)

        df = calc.calculate_all_ratios()

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert "numerator" in df.columns
        assert "denominator" in df.columns
        assert "ratio" in df.columns
        assert "stat_error" in df.columns

    def test_all_ratios_includes_required_states(
        self, mock_yields, mock_efficiencies, mock_config, tmp_test_dir
    ) -> None:
        """Test that all required ratios are calculated."""
        (tmp_test_dir / "tables").mkdir(parents=True, exist_ok=True)

        calc = BranchingFractionCalculator(mock_yields, mock_efficiencies, mock_config)

        df = calc.calculate_all_ratios()

        # Check expected ratios are present
        numerators = set(df["numerator"].values)
        assert "etac" in numerators
        assert "chic0" in numerators
        assert "chic1" in numerators
        assert "etac_2s" in numerators

        # Check chi_c1/chi_c0 derived ratio is included
        denominators = set(df["denominator"].values)
        assert "jpsi" in denominators
        assert "chic0" in denominators  # For the chi_c1/chi_c0 ratio

    def test_ratios_are_positive(
        self, mock_yields, mock_efficiencies, mock_config, tmp_test_dir
    ) -> None:
        """Test that all ratios are positive."""
        (tmp_test_dir / "tables").mkdir(parents=True, exist_ok=True)

        calc = BranchingFractionCalculator(mock_yields, mock_efficiencies, mock_config)

        df = calc.calculate_all_ratios()

        # All ratios should be positive
        assert all(df["ratio"] > 0)
        assert all(df["stat_error"] > 0)


@pytest.mark.unit
class TestYieldConsistencyCheck:
    """Test yield consistency checks across years."""

    def test_check_yield_consistency_structure(
        self, mock_yields, mock_efficiencies, mock_config, tmp_test_dir
    ) -> None:
        """Test structure of yield consistency check."""
        # Create output directories
        (tmp_test_dir / "tables").mkdir(parents=True, exist_ok=True)
        (tmp_test_dir / "plots").mkdir(parents=True, exist_ok=True)

        calc = BranchingFractionCalculator(mock_yields, mock_efficiencies, mock_config)

        df = calc.check_yield_consistency_per_year()

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert "state" in df.columns
        assert "year" in df.columns
        assert "N_over_L_eps" in df.columns
        assert "error" in df.columns

    def test_consistency_check_includes_all_states_and_years(
        self, mock_yields, mock_efficiencies, mock_config, tmp_test_dir
    ) -> None:
        """Test that consistency check covers all states and years."""
        (tmp_test_dir / "tables").mkdir(parents=True, exist_ok=True)
        (tmp_test_dir / "plots").mkdir(parents=True, exist_ok=True)

        calc = BranchingFractionCalculator(mock_yields, mock_efficiencies, mock_config)

        df = calc.check_yield_consistency_per_year()

        # Check states are present
        states = set(df["state"].values)
        assert "jpsi" in states
        assert "etac" in states

        # Check years are present
        years = set(df["year"].values)
        assert "2016" in years
        assert "2017" in years


@pytest.mark.unit
def test_branching_fraction_calculator_workflow(
    mock_yields, mock_efficiencies, mock_config, tmp_test_dir
) -> None:
    """Test complete branching fraction calculation workflow."""
    # Create necessary directories
    (tmp_test_dir / "tables").mkdir(parents=True, exist_ok=True)
    (tmp_test_dir / "plots").mkdir(parents=True, exist_ok=True)
    (tmp_test_dir / "results").mkdir(parents=True, exist_ok=True)

    calc = BranchingFractionCalculator(mock_yields, mock_efficiencies, mock_config)

    # Step 1: Calculate efficiency-corrected yields
    yield_jpsi, err_jpsi = calc.calculate_efficiency_corrected_yield("jpsi")
    assert yield_jpsi > 0
    assert err_jpsi > 0

    # Step 2: Calculate ratio to J/psi
    ratio, ratio_err = calc.calculate_ratio_to_jpsi("etac")
    assert 0 < ratio < 1.0
    assert ratio_err > 0

    # Step 3: Calculate all ratios
    ratios_df = calc.calculate_all_ratios()
    assert len(ratios_df) >= 5  # At least etac, chic0, chic1, etac_2s, and chic1/chic0

    # Step 4: Check yield consistency
    consistency_df = calc.check_yield_consistency_per_year()
    assert len(consistency_df) > 0
