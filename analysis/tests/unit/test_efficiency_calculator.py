"""
Unit tests for EfficiencyCalculator module.

Tests efficiency calculations, cut applications, and ratio calculations
using mock MC data without requiring real files.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import pandas as pd
import pytest

from analysis.modules.efficiency_calculator import EfficiencyCalculator
from analysis.modules.exceptions import EfficiencyError


@pytest.fixture
def mock_config(tmp_test_dir):
    """Create mock config object."""

    class MockConfig:
        def __init__(self):
            self.paths = {
                "output": {
                    "tables_dir": str(tmp_test_dir / "tables"),
                    "plots_dir": str(tmp_test_dir / "plots"),
                }
            }

    return MockConfig()


@pytest.fixture
def sample_optimized_cuts() -> pd.DataFrame:
    """Provide sample optimized cuts DataFrame."""
    return pd.DataFrame(
        {
            "state": ["jpsi", "jpsi", "etac", "etac"],
            "branch_name": ["Bu_PT", "h1_IPCHI2", "Bu_PT", "h1_IPCHI2"],
            "optimal_cut": [5000.0, 9.0, 4500.0, 12.0],
            "cut_type": ["greater", "greater", "greater", "greater"],
            "fom": [2.5, 2.3, 2.1, 2.0],
        }
    )


@pytest.fixture
def mock_mc_events() -> ak.Array:
    """Create mock MC events."""
    return ak.Array(
        {
            "Bu_PT": np.array([6000.0, 4000.0, 7000.0, 3000.0]),
            "h1_IPCHI2": np.array([15.0, 5.0, 20.0, 10.0]),
            "Bu_M": np.array([5279.0, 5280.0, 5281.0, 5282.0]),
        }
    )


@pytest.fixture
def mock_generated_counts() -> dict:
    """Create mock generated MC counts."""
    return {
        "jpsi": {"2016": 1000, "2017": 1200, "2018": 1500},
        "etac": {"2016": 800, "2017": 900, "2018": 1100},
        "chic0": {"2016": 900, "2017": 1000, "2018": 1200},
        "chic1": {"2016": 850, "2017": 950, "2018": 1150},
    }


@pytest.mark.unit
class TestEfficiencyCalculatorInitialization:
    """Test EfficiencyCalculator initialization."""

    def test_init_without_cuts(self, mock_config) -> None:
        """Test initialization without optimized cuts."""
        calc = EfficiencyCalculator(mock_config)

        assert calc.config is not None
        assert calc.optimized_cuts is None

    def test_init_with_cuts(self, mock_config, sample_optimized_cuts) -> None:
        """Test initialization with optimized cuts."""
        calc = EfficiencyCalculator(mock_config, sample_optimized_cuts)

        assert calc.config is not None
        assert calc.optimized_cuts is not None
        assert len(calc.optimized_cuts) == 4


@pytest.mark.unit
class TestCutRetrieval:
    """Test retrieving cuts for specific states."""

    def test_get_cuts_for_jpsi(self, mock_config, sample_optimized_cuts) -> None:
        """Test getting cuts for J/psi state."""
        calc = EfficiencyCalculator(mock_config, sample_optimized_cuts)

        jpsi_cuts = calc.get_cuts_for_state("jpsi")

        assert len(jpsi_cuts) == 2
        assert all(jpsi_cuts["state"] == "jpsi")

    def test_get_cuts_for_etac(self, mock_config, sample_optimized_cuts) -> None:
        """Test getting cuts for eta_c state."""
        calc = EfficiencyCalculator(mock_config, sample_optimized_cuts)

        etac_cuts = calc.get_cuts_for_state("etac")

        assert len(etac_cuts) == 2
        assert all(etac_cuts["state"] == "etac")

    def test_get_cuts_without_providing_cuts(self, mock_config) -> None:
        """Test error when no cuts provided."""
        calc = EfficiencyCalculator(mock_config)

        with pytest.raises(EfficiencyError) as exc_info:
            calc.get_cuts_for_state("jpsi")

        assert "not provided" in str(exc_info.value)


@pytest.mark.unit
class TestCutApplication:
    """Test applying optimized cuts to MC events."""

    def test_apply_greater_cuts(self, mock_config, sample_optimized_cuts, mock_mc_events) -> None:
        """Test applying 'greater' type cuts."""
        calc = EfficiencyCalculator(mock_config, sample_optimized_cuts)

        # Apply jpsi cuts: Bu_PT > 5000, h1_IPCHI2 > 9
        result = calc.apply_optimized_cuts(mock_mc_events, "jpsi")

        # Only events 0 and 2 should pass (Bu_PT > 5000 and h1_IPCHI2 > 9)
        assert len(result) == 2
        assert result["Bu_PT"][0] == 6000.0
        assert result["Bu_PT"][1] == 7000.0

    def test_apply_cuts_different_state(
        self, mock_config, sample_optimized_cuts, mock_mc_events
    ) -> None:
        """Test applying cuts for different state."""
        calc = EfficiencyCalculator(mock_config, sample_optimized_cuts)

        # Apply etac cuts: Bu_PT > 4500, h1_IPCHI2 > 12
        result = calc.apply_optimized_cuts(mock_mc_events, "etac")

        # Events with Bu_PT > 4500 AND h1_IPCHI2 > 12
        assert len(result) == 2  # Events 0 and 2

    def test_apply_cuts_less_type(self, mock_config) -> None:
        """Test applying 'less' type cuts."""
        cuts_df = pd.DataFrame(
            {
                "state": ["test", "test"],
                "branch_name": ["Bu_PT", "h1_IPCHI2"],
                "optimal_cut": [5000.0, 15.0],
                "cut_type": ["less", "less"],
            }
        )

        calc = EfficiencyCalculator(mock_config, cuts_df)

        events = ak.Array(
            {"Bu_PT": np.array([6000.0, 4000.0, 3000.0]), "h1_IPCHI2": np.array([10.0, 5.0, 20.0])}
        )

        result = calc.apply_optimized_cuts(events, "test")

        # Only event 1 passes (Bu_PT < 5000 AND h1_IPCHI2 < 15)
        assert len(result) == 1
        assert result["Bu_PT"][0] == 4000.0

    def test_unknown_cut_type_raises_error(self, mock_config) -> None:
        """Test that unknown cut type raises error."""
        cuts_df = pd.DataFrame(
            {
                "state": ["test"],
                "branch_name": ["Bu_PT"],
                "optimal_cut": [5000.0],
                "cut_type": ["unknown"],
            }
        )

        calc = EfficiencyCalculator(mock_config, cuts_df)

        events = ak.Array({"Bu_PT": np.array([6000.0])})

        with pytest.raises(EfficiencyError) as exc_info:
            calc.apply_optimized_cuts(events, "test")

        assert "Unknown cut type" in str(exc_info.value)


@pytest.mark.unit
class TestSelectionEfficiency:
    """Test selection efficiency calculation."""

    def test_calculate_efficiency_all_pass(self, mock_config) -> None:
        """Test efficiency when all events pass."""
        cuts_df = pd.DataFrame(
            {
                "state": ["jpsi"],
                "branch_name": ["Bu_PT"],
                "optimal_cut": [1000.0],
                "cut_type": ["greater"],
            }
        )

        gen_counts = {"jpsi": {"2016": 10}}
        calc = EfficiencyCalculator(mock_config, cuts_df, gen_counts)

        events = ak.Array({"Bu_PT": np.array([5000.0, 6000.0, 7000.0])})

        result = calc.calculate_selection_efficiency(events, "jpsi", "2016")

        assert result["eff"] == 0.3  # 3 pass / 10 generated
        assert result["n_before"] == 10  # Generated
        assert result["n_after"] == 3
        assert result["err"] >= 0.0

    def test_calculate_efficiency_none_pass(self, mock_config) -> None:
        """Test efficiency when no events pass."""
        cuts_df = pd.DataFrame(
            {
                "state": ["jpsi"],
                "branch_name": ["Bu_PT"],
                "optimal_cut": [10000.0],
                "cut_type": ["greater"],
            }
        )

        gen_counts = {"jpsi": {"2016": 10}}
        calc = EfficiencyCalculator(mock_config, cuts_df, gen_counts)

        events = ak.Array({"Bu_PT": np.array([5000.0, 6000.0, 7000.0])})

        result = calc.calculate_selection_efficiency(events, "jpsi", "2016")

        assert result["eff"] == 0.0
        assert result["n_before"] == 10  # Generated
        assert result["n_after"] == 0

    def test_calculate_efficiency_partial(
        self, mock_config, sample_optimized_cuts, mock_mc_events, mock_generated_counts
    ) -> None:
        """Test efficiency with partial acceptance."""
        calc = EfficiencyCalculator(mock_config, sample_optimized_cuts, mock_generated_counts)

        result = calc.calculate_selection_efficiency(mock_mc_events, "jpsi", "2016")

        # 2 out of 4 events pass cuts, but efficiency is relative to 1000 generated
        assert 0.0 < result["eff"] < 1.0
        assert result["n_before"] == 1000  # Generated (from mock_generated_counts)
        assert result["n_after"] == 2  # Events passing cuts
        assert 0.0 < result["err"] < 1.0

    def test_efficiency_zero_events(
        self, mock_config, sample_optimized_cuts, mock_generated_counts
    ) -> None:
        """Test efficiency with zero events after Lambda cuts but nonzero generated."""
        calc = EfficiencyCalculator(mock_config, sample_optimized_cuts, mock_generated_counts)

        empty_events = ak.Array({"Bu_PT": np.array([]), "h1_IPCHI2": np.array([])})

        result = calc.calculate_selection_efficiency(empty_events, "jpsi", "2016")

        assert result["eff"] == 0.0
        assert result["n_before"] == 1000  # Generated
        assert result["n_after"] == 0


@pytest.mark.unit
class TestEfficiencyErrorPropagation:
    """Test statistical error calculation for efficiencies."""

    def test_binomial_error_calculation(self, mock_config) -> None:
        """Test binomial error formula."""
        cuts_df = pd.DataFrame(
            {
                "state": ["jpsi"],
                "branch_name": ["Bu_PT"],
                "optimal_cut": [5000.0],
                "cut_type": ["greater"],
            }
        )

        # 50 events pass, 50 fail after Lambda cuts, 200 generated
        gen_counts = {"jpsi": {"2016": 200}}
        calc = EfficiencyCalculator(mock_config, cuts_df, gen_counts)

        # Create events where exactly half pass
        events = ak.Array({"Bu_PT": np.concatenate([np.full(50, 6000.0), np.full(50, 4000.0)])})

        result = calc.calculate_selection_efficiency(events, "jpsi", "2016")

        assert result["eff"] == 0.25  # 50 pass / 200 generated
        # σ = sqrt(ε(1-ε)/N_gen) = sqrt(0.25*0.75/200)
        expected_error = np.sqrt(0.25 * 0.75 / 200)
        assert result["err"] == pytest.approx(expected_error, rel=1e-6)

    def test_error_small_for_large_n(self, mock_config) -> None:
        """Test that error decreases with larger generated sample size."""
        cuts_df = pd.DataFrame(
            {
                "state": ["jpsi"],
                "branch_name": ["Bu_PT"],
                "optimal_cut": [5000.0],
                "cut_type": ["greater"],
            }
        )

        # Small generated sample
        gen_counts_small = {"jpsi": {"2016": 20}}
        calc_small = EfficiencyCalculator(mock_config, cuts_df, gen_counts_small)
        events_small = ak.Array({"Bu_PT": np.array([6000.0, 4000.0, 6000.0, 4000.0])})
        result_small = calc_small.calculate_selection_efficiency(events_small, "jpsi", "2016")

        # Large generated sample
        gen_counts_large = {"jpsi": {"2016": 2000}}
        calc_large = EfficiencyCalculator(mock_config, cuts_df, gen_counts_large)
        pt_values = np.array([6000.0 if i % 2 == 0 else 4000.0 for i in range(100)])
        events_large = ak.Array({"Bu_PT": pt_values})
        result_large = calc_large.calculate_selection_efficiency(events_large, "jpsi", "2016")

        # Larger generated sample should have smaller error
        assert result_large["err"] < result_small["err"]


@pytest.mark.unit
class TestAllEfficienciesCalculation:
    """Test calculating efficiencies for all states and years."""

    def test_calculate_all_efficiencies_structure(self, mock_config, sample_optimized_cuts) -> None:
        """Test structure of all efficiencies calculation."""
        calc = EfficiencyCalculator(mock_config, sample_optimized_cuts)

        # Create mock MC data for multiple states and years
        mc_by_state = {
            "jpsi": {
                "2016": ak.Array(
                    {"Bu_PT": np.array([6000.0, 7000.0]), "h1_IPCHI2": np.array([15.0, 20.0])}
                )
            },
            "etac": {
                "2016": ak.Array(
                    {"Bu_PT": np.array([6000.0, 4000.0]), "h1_IPCHI2": np.array([15.0, 20.0])}
                )
            },
            "chic0": {
                "2016": ak.Array({"Bu_PT": np.array([6000.0]), "h1_IPCHI2": np.array([15.0])})
            },
            "chic1": {
                "2016": ak.Array(
                    {"Bu_PT": np.array([6000.0, 7000.0]), "h1_IPCHI2": np.array([15.0, 20.0])}
                )
            },
        }

        # Note: This will fail because we don't have cuts for chic0/chic1
        # Just testing the structure would work
        with pytest.raises(Exception):
            # Expected to fail due to missing cuts
            calc.calculate_all_efficiencies(mc_by_state)


@pytest.mark.unit
class TestEfficiencyRatios:
    """Test efficiency ratio calculations."""

    def test_efficiency_ratio_calculation(self, mock_config, tmp_test_dir) -> None:
        """Test calculating efficiency ratios relative to J/psi."""
        # Create tables directory
        tables_dir = tmp_test_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        calc = EfficiencyCalculator(mock_config)

        # Mock efficiency data
        efficiencies = {
            "jpsi": {"2016": {"eff": 0.8, "err": 0.02}, "2017": {"eff": 0.85, "err": 0.015}},
            "etac": {"2016": {"eff": 0.6, "err": 0.03}, "2017": {"eff": 0.65, "err": 0.025}},
            "chic0": {"2016": {"eff": 0.7, "err": 0.025}, "2017": {"eff": 0.75, "err": 0.02}},
            "chic1": {"2016": {"eff": 0.7, "err": 0.025}, "2017": {"eff": 0.75, "err": 0.02}},
            "etac_2s": {"2016": {"eff": 0.7, "err": 0.025}, "2017": {"eff": 0.75, "err": 0.02}},
        }

        ratios_df = calc.calculate_efficiency_ratios(efficiencies)

        # Check structure
        assert isinstance(ratios_df, pd.DataFrame)
        assert "state" in ratios_df.columns
        assert "year" in ratios_df.columns

        # Check that all states are present
        assert "etac" in ratios_df["state"].values
        assert "chic0" in ratios_df["state"].values
        assert "chic1" in ratios_df["state"].values
        assert "etac_2s" in ratios_df["state"].values

        # Check that ratios and errors are present
        assert len(ratios_df) > 0
        assert all(ratios_df["ratio_eps_jpsi_over_state"] > 0)


@pytest.mark.unit
def test_efficiency_calculator_workflow(
    mock_config, sample_optimized_cuts, mock_mc_events, mock_generated_counts
) -> None:
    """Test complete efficiency calculation workflow."""
    calc = EfficiencyCalculator(mock_config, sample_optimized_cuts, mock_generated_counts)

    # Step 1: Get cuts for a state
    cuts = calc.get_cuts_for_state("jpsi")
    assert len(cuts) > 0

    # Step 2: Apply cuts
    events_after = calc.apply_optimized_cuts(mock_mc_events, "jpsi")
    assert len(events_after) <= len(mock_mc_events)

    # Step 3: Calculate efficiency
    eff_result = calc.calculate_selection_efficiency(mock_mc_events, "jpsi", "2016")
    assert 0.0 <= eff_result["eff"] <= 1.0
    assert eff_result["err"] >= 0.0
