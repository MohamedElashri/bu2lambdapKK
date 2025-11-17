"""
Unit tests for LambdaSelector module.

Tests Lambda selection cuts, B+ mass cuts, and efficiency calculations
using mock awkward arrays without requiring real data.
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
import pytest

from analysis.modules.exceptions import BranchMissingError
from analysis.modules.lambda_selector import LambdaSelector


@pytest.fixture
def mock_lambda_cuts() -> dict[str, Any]:
    """Provide mock Lambda cut values."""
    return {
        "mass_min": 1111.0,
        "mass_max": 1121.0,
        "fd_chisq_min": 250.0,
        "delta_z_min": 5.0,
        "proton_probnnp_min": 0.3,
    }


@pytest.fixture
def mock_bu_fixed_cuts() -> dict[str, Any]:
    """Provide mock B+ fixed cut values."""
    return {"mass_corrected_min": 5255.0, "mass_corrected_max": 5305.0}


@pytest.fixture
def mock_config_with_lambda_cuts(mock_lambda_cuts, mock_bu_fixed_cuts):
    """Create mock config object with Lambda cuts."""

    class MockConfig:
        def get_lambda_cuts(self):
            return mock_lambda_cuts

        def get_bu_fixed_cuts(self):
            return mock_bu_fixed_cuts

    return MockConfig()


@pytest.fixture
def mock_events_passing() -> ak.Array:
    """Create mock events that pass all Lambda cuts."""
    return ak.Array(
        {
            "Bu_MM": np.array([5279.0, 5280.0, 5281.0]),
            "L0_MM": np.array([1115.0, 1116.0, 1115.5]),  # Within 1111-1121
            "L0_FDCHI2_OWNPV": np.array([300.0, 350.0, 400.0]),  # > 250
            "Delta_Z_mm": np.array([10.0, -8.0, 12.0]),  # |value| > 5
            "Lp_ProbNNp": np.array([0.5, 0.6, 0.7]),  # > 0.3
        }
    )


@pytest.fixture
def mock_events_failing() -> ak.Array:
    """Create mock events that fail Lambda cuts."""
    return ak.Array(
        {
            "Bu_MM": np.array([5279.0, 5280.0, 5281.0]),
            "L0_MM": np.array([1110.0, 1125.0, 1100.0]),  # Outside 1111-1121
            "L0_FDCHI2_OWNPV": np.array([300.0, 350.0, 400.0]),
            "Delta_Z_mm": np.array([10.0, 8.0, 12.0]),
            "Lp_ProbNNp": np.array([0.5, 0.6, 0.7]),
        }
    )


@pytest.mark.unit
class TestLambdaSelectorInitialization:
    """Test LambdaSelector initialization."""

    def test_init_with_config(self, mock_config_with_lambda_cuts) -> None:
        """Test initialization with valid config."""
        selector = LambdaSelector(mock_config_with_lambda_cuts)

        assert selector.config is not None
        assert selector.cuts is not None
        assert isinstance(selector.cuts, dict)

    def test_cuts_loaded_from_config(self, mock_config_with_lambda_cuts, mock_lambda_cuts) -> None:
        """Test that cuts are loaded from config."""
        selector = LambdaSelector(mock_config_with_lambda_cuts)

        assert selector.cuts == mock_lambda_cuts
        assert "mass_min" in selector.cuts
        assert "mass_max" in selector.cuts


@pytest.mark.unit
class TestLambdaCuts:
    """Test Lambda selection cuts."""

    def test_apply_cuts_passing_events(
        self, mock_config_with_lambda_cuts, mock_events_passing
    ) -> None:
        """Test that passing events are retained."""
        selector = LambdaSelector(mock_config_with_lambda_cuts)

        result = selector.apply_lambda_cuts(mock_events_passing)

        # All events should pass
        assert len(result) == 3

    def test_apply_cuts_failing_events(
        self, mock_config_with_lambda_cuts, mock_events_failing
    ) -> None:
        """Test that failing events are rejected."""
        selector = LambdaSelector(mock_config_with_lambda_cuts)

        result = selector.apply_lambda_cuts(mock_events_failing)

        # Should reject events with bad L0_MM
        assert len(result) < 3

    def test_mass_window_cut(self, mock_config_with_lambda_cuts) -> None:
        """Test Lambda mass window cut."""
        selector = LambdaSelector(mock_config_with_lambda_cuts)

        events = ak.Array(
            {
                "Bu_MM": np.array([5279.0, 5280.0, 5281.0, 5282.0]),
                "L0_MM": np.array([1115.0, 1110.0, 1125.0, 1116.0]),  # 2nd and 3rd fail
                "L0_FDCHI2_OWNPV": np.array([300.0, 300.0, 300.0, 300.0]),
                "Delta_Z_mm": np.array([10.0, 10.0, 10.0, 10.0]),
                "Lp_ProbNNp": np.array([0.5, 0.5, 0.5, 0.5]),
            }
        )

        result = selector.apply_lambda_cuts(events)

        # Only 1st and 4th should pass (L0_MM in [1111, 1121])
        assert len(result) == 2
        assert result["L0_MM"][0] == 1115.0
        assert result["L0_MM"][1] == 1116.0

    def test_fdchi2_cut(self, mock_config_with_lambda_cuts) -> None:
        """Test Lambda FDCHI2 cut."""
        selector = LambdaSelector(mock_config_with_lambda_cuts)

        events = ak.Array(
            {
                "Bu_MM": np.array([5279.0, 5280.0, 5281.0]),
                "L0_MM": np.array([1115.0, 1115.0, 1115.0]),
                "L0_FDCHI2_OWNPV": np.array([300.0, 100.0, 400.0]),  # 2nd fails (< 250)
                "Delta_Z_mm": np.array([10.0, 10.0, 10.0]),
                "Lp_ProbNNp": np.array([0.5, 0.5, 0.5]),
            }
        )

        result = selector.apply_lambda_cuts(events)

        # Only 1st and 3rd should pass
        assert len(result) == 2

    def test_delta_z_cut(self, mock_config_with_lambda_cuts) -> None:
        """Test Delta Z cut (absolute value)."""
        selector = LambdaSelector(mock_config_with_lambda_cuts)

        events = ak.Array(
            {
                "Bu_MM": np.array([5279.0, 5280.0, 5281.0, 5282.0]),
                "L0_MM": np.array([1115.0, 1115.0, 1115.0, 1115.0]),
                "L0_FDCHI2_OWNPV": np.array([300.0, 300.0, 300.0, 300.0]),
                "Delta_Z_mm": np.array([10.0, -8.0, 3.0, -15.0]),  # 3rd fails (|3| < 5)
                "Lp_ProbNNp": np.array([0.5, 0.5, 0.5, 0.5]),
            }
        )

        result = selector.apply_lambda_cuts(events)

        # 1st, 2nd, and 4th should pass (|Delta_Z| > 5)
        assert len(result) == 3

    def test_proton_pid_cut(self, mock_config_with_lambda_cuts) -> None:
        """Test proton PID cut."""
        selector = LambdaSelector(mock_config_with_lambda_cuts)

        events = ak.Array(
            {
                "Bu_MM": np.array([5279.0, 5280.0, 5281.0]),
                "L0_MM": np.array([1115.0, 1115.0, 1115.0]),
                "L0_FDCHI2_OWNPV": np.array([300.0, 300.0, 300.0]),
                "Delta_Z_mm": np.array([10.0, 10.0, 10.0]),
                "Lp_ProbNNp": np.array([0.5, 0.1, 0.8]),  # 2nd fails (< 0.3)
            }
        )

        result = selector.apply_lambda_cuts(events)

        # Only 1st and 3rd should pass
        assert len(result) == 2


@pytest.mark.unit
class TestAlternativeBranchNames:
    """Test handling of alternative branch names."""

    def test_l0_m_instead_of_l0_mm(self, mock_config_with_lambda_cuts) -> None:
        """Test using L0_M instead of L0_MM."""
        selector = LambdaSelector(mock_config_with_lambda_cuts)

        events = ak.Array(
            {
                "Bu_M": np.array([5279.0, 5280.0]),
                "L0_M": np.array([1115.0, 1110.0]),  # 2nd fails
                "L0_FDCHI2_OWNPV": np.array([300.0, 300.0]),
                "Delta_Z_mm": np.array([10.0, 10.0]),
                "Lp_ProbNNp": np.array([0.5, 0.5]),
            }
        )

        result = selector.apply_lambda_cuts(events)
        assert len(result) == 1

    def test_bu_m_instead_of_bu_mm(self, mock_config_with_lambda_cuts) -> None:
        """Test using Bu_M instead of Bu_MM for reference."""
        selector = LambdaSelector(mock_config_with_lambda_cuts)

        events = ak.Array(
            {
                "Bu_M": np.array([5279.0, 5280.0]),
                "L0_MM": np.array([1115.0, 1116.0]),
                "L0_FDCHI2_OWNPV": np.array([300.0, 300.0]),
                "Delta_Z_mm": np.array([10.0, 10.0]),
                "Lp_ProbNNp": np.array([0.5, 0.5]),
            }
        )

        result = selector.apply_lambda_cuts(events)
        assert len(result) == 2


@pytest.mark.unit
class TestMissingBranches:
    """Test error handling for missing branches."""

    def test_missing_lambda_mass_branch(self, mock_config_with_lambda_cuts) -> None:
        """Test error when Lambda mass branch is missing."""
        selector = LambdaSelector(mock_config_with_lambda_cuts)

        events = ak.Array(
            {
                "Bu_MM": np.array([5279.0]),
                # No L0_MM or L0_M
                "L0_FDCHI2_OWNPV": np.array([300.0]),
                "Delta_Z_mm": np.array([10.0]),
                "Lp_ProbNNp": np.array([0.5]),
            }
        )

        with pytest.raises(BranchMissingError) as exc_info:
            selector.apply_lambda_cuts(events)

        assert "L0_MM" in str(exc_info.value)

    def test_missing_reference_branch(self, mock_config_with_lambda_cuts) -> None:
        """Test error when reference branch (Bu_MM/Bu_M) is missing."""
        selector = LambdaSelector(mock_config_with_lambda_cuts)

        events = ak.Array(
            {
                # No Bu_MM or Bu_M
                "L0_MM": np.array([1115.0]),
                "L0_FDCHI2_OWNPV": np.array([300.0]),
                "Delta_Z_mm": np.array([10.0]),
                "Lp_ProbNNp": np.array([0.5]),
            }
        )

        with pytest.raises(BranchMissingError):
            selector.apply_lambda_cuts(events)


@pytest.mark.unit
class TestBuMassCuts:
    """Test B+ mass cuts."""

    def test_apply_bu_mass_cut(self, mock_config_with_lambda_cuts) -> None:
        """Test B+ mass window cut."""
        selector = LambdaSelector(mock_config_with_lambda_cuts)

        events = ak.Array(
            {
                "Bu_MM_corrected": np.array([5250.0, 5279.0, 5290.0, 5310.0])
                # 1st and 4th fail (outside 5255-5305)
            }
        )

        result = selector.apply_bu_fixed_cuts(events)

        # Only 2nd and 3rd should pass (5279 and 5290)
        assert len(result) == 2
        assert result["Bu_MM_corrected"][0] == 5279.0
        assert result["Bu_MM_corrected"][1] == 5290.0

    def test_missing_bu_mm_corrected(self, mock_config_with_lambda_cuts) -> None:
        """Test error when Bu_MM_corrected is missing."""
        selector = LambdaSelector(mock_config_with_lambda_cuts)

        events = ak.Array({"Bu_MM": np.array([5279.0])})  # Wrong branch name

        with pytest.raises(BranchMissingError) as exc_info:
            selector.apply_bu_fixed_cuts(events)

        assert "Bu_MM_corrected" in str(exc_info.value)


@pytest.mark.unit
class TestEfficiencyCalculation:
    """Test Lambda efficiency calculation from MC."""

    def test_calculate_efficiency(self, mock_config_with_lambda_cuts) -> None:
        """Test efficiency calculation."""
        selector = LambdaSelector(mock_config_with_lambda_cuts)

        # Create MC events where 2 out of 3 pass
        mc_events = ak.Array(
            {
                "Bu_MM": np.array([5279.0, 5280.0, 5281.0]),
                "L0_MM": np.array([1115.0, 1110.0, 1116.0]),  # 2nd fails
                "L0_FDCHI2_OWNPV": np.array([300.0, 300.0, 300.0]),
                "Delta_Z_mm": np.array([10.0, 10.0, 10.0]),
                "Lp_ProbNNp": np.array([0.5, 0.5, 0.5]),
            }
        )

        efficiency = selector.get_lambda_efficiency_from_mc(mc_events, mc_events)

        assert efficiency == pytest.approx(2.0 / 3.0, rel=1e-6)

    def test_efficiency_zero_events(self, mock_config_with_lambda_cuts) -> None:
        """Test efficiency calculation with zero events."""
        selector = LambdaSelector(mock_config_with_lambda_cuts)

        empty_events = ak.Array(
            {
                "Bu_MM": np.array([]),
                "L0_MM": np.array([]),
                "L0_FDCHI2_OWNPV": np.array([]),
                "Delta_Z_mm": np.array([]),
                "Lp_ProbNNp": np.array([]),
            }
        )

        efficiency = selector.get_lambda_efficiency_from_mc(empty_events, empty_events)

        assert efficiency == 0.0

    def test_efficiency_all_pass(self, mock_config_with_lambda_cuts, mock_events_passing) -> None:
        """Test efficiency when all events pass."""
        selector = LambdaSelector(mock_config_with_lambda_cuts)

        efficiency = selector.get_lambda_efficiency_from_mc(
            mock_events_passing, mock_events_passing
        )

        assert efficiency == 1.0
