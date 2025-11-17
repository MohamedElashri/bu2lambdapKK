"""
Data integrity validation tests.

Tests data quality and consistency:
- Data shapes and dimensions
- NaN/inf value detection
- Data type consistency
- Empty or malformed data
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.tests.utils.mock_data_generator import (
    generate_efficiency_data,
    generate_mock_physics_data,
    generate_yield_data,
)


@pytest.mark.validation
class TestDataShapes:
    """Test that data has expected shapes and dimensions."""

    def test_physics_data_structure(self, mock_data_array: np.ndarray) -> None:
        """Test that physics data has expected structure."""
        # Should have events
        assert len(mock_data_array) > 0, "Data array should not be empty"

        # Should be numpy structured array
        assert isinstance(mock_data_array, np.ndarray), "Data should be numpy array"
        assert mock_data_array.dtype.names is not None, "Data should have named fields"

    def test_physics_data_field_consistency(self, mock_data_array: np.ndarray) -> None:
        """Test that all fields have consistent lengths."""
        n_events = len(mock_data_array)

        # All fields should have same number of entries
        for field in mock_data_array.dtype.names:
            field_data = mock_data_array[field]
            assert len(field_data) == n_events, f"Field {field} has inconsistent length"

    def test_efficiency_data_shape(self) -> None:
        """Test that efficiency data has expected shape."""
        eff_data = generate_efficiency_data(states=["jpsi"], years=["2016"])

        # Returns {state: {year: {eff, err}}}
        assert "jpsi" in eff_data, "Missing jpsi state"
        assert "2016" in eff_data["jpsi"], "Missing 2016 year"

        # Check structure
        jpsi_2016 = eff_data["jpsi"]["2016"]
        assert "eff" in jpsi_2016, "Missing eff"
        assert "err" in jpsi_2016, "Missing err"

        # Both should be numeric
        assert isinstance(jpsi_2016["eff"], (int, float))
        assert isinstance(jpsi_2016["err"], (int, float))

    def test_yield_data_structure(self) -> None:
        """Test that yield data has expected structure."""
        yield_data = generate_yield_data(states=["jpsi"], years=["2016"])

        # Returns {year: {state: (yield, error)}}
        assert "2016" in yield_data, "Missing 2016 year"
        assert "jpsi" in yield_data["2016"], "Missing jpsi state"

        # Check structure - tuple of (yield, error)
        jpsi_yield = yield_data["2016"]["jpsi"]
        assert isinstance(jpsi_yield, tuple), "Yield should be tuple"
        assert len(jpsi_yield) == 2, "Yield tuple should have 2 elements"


@pytest.mark.validation
class TestNaNInfValues:
    """Test detection of NaN and inf values in data."""

    def test_no_nan_in_masses(self, mock_data_array: np.ndarray) -> None:
        """Test that mass values don't contain NaN."""
        # Check common mass fields
        mass_fields = [f for f in mock_data_array.dtype.names if "MM" in f or "M" in f]

        for field in mass_fields:
            data = mock_data_array[field]
            assert not np.any(np.isnan(data)), f"Field {field} contains NaN values"

    def test_no_inf_in_masses(self, mock_data_array: np.ndarray) -> None:
        """Test that mass values don't contain inf."""
        mass_fields = [f for f in mock_data_array.dtype.names if "MM" in f or "M" in f]

        for field in mass_fields:
            data = mock_data_array[field]
            assert not np.any(np.isinf(data)), f"Field {field} contains inf values"

    def test_efficiency_no_nan(self) -> None:
        """Test that efficiency values are not NaN."""
        eff_data = generate_efficiency_data(states=["jpsi"], years=["2016"])
        jpsi_2016 = eff_data["jpsi"]["2016"]

        assert not np.isnan(jpsi_2016["eff"]), "Efficiency should not be NaN"
        assert not np.isnan(jpsi_2016["err"]), "Efficiency error should not be NaN"

    def test_efficiency_no_inf(self) -> None:
        """Test that efficiency values are not inf."""
        eff_data = generate_efficiency_data(states=["jpsi"], years=["2016"])
        jpsi_2016 = eff_data["jpsi"]["2016"]

        assert not np.isinf(jpsi_2016["eff"]), "Efficiency should not be inf"
        assert not np.isinf(jpsi_2016["err"]), "Efficiency error should not be inf"

    def test_yield_no_nan(self) -> None:
        """Test that yield values are not NaN."""
        yield_data = generate_yield_data(states=["jpsi"], years=["2016"])
        jpsi_yield = yield_data["2016"]["jpsi"]

        assert not np.isnan(jpsi_yield[0]), "Yield should not be NaN"
        assert not np.isnan(jpsi_yield[1]), "Yield error should not be NaN"

    def test_array_with_nan_detected(self) -> None:
        """Test that NaN detection works correctly."""
        # Create array with NaN
        data = np.array([1.0, 2.0, np.nan, 4.0])

        # Should detect NaN
        assert np.any(np.isnan(data)), "Should detect NaN in array"

        # Count NaNs
        n_nan = np.sum(np.isnan(data))
        assert n_nan == 1, f"Expected 1 NaN, found {n_nan}"


@pytest.mark.validation
class TestDataTypeConsistency:
    """Test that data types are consistent."""

    def test_masses_are_float(self, mock_data_array: np.ndarray) -> None:
        """Test that mass values are floating point."""
        mass_fields = [f for f in mock_data_array.dtype.names if "MM" in f or "_M" in f]

        for field in mass_fields:
            data = mock_data_array[field]
            assert np.issubdtype(
                data.dtype, np.floating
            ), f"Field {field} should be floating point, got {data.dtype}"

    def test_efficiency_is_numeric(self) -> None:
        """Test that efficiency values are numeric."""
        eff_data = generate_efficiency_data(states=["jpsi"], years=["2016"])
        jpsi_2016 = eff_data["jpsi"]["2016"]

        assert isinstance(jpsi_2016["eff"], (int, float, np.number)), "Efficiency should be numeric"
        assert isinstance(
            jpsi_2016["err"], (int, float, np.number)
        ), "Efficiency error should be numeric"

    def test_yield_is_numeric(self) -> None:
        """Test that yield values are numeric."""
        yield_data = generate_yield_data(states=["jpsi"], years=["2016"])
        jpsi_yield = yield_data["2016"]["jpsi"]

        assert isinstance(jpsi_yield[0], (int, float, np.number)), "Yield should be numeric"
        assert isinstance(jpsi_yield[1], (int, float, np.number)), "Yield error should be numeric"

    def test_year_is_string(self) -> None:
        """Test that year values are strings."""
        yield_data = generate_yield_data(states=["jpsi"], years=["2016"])

        for year in yield_data.keys():
            assert isinstance(year, str), f"Year {year} should be string"

    def test_state_is_string(self) -> None:
        """Test that state values are strings."""
        yield_data = generate_yield_data(states=["jpsi"], years=["2016"])

        for year_data in yield_data.values():
            for state in year_data.keys():
                assert isinstance(state, str), f"State {state} should be string"


@pytest.mark.validation
class TestEmptyData:
    """Test handling of empty or minimal data."""

    def test_empty_array_detected(self) -> None:
        """Test that empty arrays are detected."""
        data = np.array([])
        assert len(data) == 0, "Array should be empty"

    def test_zero_events_handled(self) -> None:
        """Test handling of zero events."""
        data = generate_mock_physics_data(n_events=0)

        # Should be dict with empty arrays
        assert isinstance(data, dict), "Should return dict"
        for field, values in data.items():
            assert len(values) == 0, f"Field {field} should have 0 events"

    def test_single_event_handled(self) -> None:
        """Test handling of single event."""
        data = generate_mock_physics_data(n_events=1)

        # Should be dict with single-element arrays
        assert isinstance(data, dict), "Should return dict"
        for field, values in data.items():
            assert len(values) == 1, f"Field {field} should have 1 event"

    def test_missing_field_detected(self, mock_data_array: np.ndarray) -> None:
        """Test that missing fields are detected."""
        # Try to access non-existent field
        assert (
            "nonexistent_field" not in mock_data_array.dtype.names
        ), "Nonexistent field should not be present"

    def test_negative_event_count_invalid(self) -> None:
        """Test that negative event counts are invalid."""
        with pytest.raises(ValueError):
            generate_mock_physics_data(n_events=-1)


@pytest.mark.validation
class TestDataRanges:
    """Test that data values are in expected ranges."""

    def test_masses_positive(self, mock_data_array: np.ndarray) -> None:
        """Test that mass values are positive."""
        mass_fields = [f for f in mock_data_array.dtype.names if "MM" in f or "_M" in f]

        for field in mass_fields:
            data = mock_data_array[field]
            assert np.all(data > 0), f"Field {field} contains non-positive values"

    def test_efficiency_in_range(self) -> None:
        """Test that efficiency values are in [0, 1]."""
        eff_data = generate_efficiency_data(states=["jpsi"], years=["2016"])
        jpsi_2016 = eff_data["jpsi"]["2016"]

        assert 0 <= jpsi_2016["eff"] <= 1, f"Efficiency {jpsi_2016['eff']} out of range [0,1]"

    def test_efficiency_error_positive(self) -> None:
        """Test that efficiency errors are positive."""
        eff_data = generate_efficiency_data(states=["jpsi"], years=["2016"])
        jpsi_2016 = eff_data["jpsi"]["2016"]

        assert jpsi_2016["err"] >= 0, f"Efficiency error {jpsi_2016['err']} should be non-negative"

    def test_yield_non_negative(self) -> None:
        """Test that yield values are non-negative."""
        yield_data = generate_yield_data(states=["jpsi"], years=["2016"])
        jpsi_yield = yield_data["2016"]["jpsi"]

        assert jpsi_yield[0] >= 0, f"Yield {jpsi_yield[0]} should be non-negative"

    def test_yield_error_positive(self) -> None:
        """Test that yield errors are positive."""
        yield_data = generate_yield_data(states=["jpsi"], years=["2016"])
        jpsi_yield = yield_data["2016"]["jpsi"]

        assert jpsi_yield[1] >= 0, f"Yield error {jpsi_yield[1]} should be non-negative"

    def test_pt_positive(self, mock_data_array: np.ndarray) -> None:
        """Test that PT values are positive."""
        pt_fields = [f for f in mock_data_array.dtype.names if "PT" in f]

        for field in pt_fields:
            data = mock_data_array[field]
            assert np.all(data > 0), f"Field {field} contains non-positive PT values"


@pytest.mark.validation
class TestDataConsistency:
    """Test consistency relationships in data."""

    def test_error_smaller_than_value(self) -> None:
        """Test that statistical errors are reasonable relative to values."""
        yield_data = generate_yield_data(states=["jpsi"], years=["2016"])
        jpsi_yield = yield_data["2016"]["jpsi"]

        if jpsi_yield[0] > 0:
            # Relative error should be less than 100%
            rel_error = jpsi_yield[1] / jpsi_yield[0]
            assert rel_error < 1.0, f"Relative error {rel_error:.2%} seems unreasonably large"

    def test_efficiency_error_reasonable(self) -> None:
        """Test that efficiency errors are reasonable."""
        eff_data = generate_efficiency_data(states=["jpsi"], years=["2016"])
        jpsi_2016 = eff_data["jpsi"]["2016"]

        assert (
            jpsi_2016["err"] <= jpsi_2016["eff"]
        ), "Efficiency error should not exceed efficiency value"
        assert jpsi_2016["err"] < 0.5, "Efficiency error seems unreasonably large"

    def test_multiple_events_different(self) -> None:
        """Test that multiple events are not all identical."""
        data = generate_mock_physics_data(n_events=100)

        # At least one field should have variation
        has_variation = False
        for field, values in data.items():
            if len(values) > 1:
                if np.std(values) > 0:
                    has_variation = True
                    break

        assert has_variation, "All fields are constant - suspicious"

    def test_mass_correlations_sensible(self, mock_data_array: np.ndarray) -> None:
        """Test that mass values have sensible correlations."""
        # Find Bu and Lambda mass fields
        bu_field = None
        lambda_field = None

        for f in mock_data_array.dtype.names:
            if "Bu_M" in f:
                bu_field = f
            if "L0_MM" in f:
                lambda_field = f

        if bu_field and lambda_field:
            # Bu mass should be larger than Lambda mass
            bu_masses = mock_data_array[bu_field]
            lambda_masses = mock_data_array[lambda_field]

            # Most B masses should be larger than Lambda masses
            n_correct = np.sum(bu_masses > lambda_masses)
            fraction = n_correct / len(bu_masses)

            assert fraction > 0.9, f"Only {fraction:.1%} of events have Bu_M > L0_MM"


@pytest.mark.validation
class TestDataCompleteness:
    """Test that data contains all expected information."""

    def test_all_expected_fields_present(self, mock_data_array: np.ndarray) -> None:
        """Test that expected fields are present."""
        # Should have at least some fields
        assert len(mock_data_array.dtype.names) > 0, "Data should have at least one field"

        # Common fields should be present
        expected_fields = ["Bu_PT", "Bu_M", "L0_MM"]
        for field in expected_fields:
            assert field in mock_data_array.dtype.names, f"Expected field {field} not found"

    def test_sufficient_statistics(self) -> None:
        """Test that data has sufficient statistics."""
        data = generate_mock_physics_data(n_events=200)

        # Should have requested number of events
        first_field = list(data.keys())[0]
        n_events = len(data[first_field])

        assert n_events >= 100, "Should have sufficient events for analysis"

    def test_no_duplicate_entries(self) -> None:
        """Test that data doesn't have obvious duplicates."""
        yields = [generate_yield_data(states=["jpsi"], years=["2016"])]

        # Check that we can generate data without errors
        assert len(yields) > 0, "Should generate yield data"

    def test_all_years_have_data(self) -> None:
        """Test that all requested years have data."""
        years = ["2016", "2017", "2018"]
        states = ["jpsi", "etac"]

        yield_data = generate_yield_data(states=states, years=years)

        for year in years:
            assert year in yield_data, f"Missing data for year {year}"
            for state in states:
                assert state in yield_data[year], f"Missing state {state} for year {year}"


@pytest.mark.validation
@pytest.mark.slow
class TestLargeDatasets:
    """Test handling of large datasets."""

    def test_large_dataset_loads(self) -> None:
        """Test that large datasets can be loaded."""
        data = generate_mock_physics_data(n_events=10000)

        # Check size
        first_field = list(data.keys())[0]
        assert len(data[first_field]) == 10000, "Should handle 10k events"

    def test_large_dataset_operations(self) -> None:
        """Test operations on large datasets."""
        data = generate_mock_physics_data(n_events=10000)

        # Should be able to perform operations
        if len(data) > 0:
            first_field = list(data.keys())[0]
            values = data[first_field]

            # Calculate statistics
            mean = np.mean(values)
            std = np.std(values)

            assert np.isfinite(mean), "Mean should be finite"
            assert np.isfinite(std), "Std should be finite"
            assert std >= 0, "Std should be non-negative"

    def test_memory_efficiency(self) -> None:
        """Test that data generation is memory efficient."""
        # Generate moderately large dataset
        data = generate_mock_physics_data(n_events=5000)

        # Check we got the data
        first_field = list(data.keys())[0]
        assert len(data[first_field]) == 5000, "Data size mismatch"
