"""
Infrastructure validation tests.

Smoke tests to verify that the test framework is properly configured
and all fixtures are working correctly.
"""

from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any

from .utils.test_helpers import (
    assert_arrays_close,
    assert_dicts_equal,
    assert_file_exists,
    assert_dir_exists
)
from .utils.mock_data_generator import (
    generate_mock_physics_data,
    create_mock_root_file,
    generate_efficiency_data,
    generate_yield_data
)


@pytest.mark.unit
class TestInfrastructure:
    """Test suite for validating test infrastructure."""
    
    def test_tmp_test_dir_fixture(self, tmp_test_dir: Path) -> None:
        """Verify temporary test directory fixture works."""
        assert tmp_test_dir.exists()
        assert tmp_test_dir.is_dir()
        
        # Can write to it
        test_file = tmp_test_dir / "test.txt"
        test_file.write_text("test")
        assert test_file.exists()
    
    def test_tmp_cache_dir_fixture(self, tmp_cache_dir: Path) -> None:
        """Verify cache directory fixture works."""
        assert tmp_cache_dir.exists()
        assert tmp_cache_dir.is_dir()
        assert tmp_cache_dir.name == "cache"
    
    def test_sample_config_dict_fixture(self, sample_config_dict: Dict[str, Any]) -> None:
        """Verify sample config dictionary fixture."""
        assert "branches" in sample_config_dict
        assert "presets" in sample_config_dict
        assert isinstance(sample_config_dict["branches"], dict)
    
    def test_mock_data_array_fixture(self, mock_data_array: np.ndarray) -> None:
        """Verify mock data array fixture."""
        assert len(mock_data_array) == 1000
        assert "Bu_PT" in mock_data_array.dtype.names
        assert "L0_MM" in mock_data_array.dtype.names


@pytest.mark.unit
class TestHelpers:
    """Test suite for test helper functions."""
    
    def test_assert_arrays_close(self) -> None:
        """Test array comparison helper."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 3.0])
        assert_arrays_close(arr1, arr2)
        
        # Should raise for different arrays
        arr3 = np.array([1.0, 2.0, 4.0])
        with pytest.raises(AssertionError):
            assert_arrays_close(arr1, arr3)
    
    def test_assert_dicts_equal(self) -> None:
        """Test dictionary comparison helper."""
        dict1 = {"a": 1, "b": 2.0, "c": "test"}
        dict2 = {"a": 1, "b": 2.0, "c": "test"}
        assert_dicts_equal(dict1, dict2)
        
        # Should raise for different dicts
        dict3 = {"a": 1, "b": 3.0, "c": "test"}
        with pytest.raises(AssertionError):
            assert_dicts_equal(dict1, dict3)
    
    def test_file_assertions(self, tmp_test_dir: Path) -> None:
        """Test file existence assertions."""
        test_file = tmp_test_dir / "test.txt"
        test_file.write_text("content")
        
        assert_file_exists(test_file)
        assert_dir_exists(tmp_test_dir)
        
        # Should raise for non-existent file
        with pytest.raises(AssertionError):
            assert_file_exists(tmp_test_dir / "nonexistent.txt")


@pytest.mark.unit
class TestMockDataGenerator:
    """Test suite for mock data generators."""
    
    def test_generate_mock_physics_data(self) -> None:
        """Test physics data generation."""
        data = generate_mock_physics_data(n_events=100, seed=42)
        
        # Check default branches are created
        assert "Bu_PT" in data
        assert "L0_MM" in data
        assert len(data["Bu_PT"]) == 100
        
        # Check values are reasonable
        assert np.all(data["Bu_PT"] > 0)  # PT should be positive
        assert np.all(np.abs(data["Bu_ETA"]) < 10)  # ETA in reasonable range
    
    def test_generate_mock_physics_data_custom_branches(self) -> None:
        """Test physics data generation with custom branches."""
        branches = ["Bu_M", "L0_PT", "custom_var"]
        data = generate_mock_physics_data(n_events=50, branches=branches, seed=42)
        
        assert set(data.keys()) == set(branches)
        assert len(data["Bu_M"]) == 50
    
    def test_create_mock_root_file(self, tmp_test_dir: Path) -> None:
        """Test ROOT file creation."""
        output_file = tmp_test_dir / "test.root"
        created_file = create_mock_root_file(
            output_file,
            tree_name="TestTree",
            n_events=100,
            seed=42
        )
        
        assert created_file.exists()
        assert created_file.suffix == ".root"
        
        # Verify we can read it back
        import uproot
        with uproot.open(created_file) as f:
            tree = f["TestTree"]
            assert len(tree["Bu_PT"].array()) == 100
    
    def test_generate_efficiency_data(self) -> None:
        """Test efficiency data generation."""
        states = ["jpsi", "etac"]
        years = ["2016", "2017"]
        
        eff_data = generate_efficiency_data(states=states, years=years, seed=42)
        
        assert "jpsi" in eff_data
        assert "2016" in eff_data["jpsi"]
        assert "eff" in eff_data["jpsi"]["2016"]
        assert "err" in eff_data["jpsi"]["2016"]
        
        # Check values are in reasonable range
        for state in states:
            for year in years:
                eff = eff_data[state][year]["eff"]
                assert 0.0 < eff < 1.0
    
    def test_generate_yield_data(self) -> None:
        """Test yield data generation."""
        states = ["jpsi", "etac"]
        years = ["2016", "2017"]
        
        yield_data = generate_yield_data(states=states, years=years, seed=42)
        
        assert "2016" in yield_data
        assert "jpsi" in yield_data["2016"]
        
        # Check yield structure
        yield_val, error = yield_data["2016"]["jpsi"]
        assert yield_val > 0
        assert error > 0
        assert error < yield_val  # Error should be less than value


@pytest.mark.unit
def test_pytest_markers() -> None:
    """Verify pytest markers are configured correctly."""
    # This test itself uses the 'unit' marker
    # If markers aren't configured, pytest would warn
    assert True


@pytest.mark.unit
def test_imports() -> None:
    """Verify all test utilities can be imported."""
    from .utils import test_helpers
    from .utils import mock_data_generator
    
    assert hasattr(test_helpers, 'assert_arrays_close')
    assert hasattr(mock_data_generator, 'generate_mock_physics_data')
