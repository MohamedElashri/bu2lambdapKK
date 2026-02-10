"""
Test utilities and helper functions.

Provides common functionality for test setup, data generation,
and result validation across the test suite.
"""

from .mock_data_generator import (
    create_mock_config_toml,
    create_mock_mc_data,
    create_mock_root_file,
    generate_efficiency_data,
    generate_mock_physics_data,
    generate_signal_background_mix,
    generate_yield_data,
)
from .test_helpers import (
    assert_arrays_close,
    assert_dataframes_equal,
    assert_dicts_equal,
    assert_dir_exists,
    assert_file_exists,
    assert_raises_with_message,
    assert_value_in_range,
    count_files_in_dir,
    get_file_size,
)

__all__ = [
    "assert_arrays_close",
    "assert_dicts_equal",
    "create_mock_root_file",
    "generate_mock_physics_data",
    "assert_dataframes_equal",
    "assert_dir_exists",
    "assert_file_exists",
    "assert_raises_with_message",
    "assert_value_in_range",
    "count_files_in_dir",
    "get_file_size",
    "create_mock_config_toml",
    "create_mock_mc_data",
    "generate_efficiency_data",
    "generate_signal_background_mix",
    "generate_yield_data",
]
