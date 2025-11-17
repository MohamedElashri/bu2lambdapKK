"""
Helper functions for testing pipeline components.

Provides utilities for comparing data, validating results,
and common test operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def assert_arrays_close(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol: float = 1e-7,
    atol: float = 0.0,
    msg: str | None = None,
) -> None:
    """
    Assert that two numpy arrays are element-wise close.

    Args:
        actual: Actual array from test
        expected: Expected array
        rtol: Relative tolerance
        atol: Absolute tolerance
        msg: Optional error message

    Raises:
        AssertionError: If arrays are not close
    """
    error_msg = msg or f"Arrays not close:\nActual: {actual}\nExpected: {expected}"
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=error_msg)


def assert_dicts_equal(
    actual: dict[str, Any], expected: dict[str, Any], check_types: bool = True
) -> None:
    """
    Assert that two dictionaries are equal.

    Args:
        actual: Actual dictionary from test
        expected: Expected dictionary
        check_types: Whether to check value types

    Raises:
        AssertionError: If dictionaries are not equal
    """
    assert set(actual.keys()) == set(
        expected.keys()
    ), f"Keys differ:\nActual: {set(actual.keys())}\nExpected: {set(expected.keys())}"

    for key in expected:
        if isinstance(expected[key], np.ndarray):
            assert_arrays_close(actual[key], expected[key])
        elif isinstance(expected[key], dict):
            assert_dicts_equal(actual[key], expected[key], check_types=check_types)
        else:
            assert (
                actual[key] == expected[key]
            ), f"Value mismatch for key '{key}':\nActual: {actual[key]}\nExpected: {expected[key]}"

            if check_types:
                assert type(actual[key]) == type(
                    expected[key]
                ), f"Type mismatch for key '{key}':\nActual: {type(actual[key])}\nExpected: {type(expected[key])}"


def assert_dataframes_equal(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    check_dtype: bool = True,
    check_column_order: bool = False,
) -> None:
    """
    Assert that two pandas DataFrames are equal.

    Args:
        actual: Actual DataFrame from test
        expected: Expected DataFrame
        check_dtype: Whether to check data types
        check_column_order: Whether to check column order

    Raises:
        AssertionError: If DataFrames are not equal
    """
    pd.testing.assert_frame_equal(
        actual, expected, check_dtype=check_dtype, check_like=not check_column_order
    )


def assert_file_exists(file_path: str | Path) -> None:
    """
    Assert that a file exists.

    Args:
        file_path: Path to file

    Raises:
        AssertionError: If file does not exist
    """
    path = Path(file_path)
    assert path.exists(), f"File does not exist: {path}"
    assert path.is_file(), f"Path is not a file: {path}"


def assert_dir_exists(dir_path: str | Path) -> None:
    """
    Assert that a directory exists.

    Args:
        dir_path: Path to directory

    Raises:
        AssertionError: If directory does not exist
    """
    path = Path(dir_path)
    assert path.exists(), f"Directory does not exist: {path}"
    assert path.is_dir(), f"Path is not a directory: {path}"


def assert_raises_with_message(
    exception_type: type, message_substring: str, callable_obj: callable, *args: Any, **kwargs: Any
) -> None:
    """
    Assert that a callable raises an exception with a specific message.

    Args:
        exception_type: Expected exception type
        message_substring: Substring expected in exception message
        callable_obj: Callable to execute
        *args: Positional arguments for callable
        **kwargs: Keyword arguments for callable

    Raises:
        AssertionError: If exception not raised or message doesn't match
    """
    try:
        callable_obj(*args, **kwargs)
        raise AssertionError(f"Expected {exception_type.__name__} was not raised")
    except exception_type as e:
        assert message_substring in str(
            e
        ), f"Expected substring '{message_substring}' not found in error message: {str(e)}"


def count_files_in_dir(dir_path: str | Path, pattern: str = "*") -> int:
    """
    Count files matching a pattern in a directory.

    Args:
        dir_path: Path to directory
        pattern: Glob pattern for matching files

    Returns:
        Number of matching files
    """
    path = Path(dir_path)
    return len(list(path.glob(pattern)))


def get_file_size(file_path: str | Path) -> int:
    """
    Get file size in bytes.

    Args:
        file_path: Path to file

    Returns:
        File size in bytes
    """
    return Path(file_path).stat().st_size


def assert_value_in_range(
    value: float, min_val: float, max_val: float, inclusive: bool = True
) -> None:
    """
    Assert that a value is within a specified range.

    Args:
        value: Value to check
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        inclusive: Whether range is inclusive

    Raises:
        AssertionError: If value is out of range
    """
    if inclusive:
        assert min_val <= value <= max_val, f"Value {value} not in range [{min_val}, {max_val}]"
    else:
        assert min_val < value < max_val, f"Value {value} not in range ({min_val}, {max_val})"
