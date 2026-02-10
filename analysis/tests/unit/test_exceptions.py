"""
Unit tests for custom exception classes.

Tests verify exception hierarchy, message formatting, and proper
initialization of all custom exception types.
"""

from __future__ import annotations

import pytest

from modules.exceptions import (
    AnalysisError,
    BranchMissingError,
    CacheError,
    ConfigurationError,
    DataLoadError,
    EfficiencyError,
    FittingError,
    OptimizationError,
    ValidationError,
)


@pytest.mark.unit
class TestExceptionHierarchy:
    """Test exception inheritance and hierarchy."""

    def test_all_inherit_from_analysis_error(self) -> None:
        """Verify all custom exceptions inherit from AnalysisError."""
        exceptions = [
            ConfigurationError,
            DataLoadError,
            BranchMissingError,
            OptimizationError,
            FittingError,
            EfficiencyError,
            ValidationError,
            CacheError,
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, AnalysisError)

    def test_analysis_error_inherits_from_exception(self) -> None:
        """Verify AnalysisError inherits from base Exception."""
        assert issubclass(AnalysisError, Exception)

    def test_catch_all_analysis_errors(self) -> None:
        """Test that AnalysisError catches all custom exceptions."""
        exceptions = [
            ConfigurationError("test"),
            DataLoadError("test"),
            OptimizationError("test"),
            FittingError("test"),
            EfficiencyError("test"),
            ValidationError("test"),
            CacheError("test"),
        ]

        for exc in exceptions:
            try:
                raise exc
            except AnalysisError:
                pass  # Successfully caught
            else:
                pytest.fail(f"{type(exc).__name__} not caught by AnalysisError")


@pytest.mark.unit
class TestConfigurationError:
    """Test ConfigurationError exception."""

    def test_raise_with_message(self) -> None:
        """Test raising ConfigurationError with message."""
        msg = "Missing required configuration"
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError(msg)

        assert msg in str(exc_info.value)

    def test_message_formatting(self) -> None:
        """Test multiline message formatting."""
        msg = "Error:\nLine 1\nLine 2"
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError(msg)

        assert "Line 1" in str(exc_info.value)
        assert "Line 2" in str(exc_info.value)


@pytest.mark.unit
class TestDataLoadError:
    """Test DataLoadError exception."""

    def test_raise_with_file_path(self) -> None:
        """Test DataLoadError with file path information."""
        msg = "Cannot load file: /path/to/data.root"
        with pytest.raises(DataLoadError) as exc_info:
            raise DataLoadError(msg)

        assert "/path/to/data.root" in str(exc_info.value)

    def test_empty_dataset_error(self) -> None:
        """Test error for empty dataset."""
        with pytest.raises(DataLoadError) as exc_info:
            raise DataLoadError("Empty dataset")

        assert "Empty dataset" in str(exc_info.value)


@pytest.mark.unit
class TestBranchMissingError:
    """Test BranchMissingError exception."""

    def test_basic_branch_missing(self) -> None:
        """Test basic branch missing error."""
        branch = "Bu_PT"
        exc = BranchMissingError(branch)

        assert branch in str(exc)
        assert exc.branch_name == branch
        assert exc.file_path is None

    def test_branch_missing_with_file_path(self) -> None:
        """Test branch missing error with file path."""
        branch = "L0_MM"
        file_path = "/data/test.root"
        exc = BranchMissingError(branch, file_path)

        assert branch in str(exc)
        assert file_path in str(exc)
        assert exc.branch_name == branch
        assert exc.file_path == file_path

    def test_raise_branch_missing_error(self) -> None:
        """Test raising BranchMissingError."""
        with pytest.raises(BranchMissingError) as exc_info:
            raise BranchMissingError("Bu_M", "/test/file.root")

        exc = exc_info.value
        assert exc.branch_name == "Bu_M"
        assert exc.file_path == "/test/file.root"


@pytest.mark.unit
class TestOptimizationError:
    """Test OptimizationError exception."""

    def test_no_valid_cuts_found(self) -> None:
        """Test error when no valid cuts found."""
        with pytest.raises(OptimizationError) as exc_info:
            raise OptimizationError("No valid cut combinations found")

        assert "No valid cut" in str(exc_info.value)

    def test_insufficient_statistics(self) -> None:
        """Test error for insufficient statistics."""
        with pytest.raises(OptimizationError) as exc_info:
            raise OptimizationError("Insufficient statistics for optimization")

        assert "Insufficient statistics" in str(exc_info.value)


@pytest.mark.unit
class TestFittingError:
    """Test FittingError exception."""

    def test_fit_not_converge(self) -> None:
        """Test error when fit doesn't converge."""
        with pytest.raises(FittingError) as exc_info:
            raise FittingError("Fit did not converge")

        assert "converge" in str(exc_info.value)

    def test_invalid_parameters(self) -> None:
        """Test error for invalid fit parameters."""
        with pytest.raises(FittingError) as exc_info:
            raise FittingError("Invalid fit parameters: negative width")

        assert "Invalid" in str(exc_info.value)
        assert "negative width" in str(exc_info.value)


@pytest.mark.unit
class TestEfficiencyError:
    """Test EfficiencyError exception."""

    def test_division_by_zero(self) -> None:
        """Test error for division by zero in efficiency."""
        with pytest.raises(EfficiencyError) as exc_info:
            raise EfficiencyError("Division by zero: no generated events")

        assert "Division by zero" in str(exc_info.value)

    def test_negative_efficiency(self) -> None:
        """Test error for negative efficiency."""
        with pytest.raises(EfficiencyError) as exc_info:
            raise EfficiencyError("Negative efficiency calculated")

        assert "Negative efficiency" in str(exc_info.value)


@pytest.mark.unit
class TestValidationError:
    """Test ValidationError exception."""

    def test_inconsistent_yields(self) -> None:
        """Test error for inconsistent yields."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Inconsistent yields across years")

        assert "Inconsistent yields" in str(exc_info.value)

    def test_unphysical_results(self) -> None:
        """Test error for unphysical results."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Unphysical result: negative yield")

        assert "Unphysical" in str(exc_info.value)


@pytest.mark.unit
class TestCacheError:
    """Test CacheError exception."""

    def test_cannot_write_cache(self) -> None:
        """Test error when cannot write cache."""
        with pytest.raises(CacheError) as exc_info:
            raise CacheError("Cannot write cache file: permission denied")

        assert "Cannot write cache" in str(exc_info.value)

    def test_corrupted_cache(self) -> None:
        """Test error for corrupted cache."""
        with pytest.raises(CacheError) as exc_info:
            raise CacheError("Corrupted cache file")

        assert "Corrupted cache" in str(exc_info.value)

    def test_version_mismatch(self) -> None:
        """Test error for cache version mismatch."""
        with pytest.raises(CacheError) as exc_info:
            raise CacheError("Cache version mismatch: expected 1.0, got 0.9")

        assert "version mismatch" in str(exc_info.value)


@pytest.mark.unit
def test_exception_string_representation() -> None:
    """Test string representation of exceptions."""
    exc = ConfigurationError("Test message")
    assert "Test message" in str(exc)
    assert "Test message" in repr(exc)
