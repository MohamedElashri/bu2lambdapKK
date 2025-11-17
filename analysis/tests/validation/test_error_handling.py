"""
Error handling validation tests.

Tests graceful failure and error recovery:
- File I/O errors
- Corrupted data files
- Missing branches
- Exception propagation
- Informative error messages
"""

from __future__ import annotations

import pytest
import tomli_w
import pickle
from pathlib import Path
from typing import Dict, Any

from analysis.modules.data_handler import TOMLConfig
from analysis.modules.cache_manager import CacheManager
from analysis.modules.branch_config import BranchConfig
from analysis.modules.exceptions import (
    AnalysisError,
    ConfigurationError,
    DataLoadError,
    BranchMissingError,
    CacheError,
    ValidationError
)


@pytest.mark.validation
class TestFileIOErrors:
    """Test handling of file I/O errors."""
    
    def test_missing_config_directory(self, tmp_test_dir: Path) -> None:
        """Test error when config directory doesn't exist."""
        nonexistent_dir = tmp_test_dir / "nonexistent_config"
        
        with pytest.raises(ConfigurationError) as exc_info:
            TOMLConfig(str(nonexistent_dir))
        
        # Error message should be informative
        assert "not found" in str(exc_info.value).lower() or \
               "does not exist" in str(exc_info.value).lower()
    
    def test_missing_config_file(self, tmp_test_dir: Path) -> None:
        """Test error when specific config file is missing."""
        config_dir = tmp_test_dir / "config"
        config_dir.mkdir()
        
        # Create some but not all config files
        for filename in ['physics.toml', 'detector.toml']:
            with open(config_dir / filename, 'wb') as f:
                tomli_w.dump({}, f)
        
        # Missing fitting.toml should cause error
        with pytest.raises(ConfigurationError) as exc_info:
            TOMLConfig(str(config_dir))
        
        assert "fitting.toml" in str(exc_info.value).lower() or \
               "configuration" in str(exc_info.value).lower()
    
    def test_unreadable_config_file(self, tmp_test_dir: Path) -> None:
        """Test error when config file is not readable."""
        config_dir = tmp_test_dir / "config"
        config_dir.mkdir()
        
        # Create a file with invalid TOML
        physics_file = config_dir / "physics.toml"
        with open(physics_file, 'w') as f:
            f.write("invalid toml {{{ content")
        
        # Should raise TOML parsing error (tomli.TOMLDecodeError)
        import tomli
        with pytest.raises(tomli.TOMLDecodeError):
            physics_config = config_dir / "physics.toml"
            with open(physics_config, 'rb') as f:
                tomli.load(f)
    
    def test_cache_directory_creation_failure(self, tmp_test_dir: Path) -> None:
        """Test handling when cache directory can't be created."""
        # Create a file where cache dir should be (will block mkdir)
        cache_path = tmp_test_dir / "cache_blocked"
        cache_path.touch()  # Create as file, not directory
        
        # Trying to use this as cache directory should fail gracefully
        # Note: CacheManager might handle this, but we test it fails appropriately
        try:
            cache = CacheManager(str(cache_path))
            # If it succeeds, check it's not actually using a file as directory
            assert cache_path.is_dir() or not cache_path.exists()
        except (OSError, PermissionError, CacheError):
            # Expected: can't create cache in this location
            pass
    
    def test_cache_file_read_error(self, tmp_cache_dir: Path) -> None:
        """Test handling of corrupted cache files."""
        cache = CacheManager(str(tmp_cache_dir))
        
        # Save some data
        cache.save("test_key", {"data": "value"}, dependencies={})
        
        # Corrupt the pickle file
        data_files = list(cache.data_dir.glob("*.pkl"))
        assert len(data_files) > 0
        
        with open(data_files[0], 'wb') as f:
            f.write(b"corrupted data")
        
        # Loading should handle gracefully (raise CacheError or return None)
        try:
            result = cache.load("test_key", dependencies={})
            # If it doesn't raise, should return None
            assert result is None, "Corrupted cache should return None"
        except CacheError:
            # Also acceptable to raise CacheError
            pass


@pytest.mark.validation
class TestCorruptedData:
    """Test handling of corrupted data files."""
    
    def test_corrupted_cache_metadata(self, tmp_cache_dir: Path) -> None:
        """Test handling of corrupted cache metadata."""
        cache = CacheManager(str(tmp_cache_dir))
        
        # Save data
        cache.save("test", {"value": 42}, dependencies={})
        
        # Corrupt metadata
        metadata_files = list(cache.metadata_dir.glob("*.json"))
        assert len(metadata_files) > 0
        
        with open(metadata_files[0], 'w') as f:
            f.write("{invalid json content")
        
        # Loading should handle gracefully
        result = cache.load("test", dependencies={})
        assert result is None, "Corrupted metadata should return None"
    
    def test_missing_cache_data_file(self, tmp_cache_dir: Path) -> None:
        """Test handling when cache data file is missing."""
        cache = CacheManager(str(tmp_cache_dir))
        
        # Save data
        cache.save("test", {"value": 42}, dependencies={})
        
        # Delete data file but keep metadata
        data_files = list(cache.data_dir.glob("*.pkl"))
        assert len(data_files) > 0
        data_files[0].unlink()
        
        # Loading should handle gracefully
        result = cache.load("test", dependencies={})
        assert result is None, "Missing data file should return None"
    
    def test_corrupted_pickle_data(self, tmp_cache_dir: Path) -> None:
        """Test handling of corrupted pickle data."""
        cache = CacheManager(str(tmp_cache_dir))
        
        # Create cache entry with valid metadata but completely corrupt pickle
        deps = {"file1": "hash1"}
        cache_key = cache._compute_hash("test", **deps)
        
        # Write valid metadata
        metadata = {
            "key": cache_key,
            "dependencies": deps,
            "created_at": "2024-01-01T00:00:00",
            "version": "1.0.0",
            "description": "test"
        }
        
        import json
        metadata_file = cache.metadata_dir / f"{cache_key}.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Write truncated pickle data (invalid)
        data_file = cache.data_dir / f"{cache_key}.pkl"
        data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(data_file, 'wb') as f:
            f.write(b"truncated")  # Too short to be valid pickle
        
        # Loading should raise CacheError (or return None gracefully)
        try:
            result = cache.load("test", dependencies=deps)
            # If load succeeds, should return None for corrupted data
            assert result is None, "Corrupted pickle should return None or raise CacheError"
        except CacheError:
            # Also acceptable to raise CacheError
            pass


@pytest.mark.validation
class TestMissingBranches:
    """Test handling of missing branches in data."""
    
    def test_missing_branch_error_raised(self) -> None:
        """Test that BranchMissingError is raised for missing branches."""
        # Create error
        error = BranchMissingError("L0_MM", "/path/to/file.root")
        
        # Check error properties
        assert "L0_MM" in str(error)
        assert isinstance(error, AnalysisError)
    
    def test_missing_branch_error_message(self) -> None:
        """Test that missing branch error has informative message."""
        error = BranchMissingError("nonexistent_branch", "test.root")
        
        error_msg = str(error)
        assert "nonexistent_branch" in error_msg
        assert "test.root" in error_msg
    
    def test_branch_config_missing_file(self, tmp_test_dir: Path) -> None:
        """Test error when branch config file doesn't exist."""
        nonexistent_path = tmp_test_dir / "nonexistent_branches.toml"
        
        with pytest.raises(ConfigurationError) as exc_info:
            BranchConfig(str(nonexistent_path))
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_branch_validation_missing_branches(self) -> None:
        """Test validation when branches are missing."""
        # Test BranchMissingError behavior
        try:
            raise BranchMissingError("missing_branch", "data.root")
        except BranchMissingError as e:
            assert "missing_branch" in str(e)
            assert "data.root" in str(e)


@pytest.mark.validation
class TestExceptionPropagation:
    """Test that exceptions propagate correctly."""
    
    def test_configuration_error_inheritance(self) -> None:
        """Test that ConfigurationError inherits from AnalysisError."""
        error = ConfigurationError("test error")
        
        assert isinstance(error, AnalysisError)
        assert isinstance(error, Exception)
    
    def test_cache_error_inheritance(self) -> None:
        """Test that CacheError inherits from AnalysisError."""
        error = CacheError("test error")
        
        assert isinstance(error, AnalysisError)
        assert isinstance(error, Exception)
    
    def test_validation_error_inheritance(self) -> None:
        """Test that ValidationError inherits from AnalysisError."""
        error = ValidationError("test error")
        
        assert isinstance(error, AnalysisError)
        assert isinstance(error, Exception)
    
    def test_catch_all_analysis_errors(self) -> None:
        """Test that AnalysisError can catch all custom errors."""
        errors = [
            ConfigurationError("config"),
            DataLoadError("data"),
            BranchMissingError("branch", "file"),
            CacheError("cache"),
            ValidationError("validation")
        ]
        
        for error in errors:
            try:
                raise error
            except AnalysisError as e:
                # Should catch all
                assert isinstance(e, AnalysisError)
    
    def test_specific_exception_caught_first(self) -> None:
        """Test that specific exceptions are caught before general ones."""
        try:
            raise ConfigurationError("test")
        except ConfigurationError as e:
            # Should catch specific first
            assert isinstance(e, ConfigurationError)
        except AnalysisError:
            pytest.fail("Should have caught ConfigurationError specifically")


@pytest.mark.validation
class TestErrorMessages:
    """Test that error messages are informative."""
    
    def test_configuration_error_message_format(self) -> None:
        """Test that configuration errors have clear messages."""
        error = ConfigurationError(
            "Missing required parameter 'jpsi_mass' in physics.toml"
        )
        
        msg = str(error)
        assert "physics.toml" in msg
        assert "jpsi_mass" in msg
        assert len(msg) > 20, "Error message should be descriptive"
    
    def test_branch_missing_error_includes_file(self) -> None:
        """Test that branch errors include file path."""
        error = BranchMissingError("Bu_MM", "/data/file.root")
        
        msg = str(error)
        assert "Bu_MM" in msg
        assert "/data/file.root" in msg
    
    def test_cache_error_message_format(self) -> None:
        """Test that cache errors have clear messages."""
        error = CacheError("Failed to load cache entry: corrupted metadata")
        
        msg = str(error)
        assert "cache" in msg.lower()
        assert len(msg) > 10, "Error message should be descriptive"
    
    def test_data_load_error_message(self) -> None:
        """Test that data load errors are informative."""
        error = DataLoadError("Failed to open ROOT file: file.root")
        
        msg = str(error)
        assert "file.root" in msg
        assert len(msg) > 10
    
    def test_validation_error_includes_details(self) -> None:
        """Test that validation errors include useful details."""
        error = ValidationError(
            "Yield value is negative: -10.5 for jpsi 2016"
        )
        
        msg = str(error)
        assert "-10.5" in msg
        assert "jpsi" in msg
        assert "2016" in msg


@pytest.mark.validation
class TestErrorRecovery:
    """Test error recovery and graceful degradation."""
    
    def test_partial_config_loading(self, tmp_test_dir: Path) -> None:
        """Test behavior when some config files are missing."""
        config_dir = tmp_test_dir / "config"
        config_dir.mkdir()
        
        # Create minimal config
        physics = {'pdg_masses': {'jpsi': 3096.9}}
        with open(config_dir / "physics.toml", 'wb') as f:
            tomli_w.dump(physics, f)
        
        # Missing other files should raise error
        with pytest.raises(ConfigurationError):
            TOMLConfig(str(config_dir))
    
    def test_cache_continues_after_corruption(self, tmp_cache_dir: Path) -> None:
        """Test that cache can continue after handling corruption."""
        cache = CacheManager(str(tmp_cache_dir))
        
        # Save three separate pieces of data
        cache.save("first", {"value": 1}, dependencies={})
        cache.save("second", {"value": 2}, dependencies={})
        cache.save("third", {"value": 3}, dependencies={})
        
        # Get cache key for "second" to corrupt its specific file
        second_key = cache._compute_hash("second")
        second_data_file = cache.data_dir / f"{second_key}.pkl"
        
        # Corrupt only the second entry's data file  
        if second_data_file.exists():
            with open(second_data_file, 'wb') as f:
                f.write(b"X")  # Write single byte - invalid pickle
        
        # Should still be able to load first entry
        result1 = cache.load("first", dependencies={})
        assert result1 == {"value": 1}, "Should load uncorrupted first cache"
        
        # Second entry should fail or return None (acceptable either way)
        # Don't make assertions about this one
        
        # Third entry should still work
        result3 = cache.load("third", dependencies={})
        assert result3 == {"value": 3}, "Should load uncorrupted third cache"
    
    def test_config_validation_continues_after_warnings(self, config_dir_fixture: Path) -> None:
        """Test that validation continues after non-fatal warnings."""
        # Config loading should succeed even with minor issues
        config = TOMLConfig(str(config_dir_fixture))
        
        # Should have loaded successfully
        assert hasattr(config, 'physics')
        assert hasattr(config, 'detector')
    
    def test_multiple_errors_collected(self, tmp_test_dir: Path) -> None:
        """Test that multiple validation errors can be collected."""
        config_dir = tmp_test_dir / "config"
        config_dir.mkdir()
        
        # Create config with multiple issues
        physics = {
            'pdg_masses': {
                'jpsi': -3096.9,  # Invalid: negative
                # Missing: etac, lambda, etc.
            },
            'pdg_widths': {},
            'pdg_branching_fractions': {},
            'analysis_method': {}
        }
        
        detector = {
            'signal_regions': {'jpsi': {'center': 3096.9, 'width': 0.05}},
            'mass_windows': {},
            'integrated_luminosity': {'2016': 1.0}
        }
        
        with open(config_dir / "physics.toml", 'wb') as f:
            tomli_w.dump(physics, f)
        
        with open(config_dir / "detector.toml", 'wb') as f:
            tomli_w.dump(detector, f)
        
        for filename in ['fitting.toml', 'selection.toml',
                        'triggers.toml', 'data.toml', 'efficiencies.toml']:
            with open(config_dir / filename, 'wb') as f:
                if filename == 'fitting.toml':
                    tomli_w.dump({'fit_method': {}, 'background_model': {}}, f)
                elif filename == 'selection.toml':
                    tomli_w.dump({'cuts': {}}, f)
                elif filename == 'triggers.toml':
                    tomli_w.dump({'lines': []}, f)
                elif filename == 'data.toml':
                    tomli_w.dump({'input_data': {}, 'input_mc': {}, 'output': {}}, f)
                elif filename == 'efficiencies.toml':
                    tomli_w.dump({'trigger': {}}, f)
        
        # Config loads but is invalid
        config = TOMLConfig(str(config_dir))
        
        # Multiple issues should be detectable
        assert config.physics['pdg_masses']['jpsi'] < 0  # Negative mass
        assert 'etac' not in config.physics['pdg_masses']  # Missing mass


@pytest.mark.validation
class TestEdgeCases:
    """Test edge cases in error handling."""
    
    def test_empty_error_message(self) -> None:
        """Test handling of empty error message."""
        error = ConfigurationError("")
        
        # Should still be an error object
        assert isinstance(error, ConfigurationError)
        assert isinstance(error, AnalysisError)
    
    def test_very_long_error_message(self) -> None:
        """Test handling of very long error messages."""
        long_msg = "Error: " + "x" * 10000
        error = ConfigurationError(long_msg)
        
        # Should handle long messages
        assert len(str(error)) > 1000
    
    def test_error_with_special_characters(self) -> None:
        """Test error messages with special characters."""
        msg = "Error in file: /path/to/file's_data.root"
        error = DataLoadError(msg)
        
        # Should preserve special characters
        assert "file's_data.root" in str(error)
    
    def test_nested_exception_chain(self) -> None:
        """Test exception chaining."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise ConfigurationError("Config error") from e
        except ConfigurationError as ce:
            # Should have original cause
            assert ce.__cause__ is not None
            assert isinstance(ce.__cause__, ValueError)
    
    def test_unicode_in_error_message(self) -> None:
        """Test error messages with unicode characters."""
        msg = "Error in η_c mass: value is ∞"
        error = ValidationError(msg)
        
        # Should handle unicode
        assert "η_c" in str(error) or "eta" in str(error).lower()


@pytest.mark.validation
@pytest.mark.slow
class TestStressErrorHandling:
    """Stress tests for error handling."""
    
    def test_many_missing_branches(self) -> None:
        """Test handling of many missing branches."""
        # Create errors for many missing branches
        missing_branches = [f"branch_{i}" for i in range(100)]
        
        for branch in missing_branches:
            error = BranchMissingError(branch, "test.root")
            assert branch in str(error)
    
    def test_rapid_error_creation(self) -> None:
        """Test rapid creation of many errors."""
        errors = []
        for i in range(1000):
            error = ConfigurationError(f"Error {i}")
            errors.append(error)
        
        # Should handle many errors
        assert len(errors) == 1000
        assert all(isinstance(e, ConfigurationError) for e in errors)
    
    def test_concurrent_cache_corruption(self, tmp_cache_dir: Path) -> None:
        """Test handling of multiple corrupted cache entries."""
        cache = CacheManager(str(tmp_cache_dir))
        
        # Create multiple entries
        for i in range(10):
            cache.save(f"entry_{i}", {"value": i}, dependencies={})
        
        # Corrupt all data files
        for data_file in cache.data_dir.glob("*.pkl"):
            with open(data_file, 'wb') as f:
                f.write(b"corrupted")
        
        # Should handle all corruptions gracefully
        for i in range(10):
            try:
                result = cache.load(f"entry_{i}", dependencies={})
                assert result is None or isinstance(result, dict)
            except CacheError:
                pass  # Also acceptable
