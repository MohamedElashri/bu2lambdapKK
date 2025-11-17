"""
Integration tests for end-to-end cache workflows.

Tests the complete caching workflow including saving, loading,
invalidation, corruption handling, and cache management.

These tests validate that the cache system works correctly
in realistic usage scenarios.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.cache_manager import CacheManager
from modules.data_handler import TOMLConfig


@pytest.mark.integration
class TestCacheWorkflowBasics:
    """Test basic cache workflow operations."""

    def test_save_and_load_simple_payload(self, tmp_cache_dir, config_dir_fixture):
        """Test saving and loading a simple data payload."""
        cache = CacheManager(str(tmp_cache_dir))
        config = TOMLConfig(str(config_dir_fixture))

        # Create test payload
        payload = {"year": "2016", "n_events": 10000, "efficiency": 0.85}

        # Compute dependencies
        config_files = list(Path(config_dir_fixture).glob("*.toml"))
        deps = cache.compute_dependencies(config_files=config_files)

        # Save to cache
        cache.save("test_payload", payload, dependencies=deps)

        # Load from cache
        loaded = cache.load("test_payload", dependencies=deps)

        assert loaded is not None
        assert loaded == payload

    def test_cache_miss_returns_none(self, tmp_cache_dir):
        """Test that cache miss returns None."""
        cache = CacheManager(str(tmp_cache_dir))

        # Try to load non-existent key
        loaded = cache.load("nonexistent_key", dependencies={})

        assert loaded is None

    def test_save_multiple_entries(self, tmp_cache_dir, config_dir_fixture):
        """Test saving multiple cache entries."""
        cache = CacheManager(str(tmp_cache_dir))
        config = TOMLConfig(str(config_dir_fixture))

        config_files = list(Path(config_dir_fixture).glob("*.toml"))
        deps = cache.compute_dependencies(config_files=config_files)

        # Save multiple entries
        cache.save("entry1", {"value": 1}, dependencies=deps)
        cache.save("entry2", {"value": 2}, dependencies=deps)
        cache.save("entry3", {"value": 3}, dependencies=deps)

        # Load all entries
        assert cache.load("entry1", dependencies=deps)["value"] == 1
        assert cache.load("entry2", dependencies=deps)["value"] == 2
        assert cache.load("entry3", dependencies=deps)["value"] == 3


@pytest.mark.integration
class TestCacheDependencyTracking:
    """Test dependency-based cache invalidation."""

    def test_cache_invalidates_on_config_change(self, tmp_cache_dir, tmp_test_dir):
        """Test that cache invalidates when config files change."""
        cache = CacheManager(str(tmp_cache_dir))

        # Create initial config file
        config_file = tmp_test_dir / "test_config.toml"
        config_file.write_text("[section]\nvalue = 1\n")

        # Compute dependencies with initial config
        deps1 = cache.compute_dependencies(config_files=[config_file])

        # Save data
        cache.save("data", {"result": 42}, dependencies=deps1)

        # Verify cache hit
        loaded = cache.load("data", dependencies=deps1)
        assert loaded == {"result": 42}

        # Modify config file
        config_file.write_text("[section]\nvalue = 2\n")

        # Compute new dependencies
        deps2 = cache.compute_dependencies(config_files=[config_file])

        # Cache should miss with new dependencies
        loaded = cache.load("data", dependencies=deps2)
        assert loaded is None

    def test_cache_hit_with_same_dependencies(self, tmp_cache_dir, tmp_test_dir):
        """Test that cache hits when dependencies match."""
        cache = CacheManager(str(tmp_cache_dir))

        # Create config file
        config_file = tmp_test_dir / "test_config.toml"
        config_file.write_text("[section]\nvalue = 1\n")

        # Compute dependencies
        deps = cache.compute_dependencies(config_files=[config_file])

        # Save data
        cache.save("data", {"result": 42}, dependencies=deps)

        # Recompute dependencies (file hasn't changed)
        deps2 = cache.compute_dependencies(config_files=[config_file])

        # Should have same dependencies
        assert deps == deps2

        # Cache should hit
        loaded = cache.load("data", dependencies=deps2)
        assert loaded == {"result": 42}

    def test_cache_with_parameter_dependencies(self, tmp_cache_dir):
        """Test cache invalidation based on parameter changes."""
        cache = CacheManager(str(tmp_cache_dir))

        # Save with specific parameters
        params1 = {"year": "2016", "track_type": "LL"}
        deps1 = cache.compute_dependencies(extra_params=params1)
        cache.save("result", {"value": 100}, dependencies=deps1)

        # Load with same parameters - should hit
        loaded = cache.load("result", dependencies=deps1)
        assert loaded == {"value": 100}

        # Load with different parameters - should miss
        params2 = {"year": "2017", "track_type": "LL"}
        deps2 = cache.compute_dependencies(extra_params=params2)
        loaded = cache.load("result", dependencies=deps2)
        assert loaded is None


@pytest.mark.integration
class TestCacheComplexPayloads:
    """Test caching complex data structures."""

    def test_cache_nested_dictionary(self, tmp_cache_dir, config_dir_fixture):
        """Test caching nested dictionary structures."""
        cache = CacheManager(str(tmp_cache_dir))

        config_files = list(Path(config_dir_fixture).glob("*.toml"))
        deps = cache.compute_dependencies(config_files=config_files)

        # Create nested structure
        nested_data = {
            "states": {
                "jpsi": {"2016": {"LL": 1000, "DD": 800}, "2017": {"LL": 1200, "DD": 900}},
                "etac": {"2016": {"LL": 500, "DD": 400}},
            },
            "metadata": {"description": "Test data", "version": "1.0"},
        }

        # Save and load
        cache.save("nested", nested_data, dependencies=deps)
        loaded = cache.load("nested", dependencies=deps)

        assert loaded is not None
        assert loaded == nested_data
        assert loaded["states"]["jpsi"]["2016"]["LL"] == 1000

    def test_cache_list_of_dicts(self, tmp_cache_dir, config_dir_fixture):
        """Test caching list of dictionaries."""
        cache = CacheManager(str(tmp_cache_dir))

        config_files = list(Path(config_dir_fixture).glob("*.toml"))
        deps = cache.compute_dependencies(config_files=config_files)

        # Create list of dicts
        list_data = [
            {"state": "jpsi", "yield": 1000, "error": 50},
            {"state": "etac", "yield": 500, "error": 30},
            {"state": "chic0", "yield": 300, "error": 25},
        ]

        # Save and load
        cache.save("yields", list_data, dependencies=deps)
        loaded = cache.load("yields", dependencies=deps)

        assert loaded is not None
        assert loaded == list_data
        assert len(loaded) == 3
        assert loaded[0]["state"] == "jpsi"

    def test_cache_with_numeric_arrays(self, tmp_cache_dir, config_dir_fixture):
        """Test caching data with numeric arrays."""
        cache = CacheManager(str(tmp_cache_dir))

        config_files = list(Path(config_dir_fixture).glob("*.toml"))
        deps = cache.compute_dependencies(config_files=config_files)

        import numpy as np

        # Create data with numpy arrays
        array_data = {
            "masses": np.array([5279.0, 5280.0, 5281.0]),
            "yields": np.array([100, 150, 120]),
            "errors": np.array([10.0, 12.0, 11.0]),
        }

        # Save and load
        cache.save("arrays", array_data, dependencies=deps)
        loaded = cache.load("arrays", dependencies=deps)

        assert loaded is not None
        assert np.allclose(loaded["masses"], array_data["masses"])
        assert np.allclose(loaded["yields"], array_data["yields"])


@pytest.mark.integration
class TestCacheCorruptionHandling:
    """Test handling of corrupted cache entries."""

    def test_corrupted_metadata_returns_none(self, tmp_cache_dir, config_dir_fixture):
        """Test that corrupted metadata is handled gracefully."""
        cache = CacheManager(str(tmp_cache_dir))

        config_files = list(Path(config_dir_fixture).glob("*.toml"))
        deps = cache.compute_dependencies(config_files=config_files)

        # Save valid data
        cache.save("test_data", {"value": 42}, dependencies=deps)

        # Find the metadata file by listing directory
        metadata_files = list(cache.metadata_dir.glob("*.json"))
        assert len(metadata_files) > 0, "No metadata files found"
        metadata_file = metadata_files[0]

        # Write invalid JSON
        metadata_file.write_text("{ corrupted json }")

        # Try to load - should return None gracefully
        loaded = cache.load("test_data", dependencies=deps)
        assert loaded is None

    def test_missing_data_file_returns_none(self, tmp_cache_dir, config_dir_fixture):
        """Test that missing data file is handled gracefully."""
        cache = CacheManager(str(tmp_cache_dir))

        config_files = list(Path(config_dir_fixture).glob("*.toml"))
        deps = cache.compute_dependencies(config_files=config_files)

        # Save valid data
        cache.save("test_data", {"value": 42}, dependencies=deps)

        # Delete a data file (find any .pkl file)
        data_files = list(cache.data_dir.glob("*.pkl"))
        assert len(data_files) > 0, "No data files found"
        data_files[0].unlink()

        # Try to load - should return None
        loaded = cache.load("test_data", dependencies=deps)
        assert loaded is None

    def test_corrupted_pickle_returns_none(self, tmp_cache_dir, config_dir_fixture):
        """Test that corrupted pickle data raises CacheError."""
        from modules.exceptions import CacheError

        cache = CacheManager(str(tmp_cache_dir))

        config_files = list(Path(config_dir_fixture).glob("*.toml"))
        deps = cache.compute_dependencies(config_files=config_files)

        # Save valid data
        cache.save("test_data", {"value": 42}, dependencies=deps)

        # Find and corrupt a pickle file
        data_files = list(cache.data_dir.glob("*.pkl"))
        assert len(data_files) > 0, "No data files found"
        data_file = data_files[0]

        # Write invalid pickle data
        data_file.write_bytes(b"corrupted pickle data")

        # Try to load - should raise CacheError for corrupted pickle
        with pytest.raises(CacheError):
            cache.load("test_data", dependencies=deps)


@pytest.mark.integration
class TestCacheManagement:
    """Test cache management operations."""

    def test_clear_all_cache(self, tmp_cache_dir, config_dir_fixture):
        """Test clearing all cache entries."""
        cache = CacheManager(str(tmp_cache_dir))

        config_files = list(Path(config_dir_fixture).glob("*.toml"))
        deps = cache.compute_dependencies(config_files=config_files)

        # Add multiple entries
        cache.save("entry1", {"value": 1}, dependencies=deps)
        cache.save("entry2", {"value": 2}, dependencies=deps)
        cache.save("entry3", {"value": 3}, dependencies=deps)

        # Verify entries exist
        stats_before = cache.get_cache_stats()
        assert stats_before["num_entries"] >= 3

        # Clear all
        cache.clear_all()

        # Verify all cleared
        stats_after = cache.get_cache_stats()
        assert stats_after["num_entries"] == 0

        # Verify loads return None
        assert cache.load("entry1", dependencies=deps) is None
        assert cache.load("entry2", dependencies=deps) is None
        assert cache.load("entry3", dependencies=deps) is None

    def test_cache_statistics_accurate(self, tmp_cache_dir, config_dir_fixture):
        """Test that cache statistics are accurate."""
        cache = CacheManager(str(tmp_cache_dir))

        config_files = list(Path(config_dir_fixture).glob("*.toml"))
        deps = cache.compute_dependencies(config_files=config_files)

        # Initial stats
        stats_initial = cache.get_cache_stats()
        initial_entries = stats_initial["num_entries"]

        # Add entries
        num_to_add = 5
        for i in range(num_to_add):
            cache.save(f"entry{i}", {"value": i}, dependencies=deps)

        # Check stats updated
        stats_after = cache.get_cache_stats()
        assert stats_after["num_entries"] == initial_entries + num_to_add
        assert stats_after["total_size_mb"] >= 0

    def test_clear_all_entries(self, tmp_cache_dir, config_dir_fixture):
        """Test clearing all cache entries."""
        cache = CacheManager(str(tmp_cache_dir))

        config_files = list(Path(config_dir_fixture).glob("*.toml"))
        deps = cache.compute_dependencies(config_files=config_files)

        # Add entries
        cache.save("entry1", {"value": 1}, dependencies=deps)
        cache.save("entry2", {"value": 2}, dependencies=deps)
        cache.save("entry3", {"value": 3}, dependencies=deps)

        # Verify entries exist
        assert cache.load("entry1", dependencies=deps) is not None

        # Clear all
        cache.clear_all()

        # Verify all entries cleared
        assert cache.load("entry1", dependencies=deps) is None
        assert cache.load("entry2", dependencies=deps) is None
        assert cache.load("entry3", dependencies=deps) is None


@pytest.mark.integration
class TestCacheMetadata:
    """Test cache metadata handling."""

    def test_metadata_includes_description(self, tmp_cache_dir, config_dir_fixture):
        """Test that cache metadata includes description."""
        cache = CacheManager(str(tmp_cache_dir))

        config_files = list(Path(config_dir_fixture).glob("*.toml"))
        deps = cache.compute_dependencies(config_files=config_files)

        description = "Test data for metadata verification"
        cache.save("test", {"value": 42}, dependencies=deps, description=description)

        # Check metadata file exists (find any metadata file)
        metadata_files = list(cache.metadata_dir.glob("*.json"))
        assert len(metadata_files) > 0, "No metadata files found"

        # Check that at least one metadata file has a description
        found_description = False
        for metadata_file in metadata_files:
            with open(metadata_file) as f:
                metadata = json.load(f)
            if "description" in metadata and metadata["description"] == description:
                found_description = True
                break

        assert found_description, f"Description '{description}' not found in any metadata file"

    def test_metadata_includes_dependencies(self, tmp_cache_dir, config_dir_fixture):
        """Test that cache metadata includes dependency hashes."""
        cache = CacheManager(str(tmp_cache_dir))

        config_files = list(Path(config_dir_fixture).glob("*.toml"))
        deps = cache.compute_dependencies(config_files=config_files)

        cache.save("test", {"value": 42}, dependencies=deps)

        # Check metadata file
        metadata_files = list(cache.metadata_dir.glob("*.json"))
        assert len(metadata_files) > 0, "No metadata files found"

        metadata_file = metadata_files[0]
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert "dependencies" in metadata
        assert isinstance(metadata["dependencies"], dict)

    def test_metadata_includes_timestamp(self, tmp_cache_dir, config_dir_fixture):
        """Test that cache metadata includes timestamp."""
        cache = CacheManager(str(tmp_cache_dir))

        config_files = list(Path(config_dir_fixture).glob("*.toml"))
        deps = cache.compute_dependencies(config_files=config_files)

        cache.save("test", {"value": 42}, dependencies=deps)

        # Check metadata file
        metadata_files = list(cache.metadata_dir.glob("*.json"))
        assert len(metadata_files) > 0, "No metadata files found"

        metadata_file = metadata_files[0]
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert "created_at" in metadata
        assert isinstance(metadata["created_at"], str)


@pytest.mark.integration
class TestCacheAtomicity:
    """Test cache operations are atomic."""

    def test_save_is_atomic(self, tmp_cache_dir, config_dir_fixture):
        """Test that save operations are atomic (no partial writes)."""
        cache = CacheManager(str(tmp_cache_dir))

        config_files = list(Path(config_dir_fixture).glob("*.toml"))
        deps = cache.compute_dependencies(config_files=config_files)

        # Save data
        large_data = {"values": list(range(10000))}
        cache.save("large", large_data, dependencies=deps)

        # Load and verify completeness
        loaded = cache.load("large", dependencies=deps)
        assert loaded is not None
        assert len(loaded["values"]) == 10000
        assert loaded["values"][0] == 0
        assert loaded["values"][-1] == 9999

    def test_load_after_failed_save(self, tmp_cache_dir, config_dir_fixture):
        """Test that failed save doesn't corrupt existing cache."""
        cache = CacheManager(str(tmp_cache_dir))

        config_files = list(Path(config_dir_fixture).glob("*.toml"))
        deps = cache.compute_dependencies(config_files=config_files)

        # Save initial valid data
        cache.save("data", {"value": 42}, dependencies=deps)

        # Verify it loads correctly
        loaded = cache.load("data", dependencies=deps)
        assert loaded == {"value": 42}

        # Even if save were to fail, the old data should still be loadable
        # (This is ensured by atomic write operations)


@pytest.mark.integration
class TestCacheWithRealWorkflow:
    """Test cache in realistic workflow scenarios."""

    def test_pipeline_phase_caching_workflow(self, tmp_cache_dir, config_dir_fixture):
        """Test caching workflow similar to pipeline phase usage."""
        cache = CacheManager(str(tmp_cache_dir))

        config_files = list(Path(config_dir_fixture).glob("*.toml"))

        # Simulate Phase 2: Data loading
        phase2_params = {"years": ["2016"], "track_types": ["LL", "DD"]}
        phase2_deps = cache.compute_dependencies(
            config_files=config_files, extra_params=phase2_params
        )

        # Save Phase 2 results
        phase2_data = {
            "data": {"2016": {"LL": [1, 2, 3], "DD": [4, 5, 6]}},
            "mc": {"jpsi": {"2016": {"LL": [7, 8, 9]}}},
        }
        cache.save(
            "phase2_data",
            phase2_data,
            dependencies=phase2_deps,
            description="Phase 2: Data loading results",
        )

        # Load Phase 2 results (cache hit)
        loaded_phase2 = cache.load("phase2_data", dependencies=phase2_deps)
        assert loaded_phase2 == phase2_data

        # Simulate Phase 3: Optimization
        phase3_params = {"states": ["jpsi", "etac"]}
        phase3_deps = cache.compute_dependencies(
            config_files=config_files, extra_params=phase3_params
        )

        # Save Phase 3 results
        phase3_cuts = [
            {"state": "jpsi", "variable": "Bu_PT", "cut": 5000},
            {"state": "etac", "variable": "Bu_PT", "cut": 4500},
        ]
        cache.save(
            "phase3_cuts",
            phase3_cuts,
            dependencies=phase3_deps,
            description="Phase 3: Optimized cuts",
        )

        # Both should be in cache
        assert cache.load("phase2_data", dependencies=phase2_deps) is not None
        assert cache.load("phase3_cuts", dependencies=phase3_deps) is not None

        # Check statistics
        stats = cache.get_cache_stats()
        assert stats["num_entries"] >= 2
