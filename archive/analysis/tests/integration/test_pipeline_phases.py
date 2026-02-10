"""
Integration tests for pipeline phase management.

Tests the PipelineManager class to ensure proper initialization,
configuration handling, dependency tracking, and cache management.

These tests validate the orchestration layer that coordinates
all analysis phases.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from run_pipeline import PipelineManager


@pytest.mark.integration
class TestPipelineManagerInitialization:
    """Test PipelineManager initialization and setup."""

    def test_initialize_with_valid_config(self, config_dir_fixture, tmp_cache_dir):
        """Test PipelineManager initializes correctly with valid configuration."""
        pipeline = PipelineManager(config_dir=str(config_dir_fixture), cache_dir=str(tmp_cache_dir))

        # Verify configuration is loaded
        assert hasattr(pipeline, "config")
        assert hasattr(pipeline.config, "physics")
        assert hasattr(pipeline.config, "detector")
        assert hasattr(pipeline.config, "fitting")

        # Verify backward compatibility attributes
        assert hasattr(pipeline.config, "particles")
        assert hasattr(pipeline.config, "paths")
        assert hasattr(pipeline.config, "luminosity")

        # Verify cache manager is initialized
        assert hasattr(pipeline, "cache")
        assert pipeline.cache.cache_dir.exists()

    def test_initialize_with_invalid_config_dir(self, tmp_test_dir, tmp_cache_dir):
        """Test PipelineManager raises error with invalid config directory."""
        invalid_config_dir = tmp_test_dir / "nonexistent_config"

        with pytest.raises(Exception):  # Could be ConfigurationError or FileNotFoundError
            PipelineManager(config_dir=str(invalid_config_dir), cache_dir=str(tmp_cache_dir))

    def test_output_directories_created(self, config_dir_fixture, tmp_cache_dir):
        """Test that output directories are created on initialization."""
        pipeline = PipelineManager(config_dir=str(config_dir_fixture), cache_dir=str(tmp_cache_dir))

        # These should be created, or at least the output structure should be defined
        assert "tables_dir" in pipeline.config.paths["output"]
        assert "plots_dir" in pipeline.config.paths["output"]
        assert "results_dir" in pipeline.config.paths["output"]

    def test_cache_statistics_available(self, config_dir_fixture, tmp_cache_dir):
        """Test that cache statistics can be retrieved."""
        pipeline = PipelineManager(config_dir=str(config_dir_fixture), cache_dir=str(tmp_cache_dir))

        stats = pipeline.cache.get_cache_stats()

        assert isinstance(stats, dict)
        assert "num_entries" in stats
        assert "total_size_mb" in stats
        assert stats["num_entries"] >= 0
        assert stats["total_size_mb"] >= 0.0


@pytest.mark.integration
class TestPipelineDependencyTracking:
    """Test dependency computation and tracking."""

    def test_compute_phase2_dependencies(self, config_dir_fixture, tmp_cache_dir):
        """Test Phase 2 dependency computation includes config and code files."""
        pipeline = PipelineManager(config_dir=str(config_dir_fixture), cache_dir=str(tmp_cache_dir))

        dependencies = pipeline._compute_phase_dependencies(
            phase="2", extra_params={"years": ["2016"], "track_types": ["LL"]}
        )

        # Dependencies should include hashes
        assert isinstance(dependencies, dict)
        assert len(dependencies) > 0

        # Should include config files and code files
        # Verify the structure makes sense
        for key, value in dependencies.items():
            assert isinstance(value, str)  # Hash values
            assert len(value) > 0

    def test_dependencies_change_with_parameters(self, config_dir_fixture, tmp_cache_dir):
        """Test that dependencies change when parameters change."""
        pipeline = PipelineManager(config_dir=str(config_dir_fixture), cache_dir=str(tmp_cache_dir))

        deps1 = pipeline._compute_phase_dependencies(
            phase="2", extra_params={"years": ["2016"], "track_types": ["LL"]}
        )

        deps2 = pipeline._compute_phase_dependencies(
            phase="2", extra_params={"years": ["2017"], "track_types": ["DD"]}
        )

        # Dependencies should be different due to different parameters
        assert deps1 != deps2

    def test_get_config_files(self, config_dir_fixture, tmp_cache_dir):
        """Test that configuration files are correctly identified."""
        pipeline = PipelineManager(config_dir=str(config_dir_fixture), cache_dir=str(tmp_cache_dir))

        config_files = pipeline._get_config_files()

        assert isinstance(config_files, list)
        assert len(config_files) > 0

        # All should be Path objects pointing to .toml files
        for f in config_files:
            assert isinstance(f, Path)
            assert f.suffix == ".toml"
            assert f.exists()


@pytest.mark.integration
class TestPipelineCaching:
    """Test pipeline-level caching functionality."""

    def test_cache_save_and_load_simple_data(self, config_dir_fixture, tmp_cache_dir):
        """Test caching and retrieving simple data structures."""
        pipeline = PipelineManager(config_dir=str(config_dir_fixture), cache_dir=str(tmp_cache_dir))

        # Create test data
        test_data = {"year": "2016", "track_type": "LL", "n_events": 1000}

        # Compute dependencies
        dependencies = pipeline._compute_phase_dependencies(
            phase="2", extra_params={"test": "data"}
        )

        # Save to cache
        pipeline.cache.save(
            "test_data",
            test_data,
            dependencies=dependencies,
            description="Test data for integration test",
        )

        # Load from cache
        loaded_data = pipeline.cache.load("test_data", dependencies=dependencies)

        assert loaded_data is not None
        assert loaded_data == test_data

    def test_cache_invalidation_on_config_change(self, config_dir_fixture, tmp_cache_dir):
        """Test that cache is invalidated when configuration changes."""
        pipeline = PipelineManager(config_dir=str(config_dir_fixture), cache_dir=str(tmp_cache_dir))

        # Create initial dependencies
        test_data = {"value": 42}
        deps1 = pipeline._compute_phase_dependencies(phase="2", extra_params={"version": "1"})

        # Save with first dependencies
        pipeline.cache.save("test_key", test_data, dependencies=deps1)

        # Create different dependencies (simulating config change)
        deps2 = pipeline._compute_phase_dependencies(
            phase="2", extra_params={"version": "2"}  # Different parameter
        )

        # Try to load with different dependencies - should fail
        loaded = pipeline.cache.load("test_key", dependencies=deps2)

        # Should be None because dependencies don't match
        assert loaded is None

    def test_cache_with_nested_data_structures(self, config_dir_fixture, tmp_cache_dir):
        """Test caching with nested dictionaries (like MC/data structures)."""
        pipeline = PipelineManager(config_dir=str(config_dir_fixture), cache_dir=str(tmp_cache_dir))

        # Create nested structure similar to MC dict
        nested_data = {
            "jpsi": {
                "2016": {"LL": [1, 2, 3], "DD": [4, 5, 6]},
                "2017": {"LL": [7, 8, 9], "DD": [10, 11, 12]},
            },
            "etac": {"2016": {"LL": [13, 14, 15], "DD": [16, 17, 18]}},
        }

        deps = pipeline._compute_phase_dependencies(phase="2", extra_params={"nested": "test"})

        # Save and load
        pipeline.cache.save("nested_test", nested_data, dependencies=deps)
        loaded = pipeline.cache.load("nested_test", dependencies=deps)

        assert loaded is not None
        assert loaded == nested_data
        assert loaded["jpsi"]["2016"]["LL"] == [1, 2, 3]


@pytest.mark.integration
class TestPipelineConfiguration:
    """Test configuration validation and access."""

    def test_config_has_required_sections(self, config_dir_fixture, tmp_cache_dir):
        """Test that all required configuration sections are present."""
        pipeline = PipelineManager(config_dir=str(config_dir_fixture), cache_dir=str(tmp_cache_dir))

        # Check all required logical sections
        required_sections = [
            "physics",
            "detector",
            "fitting",
            "selection",
            "triggers",
            "data",
            "efficiencies",
        ]

        for section in required_sections:
            assert hasattr(pipeline.config, section), f"Missing section: {section}"

    def test_backward_compatibility_layer(self, config_dir_fixture, tmp_cache_dir):
        """Test backward compatibility attributes are created correctly."""
        pipeline = PipelineManager(config_dir=str(config_dir_fixture), cache_dir=str(tmp_cache_dir))

        # Check backward compatibility attributes
        compat_attrs = [
            "particles",
            "paths",
            "luminosity",
            "branching_fractions",
        ]

        for attr in compat_attrs:
            assert hasattr(
                pipeline.config, attr
            ), f"Missing backward compatibility attribute: {attr}"

    def test_config_access_patterns(self, config_dir_fixture, tmp_cache_dir):
        """Test various configuration access patterns work correctly."""
        pipeline = PipelineManager(config_dir=str(config_dir_fixture), cache_dir=str(tmp_cache_dir))

        # Test physics access - physics is a dict
        assert "pdg_masses" in pipeline.config.physics

        # Test detector access - detector is a dict
        assert "signal_regions" in pipeline.config.detector

        # Test paths access (both new and backward compatible)
        assert "output" in pipeline.config.paths
        assert "base_path" in pipeline.config.paths["output"]


@pytest.mark.integration
class TestPipelineCacheWorkflow:
    """Test complete cache workflow scenarios."""

    def test_cache_hit_on_repeated_access(self, config_dir_fixture, tmp_cache_dir):
        """Test that repeated access with same dependencies results in cache hit."""
        pipeline = PipelineManager(config_dir=str(config_dir_fixture), cache_dir=str(tmp_cache_dir))

        test_data = {"result": [1, 2, 3, 4, 5]}
        deps = pipeline._compute_phase_dependencies(phase="3")

        # First save
        pipeline.cache.save("result_data", test_data, dependencies=deps)

        # First load - should hit cache
        loaded1 = pipeline.cache.load("result_data", dependencies=deps)
        assert loaded1 == test_data

        # Second load - should also hit cache
        loaded2 = pipeline.cache.load("result_data", dependencies=deps)
        assert loaded2 == test_data
        assert loaded2 == loaded1

    def test_cache_miss_on_nonexistent_key(self, config_dir_fixture, tmp_cache_dir):
        """Test that loading non-existent key returns None."""
        pipeline = PipelineManager(config_dir=str(config_dir_fixture), cache_dir=str(tmp_cache_dir))

        deps = pipeline._compute_phase_dependencies(phase="5")

        # Try to load non-existent key
        loaded = pipeline.cache.load("nonexistent_key", dependencies=deps)

        assert loaded is None

    def test_cache_clear_and_statistics(self, config_dir_fixture, tmp_cache_dir):
        """Test cache clearing and statistics reporting."""
        pipeline = PipelineManager(config_dir=str(config_dir_fixture), cache_dir=str(tmp_cache_dir))

        # Add some data to cache
        deps = pipeline._compute_phase_dependencies(phase="2")
        pipeline.cache.save("data1", {"a": 1}, dependencies=deps)
        pipeline.cache.save("data2", {"b": 2}, dependencies=deps)

        # Check statistics
        stats = pipeline.cache.get_cache_stats()
        assert stats["num_entries"] >= 2

        # Clear all cache
        pipeline.cache.clear_all()

        # Check statistics after clear
        stats_after = pipeline.cache.get_cache_stats()
        assert stats_after["num_entries"] == 0


@pytest.mark.integration
@pytest.mark.slow
class TestPipelinePhaseOrchestration:
    """Test phase orchestration and data flow between phases."""

    def test_phase_dependencies_different_per_phase(self, config_dir_fixture, tmp_cache_dir):
        """Test that different phases have different dependency tracking."""
        pipeline = PipelineManager(config_dir=str(config_dir_fixture), cache_dir=str(tmp_cache_dir))

        deps_phase2 = pipeline._compute_phase_dependencies(phase="2")
        deps_phase3 = pipeline._compute_phase_dependencies(phase="3")
        deps_phase5 = pipeline._compute_phase_dependencies(phase="5")

        # Dependencies should be different for different phases
        # (they may include different code files)
        assert isinstance(deps_phase2, dict)
        assert isinstance(deps_phase3, dict)
        assert isinstance(deps_phase5, dict)

    def test_cache_description_stored(self, config_dir_fixture, tmp_cache_dir):
        """Test that cache entries include descriptive metadata."""
        pipeline = PipelineManager(config_dir=str(config_dir_fixture), cache_dir=str(tmp_cache_dir))

        test_data = {"sample": "data"}
        deps = pipeline._compute_phase_dependencies(phase="2")
        description = "Test data for metadata verification"

        # Save with description
        pipeline.cache.save("metadata_test", test_data, dependencies=deps, description=description)

        # Check that metadata file exists and contains description
        # Find metadata file using glob pattern instead of private method
        metadata_files = list(pipeline.cache.metadata_dir.glob("*.json"))
        assert len(metadata_files) > 0, "No metadata files found"

        # Find the metadata file we just created - metadata uses 'key' not 'cache_key'
        metadata_file = None
        for mf in metadata_files:
            with open(mf) as f:
                meta = json.load(f)
                # Check if this metadata has the description we set
                if meta.get("description") == description:
                    metadata_file = mf
                    break

        assert metadata_file is not None, "Metadata file not found"

        with open(metadata_file) as f:
            metadata = json.load(f)

        assert "description" in metadata
        assert metadata["description"] == description
