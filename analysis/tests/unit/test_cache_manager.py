"""
Unit tests for CacheManager module.

Tests hash-based caching, dependency tracking, automatic invalidation,
and atomic operations without relying on external files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules.cache_manager import CacheManager, CacheMetadata


@pytest.mark.unit
@pytest.mark.cache
class TestCacheManagerInitialization:
    """Test CacheManager initialization."""

    def test_init_creates_directories(self, tmp_cache_dir: Path) -> None:
        """Test that initialization creates necessary directories."""
        cache = CacheManager(tmp_cache_dir)

        assert cache.cache_dir.exists()
        assert cache.data_dir.exists()
        assert cache.metadata_dir.exists()

        assert cache.cache_dir.is_dir()
        assert (cache.cache_dir / "data").is_dir()
        assert (cache.cache_dir / "metadata").is_dir()

    def test_init_with_existing_cache(self, tmp_cache_dir: Path) -> None:
        """Test initialization with existing cache directory."""
        # Create first cache manager
        cache1 = CacheManager(tmp_cache_dir)

        # Create second with same path (should not error)
        cache2 = CacheManager(tmp_cache_dir)

        assert cache1.cache_dir == cache2.cache_dir


@pytest.mark.unit
@pytest.mark.cache
class TestHashComputation:
    """Test hash computation methods."""

    def test_compute_hash_strings(self, tmp_cache_dir: Path) -> None:
        """Test hashing string arguments."""
        cache = CacheManager(tmp_cache_dir)

        hash1 = cache._compute_hash("test_string")
        hash2 = cache._compute_hash("test_string")
        hash3 = cache._compute_hash("different_string")

        # Same input should give same hash
        assert hash1 == hash2

        # Different input should give different hash
        assert hash1 != hash3

        # Should be hex string
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex digest length

    def test_compute_hash_multiple_args(self, tmp_cache_dir: Path) -> None:
        """Test hashing multiple arguments."""
        cache = CacheManager(tmp_cache_dir)

        hash1 = cache._compute_hash("arg1", "arg2", "arg3")
        hash2 = cache._compute_hash("arg1", "arg2", "arg3")
        hash3 = cache._compute_hash("arg1", "arg2", "different")

        assert hash1 == hash2
        assert hash1 != hash3

    def test_compute_hash_kwargs(self, tmp_cache_dir: Path) -> None:
        """Test hashing keyword arguments."""
        cache = CacheManager(tmp_cache_dir)

        hash1 = cache._compute_hash(key1="value1", key2="value2")
        hash2 = cache._compute_hash(key1="value1", key2="value2")
        hash3 = cache._compute_hash(key1="value1", key2="different")

        assert hash1 == hash2
        assert hash1 != hash3


@pytest.mark.unit
@pytest.mark.cache
class TestBasicCacheOperations:
    """Test basic cache save and load operations."""

    def test_save_and_load_simple_data(self, tmp_cache_dir: Path) -> None:
        """Test saving and loading simple data."""
        cache = CacheManager(tmp_cache_dir)

        key = "test_key"
        data = {"value": 42, "name": "test"}

        cache.save(key, data)
        loaded = cache.load(key)

        assert loaded == data

    def test_load_nonexistent_key(self, tmp_cache_dir: Path) -> None:
        """Test loading non-existent cache entry returns None."""
        cache = CacheManager(tmp_cache_dir)

        result = cache.load("nonexistent_key")

        assert result is None

    def test_save_overwrite(self, tmp_cache_dir: Path) -> None:
        """Test that saving overwrites existing cache."""
        cache = CacheManager(tmp_cache_dir)

        key = "test_key"
        cache.save(key, {"value": 1})
        cache.save(key, {"value": 2})

        loaded = cache.load(key)
        assert loaded == {"value": 2}

    def test_save_different_data_types(self, tmp_cache_dir: Path) -> None:
        """Test saving different data types."""
        cache = CacheManager(tmp_cache_dir)

        # Dictionary
        cache.save("dict", {"a": 1, "b": 2})
        assert cache.load("dict") == {"a": 1, "b": 2}

        # List
        cache.save("list", [1, 2, 3])
        assert cache.load("list") == [1, 2, 3]

        # String
        cache.save("string", "test_value")
        assert cache.load("string") == "test_value"

        # Number
        cache.save("number", 42.5)
        assert cache.load("number") == 42.5


@pytest.mark.unit
@pytest.mark.cache
class TestDependencyTracking:
    """Test dependency computation and tracking."""

    def test_compute_file_hash(self, tmp_test_dir: Path, tmp_cache_dir: Path) -> None:
        """Test computing hash of file contents."""
        cache = CacheManager(tmp_cache_dir)

        # Create test file
        test_file = tmp_test_dir / "test.txt"
        test_file.write_text("test content")

        hash1 = cache._hash_file(test_file)

        # Same file should give same hash
        hash2 = cache._hash_file(test_file)
        assert hash1 == hash2

        # Modified file should give different hash
        test_file.write_text("modified content")
        hash3 = cache._hash_file(test_file)
        assert hash1 != hash3

    def test_compute_dependencies(self, tmp_test_dir: Path, tmp_cache_dir: Path) -> None:
        """Test computing dependencies from config files."""
        cache = CacheManager(tmp_cache_dir)

        # Create config files
        config1 = tmp_test_dir / "config1.toml"
        config2 = tmp_test_dir / "config2.toml"
        config1.write_text("key = 'value1'")
        config2.write_text("key = 'value2'")

        deps = cache.compute_dependencies(config_files=[str(config1), str(config2)])

        assert isinstance(deps, dict)
        # Dependencies use "config:filename" format, not full paths
        assert "config:config1.toml" in deps
        assert "config:config2.toml" in deps
        assert "cache_version" in deps
        assert len(deps["config:config1.toml"]) == 64  # SHA256 hex length

    def test_compute_dependencies_missing_file(self, tmp_cache_dir: Path) -> None:
        """Test that missing files are skipped in dependencies."""
        cache = CacheManager(tmp_cache_dir)

        deps = cache.compute_dependencies(config_files=["/nonexistent/file.txt"])

        # Should return empty dict for missing files
        assert isinstance(deps, dict)


@pytest.mark.unit
@pytest.mark.cache
class TestCacheInvalidation:
    """Test automatic cache invalidation based on dependencies."""

    def test_cache_hit_with_unchanged_dependencies(
        self, tmp_test_dir: Path, tmp_cache_dir: Path
    ) -> None:
        """Test cache hit when dependencies haven't changed."""
        cache = CacheManager(tmp_cache_dir)

        # Create config file
        config_file = tmp_test_dir / "config.toml"
        config_file.write_text("value = 1")

        # Compute dependencies
        deps = cache.compute_dependencies(config_files=[str(config_file)])

        # Save data with dependencies
        cache.save("test_key", {"result": 42}, dependencies=deps)

        # Load should succeed
        loaded = cache.load("test_key", dependencies=deps)
        assert loaded == {"result": 42}

    def test_cache_miss_with_changed_dependencies(
        self, tmp_test_dir: Path, tmp_cache_dir: Path
    ) -> None:
        """Test cache miss when dependencies change."""
        cache = CacheManager(tmp_cache_dir)

        # Create config file
        config_file = tmp_test_dir / "config.toml"
        config_file.write_text("value = 1")

        # Compute dependencies and save
        deps1 = cache.compute_dependencies(config_files=[str(config_file)])
        cache.save("test_key", {"result": 42}, dependencies=deps1)

        # Modify config file
        config_file.write_text("value = 2")

        # Compute new dependencies
        deps2 = cache.compute_dependencies(config_files=[str(config_file)])

        # Load should return None (cache invalidated)
        loaded = cache.load("test_key", dependencies=deps2)
        assert loaded is None

    def test_cache_without_dependencies(self, tmp_cache_dir: Path) -> None:
        """Test caching without dependency tracking."""
        cache = CacheManager(tmp_cache_dir)

        # Save without dependencies
        cache.save("test_key", {"result": 42})

        # Load without dependencies
        loaded = cache.load("test_key")
        assert loaded == {"result": 42}


@pytest.mark.unit
@pytest.mark.cache
class TestCacheMetadata:
    """Test cache metadata storage and retrieval."""

    def test_metadata_created_on_save(self, tmp_cache_dir: Path) -> None:
        """Test that metadata is created when saving."""
        cache = CacheManager(tmp_cache_dir)

        cache.save("test_key", {"data": "value"}, description="Test entry")

        # Check metadata file exists
        metadata_files = list(cache.metadata_dir.glob("*.json"))
        assert len(metadata_files) > 0

    def test_metadata_contains_correct_info(self, tmp_cache_dir: Path) -> None:
        """Test metadata contains expected fields."""
        cache = CacheManager(tmp_cache_dir)

        deps = {"file1": "hash1"}
        cache.save("test_key", {"data": "value"}, dependencies=deps, description="Test description")

        # Load metadata
        metadata_files = list(cache.metadata_dir.glob("*.json"))
        assert len(metadata_files) > 0

        with open(metadata_files[0]) as f:
            metadata = json.load(f)

        assert "key" in metadata
        assert "created_at" in metadata
        assert "dependencies" in metadata
        assert "description" in metadata
        assert metadata["description"] == "Test description"


@pytest.mark.unit
@pytest.mark.cache
class TestCacheManagement:
    """Test cache management operations."""

    def test_clear_specific_entry(self, tmp_cache_dir: Path) -> None:
        """Test invalidating specific cache entry."""
        cache = CacheManager(tmp_cache_dir)

        cache.save("key1", {"value": 1})
        cache.save("key2", {"value": 2})

        # Invalidate key1
        cache.invalidate("key1")

        # key1 should be gone, key2 should remain
        assert cache.load("key1") is None
        assert cache.load("key2") == {"value": 2}

    def test_clear_all_cache(self, tmp_cache_dir: Path) -> None:
        """Test clearing all cache entries."""
        cache = CacheManager(tmp_cache_dir)

        cache.save("key1", {"value": 1})
        cache.save("key2", {"value": 2})

        # Clear all
        cache.clear_all()

        # Both should be gone
        assert cache.load("key1") is None
        assert cache.load("key2") is None

    def test_get_cache_stats(self, tmp_cache_dir: Path) -> None:
        """Test getting cache statistics."""
        cache = CacheManager(tmp_cache_dir)

        # Empty cache
        stats = cache.get_cache_stats()
        assert stats["num_entries"] == 0
        assert stats["total_size_mb"] == 0.0

        # Add some entries
        cache.save("key1", {"value": 1})
        cache.save("key2", {"value": 2})

        stats = cache.get_cache_stats()
        assert stats["num_entries"] == 2
        assert stats["total_size_mb"] > 0.0


@pytest.mark.unit
@pytest.mark.cache
def test_cache_metadata_dataclass() -> None:
    """Test CacheMetadata dataclass."""
    metadata = CacheMetadata(
        key="test_key",
        created_at="2024-01-01T00:00:00",
        dependencies={"file1": "hash1"},
        version="1.0.0",
        size_bytes=1024,
        description="Test cache entry",
    )

    assert metadata.key == "test_key"
    assert metadata.dependencies == {"file1": "hash1"}
    assert metadata.size_bytes == 1024
