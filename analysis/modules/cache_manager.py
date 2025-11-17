"""
Hash-based cache manager with automatic invalidation.

This module provides a robust caching system that:
- Uses content-based hashing (SHA256) for cache keys
- Tracks dependencies (config files, code versions, input data)
- Automatically invalidates cache when dependencies change
- Stores metadata for debugging and auditing
- Provides atomic write operations to prevent corruption
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .exceptions import CacheError


@dataclass
class CacheMetadata:
    """
    Metadata stored with each cache entry.

    Attributes:
        key: Cache key (hash of inputs)
        created_at: ISO timestamp of cache creation
        dependencies: Dict of dependency names to their hashes
        version: Cache format version
        size_bytes: Size of cached data in bytes
        description: Human-readable description
    """

    key: str
    created_at: str
    dependencies: dict[str, str]
    version: str
    size_bytes: int
    description: str


class CacheManager:
    """
    Hash-based cache manager with automatic invalidation.

    Features:
    - Content-based cache keys using SHA256 hashing
    - Dependency tracking for automatic invalidation
    - Atomic writes to prevent corruption
    - Metadata storage for debugging
    - Backward compatibility with simple pickle caching

    Usage:
        >>> cache = CacheManager("./cache")
        >>> deps = cache.compute_dependencies(config_files=["config/selection.toml"])
        >>> data = cache.load("phase2_data", dependencies=deps)
        >>> if data is None:
        >>>     data = expensive_computation()
        >>>     cache.save("phase2_data", data, dependencies=deps)
    """

    CACHE_VERSION: str = "1.0.0"

    def __init__(self, cache_dir: str | Path) -> None:
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir: Path = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Create subdirectories for organization
        self.data_dir: Path = self.cache_dir / "data"
        self.metadata_dir: Path = self.cache_dir / "metadata"
        self.data_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

    def _compute_hash(self, *args: Any, **kwargs: Any) -> str:
        """
        Compute SHA256 hash of arguments.

        Args:
            *args: Positional arguments to hash
            **kwargs: Keyword arguments to hash

        Returns:
            Hex string of hash
        """
        hasher = hashlib.sha256()

        # Hash positional arguments
        for arg in args:
            hasher.update(self._serialize_for_hash(arg))

        # Hash keyword arguments (sorted for consistency)
        for key in sorted(kwargs.keys()):
            hasher.update(key.encode("utf-8"))
            hasher.update(self._serialize_for_hash(kwargs[key]))

        return hasher.hexdigest()

    def _serialize_for_hash(self, obj: Any) -> bytes:
        """
        Serialize object to bytes for hashing.

        Args:
            obj: Object to serialize

        Returns:
            Bytes representation
        """
        if isinstance(obj, (str, int, float, bool)):
            return str(obj).encode("utf-8")
        if isinstance(obj, bytes):
            return obj
        if isinstance(obj, (list, tuple)) or isinstance(obj, dict):
            return json.dumps(obj, sort_keys=True).encode("utf-8")
        if isinstance(obj, Path):
            return str(obj).encode("utf-8")
        # Fallback: use pickle representation
        try:
            return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise CacheError(f"Cannot serialize object of type {type(obj)} for hashing: {e}")

    def _hash_file(self, filepath: str | Path) -> str:
        """
        Compute SHA256 hash of file contents.

        Args:
            filepath: Path to file

        Returns:
            Hex string of file hash
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise CacheError(f"Cannot hash non-existent file: {filepath}")

        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)

        return hasher.hexdigest()

    def compute_dependencies(
        self,
        config_files: list[str | Path] | None = None,
        code_files: list[str | Path] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """
        Compute dependency hashes for cache validation.

        This method computes hashes of all dependencies that should trigger
        cache invalidation when changed.

        Args:
            config_files: Configuration files to track
            code_files: Python code files to track (for detecting algorithm changes)
            extra_params: Additional parameters to include (e.g., years, track_types)

        Returns:
            Dict mapping dependency names to their hashes
        """
        dependencies: dict[str, str] = {}

        # Hash configuration files
        if config_files:
            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    dep_name = f"config:{config_path.name}"
                    dependencies[dep_name] = self._hash_file(config_path)

        # Hash code files
        if code_files:
            for code_file in code_files:
                code_path = Path(code_file)
                if code_path.exists():
                    dep_name = f"code:{code_path.name}"
                    dependencies[dep_name] = self._hash_file(code_path)

        # Hash extra parameters
        if extra_params:
            for param_name, param_value in sorted(extra_params.items()):
                dep_name = f"param:{param_name}"
                dependencies[dep_name] = self._compute_hash(param_value)

        # Add cache version to dependencies
        dependencies["cache_version"] = self.CACHE_VERSION

        return dependencies

    def _generate_cache_key(self, name: str, dependencies: dict[str, str] | None = None) -> str:
        """
        Generate cache key from name and dependencies.

        Args:
            name: Base name for cache entry
            dependencies: Dependency hashes

        Returns:
            Cache key (hash)
        """
        if dependencies:
            # Include dependencies in hash
            return self._compute_hash(name, dependencies)
        # Simple hash of name only (backward compatibility)
        return self._compute_hash(name)

    def _get_cache_paths(self, cache_key: str) -> tuple[Path, Path]:
        """
        Get paths for data and metadata files.

        Args:
            cache_key: Cache key

        Returns:
            (data_path, metadata_path)
        """
        data_path = self.data_dir / f"{cache_key}.pkl"
        metadata_path = self.metadata_dir / f"{cache_key}.json"
        return data_path, metadata_path

    def load(
        self, name: str, dependencies: dict[str, str] | None = None, validate: bool = True
    ) -> Any | None:
        """
        Load data from cache if valid.

        Args:
            name: Cache entry name
            dependencies: Dependency hashes for validation
            validate: Whether to validate dependencies

        Returns:
            Cached data if valid, None otherwise
        """
        cache_key = self._generate_cache_key(name, dependencies)
        data_path, metadata_path = self._get_cache_paths(cache_key)

        # Check if cache exists
        if not data_path.exists():
            return None

        # Load and validate metadata
        if validate and metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata_dict = json.load(f)
                    metadata = CacheMetadata(**metadata_dict)

                # Validate cache version
                if metadata.version != self.CACHE_VERSION:
                    print(
                        f"  ⚠️  Cache version mismatch: {metadata.version} != {self.CACHE_VERSION}"
                    )
                    return None

                # Validate dependencies
                if dependencies and metadata.dependencies != dependencies:
                    print(f"  ⚠️  Cache invalidated: dependencies changed for '{name}'")
                    return None

            except Exception as e:
                print(f"  ⚠️  Cache metadata corrupted for '{name}': {e}")
                return None

        # Load cached data
        try:
            with open(data_path, "rb") as f:
                data = pickle.load(f)

            print(
                f"  ✓ Loaded from cache: {name} ({data_path.stat().st_size / 1024 / 1024:.1f} MB)"
            )
            return data

        except Exception as e:
            raise CacheError(f"Failed to load cache for '{name}': {e}")

    def save(
        self,
        name: str,
        data: Any,
        dependencies: dict[str, str] | None = None,
        description: str = "",
    ) -> None:
        """
        Save data to cache with metadata.

        Uses atomic write (write to temp file, then rename) to prevent corruption.

        Args:
            name: Cache entry name
            data: Data to cache
            dependencies: Dependency hashes
            description: Human-readable description
        """
        cache_key = self._generate_cache_key(name, dependencies)
        data_path, metadata_path = self._get_cache_paths(cache_key)

        # Atomic write for data
        temp_data_path = data_path.with_suffix(".tmp")
        try:
            with open(temp_data_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Get size before moving
            size_bytes = temp_data_path.stat().st_size

            # Atomic rename
            shutil.move(str(temp_data_path), str(data_path))

        except Exception as e:
            if temp_data_path.exists():
                temp_data_path.unlink()
            raise CacheError(f"Failed to save cache for '{name}': {e}")

        # Save metadata
        metadata = CacheMetadata(
            key=cache_key,
            created_at=datetime.now().isoformat(),
            dependencies=dependencies or {},
            version=self.CACHE_VERSION,
            size_bytes=size_bytes,
            description=description or name,
        )

        temp_metadata_path = metadata_path.with_suffix(".tmp")
        try:
            with open(temp_metadata_path, "w") as f:
                json.dump(asdict(metadata), f, indent=2)

            # Atomic rename
            shutil.move(str(temp_metadata_path), str(metadata_path))

        except Exception as e:
            if temp_metadata_path.exists():
                temp_metadata_path.unlink()
            print(f"  ⚠️  Failed to save metadata for '{name}': {e}")

        print(f"  → Cached: {name} ({size_bytes / 1024 / 1024:.1f} MB)")

    def invalidate(self, name: str, dependencies: dict[str, str] | None = None) -> None:
        """
        Invalidate (delete) a cache entry.

        Args:
            name: Cache entry name
            dependencies: Dependency hashes (to generate correct key)
        """
        cache_key = self._generate_cache_key(name, dependencies)
        data_path, metadata_path = self._get_cache_paths(cache_key)

        if data_path.exists():
            data_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()

        print(f"  ✓ Invalidated cache: {name}")

    def clear_all(self) -> None:
        """
        Clear all cache entries.

        Warning: This deletes all cached data!
        """
        for cache_file in self.data_dir.glob("*.pkl"):
            cache_file.unlink()

        for metadata_file in self.metadata_dir.glob("*.json"):
            metadata_file.unlink()

        print("  ✓ Cleared all cache entries")

    def list_entries(self) -> list[CacheMetadata]:
        """
        List all cache entries with their metadata.

        Returns:
            List of CacheMetadata objects
        """
        entries: list[CacheMetadata] = []

        for metadata_file in sorted(self.metadata_dir.glob("*.json")):
            try:
                with open(metadata_file) as f:
                    metadata_dict = json.load(f)
                    entries.append(CacheMetadata(**metadata_dict))
            except Exception as e:
                print(f"  ⚠️  Corrupted metadata: {metadata_file.name}: {e}")

        return entries

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        entries = self.list_entries()
        total_size = sum(entry.size_bytes for entry in entries)

        return {
            "num_entries": len(entries),
            "total_size_mb": total_size / 1024 / 1024,
            "cache_dir": str(self.cache_dir),
            "version": self.CACHE_VERSION,
        }
