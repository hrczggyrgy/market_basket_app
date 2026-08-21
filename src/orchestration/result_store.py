"""Versioned result store keyed by (dataset_id, analysis_key, analysis_version, param_hash).

Provides get/has/set/invalidate operations with deterministic parameter hashing
using xxhash64 for cache versioning and schema/feature versioning.

Example:
    store = ResultStore()
    store.set("ds1", "basket", "1.0.0", "abc123", result)
    result = store.get("ds1", "basket", "1.0.0", "abc123")
"""

from __future__ import annotations

import json
from typing import Any, Optional

import xxhash

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SCHEMA_VERSION = "1.0.0"
DEFAULT_FEATURE_VERSION = "1.0.0"

# In-memory storage: key -> result
_store: dict[str, dict[str, Any]] = {}

# Schema/feature versioning
_schema_version: str = DEFAULT_SCHEMA_VERSION
_feature_version: str = DEFAULT_FEATURE_VERSION


# ---------------------------------------------------------------------------
# Deterministic param hash computation (module-level)
# ---------------------------------------------------------------------------


def param_hash(
    params: dict[str, Any],
    *,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
) -> str:
    """Compute a deterministic xxhash64 hash of sorted parameters.

    The hash is computed by serializing sorted key-value pairs as JSON,
    then hashing with xxhash64. This ensures that identical parameter
    combinations always produce the same hash, while different orders or
    values produce different hashes.

    Args:
        params: Dictionary of parameter names to values.
        schema_version: Optional schema version string included in the hash
            input for cross-version invalidation.

    Returns:
        Hex string of the xxhash64 hash (32 characters).
    """
    sorted_items = sorted(params.items())
    serialized = json.dumps(sorted_items, sort_keys=True, separators=(",", ":"))
    hash_input = f"{schema_version}:{serialized}"
    h = xxhash.xxh64(hash_input.encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Key construction (module-level)
# ---------------------------------------------------------------------------


def make_key(
    dataset_id: str,
    analysis_key: str,
    analysis_version: str,
    param_hash: str,
) -> str:
    """Construct a versioned cache key.

    Key format: "dataset_id:analysis_key:analysis_version:param_hash"
    """
    return f"{dataset_id}:{analysis_key}:{analysis_version}:{param_hash}"


# ---------------------------------------------------------------------------
# ResultStore class
# ---------------------------------------------------------------------------


class ResultStore:
    """Versioned result store keyed by (dataset_id, analysis_key, analysis_version, param_hash).

    Provides get/has/set/invalidate operations with deterministic parameter hashing
    using xxhash64 for cache versioning and schema/feature versioning.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    # -----------------------------------------------------------------
    # Core operations
    # -----------------------------------------------------------------

    def get(
        self,
        dataset_id: str,
        analysis_key: str,
        analysis_version: str,
        param_hash: str,
        *,
        default: Any = None,
    ) -> Any:
        """Retrieve a cached result.

        Args:
            dataset_id: Identifier for the dataset used in the computation.
            analysis_key: The analysis specification key.
            analysis_version: Version string of the analysis implementation.
            param_hash: Deterministic hash of the parameters.
            default: Value to return if the key is not found.

        Returns:
            The cached result, or `default` if not found.
        """
        key = make_key(dataset_id, analysis_key, analysis_version, param_hash)
        return self._store.get(key, default)

    def has(
        self,
        dataset_id: str,
        analysis_key: str,
        analysis_version: str,
        param_hash: str,
    ) -> bool:
        """Check if a cached result exists.

        Args:
            dataset_id: Identifier for the dataset used in the computation.
            analysis_key: The analysis specification key.
            analysis_version: Version string of the analysis implementation.
            param_hash: Deterministic hash of the parameters.

        Returns:
            True if the key exists in the store.
        """
        key = make_key(dataset_id, analysis_key, analysis_version, param_hash)
        return key in self._store

    def set(
        self,
        dataset_id: str,
        analysis_key: str,
        analysis_version: str,
        param_hash: str,
        result: Any,
    ) -> None:
        """Store a result in the cache.

        Args:
            dataset_id: Identifier for the dataset used in the computation.
            analysis_key: The analysis specification key.
            analysis_version: Version string of the analysis implementation.
            param_hash: Deterministic hash of the parameters.
            result: The result to cache.
        """
        key = make_key(dataset_id, analysis_key, analysis_version, param_hash)
        self._store[key] = {
            "result": result,
            "dataset_id": dataset_id,
            "analysis_key": analysis_key,
            "analysis_version": analysis_version,
            "param_hash": param_hash,
        }

    def invalidate(
        self,
        dataset_id: str,
        analysis_key: str,
        analysis_version: str,
        *,
        param_hash: Optional[str] = None,
    ) -> None:
        """Invalidate cached results.

        If param_hash is provided, only invalidate that specific parameter combination.
        If param_hash is None, invalidate all entries for the given analysis/dataset.
        """
        if param_hash is not None:
            key = make_key(dataset_id, analysis_key, analysis_version, param_hash)
            self._store.pop(key, None)
        else:
            keys_to_remove: list[str] = []
            for key in self._store:
                parts = key.split(":")
                if len(parts) == 4 and parts[0] == dataset_id and parts[1] == analysis_key and parts[2] == analysis_version:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                self._store.pop(key, None)

    def invalidate_all(self) -> None:
        """Invalidate the entire cache."""
        self._store.clear()

    # -----------------------------------------------------------------
    # Version management
    # -----------------------------------------------------------------

    def set_schema_version(self, version: str) -> None:
        """Set the global schema version. Changes invalidate all cached results."""
        global _schema_version
        _schema_version = version
        self.invalidate_all()

    def set_feature_version(self, version: str) -> None:
        """Set the global feature version."""
        global _feature_version
        _feature_version = version

    def get_schema_version(self) -> str:
        """Get the current schema version."""
        return _schema_version

    def get_feature_version(self) -> str:
        """Get the current feature version."""
        return _feature_version

    # -----------------------------------------------------------------
    # Batch operations
    # -----------------------------------------------------------------

    def get_many(
        self,
        entries: list[tuple[str, str, str, str]],
    ) -> dict[tuple[str, str, str, str], Any]:
        """Retrieve multiple cached results.

        Args:
            entries: List of (dataset_id, analysis_key, analysis_version, param_hash) tuples.

        Returns:
            Dictionary mapping (dataset_id, analysis_key, analysis_version, param_hash) -> result.
            Results that are not found are mapped to None.
        """
        result: dict[tuple[str, str, str, str], Any] = {}
        for dataset_id, analysis_key, analysis_version, param_hash in entries:
            key = make_key(dataset_id, analysis_key, analysis_version, param_hash)
            result[(dataset_id, analysis_key, analysis_version, param_hash)] = self._store.get(key)
        return result

    def set_many(
        self,
        entries: list[tuple[str, str, str, str, Any]],
    ) -> None:
        """Store multiple results in the cache.

        Args:
            entries: List of (dataset_id, analysis_key, analysis_version, param_hash, result) tuples.
        """
        for dataset_id, analysis_key, analysis_version, param_hash, result in entries:
            self.set(dataset_id, analysis_key, analysis_version, param_hash, result)

    # -----------------------------------------------------------------
    # Status/debug
    # -----------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return cache status for debugging."""
        return {
            "schema_version": _schema_version,
            "feature_version": _feature_version,
            "total_entries": len(self._store),
            "keys": list(self._store.keys()),
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions (backward compatible)
# ---------------------------------------------------------------------------

def get(
    dataset_id: str,
    analysis_key: str,
    analysis_version: str,
    param_hash: str,
    *,
    default: Any = None,
) -> Any:
    """Retrieve a cached result (module-level convenience function)."""
    return ResultStore().get(dataset_id, analysis_key, analysis_version, param_hash, default=default)


def has(
    dataset_id: str,
    analysis_key: str,
    analysis_version: str,
    param_hash: str,
) -> bool:
    """Check if a cached result exists (module-level convenience function)."""
    return ResultStore().has(dataset_id, analysis_key, analysis_version, param_hash)


def set(
    dataset_id: str,
    analysis_key: str,
    analysis_version: str,
    param_hash: str,
    result: Any,
) -> None:
    """Store a result in the cache (module-level convenience function)."""
    ResultStore().set(dataset_id, analysis_key, analysis_version, param_hash, result)


def invalidate(
    dataset_id: str,
    analysis_key: str,
    analysis_version: str,
    *,
    param_hash: Optional[str] = None,
) -> None:
    """Invalidate cached results (module-level convenience function)."""
    ResultStore().invalidate(dataset_id, analysis_key, analysis_version, param_hash=param_hash)


def invalidate_all() -> None:
    """Invalidate the entire cache (module-level convenience function)."""
    ResultStore().invalidate_all()


def set_schema_version_func(version: str) -> None:
    """Set the global schema version. Changes invalidate all cached results."""
    global _schema_version
    _schema_version = version
    invalidate_all()


def set_feature_version_func(version: str) -> None:
    """Set the global feature version."""
    global _feature_version
    _feature_version = version


def get_schema_version_func() -> str:
    """Get the current schema version."""
    return _schema_version


def get_feature_version_func() -> str:
    """Get the current feature version."""
    return _feature_version


# Backward compatibility aliases
get_schema_version = get_schema_version_func
get_feature_version = get_feature_version_func


# Global default instance
_default_store = ResultStore()


def get_default() -> ResultStore:
    """Get the default ResultStore instance."""
    return _default_store
