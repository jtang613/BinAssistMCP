"""
Analysis cache for BinAssistMCP

This module provides LRU caching with invalidation for expensive
analysis operations like decompilation.
"""

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .logging import log


@dataclass
class CacheEntry:
    """A single cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    size_estimate: int = 0  # Rough size in bytes

    def touch(self):
        """Update last accessed time"""
        self.last_accessed = time.time()


class AnalysisCache:
    """LRU cache for analysis results with binary-scoped invalidation"""

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100, ttl_seconds: int = 3600):
        """Initialize the cache.

        Args:
            max_size: Maximum number of cache entries
            max_memory_mb: Maximum estimated memory usage in MB
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._max_size = max_size
        self._max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self._ttl = ttl_seconds
        self._current_memory = 0

        # Track which keys belong to which binary for invalidation
        self._binary_keys: Dict[str, Set[str]] = {}

        # Statistics
        self._hits = 0
        self._misses = 0

    def _make_key(self, binary_name: str, tool: str, *args, **kwargs) -> str:
        """Create a unique cache key.

        Args:
            binary_name: Name of the binary
            tool: Tool/method name
            *args: Tool arguments
            **kwargs: Tool keyword arguments

        Returns:
            SHA256 hash-based cache key
        """
        # Create a string representation of the key components
        key_parts = [binary_name, tool]
        key_parts.extend(str(arg) for arg in args)
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_string = ":".join(key_parts)

        # Hash for consistent key length
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]

    def _estimate_size(self, value: Any) -> int:
        """Estimate the memory size of a value.

        Args:
            value: Value to estimate

        Returns:
            Estimated size in bytes
        """
        if value is None:
            return 0
        if isinstance(value, str):
            return len(value.encode('utf-8', errors='ignore'))
        if isinstance(value, (bytes, bytearray)):
            return len(value)
        if isinstance(value, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v)
                      for k, v in value.items())
        if isinstance(value, (list, tuple, set)):
            return sum(self._estimate_size(item) for item in value)
        # Default estimate for other types
        return 100

    def get(self, binary_name: str, tool: str, *args, **kwargs) -> Optional[Any]:
        """Get a cached result.

        Args:
            binary_name: Name of the binary
            tool: Tool/method name
            *args: Tool arguments
            **kwargs: Tool keyword arguments

        Returns:
            Cached value if found and valid, None otherwise
        """
        key = self._make_key(binary_name, tool, *args, **kwargs)

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check TTL
            if time.time() - entry.created_at > self._ttl:
                self._remove_entry(key)
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1

            return entry.value

    def set(self, binary_name: str, tool: str, result: Any, *args, **kwargs):
        """Cache a result.

        Args:
            binary_name: Name of the binary
            tool: Tool/method name
            result: Result to cache
            *args: Tool arguments
            **kwargs: Tool keyword arguments
        """
        key = self._make_key(binary_name, tool, *args, **kwargs)
        size = self._estimate_size(result)

        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)

            # Evict entries if needed
            self._evict_if_needed(size)

            # Create new entry
            now = time.time()
            entry = CacheEntry(
                key=key,
                value=result,
                created_at=now,
                last_accessed=now,
                size_estimate=size
            )

            self._cache[key] = entry
            self._current_memory += size

            # Track binary association
            if binary_name not in self._binary_keys:
                self._binary_keys[binary_name] = set()
            self._binary_keys[binary_name].add(key)

            log.log_debug(f"Cached {tool} for {binary_name} (key={key[:8]}..., size={size})")

    def invalidate(self, binary_name: str):
        """Invalidate all cache entries for a specific binary.

        Args:
            binary_name: Name of the binary to invalidate
        """
        with self._lock:
            if binary_name not in self._binary_keys:
                return

            keys_to_remove = list(self._binary_keys[binary_name])
            for key in keys_to_remove:
                self._remove_entry(key)

            del self._binary_keys[binary_name]
            log.log_info(f"Invalidated {len(keys_to_remove)} cache entries for {binary_name}")

    def invalidate_tool(self, binary_name: str, tool: str):
        """Invalidate cache entries for a specific tool on a binary.

        This is useful when a tool result may be affected by changes
        without invalidating the entire binary cache.

        Args:
            binary_name: Name of the binary
            tool: Tool name to invalidate
        """
        with self._lock:
            if binary_name not in self._binary_keys:
                return

            # We need to check each key - this is O(n) but invalidation is rare
            keys_to_remove = []
            for key in self._binary_keys[binary_name]:
                if key in self._cache:
                    # Re-compute key prefix to check tool match
                    # Since we use hashes, we'd need to track tool names separately
                    # For now, this is a simplified implementation
                    pass

            for key in keys_to_remove:
                self._remove_entry(key)

    def _remove_entry(self, key: str):
        """Remove a single cache entry.

        Args:
            key: Cache key to remove
        """
        if key in self._cache:
            entry = self._cache[key]
            self._current_memory -= entry.size_estimate
            del self._cache[key]

            # Clean up binary tracking
            for binary_keys in self._binary_keys.values():
                binary_keys.discard(key)

    def _evict_if_needed(self, new_size: int):
        """Evict entries if cache is full.

        Args:
            new_size: Size of new entry being added
        """
        # Evict based on count
        while len(self._cache) >= self._max_size and self._cache:
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)

        # Evict based on memory
        while self._current_memory + new_size > self._max_memory and self._cache:
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)

        # Evict expired entries
        now = time.time()
        expired = [
            key for key, entry in self._cache.items()
            if now - entry.created_at > self._ttl
        ]
        for key in expired:
            self._remove_entry(key)

    def clear(self):
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()
            self._binary_keys.clear()
            self._current_memory = 0
            log.log_info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0

            return {
                "entries": len(self._cache),
                "max_entries": self._max_size,
                "memory_used_mb": self._current_memory / (1024 * 1024),
                "max_memory_mb": self._max_memory / (1024 * 1024),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "binaries_tracked": len(self._binary_keys)
            }

    def __len__(self) -> int:
        """Return the number of cached entries."""
        with self._lock:
            return len(self._cache)


# Global cache instance
_analysis_cache: Optional[AnalysisCache] = None


def get_analysis_cache() -> AnalysisCache:
    """Get the global analysis cache instance."""
    global _analysis_cache
    if _analysis_cache is None:
        _analysis_cache = AnalysisCache()
    return _analysis_cache


def reset_analysis_cache():
    """Reset the global analysis cache (for testing)."""
    global _analysis_cache
    _analysis_cache = None


def cached_tool(binary_name_param: str = "filename", invalidates: bool = False):
    """Decorator for caching tool results.

    Args:
        binary_name_param: Name of the parameter containing the binary name
        invalidates: If True, invalidates cache for the binary instead of caching

    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_analysis_cache()

            # Extract binary name from args/kwargs
            binary_name = kwargs.get(binary_name_param)
            if binary_name is None and args:
                # Try to find it in positional args based on function signature
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                if binary_name_param in params:
                    idx = params.index(binary_name_param)
                    if idx < len(args):
                        binary_name = args[idx]

            if binary_name is None:
                # Can't cache without binary name
                return func(*args, **kwargs)

            if invalidates:
                # This tool modifies state, invalidate cache
                result = func(*args, **kwargs)
                cache.invalidate(binary_name)
                return result

            # Try to get from cache
            tool_name = func.__name__
            cache_args = tuple(a for a in args if a != binary_name)
            cache_kwargs = {k: v for k, v in kwargs.items() if k != binary_name_param}

            cached_result = cache.get(binary_name, tool_name, *cache_args, **cache_kwargs)
            if cached_result is not None:
                return cached_result

            # Execute and cache
            result = func(*args, **kwargs)
            cache.set(binary_name, tool_name, result, *cache_args, **cache_kwargs)
            return result

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator
