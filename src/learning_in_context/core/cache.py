"""Two-tier caching system with compression and automatic invalidation."""

import hashlib
import json
import pickle
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .constants import MAX_CACHE_SIZE_GB


class CacheManager:
    """Two-tier caching system for efficient data access."""
    
    def __init__(
        self, 
        cache_dir: Path, 
        max_size_gb: float = MAX_CACHE_SIZE_GB, 
        eviction_policy: str = "lru",
        compression: bool = True
    ):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache storage
            max_size_gb: Maximum cache size in GB
            eviction_policy: Eviction policy ("lru", "lfu", "fifo")
            compression: Whether to compress cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.eviction_policy = eviction_policy
        self.compression = compression
        
        # Memory cache (tier 1)
        self._memory_cache = OrderedDict()
        self._memory_size = 0
        self._max_memory_size = 1024 * 1024 * 1024  # 1GB memory cache
        
        # Access tracking for eviction policies
        self._access_counts = {}
        self._access_times = {}
        
        # Load cache metadata
        self._metadata_path = self.cache_dir / ".cache_metadata.json"
        self._load_metadata()
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._memory_cache or self._disk_path(key).exists()
    
    def load(self, key: str) -> Any:
        """Load data from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data
            
        Raises:
            KeyError: If key not found in cache
        """
        # Check memory cache first
        if key in self._memory_cache:
            self._update_access(key)
            return self._memory_cache[key]
        
        # Check disk cache
        disk_path = self._disk_path(key)
        if disk_path.exists():
            data = self._load_from_disk(disk_path)
            self._add_to_memory_cache(key, data)
            self._update_access(key)
            return data
        
        raise KeyError(f"Key not found in cache: {key}")
    
    def save(self, key: str, data: Any) -> None:
        """Save data to cache.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        # Save to disk
        disk_path = self._disk_path(key)
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_to_disk(data, disk_path)
        
        # Add to memory cache
        self._add_to_memory_cache(key, data)
        
        # Update metadata
        self._update_access(key)
        self._save_metadata()
        
        # Check cache size and evict if necessary
        self._check_cache_size()
    
    def load_or_compute(self, key: str, compute_fn: callable) -> Any:
        """Load from cache or compute if not found.
        
        Args:
            key: Cache key
            compute_fn: Function to compute data if not cached
            
        Returns:
            Cached or computed data
        """
        try:
            return self.load(key)
        except KeyError:
            data = compute_fn()
            self.save(key, data)
            return data
    
    def invalidate(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern.
        
        Args:
            pattern: Glob pattern for keys to invalidate
            
        Returns:
            Number of entries invalidated
        """
        import fnmatch
        
        count = 0
        
        # Invalidate memory cache
        keys_to_remove = [k for k in self._memory_cache if fnmatch.fnmatch(k, pattern)]
        for key in keys_to_remove:
            del self._memory_cache[key]
            count += 1
        
        # Invalidate disk cache
        for path in self.cache_dir.rglob("*"):
            if path.is_file() and path.name != ".cache_metadata.json":
                key = str(path.relative_to(self.cache_dir))
                if fnmatch.fnmatch(key, pattern):
                    path.unlink()
                    count += 1
        
        return count
    
    def get_size(self) -> float:
        """Get total cache size in GB."""
        total_size = 0
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size / (1024 * 1024 * 1024)
    
    def clear(self) -> None:
        """Clear entire cache."""
        self._memory_cache.clear()
        self._memory_size = 0
        
        # Remove all files except metadata
        for path in self.cache_dir.rglob("*"):
            if path.is_file() and path.name != ".cache_metadata.json":
                path.unlink()
        
        self._access_counts.clear()
        self._access_times.clear()
        self._save_metadata()
    
    def _disk_path(self, key: str) -> Path:
        """Get disk path for cache key."""
        # Handle key with slashes as subdirectories
        key_parts = key.split("/")
        
        # Add file extension based on compression
        if self.compression:
            key_parts[-1] += ".pkl.gz"
        else:
            key_parts[-1] += ".pkl"
            
        return self.cache_dir / Path(*key_parts)
    
    def _load_from_disk(self, path: Path) -> Any:
        """Load data from disk."""
        if self.compression:
            import gzip
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        else:
            with open(path, "rb") as f:
                return pickle.load(f)
    
    def _save_to_disk(self, data: Any, path: Path) -> None:
        """Save data to disk."""
        if self.compression:
            import gzip
            with gzip.open(path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _add_to_memory_cache(self, key: str, data: Any) -> None:
        """Add data to memory cache with size management."""
        # Estimate data size
        data_size = self._estimate_size(data)
        
        # Evict from memory cache if necessary
        while self._memory_size + data_size > self._max_memory_size and self._memory_cache:
            self._evict_from_memory()
        
        # Add to cache
        self._memory_cache[key] = data
        self._memory_size += data_size
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data."""
        if isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, pd.DataFrame):
            return data.memory_usage(deep=True).sum()
        else:
            # Rough estimate using pickle
            return len(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
    
    def _evict_from_memory(self) -> None:
        """Evict item from memory cache based on policy."""
        if not self._memory_cache:
            return
        
        if self.eviction_policy == "lru":
            # Remove least recently used
            key = next(iter(self._memory_cache))
        elif self.eviction_policy == "lfu":
            # Remove least frequently used
            key = min(self._memory_cache.keys(), key=lambda k: self._access_counts.get(k, 0))
        else:  # fifo
            # Remove first in
            key = next(iter(self._memory_cache))
        
        data = self._memory_cache.pop(key)
        self._memory_size -= self._estimate_size(data)
    
    def _check_cache_size(self) -> None:
        """Check total cache size and evict if necessary."""
        current_size = self.get_size() * 1024 * 1024 * 1024  # Convert to bytes
        
        if current_size > self.max_size_bytes:
            # Get all cached files with access times
            cache_files = []
            for path in self.cache_dir.rglob("*"):
                if path.is_file() and path.name != ".cache_metadata.json":
                    key = str(path.relative_to(self.cache_dir))
                    access_time = self._access_times.get(key, 0)
                    cache_files.append((path, key, access_time))
            
            # Sort based on eviction policy
            if self.eviction_policy == "lru":
                cache_files.sort(key=lambda x: x[2])
            elif self.eviction_policy == "lfu":
                cache_files.sort(key=lambda x: self._access_counts.get(x[1], 0))
            else:  # fifo
                cache_files.sort(key=lambda x: x[0].stat().st_ctime)
            
            # Evict files until under limit
            for path, key, _ in cache_files:
                if current_size <= self.max_size_bytes:
                    break
                file_size = path.stat().st_size
                path.unlink()
                current_size -= file_size
                
                # Remove from tracking
                self._access_counts.pop(key, None)
                self._access_times.pop(key, None)
    
    def _update_access(self, key: str) -> None:
        """Update access tracking for key."""
        self._access_counts[key] = self._access_counts.get(key, 0) + 1
        self._access_times[key] = time.time()
        
        # Move to end for LRU in memory cache
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
    
    def _load_metadata(self) -> None:
        """Load cache metadata."""
        if self._metadata_path.exists():
            with open(self._metadata_path, "r") as f:
                metadata = json.load(f)
                self._access_counts = metadata.get("access_counts", {})
                self._access_times = metadata.get("access_times", {})
        else:
            self._access_counts = {}
            self._access_times = {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata."""
        metadata = {
            "access_counts": self._access_counts,
            "access_times": self._access_times
        }
        with open(self._metadata_path, "w") as f:
            json.dump(metadata, f)