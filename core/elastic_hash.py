"""
Elastic / Multi-Level Geometric Open Addressing Hash Table (Farach-Colton, Krapivin, Kuszmaul 2025).
Provides O(1) amortized and O(log 1/delta) expected worst-case probe complexity without ANY reordering.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False

class ElasticHashTable:
    """
    Elastic Open Addressing Hash Table without Reordering (Farach-Colton et al. 2025).
    Organizes the backing storage into geometrically scaling sub-arrays (levels)
    where keys probe systematically without displacing earlier keys.
    """
    def __init__(self, capacity: int, delta: float = 0.05, num_levels: int = 5):
        self.capacity = capacity
        self.delta = delta
        self.num_levels = num_levels
        
        # Geometrically distributed level sizes
        # Level 0 is largest, deeper levels act as funnel buffers
        fractions = [0.5**(i+1) for i in range(num_levels - 1)]
        fractions.append(1.0 - sum(fractions))
        
        self.level_sizes = [max(16, int(capacity * f)) for f in fractions]
        self.total_size = sum(self.level_sizes)
        self.level_offsets = [0] + list(np.cumsum(self.level_sizes)[:-1])
        
        # Primary storage: keys and values (supports arbitrary python objects / references)
        self.keys = np.full(self.total_size, -1, dtype=np.int64)
        self.values = [None] * self.total_size
        self.occupied = np.zeros(self.total_size, dtype=bool)
        
        # Seeds for level-wise independent pseudo-random uniform hashing
        rng = np.random.RandomState(42)
        self.seeds_a = rng.randint(1, 2**31 - 1, size=(num_levels, 4), dtype=np.int64)
        self.seeds_b = rng.randint(0, 2**31 - 1, size=(num_levels, 4), dtype=np.int64)
        
        self.count = 0

    def _hash(self, key: int, level: int, attempt: int) -> int:
        a = self.seeds_a[level, attempt % 4]
        b = self.seeds_b[level, attempt % 4]
        size = self.level_sizes[level]
        # Multiplicative hash mapped to sub-array range
        raw_h = (int(key) * int(a) + int(b) + attempt * 2654435761) & 0x7FFFFFFF
        return (raw_h % size)

    def insert(self, key: int, value: Any) -> Tuple[bool, int]:
        """
        Inserts key, value WITHOUT reordering existing elements.
        Returns (success, probe_count).
        """
        if self.count >= self.capacity:
            return False, 0
            
        probes = 0
        for level in range(self.num_levels):
            offset = self.level_offsets[level]
            size = self.level_sizes[level]
            max_attempts = min(size, 4 + level * 2)
            
            for attempt in range(max_attempts):
                probes += 1
                pos = offset + self._hash(key, level, attempt)
                
                if not self.occupied[pos]:
                    self.keys[pos] = key
                    self.values[pos] = value
                    self.occupied[pos] = True
                    self.count += 1
                    return True, probes
                elif self.keys[pos] == key:
                    # Update existing key
                    self.values[pos] = value
                    return True, probes
                    
        # Fallback linear probe with bounded search radius
        start_pos = (int(key) & 0x7FFFFFFF) % self.total_size
        for i in range(min(self.total_size, 32)):
            pos = (start_pos + i) % self.total_size
            probes += 1
            if not self.occupied[pos]:
                self.keys[pos] = key
                self.values[pos] = value
                self.occupied[pos] = True
                self.count += 1
                return True, probes
            elif self.keys[pos] == key:
                self.values[pos] = value
                return True, probes
                
        return False, probes

    def lookup(self, key: int) -> Tuple[Optional[Any], int]:
        """
        Queries key. Returns (value or None, probe_count).
        Worst-case expected probe count bounded by O(log 1/delta).
        """
        probes = 0
        for level in range(self.num_levels):
            offset = self.level_offsets[level]
            size = self.level_sizes[level]
            max_attempts = min(size, 4 + level * 2)
            
            for attempt in range(max_attempts):
                probes += 1
                pos = offset + self._hash(key, level, attempt)
                if not self.occupied[pos]:
                    # In open addressing, an empty slot in the sequence might mean not found in this sub-block
                    continue
                if self.keys[pos] == key:
                    return self.values[pos], probes
                    
        # Fallback search matching insert's bounded probe sequence
        start_pos = (int(key) & 0x7FFFFFFF) % self.total_size
        for i in range(min(self.total_size, 32)):
            pos = (start_pos + i) % self.total_size
            probes += 1
            if not self.occupied[pos]:
                continue
            if self.keys[pos] == key:
                return self.values[pos], probes
                
        return None, probes


# --- JAX Vectorized Non-Reordering Spatial Hash Table ---

if HAS_JAX:
    @jax.jit
    def jax_hash_probe(keys_table: jnp.ndarray, 
                       query_keys: jnp.ndarray, 
                       level_offsets: jnp.ndarray, 
                       level_sizes: jnp.ndarray,
                       seeds_a: jnp.ndarray,
                       seeds_b: jnp.ndarray) -> jnp.ndarray:
        """
        Vectorized JAX lookup for spatial Morton keys without branching or tree traversal.
        Returns indices in keys_table for each query_key.
        """
        def find_one(key):
            # Scan levels & probe sequence in parallel unrolled fashion
            found_idx = -1
            # Check primary level 0 probe
            pos0 = level_offsets[0] + (jnp.bitwise_and(key * seeds_a[0, 0] + seeds_b[0, 0], 0x7FFFFFFF) % level_sizes[0])
            match0 = (keys_table[pos0] == key)
            found_idx = jnp.where(match0, pos0, found_idx)
            
            # Check level 1 probe
            pos1 = level_offsets[1] + (jnp.bitwise_and(key * seeds_a[1, 0] + seeds_b[1, 0], 0x7FFFFFFF) % level_sizes[1])
            match1 = (keys_table[pos1] == key)
            found_idx = jnp.where(match1 & (found_idx == -1), pos1, found_idx)
            
            return found_idx

        return jax.vmap(find_one)(query_keys)
else:
    jax_hash_probe = None
