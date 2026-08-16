"""
3D Elastic Spatial Hash with Morton Z-Order Keying and Farach-Colton Non-Reordering Open Addressing.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, List, Dict, Optional, Any


def morton_part1by2_64(n: np.ndarray) -> np.ndarray:
    """Inserts two 0-bits after each bit of an integer for 3D Morton encoding."""
    n = n.astype(np.uint64) & np.uint64(0x1FFFFF)  # 21 bits
    n = (n | (n << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    n = (n | (n << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    n = (n | (n << np.uint64(8)))  & np.uint64(0x100F00F00F00F00F)
    n = (n | (n << np.uint64(4)))  & np.uint64(0x10C30C30C30C30C3)
    n = (n | (n << np.uint64(2)))  & np.uint64(0x1249249249249249)
    return n


def morton_encode_3d(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
    """Computes 64-bit 3D Morton code from discrete integer coordinates."""
    return (morton_part1by2_64(ix) | (morton_part1by2_64(iy) << np.uint64(1)) | (morton_part1by2_64(iz) << np.uint64(2)))


def morton_compact1by2_64(m: int) -> int:
    """Extracts bits with 2 zero gaps for decoding."""
    x = int(m) & 0x1249249249249249
    x = (x ^ (x >> 2)) & 0x10C30C30C30C30C3
    x = (x ^ (x >> 4)) & 0x100F00F00F00F00F
    x = (x ^ (x >> 8)) & 0x1F0000FF0000FF
    x = (x ^ (x >> 16)) & 0x1F00000000FFFF
    x = (x ^ (x >> 32)) & 0x1FFFFF
    return int(x)


def morton_decode_3d(code: int) -> Tuple[int, int, int]:
    """Decodes 64-bit 3D Morton code into (ix, iy, iz)."""
    ix = morton_compact1by2_64(code)
    iy = morton_compact1by2_64(code >> 1)
    iz = morton_compact1by2_64(code >> 2)
    return ix, iy, iz


class ElasticSpatialHash3D:
    """
    Lock-Free Compatible 3D Spatial Open-Addressing Hash Table without Reordering.
    Indexes 3D atomic coordinates into uniform or multi-scale cells.
    """
    def __init__(self, cell_size: float = 6.0, capacity_hint: int = 16384, delta: float = 0.05):
        self.cell_size = float(cell_size)
        self.inv_cell_size = 1.0 / self.cell_size
        self.delta = float(delta)

        # Multi-level geometric sizes (Farach-Colton 2025)
        self.num_levels = 5
        fractions = [0.5**(i + 1) for i in range(self.num_levels - 1)]
        fractions.append(1.0 - sum(fractions))

        self.capacity = max(1024, int(capacity_hint / (1.0 - delta)))
        self.level_sizes = [max(32, int(self.capacity * f)) for f in fractions]
        self.total_size = sum(self.level_sizes)
        self.level_offsets = [0] + list(np.cumsum(self.level_sizes)[:-1])

        # Flat linear contiguous memory backing
        self.keys = np.full(self.total_size, -1, dtype=np.int64)
        self.values = [None] * self.total_size
        self.occupied = np.zeros(self.total_size, dtype=bool)

        rng = np.random.RandomState(1337)
        self.seeds_a = rng.randint(1, 2**31 - 1, size=(self.num_levels, 4), dtype=np.int64)
        self.seeds_b = rng.randint(0, 2**31 - 1, size=(self.num_levels, 4), dtype=np.int64)
        self.count = 0

    def _hash(self, key: int, level: int, attempt: int) -> int:
        a = self.seeds_a[level, attempt % 4]
        b = self.seeds_b[level, attempt % 4]
        size = self.level_sizes[level]
        raw_h = (int(key) * int(a) + int(b) + attempt * 2654435761) & 0x7FFFFFFF
        return (raw_h % size)

    def insert(self, key: int, value: Any) -> bool:
        """Inserts key-value pair without displacing any existing keys."""
        if self.count >= self.capacity:
            return False

        for level in range(self.num_levels):
            offset = self.level_offsets[level]
            size = self.level_sizes[level]
            max_attempts = min(size, 4 + level * 2)

            for attempt in range(max_attempts):
                pos = offset + self._hash(key, level, attempt)
                if not self.occupied[pos]:
                    self.keys[pos] = key
                    self.values[pos] = value
                    self.occupied[pos] = True
                    self.count += 1
                    return True
                elif self.keys[pos] == key:
                    self.values[pos] = value
                    return True

        # Fallback linear scan
        for pos in range(self.total_size):
            if not self.occupied[pos]:
                self.keys[pos] = key
                self.values[pos] = value
                self.occupied[pos] = True
                self.count += 1
                return True

        return False

    def lookup(self, key: int) -> Optional[Any]:
        """Queries value for Morton key in expected O(log 1/delta) time."""
        for level in range(self.num_levels):
            offset = self.level_offsets[level]
            size = self.level_sizes[level]
            max_attempts = min(size, 4 + level * 2)

            for attempt in range(max_attempts):
                pos = offset + self._hash(key, level, attempt)
                if not self.occupied[pos]:
                    continue
                if self.keys[pos] == key:
                    return self.values[pos]

        return None

    def build_from_coords(self, coords: np.ndarray, origin: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Assigns atomic coordinates to Morton cells and indexes into elastic hash table.
        Returns (atom_cell_indices, unique_morton_keys, cell_inverse_indices).
        """
        if origin is None:
            origin = np.min(coords, axis=0) - self.cell_size

        shifted = coords - origin
        ix = np.maximum(0, (shifted[:, 0] * self.inv_cell_size).astype(np.int64))
        iy = np.maximum(0, (shifted[:, 1] * self.inv_cell_size).astype(np.int64))
        iz = np.maximum(0, (shifted[:, 2] * self.inv_cell_size).astype(np.int64))

        morton_keys = morton_encode_3d(ix, iy, iz)
        unique_keys, inverse = np.unique(morton_keys, return_inverse=True)

        for cluster_id, key in enumerate(unique_keys):
            self.insert(int(key), cluster_id)

        return morton_keys, unique_keys, inverse
