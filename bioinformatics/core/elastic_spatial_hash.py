"""
3D Elastic Spatial Hash with Morton Z-Order Keying and Farach-Colton Non-Reordering Open Addressing.

Round-7 task T-A2: `ElasticSpatialHash3D` is now a thin façade over
`core.elastic_hash.ElasticHashTable` (the funnel-hash implementation). The
pre-funnel geometric-levels + linear-probing-fallback scheme that previously
lived here (finding F-01) has been removed. The Morton encode/decode utilities
are kept (they are correct 21-bit 3D Morton utilities used widely across the
bioinformatics modules).
"""

from __future__ import annotations
import os
import sys
import numpy as np
from typing import Tuple, List, Dict, Optional, Any

# Make `core` importable whether this file is loaded as part of a package or
# as a top-level script.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.elastic_hash import ElasticHashTable


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
    3D Spatial Open-Addressing Hash Table without Reordering.

    Façade over `core.elastic_hash.ElasticHashTable` (the funnel hash,
    Farach-Colton, Krapivin, & Kuszmaul, 2025). `insert/lookup/build_from_coords` signatures
    are unchanged from the legacy class; values are cluster ids (ints).
    """
    def __init__(self, cell_size: float = 6.0, capacity_hint: int = 16384, delta: float = 0.05):
        self.cell_size = float(cell_size)
        self.inv_cell_size = 1.0 / self.cell_size
        self.delta = float(delta)
        # Two-pass sizing: start with a small table; build_from_coords sizes
        # it to max(16, 2*K) where K is the real occupied-cell count.
        self.capacity_hint = int(capacity_hint)
        self._table: Optional[ElasticHashTable] = None
        # Eagerly allocate a small table so insert() works before a
        # build_from_coords call (backward compat for callers that insert
        # one key at a time).
        self._table = ElasticHashTable(
            capacity=max(16, int(capacity_hint)), delta=delta
        )

    @property
    def probe_bound(self) -> int:
        """Deterministic worst-case probe count of the underlying funnel table."""
        return self._table.probe_bound if self._table is not None else 0

    @property
    def count(self) -> int:
        return self._table.count if self._table is not None else 0

    def insert(self, key: int, value: Any) -> bool:
        """Inserts key-value pair without displacing any existing keys."""
        ok, _ = self._table.insert(int(key), value)
        return ok

    def lookup(self, key: int) -> Optional[Any]:
        """Queries value for Morton key (None if absent)."""
        val, _ = self._table.lookup(int(key))
        return val

    def lookup_with_probes(self, key: int) -> Tuple[Optional[Any], int]:
        """Public probe-exposing lookup (Round-7 task T-B8 / R7-F29).

        Returns (value, probe_count) so tests and callers can verify the
        probe-bound without poking the private `_table` attribute. Delegates
        to the underlying funnel table's `lookup`.
        """
        val, probes = self._table.lookup(int(key))
        return val, probes

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

        # Two-pass sizing: count unique keys (already done via np.unique) then
        # size the funnel table to max(16, 2*K).
        K = len(unique_keys)
        self._table = ElasticHashTable(
            capacity=max(16, 2 * K), delta=self.delta
        )
        for cluster_id, key in enumerate(unique_keys):
            ok, _ = self._table.insert(int(key), cluster_id)
            if not ok:
                raise RuntimeError(
                    "ElasticHashTable insert failed during build_from_coords "
                    f"(K={K}, capacity={max(16, 2 * K)})"
                )

        return morton_keys, unique_keys, inverse
