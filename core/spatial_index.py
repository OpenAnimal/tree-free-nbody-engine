"""
CellIndex: the repo-wide spatial cell index.

One implementation of the pattern used across every domain folder:

  1. quantize positions to integer grid cells (2D: (iy<<12)|ix in [0,1)^2,
     3D: 10-bit Morton interleave of world-space cell coords),
  2. bucket the item ids per occupied cell,
  3. keep the elastic (non-reordering) hash as the AUTHORITATIVE occupied-cell
     index for membership and neighborhood probes -- `cell_id`, `bucket`,
     `neighbor_keys`, and `neighborhood_indices` all resolve occupancy via a
     hash probe per candidate key (never a dict scan),
  4. rebuild on every `build()` call: the append-only table cannot unlearn
     stale keys, so the table is recreated with a capacity sized to the real
     occupied-cell count (two-pass build).

Honesty note on the exception to (3): the full-set enumerators
`occupied_keys()`, `items()`, and `far_keys(key)` DO iterate the backing
`_buckets` dict (the elastic hash is not iterable in key order), so they are
O(occupied_cells) dict scans rather than hash probes. Membership and
per-neighborhood queries -- the hot paths -- remain hash-probe only.
"""

from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.elastic_hash import ElasticHashTable


def morton_1d_key(ix: int) -> int:
    """1D cell key: just the cell index directly (grid resolution up to 4096)."""
    return int(ix) & 0xFFF


def morton_2d_key(ix: int, iy: int) -> int:
    """Row key format: (iy << 12) | ix, grid resolution up to 4096."""
    return ((int(iy) & 0xFFF) << 12) | (int(ix) & 0xFFF)


def morton_3d_key(ix: int, iy: int, iz: int) -> int:
    """10-bit-per-axis Morton interleave (1024^3 cells)."""
    ix, iy, iz = int(ix) & 0x3FF, int(iy) & 0x3FF, int(iz) & 0x3FF
    m = 0
    for b in range(10):
        m |= ((ix & (1 << b)) << (2 * b)) | ((iy & (1 << b)) << (2 * b + 1)) | ((iz & (1 << b)) << (2 * b + 2))
    return m


def morton_nd_key(coords: Tuple[int, ...], bits: int) -> int:
    """Interleave arbitrary-dimensional integer coordinates into one key.

    The specialized 1D/2D/3D encodings above remain byte-for-byte compatible.
    This generic path is used for higher-dimensional neural-operator inputs;
    callers must keep ``len(coords) * bits <= 63`` so keys remain valid signed
    64-bit hash keys.
    """
    if bits < 1 or len(coords) < 1 or len(coords) * bits > 63:
        raise ValueError("invalid Morton dimensions/bit budget")
    mask = (1 << bits) - 1
    key = 0
    for b in range(bits):
        for axis, coord in enumerate(coords):
            key |= ((int(coord) & mask) >> b & 1) << (b * len(coords) + axis)
    return key


class CellIndex:
    """
    Authoritative spatial cell index over an elastic hash table.

    Parameters
    ----------
    dims : any positive integer.
    grid_res : "unit" mode (default) — positions live in [0,1)^dims and are
        quantized with floor(p * grid_res). 1D/2D/3D keep their historical
        compact key formats; higher dimensions use generic Morton keys.
    cell_size : pass explicitly for "world" mode — positions are world units,
        quantized as floor(p / cell_size) + 512 into 1024 cells per axis.

    Note on `grid_res` semantics (Round-7 task T-C8 / finding R7-F30):
        In this class `grid_res` is exactly cells-per-side (linear). The
        `RadialTaylorFMM` engines pass `grid_res=depth` and treat `depth` as
        cells-per-side linearly. The neural-ops modules
        (`multipole_attention.py`, `flash_multipole_kernel.py`,
        `continuous_meshfree_gnn.py`) use a DIFFERENT convention:
        `grid_res = 1 << grid_depth` (exponential). The same word
        (`depth`/`grid_depth`) therefore means different things one import
        apart — check each module's convention before mixing.
    """

    def __init__(self, dims: int = 2, grid_res: int = 32, cell_size: Optional[float] = None):
        if int(dims) < 1:
            raise ValueError("dims must be positive")
        self.dims = int(dims)
        self.grid_res = int(grid_res)
        if self.grid_res < 1:
            raise ValueError("grid_res must be positive")
        self._morton_bits = (int(np.ceil(np.log2(self.grid_res)))
                             if self.grid_res > 1 else 1)
        if self.dims > 3 and self.dims * self._morton_bits > 63:
            raise ValueError(
                f"{self.dims}D grid_res={self.grid_res} exceeds the 63-bit "
                "generic Morton key budget")
        self.unit_mode = cell_size is None
        if self.unit_mode:
            self.cell_size = 1.0
        else:
            if cell_size <= 0:
                raise ValueError("cell_size must be positive in world mode")
            self.cell_size = float(cell_size)
        if self.unit_mode and (self.dims == 3 and self.grid_res > 1024):
            raise ValueError("3D unit mode supports grid_res <= 1024")
        # 1D and 2D keys use 12-bit per-axis masks (morton_1d_key and
        # morton_2d_key mask with 0xFFF), so grid_res > 4096 would silently
        # alias distinct cells to the same key. Reject explicitly.
        if self.unit_mode and self.dims in (1, 2) and self.grid_res > 4096:
            raise ValueError(
                f"{self.dims}D unit mode supports grid_res <= 4096 (12-bit "
                f"per-axis key mask); got {self.grid_res}")
        self.hash_table = ElasticHashTable(capacity=16, delta=0.05)
        self._buckets: Dict[int, List[int]] = {}
        self._cell_ids: Dict[int, int] = {}

    def _quantize_axis(self, values: np.ndarray) -> np.ndarray:
        """Quantize one coordinate column to integer cell ids (clipped)."""
        if self.unit_mode:
            hi = self.grid_res - 1
            return np.clip(np.floor(values * self.grid_res), 0, hi).astype(np.int64)
        return np.clip(np.floor(values / self.cell_size) + 512, 0, 1023).astype(np.int64)

    def _axis_limit(self) -> int:
        return self.grid_res - 1 if self.unit_mode else 1023

    # ------------------------------------------------------------------ #
    # Quantization
    # ------------------------------------------------------------------ #

    def key_of(self, pos) -> int:
        """Cell key of a single position."""
        p = np.asarray(pos, dtype=np.float64)
        if self.dims == 1:
            return morton_1d_key(int(self._quantize_axis(p[:1])[0]))
        if self.dims == 2:
            return morton_2d_key(int(self._quantize_axis(p[:1])[0]),
                                 int(self._quantize_axis(p[1:2])[0]))
        if self.dims == 3:
            return morton_3d_key(int(self._quantize_axis(p[:1])[0]),
                                 int(self._quantize_axis(p[1:2])[0]),
                                 int(self._quantize_axis(p[2:3])[0]))
        coords = tuple(int(v) for v in self._quantize_axis(p[:self.dims]))
        return morton_nd_key(coords, self._morton_bits)

    def key_ints(self, key: int) -> Tuple[int, ...]:
        """Decode a key back to integer cell coordinates."""
        if self.dims == 1:
            return (int(key) & 0xFFF,)
        if self.dims == 2:
            return (key & 0xFFF, key >> 12)
        if self.dims == 3:
            ix = iy = iz = 0
            for b in range(10):
                ix |= (key >> (2 * b)) & (1 << b)
                iy |= (key >> (2 * b + 1)) & (1 << b)
                iz |= (key >> (2 * b + 2)) & (1 << b)
            return (ix, iy, iz)
        coords = [0] * self.dims
        for b in range(self._morton_bits):
            for axis in range(self.dims):
                coords[axis] |= ((int(key) >> (b * self.dims + axis)) & 1) << b
        return tuple(coords)

    def _neighbor_key(self, key: int, d):
        """Shifted key, or None if the shifted cell falls outside the grid."""
        ints = list(self.key_ints(key))
        limit = self._axis_limit()
        for ax, delta in enumerate(d):
            ints[ax] += delta
            if ints[ax] < 0 or ints[ax] > limit:
                return None
        if self.dims == 1:
            return morton_1d_key(ints[0])
        if self.dims == 2:
            return morton_2d_key(ints[0], ints[1])
        if self.dims == 3:
            return morton_3d_key(*ints)
        return morton_nd_key(tuple(ints), self._morton_bits)

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #

    def build(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rebuilds the index for `positions` ((N, dims)). Returns
        (unique_keys, inverse) where inverse maps each item to its bucket id.
        """
        positions = np.asarray(positions, dtype=np.float64)
        n = len(positions)
        self._buckets.clear()
        if self.dims == 1:
            ix = self._quantize_axis(positions[:, 0])
            keys = ix
        elif self.dims == 2:
            ix = self._quantize_axis(positions[:, 0])
            iy = self._quantize_axis(positions[:, 1])
            keys = (iy << 12) | ix
        elif self.dims == 3:
            ix = self._quantize_axis(positions[:, 0])
            iy = self._quantize_axis(positions[:, 1])
            iz = self._quantize_axis(positions[:, 2])
            keys = np.zeros(n, dtype=np.int64)
            for b in range(10):
                keys |= ((ix & (1 << b)) << (2 * b)) | ((iy & (1 << b)) << (2 * b + 1)) | ((iz & (1 << b)) << (2 * b + 2))
        else:
            axes = [self._quantize_axis(positions[:, axis]) for axis in range(self.dims)]
            keys = np.asarray([
                morton_nd_key(tuple(int(axis[i]) for axis in axes), self._morton_bits)
                for i in range(n)
            ], dtype=np.uint64)

        unique_keys, inverse = np.unique(keys, return_inverse=True)
        for c_id, k in enumerate(unique_keys):
            self._buckets[int(k)] = []
        self._cell_ids = {int(k): c for c, k in enumerate(unique_keys)}
        # Two-pass build: size the hash to the real occupied-cell count.
        self.hash_table = ElasticHashTable(capacity=max(16, 2 * len(unique_keys)), delta=0.05)
        for c_id, k in enumerate(unique_keys):
            ok, _ = self.hash_table.insert(int(k), c_id)
            if not ok:
                raise RuntimeError("elastic hash full while building CellIndex")
        order = np.argsort(inverse, kind="stable")
        sorted_inv = inverse[order]
        boundaries = np.searchsorted(sorted_inv, np.arange(len(unique_keys)))
        for c_id, k in enumerate(unique_keys):
            lo = boundaries[c_id]
            hi = boundaries[c_id + 1] if c_id + 1 < len(unique_keys) else n
            self._buckets[int(k)] = order[lo:hi]
        return unique_keys, inverse

    # ------------------------------------------------------------------ #
    # Queries (hash-probed only)
    # ------------------------------------------------------------------ #

    def __contains__(self, key: int) -> bool:
        return self.hash_table.get(int(key)) is not None

    def __len__(self) -> int:
        return len(self._buckets)

    def cell_id(self, key: int) -> Optional[int]:
        """Bucket id for an occupied key, or None (single elastic-hash probe)."""
        return self.hash_table.get(int(key))

    def bucket(self, key: int) -> Optional[np.ndarray]:
        """Item ids in the cell `key`, or None if unoccupied."""
        if not self.__contains__(key):
            return None
        return self._buckets[int(key)]

    def occupied_keys(self) -> List[int]:
        return list(self._buckets.keys())

    def items(self):
        """Iterate (key, item-id array) over occupied cells."""
        return self._buckets.items()

    def neighbor_keys(self, key: int, ring: int = 1) -> List[int]:
        """Occupied keys within Chebyshev ring `ring` of `key` (inclusive)."""
        out = []
        if self.dims == 1:
            deltas = [(dx,) for dx in range(-ring, ring + 1)]
        elif self.dims == 2:
            deltas = [(dx, dy) for dx in range(-ring, ring + 1) for dy in range(-ring, ring + 1)]
        elif self.dims == 3:
            deltas = [(dx, dy, dz) for dx in range(-ring, ring + 1)
                      for dy in range(-ring, ring + 1) for dz in range(-ring, ring + 1)]
        else:
            deltas = product(range(-ring, ring + 1), repeat=self.dims)
        for d in deltas:
            nk = self._neighbor_key(key, d)
            if nk is not None and self.hash_table.get(nk) is not None:
                out.append(nk)
        return out

    def neighborhood_indices(self, key: int, ring: int = 1) -> np.ndarray:
        """Concatenated item ids of all occupied cells within `ring` of `key`."""
        parts = [self._buckets[nk] for nk in self.neighbor_keys(key, ring)]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    def far_keys(self, key: int, ring: int = 1) -> List[int]:
        """Occupied keys OUTSIDE the ring neighborhood of `key`."""
        near = set(self.neighbor_keys(key, ring))
        return [k for k in self._buckets if k not in near]

    # ------------------------------------------------------------------ #
    # Cluster moments
    # ------------------------------------------------------------------ #

    def moments(self, positions: np.ndarray, weights: Optional[np.ndarray] = None):
        """
        Per occupied cell: (keys, counts, centroids, total_weights).
        `weights` defaults to ones. Centroid = weighted mean position.
        """
        positions = np.asarray(positions, dtype=np.float64)
        n = len(positions)
        w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
        keys = sorted(self._buckets.keys())
        k_index = {k: c for c, k in enumerate(keys)}
        inv = np.empty(n, dtype=np.int64)
        counts = np.empty(len(keys))
        centroids = np.empty((len(keys), self.dims))
        totals = np.empty(len(keys))
        for c, k in enumerate(keys):
            idx = self._buckets[k]
            inv[idx] = c
            counts[c] = len(idx)
            totals[c] = np.sum(w[idx])
            centroids[c] = np.sum(positions[idx] * w[idx, None], axis=0) / max(1e-12, totals[c])
        return keys, inv, counts, centroids, totals
