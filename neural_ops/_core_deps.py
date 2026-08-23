"""Dependency shim: canonical core/ implementations when present,
self-contained fallbacks when neural_ops is copied standalone.

neural_ops deliberately keeps its algorithmic dependencies small. Three of
its modules reference the repository's canonical ``core/`` engines; every
other module is numpy-only. When the folder is used inside this repository
the canonical implementations are always used (elastic-hash-backed
``CellIndex``, the full radial-Taylor FGT engines). When ``neural_ops/`` is
copied into another codebase WITHOUT ``core/``, this module substitutes
dependency-free fallbacks so every public API still works:

- ``CellIndex``: dict-backed reimplementation of the exact public surface.
  Outputs (keys, buckets, neighborhoods, moments) are identical to the
  canonical class; the difference is internal — the canonical index probes
  an ``ElasticHashTable`` (O(1) worst-case), the fallback probes a dict.
- ``Gaussian2DFGT`` / ``Gaussian3DFGT``: exact direct O(N^2) evaluation of
  the same Gaussian transform the fast engines approximate:

      potentials[i] = sum_{j != i} q[j] * exp(-|x_i - x_j|^2 / h^2)

  (the near-field self-pair is masked in the canonical engines as well).
  The direct sum IS the accuracy reference of the FGT, so fallback results
  are the exact quantity — only the asymptotic speedup is lost. Copy
  ``core/`` alongside for the fast path.

``equivariant_field_layer``'s optional ``tayloryukawa`` kernel path and the
``infinite_multipole_memory_network`` example use larger core/ engines with
no compact exact fallback; they raise an informative ImportError when
``core/`` is absent (their other kernel paths remain fully standalone).

Parity of the fallbacks against the canonical implementations is pinned by
``tests/neural_ops/test_vendored_parity.py``.
"""

import numpy as np

try:  # canonical, elastic-hash-backed
    from core.spatial_index import CellIndex as _CoreCellIndex
    from core.gaussian2d_fgt import Gaussian2DFGT as _CoreGaussian2DFGT
    from core.gaussian2d_fgt import Gaussian3DFGT as _CoreGaussian3DFGT
    USING_CORE = True
except ImportError:  # standalone copy: use the fallbacks below
    _CoreCellIndex = None
    _CoreGaussian2DFGT = None
    _CoreGaussian3DFGT = None
    USING_CORE = False


# --------------------------------------------------------------------- #
# Fallback CellIndex (dict-backed; same public surface and outputs as
# core/spatial_index.py).
# --------------------------------------------------------------------- #

def _morton_1d_key(ix: int) -> int:
    return int(ix) & 0xFFF


def _morton_2d_key(ix: int, iy: int) -> int:
    return ((int(iy) & 0xFFF) << 12) | (int(ix) & 0xFFF)


def _morton_3d_key(ix: int, iy: int, iz: int) -> int:
    ix, iy, iz = int(ix) & 0x3FF, int(iy) & 0x3FF, int(iz) & 0x3FF
    m = 0
    for b in range(10):
        m |= ((ix & (1 << b)) << (2 * b)) | ((iy & (1 << b)) << (2 * b + 1)) | ((iz & (1 << b)) << (2 * b + 2))
    return m


class _FallbackCellIndex:
    """Dict-backed CellIndex with the canonical class's public surface.

    Quantization, key formats (12-bit 1D/2D, 10-bit Morton 3D), unit/world
    modes and all query outputs match ``core.spatial_index.CellIndex``;
    occupancy is resolved through a plain dict instead of the elastic hash.
    """

    def __init__(self, dims: int = 2, grid_res: int = 32, cell_size=None):
        if dims not in (1, 2, 3):
            raise ValueError("dims must be 1, 2, or 3")
        self.dims = dims
        self.grid_res = int(grid_res)
        self.unit_mode = cell_size is None
        if self.unit_mode:
            self.cell_size = 1.0
        else:
            if cell_size <= 0:
                raise ValueError("cell_size must be positive in world mode")
            self.cell_size = float(cell_size)
        if self.unit_mode and (self.dims == 3 and self.grid_res > 1024):
            raise ValueError("3D unit mode supports grid_res <= 1024")
        if self.unit_mode and self.dims in (1, 2) and self.grid_res > 4096:
            raise ValueError(
                f"{self.dims}D unit mode supports grid_res <= 4096 (12-bit "
                f"per-axis key mask); got {self.grid_res}")
        self._buckets = {}
        self._cell_ids = {}

    def _quantize_axis(self, values):
        if self.unit_mode:
            return np.clip(np.floor(values * self.grid_res), 0, self.grid_res - 1).astype(np.int64)
        return np.clip(np.floor(values / self.cell_size) + 512, 0, 1023).astype(np.int64)

    def _axis_limit(self):
        return self.grid_res - 1 if self.unit_mode else 1023

    def key_of(self, pos) -> int:
        p = np.asarray(pos, dtype=np.float64)
        if self.dims == 1:
            return _morton_1d_key(int(self._quantize_axis(p[:1])[0]))
        if self.dims == 2:
            return _morton_2d_key(int(self._quantize_axis(p[:1])[0]),
                                  int(self._quantize_axis(p[1:2])[0]))
        return _morton_3d_key(int(self._quantize_axis(p[:1])[0]),
                              int(self._quantize_axis(p[1:2])[0]),
                              int(self._quantize_axis(p[2:3])[0]))

    def key_ints(self, key):
        if self.dims == 1:
            return (int(key) & 0xFFF,)
        if self.dims == 2:
            return (key & 0xFFF, key >> 12)
        ix = iy = iz = 0
        for b in range(10):
            ix |= (key >> (2 * b)) & (1 << b)
            iy |= (key >> (2 * b + 1)) & (1 << b)
            iz |= (key >> (2 * b + 2)) & (1 << b)
        return (ix, iy, iz)

    def _neighbor_key(self, key, d):
        ints = list(self.key_ints(key))
        limit = self._axis_limit()
        for ax, delta in enumerate(d):
            ints[ax] += delta
            if ints[ax] < 0 or ints[ax] > limit:
                return None
        if self.dims == 1:
            return _morton_1d_key(ints[0])
        if self.dims == 2:
            return _morton_2d_key(ints[0], ints[1])
        return _morton_3d_key(*ints)

    def build(self, positions):
        positions = np.asarray(positions, dtype=np.float64)
        n = len(positions)
        self._buckets = {}
        if self.dims == 1:
            keys = self._quantize_axis(positions[:, 0])
        elif self.dims == 2:
            ix = self._quantize_axis(positions[:, 0])
            iy = self._quantize_axis(positions[:, 1])
            keys = (iy << 12) | ix
        else:
            ix = self._quantize_axis(positions[:, 0])
            iy = self._quantize_axis(positions[:, 1])
            iz = self._quantize_axis(positions[:, 2])
            keys = np.zeros(n, dtype=np.int64)
            for b in range(10):
                keys |= ((ix & (1 << b)) << (2 * b)) | ((iy & (1 << b)) << (2 * b + 1)) | ((iz & (1 << b)) << (2 * b + 2))
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        self._cell_ids = {int(k): c for c, k in enumerate(unique_keys)}
        order = np.argsort(inverse, kind="stable")
        sorted_inv = inverse[order]
        boundaries = np.searchsorted(sorted_inv, np.arange(len(unique_keys)))
        for c_id, k in enumerate(unique_keys):
            lo = boundaries[c_id]
            hi = boundaries[c_id + 1] if c_id + 1 < len(unique_keys) else n
            self._buckets[int(k)] = order[lo:hi]
        return unique_keys, inverse

    def __contains__(self, key):
        return int(key) in self._buckets

    def __len__(self):
        return len(self._buckets)

    def cell_id(self, key):
        return self._cell_ids.get(int(key))

    def bucket(self, key):
        return self._buckets.get(int(key))

    def occupied_keys(self):
        return list(self._buckets.keys())

    def items(self):
        return self._buckets.items()

    def neighbor_keys(self, key, ring=1):
        out = []
        if self.dims == 1:
            deltas = [(dx,) for dx in range(-ring, ring + 1)]
        elif self.dims == 2:
            deltas = [(dx, dy) for dx in range(-ring, ring + 1) for dy in range(-ring, ring + 1)]
        else:
            deltas = [(dx, dy, dz) for dx in range(-ring, ring + 1)
                      for dy in range(-ring, ring + 1) for dz in range(-ring, ring + 1)]
        for d in deltas:
            nk = self._neighbor_key(key, d)
            if nk is not None and nk in self._buckets:
                out.append(nk)
        return out

    def neighborhood_indices(self, key, ring=1):
        parts = [self._buckets[nk] for nk in self.neighbor_keys(key, ring)]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    def far_keys(self, key, ring=1):
        near = set(self.neighbor_keys(key, ring))
        return [k for k in self._buckets if k not in near]

    def moments(self, positions, weights=None):
        positions = np.asarray(positions, dtype=np.float64)
        n = len(positions)
        w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
        keys = sorted(self._buckets.keys())
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


# --------------------------------------------------------------------- #
# Fallback FGT: exact direct Gaussian transform (accuracy reference of
# the fast engine; self-pair masked exactly like the canonical near field).
# --------------------------------------------------------------------- #

class _GaussianFGTDirect:
    """Exact O(N^2) Gaussian transform.

    potentials[i] = sum_{j != i} q[j] * exp(-|x_i - x_j|^2 / h^2)

    Accepts the canonical constructor arguments (depth, p, h, ring_direct)
    and ignores the discretization ones — there is nothing to discretize.
    """

    def __init__(self, depth=6, p=8, h=0.2, ring_direct=2, dims=None):
        self.h = float(h)
        self.dims = dims
        self.depth = depth
        self.p = p

    def evaluate(self, positions, charges):
        positions = np.asarray(positions, dtype=np.float64)
        charges = np.asarray(charges, dtype=np.float64)
        n = len(positions)
        if n == 0:
            return np.empty(0, dtype=np.float64)
        h2 = self.h * self.h
        out = np.empty(n, dtype=np.float64)
        # Chunked to keep the (n, n) distance matrices bounded.
        block = 1024
        for i0 in range(0, n, block):
            i1 = min(i0 + block, n)
            diff = positions[i0:i1, None, :] - positions[None, :, :]
            r2 = np.sum(diff * diff, axis=-1)
            g = np.exp(-r2 / h2)
            g[np.arange(i1 - i0), np.arange(i0, i1)] = 0.0  # mask self-pair
            out[i0:i1] = g @ charges
        return out

    # build_operator / evaluate_prebuilt: the canonical FGT splits a
    # charge-independent build from per-charge contraction (the caller
    # evaluates many charge vectors against one position set). The direct
    # transform has no charge-independent work worth caching short of the
    # full (N, N) kernel matrix, so the "build" snapshots positions and
    # each evaluate_prebuilt recomputes chunked — same results, O(N^2)
    # per call.
    def build_operator(self, positions):
        positions = np.asarray(positions, dtype=np.float64)
        return {"positions": positions, "direct": True, "h": self.h}

    def evaluate_prebuilt(self, built, charges):
        return _GaussianFGTDirect(h=built["h"]).evaluate(built["positions"], charges)


class _Gaussian2DFGTDirect(_GaussianFGTDirect):
    dims = 2


class _Gaussian3DFGTDirect(_GaussianFGTDirect):
    dims = 3


CellIndex = _CoreCellIndex if USING_CORE else _FallbackCellIndex
Gaussian2DFGT = _CoreGaussian2DFGT if USING_CORE else _Gaussian2DFGTDirect
Gaussian3DFGT = _CoreGaussian3DFGT if USING_CORE else _Gaussian3DFGTDirect
