"""
Shared bucketing helper for neural_ops (`neural_ops/_bucketing.py`).

Wraps the canonical `core.spatial_index.CellIndex` (elastic-hash-backed
cell index) and provides per-cell moment assembly (P2M: centroid, mean-K,
sum-V, dipole, count) used by the multipole attention layers. This removes
~4 copy-pasted blocks across `multipole_attention.py`,
`continuous_meshfree_gnn.py`, and `equivariant_field_layer.py` (Round-7
task T-A1). Inside this repository the elastic hash
(`core/elastic_hash.py`) is the AUTHORITATIVE occupied-cell index — every
membership / neighborhood query goes through hash probes, never through
dict scans (finding F-01). When neural_ops/ is copied WITHOUT core/, the
dependency shim `_core_deps` substitutes a dict-backed fallback with
identical outputs (see `neural_ops/_core_deps.py`).
"""

from __future__ import annotations
import os
import sys
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Make the repo root importable whether this file is loaded as part of a
# package or as a top-level script (needed to reach canonical core/).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from neural_ops._core_deps import CellIndex


def build_cell_index(
    coords: np.ndarray,
    spatial_dim: int,
    grid_res: int,
) -> Tuple[CellIndex, np.ndarray, np.ndarray]:
    """
    Build a `CellIndex` for `coords` ((N, spatial_dim) in [0,1)^d).

    Returns
    -------
    idx : CellIndex
        Funnel-hash-backed occupied-cell index.
    unique_keys : ndarray
        Unique occupied cell keys.
    inverse : ndarray
        Cluster id per particle (maps each particle to its bucket).
    """
    idx = CellIndex(dims=spatial_dim, grid_res=grid_res)
    unique_keys, inverse = idx.build(coords)
    return idx, unique_keys, inverse


def compute_cluster_moments(
    coords: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    idx: CellIndex,
    inverse: np.ndarray,
    spatial_dim: int,
    D: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Per occupied cell: centroid, mean-K, sum-V, dipole (D x spatial_dim), count.

    Returns
    -------
    centers : (K_clusters, spatial_dim)
    mean_k  : (K_clusters, D)
    sum_v   : (K_clusters, D)
    dipoles : (K_clusters, D, spatial_dim)
    counts  : (K_clusters,)
    key_to_idx : dict mapping cell key -> cluster index
    """
    occupied = idx.occupied_keys()
    n_clusters = len(occupied)
    key_to_idx = {int(k): c for c, k in enumerate(occupied)}

    centers = np.zeros((n_clusters, spatial_dim), dtype=np.float32)
    mean_k = np.zeros((n_clusters, D), dtype=np.float32)
    sum_v = np.zeros((n_clusters, D), dtype=np.float32)
    dipoles = np.zeros((n_clusters, D, spatial_dim), dtype=np.float32)
    counts = np.zeros(n_clusters, dtype=np.float32)

    # Vectorized P2M: use np.add.at (unbuffered scatter) for the per-cluster
    # reductions, avoiding the per-cluster Python loop. This is the same
    # math as the loop above but O(N) in C instead of O(K * avg_bucket).
    inv = inverse.astype(np.int64)  # (N,) cluster id per particle
    N = coords.shape[0]

    # Counts per cluster
    np.add.at(counts, inv, 1.0)

    # Centroid: sum coords per cluster, then divide.
    coord_sum = np.zeros((n_clusters, spatial_dim), dtype=np.float64)
    np.add.at(coord_sum, inv, coords.astype(np.float64))
    centers = (coord_sum / np.maximum(counts[:, None], 1e-12)).astype(np.float32)

    # mean_k: sum K per cluster, then divide.
    k_sum = np.zeros((n_clusters, D), dtype=np.float64)
    np.add.at(k_sum, inv, K.astype(np.float64))
    mean_k = (k_sum / np.maximum(counts[:, None], 1e-12)).astype(np.float32)

    # sum_v: sum V per cluster.
    np.add.at(sum_v, inv, V.astype(np.float64))
    sum_v = sum_v.astype(np.float32)

    # Dipoles: sum_n V[n,:] * (coords[n,:] - center[cluster(n),:]) .
    # Vectorized: build per-particle delta, then scatter-add the outer products.
    delta = coords.astype(np.float64) - centers[inv].astype(np.float64)  # (N, spatial_dim)
    V_d = V.astype(np.float64)
    # dipole[c, d, s] = sum_{n in c} V[n,d] * delta[n,s]
    # Use np.add.at over the flattened (D * spatial_dim) outer product.
    outer = V_d[:, :, None] * delta[:, None, :]  # (N, D, spatial_dim)
    outer_flat = outer.reshape(N, D * spatial_dim)
    dipoles_flat = np.zeros((n_clusters, D * spatial_dim), dtype=np.float64)
    np.add.at(dipoles_flat, inv, outer_flat)
    dipoles = dipoles_flat.reshape(n_clusters, D, spatial_dim).astype(np.float32)

    return centers, mean_k, sum_v, dipoles, counts, key_to_idx
