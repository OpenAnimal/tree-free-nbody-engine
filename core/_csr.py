"""CSR (Compressed Sparse Row) helper for cell-list particle grouping.

Round-7 task T-E4: hoisted from the T-C3 pattern for reuse across engines.
Given `inverse` (particle -> cluster id) and `K` (num clusters), produces
CSR arrays `cell_start` (K+1,) and `cell_particles` (N,) such that
particles in cluster c are `cell_particles[cell_start[c]:cell_start[c+1]]`.
"""
import numpy as np


def build_csr(inverse: np.ndarray, K: int):
    """Build CSR cell lists from a particle->cluster mapping.

    Returns (cell_start, cell_particles, sorted_order) where:
      cell_start: (K+1,) int64 — prefix sum of cell counts
      cell_particles: (N,) int64 — particle indices sorted by cluster
      sorted_order: (N,) int64 — the argsort that produced cell_particles
    """
    N = len(inverse)
    counts = np.bincount(inverse, minlength=K).astype(np.int64)
    cell_start = np.zeros(K + 1, dtype=np.int64)
    np.cumsum(counts, out=cell_start[1:])
    sorted_order = np.argsort(inverse, kind="stable")
    cell_particles = sorted_order  # particles sorted by cluster id
    return cell_start, cell_particles, sorted_order
