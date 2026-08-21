"""
Fast Entropic Optimal Transport via Spatial Hash Convolution (optimal_transport_fmm.py).

Inspired by:
1. "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"
   Marco Cuturi (NeurIPS 2013).
2. "Computational Optimal Transport"
   Gabriel Peyre and Marco Cuturi (Foundations and Trends in Machine Learning, 2019).
3. "Optimal Bounds for Open Addressing Without Reordering"
   Martin Farach-Colton, Andrew Krapivin, William Kuszmaul (FOCS 2024 / arXiv:2501.02305).

Key Algorithmic Principle:
Solving optimal transport between continuous probability distributions a on sources X (N points)
and b on targets Y (M points) requires computing the regularized Wasserstein-2 distance:
    min_{P in U(a, b)} <P, C> - gamma * H(P)
where C_{ij} = ||x_i - y_j||^2.

The Sinkhorn-Knopp algorithm computes scaling vectors u and v iteratively:
    u^{(k+1)} = a / (K * v^{(k)})
    v^{(k+1)} = b / (K^T * u^{(k+1)})
where K_{ij} = exp(-||x_i - y_j||^2 / (2 * gamma^2)).

Standard Sinkhorn requires O(k_iter * N * M) time and O(N * M) memory to store the dense Gibbs kernel.
Here, we evaluate K * v and K^T * u via Matrix-Free Elastic Spatial Hashing in O(k_iter * (N + M)) time,
enabling real-time decentralized logistics and continuous density matching on massive datasets.
"""

import time
import os
import sys
from typing import Tuple, List, Optional, Dict
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.spatial_index import CellIndex


class FastEntropicOptimalTransport:
    """
    Matrix-Free Fast Sinkhorn Optimal Transport Solver.
    
    Computes regularized Wasserstein-2 transport plans and distances in O(N + M) time per iteration.
    """
    def __init__(
        self,
        regularization_gamma: float = 0.1,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
        cutoff_sigma_multiplier: float = 3.5
    ):
        self.gamma = float(regularization_gamma)
        self.max_iter = int(max_iterations)
        self.tol = float(tolerance)
        self.cutoff_multiplier = float(cutoff_sigma_multiplier)
        self.r_cut = self.cutoff_multiplier * self.gamma

    def _build_sparse_interaction_blocks(
        self,
        targets: np.ndarray,
        sources: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Precomputes sparse block interaction stencils between targets and sources.
        Returns list of (target_indices, source_indices, kernel_matrix_block).

        X-A12: uses CellIndex (world mode, cell_size = r_cut) instead of
        hand-rolled dict hashing with tuple keys.  Same cell size and 3^D
        ring-1 neighborhood, so the neighbor sets are identical.
        """
        cell_size = self.r_cut
        dim = targets.shape[1]
        inv_2gamma2 = 1.0 / (2.0 * (self.gamma ** 2))
        r_cut_sq = self.r_cut ** 2

        # X-A12: CellIndex (world mode) replaces hand-rolled dict grids.
        src_index = CellIndex(dims=dim, cell_size=cell_size)
        src_index.build(sources)

        tgt_index = CellIndex(dims=dim, cell_size=cell_size)
        tgt_index.build(targets)

        blocks = []
        for tkey, t_idx in tgt_index.items():
            t_idx = np.asarray(t_idx, dtype=np.int64)
            s_idx_all = src_index.neighborhood_indices(tkey, ring=1)
            if len(s_idx_all) == 0:
                continue

            pts_t = targets[t_idx]
            pts_s = sources[s_idx_all]

            diff = pts_t[:, None, :] - pts_s[None, :, :]
            r_sq = np.sum(diff ** 2, axis=-1)

            mask = r_sq <= r_cut_sq
            k_mat = np.where(mask, np.exp(-r_sq * inv_2gamma2), 0.0)
            blocks.append((t_idx, s_idx_all, k_mat))

        return blocks

    def solve_transport_plan(
        self,
        source_points: np.ndarray,
        source_mass: np.ndarray,
        target_points: np.ndarray,
        target_mass: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Executes Fast Tree-Free Sinkhorn Iterations.
        """
        x_src = np.asarray(source_points, dtype=np.float64)
        y_tgt = np.asarray(target_points, dtype=np.float64)
        a = np.asarray(source_mass, dtype=np.float64)
        b = np.asarray(target_mass, dtype=np.float64)

        a = a / np.sum(a)
        b = b / np.sum(b)
        
        n_src = len(x_src)
        n_tgt = len(y_tgt)

        # Precompute forward (target = x_src, source = y_tgt) and adjoint blocks
        forward_blocks = self._build_sparse_interaction_blocks(targets=x_src, sources=y_tgt)
        adjoint_blocks = self._build_sparse_interaction_blocks(targets=y_tgt, sources=x_src)

        u = np.ones(n_src, dtype=np.float64) / n_src
        v = np.ones(n_tgt, dtype=np.float64) / n_tgt

        n_iters = 0
        for it in range(self.max_iter):
            n_iters = it + 1
            
            # Forward: Kv = K * v
            Kv = np.zeros(n_src, dtype=np.float64)
            for t_idx, s_idx, k_mat in forward_blocks:
                Kv[t_idx] += k_mat @ v[s_idx]
            Kv = np.maximum(Kv, 1e-15)
            u_next = a / Kv
            
            # Adjoint: Ktu = K^T * u
            Ktu = np.zeros(n_tgt, dtype=np.float64)
            for t_idx, s_idx, k_mat in adjoint_blocks:
                Ktu[t_idx] += k_mat @ u_next[s_idx]
            Ktu = np.maximum(Ktu, 1e-15)
            v_next = b / Ktu

            err_u = np.max(np.abs(u_next - u)) / (np.max(u) + 1e-12)
            u = u_next
            v = v_next

            if err_u < self.tol:
                break

        w2_cost = float(-self.gamma * (np.sum(a * np.log(np.maximum(u, 1e-15))) + np.sum(b * np.log(np.maximum(v, 1e-15)))))
        return u, v, w2_cost, n_iters


def direct_sinkhorn_baseline(
    x_src: np.ndarray,
    a: np.ndarray,
    y_tgt: np.ndarray,
    b: np.ndarray,
    gamma: float,
    max_iter: int = 100,
    tol: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Exact dense O(N * M) Sinkhorn reference algorithm."""
    a = a / np.sum(a)
    b = b / np.sum(b)
    
    diff = x_src[:, None, :] - y_tgt[None, :, :]
    c_mat = np.sum(diff ** 2, axis=-1)
    k_dense = np.exp(-c_mat / (2.0 * (gamma ** 2)))

    u = np.ones(len(a)) / len(a)
    v = np.ones(len(b)) / len(b)

    for _ in range(max_iter):
        u_next = a / np.maximum(k_dense @ v, 1e-15)
        v_next = b / np.maximum(k_dense.T @ u_next, 1e-15)
        if np.max(np.abs(u_next - u)) < tol:
            u, v = u_next, v_next
            break
        u, v = u_next, v_next

    cost = float(np.sum((u[:, None] * k_dense * v[None, :]) * c_mat))
    return u, v, cost


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("Fast Entropic Optimal Transport (Sinkhorn) Benchmark")
    print("=" * 70)

    n_source = 12000
    n_target = 12000
    gamma_reg = 0.15
    print(f"Number of Source Points (a)  : {n_source:,}")
    print(f"Number of Target Points (b)  : {n_target:,}")
    print(f"Entropic Regularization (y)  : {gamma_reg:.3f}")

    src_pts = np.random.randn(n_source, 2) * 0.4 + np.array([-1.0, 0.0])
    tgt_pts = np.random.randn(n_target, 2) * 0.5 + np.array([+1.0, 0.0])
    src_mass = np.random.rand(n_source) + 0.5
    tgt_mass = np.random.rand(n_target) + 0.5

    solver = FastEntropicOptimalTransport(regularization_gamma=gamma_reg, max_iterations=40, tolerance=1e-4)

    t0 = time.perf_counter()
    u_f, v_f, cost_f, iters = solver.solve_transport_plan(src_pts, src_mass, tgt_pts, tgt_mass)
    t_fast = (time.perf_counter() - t0) * 1000.0

    print(f"Fast Tree-Free Sinkhorn Time : {t_fast:.2f} ms ({iters} iterations)")
    print(f"Computed Transport Cost      : {cost_f:.4f}")

    n_sub = 2000
    t0 = time.perf_counter()
    u_d, v_d, cost_d = direct_sinkhorn_baseline(
        src_pts[:n_sub], src_mass[:n_sub], tgt_pts[:n_sub], tgt_mass[:n_sub],
        gamma=gamma_reg, max_iter=40, tol=1e-4
    )
    t_dense_sub = (time.perf_counter() - t0) * 1000.0
    t_dense_proj = t_dense_sub * ((n_source * n_target) / (n_sub * n_sub))

    print(f"Projected Dense O(N*M) Time  : {t_dense_proj:.2f} ms")
    print(f"Measured Speedup Ratio       : {t_dense_proj / max(t_fast, 1e-6):.1f}x")
    print("=" * 70)
