"""
Fast Co-Optimal Transport (CO-OT) & Fused Gromov-Wasserstein Engine (`co_optimal_transport.py`)
==============================================================================================
Dense O(N^2 + M^2) reference Entropic Gromov-Wasserstein & Co-Optimal Transport.
The "linear-time O(N + M)" claim is aspirational: compute_pairwise_distances builds a dense
N x N (and M x M) distance matrix, and each GW iteration performs dense N x N matmuls, so the
implemented cost is O(N^2 + M^2) per iteration. Aligns heterogeneous multi-modal spaces of
differing dimensions and metric geometries without allocating dense 4-way N x N x M x M
interaction tensors.

Round-7 task X-A13 — honest scoping: this is a REFERENCE IMPLEMENTATION of entropic GW/FGW
with dense cost matrices. Gromov-Wasserstein with dense cost matrices is inherently O(N^2)
per iteration; FMM does not apply here (the GW quadratic form is not a radial kernel sum).
A fast path would require restricting to the entropic-Sinkhorn COOT mode with a Gaussian
kernel cutoff (same pattern as ``optimal_transport_fmm.py``), which is NOT implemented in
this module. Do not cite this module as "linear-time" — the asymptotic cost is quadratic.

Key Applications:
- Multi-Omics Cross-Modal Alignment (e.g., 20,000-gene scRNA-seq to 3D Spatial Transcriptomics).
- Heterogeneous Graph & Manifold Matching (Cross-species protein interactomes, 3D shape morphing).
- Dual sample-feature co-clustering & domain adaptation across distinct feature spaces.
"""

from __future__ import annotations
import numpy as np
import time
from typing import Tuple, Dict, List, Optional, Any, Union


def compute_pairwise_distances(X: np.ndarray, metric: str = "sqeuclidean") -> np.ndarray:
    """Computes pairwise distance matrix within space X."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or len(X) == 0 or not np.all(np.isfinite(X)):
        raise ValueError("X must be a non-empty finite array with shape (N, D)")
    if metric == "sqeuclidean":
        diff = X[:, None, :] - X[None, :, :]
        return np.sum(diff ** 2, axis=-1)
    elif metric == "euclidean":
        diff = X[:, None, :] - X[None, :, :]
        return np.linalg.norm(diff, axis=-1)
    else:
        raise NotImplementedError(f"Metric {metric} not supported.")


class FastGromovWasserstein:
    """
    Reference Entropic Gromov-Wasserstein (GW) and Fused Gromov-Wasserstein (FGW) Solver.
    Uses separable tensor contractions to avoid the 4-way N x N x M x M tensor, but the
    implemented cost is O(k_iter * (N^2 + M^2)) per iteration due to dense N x N matmuls
    (X-A13: this is a reference implementation, not a linear-time fast path — see module
    docstring). The "spatial hash Gaussian convolutions" phrase in earlier versions was
    aspirational and has been removed.
    """
    def __init__(
        self,
        epsilon: float = 0.05,
        max_iter: int = 50,
        inner_sinkhorn_iters: int = 15,
        tol: float = 1e-5,
        alpha: float = 0.5, # Weight for FGW (1.0 = pure GW, 0.0 = pure Wasserstein)
    ):
        self.eps = float(epsilon)
        self.max_iter = int(max_iter)
        self.inner_sinkhorn_iters = int(inner_sinkhorn_iters)
        self.tol = float(tol)
        self.alpha = float(alpha)
        if not np.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if self.max_iter <= 0 or self.inner_sinkhorn_iters <= 0:
            raise ValueError("iteration counts must be positive")
        if not np.isfinite(self.tol) or self.tol < 0.0:
            raise ValueError("tol must be finite and non-negative")
        if not np.isfinite(self.alpha) or not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")

    def _sinkhorn_projection(
        self,
        cost_matrix: np.ndarray,
        p: np.ndarray,
        q: np.ndarray,
        P_init: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Computes entropic regularized optimal transport plan via Sinkhorn-Knopp."""
        cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        if cost_matrix.ndim != 2 or not np.all(np.isfinite(cost_matrix)):
            raise ValueError("cost_matrix must be a finite 2D array")
        N, M = cost_matrix.shape
        if p.shape != (N,) or q.shape != (M,) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)) or np.any(p < 0) or np.any(q < 0):
            raise ValueError("marginals must be finite non-negative vectors matching cost_matrix")
        if not np.isclose(np.sum(p), 1.0) or not np.isclose(np.sum(q), 1.0):
            raise ValueError("marginals must sum to one")
        # Log-domain stabilized Sinkhorn
        K_log = -cost_matrix / self.eps
        K_max = np.max(K_log)
        K = np.exp(K_log - K_max)

        u = np.ones(N, dtype=np.float64) / N
        v = np.ones(M, dtype=np.float64) / M

        for _ in range(self.inner_sinkhorn_iters):
            u = p / (np.matmul(K, v) + 1e-15)
            v = q / (np.matmul(K.T, u) + 1e-15)

        P = u[:, None] * K * v[None, :]
        return P / (np.sum(P) + 1e-15)

    def solve_gromov_wasserstein(
        self,
        C_X: np.ndarray, # (N, N) Intra-domain cost in source space X
        C_Y: np.ndarray, # (M, M) Intra-domain cost in target space Y
        p: Optional[np.ndarray] = None, # (N,) Source marginals (default: uniform)
        q: Optional[np.ndarray] = None, # (M,) Target marginals (default: uniform)
        M_feat: Optional[np.ndarray] = None, # (N, M) Optional linear feature cost for Fused GW
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solves Gromov-Wasserstein transport plan P in R^{N x M}.
        """
        C_X = np.asarray(C_X, dtype=np.float64)
        C_Y = np.asarray(C_Y, dtype=np.float64)
        if C_X.ndim != 2 or C_X.shape[0] != C_X.shape[1] or C_Y.ndim != 2 or C_Y.shape[0] != C_Y.shape[1]:
            raise ValueError("C_X and C_Y must be square cost matrices")
        if not np.all(np.isfinite(C_X)) or not np.all(np.isfinite(C_Y)):
            raise ValueError("C_X and C_Y must contain finite values")
        N = C_X.shape[0]
        M = C_Y.shape[0]
        if N == 0 or M == 0:
            raise ValueError("cost matrices must be non-empty")

        def normalized_mass(values: Optional[np.ndarray], size: int, name: str) -> np.ndarray:
            if values is None:
                return np.ones(size, dtype=np.float64) / size
            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim != 1 or len(arr) != size or not np.all(np.isfinite(arr)) or np.any(arr < 0.0) or np.sum(arr) <= 0.0:
                raise ValueError(f"{name} must be finite, non-negative, and have positive sum")
            return arr / np.sum(arr)

        p = normalized_mass(p, N, "p")
        q = normalized_mass(q, M, "q")

        if M_feat is not None:
            M_feat = np.asarray(M_feat, dtype=np.float64)
            if M_feat.shape != (N, M) or not np.all(np.isfinite(M_feat)):
                raise ValueError("M_feat must be a finite array with shape (N, M)")

        # Precompute constants for separable GW gradient:
        # L(C_X, C_Y, P) = const - 2 * tr(C_X * P * C_Y^T * P^T)
        C_X_sq = C_X ** 2
        C_Y_sq = C_Y ** 2

        # Initialize transport plan: outer product of marginals P_0 = p (x) q
        P = np.outer(p, q)

        history_gw_cost = []
        t0 = time.perf_counter()

        for it in range(self.max_iter):
            P_old = P.copy()

            # Separable Gradient tensor evaluation:
            # G = (C_X^2 @ p)[:, None] + (q @ (C_Y^2).T)[None, :] - 2 * (C_X @ P @ C_Y.T)
            term1 = np.matmul(C_X_sq, np.sum(P, axis=1, keepdims=True)) # (N, 1)
            term2 = np.matmul(np.sum(P, axis=0, keepdims=True), C_Y_sq.T) # (1, M)
            term3 = -2.0 * np.matmul(C_X, np.matmul(P, C_Y.T))            # (N, M)

            grad_gw = term1 + term2 + term3

            if M_feat is not None and self.alpha < 1.0:
                cost_total = (1.0 - self.alpha) * M_feat + self.alpha * grad_gw
            else:
                cost_total = grad_gw

            # Entropic proximal step via Sinkhorn
            P = self._sinkhorn_projection(cost_total, p, q, P_init=P)

            # Evaluate GW loss
            gw_cost = float(np.sum(grad_gw * P))
            history_gw_cost.append(gw_cost)

            # Check convergence
            diff = np.max(np.abs(P - P_old))
            if diff < self.tol:
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        meta = {
            "num_iterations": len(history_gw_cost),
            "final_gw_cost": history_gw_cost[-1] if history_gw_cost else 0.0,
            "cost_history": history_gw_cost,
            "elapsed_ms": elapsed_ms,
            "source_nodes": N,
            "target_nodes": M,
        }
        return P, meta


class FastCoOptimalTransport:
    """
    Co-Optimal Transport (CO-OT) Engine.
    Simultaneously optimizes sample transport plan P in R^{N x M} and feature coupling Q in R^{d_x x d_y}.
    """
    def __init__(
        self,
        eps_samples: float = 0.05,
        eps_features: float = 0.05,
        max_iter: int = 30,
        inner_sinkhorn_iters: int = 15,
        tol: float = 1e-4,
    ):
        self.eps_s = float(eps_samples)
        self.eps_f = float(eps_features)
        self.max_iter = int(max_iter)
        self.inner_sinkhorn_iters = int(inner_sinkhorn_iters)
        self.tol = float(tol)
        if not np.isfinite(self.eps_s) or self.eps_s <= 0.0 or not np.isfinite(self.eps_f) or self.eps_f <= 0.0:
            raise ValueError("eps_samples and eps_features must be finite and positive")
        if self.max_iter <= 0 or self.inner_sinkhorn_iters <= 0 or not np.isfinite(self.tol) or self.tol < 0.0:
            raise ValueError("iteration counts must be positive and tol must be finite and non-negative")

    def solve_co_optimal_transport(
        self,
        X: np.ndarray, # (N, d_x) Source dataset
        Y: np.ndarray, # (M, d_y) Target dataset
        p_samples: Optional[np.ndarray] = None, # (N,) Sample weights
        q_samples: Optional[np.ndarray] = None, # (M,) Target sample weights
        p_feats: Optional[np.ndarray] = None,   # (d_x,) Source feature weights
        q_feats: Optional[np.ndarray] = None,   # (d_y,) Target feature weights
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Solves CO-OT sample plan P (N, M) and feature plan Q (d_x, d_y).
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.ndim != 2 or Y.ndim != 2 or X.shape[0] == 0 or Y.shape[0] == 0 or X.shape[1] == 0 or Y.shape[1] == 0:
            raise ValueError("X and Y must be non-empty 2D arrays")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
            raise ValueError("X and Y must contain only finite values")
        N, d_x = X.shape
        M, d_y = Y.shape

        def normalized_mass(values: Optional[np.ndarray], size: int, name: str) -> np.ndarray:
            if values is None:
                return np.ones(size, dtype=np.float64) / size
            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim != 1 or len(arr) != size or not np.all(np.isfinite(arr)) or np.any(arr < 0.0) or np.sum(arr) <= 0.0:
                raise ValueError(f"{name} must be finite, non-negative, and have positive sum")
            return arr / np.sum(arr)

        p_s = normalized_mass(p_samples, N, "p_samples")
        q_s = normalized_mass(q_samples, M, "q_samples")
        p_f = normalized_mass(p_feats, d_x, "p_feats")
        q_f = normalized_mass(q_feats, d_y, "q_feats")

        # Initialize uniform couplings
        P = np.outer(p_s, q_s)
        Q = np.outer(p_f, q_f)

        t0 = time.perf_counter()
        history_loss = []

        X_sq = X ** 2
        Y_sq = Y ** 2

        for it in range(self.max_iter):
            P_old = P.copy()
            Q_old = Q.copy()

            # 1. Update Sample Plan P given Q:
            # Cost_P(i, j) = sum_{k, l} (X_{ik} - Y_{jl})^2 Q_{kl}
            # = (X_sq @ sum(Q, axis=1))_i + (sum(Q, axis=0) @ Y_sq^T)_j - 2 (X @ Q @ Y^T)_{ij}
            q_sum_cols = np.sum(Q, axis=1) # (d_x,)
            q_sum_rows = np.sum(Q, axis=0) # (d_y,)
            cost_P = (X_sq @ q_sum_cols)[:, None] + (Y_sq @ q_sum_rows)[None, :] - 2.0 * (X @ Q @ Y.T)

            # Sinkhorn for P; subtract the minimum cost to avoid underflow on sharp kernels.
            K_P = np.exp(np.clip(-(cost_P - np.min(cost_P)) / self.eps_s, -745.0, 0.0))
            u_s = np.ones(N, dtype=np.float64) / N
            v_s = np.ones(M, dtype=np.float64) / M
            for _ in range(self.inner_sinkhorn_iters):
                u_s = p_s / (np.matmul(K_P, v_s) + 1e-15)
                v_s = q_s / (np.matmul(K_P.T, u_s) + 1e-15)
            P = u_s[:, None] * K_P * v_s[None, :]
            P /= np.sum(P) + 1e-15

            # 2. Update Feature Plan Q given P:
            # Cost_Q(k, l) = sum_{i, j} (X_{ik} - Y_{jl})^2 P_{ij}
            # = (X_sq.T @ sum(P, axis=1))_k + (sum(P, axis=0) @ Y_sq)_l - 2 (X.T @ P @ Y)_{kl}
            p_sum_cols = np.sum(P, axis=1) # (N,)
            p_sum_rows = np.sum(P, axis=0) # (M,)
            cost_Q = (X_sq.T @ p_sum_cols)[:, None] + (Y_sq.T @ p_sum_rows)[None, :] - 2.0 * (X.T @ P @ Y)

            # Sinkhorn for Q; subtract the minimum cost to avoid underflow on sharp kernels.
            K_Q = np.exp(np.clip(-(cost_Q - np.min(cost_Q)) / self.eps_f, -745.0, 0.0))
            u_f = np.ones(d_x, dtype=np.float64) / d_x
            v_f = np.ones(d_y, dtype=np.float64) / d_y
            for _ in range(self.inner_sinkhorn_iters):
                u_f = p_f / (np.matmul(K_Q, v_f) + 1e-15)
                v_f = q_f / (np.matmul(K_Q.T, u_f) + 1e-15)
            Q = u_f[:, None] * K_Q * v_f[None, :]
            Q /= np.sum(Q) + 1e-15

            loss = float(np.sum(cost_P * P))
            history_loss.append(loss)

            if max(np.max(np.abs(P - P_old)), np.max(np.abs(Q - Q_old))) < self.tol:
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        meta = {
            "num_iterations": len(history_loss),
            "final_coot_loss": history_loss[-1] if history_loss else 0.0,
            "elapsed_ms": elapsed_ms,
            "loss_history": history_loss,
        }
        return P, Q, meta


# Standardized Aliases
CoOptimalTransport = FastCoOptimalTransport
GromovWasserstein = FastGromovWasserstein


if __name__ == "__main__":
    print("=" * 70)
    print("Fast Entropic Gromov-Wasserstein (GW) & Co-Optimal Transport Benchmark")
    print("=" * 70)

    # Synthetic manifold matching: 3D Swiss Roll (N=500) to 2D Disc (M=400)
    rng = np.random.RandomState(42)
    N, M = 500, 400
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.rand(N))
    X_3d = np.stack([t * np.cos(t), 21.0 * rng.rand(N), t * np.sin(t)], axis=-1)
    
    r_2d = np.sqrt(rng.rand(M))
    theta_2d = rng.rand(M) * 2.0 * np.pi
    Y_2d = np.stack([r_2d * np.cos(theta_2d), r_2d * np.sin(theta_2d)], axis=-1)

    print(f"Source Manifold (3D): N = {N} points")
    print(f"Target Manifold (2D): M = {M} points")

    C_X = compute_pairwise_distances(X_3d, metric="euclidean")
    C_Y = compute_pairwise_distances(Y_2d, metric="euclidean")

    gw_solver = FastGromovWasserstein(epsilon=0.02, max_iter=25)
    P_gw, meta_gw = gw_solver.solve_gromov_wasserstein(C_X, C_Y)

    print(f"Fast Gromov-Wasserstein Execution: {meta_gw['elapsed_ms']:.2f} ms ({meta_gw['num_iterations']} iters)")
    print(f"Final GW Divergence Metric      : {meta_gw['final_gw_cost']:.4e}")
    print(f"Transport Plan Shape            : {P_gw.shape} | Plan Sum = {np.sum(P_gw):.4f}")

    # Co-Optimal Transport Test
    coot_solver = FastCoOptimalTransport(eps_samples=0.05, eps_features=0.05, max_iter=15)
    P_coot, Q_coot, meta_coot = coot_solver.solve_co_optimal_transport(X_3d, Y_2d)
    print(f"\nCo-Optimal Transport Execution  : {meta_coot['elapsed_ms']:.2f} ms")
    print(f"Sample Coupling P Shape         : {P_coot.shape} | Feature Coupling Q Shape: {Q_coot.shape}")
    print("=" * 70)
