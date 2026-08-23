"""
Synthetic Low-Rank Tensor Contraction Demo (algebraic_multipole_tensor.py).

Inspired by:
1. "More Asymmetry Yields Faster Matrix Multiplication"
   Josh Alman, Ran Duan, Virginia Vassilevska Williams, Yinzhan Xu, Zixuan Xu, Renfei Zhou (arXiv:2404.16349, 2024/2025).
2. "New Bounds for Matrix Multiplication: from Alpha to Omega"
   Virginia Vassilevska Williams, Yinzhan Xu, Zixuan Xu, Renfei Zhou (SODA 2024).
3. "Finite Matrix Multiplication Algorithms from Infinite Groups"
   Henry Cohn, Christopher Umans et al. (FOCS / arXiv 2024/2025).

HONESTY NOTE -- this is a SYNTHETIC low-rank contraction demo, NOT a real FMM
Multipole-to-Local (M2L) operator. The "kernel" M(r) is not derived from any
physical 1/r (or harmonic) Taylor translation: U is a fixed QR-orthonormalised
Vandermonde-pattern matrix, the latent decay s(r) is a hand-constructed
1/dist^exps * cos(phase) factor, and the "dense" baseline
(``_build_dense_m2l_kernel``) RECONSTRUCTS THE SAME U * diag(s(r)) * U^T
factorisation the low-rank path uses. The dense-vs-low-rank "Rel Error"
reported by the benchmark is therefore ~1e-16 (floating-point round-off in
reordering the same algebra), NOT an approximation error of a true M2L
kernel. The speedup is real (O(P*R) vs O(P^2) contraction of the same
synthetic factorisation), but no claim is made that this approximates an
actual FMM M2L operator at bounded epsilon error.
"""

import time
from typing import Tuple, List, Optional, Dict
import numpy as np


class AsymmetricMultipoleTensor:
    """
    Synthetic low-rank tensor factorisation demo.

    Builds a fixed rank-R orthonormal basis U (P x R) from a Vandermonde
    pattern and a synthetic distance-dependent latent decay s(r), then
    contracts source moments through U * diag(s(r)) * U^T. This is a
    synthetic low-rank contraction, not a physical M2L operator.
    """
    def __init__(self, order: int = 4, dim: int = 3, target_rank: Optional[int] = None):
        self.order = order
        self.dim = dim

        # Total number of multi-index coefficients (p + 1)^dim or homogeneous monomial combinations
        self.n_coeffs = (order + 1) ** dim

        # Target rank for asymmetric compression
        if target_rank is None:
            # Heuristic rank scaling for the synthetic demo
            self.rank = max(4, int(self.n_coeffs ** 0.65))
        else:
            self.rank = min(target_rank, self.n_coeffs)

        self.U: Optional[np.ndarray] = None
        self.V: Optional[np.ndarray] = None
        self.singular_values: Optional[np.ndarray] = None
        self._calibrate_low_rank_basis()

    def _calibrate_low_rank_basis(self, n_sample_directions: int = 64):
        """
        Computes the universal spatial Taylor / multipole projection basis U (P x R)
        and associated polynomial decay modes.
        """
        degrees = np.arange(self.n_coeffs) % (self.order + 1)
        # Construct orthogonal polynomial basis over degree multi-indices
        poly_mat = np.zeros((self.n_coeffs, self.rank), dtype=np.float64)
        for r_idx in range(self.rank):
            poly_mat[:, r_idx] = (degrees ** r_idx) * np.exp(-0.2 * degrees)
            
        # Orthonormalize basis via QR decomposition
        q, _ = np.linalg.qr(poly_mat)
        self.U = q[:, :self.rank]  # (P, R)
        self.V = self.U
        self.singular_values = np.exp(-0.5 * np.arange(self.rank))

    def evaluate_latent_decay(self, r_vectors: np.ndarray) -> np.ndarray:
        """
        Evaluates the R-dimensional latent separation factors directly in O(N * R) operations
        WITHOUT constructing dense P x P matrices.
        
        Args:
            r_vectors: (N, 3) displacement vectors
        Returns:
            (N, R) latent scaling factors
        """
        dists = np.linalg.norm(r_vectors, axis=-1, keepdims=True) + 1e-12  # (N, 1)
        exps = 1.0 + np.arange(self.rank) / max(1.0, float(self.rank - 1))  # (R,)
        # Directional harmonics projection
        r_unit = r_vectors / dists
        dir_phase = r_unit[:, 0:1] + 0.5 * r_unit[:, 1:2] + 0.25 * r_unit[:, 2:3] if self.dim == 3 else np.ones_like(dists)
        
        modes = (1.0 + 0.15 * np.cos(np.arange(self.rank) * dir_phase)) / (dists ** exps)
        return modes * self.singular_values[np.newaxis, :]  # (N, R)

    def _build_dense_m2l_kernel(self, r: np.ndarray) -> np.ndarray:
        """
        Reconstructs the full dense P x P Taylor translation kernel from the separable tensor basis.
        """
        latent_sigma = self.evaluate_latent_decay(r[np.newaxis, :])[0]  # (R,)
        # Reconstruct M = U * diag(latent_sigma) * U^T
        dense_kernel = np.dot(self.U * latent_sigma[np.newaxis, :], self.U.T)
        return dense_kernel.astype(np.float64)

    def dense_m2l_eval(self, source_moments: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Naive dense O(P^2) contraction."""
        dense_kernel = self._build_dense_m2l_kernel(r)
        return np.dot(dense_kernel, source_moments)

    def fast_low_rank_eval(self, source_moments: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Asymmetric Low-Rank O(P * R) contraction.
        """
        latent_sigma = self.evaluate_latent_decay(r[np.newaxis, :])[0]  # (R,)
        latent_src = np.dot(self.U.T, source_moments)                  # (R,)
        latent_tgt = latent_src * latent_sigma                         # (R,)
        return np.dot(self.U, latent_tgt)                             # (P,)


class LowRankFarFieldContraction:
    """
    Batch Far-Field Contraction Engine for Multi-Cluster FMM Systems.
    """
    def __init__(self, order: int = 4, dim: int = 3, target_rank: Optional[int] = None):
        self.tensor_engine = AsymmetricMultipoleTensor(order=order, dim=dim, target_rank=target_rank)

    def contract_clusters(
        self,
        source_moments_batch: np.ndarray,
        source_centers: np.ndarray,
        target_centers: np.ndarray,
        method: str = "low_rank"
    ) -> Tuple[np.ndarray, float]:
        """
        Contracts all-pairs far-field clusters.
        
        Args:
            source_moments_batch: (N_sources, P) multipole moments
            source_centers: (N_sources, 3) 3D cluster positions
            target_centers: (N_targets, 3) 3D target cluster positions
            method: 'low_rank' (asymmetric O(P*R)) or 'dense' (naive O(P^2))
            
        Returns:
            (target_local_expansions, elapsed_ms)
        """
        n_sources = len(source_centers)
        n_targets = len(target_centers)
        p = self.tensor_engine.n_coeffs
        
        target_expansions = np.zeros((n_targets, p), dtype=np.float64)
        
        t0 = time.perf_counter()
        if method == "low_rank":
            # Batch projection to latent rank space for all sources: (N_sources, R)
            latent_sources = np.dot(source_moments_batch, self.tensor_engine.U)  # (N_s, R)
            latent_target_accum = np.zeros((n_targets, self.tensor_engine.rank), dtype=np.float64)
            
            for t_idx in range(n_targets):
                t_pos = target_centers[t_idx]
                r_vectors = t_pos[np.newaxis, :] - source_centers  # (N_s, 3)
                decay_sigma = self.tensor_engine.evaluate_latent_decay(r_vectors)  # (N_s, R)
                latent_target_accum[t_idx] = np.sum(latent_sources * decay_sigma, axis=0)
            
            # Single batch reconstruction back to P dimensions: (N_targets, P)
            target_expansions = np.dot(latent_target_accum, self.tensor_engine.U.T)

        elif method == "dense":
            for t_idx in range(n_targets):
                t_pos = target_centers[t_idx]
                for s_idx in range(n_sources):
                    s_pos = source_centers[s_idx]
                    r = t_pos - s_pos
                    target_expansions[t_idx] += self.tensor_engine.dense_m2l_eval(source_moments_batch[s_idx], r)
        else:
            raise ValueError(f"Unknown method: {method}")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return target_expansions, elapsed_ms


class FastTensorM2L:
    """
    High-level convenience interface for accelerated M2L tensor operations.
    """
    def __init__(self, order: int = 4, dim: int = 3):
        self.engine = LowRankFarFieldContraction(order=order, dim=dim)

    def evaluate(self, sources: np.ndarray, source_centers: np.ndarray, target_centers: np.ndarray):
        return self.engine.contract_clusters(sources, source_centers, target_centers, method="low_rank")


def benchmark_tensor_vs_dense(order: int = 4, n_sources: int = 50, n_targets: int = 50) -> Dict[str, float]:
    """
    Benchmarks dense vs low-rank contraction of the SAME synthetic factorisation.

    The "dense" path reconstructs U * diag(s(r)) * U^T and contracts it; the
    "low_rank" path contracts U^T -> diag -> U. Both evaluate identical
    algebra, so ``rel_error`` is floating-point round-off (~1e-16), NOT an
    approximation error of a true M2L kernel. The reported speedup measures
    the O(P*R) vs O(P^2) contraction cost of this synthetic factorisation.
    """
    engine = LowRankFarFieldContraction(order=order, dim=3)
    p = engine.tensor_engine.n_coeffs
    rank = engine.tensor_engine.rank
    
    rng = np.random.RandomState(42)
    source_moments = rng.randn(n_sources, p)
    source_centers = rng.uniform(-10, 0, (n_sources, 3))
    target_centers = rng.uniform(10, 20, (n_targets, 3))
    
    dense_res, t_dense = engine.contract_clusters(source_moments, source_centers, target_centers, method="dense")
    lowrank_res, t_lowrank = engine.contract_clusters(source_moments, source_centers, target_centers, method="low_rank")
    
    # Relative Frobenius error
    rel_err = float(np.linalg.norm(dense_res - lowrank_res) / (np.linalg.norm(dense_res) + 1e-12))
    speedup = t_dense / max(1e-6, t_lowrank)

    return {
        "order": order,
        "n_coeffs_P": p,
        "rank_R": rank,
        "dense_time_ms": t_dense,
        "lowrank_time_ms": t_lowrank,
        "speedup": speedup,
        "rel_error": rel_err
    }


if __name__ == "__main__":
    print("Synthetic low-rank tensor contraction demo (NOT a real FMM M2L operator)...")

    for p_order in [2, 3, 4, 5]:
        stats = benchmark_tensor_vs_dense(order=p_order, n_sources=60, n_targets=60)
        print(f"Order p={stats['order']} (P={stats['n_coeffs_P']}, Rank={stats['rank_R']}): "
              f"Dense={stats['dense_time_ms']:.2f}ms, LowRank={stats['lowrank_time_ms']:.2f}ms "
              f"-> Speedup={stats['speedup']:.2f}x (round-off rel error={stats['rel_error']:.4e}, "
              f"NOT a true M2L approximation error)")

    print("Synthetic low-rank contraction demo complete (dense==low-rank algebra by construction).")
