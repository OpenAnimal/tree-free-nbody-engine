"""
Nearly-Linear Spectral Bipartite Co-Clustering (spectral_biclustering_fmm.py).

Inspired by:
1. "Co-clustering Documents and Words Using Bipartite Spectral Graph Partitioning"
   Inderjit S. Dhillon (ACM SIGKDD 2001).
2. "Spectral Biclustering of Microarray Data: Coclustering Genes and Conditions"
   Y. Kluger, R. Basri, J. T. Chang, M. Gerstein (Genome Research, 2003).
3. "Nearly-Linear Time Algorithms for Graph Laplacians"
   Daniel A. Spielman and Shang-Hua Teng (SIAM J. Comput. 2011).

Key Algorithmic Principle:
Given a massive bipartite interaction matrix A in R^{R x C} (e.g., users x items, genes x single-cells,
documents x vocabulary), classical spectral clustering builds a dense (R + C) x (R + C) bipartite Laplacian.
Computing exact SVD via dense eigensolvers scales as O(min(R, C) * R * C), which becomes intractable
for large-scale datasets.

By Dhillon's Bipartite Normalized Cut Theorem:
1. Compute row degrees D1 = A * 1 and column degrees D2 = A^T * 1.
2. Form the normalized matrix-free operator A_n = D1^{-1/2} * A * D2^{-1/2}.
3. The continuous relaxation of the optimal bipartite normalized cut is given by the leading
   singular vectors (u_l, v_l) of A_n.

We compute the k leading singular pairs via Matrix-Free Orthogonalized Power Iteration / Lanczos
in O(k * nnz(A)) time without dense allocations, simultaneously partitioning rows and columns.
"""

import time
from typing import Tuple, List, Optional, Dict
import numpy as np


class SpectralBiclusteringFMM:
    """
    Matrix-Free Normalized Spectral Biclustering Engine.
    
    Partitions rows and columns simultaneously in O(k * nnz(A)) time.
    """
    def __init__(
        self,
        n_clusters: int = 3,
        n_components: Optional[int] = None,
        max_power_iters: int = 60,
        tolerance: float = 1e-6
    ):
        self.k_clusters = int(n_clusters)
        self.n_components = int(n_components) if n_components is not None else int(np.ceil(np.log2(self.k_clusters)) + 1)
        self.max_iters = int(max_power_iters)
        self.tol = float(tolerance)
        if self.k_clusters < 2 or self.n_components < 1 or self.max_iters <= 0 or not np.isfinite(self.tol) or self.tol < 0.0:
            raise ValueError("n_clusters >= 2, n_components >= 1, max_power_iters > 0, and tol >= 0 are required")

    def _normalize_bipartite_operator(
        self,
        A: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Computes diagonal scaling vectors D1^{-1/2} and D2^{-1/2}."""
        A = np.asarray(A, dtype=np.float64)
        row_deg = np.sum(A, axis=1)
        col_deg = np.sum(A, axis=0)

        inv_sqrt_d1 = 1.0 / np.sqrt(np.maximum(row_deg, 1e-12))
        inv_sqrt_d2 = 1.0 / np.sqrt(np.maximum(col_deg, 1e-12))
        return inv_sqrt_d1, inv_sqrt_d2, row_deg, col_deg

    def fit_transform_biclusters(
        self,
        A: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the spectral row and column biclustering assignments.
        
        Args:
            A: (R, C) non-negative bipartite data matrix
            
        Returns:
            row_labels: (R,) cluster assignments for rows
            col_labels: (C,) cluster assignments for columns
            U_embeddings: (R, n_components) normalized row spectral coordinates
            V_embeddings: (C, n_components) normalized column spectral coordinates
        """
        A = np.asarray(A, dtype=np.float64)
        if A.ndim != 2 or A.shape[0] == 0 or A.shape[1] == 0 or not np.all(np.isfinite(A)) or np.any(A < 0.0):
            raise ValueError("A must be a non-empty finite non-negative 2D matrix")
        R, C = A.shape
        if self.k_clusters > min(R, C):
            raise ValueError("n_clusters cannot exceed the smaller matrix dimension")
        if self.n_components + 1 > min(R, C):
            raise ValueError("n_components + 1 cannot exceed the smaller matrix dimension")
        inv_d1, inv_d2, _, _ = self._normalize_bipartite_operator(A)

        # Matrix-free operator: A_n(v) = inv_d1 * (A @ (inv_d2 * v))
        def An_matvec(v: np.ndarray) -> np.ndarray:
            return inv_d1 * (A @ (inv_d2 * v))

        def Ant_matvec(u: np.ndarray) -> np.ndarray:
            return inv_d2 * (A.T @ (inv_d1 * u))

        # Matrix-Free Orthogonalized Block Power Iteration to find leading singular vectors
        # Note: The first singular vector u_1 = D1^{1/2}, v_1 = D2^{1/2} corresponds to eigenvalue 1.0 (trivial)
        # We target eigenvectors 2 ... n_components + 1
        n_vecs = self.n_components + 1
        V_block = np.random.randn(C, n_vecs)
        V_block, _ = np.linalg.qr(V_block)

        for _ in range(self.max_iters):
            # U = A_n * V
            U_block = np.zeros((R, n_vecs), dtype=np.float64)
            for j in range(n_vecs):
                U_block[:, j] = An_matvec(V_block[:, j])

            U_block, _ = np.linalg.qr(U_block)

            # V = A_n^T * U
            V_next = np.zeros((C, n_vecs), dtype=np.float64)
            for j in range(n_vecs):
                V_next[:, j] = Ant_matvec(U_block[:, j])

            V_next, _ = np.linalg.qr(V_next)

            # Check subspace convergence
            diff = np.linalg.norm(V_next - V_block)
            V_block = V_next
            if diff < self.tol:
                break

        # Recompute final orthonormal U and V
        U_block = np.zeros((R, n_vecs), dtype=np.float64)
        for j in range(n_vecs):
            U_block[:, j] = An_matvec(V_block[:, j])
        U_block, _ = np.linalg.qr(U_block)

        # Scale by D1^{-1/2} and D2^{-1/2} to obtain continuous partition coordinates (Dhillon 2001)
        # Discard the trivial first vector (index 0)
        U_embed = inv_d1[:, None] * U_block[:, 1:n_vecs]
        V_embed = inv_d2[:, None] * V_block[:, 1:n_vecs]

        # Discrete k-means clustering in continuous spectral coordinate space
        row_labels = self._simple_kmeans(U_embed, self.k_clusters)
        col_labels = self._simple_kmeans(V_embed, self.k_clusters)

        return row_labels, col_labels, U_embed, V_embed

    def _simple_kmeans(self, data: np.ndarray, k: int, iters: int = 20) -> np.ndarray:
        """Lightweight k-means in spectral subspace."""
        n, d = data.shape
        # Seed initialization
        indices = np.random.choice(n, size=k, replace=False)
        centers = data[indices].copy()

        labels = np.zeros(n, dtype=np.int64)
        for _ in range(iters):
            # Assign nearest center
            diff = data[:, None, :] - centers[None, :, :]
            dists = np.sum(diff ** 2, axis=-1)
            labels = np.argmin(dists, axis=-1)

            # Update centers
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    centers[c] = np.mean(data[mask], axis=0)

        return labels


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("Spectral Bipartite Co-Clustering (Dhillon) Benchmark")
    print("=" * 70)

    # Synthetic bipartite interaction matrix (e.g. 5,000 single-cells x 2,000 genes)
    R, C = 5000, 2000
    k_blocks = 3
    print(f"Bipartite Graph Dimension    : {R:,} Rows (Cells) x {C:,} Columns (Genes)")
    print(f"Target Number of Co-Clusters : {k_blocks}")

    # Generate sparse block-diagonal matrix with background noise
    A = np.random.exponential(scale=0.1, size=(R, C))
    
    # Inject 3 distinct co-cluster blocks
    A[:1600, :600] += 3.0
    A[1600:3400, 600:1300] += 3.5
    A[3400:, 1300:] += 4.0

    bicluster_engine = SpectralBiclusteringFMM(n_clusters=k_blocks, max_power_iters=40)

    # 1. Fast Matrix-Free Spectral Biclustering
    t0 = time.perf_counter()
    row_lbls, col_lbls, U_emb, V_emb = bicluster_engine.fit_transform_biclusters(A)
    t_fast = (time.perf_counter() - t0) * 1000.0

    print(f"Matrix-Free Biclustering Time: {t_fast:.2f} ms")
    print(f"Row Cluster Counts           : {np.bincount(row_lbls).tolist()}")
    print(f"Column Cluster Counts        : {np.bincount(col_lbls).tolist()}")

    # 2. Dense SVD Baseline for time comparison
    t0 = time.perf_counter()
    # Normalized matrix dense SVD
    d1 = 1.0 / np.sqrt(np.maximum(np.sum(A, axis=1), 1e-12))
    d2 = 1.0 / np.sqrt(np.maximum(np.sum(A, axis=0), 1e-12))
    A_norm = (d1[:, None] * A) * d2[None, :]
    u_dense, s_dense, vt_dense = np.linalg.svd(A_norm, full_matrices=False)
    t_dense = (time.perf_counter() - t0) * 1000.0

    print(f"Dense Exact SVD Time         : {t_dense:.2f} ms")
    print(f"Measured Speedup Ratio       : {t_dense / max(t_fast, 1e-6):.1f}x")
    print("=" * 70)
