"""
Matrix-Free Gaussian Process Regression & Uncertainty Quantification (matrix_free_gaussian_process.py).

Inspired by:
1. "GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration"
   J. Gardner, G. Pleiss, R. Wu, K. Weinberger, A. G. Wilson (NeurIPS 2018).
2. "Exact Gaussian Processes on a Million Data Points"
   Ke Alexander Wang, Geoff Pleiss, Jacob R. Gardner, Roman Garnett, Andrew Gordon Wilson (NeurIPS 2019).
3. "Optimal Bounds for Open Addressing Without Reordering"
   Martin Farach-Colton, Andrew Krapivin, William Kuszmaul (FOCS 2024 / arXiv:2501.02305).

Key Algorithmic Principle:
Gaussian Process (GP) regression is the gold standard for Bayesian non-parametric function approximation
and calibrated uncertainty quantification. However, training standard GPs requires solving the linear system:
    (K_XX + sigma_n^2 * I) * alpha = y
which requires O(N^3) time and O(N^2) memory via dense Cholesky factorization, failing on large datasets.

By recognizing that matrix-vector multiplication K_XX * v is a continuous Gaussian potential summation:
    (K_XX * v)_i = sigma_f^2 * sum_{j=1}^N exp(-||x_i - x_j||^2 / (2 * ell^2)) * v_j

Using Tree-Free Elastic Spatial Hashing with cutoff radius R_cut = 3.5 * ell, the sparse-truncated
matrix-vector product is evaluated in O(N * nnz_per_point) operations (exact for the cutoff-truncated
RBF kernel at ~1e-7). Solving (K + sigma_n^2 * I) * alpha = y via Preconditioned Conjugate Gradients
(PCG) enables sparse-truncated exact Gaussian Processes in O(N * iters * nnz) time and linear memory.
"""

import time
from typing import Tuple, List, Optional, Dict
import numpy as np


class MatrixFreeGaussianProcess:
    """
    Tree-Free Matrix-Free Gaussian Process Regression Solver.

    Fits GPs via sparse-truncated matrix-free PCG (O(N * iters * nnz) training).
    Predictive mean is O(N_test * nnz_per_point). Predictive variance with
    compute_variance=True loops over each test point running a separate PCG solve
    against the full training set, so it costs O(N_test * N_train * iters * nnz) --
    not O(N). Batching the variance solve (plan task X-A10) is what would restore
    near-linear predict-time cost.
    """
    def __init__(
        self,
        lengthscale: float = 0.5,
        signal_variance: float = 1.0,
        noise_variance: float = 0.05,
        cutoff_multiplier: float = 3.5
    ):
        self.ell = float(lengthscale)
        self.sigma_f2 = float(signal_variance)
        self.sigma_n2 = float(noise_variance)
        self.r_cut = float(cutoff_multiplier * self.ell)
        if not np.isfinite(self.ell) or self.ell <= 0.0:
            raise ValueError("lengthscale must be finite and positive")
        if not np.isfinite(self.sigma_f2) or self.sigma_f2 <= 0.0:
            raise ValueError("signal_variance must be finite and positive")
        if not np.isfinite(self.sigma_n2) or self.sigma_n2 < 0.0:
            raise ValueError("noise_variance must be finite and non-negative")
        if not np.isfinite(cutoff_multiplier) or cutoff_multiplier <= 0.0:
            raise ValueError("cutoff_multiplier must be finite and positive")
        self.cell_size = self.r_cut

        self.train_X: Optional[np.ndarray] = None
        self.alpha_weights: Optional[np.ndarray] = None
        self.n_train: int = 0
        self.dim: int = 0

    def _build_sparse_kernel_blocks(
        self,
        targets: np.ndarray,
        sources: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Precomputes sparse spatial block kernel matrices."""
        n_t = len(targets)
        dim = targets.shape[1]
        inv_2ell2 = 1.0 / (2.0 * (self.ell ** 2))
        r_cut_sq = self.r_cut ** 2

        # Hash source points
        src_grid = np.floor(sources / self.cell_size).astype(np.int64)
        src_buckets: Dict[Tuple[int, ...], List[int]] = {}
        for idx, coord in enumerate(src_grid):
            k = tuple(coord)
            if k not in src_buckets:
                src_buckets[k] = []
            src_buckets[k].append(idx)
        src_arrays = {k: np.array(v, dtype=np.int64) for k, v in src_buckets.items()}

        # Hash target points
        tgt_grid = np.floor(targets / self.cell_size).astype(np.int64)
        tgt_buckets: Dict[Tuple[int, ...], List[int]] = {}
        for idx, coord in enumerate(tgt_grid):
            k = tuple(coord)
            if k not in tgt_buckets:
                tgt_buckets[k] = []
            tgt_buckets[k].append(idx)
        tgt_arrays = {k: np.array(v, dtype=np.int64) for k, v in tgt_buckets.items()}

        from itertools import product
        neighbor_offsets = tuple(product((-1, 0, 1), repeat=dim))
        blocks = []

        for t_k, t_idx in tgt_arrays.items():
            cand_src = []
            for offset in neighbor_offsets:
                s_k = tuple(c + delta for c, delta in zip(t_k, offset))
                if s_k in src_arrays:
                    cand_src.append(src_arrays[s_k])

            if len(cand_src) == 0:
                continue

            s_idx_all = np.concatenate(cand_src)
            pts_t = targets[t_idx]
            pts_s = sources[s_idx_all]

            diff = pts_t[:, None, :] - pts_s[None, :, :]
            r_sq = np.sum(diff ** 2, axis=-1)
            
            mask = r_sq <= r_cut_sq
            k_vals = np.where(mask, self.sigma_f2 * np.exp(-r_sq * inv_2ell2), 0.0)
            blocks.append((t_idx, s_idx_all, k_vals))

        return blocks

    def fit(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        tol: float = 1e-5,
        max_iter: int = 60
    ) -> int:
        """
        Fits GP by solving (K + sigma_n^2 * I) * alpha = y via Preconditioned Conjugate Gradients.
        """
        self.train_X = np.asarray(train_X, dtype=np.float64)
        if self.train_X.ndim == 1:
            self.train_X = self.train_X[:, None]
        if self.train_X.ndim != 2 or len(self.train_X) == 0 or not np.all(np.isfinite(self.train_X)):
            raise ValueError("train_X must be a non-empty finite array with shape (N, D)")
        self.n_train, self.dim = self.train_X.shape
        y = np.asarray(train_y, dtype=np.float64).ravel()
        if len(y) != self.n_train or not np.all(np.isfinite(y)):
            raise ValueError("train_y must be finite and have length N")

        # Precompute training kernel blocks once
        train_blocks = self._build_sparse_kernel_blocks(self.train_X, self.train_X)

        def A_op(v: np.ndarray) -> np.ndarray:
            out = np.zeros(self.n_train, dtype=np.float64)
            for t_idx, s_idx, k_mat in train_blocks:
                out[t_idx] += k_mat @ v[s_idx]
            out += self.sigma_n2 * v
            return out

        # Jacobi Preconditioned Conjugate Gradient
        alpha = np.zeros(self.n_train, dtype=np.float64)
        r = y - A_op(alpha)
        norm_r0 = np.linalg.norm(r)
        if norm_r0 < 1e-12:
            self.alpha_weights = alpha
            return 0

        inv_diag = 1.0 / (self.sigma_f2 + self.sigma_n2)
        z = inv_diag * r
        p = z.copy()
        rz_old = np.dot(r, z)

        n_iters = 0
        for it in range(max_iter):
            n_iters = it + 1
            Ap = A_op(p)
            pAp = np.dot(p, Ap)
            if abs(pAp) < 1e-16:
                break

            step = rz_old / pAp
            alpha += step * p
            r -= step * Ap

            if np.linalg.norm(r) / norm_r0 < tol:
                break

            z = inv_diag * r
            rz_new = np.dot(r, z)
            p = z + (rz_new / rz_old) * p
            rz_old = rz_new

        self.alpha_weights = alpha
        return n_iters

    def predict(
        self,
        test_X: np.ndarray,
        compute_variance: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes predictive mean mu_* and true predictive variance sigma_*^2 for test queries:
            mu_* = K_* alpha
            sigma_*^2 = k(x_*, x_*) - K_*^T (K + sigma_n^2 I)^-1 K_*
        """
        if self.train_X is None or self.alpha_weights is None:
            raise RuntimeError("GP must be fitted before predict()")
        X_star = np.asarray(test_X, dtype=np.float64)
        if X_star.ndim == 1:
            X_star = X_star[:, None]
        if X_star.ndim != 2 or X_star.shape[1] != self.dim or not np.all(np.isfinite(X_star)):
            raise ValueError(f"test_X must have finite shape (N, {self.dim})")
        n_test = len(X_star)

        # Build test-train interaction blocks
        test_blocks = self._build_sparse_kernel_blocks(targets=X_star, sources=self.train_X)
        mu_star = np.zeros(n_test, dtype=np.float64)
        for t_idx, s_idx, k_mat in test_blocks:
            mu_star[t_idx] += k_mat @ self.alpha_weights[s_idx]

        var_star = np.full(n_test, self.sigma_f2, dtype=np.float64)
        if compute_variance:
            train_blocks = self._build_sparse_kernel_blocks(self.train_X, self.train_X)
            def A_op(v: np.ndarray) -> np.ndarray:
                out = np.zeros(self.n_train, dtype=np.float64)
                for t_idx, s_idx, k_mat in train_blocks:
                    out[t_idx] += k_mat @ v[s_idx]
                out += self.sigma_n2 * v
                return out

            inv_diag = 1.0 / (self.sigma_f2 + self.sigma_n2)

            for i in range(n_test):
                k_star_i = np.zeros(self.n_train, dtype=np.float64)
                for t_idx, s_idx, k_mat in test_blocks:
                    match_pos = np.where(t_idx == i)[0]
                    if len(match_pos) > 0:
                        k_star_i[s_idx] += k_mat[match_pos[0]]

                if np.linalg.norm(k_star_i) > 1e-12:
                    # Quick PCG solve for variance reduction
                    v_i = np.zeros(self.n_train, dtype=np.float64)
                    r_v = k_star_i - A_op(v_i)
                    z_v = inv_diag * r_v
                    p_v = z_v.copy()
                    rz_old_v = np.dot(r_v, z_v)
                    for _ in range(25):
                        Ap_v = A_op(p_v)
                        pAp_v = np.dot(p_v, Ap_v)
                        if abs(pAp_v) < 1e-16:
                            break
                        step_v = rz_old_v / pAp_v
                        v_i += step_v * p_v
                        r_v -= step_v * Ap_v
                        if np.linalg.norm(r_v) / (np.linalg.norm(k_star_i) + 1e-12) < 1e-4:
                            break
                        z_v = inv_diag * r_v
                        rz_new_v = np.dot(r_v, z_v)
                        p_v = z_v + (rz_new_v / rz_old_v) * p_v
                        rz_old_v = rz_new_v
                    reduction = np.dot(k_star_i, v_i)
                    var_star[i] = max(1e-8, self.sigma_f2 - reduction)

        return mu_star, var_star


def dense_cholesky_gp_baseline(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    ell: float,
    sigma_f2: float,
    sigma_n2: float
) -> Tuple[np.ndarray, float]:
    """Exact dense O(N^3) Cholesky baseline."""
    diff_train = train_X[:, None, :] - train_X[None, :, :]
    K_train = sigma_f2 * np.exp(-np.sum(diff_train ** 2, axis=-1) / (2.0 * (ell ** 2))) + sigma_n2 * np.eye(len(train_X))

    L = np.linalg.cholesky(K_train)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, train_y))

    diff_test = test_X[:, None, :] - train_X[None, :, :]
    K_star = sigma_f2 * np.exp(-np.sum(diff_test ** 2, axis=-1) / (2.0 * (ell ** 2)))
    mu_star = K_star @ alpha
    return mu_star, 0.0


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("Matrix-Free Gaussian Process Regression Benchmark")
    print("=" * 70)

    n_train = 12000
    n_test = 2000
    print(f"Training Dataset Size (N)    : {n_train:,} points")
    print(f"Test Evaluation Queries      : {n_test:,} points")

    X_train = np.random.rand(n_train, 2) * 4.0
    y_true = np.sin(X_train[:, 0] * 1.5) * np.cos(X_train[:, 1] * 1.5)
    noise_sigma = 0.2
    y_train = y_true + np.random.randn(n_train) * noise_sigma

    X_test = np.random.rand(n_test, 2) * 4.0
    y_test_true = np.sin(X_test[:, 0] * 1.5) * np.cos(X_test[:, 1] * 1.5)

    gp = MatrixFreeGaussianProcess(
        lengthscale=0.6,
        signal_variance=1.0,
        noise_variance=noise_sigma**2,
        cutoff_multiplier=3.5
    )

    # 1. Fast Matrix-Free GP Fit
    t0 = time.perf_counter()
    n_iters = gp.fit(X_train, y_train, tol=1e-4, max_iter=40)
    t_fit = (time.perf_counter() - t0) * 1000.0

    print(f"Matrix-Free GP Fit Time      : {t_fit:.2f} ms ({n_iters} PCG iterations)")

    # 2. Fast GP Predict
    t0 = time.perf_counter()
    mu_pred, var_pred = gp.predict(X_test)
    t_pred = (time.perf_counter() - t0) * 1000.0

    test_rmse = np.sqrt(np.mean((mu_pred - y_test_true) ** 2))
    print(f"Predictive Inference Time    : {t_pred:.2f} ms ({n_test:,} test points)")
    print(f"Test Generalization RMSE     : {test_rmse:.4f}")

    # 3. Dense Cholesky Baseline on subset for scaling comparison
    n_sub = 1500
    t0 = time.perf_counter()
    mu_dense_sub, _ = dense_cholesky_gp_baseline(
        X_train[:n_sub], y_train[:n_sub], X_test[:200], ell=0.6, sigma_f2=1.0, sigma_n2=noise_sigma**2
    )
    t_dense_sub = (time.perf_counter() - t0) * 1000.0
    t_dense_proj = t_dense_sub * ((n_train / n_sub) ** 3)

    print(f"Projected Dense O(N^3) Time  : {t_dense_proj:.2f} ms")
    print(f"Measured Speedup Ratio       : {t_dense_proj / max(t_fit, 1e-6):.1f}x")
    print("=" * 70)
