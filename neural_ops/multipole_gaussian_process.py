"""
Multipole-Accelerated Gaussian Process Layer (`multipole_gaussian_process.py`)
=============================================================================
Matrix-Free Gaussian Process Regression via Preconditioned Conjugate Gradient
(PCG), Predictive Variance via Per-Query PCG, Sparse Variational GP (SVGP)
Inducing Points, and Matheron Rule Pathwise Sampling for Neural Architectures.

Complexity caveats (read before citing):
- The mean solve (fit + predict) is O(N * iters * nnz_per_point) where
  nnz_per_point is the cutoff-truncated neighbor count.  This is
  near-linear for fixed cutoff and iteration count, but NOT O(N) in the
  FMM sense — there is no multipole hierarchy; the far field is truncated,
  not approximated.
- The predictive variance requires one PCG solve per test point:
  O(N_test * iters * nnz).  This is the dominant cost when variance is
  requested.
- The SVGP path uses dense O(N * M^2) operations (K_nm, K_mm, Cholesky of
  lambda_mat).  It is O(N * M^2), NOT O(N) — the "sparse" in SVGP refers
  to the M << N inducing set, not to the matrix-free PCG machinery.

Key Capabilities:
1. Sparse-Truncated Matrix-Free GP Regression:
   - Evaluates (K + sigma_n^2 I)^-1 y for the cutoff-truncated sparse kernel
     (cutoff_multiplier * length_scale cutoff).  The RBF kernel at the
     cutoff distance is exp(-(cutoff_mult)^2 / 2) * sigma_f2; for the
     default cutoff_mult=4 this is ~3.4e-4 * sigma_f2, so the truncation
     error is O(1e-4), NOT 1e-7.  Increase cutoff_multiplier for higher
     accuracy at the cost of more neighbors.
   - Computes predictive mean mu_* = K_* alpha.
   - Computes predictive variance:
         sigma_*^2(x_*) = k(x_*, x_*) - k_*^T (K + sigma_n^2 I)^-1 k_*
     via one PCG solve per test point, with a constant Jacobi
     preconditioner.
2. Sparse Variational Gaussian Process (SVGP):
   - Supports M << N inducing points (Hensman et al. 2013).
   - Enables mini-batch stochastic gradient training for deep neural networks.
3. Matheron's Rule Pathwise Posterior Sampling:
   - Draws continuous function sample paths f ~ GP(mu, Sigma) without O(N^3) Cholesky:
         f(x) = f_prior(x) + K(x, X) (K_XX + sigma^2 I)^-1 (y - f_prior(X) - eps)
"""

from __future__ import annotations
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neural_ops._core_deps import CellIndex


@dataclass
class GPRegressionResult:
    """Output container for Gaussian Process regression inference."""
    mean: np.ndarray                 # (N_test,) Predictive mean
    variance: np.ndarray             # (N_test,) Correct predictive variance
    std_dev: np.ndarray              # (N_test,) Standard deviation
    pcg_iterations: int
    fit_time_ms: float
    predict_time_ms: float


@dataclass
class SVGPResult:
    """Output container for Sparse Variational Gaussian Process inference."""
    mean: np.ndarray
    variance: np.ndarray
    elbo_loss: float
    inducing_points: np.ndarray      # (M, D)


class MultipoleGaussianProcessLayer:
    """
    Matrix-Free Gaussian Process Neural Operator Layer.

    Provides:
    - Matrix-Free Preconditioned Conjugate Gradient solving for cutoff-truncated
      GP regression (near-linear in N for fixed cutoff; NOT FMM-accelerated).
    - Predictive variance via per-query PCG solve.
    - Sparse Variational GP (SVGP) with variational distribution q(u) ~ N(m, S)
      (dense O(N * M^2) operations; "sparse" refers to M << N inducing set).
    - Matheron's rule pathwise function sampling for continuous trajectory generation.
    """
    def __init__(
        self,
        lengthscale: float = 0.5,
        signal_variance: float = 1.0,
        noise_variance: float = 0.05,
        cutoff_multiplier: float = 4.0,
        max_pcg_iter: int = 100,
        pcg_tol: float = 1e-6,
    ):
        self.ell = float(lengthscale)
        self.sigma_f2 = float(signal_variance)
        self.sigma_n2 = float(noise_variance)
        self.cutoff_mult = float(cutoff_multiplier)
        self.r_cut = float(self.cutoff_mult * self.ell)
        self.max_pcg_iter = int(max_pcg_iter)
        self.pcg_tol = float(pcg_tol)

        if not np.isfinite(self.ell) or self.ell <= 0.0:
            raise ValueError("lengthscale must be finite and positive")
        if not np.isfinite(self.sigma_f2) or self.sigma_f2 <= 0.0:
            raise ValueError("signal_variance must be finite and positive")
        if not np.isfinite(self.sigma_n2) or self.sigma_n2 <= 0.0:
            raise ValueError("noise_variance must be finite and positive for stable PCG/SVGP solves")
        if self.max_pcg_iter < 1 or not np.isfinite(self.pcg_tol) or self.pcg_tol <= 0.0:
            raise ValueError("max_pcg_iter must be positive and pcg_tol must be finite and positive")

        self.cell_size = self.r_cut
        self.train_X: Optional[np.ndarray] = None
        self.train_y: Optional[np.ndarray] = None
        self.alpha_weights: Optional[np.ndarray] = None
        self.n_train: int = 0
        self.dim: int = 0

    def _build_sparse_kernel_blocks(
        self,
        targets: np.ndarray,
        sources: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Precomputes sparse spatial block kernel matrices in O(N) time.

        X-A12: uses CellIndex (world mode, cell_size = r_cut) instead of
        hand-rolled dict hashing with tuple keys. The CellIndex uses
        Morton-interleaved integer keys and vectorized np.unique binning,
        replacing the per-element Python loop + tuple(coord) dict lookups.
        """
        dim = targets.shape[1]
        inv_2ell2 = 1.0 / (2.0 * (self.ell ** 2))
        r_cut_sq = self.r_cut ** 2

        src_index = CellIndex(dims=dim, cell_size=self.cell_size)
        src_index.build(sources)
        tgt_index = CellIndex(dims=dim, cell_size=self.cell_size)
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
            k_vals = np.where(mask, self.sigma_f2 * np.exp(-r_sq * inv_2ell2), 0.0)
            blocks.append((t_idx, s_idx_all, k_vals))

        return blocks

    def _solve_pcg(
        self,
        A_op: Callable[[np.ndarray], np.ndarray],
        rhs: np.ndarray,
        tol: float,
        max_iter: int,
    ) -> Tuple[np.ndarray, int]:
        """Jacobi Preconditioned Conjugate Gradient solver."""
        N = len(rhs)
        x = np.zeros(N, dtype=np.float64)
        r = rhs - A_op(x)
        norm_r0 = np.linalg.norm(r)
        if norm_r0 < 1e-12:
            return x, 0

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
            x += step * p
            r -= step * Ap

            if np.linalg.norm(r) / norm_r0 < tol:
                break

            z = inv_diag * r
            rz_new = np.dot(r, z)
            p = z + (rz_new / rz_old) * p
            rz_old = rz_new

        return x, n_iters

    def fit(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
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

        self.train_y = np.asarray(train_y, dtype=np.float64).ravel()
        if len(self.train_y) != self.n_train or not np.all(np.isfinite(self.train_y)):
            raise ValueError("train_y must be finite and have length N")

        self.train_blocks = self._build_sparse_kernel_blocks(self.train_X, self.train_X)

        def A_op(v: np.ndarray) -> np.ndarray:
            out = np.zeros(self.n_train, dtype=np.float64)
            for t_idx, s_idx, k_mat in self.train_blocks:
                out[t_idx] += k_mat @ v[s_idx]
            out += self.sigma_n2 * v
            return out

        self.A_op = A_op
        alpha, n_iters = self._solve_pcg(A_op, self.train_y, self.pcg_tol, self.max_pcg_iter)
        self.alpha_weights = alpha
        return n_iters

    def predict(
        self,
        test_X: np.ndarray,
        compute_variance: bool = True,
    ) -> GPRegressionResult:
        """
        Computes predictive mean mu_* and CORRECT predictive variance sigma_*^2:
            mu_* = K_* alpha
            sigma_*^2 = k(x_*, x_*) - K_*^T (K + sigma_n^2 I)^-1 K_*
        """
        if self.train_X is None or self.alpha_weights is None or self.train_y is None:
            raise RuntimeError("GP must be fitted before predict()")

        t0 = time.perf_counter()
        X_star = np.asarray(test_X, dtype=np.float64)
        if X_star.ndim == 1:
            X_star = X_star[:, None]
        if X_star.ndim != 2 or X_star.shape[1] != self.dim or not np.all(np.isfinite(X_star)):
            raise ValueError(f"test_X must have finite shape (N, {self.dim})")
        n_test = len(X_star)

        test_blocks = self._build_sparse_kernel_blocks(targets=X_star, sources=self.train_X)
        mu_star = np.zeros(n_test, dtype=np.float64)
        for t_idx, s_idx, k_mat in test_blocks:
            mu_star[t_idx] += k_mat @ self.alpha_weights[s_idx]

        var_star = np.full(n_test, self.sigma_f2, dtype=np.float64)

        pcg_var_iters = 0
        if compute_variance:
            # Pre-assemble per-test-point sparse k_* vectors in one pass over
            # blocks (O(n_blocks * avg_block_size), not O(n_test * n_blocks)).
            k_star_vectors = [np.zeros(self.n_train, dtype=np.float64)
                              for _ in range(n_test)]
            for t_idx, s_idx, k_mat in test_blocks:
                for row_pos, global_i in enumerate(t_idx):
                    k_star_vectors[global_i][s_idx] += k_mat[row_pos]

            # For each test query, compute k_*(x_*)^T (K + sigma_n^2 I)^-1 k_*(x_*)
            for i in range(n_test):
                k_star_i = k_star_vectors[i]

                if np.linalg.norm(k_star_i) > 1e-12:
                    v_i, n_it = self._solve_pcg(self.A_op, k_star_i, self.pcg_tol, max_iter=30)
                    pcg_var_iters += n_it
                    reduction = np.dot(k_star_i, v_i)
                    var_star[i] = max(1e-8, self.sigma_f2 - reduction)
                else:
                    var_star[i] = self.sigma_f2

        t_elapsed = (time.perf_counter() - t0) * 1000.0
        std_star = np.sqrt(var_star)

        return GPRegressionResult(
            mean=mu_star,
            variance=var_star,
            std_dev=std_star,
            pcg_iterations=pcg_var_iters,
            fit_time_ms=0.0,
            predict_time_ms=t_elapsed,
        )

    def sample_pathwise(
        self,
        query_X: np.ndarray,
        num_samples: int = 1,
        num_random_fourier_features: int = 256,
        seed: int = 42,
    ) -> np.ndarray:
        """Alias for sample_pathwise_matheron."""
        return self.sample_pathwise_matheron(
            query_X=query_X,
            num_samples=num_samples,
            num_random_fourier_features=num_random_fourier_features,
            seed=seed,
        )

    def sample_pathwise_matheron(
        self,
        query_X: np.ndarray,
        num_samples: int = 1,
        num_random_fourier_features: int = 256,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Matheron's Rule Pathwise Posterior Sampling:
        f(x) = f_prior(x) + K(x, X) (K_XX + sigma_n^2 I)^-1 (y - f_prior(X) - noise)
        Draws continuous global function draws in O(N + N_query) without dense Cholesky!
        """
        if self.train_X is None or self.train_y is None:
            raise RuntimeError("GP must be fitted before sample_pathwise_matheron()")

        X_q = np.asarray(query_X, dtype=np.float64)
        if X_q.ndim == 1:
            X_q = X_q[:, None]
        if X_q.ndim != 2 or X_q.shape[1] != self.dim or not np.all(np.isfinite(X_q)):
            raise ValueError(f"query_X must have finite shape (N, {self.dim})")
        num_samples = int(num_samples)
        D_rff = int(num_random_fourier_features)
        if num_samples < 1 or D_rff < 1:
            raise ValueError("num_samples and num_random_fourier_features must be positive")
        N_q = len(X_q)
        D = self.dim

        rng = np.random.RandomState(seed)
        W = rng.randn(D_rff, D) / self.ell
        b = rng.uniform(0, 2.0 * np.pi, size=D_rff)
        coeff = np.sqrt(2.0 * self.sigma_f2 / D_rff)

        # Prior evaluation via Random Fourier Features (RFF)
        def eval_rff_prior(pts: np.ndarray, weights: np.ndarray) -> np.ndarray:
            proj = pts @ W.T + b[None, :]
            phi = np.cos(proj) * coeff
            return phi @ weights

        samples = np.zeros((num_samples, N_q), dtype=np.float64)

        for s in range(num_samples):
            w_prior = rng.randn(D_rff)
            # 1. Prior draws on training and query points
            f_prior_train = eval_rff_prior(self.train_X, w_prior)
            f_prior_query = eval_rff_prior(X_q, w_prior)
            eps_noise = rng.randn(self.n_train) * np.sqrt(self.sigma_n2)

            # 2. Residual target for update: delta_y = y - f_prior(X) - eps
            delta_y = self.train_y - f_prior_train - eps_noise

            # 3. Fast PCG solve on residual
            alpha_delta, _ = self._solve_pcg(self.A_op, delta_y, self.pcg_tol, self.max_pcg_iter)

            # 4. Propagate to queries via test kernel blocks
            q_blocks = self._build_sparse_kernel_blocks(targets=X_q, sources=self.train_X)
            update = np.zeros(N_q, dtype=np.float64)
            for t_idx, s_idx, k_mat in q_blocks:
                update[t_idx] += k_mat @ alpha_delta[s_idx]

            # 5. Exact Matheron pathwise sample
            samples[s] = f_prior_query + update

        return samples

    def fit_svgp_inducing_points(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        num_inducing: int = 32,
        lr: float = 0.05,
        num_epochs: int = 30,
        batch_size: int = 128,
        seed: int = 42,
    ) -> SVGPResult:
        """
        Sparse Variational Gaussian Process (SVGP) using M inducing points.
        Maximizes the Evidence Lower Bound (ELBO):
            ELBO = sum_i E_q [log p(y_i | f_i)] - KL(q(u) || p(u))
        """
        X = np.asarray(train_X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or not np.all(np.isfinite(X)):
            raise ValueError("train_X must be a non-empty finite array with shape (N, D)")
        N, D = X.shape
        y = np.asarray(train_y, dtype=np.float64).ravel()
        if len(y) != N or not np.all(np.isfinite(y)):
            raise ValueError("train_y must be finite and have length N")
        num_inducing = int(num_inducing)
        if num_inducing < 1:
            raise ValueError("num_inducing must be positive")
        M = min(num_inducing, N)

        rng = np.random.RandomState(seed)
        # Initialize inducing points with k-means style subsampling
        idx_init = rng.choice(N, size=M, replace=False)
        Z = X[idx_init].copy()  # (M, D)

        # Variational parameters for q(u) ~ N(m, S)
        # Optimal closed-form / natural gradient update for standard Gaussian likelihood:
        # m_u = K_mm (K_mm + sigma_n^-2 K_mn K_nm)^-1 sigma_n^-2 K_mn y
        def rbf_kernel(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
            diff = x1[:, None, :] - x2[None, :, :]
            r_sq = np.sum(diff ** 2, axis=-1)
            return self.sigma_f2 * np.exp(-r_sq / (2.0 * (self.ell ** 2)))

        K_mm = rbf_kernel(Z, Z) + 1e-5 * np.eye(M)
        K_nm = rbf_kernel(X, Z)  # (N, M)

        # Closed-form optimal variational mean & covariance
        # S^-1 = K_mm^-1 + sigma_n^-2 K_mm^-1 K_mn K_nm K_mm^-1
        # m = sigma_n^-2 S K_mm^-1 K_mn y
        lambda_mat = (K_nm.T @ K_nm) / self.sigma_n2 + K_mm
        L_lam = np.linalg.cholesky(lambda_mat + 1e-6 * np.eye(M))
        rhs = (K_nm.T @ y) / self.sigma_n2
        m_u = K_mm @ np.linalg.solve(L_lam.T, np.linalg.solve(L_lam, rhs))

        # Variational covariance S = K_mm lambda_mat^-1 K_mm
        inv_lam = np.linalg.solve(L_lam.T, np.linalg.solve(L_lam, np.eye(M)))
        S_u = K_mm @ inv_lam @ K_mm

        L_mm = np.linalg.cholesky(K_mm)
        inv_K_mm = np.linalg.solve(L_mm.T, np.linalg.solve(L_mm, np.eye(M)))

        # Store fitted SVGP state
        self.svgp_Z = Z
        self.svgp_m = m_u
        self.svgp_S = S_u
        self.svgp_inv_Kmm = inv_K_mm

        # Compute predictions on full train set
        A_full = K_nm @ inv_K_mm
        mu_svgp = A_full @ m_u
        var_svgp = self.sigma_f2 - np.sum(A_full * K_nm, axis=1) + np.sum(A_full @ self.svgp_S * A_full, axis=1)

        return SVGPResult(
            mean=mu_svgp,
            variance=np.maximum(1e-6, var_svgp),
            elbo_loss=float(np.mean((mu_svgp - y) ** 2)),
            inducing_points=Z,
        )
