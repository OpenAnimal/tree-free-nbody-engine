"""
Matrix-Free Numerical Laplace Inversion via Modified Talbot Contours (fractional_laplace_contour.py).

Inspired by:
1. "The Accurate Numerical Inversion of the Laplace Transform"
   A. Talbot (IMA Journal of Applied Mathematics, 1979).
2. "A Comparison of Numerical Inversion Methods for the Laplace Transform"
   J. Abate and P. P. Valko (Computers & Mathematics with Applications, 2004).
3. "Nearly-Linear Time Algorithms for Graph Laplacians"
   Daniel A. Spielman and Shang-Hua Teng (SIAM J. Comput. 2011).

Key Algorithmic Principle:
Given a continuous physical or network operator L (e.g. meshfree Laplacian), transient diffusion
and anomalous fractional responses are formally expressed in the Laplace domain:
    U(s) = (s * I - L)^{-1} b

Computing time-domain solutions u(t) = exp(t * L) * b via dense matrix exponentiation requires
O(N^3) eigensolvers or dense matrix polynomial expansions.
By deforming the classical Bromwich inversion contour into a Modified Talbot contour:
    s(theta) = sigma + mu * (theta * cot(theta) + i * nu * theta),  theta in (-pi, pi)

The continuous Bromwich integral:
    u(t) = (1 / 2*pi*i) * int_Gamma exp(s * t) * (s * I - L)^{-1} b ds
reduces to an exponentially convergent K-point trapezoidal quadrature sum:
    u(t) \approx sum_{k=1}^K Real[ w_k * exp(s_k * t) * (s_k * I - L)^{-1} b ]

Each complex-shifted resolvent (s_k * I - L)^{-1} b is solved via matrix-free Preconditioned
Conjugate Gradients / BiCGStab in O(K * N) operations without assembling dense matrices.
"""

import time
from typing import Tuple, List, Optional, Dict, Callable
import numpy as np


class MatrixFreeTalbotLaplaceInverter:
    """
    Matrix-Free Numerical Laplace Transform Inverter using Modified Talbot Contours.
    
    Inverts operator equations U(s) = (s * I - L)^{-1} b into time-domain u(t)
    via multi-frequency complex-shifted Krylov solvers.
    """
    def __init__(
        self,
        n_quadrature_nodes: int = 24,
        sigma_shift: float = 0.0,
        mu_scale: float = 1.0,
        nu_scale: float = 0.6
    ):
        self.K = int(n_quadrature_nodes)
        self.sigma = float(sigma_shift)
        self.mu = float(mu_scale)
        self.nu = float(nu_scale)

    def compute_talbot_nodes(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the K complex quadrature nodes s_k and corresponding integration weights w_k
        along the Talbot contour for target time t.
        
        Args:
            t: Target time value (t > 0)
            
        Returns:
            s_nodes: (K,) complex quadrature frequencies
            weights: (K,) complex quadrature integration weights
        """
        t_safe = max(float(t), 1e-12)
        # Scaled Talbot parameters for optimal exponential convergence
        # As established by Weideman & Trefethen (2007).
        # The mu_scale parameter scales the contour radius, allowing users to
        # tune the contour geometry (e.g. for stiff or multi-scale problems).
        r_scale = self.mu * (2.0 * self.K) / (5.0 * t_safe)
        
        k_indices = np.arange(1, self.K + 1)
        theta_k = (k_indices - 0.5) * np.pi / self.K
        
        # Talbot contour parametrization: s(theta) = r * (theta * cot(theta) + i * nu * theta)
        # Handling theta -> 0 singularity via Taylor expansion: theta * cot(theta) ~ 1 - theta^2/3
        cot_theta = np.zeros_like(theta_k)
        for idx, th in enumerate(theta_k):
            if np.abs(th) < 1e-6:
                cot_theta[idx] = 1.0 - (th ** 2) / 3.0
            else:
                cot_theta[idx] = th / np.tan(th)

        s_nodes = self.sigma + r_scale * (cot_theta + 1j * self.nu * theta_k)
        
        # Derivative ds/dtheta for trapezoidal quadrature weight:
        # ds/dtheta = r * (cot(theta) - theta * csc^2(theta) + i * nu)
        ds_dtheta = np.zeros_like(theta_k, dtype=np.complex128)
        for idx, th in enumerate(theta_k):
            if np.abs(th) < 1e-6:
                ds_dtheta[idx] = r_scale * (-2.0 * th / 3.0 + 1j * self.nu)
            else:
                sin_th = np.sin(th)
                ds_dtheta[idx] = r_scale * ((1.0 / np.tan(th)) - th / (sin_th ** 2) + 1j * self.nu)

        # Trapezoidal weights with pre-multiplied 1 / (pi * i) factor
        # Since u(t) = (1 / pi) * Im[ sum w_k * exp(s_k * t) * F(s_k) ]
        weights = (np.pi / self.K) * ds_dtheta / (1j * np.pi)
        return s_nodes, weights

    def solve_complex_shifted_system(
        self,
        matvec_fn: Callable[[np.ndarray], np.ndarray],
        s_complex: complex,
        rhs: np.ndarray,
        tol: float = 1e-6,
        max_iter: int = 150
    ) -> np.ndarray:
        """
        Solves (s * I - L) x = rhs for complex shift s using complex BiCGStab.
        
        Args:
            matvec_fn: Linear operator computing L * v
            s_complex: Complex frequency shift
            rhs: Real or complex RHS vector
            tol: Residual convergence tolerance
            max_iter: Maximum Krylov iterations
            
        Returns:
            x: Complex solution vector
        """
        n = len(rhs)
        x = np.zeros(n, dtype=np.complex128)
        
        # Operator A(v) = s * v - L(v)
        def A_op(v: np.ndarray) -> np.ndarray:
            return s_complex * v - matvec_fn(v)

        r = rhs.astype(np.complex128) - A_op(x)
        r_hat = r.copy()
        
        norm_r0 = np.linalg.norm(r)
        if norm_r0 < 1e-14:
            return x

        rho_prev = 1.0 + 0.0j
        alpha = 1.0 + 0.0j
        omega = 1.0 + 0.0j
        v = np.zeros(n, dtype=np.complex128)
        p = np.zeros(n, dtype=np.complex128)

        for _ in range(max_iter):
            rho = np.vdot(r_hat, r)
            if np.abs(rho) < 1e-16:
                break
                
            beta = (rho / rho_prev) * (alpha / omega)
            p = r + beta * (p - omega * v)
            
            v = A_op(p)
            vdot_rhat_v = np.vdot(r_hat, v)
            if np.abs(vdot_rhat_v) < 1e-16:
                break
            alpha = rho / vdot_rhat_v
            
            s_vec = r - alpha * v
            if np.linalg.norm(s_vec) / norm_r0 < tol:
                x += alpha * p
                break
                
            t_vec = A_op(s_vec)
            t_norm_sq = np.vdot(t_vec, t_vec)
            if np.abs(t_norm_sq) < 1e-16:
                omega = 0.0
            else:
                omega = np.vdot(t_vec, s_vec) / t_norm_sq
                
            x += alpha * p + omega * s_vec
            r = s_vec - omega * t_vec
            
            if np.linalg.norm(r) / norm_r0 < tol:
                break
                
            rho_prev = rho

        return x

    def invert_transient_diffusion(
        self,
        matvec_fn: Callable[[np.ndarray], np.ndarray],
        rhs_b: np.ndarray,
        t: float,
        tol: float = 1e-5
    ) -> np.ndarray:
        """
        Evaluates u(t) = exp(t * L) * b via Matrix-Free Talbot Contour Inversion.

        Args:
            matvec_fn: Linear operator L(v)
            rhs_b: Initial condition vector b (real or complex)
            t: Target time point (t > 0)
            tol: Solver tolerance

        Returns:
            u_t: (N,) time-domain solution vector (real if rhs_b is real,
                 complex if rhs_b is complex)
        """
        rhs_b = np.asarray(rhs_b)

        # For real rhs, the Talbot contour symmetry gives u(t) = Re[contour sum].
        # For complex rhs, the symmetry breaks: we split b = b_re + i*b_im and
        # solve each part separately (valid because exp(t*L) is real-linear for
        # a real operator L, so exp(t*L)*(b_re + i*b_im) = exp(t*L)*b_re
        # + i*exp(t*L)*b_im).
        if np.iscomplexobj(rhs_b) and np.any(np.imag(rhs_b) != 0):
            u_re = self._invert_real(matvec_fn, np.real(rhs_b), t, tol)
            u_im = self._invert_real(matvec_fn, np.imag(rhs_b), t, tol)
            return u_re + 1j * u_im

        return self._invert_real(matvec_fn, np.asarray(rhs_b, dtype=np.float64), t, tol)

    def _invert_real(
        self,
        matvec_fn: Callable[[np.ndarray], np.ndarray],
        rhs_b: np.ndarray,
        t: float,
        tol: float = 1e-5
    ) -> np.ndarray:
        """Core Talbot inversion for a real rhs (internal helper)."""
        s_nodes, weights = self.compute_talbot_nodes(t)
        u_accum = np.zeros(len(rhs_b), dtype=np.complex128)

        for s_k, w_k in zip(s_nodes, weights):
            # Solve (s_k * I - L) x = b
            sol_k = self.solve_complex_shifted_system(matvec_fn, s_k, rhs_b, tol=tol)
            # Accumulate contour integral: w_k * exp(s_k * t) * sol_k
            u_accum += w_k * np.exp(s_k * t) * sol_k

        # The solution is the real part of the contour accumulation
        return np.real(u_accum)


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("Matrix-Free Talbot Contour Laplace Inversion Benchmark")
    print("=" * 70)

    # Build 1D/2D Laplacian test system
    n_nodes = 1500
    print(f"System Operator Dimension    : {n_nodes} x {n_nodes}")
    
    # 1D/2D 3-point stencil tridiagonal negative-definite Laplacian
    diag = -2.0 * np.ones(n_nodes)
    off_diag = 1.0 * np.ones(n_nodes - 1)
    
    def laplacian_matvec(v: np.ndarray) -> np.ndarray:
        res = -2.0 * v
        res[:-1] += v[1:]
        res[1:] += v[:-1]
        return res * (n_nodes ** 2) * 1e-4  # Scaled diffusion operator

    # Dense matrix for exact reference verification
    L_dense = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    L_dense = L_dense * (n_nodes ** 2) * 1e-4

    # Initial condition: localized heat pulse
    b_init = np.zeros(n_nodes)
    b_init[n_nodes // 2] = 1.0

    target_t = 0.05
    print(f"Target Diffusion Time (t)    : {target_t} s")

    # 1. Fast Matrix-Free Talbot Inversion
    inverter = MatrixFreeTalbotLaplaceInverter(n_quadrature_nodes=20)
    
    t0 = time.perf_counter()
    u_fast = inverter.invert_transient_diffusion(laplacian_matvec, b_init, target_t, tol=1e-6)
    t_fast = (time.perf_counter() - t0) * 1000.0

    print(f"Matrix-Free Talbot Execution : {t_fast:.2f} ms (20 Contour Quadrature Solves)")

    # 2. Exact Matrix Exponential Reference
    t0 = time.perf_counter()
    # Eigendecomposition / dense expm
    eigvals, eigvecs = np.linalg.eigh(L_dense)
    exp_diag = np.exp(eigvals * target_t)
    u_ref = eigvecs @ (exp_diag * (eigvecs.T @ b_init))
    t_ref = (time.perf_counter() - t0) * 1000.0

    print(f"Dense Eigensolver Ref Time   : {t_ref:.2f} ms")
    
    rel_error = np.linalg.norm(u_fast - u_ref) / np.linalg.norm(u_ref)
    print(f"Measured Speedup Ratio       : {t_ref / max(t_fast, 1e-6):.1f}x")
    print(f"Relative L2 Precision Error  : {rel_error:.2e}")
    print("=" * 70)
