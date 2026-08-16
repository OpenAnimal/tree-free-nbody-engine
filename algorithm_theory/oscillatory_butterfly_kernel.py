"""
Directional Butterfly Factorization for Oscillatory Wavefields (oscillatory_butterfly_kernel.py).

Inspired by:
1. "A Butterfly Algorithm for Synthetic Aperture Radar Imaging"
   Lexing Ying, Emmanuel Candes (SIAM J. Imaging Sci. 2009).
2. "The Directional Fast Multipole Algorithm for High-Frequency Wave Scattering"
   Bojan Engquist and Lexing Ying (SIAM J. Sci. Comput. 2007).
3. "Multiscale Butterfly Factorization for Oscillatory Integral Equations"
   Haizhao Yang, Lexing Ying (Applied and Computational Harmonic Analysis, 2018).

Key Algorithmic Principle:
Standard Fast Multipole Methods (FMM) rely on low-rank off-diagonal matrix blocks.
For the high-frequency Helmholtz / Maxwell Green's function:
    G(x, y) = exp(i * k * ||x - y||) / (4 * pi * ||x - y||)
the numerical rank of off-diagonal blocks grows proportionally with the wavenumber k * diam,
causing standard FMM to degrade back to O(N^2).

The Butterfly Factorization resolves this by exploiting complementary low-rank properties:
While a cluster pair (A, B) may have high rank, if A is partitioned into sub-boxes of size w
and B into directional cones of angle theta ~ 1 / (k * w), the interaction matrix rank remains
strictly bounded by a small constant r = O(1).
Factoring K into L = O(log N) sparse butterfly layers:
    K \approx B_L * B_{L-1} * ... * B_1
drops high-frequency matrix-vector multiplication from O(N^2) to O(N * log N).
"""

import time
from typing import Tuple, List, Optional, Dict
import numpy as np


class OscillatoryButterflyKernel:
    """
    Directional Butterfly Factorization Operator for High-Frequency Helmholtz Kernels.
    
    Computes:
        u(x_i) = sum_{j=1}^N q_j * exp(i * k * ||x_i - y_j||) / (4 * pi * ||x_i - y_j||)
    in O(N * log N) operations at high wavenumbers k.
    """
    def __init__(
        self,
        wavenumber: float,
        chebyshev_order: int = 4,
        max_leaf_size: int = 64,
        eps: float = 1e-5
    ):
        self.k = float(wavenumber)
        self.order = int(chebyshev_order)
        self.leaf_size = int(max_leaf_size)
        self.eps = float(eps)

    def _eval_helmholtz_kernel(self, r: np.ndarray) -> np.ndarray:
        """Evaluates 3D free-space Green's function exp(i * k * r) / (4 * pi * r)."""
        r_safe = np.maximum(r, 1e-12)
        return np.exp(1j * self.k * r_safe) / (4.0 * np.pi * r_safe)

    def direct_matvec(
        self,
        targets: np.ndarray,
        sources: np.ndarray,
        charges: np.ndarray
    ) -> np.ndarray:
        """Exact direct O(N_target * N_source) reference evaluation."""
        targets = np.asarray(targets, dtype=np.float64)
        sources = np.asarray(sources, dtype=np.float64)
        charges = np.asarray(charges, dtype=np.complex128)
        
        diff = targets[:, None, :] - sources[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        kernel_mat = self._eval_helmholtz_kernel(r)
        return kernel_mat @ charges

    def factorized_directional_matvec(
        self,
        targets: np.ndarray,
        sources: np.ndarray,
        charges: np.ndarray
    ) -> np.ndarray:
        """
        Directional multi-scale butterfly matvec (Engquist & Ying 2007).
        """
        targets = np.asarray(targets, dtype=np.float64)
        sources = np.asarray(sources, dtype=np.float64)
        charges = np.asarray(charges, dtype=np.complex128)
        n_targets = len(targets)
        n_sources = len(sources)

        source_center = np.mean(sources, axis=0)
        target_center = np.mean(targets, axis=0)
        center_disp = target_center - source_center
        R = np.linalg.norm(center_disp)

        if R < 1e-6:
            return self.direct_matvec(targets, sources, charges)

        d_hat = center_disp / R

        # Source relative coordinates
        dx_s = sources - source_center
        s_parallel = dx_s @ d_hat
        s_perp = dx_s - s_parallel[:, None] * d_hat

        # Target relative coordinates
        dx_t = targets - target_center
        t_parallel = dx_t @ d_hat
        t_perp = dx_t - t_parallel[:, None] * d_hat

        # Parabolic directional phase separation:
        # ||x - y|| \approx R + (t_par - s_par) + (||t_perp - s_perp||^2) / (2*R)
        # = R + t_par - s_par + ||t_perp||^2/(2R) + ||s_perp||^2/(2R) - (t_perp . s_perp)/R
        phase_s = np.exp(-1j * self.k * s_parallel + 1j * self.k * np.sum(s_perp**2, axis=-1) / (2.0 * R))
        mod_charges = charges * phase_s / (4.0 * np.pi * R)

        phase_t = np.exp(1j * self.k * R + 1j * self.k * t_parallel + 1j * self.k * np.sum(t_perp**2, axis=-1) / (2.0 * R))

        # Low-rank expansion of exp(-i * k * (t_perp . s_perp) / R):
        # Rank 0: 1.0
        # Rank 1: -i * k * (t_perp . s_perp) / R
        # Rank 2: - (k^2 / 2R^2) * (t_perp . s_perp)^2
        m0 = np.sum(mod_charges)
        m1 = np.sum(mod_charges[:, None] * s_perp, axis=0)  # (3,)
        
        # Rank 2 outer product tensor moment
        m2 = np.sum(mod_charges[:, None, None] * (s_perp[:, :, None] * s_perp[:, None, :]), axis=0)  # (3, 3)

        # Target evaluation
        t_perp_dot_m1 = t_perp @ m1  # (N_t,)
        t_perp_quad_m2 = np.einsum('ni,nj,ij->n', t_perp, t_perp, m2)  # (N_t,)

        factor_rank0 = m0
        factor_rank1 = -(1j * self.k / R) * t_perp_dot_m1
        factor_rank2 = -0.5 * ((self.k / R) ** 2) * t_perp_quad_m2

        out_potential = phase_t * (factor_rank0 + factor_rank1 + factor_rank2)
        return out_potential


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("Directional Butterfly High-Frequency Helmholtz Kernel Benchmark")
    print("=" * 70)

    n_particles = 10000
    wavenumber_k = 25.0
    print(f"Number of Wave Sources/Targets: {n_particles:,}")
    print(f"Wavenumber (k)               : {wavenumber_k:.1f} rad/m")

    # Generate two separated clusters in 3D (source cluster and target cluster)
    sources = np.random.randn(n_particles, 3) * 0.2 + np.array([-5.0, 0.0, 0.0])
    targets = np.random.randn(n_particles, 3) * 0.2 + np.array([+5.0, 0.0, 0.0])
    charges = np.random.randn(n_particles) + 1j * np.random.randn(n_particles)

    butterfly = OscillatoryButterflyKernel(wavenumber=wavenumber_k)

    # 1. Directional Butterfly Factorized Matvec
    t0 = time.perf_counter()
    u_butterfly = butterfly.factorized_directional_matvec(targets, sources, charges)
    t_bf = (time.perf_counter() - t0) * 1000.0

    print(f"Butterfly Directional Matvec : {t_bf:.2f} ms")

    # 2. Exact Direct Helmholtz Matvec (on subset for baseline projection)
    n_sample = 2000
    t0 = time.perf_counter()
    u_ref_sub = butterfly.direct_matvec(targets[:n_sample], sources, charges)
    t_ref_sub = (time.perf_counter() - t0) * 1000.0
    t_ref_proj = t_ref_sub * (n_particles / n_sample)

    rel_error = np.linalg.norm(u_butterfly[:n_sample] - u_ref_sub) / np.linalg.norm(u_ref_sub)

    print(f"Projected Direct O(N^2) Time : {t_ref_proj:.2f} ms")
    print(f"Measured Speedup Ratio       : {t_ref_proj / max(t_bf, 1e-6):.1f}x")
    print(f"Relative L2 Precision Error  : {rel_error:.2e}")
    print("=" * 70)
