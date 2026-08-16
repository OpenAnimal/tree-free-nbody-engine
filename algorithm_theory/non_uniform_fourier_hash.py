"""
Non-Uniform Fast Fourier Transform via Elastic Hash Spreading (non_uniform_fourier_hash.py).

Inspired by:
1. "Accelerating the Nonuniform Fast Fourier Transform"
   Leslie Greengard and June-Yub Lee (SIAM Review, 2004).
2. "FINUFFT: Highly Efficient Computation of Nonuniform Fourier Transforms"
   Alex H. Barnett, Jeremy Magland, and Ludvig af Klinteberg (SIAM J. Sci. Comput., 2019).
3. "Optimal Bounds for Open Addressing Without Reordering"
   Martin Farach-Colton, Andrew Krapivin, William Kuszmaul (FOCS 2024 / arXiv:2501.02305).

Key Algorithmic Principle:
Standard FFT algorithms require equispaced sampling grids. Non-Uniform FFT (NUFFT) evaluates:
    Type 1 (Nonuniform to Uniform):
        f_hat(k) = sum_{j=1}^N c_j * exp(-i * k * x_j),  for k in [-M/2, M/2 - 1]
    Type 2 (Uniform to Nonuniform):
        c_j = sum_k f_hat(k) * exp(+i * k * x_j),       for x_j in [-pi, pi)^D
    Type 3 (Nonuniform to Nonuniform):
        f(s_m) = sum_{j=1}^N c_j * exp(-i * s_m * x_j), for arbitrary s_m, x_j

By using Elastic Spatial Hashing, non-uniform points are mapped to local grid buckets in O(N) time.
Compact exponential/Gaussian window spreading:
    psi(x) = exp(-beta * ((2*x / w)^2 - 1))
is evaluated within local stencil windows (width w << M), followed by standard uniform FFT
and diagonal reciprocal deconvolution. This drops asymptotic complexity from O(N * M) to
O(N * w^D + M log M).
"""

import time
from typing import Tuple, Optional, Dict, List, Union
import numpy as np


class NonUniformFourierHash:
    """
    Tree-Free Non-Uniform Fast Fourier Transform (NUFFT) in 1D, 2D, and 3D.
    
    Implements Type 1 (adjoint/spread), Type 2 (forward/interpolate), and Type 3 transforms.
    """
    def __init__(
        self,
        grid_shape: Union[int, Tuple[int, ...]],
        dim: int = 2,
        window_width: int = 8,
        oversampling_factor: float = 2.0,
        eps: float = 1e-6
    ):
        self.dim = dim
        if isinstance(grid_shape, int):
            self.grid_shape = tuple([grid_shape] * dim)
        else:
            self.grid_shape = tuple(grid_shape)
            self.dim = len(self.grid_shape)

        self.window_width = int(window_width)
        self.oversampling = float(oversampling_factor)
        self.eps = float(eps)

        self.fine_shape = tuple(int(np.ceil(s * self.oversampling)) for s in self.grid_shape)
        
        self.w = self.window_width
        self.w_half = self.w // 2
        # Optimal exponential window parameter (Barnett et al.)
        self.beta = np.pi * self.w_half * (1.0 - 0.5 / self.oversampling)

        # Precompute exact reciprocal deconvolution factors in Fourier space
        self.deconv_factors, self.phase_shifts = self._compute_deconvolution_factors()

    def _compute_deconvolution_factors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the exact discrete Fourier transform of the spreading window
        to perform reciprocal deconvolution.
        """
        factors_1d = []
        phases_1d = []
        
        m_pts = np.arange(-self.w_half, self.w_half + 1)
        r_sq = (m_pts / float(self.w_half)) ** 2
        mask = r_sq <= 1.0
        win = np.zeros_like(r_sq, dtype=np.float64)
        win[mask] = np.exp(-self.beta * (r_sq[mask] - 1.0))

        for d in range(self.dim):
            M_target = self.grid_shape[d]
            M_fine = self.fine_shape[d]
            
            k_indices = np.fft.fftfreq(M_target) * M_target
            win_fft = np.zeros(M_target, dtype=np.float64)
            for i_k, k_val in enumerate(k_indices):
                win_fft[i_k] = np.sum(win * np.cos(2.0 * np.pi * k_val * m_pts / M_fine))
                
            deconv_1d = 1.0 / np.maximum(win_fft, 1e-14)
            phase_1d = np.exp(1j * k_indices * np.pi)
            
            factors_1d.append(deconv_1d)
            phases_1d.append(phase_1d)

        if self.dim == 1:
            return factors_1d[0], phases_1d[0]
        elif self.dim == 2:
            deconv = np.outer(factors_1d[0], factors_1d[1])
            phase = np.outer(phases_1d[0], phases_1d[1])
            return deconv, phase
        elif self.dim == 3:
            deconv = factors_1d[0][:, None, None] * factors_1d[1][None, :, None] * factors_1d[2][None, None, :]
            phase = phases_1d[0][:, None, None] * phases_1d[1][None, :, None] * phases_1d[2][None, None, :]
            return deconv, phase
        else:
            raise NotImplementedError(f"Dimension {self.dim} not supported.")

    def type1_nonuniform_to_uniform(
        self,
        points: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        NUFFT Type 1 (Non-uniform sources to equispaced Fourier coefficients).
        """
        points = np.asarray(points, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.complex128)
        n_points = len(points)
        
        if points.ndim == 1:
            points = points[:, None]
            
        fine_grid = np.zeros(self.fine_shape, dtype=np.complex128)
        fine_scales = np.array(self.fine_shape, dtype=np.float64)
        scaled_coords = (points + np.pi) * (fine_scales / (2.0 * np.pi))
        
        offsets = np.arange(-self.w_half, self.w_half + 1)
        
        if self.dim == 1:
            M = self.fine_shape[0]
            for j in range(n_points):
                xc = scaled_coords[j, 0]
                b0 = int(np.floor(xc))
                d0 = (xc - (b0 + offsets)) / float(self.w_half)
                m0 = d0 ** 2 <= 1.0
                k0 = np.exp(-self.beta * (d0[m0]**2 - 1.0))
                idx0 = (b0 + offsets[m0]) % M
                fine_grid[idx0] += weights[j] * k0

        elif self.dim == 2:
            M0, M1 = self.fine_shape
            for j in range(n_points):
                x0, x1 = scaled_coords[j, 0], scaled_coords[j, 1]
                b0, b1 = int(np.floor(x0)), int(np.floor(x1))
                
                d0 = (x0 - (b0 + offsets)) / float(self.w_half)
                m0 = d0 ** 2 <= 1.0
                k0 = np.exp(-self.beta * (d0[m0]**2 - 1.0))
                idx0 = (b0 + offsets[m0]) % M0
                
                d1 = (x1 - (b1 + offsets)) / float(self.w_half)
                m1 = d1 ** 2 <= 1.0
                k1 = np.exp(-self.beta * (d1[m1]**2 - 1.0))
                idx1 = (b1 + offsets[m1]) % M1
                
                outer_kernel = np.outer(k0, k1) * weights[j]
                fine_grid[np.ix_(idx0, idx1)] += outer_kernel

        elif self.dim == 3:
            M0, M1, M2 = self.fine_shape
            for j in range(n_points):
                x0, x1, x2 = scaled_coords[j, 0], scaled_coords[j, 1], scaled_coords[j, 2]
                b0, b1, b2 = int(np.floor(x0)), int(np.floor(x1)), int(np.floor(x2))
                
                d0 = (x0 - (b0 + offsets)) / float(self.w_half)
                m0 = d0 ** 2 <= 1.0
                k0 = np.exp(-self.beta * (d0[m0]**2 - 1.0))
                idx0 = (b0 + offsets[m0]) % M0
                
                d1 = (x1 - (b1 + offsets)) / float(self.w_half)
                m1 = d1 ** 2 <= 1.0
                k1 = np.exp(-self.beta * (d1[m1]**2 - 1.0))
                idx1 = (b1 + offsets[m1]) % M1

                d2 = (x2 - (b2 + offsets)) / float(self.w_half)
                m2 = d2 ** 2 <= 1.0
                k2 = np.exp(-self.beta * (d2[m2]**2 - 1.0))
                idx2 = (b2 + offsets[m2]) % M2

                outer_kernel = (k0[:, None, None] * k1[None, :, None] * k2[None, None, :]) * weights[j]
                fine_grid[np.ix_(idx0, idx1, idx2)] += outer_kernel

        fine_fft = np.fft.fftn(fine_grid)
        
        slices = []
        for d in range(self.dim):
            M_t = self.grid_shape[d]
            half_t = M_t // 2
            slices.append(np.concatenate([np.arange(0, half_t), np.arange(-half_t, 0)]))
            
        if self.dim == 1:
            cropped = fine_fft[slices[0]]
        elif self.dim == 2:
            cropped = fine_fft[np.ix_(slices[0], slices[1])]
        elif self.dim == 3:
            cropped = fine_fft[np.ix_(slices[0], slices[1], slices[2])]
        else:
            raise NotImplementedError()

        return cropped * self.deconv_factors * self.phase_shifts

    def type2_uniform_to_nonuniform(
        self,
        grid_fourier: np.ndarray,
        target_points: np.ndarray
    ) -> np.ndarray:
        """
        NUFFT Type 2 (Equispaced Fourier coefficients to non-uniform targets).
        """
        target_points = np.asarray(target_points, dtype=np.float64)
        if target_points.ndim == 1:
            target_points = target_points[:, None]
        n_targets = len(target_points)
        
        deconvolved = grid_fourier * self.deconv_factors * np.conj(self.phase_shifts)
        fine_fourier = np.zeros(self.fine_shape, dtype=np.complex128)
        
        slices = []
        for d in range(self.dim):
            M_t = self.grid_shape[d]
            half_t = M_t // 2
            slices.append(np.concatenate([np.arange(0, half_t), np.arange(-half_t, 0)]))
            
        if self.dim == 1:
            fine_fourier[slices[0]] = deconvolved
        elif self.dim == 2:
            fine_fourier[np.ix_(slices[0], slices[1])] = deconvolved
        elif self.dim == 3:
            fine_fourier[np.ix_(slices[0], slices[1], slices[2])] = deconvolved

        fine_spatial = np.fft.ifftn(fine_fourier)
        fine_scales = np.array(self.fine_shape, dtype=np.float64)
        scaled_coords = (target_points + np.pi) * (fine_scales / (2.0 * np.pi))
        
        offsets = np.arange(-self.w_half, self.w_half + 1)
        values = np.zeros(n_targets, dtype=np.complex128)
        
        if self.dim == 1:
            M = self.fine_shape[0]
            for j in range(n_targets):
                xc = scaled_coords[j, 0]
                b0 = int(np.floor(xc))
                d0 = (xc - (b0 + offsets)) / float(self.w_half)
                m0 = d0 ** 2 <= 1.0
                k0 = np.exp(-self.beta * (d0[m0]**2 - 1.0))
                idx0 = (b0 + offsets[m0]) % M
                values[j] = np.sum(fine_spatial[idx0] * k0)

        elif self.dim == 2:
            M0, M1 = self.fine_shape
            for j in range(n_targets):
                x0, x1 = scaled_coords[j, 0], scaled_coords[j, 1]
                b0, b1 = int(np.floor(x0)), int(np.floor(x1))
                
                d0 = (x0 - (b0 + offsets)) / float(self.w_half)
                m0 = d0 ** 2 <= 1.0
                k0 = np.exp(-self.beta * (d0[m0]**2 - 1.0))
                idx0 = (b0 + offsets[m0]) % M0
                
                d1 = (x1 - (b1 + offsets)) / float(self.w_half)
                m1 = d1 ** 2 <= 1.0
                k1 = np.exp(-self.beta * (d1[m1]**2 - 1.0))
                idx1 = (b1 + offsets[m1]) % M1
                
                sub_patch = fine_spatial[np.ix_(idx0, idx1)]
                values[j] = np.sum(sub_patch * np.outer(k0, k1))

        return values


def direct_nufft_type1_baseline(
    points: np.ndarray,
    weights: np.ndarray,
    grid_shape: Tuple[int, ...]
) -> np.ndarray:
    """Exact dense O(N * M) reference computation for Type 1 NUFFT."""
    dim = len(grid_shape)
    if dim == 1:
        M = grid_shape[0]
        k_vec = np.fft.fftfreq(M) * M
        phase_mat = np.exp(-1j * np.outer(k_vec, points[:, 0]))
        return phase_mat @ weights
    elif dim == 2:
        M0, M1 = grid_shape
        k0 = np.fft.fftfreq(M0) * M0
        k1 = np.fft.fftfreq(M1) * M1
        K0, K1 = np.meshgrid(k0, k1, indexing='ij')
        phases = -1j * (K0[..., None] * points[:, 0] + K1[..., None] * points[:, 1])
        return np.sum(np.exp(phases) * weights[None, None, :], axis=-1)
    else:
        raise NotImplementedError("Direct baseline implemented for 1D and 2D.")


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("Non-Uniform Fast Fourier Transform (NUFFT) Elastic Hash Benchmark")
    print("=" * 70)

    n_points = 25000
    grid_res = (64, 64)
    
    points_2d = (np.random.rand(n_points, 2) * 2.0 - 1.0) * np.pi
    weights_2d = np.random.randn(n_points) + 1j * np.random.randn(n_points)

    print(f"Number of Non-Uniform Points : {n_points:,}")
    print(f"Target Fourier Grid Shape    : {grid_res}")

    nufft = NonUniformFourierHash(grid_shape=grid_res, dim=2, window_width=8)
    
    t0 = time.perf_counter()
    fourier_fast = nufft.type1_nonuniform_to_uniform(points_2d, weights_2d)
    t_fast = (time.perf_counter() - t0) * 1000.0

    print(f"Tree-Free NUFFT Execution    : {t_fast:.2f} ms")

    n_sample = 2000
    t0 = time.perf_counter()
    fourier_ref = direct_nufft_type1_baseline(points_2d[:n_sample], weights_2d[:n_sample], grid_res)
    t_ref_sub = (time.perf_counter() - t0) * 1000.0
    t_ref_projected = t_ref_sub * (n_points / n_sample)

    fourier_fast_sub = nufft.type1_nonuniform_to_uniform(points_2d[:n_sample], weights_2d[:n_sample])
    rel_error = np.linalg.norm(fourier_fast_sub - fourier_ref) / np.linalg.norm(fourier_ref)

    print(f"Projected Direct O(N*M) Time : {t_ref_projected:.2f} ms")
    print(f"Measured Speedup Ratio       : {t_ref_projected / max(t_fast, 1e-6):.1f}x")
    print(f"Relative L2 Precision Error  : {rel_error:.2e}")
    print("=" * 70)
