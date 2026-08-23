"""
Continuous Meshfree Wavelet Transform & Multi-Resolution Filterbanks (continuous_meshfree_wavelet.py).

Inspired by:
1. "Continuous Wavelet Transforms on Arbitrary Manifolds"
   Antoine, Demanet, Jacques, Vandergheynst (Applied and Computational Harmonic Analysis, 2002).
2. "Spectral Graph Wavelets"
   David K. Hammond, Pierre Vandergheynst, Remi Gribonval (Applied and Computational Harmonic Analysis, 2011).
3. "Optimal Bounds for Open Addressing Without Reordering"
   Farach-Colton, Krapivin, & Kuszmaul (2025). IEEE FOCS 2024 / arXiv:2501.02305.

Key Algorithmic Principle:
Continuous multi-scale signal analysis and spectral feature extraction on unstructured 3D point sets
traditionally require explicit triangular meshing or dense graph Laplacian eigendecompositions (O(N^3)).
Here, we formulate a Continuous Meshfree Wavelet Transform (CWT):
    W_psi[f](a, x) = (1 / a^{D/2}) * sum_{j=1}^N f(y_j) * psi( (x - y_j) / a ) * vol_j

Using Elastic Spatial Hashing across dyadic scale hierarchies a_l = a_0 * 2^l, each scale's
wavelet response is evaluated via compact radial basis windowing (Mexican Hat / Ricker or Morlet wavelets)
in O(J * N) total operations instead of dense O(J * N^2) convolutions.
"""

import time
import os
import sys
from typing import Tuple, List, Optional, Dict
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.spatial_index import CellIndex


class ContinuousMeshfreeWavelet:
    """
    Continuous Multi-Scale Wavelet Filterbank on Unstructured 3D Point Manifolds.
    
    Supports:
    - Ricker / Mexican Hat Wavelet: psi(r) = (D - r^2) * exp(-r^2 / 2)
      (D-dimensional generalization; the 1D Ricker uses D=1, the 3D
      Mexican Hat uses D=3)
    - Morlet Wavelet: psi(r) = cos(omega_0 * r) * exp(-r^2 / 2)
    - Gaussian smoothing kernel: psi(r) = exp(-r^2 / 2)
    """
    def __init__(
        self,
        points: np.ndarray,
        num_scales: int = 5,
        base_scale: float = 0.05,
        wavelet_type: str = "mexican_hat",
        dim: int = 3,
        volumes: Optional[np.ndarray] = None
    ):
        self.points = np.asarray(points, dtype=np.float64)
        self.n_points = len(self.points)
        self.num_scales = int(num_scales)
        self.base_scale = float(base_scale)
        self.wavelet_type = wavelet_type.lower()
        self.dim = int(dim)
        # Per-particle volume weights vol_j for the continuous wavelet
        # transform W_psi[f](a, x) = (1/a^{D/2}) * sum_j f(y_j) * psi((x-y_j)/a) * vol_j.
        # Defaults to uniform weights (all ones) when not provided.
        if volumes is None:
            self.volumes = np.ones(self.n_points, dtype=np.float64)
        else:
            self.volumes = np.asarray(volumes, dtype=np.float64)

        self.scales = self.base_scale * (2.0 ** np.arange(self.num_scales))

    def _eval_mother_wavelet(self, r_normalized: np.ndarray) -> np.ndarray:
        """Evaluates normalized mother wavelet function psi(r)."""
        r_sq = r_normalized ** 2
        if self.wavelet_type in ("mexican_hat", "ricker"):
            # D-dimensional Mexican Hat / Ricker wavelet:
            # psi(r) = (D - r^2) * exp(-r^2 / 2)
            # The factor D (not 1) ensures the wavelet has zero mean in D
            # dimensions: integral of (D - r^2) * exp(-r^2/2) * r^{D-1} dr = 0.
            return (float(self.dim) - r_sq) * np.exp(-0.5 * r_sq)
        elif self.wavelet_type == "morlet":
            omega_0 = 5.0
            return np.cos(omega_0 * r_normalized) * np.exp(-0.5 * r_sq)
        elif self.wavelet_type == "gaussian":
            return np.exp(-0.5 * r_sq)
        else:
            raise ValueError(f"Unknown wavelet type: {self.wavelet_type}")

    def transform_scale_direct(self, signal: np.ndarray, scale: float) -> np.ndarray:
        """Exact dense O(N^2) continuous wavelet convolution at scale 'a'."""
        diff = self.points[:, None, :] - self.points[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        r_norm = r / scale

        psi_mat = (1.0 / (scale ** (self.dim / 2.0))) * self._eval_mother_wavelet(r_norm)
        # Apply per-particle volume weights vol_j (source-side weighting):
        # W_psi[f](a, x_i) = (1/a^{D/2}) * sum_j f(y_j) * psi((x_i-y_j)/a) * vol_j
        return psi_mat @ (signal * self.volumes)

    def transform_scale_hashed(
        self,
        signal: np.ndarray,
        scale: float,
        cutoff_radius_multiplier: float = 3.5
    ) -> np.ndarray:
        """
        Fast O(N) Wavelet convolution using vectorized spatial hash block partitioning.

        X-A12: uses CellIndex (world mode, cell_size = cutoff_r) instead of the
        hand-rolled dict grid with tuple keys.  Same cell size and 3^D ring-1
        neighborhood, so the neighbor sets are identical.
        """
        cutoff_r = cutoff_radius_multiplier * scale
        cell_size = cutoff_r

        # X-A12: CellIndex (world mode) replaces hand-rolled dict grid.
        idx = CellIndex(dims=self.dim, cell_size=cell_size)
        idx.build(self.points)

        norm_factor = 1.0 / (scale ** (self.dim / 2.0))
        out_coeffs = np.zeros(self.n_points, dtype=np.float64)

        for cell_key, idx_target in idx.items():
            idx_target = np.asarray(idx_target, dtype=np.int64)
            pts_t = self.points[idx_target]

            # Gather all source particles from 3^D adjacent cells (ring=1).
            idx_src = idx.neighborhood_indices(cell_key, ring=1)
            if len(idx_src) == 0:
                continue

            pts_s = self.points[idx_src]
            sig_s = signal[idx_src]
            vol_s = self.volumes[idx_src]

            # Vectorized block distance
            diff = pts_t[:, None, :] - pts_s[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)

            mask = dist <= cutoff_r
            r_norm = dist / scale
            psi_vals = np.where(mask, norm_factor * self._eval_mother_wavelet(r_norm), 0.0)

            # Apply per-particle volume weights vol_j (source-side weighting)
            out_coeffs[idx_target] = psi_vals @ (sig_s * vol_s)

        return out_coeffs

    def forward_multiscale_transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Computes the complete Multi-Scale Continuous Wavelet Transform across all dyadic scales.
        """
        scalogram = np.zeros((self.num_scales, self.n_points), dtype=np.float64)
        for s_idx, scale in enumerate(self.scales):
            scalogram[s_idx] = self.transform_scale_hashed(signal, scale)
        return scalogram


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("Continuous Meshfree Wavelet Transform Benchmark")
    print("=" * 70)

    n_points = 8000
    print(f"Number of Unstructured 3D Points: {n_points:,}")
    
    t = 1.5 * np.pi * (1.0 + 2.0 * np.random.rand(n_points))
    height = 21.0 * np.random.rand(n_points)
    x = t * np.cos(t) * 0.05
    y = height * 0.05
    z = t * np.sin(t) * 0.05
    points_3d = np.stack([x, y, z], axis=-1)

    signal = np.sin(5.0 * x) * np.cos(5.0 * z) + np.exp(-((x - 0.5)**2 + (z - 0.5)**2) / 0.02)

    wavelet_engine = ContinuousMeshfreeWavelet(
        points=points_3d,
        num_scales=4,
        base_scale=0.08,
        wavelet_type="mexican_hat"
    )

    t0 = time.perf_counter()
    scalogram_fast = wavelet_engine.forward_multiscale_transform(signal)
    t_fast = (time.perf_counter() - t0) * 1000.0

    print(f"Fast Multi-Scale CWT Execution : {t_fast:.2f} ms (4 Dyadic Scales)")

    target_scale = wavelet_engine.scales[0]
    t0 = time.perf_counter()
    ref_fine = wavelet_engine.transform_scale_direct(signal, target_scale)
    t_dense_single = (time.perf_counter() - t0) * 1000.0
    t_dense_total_proj = t_dense_single * 4.0

    rel_error = np.linalg.norm(scalogram_fast[0] - ref_fine) / np.linalg.norm(ref_fine)

    print(f"Projected Dense O(J*N^2) Time  : {t_dense_total_proj:.2f} ms")
    print(f"Measured Speedup Ratio        : {t_dense_total_proj / max(t_fast, 1e-6):.1f}x")
    print(f"Relative L2 Precision Error   : {rel_error:.2e}")
    print("=" * 70)
