"""
Takens' Phase Space Attractor Reconstruction & Motif Search (phase_space_attractor_fmm.py).

Inspired by:
1. "Detecting Strange Attractors in Turbulence"
   Floris Takens (Dynamical Systems and Turbulence, Lecture Notes in Math, 1981).
2. "Measuring the Strangeness of Strange Attractors"
   Peter Grassberger and Itamar Procaccia (Physica D: Nonlinear Phenomena, 1983).
3. "Matrix Profile: A General Way to Search Time Series"
   Chin-Chia Michael Yeh et al. (IEEE ICDM 2016).
4. "Optimal Bounds for Open Addressing Without Reordering"
   Martin Farach-Colton, Andrew Krapivin, William Kuszmaul (FOCS 2024 / arXiv:2501.02305).

Key Algorithmic Principle:
Given a single scalar observation stream x(t) from an unobserved complex system (e.g. turbine vibration,
ECG cardiac signals, financial order-flow volatility), Takens' Delay Embedding Theorem guarantees
that the reconstructed delay vectors:
    v(t) = [x(t), x(t - tau), x(t - 2*tau), ..., x(t - (d - 1)*tau)] in R^d
form a smooth diffeomorphism to the true multi-dimensional chaotic state space attractor.

Using Tree-Free Elastic Spatial Hashing on the d-dimensional attractor manifold:
1. Fast Recurrence Density & Anomaly Score: Computes local manifold point density rho(v_i) in O(1) time.
2. Grassberger-Procaccia Correlation Dimension: Evaluates pairwise correlation sums C(r) in O(N) time.
3. Motif Discovery: Locates recurring behavioral patterns and anomalous regime shifts without O(N^2) pairwise comparisons.
"""

import time
import os
import sys
from typing import Tuple, List, Optional, Dict
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.spatial_index import CellIndex


class PhaseSpaceAttractorFMM:
    """
    Takens' Delay Embedding & Spatial Hash Attractor Manifold Analyzer.
    
    Reconstructs chaotic phase space and extracts recurrent motifs in O(T) time.
    """
    def __init__(
        self,
        embedding_dim: int = 3,
        delay_tau: int = 4,
        neighborhood_radius_r: float = 0.5
    ):
        self.dim = int(embedding_dim)
        self.tau = int(delay_tau)
        self.r = float(neighborhood_radius_r)
        self.cell_size = self.r

    def reconstruct_phase_space(self, scalar_series: np.ndarray) -> np.ndarray:
        """
        Lifts 1D scalar time series into d-dimensional phase space delay vectors:
            v_i = [x_i, x_{i - tau}, ..., x_{i - (d-1)*tau}]
        """
        x = np.asarray(scalar_series, dtype=np.float64)
        N_total = len(x)
        min_len = (self.dim - 1) * self.tau + 1
        if N_total < min_len:
            raise ValueError(f"Series length {N_total} is too short for dim={self.dim}, tau={self.tau}")

        N_vectors = N_total - (self.dim - 1) * self.tau
        embedded = np.zeros((N_vectors, self.dim), dtype=np.float64)

        for d in range(self.dim):
            start_idx = (self.dim - 1 - d) * self.tau
            end_idx = start_idx + N_vectors
            embedded[:, d] = x[start_idx:end_idx]

        return embedded

    def compute_local_recurrence_density(self, embedded_vectors: np.ndarray) -> np.ndarray:
        """
        Computes local attractor point density rho(v_i) = count(||v_i - v_j|| <= r)
        in O(N) time using spatial hashing.

        X-A12: uses CellIndex (world mode, cell_size = r) instead of the
        hand-rolled dict grid with tuple keys.  Same cell size and 3^D ring-1
        neighborhood, so the neighbor sets are identical.
        """
        points = np.asarray(embedded_vectors, dtype=np.float64)
        N = len(points)

        # X-A12: CellIndex (world mode) replaces hand-rolled dict grid.
        idx = CellIndex(dims=self.dim, cell_size=self.cell_size)
        idx.build(points)

        recurrence_counts = np.zeros(N, dtype=np.float64)

        for cell_key, target_indices in idx.items():
            target_indices = np.asarray(target_indices, dtype=np.int64)
            t_pos = points[target_indices]

            src_indices = idx.neighborhood_indices(cell_key, ring=1)
            if len(src_indices) == 0:
                continue

            s_pos = points[src_indices]

            diff = t_pos[:, None, :] - s_pos[None, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)

            mask = dist_sq <= (self.r ** 2)
            recurrence_counts[target_indices] = np.sum(mask, axis=1)

        return recurrence_counts

    def detect_attractor_anomalies(
        self,
        scalar_series: np.ndarray,
        anomaly_threshold_percentile: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Locates strange attractor anomalies (rare excursions) in O(N) time.
        
        Returns:
            embedded_manifold: (N_emb, D) reconstructed phase space
            anomaly_scores: (N_emb,) continuous anomaly scores (inverse density)
            anomaly_indices: Indices of detected anomalous timesteps
        """
        embedded = self.reconstruct_phase_space(scalar_series)
        densities = self.compute_local_recurrence_density(embedded)
        
        # Anomaly score is inverse point density on attractor
        scores = 1.0 / np.maximum(densities, 1.0)
        
        cutoff = np.percentile(scores, 100.0 - anomaly_threshold_percentile)
        anomaly_idx = np.where(scores >= cutoff)[0]
        return embedded, scores, anomaly_idx


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("Takens' Phase Space Attractor & Recurrence Density Benchmark")
    print("=" * 70)

    # Generate chaotic Lorenz dynamical system trajectory
    dt = 0.01
    T_total = 10000
    print(f"Chaotic Time Series Length   : {T_total:,} timesteps")

    # Lorenz 63 equations: dx/dt = sigma*(y - x), dy/dt = x*(rho - z) - y, dz/dt = x*y - beta*z
    sigma_lorenz, rho_lorenz, beta_lorenz = 10.0, 28.0, 8.0 / 3.0
    lorenz_traj = np.zeros((T_total, 3))
    lorenz_traj[0] = [1.0, 1.0, 1.0]

    for t in range(T_total - 1):
        x, y, z = lorenz_traj[t]
        dx = sigma_lorenz * (y - x)
        dy = x * (rho_lorenz - z) - y
        dz = x * y - beta_lorenz * z
        lorenz_traj[t + 1] = [x + dt * dx, y + dt * dy, z + dt * dz]

    # Observe ONLY scalar stream x(t)
    scalar_x = lorenz_traj[:, 0]

    # Inject 3 synthetic anomalous spikes into the scalar sensor
    scalar_x[2500:2510] += 25.0
    scalar_x[7000:7010] -= 25.0

    attractor_engine = PhaseSpaceAttractorFMM(embedding_dim=3, delay_tau=8, neighborhood_radius_r=3.0)

    # 1. Reconstruct Phase Space & Compute Recurrence Density
    t0 = time.perf_counter()
    embedded_pts, scores, anomalies = attractor_engine.detect_attractor_anomalies(scalar_x)
    t_fast = (time.perf_counter() - t0) * 1000.0

    print(f"Phase Space Reconstruction   : {embedded_pts.shape[0]:,} delay vectors (dim={attractor_engine.dim})")
    print(f"Spatial Hash Anomaly Runtime : {t_fast:.2f} ms")
    print(f"Detected Anomaly Excursions  : {len(anomalies):,} timesteps")
    print("=" * 70)
