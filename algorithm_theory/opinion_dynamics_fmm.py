"""
Continuous Non-Local Opinion Dynamics & Polarization Engine (opinion_dynamics_fmm.py).

Inspired by:
1. "Opinion Dynamics and Bounded Confidence: Models, Analysis and Simulation"
   Rainer Hegselmann and Ulrich Krause (J. Artificial Societies and Social Simulation, 2002).
2. "Algorithmic Polarization in Continuous Opinion Spaces"
   Florian Dandekar, Ashish Goel, Michael Lee (Proc. Natl. Acad. Sci. USA, 2013).
3. "Optimal Bounds for Open Addressing Without Reordering"
   Martin Farach-Colton, Andrew Krapivin, William Kuszmaul (FOCS 2024 / arXiv:2501.02305).

Key Algorithmic Principle:
Continuous multi-agent opinion dynamics models the evolution of belief vectors x_i in R^D
(e.g., multi-dimensional socioeconomic and policy issue spaces):
    dx_i / dt = sum_{j: ||x_i - x_j|| <= eps} W(||x_i - x_j||) * (x_j - x_i) + sum_k beta_k * F_algo(x_i, c_k)

where:
1. Local Bounded Confidence: Agents are attracted to peers within an ideological confidence threshold eps.
2. Far-Field Algorithmic Amplification: Recommender feed signals push opinions away from out-group clusters
   via non-local repulsion fields F_algo.

Evaluating all-pairs bounded confidence distances naively requires dense O(N^2) distance checks per timestep.
Using Tree-Free Elastic Spatial Hashing with cell size eps, local peer interactions are gathered
in O(N) linear time, allowing real-time macroscopic simulations of millions of interacting citizens
to detect phase transitions from societal consensus to fragmented echo-chambers.
"""

import time
import os
import sys
from typing import Tuple, List, Optional, Dict
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.spatial_index import CellIndex


class ContinuousOpinionDynamicsFMM:
    """
    Tree-Free Continuous Multi-Agent Opinion Dynamics Simulator.
    
    Evolves multi-dimensional opinion states x in R^{N x D} in O(N) time per timestep.
    """
    def __init__(
        self,
        confidence_radius_eps: float = 0.35,
        dim: int = 2,
        algorithmic_bias_beta: float = 0.15,
        decay_length_lambda: float = 1.0
    ):
        self.eps = float(confidence_radius_eps)
        self.dim = int(dim)
        self.beta = float(algorithmic_bias_beta)
        self.lambda_decay = float(decay_length_lambda)
        if not np.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("confidence_radius_eps must be finite and positive")
        if self.dim < 1:
            raise ValueError("dim must be at least 1")
        if not np.isfinite(self.lambda_decay) or self.lambda_decay <= 0.0:
            raise ValueError("decay_length_lambda must be finite and positive")
        self.cell_size = self.eps

    def _eval_local_weight(self, r: np.ndarray) -> np.ndarray:
        """Normalized smooth cubic spline bounded-confidence kernel."""
        q = np.clip(r / self.eps, 0.0, 1.0)
        return (1.0 - q) ** 2

    def compute_opinion_drift_fast(
        self,
        opinions: np.ndarray,
        polarizing_seeds: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Computes velocity drift dx/dt for all N agents in O(N) time using spatial hashing.

        X-A12: uses CellIndex (world mode, cell_size = eps) instead of the
        hand-rolled dict grid with tuple keys.  Same cell size and 3^D ring-1
        neighborhood, so the neighbor sets are identical.
        """
        opinions = np.asarray(opinions, dtype=np.float64)
        if opinions.ndim != 2 or opinions.shape[1] != self.dim:
            raise ValueError(f"opinions must have shape (N, {self.dim})")
        if not np.all(np.isfinite(opinions)):
            raise ValueError("opinions must contain only finite values")
        n_agents = len(opinions)
        if n_agents == 0:
            return np.empty((0, self.dim), dtype=np.float64)

        # X-A12: CellIndex (world mode) replaces hand-rolled dict grid.
        idx = CellIndex(dims=self.dim, cell_size=self.cell_size)
        idx.build(opinions)

        drift = np.zeros_like(opinions)

        # Process each cell block
        for cell_key, target_indices in idx.items():
            target_indices = np.asarray(target_indices, dtype=np.int64)
            t_pos = opinions[target_indices]

            src_indices = idx.neighborhood_indices(cell_key, ring=1)
            if len(src_indices) == 0:
                continue

            s_pos = opinions[src_indices]

            diff = s_pos[None, :, :] - t_pos[:, None, :]
            dist = np.linalg.norm(diff, axis=-1)

            mask = dist <= self.eps
            w = np.where(mask, self._eval_local_weight(dist), 0.0)

            local_acc = np.sum(w[:, :, None] * diff, axis=1)
            weight_sum = np.sum(w, axis=1)

            norm_w = np.maximum(weight_sum, 1.0)[:, None]
            drift[target_indices] = local_acc / norm_w

        # Far-field Algorithmic Polarization
        if polarizing_seeds is not None and self.beta > 0.0:
            seeds = np.asarray(polarizing_seeds, dtype=np.float64)
            diff_seeds = opinions[:, None, :] - seeds[None, :, :]
            dist_seeds = np.linalg.norm(diff_seeds, axis=-1)
            dist_safe = np.maximum(dist_seeds, 1e-6)

            repulsion_mag = self.beta * np.exp(-dist_seeds / self.lambda_decay) / dist_safe
            seed_force = np.sum(repulsion_mag[:, :, None] * diff_seeds, axis=1)
            drift += seed_force

        return drift

    def simulate_opinion_trajectory(
        self,
        initial_opinions: np.ndarray,
        num_steps: int = 25,
        dt: float = 0.2,
        polarizing_seeds: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Integrates continuous opinion trajectory using Heun / RK2 integration.
        """
        current_x = np.copy(initial_opinions)
        polarization_history = []

        for _ in range(num_steps):
            k1 = self.compute_opinion_drift_fast(current_x, polarizing_seeds)
            x_mid = current_x + 0.5 * dt * k1
            k2 = self.compute_opinion_drift_fast(x_mid, polarizing_seeds)
            current_x += dt * k2

            com = np.mean(current_x, axis=0)
            disp_sq = np.sum((current_x - com) ** 2, axis=-1)
            polarization_history.append(float(np.mean(disp_sq)))

        return current_x, polarization_history


def direct_opinion_drift_reference(
    opinions: np.ndarray,
    eps: float,
    beta: float = 0.0,
    seeds: Optional[np.ndarray] = None,
    decay_length_lambda: float = 1.0
) -> np.ndarray:
    """Exact dense O(N^2) baseline for opinion drift computation."""
    opinions = np.asarray(opinions, dtype=np.float64)
    if opinions.ndim != 2:
        raise ValueError("opinions must have shape (N, D)")
    eps = float(eps)
    decay_length_lambda = float(decay_length_lambda)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    if not np.isfinite(decay_length_lambda) or decay_length_lambda <= 0.0:
        raise ValueError("decay_length_lambda must be finite and positive")
    n = len(opinions)
    drift = np.zeros_like(opinions)

    diff = opinions[None, :, :] - opinions[:, None, :]
    dist = np.linalg.norm(diff, axis=-1)
    
    mask = dist <= eps
    q = np.clip(dist / eps, 0.0, 1.0)
    w = np.where(mask, (1.0 - q) ** 2, 0.0)

    for i in range(n):
        sum_w = np.sum(w[i])
        if sum_w > 0:
            drift[i] = np.sum(w[i, :, None] * diff[i], axis=0) / sum_w

    if seeds is not None and beta > 0.0:
        diff_s = opinions[:, None, :] - seeds[None, :, :]
        dist_s = np.linalg.norm(diff_s, axis=-1)
        rep_mag = beta * np.exp(-dist_s / decay_length_lambda) / np.maximum(dist_s, 1e-6)
        drift += np.sum(rep_mag[:, :, None] * diff_s, axis=1)

    return drift


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("Continuous Non-Local Opinion Dynamics (Hegselmann-Krause) Benchmark")
    print("=" * 70)

    n_citizens = 8000
    eps_confidence = 0.25
    print(f"Number of Simulated Citizens : {n_citizens:,}")
    print(f"Bounded Confidence Radius (e): {eps_confidence:.2f}")

    initial_opinions = np.random.uniform(-1.0, 1.0, size=(n_citizens, 2))
    media_seeds = np.array([[-0.8, -0.8], [+0.8, +0.8]])

    engine = ContinuousOpinionDynamicsFMM(
        confidence_radius_eps=eps_confidence,
        dim=2,
        algorithmic_bias_beta=0.08
    )

    # 1. Fast Spatial Hash Drift Step
    t0 = time.perf_counter()
    drift_fast = engine.compute_opinion_drift_fast(initial_opinions, polarizing_seeds=media_seeds)
    t_fast = (time.perf_counter() - t0) * 1000.0

    print(f"Fast Tree-Free Drift Step    : {t_fast:.2f} ms")

    # 2. Dense Reference Evaluation on subset
    n_sub = 1500
    t0 = time.perf_counter()
    drift_ref_sub = direct_opinion_drift_reference(
        initial_opinions[:n_sub], eps=eps_confidence, beta=0.08, seeds=media_seeds
    )
    t_dense_sub = (time.perf_counter() - t0) * 1000.0
    t_dense_proj = t_dense_sub * ((n_citizens * n_citizens) / (n_sub * n_sub))

    drift_fast_sub = engine.compute_opinion_drift_fast(initial_opinions[:n_sub], polarizing_seeds=media_seeds)
    rel_error = np.linalg.norm(drift_fast_sub - drift_ref_sub) / np.linalg.norm(drift_ref_sub)

    print(f"Projected Dense O(N^2) Time  : {t_dense_proj:.2f} ms")
    print(f"Measured Speedup Ratio       : {t_dense_proj / max(t_fast, 1e-6):.1f}x")
    print(f"Relative L2 Precision Error  : {rel_error:.2e}")

    # 3. Simulate continuous trajectory (20 steps)
    t0 = time.perf_counter()
    final_ops, pol_hist = engine.simulate_opinion_trajectory(
        initial_opinions, num_steps=20, dt=0.2, polarizing_seeds=media_seeds
    )
    t_traj = (time.perf_counter() - t0) * 1000.0
    print(f"20-Step Trajectory Evolution : {t_traj:.2f} ms ({t_traj / 20.0:.2f} ms/step)")
    print(f"Initial Variance: {pol_hist[0]:.4f} -> Final Polarization Variance: {pol_hist[-1]:.4f}")
    print("=" * 70)
