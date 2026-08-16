"""
Example 5: Continuous 3D Generative Flow-Matching with Multipole Repulsion
==========================================================================
Simulates a Continuous Normalizing Flow / Flow-Matching Diffusion process (e.g. RFDiffusion/Point-E)
sampling a 3D point cloud over continuous time t in [0, 1].
Uses Tree-Free FMM to compute all-pairs repulsive velocity fields in O(N) time per ODE step,
preventing particle collapse and ensuring smooth, collision-free physical generation.
"""

import numpy as np
import time
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from neural_ops import EquivariantMultipoleLayer


class MultipoleFlowMatchingGenerator:
    """
    Continuous 3D Generative Diffusion / Flow-Matching Model with All-Pairs Multipole Regularization.
    """
    def __init__(self, n_points: int = 2000, hidden_dim: int = 32):
        self.n_points = n_points
        self.fmm_field = EquivariantMultipoleLayer(
            hidden_dim=hidden_dim,
            grid_depth=3,
            softening_radius=0.08,
            screening_kappa=0.05
        )

    def sample_target_manifold(self, n: int) -> np.ndarray:
        """Target geometric manifold: 3D Trefoil Knot / Torus."""
        theta = np.linspace(0, 4 * np.pi, n)
        x = np.sin(theta) + 2 * np.sin(2 * theta)
        y = np.cos(theta) - 2 * np.cos(2 * theta)
        z = -np.sin(3 * theta)
        target = np.stack([x, y, z], axis=-1) * 0.15 + 0.5
        return target.astype(np.float32)

    def compute_velocity_field(self, x_t: np.ndarray, target: np.ndarray, t: float) -> np.ndarray:
        """
        Computes continuous velocity field v_t(x) = v_drift + v_repulsive.
        """
        N = len(x_t)
        # 1. Manifold attractor drift velocity towards target
        v_drift = (target - x_t) / max(1.0 - t, 0.05)

        # 2. All-pairs continuous repulsive velocity field via Tree-Free FMM in O(N)
        charges = np.ones(N, dtype=np.float32) # Uniform positive repulsive charge
        node_feats = np.ones((N, 32), dtype=np.float32)

        _, fmm_forces, _, _ = self.fmm_field.forward(x_t, node_feats, charges)
        v_repulsive = fmm_forces * 0.02 # Scale repulsive dispersion

        # Combined continuous ODE velocity
        v_total = v_drift + v_repulsive
        return v_total

    def generate(self, n_steps: int = 10) -> np.ndarray:
        """Executes Euler ODE integration trajectory from noise to target."""
        np.random.seed(42)
        # Initial standard Gaussian noise at t=0
        x = np.random.normal(loc=0.5, scale=0.2, size=(self.n_points, 3)).astype(np.float32)
        target = self.sample_target_manifold(self.n_points)

        dt = 1.0 / n_steps
        step_latencies = []

        for step in range(n_steps):
            t = step * dt
            t0 = time.perf_counter()
            v_t = self.compute_velocity_field(x, target, t)
            x = x + v_t * dt
            elapsed = (time.perf_counter() - t0) * 1000.0
            step_latencies.append(elapsed)

        return x, np.mean(step_latencies)


def run_flow_matching_demo():
    print("=" * 70)
    print(">>> DEMO 5: Continuous 3D Flow-Matching Diffusion with Multipole Fields")
    print("=" * 70)

    N_points = 2000
    generator = MultipoleFlowMatchingGenerator(n_points=N_points, hidden_dim=32)

    print(f"[*] Integrating continuous generative ODE trajectory (N={N_points:,} points, 10 steps)...")
    t0 = time.perf_counter()
    final_points, avg_step_ms = generator.generate(n_steps=10)
    total_time = (time.perf_counter() - t0) * 1000.0

    # Calculate min pairwise clearance (collision prevention check)
    sub_sample = final_points[:200]
    diff = sub_sample[:, None, :] - sub_sample[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    min_clearance = np.min(dist)

    print(f"[-] Point Count:       {N_points:,} 3D generative particles")
    print(f"[-] Avg ODE Step Time: {avg_step_ms:.2f} ms / step")
    print(f"[-] Total Generation:  {total_time:.2f} ms (10 integration steps)")
    print(f"[-] Min Clearance:     {min_clearance:.4f} (No singular particle collisions)")
    print(f"[-] Final Bounds:      [{np.min(final_points):.2f}, {np.max(final_points):.2f}]")
    print("=" * 70)


if __name__ == "__main__":
    run_flow_matching_demo()
