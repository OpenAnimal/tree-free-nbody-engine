"""
Application 1: Dynamic N-Body Galaxy Collision Simulation
Forces computed each step by the flat tree-free FMM (FastVectorizedFMM):
CGR88 2D logarithmic multipoles on a uniform cell grid whose occupied-cell
index is the non-reordering funnel hash (Farach-Colton/Krapivin/Kuszmaul,
arXiv:2501.02305). A direct-summation accuracy check runs at start-up.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from core.fast_vectorized_fmm import FastVectorizedFMM

def generate_spiral_galaxy(n_particles: int, center: np.ndarray, velocity: np.ndarray, radius: float = 0.15, arms: int = 2):
    """Generates rotating disk galaxy with logarithmic spiral arms."""
    r = np.random.exponential(scale=radius / 2.5, size=n_particles)
    r = np.clip(r, 0.01, radius)
    theta = np.random.uniform(0, 2 * np.pi, size=n_particles)
    theta += 2.0 * np.log(r / radius + 1e-4)

    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    positions = np.stack([x, y], axis=1)

    v_mag = np.sqrt(1.0 / (r + 0.02)) * 0.04
    vx = velocity[0] - v_mag * np.sin(theta)
    vy = velocity[1] + v_mag * np.cos(theta)
    velocities = np.stack([vx, vy], axis=1)

    masses = np.ones(n_particles) * (1.0 / n_particles)
    return positions, velocities, masses

def compute_fmm_gravitational_forces(fmm: FastVectorizedFMM, pos: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Attractive 2D logarithmic forces evaluated by the flat tree-free FMM."""
    pos = np.asarray(pos, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    _, fx, fy = fmm.evaluate(pos, masses, compute_forces=True)
    forces = np.stack([fx, fy], axis=1)
    # Clip extreme near-field singularities
    f_norm = np.linalg.norm(forces, axis=1, keepdims=True)
    forces = np.where(f_norm > 50.0, forces * (50.0 / (f_norm + 1e-6)), forces)
    return forces

def validate_against_direct(pos: np.ndarray, masses: np.ndarray, eps: float = 1e-4):
    """Max relative force error of the FMM vs exact direct summation."""
    fmm = FastVectorizedFMM(depth=4, order=6, softening=eps)
    _, fx, _ = fmm.evaluate(pos, masses, compute_forces=True)
    delta = pos[:, None, :] - pos[None, :, :]
    r_sq = np.sum(delta * delta, axis=-1) + eps ** 2
    np.fill_diagonal(r_sq, np.inf)
    dx = -np.sum(masses[None, :, None] * delta / r_sq[:, :, None], axis=1)[:, 0]
    denom = np.max(np.abs(dx))
    rel = np.max(np.abs(fx - dx)) / denom
    print(f"[-] FMM vs direct-summation validation: max relative force error = {rel:.3e}")
    assert rel < 1e-3, f"FMM force validation failed ({rel:.3e} >= 1e-3)"
    print("[-] Validation PASS (< 1e-3).")

def run_galaxy_collision(n_per_galaxy: int = 300, steps: int = 30, dt: float = 0.08):
    print(">>> Running Application 1: Dynamic Galaxy Collision (Flat Tree-Free FMM)")
    p1, v1, m1 = generate_spiral_galaxy(n_per_galaxy, center=np.array([0.35, 0.4]), velocity=np.array([0.08, 0.05]), radius=0.14)
    p2, v2, m2 = generate_spiral_galaxy(n_per_galaxy, center=np.array([0.65, 0.6]), velocity=np.array([-0.08, -0.05]), radius=0.14)

    pos = np.vstack([p1, p2])
    vel = np.vstack([v1, v2])
    masses = np.concatenate([m1, m2])

    validate_against_direct(pos, masses)
    fmm_engine = FastVectorizedFMM(depth=4, order=6, softening=1e-4)

    snapshots = []
    times = [0, steps // 3, 2 * steps // 3, steps - 1]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), facecolor='#0B0E14')
    fig.suptitle("Application 1: Dynamic N-Body Galaxy Collision\n"
                 "(Forces each step: Flat Tree-Free FMM + Non-Reordering Funnel Hash)",
                 color='white', fontsize=14, fontweight='bold')

    ax_idx = 0
    t_start = time.perf_counter()

    for step in range(steps):
        # Flat tree-free FMM force evaluation (funnel-hash cell index)
        acc = compute_fmm_gravitational_forces(fmm_engine, pos, masses)

        vel += acc * dt * 0.01
        pos += vel * dt

        pos[:, 0] = np.clip(pos[:, 0], 0.02, 0.98)
        pos[:, 1] = np.clip(pos[:, 1], 0.02, 0.98)

        if step in times:
            ax = axes[ax_idx]
            ax.set_facecolor('#0B0E14')
            ax.scatter(pos[:n_per_galaxy, 0], pos[:n_per_galaxy, 1], s=8, color='#00F0FF', alpha=0.8, label='Galaxy A')
            ax.scatter(pos[n_per_galaxy:, 0], pos[n_per_galaxy:, 1], s=8, color='#FF007F', alpha=0.8, label='Galaxy B')

            grid_res = 1 << 4
            for g in np.linspace(0, 1, grid_res + 1):
                ax.axvline(g, color='#30363D', lw=0.4, alpha=0.5)
                ax.axhline(g, color='#30363D', lw=0.4, alpha=0.5)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"Step {step} (t = {step*dt:.2f}s)\n"
                         f"Occupied FMM cells (funnel hash): {fmm_engine.hash_table.count}",
                         color='#E6EDF3', fontsize=10)
            ax.tick_params(colors='#8B949E')
            for spine in ax.spines.values():
                spine.set_color('#30363D')
            if ax_idx == 0:
                ax.legend(loc='upper right', facecolor='#161B22', edgecolor='#30363D', labelcolor='white')
            ax_idx += 1

    total_time = time.perf_counter() - t_start
    print(f"[-] Galaxy collision simulated {steps} steps in {total_time:.2f}s.")

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "app1_galaxy_collision.png")
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[-] Saved high-resolution visualization to: {output_path}")

if __name__ == '__main__':
    run_galaxy_collision(n_per_galaxy=250, steps=25, dt=0.08)
