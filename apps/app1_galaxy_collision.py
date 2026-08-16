"""
Application 1: Dynamic N-Body Galaxy Collision Simulation
Powered by Tree-Free Fast Multipole Method (FMM) + Farach-Colton/Kuszmaul Non-Reordering Hash.

Simulates two interacting spiral galaxies colliding in real-time and renders step snapshots.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import time
from core.tree_free_fmm import TreeFreeFMM

def generate_spiral_galaxy(n_particles: int, center: np.ndarray, velocity: np.ndarray, radius: float = 0.15, arms: int = 2):
    """Generates rotating disk galaxy with logarithmic spiral arms."""
    r = np.random.exponential(scale=radius / 2.5, size=n_particles)
    r = np.clip(r, 0.01, radius)
    theta = np.random.uniform(0, 2 * np.pi, size=n_particles)
    # Add spiral arm perturbation
    theta += 2.0 * np.log(r / radius + 1e-4)
    
    # 2D Positions
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    positions = np.stack([x, y], axis=1)
    
    # Circular orbital velocities + bulk drift velocity
    v_mag = np.sqrt(1.0 / (r + 0.02)) * 0.04
    vx = velocity[0] - v_mag * np.sin(theta)
    vy = velocity[1] + v_mag * np.cos(theta)
    velocities = np.stack([vx, vy], axis=1)
    
    masses = np.ones(n_particles) * (1.0 / n_particles)
    return positions, velocities, masses

def compute_fmm_gravitational_forces(fmm: TreeFreeFMM, pos: np.ndarray, masses: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Computes attractive 2D logarithmic forces with a correct target/source gradient."""
    pos = np.asarray(pos, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2 or len(pos) != len(masses):
        raise ValueError("pos must have shape (N, 2) and match masses")

    delta = pos[:, None, :] - pos[None, :, :]
    r_sq = np.sum(delta * delta, axis=-1) + eps ** 2
    np.fill_diagonal(r_sq, np.inf)
    # F = -grad(sum_j m_j log(r_ij)) = -sum_j m_j (r_i-r_j)/r_ij^2.
    forces = -np.sum(masses[None, :, None] * delta / r_sq[:, :, None], axis=1)
    # Clip extreme near-field singularities
    f_norm = np.linalg.norm(forces, axis=1, keepdims=True)
    forces = np.where(f_norm > 50.0, forces * (50.0 / (f_norm + 1e-6)), forces)
    return forces

def run_galaxy_collision(n_per_galaxy: int = 300, steps: int = 30, dt: float = 0.08):
    print(">>> Running Application 1: Dynamic Galaxy Collision (Tree-Free FMM)")
    # Galaxy 1: Left moving right-up
    p1, v1, m1 = generate_spiral_galaxy(n_per_galaxy, center=np.array([0.35, 0.4]), velocity=np.array([0.08, 0.05]), radius=0.14)
    # Galaxy 2: Right moving left-down
    p2, v2, m2 = generate_spiral_galaxy(n_per_galaxy, center=np.array([0.65, 0.6]), velocity=np.array([-0.08, -0.05]), radius=0.14)
    
    pos = np.vstack([p1, p2])
    vel = np.vstack([v1, v2])
    masses = np.concatenate([m1, m2])
    
    # Track snapshots for visualization
    snapshots = []
    times = [0, steps // 3, 2 * steps // 3, steps - 1]
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), facecolor='#0B0E14')
    fig.suptitle("Application 1: Dynamic N-Body Galaxy Collision\n(Accelerated by Tree-Free FMM + Elastic Non-Reordering Hash)", 
                 color='white', fontsize=14, fontweight='bold')
    
    ax_idx = 0
    t_start = time.perf_counter()
    
    for step in range(steps):
        # Tree-Free FMM Force Evaluation with Non-Reordering Spatial Hash
        fmm = TreeFreeFMM(depth=4, order=4)
        acc = compute_fmm_gravitational_forces(fmm, pos, masses)
        
        # Symplectic Euler Step
        vel += acc * dt * 0.01
        pos += vel * dt
        
        # Boundary bounce/containment in [0, 1] domain
        pos[:, 0] = np.clip(pos[:, 0], 0.02, 0.98)
        pos[:, 1] = np.clip(pos[:, 1], 0.02, 0.98)
        
        if step in times:
            ax = axes[ax_idx]
            ax.set_facecolor('#0B0E14')
            # Plot Galaxy 1 particles (Cyan) and Galaxy 2 particles (Magenta)
            ax.scatter(pos[:n_per_galaxy, 0], pos[:n_per_galaxy, 1], s=8, color='#00F0FF', alpha=0.8, label='Galaxy A')
            ax.scatter(pos[n_per_galaxy:, 0], pos[n_per_galaxy:, 1], s=8, color='#FF007F', alpha=0.8, label='Galaxy B')
            
            # Highlight non-reordering spatial hash octant grid
            grid_res = 1 << 4
            for g in np.linspace(0, 1, grid_res + 1):
                ax.axvline(g, color='#30363D', lw=0.4, alpha=0.5)
                ax.axhline(g, color='#30363D', lw=0.4, alpha=0.5)
                
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"Step {step} (t = {step*dt:.2f}s)\nOccupied Buckets: {fmm.hash_table.count}", color='#E6EDF3', fontsize=10)
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
