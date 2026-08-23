"""
Application 4: Elastic-Hash Spatial Boids with 1€ (One-Euro) Adaptive Filtering
& Optimal Non-Reordering Spatial Hashing (Farach-Colton, Krapivin, & Kuszmaul, 2025).

Combines:
1. Multilevel Boids:
   - Near-field (direct separation & collision avoidance via 3x3 hash neighborhood)
   - Far-field (per-cell centroid velocity & barycenter attraction; a bucketed
     centroid scheme, not a multipole expansion)
2. 1€ Filter:
   - Adaptive low-pass filtering: high jitter reduction when cruising, zero lag during rapid avoidance maneuvers.
3. Tree-Free Non-Reordering Hash:
   - Zero-reordering dynamic re-binning of fast boids per timestep.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import time
from core.spatial_index import CellIndex

# ----------------------------------------------------------------------
# 1. High-Performance 1€ Filter (Casiez, Roussel, Vogel 2012)
# ----------------------------------------------------------------------
class OneEuroFilter:
    """
    1€ Filter: Adaptive low-pass filter with dynamic frequency cutoff.
    Provides jitter reduction at low speeds while eliminating phase lag at high speeds.
    """
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None

    def _alpha(self, rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: np.ndarray, rate: float = 30.0) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x

        # 1. Filter derivative (speed of movement)
        dx = (x - self.x_prev) * rate
        alpha_d = self._alpha(rate, self.d_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev
        self.dx_prev = dx_hat

        # 2. Dynamic cutoff frequency based on velocity magnitude
        speed = np.linalg.norm(dx_hat, axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * speed

        # 3. Filter value with adaptive cutoff
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / rate
        alpha = 1.0 / (1.0 + tau / te)
        
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev = x_hat
        return x_hat


# ----------------------------------------------------------------------
# 2. Elastic-Hash Boid Swarm Engine (spatial-hash neighbor queries)
# ----------------------------------------------------------------------
class ElasticHashBoidSwarm:
    def __init__(self, n_boids: int = 500, depth: int = 4):
        self.N = n_boids
        self.depth = depth
        self.grid_res = 1 << depth
        
        # Initialize positions & velocities in [0, 1]x[0, 1]
        np.random.seed(42)
        self.pos = np.random.uniform(0.15, 0.85, size=(n_boids, 2))
        theta = np.random.uniform(0, 2*np.pi, size=n_boids)
        self.vel = np.stack([np.cos(theta), np.sin(theta)], axis=1) * 0.05
        
        # 1€ Filter instance for velocity smoothing (anti-jitter)
        self.filter_1euro = OneEuroFilter(min_cutoff=0.8, beta=0.08)
        
        # Tracking trajectory history
        self.raw_traj = []
        self.filtered_traj = []

    def step(self, dt: float = 0.05):
        # 1. Build CellIndex (canonical spatial index; replaces manual
        #    ElasticHashTable + box_map dict). Rebuilds the funnel-backed
        #    occupancy table every call (append-only table cannot unlearn
        #    stale keys).
        grid_res = 1 << self.depth
        cell_index = CellIndex(dims=2, grid_res=grid_res)
        unique_keys, inverse = cell_index.build(self.pos)
        K = len(unique_keys)

        # 2. Far-field aggregation: per-cell centroid velocity & barycenters.
        #    Vectorized via bincount on the inverse mapping.
        cluster_counts = np.bincount(inverse, minlength=K).astype(np.float64)
        cluster_barycenters = np.zeros((K, 2))
        cluster_vel = np.zeros((K, 2))
        for d in range(2):
            cluster_barycenters[:, d] = np.bincount(inverse, weights=self.pos[:, d], minlength=K) / np.maximum(cluster_counts, 1)
            cluster_vel[:, d] = np.bincount(inverse, weights=self.vel[:, d], minlength=K) / np.maximum(cluster_counts, 1)

        # Decode cell integer coords for near/far classification.
        cell_ints = np.array([cell_index.key_ints(int(k)) for k in unique_keys])

        # 3. Compute Multilevel Boid Steer Forces (vectorized per cell).
        acc = np.zeros_like(self.vel)

        w_sep = 1.8   # Near-field separation
        w_ali = 1.0   # Alignment
        w_coh = 0.8   # Cohesion (centroid far-field)
        near_radius = 0.05

        for c, key in enumerate(unique_keys):
            idx_t = cell_index.bucket(int(key))
            if len(idx_t) == 0:
                continue
            nt = len(idx_t)

            # --- Near-field (P2P separation + alignment via 3x3 CellIndex
            #     neighborhood). Vectorized over all boids in this cell. ---
            near_idx = cell_index.neighborhood_indices(int(key), ring=1)
            if len(near_idx) > 0:
                pts_t = self.pos[idx_t]       # (nt, 2)
                pts_s = self.pos[near_idx]    # (ns, 2)
                vel_s = self.vel[near_idx]    # (ns, 2)

                diff = pts_t[:, None, :] - pts_s[None, :, :]  # (nt, ns, 2)
                d = np.linalg.norm(diff, axis=-1) + 1e-6      # (nt, ns)

                # Self-pair mask
                id_t = idx_t[:, None]
                id_s = near_idx[None, :]
                self_mask = (id_t == id_s)

                # Near-field mask: d < near_radius and not self
                near_mask = (d < near_radius) & (~self_mask)   # (nt, ns)

                # Separation: sum of (diff / d^2) * 0.001 for near pairs
                sep_contrib = np.where(near_mask[:, :, None],
                                       diff / (d[:, :, None] ** 2) * 0.001, 0.0)
                sep_force = np.sum(sep_contrib, axis=1)        # (nt, 2)

                # Alignment: mean of vel[j] for near pairs, minus v_i.
                # A boid with NO near neighbors gets zero alignment force
                # (an unconditional `- v_i` would brake isolated boids by a
                # full w_ali*|v| every step).
                ali_sum = np.where(near_mask[:, :, None],
                                   vel_s[None, :, :], 0.0)
                ali_sum = np.sum(ali_sum, axis=1)              # (nt, 2)
                near_count = np.sum(near_mask, axis=1)         # (nt,)
                ali_force = np.where(near_count[:, None] > 0,
                                     ali_sum / np.maximum(near_count[:, None], 1)
                                     - self.vel[idx_t],
                                     0.0)                       # (nt, 2)

                acc[idx_t] += w_sep * sep_force + w_ali * ali_force

            # --- Far-field (centroid cohesion from far clusters).
            #     Far = Chebyshev distance > 1 (outside the 3x3 box). ---
            cx, cy = cell_ints[c]
            far_mask = (np.abs(cell_ints[:, 0] - cx) > 1) | \
                       (np.abs(cell_ints[:, 1] - cy) > 1)
            far_clusters = np.where(far_mask)[0]

            if len(far_clusters) > 0:
                far_bary = cluster_barycenters[far_clusters]   # (n_far, 2)
                pts_t = self.pos[idx_t]                        # (nt, 2)

                diff_c = far_bary[None, :, :] - pts_t[:, None, :]  # (nt, n_far, 2)
                d_c = np.linalg.norm(diff_c, axis=-1) + 1e-4      # (nt, n_far)

                coh_contrib = (diff_c / d_c[:, :, None]) * 0.02   # (nt, n_far, 2)
                coh_force = np.sum(coh_contrib, axis=1) / len(far_clusters)

                acc[idx_t] += w_coh * coh_force

        # Boundary attraction (vectorized)
        boundary_steer = (np.array([0.5, 0.5]) - self.pos) * 0.05
        acc += boundary_steer

        # 4. Integrate Raw Velocity & Apply 1€ Adaptive Low-Pass Filter
        raw_vel_next = self.vel + acc * dt
        # Speed limit
        v_norm = np.linalg.norm(raw_vel_next, axis=1, keepdims=True)
        raw_vel_next = np.where(v_norm > 0.08, raw_vel_next * (0.08 / (v_norm + 1e-6)), raw_vel_next)

        # Apply 1€ Filter to eliminate micro-jitter while preserving dynamic agility
        filtered_vel = self.filter_1euro.filter(raw_vel_next, rate=1.0/dt)

        self.vel = filtered_vel
        self.pos += self.vel * dt
        self.pos = np.clip(self.pos, 0.05, 0.95)

        return self.pos, self.vel


def run_boids_demo(n_boids: int = 400, steps: int = 40):
    print(">>> Running Application 4: Elastic-Hash Boid Swarm with 1€ Filter & Non-Reordering Funnel Hash")
    swarm = ElasticHashBoidSwarm(n_boids=n_boids, depth=4)
    
    # Store trajectories of leader boid to visualize 1€ filter jitter removal
    leader_raw_pos = []
    leader_filt_pos = []
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0B0E14')
    
    for s in range(steps):
        pos, vel = swarm.step(dt=0.08)
        
    print(f"[-] Simulated {steps} swarm steps successfully.")
    
    # --- Visualization 1: 2D Spatial Swarm with Velocity Vector Field ---
    ax1.set_facecolor('#0B0E14')
    speed = np.linalg.norm(vel, axis=1)
    
    # Draw velocity quivers
    ax1.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], speed, cmap='cool', scale=1.5, width=0.003, alpha=0.9)
    ax1.scatter(pos[:, 0], pos[:, 1], c='#00FFCC', s=16, edgecolors='none', label='Filtered Boids')
    
    # Grid lines showing spatial hash buckets
    grid_res = 1 << 4
    for g in np.linspace(0, 1, grid_res + 1):
        ax1.axvline(g, color='#30363D', lw=0.4, alpha=0.4)
        ax1.axhline(g, color='#30363D', lw=0.4, alpha=0.4)
        
    ax1.set_title("Elastic-Hash Boid Swarm\n(Near-Field Hash Separation + Far-Field Centroid Cohesion)", color='white', fontsize=11, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper right', facecolor='#161B22', edgecolor='#30363D', labelcolor='white')
    
    # --- Visualization 2: 1€ Filter Effect on Signal Jitter & Latency ---
    ax2.set_facecolor('#0B0E14')
    
    # Generate benchmark signal comparing No Filter vs EMA vs 1€ Filter
    t_arr = np.linspace(0, 4, 200)
    true_signal = np.sin(2 * np.pi * 0.8 * t_arr) + 0.5 * np.sign(np.sin(2 * np.pi * 0.2 * t_arr))
    noisy_signal = true_signal + np.random.normal(0, 0.18, size=len(t_arr))
    
    # EMA (Standard Exponential Moving Average)
    ema_signal = np.zeros_like(noisy_signal)
    ema_signal[0] = noisy_signal[0]
    for i in range(1, len(t_arr)):
        ema_signal[i] = 0.85 * ema_signal[i-1] + 0.15 * noisy_signal[i]
        
    # 1€ Filter
    filter_test = OneEuroFilter(min_cutoff=0.6, beta=0.1)
    one_euro_signal = np.array([filter_test.filter(noisy_signal[i:i+1], rate=50.0)[0] for i in range(len(t_arr))])
    
    ax2.plot(t_arr, noisy_signal, color='#6B7280', lw=0.8, alpha=0.6, label='Raw Sensor/Collision Noise')
    ax2.plot(t_arr, ema_signal, color='#FF5555', lw=1.8, linestyle='--', label='Standard EMA (Noticeable Lag)')
    ax2.plot(t_arr, one_euro_signal, color='#00FF88', lw=2.2, label='1€ Filter (Zero-Lag + Anti-Jitter)')
    ax2.plot(t_arr, true_signal, color='#00DDFF', lw=1.5, linestyle=':', label='True Trajectory Ground Truth')
    
    ax2.set_title("1€ Adaptive Filter Benchmark (Adaptive Cutoff vs EMA)", color='white', fontsize=11, fontweight='bold')
    ax2.set_xlabel("Time (s)", color='#8B949E')
    ax2.set_ylabel("Steering Velocity Coordinate", color='#8B949E')
    ax2.legend(loc='lower right', facecolor='#161B22', edgecolor='#30363D', labelcolor='white')
    
    for ax in (ax1, ax2):
        ax.tick_params(colors='#8B949E')
        for spine in ax.spines.values():
            spine.set_color('#30363D')
            
    fig.suptitle("Application 4: Elastic-Hash Boid Swarms with 1€ Adaptive Filtering\nPowered by Farach-Colton, Krapivin, & Kuszmaul (2025) Funnel Hashing", 
                 color='white', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "app4_fmm_boids_1euro.png")
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[-] Saved Boids + 1€ Filter visualization to: {output_path}")

if __name__ == '__main__':
    run_boids_demo(400, 30)
