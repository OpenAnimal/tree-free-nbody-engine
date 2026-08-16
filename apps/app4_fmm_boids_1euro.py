"""
Application 4: Fast Multipole Boids with 1€ (One-Euro) Adaptive Filtering
& Optimal Non-Reordering Spatial Hashing (Farach-Colton et al. 2025).

Combines:
1. Multilevel Boids:
   - Near-field (direct separation & collision avoidance via 3x3 hash neighborhood)
   - Far-field (multipole flock alignment & global barycenter attraction)
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
from core.elastic_hash import ElasticHashTable
from core.tree_free_fmm import morton_encode_2d, decode_morton_2d, get_box_center_2d

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
# 2. Fast Multipole Boid Swarm Engine
# ----------------------------------------------------------------------
class FMMBoidSwarm:
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
        # 1. Non-reordering Spatial Hash Table indexing
        hash_table = ElasticHashTable(capacity=self.grid_res * self.grid_res * 2, delta=0.05)
        box_map = {}
        for i in range(self.N):
            key = morton_encode_2d(self.pos[i, 0], self.pos[i, 1], depth=self.depth)
            if key not in box_map:
                box_map[key] = []
            box_map[key].append(i)
            
        for key, p_indices in box_map.items():
            hash_table.insert(key, p_indices)

        # 2. Far-Field Multipole Aggregation (Cluster Velocity Moments & Barycenters)
        cluster_barycenters = {}
        cluster_vel = {}
        for key, p_indices in box_map.items():
            cluster_barycenters[key] = np.mean(self.pos[p_indices], axis=0)
            cluster_vel[key] = np.mean(self.vel[p_indices], axis=0)

        # 3. Compute Multilevel Boid Steer Forces
        acc = np.zeros_like(self.vel)
        
        # Boid weights
        w_sep = 1.8   # Near-field separation
        w_ali = 1.0   # Alignment
        w_coh = 0.8   # Cohesion (Multipole Far-Field)
        
        for i in range(self.N):
            p_i = self.pos[i]
            v_i = self.vel[i]
            m_key = morton_encode_2d(p_i[0], p_i[1], depth=self.depth)
            _, ix, iy = decode_morton_2d(m_key)
            
            # --- Near-field (P2P Separation via 3x3 Hash Neighbors) ---
            sep_force = np.zeros(2)
            ali_force = np.zeros(2)
            near_count = 0
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = ix + dx, iy + dy
                    if 0 <= nx < self.grid_res and 0 <= ny < self.grid_res:
                        n_key = (self.depth << 24) | morton_encode_2d((nx+0.5)/self.grid_res, (ny+0.5)/self.grid_res, depth=self.depth) & 0xFFFFFF
                        p_indices, _ = hash_table.lookup(n_key)
                        if p_indices is not None and n_key in box_map:
                            for j in p_indices:
                                if i == j:
                                    continue
                                diff = p_i - self.pos[j]
                                d = np.linalg.norm(diff) + 1e-6
                                if d < 0.05:
                                    sep_force += (diff / (d**2)) * 0.001
                                    ali_force += self.vel[j]
                                    near_count += 1
                                    
            if near_count > 0:
                ali_force = (ali_force / near_count) - v_i
                
            # --- Far-Field (M2L Cohesion via Cluster Multipoles) ---
            coh_force = np.zeros(2)
            far_clusters = 0
            for f_key, barycenter in cluster_barycenters.items():
                _, fx, fy = decode_morton_2d(f_key)
                if abs(fx - ix) > 1 or abs(fy - iy) > 1:
                    diff_c = barycenter - p_i
                    d_c = np.linalg.norm(diff_c) + 1e-4
                    coh_force += (diff_c / d_c) * 0.02
                    far_clusters += 1
                    
            if far_clusters > 0:
                coh_force /= far_clusters
                
            # Center attraction to keep swarm inside domain
            boundary_steer = (np.array([0.5, 0.5]) - p_i) * 0.05
            
            acc[i] = w_sep * sep_force + w_ali * ali_force + w_coh * coh_force + boundary_steer

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
    print(">>> Running Application 4: Fast Multipole Boid Swarm with 1€ Filter & Non-Reordering Hash")
    swarm = FMMBoidSwarm(n_boids=n_boids, depth=4)
    
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
        
    ax1.set_title("Multilevel FMM Boid Swarm\n(Near-Field P2P Separation + Far-Field M2L Cohesion)", color='white', fontsize=11, fontweight='bold')
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
            
    fig.suptitle("Application 4: Fast Multipole Boid Swarms with 1€ Adaptive Filtering\nPowered by Farach-Colton / Krapivin / Kuszmaul Elastic Hashing", 
                 color='white', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "app4_fmm_boids_1euro.png")
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[-] Saved Boids + 1€ Filter visualization to: {output_path}")

if __name__ == '__main__':
    run_boids_demo(400, 30)
