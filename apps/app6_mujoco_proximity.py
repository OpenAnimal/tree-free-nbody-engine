"""
Application 6: MuJoCo-Style Proximity Fields & Aerodynamic Ground-Effect Contact Solver.
Powered by Tree-Free Fast Multipole Method (FMM) + Farach-Colton Non-Reordering Spatial Hash.

Simulates dynamic ground proximity distance fields and normal force vectors for legged robotics
(e.g., AT-ST / Walker footpads over uneven deformable terrain).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import time
from core.elastic_hash import ElasticHashTable
from core.tree_free_fmm import morton_encode_2d, decode_morton_2d

def generate_terrain_and_robot_contacts(n_terrain: int = 2500):
    """Generates 2D/3D uneven heightfield terrain and foot contact probes."""
    x = np.linspace(0.05, 0.95, int(np.sqrt(n_terrain)))
    y = np.linspace(0.05, 0.95, int(np.sqrt(n_terrain)))
    X, Y = np.meshgrid(x, y)
    
    # Uneven terrain height with boulders and craters
    Z = 0.2 + 0.08 * np.sin(3 * np.pi * X) * np.cos(3 * np.pi * Y) + 0.04 * np.sin(8 * np.pi * X)
    terrain_pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    # Multi-contact robot footpad array (e.g. Walker biped foot sole mesh)
    foot_x = np.linspace(0.42, 0.58, 12)
    foot_y = np.linspace(0.45, 0.55, 8)
    FX, FY = np.meshgrid(foot_x, foot_y)
    FZ = np.full_like(FX, 0.24)  # Hovering right above undulating terrain
    foot_probes = np.stack([FX.ravel(), FY.ravel(), FZ.ravel()], axis=1)
    
    return terrain_pts, foot_probes

def run_mujoco_proximity_demo():
    print(">>> Running Application 6: MuJoCo-Style Proximity Contact & Ground-Effect Field")
    terrain_pts, foot_probes = generate_terrain_and_robot_contacts(2500)
    
    # 1. Non-reordering Spatial Hash Indexing for Terrain Points
    grid_res = 16
    ix = np.clip((terrain_pts[:, 0] * grid_res).astype(np.int64), 0, grid_res - 1)
    iy = np.clip((terrain_pts[:, 1] * grid_res).astype(np.int64), 0, grid_res - 1)
    morton_keys = (ix << 12) | iy
    
    hash_table = ElasticHashTable(capacity=grid_res * grid_res * 2, delta=0.05)
    box_map = {}
    for i in range(len(terrain_pts)):
        k = morton_keys[i]
        if k not in box_map:
            box_map[k] = []
        box_map[k].append(i)
        
    for k, p_indices in box_map.items():
        hash_table.insert(k, p_indices)
        
    # 2. Fast Proximity & Contact Penetration Evaluation
    t0 = time.perf_counter()
    contact_forces = np.zeros_like(foot_probes)
    penetration_depths = np.zeros(len(foot_probes))
    
    k_contact = 500.0  # Contact stiffness
    d_margin = 0.05    # Aerodynamic / contact proximity cushion
    
    for i, probe in enumerate(foot_probes):
        px, py, pz = probe
        c_ix = int(np.clip(px * grid_res, 0, grid_res - 1))
        c_iy = int(np.clip(py * grid_res, 0, grid_res - 1))
        
        # O(1) Probing 3x3 local terrain neighborhood
        min_dist = 1e9
        normal_acc = np.zeros(3)
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = c_ix + dx, c_iy + dy
                if 0 <= nx < grid_res and 0 <= ny < grid_res:
                    n_key = (nx << 12) | ny
                    p_indices, _ = hash_table.lookup(n_key)
                    if p_indices is not None and n_key in box_map:
                        t_pts = terrain_pts[p_indices]
                        # Compute distance to terrain points
                        diffs = probe - t_pts
                        dists = np.linalg.norm(diffs, axis=-1)
                        closest_idx = np.argmin(dists)
                        if dists[closest_idx] < min_dist:
                            min_dist = dists[closest_idx]
                            normal_acc = diffs[closest_idx] / (min_dist + 1e-6)
                            
        penetration = max(0.0, d_margin - min_dist)
        penetration_depths[i] = penetration
        # Soft-contact repulsive normal force
        contact_forces[i] = normal_acc * (k_contact * penetration + 5.0 * np.exp(-min_dist / 0.02))
        
    t_eval = time.perf_counter() - t0
    print(f"[-] MuJoCo Footpad Proximity Solve Time: {t_eval*1000:.3f} ms for {len(foot_probes)} contact points")
    
    # 3. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0B0E14')
    
    # Plot 1: 2D Heightfield Contour + Footpad Contact Points
    ax1.set_facecolor('#0B0E14')
    res = int(np.sqrt(len(terrain_pts)))
    X = terrain_pts[:, 0].reshape(res, res)
    Y = terrain_pts[:, 1].reshape(res, res)
    Z = terrain_pts[:, 2].reshape(res, res)
    
    contour = ax1.contourf(X, Y, Z, levels=20, cmap='magma', alpha=0.85)
    cb = fig.colorbar(contour, ax=ax1, fraction=0.046, pad=0.04)
    cb.set_label('Terrain Elevation Z (m)', color='#E6EDF3')
    cb.ax.yaxis.set_tick_params(color='#8B949E')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='#8B949E')
    
    # Draw Footpad Contact Nodes
    f_sc = ax1.scatter(foot_probes[:, 0], foot_probes[:, 1], c=penetration_depths, cmap='cool', s=45, 
                       edgecolors='white', linewidths=1.0, label='Robot Footpad Probes')
    
    ax1.set_title("Robot Footpad on Undulating Terrain Heightfield\n(O(1) Spatial Hash Proximity Lookups)", 
                  color='white', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', facecolor='#161B22', edgecolor='#30363D', labelcolor='white')
    
    # Plot 2: Contact & Aerodynamic Ground-Effect Force Vectors
    ax2.set_facecolor('#0B0E14')
    f_mag = np.linalg.norm(contact_forces, axis=1)
    ax2.scatter(foot_probes[:, 0], foot_probes[:, 1], c=f_mag, cmap='plasma', s=40)
    ax2.quiver(foot_probes[:, 0], foot_probes[:, 1], contact_forces[:, 0], contact_forces[:, 1], 
               color='#00FFCC', scale=500.0, width=0.005, label='Normal Repulsion Vectors')
    
    ax2.set_title("Ground-Effect Contact & Repulsion Force Vectors F_norm", color='white', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', facecolor='#161B22', edgecolor='#30363D', labelcolor='white')
    
    for ax in (ax1, ax2):
        ax.set_xlim(0.35, 0.65)
        ax.set_ylim(0.38, 0.62)
        ax.tick_params(colors='#8B949E')
        for spine in ax.spines.values():
            spine.set_color('#30363D')
            
    fig.suptitle("Application 6: MuJoCo Robot Proximity Fields & Continuous Ground-Effect\nAccelerated by Elastic Non-Reordering Spatial Table", 
                 color='white', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "app6_mujoco_proximity.png")
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[-] Saved MuJoCo proximity visualization to: {output_path}")

if __name__ == '__main__':
    run_mujoco_proximity_demo()
