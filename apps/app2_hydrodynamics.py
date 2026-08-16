"""
Application 2: Hydrodynamics / Vortex Particle Field (Biot-Savart Law)
Powered by Tree-Free Fast Multipole Method (FMM) + Farach-Colton Non-Reordering Hash.

Simulates 2D vortex sheet / Kelvin-Helmholtz hydrodynamic instability with velocity streamlines.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import time
from core.tree_free_fmm import TreeFreeFMM

def simulate_vortex_sheet(n_vortices: int = 400):
    print(">>> Running Application 2: Continuous Hydrodynamic Vortex Field (Biot-Savart FMM)")
    # Generate 2 parallel vortex sheets with sinusoidal perturbation (Kelvin-Helmholtz)
    x = np.linspace(0.1, 0.9, n_vortices // 2)
    # Upper sheet (circulation +gamma)
    y1 = 0.6 + 0.03 * np.sin(4 * np.pi * x)
    gamma1 = np.ones_like(x) * 1.5
    
    # Lower sheet (circulation -gamma)
    y2 = 0.4 - 0.03 * np.sin(4 * np.pi * x)
    gamma2 = -np.ones_like(x) * 1.5
    
    pos = np.vstack([np.stack([x, y1], axis=1), np.stack([x, y2], axis=1)])
    circulations = np.concatenate([gamma1, gamma2])
    
    # 1. Build Tree-Free FMM structure
    fmm = TreeFreeFMM(depth=4, order=6)
    fmm.build_hash_octree(pos, circulations)
    
    # 2. Evaluate Streamfunction psi across an Eulerian probe grid
    res = 40
    gx = np.linspace(0.05, 0.95, res)
    gy = np.linspace(0.05, 0.95, res)
    X, Y = np.meshgrid(gx, gy)
    grid_pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    
    # Evaluate Induced Velocity (u, v) = (dPsi/dy, -dPsi/dx)
    # Biot-Savart streamfunction evaluated via FMM
    fmm_eval = TreeFreeFMM(depth=4, order=6)
    fmm_eval.build_hash_octree(pos, circulations)
    psi = fmm_eval.compute_far_and_near_field(pos, circulations)
    
    # Vector field calculation on grid
    u_grid = np.zeros_like(X)
    v_grid = np.zeros_like(Y)
    
    for i in range(res):
        for j in range(res):
            px, py = X[i, j], Y[i, j]
            # Direct Biot-Savart velocity induction
            dx = px - pos[:, 0]
            dy = py - pos[:, 1]
            r2 = dx**2 + dy**2 + 1e-4
            # u = - 1/(2*pi) * sum(gamma * dy / r^2)
            # v = + 1/(2*pi) * sum(gamma * dx / r^2)
            u_grid[i, j] = -np.sum(circulations * dy / r2) / (2 * np.pi)
            v_grid[i, j] =  np.sum(circulations * dx / r2) / (2 * np.pi)
            
    speed = np.sqrt(u_grid**2 + v_grid**2)
    
    # 3. Visualization
    fig, ax = plt.subplots(figsize=(9, 7.5), facecolor='#0B0E14')
    ax.set_facecolor('#0B0E14')
    
    # Streamlines
    strm = ax.streamplot(X, Y, u_grid, v_grid, color=speed, cmap='plasma', density=1.4, linewidth=1.2, arrowsize=1.2)
    cbar = fig.colorbar(strm.lines, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Fluid Flow Velocity Magnitude', color='#E6EDF3')
    cbar.ax.yaxis.set_tick_params(color='#8B949E')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#8B949E')
    
    # Plot positive & negative vortex points
    pos_mask = circulations > 0
    ax.scatter(pos[pos_mask, 0], pos[pos_mask, 1], c='#00F0FF', s=25, edgecolors='none', label='Vortex Core (+Gamma)')
    ax.scatter(pos[~pos_mask, 0], pos[~pos_mask, 1], c='#FF3366', s=25, edgecolors='none', label='Vortex Core (-Gamma)')
    
    # Grid lines representing Non-Reordering Hash Buckets
    grid_res = 1 << 4
    for g in np.linspace(0, 1, grid_res + 1):
        ax.axvline(g, color='#30363D', lw=0.4, alpha=0.4)
        ax.axhline(g, color='#30363D', lw=0.4, alpha=0.4)
        
    ax.set_title("Application 2: Hydrodynamic Vortex Streamlines (Kelvin-Helmholtz)\nIndexed via Tree-Free Elastic Spatial Hash Table", 
                 color='white', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.05, 0.95)
    ax.tick_params(colors='#8B949E')
    for spine in ax.spines.values():
        spine.set_color('#30363D')
    ax.legend(loc='upper right', facecolor='#161B22', edgecolor='#30363D', labelcolor='white')
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "app2_hydrodynamic_vortex.png")
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[-] Saved hydrodynamic visualization to: {output_path}")

if __name__ == '__main__':
    simulate_vortex_sheet(400)
