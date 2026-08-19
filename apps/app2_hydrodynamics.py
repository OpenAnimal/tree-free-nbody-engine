"""
Application 2: Hydrodynamics / Vortex Particle Field (Biot-Savart Law)
The streamfunction psi(p) = -(1/2*pi) * sum_j gamma_j ln|p - p_j| is
evaluated on the probe grid by the flat tree-free FMM (FastVectorizedFMM,
CGR88 logarithmic multipoles, funnel-hash cell index). The velocity field
(u, v) = (dpsi/dy, -dpsi/dx) is then obtained by central finite
differences of the FMM-evaluated psi -- no O(N_grid * N) Biot-Savart
double loop. A direct Biot-Savart cross-check is printed and asserted.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core.fast_vectorized_fmm import FastVectorizedFMM

def biot_savart_direct(px, py, pos, circulations):
    dx = px - pos[:, 0]
    dy = py - pos[:, 1]
    r2 = dx ** 2 + dy ** 2 + 1e-4
    u = -np.sum(circulations * dy / r2) / (2 * np.pi)
    v = np.sum(circulations * dx / r2) / (2 * np.pi)
    return u, v

def simulate_vortex_sheet(n_vortices: int = 400):
    print(">>> Running Application 2: Continuous Hydrodynamic Vortex Field (FMM streamfunction)")
    x = np.linspace(0.1, 0.9, n_vortices // 2)
    y1 = 0.6 + 0.03 * np.sin(4 * np.pi * x)
    gamma1 = np.ones_like(x) * 1.5
    y2 = 0.4 - 0.03 * np.sin(4 * np.pi * x)
    gamma2 = -np.ones_like(x) * 1.5

    pos = np.vstack([np.stack([x, y1], axis=1), np.stack([x, y2], axis=1)])
    circulations = np.concatenate([gamma1, gamma2])

    # 1. Eulerian probe grid
    res = 80
    gx = np.linspace(0.05, 0.95, res)
    gy = np.linspace(0.05, 0.95, res)
    X, Y = np.meshgrid(gx, gy)
    grid_pts = np.stack([X.ravel(), Y.ravel()], axis=1)

    # 2. FMM evaluation of psi on the grid.
    # FastVectorizedFMM evaluates sources and targets together: append the
    # grid points as zero-charge particles so they contribute nothing but
    # receive the vortex potentials. phi = sum_j q_j ln|r - r_j|, and the
    # streamfunction psi = -(1/2*pi) * sum gamma_j ln|r - r_j|, so
    # psi = -phi / (2*pi).
    fmm = FastVectorizedFMM(depth=5, order=8, softening=1e-4)
    all_pos = np.vstack([pos, grid_pts])
    all_q = np.concatenate([circulations, np.zeros(len(grid_pts))])
    phi = fmm.evaluate(all_pos, all_q)[len(pos):]
    psi = -phi.reshape(res, res) / (2 * np.pi)

    # 3. Velocity field: (u, v) = (dpsi/dy, -dpsi/dx), central differences
    u_grid, v_grid = np.gradient(psi, gy, gx)
    u_grid, v_grid = u_grid, -v_grid  # np.gradient(psi, y, x): first is d/dy

    # 4. Cross-check vs direct Biot-Savart at points away from the sheets
    rng = np.random.RandomState(0)
    check_idx = rng.choice(res * res, size=60, replace=False)
    u_ex_all, v_ex_all = np.meshgrid(np.zeros(res), np.zeros(res))  # placeholder
    direct_speeds = []
    checks = []
    for idx in check_idx:
        i, j = divmod(int(idx), res)
        px, py = X[i, j], Y[i, j]
        d = np.sqrt((px - pos[:, 0]) ** 2 + (py - pos[:, 1]) ** 2)
        if d.min() < 0.05:
            continue  # skip points close to vortex cores (FD smoothing)
        checks.append((i, j))
        direct_speeds.append(np.hypot(*biot_savart_direct(px, py, pos, circulations)))
    speed_scale = np.sqrt(np.mean(np.array(direct_speeds) ** 2))  # RMS field speed
    max_rel = 0.0
    for i, j in checks:
        u_ex, v_ex = biot_savart_direct(X[i, j], Y[i, j], pos, circulations)
        rel = np.hypot(u_grid[i, j] - u_ex, v_grid[i, j] - v_ex) / speed_scale
        max_rel = max(max_rel, rel)
    print(f"[-] FMM-psi (finite-difference) vs direct Biot-Savart: max relative velocity error = {max_rel:.3e}")
    assert max_rel < 0.05, f"velocity validation failed ({max_rel:.3e} >= 5e-2)"
    print("[-] Validation PASS (< 5e-2, tolerance dominated by finite-difference grid spacing).")

    speed = np.sqrt(u_grid ** 2 + v_grid ** 2)

    # 5. Visualization
    fig, ax = plt.subplots(figsize=(9, 7.5), facecolor='#0B0E14')
    ax.set_facecolor('#0B0E14')

    strm = ax.streamplot(X, Y, u_grid, v_grid, color=speed, cmap='plasma', density=1.4, linewidth=1.2, arrowsize=1.2)
    cbar = fig.colorbar(strm.lines, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Fluid Flow Velocity Magnitude', color='#E6EDF3')
    cbar.ax.yaxis.set_tick_params(color='#8B949E')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#8B949E')

    pos_mask = circulations > 0
    ax.scatter(pos[pos_mask, 0], pos[pos_mask, 1], c='#00F0FF', s=25, edgecolors='none', label='Vortex Core (+Gamma)')
    ax.scatter(pos[~pos_mask, 0], pos[~pos_mask, 1], c='#FF3366', s=25, edgecolors='none', label='Vortex Core (-Gamma)')

    grid_res = 1 << 5
    for g in np.linspace(0, 1, grid_res + 1):
        ax.axvline(g, color='#30363D', lw=0.4, alpha=0.4)
        ax.axhline(g, color='#30363D', lw=0.4, alpha=0.4)

    ax.set_title("Application 2: Hydrodynamic Vortex Streamlines (Kelvin-Helmholtz)\n"
                 "Streamfunction evaluated by flat tree-free FMM (funnel-hash cells)",
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
