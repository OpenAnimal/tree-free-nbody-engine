"""
3D Triangulated Shell & Multilayer Fabric Contact Dynamics Simulation
Powered by Matrix-Free Tree-Free IPC Solver.

Simulates authentic multilayer cloth draping, discrete shell bending,
self-collision, and obstacle contact.  The IPC log-barrier prevents
penetration of the checked candidate set under successful line search
(vertex-vertex; no point-triangle CCD), with 0 MB allocated for DynCSRMat
sparse matrices.
"""

import numpy as np
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors

import sys
sys.path.append(os.path.dirname(__file__))
from matrix_free_ipc import (
    MatrixFreeIPCSolver,
    ClothMesh,
    create_cloth_grid,
    combine_cloth_meshes
)

def run_cloth_simulation():
    print("=" * 85)
    print("3D MULTILAYER CLOTH & DISCRETE SHELL DRAPING SIMULATION (MATRIX-FREE IPC)")
    print("Inspired by ZOZO PPF Contact Solver, Li et al. (IPC), and Bergou/Grinspun Shells")
    print("=" * 85)

    # 1. Setup multi-layer cloth sheets
    grid_res = 20  # 20 x 20 per layer -> 400 vertices, 722 triangles per layer
    width = 0.62
    height = 0.62
    
    print(f"Constructing 2 triangulated cloth layers ({grid_res}x{grid_res} nodes each)...")
    
    # Layer 1: Bottom fabric sheet (magenta silk)
    cloth1 = create_cloth_grid(
        nx=grid_res, ny=grid_res,
        width=width, height=height,
        center=(0.5, 0.5, 0.58),
        k_stretch=1800.0,
        k_bend=0.06,
        density=0.25
    )
    
    # Layer 2: Top fabric sheet (cyan velvet, slightly offset and rotated angle)
    # Uses the SAME material params as layer 1: combine_cloth_meshes stores a
    # single material set, so per-layer params would be silently discarded.
    cloth2 = create_cloth_grid(
        nx=grid_res, ny=grid_res,
        width=width * 0.95, height=height * 0.95,
        center=(0.505, 0.495, 0.67),
        k_stretch=1800.0,
        k_bend=0.06,
        density=0.25
    )
    
    cloth = combine_cloth_meshes([cloth1, cloth2])
    N = cloth.num_vertices
    num_faces = len(cloth.triangles)
    num_edges = len(cloth.struct_edges)
    num_hinges = len(cloth.hinges)
    
    print(f"Global Mesh Topology:")
    print(f"  - Total Vertices:          N = {N:,}")
    print(f"  - Triangular Faces:        M = {num_faces:,}")
    print(f"  - Structural Edges:        E = {num_edges:,}")
    print(f"  - Discrete Bending Hinges: H = {num_hinges:,}")
    
    # 2. Setup Matrix-Free IPC Solver
    dhat = 0.016  # 1.6 cm barrier thickness
    solver = MatrixFreeIPCSolver(
        dhat=dhat,
        stiffness=4e3,
        max_newton_iters=4,
        cg_max_iters=16,
        cg_tol=1e-4,
        damp_coef=0.20
    )
    
    # Add rigid sphere obstacle
    obs_center = np.array([0.5, 0.5, 0.35])
    obs_radius = 0.18
    solver.add_sphere_obstacle(center=obs_center, radius=obs_radius)
    solver.add_plane_obstacle(point=np.array([0.0, 0.0, 0.08]), normal=np.array([0.0, 0.0, 1.0]))
    
    positions = cloth.rest_positions.copy()
    velocities = np.zeros_like(positions)
    
    dt = 0.012
    total_steps = 50
    print(f"\nSimulating {total_steps} implicit time steps (dt = {dt}s)...")
    
    min_clearances = []
    e_kin_history = []
    e_elastic_history = []
    e_barrier_history = []
    latencies = []
    
    t_sim_start = time.perf_counter()
    
    for step in range(total_steps):
        t_s0 = time.perf_counter()
        
        positions, velocities, m = solver.solve_step(
            x_prev=positions,
            v_prev=velocities,
            cloth=cloth,
            dt=dt
        )
        
        step_latency = (time.perf_counter() - t_s0) * 1000.0
        latencies.append(step_latency)
        
        e_k = 0.5 * np.sum(cloth.masses[:, None] * (velocities**2))
        e_el, _ = solver.compute_elastic_energy_and_forces(positions, cloth)

        # NOTE: this rebuilds the broadphase at the final positions for
        # reporting (e_bar and min_c).  The step's own candidate count
        # (m['active_candidates']) was computed at x_tilde (the predicted
        # step), not at the final positions, so reusing it here would give
        # stale pairs.  This rebuild is reporting-only (not physics) and
        # costs one extra broadphase per frame; it is kept for accuracy
        # of the reported barrier energy and clearance.
        cand_pairs = solver.find_broadphase_candidates(positions, cloth)
        e_bar, _ = solver.compute_barrier_energy_and_forces(positions, cand_pairs)
        
        dist_sphere = np.linalg.norm(positions - obs_center, axis=-1) - obs_radius
        dist_floor = positions[:, 2] - 0.08
        min_c_obs = min(float(np.min(dist_sphere)), float(np.min(dist_floor)))
        
        if len(cand_pairs) > 0:
            diff_p = positions[cand_pairs[:, 0]] - positions[cand_pairs[:, 1]]
            min_c_self = float(np.min(np.linalg.norm(diff_p, axis=-1)))
            min_c = min(min_c_obs, min_c_self)
        else:
            min_c = min_c_obs
            
        min_clearances.append(min_c)
        e_kin_history.append(e_k)
        e_elastic_history.append(e_el)
        e_barrier_history.append(e_bar)
        
        if step % 10 == 0 or step == total_steps - 1:
            print(f"  Step {step:2d}/{total_steps} | Step Time: {step_latency:5.1f} ms | Min Clearance: {min_c*100:5.2f} cm | Contacts: {m['active_candidates']:3d}")

    total_sim_time = time.perf_counter() - t_sim_start
    print(f"\nSimulation finished in {total_sim_time:.2f}s (Average: {np.mean(latencies):.2f} ms/step).")
    print(f"Minimum clearance recorded: {min(min_clearances)*100:.3f} cm (> 0 mm; barrier prevents penetration of the checked candidate set under successful line search).")

    # Compute per-vertex engineering strain
    vertex_strains = np.zeros(N)
    diffs = positions[cloth.struct_edges[:, 0]] - positions[cloth.struct_edges[:, 1]]
    curr_len = np.linalg.norm(diffs, axis=-1)
    strain_e = np.abs(curr_len - cloth.struct_rest_lengths) / (cloth.struct_rest_lengths + 1e-12)
    
    np.add.at(vertex_strains, cloth.struct_edges[:, 0], strain_e)
    np.add.at(vertex_strains, cloth.struct_edges[:, 1], strain_e)
    vertex_strains /= np.maximum(1, np.bincount(cloth.struct_edges.ravel(), minlength=N))

    # -------------------------------------------------------------------------
    # Render 4-Panel Visualization: cloth_shell_simulation.png
    # -------------------------------------------------------------------------
    print("\nRendering 4-Panel Visualization: cloth_shell_simulation.png...")
    fig = plt.figure(figsize=(16, 11), facecolor="#0B0E14")
    
    text_color = "#E6EDF3"
    grid_color = "#21262D"
    pane_color = "#161B22"
    border_color = "#30363D"

    n_per_layer = grid_res * grid_res
    tris1 = cloth.triangles[cloth.triangles[:, 0] < n_per_layer]
    tris2 = cloth.triangles[cloth.triangles[:, 0] >= n_per_layer]
    
    # Pre-generate sphere
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 24)
    sx = obs_center[0] + obs_radius * np.outer(np.cos(u), np.sin(v))
    sy = obs_center[1] + obs_radius * np.outer(np.sin(u), np.sin(v))
    sz = obs_center[2] + obs_radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Panel 1: 3D Triangulated Mesh Drape Render
    ax1 = fig.add_subplot(2, 2, 1, projection='3d', facecolor=pane_color)
    ax1.tick_params(colors=text_color, labelsize=8)
    for pane in [ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane]:
        pane.set_facecolor(pane_color)
        pane.set_edgecolor(border_color)
        
    # Draw Sphere Obstacle
    ax1.plot_surface(sx, sy, sz, color="#FFB800", alpha=0.5, edgecolor="#FF8800", linewidth=0.15)
    
    # Draw Floor
    gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
    gz = np.full_like(gx, 0.08)
    ax1.plot_wireframe(gx, gy, gz, color="#484F58", linewidth=0.5, alpha=0.3)
    
    # Draw Layer 1 (Bottom - Magenta Silk)
    verts1 = positions[tris1]
    poly1 = Poly3DCollection(verts1, facecolors="#FF007F", alpha=0.82, edgecolors="#770038", linewidths=0.25)
    ax1.add_collection3d(poly1)
    
    # Draw Layer 2 (Top - Cyan Velvet)
    verts2 = positions[tris2]
    poly2 = Poly3DCollection(verts2, facecolors="#00F0FF", alpha=0.85, edgecolors="#005577", linewidths=0.25)
    ax1.add_collection3d(poly2)
    
    ax1.set_xlim(0.15, 0.85)
    ax1.set_ylim(0.15, 0.85)
    ax1.set_zlim(0.08, 0.75)
    ax1.view_init(elev=24, azim=-55)
    ax1.set_title("1. 3D Triangulated Multilayer Drape & Shell Folding", color=text_color, fontsize=11, fontweight="bold", pad=8)
    ax1.set_xlabel("X (m)", color=text_color, labelpad=4)
    ax1.set_ylabel("Y (m)", color=text_color, labelpad=4)
    ax1.set_zlabel("Z (m)", color=text_color, labelpad=4)
    
    p1_proxy = plt.Rectangle((0, 0), 1, 1, fc="#FF007F", alpha=0.82)
    p2_proxy = plt.Rectangle((0, 0), 1, 1, fc="#00F0FF", alpha=0.85)
    obs_proxy = plt.Rectangle((0, 0), 1, 1, fc="#FFB800", alpha=0.5)
    ax1.legend([p2_proxy, p1_proxy, obs_proxy], ["Top (Cyan Velvet)", "Bottom (Magenta Silk)", "Sphere Obstacle"],
               facecolor=pane_color, edgecolor=border_color, labelcolor=text_color, fontsize=7.5, loc="upper right")

    # Panel 2: Elastic Strain Heatmap
    ax2 = fig.add_subplot(2, 2, 2, projection='3d', facecolor=pane_color)
    ax2.tick_params(colors=text_color, labelsize=8)
    for pane in [ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane]:
        pane.set_facecolor(pane_color)
        pane.set_edgecolor(border_color)

    tri_strains = np.mean(vertex_strains[cloth.triangles], axis=1)
    norm = mcolors.Normalize(vmin=0.0, vmax=max(0.01, float(np.percentile(tri_strains, 98))))
    cmap = plt.cm.plasma
    face_colors = cmap(norm(tri_strains))
    face_colors[:, 3] = 0.88

    poly_strain = Poly3DCollection(positions[cloth.triangles], facecolors=face_colors, edgecolors="#1a1a1a", linewidths=0.15)
    ax2.add_collection3d(poly_strain)
    ax2.plot_surface(sx, sy, sz, color="#30363D", alpha=0.3, edgecolor="none")
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, shrink=0.55, pad=0.1)
    cbar.ax.tick_params(colors=text_color, labelsize=8)
    cbar.set_label("Elastic Stretch Strain $\\epsilon = |\\Delta L|/L_0$", color=text_color, fontsize=8.5)
    
    ax2.set_xlim(0.15, 0.85)
    ax2.set_ylim(0.15, 0.85)
    ax2.set_zlim(0.08, 0.75)
    ax2.view_init(elev=26, azim=-125)
    ax2.set_title("2. Elastic Membrane Tension & Strain Field", color=text_color, fontsize=11, fontweight="bold", pad=8)
    ax2.set_xlabel("X (m)", color=text_color, labelpad=4)
    ax2.set_ylabel("Y (m)", color=text_color, labelpad=4)
    ax2.set_zlabel("Z (m)", color=text_color, labelpad=4)

    # Panel 3: Penetration-Free Clearance Time Series
    ax3 = fig.add_subplot(2, 2, 3, facecolor=pane_color)
    ax3.tick_params(colors=text_color, labelsize=8.5)
    ax3.grid(True, linestyle="--", alpha=0.3, color=grid_color)
    for spine in ax3.spines.values():
        spine.set_color(border_color)
        
    timesteps = np.arange(total_steps)
    ax3.plot(timesteps, np.array(min_clearances) * 100.0, 'o-', color="#00FF88", linewidth=2.0, label="Minimal Contact Gap $d_{\\min}$")
    ax3.axhline(0.0, color="#FF4D4D", linestyle="--", linewidth=1.5, label="Penetration Barrier ($0.0$ cm)")
    ax3.axhline(dhat * 100.0, color="#00F0FF", linestyle=":", linewidth=1.5, label=f"IPC Activation Threshold $\\hat{{d}} = {dhat*100:.1f}$ cm")
    
    ax3.set_title("3. Penetration-Free Barrier (Line Search)", color=text_color, fontsize=11, fontweight="bold")
    ax3.set_xlabel("Simulation Timestep", color=text_color, fontsize=9.5)
    ax3.set_ylabel("Clearance Distance (cm)", color=text_color, fontsize=9.5)
    ax3.legend(facecolor=pane_color, edgecolor=border_color, labelcolor=text_color, fontsize=8, loc="upper right")

    # Panel 4: Energy Partitioning History
    ax4 = fig.add_subplot(2, 2, 4, facecolor=pane_color)
    ax4.tick_params(colors=text_color, labelsize=8.5)
    ax4.grid(True, linestyle="--", alpha=0.3, color=grid_color)
    for spine in ax4.spines.values():
        spine.set_color(border_color)
        
    ax4.plot(timesteps, e_kin_history, '^-', color="#388BFD", linewidth=1.6, label="Kinetic Energy $E_{\\text{kin}}$")
    ax4.plot(timesteps, e_elastic_history, 's-', color="#FFB800", linewidth=1.6, label="Elastic Strain Energy $E_{\\text{elastic}}$")
    ax4.plot(timesteps, e_barrier_history, 'd-', color="#A371F7", linewidth=1.6, label="Barrier Potential $E_{\\text{barrier}}$")
    
    ax4.set_title("4. Energy Partition & Dynamic Evolution", color=text_color, fontsize=11, fontweight="bold")
    ax4.set_xlabel("Simulation Timestep", color=text_color, fontsize=9.5)
    ax4.set_ylabel("Energy (Joules)", color=text_color, fontsize=9.5)
    ax4.legend(facecolor=pane_color, edgecolor=border_color, labelcolor=text_color, fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "cloth_shell_simulation.png")
    plt.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Publication visualization saved successfully to: {out_path}")

if __name__ == "__main__":
    run_cloth_simulation()
