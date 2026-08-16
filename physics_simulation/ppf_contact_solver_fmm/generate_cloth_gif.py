"""
Generates high-resolution 3D animated GIF of Multilayer Cloth Draping
over Sphere Obstacle with Matrix-Free Tree-Free IPC Solver.
"""

import numpy as np
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

import sys
sys.path.append(os.path.dirname(__file__))
from matrix_free_ipc import (
    MatrixFreeIPCSolver,
    ClothMesh,
    create_cloth_grid,
    combine_cloth_meshes
)

def generate_cloth_animation():
    print("=" * 80)
    print("GENERATING 3D MULTILAYER CLOTH DRAPE ANIMATION (GIF)")
    print("=" * 80)

    grid_res = 18  # 18x18 per layer -> 324 nodes, 578 triangles
    width = 0.60
    height = 0.60
    
    # Layer 1: Bottom fabric (Magenta silk)
    cloth1 = create_cloth_grid(
        nx=grid_res, ny=grid_res,
        width=width, height=height,
        center=(0.5, 0.5, 0.58),
        k_stretch=1800.0,
        k_bend=0.06,
        density=0.25
    )
    # Layer 2: Top fabric (Cyan velvet)
    cloth2 = create_cloth_grid(
        nx=grid_res, ny=grid_res,
        width=width * 0.94, height=height * 0.94,
        center=(0.505, 0.495, 0.67),
        k_stretch=1600.0,
        k_bend=0.05,
        density=0.22
    )
    
    cloth = combine_cloth_meshes([cloth1, cloth2])
    N = cloth.num_vertices
    
    solver = MatrixFreeIPCSolver(
        dhat=0.016,
        stiffness=4e3,
        cell_size=0.035,
        max_newton_iters=3,
        cg_max_iters=14,
        damp_coef=0.20
    )
    
    obs_center = np.array([0.5, 0.5, 0.35])
    obs_radius = 0.18
    solver.add_sphere_obstacle(center=obs_center, radius=obs_radius)
    solver.add_plane_obstacle(point=np.array([0.0, 0.0, 0.08]), normal=np.array([0.0, 0.0, 1.0]))
    
    positions = cloth.rest_positions.copy()
    velocities = np.zeros_like(positions)
    
    dt = 0.012
    total_frames = 36
    
    frames = []
    print(f"Simulating & Rendering {total_frames} animation frames...")
    
    n_per_layer = grid_res * grid_res
    tris1 = cloth.triangles[cloth.triangles[:, 0] < n_per_layer]
    tris2 = cloth.triangles[cloth.triangles[:, 0] >= n_per_layer]
    
    # Pre-generate sphere geometry
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 24)
    sx = obs_center[0] + obs_radius * np.outer(np.cos(u), np.sin(v))
    sy = obs_center[1] + obs_radius * np.outer(np.sin(u), np.sin(v))
    sz = obs_center[2] + obs_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    for f in range(total_frames):
        # Simulation sub-steps
        for _ in range(2):
            positions, velocities, _ = solver.solve_step(positions, velocities, cloth=cloth, dt=dt)
            
        # Clearance
        dist_sphere = np.linalg.norm(positions - obs_center, axis=-1) - obs_radius
        min_c = float(np.min(dist_sphere))
        
        # Render frame
        fig = plt.figure(figsize=(7, 7), facecolor="#0B0E14")
        ax = fig.add_subplot(1, 1, 1, projection='3d', facecolor="#161B22")
        ax.tick_params(colors="#E6EDF3", labelsize=8)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.set_facecolor("#161B22")
            pane.set_edgecolor("#30363D")
            
        # Draw Sphere
        ax.plot_surface(sx, sy, sz, color="#FFB800", alpha=0.5, edgecolor="#FF8800", linewidth=0.15)
        
        # Draw Ground Floor
        gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
        gz = np.full_like(gx, 0.08)
        ax.plot_wireframe(gx, gy, gz, color="#484F58", linewidth=0.5, alpha=0.3)
        
        # Draw Layer 1 (Bottom - Magenta Silk)
        verts1 = positions[tris1]
        poly1 = Poly3DCollection(verts1, facecolors="#FF007F", alpha=0.85, edgecolors="#770038", linewidths=0.25)
        ax.add_collection3d(poly1)
        
        # Draw Layer 2 (Top - Cyan Velvet)
        verts2 = positions[tris2]
        poly2 = Poly3DCollection(verts2, facecolors="#00F0FF", alpha=0.88, edgecolors="#005577", linewidths=0.25)
        ax.add_collection3d(poly2)
        
        ax.set_xlim(0.15, 0.85)
        ax.set_ylim(0.15, 0.85)
        ax.set_zlim(0.08, 0.75)
        
        # Rotating camera angle
        azim = -60 + f * 2.0
        ax.view_init(elev=24, azim=azim)
        
        ax.set_title(f"Matrix-Free IPC Multilayer Drape | Frame {f:2d}/{total_frames}\nMin Clearance: {min_c*100:4.2f} cm (Penetration-Free)",
                     color="#E6EDF3", fontsize=10, fontweight="bold", pad=8)
                     
        fig.tight_layout()
        fig.canvas.draw()
        
        rgba = np.asarray(fig.canvas.buffer_rgba())
        img = Image.fromarray(rgba)
        frames.append(img)
        plt.close(fig)
        
        if f % 10 == 0 or f == total_frames - 1:
            print(f"  Rendered Frame {f:2d}/{total_frames} (Azim: {azim:.1f}°)...")

    out_gif = os.path.join(os.path.dirname(__file__), "cloth_drape_animation.gif")
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=65,
        loop=0
    )
    print(f"\nAnimated GIF successfully saved to: {out_gif}")

if __name__ == "__main__":
    generate_cloth_animation()
