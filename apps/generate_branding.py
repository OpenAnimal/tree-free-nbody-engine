"""
Generates high-resolution social/header banner for Tree-Free N-Body Engine.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects

def generate_banner():
    fig = plt.figure(figsize=(12, 4.0), facecolor='#080C15', dpi=250)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#080C15')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.0)
    ax.axis('off')

    # Lattice Grid
    for i in np.linspace(0, 12, 25):
        ax.axvline(i, color='#131D2E', lw=0.5, alpha=0.35)
    for j in np.linspace(0, 4.0, 10):
        ax.axhline(j, color='#131D2E', lw=0.5, alpha=0.35)

    # Particle Scatter Layout (2 Radial Sections k=2)
    np.random.seed(42)
    n_pts = 450
    center_x, center_y = 9.8, 2.0
    r = np.random.exponential(scale=0.55, size=n_pts) + 0.12
    theta = np.random.uniform(0, 2*np.pi, size=n_pts) + 2.2 * np.log(r + 1e-3)
    px = center_x + r * np.cos(theta) * 1.55
    py = center_y + r * np.sin(theta) * 1.15
    
    # 2-fold galaxy swirl
    charges = np.sin(2 * theta)

    ax.scatter(px, py, c=charges, cmap='cool', s=16, alpha=0.85, edgecolors='none', zorder=2)

    for rad in [0.45, 0.95, 1.55, 2.25]:
        ellipse = patches.Ellipse((center_x, center_y), rad * 2 * 1.55, rad * 2 * 1.15, fill=False,
                                  edgecolor='#00F0FF', lw=1.0, linestyle='--', alpha=0.28 - rad*0.06, zorder=1)
        ax.add_patch(ellipse)

    ax.scatter([center_x], [center_y], c='#FFFFFF', s=70, edgecolors='#00F0FF', lw=2.0, zorder=5)

    # Typography
    x_pos = 0.35
    # Extra thick, rich dark stroke backdrop (linewidth=16.0) for crystal clear readability
    stroke_bg = [patheffects.withStroke(linewidth=16.0, foreground='#080C15')]
    
    # Main Header Title
    title = ax.text(x_pos, 2.70, "TREE-FREE  N-BODY", fontsize=41, fontweight='heavy',
                    color='#FFFFFF', fontfamily='sans-serif', zorder=4)
    title.set_path_effects([patheffects.withStroke(linewidth=16.0, foreground='#000000')])

    # High-tech Gradient Accent Line
    ax.plot([x_pos, 6.8], [2.42, 2.42], color='#7928CA', lw=3.0, alpha=0.8, zorder=3)
    ax.plot([x_pos, 3.8], [2.42, 2.42], color='#00DFD8', lw=3.0, alpha=1.0, zorder=4)

    # Subtitle: Clean Cyan Highlight
    sub = ax.text(x_pos, 2.02, "Pointerless Spatial Mechanics  •  Zero-Reordering Open Addressing",
                  fontsize=13.8, fontweight='semibold', color='#00F0FF', fontfamily='sans-serif', zorder=4)
    sub.set_path_effects(stroke_bg)

    # Symmetrically matched fully written out citations with prominent larger sizing & extra thick dark backdrop
    cit = ax.text(x_pos, 1.58, "Fast Multipole Method (Greengard & Rokhlin, 1987)  +  Elastic Open Addressing (Farach-Colton et al., 2025)",
                  fontsize=11.2, fontweight='medium', color='#E6EDF3', fontfamily='sans-serif', zorder=4)
    cit.set_path_effects(stroke_bg)

    # Lock-free specs line with elegant O(log(1/ε)) notation
    specs = ax.text(x_pos, 1.18, "Non-Reordering Hash  |  O(1) and O(log(1/ε)) Bounds  |  5M+ Real-Time Particles",
                    fontsize=11.0, fontweight='normal', color='#8B949E', fontfamily='sans-serif', zorder=4)
    specs.set_path_effects(stroke_bg)

    # Badges
    badges = ax.text(x_pos, 0.78, "[ SIMD / GPU Stream ]      [ WebGL 2.0 / WebGPU Real-Time ]",
                     fontsize=10.6, fontweight='medium', color='#58A6FF', fontfamily='sans-serif', alpha=0.95, zorder=4)
    badges.set_path_effects(stroke_bg)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(repo_root, "assets", "banner.png")
    root_banner = os.path.join(repo_root, "banner.png")
    plt.savefig(output_path, dpi=250, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(root_banner, dpi=250, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("[-] Successfully generated banner.png and assets/banner.png")

if __name__ == '__main__':
    generate_banner()
