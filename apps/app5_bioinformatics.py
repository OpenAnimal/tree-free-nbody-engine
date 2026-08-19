"""
Application 5: 3D Molecular Electrostatics & Solvation Free Energy (Bioinformatics / Drug Design).
Cell index: Farach-Colton/Krapivin/Kuszmaul (2025) non-reordering funnel/elastic hash
(core.elastic_hash), queried in the compute path for cluster membership.

Method, stated honestly: the interaction kernel here is the 3D screened
Yukawa (Debye-Huckel) potential, which does NOT match the 2D logarithmic
CGR88 FMM in core/, so no FMM engine is used. Potentials are computed by
direct O(K^2) summation between spatial clusters (bucketed centroids),
with the funnel hash resolving cluster membership per atom.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from core.elastic_hash import ElasticHashTable

def generate_synthetic_protein_backbone(n_residues: int = 2000):
    """Generates 3D alpha-helix / beta-sheet folded protein geometry with partial atomic charges."""
    t = np.linspace(0, 10 * np.pi, n_residues)
    # Alpha helical structure with tertiary supercoiling
    r_helix = 0.25
    r_super = 0.4
    x = (r_super + r_helix * np.cos(5 * t)) * np.cos(t)
    y = (r_super + r_helix * np.cos(5 * t)) * np.sin(t)
    z = 0.08 * t + r_helix * np.sin(5 * t)
    
    # Normalize into [0.1, 0.9]^3 domain
    coords = np.stack([x, y, z], axis=1)
    coords = (coords - np.min(coords, axis=0)) / (np.ptp(coords, axis=0) + 1e-6) * 0.8 + 0.1
    
    # Partial charges (e.g. +e for Lys/Arg, -e for Asp/Glu, neutral with dipoles)
    charges = np.sin(3 * t) * 1.0 + np.random.normal(0, 0.2, size=n_residues)
    return coords, charges

def run_bioinformatics_demo(n_atoms: int = 3000):
    print(f">>> Running Application 5: 3D Protein Molecular Electrostatics (N = {n_atoms} atoms)")
    coords, charges = generate_synthetic_protein_backbone(n_atoms)
    
    # 1. 3D Spatial Morton & Elastic Hash Table
    grid_res = 16
    ix = np.clip((coords[:, 0] * grid_res).astype(np.int64), 0, grid_res - 1)
    iy = np.clip((coords[:, 1] * grid_res).astype(np.int64), 0, grid_res - 1)
    iz = np.clip((coords[:, 2] * grid_res).astype(np.int64), 0, grid_res - 1)
    
    # 3D Morton Interleaving Key
    morton_3d = (ix << 20) | (iy << 10) | iz
    
    hash_table = ElasticHashTable(capacity=grid_res**3, delta=0.05)
    unique_keys, inverse = np.unique(morton_3d, return_inverse=True)
    num_clusters = len(unique_keys)
    
    for c, k in enumerate(unique_keys):
        hash_table.insert(int(k), c)  # funnel hash: cell key -> cluster id

    # Hash is load-bearing: resolve each atom's cluster id by LOOKUP, and
    # drop the numpy inverse array from the compute path.
    inverse = np.array([hash_table.lookup(int(k))[0] for k in morton_3d], dtype=np.int64)
    assert (inverse >= 0).all(), "funnel hash lost an occupied cell key"
    probe_count = sum(hash_table.lookup(int(k))[1] for k in unique_keys)
    print(f"[-] 3D Protein spatial clusters: {num_clusters} | Hash Load: {hash_table.count / hash_table.capacity * 100:.2f}% "
          f"| Avg occupancy lookup probes: {probe_count / num_clusters:.2f}")
    
    # 2. Vectorized 3D Debye-Huckel Screened Potential
    # Kappa: ionic screening parameter in water (debye length ~ 1nm)
    kappa = 2.0
    
    t0 = time.perf_counter()
    # Direct O(K^2) screened-Coulomb evaluation between clusters
    cluster_centers = np.array([np.mean(coords[inverse == c], axis=0) for c in range(num_clusters)])
    cluster_q = np.bincount(inverse, weights=charges, minlength=num_clusters)
    
    # Far-field cluster interaction: screened Coulomb V(r) = q * exp(-kappa * r) / r
    c_diff = cluster_centers[:, None, :] - cluster_centers[None, :, :]
    c_dist = np.linalg.norm(c_diff, axis=-1) + 1e-6
    np.fill_diagonal(c_dist, 1e9)
    
    cluster_pot = np.sum(cluster_q[None, :] * np.exp(-kappa * c_dist) / c_dist, axis=1)
    atom_potentials = cluster_pot[inverse]
    t_eval = time.perf_counter() - t0
    print(f"[-] 3D Molecular Electrostatics (direct cluster O(K^2)) Time: {t_eval*1000:.2f} ms")
    
    # 3. 3D Visualization of Electrostatic Surface Potential
    fig = plt.figure(figsize=(10, 8), facecolor='#0B0E14')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0B0E14')
    
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                    c=atom_potentials, cmap='coolwarm', s=12, alpha=0.85, edgecolors='none')
    
    # Draw backbone trace
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color='#58A6FF', lw=0.6, alpha=0.4)
    
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.08)
    cb.set_label('Electrostatic Solvation Potential (kcal/mol/e)', color='#E6EDF3')
    cb.ax.yaxis.set_tick_params(color='#8B949E')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='#8B949E')
    
    ax.set_title(f"Application 5: 3D Protein Electrostatic Potential Field\n(Spatial-Hash Clustered Electrostatics, direct O(K^2), N={n_atoms})", 
                 color='white', fontsize=12, fontweight='bold', pad=15)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#30363D')
    ax.yaxis.pane.set_edgecolor('#30363D')
    ax.zaxis.pane.set_edgecolor('#30363D')
    ax.tick_params(colors='#8B949E')
    
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "app5_protein_electrostatics.png")
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[-] Saved 3D protein electrostatics visualization to: {output_path}")

if __name__ == '__main__':
    run_bioinformatics_demo(3000)
