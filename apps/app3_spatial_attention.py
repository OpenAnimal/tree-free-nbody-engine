"""
Application 3: Fast Spatial-Hash Attention for 2D Geometric Point Clouds.
Cell index: Farach-Colton / Krapivin / Kuszmaul (2025) non-reordering
funnel/elastic hash (core.elastic_hash), used as the sole spatial index.

Method, stated honestly: near-field (3x3 cell neighborhood) attention is
computed exactly via O(1) hash lookups; far-field contributions use a
per-cell CENTROID approximation of the spatial RBF kernel (each distant
cell is summarized by its aggregated value at the cell center). This is
a bucketed centroid approximation, not a multipole expansion. The
approximation error vs dense spatial attention is measured and printed.
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
from core.tree_free_fmm import morton_encode_2d, decode_morton_2d, get_box_center_2d

def run_spatial_attention_demo(N_points: int = 1500, d_model: int = 16):
    print(">>> Running Application 3: Fast Spatial-Hash Attention (near exact + far centroid)")
    np.random.seed(42)
    
    # 1. Generate non-uniform 2D spatial point cloud (clustered robotic sensors / point clouds)
    c1 = np.random.normal(loc=[0.3, 0.3], scale=0.08, size=(N_points // 3, 2))
    c2 = np.random.normal(loc=[0.7, 0.7], scale=0.06, size=(N_points // 3, 2))
    c3 = np.random.normal(loc=[0.4, 0.7], scale=0.10, size=(N_points - 2 * (N_points // 3), 2))
    points = np.clip(np.vstack([c1, c2, c3]), 0.05, 0.95)
    
    # Features (Queries, Keys, Values)
    Q = np.random.randn(N_points, d_model)
    K = np.random.randn(N_points, d_model)
    V = np.random.randn(N_points, d_model)
    
    # --- Standard Dense O(N^2) Spatial Attention ---
    t0 = time.perf_counter()
    # Spatial RBF Kernel Attention: A_ij = exp(- ||x_i - x_j||^2 / (2 * sigma^2)) * (Q_i . K_j)
    sigma = 0.15
    diff = points[:, None, :] - points[None, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    spatial_kernel = np.exp(-dist_sq / (2 * sigma**2))
    
    feat_sim = np.matmul(Q, K.T) / np.sqrt(d_model)
    dense_attn_weights = spatial_kernel * np.exp(feat_sim - np.max(feat_sim, axis=-1, keepdims=True))
    dense_attn_weights /= (np.sum(dense_attn_weights, axis=-1, keepdims=True) + 1e-9)
    dense_output = np.matmul(dense_attn_weights, V)
    t_dense = time.perf_counter() - t0
    print(f"[-] Dense O(N^2) Spatial Attention Time: {t_dense*1000:.2f} ms")
    
    # --- Spatial-Hash Attention (near exact + far centroid approximation) ---
    t0 = time.perf_counter()
    depth = 4
    grid_res = 1 << depth
    hash_table = ElasticHashTable(capacity=grid_res * grid_res * 2, delta=0.05)
    
    # Step 1: Bucket points into the funnel hash (sole spatial index:
    # Morton cell key -> list of particle indices).
    for i in range(N_points):
        key = morton_encode_2d(points[i, 0], points[i, 1], depth=depth)
        p_indices, _ = hash_table.lookup(key)
        if p_indices is None:
            hash_table.insert(key, [i])
        else:
            p_indices.append(i)
    cell_keys = [k for k, _ in hash_table.items()]
    
    # Step 2: Compute cluster center aggregated values (centroid moments)
    cluster_centers = {}
    cluster_v = {}
    for key in cell_keys:
        p_indices = hash_table.lookup(key)[0]
        _, ix, iy = decode_morton_2d(key)
        cx, cy = get_box_center_2d(depth, ix, iy)
        cluster_centers[key] = np.array([cx, cy])
        cluster_v[key] = np.sum(V[p_indices], axis=0)  # aggregated value sum (centroid moment)
        
    # Step 3: Fast Attention Query: near-field direct + far-field centroid
    fast_output = np.zeros_like(V)
    for i in range(N_points):
        px, py = points[i, 0], points[i, 1]
        m_key = morton_encode_2d(px, py, depth=depth)
        _, ix, iy = decode_morton_2d(m_key)
        
        acc_v = np.zeros(d_model)
        acc_weight = 1e-9
        
        # Near-field search (3x3 adjacent buckets via O(1) hash probes)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = ix + dx, iy + dy
                if 0 <= nx < grid_res and 0 <= ny < grid_res:
                    n_key = (depth << 24) | morton_encode_2d((nx+0.5)/grid_res, (ny+0.5)/grid_res, depth=depth) & 0xFFFFFF
                    p_indices, _ = hash_table.lookup(n_key)
                    if p_indices is not None:
                        # Direct local attention
                        neigh_pts = points[p_indices]
                        d2 = np.sum((points[i] - neigh_pts)**2, axis=-1)
                        w = np.exp(-d2 / (2 * sigma**2))
                        acc_v += np.sum(w[:, None] * V[p_indices], axis=0)
                        acc_weight += np.sum(w)
                        
        # Far-field centroid approximation from distant clusters
        for f_key, c_center in cluster_centers.items():
            _, fx, fy = decode_morton_2d(f_key)
            if abs(fx - ix) > 1 or abs(fy - iy) > 1:
                dist_c2 = np.sum((points[i] - c_center)**2)
                w_far = np.exp(-dist_c2 / (2 * sigma**2))
                acc_v += w_far * cluster_v[f_key]
                acc_weight += w_far * len(hash_table.lookup(f_key)[0])
                
        fast_output[i] = acc_v / acc_weight
        
    t_fast = time.perf_counter() - t0
    print(f"[-] Spatial-Hash Attention Time:          {t_fast*1000:.2f} ms")

    # Approximation error vs dense SPATIAL attention (same kernel, no QK term)
    w_dense = spatial_kernel / (np.sum(spatial_kernel, axis=-1, keepdims=True) + 1e-9)
    dense_spatial_out = np.matmul(w_dense, V)
    denom = np.max(np.abs(dense_spatial_out))
    approx_err = np.max(np.abs(fast_output - dense_spatial_out)) / denom
    print(f"[-] Max relative error vs dense spatial attention: {approx_err:.3e} "
          f"(far-field centroid approximation + no QK feature term)")
    
    # 3. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), facecolor='#0B0E14')
    
    # Plot 1: Query attention field from an arbitrary query point
    query_idx = 100
    q_pt = points[query_idx]
    ax1.set_facecolor('#0B0E14')
    scatter = ax1.scatter(points[:, 0], points[:, 1], c=dense_attn_weights[query_idx], cmap='viridis', s=20, alpha=0.9)
    ax1.scatter(q_pt[0], q_pt[1], c='#FF0055', s=120, marker='*', edgecolors='white', label='Query Point')
    cb1 = fig.colorbar(scatter, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label('Exact Attention Weight', color='#E6EDF3')
    cb1.ax.yaxis.set_tick_params(color='#8B949E')
    plt.setp(plt.getp(cb1.ax.axes, 'yticklabels'), color='#8B949E')
    ax1.set_title("Exact Dense Attention Matrix A[q, :]", color='white', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', facecolor='#161B22', edgecolor='#30363D', labelcolor='white')
    
    # Plot 2: Spatial Hash Decomposition (Near vs Far buckets)
    ax2.set_facecolor('#0B0E14')
    q_key = morton_encode_2d(q_pt[0], q_pt[1], depth=depth)
    _, q_ix, q_iy = decode_morton_2d(q_key)
    
    # Color particles by Near vs Far field relative to query
    is_near = np.zeros(N_points, dtype=bool)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = q_ix + dx, q_iy + dy
            if 0 <= nx < grid_res and 0 <= ny < grid_res:
                n_key = (depth << 24) | morton_encode_2d((nx+0.5)/grid_res, (ny+0.5)/grid_res, depth=depth) & 0xFFFFFF
                p_indices, _ = hash_table.lookup(n_key)
                if p_indices is not None:
                    is_near[p_indices] = True
                    
    ax2.scatter(points[is_near, 0], points[is_near, 1], c='#00FFCC', s=22, label='Near-Field (Direct P2P via Hash)')
    ax2.scatter(points[~is_near, 0], points[~is_near, 1], c='#4B5563', s=15, alpha=0.6, label='Far-Field (Centroid Cluster Approx.)')
    ax2.scatter(q_pt[0], q_pt[1], c='#FF0055', s=120, marker='*', edgecolors='white', label='Query Point')
    
    # Grid lines
    for g in np.linspace(0, 1, grid_res + 1):
        ax2.axvline(g, color='#30363D', lw=0.4, alpha=0.4)
        ax2.axhline(g, color='#30363D', lw=0.4, alpha=0.4)
        
    ax2.set_title("Spatial Hash Partitioning (Funnel Hash, Farach-Colton et al.)", color='white', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', facecolor='#161B22', edgecolor='#30363D', labelcolor='white')
    
    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(colors='#8B949E')
        for spine in ax.spines.values():
            spine.set_color('#30363D')
            
    fig.suptitle("Application 3: Fast Spatial-Hash Attention for Point Clouds (near exact / far centroid)", 
                 color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "app3_spatial_attention.png")
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[-] Saved spatial attention visualization to: {output_path}")

if __name__ == '__main__':
    run_spatial_attention_demo(1500)
