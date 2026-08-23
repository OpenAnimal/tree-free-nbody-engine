"""
Application 3: Fast Spatial-Hash Attention for 2D Geometric Point Clouds.
Cell index: Farach-Colton, Krapivin, & Kuszmaul (2025) non-reordering
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
from core.spatial_index import CellIndex


def hash_spatial_attention(points: np.ndarray, V: np.ndarray,
                           sigma: float = 0.15, depth: int = 4) -> np.ndarray:
    """Near-exact (3x3 CellIndex neighborhood) + far-field per-cell centroid
    approximation. Vectorized per-cell using CellIndex (replaces the old
    raw ElasticHashTable + per-point Python loops).

    The dense spatial attention reference includes the self-pair w_ii = 1
    (exp(0) = 1), so this function also includes self-pairs (no masking).
    """
    n, d = V.shape
    grid_res = 1 << depth
    cell_index = CellIndex(dims=2, grid_res=grid_res)
    unique_keys, inverse = cell_index.build(points)
    K = len(unique_keys)

    # Cluster moments: barycenter and aggregated value sum per occupied cell.
    cluster_counts = np.bincount(inverse, minlength=K).astype(np.float64)
    cluster_centers = np.zeros((K, 2))
    cluster_v = np.zeros((K, d))
    for dim in range(2):
        cluster_centers[:, dim] = np.bincount(inverse, weights=points[:, dim],
                                              minlength=K) / np.maximum(cluster_counts, 1)
    for dim in range(d):
        cluster_v[:, dim] = np.bincount(inverse, weights=V[:, dim], minlength=K)

    cell_ints = np.array([cell_index.key_ints(int(k)) for k in unique_keys])
    inv_2_sigma_sq = 1.0 / (2.0 * sigma ** 2)

    out = np.zeros_like(V)
    for c, key in enumerate(unique_keys):
        idx_t = cell_index.bucket(int(key))
        if len(idx_t) == 0:
            continue
        pts_t = points[idx_t]  # (nt, 2)

        acc_v = np.zeros((len(idx_t), d))
        acc_w = np.zeros(len(idx_t)) + 1e-9

        # Near-field: 3x3 CellIndex neighborhood (includes self-pairs w_ii=1).
        near_idx = cell_index.neighborhood_indices(int(key), ring=1)
        if len(near_idx) > 0:
            pts_s = points[near_idx]   # (ns, 2)
            V_s = V[near_idx]          # (ns, d)
            diff = pts_t[:, None, :] - pts_s[None, :, :]  # (nt, ns, 2)
            d2 = np.sum(diff ** 2, axis=-1)               # (nt, ns)
            w = np.exp(-d2 * inv_2_sigma_sq)               # (nt, ns)
            acc_v += np.sum(w[:, :, None] * V_s[None, :, :], axis=1)
            acc_w += np.sum(w, axis=1)

        # Far-field: per-cell centroid approximation.
        cx, cy = cell_ints[c]
        far_mask = (np.abs(cell_ints[:, 0] - cx) > 1) | \
                   (np.abs(cell_ints[:, 1] - cy) > 1)
        far_clusters = np.where(far_mask)[0]
        if len(far_clusters) > 0:
            far_centers = cluster_centers[far_clusters]
            far_v = cluster_v[far_clusters]
            far_counts = cluster_counts[far_clusters]
            diff_c = pts_t[:, None, :] - far_centers[None, :, :]  # (nt, n_far, 2)
            d_c2 = np.sum(diff_c ** 2, axis=-1)                   # (nt, n_far)
            w_far = np.exp(-d_c2 * inv_2_sigma_sq)                 # (nt, n_far)
            acc_v += np.sum(w_far[:, :, None] * far_v[None, :, :], axis=1)
            acc_w += np.sum(w_far * far_counts[None, :], axis=1)

        out[idx_t] = acc_v / acc_w[:, None]
    return out


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
    fast_output = hash_spatial_attention(points, V, sigma=sigma, depth=depth)
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
    grid_res = 1 << depth
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
    vis_index = CellIndex(dims=2, grid_res=grid_res)
    vis_index.build(points)
    q_key = vis_index.key_of(q_pt)
    near_idx = vis_index.neighborhood_indices(int(q_key), ring=1)

    is_near = np.zeros(N_points, dtype=bool)
    is_near[near_idx] = True

    ax2.scatter(points[is_near, 0], points[is_near, 1], c='#00FFCC', s=22, label='Near-Field (Direct P2P via Hash)')
    ax2.scatter(points[~is_near, 0], points[~is_near, 1], c='#4B5563', s=15, alpha=0.6, label='Far-Field (Centroid Cluster Approx.)')
    ax2.scatter(q_pt[0], q_pt[1], c='#FF0055', s=120, marker='*', edgecolors='white', label='Query Point')

    # Grid lines
    for g in np.linspace(0, 1, grid_res + 1):
        ax2.axvline(g, color='#30363D', lw=0.4, alpha=0.4)
        ax2.axhline(g, color='#30363D', lw=0.4, alpha=0.4)
        
    ax2.set_title("Spatial Hash Partitioning (Funnel Hash, Farach-Colton, Krapivin, & Kuszmaul, 2025)", color='white', fontsize=11, fontweight='bold')
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
