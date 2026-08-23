"""
Application 10: Continuous Graph Neural Network (GNN) Message Passing without Adjacency Matrices.
Cell index: Farach-Colton, Krapivin, & Kuszmaul (2025) non-reordering funnel/elastic hash.

Executes continuous spatial graph convolutions:
h_i^(l+1) = ReLU( W_self * h_i + sum_{near} W_near * h_j + sum_{far} W_far * Centroid_k )

Method, stated honestly: near-field messages are exchanged exactly within
the 3x3 funnel-hash neighborhood; far-field messages use per-cell
centroid aggregation (a bucketed centroid scheme, not a multipole
expansion -- the Gaussian message kernel is not the adaptive FMM log kernel).
Scales without N x N adjacency matrices or stored edge lists. The
approximation error vs a dense all-pairs message pass is measured and
printed for a small graph.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from typing import Tuple, List, Dict
from core.spatial_index import CellIndex

class ContinuousSpatialGNNLayer:
    """
    GNN layer with continuous spatial message passing.
    Near-field: exact message exchange within the 3x3 hash neighborhood.
    Far-field: per-cell centroid aggregation over distant cells.
    """
    def __init__(self, in_features: int = 32, out_features: int = 32, depth: int = 4):
        self.in_features = in_features
        self.out_features = out_features
        self.depth = depth
        self.grid_res = 1 << depth
        
        # Learnable GNN weights
        np.random.seed(42)
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W_self = np.random.randn(in_features, out_features) * scale
        self.W_near = np.random.randn(in_features, out_features) * scale
        self.W_far = np.random.randn(in_features, out_features) * scale
        self.bias = np.zeros(out_features)

    def forward(self, coords: np.ndarray, node_features: np.ndarray) -> np.ndarray:
        N = len(coords)

        # 1. Build CellIndex (canonical spatial index; replaces manual
        #    ElasticHashTable + per-node Python loops).
        cell_index = CellIndex(dims=2, grid_res=self.grid_res)
        unique_keys, inverse = cell_index.build(coords)
        K = len(unique_keys)

        # 2. Far-field aggregation: per-cell centroid features (vectorized
        #    via bincount on the inverse mapping).
        cluster_counts = np.bincount(inverse, minlength=K).astype(np.float64)
        cluster_centers = np.zeros((K, 2))
        for d in range(2):
            cluster_centers[:, d] = np.bincount(inverse, weights=coords[:, d],
                                                minlength=K) / np.maximum(cluster_counts, 1)
        cluster_h = np.zeros((K, self.in_features))
        for d in range(self.in_features):
            cluster_h[:, d] = np.bincount(inverse, weights=node_features[:, d],
                                          minlength=K) / np.maximum(cluster_counts, 1)

        cell_ints = np.array([cell_index.key_ints(int(k)) for k in unique_keys])

        # 3. Self-transformation
        h_self = np.matmul(node_features, self.W_self)  # (N, out_features)

        # 4. Near-field + far-field message passing (vectorized per cell).
        near_msg_all = np.zeros((N, self.in_features))
        far_msg_all = np.zeros((N, self.in_features))

        for c, key in enumerate(unique_keys):
            idx_t = cell_index.bucket(int(key))
            if len(idx_t) == 0:
                continue
            pts_t = coords[idx_t]  # (nt, 2)

            # --- Near-field: 3x3 CellIndex neighborhood ---
            near_idx = cell_index.neighborhood_indices(int(key), ring=1)
            if len(near_idx) > 0:
                pts_s = coords[near_idx]          # (ns, 2)
                feat_s = node_features[near_idx]  # (ns, in_features)

                diff = pts_t[:, None, :] - pts_s[None, :, :]  # (nt, ns, 2)
                d = np.linalg.norm(diff, axis=-1) + 1e-4      # (nt, ns)
                w = np.exp(-d ** 2 / 0.05)                     # (nt, ns)

                # Self-pair mask (exclude self, matching the original code)
                id_t = idx_t[:, None]
                id_s = near_idx[None, :]
                self_mask = (id_t == id_s)
                w = np.where(self_mask, 0.0, w)

                # Mean of w*feat divided by count (matching original: /= near_count)
                near_count = np.sum(~self_mask, axis=1)  # (nt,)
                near_msg = np.where(
                    near_count[:, None] > 0,
                    (w[:, :, None] * feat_s[None, :, :]).sum(axis=1) / np.maximum(near_count[:, None], 1),
                    0.0)
                near_msg_all[idx_t] = near_msg

            # --- Far-field: per-cell centroid message ---
            cx, cy = cell_ints[c]
            far_mask = (np.abs(cell_ints[:, 0] - cx) > 1) | \
                       (np.abs(cell_ints[:, 1] - cy) > 1)
            far_clusters = np.where(far_mask)[0]

            if len(far_clusters) > 0:
                far_centers = cluster_centers[far_clusters]  # (n_far, 2)
                far_h = cluster_h[far_clusters]              # (n_far, in_features)

                diff_c = pts_t[:, None, :] - far_centers[None, :, :]  # (nt, n_far, 2)
                d_c = np.linalg.norm(diff_c, axis=-1) + 1e-4          # (nt, n_far)
                w_far = np.exp(-d_c ** 2 / 0.2)                        # (nt, n_far)

                # Mean of w_far*cluster_h divided by far_count (matching original)
                far_msg = (w_far[:, :, None] * far_h[None, :, :]).sum(axis=1) / len(far_clusters)
                far_msg_all[idx_t] = far_msg

        # 5. Combine transformations + ReLU
        total_h = h_self + np.matmul(near_msg_all, self.W_near) + np.matmul(far_msg_all, self.W_far) + self.bias
        return np.maximum(0.0, total_h)


def run_continuous_gnn_demo():
    print("==================================================================")
    print(" APP 10: CONTINUOUS GRAPH NEURAL NETWORK MESSAGE PASSING (spatial-hash GNN)")
    print("==================================================================")
    N_NODES = 2000
    in_dim = 32
    out_dim = 16
    print(f"Executing GNN Forward Pass on {N_NODES} graph nodes (In: {in_dim} -> Out: {out_dim})...")
    
    np.random.seed(42)
    # Generate non-uniform 2D spatial graph topology
    c1 = np.random.normal(loc=[0.3, 0.3], scale=0.08, size=(N_NODES // 3, 2))
    c2 = np.random.normal(loc=[0.7, 0.7], scale=0.06, size=(N_NODES // 3, 2))
    c3 = np.random.normal(loc=[0.4, 0.7], scale=0.10, size=(N_NODES - 2 * (N_NODES // 3), 2))
    coords = np.clip(np.vstack([c1, c2, c3]), 0.05, 0.95)
    
    node_features = np.random.randn(N_NODES, in_dim)
    
    # 1. Forward Pass Benchmark
    gnn_layer = ContinuousSpatialGNNLayer(in_features=in_dim, out_features=out_dim, depth=4)
    t0 = time.perf_counter()
    out_h = gnn_layer.forward(coords, node_features)
    t_gnn = time.perf_counter() - t0
    
    print(f"[-] GNN Layer Execution Time: {t_gnn*1000:.2f} ms")
    print(f"[-] Edge Matrix Memory Stored: 0 MB (Completely Matrix-Free)")

    # Dense all-pairs reference (small N) to quantify the far-field
    # centroid approximation error.
    N_small = 150
    ref_layer = ContinuousSpatialGNNLayer(in_features=in_dim, out_features=out_dim, depth=4)
    ref_layer.W_self = gnn_layer.W_self; ref_layer.W_near = gnn_layer.W_near
    ref_layer.W_far = gnn_layer.W_far; ref_layer.bias = gnn_layer.bias
    small_coords, small_feat = coords[:N_small], node_features[:N_small]
    out_approx = ref_layer.forward(small_coords, small_feat)
    W_near, W_far, bias = gnn_layer.W_near, gnn_layer.W_far, gnn_layer.bias
    h_self = np.matmul(small_feat, gnn_layer.W_self)
    dense = np.zeros_like(out_approx)
    for i in range(N_small):
        d = np.linalg.norm(small_coords[i] - small_coords, axis=1) + 1e-4
        w = np.exp(-d ** 2 / 0.05); w[i] = 0.0
        near_msg = (w[:, None] * small_feat).sum(axis=0) / max(w.sum(), 1e-9)
        w_far = np.exp(-d ** 2 / 0.2); w_far[i] = 0.0
        far_msg = (w_far[:, None] * small_feat).sum(axis=0) / max(w_far.sum(), 1e-9)
        dense[i] = np.maximum(0.0, h_self[i] + near_msg @ W_near + far_msg @ W_far + bias)
    rel = np.max(np.abs(out_approx - dense)) / (np.max(np.abs(dense)) + 1e-12)
    print(f"[-] Max relative error vs dense all-pairs message pass (N={N_small}): {rel:.3e}")
    
    # 2. Visualization: Node Embeddings & Message Passing Field
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), facecolor='#0B0E14')
    
    # Plot 1: Input Graph Spatial Topology & Raw Node Energy
    ax1.set_facecolor('#0B0E14')
    raw_energy = np.linalg.norm(node_features, axis=1)
    s1 = ax1.scatter(coords[:, 0], coords[:, 1], c=raw_energy, cmap='magma', s=18, alpha=0.85)
    cb1 = fig.colorbar(s1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label('Input Node Activation Magnitude', color='#E6EDF3')
    cb1.ax.yaxis.set_tick_params(color='#8B949E')
    plt.setp(plt.getp(cb1.ax.axes, 'yticklabels'), color='#8B949E')
    
    ax1.set_title("Input Dynamic Graph Nodes (No Pre-Defined Edge List)", color='white', fontsize=11, fontweight='bold')
    
    # Plot 2: Output Node Features after Dual-Scale Spatial Message Passing
    ax2.set_facecolor('#0B0E14')
    out_energy = np.linalg.norm(out_h, axis=1)
    s2 = ax2.scatter(coords[:, 0], coords[:, 1], c=out_energy, cmap='viridis', s=18, alpha=0.85)
    cb2 = fig.colorbar(s2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label('Transformed Latent Representation', color='#E6EDF3')
    cb2.ax.yaxis.set_tick_params(color='#8B949E')
    plt.setp(plt.getp(cb2.ax.axes, 'yticklabels'), color='#8B949E')
    
    # Overlay Morton Hash Grid
    grid_res = 1 << 4
    for g in np.linspace(0, 1, grid_res + 1):
        ax2.axvline(g, color='#30363D', lw=0.4, alpha=0.4)
        ax2.axhline(g, color='#30363D', lw=0.4, alpha=0.4)
        
    ax2.set_title("Transformed Latent Node States (Continuous Spatial Convolutions)", color='white', fontsize=11, fontweight='bold')
    
    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(colors='#8B949E')
        for spine in ax.spines.values():
            spine.set_color('#30363D')
            
    fig.suptitle("Application 10: Matrix-Free Continuous Graph Neural Network\nMessage Passing via Farach-Colton, Krapivin, & Kuszmaul (2025) Funnel Hash (near exact / far centroid)", 
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "app10_continuous_gnn_fmm.png")
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[-] Saved continuous GNN visualization to: {output_path}")

if __name__ == '__main__':
    run_continuous_gnn_demo()
