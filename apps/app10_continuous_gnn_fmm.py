"""
Application 10: Continuous Graph Neural Network (GNN) Message Passing without Adjacency Matrices.
Powered by Tree-Free Fast Multipole Method (FMM) & Farach-Colton Non-Reordering Hash.

Executes continuous spatial graph convolutions:
h_i^(l+1) = ReLU( W_self * h_i + sum_{near} W_near * h_j + sum_{far} W_far * Centroid_k )
Scales to massive dynamic graphs without allocating N x N adjacency matrices or storing edge lists.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple, List, Dict
from core.elastic_hash import ElasticHashTable
from core.tree_free_fmm import morton_encode_2d, decode_morton_2d, get_box_center_2d

class ContinuousFMMGNNLayer:
    """
    Graph Neural Network layer executing continuous spatial message passing in O(N).
    Near-field: exact attention message exchange within hash neighborhood.
    Far-field: multipole centroid message aggregation across distant graph clusters.
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
        grid_res = self.grid_res
        
        # 1. Non-Reordering Hash Table Dynamic Indexing
        hash_table = ElasticHashTable(capacity=grid_res * grid_res * 2, delta=0.05)
        bucket_map = {}
        for i in range(N):
            key = morton_encode_2d(coords[i, 0], coords[i, 1], depth=self.depth)
            if key not in bucket_map:
                bucket_map[key] = []
            bucket_map[key].append(i)
            
        for key, p_indices in bucket_map.items():
            hash_table.insert(key, p_indices)
            
        # 2. Far-Field Multipole Aggregation (Cluster Centroid Moments)
        cluster_h = {}
        cluster_centers = {}
        for key, p_indices in bucket_map.items():
            _, ix, iy = decode_morton_2d(key)
            cx, cy = get_box_center_2d(self.depth, ix, iy)
            cluster_centers[key] = np.array([cx, cy])
            cluster_h[key] = np.mean(node_features[p_indices], axis=0)  # Aggregated node representation
            
        # 3. Continuous Message Passing: Self + Near (P2P) + Far (M2L)
        out_features = np.zeros((N, self.out_features))
        
        # Self-transformation: (N, out_features)
        h_self = np.matmul(node_features, self.W_self)
        
        # Fast message aggregation
        for i in range(N):
            px, py = coords[i, 0], coords[i, 1]
            m_key = morton_encode_2d(px, py, depth=self.depth)
            _, ix, iy = decode_morton_2d(m_key)
            
            near_msg = np.zeros(self.in_features)
            near_count = 0
            
            # Near-field search (3x3 adjacent buckets via O(1) hash probes)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = ix + dx, iy + dy
                    if 0 <= nx < grid_res and 0 <= ny < grid_res:
                        n_key = (self.depth << 24) | morton_encode_2d((nx+0.5)/grid_res, (ny+0.5)/grid_res, depth=self.depth) & 0xFFFFFF
                        p_indices, _ = hash_table.lookup(n_key)
                        if p_indices is not None and n_key in bucket_map:
                            for j in p_indices:
                                if i != j:
                                    d = np.linalg.norm(coords[i] - coords[j]) + 1e-4
                                    w = np.exp(-d**2 / 0.05)
                                    near_msg += w * node_features[j]
                                    near_count += 1
                                    
            if near_count > 0:
                near_msg /= near_count
                
            # Far-field multipole message aggregation
            far_msg = np.zeros(self.in_features)
            far_count = 0
            for f_key, c_center in cluster_centers.items():
                _, fx, fy = decode_morton_2d(f_key)
                if abs(fx - ix) > 1 or abs(fy - iy) > 1:
                    d_c = np.linalg.norm(coords[i] - c_center) + 1e-4
                    w_far = np.exp(-d_c**2 / 0.2)
                    far_msg += w_far * cluster_h[f_key]
                    far_count += 1
                    
            if far_count > 0:
                far_msg /= far_count
                
            # Combine transformations
            total_h = h_self[i] + np.matmul(near_msg, self.W_near) + np.matmul(far_msg, self.W_far) + self.bias
            # ReLU activation
            out_features[i] = np.maximum(0.0, total_h)
            
        return out_features


def run_continuous_gnn_demo():
    print("==================================================================")
    print(" APP 10: CONTINUOUS GRAPH NEURAL NETWORK MESSAGE PASSING (FMM-GNN)")
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
    gnn_layer = ContinuousFMMGNNLayer(in_features=in_dim, out_features=out_dim, depth=4)
    t0 = time.perf_counter()
    out_h = gnn_layer.forward(coords, node_features)
    t_gnn = time.perf_counter() - t0
    
    print(f"[-] GNN Layer Execution Time: {t_gnn*1000:.2f} ms")
    print(f"[-] Edge Matrix Memory Stored: 0 MB (Completely Matrix-Free)")
    
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
    
    # Plot 2: Output Node Features after Dual-Scale Multipole Message Passing
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
        
    ax2.set_title("Transformed Latent Node States (Continuous FMM Convolutions)", color='white', fontsize=11, fontweight='bold')
    
    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(colors='#8B949E')
        for spine in ax.spines.values():
            spine.set_color('#30363D')
            
    fig.suptitle("Application 10: Matrix-Free Continuous Graph Neural Network (FMM-GNN)\nMessage Passing via Farach-Colton / Kuszmaul Non-Reordering Spatial Table", 
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "app10_continuous_gnn_fmm.png")
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[-] Saved continuous GNN visualization to: {output_path}")

if __name__ == '__main__':
    run_continuous_gnn_demo()
