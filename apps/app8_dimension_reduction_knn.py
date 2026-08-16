"""
Application 8: High-to-Low Dimensional Manifold Projection & Unfolding via Hash-Accelerated KNN.
Powered by Farach-Colton / Krapivin / Kuszmaul (2025) Non-Reordering Open Addressing.

Demonstrates the core intuitive concept:
1. Projects an 8-dimensional non-linear manifold (or 3D Swiss Roll) down to 2D.
2. Uses Random Hyperplane Projections + Non-Reordering Hash Table to construct the k-NN graph in O(N) without O(N^2) pairwise comparisons.
3. Computes 2D Spectral / Laplacian Eigenmap embedding.
"""

import sys
import os
from typing import Any, Tuple, Dict, List, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import time
from core.elastic_hash import ElasticHashTable
try:
    from scipy.sparse import coo_matrix, diags, eye as sparse_eye
    from scipy.sparse.linalg import eigsh
except ImportError:
    coo_matrix = diags = sparse_eye = eigsh = None

def generate_high_dim_swiss_roll(n_samples: int = 2500, ambient_dim: int = 8):
    """Generates a non-linear 2D manifold embedded into an 8D ambient space with noise."""
    np.random.seed(42)
    # Intrinsic 2D coordinates: t (angle along spiral) and h (height)
    t = 1.5 * np.pi * (1 + 2 * np.random.uniform(0, 1, n_samples))
    h = np.random.uniform(0, 20, n_samples)
    
    # 3D Swiss Roll
    x = t * np.cos(t)
    y = h
    z = t * np.sin(t)
    base_3d = np.stack([x, y, z], axis=1)
    
    # Embed into ambient_dim (e.g. 8D) via orthogonal random basis + slight noise
    proj_matrix = np.random.randn(ambient_dim, 3)
    Q, _ = np.linalg.qr(proj_matrix)  # (ambient_dim, 3) orthogonal
    
    ambient_pts = np.matmul(base_3d, Q.T) + np.random.normal(0, 0.05, size=(n_samples, ambient_dim))
    return ambient_pts, t, base_3d

def build_hash_knn_graph(points: np.ndarray, k_neighbors: int = 12, n_tables: int = 5, n_hyperplanes: int = 6):
    """
    Constructs an approximate k-NN adjacency graph in O(N) using
    Multi-Table Locality Sensitive Hashing (LSH) + Elastic Non-Reordering Table.
    """
    N, D = points.shape
    if N == 0 or k_neighbors < 1:
        raise ValueError("points must be non-empty and k_neighbors must be positive")
    edge_weights = {}
    table_cap = 1 << (n_hyperplanes + 2)
    
    for t in range(n_tables):
        np.random.seed(42 + t)
        hyperplanes = np.random.randn(D, n_hyperplanes)
        projections = np.matmul(points, hyperplanes) > 0
        powers_of_two = 1 << np.arange(n_hyperplanes, dtype=np.int64)
        lsh_keys = np.sum(projections * powers_of_two[None, :], axis=1)
        
        hash_table = ElasticHashTable(capacity=table_cap, delta=0.05)
        bucket_map = {}
        for i in range(N):
            key = int(lsh_keys[i])
            if key not in bucket_map:
                bucket_map[key] = []
            bucket_map[key].append(i)
            
        for key, indices in bucket_map.items():
            hash_table.insert(key, indices)
            
        for i in range(N):
            key = int(lsh_keys[i])
            bucket_indices, _ = hash_table.lookup(key)
            if bucket_indices is not None and len(bucket_indices) > 1:
                cand_pts = points[bucket_indices]
                dists = np.linalg.norm(points[i] - cand_pts, axis=1)
                top_local = np.argsort(dists)[:min(k_neighbors + 1, len(dists))]
                for idx in top_local:
                    neighbor_idx = bucket_indices[idx]
                    if neighbor_idx != i:
                        w = np.exp(-dists[idx]**2 / 8.0)
                        edge = (min(i, neighbor_idx), max(i, neighbor_idx))
                        edge_weights[edge] = max(edge_weights.get(edge, 0.0), w)

    if coo_matrix is not None:
        if not edge_weights:
            return coo_matrix((N, N), dtype=np.float32).tocsr()
        rows, cols, data = [], [], []
        for (i, j), weight in edge_weights.items():
            rows.extend((i, j))
            cols.extend((j, i))
            data.extend((weight, weight))
        return coo_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32).tocsr()
    else:
        # Pure NumPy sparse dictionary representation
        return {"N": N, "edges": edge_weights}

def compute_laplacian_eigenmap_2d(adj_matrix: Any) -> np.ndarray:
    """Unfolds manifold to 2D via Graph Laplacian eigenvectors."""
    if isinstance(adj_matrix, dict) and "edges" in adj_matrix:
        # Pure NumPy matrix-free normalized Laplacian power/Lanczos eigensolver
        N = adj_matrix["N"]
        if N < 3:
            raise ValueError("at least three nodes are required for a 2D eigenmap")
        
        degrees = np.zeros(N, dtype=np.float64)
        for (i, j), w in adj_matrix["edges"].items():
            degrees[i] += w
            degrees[j] += w
            
        inv_sqrt_d = 1.0 / np.sqrt(np.maximum(degrees, 1e-6))
        
        # Matrix-free normalized operator: M(v) = D^-1/2 A D^-1/2 v
        def M_matvec(v: np.ndarray) -> np.ndarray:
            u = v * inv_sqrt_d
            Au = np.zeros(N, dtype=np.float64)
            for (i, j), w in adj_matrix["edges"].items():
                Au[i] += w * u[j]
                Au[j] += w * u[i]
            return Au * inv_sqrt_d

        # Find top 3 eigenvectors of M (corresponding to smallest 3 eigenvectors of L_sym)
        # Power iteration with Gram-Schmidt orthogonalization
        rng = np.random.RandomState(42)
        k_vecs = 3
        V = rng.randn(k_vecs, N).astype(np.float64)
        for it in range(35):
            for k in range(k_vecs):
                V[k] = M_matvec(V[k])
                for prev in range(k):
                    V[k] -= np.dot(V[k], V[prev]) * V[prev]
                V[k] /= (np.linalg.norm(V[k]) + 1e-12)
                
        # Return eigenvectors 1 and 2 (ignoring trivial stationary eigenvector 0)
        return V[1:3].T
    else:
        if eigsh is None or not hasattr(adj_matrix, "tocsr"):
            raise TypeError("adj_matrix must be a SciPy sparse matrix or NumPy sparse dict")
        n = adj_matrix.shape[0]
        if n < 3:
            raise ValueError("at least three nodes are required for a 2D eigenmap")
        degrees = np.asarray(adj_matrix.sum(axis=1)).ravel()
        inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-6))
        normalized = diags(inv_sqrt) @ adj_matrix @ diags(inv_sqrt)
        laplacian = sparse_eye(n, format="csr") - normalized
        _, eigvecs = eigsh(laplacian, k=3, which="SM")
        return eigvecs[:, 1:3]

def run_dimension_reduction_demo():
    print("==================================================================")
    print(" APP 8: 8D MANIFOLD TO 2D PROJECTION VIA HASH-ACCELERATED KNN")
    print("==================================================================")
    N = 2500
    ambient_dim = 8
    print(f"Generating {N} points on non-linear manifold in {ambient_dim}D ambient space...")
    ambient_pts, manifold_color, base_3d = generate_high_dim_swiss_roll(N, ambient_dim)
    
    # 1. Exact O(N^2) KNN Reference Time
    t0 = time.perf_counter()
    # Compute small sample to extrapolate exact cost
    sample_diff = ambient_pts[:500, None, :] - ambient_pts[None, :500, :]
    _ = np.linalg.norm(sample_diff, axis=-1)
    t_exact_est = (time.perf_counter() - t0) * (N / 500)**2
    print(f"[-] Exact O(N^2) Full Pairwise Distance Est.: {t_exact_est*1000:.2f} ms")
    
    # 2. Hash-Accelerated KNN Graph Construction
    t0 = time.perf_counter()
    adj = build_hash_knn_graph(ambient_pts, k_neighbors=12, n_hyperplanes=10)
    t_hash_knn = time.perf_counter() - t0
    print(f"[-] Hash-Accelerated k-NN Graph Time:          {t_hash_knn*1000:.2f} ms")
    
    # 3. 2D Spectral Manifold Unfolding
    t0 = time.perf_counter()
    embedding_2d = compute_laplacian_eigenmap_2d(adj)
    t_unfold = time.perf_counter() - t0
    print(f"[-] 2D Manifold Eigen-Projection Time:         {t_unfold*1000:.2f} ms")
    
    # 4. Visualization: 3D Visual Slice vs 2D Unfolded Intrinsic Coordinates
    fig = plt.figure(figsize=(16, 6.5), facecolor='#0B0E14')
    
    # Plot 1: 3D Physical Slice of the 8D Manifold
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_facecolor('#0B0E14')
    p3d = ax1.scatter(base_3d[:, 0], base_3d[:, 1], base_3d[:, 2], c=manifold_color, 
                      cmap='turbo', s=14, alpha=0.85, edgecolors='none')
    ax1.view_init(elev=10, azim=-75)
    ax1.set_title(f"High-Dimensional Manifold (8D Ambient Space)\nCurved Non-Linear Topology (N={N})", 
                  color='white', fontsize=11, fontweight='bold', pad=12)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor('#30363D')
    ax1.yaxis.pane.set_edgecolor('#30363D')
    ax1.zaxis.pane.set_edgecolor('#30363D')
    ax1.tick_params(colors='#8B949E')
    
    # Plot 2: 2D Unfolded Intrinsic Projection via Hash-KNN
    ax2 = fig.add_subplot(122)
    ax2.set_facecolor('#0B0E14')
    p2d = ax2.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=manifold_color, 
                      cmap='turbo', s=16, alpha=0.85, edgecolors='none')
    
    # Tight bounding box with 6% margin to eliminate empty black voids
    x_min, x_max = embedding_2d[:, 0].min(), embedding_2d[:, 0].max()
    y_min, y_max = embedding_2d[:, 1].min(), embedding_2d[:, 1].max()
    dx = (x_max - x_min) * 0.06
    dy = (y_max - y_min) * 0.06
    ax2.set_xlim(x_min - dx, x_max + dx)
    ax2.set_ylim(y_min - dy, y_max + dy)
    
    cb = fig.colorbar(p2d, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label('Manifold Intrinsic Arc Length (t)', color='#E6EDF3')
    cb.ax.yaxis.set_tick_params(color='#8B949E')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='#8B949E')
    
    ax2.set_title("Unfolded 2D Intrinsic Projection (Hash-KNN Laplacian Map)\nPreserves Local Neighborhoods Without Pairwise O(N^2) Matrix", 
                  color='white', fontsize=11, fontweight='bold')
    ax2.tick_params(colors='#8B949E')
    for spine in ax2.spines.values():
        spine.set_color('#30363D')
        
    fig.suptitle("Application 8: High-to-Low Dimensional Manifold Unfolding via Non-Reordering Hash k-NN", 
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "app8_dimension_reduction_knn.png")
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[-] Saved 8D to 2D projection visualization to: {output_path}")

if __name__ == '__main__':
    run_dimension_reduction_demo()
