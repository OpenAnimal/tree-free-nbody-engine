"""
Application 8: High-to-Low Dimensional Manifold Projection & Unfolding via Hash-Accelerated KNN.
Powered by Farach-Colton / Krapivin / Kuszmaul (2025) Non-Reordering Open Addressing.

Demonstrates the core intuitive concept:
1. Projects an 8-dimensional non-linear manifold (Swiss Roll embedded in 8D) down to 2D.
2. Uses Random Hyperplane Projections + Non-Reordering Hash Table to construct the clean k-NN graph in O(N) without O(N^2) pairwise comparisons.
3. Computes isometric 2D manifold unfolding via graph geodesic distance embedding (Landmark Isomap / Shortest Paths).
"""

import sys
import os
import heapq
import time
from typing import Any, Tuple, Dict, List, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from core.elastic_hash import ElasticHashTable

def generate_high_dim_swiss_roll(n_samples: int = 2500, ambient_dim: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

def build_hash_knn_graph(points: np.ndarray, k_neighbors: int = 12, n_tables: int = 8, n_hyperplanes: int = 8) -> Dict[str, Any]:
    """
    Constructs an approximate k-NN adjacency graph in O(N) using
    Multi-Table Locality Sensitive Hashing (LSH) + Elastic Non-Reordering Table.
    Uses candidate pool aggregation across tables followed by local distance ranking.
    """
    N, D = points.shape
    if N == 0 or k_neighbors < 1:
        raise ValueError("points must be non-empty and k_neighbors must be positive")
    
    candidate_sets: List[set] = [set() for _ in range(N)]
    table_cap = 1 << (n_hyperplanes + 2)
    
    for t_idx in range(n_tables):
        np.random.seed(42 + t_idx)
        hyperplanes = np.random.randn(D, n_hyperplanes)
        projections = np.matmul(points, hyperplanes) > 0
        powers_of_two = 1 << np.arange(n_hyperplanes, dtype=np.int64)
        lsh_keys = np.sum(projections * powers_of_two[None, :], axis=1)
        
        hash_table = ElasticHashTable(capacity=table_cap, delta=0.05)
        bucket_map: Dict[int, List[int]] = {}
        for i in range(N):
            key = int(lsh_keys[i])
            if key not in bucket_map:
                bucket_map[key] = []
            bucket_map[key].append(i)
            
        for key, indices in bucket_map.items():
            hash_table.insert(key, indices)
            
        for i in range(N):
            key = int(lsh_keys[i])
            b_indices, _ = hash_table.lookup(key)
            if b_indices is not None:
                candidate_sets[i].update(b_indices)
    
    # Filter candidates to retain true top-k nearest neighbors per point
    adj_list: List[List[Tuple[int, float]]] = [[] for _ in range(N)]
    edge_weights: Dict[Tuple[int, int], float] = {}
    
    for i in range(N):
        cands = np.array(list(candidate_sets[i] - {i}), dtype=np.int64)
        if len(cands) == 0:
            continue
        cand_pts = points[cands]
        dists = np.linalg.norm(points[i] - cand_pts, axis=1)
        top_k = np.argsort(dists)[:min(k_neighbors, len(dists))]
        for idx in top_k:
            nbr = int(cands[idx])
            d = float(dists[idx])
            w = float(np.exp(-d**2 / 1.0))
            edge = (min(i, nbr), max(i, nbr))
            edge_weights[edge] = max(edge_weights.get(edge, 0.0), w)
            adj_list[i].append((nbr, d))
            adj_list[nbr].append((i, d))
            
    return {"N": N, "adj_list": adj_list, "edges": edge_weights}

def compute_geodesic_isomap_2d(graph_data: Dict[str, Any], n_landmarks: int = 60) -> np.ndarray:
    """
    Unfolds manifold to 2D via Landmark Isomap / Graph Geodesic Shortest Paths.
    Isometrically unrolls developable manifolds like the Swiss roll in O(L * (N + E)).
    """
    N = graph_data["N"]
    adj_list = graph_data["adj_list"]
    if N < 3:
        raise ValueError("at least three nodes are required for a 2D embedding")
        
    rng = np.random.RandomState(42)
    landmarks = rng.choice(N, size=min(n_landmarks, N), replace=False)
    
    def dijkstra_shortest_paths(start: int) -> np.ndarray:
        dists = np.full(N, np.inf, dtype=np.float64)
        dists[start] = 0.0
        pq = [(0.0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dists[u]:
                continue
            for v, weight in adj_list[u]:
                if dists[u] + weight < dists[v]:
                    dists[v] = dists[u] + weight
                    heapq.heappush(pq, (dists[v], v))
        return dists
    
    D_land = np.zeros((len(landmarks), N), dtype=np.float64)
    for i, l_idx in enumerate(landmarks):
        D_land[i] = dijkstra_shortest_paths(int(l_idx))
        
    # Replace any disconnected infinite values with max observed distance
    max_finite = np.max(D_land[np.isfinite(D_land)]) if np.any(np.isfinite(D_land)) else 100.0
    D_land[~np.isfinite(D_land)] = max_finite * 2.0
    
    # Classical Landmark Multidimensional Scaling (MDS)
    D_ll = D_land[:, landmarks]
    k_l = len(landmarks)
    H = np.eye(k_l) - 1.0 / k_l
    B = -0.5 * H @ (D_ll**2) @ H
    
    eigvals, eigvecs = np.linalg.eigh(B)
    sort_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]
    
    # Top 2 embedding coordinates
    mean_d2 = np.mean(D_ll**2, axis=0, keepdims=True)
    all_mean_d2 = np.mean(D_land**2, axis=0, keepdims=True)
    Delta = -0.5 * (D_land**2 - mean_d2.T - all_mean_d2 + np.mean(D_ll**2))
    
    inv_scale = 1.0 / np.sqrt(np.maximum(eigvals[:2], 1e-12))
    proj_coords = Delta.T @ (eigvecs[:, :2] * inv_scale[None, :])
    return proj_coords

def run_dimension_reduction_demo():
    print("==================================================================")
    print(" APP 8: 8D MANIFOLD TO 2D PROJECTION VIA HASH-ACCELERATED KNN")
    print("==================================================================")
    N = 2500
    ambient_dim = 8
    print(f"Generating {N} points on non-linear manifold in {ambient_dim}D ambient space...")
    ambient_pts, manifold_color, base_3d = generate_high_dim_swiss_roll(N, ambient_dim)
    h_coord = base_3d[:, 1]
    
    # 1. Exact O(N^2) KNN Reference Time
    t0 = time.perf_counter()
    sample_diff = ambient_pts[:500, None, :] - ambient_pts[None, :500, :]
    _ = np.linalg.norm(sample_diff, axis=-1)
    t_exact_est = (time.perf_counter() - t0) * (N / 500)**2
    print(f"[-] Exact O(N^2) Full Pairwise Distance Est.: {t_exact_est*1000:.2f} ms")
    
    # 2. Hash-Accelerated KNN Graph Construction
    t0 = time.perf_counter()
    graph_data = build_hash_knn_graph(ambient_pts, k_neighbors=12, n_tables=8, n_hyperplanes=8)
    t_hash_knn = time.perf_counter() - t0
    print(f"[-] Hash-Accelerated k-NN Graph Time:          {t_hash_knn*1000:.2f} ms")
    
    # 3. 2D Manifold Geodesic Unfolding
    t0 = time.perf_counter()
    embedding_2d = compute_geodesic_isomap_2d(graph_data, n_landmarks=60)
    t_unfold = time.perf_counter() - t0
    print(f"[-] 2D Manifold Geodesic Unfolding Time:       {t_unfold*1000:.2f} ms")
    
    # Align coordinate axes: assign t to X and h to Y
    if abs(np.corrcoef(embedding_2d[:, 0], manifold_color)[0, 1]) < abs(np.corrcoef(embedding_2d[:, 1], manifold_color)[0, 1]):
        embedding_2d = embedding_2d[:, [1, 0]]
    if np.corrcoef(embedding_2d[:, 0], manifold_color)[0, 1] < 0:
        embedding_2d[:, 0] = -embedding_2d[:, 0]
    if np.corrcoef(embedding_2d[:, 1], h_coord)[0, 1] < 0:
        embedding_2d[:, 1] = -embedding_2d[:, 1]
        
    r_t = abs(np.corrcoef(embedding_2d[:, 0], manifold_color)[0, 1])
    r_h = abs(np.corrcoef(embedding_2d[:, 1], h_coord)[0, 1])
    print(f"[-] Intrinsic Coordinates Correlation: r(t) = {r_t:.4f}, r(h) = {r_h:.4f}")
    
    # 4. Visualization: 3D Visual Slice vs 2D Unfolded Intrinsic Coordinates
    fig = plt.figure(figsize=(16, 6.5), facecolor='#0B0E14')
    
    # Plot 1: 3D Physical Slice of the 8D Manifold
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_facecolor('#0B0E14')
    p3d = ax1.scatter(base_3d[:, 0], base_3d[:, 1], base_3d[:, 2], c=manifold_color, 
                      cmap='turbo', s=14, alpha=0.85, edgecolors='none')
    ax1.view_init(elev=12, azim=-72)
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
    
    ax2.set_xlabel("Unfolded Intrinsic Coordinate 1 (t / Spiral Arc)", color='#E6EDF3', fontsize=10)
    ax2.set_ylabel("Unfolded Intrinsic Coordinate 2 (h / Height)", color='#E6EDF3', fontsize=10)
    ax2.set_title(f"Isometric 2D Unfolded Manifold (Hash-KNN Geodesic Embedding)\nPearson r(t) = {r_t:.3f}, r(h) = {r_h:.3f} (O(N) Graph)", 
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
