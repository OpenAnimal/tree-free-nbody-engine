"""
Application 7: High-Dimensional Continuous Graph & Memory Partitioning.
Powered by Random Hyperplane LSH + Farach-Colton / Krapivin / Kuszmaul Non-Reordering Hash.

Clusters 128-dimensional dense continuous embedding vectors (e.g. LLM memory / Vector DB retrieval)
into non-reordered, lock-free hash buckets without building expensive Hierarchical Navigable Small World (HNSW) graphs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import time
from core.elastic_hash import ElasticHashTable

def run_high_dim_graph_demo(n_embeddings: int = 5000, d_dim: int = 64, n_hyperplanes: int = 12):
    print(f">>> Running Application 7: High-Dim Graph & Vector Memory (N={n_embeddings}, D={d_dim})")
    np.random.seed(42)
    
    # Generate 5 semantic clusters of dense vectors with noise
    centers = np.random.randn(5, d_dim)
    cluster_labels = np.random.choice(5, size=n_embeddings)
    embeddings = centers[cluster_labels] + np.random.normal(0, 0.4, size=(n_embeddings, d_dim))
    # Normalize embeddings to unit sphere
    embeddings /= (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    
    # 1. Random Hyperplane Locality-Sensitive Hashing (LSH)
    # Generates a bitmask integer key for high-dimensional cosine angle partitioning
    hyperplanes = np.random.randn(d_dim, n_hyperplanes)
    projections = np.matmul(embeddings, hyperplanes)  # (N, n_hyperplanes)
    bitmasks = (projections > 0).astype(np.int64)
    
    # Convert bits to integer LSH key
    powers_of_two = 1 << np.arange(n_hyperplanes, dtype=np.int64)
    lsh_keys = np.sum(bitmasks * powers_of_two[None, :], axis=1)
    
    # 2. Populate Farach-Colton Non-Reordering Elastic Hash Table
    t0 = time.perf_counter()
    capacity = 1 << (n_hyperplanes + 1)
    hash_table = ElasticHashTable(capacity=capacity, delta=0.05)
    
    bucket_map = {}
    for i in range(n_embeddings):
        k = int(lsh_keys[i])
        if k not in bucket_map:
            bucket_map[k] = []
        bucket_map[k].append(i)
        
    for k, ids in bucket_map.items():
        hash_table.insert(k, ids)
        
    t_insert = time.perf_counter() - t0
    print(f"[-] High-Dim Embedding Partition Time: {t_insert*1000:.2f} ms ({n_embeddings / (t_insert + 1e-6):.0f} vec/s)")
    print(f"[-] Total Active Semantic Buckets: {len(bucket_map)} | Load Factor: {hash_table.count / hash_table.capacity * 100:.1f}%")
    
    # 3. Fast Vector Similarity Retrieval Query
    query_vec = centers[0] + np.random.normal(0, 0.2, size=d_dim)
    query_vec /= np.linalg.norm(query_vec)
    
    t0 = time.perf_counter()
    q_proj = np.matmul(query_vec, hyperplanes) > 0
    q_key = int(np.sum(q_proj * powers_of_two))
    
    candidate_ids, probe_steps = hash_table.lookup(q_key)
    t_query = time.perf_counter() - t0
    
    if candidate_ids is not None:
        cand_vecs = embeddings[candidate_ids]
        sims = np.matmul(cand_vecs, query_vec)
        top_k = np.argsort(sims)[::-1][:5]
        print(f"[-] Vector Query Time: {t_query*1000:.3f} ms | Probe Steps: {probe_steps} | Retreived {len(candidate_ids)} candidates")
    
    # 4. 2D PCA Dimensionality Reduction for Visualization
    # Compute 2D SVD/PCA for clean plotting
    U, S, Vt = np.linalg.svd(embeddings - np.mean(embeddings, axis=0), full_matrices=False)
    pts_2d = np.matmul(embeddings, Vt[:2].T)
    q_2d = np.matmul(query_vec, Vt[:2].T)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0B0E14')
    
    # Plot 1: Semantic Ground Truth Cluster Layout
    ax1.set_facecolor('#0B0E14')
    scatter = ax1.scatter(pts_2d[:, 0], pts_2d[:, 1], c=cluster_labels, cmap='Spectral', s=16, alpha=0.7)
    ax1.scatter(q_2d[0], q_2d[1], c='#00FFCC', marker='*', s=150, edgecolors='white', label='Query Vector')
    ax1.set_title("Ground-Truth Semantic Embedding Clusters (2D PCA)", color='white', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', facecolor='#161B22', edgecolor='#30363D', labelcolor='white')
    
    # Plot 2: High-Dimensional LSH + Non-Reordering Bucket Retrieval
    ax2.set_facecolor('#0B0E14')
    is_retrieved = np.zeros(n_embeddings, dtype=bool)
    if candidate_ids is not None:
        is_retrieved[candidate_ids] = True
        
    ax2.scatter(pts_2d[~is_retrieved, 0], pts_2d[~is_retrieved, 1], c='#21262D', s=12, alpha=0.4, label='Unvisited Embeddings')
    ax2.scatter(pts_2d[is_retrieved, 0], pts_2d[is_retrieved, 1], c='#00F0FF', s=35, alpha=0.9, label='Instant O(1) Retrieved Candidates')
    ax2.scatter(q_2d[0], q_2d[1], c='#FF0055', marker='*', s=160, edgecolors='white', label='Query Vector')
    
    ax2.set_title(f"Non-Reordering Hash Partitioning (LSH Depth = {n_hyperplanes})", color='white', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', facecolor='#161B22', edgecolor='#30363D', labelcolor='white')
    
    for ax in (ax1, ax2):
        ax.tick_params(colors='#8B949E')
        for spine in ax.spines.values():
            spine.set_color('#30363D')
            
    fig.suptitle("Application 7: High-Dimensional Vector DB / Memory Graph Clustering\nAccelerated by Farach-Colton / Kuszmaul Elastic Open Addressing", 
                 color='white', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "app7_highdim_embeddings.png")
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[-] Saved high-dim graph visualization to: {output_path}")

if __name__ == '__main__':
    run_high_dim_graph_demo(5000, 64, 12)
