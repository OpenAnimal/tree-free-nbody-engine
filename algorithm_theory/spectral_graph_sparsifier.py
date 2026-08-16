"""
Spielman-Srivastava Spectral Graph Sparsifier (spectral_graph_sparsifier.py).

Inspired by:
1. "Graph Sparsification by Effective Resistances"
   Daniel A. Spielman and Nikhil Srivastava (SIAM J. Comput. / STOC 2008, 2011).
2. "Spectral Graph Theory"
   Daniel A. Spielman (Yale University Lecture Notes / Combinatorial Scientific Computing, 2019).
3. "Optimal Bounds for Open Addressing Without Reordering"
   Martin Farach-Colton, Andrew Krapivin, William Kuszmaul (FOCS 2024 / arXiv:2501.02305).

Key Algorithmic Principle:
Given a massive, dense graph G = (V, E) with |E| = O(|V|^2) edges, solving linear systems,
random walks, and cuts is computationally expensive.
The Spielman-Srivastava Sparsification Theorem proves that any graph G can be compressed into an
ultra-sparse subgraph H = (V, E_H) containing only |E_H| = O(|V| * log |V| / eps^2) edges such that:
    (1 - eps) * x^T * L_G * x <= x^T * L_H * x <= (1 + eps) * x^T * L_G * x,   forall x in R^{|V|}

The sampling probability for each edge e = (u, v) is strictly proportional to its statistical leverage score:
    p_e = (w_e * R_eff(u, v)) / (|V| - 1)
Using our randomized Johnson-Lindenstrauss effective resistance solver (`network_power_centrality.py`),
all leverage scores are estimated in O(k * |E|) time, producing a sparsifier H with provable spectral bounds.
"""

import time
from typing import Tuple, List, Optional, Dict
import numpy as np

try:
    from .network_power_centrality import NetworkPowerCentrality
except ImportError:
    from network_power_centrality import NetworkPowerCentrality


class SpectralGraphSparsifier:
    """
    Spielman-Srivastava Effective Resistance Spectral Sparsifier.
    
    Compresses dense graphs into sparse subgraphs with bounded Laplacian quadratic forms.
    """
    def __init__(
        self,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
        target_sparsity_factor: float = 4.0,
        eps_spectral: float = 0.2
    ):
        self.n = int(num_nodes)
        self.edges = edges
        self.sparsity_factor = float(target_sparsity_factor)
        self.eps = float(eps_spectral)

    def sparsify(self) -> Tuple[List[Tuple[int, int, float]], Dict[str, float]]:
        """
        Samples edges proportional to effective resistance leverage scores.
        
        Returns:
            sparse_edges: List of (u, v, reweighted_weight) edges in sparsified graph H
            stats: Summary metrics (original edge count, sparse edge count, compression ratio)
        """
        n_edges = len(self.edges)
        if n_edges <= self.n * self.sparsity_factor:
            return self.edges, {"compression_ratio": 1.0, "original_edges": n_edges, "sparse_edges": n_edges}

        # 1. Compute Effective Resistances for all edges via JL Embedding
        analyzer = NetworkPowerCentrality(num_nodes=self.n, edges=self.edges, num_projections=24)
        
        edge_pairs = np.array([[e[0], e[1]] for e in self.edges], dtype=np.int64)
        weights = np.array([e[2] for e in self.edges], dtype=np.float64)

        r_eff_values = analyzer.batch_query_effective_resistances(edge_pairs)

        # Leverage scores: l_e = w_e * R_eff(e)
        # Note: Sum of leverage scores over all edges equals |V| - 1 (Spanning tree theorem)
        leverage_scores = weights * np.maximum(r_eff_values, 1e-6)
        total_leverage = np.sum(leverage_scores)

        # Sampling probabilities: p_e = l_e / sum(l)
        sampling_probs = leverage_scores / max(total_leverage, 1e-12)

        # Target number of samples: q = C * |V| * log(|V|) / eps^2
        target_samples = int(min(n_edges, max(self.n * 2, int(self.n * np.log(self.n) * self.sparsity_factor / (self.eps ** 2)))))
        
        # Draw samples with replacement
        sampled_indices = np.random.choice(n_edges, size=target_samples, p=sampling_probs)
        
        # Count frequencies for re-weighting
        counts = np.bincount(sampled_indices, minlength=n_edges)
        active_mask = counts > 0
        active_indices = np.where(active_mask)[0]

        # New weights: w_new = (count / target_samples) * (w_e / p_e)
        new_weights = (counts[active_indices] / float(target_samples)) * (weights[active_indices] / sampling_probs[active_indices])

        sparse_edges = []
        for i_loc, orig_i in enumerate(active_indices):
            u, v, _ = self.edges[orig_i]
            sparse_edges.append((u, v, float(new_weights[i_loc])))

        stats = {
            "original_edges": float(n_edges),
            "sparse_edges": float(len(sparse_edges)),
            "compression_ratio": float(n_edges / max(len(sparse_edges), 1)),
            "target_samples": float(target_samples)
        }
        return sparse_edges, stats


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("Spielman-Srivastava Spectral Graph Sparsifier Benchmark")
    print("=" * 70)

    n_nodes = 3000
    print(f"Original Graph Nodes (|V|)   : {n_nodes:,}")

    # Generate dense random geometric graph
    coords = np.random.rand(n_nodes, 2) * 10.0
    edges_list = []
    
    cell_sz = 1.0
    grid = {}
    for idx, (x, y) in enumerate(coords):
        k = (int(x / cell_sz), int(y / cell_sz))
        if k not in grid:
            grid[k] = []
        grid[k].append(idx)

    for (gx, gy), indices in grid.items():
        for i in indices:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nbr_k = (gx + dx, gy + dy)
                    if nbr_k in grid:
                        for j in grid[nbr_k]:
                            if i < j:
                                dist = np.linalg.norm(coords[i] - coords[j])
                                if dist < 1.0:
                                    w = 1.0 / max(dist, 0.1)
                                    edges_list.append((i, j, w))

    print(f"Original Dense Edges (|E|)   : {len(edges_list):,}")

    sparsifier = SpectralGraphSparsifier(
        num_nodes=n_nodes,
        edges=edges_list,
        target_sparsity_factor=0.8,
        eps_spectral=0.25
    )

    t0 = time.perf_counter()
    sparse_edges, stats = sparsifier.sparsify()
    t_sparsify = (time.perf_counter() - t0) * 1000.0

    print(f"Sparsification Runtime       : {t_sparsify:.2f} ms")
    print(f"Sparsified Graph Edges (|E_H|): {int(stats['sparse_edges']):,}")
    print(f"Edge Compression Factor      : {stats['compression_ratio']:.1f}x (Keeps only {100/stats['compression_ratio']:.1f}% edges)")
    print("=" * 70)
