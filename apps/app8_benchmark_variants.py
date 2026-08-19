"""Standardized variant benchmark for Application 8 (manifold unfolding via hash k-NN).

Variants:
  standard      -- exact O(N^2) k-NN graph (all pairwise distances, true
                   top-k neighbors per point -- the reference for the app's
                   graph-construction step)
  +elastichash  -- the app's compute path: multi-table random-hyperplane LSH
                   + Farach-Colton funnel hash -> bucketed candidate sets ->
                   top-k within candidates (O(N) inserts + bucket-local work)

The +fmm axis is OMITTED with reason: the task is high-dimensional k-NN
graph construction, not a 2D/3D kernel sum.

Accuracy semantics: LSH k-NN is approximate. The correctness metric is
`k-NN edge recall` (fraction of true top-k neighbor edges that appear in
the hash k-NN graph), reported in the note -- not a rel-L2.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.benchmark_kit import VariantBenchmark


def _swiss_roll(n_samples: int = 2500, ambient_dim: int = 8, seed: int = 42):
    np.random.seed(seed)
    t = 1.5 * np.pi * (1 + 2 * np.random.uniform(0, 1, n_samples))
    h = np.random.uniform(0, 20, n_samples)
    base_3d = np.stack([t * np.cos(t), h, t * np.sin(t)], axis=1)
    proj = np.random.randn(ambient_dim, 3)
    Q, _ = np.linalg.qr(proj)
    pts = base_3d @ Q.T + np.random.normal(0, 0.05, size=(n_samples, ambient_dim))
    return pts


def _brute_knn_edges(points, k=12):
    """Exact O(N^2) k-NN edge set (sorted (i,j) pairs with i<j)."""
    n = len(points)
    edges = set()
    for i in range(n):
        d = np.linalg.norm(points - points[i], axis=1)
        d[i] = np.inf
        top = np.argsort(d)[:k]
        for j in top:
            edges.add((min(i, int(j)), max(i, int(j))))
    return edges


def run_app8_variants(n_samples: int = 2500, k=12):
    from apps.app8_dimension_reduction_knn import build_hash_knn_graph
    points = _swiss_roll(n_samples=n_samples)

    # One-off recall check (LSH k-NN is approximate).
    exact_edges = _brute_knn_edges(points, k=k)
    graph = build_hash_knn_graph(points, k_neighbors=k, n_tables=8, n_hyperplanes=8)
    approx_edges = set(e for e in graph["edges"].keys())
    recall = len(exact_edges & approx_edges) / max(1, len(exact_edges))
    note = (f"k-NN edge recall@{k} = {recall*100:.1f}% "
            f"({len(exact_edges & approx_edges)}/{len(exact_edges)} true edges); "
            f"+fmm axis omitted (high-dim k-NN, not a kernel sum)")

    bench = VariantBenchmark(
        f"App 8 -- Hash-accelerated k-NN graph (N={n_samples}, k={k}, 8D Swiss roll; "
        f"+fmm axis omitted)"
    )
    bench.add(
        "standard (exact O(N^2) k-NN)",
        lambda: np.array(sorted(_brute_knn_edges(points, k=k)), dtype=np.int64),
        note=f"exact top-{k} neighbor edge set",
    )
    bench.add(
        "+elastichash (LSH k-NN graph)",
        lambda: np.array(sorted(graph["edges"].keys()), dtype=np.int64),
        note=note,
    )
    return bench.run()


if __name__ == "__main__":
    run_app8_variants()
