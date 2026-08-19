"""Standardized variant benchmark for Application 10 (continuous spatial GNN).

Variants:
  standard      -- dense all-pairs spatial message pass (the app's reference
                   at small N: Gaussian kernel over every pair, no adjacency
                   matrix)
  +elastichash  -- the app's compute path: near-field exact Gaussian messages
                   within the 3x3 funnel-hash neighborhood + far-field
                   per-cell centroid message aggregation

The +fmm axis is OMITTED with reason: the message kernel is a Gaussian,
NOT the 2D logarithmic CGR88 kernel, so the core FMM engines do not apply
(per the kit's documented policy).

Accuracy vs `standard` on the per-node output features (rel L2). The
far-field centroid approximation error is reported in the table, not hidden.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.benchmark_kit import VariantBenchmark


def _graph(n_nodes: int = 150, in_dim: int = 32, out_dim: int = 16, seed: int = 42):
    """Smaller N than the app's 2000 so the dense all-pairs reference is
    affordable -- matches the app's own N_small=150 reference comparison."""
    np.random.seed(seed)
    c1 = np.random.normal(loc=[0.3, 0.3], scale=0.08, size=(n_nodes // 3, 2))
    c2 = np.random.normal(loc=[0.7, 0.7], scale=0.06, size=(n_nodes // 3, 2))
    c3 = np.random.normal(loc=[0.4, 0.7], scale=0.10,
                          size=(n_nodes - 2 * (n_nodes // 3), 2))
    coords = np.clip(np.vstack([c1, c2, c3]), 0.05, 0.95)
    feats = np.random.randn(n_nodes, in_dim)
    return coords, feats, in_dim, out_dim


def _dense_message_pass(coords, feats, layer, n_nodes):
    """Dense all-pairs Gaussian message pass (the app's `dense` reference)."""
    W_near, W_far, bias = layer.W_near, layer.W_far, layer.bias
    h_self = feats @ layer.W_self
    out = np.zeros((n_nodes, layer.out_features))
    for i in range(n_nodes):
        d = np.linalg.norm(coords[i] - coords, axis=1) + 1e-4
        w = np.exp(-d ** 2 / 0.05); w[i] = 0.0
        near_msg = (w[:, None] * feats).sum(axis=0) / max(w.sum(), 1e-9)
        w_far = np.exp(-d ** 2 / 0.2); w_far[i] = 0.0
        far_msg = (w_far[:, None] * feats).sum(axis=0) / max(w_far.sum(), 1e-9)
        out[i] = np.maximum(0.0, h_self[i] + near_msg @ W_near + far_msg @ W_far + bias)
    return out


def run_app10_variants(n_nodes: int = 150):
    from apps.app10_continuous_gnn_fmm import ContinuousSpatialGNNLayer
    coords, feats, in_dim, out_dim = _graph(n_nodes=n_nodes, in_dim=32, out_dim=16)
    layer = ContinuousSpatialGNNLayer(in_features=32, out_features=16, depth=4)

    bench = VariantBenchmark(
        f"App 10 -- Continuous spatial GNN message pass (N={n_nodes}, 2D Gaussian kernel; "
        f"+fmm axis omitted -- not a 2D log kernel)"
    )
    bench.add(
        "standard (dense all-pairs)",
        lambda: _dense_message_pass(coords, feats, layer, n_nodes),
        note="dense Gaussian message pass reference (no adjacency matrix)",
    )
    bench.add(
        "+elastichash (near+far centroid)",
        lambda: layer.forward(coords, feats),
        accuracy_vs="standard (dense all-pairs)",
        note="near exact (3x3 funnel hash) + far per-cell centroid; "
             "lossy far-field centroid approximation",
    )
    return bench.run()


if __name__ == "__main__":
    run_app10_variants()
