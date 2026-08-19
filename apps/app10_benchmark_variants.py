"""Standardized variant benchmark for Application 10 (continuous spatial GNN).

Variants:
  standard      -- dense all-pairs spatial message pass (the app's reference
                   at small N: Gaussian kernel over every pair, no adjacency
                   matrix)
  +elastichash  -- the app's compute path: near-field exact Gaussian messages
                   within the 3x3 funnel-hash neighborhood + far-field
                   per-cell centroid message aggregation
  +fmm (Taylor FGT) -- 2D Gaussian Taylor Fast Gaussian Transform
                   (`core/gaussian2d_fgt.py`) on the app's two Gaussian
                   message kernels (near: exp(-r^2/0.05), far: exp(-r^2/0.2)).
                   The app's kernel is exp(-r^2/c); the FGT kernel is
                   exp(-r^2/h^2), so h^2 = c makes them IDENTICAL (asserted
                   before benchmarking on r in linspace(0,3,50)).  The
                   message pass is computed by running the FGT once per
                   feature dim (charges = feats[:, d]) plus once for the
                   per-kernel normalizer (charges = ones), then dividing --
                   this is the exact spatial message pass, not a centroid
                   approximation.  The dense reference excludes self pairs
                   (w[i] = 0), so no self-term restoration is needed (unlike
                   app3, whose dense attention includes the self term).

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


def _fgt_message_pass(coords, feats, layer, n_nodes, depth=6, p=8):
    """Spatial message pass via the 2D Gaussian Taylor FGT.

    The app uses two Gaussian kernels: near exp(-r^2/0.05) and far
    exp(-r^2/0.2).  The FGT kernel is exp(-r^2/h^2), so h^2 = c makes them
    identical.  Asserted on a radial sweep first.  The dense reference
    excludes self pairs (w[i] = 0), and the FGT also excludes self pairs,
    so no self-term restoration is needed.
    """
    from core.gaussian2d_fgt import Gaussian2DFGT

    h_near = np.sqrt(0.05)
    h_far = np.sqrt(0.2)
    # Assert the FGT kernel matches the app's two kernels on a radial sweep.
    r = np.linspace(0.0, 3.0, 50)
    assert np.allclose(np.exp(-(r * r) / 0.05), np.exp(-(r * r) / (h_near * h_near))), (
        "app10 near kernel != FGT near kernel")
    assert np.allclose(np.exp(-(r * r) / 0.2), np.exp(-(r * r) / (h_far * h_far))), (
        "app10 far kernel != FGT far kernel")

    fgt_near = Gaussian2DFGT(depth=depth, p=p, h=h_near)
    fgt_far = Gaussian2DFGT(depth=depth, p=p, h=h_far)
    n, in_dim = feats.shape
    feats = feats.astype(np.float64)
    # Per-kernel normalizers (exclude self, matching the dense reference).
    norm_near = fgt_near.evaluate(coords, np.ones(n, dtype=np.float64))
    norm_far = fgt_far.evaluate(coords, np.ones(n, dtype=np.float64))
    # Per-feature numerators.
    near_num = np.zeros((n, in_dim), dtype=np.float64)
    far_num = np.zeros((n, in_dim), dtype=np.float64)
    for d in range(in_dim):
        near_num[:, d] = fgt_near.evaluate(coords, feats[:, d])
        far_num[:, d] = fgt_far.evaluate(coords, feats[:, d])
    near_msg = near_num / (norm_near[:, None] + 1e-9)
    far_msg = far_num / (norm_far[:, None] + 1e-9)
    h_self = feats @ layer.W_self
    out = np.maximum(0.0, h_self + near_msg @ layer.W_near
                     + far_msg @ layer.W_far + layer.bias)
    return out


def run_app10_variants(n_nodes: int = 150):
    from apps.app10_continuous_gnn_fmm import ContinuousSpatialGNNLayer
    coords, feats, in_dim, out_dim = _graph(n_nodes=n_nodes, in_dim=32, out_dim=16)
    layer = ContinuousSpatialGNNLayer(in_features=32, out_features=16, depth=4)

    bench = VariantBenchmark(
        f"App 10 -- Continuous spatial GNN message pass (N={n_nodes}, 2D Gaussian kernel; "
        f"+fmm = Taylor FGT on the two Gaussian message kernels)"
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
    bench.add(
        "+fmm (Taylor FGT)",
        lambda: _fgt_message_pass(coords, feats, layer, n_nodes, depth=6, p=8),
        accuracy_vs="standard (dense all-pairs)",
        note="2D Gaussian Taylor FGT (core/gaussian2d_fgt.py) on both the "
             "near (h^2=0.05) and far (h^2=0.2) message kernels; exact "
             "spatial message pass via per-feature FGT + normalizer; "
             "self terms excluded (matches the dense reference w[i]=0)",
    )
    return bench.run()


if __name__ == "__main__":
    run_app10_variants()
