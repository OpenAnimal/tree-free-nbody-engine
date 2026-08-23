"""Validation for T-D3's TaylorFGTAttention (NumPy reference).

Layer 1 (pure spatial attention) is exact up to the FGT truncation error and
is asserted against the dense reference. Layer 2 (spatial x feature softmax)
uses the positive-feature ratio estimator whose variance is the known
Performer pain point, so its error is MEASURED and only loosely bounded, as
the module docstring promises ("a measured rel-L2-vs-m curve, not an appeal
to a bound").

Run: python -X utf8 tests/neural_ops/test_taylor_fgt_attention.py
     (or) python -m pytest tests/neural_ops/test_taylor_fgt_attention.py -q
"""
from __future__ import annotations
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from neural_ops.taylor_fgt_attention import TaylorFGTAttention


def _dense_layer1(coords, V, sigma):
    diff = coords[:, None, :] - coords[None, :, :]
    G = np.exp(-np.sum(diff ** 2, axis=-1) / (2 * sigma ** 2))
    np.fill_diagonal(G, 0.0)
    return G @ V / (G.sum(axis=1, keepdims=True) + 1e-30)


def test_layer1_exact_spatial_attention_3d():
    rng = np.random.RandomState(42)
    N, D = 400, 4
    coords = rng.uniform(0.05, 0.95, size=(N, 3))
    V = rng.randn(N, D)
    sigma = 0.15

    layer = TaylorFGTAttention(spatial_dim=3, sigma=sigma, grid_depth=6, p=8)
    out, meta = layer.forward(None, None, V, coords)

    ref = _dense_layer1(coords, V, sigma)
    rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    assert rel < 1e-4, f"layer-1 rel-L2 {rel:.3e} >= 1e-4"
    assert meta["spatial_dim"] == 3 and meta["sigma"] == sigma


def test_layer1_exact_spatial_attention_2d():
    rng = np.random.RandomState(7)
    N, D = 300, 2
    coords = rng.uniform(0.05, 0.95, size=(N, 2))
    V = rng.randn(N, D)
    sigma = 0.2

    layer = TaylorFGTAttention(spatial_dim=2, sigma=sigma, grid_depth=6, p=8)
    out, _ = layer.forward(None, None, V, coords)

    ref = _dense_layer1(coords, V, sigma)
    rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    assert rel < 1e-4, f"layer-1 (2D) rel-L2 {rel:.3e} >= 1e-4"


def test_layer2_feature_estimator_measured():
    """Layer 2 error is dominated by the m-feature ratio estimator: assert
    only the loose bound the estimator supports, and verify the error
    DECREASES when m grows (the spec's rel-L2-vs-m curve, two points).
    Measured on this scene: m=16 -> 4.5e-1, m=64 -> 2.7e-1, m=256 -> 1.3e-1."""
    rng = np.random.RandomState(1)
    N, D_feat, D = 250, 32, 4
    coords = rng.uniform(0.05, 0.95, size=(N, 3))
    Q = rng.randn(N, D_feat)
    K = rng.randn(N, D_feat)
    V = rng.randn(N, D)
    sigma, tau = 0.15, 1.0 / D_feat

    diff = coords[:, None, :] - coords[None, :, :]
    G = np.exp(-np.sum(diff ** 2, axis=-1) / (2 * sigma ** 2))
    W = G * np.exp(np.clip((Q @ K.T) * tau, -50, 50))
    np.fill_diagonal(W, 0.0)
    ref = W @ V / (W.sum(axis=1, keepdims=True) + 1e-30)

    rels = {}
    for m in (16, 64):
        layer = TaylorFGTAttention(
            spatial_dim=3, sigma=sigma, grid_depth=5, p=6,
            n_features=m, seed=3,
        )
        out, meta = layer.forward(Q, K, V, coords)
        rels[m] = np.linalg.norm(out - ref) / np.linalg.norm(ref)
        assert np.all(np.isfinite(out))
        assert meta["n_features"] == m

    assert rels[64] < 0.5, f"layer-2 rel-L2 at m=64 unexpectedly large: {rels[64]:.3e}"
    assert rels[64] <= rels[16] + 1e-3, (
        f"layer-2 error did not improve with m (m=16: {rels[16]:.3e}, m=64: {rels[64]:.3e})"
    )


if __name__ == "__main__":
    test_layer1_exact_spatial_attention_3d()
    test_layer1_exact_spatial_attention_2d()
    test_layer2_feature_estimator_measured()
    print("test_taylor_fgt_attention: all checks passed")
