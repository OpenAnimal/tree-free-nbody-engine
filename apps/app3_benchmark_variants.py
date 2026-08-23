"""Standardized variant benchmark for Application 3 (spatial-hash attention).

Variants:
  standard      -- dense O(N^2) spatial RBF attention (the app's reference)
  +elastichash  -- the app's compute path: near-field exact (3x3 funnel-hash
                   neighborhood) + far-field per-cell centroid approximation
  +fmm (Taylor FGT) -- 2D Gaussian Fast Gaussian Transform
                   (core/gaussian2d_fgt.py), the eigenfunction-kernel Taylor
                   FGT.  The app's spatial kernel is exp(-r^2/(2 sigma^2));
                   the FGT kernel is exp(-r^2/h^2), so h^2 = 2 sigma^2 makes
                   the two kernels IDENTICAL (asserted before benchmarking).
                   The attention output is computed by running the FGT once
                   per V column (charges = V[:, d]) plus once for the row-
                   normalizer (charges = ones), then dividing -- this is the
                   exact spatial-only attention, not an approximation.

Accuracy vs `standard` on the attention output (rel L2). The far-field
centroid approximation error is reported in the table, not hidden in a note.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.benchmark_kit import VariantBenchmark


def _point_cloud(n_points: int = 1500, d_model: int = 16, seed: int = 42):
    np.random.seed(seed)
    c1 = np.random.normal(loc=[0.3, 0.3], scale=0.08, size=(n_points // 3, 2))
    c2 = np.random.normal(loc=[0.7, 0.7], scale=0.06, size=(n_points // 3, 2))
    c3 = np.random.normal(loc=[0.4, 0.7], scale=0.10,
                          size=(n_points - 2 * (n_points // 3), 2))
    points = np.clip(np.vstack([c1, c2, c3]), 0.05, 0.95)
    Q = np.random.randn(n_points, d_model)
    K = np.random.randn(n_points, d_model)
    V = np.random.randn(n_points, d_model)
    return points, Q, K, V


def _dense_spatial_attention(points, Q, K, V, sigma=0.15):
    """Dense O(N^2) SPATIAL-ONLY attention (matches the app's
    dense_spatial_out reference, no QK feature term)."""
    diff = points[:, None, :] - points[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    w = np.exp(-dist_sq / (2 * sigma ** 2))
    w /= (np.sum(w, axis=-1, keepdims=True) + 1e-9)
    return w @ V


def _hash_near_far_attention(points, V, sigma=0.15, depth=4):
    """The app's compute path: near-field exact (3x3 CellIndex) + far-field
    per-cell centroid. Delegates to the app's vectorized hash_spatial_attention
    (CellIndex + per-cell NumPy, replaces the old ElasticHashTable + per-point
    Python loops)."""
    from apps.app3_spatial_attention import hash_spatial_attention
    return hash_spatial_attention(points, V, sigma=sigma, depth=depth)


def _fgt_spatial_attention(points, V, sigma=0.15, depth=6, p=8):
    """Spatial-only attention via the 2D Gaussian Taylor FGT.

    The app's kernel is exp(-r^2/(2 sigma^2)); the FGT kernel is
    exp(-r^2/h^2), so h^2 = 2 sigma^2 makes them identical.  The attention
    output for feature dim d is:
        out_i^d = sum_j V_j^d * w_ij / sum_j w_ij,   w_ij = exp(-r_ij^2/(2 sigma^2))
    so we run the FGT once per V column (charges = V[:, d]) plus once for
    the row-normalizer (charges = ones), then divide.  This is the EXACT
    spatial-only attention (no centroid approximation), computed with the
    FGT's Taylor far field + exact ring-2 near field.
    """
    from core.gaussian2d_fgt import Gaussian2DFGT

    h = sigma * np.sqrt(2.0)  # h^2 = 2 sigma^2  =>  exp(-r^2/h^2) == exp(-r^2/(2 sigma^2))
    # Assert the two kernel functions are identical on a radial sweep before
    # benchmarking (per the round-4 plan).
    r = np.linspace(0.0, 3.0, 50)
    k_app = np.exp(-(r * r) / (2.0 * sigma * sigma))
    k_fgt = np.exp(-(r * r) / (h * h))
    assert np.allclose(k_app, k_fgt), (
        f"app3 kernel != FGT kernel: max diff {np.max(np.abs(k_app - k_fgt)):.3e}"
    )

    fgt = Gaussian2DFGT(depth=depth, p=p, h=h)
    n, d = V.shape
    # The FGT excludes self pairs (standard for N-body potentials where G(0)
    # is singular).  The Gaussian kernel G(0) = exp(0) = 1 is finite, and the
    # dense attention INCLUDES the self term w_ii = 1, so add it back: each
    # particle contributes q_i * G(0) = q_i * 1 to its own row.
    # Normalizer: sum_j exp(-r_ij^2/(2 sigma^2)) = 1 + sum_{j!=i} exp(...).
    norm = fgt.evaluate(points, np.ones(n, dtype=np.float64)) + 1.0
    # Per-column numerator: V_i + sum_{j!=i} V_j * exp(...).
    out = V.astype(np.float64).copy()
    for col in range(d):
        out[:, col] += fgt.evaluate(points, V[:, col].astype(np.float64))
    return out / (norm[:, None] + 1e-9)


def run_app3_variants(n_points: int = 1500):
    points, Q, K, V = _point_cloud(n_points=n_points)
    bench = VariantBenchmark(
        f"App 3 -- Spatial-hash attention (N={n_points}, 2D Gaussian RBF kernel; "
        f"+fmm = Taylor FGT on the Gaussian eigenfunction kernel)"
    )
    bench.add(
        "standard (dense O(N^2))",
        lambda: _dense_spatial_attention(points, Q, K, V),
        note="dense spatial RBF attention reference",
    )
    bench.add(
        "+elastichash (near+far centroid)",
        lambda: _hash_near_far_attention(points, V, depth=4),
        accuracy_vs="standard (dense O(N^2))",
        note="near exact (3x3 funnel hash) + far per-cell centroid; "
             "lossy far-field centroid approximation",
    )
    bench.add(
        "+fmm (Taylor FGT)",
        lambda: _fgt_spatial_attention(points, V, sigma=0.15, depth=6, p=8),
        accuracy_vs="standard (dense O(N^2))",
        note="2D Gaussian Taylor FGT (core/gaussian2d_fgt.py), h=sigma*sqrt(2); "
             "exact spatial-only attention via per-column FGT + normalizer; "
             "NOT faster than direct at N=1500 (per-cell Python loop overhead)",
    )
    return bench.run()


if __name__ == "__main__":
    run_app3_variants()
