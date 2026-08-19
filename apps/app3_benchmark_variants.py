"""Standardized variant benchmark for Application 3 (spatial-hash attention).

Variants:
  standard      -- dense O(N^2) spatial RBF attention (the app's reference)
  +elastichash  -- the app's compute path: near-field exact (3x3 funnel-hash
                   neighborhood) + far-field per-cell centroid approximation

The +fmm axis is OMITTED with reason: the spatial kernel here is a Gaussian
RBF, NOT the 2D logarithmic CGR88 kernel, so the core FMM engines do not
apply (per the kit's documented policy).

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
    """The app's compute path: near-field exact (3x3 funnel-hash) + far-field
    per-cell centroid. Reuses the app's ElasticHashTable + morton helpers."""
    from core.elastic_hash import ElasticHashTable
    from core.tree_free_fmm import morton_encode_2d, decode_morton_2d, get_box_center_2d
    n, d = V.shape
    grid_res = 1 << depth
    ht = ElasticHashTable(capacity=grid_res * grid_res * 2, delta=0.05)
    for i in range(n):
        key = morton_encode_2d(points[i, 0], points[i, 1], depth=depth)
        pidx, _ = ht.lookup(key)
        if pidx is None:
            ht.insert(key, [i])
        else:
            pidx.append(i)
    cluster_centers, cluster_v = {}, {}
    for key in [k for k, _ in ht.items()]:
        pidx = ht.lookup(key)[0]
        _, ix, iy = decode_morton_2d(key)
        cx, cy = get_box_center_2d(depth, ix, iy)
        cluster_centers[key] = np.array([cx, cy])
        cluster_v[key] = np.sum(V[pidx], axis=0)
    out = np.zeros_like(V)
    for i in range(n):
        m_key = morton_encode_2d(points[i, 0], points[i, 1], depth=depth)
        _, ix, iy = decode_morton_2d(m_key)
        acc_v = np.zeros(d)
        acc_w = 1e-9
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = ix + dx, iy + dy
                if 0 <= nx < grid_res and 0 <= ny < grid_res:
                    n_key = (depth << 24) | morton_encode_2d(
                        (nx + 0.5) / grid_res, (ny + 0.5) / grid_res, depth=depth) & 0xFFFFFF
                    pidx, _ = ht.lookup(n_key)
                    if pidx is not None:
                        d2 = np.sum((points[i] - points[pidx]) ** 2, axis=-1)
                        w = np.exp(-d2 / (2 * sigma ** 2))
                        acc_v += np.sum(w[:, None] * V[pidx], axis=0)
                        acc_w += np.sum(w)
        for f_key, c_center in cluster_centers.items():
            _, fx, fy = decode_morton_2d(f_key)
            if abs(fx - ix) > 1 or abs(fy - iy) > 1:
                d_c2 = np.sum((points[i] - c_center) ** 2)
                w_far = np.exp(-d_c2 / (2 * sigma ** 2))
                acc_v += w_far * cluster_v[f_key]
                acc_w += w_far * len(ht.lookup(f_key)[0])
        out[i] = acc_v / acc_w
    return out


def run_app3_variants(n_points: int = 1500):
    points, Q, K, V = _point_cloud(n_points=n_points)
    bench = VariantBenchmark(
        f"App 3 -- Spatial-hash attention (N={n_points}, 2D Gaussian RBF kernel; "
        f"+fmm axis omitted -- not a 2D log kernel)"
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
    return bench.run()


if __name__ == "__main__":
    run_app3_variants()
