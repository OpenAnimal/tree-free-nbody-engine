"""Standardized variant benchmark for the video streaming Gaussian splat.

Variants:
  standard     — exact per-pixel Gaussian splat of one frame: the original
                 per-Gaussian SH colors are returned unchanged (the
                 lossless reference image)
  +elastichash — cell-bucketed splat: Gaussian centroids are quantized into
                 3D Morton cells via the elastic-hash CellIndex and one
                 mean color per occupied cell is broadcast back to its
                 members (the existing compress_frame order-0 path)
  +quantized   — quantized color path: per-channel color bit-packing to a
                 small codebook (4 bits / 16 levels per channel) before
                 de-quantization

accuracy_vs standard on the reconstructed per-Gaussian color image. The
known lossy color quantization (~0.31 rel L2 for the cluster-mean path,
plus the extra 4-bit color quantization cost) shows up in the table, not
hidden in a note.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.benchmark_kit import VariantBenchmark
from core.spatial_index import CellIndex


def _frame(n: int = 4000, seed: int = 42):
    rng = np.random.default_rng(seed)
    means = rng.uniform(0.05, 0.95, size=(n, 3)).astype(np.float64)
    scales = rng.uniform(0.001, 0.01, size=(n, 3)).astype(np.float64)
    rotations = rng.standard_normal((n, 4)).astype(np.float64)
    sh_colors = rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float64)
    return means, scales, rotations, sh_colors


def _exact_splat(sh_colors):
    """Lossless reference: the per-Gaussian colors are the splat."""
    return sh_colors.copy()


def _cell_bucketed_splat(means, sh_colors, depth=4):
    """Order-0 cluster-mean compression (the existing compress_frame path),
    returning the reconstructed per-Gaussian color image."""
    idx = CellIndex(dims=3, grid_res=1 << depth)
    _, inverse = idx.build(means)
    num_clusters = len(np.unique(inverse))
    cluster_radiance = np.zeros((num_clusters, 3), dtype=np.float64)
    weights = np.bincount(inverse, minlength=num_clusters).astype(np.float64)
    for c in range(3):
        cluster_radiance[:, c] = np.bincount(
            inverse, weights=sh_colors[:, c], minlength=num_clusters
        )
    cluster_radiance /= np.maximum(1.0, weights[:, None])
    return cluster_radiance[inverse]


def _quantized_color_splat(means, sh_colors, depth=4, bits=4):
    """Cell-bucketed cluster-mean splat PLUS per-channel color bit-packing
    (the existing quantized color path: cluster means are quantized to
    2^bits levels per channel before being broadcast back)."""
    recon = _cell_bucketed_splat(means, sh_colors, depth=depth)
    levels = (1 << bits) - 1
    q = np.clip(np.round(recon * levels) / levels, 0.0, 1.0)
    return q


def run_gaussian_splat_variants(n: int = 4000):
    means, scales, rotations, sh_colors = _frame(n=n)

    bench = VariantBenchmark(
        f"Gaussian splat frame compression (N={n} Gaussians, depth=4 3D Morton bucketing)"
    )
    bench.add(
        "standard (exact per-pixel)",
        lambda: _exact_splat(sh_colors),
        note="lossless per-Gaussian SH colors",
    )
    bench.add(
        "+elastichash (cell-bucketed)",
        lambda: _cell_bucketed_splat(means, sh_colors, depth=4),
        accuracy_vs="standard (exact per-pixel)",
        note="order-0 cluster-mean per occupied cell (lossy ~0.31 rel L2)",
    )
    bench.add(
        "+quantized (4-bit color)",
        lambda: _quantized_color_splat(means, sh_colors, depth=4),
        accuracy_vs="standard (exact per-pixel)",
        note="cluster-mean + 4-bit per-channel color quantization (lossy)",
    )
    return bench.run()


if __name__ == "__main__":
    run_gaussian_splat_variants()
