"""
4D Dynamic Gaussian Splatting Frame Compressor.
Tree-free 3D Morton quantization indexed by the non-reordering elastic hash.

What it actually does: buckets Gaussian centroids into Morton cells and stores
one mean color per occupied cell (lossy order-0 compression). The elastic hash
is the authoritative occupied-cell index. What it does NOT do: there is no
spherical-harmonics radiance aggregation, no "multipole M2L merging", no
rendering or decoding pipeline, and per-frame work is O(N) bucketing — the
"90 FPS VR decoding" claim of earlier revisions described nothing real. The
lossy color error is measured and reported by validate_color_compression().
"""

import numpy as np
import time
from typing import Tuple, List, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.spatial_index import CellIndex

class GaussianSplat4DStreamer:
    """
    Volumetric 4D Gaussian Splatting Video Streamer & Radiance Field Compressor.
    """
    def __init__(self, depth: int = 6, max_gaussians: int = 100000):
        self.depth = depth
        self.grid_res = 1 << depth
        self.max_gaussians = max_gaussians
        self.index = CellIndex(dims=3, grid_res=self.grid_res)
        self.cluster_map = {}

    def compress_frame(self, means: np.ndarray, scales: np.ndarray, rotations: np.ndarray, sh_colors: np.ndarray) -> Dict:
        """
        Compresses a frame of N 3D Gaussians (lossy, order-0):
        - Quantizes centroids into 3D Morton integer keys (one bucket per occupied cell).
        - Stores the per-bucket mean color (an order-0 / average moment, NOT a
          "local multipole radiance moment" — see the module header for what this
          module does and does not do).
        - `scales` and `rotations` are accepted for API completeness but are NOT
          used by this lossy color-averaging compressor.
        """
        t0 = time.perf_counter()
        N = len(means)

        # 1. Build the authoritative occupied-cell index (3D Morton, unit mode).
        unique_keys, inverse = self.index.build(means)
        num_clusters = len(unique_keys)

        # 2. Per-cluster mean color (order-0 moment)
        cluster_weights = np.bincount(inverse, minlength=num_clusters).astype(np.float32)
        cluster_radiance = np.zeros((num_clusters, 3), dtype=np.float32)
        for c in range(3):
            cluster_radiance[:, c] = np.bincount(inverse, weights=sh_colors[:, c], minlength=num_clusters)
        cluster_radiance /= np.maximum(1.0, cluster_weights[:, None])

        # 3. Lossy reconstruction: expand cluster means back to N Gaussians and
        #    measure the actual compression error honestly.
        recon = cluster_radiance[inverse]
        color_rel_l2_err = float(np.linalg.norm(recon - sh_colors) /
                                 max(1e-9, np.linalg.norm(sh_colors)))

        t_compress = (time.perf_counter() - t0) * 1000.0

        return {
            "color_rel_l2_err": color_rel_l2_err,
            "num_gaussians": N,
            "num_clusters": num_clusters,
            "compression_ratio": N / max(1, num_clusters),
            "latency_ms": t_compress,
            "throughput_gps": N / max(1e-6, t_compress / 1000.0),
            "fps_capacity": 1000.0 / max(1e-3, t_compress)
        }

def run_gaussian_splat_demo():
    print("==================================================================")
    print(" VIDEO STREAMING: 4D GAUSSIAN SPLATTING VOLUMETRIC STREAMING")
    print("==================================================================")
    N_GAUSSIANS = 200000
    print(f"Streaming frame with {N_GAUSSIANS:,} dynamic 3D Gaussians...")
    
    np.random.seed(42)
    means = np.random.uniform(0.05, 0.95, size=(N_GAUSSIANS, 3)).astype(np.float32)
    scales = np.random.uniform(0.001, 0.01, size=(N_GAUSSIANS, 3)).astype(np.float32)
    rotations = np.random.randn(N_GAUSSIANS, 4).astype(np.float32)
    sh_colors = np.random.uniform(0, 1, size=(N_GAUSSIANS, 3)).astype(np.float32)
    
    streamer = GaussianSplat4DStreamer(depth=6)
    stats = streamer.compress_frame(means, scales, rotations, sh_colors)
    
    print(f"[-] Splat Frame Ingest Time:  {stats['latency_ms']:.2f} ms")
    print(f"[-] Splat Throughput:         {stats['throughput_gps']/1e6:.2f} Million Gaussians/sec")
    print(f"[-] Frame Ingest Capacity:     {stats['fps_capacity']:.1f} FPS (bucketing only — no decoder implemented)")
    print(f"[-] Cluster Pruning Ratio:     {stats['compression_ratio']:.2f}x")
    print(f"[-] Color Compression Error:   rel L2 = {stats['color_rel_l2_err']:.3f} "
          f"(lossy order-0 cluster-mean quantization)")

if __name__ == '__main__':
    run_gaussian_splat_demo()
