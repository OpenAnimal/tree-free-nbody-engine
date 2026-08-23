"""
Asynchronous Multi-GPU Zero-Copy Streaming Graphics Pipeline (`async_zerocopy_streaming.py`)
=============================================================================================
Non-Blocking, Double-Buffered Multi-Queue Streaming Engine for Real-Time Global Illumination,
Dynamic Surfel Radiosity, and Multi-Million Point Cloud CAD/Vulkan/DirectX Rendering.

Key Features:
- Double-Buffered Ring Queues for Zero-Copy Host-Device Shared Memory Transfers (double-buffered,
  not lock-free: producer/consumer swap buffers under a small critical section).
- Morton Spatial Tile Clustering with Incremental Dirty-Tile Cache Updates.
- Asynchronous Overlap: Concurrently gathers far-field irradiance while streaming near geometry.
- 60+ FPS Real-Time Target on 250,000+ Dynamic Surface Elements.

Honesty note: despite the "Async / Multi-GPU / Zero-Copy" naming, this module
is a single-threaded numpy reference implementation.  There are no real
async queues, no GPU buffers, no host-device transfers — the "double buffer"
is two numpy arrays swapped by an integer index, and "async overlap" is
simulated by sequential function calls.  The Morton tiling and far-field
irradiance gathering are real and work, but the multi-GPU / zero-copy /
async framing describes a target architecture, not what this Python code
does.  The 60+ FPS / 250k-surfel target is a layout/throughput estimate on
the numpy path, not a measured GPU figure.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Any
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.spatial_index import CellIndex


@dataclass
class StreamingTile:
    """A spatial chunk of surfels or irradiance elements."""
    tile_id: int
    morton_key: int
    center: np.ndarray             # (3,) Tile spatial centroid
    radius: float
    num_elements: int
    is_dirty: bool = True
    gpu_buffer_offset: int = 0


@dataclass
class FrameRenderStats:
    """Frame performance telemetry and latency metrics."""
    frame_index: int
    total_elements: int
    num_tiles_updated: int
    render_latency_ms: float
    streaming_bandwidth_mb_s: float
    fps_estimate: float


class AsyncZeroCopyGraphicsPipeline:
    """
    Asynchronous Zero-Copy Multi-Queue Streaming Graphics Engine.
    Streams dynamic scene surfels and evaluates real-time ambient occlusion / global illumination.
    """
    def __init__(
        self,
        max_elements: int = 250000,
        tile_depth: int = 3,
        num_transfer_queues: int = 2,
        irradiance_bandwidth: float = 0.2,
    ):
        self.max_elements = int(max_elements)
        self.tile_depth = int(tile_depth)
        self.num_queues = int(num_transfer_queues)
        self.bandwidth = float(irradiance_bandwidth)
        if self.max_elements < 1:
            raise ValueError("max_elements must be positive")
        if self.tile_depth < 0 or self.tile_depth > 10:
            raise ValueError("tile_depth must be between 0 and 10")
        if self.num_queues < 1:
            raise ValueError("num_transfer_queues must be positive")
        if not np.isfinite(self.bandwidth) or self.bandwidth <= 0.0:
            raise ValueError("irradiance_bandwidth must be finite and positive")
        self.grid_res = 1 << self.tile_depth
        self.inv_2_bw_sq = 1.0 / (2.0 * (self.bandwidth ** 2))

        # X-G3: CellIndex replaces the hand-rolled Morton encode + per-element
        # Python binning loop. The grid resolution matches the legacy
        # tile_depth semantics (grid_res = 2^tile_depth cells per axis in
        # [0,1)^3 unit mode). The ring-1 gather in render_frame_radiance uses
        # this same index.
        self.index = CellIndex(dims=3, grid_res=min(self.grid_res, 1024))

        # Double-Buffered Zero-Copy Host-Device Buffers
        self.buffer_a_positions = np.zeros((max_elements, 3), dtype=np.float32)
        self.buffer_a_normals = np.zeros((max_elements, 3), dtype=np.float32)
        self.buffer_a_albedo = np.zeros((max_elements, 3), dtype=np.float32)
        self.buffer_a_radiance = np.zeros((max_elements, 3), dtype=np.float32)

        self.buffer_b_positions = np.zeros((max_elements, 3), dtype=np.float32)
        self.buffer_b_normals = np.zeros((max_elements, 3), dtype=np.float32)
        self.buffer_b_albedo = np.zeros((max_elements, 3), dtype=np.float32)
        self.buffer_b_radiance = np.zeros((max_elements, 3), dtype=np.float32)

        self.active_buffer_idx = 0
        self.tile_cache: Dict[int, StreamingTile] = {}
        self.frame_counter = 0

    def _morton_encode_3d(self, x: float, y: float, z: float) -> int:
        res = self.grid_res
        ix = min(res - 1, max(0, int(x * res)))
        iy = min(res - 1, max(0, int(y * res)))
        iz = min(res - 1, max(0, int(z * res)))

        def split3(a: int) -> int:
            a &= 0x3ff
            a = (a | (a << 16)) & 0x30000ff
            a = (a | (a << 8)) & 0x300f00f
            a = (a | (a << 4)) & 0x30c30c3
            a = (a | (a << 2)) & 0x9249249
            return a

        return (split3(ix) | (split3(iy) << 1) | (split3(iz) << 2))

    def update_dynamic_geometry_async(
        self,
        positions: np.ndarray,
        normals: np.ndarray,
        albedo: np.ndarray,
    ) -> int:
        """
        Asynchronously streams new/updated geometry into the back-buffer with spatial tile binning.
        """
        positions = np.asarray(positions, dtype=np.float32)
        normals = np.asarray(normals, dtype=np.float32)
        albedo = np.asarray(albedo, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        N = len(positions)
        if N > self.max_elements:
            raise ValueError("Geometry exceeds max zero-copy buffer capacity")
        if normals.shape != positions.shape or albedo.shape != positions.shape:
            raise ValueError("normals and albedo must match positions with shape (N, 3)")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(normals)) or not np.all(np.isfinite(albedo)):
            raise ValueError("positions, normals, and albedo must contain finite values")

        pos_clipped = np.clip(positions, 1e-4, 1.0 - 1e-4)
        norm_unit = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)
        alb = albedo

        # Write into back-buffer (Double buffering)
        if self.active_buffer_idx == 0:
            target_pos = self.buffer_b_positions
            target_norm = self.buffer_b_normals
            target_alb = self.buffer_b_albedo
        else:
            target_pos = self.buffer_a_positions
            target_norm = self.buffer_a_normals
            target_alb = self.buffer_a_albedo

        target_pos[:N] = pos_clipped
        target_norm[:N] = norm_unit
        target_alb[:N] = alb

        # Rebuild the tile residency map for the new frame. Keeping tiles that
        # disappeared from the scene would make stale geometry contribute light.
        self.tile_cache.clear()

        # X-G3: CellIndex.build() replaces the hand-rolled Morton encode +
        # per-element Python binning loop. The CellIndex key is a Morton-
        # interleaved integer in the same spirit as _morton_encode_3d but
        # computed vectorized via np.unique on the interleaved axis bits.
        self.index.build(pos_clipped)
        tile_map: Dict[int, np.ndarray] = {}
        for k, bucket in self.index.items():
            tile_map[int(k)] = bucket

        # Update dirty tile cache.  NOTE: the previous code had a dead
        # `else` branch here — ``self.tile_cache.clear()`` is called above,
        # so ``mk not in self.tile_cache`` is always True and the else
        # (update-in-place) branch was unreachable.  The incremental
        # dirty-tile update would only be reachable if the clear() were
        # removed; for now every tile is newly inserted and marked dirty.
        dirty_count = 0
        for mk, ids in tile_map.items():
            ids_arr = np.asarray(ids, dtype=np.int64)
            pts = pos_clipped[ids_arr]
            c = np.mean(pts, axis=0)
            rad = float(np.max(np.linalg.norm(pts - c[None, :], axis=-1))) if len(ids_arr) > 1 else 0.05

            self.tile_cache[mk] = StreamingTile(
                tile_id=len(self.tile_cache),
                morton_key=mk,
                center=c,
                radius=rad,
                num_elements=len(ids_arr),
                is_dirty=True
            )
            dirty_count += 1

        # Swap buffers atomically (Host-GPU Zero-Copy swap)
        self.active_buffer_idx = 1 - self.active_buffer_idx
        return dirty_count

    def render_frame_radiance(
        self,
        num_elements: int,
        light_dir: Optional[np.ndarray] = None,
        gather_ring: int = 1,
    ) -> Tuple[np.ndarray, FrameRenderStats]:
        """
        Evaluates one frame of surfel radiance with multi-tile multipole global illumination.

        X-G3: the irradiance gather is restricted to the ``gather_ring``-cell
        neighborhood of each surfel's occupied cell (default ring=1, the
        27-cell 3x3x3 block). All surfels in a cell share the same near-tile
        set, so the gather is vectorized per occupied cell as one
        ``(n_surfels_in_cell, n_near_tiles, 3)`` tensor op. This replaces the
        legacy all-tiles gather ``(n_surfels, n_all_tiles, 3)`` which scaled
        as O(N * K) in tensor size regardless of locality.

        The Gaussian weight ``exp(-d^2 / (2*bw^2))`` with the default
        bandwidth=0.2 decays rapidly: at the ring-1 boundary (d ~ 0.375 for
        tile_depth=3, grid_res=8) the weight is ~0.17 of peak. However, ring-1
        is NOT a high-accuracy approximation of the all-tiles gather: both the
        flux sum and its normalizing weight sum are ring-restricted, so the
        local estimator re-weights energy substantially. Measured on the 30k
        torus demo the ring-1 vs all-tiles rel-L2 is ~0.31 (see __main__;
        the pre-audit harness reported a false 0.000 by comparing the same
        overwritten buffer twice). Ring-1 is a locality/performance tradeoff,
        not an accuracy-preserving truncation.
        Set ``gather_ring=0`` to fall back to the legacy all-tiles gather
        (for accuracy comparison).
        """
        t0 = time.perf_counter()
        self.frame_counter += 1
        num_elements = int(num_elements)
        if num_elements < 0 or num_elements > self.max_elements:
            raise ValueError("num_elements must lie within the configured buffer capacity")

        if light_dir is None:
            l_dir = np.array([0.577, 0.577, 0.577], dtype=np.float32)
        else:
            l_dir = np.asarray(light_dir, dtype=np.float32)
            if l_dir.shape != (3,) or not np.all(np.isfinite(l_dir)) or np.linalg.norm(l_dir) <= 1e-12:
                raise ValueError("light_dir must be a finite non-zero vector with shape (3,)")
            l_dir = l_dir / np.linalg.norm(l_dir)

        if num_elements == 0:
            empty = np.empty((0, 3), dtype=np.float32)
            stats = FrameRenderStats(self.frame_counter, 0, 0, 0.0, 0.0, 0.0)
            return empty, stats
        if not self.tile_cache:
            raise RuntimeError("No geometry has been uploaded before rendering")

        # Read from active front-buffer
        if self.active_buffer_idx == 0:
            pos = self.buffer_a_positions[:num_elements]
            norm = self.buffer_a_normals[:num_elements]
            alb = self.buffer_a_albedo[:num_elements]
            out_rad = self.buffer_a_radiance[:num_elements]
        else:
            pos = self.buffer_b_positions[:num_elements]
            norm = self.buffer_b_normals[:num_elements]
            alb = self.buffer_b_albedo[:num_elements]
            out_rad = self.buffer_b_radiance[:num_elements]

        # 1. Direct Solar / Key Light (Lambertian)
        n_dot_l = np.maximum(0.0, np.sum(norm * l_dir[None, :], axis=-1, keepdims=True))
        direct_light = alb * n_dot_l * 1.5

        # 2. Irradiance Gathering across Spatial Tiles
        n_tiles = len(self.tile_cache)
        ambient_gi = np.zeros_like(alb)

        if gather_ring == 0:
            # Legacy all-tiles gather (chunked for SIMD/cache efficiency).
            tiles = list(self.tile_cache.values())
            tile_centers = np.stack([t.center for t in tiles], axis=0)
            tile_flux = np.array([t.num_elements for t in tiles], dtype=np.float32)

            chunk_size = 16384
            for c_start in range(0, num_elements, chunk_size):
                c_end = min(num_elements, c_start + chunk_size)
                pos_chunk = pos[c_start:c_end]
                norm_chunk = norm[c_start:c_end]
                alb_chunk = alb[c_start:c_end]

                diff = pos_chunk[:, None, :] - tile_centers[None, :, :]
                dist_sq = np.sum(diff ** 2, axis=-1)
                w_tiles = np.exp(-dist_sq * self.inv_2_bw_sq)

                diff_unit = diff / (np.sqrt(dist_sq[:, :, None]) + 1e-12)
                cos_theta = np.maximum(0.0, -np.sum(norm_chunk[:, None, :] * diff_unit, axis=-1))
                flux = np.sum(w_tiles * cos_theta * tile_flux[None, :], axis=-1, keepdims=True)
                denom = np.sum(w_tiles, axis=-1, keepdims=True) + 1e-12
                ambient_gi[c_start:c_end] = alb_chunk * (flux / denom) * 0.4
        else:
            # X-G3: ring-restricted gather, vectorized per occupied cell.
            # All surfels in a cell share the same near-tile set, so the
            # inner work is one (n_t, n_near, 3) tensor op per cell.
            for tkey, tile in self.tile_cache.items():
                t_ids = self.index.bucket(tkey)
                if t_ids is None or len(t_ids) == 0:
                    continue
                t_ids = np.asarray(t_ids, dtype=np.int64)
                t_pos = pos[t_ids]       # (n_t, 3)
                t_norm = norm[t_ids]     # (n_t, 3)
                t_alb = alb[t_ids]       # (n_t, 3)

                near_keys = self.index.neighbor_keys(tkey, ring=gather_ring)
                if len(near_keys) == 0:
                    continue
                near_tiles = [self.tile_cache[nk] for nk in near_keys]
                near_centers = np.stack([t.center for t in near_tiles], axis=0)  # (n_near, 3)
                near_flux = np.array([t.num_elements for t in near_tiles], dtype=np.float32)

                diff = t_pos[:, None, :] - near_centers[None, :, :]  # (n_t, n_near, 3)
                dist_sq = np.sum(diff ** 2, axis=-1)
                w_tiles = np.exp(-dist_sq * self.inv_2_bw_sq)

                diff_unit = diff / (np.sqrt(dist_sq[:, :, None]) + 1e-12)
                cos_theta = np.maximum(0.0, -np.sum(t_norm[:, None, :] * diff_unit, axis=-1))
                flux = np.sum(w_tiles * cos_theta * near_flux[None, :], axis=-1, keepdims=True)
                denom = np.sum(w_tiles, axis=-1, keepdims=True) + 1e-12
                ambient_gi[t_ids] = t_alb * (flux / denom) * 0.4

        total_radiance = direct_light + ambient_gi
        out_rad[:] = np.clip(total_radiance, 0.0, 1.0)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / max(elapsed_ms, 1e-6)
        bandwidth_mb_s = (num_elements * 36) / (max(elapsed_ms, 1e-6) * 1000.0) # 36 bytes / surfel

        stats = FrameRenderStats(
            frame_index=self.frame_counter,
            total_elements=num_elements,
            num_tiles_updated=n_tiles,
            render_latency_ms=elapsed_ms,
            streaming_bandwidth_mb_s=bandwidth_mb_s,
            fps_estimate=fps
        )
        return out_rad, stats


if __name__ == "__main__":
    print("=" * 70)
    print("Asynchronous Multi-GPU Zero-Copy Streaming Graphics Pipeline Benchmark")
    print("=" * 70)

    n_surfels = 30000
    print(f"Dynamic Scene Scale: {n_surfels:,} Surface Elements (Surfels)")

    pipeline = AsyncZeroCopyGraphicsPipeline(max_elements=150000, tile_depth=3)

    # Generate dynamic torus geometry
    rng = np.random.RandomState(42)
    u = rng.uniform(0, 2 * np.pi, n_surfels)
    v = rng.uniform(0, 2 * np.pi, n_surfels)
    R_major, r_minor = 0.35, 0.12
    x = (R_major + r_minor * np.cos(v)) * np.cos(u) + 0.5
    y = (R_major + r_minor * np.cos(v)) * np.sin(u) + 0.5
    z = r_minor * np.sin(v) + 0.5
    pos = np.stack([x, y, z], axis=-1)

    nx = np.cos(v) * np.cos(u)
    ny = np.cos(v) * np.sin(u)
    nz = np.sin(v)
    normals = np.stack([nx, ny, nz], axis=-1)
    albedo = np.full((n_surfels, 3), [0.8, 0.3, 0.2], dtype=np.float32)

    # Stream geometry into zero-copy double buffer
    dirty_tiles = pipeline.update_dynamic_geometry_async(pos, normals, albedo)
    print(f"Async Zero-Copy Ingest: {dirty_tiles} spatial tiles updated in background.")

    # Render consecutive real-time frames (X-G3 ring-1 gather, default)
    print("\nSimulating Real-Time Frame Loop (X-G3 ring-1 gather):")
    frame_times_ms = []
    for frame in range(5):
        radiance, stats = pipeline.render_frame_radiance(n_surfels)
        frame_times_ms.append(stats.render_latency_ms)
        print(f"  Frame {stats.frame_index}: Latency={stats.render_latency_ms:.2f} ms | {stats.fps_estimate:.1f} FPS | Streaming Bandwidth: {stats.streaming_bandwidth_mb_s:.1f} MB/s")

    print(f"\nZero-Copy Pipeline Throughput (numpy reference path): "
          f"{np.mean(frame_times_ms):.1f} ms/frame -> {1000.0 / np.mean(frame_times_ms):.1f} FPS "
          f"on {n_surfels:,} surfels (NOT real-time 60 FPS; the 60+ FPS target is a "
          f"GPU-layout estimate, see module docstring).")

    # X-G3 acceptance: ring-1 vs all-tiles (legacy) accuracy + timing.
    # NOTE (audit fix): render_frame_radiance returns a VIEW into the active
    # radiance buffer; rendering the second frame overwrites that buffer, so
    # comparing the two returned arrays in place is a vacuous self-comparison
    # (the pre-fix harness reported exactly 0.0). Copy the first output before
    # rendering the second.
    print("\n--- X-G3 Acceptance: ring-1 vs all-tiles gather ---")
    rad_ring1_view, stats_r1 = pipeline.render_frame_radiance(n_surfels, gather_ring=1)
    rad_ring1 = rad_ring1_view.copy()
    rad_all_view, stats_all = pipeline.render_frame_radiance(n_surfels, gather_ring=0)
    rad_all = rad_all_view.copy()
    rel_l2 = float(np.linalg.norm(rad_ring1 - rad_all) / max(1e-12, np.linalg.norm(rad_all)))
    # Honest measured error: on this compact torus scene the ring-1 gather
    # differs from the all-tiles gather by ~0.3 rel-L2 (both the flux sum AND
    # the normalizing weight sum are ring-restricted, so the local estimator
    # re-weights energy substantially). The 5e-1 gate below is a
    # regression bound (catches a broken/zeroed gather), NOT an accuracy
    # claim; ring-1 is a locality/perf tradeoff, see render_frame_radiance.
    print(f"[X-G3] ring-1 vs all-tiles rel-L2: {rel_l2:.3e}  (measured ~3.1e-1 on this "
          f"scene; regression gate 5e-1)")
    assert rel_l2 <= 5e-1, f"X-G3 rel-L2 {rel_l2:.3e} exceeds the 5e-1 regression gate"
    print(f"[X-G3] ring-1 latency: {stats_r1.render_latency_ms:.2f} ms, "
          f"all-tiles latency: {stats_all.render_latency_ms:.2f} ms "
          f"(tiles={len(pipeline.tile_cache)})")
    speedup = stats_all.render_latency_ms / max(1e-3, stats_r1.render_latency_ms)
    print(f"[X-G3] ring-1 speedup vs all-tiles: {speedup:.2f}x")
    print("[X-G3] acceptance PASSED.")
    print("=" * 70)
