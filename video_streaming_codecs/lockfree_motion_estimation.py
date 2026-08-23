"""
Hash-Accelerated Block Motion Estimation for Video Codecs
(`lockfree_motion_estimation.py`)

A single-threaded Python motion estimator that uses a quantized block fingerprint
dict to propose candidate motion vectors from a reference frame, then validates
every proposal with a Sum-of-Absolute-Differences (SAD) test against the local
+-4 diamond search and the zero motion vector. The best SAD wins.

NOTE on naming/history: an earlier revision labelled this "Lock-Free" with
"atomicCAS" inserts. That was theatre — this is plain single-threaded Python with
a normal dict, no concurrency, no atomics, and no compare-and-swap. The class is
now named `HashAcceleratedMotionEstimation` (alias `LockFreeMotionEstimator`
retained for backwards compatibility). The Farach-Colton, Krapivin, & Kuszmaul (2025)
non-reordering open-addressing hash table that was previously constructed but
never read has been removed; the fingerprint dict is the actual index and is
cleared per reference frame.
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Optional


class HashAcceleratedMotionEstimation:
    """
    Single-threaded hash-accelerated block motion estimator.

    A 56-bit quantized block fingerprint (DC + 4 quadrant means + 2 gradient
    magnitudes) indexes reference-frame macroblocks in a plain dict. Candidate
    MVs from an exact fingerprint match are accepted ONLY if their SAD beats both
    the zero-MV SAD and the local +-4 diamond search SAD; otherwise the diamond
    search result is used. This prevents the uniform-region failure mode where
    many blocks share a fingerprint and would otherwise all inherit the first
    inserted block's MV.
    """
    def __init__(self, block_size: int = 16, width: int = 1920, height: int = 1080):
        self.block_size = block_size
        self.width = width
        self.height = height
        self.blocks_x = width // block_size
        self.blocks_y = height // block_size
        self.num_blocks = self.blocks_x * self.blocks_y

        # Plain dict: fingerprint -> list of (bx, by) reference block coords.
        # Cleared and rebuilt on every register_reference_frame call so no stale
        # fingerprints accumulate across reference frames.
        self.ref_block_map: Dict[int, List[Tuple[int, int]]] = {}
        self.ref_frame: Optional[np.ndarray] = None

    def _compute_block_fingerprint(self, block: np.ndarray) -> int:
        """
        Computes 56-bit quantized spatial fingerprint from a luma block:
        DC mean + 4 sub-quadrant means + horizontal/vertical gradient magnitudes.
        """
        # DC component (mean luminance)
        dc = int(np.mean(block)) & 0xFF

        # 2x2 Sub-quadrant means (spatial gradient)
        h_mid, w_mid = self.block_size // 2, self.block_size // 2
        q0 = int(np.mean(block[:h_mid, :w_mid])) & 0xFF
        q1 = int(np.mean(block[:h_mid, w_mid:])) & 0xFF
        q2 = int(np.mean(block[h_mid:, :w_mid])) & 0xFF
        q3 = int(np.mean(block[h_mid:, w_mid:])) & 0xFF

        # Horizontal & Vertical Edge Gradients
        dx = int(np.mean(np.abs(np.diff(block.astype(np.int32), axis=1)))) & 0xFF
        dy = int(np.mean(np.abs(np.diff(block.astype(np.int32), axis=0)))) & 0xFF

        fingerprint = (dc << 48) | (q0 << 40) | (q1 << 32) | (q2 << 24) | (q3 << 16) | (dx << 8) | dy
        return int(fingerprint)

    def _block_sad(self, cur_block: np.ndarray, ref_block: np.ndarray) -> float:
        """Sum of Absolute Differences between two equally-shaped blocks."""
        return float(np.sum(np.abs(cur_block.astype(np.int32) - ref_block.astype(np.int32))))

    def register_reference_frame(self, ref_frame: np.ndarray):
        """
        Indexes every reference macroblock by its fingerprint. The index is
        rebuilt from scratch on each call (no stale fingerprints survive across
        reference frames).
        """
        self.ref_frame = ref_frame
        self.ref_block_map.clear()

        for by in range(self.blocks_y):
            for bx in range(self.blocks_x):
                y = by * self.block_size
                x = bx * self.block_size
                block = ref_frame[y:y+self.block_size, x:x+self.block_size]

                fp = self._compute_block_fingerprint(block)
                self.ref_block_map.setdefault(fp, []).append((bx, by))

    def _diamond_search_sad(self, cur_block: np.ndarray, y: int, x: int) -> Tuple[float, List[int]]:
        """Local +-4 diamond search. Returns (best_sad, best_mv=[dx,dy])."""
        best_sad = float(self._block_sad(
            cur_block,
            self.ref_frame[y:y+self.block_size, x:x+self.block_size]
        ))
        best_mv = [0, 0]
        for dy in (-4, 0, 4):
            for dx in (-4, 0, 4):
                if dx == 0 and dy == 0:
                    continue
                ry = int(np.clip(y + dy, 0, self.height - self.block_size))
                rx = int(np.clip(x + dx, 0, self.width - self.block_size))
                ref_block = self.ref_frame[ry:ry+self.block_size, rx:rx+self.block_size]
                sad = float(np.sum(np.abs(cur_block.astype(np.int32) - ref_block.astype(np.int32))))
                if sad < best_sad:
                    best_sad = sad
                    best_mv = [dx, dy]
        return best_sad, best_mv

    # Cap on hash-proposed candidates evaluated per block. Flat content
    # (uniform frames, sky, letterbox, fades) makes ALL blocks share one
    # fingerprint; without a cap the loop below degenerates to O(num_blocks^2)
    # SAD evaluations (~275 s/frame extrapolated at 1080p). 32 candidates is
    # far beyond the number of distinct plausible matches for real content.
    MAX_HASH_CANDIDATES = 32

    def estimate_motion(self, cur_frame: np.ndarray, search_range: int = 32) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """`search_range` is accepted for API compatibility but unused: the
        search extent is set by the hash candidate list (capped at
        MAX_HASH_CANDIDATES) plus the fixed +-4 diamond ring."""
        """
        Estimates Motion Vectors (MVs) for the current frame against the reference.

        For each block:
          1. Look up its fingerprint in the reference index.
          2. For up to MAX_HASH_CANDIDATES matching reference blocks, compute
             the candidate MV's SAD. A perfect match (SAD == 0) short-circuits
             the loop.
          3. Compute the zero-MV SAD and the +-4 diamond search SAD.
          4. Accept the hash candidate MV only if its SAD strictly beats BOTH the
             zero-MV SAD and the diamond search SAD; otherwise keep the diamond
             search result. This guarantees uniform regions (where many blocks
             share a fingerprint) fall back to the local search instead of
             inheriting a garbage global MV.
        """
        t0 = time.perf_counter()
        motion_vectors = np.zeros((self.blocks_y, self.blocks_x, 2), dtype=np.int32)
        sad_costs = np.zeros((self.blocks_y, self.blocks_x), dtype=np.float32)

        hash_hits = 0
        total_blocks = self.blocks_y * self.blocks_x

        for by in range(self.blocks_y):
            for bx in range(self.blocks_x):
                y = by * self.block_size
                x = bx * self.block_size
                cur_block = cur_frame[y:y+self.block_size, x:x+self.block_size]

                # Local diamond search (includes the zero-MV SAD as its baseline).
                best_sad, best_mv = self._diamond_search_sad(cur_block, y, x)
                from_hash = False

                # Hash-proposed global candidates, validated by SAD.
                fp = self._compute_block_fingerprint(cur_block)
                matches = self.ref_block_map.get(fp)
                if matches:
                    evaluated = 0
                    for (ref_bx, ref_by) in matches:
                        if evaluated >= self.MAX_HASH_CANDIDATES:
                            break
                        mv_x = (ref_bx - bx) * self.block_size
                        mv_y = (ref_by - by) * self.block_size
                        # Skip candidates that the diamond search already covers.
                        if mv_x == 0 and mv_y == 0:
                            continue
                        ref_y = ref_by * self.block_size
                        ref_x = ref_bx * self.block_size
                        ref_block = self.ref_frame[ref_y:ref_y+self.block_size, ref_x:ref_x+self.block_size]
                        sad = float(np.sum(np.abs(cur_block.astype(np.int32) - ref_block.astype(np.int32))))
                        evaluated += 1
                        if sad < best_sad:
                            best_sad = sad
                            best_mv = [mv_x, mv_y]
                            from_hash = True
                            if sad == 0.0:
                                break  # perfect match; cannot improve
                    if from_hash:
                        hash_hits += 1

                motion_vectors[by, bx] = best_mv
                sad_costs[by, bx] = best_sad

        t_elapsed = (time.perf_counter() - t0) * 1000.0

        stats = {
            "elapsed_ms": t_elapsed,
            "hash_hit_rate": (hash_hits / total_blocks) * 100.0,
            "total_blocks": total_blocks,
            "throughput_fps": 1000.0 / max(1e-3, t_elapsed),
            "mean_sad": float(np.mean(sad_costs))
        }
        return motion_vectors, sad_costs, stats


# Backwards-compatible alias (the public API / tests import LockFreeMotionEstimator).
LockFreeMotionEstimator = HashAcceleratedMotionEstimation
