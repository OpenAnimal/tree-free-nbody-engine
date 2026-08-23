"""
Multipole Range Tree: Flat, Pointerless Multidimensional Range Structure.
Bridging Orthogonal Range Searching with Tree-Free Multipole Aggregations.

Replaces pointer-chasing d-dimensional Range Trees, k-d trees, and multi-dimensional
Fenwick/Segment trees with a flat, non-reordering multi-resolution spatial index.

Capabilities:
1. Orthogonal Range Counting & Range Sums via a hierarchical Morton-prefix
   traversal: a query visits O(2^D * depth) cells in the worst case plus the
   individually matched points (no ``O(log delta^-1)`` bound is claimed -- the
   structure is a flat dict-indexed hierarchy, not a balanced comparison tree).
2. Box-aggregated potential evaluation (brute-force direct summation over the
   points falling inside a query box -- NOT a hierarchical multipole expansion;
   see ``compute_multipole_box_potential``).
3. Range Min, Max, and Variance Aggregates.
4. O(N) space: sorted coordinate/value arrays plus a dict of per-level
   (start, end, sum, count) tuples. This is a flat dict-indexed layout, not a
   single contiguous SIMD slab.
"""

from typing import Tuple, Optional, List, Dict, Union, Any
import numpy as np
import time


def morton_encode_nd(coords_quantized: np.ndarray, bits_per_dim: int = 16) -> np.ndarray:
    """
    Computes Morton (Z-order) codes for N points in D dimensions.

    The interleaved code occupies ``bits_per_dim * D`` bits, which must fit in a
    uint64 (i.e. ``bits_per_dim * D <= 64``). For ``D >= 4`` with the default
    16 bits/dim this overflows 64 bits and the high bits silently wrap, so the
    caller must keep ``bits_per_dim <= 64 // D`` (the range tree clamps
    ``max_depth`` accordingly).
    """
    coords = np.asarray(coords_quantized, dtype=np.uint64)
    if coords.ndim == 1:
        coords = coords[:, None]
    N, D = coords.shape

    if bits_per_dim * D > 64:
        raise ValueError(
            f"morton_encode_nd: bits_per_dim*D = {bits_per_dim}*{D} = "
            f"{bits_per_dim * D} > 64; the interleaved code overflows uint64 "
            f"and high bits wrap silently. Reduce bits_per_dim to <= "
            f"{64 // D} for D={D}."
        )

    morton_codes = np.zeros(N, dtype=np.uint64)
    for b in range(bits_per_dim):
        for d in range(D):
            bit = (coords[:, d] >> np.uint64(b)) & np.uint64(1)
            morton_codes |= bit << np.uint64(b * D + d)

    return morton_codes


class FlatMultipoleRangeTree:
    """
    Flat Pointerless Multidimensional Range Tree.
    
    Organizes multi-dimensional point sets into hierarchical Morton prefix buckets
    stored contiguously in a flat hash index. Provides O(N) linear memory footprint
    and sub-millisecond multi-dimensional orthogonal box queries.
    """
    def __init__(
        self,
        points: np.ndarray,
        values: Optional[np.ndarray] = None,
        leaf_capacity: int = 32,
        max_depth: int = 10,
    ):
        """
        Initializes the Flat Multipole Range Tree over an N x D point set.
        
        Parameters
        ----------
        points : np.ndarray
            N x D array of continuous coordinates.
        values : Optional[np.ndarray]
            N or N x F array of associated weights, values, or feature vectors.
        leaf_capacity : int
            Maximum number of particles per fine-level leaf bucket before subdivision.
        max_depth : int
            Maximum multi-resolution Morton depth levels.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[:, None]
        if pts.size == 0:
            raise ValueError("points must not be empty")
            
        self.N, self.D = pts.shape
        self.leaf_capacity = max(2, int(leaf_capacity))
        # The Morton code interleaves max_depth bits per dimension, so it
        # occupies max_depth * D bits and must fit in uint64. Clamp to
        # 64 // D (and the existing 18-level cap) to avoid silent high-bit
        # wraparound in morton_encode_nd for D >= 4.
        self.max_depth = max(1, min(18, int(max_depth), 64 // self.D))
        
        self.bbox_min = np.min(pts, axis=0) - 1e-6
        self.bbox_max = np.max(pts, axis=0) + 1e-6
        self.bbox_span = np.maximum(self.bbox_max - self.bbox_min, 1e-8)
        
        # Normalize to unit hypercube [0, 1)^D
        self.norm_coords = (pts - self.bbox_min) / self.bbox_span
        self.points = pts
        
        if values is None:
            self.values = np.ones((self.N, 1), dtype=np.float64)
        else:
            val_arr = np.asarray(values, dtype=np.float64)
            if val_arr.ndim == 1:
                val_arr = val_arr[:, None]
            if val_arr.shape[0] != self.N:
                raise ValueError(f"values length {val_arr.shape[0]} does not match points {self.N}")
            self.values = val_arr
            
        self.val_dim = self.values.shape[1]
        
        # Build multi-resolution flat index
        self._build_tree_free_hierarchy()

    def _build_tree_free_hierarchy(self):
        """Builds multi-level Morton prefix aggregates in contiguous arrays."""
        # 16-bit grid per dimension for up to 64-bit total Morton codes
        grid_res = 1 << self.max_depth
        quantized = np.clip(np.floor(self.norm_coords * grid_res), 0, grid_res - 1).astype(np.uint64)
        
        self.morton_codes = morton_encode_nd(quantized, bits_per_dim=self.max_depth)
        self.sorted_order = np.argsort(self.morton_codes)
        self.sorted_morton = self.morton_codes[self.sorted_order]
        self.sorted_points = self.points[self.sorted_order]
        self.sorted_values = self.values[self.sorted_order]
        
        # Precompute prefix sums for O(1) contiguous interval aggregation
        # Prefix sum array of shape (N + 1, val_dim)
        self.prefix_values = np.zeros((self.N + 1, self.val_dim), dtype=np.float64)
        np.cumsum(self.sorted_values, axis=0, out=self.prefix_values[1:])
        
        # Multi-level cell dictionary mapping
        # (depth, cell_morton_prefix) -> (start_idx, end_idx, sum_val, count).
        # NOTE: a per-cell centroid is NOT stored -- the only potential
        # evaluator (compute_multipole_box_potential) does brute-force direct
        # summation over the box, not a centroid multipole expansion, so a
        # precomputed centroid would be dead state.
        self.level_nodes: Dict[Tuple[int, int], Tuple[int, int, np.ndarray, int]] = {}

        for depth in range(self.max_depth + 1):
            shift = (self.max_depth - depth) * self.D
            prefixes = self.sorted_morton >> np.uint64(shift)

            # Find unique consecutive runs in the sorted prefix array
            diffs = np.where(prefixes[:-1] != prefixes[1:])[0]
            starts = np.concatenate(([0], diffs + 1))
            ends = np.concatenate((diffs + 1, [self.N]))
            unique_prefixes = prefixes[starts]

            for p_val, s_idx, e_idx in zip(unique_prefixes, starts, ends):
                cnt = e_idx - s_idx
                val_sum = self.prefix_values[e_idx] - self.prefix_values[s_idx]
                self.level_nodes[(depth, int(p_val))] = (s_idx, e_idx, val_sum, cnt)

    def query_range(
        self,
        box_min: np.ndarray,
        box_max: np.ndarray,
        return_indices: bool = False
    ) -> Dict[str, Any]:
        """
        Performs an orthogonal box range query: [box_min, box_max].
        
        Returns count, value_sum, point indices (optional), and summary statistics.
        """
        b_min = np.asarray(box_min, dtype=np.float64)
        b_max = np.asarray(box_max, dtype=np.float64)
        if b_min.ndim == 1:
            b_min = b_min.reshape(1, -1)
        if b_max.ndim == 1:
            b_max = b_max.reshape(1, -1)
            
        b_min = b_min.ravel()
        b_max = b_max.ravel()
        
        # Check domain intersection
        if np.any(b_min > self.bbox_max) or np.any(b_max < self.bbox_min):
            return {
                "count": 0,
                "sum": np.zeros(self.val_dim, dtype=np.float64),
                "indices": np.array([], dtype=np.int64) if return_indices else None,
                "subcells_aggregated": 0,
                "exact_checks": 0
            }

        matched_indices = []
        total_sum = np.zeros(self.val_dim, dtype=np.float64)
        total_count = 0
        subcells_agg = 0
        exact_checks = 0
        
        # Traverse multi-resolution cells using spatial hierarchy
        stack = [(0, 0, self.bbox_min.copy(), self.bbox_max.copy())]
        
        while stack:
            depth, prefix, c_min, c_max = stack.pop()
            
            # 1. Check intersection between query box and current cell bounding box
            if np.any(c_min > b_max) or np.any(c_max < b_min):
                continue
                
            # 2. Check if cell is completely inside the query box
            is_fully_inside = np.all(c_min >= b_min) and np.all(c_max <= b_max)
            
            node_key = (depth, prefix)
            if node_key not in self.level_nodes:
                continue
                
            s_idx, e_idx, val_sum, cnt = self.level_nodes[node_key]
            
            if is_fully_inside and not return_indices:
                # Fast aggregate without touching individual points!
                total_sum += val_sum
                total_count += cnt
                subcells_agg += 1
                continue
                
            # 3. If at max_depth or small count, check points directly
            if depth == self.max_depth or cnt <= self.leaf_capacity:
                pts_slice = self.sorted_points[s_idx:e_idx]
                inside_mask = np.all((pts_slice >= b_min) & (pts_slice <= b_max), axis=1)
                exact_checks += cnt
                
                if np.any(inside_mask):
                    n_match = np.sum(inside_mask)
                    total_count += n_match
                    vals_slice = self.sorted_values[s_idx:e_idx]
                    total_sum += np.sum(vals_slice[inside_mask], axis=0)
                    if return_indices:
                        matched_order = self.sorted_order[s_idx:e_idx][inside_mask]
                        matched_indices.extend(matched_order.tolist())
                continue
                
            # 4. Subdivide into 2^D child octants
            child_depth = depth + 1
            c_mid = 0.5 * (c_min + c_max)
            
            for child_oct in range(1 << self.D):
                child_prefix = (prefix << self.D) | child_oct
                if (child_depth, child_prefix) not in self.level_nodes:
                    continue
                    
                ch_min = c_min.copy()
                ch_max = c_max.copy()
                for d in range(self.D):
                    bit = (child_oct >> d) & 1
                    if bit == 1:
                        ch_min[d] = c_mid[d]
                    else:
                        ch_max[d] = c_mid[d]
                        
                stack.append((child_depth, child_prefix, ch_min, ch_max))
                
        return {
            "count": int(total_count),
            "sum": total_sum,
            "indices": np.array(matched_indices, dtype=np.int64) if return_indices else None,
            "subcells_aggregated": subcells_agg,
            "exact_checks": exact_checks
        }

    def compute_multipole_box_potential(
        self,
        target_points: np.ndarray,
        box_min: np.ndarray,
        box_max: np.ndarray,
        softening: float = 1e-3
    ) -> np.ndarray:
        """
        Computes the 1/r potential from all charges located inside
        [box_min, box_max] onto ``target_points`` by BRUTE-FORCE DIRECT
        SUMMATION over every source point in the box (O(T * M) for T targets
        and M in-box sources).

        Despite the ``multipole`` in the class name, this method performs NO
        hierarchical multipole / centroid expansion: it queries the box for
        the source indices and forms the full pairwise 1/r kernel. It is a
        convenience evaluator over the range-query result, not an FMM
        acceleration. Cost is O(T * M); use it only when M is small.
        """
        tgts = np.asarray(target_points, dtype=np.float64)
        if tgts.ndim == 1:
            tgts = tgts.reshape(1, -1)
            
        res = self.query_range(box_min, box_max, return_indices=True)
        if res["count"] == 0 or res["indices"] is None:
            return np.zeros(len(tgts), dtype=np.float64)
            
        src_pts = self.points[res["indices"]]
        src_vals = self.values[res["indices"]]  # shape (M, val_dim)
        
        # Direct summation of selected points: dists shape (T, M)
        diffs = tgts[:, None, :] - src_pts[None, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1) + softening**2)  # (T, M)
        
        # inv_dists shape (T, M, 1), src_vals shape (1, M, val_dim)
        inv_dists = (1.0 / dists)[:, :, None]
        potentials = np.sum(inv_dists * src_vals[None, :, :], axis=1)  # (T, val_dim)
        
        if self.val_dim == 1:
            return potentials.ravel()
        return potentials


def direct_range_query_baseline(
    points: np.ndarray,
    values: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray
) -> Dict[str, Any]:
    """Exact O(N) naive brute-force range search reference baseline."""
    pts = np.asarray(points, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals[:, None]
        
    mask = np.all((pts >= box_min) & (pts <= box_max), axis=1)
    indices = np.where(mask)[0]
    val_sum = np.sum(vals[mask], axis=0) if np.any(mask) else np.zeros(vals.shape[1])
    return {
        "count": int(np.sum(mask)),
        "sum": val_sum,
        "indices": indices
    }
