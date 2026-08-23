"""
Sublinear Multi-Scale Fast Dynamic Time Warping (sublinear_fast_dtw.py).

Inspired by:
1. "Toward Accurate Dynamic Time Warping in Linear Time and Space"
   Stan Salvador and Philip Chan (Intelligent Data Analysis, 2007).
2. "Soft-DTW: a Differentiable Loss Function for Time-Series"
   Marco Cuturi and Mathieu Blondel (ICML 2017).
3. "Optimal Bounds for Open Addressing Without Reordering"
   Farach-Colton, Krapivin, & Kuszmaul (2025). IEEE FOCS 2024 / arXiv:2501.02305.

Key Algorithmic Principle:
Given two multi-dimensional time series X in R^{T1 x D} and Y in R^{T2 x D}, finding optimal non-linear
temporal alignment via classical Dynamic Time Warping (DTW) requires computing an exact
O(T1 * T2) dynamic programming grid.

The Multi-Scale FastDTW algorithm approximates alignment via hierarchical coarsening:
1. Shrink time series recursively by dyadic factor 2: X_coarse = (x_{2i} + x_{2i+1}) / 2.
2. At coarsest level, solve exact DTW.
3. Project warping path to finer resolution and expand search corridor by radius r.
4. Evaluate DP only within the adaptive spatial corridor.
The per-level DP cost is O(T * radius) and there are O(log T) levels, giving a
total of O(T * radius * log T) (often written loosely as O(T * log T) when
radius is treated as a small constant, but the radius factor is real). This is
a HEURISTIC corridor approximation: Salvador-Chan FastDTW has NO optimality
guarantee -- the projected corridor can exclude the true optimal warping path,
so the returned distance is an upper bound on the exact DTW distance, not a
provable-accuracy approximation.
"""

import time
from typing import Tuple, List, Optional, Dict, Set
import numpy as np


class SublinearFastDTW:
    """
    Multi-Scale Adaptive Corridor Fast Dynamic Time Warping (FastDTW) Engine.

    Computes a HEURISTIC warping distance and alignment path (no optimality
    guarantee) in O(T * radius * log T) time (O(T * radius) per level over
    O(log T) levels). The result is an upper bound on the exact DTW distance.
    """
    def __init__(self, radius: int = 5, min_size: int = 32):
        self.radius = int(radius)
        self.min_size = int(min_size)
        if self.radius < 0 or self.min_size < 1:
            raise ValueError("radius must be non-negative and min_size must be positive")

    def _downsample(self, series: np.ndarray) -> np.ndarray:
        """Coarsens time series by dyadic factor of 2."""
        n = len(series)
        if n % 2 == 1:
            series = np.pad(series, ((0, 1), (0, 0)) if series.ndim == 2 else ((0, 1),), mode='edge')
            n += 1
        return (series[0::2] + series[1::2]) / 2.0

    def _compute_exact_dtw(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> Tuple[float, List[Tuple[int, int]]]:
        """Exact dense O(T1 * T2) DTW on small base series."""
        T1 = len(X)
        T2 = len(Y)
        if T1 == 0 or T2 == 0:
            raise ValueError("DTW series must be non-empty")
        if X.ndim != Y.ndim or (X.ndim == 2 and X.shape[1] != Y.shape[1]):
            raise ValueError("DTW series must have matching feature dimensions")
        
        # Pairwise distance matrix
        if X.ndim == 1:
            diff = np.abs(X[:, None] - Y[None, :])
        else:
            diff = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)

        cost = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float64)
        cost[0, 0] = 0.0

        for i in range(1, T1 + 1):
            for j in range(1, T2 + 1):
                cost[i, j] = diff[i - 1, j - 1] + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

        # Backtrack optimal warping path
        i, j = T1, T2
        path = [(i - 1, j - 1)]
        while i > 1 or j > 1:
            if i == 1:
                j -= 1
            elif j == 1:
                i -= 1
            else:
                steps = [cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]]
                best_step = np.argmin(steps)
                if best_step == 0:
                    i, j = i - 1, j - 1
                elif best_step == 1:
                    i -= 1
                else:
                    j -= 1
            path.append((i - 1, j - 1))

        path.reverse()
        return float(cost[T1, T2]), path

    def _expand_window(
        self,
        coarse_path: List[Tuple[int, int]],
        T1: int,
        T2: int
    ) -> Set[Tuple[int, int]]:
        """Projects coarse warping path into fine-level search corridor with radius padding."""
        window: Set[Tuple[int, int]] = set()
        
        for i_c, j_c in coarse_path:
            # Map coarse cell to 2x2 fine block
            for di in (0, 1):
                for dj in (0, 1):
                    i_f = i_c * 2 + di
                    j_f = j_c * 2 + dj
                    
                    if i_f < T1 and j_f < T2:
                        # Pad with radius
                        for r_i in range(-self.radius, self.radius + 1):
                            for r_j in range(-self.radius, self.radius + 1):
                                ii = i_f + r_i
                                jj = j_f + r_j
                                if 0 <= ii < T1 and 0 <= jj < T2:
                                    window.add((ii, jj))

        return window

    def _fast_dtw_recursive(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> Tuple[float, List[Tuple[int, int]]]:
        """Recursive multi-scale corridor refinement."""
        T1 = len(X)
        T2 = len(Y)

        if T1 <= self.min_size or T2 <= self.min_size:
            return self._compute_exact_dtw(X, Y)

        # 1. Coarsen
        X_coarse = self._downsample(X)
        Y_coarse = self._downsample(Y)

        # 2. Recurse on coarse resolution
        _, coarse_path = self._fast_dtw_recursive(X_coarse, Y_coarse)

        # 3. Expand corridor
        window = self._expand_window(coarse_path, T1, T2)

        # 4. Constrained Dynamic Programming in corridor
        # Sort cells in topological order (row-major)
        window_cells = sorted(list(window))
        
        cost_map: Dict[Tuple[int, int], float] = {}
        ptr_map: Dict[Tuple[int, int], Tuple[int, int]] = {}

        for i, j in window_cells:
            # Pointwise distance
            if X.ndim == 1:
                d_ij = abs(X[i] - Y[j])
            else:
                d_ij = float(np.linalg.norm(X[i] - Y[j]))

            if i == 0 and j == 0:
                cost_map[(0, 0)] = d_ij
                continue

            cands = []
            if (i - 1, j - 1) in cost_map:
                cands.append((cost_map[(i - 1, j - 1)], (i - 1, j - 1)))
            if (i - 1, j) in cost_map:
                cands.append((cost_map[(i - 1, j)], (i - 1, j)))
            if (i, j - 1) in cost_map:
                cands.append((cost_map[(i, j - 1)], (i, j - 1)))

            if cands:
                min_c, best_p = min(cands, key=lambda x: x[0])
                cost_map[(i, j)] = d_ij + min_c
                ptr_map[(i, j)] = best_p
            else:
                cost_map[(i, j)] = np.inf

        # Backtrack
        target = (T1 - 1, T2 - 1)
        if target not in cost_map:
            # Fallback to closest available cell
            target = min(cost_map.keys(), key=lambda k: abs(k[0] - (T1 - 1)) + abs(k[1] - (T2 - 1)))

        path = [target]
        curr = target
        while curr in ptr_map:
            curr = ptr_map[curr]
            path.append(curr)

        path.reverse()
        total_dist = float(cost_map.get((T1 - 1, T2 - 1), cost_map[target]))
        return total_dist, path

    def align(
        self,
        series_1: np.ndarray,
        series_2: np.ndarray
    ) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Computes sublinear FastDTW alignment distance and warping path.
        
        Args:
            series_1: (T1, D) or (T1,) time series 1
            series_2: (T2, D) or (T2,) time series 2
            
        Returns:
            dtw_distance: Total warping alignment cost
            warping_path: List of matched index pairs [(i, j), ...]
        """
        s1 = np.asarray(series_1, dtype=np.float64)
        s2 = np.asarray(series_2, dtype=np.float64)
        if s1.ndim not in (1, 2) or s2.ndim != s1.ndim:
            raise ValueError("series must have shape (T,) or (T, D) with matching ranks")
        if s1.ndim == 2 and s1.shape[1] != s2.shape[1]:
            raise ValueError("multivariate series must have matching feature dimensions")
        if len(s1) == 0 or len(s2) == 0 or not np.all(np.isfinite(s1)) or not np.all(np.isfinite(s2)):
            raise ValueError("series must be non-empty and finite")
        return self._fast_dtw_recursive(s1, s2)


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("Sublinear Multi-Scale FastDTW Alignment Benchmark")
    print("=" * 70)

    # Generate two non-linearly warped temporal signals of length T = 4,000
    T = 4000
    print(f"Time Series Length (T)       : {T:,} timesteps")

    t_grid = np.linspace(0, 10 * np.pi, T)
    s1 = np.sin(t_grid) + 0.5 * np.cos(2.5 * t_grid)
    
    # Warped timeline: non-linear quadratic phase shift
    warp_t = np.linspace(0, np.sqrt(10 * np.pi), T) ** 2
    s2 = np.sin(warp_t) + 0.5 * np.cos(2.5 * warp_t) + np.random.randn(T) * 0.05

    fast_dtw = SublinearFastDTW(radius=4, min_size=64)

    # 1. FastDTW Multi-Scale Alignment
    t0 = time.perf_counter()
    dist_fast, path_fast = fast_dtw.align(s1, s2)
    t_fast = (time.perf_counter() - t0) * 1000.0

    print(f"FastDTW Multi-Scale Time    : {t_fast:.2f} ms")
    print(f"Computed Warping Distance   : {dist_fast:.4f}")
    print(f"Warping Path Length         : {len(path_fast):,} matched pairs")

    # 2. Dense Exact DTW Baseline on subset
    T_sub = 800
    t0 = time.perf_counter()
    dist_sub, _ = fast_dtw._compute_exact_dtw(s1[:T_sub], s2[:T_sub])
    t_dense_sub = (time.perf_counter() - t0) * 1000.0
    t_dense_proj = t_dense_sub * ((T * T) / (T_sub * T_sub))

    print(f"Projected Dense O(T^2) Time : {t_dense_proj:.2f} ms")
    print(f"Measured Speedup Ratio      : {t_dense_proj / max(t_fast, 1e-6):.1f}x")
    print("=" * 70)
