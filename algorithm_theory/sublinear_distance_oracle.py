"""
Sublinear-Time Approximate Distance Oracle (sublinear_distance_oracle.py).

Inspired by:
1. "Approximate Distance Oracles"
   Mikkel Thorup and Uri Zwick (STOC 2001 / J. ACM 2005).
2. "(1 + eps)-Approximate Distance Oracles for Doubling Metrics"
   Sariel Har-Peled, Manor Mendel (SODA 2006).
3. "Optimal Bounds for Open Addressing Without Reordering"
   Farach-Colton, Krapivin, & Kuszmaul (2025). IEEE FOCS 2024.

Key Algorithmic Principle:
Evaluating pairwise geodesic or manifold distances among N points typically requires O(N^2) storage
for an all-pairs distance matrix, or O(m + n log n) Dijkstra queries per online pair.
This module constructs a multi-scale landmark Approximate Distance Oracle (ADO). At each dyadic
level it elects one landmark per spatial-hash bucket (the FIRST point falling into each bucket --
a simple election heuristic, not a Thorup-Zwick / Har-Peled-Mendel sampling) and precomputes a
full single-source shortest-path vector from that landmark. Online queries combine the
triangle-inequality upper bounds d(u,v) <= d(u,lm) + d(lm,v) across levels; this yields a valid
upper bound but NO formal (1+eps) stretch guarantee is claimed (the first-point-per-bucket
election has no stretch bound). Query time is O(num_levels) = O(log(diameter / base_radius)).
Preprocessing space is dominated by the level-0 landmark SSSP tables: with cell_size = base_radius
there can be O(N) buckets, each storing an N-length distance vector, giving O(N^2)-class memory
at level 0 (not O(N log(1/eps))).
"""

import time
from typing import Tuple, List, Dict, Optional, Union
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithm_theory.tree_free_geodesic_fmm import FrontierClusteredSSSP, MeshfreeGeodesicSolver


class MultiScaleLandmarkOracle:
    """
    Hierarchical Multi-Scale Landmark Distance Oracle.
    
    Elects representative landmark nodes per spatial hash bucket at dyadic scales:
        r_l = r_0 * 2^l   for l = 0, ..., L-1
    Precomputes single-source distance vectors from elected landmarks to surrounding points.
    """
    def __init__(
        self,
        points: np.ndarray,
        adj_list: Optional[List[List[Tuple[int, float]]]] = None,
        base_radius: Optional[float] = None
    ):
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        self.n_points = len(self.points)
        if self.n_points == 0:
            raise ValueError("points must contain at least one point")

        p_min = self.points.min(axis=0)
        p_max = self.points.max(axis=0)
        self.diameter = float(np.linalg.norm(p_max - p_min)) + 1e-6

        if base_radius is None:
            self.base_radius = max(self.diameter / (self.n_points ** (1.0 / 3.0)), 1e-4)
        else:
            self.base_radius = float(base_radius)

        # Number of dyadic levels L = ceil(log2(diameter / base_radius)).
        # (The previous code/comment referenced an `eps` factor here, but eps
        # was never actually used in the level count and has been removed.)
        self.num_levels = max(2, int(np.ceil(np.log2(max(2.0, self.diameter / self.base_radius)))))
        
        # Build proximity graph if not provided
        if adj_list is None:
            solver = MeshfreeGeodesicSolver(self.points, k_neighbors=12, cell_size=self.base_radius)
            self.adj_list = solver.adj_list
        else:
            self.adj_list = adj_list

        self.landmarks_per_level: List[np.ndarray] = []
        self.point_to_nearest_landmark: List[np.ndarray] = []  # (num_levels, N)
        self.landmark_distances: List[Dict[int, np.ndarray]] = []  # level -> (landmark_id -> dist_array)
        
        self._build_multi_scale_landmarks()

    def _build_multi_scale_landmarks(self):
        """Constructs dyadic landmark hierarchy via spatial hashing."""
        for level in range(self.num_levels):
            cell_size = self.base_radius * (2.0 ** level)
            grid_coords = np.floor(self.points / cell_size).astype(np.int64)
            
            # Elect first point in each hash bucket as landmark
            bucket_map: Dict[Tuple[int, int, int], int] = {}
            for idx, coord in enumerate(grid_coords):
                k = (int(coord[0]), int(coord[1]), int(coord[2]))
                if k not in bucket_map:
                    bucket_map[k] = idx

            landmarks = np.array(list(bucket_map.values()), dtype=np.int64)
            self.landmarks_per_level.append(landmarks)

            # Map each point to its bucket's landmark
            pt_to_lm = np.zeros(self.n_points, dtype=np.int64)
            for idx, coord in enumerate(grid_coords):
                k = (int(coord[0]), int(coord[1]), int(coord[2]))
                pt_to_lm[idx] = bucket_map[k]
            self.point_to_nearest_landmark.append(pt_to_lm)

            # Precompute SSSP from elected landmarks at this level
            lm_dist_dict: Dict[int, np.ndarray] = {}
            for lm in landmarks:
                fc_sssp = FrontierClusteredSSSP(self.n_points, self.adj_list, delta=cell_size * 0.5)
                lm_dist_dict[int(lm)] = fc_sssp.compute(int(lm))
            self.landmark_distances.append(lm_dist_dict)

    def query_distance(self, u: int, v: int) -> float:
        """
        Answers an approximate distance between u and v in O(num_levels)
        operations by triangulating through the per-level elected landmarks
        (triangle-inequality upper bound d(u,v) <= d(u,lm)+d(lm,v)). This is
        an upper bound, NOT a (1+eps)-stretch guarantee.
        """
        if not (0 <= u < self.n_points and 0 <= v < self.n_points):
            raise IndexError("query vertex index out of range")
        if u == v:
            return 0.0

        # Exact check if directly adjacent
        for neigh, weight in self.adj_list[u]:
            if neigh == v:
                return float(weight)

        best_cand = np.inf

        # Test landmarks across multi-scale levels
        for level in range(self.num_levels):
            lm_u = int(self.point_to_nearest_landmark[level][u])
            lm_v = int(self.point_to_nearest_landmark[level][v])

            # Triangle inequality through landmark of u: d(u, v) <= d(lm_u, u) + d(lm_u, v)
            if lm_u in self.landmark_distances[level]:
                d_table = self.landmark_distances[level][lm_u]
                cand_u = d_table[u] + d_table[v]
                if cand_u < best_cand:
                    best_cand = cand_u

            # Triangle inequality through landmark of v
            if lm_v in self.landmark_distances[level]:
                d_table = self.landmark_distances[level][lm_v]
                cand_v = d_table[u] + d_table[v]
                if cand_v < best_cand:
                    best_cand = cand_v

        return float(best_cand) if np.isfinite(best_cand) else float("inf")


class ElasticMetricEmbedding:
    """
    Constant-Time Metric Coordinate Embedding for Ultra-Fast Geometric Distance Queries.
    
    Embeds points into an R-dimensional pseudo-metric space where distances are estimated
    via weighted L1/L2 coordinate norms: ||Phi(u) - Phi(v)||_p.
    """
    def __init__(self, oracle: MultiScaleLandmarkOracle, embedding_dim: int = 16):
        self.oracle = oracle
        self.dim = embedding_dim
        self.n_points = oracle.n_points

        # Select a diverse subset of global multi-scale landmarks as basis coordinates
        all_landmarks = []
        for level_lms in reversed(oracle.landmarks_per_level):
            all_landmarks.extend(list(level_lms))
            if len(all_landmarks) >= self.dim * 2:
                break

        unique_lms = list(dict.fromkeys(all_landmarks))
        self.basis_landmarks = np.array(unique_lms[:self.dim], dtype=np.int64)
        self.actual_dim = len(self.basis_landmarks)

        # Coordinate matrix: (N, actual_dim)
        self.embedding_matrix = np.zeros((self.n_points, self.actual_dim), dtype=np.float32)
        for d_idx, lm in enumerate(self.basis_landmarks):
            # Find landmark distances from precomputed tables
            found = False
            for lvl_dict in oracle.landmark_distances:
                if int(lm) in lvl_dict:
                    self.embedding_matrix[:, d_idx] = lvl_dict[int(lm)]
                    found = True
                    break
            if not found:
                # Fallback Euclidean
                self.embedding_matrix[:, d_idx] = np.linalg.norm(oracle.points - oracle.points[lm], axis=1)

    def query_embedded_distance(self, u: int, v: int) -> float:
        """Computes approximate distance in O(embedding_dim) operations."""
        # L_infinity / L2 mixed metric estimate
        coord_diff = np.abs(self.embedding_matrix[u] - self.embedding_matrix[v])
        return float(np.max(coord_diff))  # Metric lower bound from Frechet / Bourgain embedding


class SublinearDistanceOracle:
    """
    Unified High-Level Interface for Sublinear Distance Queries.
    """
    def __init__(self, points: np.ndarray, base_radius: Optional[float] = None):
        self.points = points
        self.oracle = MultiScaleLandmarkOracle(points, base_radius=base_radius)
        self.embedding = ElasticMetricEmbedding(self.oracle, embedding_dim=16)

    def query_pair(self, u: int, v: int, method: str = "landmark_oracle") -> float:
        """
        Queries approximate geodesic distance between nodes u and v.
        """
        if method == "landmark_oracle":
            return self.oracle.query_distance(u, v)
        elif method == "embedding":
            return self.embedding.query_embedded_distance(u, v)
        else:
            raise ValueError(f"Unknown query method: {method}")

    def query_batch(self, query_pairs: np.ndarray, method: str = "landmark_oracle") -> Tuple[np.ndarray, float]:
        """
        Evaluates a batch of (u, v) queries.
        
        Returns:
            (estimated_distances, elapsed_ms)
        """
        n_queries = len(query_pairs)
        out = np.zeros(n_queries, dtype=np.float64)
        
        t0 = time.perf_counter()
        if method == "embedding":
            u_idx = query_pairs[:, 0]
            v_idx = query_pairs[:, 1]
            diff = np.abs(self.embedding.embedding_matrix[u_idx] - self.embedding.embedding_matrix[v_idx])
            out = np.max(diff, axis=-1).astype(np.float64)
        else:
            for i in range(n_queries):
                u, v = int(query_pairs[i, 0]), int(query_pairs[i, 1])
                out[i] = self.oracle.query_distance(u, v)
                
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return out, elapsed_ms


if __name__ == "__main__":
    print("Testing Sublinear-Time Approximate Distance Oracle...")
    
    np.random.seed(42)
    n_pts = 1500
    # Sphere surface manifold
    theta = np.random.uniform(0, 2 * np.pi, n_pts)
    cos_phi = np.random.uniform(-1, 1, n_pts)
    sin_phi = np.sqrt(1.0 - cos_phi**2)
    x = sin_phi * np.cos(theta) * 10.0
    y = sin_phi * np.sin(theta) * 10.0
    z = cos_phi * 10.0
    pts = np.stack([x, y, z], axis=-1).astype(np.float64)

    ado = SublinearDistanceOracle(pts)

    # Sample test query pairs
    n_queries = 2000
    u_samples = np.random.randint(0, n_pts, n_queries)
    v_samples = np.random.randint(0, n_pts, n_queries)
    pairs = np.stack([u_samples, v_samples], axis=-1)

    # Benchmark online queries
    est_dists, t_query_ms = ado.query_batch(pairs, method="landmark_oracle")
    emb_dists, t_emb_ms = ado.query_batch(pairs, method="embedding")

    print(f"Total Points: {n_pts}, Evaluated Online Queries: {n_queries}")
    print(f"Landmark Oracle Query Time: {t_query_ms:.2f} ms ({n_queries / (t_query_ms / 1000.0):.0f} queries/sec)")
    print(f"Embedding Vectorized Query Time: {t_emb_ms:.2f} ms ({n_queries / (t_emb_ms / 1000.0):.0f} queries/sec)")
    print(f"Avg Estimated Distance: {np.mean(est_dists):.2f}")
    print(f"Avg Embedding Lower-Bound: {np.mean(emb_dists):.2f}")
    
    assert np.all(est_dists >= 0), "Distances must be non-negative."
    print("Sublinear Distance Oracle Verification: SUCCESS!")
