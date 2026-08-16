"""
Spatial Disjoint Set & Geometric Dynamic Connectivity Engine.
Bridging Disjoint-Set Union (Union-Find) with Flat Elastic Spatial Hashing.

Replaces O(N^2) pairwise distance graphs with an O(N) tree-free spatial percolation
and geometric connected components solver.

Key Capabilities:
1. O(N) Spatial Connected Components & Density Percolation (Epsilon-Ball Clustering).
2. Lock-free flat cell neighborhood edge generation.
3. Path compression & union-by-rank disjoint set forest with component centroid tracking.
4. Approximate Euclidean Minimum Spanning Forest (EMSF).
"""

from typing import Tuple, Optional, List, Dict, Union, Any, Set
import numpy as np
import time


class SpatialDisjointSetFMM:
    """
    Spatial Disjoint Set Union (Union-Find) Engine.
    
    Identifies connected spatial components, percolation clusters, and minimum spanning
    edges across multi-dimensional point coordinates in linear O(N) time.
    """
    def __init__(self, points: np.ndarray, connectivity_radius: float):
        """
        Parameters
        ----------
        points : np.ndarray
            N x D point coordinates.
        connectivity_radius : float
            Maximum distance (epsilon) between two points to be considered connected.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[:, None]
        if pts.size == 0:
            raise ValueError("points must not be empty")
        if connectivity_radius <= 0:
            raise ValueError("connectivity_radius must be positive")
            
        self.N, self.D = pts.shape
        self.points = pts
        self.eps = float(connectivity_radius)
        self.eps_sq = self.eps * self.eps
        
        # Disjoint set parent and rank arrays
        self.parent = np.arange(self.N, dtype=np.int64)
        self.rank = np.zeros(self.N, dtype=np.int32)
        self.component_size = np.ones(self.N, dtype=np.int64)
        
        self._build_spatial_hash_and_unify()

    def find(self, i: int) -> int:
        """Finds representative root of node i with full path compression."""
        root = int(i)
        while root != self.parent[root]:
            root = self.parent[root]
            
        # Path compression pass
        curr = int(i)
        while curr != root:
            nxt = self.parent[curr]
            self.parent[curr] = root
            curr = nxt
            
        return root

    def union(self, i: int, j: int) -> bool:
        """
        Unifies components containing i and j via union-by-rank.
        Returns True if a new merge occurred, False if already connected.
        """
        root_i = self.find(i)
        root_j = self.find(j)
        
        if root_i == root_j:
            return False
            
        if self.rank[root_i] < self.rank[root_j]:
            self.parent[root_i] = root_j
            self.component_size[root_j] += self.component_size[root_i]
        elif self.rank[root_i] > self.rank[root_j]:
            self.parent[root_j] = root_i
            self.component_size[root_i] += self.component_size[root_j]
        else:
            self.parent[root_j] = root_i
            self.component_size[root_i] += self.component_size[root_j]
            self.rank[root_i] += 1
            
        return True

    def _build_spatial_hash_and_unify(self):
        """Constructs spatial grid and merges adjacent points within radius epsilon."""
        # Quantize coordinates to cell coordinates of size eps
        cell_coords = np.floor(self.points / self.eps).astype(np.int64)
        
        # Map tuple(cell_coords) -> list of point indices
        cell_map: Dict[Tuple[int, ...], List[int]] = {}
        for idx in range(self.N):
            cell_key = tuple(cell_coords[idx].tolist())
            if cell_key not in cell_map:
                cell_map[cell_key] = []
            cell_map[cell_key].append(idx)
            
        # Generate directional neighbor offsets in D dimensions: (-1, 0, 1)^D
        dim_ranges = [(-1, 0, 1) for _ in range(self.D)]
        offsets = np.array(np.meshgrid(*dim_ranges)).T.reshape(-1, self.D)
        
        visited_cells: Set[Tuple[int, ...]] = set()
        
        for cell_key, p_indices in cell_map.items():
            cell_arr = np.array(cell_key, dtype=np.int64)
            
            # 1. Intra-cell connectivity
            if len(p_indices) > 1:
                pts_in_cell = self.points[p_indices]
                for i_local in range(len(p_indices)):
                    idx_i = p_indices[i_local]
                    p_i = pts_in_cell[i_local]
                    diffs = pts_in_cell[i_local + 1:] - p_i[None, :]
                    dists_sq = np.sum(diffs**2, axis=1)
                    matches = np.where(dists_sq <= self.eps_sq)[0]
                    for m in matches:
                        idx_j = p_indices[i_local + 1 + m]
                        self.union(idx_i, idx_j)
                        
            # 2. Inter-cell neighborhood connectivity
            for off in offsets:
                # Check only non-zero positive directional offsets or unvisited neighbors
                if np.all(off == 0):
                    continue
                neighbor_key = tuple((cell_arr + off).tolist())
                if neighbor_key in visited_cells or neighbor_key not in cell_map:
                    continue
                    
                neigh_indices = cell_map[neighbor_key]
                pts_neigh = self.points[neigh_indices]
                pts_curr = self.points[p_indices]
                
                # Pairwise distance broadcast between cells
                diffs = pts_curr[:, None, :] - pts_neigh[None, :, :]
                dists_sq = np.sum(diffs**2, axis=-1)
                match_rows, match_cols = np.where(dists_sq <= self.eps_sq)
                
                for r, c in zip(match_rows, match_cols):
                    self.union(p_indices[r], neigh_indices[c])
                    
            visited_cells.add(cell_key)

    def get_components_summary(self) -> Dict[str, Any]:
        """Returns component counts, size distribution, labels, and centroids."""
        # Compress all paths to get clean root labels
        labels = np.array([self.find(i) for i in range(self.N)], dtype=np.int64)
        unique_roots, counts = np.unique(labels, return_counts=True)
        
        # Compute centroids for each connected component
        centroids: Dict[int, np.ndarray] = {}
        for root in unique_roots:
            comp_mask = (labels == root)
            centroids[int(root)] = np.mean(self.points[comp_mask], axis=0)
            
        max_cluster_size = int(np.max(counts)) if len(counts) > 0 else 0
        percolation_ratio = float(max_cluster_size / max(1, self.N))
        
        return {
            "num_components": len(unique_roots),
            "unique_roots": unique_roots,
            "component_sizes": counts,
            "labels": labels,
            "centroids": centroids,
            "max_cluster_size": max_cluster_size,
            "percolation_ratio": percolation_ratio
        }

    def compute_approximate_spanning_forest(self) -> List[Tuple[int, int, float]]:
        """
        Computes edges of the approximate Euclidean Minimum Spanning Forest (EMSF)
        connecting points within radius epsilon.
        """
        # Re-run Kruskal-style edge collection across spatial neighborhood
        cell_coords = np.floor(self.points / self.eps).astype(np.int64)
        cell_map: Dict[Tuple[int, ...], List[int]] = {}
        for idx in range(self.N):
            cell_key = tuple(cell_coords[idx].tolist())
            if cell_key not in cell_map:
                cell_map[cell_key] = []
            cell_map[cell_key].append(idx)
            
        dim_ranges = [(-1, 0, 1) for _ in range(self.D)]
        offsets = np.array(np.meshgrid(*dim_ranges)).T.reshape(-1, self.D)
        
        edges: List[Tuple[float, int, int]] = []
        visited_cells: Set[Tuple[int, ...]] = set()
        
        for cell_key, p_indices in cell_map.items():
            cell_arr = np.array(cell_key, dtype=np.int64)
            pts_in_cell = self.points[p_indices]
            
            for i_local in range(len(p_indices)):
                idx_i = p_indices[i_local]
                p_i = pts_in_cell[i_local]
                diffs = pts_in_cell[i_local + 1:] - p_i[None, :]
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                matches = np.where(dists <= self.eps)[0]
                for m in matches:
                    edges.append((float(dists[m]), idx_i, p_indices[i_local + 1 + m]))
                    
            for off in offsets:
                if np.all(off == 0):
                    continue
                neighbor_key = tuple((cell_arr + off).tolist())
                if neighbor_key in visited_cells or neighbor_key not in cell_map:
                    continue
                neigh_indices = cell_map[neighbor_key]
                pts_neigh = self.points[neigh_indices]
                
                diffs = pts_in_cell[:, None, :] - pts_neigh[None, :, :]
                dists = np.sqrt(np.sum(diffs**2, axis=-1))
                m_r, m_c = np.where(dists <= self.eps)
                for r, c in zip(m_r, m_c):
                    edges.append((float(dists[r, c]), p_indices[r], neigh_indices[c]))
                    
            visited_cells.add(cell_key)
            
        # Kruskal greedy sort & merge
        edges.sort(key=lambda x: x[0])
        forest_edges: List[Tuple[int, int, float]] = []
        
        # Temporary DSU for spanning tree
        tree_parent = np.arange(self.N, dtype=np.int64)
        def tree_find(u: int) -> int:
            while u != tree_parent[u]:
                tree_parent[u] = tree_parent[tree_parent[u]]
                u = tree_parent[u]
            return u
            
        for weight, u, v in edges:
            ru = tree_find(u)
            rv = tree_find(v)
            if ru != rv:
                tree_parent[ru] = rv
                forest_edges.append((u, v, weight))
                
        return forest_edges
