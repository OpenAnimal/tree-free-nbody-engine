"""
Continuous Meshfree GNN Layer (`continuous_meshfree_gnn.py`)
============================================================
Continuous Spatial Graph Convolution without Adjacency Matrices or Edge Lists.
Powered by Tree-Free Fast Multipole Method (FMM) & Farach-Colton (2025) Spatial Hashing.

Enables continuous message passing across dynamic point clouds, fluid particles,
and astrophysical / molecular simulations in O(N) linear time:
h_i^(l+1) = sigma( W_self h_i + sum_{near} K_near(x_i, x_j) W_near h_j + sum_{far} K_far(x_i, c_k) W_far M_k + b )
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List

try:
    from .multipole_attention import ElasticSpatialHash, morton_encode_2d, morton_encode_3d
except (ImportError, ValueError):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from neural_ops.multipole_attention import ElasticSpatialHash, morton_encode_2d, morton_encode_3d


class ContinuousMeshfreeGNNLayer:
    """
    Mesh-Free Continuous Graph Convolution Layer.
    Executes all-pairs spatial message passing in linear O(N) time without constructing edge lists.
    """
    def __init__(
        self,
        in_features: int = 32,
        out_features: int = 32,
        spatial_dim: int = 3,
        grid_depth: int = 4,
        cutoff_radius: float = 0.15,
        kernel_type: str = "rbf",
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.spatial_dim = spatial_dim
        self.grid_depth = grid_depth
        self.cutoff_radius = cutoff_radius
        self.kernel_type = kernel_type
        self.grid_res = 1 << grid_depth

        # Learnable transform weights
        scale = np.sqrt(2.0 / (in_features + out_features))
        rng = np.random.RandomState(42)
        self.W_self = rng.normal(0, scale, size=(in_features, out_features)).astype(np.float32)
        self.W_near = rng.normal(0, scale, size=(in_features, out_features)).astype(np.float32)
        self.W_far = rng.normal(0, scale, size=(in_features, out_features)).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)

    def _spatial_kernel(self, dist_sq: np.ndarray) -> np.ndarray:
        """Computes smooth continuous kernel weighting."""
        if self.kernel_type == "rbf":
            sigma_sq = (self.cutoff_radius / 2.0) ** 2
            return np.exp(-dist_sq / (2.0 * sigma_sq + 1e-8))
        elif self.kernel_type == "wendland":
            r = np.sqrt(dist_sq)
            q = np.clip(r / self.cutoff_radius, 0.0, 1.0)
            return ((1.0 - q) ** 4) * (4.0 * q + 1.0)
        else:
            return 1.0 / (1.0 + dist_sq / (self.cutoff_radius ** 2))

    def forward(
        self,
        node_features: np.ndarray,  # (N, in_features)
        coords: np.ndarray,         # (N, spatial_dim) Continuous spatial coordinates in [0, 1)^d
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward continuous graph convolution in O(N) time with vectorized bucket operations.
        Returns: (updated_features (N, out_features), metadata_dict)
        """
        N, F_in = node_features.shape
        coords_clipped = np.clip(coords, 1e-4, 1.0 - 1e-4)

        # 1. Bucket nodes into non-reordering elastic spatial hash
        max_boxes = self.grid_res ** self.spatial_dim
        hash_table = ElasticSpatialHash(capacity=max_boxes * 2)
        bucket_map: Dict[int, List[int]] = {}

        if self.spatial_dim == 2:
            for i in range(N):
                k = morton_encode_2d(coords_clipped[i, 0], coords_clipped[i, 1], depth=self.grid_depth)
                if k not in bucket_map:
                    bucket_map[k] = []
                bucket_map[k].append(i)
        else:
            for i in range(N):
                k = morton_encode_3d(coords_clipped[i, 0], coords_clipped[i, 1], coords_clipped[i, 2], depth=self.grid_depth)
                if k not in bucket_map:
                    bucket_map[k] = []
                bucket_map[k].append(i)

        for k, p_ids in bucket_map.items():
            hash_table.insert(k, p_ids)

        # 2. Linear feature projections
        h_self = np.matmul(node_features, self.W_self) # (N, out_features)
        h_near_proj = np.matmul(node_features, self.W_near) # (N, out_features)
        h_far_proj = np.matmul(node_features, self.W_far) # (N, out_features)

        # 3. Multipole cluster moments (P2M)
        cluster_keys = list(bucket_map.keys())
        n_clusters = len(cluster_keys)
        key_to_idx = {k: idx for idx, k in enumerate(cluster_keys)}

        all_centers = np.zeros((n_clusters, self.spatial_dim), dtype=np.float32)
        all_moments = np.zeros((n_clusters, self.out_features), dtype=np.float32)
        all_counts = np.zeros(n_clusters, dtype=np.float32)

        for idx, k in enumerate(cluster_keys):
            p_ids = bucket_map[k]
            pts = coords_clipped[p_ids]
            all_centers[idx] = np.mean(pts, axis=0)
            all_moments[idx] = np.mean(h_far_proj[p_ids], axis=0)
            all_counts[idx] = len(p_ids)

        # 4. Message aggregation: Vectorized Bucket-Level Near P2P + Far M2L
        out_messages = np.zeros((N, self.out_features), dtype=np.float32)
        cell_size = 1.0 / self.grid_res

        for k_src, p_src in bucket_map.items():
            p_src_arr = np.asarray(p_src, dtype=np.int32)
            M_src = len(p_src_arr)
            pts_src = coords_clipped[p_src_arr]

            center_src = all_centers[key_to_idx[k_src]]
            if self.spatial_dim == 2:
                ix = int(center_src[0] * self.grid_res)
                iy = int(center_src[1] * self.grid_res)
                near_keys = []
                for dx in (-1, 0, 1):
                    nx = ix + dx
                    if 0 <= nx < self.grid_res:
                        for dy in (-1, 0, 1):
                            ny = iy + dy
                            if 0 <= ny < self.grid_res:
                                nk = morton_encode_2d((nx + 0.5) * cell_size, (ny + 0.5) * cell_size, depth=self.grid_depth)
                                near_keys.append(nk)
            else:
                ix = int(center_src[0] * self.grid_res)
                iy = int(center_src[1] * self.grid_res)
                iz = int(center_src[2] * self.grid_res)
                near_keys = []
                for dx in (-1, 0, 1):
                    nx = ix + dx
                    if 0 <= nx < self.grid_res:
                        for dy in (-1, 0, 1):
                            ny = iy + dy
                            if 0 <= ny < self.grid_res:
                                for dz in (-1, 0, 1):
                                    nz = iz + dz
                                    if 0 <= nz < self.grid_res:
                                        nk = morton_encode_3d((nx + 0.5) * cell_size, (ny + 0.5) * cell_size, (nz + 0.5) * cell_size, depth=self.grid_depth)
                                        near_keys.append(nk)

            near_p_list = []
            near_indices_set = set()
            for nk in near_keys:
                p_n = hash_table.lookup(nk)
                if p_n is not None:
                    near_p_list.extend(p_n)
                    if nk in key_to_idx:
                        near_indices_set.add(key_to_idx[nk])

            # --- Near-Field Message Passing ---
            near_arr = np.asarray(near_p_list, dtype=np.int32)
            pts_near = coords_clipped[near_arr]
            h_near = h_near_proj[near_arr]

            diff_near = pts_src[:, None, :] - pts_near[None, :, :]
            dist_sq_near = np.sum(diff_near ** 2, axis=-1)
            k_near = self._spatial_kernel(dist_sq_near) # (M_src, N_near)

            msg_near = np.matmul(k_near, h_near) # (M_src, out_features)
            weight_near = np.sum(k_near, axis=-1, keepdims=True) + 1e-8

            # --- Far-Field Cluster Message Passing ---
            far_indices = [idx for idx in range(n_clusters) if idx not in near_indices_set]
            if far_indices:
                far_idx_arr = np.asarray(far_indices, dtype=np.int32)
                far_centers = all_centers[far_idx_arr]
                far_moments = all_moments[far_idx_arr]
                far_counts = all_counts[far_idx_arr]

                diff_far = pts_src[:, None, :] - far_centers[None, :, :]
                dist_sq_far = np.sum(diff_far ** 2, axis=-1)
                k_far = self._spatial_kernel(dist_sq_far) # (M_src, N_far)

                weighted_moments = k_far * far_counts[None, :] # (M_src, N_far)
                msg_far = np.matmul(weighted_moments, far_moments) # (M_src, out_features)
                weight_far = np.sum(weighted_moments, axis=-1, keepdims=True) + 1e-8

                total_msg = (msg_near / weight_near) + (msg_far / weight_far)
            else:
                total_msg = msg_near / weight_near

            out_messages[p_src_arr] = total_msg

        # Combined update: h_self + messages + bias -> ReLU
        total_pre_act = h_self + out_messages + self.bias
        updated_features = np.maximum(0, total_pre_act) # ReLU activation

        meta = {
            "num_nodes": N,
            "active_clusters": len(bucket_map),
            "spatial_dim": self.spatial_dim,
            "complexity": "O(N) Mesh-Free",
        }
        return updated_features, meta
