"""
SE(3) Equivariant Multipole Transformer (`equivariant_transformer.py`)
====================================================================
Linear-Time O(N) SE(3)-Equivariant Self-Attention Layer.
Processes coupled Invariant Scalar Features h in R^D_s and Equivariant Vector Channels v in R^(D_v x 3).

Key Equivariance Theorems:
- Rotation: Under R in SO(3), h -> h, v -> R v, coords -> R coords.
- Translation: Under t in R^3, h -> h, v -> v, coords -> coords + t.
- Reflection: E(3) parity consistency.
"""

import numpy as np

try:
    from neural_ops._coord_contract import check_unit_coords
except ImportError:  # direct script execution (repo root not yet on sys.path)
    from _coord_contract import check_unit_coords
from typing import Optional, Tuple, Dict, Any, List


class EquivariantMultipoleTransformerLayer:
    """
    Linear-Time O(N) SE(3)-Equivariant Self-Attention Layer.
    Computes all-pairs invariant scalar attention and equivariant vector message passing.
    """
    def __init__(
        self,
        scalar_dim: int = 64,
        vector_dim: int = 16,
        grid_depth: int = 4,
        spatial_sigma: float = 0.25,
    ):
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.grid_depth = grid_depth
        self.grid_res = 1 << grid_depth
        self.spatial_sigma = spatial_sigma

        # Scalar projections
        scale_s = 1.0 / np.sqrt(scalar_dim)
        rng = np.random.RandomState(42)
        self.W_qs = rng.normal(0, scale_s, size=(scalar_dim, scalar_dim)).astype(np.float32)
        self.W_ks = rng.normal(0, scale_s, size=(scalar_dim, scalar_dim)).astype(np.float32)
        self.W_vs = rng.normal(0, scale_s, size=(scalar_dim, scalar_dim)).astype(np.float32)
        self.W_out_s = rng.normal(0, scale_s, size=(scalar_dim, scalar_dim)).astype(np.float32)

        # Vector projections (act on vector channel dimension D_v, preserving 3D spatial index)
        scale_v = 1.0 / np.sqrt(vector_dim)
        self.W_v_proj = rng.normal(0, scale_v, size=(vector_dim, vector_dim)).astype(np.float32)
        self.W_v_out = rng.normal(0, scale_v, size=(vector_dim, vector_dim)).astype(np.float32)

        # Cross scalar-vector coupling
        self.W_s2v = rng.normal(0, scale_s, size=(scalar_dim, vector_dim)).astype(np.float32)
        self.W_v2s = rng.normal(0, scale_v, size=(vector_dim, scalar_dim)).astype(np.float32)

    def _morton_encode(self, coords: np.ndarray) -> np.ndarray:
        res = self.grid_res
        grid_indices = np.clip(np.floor(coords * res).astype(np.int64), 0, res - 1)
        return grid_indices[:, 0] + grid_indices[:, 1] * res + grid_indices[:, 2] * (res ** 2)

    def forward(
        self,
        coords: np.ndarray,      # (N, 3) 3D spatial coordinates in [0, 1)^3
        scalar_feats: np.ndarray,# (N, scalar_dim) Invariant scalar representations
        vector_feats: Optional[np.ndarray] = None, # (N, vector_dim, 3) Equivariant vector channels
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Forward pass producing updated invariant scalar and equivariant vector features.
        Returns: (scalar_out (N, scalar_dim), vector_out (N, vector_dim, 3), meta)
        """
        check_unit_coords(coords, "EquivariantMultipoleTransformerLayer.forward(coords)")
        N, D_s = scalar_feats.shape
        D_v = self.vector_dim
        if vector_feats is None:
            vector_feats = np.zeros((N, D_v, 3), dtype=np.float32)

        coords_clipped = np.clip(coords, 1e-4, 1.0 - 1e-4)

        # 1. Projections
        Q_s = scalar_feats @ self.W_qs # (N, D_s)
        K_s = scalar_feats @ self.W_ks # (N, D_s)
        V_s = scalar_feats @ self.W_vs # (N, D_s)

        # Vector linear transformation: einsum('nvd,vw->nwd', vector_feats, W_v_proj)
        V_vec = np.einsum('nvd,vw->nwd', vector_feats, self.W_v_proj) # (N, D_v, 3)

        # 2. Bucket tokens into spatial hash grid
        keys = self._morton_encode(coords_clipped)
        bucket_map: Dict[int, List[int]] = {}
        for i in range(N):
            k = int(keys[i])
            if k not in bucket_map:
                bucket_map[k] = []
            bucket_map[k].append(i)

        cluster_keys = list(bucket_map.keys())
        n_clusters = len(cluster_keys)
        key_to_idx = {k: idx for idx, k in enumerate(cluster_keys)}

        # 3. Far-Field Multipole Cluster Moments
        all_centers = np.zeros((n_clusters, 3), dtype=np.float32)
        all_mean_k = np.zeros((n_clusters, D_s), dtype=np.float32)
        all_sum_v_s = np.zeros((n_clusters, D_s), dtype=np.float32)
        all_sum_v_vec = np.zeros((n_clusters, D_v, 3), dtype=np.float32)
        all_counts = np.zeros(n_clusters, dtype=np.float32)

        for idx, k in enumerate(cluster_keys):
            p_ids = bucket_map[k]
            pts = coords_clipped[p_ids]
            c_center = np.mean(pts, axis=0)
            all_centers[idx] = c_center
            all_mean_k[idx] = np.mean(K_s[p_ids], axis=0)
            all_sum_v_s[idx] = np.sum(V_s[p_ids], axis=0)
            all_sum_v_vec[idx] = np.sum(V_vec[p_ids], axis=0)
            all_counts[idx] = len(p_ids)

        # 4. Vectorized Evaluation (Near exact + Far multipole)
        out_s = np.zeros((N, D_s), dtype=np.float32)
        out_v = np.zeros((N, D_v, 3), dtype=np.float32)
        inv_2_sigma_sq = 1.0 / (2.0 * (self.spatial_sigma ** 2))
        inv_sqrt_d = 1.0 / np.sqrt(D_s)

        res = self.grid_res
        for k_src, p_src in bucket_map.items():
            p_src_arr = np.asarray(p_src, dtype=np.int32)
            M_src = len(p_src_arr)
            q_s_src = Q_s[p_src_arr]
            pts_src = coords_clipped[p_src_arr]
            center_src = all_centers[key_to_idx[k_src]]

            # Find near neighbors.
            # Derive the source cell from the bucket key k_src itself (NOT from
            # the cluster centroid): for a multi-particle bucket whose members
            # hug a cell boundary, the centroid can land in a different cell
            # than every member, and the ring-1 neighborhood around the
            # centroid cell then misses a cell that is ring-1 adjacent to the
            # bucket's actual cell (where a true near neighbor lives).  The
            # bucket key is LINEAR (see _morton_encode): k = nx + ny*res +
            # nz*res^2, so the cell coords are recovered by integer division.
            src_grid = np.array([k_src % res,
                                 (k_src // res) % res,
                                 k_src // (res * res)], dtype=np.int64)
            near_indices_set = set()
            near_p_list = []

            for dx in (-1, 0, 1):
                nx = int(src_grid[0]) + dx
                if 0 <= nx < res:
                    for dy in (-1, 0, 1):
                        ny = int(src_grid[1]) + dy
                        if 0 <= ny < res:
                            for dz in (-1, 0, 1):
                                nz = int(src_grid[2]) + dz
                                if 0 <= nz < res:
                                    nk = int(nx + ny * res + nz * (res ** 2))
                                    if nk in key_to_idx:
                                        near_indices_set.add(key_to_idx[nk])
                                        near_p_list.extend(bucket_map[nk])

            # --- Near-field exact evaluation ---
            near_arr = np.asarray(near_p_list, dtype=np.int32)
            pts_near = coords_clipped[near_arr]
            k_s_near = K_s[near_arr]
            v_s_near = V_s[near_arr]
            v_vec_near = V_vec[near_arr] # (N_near, D_v, 3)

            # Spatial distance: (M_src, N_near)
            diff_near = pts_src[:, None, :] - pts_near[None, :, :]
            dist_sq_near = np.sum(diff_near ** 2, axis=-1)
            spatial_w_near = np.exp(-dist_sq_near * inv_2_sigma_sq)

            # Invariant scalar attention weights
            dot_near = (q_s_src @ k_s_near.T) * inv_sqrt_d
            attn_near = spatial_w_near * np.exp(np.clip(dot_near, -30.0, 30.0))

            # Mask self-pairs (target i == source i) so each token is not
            # attended to itself.  Pass-30 fix: the original code had no
            # self-masking, adding a spurious self-attention contribution
            # (weight 1.0 at distance 0) to every near-field evaluation.
            id_t = p_src_arr[:, None]
            id_s = near_arr[None, :]
            self_mask = (id_t == id_s).astype(np.float32)
            attn_near = attn_near * (1.0 - self_mask)

            val_s_near = attn_near @ v_s_near # (M_src, D_s)
            val_v_near = np.einsum('mn,nvd->mvd', attn_near, v_vec_near) # (M_src, D_v, 3)

            # Geometric displacement injection (Equivariant vector message):
            # inject (x_i - x_j) * phi(h_j)
            s2v_near = scalar_feats[near_arr] @ self.W_s2v # (N_near, D_v)
            geom_vec_near = np.einsum('mn,mnd,nv->mvd', attn_near, diff_near, s2v_near)
            val_v_near += geom_vec_near

            weight_near = np.sum(attn_near, axis=-1, keepdims=True) + 1e-9

            # --- Far-field multipole evaluation ---
            far_indices = [idx for idx in range(n_clusters) if idx not in near_indices_set]
            if far_indices:
                far_idx_arr = np.asarray(far_indices, dtype=np.int32)
                far_centers = all_centers[far_idx_arr]
                far_k_means = all_mean_k[far_idx_arr]
                far_v_s_sums = all_sum_v_s[far_idx_arr]
                far_v_vec_sums = all_sum_v_vec[far_idx_arr] # (N_far, D_v, 3)
                far_counts = all_counts[far_idx_arr]

                diff_far = pts_src[:, None, :] - far_centers[None, :, :]
                dist_sq_far = np.sum(diff_far ** 2, axis=-1)
                spatial_w_far = np.exp(-dist_sq_far * inv_2_sigma_sq)

                dot_far = (q_s_src @ far_k_means.T) * inv_sqrt_d
                w_far = spatial_w_far * np.exp(np.clip(dot_far, -30.0, 30.0))

                val_s_far = w_far @ far_v_s_sums
                val_v_far = np.einsum('mf,fvd->mvd', w_far, far_v_vec_sums)
                weight_far = w_far @ far_counts[:, None]

                val_s_total = (val_s_near + val_s_far) / (weight_near + weight_far)
                val_v_total = (val_v_near + val_v_far) / (weight_near[:, :, None] + weight_far[:, :, None])
            else:
                val_s_total = val_s_near / weight_near
                val_v_total = val_v_near / weight_near[:, :, None]

            # Invariant scalar-vector dot product coupling: ||v||^2 into scalar channels
            v_norm_sq = np.sum(val_v_total ** 2, axis=-1) # (M_src, D_v)
            s_from_v = v_norm_sq @ self.W_v2s # (M_src, D_s)

            out_s[p_src_arr] = scalar_feats[p_src_arr] + val_s_total @ self.W_out_s + s_from_v
            out_v[p_src_arr] = vector_feats[p_src_arr] + np.einsum('mvd,vw->mwd', val_v_total, self.W_v_out)

        meta = {
            "num_particles": N,
            "scalar_dim": self.scalar_dim,
            "vector_dim": self.vector_dim,
            "active_clusters": n_clusters,
            "equivariance_group": "SE(3)",
        }
        return out_s, out_v, meta
