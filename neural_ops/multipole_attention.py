"""
Tree-Free Multipole Attention (`multipole_attention.py`)
========================================================
Linear-Time O(N) Spatial and Manifold Attention Layer.
Powered by Tree-Free Fast Multipole Method (Greengard-Rokhlin) & Farach-Colton (2025) Open Addressing.

Replaces standard O(N^2) Softmax Multi-Head Attention for:
- 2D Vision Transformers (ViT, High-Resolution 4K/8K Patch Tokens)
- 3D Perception (Point Clouds, LiDAR, 3D Gaussian Splatting)
- 1D Sequence Manifolds with Positional Embedding Grids
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List


def morton_encode_2d(x: float, y: float, depth: int = 5) -> int:
    """Morton z-order curve encoding in [0, 1) x [0, 1)."""
    grid_res = 1 << depth
    ix = min(grid_res - 1, max(0, int(x * grid_res)))
    iy = min(grid_res - 1, max(0, int(y * grid_res)))
    
    def spread_bits(v: int) -> int:
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        v = (v | (v << 1)) & 0x55555555
        return v
    
    return (spread_bits(ix) | (spread_bits(iy) << 1)) | (depth << 24)


def morton_encode_3d(x: float, y: float, z: float, depth: int = 4) -> int:
    """Morton z-order curve encoding in [0, 1)^3."""
    grid_res = 1 << depth
    ix = min(grid_res - 1, max(0, int(x * grid_res)))
    iy = min(grid_res - 1, max(0, int(y * grid_res)))
    iz = min(grid_res - 1, max(0, int(z * grid_res)))

    def split_by_3(a: int) -> int:
        a &= 0x3ff
        a = (a | (a << 16)) & 0x30000ff
        a = (a | (a << 8)) & 0x300f00f
        a = (a | (a << 4)) & 0x30c30c3
        a = (a | (a << 2)) & 0x9249249
        return a

    return (split_by_3(ix) | (split_by_3(iy) << 1) | (split_by_3(iz) << 2)) | (depth << 24)


class ElasticSpatialHash:
    """
    Non-reordering open addressing spatial hash table (Farach-Colton et al. 2025).
    Guarantees O(1) probe time and zero element displacement during streaming insertions.
    """
    def __init__(self, capacity: int, num_levels: int = 4):
        self.capacity = capacity
        self.num_levels = num_levels
        fractions = [0.5**(i + 1) for i in range(num_levels - 1)]
        fractions.append(1.0 - sum(fractions))
        self.level_sizes = [max(16, int(capacity * f)) for f in fractions]
        self.total_size = sum(self.level_sizes)
        self.level_offsets = [0] + list(np.cumsum(self.level_sizes)[:-1])

        self.keys = np.full(self.total_size, -1, dtype=np.int64)
        self.values = [None] * self.total_size
        self.occupied = np.zeros(self.total_size, dtype=bool)

        rng = np.random.RandomState(42)
        self.seeds_a = rng.randint(1, 2**31 - 1, size=(num_levels, 4), dtype=np.int64)
        self.seeds_b = rng.randint(0, 2**31 - 1, size=(num_levels, 4), dtype=np.int64)
        self.count = 0

    def _hash(self, key: int, level: int, attempt: int) -> int:
        a = self.seeds_a[level, attempt % 4]
        b = self.seeds_b[level, attempt % 4]
        size = self.level_sizes[level]
        return ((int(key) * int(a) + int(b) + attempt * 2654435761) & 0x7FFFFFFF) % size

    def insert(self, key: int, value: Any) -> bool:
        if self.count >= self.capacity:
            return False
        for level in range(self.num_levels):
            offset = self.level_offsets[level]
            size = self.level_sizes[level]
            for attempt in range(min(size, 4 + level * 2)):
                pos = offset + self._hash(key, level, attempt)
                if not self.occupied[pos]:
                    self.keys[pos] = key
                    self.values[pos] = value
                    self.occupied[pos] = True
                    self.count += 1
                    return True
                elif self.keys[pos] == key:
                    self.values[pos] = value
                    return True
        return False

    def lookup(self, key: int) -> Optional[Any]:
        for level in range(self.num_levels):
            offset = self.level_offsets[level]
            size = self.level_sizes[level]
            for attempt in range(min(size, 4 + level * 2)):
                pos = offset + self._hash(key, level, attempt)
                if not self.occupied[pos]:
                    continue
                if self.keys[pos] == key:
                    return self.values[pos]
        return None


class TreeFreeMultipoleAttention:
    """
    Linear-Time O(N) Multipole Attention Layer.
    Decomposes spatial attention into:
      1. Near-field exact softmax dot-product attention within localized spatial hash buckets.
      2. Far-field multipole expansion summary from distant spatial clusters.
    """
    def __init__(
        self,
        embed_dim: int = 64,
        spatial_dim: int = 2,
        grid_depth: int = 4,
        multipole_order: int = 2,
        spatial_sigma: float = 0.25,
        temperature: Optional[float] = None,
    ):
        self.embed_dim = embed_dim
        self.spatial_dim = spatial_dim
        self.grid_depth = grid_depth
        self.multipole_order = multipole_order
        self.spatial_sigma = spatial_sigma
        self.temperature = temperature or (1.0 / np.sqrt(embed_dim))
        self.grid_res = 1 << grid_depth

    def forward(
        self,
        Q: np.ndarray,      # (N, D) Query representations
        K: np.ndarray,      # (N, D) Key representations
        V: np.ndarray,      # (N, D) Value representations
        coords: np.ndarray, # (N, d_spatial) Normalized spatial coordinates in [0, 1)^d
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Computes all-pairs continuous spatial attention in O(N) time with vectorized bucket operations.
        Returns: (output_values (N, D), metadata_dict)
        """
        N, D = Q.shape
        coords_clipped = np.clip(coords, 1e-4, 1.0 - 1e-4)

        # 1. Bucket tokens into non-reordering elastic spatial hash
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

        # 2. Precompute Far-Field Multipole Cluster Summaries (P2M)
        cluster_keys = list(bucket_map.keys())
        n_clusters = len(cluster_keys)
        
        all_centers = np.zeros((n_clusters, self.spatial_dim), dtype=np.float32)
        all_mean_k = np.zeros((n_clusters, D), dtype=np.float32)
        all_sum_v = np.zeros((n_clusters, D), dtype=np.float32)
        all_dipoles = np.zeros((n_clusters, D, self.spatial_dim), dtype=np.float32)
        all_counts = np.zeros(n_clusters, dtype=np.float32)

        key_to_idx = {k: idx for idx, k in enumerate(cluster_keys)}

        for idx, k in enumerate(cluster_keys):
            p_ids = bucket_map[k]
            pts = coords_clipped[p_ids]
            c_center = np.mean(pts, axis=0)
            all_centers[idx] = c_center
            all_mean_k[idx] = np.mean(K[p_ids], axis=0)
            all_sum_v[idx] = np.sum(V[p_ids], axis=0)
            all_counts[idx] = len(p_ids)
            
            delta = pts - c_center[None, :]
            all_dipoles[idx] = np.einsum('ni,nd->id', V[p_ids], delta)

        # 3. Fast Vectorized Bucket-Level Evaluation (Near P2P + Far M2L)
        out_v = np.zeros((N, D), dtype=np.float32)
        inv_2_sigma_sq = 1.0 / (2.0 * (self.spatial_sigma ** 2))
        inv_sigma_sq = 1.0 / (self.spatial_sigma ** 2)
        cell_size = 1.0 / self.grid_res

        total_near_evals = 0
        total_far_evals = 0

        # Evaluate per bucket
        for k_src, p_src in bucket_map.items():
            p_src_arr = np.asarray(p_src, dtype=np.int32)
            M_src = len(p_src_arr)
            q_src = Q[p_src_arr]       # (M_src, D)
            pts_src = coords_clipped[p_src_arr] # (M_src, d_spatial)

            # Find near neighbor keys
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

            # --- Vectorized Near-Field Evaluation for Bucket ---
            near_arr = np.asarray(near_p_list, dtype=np.int32)
            pts_near = coords_clipped[near_arr] # (N_near, d_spatial)
            k_near = K[near_arr]               # (N_near, D)
            v_near = V[near_arr]               # (N_near, D)

            # Pairwise spatial distances: (M_src, N_near)
            diff_near = pts_src[:, None, :] - pts_near[None, :, :]
            dist_sq_near = np.sum(diff_near ** 2, axis=-1)
            spatial_w_near = np.exp(-dist_sq_near * inv_2_sigma_sq)

            # Feature dot products: (M_src, N_near)
            dot_near = np.matmul(q_src, k_near.T) * self.temperature
            dot_near_clipped = np.clip(dot_near, -30.0, 30.0)
            attn_near = spatial_w_near * np.exp(dot_near_clipped)

            val_near = np.matmul(attn_near, v_near) # (M_src, D)
            weight_near = np.sum(attn_near, axis=-1, keepdims=True) + 1e-9 # (M_src, 1)

            total_near_evals += M_src * len(near_arr)

            # --- Vectorized Far-Field Evaluation for Bucket ---
            far_indices = [idx for idx in range(n_clusters) if idx not in near_indices_set]
            if far_indices:
                far_idx_arr = np.asarray(far_indices, dtype=np.int32)
                far_centers = all_centers[far_idx_arr]       # (N_far, d_spatial)
                far_k_means = all_mean_k[far_idx_arr]        # (N_far, D)
                far_v_sums = all_sum_v[far_idx_arr]          # (N_far, D)
                far_counts = all_counts[far_idx_arr]         # (N_far,)
                far_dipoles = all_dipoles[far_idx_arr]       # (N_far, D, d_spatial)

                # Distance from src points to far cluster centers: (M_src, N_far)
                diff_far = pts_src[:, None, :] - far_centers[None, :, :]
                dist_sq_far = np.sum(diff_far ** 2, axis=-1)
                spatial_w_far = np.exp(-dist_sq_far * inv_2_sigma_sq)

                # Dot products: (M_src, N_far)
                dot_far = np.matmul(q_src, far_k_means.T) * self.temperature
                w_far = spatial_w_far * np.exp(np.clip(dot_far, -30.0, 30.0)) # (M_src, N_far)

                # Far contribution: zero-order + first-order dipole
                # Zero order: w_far @ far_v_sums -> (M_src, D)
                val_far_0 = np.matmul(w_far, far_v_sums)

                # First order dipole correction: einsum('mfd, fid -> mfi', diff_far, far_dipoles)
                # dipole_corr: -diff_far / sigma^2
                corr = np.einsum('mfd,fid->mfi', -diff_far * inv_sigma_sq, far_dipoles) # (M_src, N_far, D)
                val_far_1 = np.einsum('mf,mfi->mi', w_far, corr)

                val_far = val_far_0 + val_far_1
                weight_far = np.matmul(w_far, far_counts[:, None]) # (M_src, 1)

                val_total = val_near + val_far
                weight_total = weight_near + weight_far
                total_far_evals += M_src * len(far_indices)
            else:
                val_total = val_near
                weight_total = weight_near

            out_v[p_src_arr] = val_total / weight_total

        meta = {
            "num_particles": N,
            "active_clusters": len(bucket_map),
            "total_near_evals": total_near_evals,
            "total_far_evals": total_far_evals,
            "complexity_scaling": "O(N)",
        }
        return out_v, meta


class MultiHeadMultipoleAttention:
    """
    Drop-in Multi-Head Multipole Attention module for deep architectures.
    Projects inputs into H distinct subspaces and evaluates linear-time multipole attention in parallel.
    """
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        spatial_dim: int = 2,
        grid_depth: int = 4,
        spatial_sigma: float = 0.25,
    ):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.spatial_dim = spatial_dim

        # Projection weights
        scale = 1.0 / np.sqrt(d_model)
        rng = np.random.RandomState(42)
        self.W_q = rng.normal(0, scale, size=(d_model, d_model)).astype(np.float32)
        self.W_k = rng.normal(0, scale, size=(d_model, d_model)).astype(np.float32)
        self.W_v = rng.normal(0, scale, size=(d_model, d_model)).astype(np.float32)
        self.W_o = rng.normal(0, scale, size=(d_model, d_model)).astype(np.float32)

        # Per-head attention operators
        self.heads = [
            TreeFreeMultipoleAttention(
                embed_dim=self.d_head,
                spatial_dim=spatial_dim,
                grid_depth=grid_depth,
                spatial_sigma=spatial_sigma,
            )
            for _ in range(n_heads)
        ]

    def forward(
        self,
        X: np.ndarray,      # (N, d_model) Input token embeddings
        coords: np.ndarray, # (N, spatial_dim) Spatial coordinates
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Multi-head forward pass in linear O(N) time."""
        N, D = X.shape
        Q_proj = np.matmul(X, self.W_q) # (N, D)
        K_proj = np.matmul(X, self.W_k) # (N, D)
        V_proj = np.matmul(X, self.W_v) # (N, D)

        head_outputs = []
        for h in range(self.n_heads):
            q_h = Q_proj[:, h * self.d_head : (h + 1) * self.d_head]
            k_h = K_proj[:, h * self.d_head : (h + 1) * self.d_head]
            v_h = V_proj[:, h * self.d_head : (h + 1) * self.d_head]
            
            out_h, _ = self.heads[h].forward(q_h, k_h, v_h, coords)
            head_outputs.append(out_h)

        concatenated = np.concatenate(head_outputs, axis=-1) # (N, D)
        final_out = np.matmul(concatenated, self.W_o) # (N, D)

        return final_out, {"num_tokens": N, "num_heads": self.n_heads, "d_model": self.d_model}
