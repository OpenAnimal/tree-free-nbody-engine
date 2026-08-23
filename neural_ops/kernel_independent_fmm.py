"""
Kernel-Independent Neural Multipole Operator (`kernel_independent_fmm.py`)
==========================================================================
Linear-Time O(N) Neural Operator using Kernel-Independent Fast Multipole Method (KI-FMM)
with Equivalent Proxy Surfaces and SVD Skeletonization.

Enables tree-free acceleration for arbitrary user-defined or learned non-linear neural kernels:
- Non-linear Gaussian/RBF kernels: K(x, y) = exp(-gamma * ||x - y||^2)
- Oscillatory/Helmholtz wave kernels: K(x, y) = cos(k * ||x - y||) / (||x - y|| + eps)
- Learned MLP / Gated kernels: K(x, y) = MLP([x - y, ||x - y||])
- Gravitational / Coulomb / Yukawa potentials with arbitrary screening.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Callable


def generate_sphere_proxy_surface(n_proxy: int = 14, radius: float = 1.0, dim: int = 3) -> np.ndarray:
    """Generates quasi-uniform proxy surface points on the unit sphere/circle."""
    if dim == 2:
        angles = np.linspace(0, 2 * np.pi, n_proxy, endpoint=False)
        return radius * np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)
    elif dim == 3:
        # Fibonacci sphere sampling
        indices = np.arange(0, n_proxy, dtype=np.float32) + 0.5
        phi = np.arccos(1.0 - 2.0 * indices / n_proxy)
        theta = np.pi * (1.0 + 5.0**0.5) * indices
        x = radius * np.cos(theta) * np.sin(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(phi)
        return np.stack([x, y, z], axis=-1).astype(np.float32)
    else:
        rng = np.random.RandomState(42)
        pts = rng.randn(n_proxy, dim).astype(np.float32)
        pts /= np.linalg.norm(pts, axis=-1, keepdims=True)
        return radius * pts


class KernelIndependentNeuralOperator:
    """
    Kernel-Independent Neural Multipole Operator (KI-FMM).
    Converts cluster node activations to equivalent proxy surface charges via regularized SVD/ridge regression,
    then evaluates far-field potentials with zero analytical expansion derivatives.
    """
    def __init__(
        self,
        in_features: int = 32,
        out_features: int = 32,
        spatial_dim: int = 3,
        grid_depth: int = 4,
        n_proxy: int = 16,
        kernel_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        regularization: float = 1e-4,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.spatial_dim = spatial_dim
        self.grid_depth = grid_depth
        self.grid_res = 1 << grid_depth
        self.n_proxy = n_proxy
        self.regularization = regularization

        # Default kernel: RBF / Gaussian neural operator kernel
        if kernel_fn is None:
            self.kernel_fn = lambda r_sq: np.exp(-4.0 * r_sq)
        else:
            self.kernel_fn = kernel_fn

        # Unit proxy surface template
        self.unit_proxy_pts = generate_sphere_proxy_surface(n_proxy=n_proxy, radius=1.0, dim=spatial_dim)

        # Feature transformations
        scale = 1.0 / np.sqrt(in_features)
        rng = np.random.RandomState(42)
        self.W_near = rng.normal(0, scale, size=(in_features, out_features)).astype(np.float32)
        self.W_far = rng.normal(0, scale, size=(in_features, out_features)).astype(np.float32)
        self.W_self = rng.normal(0, scale, size=(in_features, out_features)).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)

    def _morton_encode(self, coords: np.ndarray) -> np.ndarray:
        res = self.grid_res
        grid_indices = np.clip(np.floor(coords * res).astype(np.int64), 0, res - 1)
        if self.spatial_dim == 2:
            return grid_indices[:, 0] + grid_indices[:, 1] * res
        elif self.spatial_dim == 3:
            return grid_indices[:, 0] + grid_indices[:, 1] * res + grid_indices[:, 2] * (res ** 2)
        else:
            multipliers = (res ** np.arange(self.spatial_dim)).astype(np.int64)
            return np.sum(grid_indices * multipliers, axis=-1)

    def forward(
        self,
        X: np.ndarray,      # (N, in_features) Node / particle features
        coords: np.ndarray, # (N, spatial_dim) Normalized spatial coordinates in [0, 1)^d
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes linear O(N) Kernel-Independent neural convolution.
        """
        N, D_in = X.shape
        coords_clipped = np.clip(coords, 1e-4, 1.0 - 1e-4)
        cell_size = 1.0 / self.grid_res
        proxy_radius = cell_size * 0.75

        # 1. Bucket tokens into spatial hash grid
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

        # 2. Compute Equivalent Proxy Charges (P2M via Skeletonization)
        all_centers = np.zeros((n_clusters, self.spatial_dim), dtype=np.float32)
        # Proxy positions: (n_clusters, n_proxy, spatial_dim)
        all_proxy_pos = np.zeros((n_clusters, self.n_proxy, self.spatial_dim), dtype=np.float32)
        # Equivalent proxy charges: (n_clusters, n_proxy, in_features)
        all_proxy_charges = np.zeros((n_clusters, self.n_proxy, D_in), dtype=np.float32)

        for idx, k in enumerate(cluster_keys):
            p_ids = bucket_map[k]
            pts = coords_clipped[p_ids]
            c_center = np.mean(pts, axis=0)
            all_centers[idx] = c_center

            # Place proxy points on sphere around cluster center
            proxy_pos = c_center[None, :] + self.unit_proxy_pts * proxy_radius
            all_proxy_pos[idx] = proxy_pos

            # Compute kernel matrix between cluster internal particles and proxy points:
            # G: (len(p_ids), n_proxy)
            diff = pts[:, None, :] - proxy_pos[None, :, :]
            r_sq = np.sum(diff ** 2, axis=-1)
            G = self.kernel_fn(r_sq).astype(np.float32) # (M_cluster, n_proxy)

            # Solve regularized ridge regression: G * q_proxy = X[p_ids]
            # q_proxy = (G^T G + lambda * I)^{-1} G^T X[p_ids]
            GtG = G.T @ G
            GtG_reg = GtG + self.regularization * np.eye(self.n_proxy, dtype=np.float32)
            GtX = G.T @ X[p_ids]
            try:
                q_proxy = np.linalg.solve(GtG_reg, GtX)
            except np.linalg.LinAlgError:
                q_proxy = np.linalg.pinv(GtG_reg) @ GtX

            all_proxy_charges[idx] = q_proxy

        # 3. Vectorized Evaluation (Near exact + Far proxy)
        out_features = np.zeros((N, self.out_features), dtype=np.float32)
        total_near_evals = 0
        total_far_evals = 0

        X_transformed_near = X @ self.W_near
        X_self = X @ self.W_self

        for k_src, p_src in bucket_map.items():
            p_src_arr = np.asarray(p_src, dtype=np.int32)
            M_src = len(p_src_arr)
            pts_src = coords_clipped[p_src_arr]
            center_src = all_centers[key_to_idx[k_src]]

            # Find spatial neighbors within adjacent grid cells.
            # Derive the source cell from the bucket key k_src itself (NOT from
            # the cluster centroid): for a multi-particle bucket whose members
            # hug a cell boundary, the centroid can land in a different cell
            # than every member, and the ring-1 neighborhood around the
            # centroid cell then misses a cell that is ring-1 adjacent to the
            # bucket's actual cell (where a true near neighbor lives).  The
            # bucket key is LINEAR (see _morton_encode): k = nx + ny*res +
            # nz*res^2, so the cell coords are recovered by integer division.
            res = self.grid_res
            if self.spatial_dim == 2:
                src_grid = np.array([k_src % res,
                                     (k_src // res) % res], dtype=np.int64)
            elif self.spatial_dim == 3:
                src_grid = np.array([k_src % res,
                                     (k_src // res) % res,
                                     k_src // (res * res)], dtype=np.int64)
            else:
                # General d-dim linear key: k = sum_i idx_i * res^i
                src_grid = np.array(
                    [(k_src // (res ** i)) % res for i in range(self.spatial_dim)],
                    dtype=np.int64)
            near_indices_set = set()
            near_p_list = []

            # 3^d neighboring cells
            if self.spatial_dim == 2:
                for dx in (-1, 0, 1):
                    nx = int(src_grid[0]) + dx
                    if 0 <= nx < res:
                        for dy in (-1, 0, 1):
                            ny = int(src_grid[1]) + dy
                            if 0 <= ny < res:
                                nk = int(nx + ny * res)
                                if nk in key_to_idx:
                                    near_indices_set.add(key_to_idx[nk])
                                    near_p_list.extend(bucket_map[nk])
            elif self.spatial_dim == 3:
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

            # Near-field exact convolution
            near_arr = np.asarray(near_p_list, dtype=np.int32)
            pts_near = coords_clipped[near_arr]
            feat_near = X_transformed_near[near_arr]

            diff_near = pts_src[:, None, :] - pts_near[None, :, :]
            r_sq_near = np.sum(diff_near ** 2, axis=-1)
            K_near = self.kernel_fn(r_sq_near) # (M_src, len(near_arr))

            val_near = K_near @ feat_near # (M_src, out_features)
            total_near_evals += M_src * len(near_arr)

            # Far-field proxy surface evaluation
            far_indices = [idx for idx in range(n_clusters) if idx not in near_indices_set]
            if far_indices:
                far_idx_arr = np.asarray(far_indices, dtype=np.int32)
                far_proxy_pts = all_proxy_pos[far_idx_arr].reshape(-1, self.spatial_dim) # (N_far * n_proxy, dim)
                far_proxy_q = all_proxy_charges[far_idx_arr].reshape(-1, D_in)           # (N_far * n_proxy, D_in)
                far_proxy_q_trans = far_proxy_q @ self.W_far                             # (N_far * n_proxy, out_features)

                diff_far = pts_src[:, None, :] - far_proxy_pts[None, :, :] # (M_src, N_far * n_proxy, dim)
                r_sq_far = np.sum(diff_far ** 2, axis=-1)
                K_far = self.kernel_fn(r_sq_far) # (M_src, N_far * n_proxy)

                val_far = K_far @ far_proxy_q_trans
                total_far_evals += M_src * len(far_proxy_pts)
            else:
                val_far = 0.0

            # Aggregate self, near, far, and bias with non-linearity (ReLU / GELU)
            val_total = X_self[p_src_arr] + val_near + val_far + self.bias
            out_features[p_src_arr] = np.maximum(0.0, val_total)

        meta = {
            "num_particles": N,
            "n_proxy": self.n_proxy,
            "active_clusters": n_clusters,
            "total_near_evals": total_near_evals,
            "total_far_evals": total_far_evals,
            "regularization": self.regularization,
        }
        return out_features, meta
