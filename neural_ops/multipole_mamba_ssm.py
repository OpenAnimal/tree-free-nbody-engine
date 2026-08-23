"""
Multipole Spatial State Space Model (`multipole_mamba_ssm.py`)
=============================================================
Linear-Time O(N) Multi-Dimensional Spatial-Temporal Neural Operator.
Marries 1D Selective State Space Models (Mamba / S4 selective scan) with
Tree-Free Multipole Spatial-Temporal Mixing.

Key Features:
- Preserves continuous multi-dimensional geometric locality without rasterization degradation.
- Combines 1D selective recurrent memory with isotropic O(N) multipole far-field aggregation.
- Sub-quadratic memory and constant step inference latency.
"""

import numpy as np

try:
    from neural_ops._coord_contract import check_unit_coords
except ImportError:  # direct script execution (repo root not yet on sys.path)
    from _coord_contract import check_unit_coords
from typing import Optional, Tuple, Dict, Any, List


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU (Swish) activation function."""
    return x / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class SelectiveScan1D:
    """
    1D Selective State Space Scan (Mamba-style continuous discretization).
    h_t = exp(Delta_t * A) h_{t-1} + Delta_t * B_t * x_t
    y_t = C_t * h_t + D * x_t
    """
    def __init__(self, d_model: int = 64, d_state: int = 16):
        self.d_model = d_model
        self.d_state = d_state

        rng = np.random.RandomState(42)
        # Initialize diagonal state decay A (negative real values)
        self.A_log = np.log(np.repeat(np.arange(1, d_state + 1, dtype=np.float32)[None, :], d_model, axis=0))
        self.A = -np.exp(self.A_log) # (d_model, d_state)

        scale = 1.0 / np.sqrt(d_model)
        self.W_delta = rng.normal(0, scale, size=(d_model, d_model)).astype(np.float32)
        self.b_delta = np.zeros(d_model, dtype=np.float32)

        self.W_B = rng.normal(0, scale, size=(d_model, d_state)).astype(np.float32)
        self.W_C = rng.normal(0, scale, size=(d_model, d_state)).astype(np.float32)
        self.D = np.ones(d_model, dtype=np.float32)

    def forward(self, u: np.ndarray) -> np.ndarray:
        """
        Executes selective scan along sequence dimension.
        Args:
            u: (N, d_model) input sequence
        Returns:
            y: (N, d_model) scan output
        """
        N, D = u.shape
        # Input-dependent parameters
        delta = np.log(1.0 + np.exp(u @ self.W_delta + self.b_delta)) # (N, d_model)
        B = u @ self.W_B # (N, d_state)
        C = u @ self.W_C # (N, d_state)

        # Discretize continuous state space
        # dA = exp(delta * A) -> (N, d_model, d_state)
        # dB = delta * B      -> (N, d_model, d_state)
        dA = np.exp(delta[:, :, None] * self.A[None, :, :])
        dB = delta[:, :, None] * B[:, None, :] # (N, d_model, d_state)

        # Sequential associative recurrence: (N, d_model)
        y = np.zeros((N, D), dtype=np.float32)
        h = np.zeros((D, self.d_state), dtype=np.float32)

        for t in range(N):
            x_t = u[t, :, None] # (D, 1)
            h = dA[t] * h + dB[t] * x_t # (D, d_state)
            # y_t = sum(C_t * h_t) + D * u_t
            y[t] = np.sum(h * C[t, None, :], axis=-1) + self.D * u[t]

        return y


class MultipoleSpatialSSM:
    """
    Continuous Multi-Dimensional Multipole Spatial State Space Model.
    Fuses 1D selective scan with O(N) Far-field Spatial Multipole Mixing.
    """
    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        spatial_dim: int = 3,
        grid_depth: int = 4,
        spatial_sigma: float = 0.25,
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.spatial_dim = spatial_dim
        self.grid_depth = grid_depth
        self.grid_res = 1 << grid_depth
        self.spatial_sigma = spatial_sigma

        # 1D Selective Scan Module
        self.ssm_1d = SelectiveScan1D(d_model=d_model, d_state=d_state)

        # Multi-dimensional projections & gating
        scale = 1.0 / np.sqrt(d_model)
        rng = np.random.RandomState(42)
        self.W_in = rng.normal(0, scale, size=(d_model, 2 * d_model)).astype(np.float32)
        self.W_multipole = rng.normal(0, scale, size=(d_model, d_model)).astype(np.float32)
        self.W_out = rng.normal(0, scale, size=(d_model, d_model)).astype(np.float32)

    def _morton_order(self, coords: np.ndarray) -> np.ndarray:
        res = self.grid_res
        grid_indices = np.clip(np.floor(coords * res).astype(np.int64), 0, res - 1)
        if self.spatial_dim == 2:
            return grid_indices[:, 0] + grid_indices[:, 1] * res
        else:
            return grid_indices[:, 0] + grid_indices[:, 1] * res + grid_indices[:, 2] * (res ** 2)

    def forward(
        self,
        X: np.ndarray,      # (N, d_model) Input representations
        coords: np.ndarray, # (N, spatial_dim) Multi-dimensional spatial coordinates in [0, 1)^d
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Computes forward pass of Multipole Spatial SSM in O(N) time.
        """
        check_unit_coords(coords, "MultipoleSpatialSSM.forward(coords)")
        N, D = X.shape
        coords_clipped = np.clip(coords, 1e-4, 1.0 - 1e-4)

        # 1. Sort along Space-Filling Morton Curve for 1D Sequential Locality
        morton_keys = self._morton_order(coords_clipped)
        sort_perm = np.argsort(morton_keys)
        inv_sort_perm = np.argsort(sort_perm)

        X_sorted = X[sort_perm]
        coords_sorted = coords_clipped[sort_perm]

        # 2. Input Projection & Gated Branching
        X_proj = X_sorted @ self.W_in # (N, 2 * d_model)
        u_ssm = X_proj[:, :D]
        u_gate = X_proj[:, D:]

        # 3. 1D Selective Scan Recurrence
        ssm_out = self.ssm_1d.forward(u_ssm) # (N, D)
        ssm_activated = ssm_out * silu(u_gate)

        # 4. Multi-Dimensional Tree-Free Multipole Far-Field Mixing
        # Bucket sorted tokens into spatial hash grid
        bucket_map: Dict[int, List[int]] = {}
        for i in range(N):
            k = int(morton_keys[sort_perm[i]])
            if k not in bucket_map:
                bucket_map[k] = []
            bucket_map[k].append(i)

        cluster_keys = list(bucket_map.keys())
        n_clusters = len(cluster_keys)
        key_to_idx = {k: idx for idx, k in enumerate(cluster_keys)}

        all_centers = np.zeros((n_clusters, self.spatial_dim), dtype=np.float32)
        all_means = np.zeros((n_clusters, D), dtype=np.float32)
        all_counts = np.zeros(n_clusters, dtype=np.float32)

        feat_mixed = ssm_activated @ self.W_multipole

        for idx, k in enumerate(cluster_keys):
            p_ids = bucket_map[k]
            pts = coords_sorted[p_ids]
            c_center = np.mean(pts, axis=0)
            all_centers[idx] = c_center
            all_means[idx] = np.mean(feat_mixed[p_ids], axis=0)
            all_counts[idx] = len(p_ids)
            # NOTE: dipoles were computed here but never used in the far-field
            # evaluation (the far pass uses only centers and means). Removed
            # as dead code.

        # Vectorized multipole interaction
        multipole_out = np.zeros((N, D), dtype=np.float32)
        inv_2_sigma_sq = 1.0 / (2.0 * (self.spatial_sigma ** 2))

        for k_src, p_src in bucket_map.items():
            p_src_arr = np.asarray(p_src, dtype=np.int32)
            M_src = len(p_src_arr)
            pts_src = coords_sorted[p_src_arr]
            c_idx = key_to_idx[k_src]

            # Far-field clusters
            far_indices = [idx for idx in range(n_clusters) if idx != c_idx]
            if far_indices:
                far_idx_arr = np.asarray(far_indices, dtype=np.int32)
                far_centers = all_centers[far_idx_arr]
                far_means = all_means[far_idx_arr]

                diff_far = pts_src[:, None, :] - far_centers[None, :, :]
                dist_sq_far = np.sum(diff_far ** 2, axis=-1)
                w_far = np.exp(-dist_sq_far * inv_2_sigma_sq) # (M_src, N_far)
                w_norm = w_far / (np.sum(w_far, axis=-1, keepdims=True) + 1e-9)

                val_far = w_norm @ far_means # (M_src, D)
                multipole_out[p_src_arr] = val_far

        # 5. Output Projection and Residual Fusion
        combined = ssm_activated + multipole_out
        out_sorted = combined @ self.W_out
        out_final = out_sorted[inv_sort_perm]

        meta = {
            "num_tokens": N,
            "d_model": self.d_model,
            "d_state": self.d_state,
            "num_clusters": n_clusters,
            "complexity": "O(N)",
        }
        return out_final, meta
