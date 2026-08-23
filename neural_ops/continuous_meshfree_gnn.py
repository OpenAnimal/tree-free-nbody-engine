"""
Continuous Meshfree GNN Layer (`continuous_meshfree_gnn.py`)
============================================================
Continuous Spatial Graph Convolution without Adjacency Matrices or Edge Lists.
Powered by Tree-Free Fast Multipole Method (FMM) & Farach-Colton, Krapivin, & Kuszmaul (2025) Spatial Hashing.

Enables continuous message passing across dynamic point clouds, fluid particles,
and astrophysical / molecular simulations in O(N) linear time:
h_i^(l+1) = sigma( W_self h_i + sum_{near} K_near(x_i, x_j) W_near h_j + sum_{far} K_far(x_i, c_k) W_far M_k + b )
"""

import numpy as np

try:
    from neural_ops._coord_contract import check_unit_coords
except ImportError:  # direct script execution (repo root not yet on sys.path)
    from _coord_contract import check_unit_coords
from typing import Optional, Tuple, Dict, Any, List

try:
    from ._bucketing import build_cell_index
except (ImportError, ValueError):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from neural_ops._bucketing import build_cell_index

# Tri-backend acceleration (NumPy reference / torch / jax). See
# `neural_ops/_accel.py`. Bucketing stays CPU-only (funnel hash); only the
# per-bucket dense math (spatial kernel + matmul + sum) moves to the device.
try:
    from ._accel import (
        resolve_backend as _resolve_backend,
        get_ns as _get_ns,
        to_backend as _to_backend,
        as_numpy as _as_numpy,
        get_compiled as _get_compiled,
        HAS_TORCH as _HAS_TORCH,
        HAS_JAX as _HAS_JAX,
    )
except ImportError:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
    try:
        from neural_ops._accel import (
            resolve_backend as _resolve_backend,
            get_ns as _get_ns,
            to_backend as _to_backend,
            as_numpy as _as_numpy,
            get_compiled as _get_compiled,
            HAS_TORCH as _HAS_TORCH,
            HAS_JAX as _HAS_JAX,
        )
    except ImportError:
        _resolve_backend = None
        _get_ns = None
        _to_backend = None
        _as_numpy = None
        _get_compiled = None
        _HAS_TORCH = False
        _HAS_JAX = False

if _HAS_TORCH:
    import torch
if _HAS_JAX:
    import jax.numpy as jnp


def _make_spatial_kernel(kernel_type, cutoff_radius, exp, sqrt, clip, maximum):
    """Backend-bound continuous spatial kernel: dist_sq -> weight."""
    if kernel_type == "rbf":
        sigma_sq = (cutoff_radius / 2.0) ** 2
        denom = 2.0 * sigma_sq + 1e-8
        def kern(dist_sq):
            return exp(-dist_sq / denom)
        return kern
    elif kernel_type == "wendland":
        def kern(dist_sq):
            r = sqrt(dist_sq)
            q = clip(r / cutoff_radius, 0.0, 1.0)
            return ((1.0 - q) ** 4) * (4.0 * q + 1.0)
        return kern
    else:  # inverse
        def kern(dist_sq):
            return 1.0 / (1.0 + dist_sq / (cutoff_radius ** 2))
        return kern


def _make_gnn_near_kernel(sk_fn, matmul, sumop):
    """Branch-free per-bucket near-field message kernel (ops closure-bound)."""
    def kern(pts_src, pts_near, h_near):
        diff_near = pts_src[:, None, :] - pts_near[None, :, :]
        dist_sq_near = sumop(diff_near * diff_near, axis=-1)
        k_near = sk_fn(dist_sq_near)
        msg_near = matmul(k_near, h_near)
        weight_near = sumop(k_near, axis=-1, keepdims=True) + 1e-8
        return msg_near, weight_near
    return kern


def _make_gnn_far_kernel(sk_fn, matmul, sumop):
    """Branch-free per-bucket far-field cluster message kernel."""
    def kern(pts_src, far_centers, far_moments, far_counts):
        diff_far = pts_src[:, None, :] - far_centers[None, :, :]
        dist_sq_far = sumop(diff_far * diff_far, axis=-1)
        k_far = sk_fn(dist_sq_far)
        weighted_moments = k_far * far_counts[None, :]
        msg_far = matmul(weighted_moments, far_moments)
        weight_far = sumop(weighted_moments, axis=-1, keepdims=True) + 1e-8
        return msg_far, weight_far
    return kern


def _sum_np(x, axis=-1, keepdims=False):
    return x.sum(axis=axis, keepdims=keepdims)


_GNN_KERNELS = None


def _gnn_kernels(kernel_type, cutoff_radius):
    """Build per-backend (spatial_kernel, near_kernel, far_kernel) triples."""
    global _GNN_KERNELS
    if _GNN_KERNELS is None:
        _GNN_KERNELS = {}
    key = (kernel_type, cutoff_radius)
    cached = _GNN_KERNELS.get(key)
    if cached is not None:
        return cached
    def _torch_sum(x, axis=-1, keepdims=False):
        return x.sum(dim=axis, keepdim=keepdims)
    def _jax_sum(x, axis=-1, keepdims=False):
        return x.sum(axis=axis, keepdims=keepdims)
    triples = {}
    sk_np = _make_spatial_kernel(kernel_type, cutoff_radius, np.exp, np.sqrt, np.clip, np.maximum)
    triples["numpy"] = (
        sk_np,
        _make_gnn_near_kernel(sk_np, np.matmul, _sum_np),
        _make_gnn_far_kernel(sk_np, np.matmul, _sum_np),
    )
    if _HAS_TORCH:
        sk_t = _make_spatial_kernel(kernel_type, cutoff_radius, torch.exp, torch.sqrt, torch.clamp, torch.maximum)
        triples["torch"] = (
            sk_t,
            _make_gnn_near_kernel(sk_t, torch.matmul, _torch_sum),
            _make_gnn_far_kernel(sk_t, torch.matmul, _torch_sum),
        )
    if _HAS_JAX:
        sk_j = _make_spatial_kernel(kernel_type, cutoff_radius, jnp.exp, jnp.sqrt, jnp.clip, jnp.maximum)
        triples["jax"] = (
            sk_j,
            _make_gnn_near_kernel(sk_j, jnp.matmul, _jax_sum),
            _make_gnn_far_kernel(sk_j, jnp.matmul, _jax_sum),
        )
    _GNN_KERNELS[key] = triples
    return triples


class ContinuousMeshfreeGNNLayer:
    """
    Mesh-Free Continuous Graph Convolution Layer.
    Executes all-pairs spatial message passing in linear O(N) time without constructing edge lists.

    Shapes / dtypes
    ---------------
    node_features : float32 (N, in_features)
    coords        : float32 (N, spatial_dim), NORMALIZED to [0, 1)^d
        (out-of-range values are clipped and trigger a RuntimeWarning)
    returns       : (updated float32 (N, out_features), meta dict)

    Example
    -------
    >>> import numpy as np
    >>> from neural_ops.continuous_meshfree_gnn import ContinuousMeshfreeGNNLayer
    >>> rng = np.random.default_rng(0)
    >>> feats = rng.standard_normal((256, 32)).astype(np.float32)
    >>> coords = rng.random((256, 3)).astype(np.float32)
    >>> layer = ContinuousMeshfreeGNNLayer(in_features=32, out_features=32, spatial_dim=3)
    >>> out, meta = layer.forward(feats, coords)
    """
    def __init__(
        self,
        in_features: int = 32,
        out_features: int = 32,
        spatial_dim: int = 3,
        grid_depth: int = 4,
        cutoff_radius: float = 0.15,
        kernel_type: str = "rbf",
        backend: str = "numpy",
        jit: bool = False,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.spatial_dim = spatial_dim
        self.grid_depth = grid_depth
        self.cutoff_radius = float(cutoff_radius)  # prevent numpy.float64 promotion
        self.kernel_type = kernel_type
        self.grid_res = 1 << grid_depth
        # Acceleration backend: "numpy" (reference) | "torch" | "jax".
        self.backend = _resolve_backend(backend) if _resolve_backend is not None else "numpy"
        self.jit = bool(jit)

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

        The `backend` selects where the per-bucket dense math runs (numpy ref /
        torch / jax); spatial bucketing is always CPU (funnel hash).
        """
        check_unit_coords(coords, "ContinuousMeshfreeGNNLayer.forward(coords)")
        if self.backend != "numpy":
            return self._forward_accel(node_features, coords)
        N, F_in = node_features.shape
        coords_clipped = np.clip(coords, 1e-4, 1.0 - 1e-4).astype(np.float64)

        # 1. Bucket nodes via the funnel-hash-backed CellIndex
        idx, unique_keys, inverse = build_cell_index(
            coords_clipped, self.spatial_dim, self.grid_res
        )

        # 2. Linear feature projections
        h_self = np.matmul(node_features, self.W_self) # (N, out_features)
        h_near_proj = np.matmul(node_features, self.W_near) # (N, out_features)
        h_far_proj = np.matmul(node_features, self.W_far) # (N, out_features)

        # 3. Multipole cluster moments (P2M)
        occupied = idx.occupied_keys()
        n_clusters = len(occupied)
        key_to_idx = {int(k): c for c, k in enumerate(occupied)}

        all_centers = np.zeros((n_clusters, self.spatial_dim), dtype=np.float32)
        all_moments = np.zeros((n_clusters, self.out_features), dtype=np.float32)
        all_counts = np.zeros(n_clusters, dtype=np.float32)

        for k in occupied:
            c = key_to_idx[int(k)]
            p_ids = idx.bucket(k)
            pts = coords_clipped[p_ids]
            all_centers[c] = np.mean(pts, axis=0)
            all_moments[c] = np.mean(h_far_proj[p_ids], axis=0)
            all_counts[c] = len(p_ids)

        # 4. Message aggregation: Vectorized Bucket-Level Near P2P + Far M2L
        out_messages = np.zeros((N, self.out_features), dtype=np.float32)

        for k_src in occupied:
            p_src_arr = idx.bucket(k_src)
            M_src = len(p_src_arr)
            pts_src = coords_clipped[p_src_arr]

            # Find near neighbor keys via CellIndex (funnel-hash probes)
            near_keys = idx.neighbor_keys(k_src, ring=1)

            near_p_list = []
            near_indices_set = set()
            for nk in near_keys:
                p_n = idx.bucket(nk)
                if p_n is not None:
                    near_p_list.extend(p_n)
                    if int(nk) in key_to_idx:
                        near_indices_set.add(key_to_idx[int(nk)])

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
            far_indices = [c for c in range(n_clusters) if c not in near_indices_set]
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
            "active_clusters": len(idx),
            "spatial_dim": self.spatial_dim,
            "complexity": "O(N) Mesh-Free",
        }
        return updated_features, meta

    def _forward_accel(
        self,
        node_features: np.ndarray,
        coords: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """torch / jax backend path. Bucketing + P2M moments stay CPU; the
        feature projections, per-bucket near/far message math, and the final
        ReLU update run on the device backend via compiled branch-free
        kernels. Output is pulled back to a NumPy array."""
        ns = _get_ns(self.backend)
        N, F_in = node_features.shape
        F_out = self.out_features
        coords_clipped = np.clip(coords, 1e-4, 1.0 - 1e-4).astype(np.float64)

        # 1. CPU bucketing.
        idx, unique_keys, inverse = build_cell_index(
            coords_clipped, self.spatial_dim, self.grid_res
        )
        occupied = idx.occupied_keys()
        n_clusters = len(occupied)
        key_to_idx = {int(k): c for c, k in enumerate(occupied)}

        # 2. CPU P2M cluster moments (centroids + far-projected feature means).
        all_centers = np.zeros((n_clusters, self.spatial_dim), dtype=np.float32)
        all_moments = np.zeros((n_clusters, F_out), dtype=np.float32)
        all_counts = np.zeros(n_clusters, dtype=np.float32)
        # h_far_proj is needed for moments; compute on CPU here (small matmul)
        # so the moment assembly matches the numpy path exactly.
        h_far_proj_np = np.matmul(node_features, self.W_far)
        for k in occupied:
            c = key_to_idx[int(k)]
            p_ids = idx.bucket(k)
            all_centers[c] = np.mean(coords_clipped[p_ids], axis=0)
            all_moments[c] = np.mean(h_far_proj_np[p_ids], axis=0)
            all_counts[c] = len(p_ids)

        # Move full arrays + moments to the device once.
        coords_d = _to_backend(coords_clipped.astype(np.float32), self.backend, ns.float32)
        h_near_d = _to_backend(
            np.matmul(node_features, self.W_near).astype(np.float32), self.backend, ns.float32
        )
        centers_d = _to_backend(all_centers, self.backend, ns.float32)
        moments_d = _to_backend(all_moments, self.backend, ns.float32)
        counts_d = _to_backend(all_counts, self.backend, ns.float32)

        out_messages = ns.zeros((N, F_out), dtype=ns.float32)
        triples = _gnn_kernels(self.kernel_type, self.cutoff_radius)
        _, near_fn_eager, far_fn_eager = triples[self.backend]
        # Cache key MUST include cutoff_radius: the spatial kernel captures it
        # in its closure (denom = 2*sigma_sq+1e-8 where sigma_sq=(cutoff/2)^2).
        # Without this, two layers with the same kernel_type but different
        # cutoff_radius would share the wrong compiled kernel (88% error).
        # Use repr() not f"{:.6f}" to avoid truncation collisions.
        _ck = f"{self.kernel_type}_cr{repr(self.cutoff_radius)}"
        near_fn = _get_compiled(self.backend, f"gnn_near_{_ck}", near_fn_eager, jit=self.jit)
        far_fn = _get_compiled(self.backend, f"gnn_far_{_ck}", far_fn_eager, jit=self.jit)

        for k_src in occupied:
            p_src_arr = np.asarray(idx.bucket(k_src), dtype=np.int32)
            pts_src = ns.index(coords_d, p_src_arr)

            near_keys = idx.neighbor_keys(k_src, ring=1)
            near_p_list = []
            near_indices_set = set()
            for nk in near_keys:
                p_n = idx.bucket(nk)
                if p_n is not None:
                    near_p_list.extend(p_n)
                    if int(nk) in key_to_idx:
                        near_indices_set.add(key_to_idx[int(nk)])

            near_arr = np.asarray(near_p_list, dtype=np.int32)
            pts_near = ns.index(coords_d, near_arr)
            h_near = ns.index(h_near_d, near_arr)

            msg_near, weight_near = near_fn(pts_src, pts_near, h_near)

            far_indices = [c for c in range(n_clusters) if c not in near_indices_set]
            if far_indices:
                far_idx_arr = np.asarray(far_indices, dtype=np.int32)
                far_centers = ns.index(centers_d, far_idx_arr)
                far_moments = ns.index(moments_d, far_idx_arr)
                far_counts = ns.index(counts_d, far_idx_arr)
                msg_far, weight_far = far_fn(pts_src, far_centers, far_moments, far_counts)
                total_msg = (msg_near / weight_near) + (msg_far / weight_far)
            else:
                total_msg = msg_near / weight_near

            out_messages = ns.index_set(out_messages, p_src_arr, total_msg)

        # Final update on device: h_self + messages + bias -> ReLU.
        h_self_d = _to_backend(
            np.matmul(node_features, self.W_self).astype(np.float32), self.backend, ns.float32
        )
        bias_d = _to_backend(self.bias, self.backend, ns.float32)
        total_pre_act = h_self_d + out_messages + bias_d
        updated = ns.maximum(total_pre_act, ns.zeros((1,), dtype=ns.float32))
        updated_np = _as_numpy(updated).astype(np.float32)

        meta = {
            "num_nodes": N,
            "active_clusters": len(idx),
            "spatial_dim": self.spatial_dim,
            "complexity": "O(N) Mesh-Free",
            "backend": self.backend,
            "jit": self.jit,
        }
        return updated_np, meta
