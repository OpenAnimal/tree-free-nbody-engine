"""
Tree-Free Multipole Attention (`multipole_attention.py`)
========================================================
Linear-Time O(N) Spatial and Manifold Attention Layer.
Powered by Tree-Free Fast Multipole Method (Greengard & Rokhlin, 1987) & Farach-Colton, Krapivin, & Kuszmaul (2025) Open Addressing.

Replaces standard O(N^2) Softmax Multi-Head Attention for:
- 2D Vision Transformers (ViT, High-Resolution 4K/8K Patch Tokens)
- 3D Perception (Point Clouds, LiDAR, 3D Gaussian Splatting)
- 1D Sequence Manifolds with Positional Embedding Grids

Round-7 task T-A1: the legacy pre-funnel `ElasticSpatialHash` class that
lived here has been removed. Spatial bucketing now goes through
`core.spatial_index.CellIndex` (the funnel-hash-backed cell index) via the
shared `neural_ops/_bucketing.py` helper. The `morton_encode_2d/3d`
utilities are kept (used by other modules).
"""

import numpy as np

try:
    from neural_ops._coord_contract import check_unit_coords
except ImportError:  # direct script execution (repo root not yet on sys.path)
    from _coord_contract import check_unit_coords
import os
import sys
from typing import Optional, Tuple, Dict, Any, List

try:
    from ._bucketing import build_cell_index, compute_cluster_moments
except ImportError:  # direct-run fallback (mirrors the sibling modules)
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from neural_ops._bucketing import build_cell_index, compute_cluster_moments

# Tri-backend acceleration (NumPy reference / torch / jax). See
# `neural_ops/_accel.py` for the backend shim. The spatial bucketing stays
# CPU-only (funnel hash); only the per-bucket dense math moves to the device.
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
    # direct-run fallback (mirrors the _bucketing import above)
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
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
        # _accel truly unavailable -> only numpy backend works
        _resolve_backend = None
        _get_ns = None
        _to_backend = None
        _as_numpy = None
        _get_compiled = None
        _HAS_TORCH = False
        _HAS_JAX = False

# torch / jax are imported only when available so this module still imports on
# a numpy-only interpreter. The kernel factories below reference these names.
if _HAS_TORCH:
    import torch
if _HAS_JAX:
    import jax.numpy as jnp


def _make_near_kernel(exp, clip, matmul, sumop, bool_to_f32):
    """Branch-free per-bucket near-field kernel (factory binds the backend ops
    so torch.compile / jax.jit can trace the body without a traced `ns` arg)."""
    def kern(q_src, pts_src, pts_near, k_near, v_near, src_ids, near_ids,
             inv_2_sigma_sq, temperature):
        diff_near = pts_src[:, None, :] - pts_near[None, :, :]
        dist_sq_near = sumop(diff_near * diff_near, axis=-1)
        spatial_w_near = exp(-dist_sq_near * inv_2_sigma_sq)
        dot_near = matmul(q_src, k_near.T) * temperature
        attn_near = spatial_w_near * exp(clip(dot_near, -30.0, 30.0))
        self_mask = bool_to_f32(src_ids[:, None] != near_ids[None, :])
        attn_near = attn_near * self_mask
        val_near = matmul(attn_near, v_near)
        weight_near = sumop(attn_near, axis=-1, keepdims=True) + 1e-9
        return val_near, weight_near
    return kern


def _make_far_kernel(exp, clip, matmul, einsum, sumop):
    """Branch-free per-bucket far-field kernel (zero-order monopole +
    first-order dipole correction)."""
    def kern(q_src, pts_src, far_centers, far_k_means, far_v_sums,
             far_counts, far_dipoles, inv_2_sigma_sq, inv_sigma_sq, temperature):
        diff_far = pts_src[:, None, :] - far_centers[None, :, :]
        dist_sq_far = sumop(diff_far * diff_far, axis=-1)
        spatial_w_far = exp(-dist_sq_far * inv_2_sigma_sq)
        dot_far = matmul(q_src, far_k_means.T) * temperature
        w_far = spatial_w_far * exp(clip(dot_far, -30.0, 30.0))
        val_far_0 = matmul(w_far, far_v_sums)
        corr = einsum('mfd,fid->mfi', diff_far * inv_sigma_sq, far_dipoles)
        val_far_1 = einsum('mf,mfi->mi', w_far, corr)
        val_far = val_far_0 + val_far_1
        weight_far = matmul(w_far, far_counts[:, None])
        return val_far, weight_far
    return kern


def _sum_np(x, axis=-1, keepdims=False):
    return x.sum(axis=axis, keepdims=keepdims)


# Per-backend eager kernel instances (compiled lazily via _get_compiled).
_NEAR_KERNELS = None
_FAR_KERNELS = None


def _kernels():
    global _NEAR_KERNELS, _FAR_KERNELS
    if _NEAR_KERNELS is None:
        _NEAR_KERNELS = {
            "numpy": _make_near_kernel(np.exp, np.clip, np.matmul, _sum_np,
                                       lambda x: x.astype(np.float32)),
            "torch": _make_near_kernel(torch.exp, torch.clamp, torch.matmul,
                                       lambda x, axis=-1, keepdims=False: x.sum(dim=axis, keepdim=keepdims),
                                       lambda x: x.to(torch.float32)) if _HAS_TORCH else None,
            "jax": _make_near_kernel(jnp.exp, jnp.clip, jnp.matmul,
                                     lambda x, axis=-1, keepdims=False: x.sum(axis=axis, keepdims=keepdims),
                                     lambda x: x.astype(jnp.float32)) if _HAS_JAX else None,
        }
        _FAR_KERNELS = {
            "numpy": _make_far_kernel(np.exp, np.clip, np.matmul, np.einsum, _sum_np),
            "torch": _make_far_kernel(torch.exp, torch.clamp, torch.matmul, torch.einsum,
                                      lambda x, axis=-1, keepdims=False: x.sum(dim=axis, keepdim=keepdims)) if _HAS_TORCH else None,
            "jax": _make_far_kernel(jnp.exp, jnp.clip, jnp.matmul, jnp.einsum,
                                   lambda x, axis=-1, keepdims=False: x.sum(axis=axis, keepdims=keepdims)) if _HAS_JAX else None,
        }
    return _NEAR_KERNELS, _FAR_KERNELS


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


class TreeFreeMultipoleAttention:
    """
    Linear-Time O(N) Multipole Attention Layer.
    Decomposes spatial attention into:
      1. Near-field exact softmax dot-product attention within localized spatial hash buckets.
      2. Far-field multipole expansion summary from distant spatial clusters.

    Shapes / dtypes
    ---------------
    Q, K, V : float32 (N, embed_dim)
    coords  : float32 (N, spatial_dim), NORMALIZED to [0, 1)^d (out-of-range
        values are clipped and trigger a RuntimeWarning — see
        neural_ops/_coord_contract.py)
    returns : (out float32 (N, embed_dim), meta dict)

    Example
    -------
    >>> import numpy as np
    >>> from neural_ops.multipole_attention import TreeFreeMultipoleAttention
    >>> rng = np.random.default_rng(0)
    >>> Q, K, V = (rng.standard_normal((512, 64)).astype(np.float32) for _ in range(3))
    >>> coords = rng.random((512, 2)).astype(np.float32)      # already in [0,1)^2
    >>> att = TreeFreeMultipoleAttention(embed_dim=64, spatial_dim=2, grid_depth=4)
    >>> out, meta = att.forward(Q, K, V, coords)
    """
    def __init__(
        self,
        embed_dim: int = 64,
        spatial_dim: int = 2,
        grid_depth: int = 4,
        multipole_order: int = 2,
        spatial_sigma: float = 0.25,
        temperature: Optional[float] = None,
        backend: str = "numpy",
        jit: bool = False,
    ):
        self.embed_dim = embed_dim
        self.spatial_dim = spatial_dim
        self.grid_depth = grid_depth
        self.multipole_order = multipole_order
        # float() wrapper: prevents numpy.float64 from promoting float32 JAX
        # arrays to float64 when jax_enable_x64=True (see Round-10 audit).
        self.spatial_sigma = float(spatial_sigma)
        # temperature=0.0 must survive (falsy `or` clobbers it).
        # float() wrapper is critical: 1.0/np.sqrt() returns numpy.float64,
        # which promotes float32 JAX arrays to float64 when jax_enable_x64=True.
        self.temperature = float(1.0 / np.sqrt(embed_dim)) if temperature is None else float(temperature)
        self.grid_res = 1 << grid_depth
        # Acceleration backend: "numpy" (reference, default) | "torch" | "jax".
        # jit=True enables torch.compile / jax.jit on the per-bucket kernels.
        # See neural_ops/_accel.py for the triton-missing fallback behavior.
        self.backend = _resolve_backend(backend) if _resolve_backend is not None else "numpy"
        self.jit = bool(jit)

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

        Input contract: Q/K/V are float32 (N, D) arrays; `coords` must be
        NORMALIZED to the unit cube [0, 1)^d. Coordinates outside [0, 1)^d
        are silently clipped onto the cube boundary, which collapses distant
        tokens into the same boundary cells and silently degrades the output
        (measured: [0, 10) world coords reduced 189 active clusters to 15
        with no error raised). Rescale world coordinates to [0, 1)^d before
        calling (e.g. (x - x.min()) / (x.max() - x.min()) per axis, or an
        isotropic rescale if you need to preserve aspect ratio).

        The `backend` set at construction selects where the per-bucket dense
        math runs: "numpy" (reference, CPU) | "torch" (GPU/CPU, optionally
        torch.compile'd) | "jax" (CPU/GPU, jax.jit'd). Spatial bucketing is
        always CPU (funnel hash); only the per-bucket matmul/exp/einsum move
        to the device backend.
        """
        check_unit_coords(coords, "TreeFreeMultipoleAttention.forward(coords)")
        if self.backend != "numpy":
            return self._forward_accel(Q, K, V, coords)
        N, D = Q.shape
        coords_clipped = np.clip(coords, 1e-4, 1.0 - 1e-4).astype(np.float64)

        # 1. Bucket tokens via the funnel-hash-backed CellIndex
        idx, unique_keys, inverse = build_cell_index(
            coords_clipped, self.spatial_dim, self.grid_res
        )

        # 2. Precompute Far-Field Multipole Cluster Summaries (P2M)
        all_centers, all_mean_k, all_sum_v, all_dipoles, all_counts, key_to_idx = \
            compute_cluster_moments(
                coords_clipped, K, V, idx, inverse, self.spatial_dim, D
            )
        n_clusters = len(key_to_idx)

        # 3. Fast Vectorized Bucket-Level Evaluation (Near P2P + Far M2L)
        out_v = np.zeros((N, D), dtype=np.float32)
        inv_2_sigma_sq = 1.0 / (2.0 * (self.spatial_sigma ** 2))
        inv_sigma_sq = 1.0 / (self.spatial_sigma ** 2)

        total_near_evals = 0
        total_far_evals = 0

        # Evaluate per bucket
        for k_src in idx.occupied_keys():
            p_src_arr = idx.bucket(k_src)
            M_src = len(p_src_arr)
            q_src = Q[p_src_arr]       # (M_src, D)
            pts_src = coords_clipped[p_src_arr] # (M_src, d_spatial)

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

            # Mask the j==i self-pair: when a query and a near token share the
            # same original index, exclude that pair (the dense reference
            # excludes self-interaction). Without this, the i==i term has
            # spatial_w=1 and dot=q·k, artificially dominating the row.
            src_ids = np.asarray(p_src_arr, dtype=np.int32)[:, None]
            near_ids = near_arr[None, :]
            self_mask = (src_ids != near_ids).astype(np.float32)
            attn_near = attn_near * self_mask

            val_near = np.matmul(attn_near, v_near) # (M_src, D)
            weight_near = np.sum(attn_near, axis=-1, keepdims=True) + 1e-9 # (M_src, 1)

            total_near_evals += M_src * len(near_arr)

            # --- Vectorized Far-Field Evaluation for Bucket ---
            far_indices = [c for c in range(n_clusters) if c not in near_indices_set]
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

                # First order dipole correction (Taylor expansion of the spatial
                # Gaussian about the cluster center).  With x_j = c_f + delta_j,
                #   ||x_i - x_j||^2 = ||diff_far||^2 - 2*diff_far·delta_j + ||delta_j||^2
                # so exp(-||x_i-x_j||^2/(2 sigma^2)) ≈ w_far*(1 + diff_far·delta_j/sigma^2).
                # The far contribution is then w_far * [sum V_j + (diff_far/sigma^2)·sum V_j⊗delta_j]
                # = w_far * [far_v_sums + (diff_far/sigma^2) · far_dipoles]  -- POSITIVE sign.
                # (The previous revision used -diff_far, which is the wrong sign
                # and makes the dipole term worse than the monopole alone.)
                corr = np.einsum('mfd,fid->mfi', diff_far * inv_sigma_sq, far_dipoles) # (M_src, N_far, D)
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
            "active_clusters": len(idx),
            "total_near_evals": total_near_evals,
            "total_far_evals": total_far_evals,
            "complexity_scaling": "O(N)",
        }
        return out_v, meta

    def _forward_accel(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        coords: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """torch / jax backend path. Bucketing stays CPU (CellIndex); the full
        Q/K/V/coords and cluster-moment arrays are moved to the device once,
        then per-bucket gathers + the near/far dense math run on-device via
        the (optionally JIT-compiled) branch-free kernels. Output is pulled
        back to a NumPy array so the return contract matches the numpy path."""
        ns = _get_ns(self.backend)
        N, D = Q.shape
        coords_clipped = np.clip(coords, 1e-4, 1.0 - 1e-4).astype(np.float64)

        # 1-2. CPU bucketing + P2M cluster moments (funnel hash, CPU-only).
        idx, unique_keys, inverse = build_cell_index(
            coords_clipped, self.spatial_dim, self.grid_res
        )
        all_centers, all_mean_k, all_sum_v, all_dipoles, all_counts, key_to_idx = \
            compute_cluster_moments(
                coords_clipped, K, V, idx, inverse, self.spatial_dim, D
            )
        n_clusters = len(key_to_idx)

        # Move the full arrays to the device ONCE; per-bucket gathers happen
        # on-device (one big transfer instead of N_buckets small ones).
        Q_d = _to_backend(Q.astype(np.float32), self.backend, ns.float32)
        K_d = _to_backend(K.astype(np.float32), self.backend, ns.float32)
        V_d = _to_backend(V.astype(np.float32), self.backend, ns.float32)
        coords_d = _to_backend(coords_clipped.astype(np.float32), self.backend, ns.float32)
        centers_d = _to_backend(all_centers, self.backend, ns.float32)
        mean_k_d = _to_backend(all_mean_k, self.backend, ns.float32)
        sum_v_d = _to_backend(all_sum_v, self.backend, ns.float32)
        dipoles_d = _to_backend(all_dipoles, self.backend, ns.float32)
        counts_d = _to_backend(all_counts, self.backend, ns.float32)

        out_v = ns.zeros((N, D), dtype=ns.float32)
        inv_2_sigma_sq = 1.0 / (2.0 * (self.spatial_sigma ** 2))
        inv_sigma_sq = 1.0 / (self.spatial_sigma ** 2)
        temperature = self.temperature

        near_kernels, far_kernels = _kernels()
        near_fn = _get_compiled(self.backend, "mpa_near", near_kernels[self.backend], jit=self.jit)
        far_fn = _get_compiled(self.backend, "mpa_far", far_kernels[self.backend], jit=self.jit)

        total_near_evals = 0
        total_far_evals = 0

        for k_src in idx.occupied_keys():
            p_src_arr = np.asarray(idx.bucket(k_src), dtype=np.int32)

            # Near neighbor keys via CPU funnel-hash probes.
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
            # On-device gathers (numpy int arrays index backend tensors).
            q_src = ns.index(Q_d, p_src_arr)
            pts_src = ns.index(coords_d, p_src_arr)
            k_near = ns.index(K_d, near_arr)
            v_near = ns.index(V_d, near_arr)
            pts_near = ns.index(coords_d, near_arr)
            src_ids = _to_backend(p_src_arr, self.backend)
            near_ids = _to_backend(near_arr, self.backend)

            val_near, weight_near = near_fn(
                q_src, pts_src, pts_near, k_near, v_near, src_ids, near_ids,
                inv_2_sigma_sq, temperature,
            )
            total_near_evals += len(p_src_arr) * len(near_arr)

            far_indices = [c for c in range(n_clusters) if c not in near_indices_set]
            if far_indices:
                far_idx_arr = np.asarray(far_indices, dtype=np.int32)
                far_centers = ns.index(centers_d, far_idx_arr)
                far_k_means = ns.index(mean_k_d, far_idx_arr)
                far_v_sums = ns.index(sum_v_d, far_idx_arr)
                far_counts = ns.index(counts_d, far_idx_arr)
                far_dipoles = ns.index(dipoles_d, far_idx_arr)
                val_far, weight_far = far_fn(
                    q_src, pts_src, far_centers, far_k_means, far_v_sums,
                    far_counts, far_dipoles, inv_2_sigma_sq, inv_sigma_sq, temperature,
                )
                val_total = val_near + val_far
                weight_total = weight_near + weight_far
                total_far_evals += len(p_src_arr) * len(far_indices)
            else:
                val_total = val_near
                weight_total = weight_near

            out_v = ns.index_set(out_v, p_src_arr, val_total / weight_total)

        out_np = _as_numpy(out_v).astype(np.float32)
        meta = {
            "num_particles": N,
            "active_clusters": len(idx),
            "total_near_evals": total_near_evals,
            "total_far_evals": total_far_evals,
            "complexity_scaling": "O(N)",
            "backend": self.backend,
            "jit": self.jit,
        }
        return out_np, meta


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
        backend: str = "numpy",
        jit: bool = False,
    ):
        """backend/jit pass through to every head (see
        TreeFreeMultipoleAttention: "numpy" reference, or "torch"/"jax"
        when installed — the per-bucket dense math then runs on the
        selected accelerator; see neural_ops/_accel.py)."""
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
                backend=backend,
                jit=jit,
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
