"""
Visual Transformer & Multimodal Multipole Attention (`visual_transformer_ops.py`)
================================================================================
Advanced Linear-Time O(N) Vision Transformer (ViT), CNN-Transformer Hybrid,
and Gemma 4 / Modern Multimodal Multi-Scale Spatial-Cross-Attention Architectures.

Key Architectures:
1. MultiScaleVisualMultipoleAttention: Multi-frequency spatial patch attention across 2D/3D grids.
2. MultimodalCrossMultipoleAttention: Cross-attention aligning 1D text tokens with 2D image / 3D point cloud tokens.
3. ConvMultipoleHybridLayer: Local depthwise convolution (CNN inductive bias) fused with global O(N) far-field multipole mixing.
"""

from __future__ import annotations
import numpy as np
import time
from typing import Tuple, Dict, List, Optional, Any, Union


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish non-linear activation."""
    return x / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class MultiScaleVisualMultipoleAttention:
    """
    Multi-Scale Spatial Attention for High-Resolution Vision Transformers (ViT / Gemma 4 Vision).
    Processes image patch tokens across fine, medium, and coarse spatial Morton resolutions in O(N) time.
    """
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        spatial_dim: int = 2,
        base_depth: int = 4,
        num_scales: int = 3,
        temperature: Optional[float] = None,
    ):
        self.d_model = int(embed_dim)
        self.num_heads = int(num_heads)
        self.spatial_dim = int(spatial_dim)
        self.base_depth = int(base_depth)
        self.num_scales = int(num_scales)
        if self.d_model < 1 or self.num_heads < 1 or self.d_model % self.num_heads != 0:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        if self.spatial_dim != 2 or self.base_depth < 1 or self.num_scales < 1:
            raise ValueError("spatial_dim=2, base_depth>=1, and num_scales>=1 are required")
        self.d_head = self.d_model // self.num_heads
        self.scale = float(temperature) if temperature is not None else (1.0 / np.sqrt(self.d_head))
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("temperature must be finite and positive")

        # Linear projections
        scale_init = 1.0 / np.sqrt(self.d_model)
        rng = np.random.RandomState(42)
        self.W_q = rng.normal(0, scale_init, size=(self.d_model, self.d_model)).astype(np.float32)
        self.W_k = rng.normal(0, scale_init, size=(self.d_model, self.d_model)).astype(np.float32)
        self.W_v = rng.normal(0, scale_init, size=(self.d_model, self.d_model)).astype(np.float32)
        self.W_out = rng.normal(0, scale_init, size=(self.d_model, self.d_model)).astype(np.float32)

    def _morton_encode_2d(self, x: np.ndarray, y: np.ndarray, depth: int) -> np.ndarray:
        res = 1 << depth
        ix = np.clip(np.floor(x * res).astype(np.int64), 0, res - 1)
        iy = np.clip(np.floor(y * res).astype(np.int64), 0, res - 1)

        def spread_bits(v: np.ndarray) -> np.ndarray:
            v = (v | (v << 8)) & 0x00FF00FF
            v = (v | (v << 4)) & 0x0F0F0F0F
            v = (v | (v << 2)) & 0x33333333
            v = (v | (v << 1)) & 0x55555555
            return v

        return (spread_bits(ix) | (spread_bits(iy) << 1))

    def forward(
        self,
        X: np.ndarray,      # (N, d_model) Visual patch tokens
        coords: np.ndarray, # (N, 2) Normalized 2D spatial coordinates in [0, 1)^2
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Multi-scale visual forward pass in linear O(N) time.
        """
        X = np.asarray(X, dtype=np.float32)
        coords = np.asarray(coords, dtype=np.float32)
        if X.ndim != 2 or coords.ndim != 2 or coords.shape[1] != 2 or len(X) != len(coords):
            raise ValueError("X must have shape (N, d_model) and coords must have shape (N, 2)")
        N, D = X.shape
        if D != self.d_model or not np.all(np.isfinite(X)) or not np.all(np.isfinite(coords)):
            raise ValueError(f"X must have shape (N, {self.d_model}) and inputs must be finite")
        t0 = time.perf_counter()

        if N == 0:
            return np.empty((0, D), dtype=np.float32), {"num_tokens": 0, "complexity": "O(N)"}

        # Project Q, K, V
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        # Multi-Scale Pyramid Aggregation
        multi_scale_outputs = []

        for s_idx in range(self.num_scales):
            current_depth = max(1, self.base_depth - s_idx)
            grid_res = 1 << current_depth
            cell_size = 1.0 / grid_res
            sigma = cell_size * 1.5
            inv_2_sig_sq = 1.0 / (2.0 * (sigma ** 2))

            keys = self._morton_encode_2d(coords[:, 0], coords[:, 1], current_depth)
            
            # Bucket spatial tokens
            bucket_map: Dict[int, List[int]] = {}
            for i in range(N):
                k_val = int(keys[i])
                if k_val not in bucket_map:
                    bucket_map[k_val] = []
                bucket_map[k_val].append(i)

            cluster_keys = list(bucket_map.keys())
            n_clusters = len(cluster_keys)
            
            cluster_centers = np.zeros((n_clusters, 2), dtype=np.float32)
            cluster_k_means = np.zeros((n_clusters, D), dtype=np.float32)
            cluster_v_sums = np.zeros((n_clusters, D), dtype=np.float32)
            cluster_counts = np.zeros(n_clusters, dtype=np.float32)

            for idx, k_val in enumerate(cluster_keys):
                p_ids = bucket_map[k_val]
                pts = coords[p_ids]
                cluster_centers[idx] = np.mean(pts, axis=0)
                cluster_k_means[idx] = np.mean(K[p_ids], axis=0)
                cluster_v_sums[idx] = np.sum(V[p_ids], axis=0)
                cluster_counts[idx] = len(p_ids)

            # Evaluate far-field multipole gathering
            out_scale = np.zeros((N, D), dtype=np.float32)

            # Vectorized cluster interaction
            diff = coords[:, None, :] - cluster_centers[None, :, :] # (N, n_clusters, 2)
            dist_sq = np.sum(diff ** 2, axis=-1)
            spatial_w = np.exp(-dist_sq * inv_2_sig_sq)

            dot = (Q @ cluster_k_means.T) * self.scale
            w = spatial_w * np.exp(np.clip(dot, -30.0, 30.0))

            val = w @ cluster_v_sums
            norm = w @ cluster_counts[:, None] + 1e-12

            out_scale = val / norm
            multi_scale_outputs.append(out_scale)

        # Fuse multi-scale representations and project
        fused = np.mean(multi_scale_outputs, axis=0)
        out_final = X + fused @ self.W_out
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        meta = {
            "num_tokens": N,
            "d_model": self.d_model,
            "num_scales": self.num_scales,
            "elapsed_ms": elapsed_ms,
            "complexity": "O(N) Multi-Scale Visual",
        }
        return out_final, meta


class MultimodalCrossMultipoleAttention:
    """
    Linear-Time Multimodal Cross-Attention Layer.
    Aligns sequence text tokens with high-resolution visual patch tokens without NxM matrix materialization.
    """
    def __init__(
        self,
        d_model: int = 64,
        spatial_dim: int = 2,
        grid_depth: int = 4,
    ):
        self.d_model = int(d_model)
        self.spatial_dim = int(spatial_dim)
        self.grid_depth = int(grid_depth)
        if self.d_model < 1 or self.spatial_dim != 2 or self.grid_depth < 0 or self.grid_depth > 10:
            raise ValueError("d_model must be positive, spatial_dim must be 2, and grid_depth must be in [0, 10]")
        self.scale = 1.0 / np.sqrt(self.d_model)

        rng = np.random.RandomState(42)
        scale_init = 1.0 / np.sqrt(d_model)
        self.W_q = rng.normal(0, scale_init, size=(d_model, d_model)).astype(np.float32)
        self.W_k = rng.normal(0, scale_init, size=(d_model, d_model)).astype(np.float32)
        self.W_v = rng.normal(0, scale_init, size=(d_model, d_model)).astype(np.float32)
        self.W_out = rng.normal(0, scale_init, size=(d_model, d_model)).astype(np.float32)

    def forward(
        self,
        text_tokens: np.ndarray,    # (N_text, d_model) Query text modality
        visual_tokens: np.ndarray,  # (N_vis, d_model) Key-Value visual modality
        visual_coords: np.ndarray,  # (N_vis, 2) Visual spatial coordinates
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cross-modal attention forward pass in linear O(N_text + N_vis) time.
        """
        text_tokens = np.asarray(text_tokens, dtype=np.float32)
        visual_tokens = np.asarray(visual_tokens, dtype=np.float32)
        visual_coords = np.asarray(visual_coords, dtype=np.float32)
        if text_tokens.ndim != 2 or visual_tokens.ndim != 2 or text_tokens.shape[1] != self.d_model or visual_tokens.shape[1] != self.d_model:
            raise ValueError(f"text_tokens and visual_tokens must have shape (N, {self.d_model})")
        if visual_coords.shape != (len(visual_tokens), self.spatial_dim):
            raise ValueError(f"visual_coords must have shape (N, {self.spatial_dim})")
        if len(text_tokens) == 0 or len(visual_tokens) == 0:
            return np.zeros((len(text_tokens), self.d_model), dtype=np.float32), {"num_text_tokens": len(text_tokens), "num_visual_tokens": len(visual_tokens), "visual_clusters": 0, "complexity": "O(N_text + N_vis) Cross-Modal"}
        if not np.all(np.isfinite(text_tokens)) or not np.all(np.isfinite(visual_tokens)) or not np.all(np.isfinite(visual_coords)):
            raise ValueError("text_tokens, visual_tokens, and visual_coords must be finite")
        N_text = len(text_tokens)
        N_vis = len(visual_tokens)
        t0 = time.perf_counter()

        Q_text = text_tokens @ self.W_q      # (N_text, D)
        K_vis = visual_tokens @ self.W_k     # (N_vis, D)
        V_vis = visual_tokens @ self.W_v     # (N_vis, D)

        # Cluster visual tokens into spatial multipole moments
        grid_res = 1 << self.grid_depth
        res = grid_res
        ix = np.clip(np.floor(visual_coords[:, 0] * res).astype(np.int64), 0, res - 1)
        iy = np.clip(np.floor(visual_coords[:, 1] * res).astype(np.int64), 0, res - 1)
        vis_keys = ix + iy * res

        bucket_map: Dict[int, List[int]] = {}
        for i in range(N_vis):
            k = int(vis_keys[i])
            if k not in bucket_map:
                bucket_map[k] = []
            bucket_map[k].append(i)

        n_clusters = len(bucket_map)
        cluster_k_means = np.zeros((n_clusters, self.d_model), dtype=np.float32)
        cluster_v_sums = np.zeros((n_clusters, self.d_model), dtype=np.float32)
        cluster_counts = np.zeros(n_clusters, dtype=np.float32)

        for idx, (k_val, p_ids) in enumerate(bucket_map.items()):
            cluster_k_means[idx] = np.mean(K_vis[p_ids], axis=0)
            cluster_v_sums[idx] = np.sum(V_vis[p_ids], axis=0)
            cluster_counts[idx] = len(p_ids)

        # Cross-modal dot product between text queries and visual cluster centroids
        dot = (Q_text @ cluster_k_means.T) * self.scale # (N_text, n_clusters)
        w = np.exp(np.clip(dot, -30.0, 30.0))

        val = w @ cluster_v_sums
        norm = w @ cluster_counts[:, None] + 1e-12

        cross_out = text_tokens + (val / norm) @ self.W_out
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        meta = {
            "num_text_tokens": N_text,
            "num_visual_tokens": N_vis,
            "visual_clusters": n_clusters,
            "elapsed_ms": elapsed_ms,
            "complexity": "O(N_text + N_vis) Cross-Modal",
        }
        return cross_out, meta


class ConvMultipoleHybridLayer:
    """
    CNN-Transformer Hybrid Layer.
    Fuses local depthwise spatial convolutions (2D CNN inductive bias) with global O(N) Far-Field Multipoles.
    """
    def __init__(
        self,
        channels: int = 64,
        kernel_size: int = 3,
        spatial_dim: int = 2,
    ):
        self.channels = int(channels)
        self.k = int(kernel_size)
        if self.channels < 1 or self.k < 1 or self.k % 2 == 0 or spatial_dim != 2 or self.channels % 4 != 0:
            raise ValueError("channels must be positive and divisible by 4; kernel_size must be positive odd; spatial_dim must be 2")
        self.pad = self.k // 2
        
        # Depthwise conv kernel: (C, K, K)
        rng = np.random.RandomState(42)
        self.dw_weights = rng.normal(0, 0.1, size=(self.channels, self.k, self.k)).astype(np.float32)
        self.pointwise = rng.normal(0, 1.0 / np.sqrt(channels), size=(self.channels, self.channels)).astype(np.float32)
        self.multipole = MultiScaleVisualMultipoleAttention(embed_dim=channels, num_heads=4, spatial_dim=spatial_dim)

    def forward(
        self,
        feature_map: np.ndarray, # (H, W, C) 2D visual feature tensor
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass executing local depthwise conv + global multipole attention in O(H*W).
        """
        feature_map = np.asarray(feature_map, dtype=np.float32)
        if feature_map.ndim != 3 or feature_map.shape[2] != self.channels or not np.all(np.isfinite(feature_map)):
            raise ValueError(f"feature_map must have finite shape (H, W, {self.channels})")
        H, W, C = feature_map.shape
        if H == 0 or W == 0:
            return np.empty_like(feature_map), {"num_tokens": 0, "complexity": "O(H*W)"}
        t0 = time.perf_counter()

        # 1. Local Depthwise Convolution
        padded = np.pad(feature_map, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='edge')
        conv_out = np.zeros_like(feature_map)

        for c in range(C):
            for i in range(self.k):
                for j in range(self.k):
                    conv_out[:, :, c] += padded[i:i+H, j:j+W, c] * self.dw_weights[c, i, j]

        conv_activated = silu(conv_out) @ self.pointwise

        # 2. Global Multipole Long-Range Context
        tokens = conv_activated.reshape(H * W, C)
        grid_y, grid_x = np.meshgrid(np.linspace(0.01, 0.99, H), np.linspace(0.01, 0.99, W), indexing='ij')
        coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).astype(np.float32)

        global_out, meta = self.multipole.forward(tokens, coords)
        final_map = global_out.reshape(H, W, C)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        meta["elapsed_total_ms"] = elapsed_ms
        return final_map, meta


if __name__ == "__main__":
    print("=" * 70)
    print("Visual Transformer & Multimodal Multipole Attention Benchmark")
    print("=" * 70)

    # 1. Multi-Scale ViT Patch Attention
    N_patches, D = 4096, 64 # 64x64 image grid
    rng = np.random.RandomState(42)
    patches = rng.randn(N_patches, D).astype(np.float32)
    coords = rng.uniform(0.0, 1.0, size=(N_patches, 2)).astype(np.float32)

    vit_op = MultiScaleVisualMultipoleAttention(embed_dim=D, num_heads=4, num_scales=3)
    out_vit, meta_vit = vit_op.forward(patches, coords)
    print(f"1. Multi-Scale ViT Attention (4,096 patches) : {meta_vit['elapsed_ms']:.2f} ms")

    # 2. Multimodal Cross-Attention (Text -> Image)
    N_text = 128
    text_emb = rng.randn(N_text, D).astype(np.float32)
    cross_op = MultimodalCrossMultipoleAttention(d_model=D)
    out_cross, meta_cross = cross_op.forward(text_emb, patches, coords)
    print(f"2. Multimodal Cross-Attention (128 Text x 4096 Vis): {meta_cross['elapsed_ms']:.2f} ms")

    # 3. CNN-Transformer Hybrid
    H, W = 32, 32
    img_map = rng.randn(H, W, D).astype(np.float32)
    hybrid_op = ConvMultipoleHybridLayer(channels=D, kernel_size=3)
    out_hybrid, meta_hybrid = hybrid_op.forward(img_map)
    print(f"3. ConvMultipole Hybrid Layer (32x32x64 image)    : {meta_hybrid['elapsed_total_ms']:.2f} ms")
    print("=" * 70)
