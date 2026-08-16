"""
Example 1: Drop-In Vision Transformer (ViT) Spatial Multipole Attention
=======================================================================
Demonstrates dropping TreeFreeMultipoleAttention into a high-resolution ViT block.
Processes a 64x64 grid (4,096 patch tokens) or 128x128 grid (16,384 tokens) in O(N) time
without allocating an N x N attention matrix.
"""

import numpy as np
import time
import sys
import os

# Ensure neural_ops is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from neural_ops import MultiHeadMultipoleAttention


class ViTMultipoleBlock:
    """
    Drop-in Vision Transformer Block replacing standard O(N^2) Multi-Head Attention.
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ffn: int = 256):
        self.d_model = d_model
        self.attn = MultiHeadMultipoleAttention(
            d_model=d_model,
            n_heads=n_heads,
            spatial_dim=2,
            grid_depth=4,
            spatial_sigma=0.20
        )
        # Feed-forward network (FFN) weights
        scale = 1.0 / np.sqrt(d_model)
        self.W1 = np.random.normal(0, scale, size=(d_model, d_ffn)).astype(np.float32)
        self.W2 = np.random.normal(0, 1.0 / np.sqrt(d_ffn), size=(d_ffn, d_model)).astype(np.float32)
        self.b1 = np.zeros(d_ffn, dtype=np.float32)
        self.b2 = np.zeros(d_model, dtype=np.float32)

    def forward(self, x: np.ndarray, patch_coords: np.ndarray) -> np.ndarray:
        """
        x: (N, d_model) Patch token representations
        patch_coords: (N, 2) Normalized 2D pixel/grid positions in [0, 1)^2
        """
        # 1. Multi-head Multipole Attention with residual connection
        attn_out, _ = self.attn.forward(x, patch_coords)
        x = x + attn_out

        # 2. MLP / Feed-Forward Network with GELU/ReLU
        ffn_hidden = np.maximum(0, np.matmul(x, self.W1) + self.b1)
        ffn_out = np.matmul(ffn_hidden, self.W2) + self.b2
        x = x + ffn_out
        return x


def run_vit_demo():
    print("=" * 70)
    print(">>> DEMO 1: ViT High-Resolution Spatial Multipole Attention")
    print("=" * 70)

    # Simulate 64x64 image patches = 4,096 tokens
    grid_h, grid_w = 64, 64
    N_tokens = grid_h * grid_w
    d_model = 64

    # Generate 2D normalized patch grid coordinates
    gx = np.linspace(0.01, 0.99, grid_w)
    gy = np.linspace(0.01, 0.99, grid_h)
    grid_x, grid_y = np.meshgrid(gx, gy)
    coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).astype(np.float32)

    # Initial random patch embeddings
    patch_tokens = np.random.randn(N_tokens, d_model).astype(np.float32)

    vit_block = ViTMultipoleBlock(d_model=d_model, n_heads=4)

    t0 = time.perf_counter()
    output_tokens = vit_block.forward(patch_tokens, coords)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"[-] Grid Resolution: {grid_h}x{grid_w} ({N_tokens:,} patch tokens)")
    print(f"[-] Embedding Dim:   {d_model} (4 Attention Heads)")
    print(f"[-] Forward Pass:    {elapsed_ms:.2f} ms")
    print(f"[-] Output Shape:    {output_tokens.shape}")
    print(f"[-] NxN Matrix RAM:  0 MB (Never materialized!)")
    print("=" * 70)


if __name__ == "__main__":
    run_vit_demo()
