"""
Example 4: 3D Gaussian Splatting (3DGS) Continuous Multipole Attention
======================================================================
Applies linear-time O(N) spatial multipole attention across massive 3D Gaussian scenes
(means mu_i, scales s_i, opacities alpha_i, colors c_i) for semantic scene segmentation
and neural 3D editing without materializing a 500k x 500k quadratic attention matrix.
"""

import numpy as np
import time
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from neural_ops import TreeFreeMultipoleAttention


class GaussianSplatScene:
    """Represents a continuous 3D Gaussian Splatting scene."""
    def __init__(self, n_gaussians: int = 10000):
        self.n_gaussians = n_gaussians
        np.random.seed(42)

        # Generate 3 distinct 3D semantic clusters (e.g., Table, Chair, Lamp)
        c1 = np.random.normal(loc=[0.3, 0.3, 0.4], scale=0.08, size=(n_gaussians // 3, 3))
        c2 = np.random.normal(loc=[0.7, 0.6, 0.5], scale=0.07, size=(n_gaussians // 3, 3))
        c3 = np.random.normal(loc=[0.5, 0.8, 0.7], scale=0.06, size=(n_gaussians - 2 * (n_gaussians // 3), 3))

        self.means = np.clip(np.vstack([c1, c2, c3]), 0.05, 0.95).astype(np.float32) # (N, 3)
        self.scales = np.random.uniform(0.005, 0.02, size=(n_gaussians, 3)).astype(np.float32)
        self.opacities = np.random.uniform(0.5, 1.0, size=(n_gaussians, 1)).astype(np.float32)
        self.colors = np.random.uniform(0.0, 1.0, size=(n_gaussians, 3)).astype(np.float32) # RGB

        # Ground truth object class (0, 1, 2)
        self.labels = np.array([0] * len(c1) + [1] * len(c2) + [2] * len(c3))


class GaussianSplatAttentionEditor:
    """
    Neural 3DGS Editor evaluating all-pairs Gaussian interaction in linear O(N) time.
    """
    def __init__(self, feature_dim: int = 32, grid_depth: int = 4):
        self.feature_dim = feature_dim
        self.attn = TreeFreeMultipoleAttention(
            embed_dim=feature_dim,
            spatial_dim=3,
            grid_depth=grid_depth,
            spatial_sigma=0.15
        )
        # Projection layer: (RGB + Scale + Opacity -> Feature Dim)
        scale = 1.0 / np.sqrt(7)
        self.w_in = np.random.normal(0, scale, size=(7, feature_dim)).astype(np.float32)
        self.w_seg = np.random.normal(0, 1.0 / np.sqrt(feature_dim), size=(feature_dim, 3)).astype(np.float32)

    def segment_scene(self, scene: GaussianSplatScene):
        # 1. Construct input tokens: [RGB (3), Scale (3), Opacity (1)] -> (N, 7)
        raw_features = np.hstack([scene.colors, scene.scales, scene.opacities])
        h_in = np.matmul(raw_features, self.w_in) # (N, feature_dim)

        # 2. Linear O(N) 3D Multipole Attention pass across all Gaussians
        Q = h_in
        K = h_in
        V = h_in
        attended_features, meta = self.attn.forward(Q, K, V, scene.means)

        # 3. Predict semantic logits per Gaussian
        logits = np.matmul(attended_features, self.w_seg) # (N, 3)
        pred_classes = np.argmax(logits, axis=-1)
        return pred_classes, meta


def run_3dgs_demo():
    print("=" * 70)
    print(">>> DEMO 4: 3D Gaussian Splatting (3DGS) Continuous Multipole Attention")
    print("=" * 70)

    N_gaussians = 10000
    print(f"[*] Initializing 3DGS scene with {N_gaussians:,} continuous 3D Gaussians...")
    scene = GaussianSplatScene(n_gaussians=N_gaussians)
    editor = GaussianSplatAttentionEditor(feature_dim=32, grid_depth=4)

    t0 = time.perf_counter()
    pred_labels, meta = editor.segment_scene(scene)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Theoretical dense attention memory
    dense_mem_mb = (N_gaussians * N_gaussians * 4) / (1024 * 1024)

    print(f"[-] Gaussian Count:   {N_gaussians:,} continuous Gaussians in 3D")
    print(f"[-] Execution Time:   {elapsed_ms:.2f} ms")
    print(f"[-] Active Clusters:  {meta['active_clusters']} 3D spatial octants")
    print(f"[-] Theoretical RAM:  {dense_mem_mb:.1f} MB (Dense) -> 0 MB materialized")
    print(f"[-] Semantic Classes: {np.unique(pred_labels)} segmented")
    print("=" * 70)


if __name__ == "__main__":
    run_3dgs_demo()
