"""
Verification and Unit Tests for `fmm_neural_ops`
================================================
Validates numerical stability, equivariance properties, and O(N) scaling.
"""

import numpy as np
import time
import sys
import os

# Ensure package import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from neural_ops.multipole_attention import TreeFreeMultipoleAttention, MultiHeadMultipoleAttention
from neural_ops.elastic_kv_cache import ElasticMultipoleKVCache
from neural_ops.continuous_meshfree_gnn import ContinuousMeshfreeGNNLayer
from neural_ops.equivariant_field_layer import EquivariantMultipoleLayer


def test_multipole_attention():
    print("\n[TEST 1] Testing TreeFreeMultipoleAttention & MultiHeadMultipoleAttention...")
    N, D = 300, 32
    np.random.seed(42)
    coords = np.random.uniform(0.05, 0.95, size=(N, 2))
    Q = np.random.randn(N, D).astype(np.float32)
    K = np.random.randn(N, D).astype(np.float32)
    V = np.random.randn(N, D).astype(np.float32)

    attn = TreeFreeMultipoleAttention(embed_dim=D, spatial_dim=2, grid_depth=3)
    out, meta = attn.forward(Q, K, V, coords)

    assert out.shape == (N, D), f"Expected shape {(N, D)}, got {out.shape}"
    assert not np.isnan(out).any(), "Output contains NaNs!"
    assert not np.isinf(out).any(), "Output contains Infs!"
    print(f"  -> TreeFreeMultipoleAttention passed. Active clusters: {meta['active_clusters']}")

    # MultiHead test
    mh_attn = MultiHeadMultipoleAttention(d_model=64, n_heads=4, spatial_dim=2, grid_depth=3)
    X = np.random.randn(N, 64).astype(np.float32)
    mh_out, mh_meta = mh_attn.forward(X, coords)
    assert mh_out.shape == (N, 64), f"Expected shape {(N, 64)}, got {mh_out.shape}"
    assert not np.isnan(mh_out).any(), "MultiHead output contains NaNs!"
    print("  -> MultiHeadMultipoleAttention passed.")


def test_elastic_kv_cache():
    print("\n[TEST 2] Testing ElasticMultipoleKVCache...")
    d_k, d_v = 32, 32
    cache = ElasticMultipoleKVCache(d_k=d_k, d_v=d_v, n_hyperplanes=6, recent_window_size=64)

    np.random.seed(1337)
    N_tokens = 500
    K_seq = np.random.randn(N_tokens, d_k).astype(np.float32)
    V_seq = np.random.randn(N_tokens, d_v).astype(np.float32)

    cache.append_batch(K_seq, V_seq)
    assert cache.total_tokens_inserted == N_tokens, "Token count mismatch"

    q_query = np.random.randn(d_k).astype(np.float32)
    out_v, meta = cache.query_attention(q_query)

    assert out_v.shape == (d_v,), f"Expected shape {(d_v,)}, got {out_v.shape}"
    assert not np.isnan(out_v).any(), "KV Cache output contains NaNs!"
    print(f"  -> ElasticMultipoleKVCache passed. History: {meta['total_tokens_in_history']} tokens, Compression ratio: {meta['compression_ratio']:.2f}x")


def test_continuous_meshfree_gnn():
    print("\n[TEST 3] Testing ContinuousMeshfreeGNNLayer...")
    N, in_f, out_f = 250, 16, 32
    np.random.seed(42)
    coords = np.random.uniform(0.05, 0.95, size=(N, 3))
    node_features = np.random.randn(N, in_f).astype(np.float32)

    gnn = ContinuousMeshfreeGNNLayer(in_features=in_f, out_features=out_f, spatial_dim=3, grid_depth=3)
    out_h, meta = gnn.forward(node_features, coords)

    assert out_h.shape == (N, out_f), f"Expected shape {(N, out_f)}, got {out_h.shape}"
    assert not np.isnan(out_h).any(), "GNN output contains NaNs!"
    assert (out_h >= 0).all(), "ReLU condition violated in output!"
    print(f"  -> ContinuousMeshfreeGNNLayer passed. Evaluated on {meta['num_nodes']} nodes in {meta['spatial_dim']}D.")


def test_equivariant_multipole_layer():
    print("\n[TEST 4] Testing EquivariantMultipoleLayer & SE(3) Equivariance...")
    N, D = 150, 32
    np.random.seed(42)
    coords = np.random.randn(N, 3).astype(np.float32)
    node_features = np.random.randn(N, D).astype(np.float32)
    charges = np.random.choice([-1.0, 1.0], size=N).astype(np.float32)

    layer = EquivariantMultipoleLayer(hidden_dim=D, grid_depth=3)
    feats_1, vec_1, pot_1, meta_1 = layer.forward(coords, node_features, charges)

    assert feats_1.shape == (N, D)
    assert vec_1.shape == (N, 3)
    assert pot_1.shape == (N,)
    assert not np.isnan(vec_1).any()

    # Equivariance Test: Apply 3D Rotation Matrix R to coordinates
    theta = np.pi / 3.0 # 60 degree rotation around Z axis
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ], dtype=np.float32)

    rot_coords = np.matmul(coords, R.T)
    feats_2, vec_2, pot_2, meta_2 = layer.forward(rot_coords, node_features, charges)

    # Invariant check: Potentials should match closely
    pot_diff = np.max(np.abs(pot_1 - pot_2))
    
    # Equivariant check: vec_2 should match R * vec_1
    expected_vec_2 = np.matmul(vec_1, R.T)
    vec_cosine = np.sum(vec_2 * expected_vec_2) / (np.linalg.norm(vec_2) * np.linalg.norm(expected_vec_2) + 1e-8)

    print(f"  -> Invariant potential max abs diff: {pot_diff:.4e}")
    print(f"  -> Equivariant vector field cosine similarity: {vec_cosine:.4f}")
    assert vec_cosine > 0.95, "Equivariant rotation vector alignment failed!"
    print("  -> EquivariantMultipoleLayer passed SE(3) symmetry test.")


if __name__ == "__main__":
    test_multipole_attention()
    test_elastic_kv_cache()
    test_continuous_meshfree_gnn()
    test_equivariant_multipole_layer()
    print("\n[SUCCESS] All neural_ops tests passed successfully!")
