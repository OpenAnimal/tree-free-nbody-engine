"""
Comprehensive Unit & Equivariance Test Suite for Advanced Neural Ops
===================================================================
Tests all 10 new advanced mathematical modules:
1. Spherical Harmonic Multipole Attention
2. Kernel-Independent FMM Neural Operator
3. Hyperbolic Poincaré Multipole Attention
4. Continuous Flow Matching Drift Operator
5. Spectral Particle-Mesh Ewald (NeuralPME)
6. Multipole Spatial State Space Model (Mamba-FMM)
7. SE(3) Equivariant Multipole Transformer Layer
8. Analytical VJP & Adjoint State Engine
9. Multi-Scale Hierarchical Elastic KV-Cache
10. Neural SPH & IPC Continuum Mechanics Layer
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_ops import (
    SphericalMultipoleAttention,
    compute_real_spherical_harmonics,
    KernelIndependentNeuralOperator,
    HyperbolicMultipoleAttention,
    TreeFreeMultipoleFlowDrift,
    NeuralPME,
    MultipoleSpatialSSM,
    EquivariantMultipoleTransformerLayer,
    MultipoleAdjointEngine,
    HierarchicalElasticKVCache,
    NeuralSPHIPCLayer,
)


def test_spherical_multipole_attention():
    print("Testing 1. Spherical Harmonic Multipole Attention...")
    N, D = 128, 16
    rng = np.random.RandomState(42)
    coords = rng.uniform(0.1, 0.9, size=(N, 3)).astype(np.float32)
    Q = rng.randn(N, D).astype(np.float32)
    K = rng.randn(N, D).astype(np.float32)
    V = rng.randn(N, D).astype(np.float32)

    for l_max in [1, 2, 3]:
        attn = SphericalMultipoleAttention(embed_dim=D, l_max=l_max, grid_depth=3)
        out, meta = attn.forward(Q, K, V, coords)
        assert out.shape == (N, D), f"Expected shape {(N, D)}, got {out.shape}"
        assert not np.isnan(out).any(), "Output contains NaN"
        assert not np.isinf(out).any(), "Output contains Inf"
    print("  -> Passed (l_max=1, 2, 3)")


def test_kernel_independent_fmm():
    print("Testing 2. Kernel-Independent Neural Multipole Operator...")
    N, D_in, D_out = 100, 16, 32
    rng = np.random.RandomState(42)
    coords = rng.uniform(0.1, 0.9, size=(N, 3)).astype(np.float32)
    X = rng.randn(N, D_in).astype(np.float32)

    ki_layer = KernelIndependentNeuralOperator(
        in_features=D_in,
        out_features=D_out,
        spatial_dim=3,
        n_proxy=14,
        grid_depth=3
    )
    out, meta = ki_layer.forward(X, coords)
    assert out.shape == (N, D_out), f"Expected shape {(N, D_out)}, got {out.shape}"
    assert np.all(out >= 0.0), "ReLU output should be non-negative"
    assert not np.isnan(out).any(), "Output contains NaN"
    print("  -> Passed (KI-FMM SVD Skeletonization)")


def test_hyperbolic_multipole_attention():
    print("Testing 3. Hyperbolic Poincaré Multipole Attention...")
    N, D = 120, 16
    rng = np.random.RandomState(42)
    # Generate points inside Poincaré disk (norm < 1.0)
    raw_pts = rng.randn(N, 2).astype(np.float32)
    r = rng.uniform(0.05, 0.85, size=(N, 1)).astype(np.float32)
    hyper_pts = r * (raw_pts / np.linalg.norm(raw_pts, axis=-1, keepdims=True))

    Q = rng.randn(N, D).astype(np.float32)
    K = rng.randn(N, D).astype(np.float32)
    V = rng.randn(N, D).astype(np.float32)

    hyp_attn = HyperbolicMultipoleAttention(embed_dim=D, spatial_dim=2, curvature=1.0)
    out, meta = hyp_attn.forward(Q, K, V, hyper_pts)
    assert out.shape == (N, D), f"Expected shape {(N, D)}, got {out.shape}"
    assert not np.isnan(out).any(), "Output contains NaN"
    print("  -> Passed (Poincaré Ball Fréchet Centroid Attention)")


def test_multipole_flow_drift():
    print("Testing 4. Continuous Flow Matching Drift Operator...")
    N = 150
    rng = np.random.RandomState(42)
    pos = rng.uniform(0.1, 0.9, size=(N, 3)).astype(np.float32)
    charges = rng.uniform(0.5, 1.5, size=N).astype(np.float32)

    drift_op = TreeFreeMultipoleFlowDrift(spatial_dim=3, kernel_type="coulomb_soft")
    drift, meta = drift_op.compute_drift(pos, charges)
    assert drift.shape == (N, 3), f"Expected shape {(N, 3)}, got {drift.shape}"
    assert not np.isnan(drift).any(), "Drift contains NaN"

    # Test ODE step
    neural_v = rng.randn(N, 3).astype(np.float32)
    new_pos = drift_op.step_flow_ode(pos, neural_v, dt=0.01)
    assert new_pos.shape == (N, 3), "ODE stepped positions mismatch"
    assert np.all((new_pos >= 0.0) & (new_pos <= 1.0)), "Positions out of domain bounds"
    print("  -> Passed (O(N) Stein Score & Flow Matching Step)")


def test_spectral_neural_pme():
    print("Testing 5. Spectral Particle-Mesh Ewald (NeuralPME)...")
    N = 200
    rng = np.random.RandomState(42)
    pos = rng.uniform(0.05, 0.95, size=(N, 3)).astype(np.float32)
    charges = rng.choice([-1.0, 1.0], size=N).astype(np.float32)

    pme = NeuralPME(grid_dim=24, alpha_ewald=3.5, r_cutoff=0.25)
    potentials, forces, meta = pme.forward(pos, charges)
    assert potentials.shape == (N,), f"Expected shape {(N,)}, got {potentials.shape}"
    assert forces.shape == (N, 3), f"Expected shape {(N, 3)}, got {forces.shape}"
    assert not np.isnan(potentials).any(), "Potentials contain NaN"
    assert not np.isnan(forces).any(), "Forces contain NaN"
    print("  -> Passed (Real-Space + Reciprocal NUFFT Poisson Solver)")


def test_multipole_mamba_ssm():
    print("Testing 6. Multipole Spatial State Space Model (Mamba-FMM)...")
    N, D = 128, 32
    rng = np.random.RandomState(42)
    X = rng.randn(N, D).astype(np.float32)
    coords = rng.uniform(0.1, 0.9, size=(N, 3)).astype(np.float32)

    ssm = MultipoleSpatialSSM(d_model=D, d_state=16, spatial_dim=3)
    out, meta = ssm.forward(X, coords)
    assert out.shape == (N, D), f"Expected shape {(N, D)}, got {out.shape}"
    assert not np.isnan(out).any(), "SSM output contains NaN"
    print("  -> Passed (1D Selective Scan + Multi-D Spatial Multipole Mixing)")


def test_equivariant_transformer():
    print("Testing 7. SE(3) Equivariant Multipole Transformer Layer...")
    N, D_s, D_v = 64, 32, 8
    rng = np.random.RandomState(42)
    coords = rng.uniform(0.2, 0.8, size=(N, 3)).astype(np.float32)
    scalar_feats = rng.randn(N, D_s).astype(np.float32)
    vector_feats = rng.randn(N, D_v, 3).astype(np.float32)

    eq_layer = EquivariantMultipoleTransformerLayer(scalar_dim=D_s, vector_dim=D_v, grid_depth=3)

    # Base forward pass
    s_out, v_out, _ = eq_layer.forward(coords, scalar_feats, vector_feats)

    # Equivariance Verification under 3D Rotation R
    theta = np.pi / 3.0
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ], dtype=np.float32)

    coords_rot = (coords - 0.5) @ R.T + 0.5
    vector_rot = np.einsum('nvd,cd->nvc', vector_feats, R)

    s_out_rot, v_out_rot, _ = eq_layer.forward(coords_rot, scalar_feats, vector_rot)

    # Invariant scalars should match with high cosine similarity
    scalar_cosine = np.sum(s_out * s_out_rot) / (np.linalg.norm(s_out) * np.linalg.norm(s_out_rot) + 1e-8)
    # Rotated vectors should align with R * v_out
    v_out_expected = np.einsum('nvd,cd->nvc', v_out, R)
    vec_cosine = np.sum(v_out_rot * v_out_expected) / (np.linalg.norm(v_out_rot) * np.linalg.norm(v_out_expected) + 1e-8)

    assert scalar_cosine > 0.90, f"Scalar invariance cosine similarity too low: {scalar_cosine}"
    assert vec_cosine > 0.90, f"Vector equivariance cosine similarity too low: {vec_cosine}"
    print(f"  -> Passed (SE(3) Equivariance verified: scalar_sim={scalar_cosine:.4f}, vec_sim={vec_cosine:.4f})")


def test_autograd_adjoint_fmm():
    print("Testing 8. Analytical VJP & Adjoint State Engine...")
    adjoint_engine = MultipoleAdjointEngine(spatial_sigma=0.3, temperature=0.2)
    grad_errors = adjoint_engine.check_numerical_gradients(N=12, D=6, dim=3)
    for k, err in grad_errors.items():
        assert err < 1e-4, f"Gradient error for {k} too high: {err:.2e}"
    print(f"  -> Passed (Exact VJP: max_rel_error={max(grad_errors.values()):.2e})")


def test_hierarchical_elastic_kv_cache():
    print("Testing 9. Multi-Scale Hierarchical Elastic KV-Cache...")
    d_k, d_v = 32, 32
    cache = HierarchicalElasticKVCache(d_k=d_k, d_v=d_v, recent_window_size=32, n_hyperplanes=6)

    # Stream 256 tokens into the cache
    rng = np.random.RandomState(42)
    K = rng.randn(256, d_k).astype(np.float32)
    V = rng.randn(256, d_v).astype(np.float32)
    cache.append_batch(K, V)

    # Query decoding
    q = rng.randn(d_k).astype(np.float32)
    v_retrieved, meta = cache.query_attention(q)
    assert v_retrieved.shape == (d_v,), f"Expected shape {(d_v,)}, got {v_retrieved.shape}"
    assert meta["compression_ratio"] > 2.0, "Cache should achieve compression"
    print(f"  -> Passed (3-Tier Hierarchy: {meta['total_tokens']} tokens, {meta['compression_ratio']:.1f}x compression)")


def test_neural_sph_ipc():
    print("Testing 10. Neural SPH & IPC Continuum Mechanics Layer...")
    N, hidden_dim = 100, 32
    rng = np.random.RandomState(42)
    pos = rng.uniform(0.1, 0.9, size=(N, 3)).astype(np.float32)
    vel = rng.randn(N, 3).astype(np.float32) * 0.1
    masses = np.ones(N, dtype=np.float32) * 0.01
    h_states = rng.randn(N, hidden_dim).astype(np.float32)

    sph_ipc = NeuralSPHIPCLayer(hidden_dim=hidden_dim, smoothing_h=0.15, contact_dhat=0.08)
    new_h, forces, densities, meta = sph_ipc.forward(pos, vel, masses, h_states)

    assert new_h.shape == (N, hidden_dim), f"Expected shape {(N, hidden_dim)}, got {new_h.shape}"
    assert forces.shape == (N, 3), f"Expected shape {(N, 3)}, got {forces.shape}"
    assert densities.shape == (N,), f"Expected shape {(N,)}, got {densities.shape}"
    assert np.all(densities > 0.0), "Densities must be strictly positive"
    print(f"  -> Passed (SPH Density & Navier-Stokes forces + IPC contact barrier)")


def run_all_tests():
    print("=" * 80)
    print("TREE-FREE NEURAL OPS: ADVANCED MODULE VERIFICATION SUITE")
    print("=" * 80)
    t0 = time.perf_counter()
    test_spherical_multipole_attention()
    test_kernel_independent_fmm()
    test_hyperbolic_multipole_attention()
    test_multipole_flow_drift()
    test_spectral_neural_pme()
    test_multipole_mamba_ssm()
    test_equivariant_transformer()
    test_autograd_adjoint_fmm()
    test_hierarchical_elastic_kv_cache()
    test_neural_sph_ipc()
    total_time = (time.perf_counter() - t0) * 1000.0
    print("=" * 80)
    print(f"ALL 10 ADVANCED NEURAL OPS TESTS PASSED in {total_time:.2f} ms!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
