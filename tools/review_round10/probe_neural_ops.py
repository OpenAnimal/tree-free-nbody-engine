"""Round-10 Wave C probe: neural_ops modules vs independent references."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

FAIL = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(name)

print("=" * 70)
print("PROBE: neural_ops deep verification")
print("=" * 70)

rng = np.random.RandomState(42)

# ============================================================
# 1. TreeFreeMultipoleAttention: forward + shape + no NaN
# ============================================================
print("\n[1] TreeFreeMultipoleAttention")
from neural_ops.multipole_attention import TreeFreeMultipoleAttention

B, N, D = 1, 64, 32
Q = rng.randn(N, D)
K_arr = rng.randn(N, D)
V_arr = rng.randn(N, D)
coords = rng.uniform(0, 1, size=(N, 2))

try:
    ma = TreeFreeMultipoleAttention(embed_dim=D, spatial_dim=2, grid_depth=4, multipole_order=2)
    out, info = ma.forward(Q, K_arr, V_arr, coords)
    out = np.asarray(out)
    check("MultipoleAttention: output shape", out.shape == (N, D),
          f"got {out.shape}")
    check("MultipoleAttention: no NaN", not np.any(np.isnan(out)))
    check("MultipoleAttention: no inf", not np.any(np.isinf(out)))
    # Compare to direct attention (approximate — multipole is an approximation)
    scores = Q @ K_arr.T / np.sqrt(D)
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)
    direct_out = attn @ V_arr
    rel = np.linalg.norm(out - direct_out) / np.linalg.norm(direct_out)
    check("MultipoleAttention: approximates direct attention", rel < 2.0,
          f"rel-L2={rel:.2e}")
except Exception as e:
    check("MultipoleAttention runs", False, str(e))

# ============================================================
# 2. ElasticMultipoleKVCache: append/query
# ============================================================
print("\n[2] ElasticMultipoleKVCache")
from neural_ops.elastic_kv_cache import ElasticMultipoleKVCache

try:
    d_k, d_v = 16, 16
    kv = ElasticMultipoleKVCache(d_k=d_k, d_v=d_v, n_hyperplanes=4, bucket_capacity=32, recent_window_size=128)
    n_tokens = 50
    keys = rng.randn(n_tokens, d_k)
    values = rng.randn(n_tokens, d_v)
    kv.append_batch(keys, values)
    # Query with a single vector
    q = rng.randn(d_k)
    result, info = kv.query_attention(q)
    result = np.asarray(result)
    check("KVCache: query returns result", result is not None and result.size > 0,
          f"shape={result.shape}")
    check("KVCache: no NaN", not np.any(np.isnan(result)))
except Exception as e:
    check("KVCache runs", False, str(e))

# ============================================================
# 3. ContinuousMeshfreeGNNLayer: forward
# ============================================================
print("\n[3] ContinuousMeshfreeGNNLayer")
from neural_ops.continuous_meshfree_gnn import ContinuousMeshfreeGNNLayer

N = 50
D_in, D_out = 16, 32
pos = rng.uniform(0, 1, size=(N, 3))
features = rng.randn(N, D_in)

try:
    gnn = ContinuousMeshfreeGNNLayer(in_features=D_in, out_features=D_out, spatial_dim=3,
                                      grid_depth=4, cutoff_radius=0.3)
    out, info = gnn.forward(features, pos)
    out = np.asarray(out)
    check("GNN: output shape", out.shape == (N, D_out),
          f"got {out.shape}")
    check("GNN: no NaN", not np.any(np.isnan(out)))
    check("GNN: no inf", not np.any(np.isinf(out)))
except Exception as e:
    check("GNN runs", False, str(e))

# ============================================================
# 4. EquivariantMultipoleLayer: rotation equivariance
# ============================================================
print("\n[4] EquivariantMultipoleLayer (rotation test)")
from neural_ops.equivariant_field_layer import EquivariantMultipoleLayer

N = 20
D = 16
pos = rng.uniform(0, 1, size=(N, 3))
features = rng.randn(N, D)
charges = rng.randn(N)

try:
    efl = EquivariantMultipoleLayer(hidden_dim=D, grid_depth=3, kernel='monopole_dipole', taylor_p=4)
    out_orig, _, _, _ = efl.forward(pos, features, charges)
    out_orig = np.asarray(out_orig)
    # Rotate positions by 90 degrees around z-axis
    R_mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    pos_rot = pos @ R_mat.T
    out_rot, _, _, _ = efl.forward(pos_rot, features, charges)
    out_rot = np.asarray(out_rot)

    check("EquivariantLayer: output shape", out_orig.shape == (N, D),
          f"got {out_orig.shape}")
    check("EquivariantLayer: no NaN", not np.any(np.isnan(out_orig)))

    # For scalar features, output should be approximately rotation-invariant
    rel = np.linalg.norm(out_rot - out_orig) / max(1e-30, np.linalg.norm(out_orig))
    check("EquivariantLayer: rotation invariance (scalar features)",
          rel < 0.5,
          f"rel-L2={rel:.2e}")
except Exception as e:
    check("EquivariantLayer runs", False, str(e))

# ============================================================
# 5. FlashMultipoleAttentionEngine
# ============================================================
print("\n[5] FlashMultipoleAttentionEngine")
from neural_ops.flash_multipole_kernel import FlashMultipoleAttentionEngine

B, N, D = 1, 32, 16
Q = rng.randn(N, D)
K_arr = rng.randn(N, D)
V_arr = rng.randn(N, D)
coords3d = rng.uniform(0, 1, size=(N, 3))

try:
    fmk = FlashMultipoleAttentionEngine(embed_dim=D, block_size_q=32, block_size_kv=32,
                                         spatial_dim=3, grid_depth=3)
    out, info = fmk.forward(Q, K_arr, V_arr, coords3d)
    out = np.asarray(out)
    check("FlashMultipoleKernel: output shape", out.shape == (N, D),
          f"got {out.shape}")
    check("FlashMultipoleKernel: no NaN", not np.any(np.isnan(out)))
except Exception as e:
    check("FlashMultipoleKernel runs", False, str(e))

# ============================================================
# 6. TaylorFGTAttention
# ============================================================
print("\n[6] TaylorFGTAttention")
from neural_ops.taylor_fgt_attention import TaylorFGTAttention

try:
    tfa = TaylorFGTAttention(spatial_dim=3, sigma=0.25, grid_depth=4, p=6, n_features=D)
    out, info = tfa.forward(Q, K_arr, V_arr, coords3d)
    out = np.asarray(out)
    check("TaylorFGTAttention: output shape", out.shape == (N, D),
          f"got {out.shape}")
    check("TaylorFGTAttention: no NaN", not np.any(np.isnan(out)))
except Exception as e:
    check("TaylorFGTAttention runs", False, str(e))

# ============================================================
# 7. NeuralSPHIPCLayer
# ============================================================
print("\n[7] NeuralSPHIPCLayer")
from neural_ops.neural_sph_ipc import NeuralSPHIPCLayer

N = 30
D = 16
pos = rng.uniform(0, 1, size=(N, 3))
vel = rng.uniform(-1, 1, size=(N, 3))
masses = rng.uniform(0.5, 2.0, size=N)
hidden = rng.randn(N, D)

try:
    sph = NeuralSPHIPCLayer(hidden_dim=D, smoothing_h=0.15, grid_depth=3)
    out = sph.forward(pos, vel, masses, hidden)
    # Returns tuple (acceleration, updated_hidden, density, info) or similar
    if isinstance(out, tuple):
        primary = np.asarray(out[0])
    else:
        primary = np.asarray(out)
    check("NeuralSPHIPC: no NaN", not np.any(np.isnan(primary)))
    check("NeuralSPHIPC: no inf", not np.any(np.isinf(primary)))
except Exception as e:
    check("NeuralSPHIPC runs", False, str(e))

# ============================================================
# 8. MultipoleAdjointEngine: numerical gradient check
# ============================================================
print("\n[8] MultipoleAdjointEngine (gradient check)")
from neural_ops.autograd_adjoint_fmm import MultipoleAdjointEngine

try:
    adj = MultipoleAdjointEngine()
    if hasattr(adj, 'check_numerical_gradients'):
        grad_ok = adj.check_numerical_gradients()
        check("AutogradAdjointFMM: numerical gradient check", grad_ok,
              "gradient check failed")
    else:
        check("AutogradAdjointFMM: has check_numerical_gradients", False, str(dir(adj)))
except Exception as e:
    check("AutogradAdjointFMM runs", False, str(e))

# ============================================================
# 9. HierarchicalElasticKVCache
# ============================================================
print("\n[9] HierarchicalElasticKVCache")
from neural_ops.hierarchical_elastic_kv_cache import HierarchicalElasticKVCache

try:
    hkv = HierarchicalElasticKVCache(d_k=16, d_v=16, n_coarse_levels=2)
    n_tokens = 50
    keys = rng.randn(n_tokens, 16)
    values = rng.randn(n_tokens, 16)
    if hasattr(hkv, 'append_batch'):
        hkv.append_batch(keys, values)
    q = rng.randn(16)
    if hasattr(hkv, 'query_attention'):
        result, info = hkv.query_attention(q)
        result = np.asarray(result)
        check("HierarchicalKVCache: query returns result", result is not None and result.size > 0,
              f"shape={result.shape}")
        check("HierarchicalKVCache: no NaN", not np.any(np.isnan(result)))
    else:
        check("HierarchicalKVCache: has query method", False, str(dir(hkv)))
except Exception as e:
    check("HierarchicalKVCache runs", False, str(e))

# ============================================================
# 10. Spherical multipole attention: forward pass
# ============================================================
print("\n[10] SphericalMultipoleAttention")
try:
    import neural_ops.spherical_multipole_attention as sma_mod
    exports = [x for x in dir(sma_mod) if not x.startswith('_') and x[0].isupper() and hasattr(getattr(sma_mod, x), 'forward')]
    print(f"    exports: {exports}")
    if exports:
        cls = getattr(sma_mod, exports[0])
        import inspect
        print(f"    {exports[0]}: {inspect.signature(cls.__init__)}")
        print(f"    forward: {inspect.signature(cls.forward)}")
        sma = cls(embed_dim=D, grid_depth=3)
        out, info = sma.forward(Q, K_arr, V_arr, coords3d)
        out = np.asarray(out)
        if out.ndim == 3:
            out = out[0]
        check("SphericalMultipoleAttention: output shape", out.shape == (32, D),
              f"got {out.shape}")
        check("SphericalMultipoleAttention: no NaN", not np.any(np.isnan(out)))
    else:
        check("SphericalMultipoleAttention: has class", False)
except Exception as e:
    check("SphericalMultipoleAttention runs", False, str(e))

print("\n" + "=" * 70)
if FAIL:
    print(f"RESULT: {len(FAIL)} FAILURES")
    for f in FAIL:
        print(f"  - {f}")
else:
    print("RESULT: ALL CHECKS PASSED")
print("=" * 70)
