"""
Tri-backend parity & timing harness (`tests/neural_ops/test_backend_parity.py`).
===============================================================================
Validates that the torch / jax acceleration drop-ins in `neural_ops/_accel.py`
match the NumPy reference path to float32 round-off, and reports per-backend
timing so the JIT crossover is visible.

Covers the three flagship modules wired up in Round 9's accel drop-in:
- `TreeFreeMultipoleAttention` (per-bucket near P2P + far M2L + dipole)
- `ContinuousMeshfreeGNNLayer` (rbf / wendland / inverse spatial kernels)
- `ElasticMultipoleKVCache`    (streaming cache query: 3-tier softmax)

Design notes (see `neural_ops/_accel.py`):
- Spatial bucketing is CPU-only (funnel hash); only the per-bucket dense math
  moves to the device backend, so `active_clusters` / `exact_tokens_evaluated`
  must match the numpy path exactly (asserted).
- torch.compile falls back from `inductor` -> `aot_eager` when Triton is
  missing (the default on this Windows + torch 2.6 setup). Override with
  NEURAL_OPS_TORCH_BACKEND=inductor|aot_eager|cudagraphs|eager.
- jax runs on CPU here (aivenv has the CPU jaxlib wheel); point at a CUDA
  jaxlib (e.g. the WSL env) for GPU JAX timing.

Run:  python tests/neural_ops/test_backend_parity.py
      pytest tests/neural_ops/test_backend_parity.py -q
"""

import sys
import os
import time
import numpy as np

# Ensure the repo root (parent of tests/) is importable.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from neural_ops import _accel as A
from neural_ops.multipole_attention import TreeFreeMultipoleAttention
from neural_ops.continuous_meshfree_gnn import ContinuousMeshfreeGNNLayer
from neural_ops.elastic_kv_cache import ElasticMultipoleKVCache

# Float32 round-off parity gate. The accel backends do identical math to the
# numpy reference (same op order per bucket); differences are only float32
# accumulation noise, so this is tight.
REL_L2_TOL = 1e-4
COSINE_FLOOR = 0.9999

_AVAILABLE = [b for b in ("numpy", "torch", "jax") if A.has_backend(b)]


def _rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def _cosine(a, b):
    return float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _time_call(fn, repeats=3):
    """Best-of-repeats wall time (ms). Excludes JIT compile cost: the first
    call is a warmup that triggers torch.compile / jax.jit specialization."""
    fn()  # warmup (compile)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


# ---------------------------------------------------------------------------
# TreeFreeMultipoleAttention
# ---------------------------------------------------------------------------
def _mpa_inputs(seed=42, N=600, D=32):
    rng = np.random.RandomState(seed)
    coords = rng.uniform(0.05, 0.95, size=(N, 2))
    Q = rng.randn(N, D).astype(np.float32)
    K = rng.randn(N, D).astype(np.float32)
    V = rng.randn(N, D).astype(np.float32)
    return Q, K, V, coords


def test_multipole_attention_parity():
    print("\n[PARITY 1] TreeFreeMultipoleAttention (N=600, D=32, grid_depth=3)")
    Q, K, V, coords = _mpa_inputs()
    ref, meta_ref = TreeFreeMultipoleAttention(embed_dim=32, spatial_dim=2, grid_depth=3, backend="numpy").forward(Q, K, V, coords)
    print(f"  numpy  ref  active={meta_ref['active_clusters']} near={meta_ref['total_near_evals']} far={meta_ref['total_far_evals']}")
    rows = []
    for b in _AVAILABLE:
        if b == "numpy":
            continue
        for jit in (False, True):
            layer = TreeFreeMultipoleAttention(embed_dim=32, spatial_dim=2, grid_depth=3, backend=b, jit=jit)
            out, meta = layer.forward(Q, K, V, coords)
            rel = _rel_l2(out, ref)
            cos = _cosine(out, ref)
            ms = _time_call(lambda: layer.forward(Q, K, V, coords))
            rows.append((b, jit, rel, cos, ms, meta["active_clusters"]))
            print(f"  {b:5s} jit={str(jit):5s} rel-L2={rel:.3e} cosine={cos:.6f} {ms:7.2f}ms active={meta['active_clusters']}")
            assert rel < REL_L2_TOL, f"{b} jit={jit} rel-L2 {rel} > {REL_L2_TOL}"
            assert cos > COSINE_FLOOR, f"{b} jit={jit} cosine {cos} < {COSINE_FLOOR}"
            assert meta["active_clusters"] == meta_ref["active_clusters"]
            assert meta["total_near_evals"] == meta_ref["total_near_evals"]
            assert meta["total_far_evals"] == meta_ref["total_far_evals"]
    assert not np.isnan(ref).any()
    print("  -> PASS")


# ---------------------------------------------------------------------------
# ContinuousMeshfreeGNNLayer
# ---------------------------------------------------------------------------
def _gnn_inputs(seed=7, N=600, F=32):
    rng = np.random.RandomState(seed)
    coords = rng.uniform(0.05, 0.95, size=(N, 3))
    feats = rng.randn(N, F).astype(np.float32)
    return feats, coords


def test_continuous_meshfree_gnn_parity():
    print("\n[PARITY 2] ContinuousMeshfreeGNNLayer (N=600, F=32, 3D, grid_depth=3)")
    feats, coords = _gnn_inputs()
    for ktype in ("rbf", "wendland", "inverse"):
        ref, meta_ref = ContinuousMeshfreeGNNLayer(
            in_features=32, out_features=32, spatial_dim=3, grid_depth=3,
            cutoff_radius=0.2, kernel_type=ktype, backend="numpy").forward(feats, coords)
        print(f"  [{ktype:8s}] numpy ref active={meta_ref['active_clusters']}")
        for b in _AVAILABLE:
            if b == "numpy":
                continue
            for jit in (False, True):
                layer = ContinuousMeshfreeGNNLayer(
                    in_features=32, out_features=32, spatial_dim=3, grid_depth=3,
                    cutoff_radius=0.2, kernel_type=ktype, backend=b, jit=jit)
                out, meta = layer.forward(feats, coords)
                rel = _rel_l2(out, ref)
                cos = _cosine(out, ref)
                ms = _time_call(lambda: layer.forward(feats, coords))
                print(f"    {b:5s} jit={str(jit):5s} rel-L2={rel:.3e} cosine={cos:.6f} {ms:7.2f}ms active={meta['active_clusters']}")
                assert rel < REL_L2_TOL, f"{ktype} {b} jit={jit} rel-L2 {rel} > {REL_L2_TOL}"
                assert cos > COSINE_FLOOR
                assert meta["active_clusters"] == meta_ref["active_clusters"]
    print("  -> PASS")


# ---------------------------------------------------------------------------
# ElasticMultipoleKVCache
# ---------------------------------------------------------------------------
def _kv_inputs(seed=2024, N=800, d=32):
    rng = np.random.RandomState(seed)
    K_seq = rng.randn(N, d).astype(np.float32)
    V_seq = rng.randn(N, d).astype(np.float32)
    queries = rng.randn(8, d).astype(np.float32)
    return K_seq, V_seq, queries


def _kv_run(backend, jit, K_seq, V_seq, queries):
    c = ElasticMultipoleKVCache(d_k=32, d_v=32, n_hyperplanes=8,
                                bucket_capacity=16, recent_window_size=64,
                                backend=backend, jit=jit)
    c.append_batch(K_seq, V_seq)
    return [c.query_attention(q) for q in queries]


def test_elastic_kv_cache_parity():
    print("\n[PARITY 3] ElasticMultipoleKVCache (N=800 tokens, 8 queries, hp=8)")
    K_seq, V_seq, queries = _kv_inputs()
    ref = _kv_run("numpy", False, K_seq, V_seq, queries)
    print(f"  numpy  ref  q0 exact={ref[0][1]['exact_tokens_evaluated']} far={ref[0][1]['far_clusters_evaluated']} comp={ref[0][1]['compression_ratio']:.2f}")
    for b in _AVAILABLE:
        if b == "numpy":
            continue
        for jit in (False, True):
            got = _kv_run(b, jit, K_seq, V_seq, queries)
            rels = [_rel_l2(o, ro) for (o, _), (ro, _) in zip(got, ref)]
            coss = [_cosine(o, ro) for (o, _), (ro, _) in zip(got, ref)]
            # time the 8-query batch (warmup already done via the got run)
            t0 = time.perf_counter()
            _kv_run(b, jit, K_seq, V_seq, queries)
            ms = (time.perf_counter() - t0) * 1000.0
            print(f"  {b:5s} jit={str(jit):5s} rel-L2 max={max(rels):.3e} min={min(rels):.3e} cosine min={min(coss):.6f} {ms:7.2f}ms/8q")
            assert max(rels) < REL_L2_TOL, f"{b} jit={jit} rel-L2 {max(rels)} > {REL_L2_TOL}"
            assert min(coss) > COSINE_FLOOR
            for (o, m), (ro, m_ref) in zip(got, ref):
                assert m["exact_tokens_evaluated"] == m_ref["exact_tokens_evaluated"]
                assert m["far_clusters_evaluated"] == m_ref["far_clusters_evaluated"]
    print("  -> PASS")


def main():
    print("=" * 78)
    print("neural_ops tri-backend parity & timing harness")
    print("=" * 78)
    print("backends: " + A.status_line())
    print(f"available for this run: {_AVAILABLE}")
    print(f"parity gate: rel-L2 < {REL_L2_TOL}, cosine > {COSINE_FLOOR}")
    if "torch" in _AVAILABLE:
        print(f"torch.compile backend: {A._resolve_torch_compile_backend()} "
              f"(override via NEURAL_OPS_TORCH_BACKEND)")
    A.warn_jax_gpu_prealloc()
    tests = [
        test_multipole_attention_parity,
        test_continuous_meshfree_gnn_parity,
        test_elastic_kv_cache_parity,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  -> FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  -> ERROR: {type(e).__name__}: {str(e)[:200]}")
            failed += 1
    print("\n" + "=" * 78)
    print("ALL PASS" if failed == 0 else f"{failed} TEST(S) FAILED")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
