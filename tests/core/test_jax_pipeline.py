"""Round-7 task T-D4: test the assembled JAX flat FMM pipeline.

Cross-validates `jax_flat_fmm_evaluate` against (i) the dense direct
reference `jax_direct_nbody_reference`, (ii) `core.FastVectorizedFMM` on the
same data, (iii) a clustered scene with many UNOCCUPIED cells (catches the
raw-key / compact-rank index-space mixing regression), (iv) a JIT cache-hit
assertion (no recompilation for the same N / static args), (v) a real
finite-difference gradcheck through the pipeline, and (vi) a Round-11
machine-noise parity check of the CSR cell-list near field against a frozen
verbatim copy of the retired dense O(N^2) masked near-field implementation.

JAX is now installed in this environment, so every sub-test runs for real
(no more SKIP masquerading as PASS). x64 is enabled in
`core.jax_tree_free_fmm` so the rel-L2 <= 1e-6 acceptance is reachable in
complex128 roundoff.

Run:  python -X utf8 -m core.test_jax_pipeline
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.jax_tree_free_fmm import HAS_JAX

# Grid / order chosen so the adaptive FMM far-field truncation error is comfortably
# below the 1e-6 acceptance for the largest N tested. The near field is an
# exact masked direct sum, so the only approximation error is the M2L/L2P
# truncation, which is ~1e-9 at these settings (measured).
DEPTH = 16
ORDER = 10
TOL = 1e-6


if HAS_JAX:
    import math
    from functools import partial
    import jax
    import jax.numpy as jnp
    # segment_sum import parity with core.jax_tree_free_fmm (jax 0.10.2 has
    # no public jax.lax.segment_sum; see the audit note there).
    try:
        from jax.lax import segment_sum
    except ImportError:  # pragma: no cover - JAX >= 0.10 layout
        from jax._src.ops.scatter import segment_sum

    @partial(jax.jit, static_argnums=(2, 3))
    def _legacy_dense_flat_fmm(pos, q, depth=5, order=8, softening=0.0):
        """Verbatim freeze of `jax_flat_fmm_evaluate` as of Round-10 -- the
        dense O(N^2) masked near field that the Round-11 CSR cell-list near
        field replaced -- kept here as the parity oracle for sub-test (vi).

        The far field (steps 1-4) shares semantics with the live
        implementation; if the live far field ever changes, update this
        copy to match so sub-test (vi) keeps isolating near-field
        differences only."""
        N = pos.shape[0]
        grid_res = depth
        h = 1.0 / depth
        max_K = depth * depth
        ring = 2

        ix = jnp.clip((pos[:, 0] * grid_res).astype(jnp.int32), 0, grid_res - 1)
        iy = jnp.clip((pos[:, 1] * grid_res).astype(jnp.int32), 0, grid_res - 1)
        keys = iy * grid_res + ix
        key_mask = jnp.zeros(max_K, dtype=jnp.bool_).at[keys].set(True)
        cell_id = jnp.arange(max_K)
        cell_ix = cell_id % grid_res
        cell_iy = cell_id // grid_res
        centers = jnp.stack([
            (cell_ix.astype(jnp.float64) + 0.5) * h,
            (cell_iy.astype(jnp.float64) + 0.5) * h,
        ], axis=-1)
        centers_c = centers[:, 0] + 1j * centers[:, 1]

        z_pts = pos[:, 0] + 1j * pos[:, 1]
        dz_p = z_pts - centers_c[keys]
        a0 = segment_sum(q, keys, num_segments=max_K)
        powers = jnp.arange(1, order + 1)
        dz_pow = dz_p[:, None] ** powers[None, :]
        weighted = q[:, None] * dz_pow
        ak = -segment_sum(weighted, keys, num_segments=max_K) / powers
        M_all = jnp.concatenate([a0[:, None], ak], axis=1)
        M_all = M_all * key_mask[:, None]

        t_ix = cell_ix[:, None]
        t_iy = cell_iy[:, None]
        s_ix = cell_ix[None, :]
        s_iy = cell_iy[None, :]
        cheb = jnp.maximum(jnp.abs(s_ix - t_ix), jnp.abs(s_iy - t_iy))
        far_mask = (cheb > ring) & key_mask[None, :]

        def m2l_for_target(t_idx):
            s_idx = cell_id
            mask = far_mask[t_idx, :]
            delta = centers_c[t_idx] - centers_c[s_idx]
            delta_safe = jnp.where(mask, delta, 1.0 + 0.0j)
            a0_s = M_all[s_idx, 0]
            ak_s = M_all[s_idx, 1:order + 1]
            k_idx = jnp.arange(1, order + 1)
            c0 = a0_s * jnp.log(delta_safe) + \
                jnp.sum(ak_s / (delta_safe[:, None] ** k_idx[None, :]), axis=1)
            c_list = [c0]
            for l in range(1, order + 1):
                term = a0_s * ((-1.0) ** (l - 1)) / (l * (delta_safe ** l))
                for k in range(1, order + 1):
                    binom_factor = ((-1.0) ** l) * float(math.comb(k + l - 1, l))
                    term = term + binom_factor * ak_s[:, k - 1] / (delta_safe ** (k + l))
                c_list.append(term)
            L = jnp.stack(c_list, axis=1)
            return jnp.sum(L * mask[:, None], axis=0)

        L_all = jax.vmap(m2l_for_target)(cell_id)
        L_all = L_all * key_mask[:, None]

        def l2p_one(p_idx):
            c = keys[p_idx]
            dz = z_pts[p_idx] - centers_c[c]
            pw = jnp.arange(order + 1)
            phi = jnp.sum(L_all[c, :] * dz ** pw)
            return jnp.real(phi)

        far_pot = jax.vmap(l2p_one)(jnp.arange(N))

        # Legacy step 5: dense (N, N) masked near field (the O(N^2) form).
        c_ix_p = keys[:, None] % grid_res
        c_iy_p = keys[:, None] // grid_res
        c_ix_q = keys[None, :] % grid_res
        c_iy_q = keys[None, :] // grid_res
        cheb_nn = jnp.maximum(jnp.abs(c_ix_p - c_ix_q),
                              jnp.abs(c_iy_p - c_iy_q))  # (N, N)
        near_mask = (cheb_nn <= ring) & (jnp.eye(N, dtype=jnp.bool_) == jnp.bool_(False))
        diff = pos[:, None, :] - pos[None, :, :]         # (N, N, 2)
        r_sq = jnp.sum(diff ** 2, axis=-1) + (softening ** 2)
        r_sq_safe = jnp.where(near_mask, r_sq, 1.0)
        r = jnp.sqrt(r_sq_safe)
        near_pot = jnp.sum(q[None, :] * jnp.log(r) * near_mask, axis=1)

        return far_pot + near_pot


def _rel_l2(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) /
                 np.maximum(1e-30, np.linalg.norm(np.asarray(b))))


def test_flat_fmm_vs_direct():
    """(i) Cross-validate jax_flat_fmm_evaluate vs jax_direct_nbody_reference
    at N=2000 and N=8000 (rel-L2 <= 1e-6)."""
    if not HAS_JAX:
        print("SKIP: JAX not installed (test_jax_pipeline)")
        return True
    from core.jax_tree_free_fmm import jax_flat_fmm_evaluate, jax_direct_nbody_reference
    import jax.numpy as jnp

    rng = np.random.RandomState(42)
    for N in [2000, 8000]:
        pos = rng.uniform(0.05, 0.95, size=(N, 2)).astype(np.float64)
        q = rng.uniform(-1.0, 1.0, size=N).astype(np.float64)
        pos_j = jnp.array(pos)
        q_j = jnp.array(q)

        pot_direct = jax_direct_nbody_reference(pos_j, q_j, softening=0.0)
        pot_flat = jax_flat_fmm_evaluate(pos_j, q_j, depth=DEPTH, order=ORDER,
                                         softening=0.0)
        rel = _rel_l2(pot_flat, pot_direct)
        print(f"  (i) N={N}: rel-L2 vs direct = {rel:.4e} (target < {TOL})")
        assert rel < TOL, f"jax_flat_fmm rel-L2 {rel} >= {TOL} at N={N}"
    return True


def test_flat_fmm_vs_fast_vectorized():
    """(ii) Cross-validate vs core.FastVectorizedFMM on the same data
    (rel-L2 <= 1e-6). Both methods must independently agree with the direct
    sum, so they must agree with each other to well under tolerance."""
    if not HAS_JAX:
        print("SKIP: JAX not installed (test_jax_pipeline)")
        return True
    from core.jax_tree_free_fmm import jax_flat_fmm_evaluate
    from core import FastVectorizedFMM
    import jax.numpy as jnp

    rng = np.random.RandomState(11)
    N = 2000
    pos = rng.uniform(0.05, 0.95, size=(N, 2)).astype(np.float64)
    q = rng.uniform(-1.0, 1.0, size=N).astype(np.float64)

    pot_jax = np.asarray(jax_flat_fmm_evaluate(
        jnp.array(pos), jnp.array(q), depth=DEPTH, order=ORDER, softening=0.0))
    # FastVectorizedFMM(depth=d) uses grid_res = 2**d; depth=5 -> 32 cells/side,
    # order=12 keeps its ring-1 far field well under 1e-6.
    fmm = FastVectorizedFMM(depth=5, order=12, softening=0.0)
    pot_vec = fmm.evaluate(pos, q)

    rel = _rel_l2(pot_jax, pot_vec)
    print(f"  (ii) N={N}: rel-L2 vs FastVectorizedFMM = {rel:.4e} "
          f"(target < {TOL})")
    assert rel < TOL, f"jax vs FastVectorizedFMM rel-L2 {rel} >= {TOL}"
    return True


def test_flat_fmm_unoccupied_cells():
    """(iii) A scene with UNOCCUPIED cells (clustered data at depth>=8) must
    still match the direct reference to rel-L2 <= 1e-6. This catches the
    raw-key / compact-rank index-space mixing regression: the old pipeline
    computed P2M moments about the wrong cell centers whenever any cell was
    unoccupied."""
    if not HAS_JAX:
        print("SKIP: JAX not installed (test_jax_pipeline)")
        return True
    from core.jax_tree_free_fmm import jax_flat_fmm_evaluate, jax_direct_nbody_reference
    import jax.numpy as jnp

    rng = np.random.RandomState(7)
    cluster_centers = rng.uniform(0.2, 0.8, size=(8, 2))
    N = 3000
    member = rng.randint(0, 8, size=N)
    pos = cluster_centers[member] + rng.normal(0.0, 0.015, size=(N, 2))
    pos = np.clip(pos, 0.01, 0.99).astype(np.float64)
    q = rng.uniform(-1.0, 1.0, size=N).astype(np.float64)

    # depth=16 -> 256 cells; 8 tight clusters occupy far fewer -> many empty.
    pot_direct = jax_direct_nbody_reference(jnp.array(pos), jnp.array(q),
                                            softening=0.0)
    pot_flat = jax_flat_fmm_evaluate(jnp.array(pos), jnp.array(q),
                                     depth=DEPTH, order=ORDER, softening=0.0)
    rel = _rel_l2(pot_flat, pot_direct)
    occupied = len(np.unique((np.clip(pos[:,0]*DEPTH,0,DEPTH-1).astype(int) +
                              np.clip(pos[:,1]*DEPTH,0,DEPTH-1).astype(int)*DEPTH)))
    print(f"  (iii) clustered N={N} depth={DEPTH} ({occupied}/{DEPTH*DEPTH} "
          f"cells occupied): rel-L2 = {rel:.4e} (target < {TOL})")
    assert rel < TOL, f"unoccupied-cells rel-L2 {rel} >= {TOL}"
    assert occupied < DEPTH * DEPTH, "scene must actually have empty cells"
    return True


def test_jit_cache_hit():
    """(iv) Call jax_flat_fmm_evaluate twice with identical N and static
    args; the second call must hit the XLA compilation cache (no
    recompilation). Asserted by timing: the cached call is orders of
    magnitude faster than the compiling call."""
    if not HAS_JAX:
        print("SKIP: JAX not installed (test_jax_pipeline)")
        return True
    from core.jax_tree_free_fmm import jax_flat_fmm_evaluate
    import jax.numpy as jnp

    rng = np.random.RandomState(7)
    pos = jnp.array(rng.uniform(0.05, 0.95, size=(100, 2)))
    q = jnp.array(rng.uniform(-1, 1, size=100))
    # Use a distinct (depth, order) so this compiles fresh regardless of the
    # settings used by the accuracy sub-tests above.
    t0 = time.perf_counter()
    _ = jax_flat_fmm_evaluate(pos, q, depth=6, order=6).block_until_ready()
    t_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = jax_flat_fmm_evaluate(pos, q, depth=6, order=6).block_until_ready()
    t_cached = time.perf_counter() - t0
    ratio = t_compile / max(t_cached, 1e-9)
    print(f"  (iv) JIT compile {t_compile*1000:.1f} ms, cached {t_cached*1000:.3f} ms "
          f"(speedup {ratio:.1f}x; target >= 5x => cache hit)")
    assert ratio >= 5.0, (
        f"expected cache hit (2nd call much faster), got compile={t_compile*1000:.1f}ms "
        f"cached={t_cached*1000:.3f}ms (speedup {ratio:.1f}x)")
    return True


def test_gradcheck():
    """(v) Real finite-difference gradcheck through the pipeline.

    Perturbs each charge entry one at a time with central differences and
    compares to jax.grad. The previous version computed `fd` but never used
    it, computed `grad_fd` with jax.grad identically to `grad_jax` (so it
    compared JAX to itself), and had no assert -- it was vacuous.
    """
    if not HAS_JAX:
        print("SKIP: JAX not installed (test_jax_pipeline gradcheck)")
        return True
    import jax
    from core.jax_tree_free_fmm import jax_flat_fmm_evaluate
    import jax.numpy as jnp

    N = 48
    rng = np.random.RandomState(42)
    pos = jnp.array(rng.uniform(0.1, 0.9, size=(N, 2)))
    q = jnp.array(rng.uniform(-1.0, 1.0, size=N))

    def loss(q_arg):
        return jnp.sum(jax_flat_fmm_evaluate(pos, q_arg, depth=6, order=4))

    grad_jax = np.array(jax.grad(loss)(q))
    # Central finite differences: perturb one q entry at a time.
    eps = 1e-5
    q_np = np.array(q)
    grad_fd = np.zeros(N, dtype=np.float64)
    for i in range(N):
        qp = q_np.copy()
        qp[i] += eps
        qm = q_np.copy()
        qm[i] -= eps
        grad_fd[i] = (float(loss(jnp.array(qp))) -
                      float(loss(jnp.array(qm)))) / (2.0 * eps)
    denom = np.maximum(1e-12, np.linalg.norm(grad_fd))
    rel = float(np.linalg.norm(grad_jax - grad_fd) / denom)
    max_rel = float(np.max(np.abs(grad_jax - grad_fd) /
                           np.maximum(1e-12, np.abs(grad_fd))))
    print(f"  (v) gradcheck: rel-L2 = {rel:.4e}, max rel = {max_rel:.4e} "
          f"(target < 1e-5)")
    assert max_rel < 1e-5, (
        f"gradcheck max rel error {max_rel:.3e} >= 1e-5 "
        f"(rel-L2 {rel:.3e})")
    return True


def test_flat_fmm_cell_list_parity_vs_dense():
    """(vi) Round-11: the CSR cell-list (spatial-hash) near field must agree
    with the retired dense O(N^2) masked near field at MACHINE-NOISE level --
    both compute the identical pair partition (ring-2 Chebyshev neighborhood,
    self pairs excluded) with the identical kernel, so any deviation beyond
    f64 summation-order noise is a real partitioning bug.

    Scenes: a uniform cloud and the 8-cluster scene (a few cells hold ~10x
    the mean occupancy, exercising multi-block while_loop sweeps of long
    CSR runs); both with softening 0.0 and 1e-4. Errors are normalized by
    the reference magnitude scale (per-element relative error is ill-defined
    where the potential crosses zero)."""
    if not HAS_JAX:
        print("SKIP: JAX not installed (test_jax_pipeline)")
        return True
    from core.jax_tree_free_fmm import jax_flat_fmm_evaluate

    rng = np.random.RandomState(2026)
    scenes = [
        ("uniform", rng.uniform(0.05, 0.95, size=(600, 2)).astype(np.float64),
         rng.uniform(-1.0, 1.0, size=600).astype(np.float64)),
    ]
    rng = np.random.RandomState(7)
    cc = rng.uniform(0.2, 0.8, size=(8, 2))
    member = rng.randint(0, 8, size=800)
    pos = np.clip(cc[member] + rng.normal(0.0, 0.015, size=(800, 2)),
                  0.01, 0.99).astype(np.float64)
    scenes.append(("clustered", pos,
                   rng.uniform(-1.0, 1.0, size=800).astype(np.float64)))

    for name, pos, q in scenes:
        for softening in (0.0, 1e-4):
            pos_j = jnp.asarray(pos)
            q_j = jnp.asarray(q)
            pot_new = np.asarray(jax_flat_fmm_evaluate(
                pos_j, q_j, depth=DEPTH, order=ORDER, softening=softening))
            pot_old = np.asarray(_legacy_dense_flat_fmm(
                pos_j, q_j, depth=DEPTH, order=ORDER, softening=softening))
            d = np.abs(pot_new - pot_old)
            scale = float(np.max(np.abs(pot_old)))
            max_rel = float(np.max(d) / max(scale, 1e-30))
            rel_l2 = _rel_l2(pot_new, pot_old)
            print(f"  (vi) {name} N={len(pos)} softening={softening}: "
                  f"max|d|/max|ref| = {max_rel:.3e}, rel-L2 = {rel_l2:.3e} "
                  f"(target < 1e-12)")
            assert max_rel < 1e-12 and rel_l2 < 1e-12, (
                f"cell-list near field deviates from dense oracle beyond "
                f"machine noise ({name}, softening={softening}): "
                f"max_rel={max_rel:.3e}, rel-L2={rel_l2:.3e}")
    return True


def test_pos_gradcheck():
    """(vii) Round-11: finite-difference gradcheck of the POSITION gradient
    through the CSR near field's hand-written custom_vjp transpose rule
    (dL/dx = t*A(q) + q*A(t); section 3b of core/jax_tree_free_fmm.py).
    The charge-gradient path is covered by (v); this pins the pos rule of
    the analytic transpose against central differences. Coordinates are
    kept clear of cell boundaries so the FD never straddles a bin edge
    (the near/far pair partition is discontinuous there by design)."""
    if not HAS_JAX:
        print("SKIP: JAX not installed (test_jax_pipeline gradcheck)")
        return True
    import jax
    from core.jax_tree_free_fmm import jax_flat_fmm_evaluate

    N = 40
    d_t, o_t = 6, 4
    rng = np.random.RandomState(23)
    pos0 = rng.uniform(0.1, 0.9, size=(N, 2))
    # Redraw any particle sitting within 1e-3 of a cell boundary (in units
    # of the cell size) so no FD probe crosses a bin edge.
    h = 1.0 / d_t
    for _ in range(200):
        frac = np.abs((pos0 / h) - np.round(pos0 / h))
        bad = np.min(frac, axis=1) < 1e-3
        if not np.any(bad):
            break
        pos0[bad] = rng.uniform(0.1, 0.9, size=(int(bad.sum()), 2))
    pos_j = jnp.asarray(pos0)
    q_j = jnp.asarray(rng.uniform(-1.0, 1.0, size=N))

    def loss(p):
        return jnp.sum(jax_flat_fmm_evaluate(p, q_j, depth=d_t, order=o_t))

    g = np.array(jax.grad(loss)(pos_j))
    eps = 1e-6
    scale = float(np.max(np.abs(g)))
    worst = 0.0
    coords = [(i, ax) for i in range(0, N, 5) for ax in (0, 1)]
    for (i, ax) in coords:
        pp = pos0.copy()
        pp[i, ax] += eps
        pm = pos0.copy()
        pm[i, ax] -= eps
        fd = (float(loss(jnp.asarray(pp))) -
              float(loss(jnp.asarray(pm)))) / (2.0 * eps)
        denom = max(abs(fd), abs(g[i, ax]), 1e-4 * scale)
        worst = max(worst, abs(fd - g[i, ax]) / denom)
    print(f"  (vii) pos gradcheck over {len(coords)} coords: worst rel = "
          f"{worst:.3e} (target < 1e-5)")
    assert worst < 1e-5, (
        f"pos gradcheck worst rel error {worst:.3e} >= 1e-5 "
        f"(custom_vjp transpose rule may be wrong)")
    return True


def main():
    print("=" * 70)
    print("test_jax_pipeline: JAX flat FMM (Round-7 task T-D4)")
    print("=" * 70)
    results = [
        test_flat_fmm_vs_direct(),
        test_flat_fmm_vs_fast_vectorized(),
        test_flat_fmm_unoccupied_cells(),
        test_jit_cache_hit(),
        test_gradcheck(),
        test_flat_fmm_cell_list_parity_vs_dense(),
        test_pos_gradcheck(),
    ]
    if all(results):
        if HAS_JAX:
            print("\nAll jax_pipeline tests PASS")
        else:
            # Distinguishable final line so tools/run_all.py classifies the
            # no-JAX case as SKIP without the word "SKIP" leaking into the
            # passing banner (which would misclassify a real PASS as SKIP).
            print("\nSKIP: JAX not installed (test_jax_pipeline)")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
