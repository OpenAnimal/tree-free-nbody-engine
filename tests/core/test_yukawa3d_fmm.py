"""Tests for core.yukawa3d_fmm.Yukawa3DFMM.

Test matrix (round-3 plan section 3.4):
  1. derivative-vs-FD guard (D_alpha vs central finite differences, |alpha|<=2)
  2. kappa -> 0 limit vs exact 1/r Coulomb direct (rel-L2 < 1e-6)
  3. accuracy vs direct on the app5-style clustered distribution, N=2000,
     rel-L2 < 1e-5 (the acceptance number; if unreachable after p=12, STOP)
  4. occupied-cell set in the funnel hash matches np.unique cell keys

Run standalone:  python -X utf8 -m core.test_yukawa3d_fmm
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.yukawa3d_fmm import (
    Yukawa3DFMM,
    derivative_fd_guard,
    toy_2cell_check,
    toy_2cell_check_forces,
)
from core.spatial_index import CellIndex


def _clustered3d(n=2000, seed=707):
    """3D clustered multi-scale distribution (the spirit of the core 2D
    clustered generator, lifted to 3D for the app5-style Yukawa test)."""
    rng = np.random.default_rng(seed)
    n1 = max(1, int(n * 0.20))
    n2 = max(1, int(n * 0.30))
    n3 = max(1, int(n * 0.40))
    nbg = max(0, n - (n1 + n2 + n3))
    c1 = rng.random((n1, 3)) * 0.10 + 0.10
    c2 = rng.random((n2, 3)) * 0.15 + 0.70
    c3 = rng.random((n3, 3)) * 0.30 + 0.40
    bg = rng.random((nbg, 3)) * 0.94 + 0.03 if nbg > 0 else np.empty((0, 3))
    pts = np.vstack([c1, c2, c3, bg]).astype(np.float64)
    pts = np.clip(pts, 0.01, 0.99)
    q = rng.uniform(-1.0, 1.0, size=len(pts)).astype(np.float64)
    return pts, q


def _direct_yukawa(pts, q, kappa):
    """Exact O(N^2) direct Yukawa / Coulomb sum (excludes self)."""
    N = len(pts)
    diff = pts[:, None, :] - pts[None, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=-1)) + 1e-12
    np.fill_diagonal(r, 1e18)
    return np.sum(q[None, :] * np.exp(-kappa * r) / r, axis=1)


# =====================================================================
# Test 1: derivative tensor vs finite differences
# =====================================================================

def test_derivative_fd_guard():
    assert derivative_fd_guard(kappa=1.0, p=8), "D_alpha FD guard failed (kappa=1)"
    assert derivative_fd_guard(kappa=2.0, p=8), "D_alpha FD guard failed (kappa=2)"
    print("test_derivative_fd_guard: PASS")


# =====================================================================
# Test 2: kappa -> 0 limit vs exact 1/r Coulomb direct
# =====================================================================

def test_kappa_zero_coulomb_limit():
    rng = np.random.default_rng(11)
    # Two well-separated clusters so the FMM far field is exercised and the
    # 1/r Coulomb singularity is not directly hit (particles within a cell
    # are handled by the exact near field).
    n = 400
    a = rng.random((n // 2, 3)) * 0.05 + 0.10
    b = rng.random((n - n // 2, 3)) * 0.05 + 0.80
    pts = np.vstack([a, b]).astype(np.float64)
    q = rng.uniform(-1.0, 1.0, size=len(pts)).astype(np.float64)

    fmm = Yukawa3DFMM(depth=5, p=8, kappa=0.0)
    pot_fmm = fmm.evaluate(pts, q)
    pot_exact = _direct_yukawa(pts, q, kappa=0.0)
    rel = np.linalg.norm(pot_fmm - pot_exact) / np.linalg.norm(pot_exact)
    print(f"test_kappa_zero_coulomb_limit: rel-L2 = {rel:.3e} (target < 1e-6)")
    assert rel < 1e-6, f"kappa=0 Coulomb limit rel-L2 {rel:.2e} >= 1e-6"


# =====================================================================
# Test 3: accuracy vs direct on app5-style clustered, N=2000, rel-L2 < 1e-5
# =====================================================================

def test_clustered_accuracy_n2000():
    pts, q = _clustered3d(n=2000, seed=707)
    pot_exact = _direct_yukawa(pts, q, kappa=1.0)
    # p=8 should reach ~1e-6 per the plan's convergence estimate.
    fmm = Yukawa3DFMM(depth=6, p=8, kappa=1.0)
    pot_fmm = fmm.evaluate(pts, q)
    rel = np.linalg.norm(pot_fmm - pot_exact) / np.linalg.norm(pot_exact)
    print(f"test_clustered_accuracy_n2000 (p=8): rel-L2 = {rel:.3e} (target < 1e-5)")
    if rel >= 1e-5:
        # Plan says: raise p to 10, then 12; if still failing, STOP and report.
        for p in (10, 12):
            fmm = Yukawa3DFMM(depth=6, p=p, kappa=1.0)
            pot_fmm = fmm.evaluate(pts, q)
            rel = np.linalg.norm(pot_fmm - pot_exact) / np.linalg.norm(pot_exact)
            print(f"  retry p={p}: rel-L2 = {rel:.3e}")
            if rel < 1e-5:
                break
    assert rel < 1e-5, f"clustered N=2000 rel-L2 {rel:.2e} >= 1e-5 (STOP per plan)"


# =====================================================================
# Test 4: occupied-cell set in the hash matches np.unique cell keys
# =====================================================================

def test_occupied_cell_set_matches_unique():
    pts, q = _clustered3d(n=500, seed=123)
    fmm = Yukawa3DFMM(depth=5, p=6, kappa=1.0)
    fmm.evaluate(pts, q)
    ci = fmm.cell_index
    # Re-derive the unique cell keys the way CellIndex.build does.
    grid_res = ci.grid_res
    ix = np.clip((pts[:, 0] * grid_res).astype(np.int64), 0, grid_res - 1)
    iy = np.clip((pts[:, 1] * grid_res).astype(np.int64), 0, grid_res - 1)
    iz = np.clip((pts[:, 2] * grid_res).astype(np.int64), 0, grid_res - 1)
    keys = np.zeros(len(pts), dtype=np.int64)
    for b in range(10):
        keys |= ((ix & (1 << b)) << (2 * b)) | ((iy & (1 << b)) << (2 * b + 1)) | ((iz & (1 << b)) << (2 * b + 2))
    unique_keys = np.unique(keys)
    hash_keys = set(int(k) for k in ci.occupied_keys())
    unique_set = set(int(k) for k in unique_keys)
    assert hash_keys == unique_set, (
        f"occupied cell set mismatch: hash has {len(hash_keys)} cells, "
        f"np.unique has {len(unique_set)}; symmetric diff size "
        f"{len(hash_keys.symmetric_difference(unique_set))}"
    )
    # And every hash probe of an occupied key resolves.
    for k in unique_keys:
        assert ci.cell_id(int(k)) is not None, f"occupied key {k} missing from hash"
    print(f"test_occupied_cell_set_matches_unique: PASS ({len(unique_set)} occupied cells)")


# =====================================================================
# Toy 2-cell check (mandatory sign-convention guard before scaling up)
# =====================================================================

def test_toy_2cell_check():
    assert toy_2cell_check(kappa=1.0, p=8), "2-cell toy check failed (sign convention)"


# =====================================================================
# Test 5: round-5 p-floor regression (root cause: reference +1e-6 bias)
# =====================================================================

def test_yukawa3d_pfloor_regression():
    """Pin the round-5 task 5.2 root-cause fix.

    ROOT CAUSE: the app5 direct reference `_direct_debye_huckel` added 1e-6
    to EVERY pairwise distance, introducing a systematic ~6.27e-5 rel-L2
    bias that was independent of p and was wrongly attributed to "ring-2
    near field + f64 round-off".  The FMM computes the true (unregularized)
    kernel and converges geometrically with p.

    This test asserts:
      (a) the FMM vs the TRUE direct reference (no +1e-6) decays
          geometrically and drops below 1e-8 at p=12;
      (b) the +1e-6-regularized reference disagrees with the true reference
          by ~6.3e-5 (the old floor), confirming the root cause.
    """
    from apps.app5_benchmark_variants import _protein, _direct_debye_huckel
    coords, charges = _protein(n_atoms=2000, seed=42)
    kappa = 2.0
    # True direct reference (the fixed _direct_debye_huckel: diagonal-only).
    ref_true = _direct_debye_huckel(coords, charges, kappa=kappa)
    ref_norm = np.linalg.norm(ref_true)
    # The buggy +1e-6-everywhere reference, reconstructed here to pin the
    # root cause independently of the fix in _direct_debye_huckel.
    diff = coords[:, None, :] - coords[None, :, :]
    r_reg = np.linalg.norm(diff, axis=-1) + 1e-6
    np.fill_diagonal(r_reg, 1e9)
    ref_reg = np.sum(charges[None, :] * np.exp(-kappa * r_reg) / r_reg, axis=1)
    bias = np.linalg.norm(ref_reg - ref_true) / ref_norm
    print(f"  +1e-6-regularized reference bias vs true = {bias:.3e} "
          f"(the old ~6.3e-5 floor)")
    assert 5e-5 < bias < 8e-5, (
        f"regularized-reference bias {bias:.2e} not in the expected "
        f"~6.3e-5 band; root-cause characterization drifted")

    # FMM convergence vs the TRUE reference: must drop below 1e-8 at p=12.
    rels = {}
    for p in (6, 8, 10, 12):
        fmm = Yukawa3DFMM(depth=6, p=p, kappa=kappa)
        est = fmm.evaluate(coords, charges)
        rel = np.linalg.norm(est - ref_true) / ref_norm
        rels[p] = rel
        print(f"  p={p}: rel-L2 vs true = {rel:.3e}")
    assert rels[12] < 1e-8, (
        f"FMM rel-L2 at p=12 = {rels[12]:.2e} >= 1e-8; the p-floor was NOT "
        f"caused by the reference and the root-cause analysis is wrong")
    # Geometric decay sanity: each +2 in p should reduce rel by ~10x.
    assert rels[10] < rels[8] < rels[6], (
        "FMM error is not monotonically decreasing with p")
    print("test_yukawa3d_pfloor_regression: PASS")


def test_evaluate_forces_2cell():
    """Round-7 task T-D6: evaluate_forces vs exact direct on 2-cell toy.

    Uses the mandatory 2-cell sign-convention guard pattern (the plan says:
    "the toy 2-cell check pattern (extend it to forces) is the mandatory
    guard before any claim").
    """
    assert toy_2cell_check_forces(kappa=1.0, p=8), "2-cell force toy check failed"


# =====================================================================
# Test 6: build_operator / evaluate_prebuilt accuracy + speedup (T-D7)
# =====================================================================

def test_build_operator_parity_and_speedup():
    """Round-7 task T-D7: build_operator + evaluate_prebuilt split.

    (a) Accuracy (meaningful): evaluate_prebuilt is cross-validated against
        an INDEPENDENT direct-sum reference (`_direct_yukawa`, the O(N^2)
        kernel in this file) at engine tolerance (rel-L2 < 1e-5, the same
        bar the clustered-accuracy test uses). The previous version of this
        test compared evaluate() against build_operator()+evaluate_prebuilt()
        -- but evaluate() now delegates to exactly that pair, so the
        comparison was the code against itself and could never fail. An
        external audit already verified bit-parity vs the old monolithic
        code; this guard is what actually catches a regression in the
        operator math.
    (b) Speedup: for 8 repeated charge vectors at N=8k, reusing the built
        operator must be faster than re-running evaluate() 8 times (the
        charge-independent cell index, near-field kernel values, and M2L
        D_gamma tensors are built once instead of 8 times).
    """
    import time
    rng = np.random.default_rng(2024)
    # (a) Accuracy vs the independent direct reference on a clustered scene
    #     (exercises the far field, not just the trivial near field).
    pts, q = _clustered3d(n=1000, seed=2024)
    fmm = Yukawa3DFMM(depth=6, p=6, kappa=1.0)
    built = fmm.build_operator(pts)
    pot_pre = fmm.evaluate_prebuilt(built, q)
    pot_direct = _direct_yukawa(pts, q, kappa=1.0)
    rel = float(np.linalg.norm(pot_pre - pot_direct) /
                np.linalg.norm(pot_direct))
    print(f"  (a) accuracy: rel-L2 vs direct = {rel:.3e} (target < 1e-5)")
    assert rel < 1e-5, (
        f"evaluate_prebuilt rel-L2 vs direct {rel:.3e} >= 1e-5 -- the "
        f"operator math regressed (the old self-parity guard could not "
        f"catch this)")

    # (b) Speedup: 8 charge vectors at N=8k, fixed positions.
    N = 8000
    pts8 = rng.uniform(0.05, 0.95, (N, 3))
    qs8 = [rng.uniform(-1.0, 1.0, N) for _ in range(8)]
    fmm8 = Yukawa3DFMM(depth=6, p=6, kappa=1.0)
    # Reference: rebuild every call (what evaluate does).
    t0 = time.perf_counter()
    ref_outs = [fmm8.evaluate(pts8, qi) for qi in qs8]
    t_rebuild = time.perf_counter() - t0
    # Prebuilt: build once, evaluate many.
    t0 = time.perf_counter()
    built8 = fmm8.build_operator(pts8)
    pre_outs = [fmm8.evaluate_prebuilt(built8, qi) for qi in qs8]
    t_prebuilt = time.perf_counter() - t0
    # Bit-parity between the rebuild and prebuilt paths (consistency only --
    # NOT the accuracy guard, which is (a) above).
    worst = max(float(np.max(np.abs(a - b))) for a, b in zip(ref_outs, pre_outs))
    assert worst == 0.0, f"prebuilt vs rebuild consistency failed: {worst:.3e}"
    speedup = t_rebuild / t_prebuilt if t_prebuilt > 0 else float("inf")
    print(f"  (b) N={N}, 8 charge vectors: rebuild={t_rebuild:.2f}s "
          f"prebuilt={t_prebuilt:.2f}s speedup={speedup:.2f}x "
          f"(consistency max abs diff {worst:.3e})")
    assert speedup > 1.0, (
        f"build_operator reuse not faster for 8 charge vectors: "
        f"speedup={speedup:.2f}x")
    print("test_build_operator_parity_and_speedup: PASS")



def test_evaluate_targets_distinct_targets():
    """evaluate_targets (the T-G1/T-G3 enabler) must match the direct
    source-target sum for sources != targets, including targets that fall in
    EMPTY source cells (nearest-occupied-cell far-field branch)."""
    rng = np.random.default_rng(77)
    N_s, N_t = 400, 300
    # Sources clustered in a corner; targets spread over the whole box, many
    # of them far from any occupied source cell.
    src = np.clip(rng.normal(loc=0.25, scale=0.08, size=(N_s, 3)), 0.001, 0.999)
    tgt = rng.uniform(0.02, 0.98, size=(N_t, 3))
    q = rng.uniform(0.5, 1.5, size=N_s)

    fmm = Yukawa3DFMM(depth=6, p=8, kappa=8.0, ring_direct=2)
    pot = fmm.evaluate_targets(src, q, tgt)

    # Direct reference
    ref = np.zeros(N_t)
    kappa = 8.0
    for i in range(N_t):
        d = tgt[i] - src
        r = np.sqrt(np.sum(d * d, axis=1))
        ref[i] = np.sum(q * np.exp(-kappa * r) / np.maximum(r, 1e-12))

    rel = np.linalg.norm(pot - ref) / np.linalg.norm(ref)
    assert rel < 5e-5, f"evaluate_targets rel-L2 {rel:.3e} >= 5e-5"
    print(f"    evaluate_targets: N_s={N_s}, N_t={N_t}, rel-L2={rel:.2e} PASS")


def main():
    print("=== core.test_yukawa3d_fmm ===")
    test_derivative_fd_guard()
    test_toy_2cell_check()
    test_kappa_zero_coulomb_limit()
    test_occupied_cell_set_matches_unique()
    test_clustered_accuracy_n2000()
    test_yukawa3d_pfloor_regression()
    test_evaluate_forces_2cell()
    test_build_operator_parity_and_speedup()
    test_evaluate_targets_distinct_targets()
    print("\nAll yukawa3d_fmm tests PASS")


if __name__ == "__main__":
    main()
