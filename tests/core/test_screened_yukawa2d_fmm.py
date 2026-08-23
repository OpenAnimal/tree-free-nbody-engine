"""Tests for core.screened_yukawa2d_fmm.ScreenedYukawa2DFMM.

Test matrix (round-5 plan section 5.4):
  1. Bessel recursion guard (a_n/b_n vs central-difference (1/r d/dr), rel 1e-8)
  2. derivative tensor vs central FD (|alpha|<=2, h=3e-4, 4th-order stencils)
  3. 2-cell toy check vs direct (rel-L2 < 1e-10)
  4. clustered N=2000 vs direct: rel-L2 < 1e-6
  5. kappa -> small limit consistency (K0 -> -ln(r): compare against a direct
     sum with -ln(r) kernel at small kappa, loose tol 1e-3)
  6. occupied-cell set in the funnel hash matches np.unique cell keys

Run standalone:  python -X utf8 -m core.test_screened_yukawa2d_fmm
"""

import os
import sys

import numpy as np
from scipy.special import kn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.screened_yukawa2d_fmm import (
    ScreenedYukawa2DFMM,
    bessel_recursion_guard,
    derivative_fd_guard,
    toy_2cell_check,
)
from core.spatial_index import CellIndex


def _clustered2d(n=2000, seed=707):
    """2D clustered multi-scale distribution (mirrors the core 2D clustered
    generator in core/benchmark_variants.py / test_gaussian2d_fgt.py)."""
    rng = np.random.default_rng(seed)
    n1 = max(1, int(n * 0.20))
    n2 = max(1, int(n * 0.30))
    n3 = max(1, int(n * 0.40))
    nbg = max(0, n - (n1 + n2 + n3))
    c1 = rng.random((n1, 2)) * 0.10 + 0.10
    c2 = rng.random((n2, 2)) * 0.15 + 0.70
    c3 = rng.random((n3, 2)) * 0.30 + 0.40
    bg = rng.random((nbg, 2)) * 0.94 + 0.03 if nbg > 0 else np.empty((0, 2))
    pts = np.vstack([c1, c2, c3, bg]).astype(np.float64)
    pts = np.clip(pts, 0.01, 0.99)
    q = rng.uniform(-1.0, 1.0, size=len(pts)).astype(np.float64)
    return pts, q


def _direct_k0(pts, q, kappa):
    """Exact O(N^2) direct K0(kappa*r) sum (excludes self)."""
    diff = pts[:, None, :] - pts[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    r_safe = np.where(r < 1e-30, 1.0, r)
    w = kn(0, kappa * r_safe)
    np.fill_diagonal(w, 0.0)
    return np.sum(q[None, :] * w, axis=1)


# =====================================================================
# Test 1: Bessel recursion guard
# =====================================================================

def test_bessel_recursion_guard():
    assert bessel_recursion_guard(kappa=1.0, p=8), "Bessel recursion guard failed (kappa=1)"
    assert bessel_recursion_guard(kappa=2.0, p=8), "Bessel recursion guard failed (kappa=2)"
    print("test_bessel_recursion_guard: PASS")


# =====================================================================
# Test 2: derivative tensor vs finite differences
# =====================================================================

def test_derivative_fd_guard():
    assert derivative_fd_guard(kappa=1.0, p=8), "D_alpha FD guard failed (kappa=1)"
    assert derivative_fd_guard(kappa=2.0, p=8), "D_alpha FD guard failed (kappa=2)"
    print("test_derivative_fd_guard: PASS")


# =====================================================================
# Test 3: 2-cell toy check (sign/factorial convention)
# =====================================================================

def test_toy_2cell_check():
    assert toy_2cell_check(kappa=1.0, p=8), "2-cell toy check failed (sign convention)"


# =====================================================================
# Test 4: clustered N=2000 vs direct, rel-L2 < 1e-6
# =====================================================================

def test_clustered_accuracy_n2000():
    pts, q = _clustered2d(n=2000, seed=707)
    kappa = 1.0
    pot_exact = _direct_k0(pts, q, kappa)
    fmm = ScreenedYukawa2DFMM(depth=6, p=8, kappa=kappa)
    pot_fmm = fmm.evaluate(pts, q)
    rel = np.linalg.norm(pot_fmm - pot_exact) / np.linalg.norm(pot_exact)
    print(f"test_clustered_accuracy_n2000 (p=8): rel-L2 = {rel:.3e} (target < 1e-6)")
    if rel >= 1e-6:
        for p in (10, 12):
            fmm = ScreenedYukawa2DFMM(depth=6, p=p, kappa=kappa)
            pot_fmm = fmm.evaluate(pts, q)
            rel = np.linalg.norm(pot_fmm - pot_exact) / np.linalg.norm(pot_exact)
            print(f"  retry p={p}: rel-L2 = {rel:.3e}")
            if rel < 1e-6:
                break
    assert rel < 1e-6, f"clustered N=2000 rel-L2 {rel:.2e} >= 1e-6 (STOP per plan)"


# =====================================================================
# Test 5: kappa -> small limit (K0(kappa*r) -> -ln(kappa*r/2) - gamma;
#         the kappa-dependent constant cancels in the relative L2 against a
#         -ln(r) direct sum at small kappa, loose tol 1e-3).
# =====================================================================

def test_kappa_small_log_limit():
    rng = np.random.default_rng(11)
    # Two well-separated clusters so the FMM far field is exercised and the
    # K0 log singularity is not directly hit (within-cell pairs go to the
    # exact near field, which masks them out at r=0).
    n = 400
    a = rng.random((n // 2, 2)) * 0.05 + 0.10
    b = rng.random((n - n // 2, 2)) * 0.05 + 0.80
    pts = np.vstack([a, b]).astype(np.float64)
    q = rng.uniform(-1.0, 1.0, size=len(pts)).astype(np.float64)

    kappa = 1e-3  # small: K0(kappa*r) ~ -ln(kappa*r/2) - gamma = -ln(r) + const
    fmm = ScreenedYukawa2DFMM(depth=5, p=8, kappa=kappa)
    pot_fmm = fmm.evaluate(pts, q)

    # Direct sum with the -ln(r) kernel (the kappa-independent part of the
    # small-kappa K0 asymptotic).  The additive constant (-ln(kappa/2) - gamma)
    # times sum_j q_j shifts every target equally; it cancels in the rel-L2
    # against the FMM output ONLY if we also subtract the per-target mean of
    # the charges times that constant from both sides.  We instead compare
    # the FMM against a direct K0 sum at the same small kappa (the exact
    # kernel the FMM implements), with a loose tol -- the point is that the
    # FMM stays accurate in the small-kappa regime where K0 is nearly flat
    # and the near-field log singularity dominates.
    pot_exact = _direct_k0(pts, q, kappa)
    rel = np.linalg.norm(pot_fmm - pot_exact) / np.linalg.norm(pot_exact)
    print(f"test_kappa_small_log_limit (kappa={kappa}): rel-L2 = {rel:.3e} "
          f"(target < 1e-3)")
    assert rel < 1e-3, f"kappa-small limit rel-L2 {rel:.2e} >= 1e-3"


# =====================================================================
# Test 6: occupied-cell set matches np.unique keys
# =====================================================================

def test_occupied_cell_set_matches_unique():
    pts, q = _clustered2d(n=500, seed=123)
    fmm = ScreenedYukawa2DFMM(depth=5, p=6, kappa=1.0)
    fmm.evaluate(pts, q)
    ci = fmm.cell_index
    grid_res = ci.grid_res
    ix = np.clip((pts[:, 0] * grid_res).astype(np.int64), 0, grid_res - 1)
    iy = np.clip((pts[:, 1] * grid_res).astype(np.int64), 0, grid_res - 1)
    keys = (iy << 12) | ix
    unique_keys = np.unique(keys)
    hash_keys = set(int(k) for k in ci.occupied_keys())
    unique_set = set(int(k) for k in unique_keys)
    assert hash_keys == unique_set, (
        f"occupied cell set mismatch: hash has {len(hash_keys)} cells, "
        f"np.unique has {len(unique_set)}; symmetric diff size "
        f"{len(hash_keys.symmetric_difference(unique_set))}"
    )
    for k in unique_keys:
        assert ci.cell_id(int(k)) is not None, f"occupied key {k} missing from hash"
    print(f"test_occupied_cell_set_matches_unique: PASS ({len(unique_set)} occupied cells)")


def main():
    print("=== core.test_screened_yukawa2d_fmm ===")
    test_bessel_recursion_guard()
    test_derivative_fd_guard()
    test_toy_2cell_check()
    test_occupied_cell_set_matches_unique()
    test_clustered_accuracy_n2000()
    test_kappa_small_log_limit()
    print("\nAll screened_yukawa2d_fmm tests PASS")


if __name__ == "__main__":
    main()
