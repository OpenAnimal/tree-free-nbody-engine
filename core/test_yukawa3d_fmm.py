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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.yukawa3d_fmm import (
    Yukawa3DFMM,
    derivative_fd_guard,
    toy_2cell_check,
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


def main():
    print("=== core.test_yukawa3d_fmm ===")
    test_derivative_fd_guard()
    test_toy_2cell_check()
    test_kappa_zero_coulomb_limit()
    test_occupied_cell_set_matches_unique()
    test_clustered_accuracy_n2000()
    print("\nAll yukawa3d_fmm tests PASS")


if __name__ == "__main__":
    main()
