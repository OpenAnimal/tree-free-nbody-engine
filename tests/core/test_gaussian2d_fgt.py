"""Tests for core.gaussian2d_fgt.Gaussian2DFGT.

Test matrix (round-4 plan section 4.3):
  1. G_n eigenfunction sanity: numeric (1/r d/dr)G at 5 radii vs closed form
  2. derivative tensor vs central FD (|alpha|<=2, h=3e-4, 4th-order stencils
     for first and pure-second derivatives)
  3. 2-cell toy check vs direct (rel-L2 < 1e-12)
  4. clustered N=2000 vs direct: rel-L2 < 1e-6 (Gaussian decay makes this
     easy; if not met raise p 8 -> 10, else STOP and report)
  5. kappa-free occupied-cell membership matches np.unique keys

Run standalone:  python -X utf8 -m core.test_gaussian2d_fgt
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.gaussian2d_fgt import (
    Gaussian2DFGT,
    Gaussian3DFGT,
    gn_eigenfunction_sanity,
    derivative_fd_guard,
    toy_2cell_check,
)
from core.spatial_index import CellIndex


def _clustered2d(n=2000, seed=707):
    """2D clustered multi-scale distribution (mirrors the core 2D clustered
    generator in core/benchmark_variants.py)."""
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


def _direct_gaussian(pts, q, h):
    """Exact O(N^2) direct Gaussian sum (excludes self)."""
    N = len(pts)
    diff = pts[:, None, :] - pts[None, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    h2 = h * h
    w = np.exp(-r2 / h2)
    np.fill_diagonal(w, 0.0)
    return np.sum(q[None, :] * w, axis=1)


# =====================================================================
# Test 1: G_n eigenfunction sanity
# =====================================================================

def test_gn_eigenfunction_sanity():
    assert gn_eigenfunction_sanity(h=0.2), "G_n eigenfunction sanity failed (h=0.2)"
    assert gn_eigenfunction_sanity(h=0.15), "G_n eigenfunction sanity failed (h=0.15)"
    print("test_gn_eigenfunction_sanity: PASS")


# =====================================================================
# Test 2: derivative tensor vs finite differences
# =====================================================================

def test_derivative_fd_guard():
    assert derivative_fd_guard(h=0.2, p=8), "D_alpha FD guard failed (h=0.2)"
    assert derivative_fd_guard(h=0.15, p=8), "D_alpha FD guard failed (h=0.15)"
    print("test_derivative_fd_guard: PASS")


# =====================================================================
# Test 3: 2-cell toy check (sign/factorial convention)
# =====================================================================

def test_toy_2cell_check():
    assert toy_2cell_check(h=0.2, p=8), "2-cell toy check failed (sign convention)"


# =====================================================================
# Test 4: clustered N=2000 vs direct, rel-L2 < 1e-6
# =====================================================================

def test_clustered_accuracy_n2000():
    pts, q = _clustered2d(n=2000, seed=707)
    h = 0.2
    pot_exact = _direct_gaussian(pts, q, h)
    # p=8 should reach well below 1e-6 (Gaussian decay makes this easy).
    fgt = Gaussian2DFGT(depth=6, p=8, h=h)
    pot_fgt = fgt.evaluate(pts, q)
    rel = np.linalg.norm(pot_fgt - pot_exact) / np.linalg.norm(pot_exact)
    print(f"test_clustered_accuracy_n2000 (p=8): rel-L2 = {rel:.3e} (target < 1e-6)")
    if rel >= 1e-6:
        # Plan says: raise p to 10; if still failing, STOP and report.
        for p in (10,):
            fgt = Gaussian2DFGT(depth=6, p=p, h=h)
            pot_fgt = fgt.evaluate(pts, q)
            rel = np.linalg.norm(pot_fgt - pot_exact) / np.linalg.norm(pot_exact)
            print(f"  retry p={p}: rel-L2 = {rel:.3e}")
            if rel < 1e-6:
                break
    assert rel < 1e-6, f"clustered N=2000 rel-L2 {rel:.2e} >= 1e-6 (STOP per plan)"


# =====================================================================
# Test 5: occupied-cell set matches np.unique keys
# =====================================================================

def test_occupied_cell_set_matches_unique():
    pts, q = _clustered2d(n=500, seed=123)
    fgt = Gaussian2DFGT(depth=5, p=6, h=0.2)
    fgt.evaluate(pts, q)
    ci = fgt.cell_index
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


# =====================================================================
# Test 6: 3D toy 2-cell separation check (sign/factorial convention)
# =====================================================================

def _direct_gaussian3d(pts, q, h):
    """Exact O(N^2) direct 3D Gaussian sum (excludes self)."""
    N = len(pts)
    diff = pts[:, None, :] - pts[None, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    h2 = h * h
    w = np.exp(-r2 / h2)
    np.fill_diagonal(w, 0.0)
    return np.sum(q[None, :] * w, axis=1)


def test_gaussian3d_toy_2cell_check():
    """3D two-cell toy check: two clusters in strictly different cells with
    Chebyshev separation > 2*ring_direct, compare FGT vs exact direct.

    Pattern mirrors the 2D toy_2cell_check (gaussian2d_fgt.py:325-373):
    depth=10, cells (1,1,1) and (8,8,8), Chebyshev separation 7 > 2*2=4,
    genuinely well-separated so the far (M2L/L2P) path is exercised.
    """
    rng = np.random.default_rng(0)
    h = 0.2
    depth = 10
    h_grid = 1.0 / depth
    cell1 = np.array([1, 1, 1])
    cell2 = np.array([8, 8, 8])
    cheb_sep = int(np.max(np.abs(cell1 - cell2)))
    ring_direct = 2
    assert cheb_sep > 2 * ring_direct, (
        f"3D toy check degenerate: cells {cell1} and {cell2} have "
        f"Chebyshev separation {cheb_sep} <= 2*ring={2*ring_direct}; "
        f"the far path would never be exercised."
    )
    c1 = (cell1 + 0.5) * h_grid
    c2 = (cell2 + 0.5) * h_grid
    n1, n2 = 4, 5
    pts1 = c1 + rng.uniform(-h_grid * 0.4, h_grid * 0.4, size=(n1, 3))
    pts2 = c2 + rng.uniform(-h_grid * 0.4, h_grid * 0.4, size=(n2, 3))
    pts = np.vstack([pts1, pts2])
    q = rng.uniform(-1.0, 1.0, size=len(pts))
    pot_exact = _direct_gaussian3d(pts, q, h)
    fgt = Gaussian3DFGT(depth=depth, p=8, h=h)
    pot_fgt = fgt.evaluate(pts, q)
    rel = np.linalg.norm(pot_fgt - pot_exact) / max(1e-30, np.linalg.norm(pot_exact))
    print(f"test_gaussian3d_toy_2cell_check: rel-L2 = {rel:.3e} "
          f"(target < 1e-12) {'PASS' if rel < 1e-12 else 'FAIL'}")
    assert rel < 1e-12, (
        f"3D toy 2-cell check rel-L2 {rel:.2e} >= 1e-12")


# =====================================================================
# Test 7: 3D accuracy vs direct on N=1500 random points, rel-L2 < 1e-8
# =====================================================================

def test_gaussian3d_accuracy_n1500():
    """3D Gaussian FGT accuracy vs direct summation on N=1500 random 3D
    points.

    Honest convergence behavior on this scene (measured): p=8 / depth=6
    lands around 1.7e-6 -- NOT 'well below 1e-8' -- because the 3D
    eigenfunction translation accumulates more truncation error than the 2D
    case at the same p. The 1e-8 target is reached only on the retry ladder
    (p=12 passes; p=10 is borderline). The docstring previously claimed
    p=8 already cleared 1e-8, which was wrong and masked the retry. The
    retry ladder below is the real acceptance path, not a safety net."""
    rng = np.random.default_rng(424242)
    N = 1500
    pts = rng.uniform(0.05, 0.95, size=(N, 3))
    q = rng.uniform(-1.0, 1.0, size=N)
    h = 0.2
    pot_exact = _direct_gaussian3d(pts, q, h)
    fgt = Gaussian3DFGT(depth=6, p=8, h=h)
    pot_fgt = fgt.evaluate(pts, q)
    rel = np.linalg.norm(pot_fgt - pot_exact) / np.linalg.norm(pot_exact)
    print(f"test_gaussian3d_accuracy_n1500 (p=8): rel-L2 = {rel:.3e} "
          f"(target < 1e-8)")
    if rel >= 1e-8:
        # Retry with higher p before failing.
        for p in (10, 12):
            fgt = Gaussian3DFGT(depth=6, p=p, h=h)
            pot_fgt = fgt.evaluate(pts, q)
            rel = np.linalg.norm(pot_fgt - pot_exact) / np.linalg.norm(pot_exact)
            print(f"  retry p={p}: rel-L2 = {rel:.3e}")
            if rel < 1e-8:
                break
    assert rel < 1e-8, (
        f"3D Gaussian FGT N=1500 rel-L2 {rel:.2e} >= 1e-8")


def main():
    print("=== core.test_gaussian2d_fgt ===")
    test_gn_eigenfunction_sanity()
    test_derivative_fd_guard()
    test_toy_2cell_check()
    test_occupied_cell_set_matches_unique()
    test_clustered_accuracy_n2000()
    test_gaussian3d_toy_2cell_check()
    test_gaussian3d_accuracy_n1500()
    print("\nAll gaussian2d_fgt tests PASS")


if __name__ == "__main__":
    main()
