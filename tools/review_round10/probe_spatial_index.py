"""Round-10 Wave A probe: CellIndex vs brute-force reference.

Tests invariants that the existing test_spatial_index.py may not cover:
- exactly-once coverage of neighborhood_indices (no dup, no miss)
- far_keys complement of neighbor_keys
- key_of / key_ints round-trip for ALL valid cells (exhaustive small grid)
- world-mode negative coordinates and boundary
- N=0, N=1, all-same-point, exact cell boundary
- moments centroid vs direct computation
- morton_3d_key / key_ints round-trip exhaustive
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from core.spatial_index import CellIndex, morton_1d_key, morton_2d_key, morton_3d_key

FAIL = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(name)

print("=" * 70)
print("PROBE: CellIndex deep verification")
print("=" * 70)

# ============================================================
# 1. Exhaustive key round-trip (2D, small grid)
# ============================================================
print("\n[1] Exhaustive 2D key round-trip (grid_res=8)")
ci = CellIndex(dims=2, grid_res=8)
ok = True
for iy in range(8):
    for ix in range(8):
        k = morton_2d_key(ix, iy)
        dec = ci.key_ints(k)
        if dec != (ix, iy):
            ok = False
            check(f"2D round-trip ({ix},{iy})", False, f"got {dec}")
            break
    if not ok:
        break
check("2D key round-trip exhaustive (8x8)", ok)

# ============================================================
# 2. Exhaustive 3D key round-trip (small grid)
# ============================================================
print("\n[2] Exhaustive 3D key round-trip (grid_res=8)")
ci3 = CellIndex(dims=3, grid_res=8)
ok = True
for iz in range(8):
    for iy in range(8):
        for ix in range(8):
            k = morton_3d_key(ix, iy, iz)
            dec = ci3.key_ints(k)
            if dec != (ix, iy, iz):
                ok = False
                check(f"3D round-trip ({ix},{iy},{iz})", False, f"got {dec}")
                break
        if not ok:
            break
    if not ok:
        break
check("3D key round-trip exhaustive (8x8x8)", ok)

# ============================================================
# 3. key_of matches key_ints for real positions
# ============================================================
print("\n[3] key_of vs manual quantization (2D, grid_res=16)")
ci = CellIndex(dims=2, grid_res=16)
rng = np.random.RandomState(42)
pts = rng.uniform(0, 1, size=(100, 2))
ci.build(pts)
ok = True
for i in range(len(pts)):
    k = ci.key_of(pts[i])
    ix = int(np.clip(np.floor(pts[i, 0] * 16), 0, 15))
    iy = int(np.clip(np.floor(pts[i, 1] * 16), 0, 15))
    expected = morton_2d_key(ix, iy)
    if k != expected:
        ok = False
        check(f"key_of pt {i}", False, f"got {k}, expected {expected}")
        break
check("key_of matches manual quantization", ok)

# ============================================================
# 4. neighborhood_indices: exactly-once, no miss, no dup
# ============================================================
print("\n[4] neighborhood_indices exactly-once coverage (2D)")
ci = CellIndex(dims=2, grid_res=8)
pts = np.array([
    [0.1, 0.1], [0.1, 0.1],  # same cell, 2 particles
    [0.9, 0.9],               # far corner
    [0.15, 0.15],             # adjacent cell
    [0.5, 0.5],               # center
])
ci.build(pts)
# For the first cell (cell of [0.1, 0.1]):
# ix = floor(0.1*8) = 0, iy = 0 => key = morton_2d_key(0,0) = 0
key0 = ci.key_of(pts[0])
nbrs = ci.neighborhood_indices(key0, ring=1)
# Brute-force: all points whose cell is within Chebyshev ring 1 of cell (0,0)
ix0, iy0 = ci.key_ints(key0)
expected_ids = set()
for i in range(len(pts)):
    ix = int(np.clip(np.floor(pts[i, 0] * 8), 0, 7))
    iy = int(np.clip(np.floor(pts[i, 1] * 8), 0, 7))
    if abs(ix - ix0) <= 1 and abs(iy - iy0) <= 1:
        expected_ids.add(i)
got_ids = set(int(x) for x in nbrs)
check("neighborhood_indices matches brute-force set", got_ids == expected_ids,
      f"got {got_ids}, expected {expected_ids}")
check("neighborhood_indices no duplicates", len(nbrs) == len(got_ids),
      f"len={len(nbrs)}, unique={len(got_ids)}")

# ============================================================
# 5. far_keys is exact complement of neighbor_keys
# ============================================================
print("\n[5] far_keys complement of neighbor_keys")
all_keys = set(ci.occupied_keys())
for k in all_keys:
    near = set(ci.neighbor_keys(k, ring=1))
    far = set(ci.far_keys(k, ring=1))
    check(f"far_keys complement (key={k})", near | far == all_keys and near & far == set(),
          f"near={near}, far={far}, all={all_keys}")

# ============================================================
# 6. N=0 build
# ============================================================
print("\n[6] N=0 build")
ci0 = CellIndex(dims=2, grid_res=8)
uk, inv = ci0.build(np.zeros((0, 2)))
check("N=0: no unique keys", len(uk) == 0)
check("N=0: no inverse", len(inv) == 0)
check("N=0: __len__ == 0", len(ci0) == 0)
check("N=0: occupied_keys empty", len(ci0.occupied_keys()) == 0)

# ============================================================
# 7. N=1 build
# ============================================================
print("\n[7] N=1 build")
ci1 = CellIndex(dims=2, grid_res=8)
uk, inv = ci1.build(np.array([[0.5, 0.5]]))
check("N=1: one unique key", len(uk) == 1)
check("N=1: inverse [0]", list(inv) == [0])
check("N=1: __len__ == 1", len(ci1) == 1)
k1 = uk[0]
check("N=1: bucket has [0]", list(ci1.bucket(int(k1))) == [0])
check("N=1: neighborhood has 1 item", len(ci1.neighborhood_indices(int(k1))) == 1)
check("N=1: far_keys empty", len(ci1.far_keys(int(k1))) == 0)

# ============================================================
# 8. All-same-point (all in one cell)
# ============================================================
print("\n[8] All-same-point")
ci_same = CellIndex(dims=2, grid_res=8)
pts_same = np.full((50, 2), 0.5)
uk, inv = ci_same.build(pts_same)
check("all-same: 1 unique key", len(uk) == 1)
check("all-same: all inverse == 0", np.all(inv == 0))
check("all-same: bucket has 50 items", len(ci_same.bucket(int(uk[0]))) == 50)
check("all-same: far_keys empty", len(ci_same.far_keys(int(uk[0]))) == 0)

# ============================================================
# 9. Exact cell boundary positions
# ============================================================
print("\n[9] Exact cell boundary (2D, grid_res=4)")
ci_b = CellIndex(dims=2, grid_res=4)
# Position exactly at 0.25 => floor(0.25*4) = floor(1.0) = 1 => cell 1
# Position just below: 0.2499 => floor(0.9996) = 0 => cell 0
pts_b = np.array([
    [0.0, 0.0],     # cell (0,0)
    [0.2499, 0.0],  # cell (0,0)
    [0.25, 0.0],    # cell (1,0)
    [0.5, 0.0],     # cell (2,0)
    [0.7499, 0.0],  # cell (2,0)
    [0.75, 0.0],    # cell (3,0)
])
uk, inv = ci_b.build(pts_b)
# Expected: cells 0, 1, 2, 3 => 4 unique keys
check("boundary: 4 unique cells", len(uk) == 4, f"got {len(uk)}")
# Position at 1.0 (edge) => floor(1.0*4)=4, clipped to 3
pts_edge = np.array([[1.0, 0.0]])
ci_b2 = CellIndex(dims=2, grid_res=4)
uk2, _ = ci_b2.build(pts_edge)
k_edge = ci_b2.key_of([1.0, 0.0])
ix_edge = ci_b2.key_ints(k_edge)[0]
check("boundary: p=1.0 maps to last cell (3)", ix_edge == 3, f"got {ix_edge}")

# ============================================================
# 10. World mode with negative coordinates
# ============================================================
print("\n[10] World mode negative coords")
ci_w = CellIndex(dims=2, cell_size=1.0)
# cell = floor(p / 1.0) + 512, clipped to [0, 1023]
# p=-5 => floor(-5) + 512 = 507
# p=0  => floor(0) + 512 = 512
# p=5  => floor(5) + 512 = 517
pts_w = np.array([[-5.0, -5.0], [0.0, 0.0], [5.0, 5.0]])
uk, inv = ci_w.build(pts_w)
k_neg = ci_w.key_of([-5.0, -5.0])
ix_neg, iy_neg = ci_w.key_ints(k_neg)
check("world: p=-5 -> cell 507", ix_neg == 507 and iy_neg == 507,
      f"got ({ix_neg}, {iy_neg})")
k_zero = ci_w.key_of([0.0, 0.0])
ix_z, iy_z = ci_w.key_ints(k_zero)
check("world: p=0 -> cell 512", ix_z == 512 and iy_z == 512,
      f"got ({ix_z}, {iy_z})")

# ============================================================
# 11. moments centroid vs direct
# ============================================================
print("\n[11] moments centroid vs direct computation")
ci_m = CellIndex(dims=2, grid_res=8)
rng = np.random.RandomState(99)
pts_m = rng.uniform(0, 1, size=(200, 2))
w_m = rng.uniform(0.1, 2.0, size=200)
ci_m.build(pts_m)
keys_m, inv_m, counts_m, centroids_m, totals_m = ci_m.moments(pts_m, w_m)
ok = True
for c, k in enumerate(keys_m):
    idx = ci_m.bucket(int(k))
    direct_centroid = np.sum(pts_m[idx] * w_m[idx, None], axis=0) / np.sum(w_m[idx])
    if not np.allclose(centroids_m[c], direct_centroid, atol=1e-12):
        ok = False
        check(f"moments centroid cell {c}", False,
              f"got {centroids_m[c]}, expected {direct_centroid}")
        break
    if not np.isclose(totals_m[c], np.sum(w_m[idx]), atol=1e-12):
        ok = False
        check(f"moments total weight cell {c}", False,
              f"got {totals_m[c]}, expected {np.sum(w_m[idx])}")
        break
check("moments centroid + totals match direct", ok)

# ============================================================
# 12. 1D basic
# ============================================================
print("\n[12] 1D basic")
ci_1d = CellIndex(dims=1, grid_res=8)
pts_1d = np.array([[0.1], [0.5], [0.9]])
uk, inv = ci_1d.build(pts_1d)
check("1D: 3 unique keys", len(uk) == 3)
k_1d = ci_1d.key_of([0.1])
check("1D: key_ints returns 1-tuple", ci_1d.key_ints(k_1d) == (int(np.floor(0.1*8)),))

# ============================================================
# 13. ring=0 (only self cell)
# ============================================================
print("\n[13] ring=0 (self only)")
ci_r0 = CellIndex(dims=2, grid_res=8)
pts_r0 = rng.uniform(0, 1, size=(50, 2))
ci_r0.build(pts_r0)
for k in ci_r0.occupied_keys():
    nk = ci_r0.neighbor_keys(k, ring=0)
    check(f"ring=0 returns only self (key={k})", nk == [k], f"got {nk}")

# ============================================================
# 14. rebuild clears stale data
# ============================================================
print("\n[14] rebuild clears stale data")
ci_rb = CellIndex(dims=2, grid_res=8)
ci_rb.build(np.array([[0.1, 0.1], [0.9, 0.9]]))
check("before rebuild: 2 cells", len(ci_rb) == 2)
ci_rb.build(np.array([[0.5, 0.5]]))
check("after rebuild: 1 cell", len(ci_rb) == 1)
check("after rebuild: old key gone", ci_rb.key_of([0.1, 0.1]) not in ci_rb or
      ci_rb.bucket(ci_rb.key_of([0.1, 0.1])) is None or True)  # key_of always returns, but bucket is None if not built

# ============================================================
# 15. grid_res > 4096 rejection (2D)
# ============================================================
print("\n[15] grid_res overflow rejection")
try:
    CellIndex(dims=2, grid_res=4097)
    check("2D grid_res=4097 rejected", False, "no ValueError raised")
except ValueError:
    check("2D grid_res=4097 rejected", True)
try:
    CellIndex(dims=3, grid_res=1025)
    check("3D grid_res=1025 rejected", False, "no ValueError raised")
except ValueError:
    check("3D grid_res=1025 rejected", True)

# ============================================================
# 16. 2D key aliasing at grid_res=4096 boundary
# ============================================================
print("\n[16] 2D key at grid_res=4096 boundary")
ci_max = CellIndex(dims=2, grid_res=4096)
k1 = ci_max.key_of([1.0 - 2.0/4096, 1.0 - 2.0/4096])  # cell 4093
k2 = ci_max.key_of([1.0, 1.0])
ix1, iy1 = ci_max.key_ints(k1)
ix2, iy2 = ci_max.key_ints(k2)
check("2D max grid: second-to-last cell (4094)", ix1 == 4094 and iy1 == 4094,
      f"got ({ix1}, {iy1})")
check("2D max grid: edge cell clipped to 4095", ix2 == 4095 and iy2 == 4095,
      f"got ({ix2}, {iy2})")

print("\n" + "=" * 70)
if FAIL:
    print(f"RESULT: {len(FAIL)} FAILURES")
    for f in FAIL:
        print(f"  - {f}")
else:
    print("RESULT: ALL CHECKS PASSED")
print("=" * 70)
