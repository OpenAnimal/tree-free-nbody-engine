"""Wave E probe 2: quantized_bitpacked_optimization deep verification.

Independent scalar references for:
  - Morton inc/dec register arithmetic (decode -> offset -> re-encode)
  - FastMortonNeighborTable2D batch neighbors (exhaustive small depths)
  - 64-bit 3D / 32-bit 2D particle pack-unpack roundtrips and boundaries
  - Morton bitboards (2D/3D) occupancy sets
  - VoxelPackedTreeFreeFMM vs direct O(N^2) log-kernel sum; flag isolation
Run: python tools/review_round10/probe_wavee_2_quantized.py
"""
import os
import sys
import itertools

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from quantized_bitpacked_optimization.direct_morton_stride import (
    FastMortonNeighborTable2D,
    morton_inc_x_2d,
    morton_dec_x_2d,
    morton_inc_y_2d,
    morton_dec_y_2d,
    MASK_2D_X,
    MASK_2D_Y,
)
from quantized_bitpacked_optimization.packed_particle_types import (
    pack_particles_64bit_3d,
    unpack_particles_64bit_3d,
    pack_particles_32bit_2d,
    unpack_particles_32bit_2d,
)
from quantized_bitpacked_optimization.bitboard_occupancy import (
    MortonBitboard2D,
    MortonBitboard3D,
)
from quantized_bitpacked_optimization.greedy_multipole_mesh import (
    GreedyMultipoleAggregator2D,
)
from quantized_bitpacked_optimization.packed_vectorized_fmm import (
    VoxelPackedTreeFreeFMM,
)

PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


# Scalar reference encode/decode (independent of module code paths)
def enc2d(ix, iy, depth):
    m = 0
    for b in range(depth):
        m |= ((ix >> b) & 1) << (2 * b)
        m |= ((iy >> b) & 1) << (2 * b + 1)
    return m


def dec2d(m, depth):
    ix = iy = 0
    for b in range(depth):
        ix |= ((m >> (2 * b)) & 1) << b
        iy |= ((m >> (2 * b + 1)) & 1) << b
    return ix, iy


# =========================================================================
print("== 1. Morton inc/dec register arithmetic vs scalar reference ==")
bad = 0
wrap_x = wrap_y = 0
for depth in (2, 3, 4):
    g = 1 << depth
    for ix in range(g):
        for iy in range(g):
            m = enc2d(ix, iy, depth)
            # inc_x
            if ix + 1 < g:
                if morton_inc_x_2d(m) != enc2d(ix + 1, iy, depth):
                    bad += 1
            else:
                wrap_x += 1  # documented wraparound domain edge (out of grid)
            if ix - 1 >= 0:
                if morton_dec_x_2d(m) != enc2d(ix - 1, iy, depth):
                    bad += 1
            else:
                wrap_x += 1
            if iy + 1 < g:
                if morton_inc_y_2d(m) != enc2d(ix, iy + 1, depth):
                    bad += 1
            else:
                wrap_y += 1
            if iy - 1 >= 0:
                if morton_dec_y_2d(m) != enc2d(ix, iy - 1, depth):
                    bad += 1
            else:
                wrap_y += 1
check("morton inc/dec x/y matches decode-offset-encode on all in-grid keys",
      bad == 0, f"mismatches={bad} (grid wraps skipped: x={wrap_x}, y={wrap_y})")
# y-plane untouched by x ops and vice versa (pure bit-plane isolation)
m = enc2d(5, 9, 4)
check("inc_x leaves y bit-plane untouched",
      (morton_inc_x_2d(m) & MASK_2D_Y) == (m & MASK_2D_Y))
check("inc_y leaves x bit-plane untouched",
      (morton_inc_y_2d(m) & MASK_2D_X) == (m & MASK_2D_X))

# =========================================================================
print("== 2. FastMortonNeighborTable2D exhaustive vs scalar reference ==")
for depth in (2, 3, 4):
    g = 1 << depth
    keys = np.array([enc2d(ix, iy, depth) for ix in range(g) for iy in range(g)],
                    dtype=np.int64)
    keys_with_depth = (np.int64(depth) << 24) | keys
    tbl = FastMortonNeighborTable2D(depth=depth)
    nb = tbl.get_all_neighbors_batch(keys_with_depth)
    ok = True
    for t, key in enumerate(keys_with_depth):
        ix, iy = dec2d(int(keys[t]), depth)
        for k, (dx, dy) in enumerate(tbl.offsets):
            want = -1
            nx, ny = ix + dx, iy + dy
            if 0 <= nx < g and 0 <= ny < g:
                want = (depth << 24) | enc2d(nx, ny, depth)
            if nb[t, k] != want:
                ok = False
                print(f"    mismatch depth={depth} key={key} off=({dx},{dy}) "
                      f"got={nb[t, k]} want={want}")
                break
        if not ok:
            break
    check(f"neighbor table exhaustive depth={depth} ({g*g} keys, 9 offsets)", ok)

# ordering of offsets matches the documented row-major layout
tbl = FastMortonNeighborTable2D(depth=3)
check("offsets are the canonical 3x3",
      tbl.offsets == [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0),
                      (-1, 1), (0, 1), (1, 1)])
# empty input
nb0 = FastMortonNeighborTable2D(depth=3).get_all_neighbors_batch(
    np.array([], dtype=np.int64))
check("neighbor table empty input -> (0, 9)", nb0.shape == (0, 9))

# =========================================================================
print("== 3. 64-bit 3D pack/unpack: boundaries, roundtrip, duplicates ==")
rng = np.random.default_rng(20260822)
for depth in (4, 6, 8):
    g = 1 << depth
    pos = rng.random((500, 3))
    # boundary values per axis: 0, exact cell edges, max representable
    edges = np.array([k / g for k in (0, 1, 2, g // 2, g - 2, g - 1)])
    for r in range(3):
        pos[:8, r] = rng.choice(edges, 8)
    q = rng.uniform(-10, 10, 500)
    pck = pack_particles_64bit_3d(pos, q, depth=depth)
    up, uq = unpack_particles_64bit_3d(pck, depth=depth)
    err = float(np.max(np.abs(up - pos)))
    bound = 1.5 / (256.0 * g)  # floor (1/g) + frac (1/(256 g)) quantization
    check(f"pack64 depth={depth}: roundtrip pos err <= bound", err <= bound,
          f"err={err:.2e} bound={bound:.2e}")
    qerr = float(np.max(np.abs(uq - q) / np.maximum(1e-6, np.abs(q))))
    check(f"pack64 depth={depth}: charge fp16 rel err <= 2^-10", qerr <= 2 ** -10,
          f"err={qerr:.2e}")
check("pack64: position 1.0 clips to last cell",
      float(np.max(unpack_particles_64bit_3d(
          pack_particles_64bit_3d(np.ones((3, 3)), np.ones(3)))[0])) < 1.0)
# duplicates pack identically (same word)
p = np.array([[0.5, 0.5, 0.5]] * 4)
qq = np.array([1.0] * 4)
pw = pack_particles_64bit_3d(p, qq)
check("pack64: duplicates -> identical words", bool(np.all(pw == pw[0])))
# empty input
e = pack_particles_64bit_3d(np.empty((0, 3)), np.empty(0))
eu, eq = unpack_particles_64bit_3d(e)
check("pack64: empty input roundtrip", e.shape == (0,) and eu.shape == (0, 2) or eu.ndim == 2,
      f"shapes {e.shape} {eu.shape}")

# =========================================================================
print("== 4. 32-bit 2D pack/unpack: boundaries, signedness, clipping ==")
for depth in (4, 5, 6):
    g = 1 << depth
    pos = rng.random((500, 2))
    edges = np.array([k / g for k in (0, 1, 2, g // 2, g - 2, g - 1)])
    for r in range(2):
        pos[:8, r] = rng.choice(edges, 8)
    q = rng.uniform(-2, 2, 500)
    pck = pack_particles_32bit_2d(pos, q, depth=depth)
    up, uq = unpack_particles_32bit_2d(pck, depth=depth)
    bound = 1.5 / (64.0 * g)
    err = float(np.max(np.abs(up - pos)))
    check(f"pack32 depth={depth}: roundtrip pos err <= bound", err <= bound,
          f"err={err:.2e} bound={bound:.2e}")
    qerr = float(np.max(np.abs(uq - q)))
    check(f"pack32 depth={depth}: charge quant err <= 1/64 + clip",
          qerr <= 1.0 / 64 + 1e-12, f"err={qerr:.2e}")
# exact multiples of 1/64 are lossless in the charge field
q = np.arange(-128, 128) / 64.0
pos = np.zeros((len(q), 2)) + 0.5
_, uq = unpack_particles_32bit_2d(pack_particles_32bit_2d(pos, q))
check("pack32: charge grid multiples roundtrip exactly",
      float(np.max(np.abs(uq - q))) == 0.0)
# clipping symmetry at |q| = 2 (the documented -128..127 int8 range)
pos1 = np.zeros((1, 2)) + 0.3
_, uq = unpack_particles_32bit_2d(pack_particles_32bit_2d(pos1, np.array([2.0, -2.0])))
check("pack32: |q|=2 clips to 127/64 and -128/64",
      abs(uq[0] - 127 / 64) < 1e-12 and abs(uq[1] + 2.0) < 1e-12, f"got {uq}")
# out-of-domain positions clip (no wraparound)
_, up = unpack_particles_32bit_2d(pack_particles_32bit_2d(
    np.array([[-0.5, 1.5], [2.0, -3.0]]), np.zeros(2)))
check("pack32: out-of-domain positions clip into [0,1)",
      bool(np.all(up >= 0.0) and np.all(up < 1.0)), f"{up}")
# empty
e = pack_particles_32bit_2d(np.empty((0, 2)), np.empty(0))
check("pack32: empty input", e.shape == (0,))

# =========================================================================
print("== 5. Morton bitboards vs reference occupancy sets ==")
for trial in range(10):
    n = int(rng.integers(1, 400))
    ix = rng.integers(0, 64, n)
    iy = rng.integers(0, 64, n)
    bb = MortonBitboard2D()
    bb.populate(ix, iy, depth=6)
    got = set(bb.iter_active_cells())
    want = set(zip(ix.tolist(), iy.tolist()))
    if got != want:
        check(f"bitboard2D occupancy trial {trial}", False,
              f"missing={list(want - got)[:3]} extra={list(got - want)[:3]}")
        break
    if bb.active_cell_count() != len(want):
        check(f"bitboard2D popcount trial {trial}", False)
        break
else:
    check("bitboard2D occupancy+popcount == reference (10 random trials)", True)

bb = MortonBitboard2D()
bb.populate(np.array([], dtype=int), np.array([], dtype=int))
check("bitboard2D empty populate", len(list(bb.iter_active_cells())) == 0
      and bb.active_cell_count() == 0)
full = np.repeat(np.arange(64), 64), np.tile(np.arange(64), 64)
bb.populate(*full)
check("bitboard2D full 64x64 grid", bb.active_cell_count() == 4096)

for trial in range(5):
    n = int(rng.integers(1, 300))
    ix = rng.integers(0, 64, n)
    iy = rng.integers(0, 64, n)
    iz = rng.integers(0, 64, n)
    bb3 = MortonBitboard3D()
    bb3.populate(ix, iy, iz, depth=6)
    got = set(bb3.iter_active_cells())
    want = set(zip(ix.tolist(), iy.tolist(), iz.tolist()))
    if got != want:
        check(f"bitboard3D occupancy trial {trial}", False,
              f"missing={list(want - got)[:3]} extra={list(got - want)[:3]}")
        break
else:
    check("bitboard3D occupancy == reference (5 random trials)", True)

# =========================================================================
print("== 6. Greedy aggregator edge cases (K<=4 identity, stable mapping) ==")
agg = GreedyMultipoleAggregator2D(order=4)
keys = np.array([(6 << 24) | (ix << 12) | iy for (ix, iy) in
                 [(3, 3), (3, 4), (4, 3), (4, 4)]], dtype=np.int64)
box = 1.0 / 64
centers = np.array([(ix + 0.5) * box + 1j * (iy + 0.5) * box for (ix, iy) in
                    [(3, 3), (3, 4), (4, 3), (4, 4)]])
cm = rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))
mc, mm, pmap, rr = agg.aggregate_runs(keys, centers, cm, depth=6)
check("greedy: K<=4 early return is identity",
      np.array_equal(centers, mc) and np.array_equal(cm, mm)
      and np.array_equal(pmap, np.arange(4)) and rr == 1.0)

# K=5 with one full sibling run + 1 distant leaf: only the sibling run merges
# (siblings of parent (1,1) are leaves (2,2),(2,3),(3,2),(3,3) — a 2x2 LEAF
# block like (3,3),(3,4),(4,3),(4,4) is four DIFFERENT parents, no merge)
keys5 = np.array([(6 << 24) | (ix << 12) | iy for (ix, iy) in
                  [(2, 2), (2, 3), (3, 2), (3, 3), (40, 40)]], dtype=np.int64)
c5 = np.array([(ix + 0.5) * box + 1j * (iy + 0.5) * box for (ix, iy) in
               [(2, 2), (2, 3), (3, 2), (3, 3), (40, 40)]])
cm5 = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
mc5, mm5, pm5, rr5 = agg.aggregate_runs(keys5, c5, cm5, depth=6)
check("greedy: sibling run + distant leaf -> 2 macros, ratio 2.5",
      len(mc5) == 2 and abs(rr5 - 2.5) < 1e-12,
      f"M={len(mc5)} rr={rr5}")
# The lone distant leaf's macro center must be its PARENT box center
# (its moments are M2M-translated there — moment correctness vs direct P2M
# is covered by tests/quantized_bitpacked_optimization/test_greedy_multipole_mesh.py)
pbox = 1.0 / 32.0
want_center = (20 + 0.5) * pbox + 1j * (20 + 0.5) * pbox
check("greedy: distant leaf's macro center is its parent box center",
      abs(mc5[pm5[4]] - want_center) < 1e-15, f"{mc5[pm5[4]]} vs {want_center}")
check("greedy: siblings share one macro id",
      len(set(pm5[:4].tolist())) == 1)

# =========================================================================
print("== 7. VoxelPackedTreeFreeFMM vs direct O(N^2) oracle ==")


def direct_pot(pos, q):
    N = len(pos)
    z = pos[:, 0] + 1j * pos[:, 1]
    pot = np.zeros(N)
    for i in range(N):
        d = z[i] - z
        d[i] = 1.0
        pot[i] = np.sum(q * np.log(np.abs(d)))
    return pot


rngf = np.random.default_rng(42)
pos = rngf.random((400, 2))
q = rngf.uniform(-1, 1, 400)
ref = direct_pot(pos, q)

pot_base, _ = VoxelPackedTreeFreeFMM(depth=6, order=4, enable_packing=False,
                                     enable_greedy_aggregation=False,
                                     enable_bitboard_skip=True,
                                     enable_direct_strides=True).evaluate(pos, q)
rel = np.linalg.norm(pot_base - ref) / np.linalg.norm(ref)
check("FMM baseline (no lossy flags) vs direct: rel-L2 < 5e-3", rel < 5e-3,
      f"rel={rel:.2e}")

# direct strides / bitboard toggles are lossless
pot_ns, _ = VoxelPackedTreeFreeFMM(depth=6, order=4, enable_packing=False,
                                   enable_greedy_aggregation=False,
                                   enable_bitboard_skip=False,
                                   enable_direct_strides=False).evaluate(pos, q)
check("direct_strides+bitboard lossless vs disabled",
      np.array_equal(pot_base, pot_ns))

# bitboard skip alone
pot_bb, _ = VoxelPackedTreeFreeFMM(depth=6, order=4, enable_packing=False,
                                   enable_greedy_aggregation=False,
                                   enable_bitboard_skip=True,
                                   enable_direct_strides=False).evaluate(pos, q)
check("bitboard skip lossless", np.array_equal(pot_bb, pot_ns))

# lossy flags: measured and bounded (documented ~0.12 packing, ~0.25 greedy)
pot_p, _ = VoxelPackedTreeFreeFMM(depth=6, order=4, enable_packing=True,
                                  enable_greedy_aggregation=False).evaluate(pos, q)
relp = np.linalg.norm(pot_p - ref) / np.linalg.norm(ref)
check("packing lossy but bounded (< 0.2)", relp < 0.2, f"rel={relp:.2e}")
pot_g, _ = VoxelPackedTreeFreeFMM(depth=6, order=4, enable_packing=False,
                                  enable_greedy_aggregation=True).evaluate(pos, q)
relg = np.linalg.norm(pot_g - ref) / np.linalg.norm(ref)
check("greedy lossy but bounded (< 0.4)", relg < 0.4, f"rel={relg:.2e}")

# N=0 and N=1
p0, m0 = VoxelPackedTreeFreeFMM().evaluate(np.empty((0, 2)), np.empty(0))
check("FMM N=0 -> empty potentials", p0.shape == (0,))
p1, m1 = VoxelPackedTreeFreeFMM().evaluate(np.array([[0.31, 0.72]]), np.array([0.7]))
check("FMM N=1 -> zero self potential", p1.shape == (1,) and abs(p1[0]) < 1e-12,
      f"pot={p1}")

# coincident particles: no NaN
pc = np.tile([0.4, 0.4], (20, 1)) + rngf.standard_normal((20, 2)) * 1e-9
pcc, _ = VoxelPackedTreeFreeFMM(enable_packing=False, enable_greedy_aggregation=False).evaluate(pc, np.ones(20))
check("FMM coincident cluster: finite output", bool(np.all(np.isfinite(pcc))))

# depth > 6 with packing raises (contract guard)
try:
    VoxelPackedTreeFreeFMM(depth=7, enable_packing=True)
    check("depth>6 + packing raises", False)
except ValueError:
    check("depth>6 + packing raises", True)

# =========================================================================
print()
print(f"QUANTIZED PROBE: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILURES:")
    for f in FAIL:
        print("  -", f)
sys.exit(1 if FAIL else 0)
