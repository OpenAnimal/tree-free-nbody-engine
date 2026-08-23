"""Round-10 Wave E regression + durable gates for quantized_bitpacked_optimization.

Regression test (was RED before the fix):
  - R10-E3: pack_particles_64bit_3d silently truncated coordinates for
    depth > 8 (8-bit fields) — pack(0.9) at depth=9 unpacked to 0.4.

Durable independent-oracle gates promoted from
tools/review_round10/probe_wavee_2_quantized.py:
  - Morton inc/dec register arithmetic vs scalar decode-offset-encode
  - FastMortonNeighborTable2D exhaustive vs scalar reference
  - pack/unpack roundtrips and boundaries (64-bit 3D, 32-bit 2D)
  - bitboards vs reference occupancy sets
  - VoxelPackedTreeFreeFMM vs direct O(N^2) sum; lossless flag parity
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from quantized_bitpacked_optimization.direct_morton_stride import (
    FastMortonNeighborTable2D,
    morton_inc_x_2d,
    morton_dec_x_2d,
    morton_inc_y_2d,
    morton_dec_y_2d,
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
from quantized_bitpacked_optimization.packed_vectorized_fmm import (
    VoxelPackedTreeFreeFMM,
)


def _enc2d(ix, iy, depth):
    m = 0
    for b in range(depth):
        m |= ((ix >> b) & 1) << (2 * b)
        m |= ((iy >> b) & 1) << (2 * b + 1)
    return m


def _dec2d(m, depth):
    ix = iy = 0
    for b in range(depth):
        ix |= ((m >> (2 * b)) & 1) << b
        iy |= ((m >> (2 * b + 1)) & 1) << b
    return ix, iy


# ---------------------------------------------------------------------------
# R10-E3 regression: 64-bit packer depth overflow
# ---------------------------------------------------------------------------

def test_pack64_depth_gt_8_raises():
    """The 64-bit layout allocates 8 bits per axis; depth > 8 previously
    wrapped coordinates silently (pack(0.9, depth=9) -> unpack 0.4)."""
    try:
        pack_particles_64bit_3d(np.array([[0.9, 0.5, 0.5]]),
                                np.array([1.0]), depth=9)
    except ValueError:
        return
    raise AssertionError("depth=9 must raise ValueError (8-bit axis fields)")


# ---------------------------------------------------------------------------
# Durable oracle gates
# ---------------------------------------------------------------------------

def test_morton_inc_dec_vs_scalar_reference():
    for depth in (2, 3):
        g = 1 << depth
        for ix in range(g):
            for iy in range(g):
                m = _enc2d(ix, iy, depth)
                if ix + 1 < g:
                    assert morton_inc_x_2d(m) == _enc2d(ix + 1, iy, depth)
                if ix - 1 >= 0:
                    assert morton_dec_x_2d(m) == _enc2d(ix - 1, iy, depth)
                if iy + 1 < g:
                    assert morton_inc_y_2d(m) == _enc2d(ix, iy + 1, depth)
                if iy - 1 >= 0:
                    assert morton_dec_y_2d(m) == _enc2d(ix, iy - 1, depth)


def test_neighbor_table_exhaustive_depth3():
    depth = 3
    g = 1 << depth
    keys = (np.int64(depth) << 24) | np.array(
        [_enc2d(ix, iy, depth) for ix in range(g) for iy in range(g)],
        dtype=np.int64)
    nb = FastMortonNeighborTable2D(depth=depth).get_all_neighbors_batch(keys)
    for t in range(len(keys)):
        ix, iy = _dec2d(int(keys[t]) & 0xFFFFFF, depth)
        for k, (dx, dy) in enumerate([(-1, -1), (0, -1), (1, -1),
                                      (-1, 0), (0, 0), (1, 0),
                                      (-1, 1), (0, 1), (1, 1)]):
            nx, ny = ix + dx, iy + dy
            want = (-1 if not (0 <= nx < g and 0 <= ny < g)
                    else (depth << 24) | _enc2d(nx, ny, depth))
            assert nb[t, k] == want, (t, dx, dy, nb[t, k], want)


def test_pack_unpack_64bit_boundaries():
    rng = np.random.default_rng(20260822)
    depth = 8
    g = 1 << depth
    pos = rng.random((200, 3))
    edges = np.array([k / g for k in (0, 1, 2, g // 2, g - 2, g - 1)])
    for r in range(3):
        pos[:6, r] = rng.choice(edges, 6)
    q = rng.uniform(-10, 10, 200)
    up, uq = unpack_particles_64bit_3d(pack_particles_64bit_3d(pos, q, depth), depth)
    # floor(1/g) + frac(1/(256 g)) quantization, plus float32 output rounding
    assert np.max(np.abs(up - pos)) <= 1.6 / (256.0 * g)
    assert np.max(np.abs(uq - q) / np.maximum(1e-6, np.abs(q))) <= 2 ** -10
    # position 1.0 clips into the last cell (no wraparound)
    upp, _ = unpack_particles_64bit_3d(
        pack_particles_64bit_3d(np.ones((3, 3)), np.ones(3)))
    assert np.all(upp < 1.0)


def test_pack_unpack_32bit_signedness_and_clipping():
    rng = np.random.default_rng(5)
    pos = rng.random((200, 2))
    q = rng.uniform(-2, 2, 200)
    up, uq = unpack_particles_32bit_2d(pack_particles_32bit_2d(pos, q))
    assert np.max(np.abs(up - pos)) <= 1.6 / (64.0 * 64.0)
    assert np.max(np.abs(uq - q)) <= 1.0 / 64 + 1e-12
    # exact 1/64 grid multiples roundtrip exactly (signedness preserved)
    qg = np.arange(-128, 128) / 64.0
    _, uqg = unpack_particles_32bit_2d(
        pack_particles_32bit_2d(np.full((len(qg), 2), 0.5), qg))
    assert np.array_equal(uqg, qg)
    # out-of-domain positions clip (no wraparound)
    _, upo = unpack_particles_32bit_2d(pack_particles_32bit_2d(
        np.array([[-0.5, 1.5], [2.0, -3.0]]), np.zeros(2)))
    assert np.all(upo >= 0.0) and np.all(upo < 1.0)


def test_bitboards_match_reference_occupancy():
    rng = np.random.default_rng(9)
    for _ in range(5):
        n = int(rng.integers(1, 300))
        ix = rng.integers(0, 64, n)
        iy = rng.integers(0, 64, n)
        bb = MortonBitboard2D()
        bb.populate(ix, iy, depth=6)
        assert set(bb.iter_active_cells()) == set(zip(ix.tolist(), iy.tolist()))
        assert bb.active_cell_count() == len(set(zip(ix.tolist(), iy.tolist())))
    for _ in range(3):
        n = int(rng.integers(1, 200))
        ix = rng.integers(0, 64, n)
        iy = rng.integers(0, 64, n)
        iz = rng.integers(0, 64, n)
        bb3 = MortonBitboard3D()
        bb3.populate(ix, iy, iz, depth=6)
        assert set(bb3.iter_active_cells()) == set(zip(ix.tolist(), iy.tolist(), iz.tolist()))


def _direct_pot(pos, q):
    z = pos[:, 0] + 1j * pos[:, 1]
    pot = np.zeros(len(pos))
    for i in range(len(pos)):
        d = z[i] - z
        d[i] = 1.0
        pot[i] = np.sum(q * np.log(np.abs(d)))
    return pot


def test_fmm_baseline_vs_direct_and_lossless_flags():
    rng = np.random.default_rng(42)
    pos = rng.random((300, 2))
    q = rng.uniform(-1, 1, 300)
    ref = _direct_pot(pos, q)

    pot, _ = VoxelPackedTreeFreeFMM(depth=6, order=4, enable_packing=False,
                                    enable_greedy_aggregation=False,
                                    enable_bitboard_skip=True,
                                    enable_direct_strides=True).evaluate(pos, q)
    rel = np.linalg.norm(pot - ref) / np.linalg.norm(ref)
    assert rel < 5e-3, f"baseline FMM rel-L2 vs direct = {rel:.3e}"

    # direct strides + bitboard skip must be lossless relative to disabled
    pot_off, _ = VoxelPackedTreeFreeFMM(depth=6, order=4, enable_packing=False,
                                        enable_greedy_aggregation=False,
                                        enable_bitboard_skip=False,
                                        enable_direct_strides=False).evaluate(pos, q)
    assert np.array_equal(pot, pot_off)

    # documented lossy flags stay bounded (quantization error is N-dependent:
    # measured 0.29 packing / 0.55 greedy at this N; the bound guards against
    # catastrophic regressions, not exact quantization error)
    pot_p, _ = VoxelPackedTreeFreeFMM(depth=6, order=4, enable_packing=True,
                                      enable_greedy_aggregation=False).evaluate(pos, q)
    assert np.linalg.norm(pot_p - ref) / np.linalg.norm(ref) < 0.5
    pot_g, _ = VoxelPackedTreeFreeFMM(depth=6, order=4, enable_packing=False,
                                      enable_greedy_aggregation=True).evaluate(pos, q)
    assert np.linalg.norm(pot_g - ref) / np.linalg.norm(ref) < 0.7


def test_fmm_degenerate_inputs():
    p0, _ = VoxelPackedTreeFreeFMM().evaluate(np.empty((0, 2)), np.empty(0))
    assert p0.shape == (0,)
    p1, _ = VoxelPackedTreeFreeFMM().evaluate(np.array([[0.31, 0.72]]),
                                              np.array([0.7]))
    assert p1.shape == (1,) and abs(p1[0]) < 1e-12
    rng = np.random.default_rng(2)
    pc = np.tile([0.4, 0.4], (15, 1)) + rng.standard_normal((15, 2)) * 1e-9
    pcc, _ = VoxelPackedTreeFreeFMM(enable_packing=False,
                                    enable_greedy_aggregation=False).evaluate(pc, np.ones(15))
    assert np.all(np.isfinite(pcc))
