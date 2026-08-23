"""
Tests for the repo-wide CellIndex / validation / benchmark-kit plumbing.
Run: python -X utf8 core/test_spatial_index.py
     python -m core.test_spatial_index
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.elastic_hash import ElasticHashTable
from core.spatial_index import CellIndex, morton_2d_key, morton_3d_key
from core.validation import assert_accuracy, cross_validate, fmt_validation


def test_key_roundtrip_2d():
    idx = CellIndex(dims=2, grid_res=32)
    for k in (0, 1, (5 << 12) | 7, (31 << 12) | 31):
        assert idx.key_ints(k) == (k & 0xFFF, k >> 12)
    assert morton_2d_key(7, 5) == (5 << 12) | 7


def test_key_roundtrip_3d():
    for ix, iy, iz in ((0, 0, 0), (5, 3, 9), (1023, 1023, 1023), (17, 999, 42)):
        assert morton_3d_key(ix, iy, iz) == morton_3d_key(ix, iy, iz)
        # interleave then deinterleave must return the original triple
        m = morton_3d_key(ix, iy, iz)
        rx = ry = rz = 0
        for b in range(10):
            rx |= (m >> (2 * b)) & (1 << b)
            ry |= (m >> (2 * b + 1)) & (1 << b)
            rz |= (m >> (2 * b + 2)) & (1 << b)
        assert (rx, ry, rz) == (ix, iy, iz)


def test_build_and_membership():
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 1, (500, 2))
    idx = CellIndex(dims=2, grid_res=16)
    unique, inverse = idx.build(pos)
    assert len(idx) == len(unique)
    # every occupied key is found by hash probe, every item in exactly one bucket
    total = 0
    for k, bucket in idx.items():
        assert idx.cell_id(k) is not None
        assert idx.bucket(k) is not None and len(bucket) > 0
        total += len(bucket)
    assert total == 500
    assert sum(1 for k in idx.occupied_keys()) == len(unique)
    # unoccupied-key rejection: pick an in-grid key (coords within
    # 0..grid_res-1) that is NOT occupied and assert cell_id returns None.
    # (The previous code used `if False else None` and an `... or True`
    # tautology, so the assert was vacuous.)
    occupied = set(int(k) for k in idx.occupied_keys())
    free_key = None
    for ix in range(16):
        for iy in range(16):
            k = morton_2d_key(ix, iy)
            if k not in occupied:
                free_key = k
                break
        if free_key is not None:
            break
    assert free_key is not None, "no unoccupied in-grid key found (grid full?)"
    assert idx.cell_id(free_key) is None, (
        f"unoccupied key {free_key} (coords "
        f"{idx.key_ints(free_key)}) reported as occupied")


def test_neighbor_superset_property_2d():
    """Any pair closer than one cell width lies in each other's ring-1 union."""
    rng = np.random.default_rng(2)
    pos = rng.uniform(0, 1, (300, 2))
    idx = CellIndex(dims=2, grid_res=8)
    idx.build(pos)
    keys = [idx.key_of(p) for p in pos]
    for i in rng.choice(len(pos), 40, replace=False):
        d = np.linalg.norm(pos - pos[i], axis=1)
        true_disk = set(np.flatnonzero(d < 1.0 / 8).tolist()) - {int(i)}
        near = set(idx.neighborhood_indices(keys[i], ring=1).tolist()) - {int(i)}
        assert true_disk <= near, f"pair missed by hash neighborhood for agent {i}"


def test_neighbor_superset_property_3d():
    rng = np.random.default_rng(3)
    pos = rng.uniform(-5, 5, (300, 3))
    idx = CellIndex(dims=3, cell_size=1.0)
    idx.build(pos)
    keys = [idx.key_of(p) for p in pos]
    for i in rng.choice(len(pos), 30, replace=False):
        d = np.linalg.norm(pos - pos[i], axis=1)
        true_ball = set(np.flatnonzero(d < 1.0).tolist()) - {int(i)}
        near = set(idx.neighborhood_indices(keys[i], ring=1).tolist()) - {int(i)}
        assert true_ball <= near


def test_far_keys_partition():
    rng = np.random.default_rng(4)
    pos = rng.uniform(0, 1, (200, 2))
    idx = CellIndex(dims=2, grid_res=8)
    idx.build(pos)
    k = idx.occupied_keys()[0]
    near = set(idx.neighbor_keys(k, ring=1))
    far = set(idx.far_keys(k, ring=1))
    assert near | far == set(idx.occupied_keys())
    assert not (near & far)
    assert k in near


def test_moments():
    rng = np.random.default_rng(5)
    pos = rng.uniform(0, 1, (400, 2))
    w = rng.uniform(0.5, 2.0, 400)
    idx = CellIndex(dims=2, grid_res=8)
    idx.build(pos)
    keys, inv, counts, centroids, totals = idx.moments(pos, w)
    assert abs(counts.sum() - 400) < 1e-9
    assert abs(totals.sum() - w.sum()) < 1e-9
    # centroid of each cell lies within its own cell bounds
    r = 8
    for c, k in enumerate(keys):
        ix, iy = idx.key_ints(k)
        cx, cy = centroids[c]
        assert ix / r <= cx < (ix + 1) / r + 1e-12
        assert iy / r <= cy < (iy + 1) / r + 1e-12


def test_rebuild_drops_stale_keys():
    rng = np.random.default_rng(6)
    a = rng.uniform(0, 0.5, (100, 2))
    b = rng.uniform(0.5, 1.0, (100, 2))
    idx = CellIndex(dims=2, grid_res=16)
    ka, _ = idx.build(a)
    kb, _ = idx.build(b)
    for k in ka:
        if k not in kb:
            assert idx.cell_id(int(k)) is None, "stale key survived rebuild"


def test_cross_validate_convention():
    exact = np.array([1.0, 2.0, 3.0])
    res = cross_validate(exact * 1.01, exact, name="unit-test")
    assert 0.005 < res["rel_l2"] < 0.02
    assert res["cosine"] > 0.9999
    assert "rel L2" in fmt_validation(res)
    assert_accuracy(res, 0.05, "unit-test")


def test_benchmark_kit_table():
    from core.benchmark_kit import VariantBenchmark
    rng = np.random.default_rng(7)
    A = rng.uniform(0, 1, (50, 50))
    bench = VariantBenchmark("unit")
    bench.add("standard", lambda: A @ A, note="dense reference")
    bench.add("+elastichash", lambda: A @ (A * 0.999), accuracy_vs="standard", note="approx")
    rows = bench.run(print_table=True)
    assert len(rows) == 2
    assert "rel_l2" in rows[1] and rows[1]["rel_l2"] > 0


if __name__ == "__main__":
    tests = [
        test_key_roundtrip_2d, test_key_roundtrip_3d, test_build_and_membership,
        test_neighbor_superset_property_2d, test_neighbor_superset_property_3d,
        test_far_keys_partition, test_moments, test_rebuild_drops_stale_keys,
        test_cross_validate_convention, test_benchmark_kit_table,
    ]
    passed = 0
    for fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            print(f" [PASS] {fn.__name__:<45} ({(time.perf_counter()-t0)*1000:.1f} ms)")
            passed += 1
        except AssertionError as e:
            print(f" [FAIL] {fn.__name__}: {e}")
    print("=" * 70)
    print(f" Result: {passed}/{len(tests)} tests passed.")
    assert passed == len(tests)
