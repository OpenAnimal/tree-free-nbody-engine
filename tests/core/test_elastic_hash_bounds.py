"""Worst-case bound verification for the funnel hash table
(core.elastic_hash.ElasticHashTable, Farach-Colton, Krapivin, & Kuszmaul
(2025), Section 3).

The module docstring's deterministic guarantee: every search inspects at
most

    alpha*beta + b_attempts + 2*c_bucket_slots

slot inspections (alpha slabs x beta-slot sub-array scans + overflow B's
b_attempts uniform probes + the two C two-choice buckets of c_bucket_slots
slots), for loads up to 1 - delta, with inserts never displacing a resident
key and no key dropping (probability 1 - n^{-omega(1)}).

These tests instantiate that bound empirically at the design load with
seeded randomized key sets, delta = 1/8 and one smaller (1/64), and after
delete/reinsert churn:

1. Insert to load >= 1 - delta (full rated capacity) -> ZERO drops,
   every key found with its value, and the measured MAX search-probe
   count (hits and absent keys) within the theoretical bound.
2. Delete/reinsert churn (bounded by the physically free slots, since
   funnel tombstones are never reclaimed): all removals succeed,
   survivors stay findable at IDENTICAL probe counts (no reordering),
   reinserted fresh keys cause ZERO drops, and the post-churn max
   search-probe count is still within the bound.

Run:  python -m pytest tests/core/test_elastic_hash_bounds.py -q
      python -m tests.core.test_elastic_hash_bounds   (standalone)
"""

import math
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.elastic_hash import ElasticHashTable  # noqa: E402

CAPACITY = 50_000
DELTAS = (0.125, 1.0 / 64.0)  # 1/8 and one smaller
SEEDS = (1234, 5678)
ABSENT_N = 300


def _distinct_keys(rng: np.random.RandomState, n: int, lo: int = 0,
                    hi: int = 1 << 48) -> np.ndarray:
    keys = rng.randint(lo, hi, size=n, dtype=np.int64)
    while np.unique(keys).size != n:
        keys = np.unique(np.concatenate([keys, rng.randint(lo, hi, size=n, dtype=np.int64)]))[:n]
    return keys


def _theoretical_bound(t: ElasticHashTable) -> int:
    """alpha*beta + B attempts + 2*C bucket slots (module docstring form)."""
    return (t.alpha * t.beta) + t.b_attempts + 2 * t.c_bucket_slots


def _build_full(delta: float, seed: int):
    rng = np.random.RandomState(seed)
    keys = _distinct_keys(rng, CAPACITY)
    t = ElasticHashTable(capacity=CAPACITY, delta=delta, seed=seed)

    drops = 0
    for i, k in enumerate(keys):
        ok, _ = t.insert(int(k), i)
        drops += 0 if ok else 1
    return t, keys, rng, drops


def _lookup_all(t: ElasticHashTable, keys: np.ndarray):
    """Returns (probes array, number of misses, number of wrong values)."""
    probes = np.empty(len(keys), dtype=np.int64)
    misses = wrong = 0
    for i, k in enumerate(keys):
        v, pr = t.lookup(int(k))
        probes[i] = pr
        if v is None:
            misses += 1
        elif v != i:
            wrong += 1
    return probes, misses, wrong


@pytest.mark.parametrize("delta", DELTAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_funnel_max_probes_within_bound_at_design_load(delta, seed):
    t, keys, rng, drops = _build_full(delta, seed)

    # Realized load must be at the design load 1-delta (the slab sizes are
    # rounded up to multiples of beta, so total_size can exceed
    # capacity/(1-delta) by a few slots -- tolerate that rounding).
    load = t.count / t.total_size
    assert load >= 1.0 - delta - 16.0 / t.total_size, (
        f"load {load} below design 1-delta={1.0 - delta}")
    assert load <= 1.0, "load cannot exceed 1"

    bound = _theoretical_bound(t)
    assert bound == t.probe_bound  # property and docstring bound agree

    # No key ever drops at the rated capacity.
    assert drops == 0, f"{drops} keys DROPPED at delta={delta}, seed={seed}"
    assert t.count == CAPACITY

    probes, misses, wrong = _lookup_all(t, keys)
    assert misses == 0 and wrong == 0

    # THE assertion: measured max search-probe count within the bound.
    assert int(probes.max()) <= bound, (
        f"max hit probes {int(probes.max())} > bound {bound} "
        f"(delta={delta}, seed={seed})")

    # Absent-key searches: every probe count within the bound (the funnel
    # absent search is a fixed-length sequence, so max == bound exactly).
    absent = _distinct_keys(rng, ABSENT_N, lo=1 << 48, hi=1 << 49)
    abs_max = 0
    false_hits = 0
    for k in absent:
        v, pr = t.lookup(int(k))
        abs_max = max(abs_max, pr)
        if v is not None:
            false_hits += 1
    assert abs_max <= bound
    assert false_hits == 0


@pytest.mark.parametrize("delta", DELTAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_funnel_bound_and_no_drops_after_churn(delta, seed):
    t, keys, rng, drops = _build_full(delta, seed)
    assert drops == 0

    bound = _theoretical_bound(t)
    probes_before, misses, _ = _lookup_all(t, keys)
    assert misses == 0

    # Churn volume: bounded by the physically free slots, because funnel
    # tombstones are never reclaimed (the slot is not reused). Keep 20%
    # headroom below the free-slot count.
    free_slots = int(t.total_size - t.occupied.sum())
    d = max(1, min(CAPACITY // 10, int(0.8 * free_slots)))

    del_idx = rng.choice(CAPACITY, size=d, replace=False)
    removed = sum(1 for i in del_idx if t.remove(int(keys[i])))
    assert removed == d, "every removal of a present key must succeed"
    assert t.count == CAPACITY - d

    # Survivors: still findable, at IDENTICAL probe counts (inserts never
    # reorder; deletions tombstone in place) and within the bound.
    survivors = np.delete(keys, del_idx)
    surv_idx = np.delete(np.arange(CAPACITY), del_idx)
    probes_after = np.empty(len(survivors), dtype=np.int64)
    misses = 0
    for j, k in enumerate(survivors):
        v, pr = t.lookup(int(k))
        probes_after[j] = pr
        if v != surv_idx[j]:
            misses += 1
    assert misses == 0
    assert int(probes_after.max()) <= bound
    assert np.array_equal(probes_after, probes_before[surv_idx]), (
        "survivor probe sequences changed after churn (reordering!)")

    # Reinsert d FRESH keys: none may drop.
    fresh = _distinct_keys(rng, d, lo=1 << 49, hi=1 << 50)
    churn_drops = 0
    for k in fresh:
        ok, _ = t.insert(int(k), -1)
        if not ok:
            churn_drops += 1
    assert churn_drops == 0, (
        f"{churn_drops}/{d} fresh keys DROPPED after churn "
        f"(delta={delta}, seed={seed}, free_slots={free_slots})")

    # Post-churn: every survivor AND fresh key findable, bound still holds.
    all_keys = np.concatenate([survivors, fresh])
    expected = np.concatenate([surv_idx, np.full(d, -1, dtype=np.int64)])
    probes_post = np.empty(len(all_keys), dtype=np.int64)
    misses = 0
    for j, k in enumerate(all_keys):
        v, pr = t.lookup(int(k))
        probes_post[j] = pr
        if v != expected[j]:
            misses += 1
    assert misses == 0
    assert int(probes_post.max()) <= bound, (
        f"post-churn max probes {int(probes_post.max())} > bound {bound} "
        f"(delta={delta}, seed={seed})")


if __name__ == "__main__":
    t0 = time.perf_counter()
    failures = 0
    for delta in DELTAS:
        for seed in SEEDS:
            for fn in (test_funnel_max_probes_within_bound_at_design_load,
                       test_funnel_bound_and_no_drops_after_churn):
                try:
                    fn(delta, seed)
                    print(f" [PASS] {fn.__name__} delta={delta:.5f} seed={seed}")
                except AssertionError as e:
                    failures += 1
                    print(f" [FAIL] {fn.__name__} delta={delta:.5f} seed={seed}: {e}")
    print(f"done in {time.perf_counter() - t0:.1f}s, "
          f"{8 - failures}/8 passed")
    sys.exit(1 if failures else 0)
