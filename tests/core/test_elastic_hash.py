"""Property and stress tests for core/elastic_hash.py.

Run:  python core/test_elastic_hash.py      (repo runner style)
      python -m core.test_elastic_hash      (or pytest core/test_elastic_hash.py)

Covers BOTH tables from Farach-Colton, Krapivin, & Kuszmaul (2025),
"Optimal Bounds for Open Addressing Without Reordering"
(arXiv:2501.02305, FOCS 2024):

A. ElasticHashTable — the DEFAULT funnel-hashing table (paper Section 3):
   1. Geometry conformance: alpha = ceil(4 log2(1/delta) + 10) slabs of
      beta = ceil(2 log2(1/delta))-slot sub-arrays shrinking by ~3/4,
      overflow region within [ceil(delta*n/2), floor(3*delta*n/4)], and
      the structural funnel fact sum_{j>i}|A_j| > 2.5|A_i|.
   2. Stress at loads 0.5/0.8/0.95/0.99: ZERO dropped keys, ZERO false
      negatives, no false positives, every lookup within the deterministic
      probe bound alpha*beta + O(log log n), monotone per-decile insert
      probe counts, and the worst-case sanity bound
      max probes <= 8 * log2(1/delta) * beta.
   3. The vectorized batch probe (funnel_probe, historical alias
      jax_hash_probe) follows the identical probe sequence and finds every
      inserted key / rejects absent ones.
   4. No reordering: a resident key's slot never changes.
   5. Small tables, capacity gating, update-in-place, API compatibility
      with the previous release.

B. ElasticBatchingHashTable — the secondary elastic-hashing variant
   (paper Section 2, simplified greedy form): soundness/completeness at
   high load, no reordering, tombstone deletes, probe statistics.
"""

import os
import random
import statistics
import sys
import time
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from .elastic_hash import (
        ElasticHashTable,
        ElasticBatchingHashTable,
        funnel_probe,
        jax_hash_probe,
        _hash2d,
    )
except ImportError:
    from core.elastic_hash import (
        ElasticHashTable,
        ElasticBatchingHashTable,
        ElasticIntTable,
        funnel_probe,
        jax_hash_probe,
        _hash2d,
    )

CAPACITY = 100_000
DELTA = 0.05
LOADS = (0.5, 0.8, 0.95, 0.99)


def _distinct_keys(rng: np.random.RandomState, n: int) -> np.ndarray:
    """n distinct random keys from a 2^48 universe (collision chance ~1e-5)."""
    keys = rng.randint(0, 1 << 48, size=n, dtype=np.int64)
    while np.unique(keys).size != n:
        keys = rng.randint(0, 1 << 48, size=n, dtype=np.int64)
    return keys


# =============================================================================
# A. FUNNEL HASH TABLE (DEFAULT) — geometry, stress, bounds
# =============================================================================

def test_funnel_geometry_matches_paper():
    for delta in (0.125, 0.05, 0.01):
        t = ElasticHashTable(capacity=100_000, delta=delta)
        log_inv = math.log2(1.0 / delta)

        # Paper: alpha = ceil(4 log 1/delta + 10), beta = ceil(2 log 1/delta).
        assert t.alpha == math.ceil(4.0 * log_inv + 10.0), \
            f"alpha {t.alpha} != ceil(4 log 1/delta + 10)"
        assert t.beta == math.ceil(2.0 * log_inv), \
            f"beta {t.beta} != ceil(2 log 1/delta)"

        # Slab sizes are multiples of beta, non-increasing.
        assert all(s % t.beta == 0 for s in t.slab_sizes), "slab size not multiple of beta"
        assert all(t.slab_sizes[i] >= t.slab_sizes[i + 1]
                   for i in range(t.alpha - 1)), "slab sizes not non-increasing"

        # Total accounting: slabs + overflow == total slots.
        assert sum(t.slab_sizes) + t.overflow_size == t.total_size, \
            "slabs + overflow != total slots"

        # Overflow region inside the paper's range when the table is large
        # enough for that range to exist (clamped to >= 16 on tiny tables).
        n = t.total_size
        if math.floor(3.0 * delta * n / 4.0) >= 16:
            assert math.ceil(delta * n / 2.0) <= t.overflow_size, \
                f"overflow {t.overflow_size} < ceil(delta*n/2) = {math.ceil(delta * n / 2.0)}"
            assert t.overflow_size <= math.floor(3.0 * delta * n / 4.0), \
                f"overflow {t.overflow_size} > floor(3*delta*n/4) = {math.floor(3.0 * delta * n / 4.0)}"

        # Shrink factor ~3/4 between consecutive slabs (away from tiny tail).
        a = t.subarray_counts
        for i in range(t.alpha - 1):
            if a[i] >= 8:
                assert 0.60 * a[i] <= a[i + 1] <= 0.90 * a[i], \
                    f"slab shrink ratio broken at i={i}: {a[i]} -> {a[i+1]}"

        # Structural funnel fact (paper): sum_{j>i} |A_j| > 2.5 |A_i|.
        for i in range(t.alpha - 10):
            if a[i] >= 100:
                assert sum(a[i + 1:]) > 2.5 * a[i]


def _stress_single_load(load: float, seed: int) -> dict:
    rng = np.random.RandomState(seed)
    n_keys = int(round(load * CAPACITY))
    keys = _distinct_keys(rng, n_keys)

    t = ElasticHashTable(capacity=CAPACITY, delta=DELTA)

    drops = 0
    insert_probes = np.empty(n_keys, dtype=np.int64)
    for i in range(n_keys):
        ok, pr = t.insert(int(keys[i]), i)
        insert_probes[i] = pr
        if not ok:
            drops += 1

    false_negatives = 0
    wrong_values = 0
    lookup_probes = np.empty(n_keys, dtype=np.int64)
    for i in range(n_keys):
        v, pr = t.lookup(int(keys[i]))
        lookup_probes[i] = pr
        if v is None:
            false_negatives += 1
        elif v != i:
            wrong_values += 1

    # Absent keys must return None (no false positives).
    keyset = set(int(k) for k in keys)
    absent = []
    while len(absent) < 2000:
        cand = int(rng.randint(0, 1 << 48, dtype=np.int64))
        if cand not in keyset:
            absent.append(cand)
    false_positives = 0
    max_absent_probes = 0
    for k in absent:
        v, pr = t.lookup(k)
        if v is not None:
            false_positives += 1
        max_absent_probes = max(max_absent_probes, pr)

    # Vectorized batch probe must mirror the scalar search exactly.
    sample = keys[:: max(1, n_keys // 10000)][:10000]
    slots = funnel_probe(t, sample)
    batch_ok = bool(np.all(t.keys[slots] == sample))
    absent_arr = np.array(absent[:1000], dtype=np.int64)
    batch_absent_ok = bool(np.all(funnel_probe(t, absent_arr) == -1))
    assert jax_hash_probe is funnel_probe  # historical alias

    # Mean insert probes per fill-decile (monotonicity check).
    decile_means = [float(insert_probes[d * n_keys // 10:(d + 1) * n_keys // 10].mean())
                    for d in range(10)]

    return {
        "load": load, "n_keys": n_keys, "slots": t.total_size,
        "alpha": t.alpha, "beta": t.beta, "probe_bound": t.probe_bound,
        "drops": drops, "false_negatives": false_negatives,
        "wrong_values": wrong_values, "false_positives": false_positives,
        "batch_ok": batch_ok, "batch_absent_ok": batch_absent_ok,
        "mean_insert_probes": float(insert_probes.mean()),
        "mean_lookup_probes": float(lookup_probes.mean()),
        "max_lookup_probes": int(lookup_probes.max()),
        "max_absent_probes": max_absent_probes,
        "decile_means": decile_means,
    }


def test_stress_all_loads():
    log_inv = math.log2(1.0 / DELTA)
    sanity_multiple = 8  # max probes must stay <= 8 * log2(1/delta) * beta
    print()
    for idx, load in enumerate(LOADS):
        t0 = time.perf_counter()
        s = _stress_single_load(load, seed=1234 + idx)
        dt = time.perf_counter() - t0

        assert s["drops"] == 0, f"load {load}: {s['drops']} keys DROPPED"
        assert s["false_negatives"] == 0
        assert s["wrong_values"] == 0
        assert s["false_positives"] == 0
        assert s["batch_ok"] and s["batch_absent_ok"]
        assert s["max_lookup_probes"] <= s["probe_bound"]
        assert s["max_absent_probes"] <= s["probe_bound"]
        assert max(s["max_lookup_probes"], s["max_absent_probes"]) \
            <= sanity_multiple * log_inv * s["beta"]

        # Monotone probe counts: decile means non-decreasing (2% slack).
        dm = s["decile_means"]
        for d in range(len(dm) - 1):
            assert dm[d + 1] >= 0.98 * dm[d], \
                f"load {load}: insert probes decreased at decile {d}: " \
                f"{dm[d]:.2f} -> {dm[d+1]:.2f}"
        assert dm[-1] > dm[0]

        print(f"  load {load:.2f}: keys={s['n_keys']:6d}/{s['slots']} slots | "
              f"drops={s['drops']} FN={s['false_negatives']} FP={s['false_positives']} | "
              f"mean ins/look probes={s['mean_insert_probes']:6.2f}/"
              f"{s['mean_lookup_probes']:5.2f} | "
              f"max look={s['max_lookup_probes']} (bound {s['probe_bound']}) | "
              f"batch ok={s['batch_ok']} | {dt:.1f}s")
        print(f"           decile insert-probe means: " +
              " ".join(f"{m:.1f}" for m in dm))


def test_funnel_small_tables():
    for cap in (1, 2, 3, 7, 16, 64, 256, 1024):
        rng = np.random.RandomState(100 + cap)
        keys = _distinct_keys(rng, cap)
        t = ElasticHashTable(capacity=cap, delta=DELTA)
        drops = sum(0 if t.insert(int(k), i)[0] else 1 for i, k in enumerate(keys))
        assert drops == 0, f"capacity {cap}: {drops} drops"
        fn = sum(1 for i, k in enumerate(keys) if t.lookup(int(k))[0] != i)
        assert fn == 0, f"capacity {cap}: {fn} false negatives"
        assert len(t) == cap
        # Full table rejects one more (capacity == load 1-delta reached).
        ok, _ = t.insert(int(rng.randint(1 << 40, dtype=np.int64)), None)
        assert ok is False


def test_funnel_no_reordering():
    rng = np.random.RandomState(3)
    keys = _distinct_keys(rng, 1900)
    t = ElasticHashTable(capacity=2048, delta=DELTA)
    slot_of = {}
    for k in keys:
        t.insert(int(k), int(k))
        slot_of[int(k)] = int(funnel_probe(t, np.array([k]))[0])
    # after all insertions, no earlier key may have moved
    for k, pos in slot_of.items():
        assert t.keys[pos] == k, f"key {k} was displaced (reordering!)"


def test_funnel_api_compatibility_and_updates():
    rng = np.random.RandomState(7)
    keys = _distinct_keys(rng, 500)
    t = ElasticHashTable(capacity=1000, delta=DELTA)

    # Historical attributes exist.
    assert t.capacity == 1000 and t.delta == DELTA and t.count == 0
    assert isinstance(t.level_sizes, list) and isinstance(t.level_offsets, list)
    assert t.total_size == sum(t.level_sizes) + t.overflow_size
    assert t.keys.dtype == np.int64 and t.occupied.dtype == bool
    assert len(t.values) == t.total_size

    for i, k in enumerate(keys):
        ok, probes = t.insert(int(k), i)
        assert ok is True and probes >= 1

    # Update-in-place semantics.
    ok, _ = t.insert(int(keys[10]), "new")
    assert ok and t.lookup(int(keys[10]))[0] == "new"
    assert t.count == 500  # update must not create a duplicate

    # Dict-style helpers.
    assert t.get(int(keys[10])) == "new"
    assert t.get(-12345, "miss") == "miss"
    assert int(keys[10]) in t and -12345 not in t
    assert len(t) == 500
    assert 0.0 < t.load_factor() < 1.0
    assert set(k for k, _ in t.items()) == set(int(x) for x in keys)

    # lookup returns the (value, probes) tuple; miss -> (None, probes).
    v, pr = t.lookup(-12345)
    assert v is None and pr == t.probe_bound


# =============================================================================
# B. ELASTIC BATCHING VARIANT (secondary class) — property tests
# =============================================================================

def test_elastic_insert_find_all_keys_high_load():
    rng = random.Random(0)
    for cap, n in [(1024, 900), (16384, 15000), (1000, 950), (64, 60)]:
        h = ElasticBatchingHashTable(capacity=cap, delta=0.05)
        keys = rng.sample(range(10**9), n)
        for k in keys:
            ok, _ = h.insert(k, k * 7)
            assert ok
        for k in keys:
            v, _ = h.lookup(k)
            assert v == k * 7, f"key {k} lost at cap={cap}"
        for k in rng.sample(range(10**9, 2 * 10**9), min(200, cap)):
            assert h.lookup(k)[0] is None, f"false positive for {k}"
        assert h.count == len(set(keys))


def test_elastic_no_reordering():
    h = ElasticBatchingHashTable(capacity=2048, delta=0.05)
    rng = random.Random(1)
    keys = rng.sample(range(10**9), 1900)
    slot_of = {}
    for k in keys:
        h.insert(k, k)
        # locate slot to record position
        for i in range(len(h.subarray_sizes)):
            size_i = h.subarray_sizes[i]
            for j in range(size_i):
                pos = h.offsets[i] + _hash2d(k, i, j, size_i)
                if h.keys[pos] == k:
                    slot_of[k] = pos
                    break
            if k in slot_of and slot_of[k] is not None:
                break
    # after all insertions, no earlier key may have moved
    for k, pos in slot_of.items():
        assert h.keys[pos] == k, f"key {k} was displaced (reordering!)"


def test_elastic_delete_reinsert():
    h = ElasticBatchingHashTable(capacity=256, delta=0.05)
    for k in range(200):
        h.insert(k, str(k))
    for k in range(0, 200, 2):
        assert h.remove(k)
    for k in range(0, 200, 2):
        assert h.lookup(k)[0] is None
    for k in range(1, 200, 2):
        assert h.lookup(k)[0] == str(k)
    for k in range(0, 200, 2):
        ok, _ = h.insert(k, str(k))
        assert ok
    for k in range(200):
        assert h.lookup(k)[0] == str(k)
    assert h.count == 200


def test_elastic_probe_statistics():
    # Measured on the greedy cascade variant: mean ins ~54, mean look ~38,
    # max look ~502 at 95% load. The insert mean increased from ~3.8 to ~54
    # after the correctness fix that prevents duplicate keys by scanning all
    # sub-arrays for an existing copy before placing a new entry.
    rng = random.Random(2)
    h = ElasticBatchingHashTable(capacity=16384, delta=0.05)
    keys = rng.sample(range(10**9), 15500)  # ~95% load
    ins = [h.insert(k, k)[1] for k in keys]
    look = [h.lookup(k)[1] for k in keys]
    assert statistics.mean(ins) < 100, statistics.mean(ins)
    assert statistics.mean(look) < 60, statistics.mean(look)
    assert max(look) < 1500, max(look)


def test_elastic_update_in_place():
    h = ElasticBatchingHashTable(capacity=128, delta=0.05)
    h.insert(5, "a")
    ok, _ = h.insert(5, "b")
    assert ok
    assert h.lookup(5)[0] == "b"
    assert h.count == 1


if __name__ == '__main__':
    print("=" * 80)
    print(" ELASTIC HASH MODULE TEST SUITE")
    print(" A. Funnel hashing (ElasticHashTable, DEFAULT)  -- arXiv:2501.02305 S.3")
    print(" B. Elastic hashing (ElasticBatchingHashTable)   -- arXiv:2501.02305 S.2")
    print("=" * 80)
    tests = [
        ("A: test_funnel_geometry_matches_paper", test_funnel_geometry_matches_paper),
        ("A: test_stress_all_loads", test_stress_all_loads),
        ("A: test_funnel_small_tables", test_funnel_small_tables),
        ("A: test_funnel_no_reordering", test_funnel_no_reordering),
        ("A: test_funnel_api_compatibility_and_updates", test_funnel_api_compatibility_and_updates),
        ("B: test_elastic_insert_find_all_keys_high_load", test_elastic_insert_find_all_keys_high_load),
        ("B: test_elastic_no_reordering", test_elastic_no_reordering),
        ("B: test_elastic_delete_reinsert", test_elastic_delete_reinsert),
        ("B: test_elastic_probe_statistics", test_elastic_probe_statistics),
        ("B: test_elastic_update_in_place", test_elastic_update_in_place),
    ]
    passed = 0
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            dt = (time.perf_counter() - t0) * 1000
            print(f" [PASS] {name:<55} ({dt:.2f} ms)")
            passed += 1
        except Exception as e:
            dt = (time.perf_counter() - t0) * 1000
            print(f" [FAIL] {name:<55} ({dt:.2f} ms) -> {e}")

    print("=" * 80)
    print(f" Result: {passed}/{len(tests)} tests passed successfully.")
    print("=" * 80)
    if passed < len(tests):
        sys.exit(1)


def test_elastic_overflow_count_decremented_on_remove():
    """R10-A1: remove() of an overflow-resident key must decrement
    _overflow_count, otherwise the insert() early-stop stays disabled
    forever after overflow churn (performance-only regression).

    Natural overflow placement is a tail event that sequential keys never
    trigger under random salts, so the overflow-resident entry is placed
    directly (white-box) the way `_fill` would record it."""
    h = ElasticHashTable(capacity=16, delta=0.05)
    pos = h.b_offset  # first slot of the B (uniform probing) region
    h.keys[pos] = 999
    h.values[pos] = "x"
    h.occupied[pos] = True
    h.count += 1
    h._overflow_count += 1
    assert h._overflow_count == 1
    assert h.remove(999) is True
    assert h._overflow_count == 0, (
        "remove() of an overflow key must decrement _overflow_count (R10-A1): "
        "a stale positive count permanently disables the insert() early-stop")
    assert h.lookup(999)[0] is None
    ok, _ = h.insert(999, "again")
    assert ok
    assert h.lookup(999)[0] == "again"


def test_elastic_int_table_remove_overflow_count():
    """Same R10-A1 invariant for the int-valued subclass."""
    h = ElasticIntTable(capacity=16, delta=0.05)
    pos = h.c_offset  # a C-region (two-choice) slot
    h.keys[pos] = 123
    h.values[pos] = 7
    h.occupied[pos] = True
    h.count += 1
    h._overflow_count += 1
    assert h.remove(123) is True
    assert h._overflow_count == 0
