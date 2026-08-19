"""Round-5 task 5.3: Python vs Zig funnel-hash probe-count and throughput
comparison.

Seeds the SAME 1M splitmix64 keys as `zig/bench.zig` (capacity=1_000_000,
delta=0.05, seed=42), inserts them into the Python
`core.elastic_hash.ElasticHashTable`, then runs the hit-lookup and
absent-key-lookup passes. Prints mean probes per pass next to the Zig
reference numbers (hardcoded below from a fresh `zig build run --release=fast`
run on this machine, 2026-08-19) and a Python-vs-Zig throughput row.

The Python salts use MT19937 while the Zig port uses a splitmix64-seeded
LCG, so exact slot assignments differ — but the slab geometry (alpha, beta,
overflow sizes, probe_bound) is identical and the probe COUNTS are what we
compare (geometry-driven, not salt-driven).

Standalone:  python -X utf8 tools/compare_hash_python_zig.py
"""
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.elastic_hash import ElasticHashTable


# ---- Zig reference (fresh run, zig build run --release=fast, 2026-08-19) ----
# geometry: alpha=28 beta=9 total_slots=1,052,640 probe_bound=277
# --- INSERT ---           14.64 M keys/s, mean probes 32.9630
# --- LOOKUP (hits) ---    14.07 M keys/s, mean probes 28.9559
# --- LOOKUP (absent) ---   2.62 M keys/s, mean probes 277.0000
ZIG_REF = {
    "geometry": "alpha=28 beta=9 total_slots=1052640 probe_bound=277",
    "insert_mps": 14.64,
    "insert_probes": 32.9630,
    "hit_mps": 14.07,
    "hit_probes": 28.9559,
    "absent_mps": 2.62,
    "absent_probes": 277.0000,
}


def _splitmix64_next(state: int) -> int:
    """splitmix64 step (matches zig/bench.zig nextKey)."""
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
    z = z ^ (z >> 31)
    return state, z


def _generate_keys(capacity: int, seed: int) -> np.ndarray:
    state = seed
    out = np.empty(capacity, dtype=np.uint64)
    for i in range(capacity):
        state, k = _splitmix64_next(state)
        out[i] = k
    return out


def main(capacity: int = 1_000_000, delta: float = 0.05, seed: int = 42):
    print("Python vs Zig funnel-hash comparison (round-5 task 5.3)")
    print(f"  capacity = {capacity}")
    print(f"  delta    = {delta}")
    print(f"  seed     = {seed}")

    keys = _generate_keys(capacity, seed)
    # Zig treats the u64 key as i64 via @bitCast; Python ElasticHashTable
    # takes ints that fit in signed 64-bit. Reinterpret as signed.
    keys_signed = keys.astype(np.int64)

    ht = ElasticHashTable(capacity=capacity, delta=delta, seed=seed)
    print(f"  Python geometry: alpha={ht.alpha} beta={ht.beta} "
          f"total_slots={ht.total_size} probe_bound={ht.probe_bound}")
    print(f"  Zig    geometry: {ZIG_REF['geometry']}")
    assert (ht.alpha, ht.beta) == (28, 9), (
        f"geometry mismatch: Python alpha/beta={ht.alpha}/{ht.beta} != 28/9")
    assert ht.probe_bound == 277, (
        f"probe_bound mismatch: Python={ht.probe_bound} != 277")

    # ---- INSERT ----
    t0 = time.perf_counter()
    total_insert_probes = 0
    insert_ok = 0
    for i in range(capacity):
        ok, pr = ht.insert(int(keys_signed[i]), i)
        total_insert_probes += pr
        if ok:
            insert_ok += 1
    t_ins = time.perf_counter() - t0
    ins_mps = capacity / t_ins / 1e6
    ins_mean_probes = total_insert_probes / capacity

    # ---- LOOKUP (hits, shuffled) ----
    rng = np.random.default_rng(seed ^ 0xDEADBEEF)
    order = rng.permutation(capacity)
    t0 = time.perf_counter()
    total_hit_probes = 0
    hit_count = 0
    for idx in order:
        v, pr = ht.lookup(int(keys_signed[idx]))
        total_hit_probes += pr
        if v is not None:
            hit_count += 1
    t_hit = time.perf_counter() - t0
    hit_mps = capacity / t_hit / 1e6
    hit_mean_probes = total_hit_probes / capacity

    # ---- LOOKUP (absent, high-bit space) ----
    absent_state = seed ^ 0xCAFEBABE
    t0 = time.perf_counter()
    total_abs_probes = 0
    false_hits = 0
    for _ in range(capacity):
        absent_state, k = _splitmix64_next(absent_state)
        k |= 0x8000000000000000
        # reinterpret as signed
        if k >= (1 << 63):
            k_signed = k - (1 << 64)
        else:
            k_signed = k
        v, pr = ht.lookup(k_signed)
        total_abs_probes += pr
        if v is not None:
            false_hits += 1
    t_abs = time.perf_counter() - t0
    abs_mps = capacity / t_abs / 1e6
    abs_mean_probes = total_abs_probes / capacity

    # ---- Report ----
    print()
    print(f"{'Pass':<22} {'Python mps':>12} {'Zig mps':>10} {'Py/Zig':>8} "
          f"{'Py probes':>10} {'Zig probes':>11} {'Py/Zig':>8}")
    print("-" * 90)
    for label, py_mps, py_pr, zig_mps, zig_pr in (
        ("INSERT", ins_mps, ins_mean_probes, ZIG_REF["insert_mps"], ZIG_REF["insert_probes"]),
        ("LOOKUP (hits)", hit_mps, hit_mean_probes, ZIG_REF["hit_mps"], ZIG_REF["hit_probes"]),
        ("LOOKUP (absent)", abs_mps, abs_mean_probes, ZIG_REF["absent_mps"], ZIG_REF["absent_probes"]),
    ):
        mps_ratio = py_mps / zig_mps
        pr_ratio = py_pr / zig_pr
        print(f"{label:<22} {py_mps:>12.2f} {zig_mps:>10.2f} "
              f"{mps_ratio:>8.4f} {py_pr:>10.4f} {zig_pr:>11.4f} "
              f"{pr_ratio:>8.4f}")
    print("-" * 90)
    print(f"insert_ok     = {insert_ok} / {capacity}")
    print(f"lookup hits   = {hit_count} / {capacity}")
    print(f"absent false  = {false_hits} (must be 0)")
    print(f"probe_bound   = {ht.probe_bound} (Zig: 277)")
    print(f"absent probes = {abs_mean_probes:.4f} (worst case = probe_bound)")

    # Acceptance check: Python mean probes within 2x of Zig.
    ok = True
    for label, py_pr, zig_pr in (
        ("INSERT", ins_mean_probes, ZIG_REF["insert_probes"]),
        ("LOOKUP (hits)", hit_mean_probes, ZIG_REF["hit_probes"]),
        ("LOOKUP (absent)", abs_mean_probes, ZIG_REF["absent_probes"]),
    ):
        ratio = py_pr / zig_pr
        if not (0.5 <= ratio <= 2.0):
            print(f"  *** {label} probe ratio {ratio:.3f} outside [0.5, 2.0] ***")
            ok = False
    if false_hits != 0:
        print(f"  *** absent false hits = {false_hits} != 0 ***")
        ok = False
    if insert_ok != capacity or hit_count != capacity:
        print("  *** insert/lookup completeness failed ***")
        ok = False
    print()
    print(f"=== {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
