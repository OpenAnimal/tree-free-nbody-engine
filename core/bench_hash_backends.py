"""
Head-to-head hash-backend benchmark: funnel hashing vs baselines.

Backends
--------
  funnel   core.elastic_hash.ElasticHashTable -- the funnel hash table of
           Farach-Colton, Krapivin, & Kuszmaul (2025), "Optimal Bounds for
           Open Addressing Without Reordering" (arXiv:2501.02305, FOCS 2024),
           Section 3. The default occupied-cell index of every core FMM
           engine in this repo.
  linear   A fair open-addressing LINEAR-PROBE baseline implemented here
           (none existed in the repo): same splitmix64 finalizer, same
           slot budget, in-band -1/-2 sentinels, classic tombstone
           deletion. Scalar per-key path mirrors the funnel table's Python
           conditions exactly (numpy-array scalar indexing, Python loop);
           a NumPy-vectorized driver replaces it only where the scalar
           path would exceed the probe budget (alpha >= 0.95 at large n) --
           the row's "impl" column says which.
  dict     CPython dict (the practical pure-Python baseline; C
           implementation, included for honesty).
  zig      The compiled funnel port (zig/funnel_hash.zig), run via
           `zig build run --release=fast` (skipped with a note if the Zig
           toolchain is missing). Loads >= 0.9 only: the Zig harness
           inserts to its rated capacity so realized load = 1 - delta,
           and delta is clamped to 1/8 by the paper's analysis.

What this benchmark measures per (n, alpha): build throughput, hit-lookup
throughput, MEAN and WORST-CASE probe counts per op (hits and absent keys),
key drops, structural memory, and delete/reinsert churn behavior. The
funnel table's selling points are the DETERMINISTIC worst-case probe bound
(alpha*beta + b_attempts + 2*c_bucket_slots slot inspections for ANY
search), no-reordering inserts that never displace a resident key, and
compactness -- NOT raw pure-Python throughput (the C dict wins that; the
prose in BENCHMARKS.md says so plainly).

Standalone (repo root):
    python -X utf8 core/bench_hash_backends.py --quick
    python -X utf8 core/bench_hash_backends.py            # few minutes
    python -X utf8 core/bench_hash_backends.py --ns 100000
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.elastic_hash import (ElasticHashTable, funnel_probe,  # noqa: E402
                               _mix64, _mix64_arr)

_U63 = np.int64((1 << 63) - 1)
_LINEAR_EMPTY = -1
_LINEAR_TOMB = -2


# =============================================================================
# Key generation (same splitmix64 stream family as zig/bench.zig, masked to
# 63 bits so keys never collide with the linear baseline's -1/-2 sentinels;
# the funnel table uses an explicit occupied flag and is unaffected).
# =============================================================================

def gen_keys(n: int, seed: int) -> np.ndarray:
    """n distinct non-negative int64 keys from a splitmix64 stream."""
    with np.errstate(over="ignore"):
        states = (np.uint64(seed)
                  + np.arange(n, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15))
    k = _mix64_arr(states) & np.uint64(_U63)
    out = k.astype(np.int64)
    assert np.unique(out).size == n, "key generator produced duplicates"
    return out


def gen_absent(n: int, seed: int) -> np.ndarray:
    """n distinct keys disjoint from gen_keys(n, seed) (disjoint bit range)."""
    with np.errstate(over="ignore"):
        states = (np.uint64(seed ^ 0xCAFEBABE)
                  + np.arange(n, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15))
    k = (_mix64_arr(states) & np.uint64((1 << 62) - 1)) | np.uint64(1 << 62)
    out = k.astype(np.int64)
    assert np.unique(out).size == n
    return out


def delta_for_alpha(alpha: float) -> float:
    """Funnel delta realizing load alpha (clamped to the paper's 1/8)."""
    return min(0.125, 1.0 - alpha)


# =============================================================================
# Linear-probe baseline (scalar path -- same interpreter conditions as the
# funnel table's per-key loops).
# =============================================================================

class LinearProbeTable:
    """
    Open-addressing linear probing with splitmix64 hashing, in-band
    sentinels (-1 empty, -2 tombstone), classic tombstone deletion and
    tombstone-reusing insertion. 8 bytes per slot (no occupied flag).
    """

    def __init__(self, slots: int, seed: int = 42):
        self.m = int(slots)
        self.keys = np.full(self.m, _LINEAR_EMPTY, dtype=np.int64)
        self.values = [None] * self.m
        self.salt = int(_mix64(seed * 0x9E3779B97F4A7C15 + 1) & ((1 << 63) - 1))
        self.count = 0

    def _h(self, key: int) -> int:
        return _mix64((int(key) ^ self.salt) & ((1 << 64) - 1)) % self.m

    def insert(self, key: int, value) -> tuple:
        """Insert without reordering. Returns (ok, probes)."""
        pos = self._h(key)
        probes = 0
        first_tomb = -1
        keys = self.keys
        while probes < self.m:
            probes += 1
            k = keys[pos]
            if k == key:
                self.values[pos] = value
                return True, probes
            if k == _LINEAR_EMPTY:
                if first_tomb >= 0:
                    pos = first_tomb
                keys[pos] = key
                self.values[pos] = value
                self.count += 1
                return True, probes
            if k == _LINEAR_TOMB and first_tomb < 0:
                first_tomb = pos
            pos += 1
            if pos == self.m:
                pos = 0
        return False, probes

    def lookup(self, key: int) -> tuple:
        """Returns (value or None, probes)."""
        pos = self._h(key)
        probes = 0
        keys = self.keys
        while probes < self.m:
            probes += 1
            k = keys[pos]
            if k == key:
                return self.values[pos], probes
            if k == _LINEAR_EMPTY:
                return None, probes
            pos += 1
            if pos == self.m:
                pos = 0
        return None, probes

    def remove(self, key: int) -> bool:
        pos = self._h(key)
        probes = 0
        keys = self.keys
        while probes < self.m:
            probes += 1
            k = keys[pos]
            if k == key:
                keys[pos] = _LINEAR_TOMB
                self.values[pos] = None
                self.count -= 1
                return True
            if k == _LINEAR_EMPTY:
                return False
            pos += 1
            if pos == self.m:
                pos = 0
        return False


# =============================================================================
# NumPy-vectorized linear-probe drivers (used only where the scalar path
# would exceed the probe budget; exact probe counts, structure-only -- the
# Python values list is not written in this path).
# =============================================================================

def lin_build_vec(slots: int, keys: np.ndarray, salt: int):
    """Vectorized linear-probe build. Returns (keys_arr, probes[n])."""
    m = int(slots)
    arr = np.full(m, _LINEAR_EMPTY, dtype=np.int64)
    with np.errstate(over="ignore"):
        pos = (_mix64_arr(keys.astype(np.uint64) ^ np.uint64(salt))
               % np.uint64(m)).astype(np.int64)
    active_idx = np.arange(keys.size, dtype=np.int64)
    active_pos = pos
    probes = np.zeros(keys.size, dtype=np.int64)
    guard = 0
    while active_idx.size:
        guard += 1
        if guard > 4 * m + 64:
            raise RuntimeError("linear build vec did not converge")
        probes[active_idx] += 1
        slot_val = arr[active_pos]
        usable = slot_val < 0  # EMPTY or TOMB
        if usable.any():
            cand_pos = active_pos[usable]
            # One winner per distinct slot: lowest active index.
            _, first = np.unique(cand_pos, return_index=True)
            winners = usable.nonzero()[0][first]
            wpos = active_pos[winners]
            arr[wpos] = keys[active_idx[winners]]
        else:
            winners = np.empty(0, dtype=np.int64)
        # Everyone advances except the winners (occupied-slot keys AND
        # usable-slot keys that lost the dedup both move one slot on).
        advance_mask = np.ones(active_idx.size, dtype=bool)
        advance_mask[winners] = False
        active_idx = active_idx[advance_mask]
        active_pos = (active_pos[advance_mask] + 1) % m
    return arr, probes


def lin_lookup_vec(arr: np.ndarray, qkeys: np.ndarray, salt: int):
    """Vectorized linear-probe lookup. Returns (positions(-1 miss), probes)."""
    m = arr.size
    with np.errstate(over="ignore"):
        pos = (_mix64_arr(qkeys.astype(np.uint64) ^ np.uint64(salt))
               % np.uint64(m)).astype(np.int64)
    found = np.full(qkeys.size, -1, dtype=np.int64)
    probes = np.zeros(qkeys.size, dtype=np.int64)
    active = np.arange(qkeys.size, dtype=np.int64)
    apos = pos
    guard = 0
    while active.size:
        guard += 1
        if guard > 4 * m + 64:
            raise RuntimeError("linear lookup vec did not converge")
        probes[active] += 1
        vals = arr[apos]
        match = vals == qkeys[active]
        if match.any():
            found[active[match]] = apos[match]
        cont = ~match & (vals != _LINEAR_EMPTY)
        active = active[cont]
        apos = (apos[cont] + 1) % m
    return found, probes


# Scalar ops on a raw array (for churn on vec-built tables).
def _lin_scalar_insert_arr(arr: np.ndarray, key: int, salt: int) -> bool:
    m = arr.size
    pos = _mix64((int(key) ^ salt) & ((1 << 64) - 1)) % m
    probes = 0
    first_tomb = -1
    while probes < m:
        probes += 1
        k = arr[pos]
        if k == key:
            return True
        if k == _LINEAR_EMPTY:
            arr[first_tomb if first_tomb >= 0 else pos] = key
            return True
        if k == _LINEAR_TOMB and first_tomb < 0:
            first_tomb = pos
        pos += 1
        if pos == m:
            pos = 0
    return False


def _lin_scalar_remove_arr(arr: np.ndarray, key: int, salt: int) -> bool:
    m = arr.size
    pos = _mix64((int(key) ^ salt) & ((1 << 64) - 1)) % m
    probes = 0
    while probes < m:
        probes += 1
        k = arr[pos]
        if k == key:
            arr[pos] = _LINEAR_TOMB
            return True
        if k == _LINEAR_EMPTY:
            return False
        pos += 1
        if pos == m:
            pos = 0
    return False


def knuth_unsuccessful(alpha: float) -> float:
    """Knuth's linear-probe mean unsuccessful-search probes (insert cost)."""
    a = min(alpha, 0.9999)
    return 0.5 * (1.0 + 1.0 / (1.0 - a) ** 2)


# =============================================================================
# Backend drivers
# =============================================================================

def bench_funnel(n, alpha, keys, absent, rng):
    slots_target = math.ceil(n / alpha)
    delta = delta_for_alpha(alpha)
    capacity = max(n, int(round(slots_target * (1.0 - delta))))
    ht = ElasticHashTable(capacity=capacity, delta=delta, seed=42)

    t0 = time.perf_counter()
    ins_probes = np.empty(n, dtype=np.int64)
    drops = 0
    for i in range(n):
        ok, pr = ht.insert(int(keys[i]), i)
        ins_probes[i] = pr
        if not ok:
            drops += 1
    t_build = time.perf_counter() - t0

    order = rng.permutation(n)
    t0 = time.perf_counter()
    hit_probes = np.empty(n, dtype=np.int64)
    misses = 0
    for idx in order:
        v, pr = ht.lookup(int(keys[idx]))
        hit_probes[idx] = pr
        if v != idx:
            misses += 1
    t_hit = time.perf_counter() - t0

    t0 = time.perf_counter()
    abs_probes = np.empty(len(absent), dtype=np.int64)
    false_hits = 0
    for j, k in enumerate(absent):
        v, pr = ht.lookup(int(k))
        abs_probes[j] = pr
        if v is not None:
            false_hits += 1
    t_abs = time.perf_counter() - t0

    # Vectorized batch probe (funnel_probe): the per-key probe sequence is
    # deterministic, so NumPy can evaluate it in parallel -- the
    # vectorizability selling point, with a number.
    qkeys = keys[order]
    t0 = time.perf_counter()
    slots = funnel_probe(ht, qkeys)
    t_vec = time.perf_counter() - t0
    vec_found = int((ht.keys[slots] == qkeys).sum())
    assert vec_found == n, f"funnel_probe found only {vec_found}/{n} keys"

    return {
        "backend": "funnel", "impl": "scalar", "alpha": alpha,
        "load": n / ht.total_size, "slots": ht.total_size,
        "build_mps": n / t_build / 1e6, "build_probes_mean": float(ins_probes.mean()),
        "build_probes_max": int(ins_probes.max()),
        "hit_mps": n / t_hit / 1e6, "hit_probes_mean": float(hit_probes.mean()),
        "hit_probes_max": int(hit_probes.max()),
        "abs_mps": len(absent) / t_abs / 1e6, "abs_probes_mean": float(abs_probes.mean()),
        "abs_probes_max": int(abs_probes.max()),
        "vec_mps": n / t_vec / 1e6, "vec_found": vec_found,
        "drops": drops, "misses": misses, "false_hits": false_hits,
        "probe_bound": ht.probe_bound,
        "bytes_per_key": 9.0 * ht.total_size / n,  # i64 keys + bool flag; values excluded
        "_table": ht, "_keys": keys,
    }


def bench_linear_scalar(n, alpha, keys, absent, rng):
    m = math.ceil(n / alpha)
    lt = LinearProbeTable(slots=m, seed=42)

    t0 = time.perf_counter()
    ins_probes = np.empty(n, dtype=np.int64)
    drops = 0
    for i in range(n):
        ok, pr = lt.insert(int(keys[i]), i)
        ins_probes[i] = pr
        if not ok:
            drops += 1
    t_build = time.perf_counter() - t0

    order = rng.permutation(n)
    t0 = time.perf_counter()
    hit_probes = np.empty(n, dtype=np.int64)
    misses = 0
    for idx in order:
        v, pr = lt.lookup(int(keys[idx]))
        hit_probes[idx] = pr
        if v != idx:
            misses += 1
    t_hit = time.perf_counter() - t0

    t0 = time.perf_counter()
    abs_probes = np.empty(len(absent), dtype=np.int64)
    false_hits = 0
    for j, k in enumerate(absent):
        v, pr = lt.lookup(int(k))
        abs_probes[j] = pr
        if v is not None:
            false_hits += 1
    t_abs = time.perf_counter() - t0

    return {
        "backend": "linear", "impl": "scalar", "alpha": alpha,
        "load": n / m, "slots": m,
        "build_mps": n / t_build / 1e6, "build_probes_mean": float(ins_probes.mean()),
        "build_probes_max": int(ins_probes.max()),
        "hit_mps": n / t_hit / 1e6, "hit_probes_mean": float(hit_probes.mean()),
        "hit_probes_max": int(hit_probes.max()),
        "abs_mps": len(absent) / t_abs / 1e6, "abs_probes_mean": float(abs_probes.mean()),
        "abs_probes_max": int(abs_probes.max()),
        "vec_mps": None, "vec_found": None,
        "drops": drops, "misses": misses, "false_hits": false_hits,
        "probe_bound": None,
        "bytes_per_key": 8.0 * m / n,
        "_table": lt, "_keys": keys,
    }


def bench_linear_vec(n, alpha, keys, absent, rng):
    m = math.ceil(n / alpha)
    salt = LinearProbeTable(slots=8, seed=42).salt  # same salt derivation

    t0 = time.perf_counter()
    arr, ins_probes = lin_build_vec(m, keys, salt)
    t_build = time.perf_counter() - t0

    order = rng.permutation(n)
    qkeys = keys[order]
    t0 = time.perf_counter()
    positions, hit_probes_order = lin_lookup_vec(arr, qkeys, salt)
    t_hit = time.perf_counter() - t0
    hit_probes = np.empty(n, dtype=np.int64)
    hit_probes[order] = hit_probes_order
    misses = int((positions < 0).sum())

    t0 = time.perf_counter()
    apos, abs_probes = lin_lookup_vec(arr, absent, salt)
    t_abs = time.perf_counter() - t0
    false_hits = int((apos >= 0).sum())

    class _ArrTable:  # minimal handle for the churn phase
        pass

    handle = _ArrTable()
    handle.keys = arr
    handle.salt = salt
    handle.m = m
    handle.is_vec = True

    return {
        "backend": "linear", "impl": "vec", "alpha": alpha,
        "load": n / m, "slots": m,
        "build_mps": n / t_build / 1e6, "build_probes_mean": float(ins_probes.mean()),
        "build_probes_max": int(ins_probes.max()),
        "hit_mps": n / t_hit / 1e6, "hit_probes_mean": float(hit_probes.mean()),
        "hit_probes_max": int(hit_probes.max()),
        "abs_mps": len(absent) / t_abs / 1e6, "abs_probes_mean": float(abs_probes.mean()),
        "abs_probes_max": int(abs_probes.max()),
        "vec_mps": None, "vec_found": None,
        "drops": 0, "misses": misses, "false_hits": false_hits,
        "probe_bound": None,
        "bytes_per_key": 8.0 * m / n,
        "_table": handle, "_keys": keys,
    }


def bench_dict(n, alpha, keys, absent, rng):
    t0 = time.perf_counter()
    d = {}
    for i in range(n):
        d[int(keys[i])] = i
    t_build = time.perf_counter() - t0

    order = rng.permutation(n)
    t0 = time.perf_counter()
    misses = 0
    for idx in order:
        if d.get(int(keys[idx]), -1) != idx:
            misses += 1
    t_hit = time.perf_counter() - t0

    import sys as _sys
    t0 = time.perf_counter()
    false_hits = 0
    for k in absent:
        if int(k) in d:
            false_hits += 1
    t_abs = time.perf_counter() - t0

    return {
        "backend": "dict", "impl": "C", "alpha": alpha,
        "load": n / (n / alpha), "slots": None,
        "build_mps": n / t_build / 1e6, "build_probes_mean": None,
        "build_probes_max": None,
        "hit_mps": n / t_hit / 1e6, "hit_probes_mean": None,
        "hit_probes_max": None,
        "abs_mps": len(absent) / t_abs / 1e6, "abs_probes_mean": None,
        "abs_probes_max": None,
        "vec_mps": None, "vec_found": None,
        "drops": 0, "misses": misses, "false_hits": false_hits,
        "probe_bound": None,
        "bytes_per_key": _sys.getsizeof(d) / n,
        "_table": d, "_keys": keys,
    }


def churn_study(res, n, alpha, rng, frac):
    """Delete/reinsert churn on an already-built table from bench_*()."""
    d = max(1, int(round(frac * n)))
    del_idx = rng.choice(n, size=d, replace=False)
    fresh = gen_absent(d, seed=777)
    fresh = (fresh >> 1) | np.int64(1 << 61)  # disjoint from keys AND absent streams

    out = {"backend": res["backend"], "alpha": alpha, "n": n, "churn": d}

    if res["backend"] == "funnel":
        ht = res["_table"]
        keys = res["_keys"]
        removed = sum(1 for i in del_idx if ht.remove(int(keys[i])))
        survivors = np.delete(keys, del_idx)
        pr = [ht.lookup(int(k))[1] for k in survivors]
        out["removed"] = removed
        out["surv_mean"] = float(np.mean(pr))
        out["surv_max"] = int(np.max(pr))
        drops = sum(0 if ht.insert(int(k), -1)[0] else 1 for k in fresh)
        pr2 = [ht.lookup(int(k))[1] for k in np.concatenate([survivors, fresh])]
        found = sum(1 for k in np.concatenate([survivors, fresh]) if ht.lookup(int(k))[0] is not None)
        out["reinsert_drops"] = drops
        out["post_mean"] = float(np.mean(pr2))
        out["post_max"] = int(np.max(pr2))
        out["post_found"] = found
        out["post_expected"] = int(len(survivors) + d - drops)
    elif res["backend"] == "linear":
        lt = res["_table"]
        keys = res["_keys"]
        if getattr(lt, "is_vec", False):
            removed = sum(1 for i in del_idx if _lin_scalar_remove_arr(lt.keys, int(keys[i]), lt.salt))
            survivors = np.delete(keys, del_idx)
            _, p1 = lin_lookup_vec(lt.keys, survivors, lt.salt)
            drops = sum(0 if _lin_scalar_insert_arr(lt.keys, int(k), lt.salt) else 1 for k in fresh)
            _, p2 = lin_lookup_vec(lt.keys, np.concatenate([survivors, fresh]), lt.salt)
        else:
            removed = sum(1 for i in del_idx if lt.remove(int(keys[i])))
            survivors = np.delete(keys, del_idx)
            p1 = np.array([lt.lookup(int(k))[1] for k in survivors])
            drops = sum(0 if lt.insert(int(k), -1)[0] else 1 for k in fresh)
            p2 = np.array([lt.lookup(int(k))[1] for k in np.concatenate([survivors, fresh])])
        out["removed"] = removed
        out["surv_mean"] = float(np.mean(p1))
        out["surv_max"] = int(np.max(p1))
        out["reinsert_drops"] = drops
        out["post_mean"] = float(np.mean(p2))
        out["post_max"] = int(np.max(p2))
        out["post_found"] = None
        out["post_expected"] = None
    else:
        return None
    return out


# =============================================================================
# Zig backend runner
# =============================================================================

def run_zig(capacity, delta, seed, zig_dir, timeout=300):
    cmd = ["zig", "build", "run", "--release=fast", "--",
           "--capacity", str(capacity), "--delta", repr(delta), "--seed", str(seed)]
    try:
        proc = subprocess.run(cmd, cwd=zig_dir, capture_output=True, text=True,
                              timeout=timeout, errors="replace")
    except FileNotFoundError:
        return {"error": "zig not on PATH"}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    txt = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return {"error": f"exit {proc.returncode}: {txt[-400:]}"}

    def f(pattern, cast=float):
        mm = re.search(pattern, txt, re.DOTALL)
        return cast(mm.group(1)) if mm else None

    return {
        "capacity": capacity, "delta": delta,
        "geometry": f(r"geometry: ([^\n]*)", str),
        "probe_bound": f(r"probe_bound=(\d+)", int),
        "insert_mps": f(r"--- INSERT ---.*?throughput\s+= ([\d.]+) M keys/s"),
        "insert_probes": f(r"--- INSERT ---.*?mean probes\s+= ([\d.]+)"),
        "hit_mps": f(r"--- LOOKUP \(shuffled.*?throughput\s+= ([\d.]+) M keys/s"),
        "hit_probes": f(r"--- LOOKUP \(shuffled.*?mean probes\s+= ([\d.]+)"),
        "abs_mps": f(r"--- LOOKUP \(absent.*?throughput\s+= ([\d.]+) M keys/s"),
        "abs_probes": f(r"--- LOOKUP \(absent.*?mean probes\s+= ([\d.]+)"),
        "inserted": f(r"inserted\s+= (\d+) / (\d+)", int),
        "pass": "=== PASS" in txt,
    }


# =============================================================================
# Reporting
# =============================================================================

def _c(v, fmt="{:.2f}"):
    if v is None:
        return "-"
    return fmt.format(v)


def print_main_table(results, n):
    print(f"\n=== n = {n:,} : main comparison (per-backend rows) ===")
    hdr = (f"{'backend':<8}{'alpha':>6}{'load':>7}{'impl':>7}"
           f"{'build M/s':>10}{'look M/s':>9}{'vecLk M/s':>10}"
           f"{'insPrMn':>8}{'insPrMx':>8}"
           f"{'hitPrMn':>8}{'hitPrMx':>8}{'absPrMn':>8}{'absPrMx':>8}"
           f"{'drops':>6}{'B/key':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['backend']:<8}{r['alpha']:>6.2f}{r['load']:>7.3f}{r['impl']:>7}"
              f"{r['build_mps']:>10.2f}{r['hit_mps']:>9.2f}{_c(r['vec_mps']):>10}"
              f"{_c(r['build_probes_mean']):>8}{_c(r['build_probes_max'], '{:d}'):>8}"
              f"{_c(r['hit_probes_mean']):>8}{_c(r['hit_probes_max'], '{:d}'):>8}"
              f"{_c(r['abs_probes_mean']):>8}{_c(r['abs_probes_max'], '{:d}'):>8}"
              f"{r['drops']:>6}{r['bytes_per_key']:>7.1f}")


def print_worst_case_table(results, n):
    """THE headline: worst-case probe counts, funnel bound vs linear reality."""
    fun = {r["alpha"]: r for r in results if r["backend"] == "funnel"}
    lin = {r["alpha"]: r for r in results if r["backend"] == "linear"}
    print(f"\n=== n = {n:,} : WORST-CASE probes (the funnel guarantee, made visible) ===")
    hdr = (f"{'alpha':>6} | {'funnel bound':>12} {'hit max':>8} {'abs max':>8} | "
           f"{'linear hit max':>14} {'linear abs max':>14} | {'abs max lin/fun':>15}")
    print(hdr)
    print("-" * len(hdr))
    for a in sorted(fun):
        fr, lr = fun[a], lin[a]
        ratio = (lr["abs_probes_max"] / fr["abs_probes_max"]
                 if fr["abs_probes_max"] else float("nan"))
        print(f"{a:>6.2f} | {fr['probe_bound']:>12d} {fr['hit_probes_max']:>8d} "
              f"{fr['abs_probes_max']:>8d} | {lr['hit_probes_max']:>14d} "
              f"{lr['abs_probes_max']:>14d} | {ratio:>15.1f}x")


def print_churn_table(churn_rows, n):
    if not churn_rows:
        return
    print(f"\n=== n = {n:,} : delete/reinsert churn ({churn_rows[0]['churn']} keys) ===")
    hdr = (f"{'backend':<8}{'alpha':>6}{'removed':>9}{'survPrMn':>9}{'survPrMx':>9}"
           f"{'reinsDrops':>11}{'postPrMn':>9}{'postPrMx':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in churn_rows:
        print(f"{r['backend']:<8}{r['alpha']:>6.2f}{r['removed']:>9d}"
              f"{r['surv_mean']:>9.1f}{r['surv_max']:>9d}"
              f"{r['reinsert_drops']:>11d}{r['post_mean']:>9.1f}{r['post_max']:>9d}")


def print_zig_table(zig_rows):
    if not zig_rows:
        return
    print("\n=== Zig funnel port (compiled; zig build run --release=fast) ===")
    hdr = (f"{'n':>10}{'alpha':>7} | {'ins M/s':>8}{'insPrMn':>9}"
           f"{'hit M/s':>8}{'hitPrMn':>9}{'abs M/s':>8}{'absPrMn':>9}"
           f"{'bound':>7}{'ok':>4}")
    print(hdr)
    print("-" * len(hdr))
    for r in zig_rows:
        if "error" in r:
            print(f"  n={r.get('capacity')}: ERROR {r['error']}")
            continue
        alpha = 1.0 - r["delta"]
        print(f"{r['capacity']:>10,}{alpha:>7.2f} | {r['insert_mps']:>8.2f}"
              f"{r['insert_probes']:>9.2f}{r['hit_mps']:>8.2f}{r['hit_probes']:>9.2f}"
              f"{r['abs_mps']:>8.2f}{r['abs_probes']:>9.2f}{r['probe_bound']:>7d}"
              f"{'Y' if r['pass'] else 'N':>4}")


def cross_check(seed=4242, n=20_000, alphas=(0.9, 0.95, 0.99)):
    """Scalar vs vectorized linear-probe placement statistics (validity of
    the vec driver used where the scalar path would exceed the budget)."""
    print("\n=== cross-check: linear scalar vs vectorized drivers "
          f"(n={n:,}) ===")
    print(f"{'alpha':>6}{'impl':>8}{'insPrMn':>9}{'insPrMx':>9}"
          f"{'hitPrMn':>9}{'hitPrMx':>9}{'absPrMn':>9}{'absPrMx':>9}")
    out = []
    for a in alphas:
        keys = gen_keys(n, seed + int(a * 100))
        absent = gen_absent(500, seed + int(a * 100))
        rng = np.random.RandomState(seed)
        s = bench_linear_scalar(n, a, keys, absent, rng)
        rng = np.random.RandomState(seed)
        v = bench_linear_vec(n, a, keys, absent, rng)
        for tag, r in (("scalar", s), ("vec", v)):
            print(f"{a:>6.2f}{tag:>8}{r['build_probes_mean']:>9.1f}"
                  f"{r['build_probes_max']:>9d}{r['hit_probes_mean']:>9.1f}"
                  f"{r['hit_probes_max']:>9d}{r['abs_probes_mean']:>9.1f}"
                  f"{r['abs_probes_max']:>9d}")
            r.pop("_table", None)
            r.pop("_keys", None)
        out.append({"alpha": a, "scalar": s, "vec": v})
    return out


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--quick", action="store_true",
                    help="smaller grid (n=1e4,1e5; fewer absent samples)")
    ap.add_argument("--ns", type=int, nargs="+", default=None,
                    help="key counts (default 1e4 1e5 1e6)")
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="load factors (default 0.5 0.75 0.9 0.95 0.99)")
    ap.add_argument("--absent", type=int, default=None,
                    help="absent-key sample size (default 2000; quick 500)")
    ap.add_argument("--churn-frac", type=float, default=0.10,
                    help="fraction of keys deleted/reinserted in churn study")
    ap.add_argument("--churn-n", type=int, default=None,
                    help="n at which the churn study runs (default 1e5; quick 1e4)")
    ap.add_argument("--no-zig", action="store_true", help="skip the Zig backend")
    ap.add_argument("--zig-only", action="store_true", help="run only the Zig backend")
    ap.add_argument("--no-xcheck", action="store_true",
                    help="skip the scalar-vs-vec cross-check")
    ap.add_argument("--json", type=str, default=None, help="dump results to JSON")
    args = ap.parse_args()

    ns = args.ns or ([10_000, 100_000] if args.quick else [10_000, 100_000, 1_000_000])
    alphas = args.alphas or [0.5, 0.75, 0.9, 0.95, 0.99]
    n_absent = args.absent or (500 if args.quick else 2000)
    churn_n = args.churn_n or (10_000 if args.quick else 100_000)
    seed = 42

    all_results = {"meta": {"ns": ns, "alphas": alphas, "n_absent": n_absent,
                            "churn_frac": args.churn_frac, "seed": seed,
                            "quick": args.quick},
                   "cells": [], "churn": [], "zig": [], "xcheck": []}

    if not args.zig_only and not args.no_xcheck:
        all_results["xcheck"] = cross_check(seed=seed)

    zig_rows = []
    if not args.no_zig:
        zig_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "zig")
        zig_ns = [100_000] if args.quick else [n for n in ns if n >= 10_000]
        for n in zig_ns:
            for a in alphas:
                if a < 0.875:  # delta would clamp: Zig harness realizes load 1-delta
                    continue
                row = run_zig(capacity=n, delta=round(1.0 - a, 6), seed=seed,
                              zig_dir=zig_dir)
                row.setdefault("capacity", n)
                row.setdefault("delta", round(1.0 - a, 6))
                zig_rows.append(row)
        print_zig_table(zig_rows)
        all_results["zig"] = zig_rows
    if args.zig_only:
        return 0

    for n in ns:
        rows = []
        churn_rows = []
        for a in alphas:
            keys = gen_keys(n, seed + int(a * 1000))
            absent = gen_absent(n_absent, seed + int(a * 1000))
            rng = np.random.RandomState(seed)

            rows.append(bench_funnel(n, a, keys, absent, rng))
            rows.append(bench_dict(n, a, keys, absent, rng))

            # Linear baseline: scalar unless the probe budget explodes.
            est_probes = n * knuth_unsuccessful(a)
            if est_probes <= 3e7:
                lin = bench_linear_scalar(n, a, keys, absent, rng)
            else:
                lin = bench_linear_vec(n, a, keys, absent, rng)
            rows.append(lin)

            if n == churn_n:
                for r in (rows[-3], rows[-1]):  # funnel + linear churn
                    cr = churn_study(r, n, a, rng, args.churn_frac)
                    if cr:
                        churn_rows.append(cr)

        print_main_table(rows, n)
        print_worst_case_table(rows, n)
        print_churn_table(churn_rows, n)
        for r in rows:
            r.pop("_table", None)
            r.pop("_keys", None)
        all_results["cells"].extend([{"n": n, **r} for r in rows])
        all_results["churn"].extend(churn_rows)

    print("\nNotes:")
    print("  funnel: ElasticHashTable (Farach-Colton, Krapivin, & Kuszmaul, 2025, S.3);")
    print("    probe_bound = alpha*beta + b_attempts + 2*c_bucket_slots is a")
    print("    DETERMINISTIC cap on every search; absent lookups always pay it.")
    print("  funnel vecLk: funnel_probe vectorized batch lookup (NumPy) -- the")
    print("    deterministic per-key sequence evaluated in parallel; verified")
    print("    to find every key (asserted per cell).")
    print("  linear impl=vec rows: NumPy-vectorized driver (exact probe counts,")
    print("    structure-only, parallel collision tie-break instead of FIFO);")
    print("    used where the scalar path's probe total exceeds 3e7 (see xcheck).")
    print("  dict rows: CPython dict, C implementation -- wins raw throughput;")
    print("    B/key is sys.getsizeof(dict)/n (excludes key/value objects).")
    print("  funnel/linear B/key: slot-array bytes only (values excluded for both).")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=1, default=float)
        print(f"\nresults written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
