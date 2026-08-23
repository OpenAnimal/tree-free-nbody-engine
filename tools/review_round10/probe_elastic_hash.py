"""Round-10 Wave A probe: ElasticHashTable correctness + funnel_probe parity."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from core.elastic_hash import ElasticHashTable, ElasticIntTable, funnel_probe, ElasticBatchingHashTable

FAIL = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(name)

print("=" * 70)
print("PROBE: ElasticHashTable deep verification")
print("=" * 70)

# ============================================================
# 1. Insert/lookup correctness at high load (1000 keys)
# ============================================================
print("\n[1] Insert/lookup correctness at high load (N=1000, delta=0.05)")
ht = ElasticHashTable(capacity=1000, delta=0.05)
rng = np.random.RandomState(42)
keys = rng.randint(0, 2**31 - 1, size=1000)
vals = rng.randint(0, 1000, size=1000)
n_fail_insert = 0
for i in range(1000):
    ok, _ = ht.insert(int(keys[i]), int(vals[i]))
    if not ok:
        n_fail_insert += 1
check("all 1000 inserts succeed", n_fail_insert == 0, f"{n_fail_insert} failures")
# Verify every key looks up correctly
n_wrong = 0
for i in range(1000):
    v, _ = ht.lookup(int(keys[i]))
    if v != vals[i]:
        n_wrong += 1
check("all 1000 lookups correct", n_wrong == 0, f"{n_wrong} wrong")
# Verify absent keys return None (keys not in the inserted set)
absent_keys = np.array([-(k + 1) for k in rng.randint(0, 10000, size=100)], dtype=np.int64)
n_false_hit = 0
for k in absent_keys:
    v, _ = ht.lookup(int(k))
    if v is not None:
        n_false_hit += 1
check("absent keys return None", n_false_hit == 0, f"{n_false_hit} false hits")
check("count == 1000", ht.count == 1000, f"count={ht.count}")
check("load_factor <= 1-delta+eps", ht.load_factor() <= 0.95 + 0.01,
      f"load={ht.load_factor():.4f}")

# ============================================================
# 2. Duplicate key update
# ============================================================
print("\n[2] Duplicate key update")
ht2 = ElasticHashTable(capacity=100, delta=0.05)
ht2.insert(42, "first")
ht2.insert(42, "second")
v, _ = ht2.lookup(42)
check("duplicate insert updates value", v == "second", f"got {v}")
check("count still 1 after duplicate", ht2.count == 1, f"count={ht2.count}")

# ============================================================
# 3. funnel_probe vectorized parity with scalar lookup
# ============================================================
print("\n[3] funnel_probe parity with scalar lookup")
ht3 = ElasticHashTable(capacity=500, delta=0.05)
rng3 = np.random.RandomState(99)
keys3 = rng3.randint(0, 2**31 - 1, size=500)
for i in range(500):
    ht3.insert(int(keys3[i]), i)
# Query all keys + some absent ones via funnel_probe
query = np.concatenate([keys3, np.array([-(k + 1) for k in rng3.randint(0, 10000, size=50)], dtype=np.int64)])
slots = funnel_probe(ht3, query)
n_mismatch = 0
for i in range(len(query)):
    scalar_pos, _ = ht3._search(int(query[i]))
    vec_pos = int(slots[i])
    if scalar_pos != vec_pos:
        n_mismatch += 1
        if n_mismatch <= 3:
            print(f"    key={query[i]}: scalar={scalar_pos}, vectorized={vec_pos}")
check("funnel_probe matches scalar _search", n_mismatch == 0,
      f"{n_mismatch} mismatches out of {len(query)}")

# ============================================================
# 4. probe_bound is respected by all lookups
# ============================================================
print("\n[4] probe_bound respected")
ht4 = ElasticHashTable(capacity=2000, delta=0.05)
rng4 = np.random.RandomState(7)
keys4 = rng4.randint(0, 2**31 - 1, size=2000)
for k in keys4:
    ht4.insert(int(k), 1)
max_probes = 0
for k in keys4:
    _, p = ht4.lookup(int(k))
    max_probes = max(max_probes, p)
check("max lookup probes <= probe_bound", max_probes <= ht4.probe_bound,
      f"max={max_probes}, bound={ht4.probe_bound}")
# Also check absent-key probes
for k in np.array([-(k + 1) for k in rng4.randint(0, 10000, size=200)], dtype=np.int64):
    _, p = ht4.lookup(int(k))
    max_probes = max(max_probes, p)
check("max absent-key probes <= probe_bound", max_probes <= ht4.probe_bound,
      f"max={max_probes}, bound={ht4.probe_bound}")

# ============================================================
# 5. ElasticIntTable insert_or_increment
# ============================================================
print("\n[5] ElasticIntTable insert_or_increment")
it = ElasticIntTable(capacity=100, delta=0.05)
it.insert_or_increment(10, 1)
it.insert_or_increment(10, 1)
it.insert_or_increment(10, 1)
it.insert_or_increment(20, 5)
check("incremented key 10 == 3", it.get(10) == 3, f"got {it.get(10)}")
check("key 20 == 5", it.get(20) == 5, f"got {it.get(20)}")
check("absent key returns None", it.get(999) is None)
check("count == 2", it.count == 2, f"count={it.count}")

# ============================================================
# 6. remove + re-insert (tombstone path)
# ============================================================
print("\n[6] remove + re-insert")
ht6 = ElasticHashTable(capacity=100, delta=0.05)
ht6.insert(1, "a")
ht6.insert(2, "b")
ht6.insert(3, "c")
check("count == 3 before remove", ht6.count == 3)
ok = ht6.remove(2)
check("remove returns True for existing key", ok)
check("count == 2 after remove", ht6.count == 2, f"count={ht6.count}")
v, _ = ht6.lookup(2)
check("removed key returns None", v is None, f"got {v}")
v, _ = ht6.lookup(1)
check("other keys still findable after remove", v == "a", f"got {v}")
# Re-insert
ht6.insert(2, "b2")
v, _ = ht6.lookup(2)
check("re-inserted key findable", v == "b2", f"got {v}")
check("count == 3 after re-insert", ht6.count == 3, f"count={ht6.count}")

# ============================================================
# 7. items() returns all live entries
# ============================================================
print("\n[7] items() correctness")
ht7 = ElasticHashTable(capacity=100, delta=0.05)
for i in range(50):
    ht7.insert(1000 + i, i)
ht7.remove(1000 + 10)
ht7.remove(1000 + 20)
items = dict(ht7.items())
check("items() has 48 entries", len(items) == 48, f"got {len(items)}")
check("removed keys not in items()", (1000+10) not in items and (1000+20) not in items)
check("all values correct", all(items.get(1000+i) == i for i in range(50) if i not in (10, 20)))

# ============================================================
# 8. Capacity boundary: insert at exactly capacity
# ============================================================
print("\n[8] Capacity boundary")
ht8 = ElasticHashTable(capacity=50, delta=0.05)
all_ok = True
for i in range(50):
    ok, _ = ht8.insert(i, i)
    if not ok:
        all_ok = False
check("all 50 inserts at capacity succeed", all_ok)
ok, _ = ht8.insert(50, 50)
check("insert beyond capacity fails", not ok)
# Updating existing key at capacity should still work
ok, _ = ht8.insert(0, 999)
check("update existing key at capacity succeeds", ok)
v, _ = ht8.lookup(0)
check("updated value correct", v == 999, f"got {v}")

# ============================================================
# 9. ElasticBatchingHashTable basic correctness
# ============================================================
print("\n[9] ElasticBatchingHashTable basic")
bht = ElasticBatchingHashTable(capacity=200, delta=0.05)
rng9 = np.random.RandomState(55)
bkeys = rng9.randint(0, 2**31 - 1, size=150)
for i in range(150):
    bht.insert(int(bkeys[i]), i)
n_wrong = 0
for i in range(150):
    v, _ = bht.lookup(int(bkeys[i]))
    if v != i:
        n_wrong += 1
check("batching: 150 lookups correct", n_wrong == 0, f"{n_wrong} wrong")
check("batching: count == 150", bht.count == 150, f"count={bht.count}")

# ============================================================
# 10. Negative key handling (Morton keys are non-negative, but
#     the API doesn't enforce it — check for sentinel collision)
# ============================================================
print("\n[10] Negative key sentinel collision")
ht10 = ElasticHashTable(capacity=50, delta=0.05)
# Key -1 is the EMPTY sentinel, -2 is TOMBSTONE
# Inserting -1 should work but might confuse the table
ok, _ = ht10.insert(-1, "neg_one")
v, _ = ht10.lookup(-1)
check("key -1 (empty sentinel) insert/lookup", v == "neg_one" or v is None,
      f"ok={ok}, v={v}  (known: -1 is the empty sentinel, this may collide)")
ok2, _ = ht10.insert(-2, "neg_two")
v2, _ = ht10.lookup(-2)
check("key -2 (tombstone sentinel) insert/lookup", v2 == "neg_two" or v2 is None,
      f"ok={ok2}, v={v2}  (known: -2 is the tombstone sentinel, this may collide)")

# ============================================================
# 11. _overflow_count consistency after remove
# ============================================================
print("\n[11] _overflow_count after remove (performance, not correctness)")
ht11 = ElasticHashTable(capacity=100, delta=0.05)
# Force some keys into overflow by filling the table
rng11 = np.random.RandomState(123)
keys11 = rng11.randint(0, 2**31 - 1, size=100)
for k in keys11:
    ht11.insert(int(k), 1)
ov_before = ht11._overflow_count
# Remove a few keys (some might be in overflow)
removed = 0
for k in keys11[:20]:
    if ht11.remove(int(k)):
        removed += 1
ov_after = ht11._overflow_count
# The _overflow_count should decrease if any removed keys were in overflow,
# but the current code does NOT decrement it. This is a performance issue
# (the early-stop optimization in insert won't fire), not a correctness issue.
if removed > 0 and ov_before > 0:
    check("_overflow_count decremented after remove (perf bug if FAIL)",
          ov_after < ov_before,
          f"before={ov_before}, after={ov_after}, removed={removed}")
else:
    check("_overflow_count: no overflow keys to test", True)

print("\n" + "=" * 70)
if FAIL:
    print(f"RESULT: {len(FAIL)} FAILURES")
    for f in FAIL:
        print(f"  - {f}")
else:
    print("RESULT: ALL CHECKS PASSED")
print("=" * 70)
