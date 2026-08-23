"""
Open addressing WITHOUT reordering — Farach-Colton, Krapivin, & Kuszmaul (2025),
"Optimal Bounds for Open Addressing Without Reordering",
arXiv:2501.02305 / FOCS 2024.

Two schemes from the paper live in this module:

1. `ElasticHashTable` (DEFAULT — funnel hashing, paper Section 3)
   ---------------------------------------------------------------
   * The backing array of n slots is split into a funnel region A' and a
     small overflow region A_{alpha+1} of size within
     [ceil(delta*n/2), floor(3*delta*n/4)] (clamped to >= 16 slots for very
     small tables so the overflow mechanics exist).
   * A' is partitioned into alpha slabs A_1..A_alpha with
         alpha = ceil(4*log2(1/delta) + 10)   (paper: ceil(4 log 1/delta + 10))
     and every slab size is a multiple of
         beta  = ceil(2*log2(1/delta))        (paper: ceil(2 log 1/delta))
     with slab sizes shrinking geometrically by factor ~3/4
     (a_{i+1} = 3*a_i/4 +- 1 in the paper's notation, a_i = |A_i|/beta).
     The paper leaves the logarithm base unspecified (it only affects
     constant factors); we use base 2, standard for probe bounds.
   * Each slab is cut into sub-arrays of EXACTLY beta consecutive slots.
   * INSERT of key k attempts A_1, A_2, ..., A_alpha in order (A_1 first and
     largest — the funnel mouth). In slab A_i, k hashes to ONE uniformly
     random beta-slot sub-array which is scanned exhaustively; k occupies
     the first empty slot seen. Greedy, never displaces resident keys.
   * If every slab fails, k goes to the overflow region A_{alpha+1}, split
     into halves B and C: B is uniform probing with a cutoff of
     ceil(log2 log2 n) attempts (load <= 1/2, expected O(1) probes); C is a
     two-choice table whose buckets hold 2*ceil(log2 log2 n) slots, probed
     by alternating between the two hashed buckets.
   * SEARCH follows exactly the insertion order (A_1..A_alpha, then B, then
     C), so the probe sequence is deterministic and bounded by
         alpha*beta + b_attempts + 2*c_bucket_slots
     slot inspections in the worst case — no linear-probing fallback exists.
   * Guarantees (paper Theorem 2) for delta in (0, 1/8] and loads up to
     1 - delta: amortized expected insert O(log 1/delta), worst-case
     expected search O(log^2 1/delta), inserts succeed with probability
     1 - n^{-omega(1)} (no key drops).

2. `ElasticBatchingHashTable` (paper Section 2, simplified greedy variant)
   ---------------------------------------------------------------
   The paper's elastic hashing: sub-arrays with sizes halving, batch-style
   insertion cascading down the first sub-array still below (1 - delta/2)
   fill, a bounded probe budget f(eps) = c*min(log^2(1/eps), log(1/delta))
   in the primary sub-array and unlimited probes in the secondary one. The
   class kept here follows that schedule greedily and adds an exhaustive
   safety-net pass; it deviates from the paper in that the paper's insertion
   is non-greedy (a 2-D probe sequence h_{i,j} mapped through an injection
   phi with cost O(i*j^2), decoupling insertion probes from search probes).
   It also supports tombstone deletion, which the paper does not study. It
   is a secondary experimental class; the funnel table above is the default
   and the one used by the FMM engines.

Backward compatibility: `ElasticHashTable` keeps the historical constructor
and insert/lookup API of this repository (early releases labelled a
geometric-levels + global-linear-probing-fallback table with the same name;
that implementation did NOT realize the paper's bounds and could drop keys
at high load — it has been fully replaced by the funnel table).
"""

import math
from typing import Tuple, Optional, List, Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False


# =============================================================================
# 64-bit mixing (splitmix64 finalizer). Shared by the scalar (per-key) code
# path and the vectorized NumPy probe below, so both produce identical
# sub-array / slot sequences.
# =============================================================================

_U64_MASK = (1 << 64) - 1


_TOMBSTONE = -2  # sentinel key for deleted-but-still-occupied funnel slots


def _mix64(z: int) -> int:
    """Scalar splitmix64 finalizer on a Python int (wraps mod 2^64)."""
    z &= _U64_MASK
    z ^= z >> 30
    z = (z * 0xBF58476D1CE4E5B9) & _U64_MASK
    z ^= z >> 27
    z = (z * 0x94D049BB133111EB) & _U64_MASK
    z ^= z >> 31
    return z


def _mix64_arr(z: np.ndarray) -> np.ndarray:
    """Vectorized splitmix64 finalizer on uint64 NumPy arrays (wraps mod 2^64)."""
    z = z ^ (z >> np.uint64(30))
    z = z * np.uint64(0xBF58476D1CE4E5B9)
    z = z ^ (z >> np.uint64(27))
    z = z * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return z


class ElasticHashTable:
    """
    Funnel hash table (Farach-Colton, Krapivin, & Kuszmaul, 2025, Section 3),
    exposed under its historical class name for API compatibility.

    Geometry (see module docstring for the paper's parameterization):
        alpha = ceil(4*log2(1/delta) + 10)   slabs, sizes shrinking by ~3/4,
        beta  = ceil(2*log2(1/delta))        slots per sub-array,
        A_{alpha+1} in [ceil(delta*n/2), floor(3*delta*n/4)] overflow slots
        (>= 16 for tiny tables), split into halves B (uniform probing,
        ceil(log2 log2 n) attempt cutoff) and C (two-choice buckets of
        2*ceil(log2 log2 n) slots).

    The table supports `capacity` insertions (slots = capacity/(1-delta),
    i.e. final load 1-delta) with probability 1 - n^{-omega(1)}; insertion
    never displaces an existing key and any search inspects at most
    `probe_bound` slots deterministically.

    Public API (unchanged from the previous release):
        __init__(capacity, delta=0.05, num_levels=None, seed=42)
        insert(key, value) -> (success: bool, probes: int)
        lookup(key)        -> (value or None, probes: int)
        count, capacity, delta, total_size, level_sizes, level_offsets,
        keys, values, occupied
    `num_levels` is accepted for backward compatibility but deliberately
    ignored: the funnel scheme derives its slab count from delta (passing a
    hand-picked level count was part of the old, non-faithful design).
    New helpers: __len__, __contains__, get(), load_factor(), items(),
    probe_bound. The table is append-only (the paper studies insertion and
    search without deletions; tombstone deletion would invalidate the
    monotone-fill argument insert() relies on — use
    ElasticBatchingHashTable if you need remove()).

    Keys must be NON-NEGATIVE integers that fit in a signed 64-bit int
    (Morton cell keys, LSH bucket ids, ...): -1 and -2 are reserved as the
    empty/tombstone sentinels. Values are arbitrary Python objects.

    Primary in-repo consumer: `core.spatial_index.CellIndex` uses this table
    as its occupied-cell index for membership / neighborhood probes (the
    generic elastic hash backbone behind the spatial cell index).
    """

    def __init__(self, capacity: int, delta: float = 0.05,
                 num_levels: Optional[int] = None, seed: int = 42):
        if int(capacity) < 1:
            raise ValueError("capacity must be a positive integer")
        if not (0.0 < float(delta) <= 0.125):
            # The paper's analysis assumes delta <= 1/8.
            delta = min(max(float(delta), 1e-9), 0.125)
        self.capacity = int(capacity)
        self.delta = float(delta)

        log_inv = math.log2(1.0 / self.delta)
        self.alpha = int(math.ceil(4.0 * log_inv + 10.0))
        self.beta = max(2, int(math.ceil(2.0 * log_inv)))
        self.num_levels = self.alpha  # compatibility alias

        # Total slot count: capacity keys at final load 1 - delta.
        n_target = int(math.ceil(self.capacity / (1.0 - self.delta)))

        # Overflow region A_{alpha+1}: paper bounds
        # [ceil(delta*n/2), floor(3*delta*n/4)] against the FINAL array size
        # n. Rounding A' up to a multiple of beta perturbs n after the fact,
        # so the overflow size is fixed-pointed until the lower bound holds
        # (each pass grows it by < 1 slot on average, so this converges
        # immediately); clamped to >= 16 slots so B and C have room to
        # function on tiny tables.
        overflow_size = max(16, int(math.ceil(self.delta * n_target / 2.0)))
        for _ in range(8):
            a_probe = max(self.alpha,
                          int(math.ceil((n_target - overflow_size) / self.beta)))
            n_final = a_probe * self.beta + overflow_size
            o_low = int(math.ceil(self.delta * n_final / 2.0))
            if o_low <= overflow_size:
                break
            overflow_size = max(16, o_low)
        self.overflow_size = overflow_size

        # Funnel region A' = a_total sub-arrays of beta slots
        # (>= alpha sub-arrays total).
        a_total = max(self.alpha,
                      int(math.ceil((n_target - self.overflow_size) / self.beta)))

        # Sub-array counts per slab: geometric 3/4 shrink, non-increasing,
        # summing to a_total.
        r = 0.75
        weights = np.array([r ** i for i in range(self.alpha)], dtype=np.float64)
        weights /= weights.sum()
        a = np.maximum(np.floor(a_total * weights).astype(np.int64), 1)
        remainder = a_total - int(a.sum())
        if remainder > 0:
            a[:min(remainder, self.alpha)] += 1
        self.subarray_counts: List[int] = [int(v) for v in a]

        self.slab_sizes: List[int] = [c * self.beta for c in self.subarray_counts]
        self.slab_offsets: List[int] = [0]
        for sz in self.slab_sizes[:-1]:
            self.slab_offsets.append(self.slab_offsets[-1] + sz)
        funnel_end = self.slab_offsets[-1] + self.slab_sizes[-1]

        # Overflow halves: B (uniform probing) then C (two-choice buckets).
        self.b_size = self.overflow_size // 2
        self.c_size = self.overflow_size - self.b_size
        self.b_offset = funnel_end
        self.c_offset = funnel_end + self.b_size
        self.total_size = self.c_offset + self.c_size

        # Probe caps from log log n (paper: B cutoff log log n;
        # C buckets of 2 log log n slots).
        ll = math.log2(math.log2(max(self.total_size, 4.0)))
        self.b_attempts = max(4, int(math.ceil(ll)))
        self.c_bucket_slots = 2 * max(2, int(math.ceil(ll)))
        self.c_num_buckets = max(1, self.c_size // self.c_bucket_slots)

        # Backing storage (flat; key sentinel -1 = empty).
        self.keys = np.full(self.total_size, -1, dtype=np.int64)
        self.values: List[Any] = [None] * self.total_size
        self.occupied = np.zeros(self.total_size, dtype=bool)
        self.count = 0
        self._overflow_count = 0

        # Per-context hash salts (independent pseudo-random functions).
        rng = np.random.RandomState(seed)
        self.salt_slab = [int(s) for s in
                          rng.randint(1, 2 ** 63, size=self.alpha, dtype=np.int64)]
        self.salt_b = [int(s) for s in
                       rng.randint(1, 2 ** 63, size=self.b_attempts, dtype=np.int64)]
        self.salt_c1 = int(rng.randint(1, 2 ** 63, dtype=np.int64))
        self.salt_c2 = int(rng.randint(1, 2 ** 63, dtype=np.int64))
        # Compatibility aliases for the historical attribute names.
        self.seeds_a = np.array(self.salt_slab, dtype=np.int64)
        self.seeds_b = np.array(self.salt_b + [self.salt_c1, self.salt_c2],
                                dtype=np.int64)

        # Pre-baked (offset, sub-array count, salt) tuples for the hot loops.
        self._slabs = list(zip(self.slab_offsets, self.subarray_counts,
                               self.salt_slab))

        # Round-5 task 5.3 probe-count instrumentation: expose the probe
        # count of the most recent insert/lookup as `last_probes` (and the
        # `mean_probes_last_op` property) so external tooling can compare
        # against the Zig port WITHOUT altering the probe behavior itself.
        # The probe counts are already computed in the hot loops; this only
        # records them.
        self.last_probes: int = 0

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @property
    def probe_bound(self) -> int:
        """Deterministic worst-case number of slot inspections for any search."""
        return self.alpha * self.beta + self.b_attempts + 2 * self.c_bucket_slots

    @property
    def mean_probes_last_op(self) -> float:
        """Probe count of the most recent insert/lookup (single-op mean).

        Round-5 task 5.3 instrumentation: the hot loops already count
        probes; this exposes the last value for cross-language comparison
        with the Zig port. Does NOT alter probe behavior.
        """
        return float(self.last_probes)

    def _fill(self, pos: int, key: int, value: Any) -> None:
        self.keys[pos] = key
        self.values[pos] = value
        self.occupied[pos] = True
        self.count += 1
        if pos >= self.b_offset:
            self._overflow_count += 1

    def _search_overflow(self, key: int) -> Tuple[int, int]:
        """Search B (uniform probing) then C (two-choice) for `key`."""
        probes = 0
        occupied = self.occupied
        keys = self.keys
        for t in range(self.b_attempts):
            pos = self.b_offset + _mix64(key ^ self.salt_b[t]) % self.b_size
            probes += 1
            if occupied[pos] and keys[pos] == key:
                return pos, probes
        b1 = _mix64(key ^ self.salt_c1) % self.c_num_buckets
        b2 = _mix64(key ^ self.salt_c2) % self.c_num_buckets
        for t in range(self.c_bucket_slots):
            for bkt in (b1, b2):
                pos = self.c_offset + bkt * self.c_bucket_slots + t
                probes += 1
                if occupied[pos] and keys[pos] == key:
                    return pos, probes
        return -1, probes

    def _place_overflow(self, key: int, value: Any) -> Tuple[int, int]:
        """Place a new key into B, failing that into C (first empty slot)."""
        probes = 0
        occupied = self.occupied
        for t in range(self.b_attempts):
            pos = self.b_offset + _mix64(key ^ self.salt_b[t]) % self.b_size
            probes += 1
            if not occupied[pos]:
                self._fill(pos, key, value)
                return pos, probes
        b1 = _mix64(key ^ self.salt_c1) % self.c_num_buckets
        b2 = _mix64(key ^ self.salt_c2) % self.c_num_buckets
        for t in range(self.c_bucket_slots):
            for bkt in (b1, b2):
                pos = self.c_offset + bkt * self.c_bucket_slots + t
                probes += 1
                if not self.occupied[pos]:
                    self._fill(pos, key, value)
                    return pos, probes
        return -1, probes

    def _search(self, key: int) -> Tuple[int, int]:
        """
        Full funnel search: slabs A_1..A_alpha in order (one beta-slot
        sub-array scanned per slab), then overflow B, then C.
        Returns (slot index or -1, probes).
        """
        probes = 0
        occupied = self.occupied
        keys = self.keys
        for offset, a_count, salt in self._slabs:
            base = offset + (_mix64(key ^ salt) % a_count) * self.beta
            for s in range(self.beta):
                pos = base + s
                probes += 1
                if occupied[pos] and keys[pos] == key:
                    return pos, probes
        pos, p = self._search_overflow(key)
        return pos, probes + p

    # ------------------------------------------------------------------ #
    # Public API (backward compatible)
    # ------------------------------------------------------------------ #

    def insert(self, key: int, value: Any) -> Tuple[bool, int]:
        """
        Inserts (key, value) WITHOUT displacing any resident key (no
        reordering).

        The key descends the funnel A_1 -> A_2 -> ... ; in each slab its
        single hashed beta-slot sub-array is scanned exhaustively: an
        existing copy of the key is updated in place, otherwise the key
        takes the first empty slot of the first sub-array that has one.
        Keys that exhaust every slab go to the two-part overflow region
        (B then C).

        Correctness note (why stopping at the first unfilled sub-array is
        safe): the table is append-only, so sub-arrays only ever go
        empty -> full. If an existing key lives in slab j (or in overflow),
        every sub-array it skipped on the way down — including this one —
        was full at placement time and is still full now. Hence, whenever
        the overflow region is empty, seeing a free slot in the scanned
        sub-array proves no copy of the key exists deeper in the table.
        When the overflow region is non-empty (probability <= n^{-omega(1)}
        under the paper's load assumption) the full probe sequence is
        searched first instead.

        Returns (success, probe_count). A miss can only occur when the table
        already holds `capacity` keys (load 1 - delta reached) AND the key is
        not already present, or — with probability n^{-omega(1)} at any load
        below that — if the overflow region is also exhausted.
        """
        if self.count >= self.capacity:
            # Table is full: check if the key already exists so it can be
            # updated in place. Only genuinely new keys fail at capacity.
            pos, probes = self._search(int(key))
            if pos >= 0:
                self.values[pos] = value
                self.last_probes = probes
                return True, probes
            self.last_probes = 0
            return False, 0

        k = int(key)
        probes = 0
        occupied = self.occupied
        keys = self.keys
        overflow_was_empty = (self._overflow_count == 0)
        first_free = -1

        for offset, a_count, salt in self._slabs:
            base = offset + (_mix64(k ^ salt) % a_count) * self.beta
            sub_first_free = -1
            for s in range(self.beta):
                pos = base + s
                probes += 1
                if occupied[pos]:
                    if keys[pos] == k:
                        self.values[pos] = value
                        self.last_probes = probes
                        return True, probes
                elif sub_first_free < 0:
                    sub_first_free = pos
            if sub_first_free >= 0:
                if overflow_was_empty:
                    self._fill(sub_first_free, k, value)
                    self.last_probes = probes
                    return True, probes
                if first_free < 0:
                    first_free = sub_first_free

        # Not found in any slab: check the overflow region for the key.
        pos, p = self._search_overflow(k)
        probes += p
        if pos >= 0:
            self.values[pos] = value
            self.last_probes = probes
            return True, probes

        # Genuinely new key: remembered slab slot, else overflow B/C.
        if first_free >= 0:
            self._fill(first_free, k, value)
            self.last_probes = probes
            return True, probes
        pos, p = self._place_overflow(k, value)
        probes += p
        if pos >= 0:
            self.last_probes = probes
            return True, probes
        self.last_probes = probes
        return False, probes

    def lookup(self, key: int) -> Tuple[Optional[Any], int]:
        """
        Queries key. Returns (value or None, probe_count).

        The probe sequence is exactly the insertion order (slabs, then B,
        then C) and is bounded deterministically by `probe_bound`
        = alpha*beta + O(log log n) slot inspections — there is no linear
        probing fallback.
        """
        pos, probes = self._search(int(key))
        self.last_probes = probes
        return (self.values[pos] if pos >= 0 else None), probes

    def get(self, key: int, default: Optional[Any] = None) -> Any:
        """Dict-style access: value for key, or `default` if absent."""
        pos, _ = self._search(int(key))
        return self.values[pos] if pos >= 0 else default

    def remove(self, key: int) -> bool:
        """
        Delete key via tombstone (no displacement, no compaction).

        The slot stays marked occupied with a sentinel key so that the
        funnel search order — and the correctness argument that early
        stops are safe because sub-arrays only fill monotonically — is
        unaffected by deletions.  The slot is not reclaimed, so heavy
        remove/insert churn reduces effective capacity; rebuild the table
        for workloads with many deletions.
        """
        pos, _ = self._search(int(key))
        if pos < 0:
            return False
        self.keys[pos] = _TOMBSTONE
        self.values[pos] = None
        if pos >= self.b_offset:
            self._overflow_count -= 1
        self.count -= 1
        return True

    def __contains__(self, key: int) -> bool:
        return self._search(int(key))[0] >= 0

    def __len__(self) -> int:
        return self.count

    def load_factor(self) -> float:
        """Fraction of backing slots that are occupied."""
        return self.count / self.total_size

    def items(self):
        """Iterate (key, value) over all stored entries."""
        for pos in range(self.total_size):
            if self.occupied[pos] and self.keys[pos] != _TOMBSTONE:
                yield int(self.keys[pos]), self.values[pos]

    # Compatibility aliases for the historical level-based attribute names.
    @property
    def level_sizes(self) -> List[int]:
        return self.slab_sizes

    @property
    def level_offsets(self) -> List[int]:
        return self.slab_offsets


# =============================================================================
# ElasticIntTable — same funnel geometry, int64 values + insert_or_increment
# (Round-7 task T-A3: used by `bioinformatics/kmer_elastic_hash.py`).
# =============================================================================

class ElasticIntTable(ElasticHashTable):
    """
    Funnel-hash table with `int64` values and an `insert_or_increment` method.

    Same probe sequence as `ElasticHashTable._search`; the increment is safe
    because the table is append-only + single-threaded here (no concurrent
    inserts racing the read-modify-write on the same key).

    Round-7 task T-A3: replaces the legacy pre-funnel `KmerElasticHashTable`
    that lived in `bioinformatics/kmer_elastic_hash.py` (finding F-01).
    """

    def __init__(self, capacity: int, delta: float = 0.05, seed: int = 42):
        super().__init__(capacity=capacity, delta=delta, seed=seed)
        # Override the Python-list `values` with an int64 ndarray.
        self.values = np.zeros(self.total_size, dtype=np.int64)

    def insert(self, key: int, value: int) -> Tuple[bool, int]:
        """Insert (key, int value). `value` must be a Python int or np.int64."""
        return super().insert(int(key), int(value))

    def insert_or_increment(self, key: int, inc: int = 1) -> Tuple[bool, int]:
        """
        Insert `key` with value `inc` if absent, otherwise add `inc` to the
        existing value. Returns (ok, probe_count). The probe sequence is
        identical to `ElasticHashTable._search`.
        """
        pos, probes = self._search(int(key))
        if pos >= 0:
            self.values[pos] = self.values[pos] + int(inc)
            return True, probes
        # Key absent: descend the funnel as a fresh insert.
        return self.insert(int(key), int(inc))

    def get(self, key: int, default: Optional[int] = None) -> Optional[int]:
        """Dict-style access: int value for key, or `default` if absent."""
        pos, _ = self._search(int(key))
        return int(self.values[pos]) if pos >= 0 else default

    def lookup(self, key: int) -> Tuple[Optional[int], int]:
        """Returns (int value or None, probe_count)."""
        pos, probes = self._search(int(key))
        self.last_probes = probes
        return (int(self.values[pos]) if pos >= 0 else None), probes

    def remove(self, key: int) -> bool:
        """Delete key (tombstone).  Overrides parent to use 0 instead of
        None for the int64 values array."""
        pos, _ = self._search(int(key))
        if pos < 0:
            return False
        self.keys[pos] = _TOMBSTONE
        self.values[pos] = 0
        if pos >= self.b_offset:
            self._overflow_count -= 1
        self.count -= 1
        return True

    def items(self):
        """Iterate (key, int value) over all stored entries."""
        for pos in range(self.total_size):
            if self.occupied[pos] and self.keys[pos] != _TOMBSTONE:
                yield int(self.keys[pos]), int(self.values[pos])


# =============================================================================
# Vectorized batch probe
# =============================================================================

def funnel_probe(table: ElasticHashTable, query_keys: np.ndarray,
                 chunk: int = 4096) -> np.ndarray:
    """
    Vectorized (NumPy) mirror of `ElasticHashTable._search`: for each query
    key, evaluates the identical deterministic funnel probe sequence (slabs
    A_1..A_alpha with exhaustive beta-slot sub-array scans, then overflow B,
    then the two-choice C buckets) and returns the slot index holding the
    key, or -1 if the key is absent. Because the sequence is deterministic,
    every key inserted via `insert` is found here, and absent keys return -1.

    Implemented in NumPy on purpose: JAX's default 32-bit mode
    (JAX_ENABLE_X64=False) silently truncates the 64-bit mixer arithmetic
    the funnel geometry relies on, so a JAX port would not agree with the
    reference table without fragile workarounds. This function is the honest
    replacement for the historical `jax_hash_probe`, which probed only 2
    slots and could not find most keys.
    """
    q = np.asarray(query_keys, dtype=np.int64).ravel()
    out = np.full(q.shape, -1, dtype=np.int64)
    if q.size == 0:
        return out

    salts = np.array([s for (_, _, s) in table._slabs], dtype=np.uint64)
    counts = np.array([a for (_, a, _) in table._slabs], dtype=np.uint64)
    offsets = np.array([o for (o, _, _) in table._slabs], dtype=np.uint64)
    beta = np.uint64(table.beta)
    ar_beta = np.arange(table.beta, dtype=np.uint64)

    do_b = table.b_size > 0 and table.b_attempts > 0
    salt_b = np.array(table.salt_b, dtype=np.uint64)
    b_off = np.uint64(table.b_offset)
    b_size = np.uint64(table.b_size)
    salt_c1 = np.uint64(table.salt_c1)
    salt_c2 = np.uint64(table.salt_c2)
    c_off = np.uint64(table.c_offset)
    nb = np.uint64(table.c_num_buckets)
    cbs = np.uint64(table.c_bucket_slots)
    ar_c = np.arange(table.c_bucket_slots, dtype=np.uint64)

    for lo in range(0, q.size, chunk):
        qc = q[lo:lo + chunk]
        nqc = qc.shape[0]
        u = qc.astype(np.uint64)

        parts = []
        # Slabs: (nq, alpha, beta) — slab-major, slot-minor (search order).
        base = offsets[None, :] + \
            (_mix64_arr(u[:, None] ^ salts[None, :]) % counts[None, :]) * beta
        parts.append((base[:, :, None] + ar_beta[None, None, :]).reshape(nqc, -1))
        # B: (nq, b_attempts)
        if do_b:
            parts.append(b_off + _mix64_arr(u[:, None] ^ salt_b[None, :]) % b_size)
        # C: (nq, c_bucket_slots, 2) — slot-major, bucket1 then bucket2.
        b1 = _mix64_arr(u ^ salt_c1) % nb
        b2 = _mix64_arr(u ^ salt_c2) % nb
        buckets = np.stack([b1, b2], axis=1)[:, None, :]  # (nq, 1, 2)
        parts.append((c_off + buckets * cbs + ar_c[None, :, None]).reshape(nqc, -1))

        positions = np.concatenate(parts, axis=1).astype(np.int64)
        found = table.occupied[positions] & (table.keys[positions] == qc[:, None])
        first = np.argmax(found, axis=1)
        hit = found[np.arange(nqc), first]
        out[lo:lo + nqc] = np.where(hit, positions[np.arange(nqc), first], -1)
    return out


# Historical name (imported by core/__init__.py). The JAX-era decoration has
# been replaced by the genuine funnel search above; see funnel_probe's
# docstring for why the implementation is NumPy.
jax_hash_probe = funnel_probe


# =============================================================================
# Elastic hashing (paper Section 2) — simplified greedy variant with halved
# sub-arrays, cascade insertion and tombstone deletion. Secondary
# experimental class; the funnel table above is the default and the one used
# by the FMM engines.
# =============================================================================

_TOMBSTONE_KEY = -2
_EMPTY_KEY = -1


def _splitmix64(x: int) -> int:
    """SplitMix64 (public domain), used as the 2-D probe hasher's core."""
    x = (x + 0x9E3779B97F4A7C15) & _U64_MASK
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _U64_MASK
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _U64_MASK
    x ^= x >> 31
    return x


def _hash2d(key: int, i: int, j: int, size: int) -> int:
    """Two-dimensional probe hash h(key, subarray=i, attempt=j) -> [0, size)."""
    h = _splitmix64(_splitmix64(int(key) & _U64_MASK)
                    ^ ((i * 0x9E3779B1 + j * 0x85EBCA77) & _U64_MASK))
    return h % size


class ElasticBatchingHashTable:
    """
    Elastic-hashing open-addressed table (no reordering, tombstone deletes).

    Simplified greedy variant of the paper's Section 2 scheme: sub-arrays
    with sizes halving, insertion cascading from the first sub-array still
    below (1 - delta/2) fill, a bounded probe budget
    f(eps) = c*min(log^2(1/eps), log(1/delta)) in the primary sub-array,
    unlimited probes in the secondary one, and an exhaustive safety-net
    pass. insert() additionally runs a duplicate-prevention pre-scan over
    every sub-array's probe sequence before placing a new key (a key
    placed in an earlier primary sub-array would otherwise be re-inserted
    as a duplicate once the cascade advances), so a worst-case insert
    inspects O(total capacity) slots -- correctness over speed; the
    empirical probe counts in core/test_elastic_hash.py reflect this.
    Unlike the paper (whose insertion is non-greedy, with a 2-D probe
    sequence mapped through an injection phi), this implementation is
    greedy -- the bound it certifies in practice is the empirical one
    measured by core/test_elastic_hash.py, not Theorem 1.

    Keys must be non-negative integers (Morton keys, bucket ids, ...);
    values may be arbitrary Python objects.
    """

    def __init__(self, capacity: int, delta: float = 0.05):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")

        self.capacity = capacity
        self.delta = delta
        self.count = 0

        # Sub-arrays of geometrically decreasing size: capacity/2, capacity/4, ...
        self.subarray_sizes: List[int] = []
        remaining = capacity
        size = capacity // 2
        while remaining > 0 and size >= 1:
            actual = min(size, remaining)
            self.subarray_sizes.append(actual)
            remaining -= actual
            size //= 2
        if remaining > 0:
            self.subarray_sizes.append(remaining)

        self.keys: List[int] = [_EMPTY_KEY] * capacity
        self.values: List[Optional[Any]] = [None] * capacity
        self.offsets: List[int] = []
        off = 0
        for s in self.subarray_sizes:
            self.offsets.append(off)
            off += s

        self._fills = [0] * len(self.subarray_sizes)

        # probe-limit constant from the paper's analysis (tunable)
        self._probe_limit_c = 32.0

    # -- internal helpers ---------------------------------------------------

    def _probe_limit(self, i: int) -> int:
        eps = 1.0 - self._fills[i] / self.subarray_sizes[i]
        if eps <= self.delta / 2:
            return 0
        log_term = min(
            math.log2(1.0 / eps) ** 2,
            math.log2(1.0 / self.delta),
        )
        return max(1, int(self._probe_limit_c * log_term))

    def _insertion_subarrays(self) -> Tuple[int, int]:
        """Cascade rule: primary = first (largest) sub-array still below
        (1 - delta/2) fill; secondary = the next sub-array after it."""
        p = len(self.subarray_sizes) - 1
        for i in range(len(self.subarray_sizes)):
            if self._fills[i] < (1 - self.delta / 2) * self.subarray_sizes[i]:
                p = i
                break
        sec = min(p + 1, len(self.subarray_sizes) - 1)
        return p, sec

    def _slot_free(self, pos: int) -> bool:
        return self.keys[pos] == _EMPTY_KEY or self.keys[pos] == _TOMBSTONE_KEY

    def _place(self, pos: int, key: int, value: Any) -> int:
        """Write the entry; returns 1 if the sub-array fill increased."""
        was_empty = self.keys[pos] == _EMPTY_KEY
        self.keys[pos] = key
        self.values[pos] = value
        self.count += 1
        return 1 if was_empty else 0

    # -- public API ----------------------------------------------------------

    def insert(self, key: int, value: Any) -> Tuple[bool, int]:
        """Insert (key, value) without displacing existing entries.

        Returns (success, probe_count).  An existing key has its value
        updated in place (success, probes).
        """
        key = int(key)
        if key < 0:
            raise ValueError("keys must be non-negative integers")

        # First: search ALL sub-arrays for an existing copy of the key.
        # This is necessary because the primary/secondary probe phases
        # only check their own sub-arrays; a key placed in an earlier
        # sub-array (when it was primary) would be missed, creating a
        # duplicate.  The full scan is O(total_capacity) but is only
        # needed once per insert; the per-sub-array early-exit on
        # _EMPTY_KEY keeps it fast for absent keys.
        probes = 0
        for i in range(len(self.subarray_sizes)):
            size_i = self.subarray_sizes[i]
            for j in range(size_i):
                pos = self.offsets[i] + _hash2d(key, i, j, size_i)
                probes += 1
                k = self.keys[pos]
                if k == _EMPTY_KEY:
                    break
                if k == key:
                    self.values[pos] = value
                    return True, probes

        # Key not found: insert as new entry.
        if self.count >= self.capacity:
            return False, 0

        primary, secondary = self._insertion_subarrays()

        # 1) primary sub-array, bounded probe sequence
        limit = self._probe_limit(primary)
        size_p = self.subarray_sizes[primary]
        for j in range(min(limit, size_p)):
            pos = self.offsets[primary] + _hash2d(key, primary, j, size_p)
            probes += 1
            if self._slot_free(pos):
                self._fills[primary] += self._place(pos, key, value)
                return True, probes

        # 2) secondary sub-array, unlimited probes (amortized O(1))
        size_s = self.subarray_sizes[secondary]
        for j in range(size_s):
            pos = self.offsets[secondary] + _hash2d(key, secondary, j, size_s)
            probes += 1
            if self._slot_free(pos):
                self._fills[secondary] += self._place(pos, key, value)
                return True, probes

        # 3) exhaustive fallback over every sub-array in probe order
        for i in range(len(self.subarray_sizes)):
            size_i = self.subarray_sizes[i]
            for j in range(size_i):
                pos = self.offsets[i] + _hash2d(key, i, j, size_i)
                probes += 1
                if self._slot_free(pos):
                    self._fills[i] += self._place(pos, key, value)
                    return True, probes

        return False, probes

    def lookup(self, key: int) -> Tuple[Optional[Any], int]:
        """Return (value, probe_count); value is None if absent.

        Every inserted key is guaranteed findable as long as it has not
        been removed.

        Cost-class honesty note: an absent-key lookup terminates early only
        when it hits a never-occupied (_EMPTY_KEY) slot along a sub-array's
        probe order. In a mostly-full table -- or one with many tombstones
        from removals -- most slots are either occupied or tombstoned, so
        the early-exit rarely fires and the lookup degrades to scanning the
        full sub-array(s): worst-case O(total capacity) probes for an absent
        key. Present-key lookups are unaffected (they stop at the matching
        slot). Callers that probe for many absent keys against a full table
        should keep load factor low or use a separate occupancy set.
        """
        key = int(key)
        probes = 0
        for i in range(len(self.subarray_sizes)):
            size_i = self.subarray_sizes[i]
            for j in range(size_i):
                pos = self.offsets[i] + _hash2d(key, i, j, size_i)
                probes += 1
                k = self.keys[pos]
                if k == _EMPTY_KEY:
                    # never-occupied slot: key cannot be further along
                    # this sub-array's probe order
                    break
                if k == key:
                    return self.values[pos], probes
        return None, probes

    def remove(self, key: int) -> bool:
        """Delete key (tombstone, no displacement). Returns True if found."""
        key = int(key)
        for i in range(len(self.subarray_sizes)):
            size_i = self.subarray_sizes[i]
            for j in range(size_i):
                pos = self.offsets[i] + _hash2d(key, i, j, size_i)
                k = self.keys[pos]
                if k == _EMPTY_KEY:
                    break
                if k == key:
                    self.keys[pos] = _TOMBSTONE_KEY
                    self.values[pos] = None
                    self.count -= 1
                    return True
        return False

    # conveniences -----------------------------------------------------------

    def get(self, key: int, default: Optional[Any] = None) -> Any:
        v, _ = self.lookup(key)
        return default if v is None else v

    def __contains__(self, key: int) -> bool:
        key = int(key)
        for i in range(len(self.subarray_sizes)):
            size_i = self.subarray_sizes[i]
            for j in range(size_i):
                pos = self.offsets[i] + _hash2d(key, i, j, size_i)
                k = self.keys[pos]
                if k == _EMPTY_KEY:
                    break
                if k == key:
                    return True
        return False

    def __len__(self) -> int:
        return self.count

    def load_factor(self) -> float:
        return self.count / self.capacity

    def items(self):
        for pos in range(self.capacity):
            if self.keys[pos] >= 0:
                yield self.keys[pos], self.values[pos]
