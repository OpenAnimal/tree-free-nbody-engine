"""
Non-Reordering Elastic Quotient Filter & Approximate Membership Query (AMQ) Suite.
Based on Optimal Open Addressing Without Reordering (Farach-Colton, Krapivin, & Kuszmaul, 2025).

Replaces classical Bloom Filters, Quotient Filters, and Cuckoo Filters with a zero-displacement,
lock-free compatible multi-level geometric quotient filter.

Key Features:
1. Zero Element Displacement: No shifting runs or Robin-Hood cascading evictions on collision.
2. O(1) Amortized Probe Complexity and O(log delta^-1) Worst-Case Expected Search.
3. Frequency Counting: Dual-purpose AMQ and exact/approximate multiset frequency sketch.
4. Set Algebra: Lock-free set intersection, union, and Jaccard similarity estimation.

--------------------------------------------------------------------------------
Round-7 task T-A4 banner: this module is a LEGACY PRE-FUNNEL scheme. The
geometric-levels + per-level linear-probing body below is NOT the FKK
funnel schedule implemented in `core/elastic_hash.py` (which has a
deterministic `probe_bound` and the A_1..A_alpha / B / C funnel geometry).
Porting the quotient/remainder + fingerprint semantics to `ElasticIntTable`
is awkward because the funnel table stores (key, int value) pairs rather
than (fingerprint, counter) pairs, so this module is left as-is with this
honesty banner. New code should prefer `core.elastic_hash.ElasticHashTable`
/ `ElasticIntTable`; this module is kept for the AMQ / Jaccard-similarity
reference baseline. See `algorithm_theory/STATUS.md`.
--------------------------------------------------------------------------------
"""

from typing import Tuple, Optional, List, Dict, Union, Any
import numpy as np
import time


class ElasticQuotientFilter:
    """
    Zero-Displacement Elastic Quotient Filter (EQF).
    
    Splits a 64-bit cryptographic/multiplicative hash into:
      - Quotient (q): Target bucket address in geometric sub-table.
      - Remainder / Fingerprint (r): Compact integer signature stored in the slot.
      - Counter (c): Optional multiplicity counter for multiset frequency queries.
    """
    def __init__(
        self,
        capacity: int,
        fingerprint_bits: int = 16,
        num_levels: int = 5,
        enable_counters: bool = True,
        delta: float = 0.05,
    ):
        """
        Initializes the Elastic Quotient Filter.
        
        Parameters
        ----------
        capacity : int
            Target item capacity.
        fingerprint_bits : int
            Number of bits per remainder fingerprint (controls false-positive rate ~ 2^-r).
        num_levels : int
            Number of geometric funnel buffer levels.
        enable_counters : bool
            Whether to track multiset occurrence frequencies.
        delta : float
            Target overflow factor for level sizing.
        """
        self.capacity = max(64, int(capacity))
        self.fingerprint_bits = max(4, min(32, int(fingerprint_bits)))
        self.max_fingerprint = (1 << self.fingerprint_bits) - 1
        self.num_levels = max(2, int(num_levels))
        self.enable_counters = bool(enable_counters)
        self.delta = float(delta)
        
        # Geometrically distributed level sizes
        fractions = [0.5**(i + 1) for i in range(self.num_levels - 1)]
        fractions.append(1.0 - sum(fractions))
        
        self.level_sizes = [max(16, int(self.capacity * f * 1.3)) for f in fractions]
        self.total_slots = sum(self.level_sizes)
        self.level_offsets = [0] + list(np.cumsum(self.level_sizes)[:-1])
        
        # Storage arrays
        self.fingerprints = np.zeros(self.total_slots, dtype=np.uint32)
        self.occupied = np.zeros(self.total_slots, dtype=bool)
        if self.enable_counters:
            self.counters = np.zeros(self.total_slots, dtype=np.uint32)
        else:
            self.counters = None
            
        self.count = 0
        
        # Hash salt parameters
        rng = np.random.RandomState(1337)
        self.salts_a = rng.randint(1, 2**31 - 1, size=(self.num_levels, 4), dtype=np.int64)
        self.salts_b = rng.randint(0, 2**31 - 1, size=(self.num_levels, 4), dtype=np.int64)

    def _hash(self, key: int) -> Tuple[int, int]:
        """Maps an integer key into 64-bit hash -> (raw_quotient, fingerprint)."""
        # 64-bit Murmur-style mixer using python native 64-bit bitmasks
        k = int(key) & 0xFFFFFFFFFFFFFFFF
        k ^= (k >> 33)
        k = (k * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
        k ^= (k >> 33)
        k = (k * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
        k ^= (k >> 33)
        
        fingerprint = int(k & self.max_fingerprint)
        if fingerprint == 0:
            fingerprint = 1  # Reserve 0 for empty slot
            
        raw_quotient = int(k >> self.fingerprint_bits)
        return raw_quotient, fingerprint

    def insert(self, key: int, count: int = 1) -> Tuple[bool, int]:
        """
        Inserts a key into the filter without displacing existing elements.
        
        Returns
        -------
        Tuple[bool, int]
            (Success status, number of probes required).
        """
        raw_q, fp = self._hash(key)
        probes = 0
        
        for lvl in range(self.num_levels):
            size = self.level_sizes[lvl]
            offset = self.level_offsets[lvl]
            max_probes = min(size, 4 + lvl * 2)
            
            for attempt in range(max_probes):
                probes += 1
                a = int(self.salts_a[lvl, attempt % 4])
                b = int(self.salts_b[lvl, attempt % 4])
                bucket = int((raw_q * a + b + attempt * 2654435761) & 0x7FFFFFFF) % size
                slot = offset + bucket
                
                # Check if slot is empty -> Insert without displacement
                if not self.occupied[slot]:
                    self.occupied[slot] = True
                    self.fingerprints[slot] = fp
                    if self.enable_counters:
                        self.counters[slot] = count
                    self.count += 1
                    return True, probes
                    
                # If slot contains matching fingerprint -> Increment frequency counter
                if self.fingerprints[slot] == fp:
                    if self.enable_counters:
                        self.counters[slot] += count
                    return True, probes
                    
        return False, probes

    def contains(self, key: int) -> Tuple[bool, int]:
        """
        Checks membership of key in the filter.
        
        Returns
        -------
        Tuple[bool, int]
            (True if present (subject to ~2^-r false positive rate), probes made).
        """
        raw_q, fp = self._hash(key)
        probes = 0
        
        for lvl in range(self.num_levels):
            size = self.level_sizes[lvl]
            offset = self.level_offsets[lvl]
            max_probes = min(size, 4 + lvl * 2)
            
            for attempt in range(max_probes):
                probes += 1
                a = int(self.salts_a[lvl, attempt % 4])
                b = int(self.salts_b[lvl, attempt % 4])
                bucket = int((raw_q * a + b + attempt * 2654435761) & 0x7FFFFFFF) % size
                slot = offset + bucket
                
                if not self.occupied[slot]:
                    continue
                    
                if self.fingerprints[slot] == fp:
                    return True, probes
                    
        return False, probes

    def get_frequency(self, key: int) -> int:
        """Returns estimated multiset frequency count for key."""
        if not self.enable_counters:
            present, _ = self.contains(key)
            return 1 if present else 0
            
        raw_q, fp = self._hash(key)
        for lvl in range(self.num_levels):
            size = self.level_sizes[lvl]
            offset = self.level_offsets[lvl]
            max_probes = min(size, 4 + lvl * 2)
            
            for attempt in range(max_probes):
                a = int(self.salts_a[lvl, attempt % 4])
                b = int(self.salts_b[lvl, attempt % 4])
                bucket = int((raw_q * a + b + attempt * 2654435761) & 0x7FFFFFFF) % size
                slot = offset + bucket
                
                if self.occupied[slot] and self.fingerprints[slot] == fp:
                    return int(self.counters[slot])
                    
        return 0

    def insert_batch(self, keys: np.ndarray) -> Dict[str, float]:
        """Batch inserts keys and returns performance statistics."""
        k_arr = np.asarray(keys, dtype=np.int64).ravel()
        t0 = time.perf_counter()
        total_probes = 0
        success_count = 0
        
        for k in k_arr:
            ok, p = self.insert(int(k))
            total_probes += p
            if ok:
                success_count += 1
                
        elapsed = time.perf_counter() - t0
        return {
            "num_keys": len(k_arr),
            "inserted": success_count,
            "avg_probes": total_probes / max(1, len(k_arr)),
            "throughput_keys_sec": len(k_arr) / max(1e-9, elapsed),
            "load_factor": self.count / self.total_slots
        }

    def compute_jaccard_similarity(self, other: "ElasticQuotientFilter") -> float:
        """
        Computes approximate Jaccard similarity between two Elastic Quotient Filters
        by matching common occupied slot fingerprints.
        """
        if self.total_slots != other.total_slots:
            raise ValueError("Filters must have matching geometry for fast SIMD comparison")
            
        both_occ = self.occupied & other.occupied
        fp_match = both_occ & (self.fingerprints == other.fingerprints)
        
        intersection_count = np.sum(fp_match)
        either_occ = self.occupied | other.occupied
        union_count = np.sum(either_occ)
        
        if union_count == 0:
            return 1.0
        return float(intersection_count / union_count)


class ClassicBloomFilterBaseline:
    """Standard k-hash Bloom Filter reference baseline for comparison."""
    def __init__(self, capacity: int, bits_per_element: int = 10, num_hashes: int = 7):
        self.capacity = capacity
        self.num_bits = capacity * bits_per_element
        self.num_hashes = num_hashes
        self.bit_array = np.zeros(self.num_bits, dtype=bool)
        self.count = 0

    def insert(self, key: int):
        for h in range(self.num_hashes):
            idx = int((hash((key, h)) & 0x7FFFFFFF) % self.num_bits)
            self.bit_array[idx] = True
        self.count += 1

    def contains(self, key: int) -> bool:
        for h in range(self.num_hashes):
            idx = int((hash((key, h)) & 0x7FFFFFFF) % self.num_bits)
            if not self.bit_array[idx]:
                return False
        return True


# =============================================================================
# Round-7 task T-A4b: Funnel Quotient Filter (the real port)
# =============================================================================

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from core.elastic_hash import ElasticIntTable


class FunnelQuotientFilter:
    """Round-7 task T-A4b: a quotient filter built on the FKK funnel hash.

    Stores (remainder, count) packed in one int64 value, keyed by the
    quotient bits of a 64-bit hash. The funnel hash's deterministic
    ``probe_bound = αβ + bAttempts + 2·cBucketSlots`` gives the quotient
    filter's first worst-case-bounded search — versus classic quotient
    filters' amortized-only bounds.

    Parameters
    ----------
    capacity : int — expected number of distinct items
    delta : float — funnel hash load-factor slack (default 0.05)
    r_remainder_bits : int — number of remainder/fingerprint bits (default 8).
        The false-positive rate is ~ n_stored / 2^64 (full-hash-collision bound,
        NOT 2^(-r); see the false-positive section below). Keep r <= 24 so the
        count field has >= 40 bits in the int64 value packing.
    seed : int — funnel hash seed

    Entry encoding (int64 value):
        bits [0, r)       = remainder (fingerprint)
        bits [r, 64)      = count (multiplicity)
    Key = quotient = hash64(item) >> r  (the full 64-r high bits, NOT masked
    to 32 bits -- the funnel table accepts the full int64 key range).

    False-positive behaviour
    ------------------------
    A *false positive* requires an absent item to (a) share the quotient of an
    existing entry AND (b) share that entry's stored remainder -- i.e. the two
    items' full 64-bit hashes collide. The per-absent-item FP probability is
    therefore ~ n_stored / 2^64 (a full-hash-collision / birthday-class bound),
    which is *vastly* smaller than the naive 2^(-r) remainder-collision figure
    one would get if quotients were ignored. With r=8 and 10^5 items this is
    ~ 10^5 / 2^64 ~ 5e-15, so the <0.5% FPR test in
    `test_basic_datatypes_fmm.py` holds trivially; it is retained as a
    sanity guard, not because 2^(-r) is the operating FP rate.
    """

    def __init__(self, capacity: int, delta: float = 0.05,
                 r_remainder_bits: int = 8, seed: int = 42):
        self.r = int(r_remainder_bits)
        assert 1 <= self.r <= 24, f"r_remainder_bits must be in [1, 24], got {self.r}"
        self.remainder_mask = (1 << self.r) - 1
        self.count_shift = self.r
        self.capacity = int(capacity)
        self.delta = float(delta)
        self.seed = int(seed)
        # The funnel table stores (quotient_key, packed_value) pairs.
        self.table = ElasticIntTable(capacity=self.capacity, delta=self.delta,
                                     seed=self.seed)
        self._n_items = 0  # total insertions (including duplicates)

    def _hash64(self, item) -> int:
        """64-bit hash of an item (int, str, or bytes).

        ``int`` items use a splitmix64-style mix. ``str``/``bytes`` items use a
        deterministic FNV-1a-64 hash of the UTF-8 bytes followed by a splitmix64
        mix, so that filters are **reproducible across processes** (Python's
        built-in ``hash()`` is randomised by PYTHONHASHSEED for str/bytes and
        would make two independent constructions of the same filter disagree).
        Other types fall back to ``hash()`` and are NOT guaranteed reproducible.
        """
        if isinstance(item, int):
            # Use a mixing function for ints (splitmix64-style)
            x = (item ^ 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
            x = (x ^ (x >> 30)) & 0xFFFFFFFFFFFFFFFF
            x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
            x = (x ^ (x >> 31)) & 0xFFFFFFFFFFFFFFFF
            return x
        elif isinstance(item, (str, bytes)):
            b = item.encode("utf-8") if isinstance(item, str) else bytes(item)
            # FNV-1a 64-bit over the bytes (deterministic, process-independent).
            h = 0xCBF29CE484222325
            for byte in b:
                h ^= byte
                h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
            # splitmix64 finaliser for better bit diffusion.
            h = (h + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            h = (h ^ (h >> 30)) & 0xFFFFFFFFFFFFFFFF
            h = (h * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
            h = (h ^ (h >> 27)) & 0xFFFFFFFFFFFFFFFF
            h = (h * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
            h = (h ^ (h >> 31)) & 0xFFFFFFFFFFFFFFFF
            return h
        else:
            return int(hash(item)) & 0xFFFFFFFFFFFFFFFF

    def insert(self, item) -> Tuple[bool, int]:
        """Insert item. Returns (was_new, probe_count).

        On first insert of a new item: creates a new entry with count=1.
        On duplicate (same quotient AND same remainder): increments count.
        On quotient collision (same quotient, different remainder): overwrites
        with the new item (rare: with r=8, quotient is 56 bits, so collision
        probability at 10^5 items is ~10^-7).

        The false-positive rate is ~ n_stored / 2^64 (full 64-bit hash
        collision), NOT 2^(-r): an absent item must match both the quotient
        AND the stored remainder of an existing entry, i.e. collide on the
        full 64-bit hash.

        Raises
        ------
        RuntimeError
            If the underlying funnel table is full and cannot accept the new
            key. A membership sketch must never silently drop an item, so
            overflow is surfaced loudly with capacity diagnostics rather than
            reported as a successful insert.
        """
        h = self._hash64(item)
        q = h >> self.r             # quotient (64-r bits, e.g. 56 for r=8)
        rem = h & self.remainder_mask  # r-bit remainder
        packed = (1 << self.count_shift) | rem  # count=1, remainder=rem

        existing, probes = self.table.lookup(q)
        if existing is not None:
            existing_rem = existing & self.remainder_mask
            if existing_rem == rem:
                # Same item: increment count
                count = existing >> self.count_shift
                new_val = ((count + 1) << self.count_shift) | rem
                pos, _ = self.table._search(q)
                self.table.values[pos] = new_val
                self._n_items += 1
                return False, probes
            else:
                # Quotient collision with different remainder: overwrite.
                # This is a rare event (quotient is 56+ bits); the old item
                # is lost (false negative for it), but the new item is stored.
                ok, _ = self.table.insert(q, packed)
                if not ok:
                    raise RuntimeError(
                        f"FunnelQuotientFilter: underlying table is full "
                        f"(capacity={self.capacity}, stored distinct="
                        f"{self.table.count}, total insertions="
                        f"{self._n_items}); cannot insert new quotient. "
                        f"Enlarge `capacity`."
                    )
                self._n_items += 1
                return True, probes
        else:
            ok, _ = self.table.insert(q, packed)
            if not ok:
                raise RuntimeError(
                    f"FunnelQuotientFilter: underlying table is full "
                    f"(capacity={self.capacity}, stored distinct="
                    f"{self.table.count}, total insertions="
                    f"{self._n_items}); cannot insert new quotient. "
                    f"Enlarge `capacity`."
                )
            self._n_items += 1
            return True, probes

    def contains(self, item) -> Tuple[bool, int]:
        """Check if item is probably in the filter. Returns (maybe_present, probe_count).

        False positives: an item whose quotient matches an existing key AND
        whose remainder matches the stored remainder, but was never inserted --
        i.e. a full 64-bit hash collision. Probability ~ n_stored / 2^64
        (NOT 2^(-r); see the class docstring).
        False negatives: never (except for the rare quotient-collision
        overwrite case, ~10^-7 at 10^5 items with r=8).
        """
        h = self._hash64(item)
        q = h >> self.r
        rem = h & self.remainder_mask

        val, probes = self.table.lookup(q)
        if val is None:
            return False, probes
        existing_rem = val & self.remainder_mask
        if existing_rem == rem:
            return True, probes
        return False, probes

    def count_of(self, item) -> int:
        """Return the approximate multiplicity count of item, or 0 if absent."""
        h = self._hash64(item)
        q = h >> self.r
        rem = h & self.remainder_mask
        val, _ = self.table.lookup(q)
        if val is not None and (val & self.remainder_mask) == rem:
            return val >> self.count_shift
        return 0

    @property
    def probe_bound(self) -> int:
        """The funnel hash's deterministic worst-case probe bound."""
        return self.table.probe_bound

    @property
    def length(self) -> int:
        """Number of distinct keys in the table."""
        return self.table.count

    @property
    def n_items(self) -> int:
        """Total insertions (including duplicates)."""
        return self._n_items
