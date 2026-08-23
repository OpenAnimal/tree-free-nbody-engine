"""
Non-FMM Module: High-Throughput Genomic k-mer Counter & De Bruijn Graph Indexer.
Powered by Farach-Colton Non-Reordering Multi-Level Open Addressing Hashing.
Enables lock-free streaming k-mer ingestion with zero element displacement during sequencing.

Round-7 task T-A3: the legacy pre-funnel hash table body that lived here
(finding F-01) has been replaced by a thin wrapper over
`core.elastic_hash.ElasticIntTable` (the funnel-hash-backed int-value table).
The canonical-kmer and 2-bit packing code is kept. `bioinformatics/STATUS.md`'s
claim that this module "uses the core `ElasticHashTable` for real (queried,
load-bearing) k-mer indexing" is now true.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import time
from typing import Tuple, Dict, List, Optional, Iterator

# Make `core` importable whether this file is loaded as part of a package or
# as a top-level script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.elastic_hash import ElasticIntTable


# 2-bit nucleotide bit encoding: A=00, C=01, G=10, T=11
NUC_MAP = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3, "a": 0, "c": 1, "g": 2, "t": 3, "u": 3}
REV_NUC_MAP = {0: "A", 1: "C", 2: "G", 3: "T"}
COMP_MAP = {0: 3, 1: 2, 2: 1, 3: 0}  # A<->T, C<->G

# 256-entry ASCII lookup table for vectorized 2-bit encoding.
# Valid nucleotides map to 0-3; everything else (N, X, punctuation, etc.)
# maps to 0xFF to mark the position as invalid so k-mers spanning it are
# skipped (preserving the legacy "N resets the rolling frame" semantics).
_NUC_LUT = np.full(256, 0xFF, dtype=np.uint8)
for _ch, _val in NUC_MAP.items():
    _NUC_LUT[ord(_ch)] = _val


class KmerElasticHashTable:
    """
    Genomic k-mer Hash Table using Farach-Colton funnel hashing.
    Stores canonical 2-bit bitpacked k-mers and their integer occurrence counts.

    Backed by `core.elastic_hash.ElasticIntTable` (the funnel-hash int-value
    table). The probe sequence is the deterministic funnel schedule with a
    worst-case bound of `probe_bound` slot inspections per lookup.
    """
    def __init__(self, k: int = 21, capacity: int = 1000000, delta: float = 0.05, num_levels: int = 5):
        assert 1 <= k <= 31, "k must be between 1 and 31 for 64-bit packing."
        self.k = int(k)
        self.k_mask = (1 << (2 * self.k)) - 1
        self.delta = float(delta)
        # `num_levels` is kept for backward-compat with the old constructor
        # signature; the funnel table derives its own alpha from delta.
        self.num_levels = int(num_levels)
        self.capacity = max(1024, int(capacity))
        self._table = ElasticIntTable(capacity=self.capacity, delta=delta)
        self.num_unique_kmers = 0

    @property
    def probe_bound(self) -> int:
        """Deterministic worst-case probe count of the underlying funnel table."""
        return self._table.probe_bound

    def canonical_kmer_key(self, forward_key: int) -> int:
        """Computes the canonical k-mer (lexicographical minimum of forward and reverse-complement)."""
        rev_key = 0
        temp = forward_key
        for _ in range(self.k):
            nuc = temp & 3
            rev_nuc = COMP_MAP[nuc]
            rev_key = (rev_key << 2) | rev_nuc
            temp >>= 2
        return min(forward_key, rev_key)

    def insert_or_increment(self, key: int, increment: int = 1) -> Tuple[bool, int]:
        """
        Inserts canonical k-mer or increments existing count without ANY element displacement.
        Returns (success, probe_count).
        """
        ok, probes = self._table.insert_or_increment(int(key), int(increment))
        if not ok:
            # Insert failed: the funnel table is full. With two-pass sizing
            # (capacity sized from the sequence length) this should never
            # happen in normal use; surface it loudly with diagnostics rather
            # than silently dropping the k-mer (R7-F24).
            raise RuntimeError(
                f"KmerElasticHashTable insert failed (table full): "
                f"capacity={self.capacity}, delta={self.delta}, "
                f"occupied={self._table.count}, probe_bound={self._table.probe_bound}. "
                f"Re-instantiate with a larger capacity."
            )
        # Unique k-mer count is the funnel table's occupied-slot count; keep
        # it fresh on every insert so direct insert_or_increment callers see
        # an accurate count (R7-F24b — previously only ingest_sequence refreshed).
        self.num_unique_kmers = self._table.count
        return ok, probes

    def query_count(self, key: int) -> int:
        """Queries count of canonical k-mer (0 if absent)."""
        val = self._table.get(int(key), default=0)
        return int(val) if val is not None else 0

    def ingest_sequence(self, sequence: str) -> int:
        """
        Streams a fasta/fastq DNA string, extracting and counting rolling k-mers.

        Vectorized O(L) ingestion: the 2-bit encoding, sliding-window k-mer
        extraction, and canonical-form (min(kmer, revcomp)) computation are all
        performed in numpy. The legacy per-character Python loop with per-k-mer
        Python reverse-complement (O(L*k) interpreted) is replaced by a single
        sliding-window pass plus a vectorized bit-pair-reversal XOR. Counts are
        aggregated via ``np.unique`` before insertion, preserving exact count
        semantics (each occurrence contributes +1 to its canonical k-mer).

        Ambiguous characters (N, etc.) mark positions invalid; any window
        spanning an invalid position is skipped, matching the legacy rolling-
        frame reset behavior.
        """
        seq_len = len(sequence)
        if seq_len < self.k:
            return 0

        # 1. Vectorized 2-bit encoding via the ASCII LUT.
        arr = np.frombuffer(sequence.encode("ascii", errors="replace"), dtype=np.uint8)
        n = _NUC_LUT[arr]                       # (L,) uint8, 0xFF for invalid
        valid = (n != 0xFF)
        n_clean = n.copy()
        n_clean[~valid] = 0                      # placeholder for invalid positions

        # 2. Sliding-window views of length k (zero-copy strided views).
        win_codes = np.lib.stride_tricks.sliding_window_view(n_clean, self.k)
        win_valid = np.lib.stride_tricks.sliding_window_view(valid, self.k)
        fully_valid = win_valid.all(axis=1)      # (L-k+1,) bool

        if not np.any(fully_valid):
            return 0

        # 3. Forward k-mer keys: key = sum(n[i] << 2*(k-1-i)) for i in 0..k-1.
        #    Precompute 4^i powers (k <= 31 so 4^30 = 2^60 fits in uint64).
        powers = np.array(
            [np.uint64(4) ** np.uint64(i) for i in range(self.k)],
            dtype=np.uint64,
        )[::-1]                                   # (k,) = [4^(k-1), ..., 4^0]
        fwd = win_codes[fully_valid].astype(np.uint64)
        forward_keys = (fwd * powers[None, :]).sum(axis=1, dtype=np.uint64)

        # 4. Reverse-complement keys: reverse each window and XOR each 2-bit
        #    code with 3 (A<->T, C<->G). This is the vectorized equivalent of
        #    canonical_kmer_key's bit-pair-reversal + complement.
        rev = fwd[:, ::-1] ^ np.uint64(3)
        revcomp_keys = (rev * powers[None, :]).sum(axis=1, dtype=np.uint64)

        # 5. Canonical = min(forward, revcomp).
        canon_keys = np.minimum(forward_keys, revcomp_keys)

        # 6. Aggregate per-canonical-key counts and insert into the funnel
        #    table. np.unique returns sorted unique keys with occurrence counts;
        #    inserting each with increment=count reproduces the exact per-k-mer
        #    increment-1 semantics of the legacy loop (addition is commutative).
        uniq_keys, counts = np.unique(canon_keys, return_counts=True)
        for key, cnt in zip(uniq_keys, counts):
            self.insert_or_increment(int(key), int(cnt))

        total_ingested = int(canon_keys.size)
        self.num_unique_kmers = self._table.count
        return total_ingested

    def get_kmer_spectrum(self, max_depth: int = 50) -> np.ndarray:
        """Calculates k-mer frequency distribution spectrum (1x, 2x, 3x ... coverage)."""
        counts = np.array([v for _, v in self._table.items()], dtype=np.int32)
        if counts.size == 0:
            return np.zeros(max_depth + 1, dtype=np.int32)
        spectrum = np.bincount(counts, minlength=max_depth + 1)
        return spectrum[:max_depth + 1]

    def decode_kmer(self, key: int) -> str:
        """Decodes 64-bit integer key back to nucleotide string."""
        chars = []
        temp = key
        for _ in range(self.k):
            nuc = temp & 3
            chars.append(REV_NUC_MAP[nuc])
            temp >>= 2
        return "".join(reversed(chars))
