"""
Non-FMM Module: High-Throughput Genomic k-mer Counter & De Bruijn Graph Indexer.
Powered by Farach-Colton Non-Reordering Multi-Level Open Addressing Hashing.
Enables lock-free streaming k-mer ingestion with zero element displacement during sequencing.
"""

from __future__ import annotations
import numpy as np
import time
from typing import Tuple, Dict, List, Optional, Iterator


# 2-bit nucleotide bit encoding: A=00, C=01, G=10, T=11
NUC_MAP = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3, "a": 0, "c": 1, "g": 2, "t": 3, "u": 3}
REV_NUC_MAP = {0: "A", 1: "C", 2: "G", 3: "T"}
COMP_MAP = {0: 3, 1: 2, 2: 1, 3: 0}  # A<->T, C<->G


class KmerElasticHashTable:
    """
    Lock-Free Compatible Genomic k-mer Hash Table using Farach-Colton Open Addressing.
    Stores canonical 2-bit bitpacked k-mers and their integer occurrence counts.
    """
    def __init__(self, k: int = 21, capacity: int = 1000000, delta: float = 0.05, num_levels: int = 5):
        assert 1 <= k <= 31, "k must be between 1 and 31 for 64-bit packing."
        self.k = int(k)
        self.k_mask = (1 << (2 * self.k)) - 1
        self.delta = float(delta)
        self.num_levels = int(num_levels)

        # Geometrically distributed sub-array level sizes (Farach-Colton 2025)
        fractions = [0.5**(i + 1) for i in range(self.num_levels - 1)]
        fractions.append(1.0 - sum(fractions))

        self.capacity = max(1024, int(capacity / (1.0 - delta)))
        self.level_sizes = [max(64, int(self.capacity * f)) for f in fractions]
        self.total_size = sum(self.level_sizes)
        self.level_offsets = [0] + list(np.cumsum(self.level_sizes)[:-1])

        # Flat contiguous backing memory
        self.keys = np.full(self.total_size, -1, dtype=np.int64)
        self.counts = np.zeros(self.total_size, dtype=np.int32)
        self.occupied = np.zeros(self.total_size, dtype=bool)

        rng = np.random.RandomState(42)
        self.seeds_a = rng.randint(1, 2**31 - 1, size=(self.num_levels, 4), dtype=np.int64)
        self.seeds_b = rng.randint(0, 2**31 - 1, size=(self.num_levels, 4), dtype=np.int64)
        self.num_unique_kmers = 0

    def _hash(self, key: int, level: int, attempt: int) -> int:
        a = self.seeds_a[level, attempt % 4]
        b = self.seeds_b[level, attempt % 4]
        size = self.level_sizes[level]
        raw_h = (int(key) * int(a) + int(b) + attempt * 2654435761) & 0x7FFFFFFF
        return (raw_h % size)

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
        probes = 0
        for level in range(self.num_levels):
            offset = self.level_offsets[level]
            size = self.level_sizes[level]
            max_attempts = min(size, 4 + level * 2)

            for attempt in range(max_attempts):
                probes += 1
                pos = offset + self._hash(key, level, attempt)

                if not self.occupied[pos]:
                    self.keys[pos] = key
                    self.counts[pos] = increment
                    self.occupied[pos] = True
                    self.num_unique_kmers += 1
                    return True, probes
                elif self.keys[pos] == key:
                    self.counts[pos] += increment
                    return True, probes

        # Fallback linear scan
        for pos in range(self.total_size):
            probes += 1
            if not self.occupied[pos]:
                self.keys[pos] = key
                self.counts[pos] = increment
                self.occupied[pos] = True
                self.num_unique_kmers += 1
                return True, probes
            elif self.keys[pos] == key:
                self.counts[pos] += increment
                return True, probes

        return False, probes

    def query_count(self, key: int) -> int:
        """Queries count of canonical k-mer in bounded O(log 1/delta) expected probes."""
        for level in range(self.num_levels):
            offset = self.level_offsets[level]
            size = self.level_sizes[level]
            max_attempts = min(size, 4 + level * 2)

            for attempt in range(max_attempts):
                pos = offset + self._hash(key, level, attempt)
                if not self.occupied[pos]:
                    continue
                if self.keys[pos] == key:
                    return int(self.counts[pos])

        for pos in range(self.total_size):
            if self.occupied[pos] and self.keys[pos] == key:
                return int(self.counts[pos])

        return 0

    def ingest_sequence(self, sequence: str) -> int:
        """
        Streams a fasta/fastq DNA string, extracting and counting rolling k-mers in O(L) time.
        """
        seq_len = len(sequence)
        if seq_len < self.k:
            return 0

        rolling_key = 0
        valid_len = 0
        total_ingested = 0

        for ch in sequence:
            if ch in NUC_MAP:
                val = NUC_MAP[ch]
                rolling_key = ((rolling_key << 2) | val) & self.k_mask
                valid_len += 1

                if valid_len >= self.k:
                    canon_key = self.canonical_kmer_key(rolling_key)
                    self.insert_or_increment(canon_key, increment=1)
                    total_ingested += 1
            else:
                # Ambiguous character (N) resets rolling frame
                rolling_key = 0
                valid_len = 0

        return total_ingested

    def get_kmer_spectrum(self, max_depth: int = 50) -> np.ndarray:
        """Calculates k-mer frequency distribution spectrum (1x, 2x, 3x ... coverage)."""
        valid_counts = self.counts[self.occupied]
        spectrum = np.bincount(valid_counts, minlength=max_depth + 1)
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
