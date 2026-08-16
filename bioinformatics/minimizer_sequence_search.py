"""
Module 9: High-Throughput Genomic Minimizer Indexing & Seed-and-Extend Sequence Alignment Engine.
Implements (w, k)-minimizer hashing, Farach-Colton non-reordering seed tables,
and co-linear anchor chaining (Minimap2-style) for ultra-fast DNA/RNA database search.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

try:
    from .kmer_elastic_hash import NUC_MAP, REV_NUC_MAP, COMP_MAP, KmerElasticHashTable
except (ImportError, ValueError):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from bioinformatics.kmer_elastic_hash import NUC_MAP, REV_NUC_MAP, COMP_MAP, KmerElasticHashTable


@dataclass
class MinimizerSeed:
    """Represents an extracted (w, k)-minimizer seed."""
    kmer_key: int
    pos: int
    is_rev_comp: bool


@dataclass
class SeedHit:
    """Exact seed match between query and target reference."""
    query_pos: int
    ref_id: str
    ref_pos: int
    kmer_len: int


@dataclass
class AlignmentChain:
    """Chained alignment block formed by co-linear seed anchors."""
    ref_id: str
    query_start: int
    query_end: int
    ref_start: int
    ref_end: int
    chain_score: int
    num_anchors: int
    identity_approx: float
    cigar_approx: str


class MinimizerSequenceSearchEngine:
    """
    High-Throughput (w, k)-Minimizer Indexer and Anchor Chaining Sequence Search Engine.
    Powered by Farach-Colton non-reordering multi-level open addressing.
    """
    def __init__(
        self,
        k: int = 15,
        w: int = 10,
        capacity: int = 500000,
        max_chain_gap: int = 500
    ):
        assert 1 <= k <= 31, "k must be in [1, 31] for 64-bit integer bitpacking."
        assert w >= 1, "Window size w must be >= 1."
        self.k = int(k)
        self.w = int(w)
        self.k_mask = (1 << (2 * self.k)) - 1
        self.max_chain_gap = int(max_chain_gap)

        # Multi-level non-reordering hash table for minimizer occurrence indexing
        self.kmer_table = KmerElasticHashTable(k=self.k, capacity=capacity)
        
        # Position repository: maps minimizer key -> list of (ref_id, ref_pos)
        self.seed_postings: Dict[int, List[Tuple[str, int]]] = {}
        self.indexed_references: Dict[str, int] = {} # ref_id -> length

    def _canonical_kmer(self, forward_key: int) -> Tuple[int, bool]:
        """Computes canonical k-mer and flag indicating if reverse complement was chosen."""
        rev_key = 0
        temp = forward_key
        for _ in range(self.k):
            nuc = temp & 3
            rev_key = (rev_key << 2) | COMP_MAP[nuc]
            temp >>= 2
        if forward_key <= rev_key:
            return forward_key, False
        return rev_key, True

    def extract_minimizers(self, sequence: str) -> List[MinimizerSeed]:
        """
        Extracts (w, k)-minimizers in linear O(L) time using a rolling window buffer.
        """
        seq_len = len(sequence)
        if seq_len < self.k:
            return []

        # 1. Extract all canonical k-mers with position and orientation
        kmers: List[Tuple[int, int, bool]] = []
        rolling_key = 0
        valid_len = 0

        for i, ch in enumerate(sequence):
            if ch in NUC_MAP:
                val = NUC_MAP[ch]
                rolling_key = ((rolling_key << 2) | val) & self.k_mask
                valid_len += 1

                if valid_len >= self.k:
                    pos = i - self.k + 1
                    canon_key, is_rev = self._canonical_kmer(rolling_key)
                    kmers.append((canon_key, pos, is_rev))
            else:
                rolling_key = 0
                valid_len = 0

        if not kmers:
            return []

        # 2. Sliding window of size w to select minimum hash in each window
        minimizers: List[MinimizerSeed] = []
        last_minimizer_pos = -1

        for start_idx in range(len(kmers) - self.w + 1):
            window = kmers[start_idx : start_idx + self.w]
            # Minimum by canonical kmer key value (or randomized hash)
            min_item = min(window, key=lambda item: item[0])

            if min_item[1] != last_minimizer_pos:
                minimizers.append(MinimizerSeed(
                    kmer_key=min_item[0],
                    pos=min_item[1],
                    is_rev_comp=min_item[2]
                ))
                last_minimizer_pos = min_item[1]

        return minimizers

    def index_reference(self, ref_id: str, sequence: str) -> int:
        """
        Indexes a reference genome or transcript sequence into the seed table.
        Returns the number of indexed minimizers.
        """
        self.indexed_references[ref_id] = len(sequence)
        minimizers = self.extract_minimizers(sequence)

        for m in minimizers:
            # Increment frequency in lock-free elastic hash table
            self.kmer_table.insert_or_increment(m.kmer_key, increment=1)

            if m.kmer_key not in self.seed_postings:
                self.seed_postings[m.kmer_key] = []
            self.seed_postings[m.kmer_key].append((ref_id, m.pos))

        return len(minimizers)

    def find_seed_hits(self, query_sequence: str, max_occurrence: int = 500) -> List[SeedHit]:
        """
        Finds exact seed matches for query minimizers against the reference index.
        Filters out highly repetitive k-mers (occurrence > max_occurrence).
        """
        q_minimizers = self.extract_minimizers(query_sequence)
        hits: List[SeedHit] = []

        for q_m in q_minimizers:
            count = self.kmer_table.query_count(q_m.kmer_key)
            if 0 < count <= max_occurrence and q_m.kmer_key in self.seed_postings:
                for ref_id, ref_pos in self.seed_postings[q_m.kmer_key]:
                    hits.append(SeedHit(
                        query_pos=q_m.pos,
                        ref_id=ref_id,
                        ref_pos=ref_pos,
                        kmer_len=self.k
                    ))

        return hits

    def chain_anchors(self, hits: List[SeedHit]) -> List[AlignmentChain]:
        """
        Performs dynamic programming anchor chaining (Minimap2 style).
        Connects co-linear seed matches (q_pos[i] < q_pos[j] and ref_pos[i] < ref_pos[j]).
        """
        if not hits:
            return []

        # Group hits by reference ID
        hits_by_ref: Dict[str, List[SeedHit]] = {}
        for h in hits:
            if h.ref_id not in hits_by_ref:
                hits_by_ref[h.ref_id] = []
            hits_by_ref[h.ref_id].append(h)

        chains: List[AlignmentChain] = []

        for ref_id, ref_hits in hits_by_ref.items():
            # Sort anchors primarily by reference position, secondarily by query position
            sorted_hits = sorted(ref_hits, key=lambda h: (h.ref_pos, h.query_pos))
            n = len(sorted_hits)
            dp_scores = np.zeros(n, dtype=np.int32)
            dp_parents = np.full(n, -1, dtype=np.int32)

            for i in range(n):
                hi = sorted_hits[i]
                dp_scores[i] = self.k  # Base score for single anchor

                # Look back at preceding anchors within max_chain_gap
                for j in range(max(0, i - 64), i):
                    hj = sorted_hits[j]
                    dq = hi.query_pos - hj.query_pos
                    dr = hi.ref_pos - hj.ref_pos

                    if 0 < dq <= self.max_chain_gap and 0 < dr <= self.max_chain_gap:
                        # Gap penalty proportional to difference in distances (insertion/deletion)
                        gap = abs(dq - dr)
                        gap_penalty = int(0.1 * gap + 0.5 * np.log2(max(1, gap)))
                        score_gain = min(dq, dr, self.k) - gap_penalty
                        cand_score = dp_scores[j] + score_gain

                        if cand_score > dp_scores[i]:
                            dp_scores[i] = cand_score
                            dp_parents[i] = j

            # Extract best chain
            best_idx = int(np.argmax(dp_scores))
            if dp_scores[best_idx] > 0:
                # Traceback
                chain_hits = []
                curr = best_idx
                while curr != -1:
                    chain_hits.append(sorted_hits[curr])
                    curr = dp_parents[curr]

                chain_hits.reverse()
                q_start = chain_hits[0].query_pos
                q_end = chain_hits[-1].query_pos + self.k
                r_start = chain_hits[0].ref_pos
                r_end = chain_hits[-1].ref_pos + self.k

                span_q = max(1, q_end - q_start)
                span_r = max(1, r_end - r_start)
                approx_id = float(np.clip(dp_scores[best_idx] / max(span_q, span_r), 0.1, 1.0))

                chains.append(AlignmentChain(
                    ref_id=ref_id,
                    query_start=q_start,
                    query_end=q_end,
                    ref_start=r_start,
                    ref_end=r_end,
                    chain_score=int(dp_scores[best_idx]),
                    num_anchors=len(chain_hits),
                    identity_approx=approx_id,
                    cigar_approx=f"{span_q}M"
                ))

        chains.sort(key=lambda c: c.chain_score, reverse=True)
        return chains

    def align_query(self, query_sequence: str) -> List[AlignmentChain]:
        """
        Executes end-to-end seed-and-extend search for a query read against the reference index.
        """
        hits = self.find_seed_hits(query_sequence)
        chains = self.chain_anchors(hits)
        return chains
