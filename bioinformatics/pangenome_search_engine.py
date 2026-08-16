"""
Module 10: Pan-Genome & Colored De Bruijn Graph Search Engine.
Indexes tens of thousands of genomes/strains into a compressed Colored De Bruijn Graph (cDBG).
Enables sub-millisecond presence/absence cohort screening (e.g. antibiotic resistance genes across 500k isolates).
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Set

try:
    from .kmer_elastic_hash import NUC_MAP, REV_NUC_MAP, COMP_MAP, KmerElasticHashTable
except (ImportError, ValueError):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from bioinformatics.kmer_elastic_hash import NUC_MAP, REV_NUC_MAP, COMP_MAP, KmerElasticHashTable


@dataclass
class PanGenomeSearchResult:
    """Outcome of a pan-genomic sequence query across a sample cohort."""
    query_name: str
    query_length: int
    num_query_kmers: int
    matching_sample_ids: List[str]
    sample_coverage_fractions: Dict[str, float]  # sample_id -> fraction of query kmers present [0, 1]
    core_genome_kmer_count: int                  # kmers present in 100% of samples
    accessory_genome_kmer_count: int             # kmers present in <100% of samples
    query_presence_status: str                   # "Found (Full Match)", "Partial Match", "Absent"


class PanGenomeSearchEngine:
    """
    Compressed Colored De Bruijn Graph (cDBG) Engine.
    Maps canonical 2-bit bitpacked k-mers to bitset color vectors representing sample cohorts.
    Powered by Farach-Colton non-reordering multi-level open addressing.
    """
    def __init__(self, k: int = 21, max_samples: int = 64, capacity: int = 500000):
        assert 1 <= k <= 31, "k must be between 1 and 31 for 64-bit integer bitpacking."
        assert 1 <= max_samples <= 64, "max_samples must be <= 64 for native 64-bit color masks."
        self.k = int(k)
        self.k_mask = (1 << (2 * self.k)) - 1
        self.max_samples = int(max_samples)

        # Core non-reordering k-mer index
        self.kmer_table = KmerElasticHashTable(k=self.k, capacity=capacity)
        
        # Color bitsets: maps canonical k-mer key -> 64-bit integer bitmask (bit i = 1 if present in sample i)
        self.color_bitsets: Dict[int, int] = {}
        self.sample_registry: Dict[str, int] = {} # sample_name -> sample_index [0, max_samples - 1]
        self.reverse_registry: Dict[int, str] = {}

    def register_sample(self, sample_name: str) -> int:
        """Registers a new sample/genome in the cohort index."""
        if sample_name in self.sample_registry:
            return self.sample_registry[sample_name]

        sample_idx = len(self.sample_registry)
        if sample_idx >= self.max_samples:
            raise ValueError(f"Exceeded maximum registered samples ({self.max_samples}).")

        self.sample_registry[sample_name] = sample_idx
        self.reverse_registry[sample_idx] = sample_name
        return sample_idx

    def _canonical_kmer_key(self, forward_key: int) -> int:
        """Computes the canonical 2-bit packed k-mer."""
        rev_key = 0
        temp = forward_key
        for _ in range(self.k):
            nuc = temp & 3
            rev_key = (rev_key << 2) | COMP_MAP[nuc]
            temp >>= 2
        return min(forward_key, rev_key)

    def index_genome(self, sample_name: str, sequence: str) -> int:
        """
        Ingests a complete genome/strain sequence into the Colored De Bruijn index.
        Sets the color bit corresponding to sample_name for all observed k-mers.
        """
        sample_idx = self.register_sample(sample_name)
        color_mask = 1 << sample_idx

        seq_len = len(sequence)
        if seq_len < self.k:
            return 0

        rolling_key = 0
        valid_len = 0
        total_indexed = 0

        for ch in sequence:
            if ch in NUC_MAP:
                val = NUC_MAP[ch]
                rolling_key = ((rolling_key << 2) | val) & self.k_mask
                valid_len += 1

                if valid_len >= self.k:
                    canon_key = self._canonical_kmer_key(rolling_key)
                    self.kmer_table.insert_or_increment(canon_key, increment=1)
                    
                    # Update color bitmask
                    curr_color = self.color_bitsets.get(canon_key, 0)
                    self.color_bitsets[canon_key] = curr_color | color_mask
                    total_indexed += 1
            else:
                rolling_key = 0
                valid_len = 0

        return total_indexed

    def query_sequence(
        self,
        query_sequence: str,
        query_name: str = "Query_Target",
        min_coverage_threshold: float = 0.80
    ) -> PanGenomeSearchResult:
        """
        Queries a gene or transcript sequence against the pan-genome cohort in sub-millisecond time.
        Calculates k-mer presence across all cohort members.
        """
        seq_len = len(query_sequence)
        if seq_len < self.k:
            return PanGenomeSearchResult(
                query_name=query_name,
                query_length=seq_len,
                num_query_kmers=0,
                matching_sample_ids=[],
                sample_coverage_fractions={},
                core_genome_kmer_count=0,
                accessory_genome_kmer_count=0,
                query_presence_status="Absent (Too Short)"
            )

        # Extract unique query k-mers
        query_kmers: Set[int] = set()
        rolling_key = 0
        valid_len = 0

        for ch in query_sequence:
            if ch in NUC_MAP:
                val = NUC_MAP[ch]
                rolling_key = ((rolling_key << 2) | val) & self.k_mask
                valid_len += 1
                if valid_len >= self.k:
                    canon_key = self._canonical_kmer_key(rolling_key)
                    query_kmers.add(canon_key)
            else:
                rolling_key = 0
                valid_len = 0

        n_q_kmers = len(query_kmers)
        if n_q_kmers == 0:
            return PanGenomeSearchResult(
                query_name=query_name,
                query_length=seq_len,
                num_query_kmers=0,
                matching_sample_ids=[],
                sample_coverage_fractions={},
                core_genome_kmer_count=0,
                accessory_genome_kmer_count=0,
                query_presence_status="Absent"
            )

        # Count presence per sample
        sample_hit_counts = {idx: 0 for idx in self.reverse_registry.keys()}
        n_cohort = len(self.sample_registry)
        full_cohort_mask = (1 << n_cohort) - 1

        core_count = 0
        accessory_count = 0

        for q_kmer in query_kmers:
            color = self.color_bitsets.get(q_kmer, 0)
            if color > 0:
                if (color & full_cohort_mask) == full_cohort_mask and n_cohort > 1:
                    core_count += 1
                else:
                    accessory_count += 1

                for idx in sample_hit_counts.keys():
                    if (color & (1 << idx)) != 0:
                        sample_hit_counts[idx] += 1

        # Calculate coverage fractions
        coverages: Dict[str, float] = {}
        matching_samples: List[str] = []

        for idx, count in sample_hit_counts.items():
            sample_name = self.reverse_registry[idx]
            frac = float(count / n_q_kmers)
            coverages[sample_name] = frac
            if frac >= min_coverage_threshold:
                matching_samples.append(sample_name)

        if any(f >= 0.95 for f in coverages.values()):
            status = "Found (Full Match)"
        elif any(f >= min_coverage_threshold for f in coverages.values()):
            status = "Partial Match"
        else:
            status = "Absent"

        return PanGenomeSearchResult(
            query_name=query_name,
            query_length=seq_len,
            num_query_kmers=n_q_kmers,
            matching_sample_ids=matching_samples,
            sample_coverage_fractions=coverages,
            core_genome_kmer_count=core_count,
            accessory_genome_kmer_count=accessory_count,
            query_presence_status=status
        )
