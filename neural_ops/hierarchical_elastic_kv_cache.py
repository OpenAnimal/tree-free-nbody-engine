"""
Multi-Scale Hierarchical Elastic KV-Cache (`hierarchical_elastic_kv_cache.py`)
=============================================================================
Hierarchical Multi-Resolution Streaming KV-Cache for 100k-1M+ Token LLM Inference.
Combines 2025 Non-Reordering Elastic Open Addressing with Multi-Level Multipole Hierarchy.

3-Tier Memory Architecture:
  Level 0: Exact Recent Sliding Window (Uncompressed full-rank tokens)
  Level 1: Semantic LSH Elastic Hash Clusters (Centroid + Dipole moments)
  Level 2: Multi-Scale Global Multipole Pyramid (Coarse global summary)
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, Any, List


class HierarchicalElasticKVCache:
    """
    Multi-Scale Hierarchical KV-Cache.
    Provides O(1) streaming token append and O(K) decoding attention retrieval.
    """
    def __init__(
        self,
        d_k: int = 64,
        d_v: int = 64,
        recent_window_size: int = 256,
        n_hyperplanes: int = 8,
        n_coarse_levels: int = 3,
        temperature: Optional[float] = None,
    ):
        self.d_k = d_k
        self.d_v = d_v
        self.recent_window_size = recent_window_size
        self.n_hyperplanes = n_hyperplanes
        self.n_coarse_levels = n_coarse_levels
        # temperature=0.0 must survive (falsy `or` clobbers it).
        self.temperature = (1.0 / np.sqrt(d_k)) if temperature is None else float(temperature)

        # Random hyperplane projections for Locality Sensitive Hashing (LSH)
        rng = np.random.RandomState(42)
        self.hyperplanes = rng.randn(n_hyperplanes, d_k).astype(np.float32)
        self.hyperplanes /= np.linalg.norm(self.hyperplanes, axis=-1, keepdims=True)

        # Tier 0: Recent Sliding Window Buffer (deque for O(1) popleft).
        # NOTE: no maxlen — the eviction in append_token manually poplefts when
        # len > recent_window_size so the oldest token is compressed into
        # tier 1 / tier 2. A maxlen here would silently cap the deque and the
        # `len > recent_window_size` branch would NEVER fire (the deque drops
        # the oldest element on append before len can exceed the cap), leaving
        # tiers 1 and 2 permanently empty — the Round-7 retrieval bug.
        self.recent_k: deque = deque()
        self.recent_v: deque = deque()

        # Tier 1: Elastic Semantic LSH Hash Buckets
        # bucket_id -> { 'k_sum': (d_k,), 'v_sum': (d_v,), 'count': int, 'keys': [(d_k,)], 'vals': [(d_v,)] }
        self.lsh_buckets: Dict[int, Dict[str, Any]] = {}

        # Tier 2: Coarse Multipole Moments
        self.coarse_k_moments = np.zeros((n_coarse_levels, d_k), dtype=np.float32)
        self.coarse_v_moments = np.zeros((n_coarse_levels, d_v), dtype=np.float32)
        self.coarse_counts = np.zeros(n_coarse_levels, dtype=np.float32)

        self.total_tokens_appended = 0
        # Number of tokens compressed out of the recent window into tiers 1/2.
        # Exposed in query meta so tests can assert eviction actually fired.
        self.total_tokens_evicted = 0

    def _hash_key(self, key_vec: np.ndarray) -> int:
        """Projects key vector to discrete LSH bucket integer."""
        proj = self.hyperplanes @ key_vec
        bits = (proj > 0).astype(np.int32)
        return int(np.dot(bits, 1 << np.arange(self.n_hyperplanes)))

    def append_token(self, k: np.ndarray, v: np.ndarray) -> None:
        """
        Appends a single token (k, v) to the hierarchical cache in O(1) time.
        """
        k = k.astype(np.float32)
        v = v.astype(np.float32)

        # Add to recent sliding window
        self.recent_k.append(k)
        self.recent_v.append(v)
        self.total_tokens_appended += 1

        # When sliding window exceeds capacity, compress oldest token into Tier 1 & Tier 2
        if len(self.recent_k) > self.recent_window_size:
            evicted_k = self.recent_k.popleft()
            evicted_v = self.recent_v.popleft()
            self.total_tokens_evicted += 1

            # Tier 1: Semantic LSH cluster
            bucket_id = self._hash_key(evicted_k)
            if bucket_id not in self.lsh_buckets:
                self.lsh_buckets[bucket_id] = {
                    'k_sum': evicted_k.copy(),
                    'v_sum': evicted_v.copy(),
                    'count': 1,
                    'k_mean': evicted_k.copy(),
                    'dipole': np.zeros((self.d_v, self.d_k), dtype=np.float32),
                    # Per-level contributions for tier-2 dedup at query time.
                    'level_k': {}, 'level_v': {}, 'level_c': {},
                }
            else:
                b = self.lsh_buckets[bucket_id]
                old_mean = b['k_mean'].copy()
                old_v_sum = b['v_sum'].copy()  # Σ v_j before this token
                b['k_sum'] += evicted_k
                b['v_sum'] += evicted_v
                b['count'] += 1
                b['k_mean'] = b['k_sum'] / b['count']
                # Incremental dipole with mean-shift correction:
                # S_{n+1} = S_n + v(x)(k - mu_{n+1}) + (Σ v_j)(mu_n - mu_{n+1})
                delta_k = evicted_k - b['k_mean']
                b['dipole'] += np.outer(evicted_v, delta_k) + np.outer(old_v_sum, old_mean - b['k_mean'])

            # Tier 2: Coarse Global Multipole Pyramid
            # Assign into dyadic level
            level = min(self.n_coarse_levels - 1, int(np.log2(max(1, self.total_tokens_appended // self.recent_window_size))))
            self.coarse_k_moments[level] += evicted_k
            self.coarse_v_moments[level] += evicted_v
            self.coarse_counts[level] += 1
            # Track this bucket's per-level contribution for tier-2 dedup.
            b = self.lsh_buckets[bucket_id]
            if level not in b['level_k']:
                b['level_k'][level] = evicted_k.copy()
                b['level_v'][level] = evicted_v.copy()
                b['level_c'][level] = 1
            else:
                b['level_k'][level] += evicted_k
                b['level_v'][level] += evicted_v
                b['level_c'][level] += 1

    def append_batch(self, K: np.ndarray, V: np.ndarray) -> None:
        """Appends a batch of tokens during prompt prefill."""
        for i in range(len(K)):
            self.append_token(K[i], V[i])

    def query_attention(self, query: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Decodes attention output for a single query token across all 3 tiers.
        """
        q = query.astype(np.float32)

        # 1. Tier 0: Exact Recent Sliding Window Attention
        if len(self.recent_k) > 0:
            K_recent = np.stack(self.recent_k, axis=0) # (W, d_k)
            V_recent = np.stack(self.recent_v, axis=0) # (W, d_v)

            dot_recent = (K_recent @ q) * self.temperature # (W,)
            w_recent = np.exp(np.clip(dot_recent, -30.0, 30.0))
            out_recent = np.sum(w_recent[:, None] * V_recent, axis=0)
            weight_recent = np.sum(w_recent)
        else:
            out_recent = np.zeros(self.d_v, dtype=np.float32)
            weight_recent = 0.0

        # 2. Tier 1: Semantic LSH Multipole Hash Clusters
        q_bucket = self._hash_key(q)
        out_tier1 = np.zeros(self.d_v, dtype=np.float32)
        weight_tier1 = 0.0
        probed_bucket_ids = set()

        # Probe candidate buckets (exact match + 1-bit Hamming neighbors)
        probed_buckets = [q_bucket]
        for h in range(self.n_hyperplanes):
            probed_buckets.append(q_bucket ^ (1 << h))

        for b_id in probed_buckets:
            if b_id in self.lsh_buckets:
                b = self.lsh_buckets[b_id]
                k_mean = b['k_mean']
                count = b['count']
                v_sum = b['v_sum']

                dot_b = np.dot(q, k_mean) * self.temperature
                w_b = np.exp(np.clip(dot_b, -30.0, 30.0))

                # Zero order + dipole correction
                val_0 = w_b * v_sum
                # Dipole correction: w_b * (dipole @ q * temperature)
                dip_corr = w_b * (b['dipole'] @ q) * self.temperature

                out_tier1 += val_0 + dip_corr
                weight_tier1 += w_b * count
                probed_bucket_ids.add(b_id)

        # 3. Tier 2: Coarse Global Multipole Summary
        # Dedup: subtract probed buckets' per-level contributions from tier-2
        # sums so evicted tokens evaluated in tier 1 are not double-counted.
        out_tier2 = np.zeros(self.d_v, dtype=np.float32)
        weight_tier2 = 0.0

        for lvl in range(self.n_coarse_levels):
            if self.coarse_counts[lvl] > 0:
                k_sum_lvl = self.coarse_k_moments[lvl].copy()
                v_sum_lvl = self.coarse_v_moments[lvl].copy()
                count_lvl = float(self.coarse_counts[lvl])
                # Subtract probed buckets' contributions at this level.
                for b_id in probed_bucket_ids:
                    b = self.lsh_buckets[b_id]
                    if lvl in b['level_k']:
                        k_sum_lvl -= b['level_k'][lvl]
                        v_sum_lvl -= b['level_v'][lvl]
                        count_lvl -= b['level_c'][lvl]
                if count_lvl <= 0:
                    continue
                k_mean_lvl = k_sum_lvl / count_lvl
                dot_lvl = np.dot(q, k_mean_lvl) * self.temperature
                w_lvl = np.exp(np.clip(dot_lvl, -30.0, 30.0))

                out_tier2 += w_lvl * v_sum_lvl
                weight_tier2 += w_lvl * count_lvl

        total_out = out_recent + out_tier1 + out_tier2
        total_weight = weight_recent + weight_tier1 + weight_tier2 + 1e-12

        final_v = total_out / total_weight

        # Compute compression ratio
        active_cache_entries = len(self.recent_k) + len(self.lsh_buckets) + self.n_coarse_levels
        compression_ratio = self.total_tokens_appended / max(1, active_cache_entries)

        meta = {
            "total_tokens": self.total_tokens_appended,
            "exact_tokens_evaluated": len(self.recent_k),
            "recent_window_size": len(self.recent_k),
            "lsh_active_buckets": len(self.lsh_buckets),
            "total_tokens_evicted": self.total_tokens_evicted,
            "compression_ratio": float(compression_ratio),
        }
        return final_v, meta
