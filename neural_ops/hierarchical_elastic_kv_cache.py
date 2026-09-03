"""
Multi-Scale Hierarchical KV-Cache (`hierarchical_elastic_kv_cache.py`)
=============================================================================
What this IS (honest description): a 3-tier NumPy streaming KV-cache —
  Level 0: exact recent sliding window (deque, uncompressed tokens)
  Level 1: per-LSH-bucket centroid + dipole summaries of evicted tokens
  Level 2: a small dyadic pyramid of global key/value sums

What this is NOT: the funnel/elastic open-addressing construction of
Farach-Colton, Krapivin, & Kuszmaul (2025) — the bucket index is a plain
Python dict keyed by the SimHash signature, with no funnel slab layout, no
ordered overflow region and no worst-case probe bounds. The faithful
funnel implementation is ``core/elastic_hash.py``; this module does not
import it. Shared LSH-keying and tier-accumulation helpers live in
``neural_ops/_kv_tiers.py``.
"""
from collections import deque

import numpy as np
from typing import Optional, Tuple, Dict, Any

from ._kv_tiers import (
    HyperplaneLSH, accumulate_tier, accumulate_cluster, tier_meta,
)


class HierarchicalElasticKVCache:
    """Multi-scale hierarchical KV-cache: O(1) appends, tiered decoding."""

    def __init__(self, d_k: int = 64, d_v: int = 64,
                 recent_window_size: int = 256, n_hyperplanes: int = 8,
                 n_coarse_levels: int = 3, temperature: Optional[float] = None):
        self.d_k, self.d_v = d_k, d_v
        self.recent_window_size = recent_window_size
        self.n_coarse_levels = n_coarse_levels
        # temperature=0.0 must survive (falsy `or` clobbers it).
        self.temperature = ((1.0 / np.sqrt(d_k)) if temperature is None
                            else float(temperature))
        self._lsh = HyperplaneLSH(n_hyperplanes, d_k, seed=42)
        # Tier 0: no maxlen — eviction in append_token poplefts when
        # len > recent_window_size so the oldest token is compressed into
        # tiers 1/2 (a maxlen would silently cap the deque and tiers 1/2
        # would stay permanently empty — the Round-7 retrieval bug).
        self.recent_k: deque = deque()
        self.recent_v: deque = deque()
        # Tier 1: bucket_id -> centroid/dipole summary + per-level split.
        self.lsh_buckets: Dict[int, Dict[str, Any]] = {}
        # Tier 2: dyadic coarse pyramid.
        self.coarse_k = np.zeros((n_coarse_levels, d_k), dtype=np.float32)
        self.coarse_v = np.zeros((n_coarse_levels, d_v), dtype=np.float32)
        self.coarse_counts = np.zeros(n_coarse_levels, dtype=np.float32)
        self.total_tokens_appended = 0
        self.total_tokens_evicted = 0

    def append_token(self, k: np.ndarray, v: np.ndarray) -> None:
        """Append one (k, v) token in O(1); evict+compress on overflow."""
        k, v = k.astype(np.float32), v.astype(np.float32)
        self.recent_k.append(k)
        self.recent_v.append(v)
        self.total_tokens_appended += 1
        if len(self.recent_k) <= self.recent_window_size:
            return
        ek, ev = self.recent_k.popleft(), self.recent_v.popleft()
        self.total_tokens_evicted += 1

        b_id = self._lsh.key(ek)
        b = self.lsh_buckets.get(b_id)
        if b is None:
            self.lsh_buckets[b_id] = b = {
                'k_sum': np.zeros(self.d_k, dtype=np.float32),
                'v_sum': np.zeros(self.d_v, dtype=np.float32),
                'count': 0, 'k_mean': np.zeros(self.d_k, dtype=np.float32),
                'dipole': np.zeros((self.d_v, self.d_k), dtype=np.float32),
                'level_k': {}, 'level_v': {}, 'level_c': {},
            }
        old_mean, old_v_sum = b['k_mean'].copy(), b['v_sum'].copy()
        b['k_sum'] += ek
        b['v_sum'] += ev
        b['count'] += 1
        b['k_mean'] = b['k_sum'] / b['count']
        # Incremental dipole with mean-shift correction:
        # S_{n+1} = S_n + v(x)(k - mu_{n+1}) + (Σ v_j)(mu_n - mu_{n+1})
        b['dipole'] += (np.outer(ev, ek - b['k_mean'])
                        + np.outer(old_v_sum, old_mean - b['k_mean']))

        # Tier 2: dyadic level assignment + per-level dedup bookkeeping.
        level = min(self.n_coarse_levels - 1, int(np.log2(max(
            1, self.total_tokens_appended // self.recent_window_size))))
        self.coarse_k[level] += ek
        self.coarse_v[level] += ev
        self.coarse_counts[level] += 1
        b['level_k'][level] = b['level_k'].get(level, 0) + ek
        b['level_v'][level] = b['level_v'].get(level, 0) + ev
        b['level_c'][level] = b['level_c'].get(level, 0) + 1

    def append_batch(self, K: np.ndarray, V: np.ndarray) -> None:
        """Append a prompt-prefill batch of tokens."""
        for i in range(len(K)):
            self.append_token(K[i], V[i])

    def query_attention(self, query: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Decode attention for one query across all three tiers."""
        q = query.astype(np.float32)
        out, weight = np.zeros(self.d_v, dtype=np.float32), 1e-12
        # 1. Tier 0: exact recent window.
        if len(self.recent_k) > 0:
            K_r = np.stack(self.recent_k)
            V_r = np.stack(self.recent_v)
            out, weight = accumulate_tier(
                out, weight, (K_r @ q) * self.temperature, V_r,
                self.temperature)
        # 2. Tier 1: target bucket + 1-bit Hamming neighbours
        #    (centroid + dipole correction).
        q_bucket = self._lsh.key(q)
        probed = [q_bucket] + [q_bucket ^ (1 << h)
                               for h in range(len(self._lsh.hyperplanes))]
        probed_ids = set()
        for b_id in probed:
            b = self.lsh_buckets.get(b_id)
            if b is None:
                continue
            dot = float(np.dot(q, b['k_mean'])) * self.temperature
            val = (np.exp(np.clip(dot, -30.0, 30.0)) * b['v_sum']
                   + np.exp(np.clip(dot, -30.0, 30.0))
                   * (b['dipole'] @ q) * self.temperature)
            out = out + val
            weight += np.exp(np.clip(dot, -30.0, 30.0)) * b['count']
            probed_ids.add(b_id)
        # 3. Tier 2: coarse pyramid, minus probed buckets' contributions
        #    (evicted tokens already evaluated in tier 1 are not
        #    double-counted).
        for lvl in range(self.n_coarse_levels):
            if self.coarse_counts[lvl] <= 0:
                continue
            k_l = self.coarse_k[lvl].copy()
            v_l = self.coarse_v[lvl].copy()
            c_l = float(self.coarse_counts[lvl])
            for b_id in probed_ids:
                b = self.lsh_buckets[b_id]
                if lvl in b['level_k']:
                    k_l -= b['level_k'][lvl]
                    v_l -= b['level_v'][lvl]
                    c_l -= b['level_c'][lvl]
            if c_l <= 0:
                continue
            out, weight = accumulate_cluster(
                out, weight, float(np.dot(q, k_l / c_l)) * self.temperature,
                v_l, c_l)
        active = len(self.recent_k) + len(self.lsh_buckets) + self.n_coarse_levels
        return out / weight, tier_meta(
            exact=len(self.recent_k), entries=active,
            total=self.total_tokens_appended,
            recent_window_size=len(self.recent_k),
            lsh_active_buckets=len(self.lsh_buckets),
            total_tokens_evicted=self.total_tokens_evicted)
