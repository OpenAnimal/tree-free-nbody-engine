"""Shared 3-tier KV-cache helpers (`neural_ops/_kv_tiers.py`).

Factored out of ``elastic_kv_cache.py`` and
``hierarchical_elastic_kv_cache.py`` (claim-hygiene review): both caches
key tokens with random-hyperplane SimHash and accumulate attention tiers
as clipped-exponential weighted sums. The two caches differ in what the
tiers CONTAIN (dict-of-lists with bucket compression vs deque + dipole +
coarse pyramid); these helpers are the genuinely shared parts.

Nothing here is the funnel/elastic open-addressing construction of
Farach-Colton, Krapivin, & Kuszmaul (2025) — see each cache's module
docstring. The faithful funnel implementation is ``core/elastic_hash.py``.
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np


class HyperplaneLSH:
    """Random-hyperplane SimHash keying shared by the KV caches.

    ``key(vec)`` -> int bucket id in [0, 2**n_hyperplanes).
    """

    def __init__(self, n_hyperplanes: int, d_k: int, seed: int,
                 normalize: bool = False):
        rng = np.random.RandomState(seed)
        hp = rng.randn(n_hyperplanes, d_k).astype(np.float32)
        hp /= np.linalg.norm(hp, axis=-1, keepdims=True)
        self.hyperplanes = hp          # (n_hyperplanes, d_k), rows normalized
        self.normalize = normalize

    def key(self, vec: np.ndarray) -> int:
        v = vec / (np.linalg.norm(vec) + 1e-8) if self.normalize else vec
        bits = (self.hyperplanes @ v) > 0
        return int(np.sum(bits * (1 << np.arange(len(bits),
                                                dtype=np.int64))))


def accumulate_tier(acc_val: np.ndarray, acc_weight: float,
                    scores: np.ndarray, values: np.ndarray,
                    temperature: float) -> tuple:
    """One exact tier: out += Σ softmax-ish(w)_i · V_i (no renormalization).

    ``scores`` are pre-scaled dot products; the clip bounds match both
    caches' historical behavior (logits beyond ±30 saturate).
    """
    w = np.exp(np.clip(scores, -30.0, 30.0))
    acc_val = acc_val + np.sum(w[:, None] * values, axis=0)
    acc_weight = acc_weight + float(np.sum(w))
    return acc_val, acc_weight


def accumulate_cluster(acc_val: np.ndarray, acc_weight: float,
                       score: float, v_sum: np.ndarray,
                       count: float) -> tuple:
    """One far cluster: centroid-scored, count-weighted bucket summary."""
    w = np.exp(np.clip(score, -30.0, 30.0))
    acc_val = acc_val + w * v_sum
    acc_weight = acc_weight + w * count
    return acc_val, acc_weight


def tier_meta(exact: int, entries: int, total: int, **extra) -> Dict[str, Any]:
    """Common meta block: compression ratio of exact vs stored entries.

    ``total_tokens`` is kept as an alias of ``total_tokens_in_history``
    (the hierarchical cache's historical key).
    """
    meta = {
        "total_tokens_in_history": total,
        "total_tokens": total,
        "exact_tokens_evaluated": exact,
        "compression_ratio": float(total) / max(1, entries),
    }
    meta.update(extra)
    return meta
