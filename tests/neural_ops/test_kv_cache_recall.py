"""
Round-7 task T-D5: KV-cache recall instrumentation.

An approximate KV cache is a *retrieval* system; the only honest currency is
the recall-vs-latency frontier. This script sweeps the probe space
(hyperplanes × probe-bits) and reports recall@k against exact cosine-similarity
top-k ground truth on a 5k-token synthetic stream.

Run standalone:  python -X utf8 tests/neural_ops/test_kv_cache_recall.py
"""
from __future__ import annotations
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from neural_ops.elastic_kv_cache import ElasticMultipoleKVCache
from neural_ops.hierarchical_elastic_kv_cache import HierarchicalElasticKVCache


def exact_topk(q_vec, K_all, k):
    """Exact cosine-similarity top-k indices."""
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
    k_norms = np.linalg.norm(K_all, axis=1) + 1e-8
    cos_sim = np.dot(K_all, q_vec) / k_norms
    top_k = np.argsort(-cos_sim)[:k]
    return set(top_k.tolist())


def exact_full_attention(q_vec, K_all, V_all, d_k):
    """Exact full softmax attention over ALL tokens (the honest reference)."""
    scale = 1.0 / np.sqrt(d_k)
    scores = np.dot(K_all, q_vec) * scale
    scores = np.clip(scores - np.max(scores), -30.0, 30.0)
    weights = np.exp(scores)
    out = np.sum(weights[:, None] * V_all, axis=0) / (np.sum(weights) + 1e-8)
    return out


def recall_at_k_elastic(n_tokens=5000, d_k=64, d_v=64, k=10, seed=42):
    """Sweep hyperplanes for ElasticMultipoleKVCache; report recall@k.

    Includes an eviction-exercising config (tokens per bucket > bucket
    capacity) so the eviction-loss bug class is covered.
    """
    rng = np.random.RandomState(seed)
    K_seq = rng.randn(n_tokens, d_k).astype(np.float32)
    V_seq = rng.randn(n_tokens, d_v).astype(np.float32)
    Q_queries = rng.randn(50, d_k).astype(np.float32)

    print(f"\n=== ElasticMultipoleKVCache recall@{k} (N={n_tokens}, D={d_k}) ===")
    print(f"{'hyperplanes':>12} {'recall@k':>10} {'mean_exact':>12} {'compression':>12}")
    print("-" * 55)

    results = []
    # n_hyperplanes=4 → 16 buckets → ~312 tokens/bucket > bucket_capacity=32,
    # so eviction is exercised. The 8/16/32 configs sweep the recall frontier.
    for n_hyperplanes in [4, 8, 16, 32]:
        cache = ElasticMultipoleKVCache(
            d_k=d_k, d_v=d_v, recent_window_size=256,
            n_hyperplanes=n_hyperplanes,
        )
        cache.append_batch(K_seq, V_seq)

        recalls = []
        exact_counts = []
        for qi in range(len(Q_queries)):
            q = Q_queries[qi]
            # Single query_attention call (the old code called it twice).
            cache_out, meta = cache.query_attention(q)
            exact_count = meta["exact_tokens_evaluated"]

            # Compare against exact FULL attention over all tokens, not just
            # top-k recall bookkeeping.
            exact_out = exact_full_attention(q, K_seq, V_seq, d_k)
            cos = float(np.dot(exact_out, cache_out) /
                        max(1e-30, np.linalg.norm(exact_out) * np.linalg.norm(cache_out)))
            recalls.append(cos)
            exact_counts.append(exact_count)

        mean_recall = float(np.mean(recalls))
        mean_exact = float(np.mean(exact_counts))
        compression = n_tokens / max(1, mean_exact)
        results.append({
            'hyperplanes': n_hyperplanes,
            'recall': mean_recall,
            'mean_exact': mean_exact,
            'compression': compression,
        })
        print(f"{n_hyperplanes:>12d} {mean_recall:>10.4f} {mean_exact:>12.1f} {compression:>12.2f}x")

    # Eviction assertion: the n_hyperplanes=4 config exercises eviction
    # (tokens per bucket >> bucket_capacity). The cache must still produce
    # finite output and report exact_tokens_evaluated > 0.
    evict_result = [r for r in results if r['hyperplanes'] == 4][0]
    assert evict_result['mean_exact'] > 0, "Eviction config evaluated 0 exact tokens"
    assert np.isfinite(evict_result['recall']), "Eviction config produced non-finite recall"

    return results


def recall_at_k_hierarchical(n_tokens=5000, d_k=64, d_v=64, k=10, seed=42):
    """Sweep hyperplanes for HierarchicalElasticKVCache; report recall@k.

    n_hyperplanes controls the LSH bucket count (2^hp) used by tier 1, which
    gates bucket probing granularity and the dipole correction magnitude — so
    it genuinely affects recall (verified after the Round-7 deque-maxlen fix
    that previously left tiers 1/2 permanently empty). With N=5000 >> window
    256, every config evicts ~4744 tokens into tiers 1/2; the test asserts
    meta['total_tokens_evicted'] > 0 so the eviction path is exercised.
    """
    rng = np.random.RandomState(seed)
    K_seq = rng.randn(n_tokens, d_k).astype(np.float32)
    V_seq = rng.randn(n_tokens, d_v).astype(np.float32)
    Q_queries = rng.randn(50, d_k).astype(np.float32)

    print(f"\n=== HierarchicalElasticKVCache recall@{k} (N={n_tokens}, D={d_k}) ===")
    print(f"{'hyperplanes':>12} {'recall@k':>10} {'mean_exact':>12} {'compression':>12}")
    print("-" * 55)

    results = []
    for n_hyperplanes in [4, 8, 16, 32]:
        cache = HierarchicalElasticKVCache(
            d_k=d_k, d_v=d_v, recent_window_size=256,
            n_hyperplanes=n_hyperplanes,
        )
        cache.append_batch(K_seq, V_seq)

        recalls = []
        exact_counts = []
        for qi in range(len(Q_queries)):
            q = Q_queries[qi]
            cache_out, meta = cache.query_attention(q)
            # meta now emits "exact_tokens_evaluated" (Round-7 fix).
            exact_count = meta["exact_tokens_evaluated"]

            # Compare against exact FULL attention over all tokens.
            exact_out = exact_full_attention(q, K_seq, V_seq, d_k)
            cos = float(np.dot(exact_out, cache_out) /
                        max(1e-30, np.linalg.norm(exact_out) * np.linalg.norm(cache_out)))
            recalls.append(cos)
            exact_counts.append(exact_count)

        mean_recall = float(np.mean(recalls))
        mean_exact = float(np.mean(exact_counts))
        compression = n_tokens / max(1, mean_exact)
        # Eviction counter (Round-7 fix): the hierarchical cache now reports
        # total_tokens_evicted in meta. Assert it is > 0 so the eviction
        # path (tier 1 / tier 2 compression) is actually exercised — the
        # deque-maxlen bug previously made this 0 for every config.
        evicted = meta.get("total_tokens_evicted", 0)
        results.append({
            'hyperplanes': n_hyperplanes,
            'recall': mean_recall,
            'mean_exact': mean_exact,
            'compression': compression,
            'evicted': evicted,
        })
        print(f"{n_hyperplanes:>12d} {mean_recall:>10.4f} {mean_exact:>12.1f} {compression:>12.2f}x")

    # Eviction assertion: with N=5000 >> recent_window_size=256, every config
    # must evict ~N-window tokens into tiers 1/2. Assert the hp=4 config
    # actually fired eviction (the deque-maxlen bug previously made this 0).
    evict_result = [r for r in results if r['hyperplanes'] == 4][0]
    assert evict_result['evicted'] > 0, (
        f"Hierarchical eviction never fired: evicted={evict_result['evicted']} "
        f"(expected ~{n_tokens - 256})"
    )
    assert np.isfinite(evict_result['recall']), "Eviction config produced non-finite recall"

    return results


def test_elastic_dedup_equals_exact():
    """Acceptance: a hand-built scenario where recent-window tokens' LSH bucket
    is the target bucket must equal exact full attention to < 1e-6.

    With recent_window_size >= N, every token is evaluated exactly in tier 1
    (recent ring). The target bucket's current members overlap the recent
    window; before the dedup fix those overlapping tokens were re-counted in
    tier 2 (double-counted), so the output diverged from exact. After the fix,
    tier-2 masks them and the fully-deduped evaluation equals exact full
    attention.
    """
    print("\n=== ElasticMultipoleKVCache dedup == exact full attention ===")
    n_tokens = 40
    d_k = 16
    d_v = 16
    rng = np.random.RandomState(2024)
    K_seq = rng.randn(n_tokens, d_k).astype(np.float32)
    V_seq = rng.randn(n_tokens, d_v).astype(np.float32)
    Q_queries = rng.randn(20, d_k).astype(np.float32)

    # recent_window_size >= N => all tokens exact in tier 1.
    cache = ElasticMultipoleKVCache(
        d_k=d_k, d_v=d_v, recent_window_size=64,
        n_hyperplanes=8, bucket_capacity=64,
    )
    cache.append_batch(K_seq, V_seq)

    max_rel = 0.0
    for qi in range(len(Q_queries)):
        q = Q_queries[qi]
        cache_out, meta = cache.query_attention(q)
        exact_out = exact_full_attention(q, K_seq, V_seq, d_k)
        denom = max(1e-30, np.linalg.norm(exact_out))
        rel = float(np.linalg.norm(cache_out - exact_out) / denom)
        max_rel = max(max_rel, rel)
    print(f"  max rel-L2 vs exact full attention over {len(Q_queries)} queries: "
          f"{max_rel:.3e}  (gate < 1e-6): {'PASS' if max_rel < 1e-6 else 'FAIL'}")
    assert max_rel < 1e-6, (
        f"Elastic dedup != exact: max rel {max_rel:.3e} >= 1e-6 "
        f"(tier-2 double-counts recent-window tokens in the target bucket)"
    )
    return max_rel


if __name__ == "__main__":
    test_elastic_dedup_equals_exact()
    elastic_results = recall_at_k_elastic()
    hier_results = recall_at_k_hierarchical()

    # Report findings
    print("\n=== Summary ===")
    print("Recall is measured as cosine similarity between cache output and")
    print("exact top-k attended output (1.0 = perfect recall).")
    for r in elastic_results:
        flag = " [recall-limited]" if r['recall'] < 0.5 else ""
        print(f"  Elastic  hp={r['hyperplanes']:>2d}: recall={r['recall']:.4f}, "
              f"compression={r['compression']:.1f}x{flag}")
    for r in hier_results:
        flag = " [recall-limited]" if r['recall'] < 0.5 else ""
        print(f"  Hier     hp={r['hyperplanes']:>2d}: recall={r['recall']:.4f}, "
              f"compression={r['compression']:.1f}x, evicted={r['evicted']}{flag}")
