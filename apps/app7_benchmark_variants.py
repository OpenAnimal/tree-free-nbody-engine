"""Standardized variant benchmark for Application 7 (high-dim LSH partition).

Variants:
  standard      -- brute-force exact top-k cosine-similarity retrieval
                   (scan all N vectors: O(N*d) per query -- the natural
                   reference for the app's retrieval task)
  +elastichash  -- the app's compute path: random hyperplane LSH bitmask ->
                   Farach-Colton funnel-hash bucket -> top-k over the bucket
                   candidates only

The +fmm axis is OMITTED with reason: the task is high-dimensional cosine
retrieval, not a 2D/3D kernel sum.

Accuracy semantics: LSH retrieval is approximate. The correctness metric is
`recall@k` (fraction of the true top-k that appear in the LSH candidate
set's top-k), reported in the note -- not a rel-L2 against an exact array.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.benchmark_kit import VariantBenchmark


def _embeddings(n_embeddings: int = 5000, d_dim: int = 64, seed: int = 42):
    np.random.seed(seed)
    centers = np.random.randn(5, d_dim)
    labels = np.random.choice(5, size=n_embeddings)
    emb = centers[labels] + np.random.normal(0, 0.4, size=(n_embeddings, d_dim))
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return emb, centers


def _brute_topk(emb, query, k=5):
    sims = emb @ query
    return np.argsort(sims)[::-1][:k]


def _lsh_topk(emb, query, hyperplanes, powers_of_two, ht, bucket_map, k=5):
    q_proj = (query @ hyperplanes) > 0
    q_key = int(np.sum(q_proj * powers_of_two))
    cand_ids, _ = ht.lookup(q_key)
    if cand_ids is None or len(cand_ids) == 0:
        return np.array([], dtype=np.int64)
    sims = emb[cand_ids] @ query
    top = np.argsort(sims)[::-1][:min(k, len(sims))]
    return np.array(cand_ids)[top]


def run_app7_variants(n_embeddings: int = 5000, d_dim: int = 64, n_hyperplanes: int = 12,
                      n_queries: int = 50, k=5):
    from core.elastic_hash import ElasticHashTable
    emb, centers = _embeddings(n_embeddings=n_embeddings, d_dim=d_dim)
    np.random.seed(42)
    hyperplanes = np.random.randn(d_dim, n_hyperplanes)
    powers_of_two = 1 << np.arange(n_hyperplanes, dtype=np.int64)
    proj = (emb @ hyperplanes) > 0
    lsh_keys = np.sum(proj * powers_of_two[None, :], axis=1)

    ht = ElasticHashTable(capacity=1 << (n_hyperplanes + 1), delta=0.05)
    bucket_map = {}
    for i in range(n_embeddings):
        key = int(lsh_keys[i])
        bucket_map.setdefault(key, []).append(i)
    for key, ids in bucket_map.items():
        ht.insert(key, ids)

    rng = np.random.RandomState(7)
    queries = centers[rng.choice(5, size=n_queries)] + np.random.normal(0, 0.2, size=(n_queries, d_dim))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    recalls = []
    for q in queries:
        exact = set(_brute_topk(emb, q, k=k).tolist())
        approx = set(_lsh_topk(emb, q, hyperplanes, powers_of_two, ht, bucket_map, k=k).tolist())
        recalls.append(len(exact & approx) / k)
    avg_recall = float(np.mean(recalls))
    note = (f"recall@{k} over {n_queries} queries = {avg_recall*100:.1f}%; "
            f"+fmm axis omitted (cosine retrieval, not a kernel sum)")

    bench = VariantBenchmark(
        f"App 7 -- High-dim LSH partition + retrieval (N={n_embeddings}, d={d_dim}, "
        f"{n_hyperplanes} hyperplanes; +fmm axis omitted)"
    )
    # Results are variable-length per query (some LSH buckets hold < k
    # candidates), so return a flat concatenation -- the benchmark only times
    # the call; recall is reported in the note, not via accuracy_vs.
    bench.add(
        "standard (brute exact top-k)",
        lambda: np.concatenate([_brute_topk(emb, q, k=k) for q in queries]),
        note=f"O(N*d) exact cosine top-{k} per query",
    )
    bench.add(
        "+elastichash (LSH bucket top-k)",
        lambda: np.concatenate([_lsh_topk(emb, q, hyperplanes, powers_of_two, ht, bucket_map, k=k)
                                for q in queries]),
        note=note,
    )
    return bench.run()


if __name__ == "__main__":
    run_app7_variants()
