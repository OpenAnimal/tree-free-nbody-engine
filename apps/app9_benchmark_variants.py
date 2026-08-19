"""Standardized variant benchmark for Application 9 (streaming vector DB).

Variants:
  standard      -- brute-force exact top-k cosine retrieval: scan all N
                   stored vectors per query (O(N*d) -- the reference for the
                   app's retrieval task)
  +elastichash  -- the app's compute path: random-hyperplane LSH -> Farach-
                   Colton funnel hash -> multi-probe (primary + 1-bit flip)
                   bucket candidates -> top-k over candidates only

The +fmm axis is OMITTED with reason: the task is high-dimensional
approximate nearest-neighbor retrieval, not a 2D/3D kernel sum.

Accuracy semantics: LSH retrieval is approximate. The correctness metric is
`recall@k` averaged over the query batch, reported in the note -- not a
rel-L2 against an exact array.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.benchmark_kit import VariantBenchmark


def _vectors(n_vectors: int = 10000, d_dim: int = 128, seed: int = 42):
    np.random.seed(seed)
    n_clusters = 20
    centers = np.random.randn(n_clusters, d_dim)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    labels = np.random.choice(n_clusters, size=n_vectors)
    vecs = centers[labels] + np.random.normal(0, 0.3, size=(n_vectors, d_dim))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs, centers


def run_app9_variants(n_vectors: int = 10000, d_dim: int = 128, n_queries: int = 200, k=10):
    from apps.app9_streaming_vector_db import StreamingVectorDB
    vecs, centers = _vectors(n_vectors=n_vectors, d_dim=d_dim)
    vdb = StreamingVectorDB(d_dim=d_dim, n_hyperplanes=13)
    for i in range(n_vectors):
        vdb.insert(vecs[i])

    rng = np.random.RandomState(7)
    queries = centers[rng.choice(len(centers), size=n_queries)] + \
        np.random.normal(0, 0.25, size=(n_queries, d_dim))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    # One-off recall check (LSH retrieval is approximate).
    recalls = []
    for q in queries:
        exact = set(np.argsort(vecs @ q)[::-1][:k].tolist())
        approx_ids, _, _ = vdb.query(q, top_k=k)
        approx = set(int(x) for x in approx_ids)
        recalls.append(len(exact & approx) / k)
    avg_recall = float(np.mean(recalls))
    note = (f"recall@{k} over {n_queries} queries = {avg_recall*100:.1f}%; "
            f"zero-reorder funnel-hash ingestion; "
            f"+fmm axis omitted (cosine ANN, not a kernel sum)")

    bench = VariantBenchmark(
        f"App 9 -- Streaming vector DB (N={n_vectors}, d={d_dim}, multi-probe LSH; "
        f"+fmm axis omitted)"
    )
    # Results are variable-length per query (multi-probe may return < k
    # candidates when buckets are sparse), so return a flat concatenation --
    # the benchmark only times the call; recall is reported in the note.
    bench.add(
        "standard (brute exact top-k)",
        lambda: np.concatenate([np.argsort(vecs @ q)[::-1][:k] for q in queries]),
        note=f"O(N*d) exact cosine top-{k} per query",
    )
    bench.add(
        "+elastichash (LSH multi-probe)",
        lambda: np.concatenate([vdb.query(q, top_k=k)[0] for q in queries]),
        note=note,
    )
    return bench.run()


if __name__ == "__main__":
    run_app9_variants()
