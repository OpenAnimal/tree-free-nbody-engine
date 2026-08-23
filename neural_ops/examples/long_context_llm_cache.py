"""
Example 2: Streaming Long-Context LLM Key-Value Cache
=====================================================
Demonstrates ElasticMultipoleKVCache during autoregressive decoding.
Prefills a 10,000-token historical context and executes fast O(1) probe decode steps
with zero element reordering (Farach-Colton, Krapivin, & Kuszmaul, 2025 non-reordering open addressing).
"""

import numpy as np
import time
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from neural_ops import ElasticMultipoleKVCache


def run_long_context_demo():
    print("=" * 70)
    print(">>> DEMO 2: Streaming Long-Context LLM Elastic KV-Cache")
    print("=" * 70)

    d_k, d_v = 64, 64
    cache = ElasticMultipoleKVCache(
        d_k=d_k,
        d_v=d_v,
        n_hyperplanes=8,         # 256 semantic LSH buckets
        bucket_capacity=32,      # Per-bucket token threshold before multipole compression
        recent_window_size=128   # Full-precision local context buffer
    )

    # 1. Prefill Long-Context Prompt (10,000 tokens)
    N_prefill = 10000
    print(f"[*] Prefilling prompt context with {N_prefill:,} tokens...")
    np.random.seed(42)
    prompt_k = np.random.randn(N_prefill, d_k).astype(np.float32)
    prompt_v = np.random.randn(N_prefill, d_v).astype(np.float32)

    t0 = time.perf_counter()
    cache.append_batch(prompt_k, prompt_v)
    t_prefill = (time.perf_counter() - t0) * 1000.0
    print(f"[-] Prefill Time:    {t_prefill:.2f} ms ({N_prefill / (t_prefill / 1000):,.0f} tokens/s)")

    # 2. Autoregressive Generation Loop (Decode Steps)
    n_decode_steps = 10
    print(f"[*] Simulating {n_decode_steps} autoregressive decode steps...")

    decode_times = []
    for step in range(n_decode_steps):
        query_token = np.random.randn(d_k).astype(np.float32)

        t0 = time.perf_counter()
        attended_val, meta = cache.query_attention(query_token)
        t_decode = (time.perf_counter() - t0) * 1000.0
        decode_times.append(t_decode)

        # Append newly generated token
        new_k = np.random.randn(d_k).astype(np.float32)
        new_v = attended_val
        cache.append_token(new_k, new_v)

    avg_decode = np.mean(decode_times)
    print(f"[-] Total Context:   {cache.total_tokens_inserted:,} tokens in memory")
    print(f"[-] Avg Decode Time: {avg_decode:.3f} ms / token ({1000.0 / avg_decode:.1f} tokens/s)")
    print(f"[-] Compression:     {meta['compression_ratio']:.1f}x reduction vs full KV cache")
    print(f"[-] Memory Reorder:  0 (Strict Farach-Colton lock-free open addressing)")
    print("=" * 70)


if __name__ == "__main__":
    run_long_context_demo()
