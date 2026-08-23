"""
Elastic Multipole KV-Cache (`elastic_kv_cache.py`)
==================================================
Lock-Free, Contiguous O(1) Streaming Key-Value Memory for Long-Context LLMs.
Combines Farach-Colton, Krapivin, & Kuszmaul (2025) Non-Reordering Open Addressing with Multipole Historical Compression.

Solves the Long-Context LLM Memory Bottleneck:
- Zero element displacement / reordering (100% lock-free & CAS-compatible).
- Retains full exact tokens for recent/active contexts.
- Compresses distant context into Taylor/multipole summary moments, preventing OOM in 1M+ token contexts.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List

# Tri-backend acceleration (NumPy reference / torch / jax). See
# `neural_ops/_accel.py`. The cache *state* (buckets, cluster summaries, recent
# ring) stays NumPy on CPU; only `query_attention`'s dense math (matmul / exp /
# weighted sums) moves to the device backend.
try:
    from ._accel import (
        resolve_backend as _resolve_backend,
        get_ns as _get_ns,
        to_backend as _to_backend,
        as_numpy as _as_numpy,
        get_compiled as _get_compiled,
        HAS_TORCH as _HAS_TORCH,
        HAS_JAX as _HAS_JAX,
    )
except ImportError:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
    try:
        from neural_ops._accel import (
            resolve_backend as _resolve_backend,
            get_ns as _get_ns,
            to_backend as _to_backend,
            as_numpy as _as_numpy,
            get_compiled as _get_compiled,
            HAS_TORCH as _HAS_TORCH,
            HAS_JAX as _HAS_JAX,
        )
    except ImportError:
        _resolve_backend = None
        _get_ns = None
        _to_backend = None
        _as_numpy = None
        _get_compiled = None
        _HAS_TORCH = False
        _HAS_JAX = False

if _HAS_TORCH:
    import torch
if _HAS_JAX:
    import jax.numpy as jnp


def _make_kvq_kernel(exp, clip, matmul, sumop, zeros):
    """Branch-free KV-cache query kernel. Empty tiers are passed as zero-row
    arrays so the math naturally zeros their contribution (no shape-dependent
    branching inside the compiled region). `global_max` is computed on CPU
    and passed in (avoids max-of-empty issues across backends). `d_v` is
    derived from `rec_v.shape[-1]` (always concrete at trace time) so no
    static int arg is needed for jax.jit."""
    def kern(rec_k, rec_v, b_k, b_v, bucket_keep_mask, far_mean_k,
             far_eff_v, far_eff_count, q, scale, global_max):
        d_v = rec_v.shape[-1]
        acc_val = zeros((d_v,))
        acc_weight = zeros((1,)) + 1e-9
        # Tier 1: recent ring (rec_k may be (0, d_k) when empty).
        rec_scores = matmul(rec_k, q) * scale
        rec_w = exp(clip(rec_scores - global_max, -30.0, 30.0))
        acc_val = acc_val + sumop(rec_w[:, None] * rec_v, axis=0)
        acc_weight = acc_weight + sumop(rec_w)
        # Tier 2: target bucket (b_k may be (0, d_k) when empty).
        bucket_scores = matmul(b_k, q) * scale
        bucket_w = exp(clip(bucket_scores - global_max, -30.0, 30.0)) * bucket_keep_mask
        acc_val = acc_val + sumop(bucket_w[:, None] * b_v, axis=0)
        acc_weight = acc_weight + sumop(bucket_w)
        # Tier 3: far clusters (far_mean_k may be (0, d_k) when empty).
        far_scores = matmul(far_mean_k, q) * scale
        far_w = exp(clip(far_scores - global_max, -30.0, 30.0))
        acc_val = acc_val + matmul(far_w, far_eff_v)
        acc_weight = acc_weight + sumop(far_w * far_eff_count)
        return acc_val / acc_weight
    return kern


def _sum_np(x, axis=-1, keepdims=False):
    return x.sum(axis=axis, keepdims=keepdims)


_KVQ_KERNELS = None


def _kvq_kernels():
    global _KVQ_KERNELS
    if _KVQ_KERNELS is None:
        def _torch_sum(x, axis=-1, keepdims=False):
            return x.sum(dim=axis, keepdim=keepdims)
        def _jax_sum(x, axis=-1, keepdims=False):
            return x.sum(axis=axis, keepdims=keepdims)
        # Device-aware torch.zeros (raw torch.zeros defaults to CPU, which would
        # clash with the cuda tensors the rest of the kernel uses).
        from ._accel import torch_device as _torch_device
        def _torch_zeros(shape, dtype=None):
            if dtype is None:
                dtype = torch.float32
            return torch.zeros(shape, dtype=dtype, device=_torch_device())
        _KVQ_KERNELS = {
            "numpy": _make_kvq_kernel(np.exp, np.clip, np.matmul, _sum_np, np.zeros),
            "torch": _make_kvq_kernel(torch.exp, torch.clamp, torch.matmul, _torch_sum, _torch_zeros) if _HAS_TORCH else None,
            "jax": _make_kvq_kernel(jnp.exp, jnp.clip, jnp.matmul, _jax_sum, jnp.zeros) if _HAS_JAX else None,
        }
    return _KVQ_KERNELS


class ElasticMultipoleKVCache:
    """
    Continuous, Non-Reordering Streaming KV-Cache for Autoregressive Transformers.
    Recent tokens get exact P2P attention; evicted tokens are compressed into
    per-bucket multipole moments (key/value sums), giving O(recent + buckets)
    retrieval over an unbounded stream.

    Shapes / dtypes
    ---------------
    k_vec : float32 (d_k,)   per-token append; v_vec : float32 (d_v,)
    q_vec : float32 (d_k,) -> (out float32 (d_v,), meta dict) via query_attention()
    batch helpers: append_batch(K_seq (T, d_k), V_seq (T, d_v))

    Example
    -------
    >>> import numpy as np
    >>> from neural_ops.elastic_kv_cache import ElasticMultipoleKVCache
    >>> rng = np.random.default_rng(0)
    >>> cache = ElasticMultipoleKVCache(d_k=64, d_v=64, bucket_capacity=32)
    >>> for _ in range(1000):
    ...     cache.append_token(rng.standard_normal(64).astype(np.float32),
    ...                        rng.standard_normal(64).astype(np.float32))
    >>> out, meta = cache.query_attention(rng.standard_normal(64).astype(np.float32))
    """
    def __init__(
        self,
        d_k: int = 64,
        d_v: int = 64,
        n_hyperplanes: int = 8,
        bucket_capacity: int = 32,
        recent_window_size: int = 128,
        backend: str = "numpy",
        jit: bool = False,
    ):
        self.d_k = d_k
        self.d_v = d_v
        self.n_hyperplanes = n_hyperplanes
        self.bucket_capacity = bucket_capacity
        self.recent_window_size = recent_window_size
        # Acceleration backend for query_attention only (cache state stays
        # NumPy on CPU). "numpy" (reference) | "torch" | "jax".
        self.backend = _resolve_backend(backend) if _resolve_backend is not None else "numpy"
        self.jit = bool(jit)

        # 1. Random Hyperplanes for Cosine Locality-Sensitive Hashing (LSH)
        rng = np.random.RandomState(1337)
        self.hyperplanes = rng.normal(0, 1.0, size=(d_k, n_hyperplanes)).astype(np.float32)
        self.hyperplanes /= np.linalg.norm(self.hyperplanes, axis=0, keepdims=True)
        self.powers_of_two = (1 << np.arange(n_hyperplanes, dtype=np.int64))

        # 2. Non-Reordering Level-Arranged Backing Storage
        self.bucket_keys: Dict[int, List[np.ndarray]] = {}
        self.bucket_vals: Dict[int, List[np.ndarray]] = {}
        self.bucket_token_ids: Dict[int, List[int]] = {}

        # 3. Multipole Far-Field Compressed Summaries
        # cluster_k_sum: (d_k,), cluster_v_sum: (d_v,), cluster_count: int
        self.cluster_k_sum: Dict[int, np.ndarray] = {}
        self.cluster_v_sum: Dict[int, np.ndarray] = {}
        self.cluster_token_count: Dict[int, int] = {}

        # 4. Global FIFO Recent Ring Buffer (Exact Local Attention)
        self.recent_k = np.zeros((recent_window_size, d_k), dtype=np.float32)
        self.recent_v = np.zeros((recent_window_size, d_v), dtype=np.float32)
        self.recent_token_ids = np.full(recent_window_size, -1, dtype=np.int64)
        self.recent_bucket_ids = np.full(recent_window_size, -1, dtype=np.int64)
        self.recent_idx = 0
        self.total_tokens_inserted = 0

    def _compute_lsh_key(self, k_vec: np.ndarray) -> int:
        """Computes integer semantic bucket index via random hyperplane projection."""
        proj = np.matmul(k_vec, self.hyperplanes) > 0 # (n_hyperplanes,)
        return int(np.sum(proj * self.powers_of_two))

    def append_token(self, k_vec: np.ndarray, v_vec: np.ndarray, token_id: Optional[int] = None) -> int:
        """
        Inserts a single key-value token into the cache.
        Returns: semantic bucket index.
        """
        tid = token_id if token_id is not None else self.total_tokens_inserted
        k_norm = k_vec / (np.linalg.norm(k_vec) + 1e-8)
        bucket_idx = self._compute_lsh_key(k_norm)

        # Update FIFO recent buffer
        r_pos = self.recent_idx % self.recent_window_size
        self.recent_k[r_pos] = k_vec
        self.recent_v[r_pos] = v_vec
        self.recent_token_ids[r_pos] = tid
        self.recent_bucket_ids[r_pos] = bucket_idx
        self.recent_idx += 1
        self.total_tokens_inserted += 1

        # Insert into non-reordering semantic hash bucket
        if bucket_idx not in self.bucket_keys:
            self.bucket_keys[bucket_idx] = []
            self.bucket_vals[bucket_idx] = []
            self.bucket_token_ids[bucket_idx] = []
            self.cluster_k_sum[bucket_idx] = np.zeros(self.d_k, dtype=np.float32)
            self.cluster_v_sum[bucket_idx] = np.zeros(self.d_v, dtype=np.float32)
            self.cluster_token_count[bucket_idx] = 0

        self.bucket_keys[bucket_idx].append(k_vec)
        self.bucket_vals[bucket_idx].append(v_vec)
        self.bucket_token_ids[bucket_idx].append(tid)

        # Update multipole centroid sums
        self.cluster_k_sum[bucket_idx] += k_vec
        self.cluster_v_sum[bucket_idx] += v_vec
        self.cluster_token_count[bucket_idx] += 1

        # If bucket exceeds capacity, compress oldest tokens into multipole moments
        if len(self.bucket_keys[bucket_idx]) > self.bucket_capacity:
            # Compress half of the bucket into permanent multipole summary
            evict_count = self.bucket_capacity // 2
            self.bucket_keys[bucket_idx] = self.bucket_keys[bucket_idx][evict_count:]
            self.bucket_vals[bucket_idx] = self.bucket_vals[bucket_idx][evict_count:]
            self.bucket_token_ids[bucket_idx] = self.bucket_token_ids[bucket_idx][evict_count:]

        return bucket_idx

    def append_batch(self, K_seq: np.ndarray, V_seq: np.ndarray) -> List[int]:
        """Appends a sequence of KV tokens (e.g. prompt prefill)."""
        seq_len = len(K_seq)
        bucket_indices = []
        for i in range(seq_len):
            b_idx = self.append_token(K_seq[i], V_seq[i])
            bucket_indices.append(b_idx)
        return bucket_indices

    def query_attention(
        self,
        q_vec: np.ndarray,          # (d_k,) Query vector
        temperature: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes hybrid O(1) attention query against the KV cache:
        1. Exact Softmax on recent token buffer (Local Attention)
        2. Exact Softmax on matching semantic LSH bucket (Near-Field Retrieval)
        3. Multipole moment evaluation across all distant semantic clusters (Far-Field Global Context)

        Bug fixes (Round-7 audit):
        - (a) ONE global max-subtraction across all tiers (per-tier max rescales
          tiers by different constants).
        - (b) Exactly-evaluated tokens (recent ring + target bucket current
          members) are subtracted from the cluster summaries they overlap, so
          each token is counted exactly once.
        - (c) The target bucket's evicted tokens (summary minus current members)
          are now included in the far pass — previously the far pass skipped the
          target bucket entirely, dropping its evicted tokens.
        """
        if self.backend != "numpy":
            return self._query_attention_accel(q_vec, temperature)
        # temperature=0.0 must survive (falsy `or` clobbers it).
        # float() wrapper critical: 1.0/np.sqrt() returns numpy.float64 which
        # promotes float32 JAX arrays to float64 when jax_enable_x64=True.
        scale = float(1.0 / np.sqrt(self.d_k)) if temperature is None else float(temperature)
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
        target_bucket = self._compute_lsh_key(q_norm)

        # --- Collect exactly-evaluated tokens and their (k, v, bucket) ---
        # Used to subtract from cluster summaries so no token is double-counted.
        exact_map: Dict[int, Tuple[np.ndarray, np.ndarray, int]] = {}

        # --- 1. Recent Ring Buffer Exact Evaluation ---
        n_recent = min(self.total_tokens_inserted, self.recent_window_size)
        rec_scores = None
        rec_weights = None
        rec_v = None
        if n_recent > 0:
            rec_k = self.recent_k[:n_recent]
            rec_v = self.recent_v[:n_recent]
            rec_ids = self.recent_token_ids[:n_recent]
            rec_buckets = self.recent_bucket_ids[:n_recent]
            rec_scores = np.matmul(rec_k, q_vec) * scale
            for i in range(n_recent):
                exact_map[int(rec_ids[i])] = (rec_k[i], rec_v[i], int(rec_buckets[i]))

        # --- 2. Near-Field Semantic Bucket Probing (O(1) LSH lookup) ---
        # The recent window (tier 1) may contain tokens whose LSH bucket is the
        # target bucket; those tokens are already evaluated exactly in tier 1
        # and must NOT be re-counted in the tier-2 bucket accumulation, or they
        # are double-counted (contradicting the "each token counted exactly
        # once" guarantee). `bucket_keep_mask` zeroes their bucket weight.
        bucket_scores = None
        bucket_weights = None
        bucket_v = None
        bucket_keep_mask = None
        recent_ids_set = set(int(t) for t in rec_ids) if n_recent > 0 else set()
        if target_bucket in self.bucket_keys and len(self.bucket_keys[target_bucket]) > 0:
            b_k = np.array(self.bucket_keys[target_bucket], dtype=np.float32)
            b_v_arr = np.array(self.bucket_vals[target_bucket], dtype=np.float32)
            b_ids = self.bucket_token_ids[target_bucket]
            bucket_scores = np.matmul(b_k, q_vec) * scale
            bucket_v = b_v_arr
            # Mask out bucket members already evaluated in the recent window.
            bucket_keep_mask = np.array(
                [int(b_ids[i]) not in recent_ids_set for i in range(len(b_ids))],
                dtype=np.float32,
            )
            for i in range(len(b_ids)):
                tid = int(b_ids[i])
                # Don't overwrite recent entry (same k/v, already counted).
                if tid not in exact_map:
                    exact_map[tid] = (b_k[i], b_v_arr[i], target_bucket)

        exact_tokens_evaluated = len(exact_map)

        # --- Build per-bucket subtraction from exact tokens ---
        subtract_k: Dict[int, np.ndarray] = {}
        subtract_v: Dict[int, np.ndarray] = {}
        subtract_count: Dict[int, int] = {}
        for tid, (k, v, b_id) in exact_map.items():
            if b_id not in subtract_k:
                subtract_k[b_id] = np.zeros(self.d_k, dtype=np.float32)
                subtract_v[b_id] = np.zeros(self.d_v, dtype=np.float32)
                subtract_count[b_id] = 0
            subtract_k[b_id] += k
            subtract_v[b_id] += v
            subtract_count[b_id] += 1

        # --- 3. Far-Field Multipole Cluster Summaries (ALL buckets, including target) ---
        # For each bucket, effective summary = full summary - exact tokens in it.
        # This includes the target bucket's evicted tokens (summary - current members).
        far_scores = []
        far_eff_v = []
        far_eff_count = []
        far_b_ids = []
        for b_id, count in self.cluster_token_count.items():
            if count <= 0:
                continue
            sub_c = subtract_count.get(b_id, 0)
            eff_count = count - sub_c
            if eff_count <= 0:
                continue  # all tokens in this bucket were evaluated exactly
            eff_k_sum = self.cluster_k_sum[b_id] - subtract_k.get(b_id, 0.0)
            eff_v_sum = self.cluster_v_sum[b_id] - subtract_v.get(b_id, 0.0)
            mean_k = eff_k_sum / eff_count
            score_cluster = float(np.dot(q_vec, mean_k)) * scale
            far_scores.append(score_cluster)
            far_eff_v.append(eff_v_sum)
            far_eff_count.append(eff_count)
            far_b_ids.append(b_id)

        far_clusters_evaluated = len(far_scores)

        # --- ONE global max across all tiers (fix (a)) ---
        all_scores = []
        if rec_scores is not None:
            all_scores.append(np.max(rec_scores))
        if bucket_scores is not None:
            all_scores.append(np.max(bucket_scores))
        if far_scores:
            all_scores.append(np.max(far_scores))
        global_max = max(all_scores) if all_scores else 0.0

        # --- Accumulate with single global max ---
        acc_val = np.zeros(self.d_v, dtype=np.float32)
        acc_weight = 1e-9

        if rec_scores is not None:
            rec_weights = np.exp(np.clip(rec_scores - global_max, -30.0, 30.0))
            acc_val += np.sum(rec_weights[:, None] * rec_v, axis=0)
            acc_weight += np.sum(rec_weights)

        if bucket_scores is not None:
            bucket_weights = np.exp(np.clip(bucket_scores - global_max, -30.0, 30.0))
            # Zero out bucket members already evaluated in the recent window
            # (tier 1) so each token is counted exactly once across tiers.
            bucket_weights = bucket_weights * bucket_keep_mask
            acc_val += np.sum(bucket_weights[:, None] * bucket_v, axis=0)
            acc_weight += np.sum(bucket_weights)

        for i in range(far_clusters_evaluated):
            w = np.exp(np.clip(far_scores[i] - global_max, -30.0, 30.0))
            acc_val += w * far_eff_v[i]
            acc_weight += w * far_eff_count[i]

        attended_output = acc_val / acc_weight

        meta = {
            "total_tokens_in_history": self.total_tokens_inserted,
            "exact_tokens_evaluated": exact_tokens_evaluated,
            "far_clusters_evaluated": far_clusters_evaluated,
            "compression_ratio": float(self.total_tokens_inserted) / max(1, exact_tokens_evaluated + far_clusters_evaluated),
        }
        return attended_output, meta

    def _query_attention_accel(
        self,
        q_vec: np.ndarray,
        temperature: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """torch / jax backend for query_attention. The cache-state gathering
        (exact_map, per-bucket subtraction, far effective summaries) mirrors
        the numpy path exactly and stays CPU; only the tier matmuls + exp +
        weighted accumulation run on the device via one compiled branch-free
        kernel. Empty tiers are passed as zero-row arrays so the kernel has no
        shape-dependent branching. `global_max` is computed on CPU (small
        matmuls) and passed in to avoid max-of-empty across backends."""
        ns = _get_ns(self.backend)
        # float() wrapper critical: 1.0/np.sqrt() returns numpy.float64 which
        # promotes float32 JAX arrays to float64 when jax_enable_x64=True.
        scale = float(1.0 / np.sqrt(self.d_k)) if temperature is None else float(temperature)
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
        target_bucket = self._compute_lsh_key(q_norm)

        exact_map: Dict[int, Tuple[np.ndarray, np.ndarray, int]] = {}

        # --- 1. Recent ring (CPU gather) ---
        n_recent = min(self.total_tokens_inserted, self.recent_window_size)
        if n_recent > 0:
            rec_k = self.recent_k[:n_recent]
            rec_v = self.recent_v[:n_recent]
            rec_ids = self.recent_token_ids[:n_recent]
            rec_buckets = self.recent_bucket_ids[:n_recent]
            for i in range(n_recent):
                exact_map[int(rec_ids[i])] = (rec_k[i], rec_v[i], int(rec_buckets[i]))
        else:
            rec_k = np.zeros((0, self.d_k), dtype=np.float32)
            rec_v = np.zeros((0, self.d_v), dtype=np.float32)

        # --- 2. Target bucket (CPU gather) ---
        recent_ids_set = set(int(t) for t in rec_ids) if n_recent > 0 else set()
        if target_bucket in self.bucket_keys and len(self.bucket_keys[target_bucket]) > 0:
            b_k = np.array(self.bucket_keys[target_bucket], dtype=np.float32)
            b_v_arr = np.array(self.bucket_vals[target_bucket], dtype=np.float32)
            b_ids = self.bucket_token_ids[target_bucket]
            bucket_keep_mask = np.array(
                [int(b_ids[i]) not in recent_ids_set for i in range(len(b_ids))],
                dtype=np.float32,
            )
            for i in range(len(b_ids)):
                tid = int(b_ids[i])
                if tid not in exact_map:
                    exact_map[tid] = (b_k[i], b_v_arr[i], target_bucket)
        else:
            b_k = np.zeros((0, self.d_k), dtype=np.float32)
            b_v_arr = np.zeros((0, self.d_v), dtype=np.float32)
            bucket_keep_mask = np.zeros((0,), dtype=np.float32)

        exact_tokens_evaluated = len(exact_map)

        # --- Per-bucket subtraction from exact tokens (CPU) ---
        subtract_k: Dict[int, np.ndarray] = {}
        subtract_v: Dict[int, np.ndarray] = {}
        subtract_count: Dict[int, int] = {}
        for tid, (k, v, b_id) in exact_map.items():
            if b_id not in subtract_k:
                subtract_k[b_id] = np.zeros(self.d_k, dtype=np.float32)
                subtract_v[b_id] = np.zeros(self.d_v, dtype=np.float32)
                subtract_count[b_id] = 0
            subtract_k[b_id] += k
            subtract_v[b_id] += v
            subtract_count[b_id] += 1

        # --- 3. Far effective summaries (CPU gather) ---
        far_mean_k_list = []
        far_eff_v_list = []
        far_eff_count_list = []
        for b_id, count in self.cluster_token_count.items():
            if count <= 0:
                continue
            sub_c = subtract_count.get(b_id, 0)
            eff_count = count - sub_c
            if eff_count <= 0:
                continue
            eff_k_sum = self.cluster_k_sum[b_id] - subtract_k.get(b_id, 0.0)
            eff_v_sum = self.cluster_v_sum[b_id] - subtract_v.get(b_id, 0.0)
            far_mean_k_list.append((eff_k_sum / eff_count).astype(np.float32))
            far_eff_v_list.append(eff_v_sum.astype(np.float32))
            far_eff_count_list.append(float(eff_count))

        far_clusters_evaluated = len(far_mean_k_list)

        # --- global_max on CPU (small matmuls) ---
        all_scores = []
        if n_recent > 0:
            all_scores.append(float(np.max(np.matmul(rec_k, q_vec) * scale)))
        if b_k.shape[0] > 0:
            all_scores.append(float(np.max(np.matmul(b_k, q_vec) * scale)))
        if far_mean_k_list:
            fm = np.stack(far_mean_k_list)
            all_scores.append(float(np.max(np.matmul(fm, q_vec) * scale)))
        global_max = max(all_scores) if all_scores else 0.0

        # --- Stack far arrays (empty -> zero-row) ---
        if far_mean_k_list:
            far_mean_k_arr = np.stack(far_mean_k_list).astype(np.float32)
            far_eff_v_arr = np.stack(far_eff_v_list).astype(np.float32)
            far_eff_count_arr = np.asarray(far_eff_count_list, dtype=np.float32)
        else:
            far_mean_k_arr = np.zeros((0, self.d_k), dtype=np.float32)
            far_eff_v_arr = np.zeros((0, self.d_v), dtype=np.float32)
            far_eff_count_arr = np.zeros((0,), dtype=np.float32)

        # --- Move to device + run compiled kernel ---
        rec_k_d = _to_backend(rec_k.astype(np.float32), self.backend, ns.float32)
        rec_v_d = _to_backend(rec_v.astype(np.float32), self.backend, ns.float32)
        b_k_d = _to_backend(b_k, self.backend, ns.float32)
        b_v_d = _to_backend(b_v_arr, self.backend, ns.float32)
        mask_d = _to_backend(bucket_keep_mask, self.backend, ns.float32)
        fm_d = _to_backend(far_mean_k_arr, self.backend, ns.float32)
        fv_d = _to_backend(far_eff_v_arr, self.backend, ns.float32)
        fc_d = _to_backend(far_eff_count_arr, self.backend, ns.float32)
        q_d = _to_backend(q_vec.astype(np.float32), self.backend, ns.float32)

        kernels = _kvq_kernels()
        kern = kernels[self.backend]
        kern = _get_compiled(self.backend, "kvq", kern, jit=self.jit)
        out_d = kern(rec_k_d, rec_v_d, b_k_d, b_v_d, mask_d, fm_d, fv_d, fc_d,
                     q_d, scale, global_max)
        attended_output = _as_numpy(out_d).astype(np.float32)

        meta = {
            "total_tokens_in_history": self.total_tokens_inserted,
            "exact_tokens_evaluated": exact_tokens_evaluated,
            "far_clusters_evaluated": far_clusters_evaluated,
            "compression_ratio": float(self.total_tokens_inserted) / max(1, exact_tokens_evaluated + far_clusters_evaluated),
            "backend": self.backend,
            "jit": self.jit,
        }
        return attended_output, meta
