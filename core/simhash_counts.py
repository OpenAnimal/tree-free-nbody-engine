"""
Dense SimHash count table for count-based exploration / state visitation.

What this is
------------
A flat ``int32`` array of ``2**k`` buckets indexed DIRECTLY by the SimHash
signature of a state vector. SimHash (Charikar, 2002, "Similarity Estimation
Techniques from Rounding Algorithms") maps a continuous state ``s`` to a
``k``-bit key via ``sign(W @ s)`` (random hyperplane LSH), so a neighbourhood
of similar states shares one bucket — exactly the mechanism count-based
exploration needs (Tang et al., 2017, "#Exploration: A Study of Count-Based
Exploration for Deep Reinforcement Learning"). The count for a key is then a
single array index: ``counts[keys]`` — one probe, no collision resolution,
no host round-trip, fully jittable in XLA.

Why a dense array, not a hash table
-----------------------------------
At ``k = 20`` bits the table is ``2**20 = 1,048,576`` int32 slots = 4 MB —
the SAME slot count as a funnel hash sized for 1M keys at load 0.95
(``core/elastic_hash.py``, ~1,052,640 slots). The funnel table needs 29
probes/hit and 277/miss (measured, ``BENCHMARKS.md`` §"Core hash tables");
the dense array needs 1. The funnel hash exists for the case where the key
space CANNOT be materialized (``k > 24``, full float observations, per-
``(s,a)`` keys) and load approaches 0.95 in bounded memory. At ``k <= 22``
that constraint does not bind, and paying ~29 dependent random memory
accesses to avoid allocating 4-16 MB is a bad trade — worse on GPU, where
each probe is an uncoalesced global-memory read with a data dependency while
the dense index is one coalesced scatter-add.

``k`` is configurable in ``[12, 22]``. Above 22 a ``ValueError`` is raised
whose message points at ``core/elastic_hash.py`` (``FunnelHashTable``) as
the structure to reach for IF AND ONLY IF all three of these hold:
  1. Key space cannot be materialized (``k > 24`` bits of resolution).
  2. Load factor is genuinely high (approaching 0.95) and memory is the
     binding constraint.
  3. The work per key dwarfs the ~29 probes (true for an FMM cell index
     driving a P2P kernel; false for a 2-layer 64-unit MLP).

NumPy and JAX paths
-------------------
``SimHashCounts`` (NumPy) is the reference implementation and the one the
benchmark harness uses. ``jax_simhash_counts`` provides a JAX-compatible
factory + ``increment`` / ``lookup`` that stay jittable under
``jax.jit`` (``counts.at[keys].add(1)`` is a scatter-add, no host
round-trip). Both share the same SimHash key derivation so counts agree
bit-for-bit when the same hyperplanes and ``k`` are used.

Standalone (repo root):
    python -X utf8 core/simhash_counts.py            # smoke test
    python -X utf8 core/simhash_counts.py --bench    # quick self-bench
"""
from __future__ import annotations

import argparse
import math
import time
from typing import Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False


# =============================================================================
# k bounds
# =============================================================================

MIN_K = 12
MAX_K = 22
# Above MAX_K the dense array exceeds ~16 MB (2**22 * 4 B = 16 MB) and the
# funnel hash becomes the right tool; see the module docstring's three
# conditions. The hard cap is MAX_K; the docstring / error message explains
# when to graduate to core/elastic_hash.py.

_K_OVERFLOW_MSG = (
    "k={k} exceeds the dense-array limit of {max_k}. A 2**{k} int32 table "
    "would be {mb:.0f} MB. Reach for core/elastic_hash.py (FunnelHashTable, "
    "the funnel hash of Farach-Colton, Krapivin, & Kuszmaul, 2025, Section 3) "
    "IF AND ONLY IF all three hold: (1) the key space cannot be materialized "
    "(k > 24 bits of state resolution, e.g. hashing full float observations "
    "or per-(s,a) keys); (2) load factor is genuinely high (approaching 0.95) "
    "and memory is the binding constraint; (3) the work per key dwarfs the "
    "~29 probes the funnel table costs per hit (true for an FMM cell index "
    "driving a P2P kernel; false for a small MLP). Otherwise lower k."
)


def _check_k(k: int) -> int:
    """Validate k is in [MIN_K, MAX_K]; raise ValueError with guidance above."""
    k = int(k)
    if k < MIN_K:
        raise ValueError(
            f"k={k} is below the minimum of {MIN_K}; a 2**{k} table is too "
            f"small for meaningful state discrimination. Use k >= {MIN_K}."
        )
    if k > MAX_K:
        mb = (2 ** k) * 4 / (1024 ** 2)
        raise ValueError(_K_OVERFLOW_MSG.format(k=k, max_k=MAX_K, mb=mb))
    return k


# =============================================================================
# SimHash key derivation (shared by NumPy and JAX paths)
# =============================================================================

def simhash_keys_numpy(hyperplanes: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Compute SimHash keys for a batch of states (NumPy).

    Parameters
    ----------
    hyperplanes : (k, dim) float
        Random projection matrix (one hyperplane per bit).
    states : (..., dim) float
        State vectors to hash.

    Returns
    -------
    keys : (...) int32
        SimHash signatures in [0, 2**k).
    """
    proj = states @ hyperplanes.T  # (..., k)
    bits = (proj > 0.0).astype(np.int32)  # (..., k)
    k = bits.shape[-1]
    powers = (np.int64(1) << np.arange(k - 1, -1, -1, dtype=np.int64))  # (k,)
    keys = (bits.astype(np.int64) * powers).sum(axis=-1)
    return keys.astype(np.int32)


# =============================================================================
# NumPy reference: SimHashCounts
# =============================================================================

class SimHashCounts:
    """Dense SimHash count table (NumPy reference).

    A flat ``int32[2**k]`` array indexed directly by the SimHash signature.
    Increment is ``np.add.at``; lookup is fancy indexing — one probe, no
    collision resolution, no host round-trip.

    Parameters
    ----------
    k : int
        Number of SimHash bits (table size = 2**k). Must be in [12, 22].
    dim : int
        Dimensionality of the state vectors to be hashed.
    seed : int
        Seed for the random hyperplanes (reproducibility).

    Attributes
    ----------
    k, dim, seed : as above
    hyperplanes : (k, dim) float64
        Random projection matrix.
    counts : (2**k,) int32
        The count table (zero-initialised).
    """

    def __init__(self, k: int = 20, dim: int = 32, seed: int = 0):
        self.k = _check_k(k)
        self.dim = int(dim)
        self.seed = int(seed)
        rng = np.random.RandomState(seed)
        self.hyperplanes = rng.randn(self.k, self.dim).astype(np.float64)
        self.counts = np.zeros(1 << self.k, dtype=np.int32)

    # -- key derivation ------------------------------------------------------

    def keys(self, states: np.ndarray) -> np.ndarray:
        """Hash a batch of states to their SimHash bucket keys.

        states : (..., dim) float -> keys : (...) int32
        """
        return simhash_keys_numpy(self.hyperplanes, states)

    # -- core ops -----------------------------------------------------------

    def increment(self, states: np.ndarray, inc: int = 1) -> np.ndarray:
        """Increment counts for the SimHash buckets of ``states``.

        Returns the keys that were incremented (for logging / bonus
        computation). Uses ``np.add.at`` for correct duplicate-key
        accumulation.
        """
        keys = self.keys(states)
        np.add.at(self.counts, keys, np.int32(inc))
        return keys

    def lookup(self, states: np.ndarray) -> np.ndarray:
        """Return the count for each state's SimHash bucket.

        states : (..., dim) float -> counts : (...) int32
        """
        keys = self.keys(states)
        return self.counts[keys]

    def reset(self) -> None:
        """Zero all counts (hyperplanes are kept)."""
        self.counts[:] = 0

    # -- introspection ------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of buckets (2**k)."""
        return 1 << self.k

    @property
    def memory_mb(self) -> float:
        """Table memory in MB (counts array only; hyperplanes negligible)."""
        return self.counts.nbytes / (1024 ** 2)

    @property
    def occupied(self) -> int:
        """Number of non-zero buckets."""
        return int((self.counts > 0).sum())

    @property
    def total(self) -> int:
        """Sum of all counts (total increments)."""
        return int(self.counts.sum())

    def __len__(self) -> int:
        return self.size


# =============================================================================
# JAX path: jittable SimHash counts
# =============================================================================

if HAS_JAX:
    from typing import NamedTuple

    class JAXSimHashCounts(NamedTuple):
        """JAX-compatible dense SimHash count table (pytree).

        Fields
        ------
        hyperplanes : (k, dim) float
            Random projection matrix.
        counts : (2**k,) int32
            The count table.
        """
        hyperplanes: jnp.ndarray
        counts: jnp.ndarray

    def make_jax_simhash_counts(
        k: int = 20, dim: int = 32, seed: int = 0,
    ) -> "JAXSimHashCounts":
        """Construct an empty JAX SimHash count table.

        ``k`` is validated eagerly (outside jit); the returned pytree is
        jit-compatible. ``2**k`` must fit in a positive int32, so k <= 30
        is the JAX hard limit; the [12, 22] policy is enforced here for
        consistency with the NumPy path.
        """
        k = _check_k(k)
        rng = jax.random.PRNGKey(seed)
        hp = jax.random.normal(rng, (k, dim))
        return JAXSimHashCounts(
            hyperplanes=hp,
            counts=jnp.zeros(1 << k, dtype=jnp.int32),
        )

    def jax_simhash_keys(table: "JAXSimHashCounts",
                         states: jnp.ndarray) -> jnp.ndarray:
        """Compute SimHash keys for a batch of states (JAX).

        states : (..., dim) -> keys : (...) int32
        """
        proj = states @ table.hyperplanes.T  # (..., k)
        bits = (proj > 0.0).astype(jnp.int32)  # (..., k)
        k = bits.shape[-1]
        powers = jnp.power(jnp.int32(2), jnp.arange(k - 1, -1, -1, dtype=jnp.int32))
        keys = jnp.sum(bits * powers, axis=-1)
        return keys.astype(jnp.int32)

    def jax_increment(table: "JAXSimHashCounts",
                      states: jnp.ndarray,
                      inc: int = 1) -> "Tuple[JAXSimHashCounts, jnp.ndarray]":
        """Increment counts for ``states``'s SimHash buckets (scatter-add).

        Returns (new_table, keys). Jittable: ``counts.at[keys].add(inc)``
        is a single XLA scatter-add — one probe, no host round-trip.
        """
        keys = jax_simhash_keys(table, states)
        new_counts = table.counts.at[keys].add(jnp.int32(inc))
        return table._replace(counts=new_counts), keys

    def jax_lookup(table: "JAXSimHashCounts",
                   states: jnp.ndarray) -> jnp.ndarray:
        """Return the count for each state's SimHash bucket (JAX)."""
        keys = jax_simhash_keys(table, states)
        return table.counts[keys]


# =============================================================================
# Smoke test / self-bench
# =============================================================================

def _smoke_test():
    """Quick correctness check: identical states share a bucket, distinct
    states (with enough hyperplanes) do not, counts accumulate."""
    t = SimHashCounts(k=16, dim=8, seed=42)
    # Two identical states -> same key -> count 2.
    s = np.random.RandomState(0).randn(1, 8)
    t.increment(s)
    t.increment(s)
    c = t.lookup(s)
    assert int(c[0]) == 2, f"identical-state count should be 2, got {c[0]}"
    # A nearby state (small perturbation) should share the bucket with high
    # probability at k=16, dim=8 — this is the exploration mechanism.
    s_near = s + np.random.RandomState(1).randn(1, 8) * 0.01
    c_near = t.lookup(s_near)
    # Not asserting equality (probabilistic), just that it runs and returns
    # a sensible count (0 or 2).
    assert int(c_near[0]) in (0, 2), f"nearby count unexpected: {c_near[0]}"
    # A very different state should get its own bucket (count 0).
    s_far = np.random.RandomState(99).randn(1, 8) * 100
    c_far = t.lookup(s_far)
    assert int(c_far[0]) == 0, f"far-state count should be 0, got {c_far[0]}"
    # Reset works.
    t.reset()
    assert t.lookup(s)[0] == 0, "reset failed"
    # k validation.
    for bad_k in (11, 23, 24, 32):
        try:
            SimHashCounts(k=bad_k, dim=8)
        except ValueError as e:
            assert "elastic_hash" in str(e) or "minimum" in str(e), \
                f"ValueError for k={bad_k} should mention elastic_hash or minimum"
        else:
            raise AssertionError(f"k={bad_k} should have raised ValueError")
    print(f"smoke test PASS  (k=16, dim=8, {t.size} buckets, {t.memory_mb:.2f} MB)")

    if HAS_JAX:
        jt = make_jax_simhash_counts(k=16, dim=8, seed=42)
        states = jnp.array(s)
        jt2, keys = jax_increment(jt, states)
        jt2, keys = jax_increment(jt2, states)
        c_jax = jax_lookup(jt2, states)
        assert int(c_jax[0]) == 2, f"JAX count should be 2, got {c_jax[0]}"
        # Jitted path agrees with eager.
        jitted_inc = jax.jit(lambda t, s: jax_increment(t, s))
        jitted_lk = jax.jit(lambda t, s: jax_lookup(t, s))
        jt3 = JAXSimHashCounts(hyperplanes=jt.hyperplanes,
                               counts=jnp.zeros(jt.counts.shape[0], dtype=jnp.int32))
        jt3, _ = jitted_inc(jt3, states)
        jt3, _ = jitted_inc(jt3, states)
        assert int(jitted_lk(jt3, states)[0]) == 2
        print("JAX smoke test PASS (jit + eager agree)")


def _self_bench():
    """Quick throughput self-benchmark vs the funnel table's published numbers."""
    rng = np.random.RandomState(0)
    for k in (16, 18, 20, 22):
        t = SimHashCounts(k=k, dim=32, seed=42)
        n = 1_000_000
        states = rng.randn(n, 32)
        t0 = time.perf_counter()
        keys = t.increment(states)
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        counts = t.lookup(states)
        t_lk = time.perf_counter() - t0
        print(f"k={k:2d}  2**k={t.size:>10,}  {t.memory_mb:>6.1f} MB  "
              f"build {n/t_build/1e6:>6.1f} M/s  lookup {n/t_lk/1e6:>6.1f} M/s  "
              f"probes=1  occupied={t.occupied:>8,}  total={t.total:,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--bench", action="store_true",
                    help="run the quick self-benchmark instead of the smoke test")
    args = ap.parse_args()
    if args.bench:
        _self_bench()
    else:
        _smoke_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
