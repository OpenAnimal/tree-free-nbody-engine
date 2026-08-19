"""Standardized variant benchmark for the FMM core (2D log kernel).

Variants:
  standard             — exact direct O(N^2) summation (the reference)
  +fmm (CGR88 adaptive) — TreeFreeElasticAdaptiveFMM(p=10), the funnel-hash
                          indexed adaptive CGR88 engine
  +fmm (flat vectorized) — FastVectorizedFMM(depth=5, order=8), single-level
                           vectorized CGR88 expansions on the elastic hash
  +quantized           — VoxelPackedTreeFreeFMM (32-bit coordinate packing);
                          the CALLER inputs are adapted to its
                          (positions, charges) -> (potentials, metrics) API;
                          the quantized module itself is NOT modified.

Acceptance: the table prints; adaptive FMM rel-L2 < 1e-6. If an FMM variant
is NOT the fastest, that is reported honestly in the note — no tuning to win.
"""
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.benchmark_kit import VariantBenchmark


def _clustered_distribution(n: int = 2000, seed: int = 707):
    """Clustered, multi-scale distribution matching the spirit of
    test_flat_fmm_elastic_hash_occupancy in core/test_cgr88_cross_validation.py:
    a few tight Gaussian clusters of different scales on a sparse background."""
    rng = np.random.default_rng(seed)
    # Cluster sizes chosen so the total rounds to `n`.
    n1 = max(1, int(n * 0.20))
    n2 = max(1, int(n * 0.30))
    n3 = max(1, int(n * 0.40))
    n_bg = max(0, n - (n1 + n2 + n3))
    c1 = rng.random((n1, 2)) * 0.10 + 0.10
    c2 = rng.random((n2, 2)) * 0.15 + 0.70
    c3 = rng.random((n3, 2)) * 0.30 + 0.40
    bg = rng.random((n_bg, 2)) * 0.94 + 0.03 if n_bg > 0 else np.empty((0, 2))
    pts = np.vstack([c1, c2, c3, bg]).astype(np.float64)
    pts = np.clip(pts, 0.01, 0.99)
    q = rng.uniform(-1.0, 1.0, size=len(pts)).astype(np.float64)
    return pts, q


def run_core_fmm_variants(n: int = 2000):
    from core import (
        exact_direct_nbody_2d,
        TreeFreeElasticAdaptiveFMM,
        FastVectorizedFMM,
    )
    from quantized_bitpacked_optimization.packed_vectorized_fmm import (
        VoxelPackedTreeFreeFMM,
    )

    pts, q = _clustered_distribution(n=n)

    # Pre-instantiate the FMM engines (build cost is part of the timed call,
    # matching how a caller actually uses them).
    adaptive = TreeFreeElasticAdaptiveFMM(p=10)
    flat = FastVectorizedFMM(depth=5, order=8)
    quantized = VoxelPackedTreeFreeFMM(depth=5, order=4)

    bench = VariantBenchmark(
        f"FMM core (2D log kernel; N={n}, clustered multi-scale distribution)"
    )

    bench.add(
        "standard (exact direct)",
        lambda: exact_direct_nbody_2d(pts, q),
        note="O(N^2) reference",
    )
    bench.add(
        "+fmm (CGR88 adaptive)",
        lambda: adaptive.evaluate(pts, q, compute_forces=False),
        accuracy_vs="standard (exact direct)",
        note="funnel-hash adaptive CGR88, p=10; NOT faster than direct at N=2000 (Python tree traversal overhead)",
    )
    bench.add(
        "+fmm (flat vectorized)",
        lambda: flat.evaluate(pts, q, compute_forces=False),
        accuracy_vs="standard (exact direct)",
        note="single-level vectorized CGR88, depth=5 order=8; NOT faster than direct at N=2000 (K^2 M2L dominates at this scale)",
    )
    bench.add(
        "+quantized (32-bit packed)",
        lambda: quantized.evaluate(pts, q)[0],
        accuracy_vs="standard (exact direct)",
        note="VoxelPackedTreeFreeFMM; module documents ~1.2e-1 rel-L2 packed cost (this clustered N=2000 distribution measures higher — see table)",
    )
    return bench.run()


def _direct_chunked(positions, charges, softening=0.0, block=2048):
    """Chunked exact O(N^2) 2D log direct sum.

    The existing `exact_direct_nbody_2d` is a Python loop over targets with a
    vectorized inner; at N>=8000 the Python-loop overhead dominates. This
    version blocks the target axis so each block builds an (block, N) r2
    matrix with one vectorized op, keeping peak memory at block*N*8 bytes
    (block=2048, N=32000 -> ~524 MB) and the Python loop count at N/block.
    """
    N = len(positions)
    pot = np.zeros(N, dtype=np.float64)
    eps2 = softening * softening
    px = positions[:, 0]
    py = positions[:, 1]
    for lo in range(0, N, block):
        hi = min(lo + block, N)
        dx = px[lo:hi, None] - px[None, :]
        dy = py[lo:hi, None] - py[None, :]
        r2 = dx * dx + dy * dy + eps2
        # zero out self terms for the diagonal block
        for k in range(lo, hi):
            r2[k - lo, k] = 1.0
        pot[lo:hi] = np.sum(charges[None, :] * 0.5 * np.log(r2), axis=1)
        # subtract the self term that was included as q_i * 0.5 * log(1) = 0,
        # so no correction needed beyond zeroing r2 above.
    return pot


def run_scaling(ns=(2000, 8000, 32000), direct_budget_s=120.0):
    """Show the O(N) crossover between direct O(N^2) and the flat vectorized FMM.

    For each N in `ns` (clustered distribution, same generator as the N=2000
    table), variants:
      standard (direct)  -- chunked vectorized direct O(N^2)
      +fmm (flat vectorized) -- FastVectorizedFMM(depth=5, order=8)

    The adaptive CGR88 engine is OMITTED at these N: its Python tree
    traversal is even slower than the flat scheme and would not change the
    crossover conclusion, only add wall-clock time to the benchmark run.

    Direct O(N^2) at N=32000 is ~1e9 pairs; the chunked direct keeps memory
    bounded but the wall-clock may still be large. If the first N's direct
    time projects to > direct_budget_s for the largest N, the largest N is
    dropped to 16000 (per the round-3 plan).
    """
    from core import FastVectorizedFMM

    print(f"\n=== Core FMM scaling (clustered distribution; direct budget {direct_budget_s:.0f}s) ===")
    rows = []
    ns = list(ns)
    for n in ns:
        pts, q = _clustered_distribution(n=n)
        flat = FastVectorizedFMM(depth=5, order=8)

        # Time direct (chunked); if it blows the budget for the largest N,
        # shrink the schedule and re-run from the dropped point.
        t0 = time.perf_counter()
        pot_direct = _direct_chunked(pts, q)
        t_direct = time.perf_counter() - t0

        t0 = time.perf_counter()
        pot_fmm = flat.evaluate(pts, q, compute_forces=False)
        t_fmm = time.perf_counter() - t0

        rel_l2 = float(np.linalg.norm(pot_fmm - pot_direct) /
                       max(1e-300, np.linalg.norm(pot_direct)))
        speedup = t_direct / t_fmm if t_fmm > 0 else float("inf")
        rows.append({"N": n, "direct_ms": t_direct * 1000.0,
                     "fmm_ms": t_fmm * 1000.0, "speedup": speedup,
                     "rel_l2": rel_l2})
        print(f"N={n:>6d}  direct={t_direct*1000:10.2f} ms  fmm={t_fmm*1000:10.2f} ms  "
              f"speedup={speedup:5.2f}x  rel_l2={rel_l2:.3e}")

        # Projection guard: if direct at this N already exceeds the budget,
        # and this is not the last N, drop the remaining schedule to the
        # plan's fallback (16000) once.
        if t_direct > direct_budget_s and n != ns[-1]:
            print(f"  direct exceeded {direct_budget_s:.0f}s budget at N={n}; "
                  f"dropping largest N to 16000 per plan")
            ns = ns[:ns.index(n) + 1] + ([16000] if 16000 not in ns else [])
            # continue with the new schedule

    # Honest takeaway: state the observed crossover N (or its absence up to
    # N_max) with the per-N ratios so the trend is visible.
    faster = [r for r in rows if r["speedup"] > 1.0]
    if faster:
        first = next(r for r in rows if r["speedup"] > 1.0)
        takeaway = (f"Crossover observed at N={first['N']}: flat FMM becomes "
                    f"faster than direct O(N^2) at N={first['N']} "
                    f"(speedup {first['speedup']:.2f}x, rel_l2 {first['rel_l2']:.2e}). "
                    f"Per-N speedup: " +
                    ", ".join(f"N={r['N']}->{r['speedup']:.2f}x" for r in rows) + ".")
    else:
        takeaway = (f"No crossover up to N_max={rows[-1]['N']}: the flat FMM is "
                    f"still slower than direct O(N^2) at every N tested. Per-N "
                    f"speedup (direct/fmm, <1 means FMM slower): " +
                    ", ".join(f"N={r['N']}->{r['speedup']:.2f}x" for r in rows) +
                    ". The asymptotic win exists but needs larger N or a "
                    "compiled kernel (see docs/GPU_NOTES.md).")
    print(f"\nTakeaway: {takeaway}")
    return rows, takeaway


if __name__ == "__main__":
    run_core_fmm_variants()
    run_scaling()
