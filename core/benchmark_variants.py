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


if __name__ == "__main__":
    run_core_fmm_variants()
