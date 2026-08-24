"""Standardized variant benchmark for the FMM core (2D log kernel).

Variants:
  standard             — exact direct O(N^2) summation (the reference)
  +fmm (adaptive FMM) — TreeFreeElasticAdaptiveFMM(p=10), the classical
                          funnel-hash indexed adaptive FMM slow reference
                          engine kept for cross-validation
  +fmm (adaptive, vectorized) — core.adaptive_fmm.AdaptiveFMM(p=10) (alias
                          FastAdaptiveFMM): the canonical level-batched
                          2:1-balanced engine (offset-matrix M2L, CSR P2P)
  +fmm (flat vectorized) — FastVectorizedFMM(depth=5, order=8), single-level
                           FMM with FFT-convolution M2L on the elastic hash
  +quantized           — VoxelPackedTreeFreeFMM (32-bit coordinate packing);
                          the CALLER inputs are adapted to its
                          (positions, charges) -> (potentials, metrics) API;
                          the quantized module itself is NOT modified.

Acceptance: the table prints; both adaptive FMM variants rel-L2 < 1e-6.
Speed is reported as measured — no tuning to win — and run_scaling emits an
automated crossover headline plus log-log and linear-scale plots to assets/.
"""
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.benchmark_kit import VariantBenchmark


def _clustered_distribution(n: int = 2000, seed: int = 707):
    """Clustered, multi-scale distribution matching the spirit of
    test_flat_fmm_elastic_hash_occupancy in tests/core/test_adaptive_fmm_cross_validation.py:
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
    from core.adaptive_fmm import AdaptiveFMM as FastAdaptiveFMM
    from quantized_bitpacked_optimization.packed_vectorized_fmm import (
        VoxelPackedTreeFreeFMM,
    )

    pts, q = _clustered_distribution(n=n)

    # Pre-instantiate the FMM engines (build cost is part of the timed call,
    # matching how a caller actually uses them).
    adaptive = TreeFreeElasticAdaptiveFMM(p=10)
    fast_adaptive = FastAdaptiveFMM(max_leaf_particles=24, base_depth=2,
                                    max_depth=9, p=10)
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
        "+fmm (adaptive FMM)",
        lambda: adaptive.evaluate(pts, q, compute_forces=False),
        accuracy_vs="standard (exact direct)",
        note="funnel-hash adaptive FMM, p=10; slow classical reference engine kept for cross-validation (per-box Python loops; canonical engine is +fmm (adaptive, vectorized))",
    )
    bench.add(
        "+fmm (adaptive, vectorized)",
        lambda: fast_adaptive.evaluate(pts, q, compute_forces=False),
        accuracy_vs="standard (exact direct)",
        note="CANONICAL core.adaptive_fmm.AdaptiveFMM (alias FastAdaptiveFMM): level-batched 2:1-balanced adaptive FMM, p=10 (offset-matrix M2L, CSR P2P)",
    )
    bench.add(
        "+fmm (flat vectorized)",
        lambda: flat.evaluate(pts, q, compute_forces=False),
        accuracy_vs="standard (exact direct)",
        note="single-level vectorized adaptive FMM, depth=5 order=8 (FFT-convolution M2L)",
    )
    bench.add(
        "+quantized (32-bit packed)",
        lambda: quantized.evaluate(pts, q)[0],
        accuracy_vs="standard (exact direct)",
        note="VoxelPackedTreeFreeFMM; module documents ~1.2e-1 rel-L2 packed cost (this clustered N=2000 distribution measures higher — see table)",
    )
    rows = bench.run()
    # Regression gates: the table used to print accuracy without ever
    # asserting, so a silent accuracy regression could not fail the run.
    # Thresholds carry ~10x headroom above the measured values on this
    # clustered N=2000 distribution (adaptive ~2e-7, fast ~2e-7, flat ~7.5e-7).
    by_name = {r["variant"]: r for r in rows}
    adapt_row = by_name.get("+fmm (adaptive FMM)")
    fast_row = by_name.get("+fmm (adaptive, vectorized)")
    flat_row = by_name.get("+fmm (flat vectorized)")
    for row, gate in ((adapt_row, 1e-6), (fast_row, 1e-6), (flat_row, 1e-3)):
        if row is not None and "rel_l2" in row:
            rel = float(row["rel_l2"])
            assert np.isfinite(rel) and rel < gate, (
                f"{row['variant']} rel-L2 {rel:.3e} >= {gate} regression gate")
    print("accuracy regression gates: PASS "
          "(adaptive < 1e-6, adaptive-vectorized < 1e-6, flat < 1e-3)")
    return rows


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
        # zero out self terms for the diagonal block (vectorized: the self
        # pair for target row (k-lo) is source column k, i.e. the diagonal
        # of the submatrix r2[0:hi-lo, lo:hi]).
        rows = np.arange(hi - lo)
        cols = np.arange(lo, hi)
        r2[rows, cols] = 1.0
        pot[lo:hi] = np.sum(charges[None, :] * 0.5 * np.log(r2), axis=1)
        # subtract the self term that was included as q_i * 0.5 * log(1) = 0,
        # so no correction needed beyond zeroing r2 above.
    return pot


def run_scaling(ns=(2000, 4000, 8000, 32000, 128000), direct_budget_s=120.0,
                direct_max_n=32000, make_plots=True, reps=3):
    """Scaling sweep: direct O(N^2) vs flat FMM vs vectorized adaptive FMM.

    For each N in `ns` (clustered distribution, same generator as the N=2000
    table), variants:
      standard (direct)         -- chunked vectorized direct O(N^2); skipped
                                   above `direct_max_n` (the quadratic term
                                   would dominate the run for minutes)
      +fmm (flat vectorized)    -- FastVectorizedFMM(depth=5, order=8)
      +fmm (adaptive, vectorized) -- core.adaptive_fmm.AdaptiveFMM (alias
                                   FastAdaptiveFMM; level-batched 2:1
                                   balanced CGR88, p=10)

    Each point is the MINIMUM of `reps` fresh build+evaluate runs (the
    machine runs concurrent background training, so single shots are
    noise-dominated; min-of-k is the standard interference-robust timing
    statistic and is applied identically to every variant).

    The classical per-box adaptive engine is OMITTED at these N: its Python
    tree traversal is ~45x slower than the vectorized engine at N=2000 and
    would only add wall-clock, not information.

    Emits the automated headline (measured crossovers for BOTH the adaptive
    and the flat engine and per-N speedups), a JSON record to
    assets/core_fmm_scaling.json, and two plots to assets/ (log-log runtime
    and linear-scale speedup vs direct with the crossover annotated)."""
    import json

    from core import FastVectorizedFMM
    from core.adaptive_fmm import AdaptiveFMM as FastAdaptiveFMM

    def best_of(fn):
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn()
            best = min(best, time.perf_counter() - t0)
        return best * 1000.0, out

    print(f"\n=== Core FMM scaling (clustered distribution; direct budget "
          f"{direct_budget_s:.0f}s; min of {reps} runs) ===")
    rows = []
    schedule = list(ns)
    i = 0
    dropped = False
    while i < len(schedule):
        n = schedule[i]
        pts, q = _clustered_distribution(n=n)
        flat = FastVectorizedFMM(depth=5, order=8)
        adaptive = FastAdaptiveFMM(max_leaf_particles=24, base_depth=2,
                                   max_depth=9, p=10)

        row = {"N": n}
        if n <= direct_max_n:
            row["direct_ms"], pot_direct = best_of(
                lambda: _direct_chunked(pts, q))
        else:
            pot_direct = None

        row["flat_ms"], pot_flat = best_of(
            lambda: flat.evaluate(pts, q, compute_forces=False))
        row["adaptive_ms"], pot_adapt = best_of(
            lambda: adaptive.evaluate(pts, q, compute_forces=False))
        row["cells"] = adaptive.n_cells

        # accuracy vs direct where direct ran
        if pot_direct is not None:
            row["rel_l2_flat"] = float(
                np.linalg.norm(pot_flat - pot_direct) /
                max(1e-300, np.linalg.norm(pot_direct)))
            row["rel_l2_adaptive"] = float(
                np.linalg.norm(pot_adapt - pot_direct) /
                max(1e-300, np.linalg.norm(pot_direct)))
            row["speedup_flat"] = row["direct_ms"] / row["flat_ms"]
            row["speedup_adaptive"] = row["direct_ms"] / row["adaptive_ms"]
        rows.append(row)

        d = (f"direct={row.get('direct_ms', float('nan')):10.1f} ms"
             if pot_direct is not None else "direct=      skipped")
        sp_f = row.get("speedup_flat")
        sp_a = row.get("speedup_adaptive")
        print(f"N={n:>6}  {d}  flat={row['flat_ms']:9.1f} ms  "
              f"adaptive={row['adaptive_ms']:8.1f} ms"
              + (f"  speedup flat={sp_f:6.1f}x adaptive={sp_a:6.1f}x"
                 f"  rel-L2 {row['rel_l2_adaptive']:.1e}"
                 if pot_direct is not None else ""))

        # Projection guard on the direct budget (only while direct runs)
        if (pot_direct is not None and row["direct_ms"] / 1000.0 > direct_budget_s
                and i < len(schedule) - 1 and not dropped):
            print(f"  direct exceeded {direct_budget_s:.0f}s budget at N={n}; "
                  f"dropping remaining schedule to [16000] per plan")
            kept = schedule[:i + 1]
            schedule = kept + ([] if 16000 in kept else [16000])
            dropped = True
        i += 1

    # --- automated headline (crossover + per-N ratios) ---------------------
    with_direct = [r for r in rows if "direct_ms" in r]
    without_direct = [r for r in rows if "direct_ms" not in r]
    faster = [r for r in with_direct if r["speedup_adaptive"] > 1.0]
    ratio_txt = ", ".join(
        f"N={r['N']}->{r['speedup_adaptive']:.1f}x" for r in with_direct)
    flat_txt = ", ".join(
        f"N={r['N']}->{r['speedup_flat']:.1f}x" for r in with_direct)
    if faster:
        first = faster[0]
        headline = (f"Adaptive FMM is faster than direct O(N^2) at every N "
                    f"tested from N={first['N']} up (speedup {ratio_txt}); "
                    f"flat single-level FMM reaches {flat_txt}.")
    else:
        headline = (f"No crossover up to N_max={with_direct[-1]['N']}: "
                    f"per-N adaptive speedup (direct/fmm): {ratio_txt}.")
    flat_first = next((r for r in with_direct if r["speedup_flat"] > 1.0), None)
    if flat_first is not None:
        prev = [r for r in with_direct if r["N"] < flat_first["N"]]
        prev_txt = (f" (below parity at "
                    + ", ".join(f"N={r['N']}->{r['speedup_flat']:.2f}x"
                                for r in prev) + ")") if prev else ""
        headline += (f" Flat FMM overtakes direct at N={flat_first['N']} "
                     f"({flat_first['speedup_flat']:.1f}x and rising)"
                     f"{prev_txt}.")
    elif with_direct:
        headline += (" Flat single-level FMM does not overtake direct within "
                     f"the measured range (max N={with_direct[-1]['N']}).")
    if without_direct:
        dmax = with_direct[-1]["N"] if with_direct else 0
        ext = "; ".join(f"N={r['N']}: adaptive={r['adaptive_ms']:.0f} ms, "
                        f"flat={r['flat_ms']:.0f} ms"
                        for r in without_direct)
        headline += (f" Beyond N={dmax} direct was skipped (quadratic cost); "
                     f"measured: {ext}.")
    print(f"\nAutomated headline: {headline}")

    # --- artifacts: JSON + plots -------------------------------------------
    out = {"rows": [{k: (float(v) if isinstance(v, (int, float, np.floating))
                         else v) for k, v in r.items()} for r in rows],
           "headline": headline}
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "assets", "core_fmm_scaling.json"),
                "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    except OSError as e:  # pragma: no cover - plot/json failures are non-fatal
        print(f"warning: could not write scaling JSON: {e}")

    if make_plots:
        _scaling_plots(rows)

    return rows, headline


def _scaling_plots(rows):
    """Log-log runtime plot + linear-scale speedup plot with annotated
    crossover, written to assets/core_fmm_scaling_{loglog,linear}.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("warning: matplotlib unavailable; skipping scaling plots")
        return

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ns = [r["N"] for r in rows]

    # log-log runtime vs N
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if any("direct_ms" in r for r in rows):
        nd = [r["N"] for r in rows if "direct_ms" in r]
        td = [r["direct_ms"] for r in rows if "direct_ms" in r]
        ax.plot(nd, td, "o-", label="direct O(N$^2$)", color="#444444")
    ax.plot(ns, [r["adaptive_ms"] for r in rows], "s-",
            label="adaptive FMM (vectorized, p=10)", color="#1a7f37")
    ax.plot(ns, [r["flat_ms"] for r in rows], "^-",
            label="flat single-level FMM (FFT M2L)", color="#7f1a7f")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N particles (clustered multi-scale)")
    ax.set_ylabel("evaluation time [ms]")
    ax.set_title("Core 2D FMM scaling (NumPy, one core)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(root, "assets", "core_fmm_scaling_loglog.png"),
                dpi=150)
    plt.close(fig)

    # linear-scale speedup vs direct with crossover annotation
    rows_d = [r for r in rows if "direct_ms" in r]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if rows_d:
        nd = [r["N"] for r in rows_d]
        ax.plot(nd, [r["speedup_adaptive"] for r in rows_d], "s-",
                label="adaptive FMM (vectorized)", color="#1a7f37")
        ax.plot(nd, [r["speedup_flat"] for r in rows_d], "^-",
                label="flat single-level FMM", color="#7f1a7f")
        ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1,
                   label="parity with direct")
        first = next((r for r in rows_d if r["speedup_adaptive"] > 1.0), None)
        if first is not None:
            ax.annotate(f"faster than direct\nfrom N={first['N']}",
                        xy=(first["N"], first["speedup_adaptive"]),
                        xytext=(first["N"] * 1.4, max(1.6, first["speedup_adaptive"] * 0.35)),
                        arrowprops=dict(arrowstyle="->", color="#1a7f37"),
                        fontsize=9, color="#1a7f37")
        ax.set_xscale("log")  # N spans decades; speedup axis stays linear
        ax.set_xlabel("N particles (clustered multi-scale)")
        ax.set_ylabel("speedup vs direct O(N$^2$) [x, linear scale]")
        ax.set_title("FMM speedup over direct summation (linear scale)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(root, "assets", "core_fmm_scaling_linear.png"),
                    dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    # Skippable via the SKIP_CORE_BENCH env var: the core benchmark runs the
    # full scaling sweep (direct O(N^2) at N=32000, ~36s) plus the adaptive
    # adaptive FMM engine on every invocation, which is wasteful in CI contexts
    # that only need the lint+sync+unit-test matrix.  Set SKIP_CORE_BENCH=1
    # to print "SKIP: ..." and exit 0 (tools/run_all.py treats this as a
    # legitimate SKIP, not a failure).  The full BENCHMARKS.md tables are
    # regenerated on demand by running without the env var.
    if os.environ.get("SKIP_CORE_BENCH") == "1":
        print("SKIP: SKIP_CORE_BENCH=1 set (core/benchmark_variants.py "
              "omitted — full scaling sweep + adaptive FMM engine; "
              "run directly to regenerate BENCHMARKS.md tables)")
        sys.exit(0)
    run_core_fmm_variants()
    run_scaling()
