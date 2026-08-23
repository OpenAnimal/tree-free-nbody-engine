"""
Empirical Benchmark & Scaling Suite (validity-first revision)
=============================================================
Compares MEASURED timings only — no extrapolated data points:
1. Classical Naive Python CPU Direct O(N^2)         (N <= 2,000)
2. Vectorized NumPy Dense Matrix Direct O(N^2)      (N <= 8,000)
3. JAX JIT-Compiled Direct O(N^2)                   (optional, N <= 16,000)
4. Flat Tree-Free FMM + Elastic Hash (NumPy)        (all N)

The FMM curve is labeled honestly: the flat single-level scheme costs
O(N + K^2) with K occupied cells (K <= 4^depth), i.e. linear in N for a
fixed depth with a depth-dependent constant — NOT asymptotically O(N).

Before any timing, the suite cross-validates the FMM against the exact
direct O(N^2) summation and aborts if the error is too large: speedups
for wrong answers are not reported.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
try:
    import jax
    import jax.numpy as jnp
    from core.jax_tree_free_fmm import jax_direct_nbody
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    jax_direct_nbody = None
    HAS_JAX = False
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from core.fast_vectorized_fmm import FastVectorizedFMM
from core.adaptive_fmm import exact_direct_nbody_2d
from core.elastic_hash import ElasticHashTable

NAIVE_MAX_N = 2000
NUMPY_MAX_N = 8000
JAX_MAX_N = 16000


def validate_fmm_accuracy(n=2000, seed=123):
    """Accuracy gate: FMM vs exact direct summation (float64).

    Validates the SAME order the benchmark uses (order=4), not a higher
    order — the gate must verify the configuration being timed, not a
    more accurate one that would hide order-4 errors.  The threshold is
    1e-3 (not 1e-4) because order=4 on a uniform distribution measures
    ~3.5e-4 max relative error — the old 1e-4 threshold was calibrated
    for order=8 and would reject the correct order=4 result.  1e-3 gives
    ~3x headroom above the measured value while still catching real
    regressions (a broken FMM would be off by orders of magnitude).
    """
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.05, 0.95, size=(n, 2))
    q = rng.uniform(-1.0, 1.0, size=n)
    # Match the benchmark's order=4 (line ~130).  The old gate used order=8,
    # which passes at higher accuracy than what the benchmark actually times.
    fmm = FastVectorizedFMM(depth=4, order=4)
    pot = fmm.evaluate(pos, q)
    exact = exact_direct_nbody_2d(pos, q)
    rel = np.max(np.abs(pot - exact)) / np.max(np.abs(exact))
    print(f"[ACCURACY] N={n}, order=4: max relative potential error vs direct = {rel:.3e}")
    if rel >= 1e-3:
        print("[ACCURACY] FAIL — refusing to benchmark an incorrect FMM result.")
        raise AssertionError(f"FMM accuracy gate failed: rel err {rel:.3e} >= 1e-3")
    print(f"[ACCURACY] PASS (< 1e-3, measured {rel:.3e}) — benchmark may proceed.")


def run_comprehensive_benchmarks():
    print("=" * 82)
    print(" PERFORMANCE & SCALING BENCHMARK (measured points only)")
    print("=" * 82)

    validate_fmm_accuracy()

    if HAS_JAX:
        # JAX JIT compiles per concrete input shape: each new N triggers a
        # recompilation that would otherwise be billed to the first timed
        # run.  Warm EVERY JAX-timed shape once (un-timed, single iteration,
        # block_until_ready) so the JIT cache truly covers all sizes the
        # timing loop will hit.  Warming only N=128 and N=JAX_MAX_N (the old
        # behavior) left the other 7 timed shapes paying compile cost on
        # their first measured call.
        jax_warmup_sizes = [n for n in
                            [100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000]
                            if n <= JAX_MAX_N]
        for warmup_n in jax_warmup_sizes:
            dummy_pos = jax.random.uniform(jax.random.PRNGKey(0), (warmup_n, 2))
            dummy_q = jax.random.uniform(jax.random.PRNGKey(1), (warmup_n,))
            _ = jax_direct_nbody(dummy_pos, dummy_q).block_until_ready()
        print(f"[INFO] JAX warmed up (un-timed, 1 iter each) at N={jax_warmup_sizes}; "
              f"JIT cache now covers every JAX-timed benchmark shape.")
    else:
        print("[INFO] JAX is unavailable; JAX benchmark columns will be omitted.")

    test_sizes = [100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000]

    t_naive_cpu, t_numpy_direct, t_jax_direct, t_fmm_vectorized = [], [], [], []

    for n in test_sizes:
        print(f"\n[Evaluating N = {n} particles]")
        pos = np.random.uniform(0.05, 0.95, size=(n, 2)).astype(np.float32)
        q = np.random.uniform(-1.0, 1.0, size=n).astype(np.float32)

        # 1. Naive Python loop direct (measured only; NaN beyond cap)
        if n <= NAIVE_MAX_N:
            t0 = time.perf_counter()
            pot_naive = np.zeros(n)
            for i in range(n):
                d = pos[i] - pos
                r = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2) + 1e-12
                r[i] = 1.0
                pot_naive[i] = np.sum(q * np.log(r))
            t_naive = time.perf_counter() - t0
            print(f"  [-] Naive Python CPU O(N^2):        {t_naive*1000:.2f} ms")
        else:
            t_naive = np.nan
            print("  [-] Naive Python CPU O(N^2):        (not measured above N=2000)")
        t_naive_cpu.append(t_naive)

        # 2. Vectorized NumPy dense direct (measured only)
        if n <= NUMPY_MAX_N:
            t0 = time.perf_counter()
            diff = pos[:, None, :] - pos[None, :, :]
            r = np.linalg.norm(diff, axis=-1) + 1e-12
            np.fill_diagonal(r, 1.0)
            _ = np.sum(q[None, :] * np.log(r) * (1.0 - np.eye(n)), axis=-1)
            t_np_direct = time.perf_counter() - t0
            print(f"  [-] Vectorized NumPy Direct O(N^2): {t_np_direct*1000:.2f} ms")
        else:
            t_np_direct = np.nan
            print("  [-] Vectorized NumPy Direct O(N^2): (not measured above N=8000)")
        t_numpy_direct.append(t_np_direct)

        # 3. JAX JIT direct (optional, measured only)
        if HAS_JAX and n <= JAX_MAX_N:
            pos_j = jnp.array(pos)
            q_j = jnp.array(q)
            t0 = time.perf_counter()
            _ = jax_direct_nbody(pos_j, q_j).block_until_ready()
            t_jax = time.perf_counter() - t0
            print(f"  [-] JAX JIT Direct O(N^2):          {t_jax*1000:.2f} ms")
        else:
            t_jax = np.nan
            print("  [-] JAX JIT Direct O(N^2):          unavailable")
        t_jax_direct.append(t_jax)

        # 4. Flat Tree-Free FMM + elastic hash
        depth = 4 if n < 4000 else (5 if n < 32000 else 6)
        fmm_engine = FastVectorizedFMM(depth=depth, order=4)
        t0 = time.perf_counter()
        _ = fmm_engine.evaluate(pos, q)
        t_fmm = time.perf_counter() - t0
        t_fmm_vectorized.append(t_fmm)
        print(f"  [-] Flat Tree-Free FMM O(N + K^2):  {t_fmm*1000:.2f} ms  (depth={depth})")

    # --- Elastic hash load-factor stress (measured probes) ---
    print("\n[Elastic/Funnel Hash Stress Test (Farach-Colton, Krapivin, & Kuszmaul, 2025)]")
    for cap in [10000, 50000, 200000]:
        n_keys = int(cap * 0.92)
        h_table = ElasticHashTable(capacity=cap, delta=0.05)
        keys = np.random.randint(1, 1000000000, size=n_keys, dtype=np.int64)
        t0 = time.perf_counter()
        probes_tot = 0
        for k in keys:
            _, p = h_table.insert(int(k), int(k))
            probes_tot += p
        t_ins = time.perf_counter() - t0
        print(f"[-] Capacity {cap:7d} | {n_keys:7d} keys (92% load) | Ingest: {t_ins*1000:6.1f} ms "
              f"({n_keys/t_ins:8.0f} ops/s) | Avg Probes: {probes_tot / n_keys:.2f}")

    # --- Plots (measured points only; NaNs masked out) ---
    ts = np.array(test_sizes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), facecolor='#0B0E14')

    def masked(ax, times, **kw):
        t = np.array(times, dtype=float)
        m = ~np.isnan(t)
        ax.plot(ts[m], t[m] * 1000, **kw)

    ax1.set_facecolor('#0B0E14')
    masked(ax1, t_naive_cpu, marker='o', ls='--', color='#FF5555', lw=1.8,
           label='Naive Python Direct $O(N^2)$ (measured)')
    masked(ax1, t_numpy_direct, marker='s', ls='--', color='#FF79C6', lw=1.8,
           label='Vectorized NumPy Direct $O(N^2)$ (measured)')
    masked(ax1, t_jax_direct, marker='d', ls='--', color='#BD93F9', lw=2.0,
           label='JAX JIT Direct $O(N^2)$ (measured)')
    masked(ax1, t_fmm_vectorized, marker='^', ls='-', color='#50FA7B', lw=2.5,
           label='Flat Tree-Free FMM $O(N+K^2)$ (measured)')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Particle Count $N$ (log scale)', color='#8B949E', fontsize=11)
    ax1.set_ylabel('Execution Latency (ms, log scale)', color='#8B949E', fontsize=11)
    ax1.set_title('Measured Scaling: Direct $O(N^2)$ vs. Flat Tree-Free FMM', color='white',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, which='both', color='#21262D', linestyle='-', lw=0.5)
    ax1.legend(facecolor='#161B22', edgecolor='#30363D', labelcolor='white', fontsize=9)

    ax2.set_facecolor('#0B0E14')
    with np.errstate(invalid='ignore'):
        speedup = np.array(t_numpy_direct, dtype=float) / np.array(t_fmm_vectorized)
    m = ~np.isnan(speedup)
    bars = ax2.bar([f"{n//1000}k" if n >= 1000 else str(n) for n, ok in zip(test_sizes, m) if ok],
                   speedup[m], color='#8BE9FD', edgecolor='#30363D', alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        if h >= 1.0:
            ax2.annotate(f"{h:.1f}x", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', color='#E6EDF3', fontsize=8, fontweight='bold')
    ax2.axhline(1.0, color='#FF5555', linestyle=':', lw=1.5, label='Parity (1.0x)')
    ax2.set_xlabel('Particle Count $N$ (measured comparisons only)', color='#8B949E', fontsize=11)
    ax2.set_ylabel('Speedup vs. Vectorized NumPy Direct', color='#8B949E', fontsize=11)
    ax2.set_title('Flat Tree-Free FMM Speedup (measured baseline only)', color='white',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', color='#21262D')
    ax2.legend(facecolor='#161B22', edgecolor='#30363D', labelcolor='white', fontsize=10)

    for ax in (ax1, ax2):
        ax.tick_params(colors='#8B949E')
        for spine in ax.spines.values():
            spine.set_color('#30363D')

    fig.suptitle("Measured Scaling Benchmark: Tree-Free FMM vs. Direct Baselines",
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "assets", "benchmark_scaling_analysis.png")
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"\n[-] Saved benchmark scaling figure to: {output_path}")


if __name__ == '__main__':
    run_comprehensive_benchmarks()
