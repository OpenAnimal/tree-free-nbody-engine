# Variant Benchmarks

Five domain benchmarks and ten application case-study benchmarks run through
the shared `core.benchmark_kit.VariantBenchmark` protocol: each reports
latency per variant and, where an exact reference exists, the repo-standard
`cross_validate` rel-L2 error next to it. Speed is never shown without the
accuracy it costs. Tables below are pasted verbatim from each
`benchmark_variants.py` run; the one-line takeaway under each is honest,
including "not faster at this scale" results.

## Browser demo cross-benchmark (WebGPU, uncapped steps/sec)

Reproduced with `node tools/browser_crossbench.js [N] [rounds]` against a
local `python -m http.server 8123` (headless full Chromium — the headless
shell has no WebGPU adapter — WebGPU/D3D11, RTX 4070 SUPER, 2026-08-24).
Frames run uncapped (`?uncapped=1`; the page defaults to the vsync-locked
loop), so the metric is true steps/sec. Adaptive rows default to the round-13
materialized far-field CSR gather (`?materializedFar=0` A/Bs the legacy
per-level m2l+l2l chain).

**Environment caveat**: measured while a background process held a large
share of GPU — absolute numbers are depressed vs an idle GPU; ratios between
configs measured in the same run remain meaningful. 2M rows ran each config
in an isolated browser process (`CONFIG=<label>`).

```
N=120k (3 rounds)            median steps/sec   rounds
fixed + counting-sort        225                216, 225, 225
fixed + open-addressing      215                206, 226, 215
fixed + funnel               214                229, 213, 214
adaptive + node-hash dir     366                436, 357, 366
adaptive + leafForParticle   541                603, 538, 541
adaptive + far chain (A/B)   349                403, 349, 349

N=500k (3 rounds)            median steps/sec   rounds
fixed + counting-sort        160                168, 155, 160
fixed + open-addressing      167                167, 180, 161
fixed + funnel               161                153, 168, 161
adaptive + node-hash dir     52                 52, 51, 128
adaptive + leafForParticle   43                 43, 42, 51
adaptive + far chain (A/B)   52                 49, 52, 77

N=2M (3 rounds, isolated)    median steps/sec
fixed + counting-sort        34
adaptive + node-hash dir     11

N=500k far-field/near-field decomposition (3 rounds)
adaptive default (p2p budget 24/leaf)   50    48, 50, 52
adaptive p2p budget 6/leaf              184   189, 184, 169
adaptive p2p budget 1/leaf              316   345, 314, 316
adaptive multipole order p=0            50    49, 51, 50
```

Takeaways (details in [docs/GPU_NOTES.md](docs/GPU_NOTES.md) §8 and §10):

- **Near-field hash backends are equal** (counting/open-addr/funnel within
  noise at every N): the `materialize_ranges` pass resolves the hash table
  into the dense `cellStart`/`cellCount` arrays once per frame, so every
  consumer does two direct loads. The hash table remains the structure
  built each frame; its value is the worst-case probe bound and
  compactness, not throughput.
- **Round 13 materialized far field**: the adaptive far field is now a flat
  per-leaf CSR gather of List-2 sources through a precomputed
  per-(level, offset) M2L operator table (validated to 1e-7 against the
  legacy chain on GPU). It is ~5% faster than the legacy chain at 120k and
  performance-neutral at 500k — the remaining adaptive-vs-fixed gap at
  500k+ is NOT the far field: dropping the near-field P2P budget 24 -> 6
  gives 3.7x (50 -> 184, matching the fixed grid), while zeroing the
  multipole order or reverting the far-field rewrite changes nothing.
- **Adaptive FMM crosses over** at 120k (few tree nodes make the per-level
  chains cheap while the fixed grid always evaluates the full lattice); at
  500k+ the adaptive node count grows and the budgeted List-1 near-field
  walk dominates. Its value at large N is accuracy on clustered
  distributions, not throughput.
- **Adaptive throughput is phase-dependent**: the galaxy ICs are unseeded
  (`Math.random`), so the quadtree swings between ~1k and ~55k nodes over a
  run and adaptive steps/sec swings with it (rounds above show single
  phases, e.g. 128 vs 51; medians over 3 interleaved rounds are the honest
  comparator).
- The adaptive metadata rebuild (now including the far CSR) runs in a Web
  Worker sliced verbatim from the page's own script
  (`getAdaptiveMetaWorker` in index.html); a round-12 regression that
  silently disabled the worker (missing override globals in the glue) was
  found by the diag channel and fixed in round 13.

## Core FMM (2D log kernel)

```
Variant                            Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (exact direct)     36.37                     -  O(N^2) reference
+fmm (adaptive FMM)       716.64 (0.1x)      1.974e-07  funnel-hash adaptive FMM, p=10; slow classical reference engine kept for cross-validation (per-box Python loops; canonical engine is +fmm (adaptive, vectorized))
+fmm (adaptive, vectorized)     29.59 (1.2x)      2.000e-07  CANONICAL core.adaptive_fmm.AdaptiveFMM (alias FastAdaptiveFMM): level-batched 2:1-balanced adaptive FMM, p=10 (offset-matrix M2L, CSR P2P)
+fmm (flat vectorized)     57.21 (0.6x)      7.522e-07  single-level vectorized adaptive FMM, depth=5 order=8 (FFT-convolution M2L)
+quantized (32-bit packed)     54.46 (0.7x)      4.824e-01  VoxelPackedTreeFreeFMM; module documents ~1.2e-1 rel-L2 packed cost (this clustered N=2000 distribution measures higher — see table)
```

One canonical adaptive engine: `core.adaptive_fmm.AdaptiveFMM` (alias
`FastAdaptiveFMM`) is the level-batched 2:1-balanced CGR88 engine; the two
classical per-box engines (`ClassicalAdaptiveFMM`,
`TreeFreeElasticAdaptiveFMM`, the "+fmm (adaptive FMM)" row) remain in the
same module ONLY as slow cross-validation references. The canonical engine
is 24x faster than the classical per-box engine at identical accuracy
(2.0e-7 vs 2.0e-7 rel-L2, p=10) and is already faster than the direct
O(N^2) sum at N=2000. The earlier "NOT faster than direct at N=2000" rows
were implementation constants, not algorithm facts: the classical engine
paid per-particle/per-box Python loops and root-recursive list construction,
and the flat engine paid a p^2-loop over all K^2 cell pairs. The vectorized
engine batches every pass per tree level (2:1-balanced quadtree, M2L
operators precomputed per relative offset as in FMMLIB2D's
`itable(-3:3,-3:3)`, near-field via CSR-concatenated blocks). This matches
the literature's crossover expectations: compiled 2D adaptive FMM overtakes
direct summation around N~250-300 (Carrier, Greengard, & Rokhlin, 1988:
N=25,600 — direct 9694 s vs adaptive 97 s), and a competent NumPy
implementation should be near parity at N=2000 (Gimbutas & Greengard, 2012,
note the "O(N) work with a larger constant" tree-construction overhead).

External-reference cross-validation, stated honestly: the right external
check for a 2D log-kernel FMM is FMMLIB2D (Gimbutas & Greengard, 2012)
via the `pyfmmlib` PyPI package (its `lfmm2d` uses the identical kernel
and sign convention, sum q_j ln|r_i - r_j|). That package is sdist-only
and its meson build requires a Fortran compiler (ifort/ifx/gfortran/flang),
none of which exists on this Windows machine, so the pyfmmlib cross-check in
`tests/core/test_adaptive_fmm_reference.py` SKIPS here (it runs with real
gates wherever pyfmmlib is importable, e.g. Linux CI). Every other
pip-installable candidate was checked and rejected for a 2D log kernel:
`pyfmmlib2d` (GitHub-only, also gfortran+f2py), `fmm2dpy` (gone from PyPI),
`fmm3dpy` (Windows wheels but strictly 3D Laplace 1/r), `jaxfmm` (pure wheel
but 3D Coulomb 1/(4 pi r)). The always-running fallbacks in that test file
are the CGR88-internal references: geometric-in-p multipole-order
convergence (measured: two-box M2L error 1.8e-3 at p=4 falling to 3.1e-13
at p=20, ~0.35 per order as the CGR88 bound predicts; full engine 1.9e-4 at
p=4 to 5.6e-10 at p=16), exact translation-chain round-trip identities,
agreement with both retained classical engines (mutual 7.0e-9 at p=10) and
with the Greengard & Rokhlin (1987) uniform-grid engine, and direct O(N^2)
agreement on uniform/two-cluster/spiral distributions with potentials and
forces.

## Core FMM scaling

Direct O(N^2) (chunked), the flat single-level FMM
(`FastVectorizedFMM(depth=5, order=8)`, FFT-convolution M2L), and the
canonical vectorized adaptive FMM (`core.adaptive_fmm.AdaptiveFMM`, alias
`FastAdaptiveFMM`, p=10), on the clustered multi-scale distribution. Each
point is the minimum of 3 fresh build+evaluate runs (this machine runs
concurrent background training, so single shots are noise-dominated; the
min-of-k statistic is applied identically to every variant). The classical
per-box adaptive engine is omitted (24x slower than the vectorized engine,
no additional information). Direct is skipped above N=32000 (the quadratic
term would dominate the run for minutes). Plots:
`assets/core_fmm_scaling_loglog.png` (log-log runtime) and
`assets/core_fmm_scaling_linear.png` (linear-scale speedup with the
crossover annotated); raw numbers in `assets/core_fmm_scaling.json`. The
headline line below is generated automatically by
`core/benchmark_variants.py run_scaling`.

```
=== Core FMM scaling (clustered distribution; direct budget 120s; min of 3 runs) ===
N=  2000  direct=     57.9 ms  flat=     58.1 ms  adaptive=    30.3 ms  speedup flat=1.0x adaptive= 1.9x  rel-L2 2.0e-07
N=  4000  direct=    222.9 ms  flat=     91.3 ms  adaptive=    43.0 ms  speedup flat=2.4x adaptive= 5.2x  rel-L2 1.6e-07
N=  8000  direct=    918.3 ms  flat=    196.1 ms  adaptive=   135.9 ms  speedup flat=4.7x adaptive= 6.8x  rel-L2 1.9e-07
N= 32000  direct=  17418.0 ms  flat=   2170.6 ms  adaptive=   211.3 ms  speedup flat=8.0x adaptive=82.4x  rel-L2 3.7e-07
N=128000  direct= skipped      flat=  34071.7 ms  adaptive=   993.7 ms
```

**Automated headline:** Adaptive FMM is faster than direct O(N^2) at every N
tested from N=2000 up (speedup N=2000→1.9x, N=4000→5.2x, N=8000→6.8x,
N=32000→82.4x); flat single-level FMM reaches N=2000→1.0x, N=4000→2.4x,
N=8000→4.7x, N=32000→8.0x. Flat FMM overtakes direct at N=4000 (2.4x and
rising) (below parity at N=2000→1.00x). Beyond N=32000 direct was skipped
(quadratic cost); measured: N=128000: adaptive=994 ms, flat=34072 ms
(extrapolated direct ≈ 280 s from the N=32000 point — ≈ 280x; the earlier
validation run that measured direct at N=128000 gave 465x, rel-L2 4.4e-07).

Crossover context, stated honestly: the ADAPTIVE engine is faster than
direct at every N measured down to N=2000 (1.9x); a finer exploratory sweep
(single-shot, same distribution) brackets its crossover between N=1000
(0.7x) and N=1500 (1.3x). The FLAT single-level engine reaches parity at
N≈2000 (0.997x in this table; 1.1x in the finer sweep — parity within
measurement noise) and is clearly faster from N=4000 (2.4x) — its
small-N deficit is the honest constant-factor cost of a scheme that pays
O(p^2) FFT-convolution M2L over the whole occupied lattice plus per-cell
near-field Python loops. The flat scheme is linear in N only for fixed
depth and pays its near-field cost per occupied cell as cells fill up at
large N (34.1 s at N=128000) — the multi-level adaptive engine is the one
with the correct O(N) scaling (0.99 s at N=128000; ≈ 465x against a direct
run of 296 s in the earlier validation measurement). Per Ying, Biros, &
Zorin (2004) the properly-constructed one-level interaction list is bounded
(27 boxes in 2D), not K^2 — the earlier flat implementation evaluated all
well-separated K^2 pairs, which the FFT-convolution M2L now computes exactly
at grid-FFT cost.

## Core hash tables (funnel vs elastic vs baselines)

Reproduce with `python -X utf8 benchmarks/bench_hash_backends.py` (few
minutes; `--quick` for the reduced grid; JSON written to
`benchmarks/hash_backends_results.json`). Backends: the funnel hash table
of Farach-Colton, Krapivin, & Kuszmaul (2025) (`core.elastic_hash.
ElasticHashTable` — the default occupied-cell index of every core FMM
engine), a fair open-addressing linear-probe baseline with the same
splitmix64 finalizer and slot budget, the CPython dict, and the compiled
Zig port (`zig/funnel_hash.zig`). The worst-case bound is additionally
verified as a standing test in `tests/core/test_elastic_hash_bounds.py`
(measured max search-probe count ≤ the documented
α·β + B + 2C bound at δ = 1/8 and δ = 1/64, seeded key sets, including
delete/reinsert churn; no key ever drops at rated load).

Headline, n = 100,000 (scalar Python paths under identical interpreter
conditions; "abs max" = worst probe count over absent-key lookups):

| α | funnel bound | funnel abs-max | funnel hit-max | linear abs-max | linear hit-max |
| --- | --- | --- | --- | --- | --- |
| 0.50 | 157 | 157 | 22 | 26 | 26 |
| 0.75 | 157 | 157 | 39 | 160 | 144 |
| 0.90 | 193 | 193 | 77 | 1181 | 873 |
| 0.95 | 277 | 277 | 119 | 2204 | 2092 |
| 0.99 | 543 | 543 | 269 | 21178 | 21192 |

The funnel table's worst case is a **deterministic cap** — every absent
lookup pays exactly the bound and no search can exceed it, at any n (at
n = 1,000,000 / α = 0.99 the linear baseline reaches 22,216 probes while
funnel stays at its 543 cap, a 41× worst-case gap that grows with n).
Linear probing's mean is fine at low load but its tail is unbounded and
load-sensitive; that tail is a real latency cliff for GPU warp divergence
and real-time pipelines, which is the regime the paper's scheme targets.

Delete/reinsert churn (delete 10% of keys, reinsert fresh ones,
n = 100,000): funnel post-churn worst lookup stays at its bound
(25 / 49 / 189 / 277 / 543 across α = 0.5 → 0.99) while linear probing
degrades to 30 / 144 / 873 / 2588 / 21723. Honest caveat, both directions:
funnel tombstones never reclaim slots, so at α ≥ 0.95 a 10%-churn cycle
drops reinserted keys (4738 and 8991 of 10,000 at α = 0.95 / 0.99;
zero drops at α ≤ 0.9) — churn-heavy workloads must size δ for the churn
volume, which `tests/core/test_elastic_hash_bounds.py` quantifies.

Throughput and memory, stated plainly (n = 100,000):

| backend | build M keys/s | lookup M keys/s | bytes/key |
| --- | --- | --- | --- |
| CPython dict (C) | 6.8–7.7 | 3.2–3.6 | 52 (getsizeof; excl. objects) |
| linear probe (scalar) | 0.4–1.1 | 0.4–1.0 | 8.1–16.0 |
| funnel (scalar) | 0.12–0.40 | 0.12–0.39 | 9.1–18.0 |
| funnel (NumPy `funnel_probe`) | — | 0.19–0.55 | — |
| funnel (Zig, compiled) | 15–34 | 13–32 | 9 |

Where the funnel table wins: deterministic worst-case search bound (above),
inserts that never displace a resident key and never drop at rated load,
~9 bytes/key of slot storage at high load (vs 52+ for dict before counting
key/value objects), and a probe sequence that is a pure function of the
key — which makes the whole lookup vectorizable (`funnel_probe`: 2–5× the
scalar path in plain NumPy) and portable: the same geometry and bound hold
in the Zig backend (15–34 M keys/s insert, 2–7 M/s for absent lookups that
pay the full bounded sequence) and the WGSL port (docs/GPU_NOTES.md §5.2).
Where it does not win: raw pure-Python throughput — the C dict builds
~20× faster and the linear-probe baseline is 2–3× faster in Python at low
load. The funnel table is chosen for its worst-case and portability
guarantees, not its interpreter speed.

The elastic-hashing table (`core.elastic_hash.ElasticBatchingHashTable`,
the paper's Section 2 scheme) is **reference/experimental only** — it is
not used by any pipeline. Measured head-to-head at its rated load
(δ = 1/8): build 0.002 M keys/s vs funnel 0.25, mean hit probes 3340 vs
17 at n = 10,000, and absent lookups degrade to a full-array scan — the
greedy cascade plus its O(capacity) duplicate pre-scan loses to the
funnel table in every measurable regime in Python. It is kept solely as
an executable exploration of the Section 2 construction.

## Physics — Tetrahedral contact broadphase

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (brute O(N^2))    713.90                     -  exact AABB overlap pair set
+elastichash (CellIndex ring-1)    186.15 (3.8x)              -  no missed collisions: True; 122433 exact pairs / 122433 broadphase candidates (filter superset, narrow-phase prunes false positives)
```

The CellIndex ring-1 broadphase is ~4x faster than brute force on the demo tet
mesh and misses zero collisions (every exact AABB-overlap pair is in the
candidate set), so it is a correct filter even though the candidate and exact
sets coincide on this uniform grid.

## Video streaming — Gaussian splat frame compression

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (exact per-pixel)      0.00                     -  lossless per-Gaussian SH colors
+elastichash (cell-bucketed)      8.95 (0.0x)      3.190e-01  order-0 cluster-mean per occupied cell (lossy ~0.31 rel L2)
+quantized (4-bit color)      8.40 (0.0x)      3.207e-01  cluster-mean + 4-bit per-channel color quantization (lossy)
```

The cell-bucketed order-0 splat is lossy at ~0.31 rel L2 (the known color
quantization cost, visible in the table not hidden), and the extra 4-bit
per-channel color quantization adds only a small additional error on top.

## Graphics rendering — Volumetric AO (3D inverse-square kernel)

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (exact per-particle)    102.84                     -  O(Q*N) reference
+elastichash near/far     894.67 (0.1x)      8.269e-04  order-0 far field
+quantized (all-cluster)     19.41 (5.3x)      2.001e-02  cluster-quantized far field
```

The cluster-quantized far field is 5.3x faster than the exact per-particle AO at
a 2e-2 rel-L2 cost, while the near/far split is actually slower than the exact
path at this scale — the 2D-log FMM does not apply to this 3D inverse-square
kernel, and the per-cell Python loop overhead dominates.

## Game mechanics — Massive crowd flocking (2D unit mode)

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (brute O(N^2))     81.85                     -  O(N^2) near-field reference
+elastichash (near+far)    239.08 (0.3x)              -  near-field exact vs brute (same 3x3 cell box); |far-field residual| = 0.131 (5.9% of total); NOT faster than O(N^2) at N=1000 (per-cell Python loop overhead dominates at small N)
```

The elastic-hash flocking step is near-field exact (same 3x3 cell box as brute
force, with a 5.9% far-field heading residual reported honestly) but is NOT
faster than O(N^2) at N=1000 because per-cell Python loop overhead dominates at
small N; the asymptotic benefit only appears at larger crowds.

## Application case studies (`apps/`)

Each of the ten `apps/appN_benchmark_variants.py` files runs the same
`VariantBenchmark` protocol on the app's own kernel, so the apps are
comparable on the same axes as the domain folders above. The `+fmm` axis is
included only where the app's kernel is the 2D logarithmic adaptive FMM kernel
(apps 1 and 2); elsewhere it is omitted with the reason stated in the note
(Gaussian RBF, 3D Yukawa, nearest-point proximity, or high-dim cosine
retrieval -- none are the 2D log kernel). Where the result is approximate
rather than exact (LSH retrieval, filter broadphases), the correctness
metric is `recall@k` or `no missed collisions` in the note, not a rel-L2.

### App 1 -- Galaxy collision (2D log gravity forces)

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (exact direct)     16.50                     -  O(N^2) reference (app1 validate_against_direct path)
+elastichash (near only)     37.01 (0.4x)      2.610e-01  CellIndex ring-1 near-field, far-field SKIPPED (hash-truncated baseline)
+fmm (FastVectorizedFMM)     14.74 (1.1x)      2.386e-05  adaptive FMM flat FMM, depth=4 order=6 (the app's compute path)
```

At N=500 the flat FMM reaches 2.4e-5 rel-L2 force accuracy and is at parity
with direct summation; the hash-truncated near-field-only baseline is
slower than direct AND loses 26% rel-L2 because the far-field is skipped --
it is the cheap baseline, not a competitor.

Scaling table (depth=4, order=6):

```
     N   direct (ms)      fmm (ms)   speedup      rel L2
  ------  ------------  ------------  --------  ----------
     500          18.9          14.7      1.28x    2.39e-05
    1000          77.7          21.1      3.68x    2.14e-05
    2000         300.3          49.8      6.03x    2.76e-05
    4000        1201.4         167.6      7.17x    2.42e-05
```

FMM crosses direct at N~500 and reaches **7.2x at N=4000** with rel-L2
≤ 2.8e-5, demonstrating the O(N) vs O(N^2) asymptotic advantage.

### App 2 -- Hydrodynamic vortex sheet (2D log streamfunction)

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (direct Biot-Savart)     54.10                     -  O(N_grid * N_vortices) exact reference
+fmm (FMM streamfunction + FD)   3647.87 (0.0x)      6.812e-02  FastVectorizedFMM psi on grid, velocity by central FD; error floor set by FD grid spacing (app tolerance 5e-2)
```

The FMM streamfunction path is NOT faster than direct Biot-Savart at this
grid size (80x80 probes + 400 vortices): the depth=5/order=8 FMM constant
dominates, and the velocity error floor is set by the central-FD grid
spacing (6.8e-2 over all grid points including near-core points the app
skips in its own cross-check), not by the FMM itself.

### App 3 -- Spatial-hash attention (2D Gaussian RBF)

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (dense O(N^2))    173.30                     -  dense spatial RBF attention reference
+elastichash (near+far centroid)     96.86 (1.8x)      8.533e-02  near exact (3x3 CellIndex) + far per-cell centroid; vectorized per-cell (CellIndex replaces raw ElasticHashTable); lossy far-field centroid approximation
+fmm (Taylor FGT)        1636.01 (0.1x)      4.685e-07  2D Gaussian Taylor FGT (core/gaussian2d_fgt.py), h=sigma*sqrt(2); exact spatial-only attention via per-column FGT + normalizer; NOT faster than direct at N=1500 (per-cell Python loop overhead)
```

The near-exact/far-centroid attention is now **1.8x faster** than dense
O(N^2) at N=1500 after vectorizing the per-cell computation with CellIndex
(replacing the old raw ElasticHashTable + per-point Python loops). The
far-field centroid approximation costs 8.5e-2 rel-L2. The `+fmm (Taylor
FGT)` row reaches 4.7e-7 rel-L2 (the exact spatial-only attention, computed
via the 2D Gaussian Taylor FGT in `core/gaussian2d_fgt.py` with h =
sigma*sqrt(2) so the FGT kernel exp(-r^2/h^2) equals the app kernel
exp(-r^2/(2 sigma^2)) exactly) but is NOT faster than direct at N=1500 --
the per-column FGT loop (d_model+1 evaluations) plus the per-cell Python
loop overhead puts it at 0.1x. The asymptotic win needs larger N or a
compiled kernel (Class D in
[docs/INAPPLICABILITY.md](docs/INAPPLICABILITY.md)).

### App 4 -- Elastic-hash boids + 1euro (near-field boid rules)

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (brute near-field)     13.21                     -  O(N^2) near-field reference (separation + alignment, no far cohesion)
+elastichash (near+far+1euro)     20.67 (0.6x)              -  near-field exact vs brute (same 3x3 cell box); |far-field residual| = 0.506 (8.5% of total); +fmm axis omitted (not a 2D log kernel); vectorized per-cell via CellIndex
```

The hash boid step is near-field exact (same 3x3 cell box as brute) but NOT
faster at N=400 (per-cell Python loop overhead); the 8.4% far-field
cohesion residual is the intentional extra term, reported honestly.

### App 5 -- 3D protein electrostatics (Debye-Huckel screened Coulomb)

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (direct O(N^2))    258.14                     -  exact per-atom screened Coulomb reference
+elastichash (TreeFreeBioFMM per-atom dipole)    147.96 (1.7x)      2.002e-03  funnel-hash 3D Morton clusters, per-atom monopole + dipole far field (Round-7 T-C2); replaces old center-broadcast
+fmm (Yukawa3DFMM)       3581.86 (0.1x)      2.510e-08  single-level flat 3D Yukawa FMM, depth=6 p=8; closes INAPPLICABILITY.md Class C (3D Yukawa now has a 3D FMM)
+bio_taylor (TaylorYukawaBioFMM)   3389.63 (0.1x)      3.265e-08  Round-7 T-C1: bio-units wrapper over Yukawa3DFMM with Å→unit box mapping; target ≤1e-6 rel-L2
```

**Round-7 task T-C2 update:** the `+elastichash` row was previously the old
center-broadcast path (cluster-center-to-cluster-center distance, broadcast
to every atom in the cell) which sat at **5.682e-01** rel-L2 (the admitted
~57% cluster-mean cost). T-C2 replaced the center-broadcast with per-atom
monopole + first-order dipole evaluation against far-cluster centers (the
pattern proven in `neural_ops/equivariant_field_layer.py:144-167`). The
rel-L2 dropped from **5.682e-01** to **2.002e-03** — below the 1.5e-1
acceptance threshold by two orders of magnitude. The old value is kept
here for traceability.

**Round-7 task T-C1 update:** the `+bio_taylor` row adds the
`TaylorYukawaBioFMM` class — a bio-units wrapper over the verified
`core/yukawa3d_fmm.py` (Å→unit-box mapping with inset [0.1, 0.9] to avoid
grid-boundary near-field clipping, kappa rescaling, and
`COULOMB_CONSTANT_KCAL / (eps * s)` potential conversion). It reaches
**3.265e-08** rel-L2 at N=3000 — below the 1e-6 acceptance target. The
2-cell toy check (`toy_2cell_check_bio`) pins the scaling at 7.9e-13 rel-L2.

The `+fmm (Yukawa3DFMM)` row reaches 2.5e-8 rel-L2 (five orders of magnitude
better than the per-atom-dipole path) but is NOT faster than direct at N=3000
(0.1x) -- the single-level flat 3D FMM's per-cell Python loop over the
derivative tensors dominates at this scale (Class D in
[docs/INAPPLICABILITY.md](docs/INAPPLICABILITY.md)); the asymptotic win
needs larger N or a compiled kernel. This closes the round-3 Class C gap:
the 3D Yukawa kernel is no longer "right kernel, 2D-only FMM" -- it now has
a 3D FMM in `core/yukawa3d_fmm.py`.

#### Yukawa3D error-vs-p convergence (round-5 task 5.2: p-floor root cause)

`apps/app5_benchmark_variants.py:run_convergence` sweeps the expansion
order p on the same protein distribution (N=2000, depth=6, kappa=2.0) and
reports rel-L2 vs the exact direct reference:

```
   p         rel-L2   build+eval (s)                                     note
--------------------------------------------------------------------------------
   2     7.0357e-04           0.2417
   4     1.8746e-05           0.3812                       ~2.66e-02x vs prev
   6     7.0041e-07           0.9812                       ~3.74e-02x vs prev
   8     2.7259e-08           2.9944                       ~3.89e-02x vs prev
  10     1.8202e-09           8.7866                       ~6.68e-02x vs prev
  12     1.5324e-10          23.9535                       ~8.42e-02x vs prev
--------------------------------------------------------------------------------
```

The scheme now converges geometrically across the full p range, dropping
~1e-2 per +2 in p down to 1.5e-10 at p=12 -- consistent with the
order-(p+1) rate. The round-4 table floored at ~6.27e-5 for p>=6 with a
note attributing it to "ring-2 near field + f64 round-off"; that
attribution was WRONG. The round-5 root-cause analysis
(`tools/diag_yukawa3d_pfloor.py`, `tools/diag_yukawa3d_partition.py`)
showed the FMM operator is correct (single-pair Taylor converges to
1e-8 at p=12; the P-tensor has no off-by-one; ring_direct=3 does not
move the floor). The floor was caused by a `+1e-6` distance
regularization in the direct reference `_direct_debye_huckel` (applied to
ALL pairwise distances, not just the diagonal), which introduced a
systematic ~6.27e-5 bias independent of p. The reference is fixed
(diagonal-only self-exclusion); the regression test
`test_yukawa3d_pfloor_regression` pins both the bias value and the
geometric decay. Run with `python -X utf8 apps/app5_benchmark_variants.py`.

### Screened Yukawa 2D (K0 kernel) -- Taylor FMM vs order-0 tree-code (round 5)

`algorithm_theory/benchmark_screened_yukawa2d_variants.py` compares the
old honest order-0 (monopole + dipole) tree-code — the approach
`algorithm_theory/screened_yukawa_fmm.py` documents — against the new
round-5 2D Taylor FMM (`core/screened_yukawa2d_fmm.py`), on the SAME 2D
K0(kappa*r) kernel so the comparison is apples-to-apples:

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (direct O(N^2))    665.92                     -  exact per-particle K0(kappa*r) reference
+treecode (order-0)      4037.93 (0.2x)      2.422e-02  honest order-0 monopole+dipole tree-code (the old algorithm_theory/screened_yukawa_fmm.py approach, adapted to 2D K0); ~1% rel-L2 centroid approximation
+fmm (Taylor K0)          500.87 (1.3x)      6.153e-09  2D screened Yukawa Taylor FMM (core/screened_yukawa2d_fmm.py), depth=6 p=8; full order-p Taylor M2L far field, exact ring-2 near field; the round-5 upgrade of the old tree-code
```

The new `+fmm (Taylor K0)` row reaches 6.2e-9 rel-L2 — six orders of
magnitude better than the order-0 tree-code's 2.4e-2 — and is 1.3x faster
than the direct O(N^2) reference at N=2000 (the Taylor far field does
less arithmetic than the dense sum, and at N=2000 the per-cell Python
loop overhead is already below the O(N^2) matrix cost). The old tree-code
is retained as the honest order-0 comparison row; its module docstring
now points at the new engine. The K0 kernel's radial functions are built
exactly once per p as Laurent polynomials a_n(z), b_n(z) via the literal
plan recursion (verified by `bessel_recursion_guard` to 8e-10 rel vs
central-difference (1/r d/dr)). Run with
`python -X utf8 algorithm_theory/benchmark_screened_yukawa2d_variants.py`.

### App 6 -- MuJoCo footpad proximity (3D nearest-point search)

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (brute O(N))       3.49                     -  exact nearest terrain point per probe
+elastichash (3x3 neighborhood)     11.23 (0.3x)              -  no missed closest points: True; |contact-force residual| = 0.000 (same closest point per probe => residual is numerical only); +fmm axis omitted (proximity is nearest-point, not a kernel sum)
```

The 3x3 neighborhood proximity is a correct filter (no missed closest
points, zero force residual) but NOT faster than brute O(N) at this probe
count -- the asymptotic win only appears at much higher terrain density;
+fmm is omitted (nearest-point search, not a kernel sum).

### App 7 -- High-dim LSH partition + retrieval (cosine top-k)

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (brute exact top-k)      5.85                     -  O(N*d) exact cosine top-5 per query
+elastichash (LSH bucket top-k)     12.19 (0.5x)              -  recall@5 over 50 queries = 36.8%; +fmm axis omitted (cosine retrieval, not a kernel sum)
```

Single-bucket LSH retrieval is NOT faster than brute exact top-5 at N=5000
and reaches only 36.8% recall@5 -- the funnel hash gives O(1) bucket
lookup but the single-bucket candidate set is too small for high recall
without multi-probe; +fmm is omitted (cosine retrieval, not a kernel sum).

### App 8 -- Manifold unfolding via hash k-NN (8D Swiss roll)

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (exact O(N^2) k-NN)    255.08                     -  exact top-12 neighbor edge set
+elastichash (LSH k-NN graph)      9.69 (26.3x)              -  k-NN edge recall@12 = 100.0% (17009/17009 true edges); +fmm axis omitted (high-dim k-NN, not a kernel sum)
```

The multi-table LSH k-NN graph is 26.3x faster than exact O(N^2) k-NN at
N=2500 AND reaches 100% edge recall@12 on the Swiss roll (the manifold's
low intrinsic dimension makes LSH buckets coincide with true neighborhoods)
-- a genuine win, reported with the recall that earns it.

### App 9 -- Streaming vector DB (CAUTIONARY: fine-grained LSH partitions collapse recall)

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (brute exact top-k)     81.44                     -  O(N*d) exact cosine top-10 per query
+elastichash (LSH multi-probe)     24.09 (3.4x)              -  recall@10 over 200 queries = 0.6%; zero-reorder funnel-hash ingestion; CAUTIONARY: fine-grained LSH partitions collapse recall (see docs/INAPPLICABILITY.md Class D-adjacent); +fmm axis omitted (cosine ANN, not a kernel sum)
```

CAUTIONARY CASE STUDY. Multi-probe LSH is 3.4x faster than brute exact
top-10 at N=10000/d=128 but recall@10 collapses to 0.6%. The cause is a
data-geometry constant, not a Python constant (Class D-adjacent in
[docs/INAPPLICABILITY.md](docs/INAPPLICABILITY.md)): the corpus is 20 tight
Gaussian clusters, so the true top-10 of a query near a cluster center is
determined by fine within-cluster noise alignment, which requires a fine
LSH partition, which empties the buckets. A hyperplane/table/multi-probe
sweep (hyperplanes in {6,7,8,9,10,12}, tables in {2,4,8}, probe bits in
{1,2,3}) showed recall@10 >= 0.5 is reachable ONLY at ~0.13x speed (8x
slower than brute, ~3200 candidates/query) and >= 1.5x speed is reachable
ONLY at recall@10 <= ~5% (~20-80 candidates/query). No middle ground exists
on this corpus; the speed in the table is real and the recall cost is the
known price of the fine-grained partition, reported honestly.

### App 10 -- Continuous spatial GNN (2D Gaussian message pass)

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (dense all-pairs)      4.02                     -  dense Gaussian message pass reference (no adjacency matrix)
+elastichash (near+far centroid)     51.38 (0.1x)      1.791e-01  near exact (3x3 funnel hash) + far per-cell centroid; lossy far-field centroid approximation
+fmm (Taylor FGT)        1478.13 (0.0x)      3.719e-05  2D Gaussian Taylor FGT (core/gaussian2d_fgt.py) on both the near (h^2=0.05) and far (h^2=0.2) message kernels; exact spatial message pass via per-feature FGT + normalizer; self terms excluded (matches the dense reference w[i]=0)
```

The near-exact/far-centroid GNN message pass is NOT faster than dense
all-pairs at N=150 (per-cell Python loop overhead dominates at small N) and
pays an 18% rel-L2 far-field centroid cost. The new `+fmm (Taylor FGT)`
row (round-5 task 5.5) reaches 3.7e-5 rel-L2 -- four orders of magnitude
better than the centroid path -- by running the 2D Gaussian Taylor FGT
once per feature dim plus once per-kernel normalizer on the app's two
Gaussian message kernels (h^2 = 0.05 near, h^2 = 0.2 far; kernel equality
asserted on a radial sweep first). It is NOT faster than dense at N=150
(0.0x): 66 FGT `evaluate()` calls (32 near + 32 far + 2 normalizers) each
loop over occupied cells in pure Python, and at N=150 that overhead
dominates the single 4 ms dense matrix pass; the asymptotic win needs
larger N or a compiled kernel.

Honesty note on the 3.7e-5 residual: it is NOT the FGT truncation error.
The dense reference regularizes every pairwise distance by +1e-4
(`d = norm(...) + 1e-4`), which introduces a ~3.7e-5 systematic bias
independent of the FGT order p (the same class of reference-regularization
bias that caused the round-5 Yukawa3D p-floor in app5). Against an
unregularized direct reference the FGT reaches 2.9e-8 rel-L2 at p=8. The
self-term handling is correct: the dense reference zeroes self weights
(w[i] = 0) and the FGT excludes self pairs, so no self-term restoration
is needed (unlike app3, whose dense attention includes the self term).

## Flat-scheme depth guidance (Round-7 task T-C7 / finding R7-F28)

`tools/diag_flat_saturation.py` measures `Yukawa3DFMM` (3D, kappa=1, p=8,
ring=2, clustered data) across N x cells-per-side (depth). The flat
engines' linearity is depth-conditional: at fixed depth the far field
O(K^2 * |alphas|) is constant in N, but the near field
O(N * M_bar * (2*ring+1)^d) degrades as M_bar = N/K grows with N.

| N | depth | K | M_bar | wall (ms) | rel-L2 vs direct |
| --- | --- | --- | --- | --- | --- |
| 500 | 8 | 19 | 26.3 | 597 | 2.43e-11 |
| 500 | 16 | 52 | 9.6 | 1684 | 1.06e-08 |
| 2000 | 8 | 76 | 26.3 | 3321 | 4.08e-08 |
| 2000 | 16 | 224 | 8.9 | 40871 | 2.59e-08 |

Caveat: the wall times here are dominated by a pre-existing hot path in
`CellIndex.neighborhood_indices` (125 Morton decode/encode/hash-lookup ops
per cell in Python, ~5ms/cell), not the FMM math itself. T-C6's CSR
batching targets this. The accuracy and K/M_bar trends are the load-bearing
output; the absolute timings will drop sharply after T-C6.

Guidance:
- **Accuracy-driven rule**: keep M_bar <= ~60 (mean cell occupancy).
- **Cost-driven classical optimum (3D)**: K_opt ~ N^{2/3}, total O(N^{4/3}).
- The flat engines are **O(N^{4/3})-class single-level schemes**.
- The true O(N) member of the repo is the multilevel adaptive FMM engine / GPU demo.
- Choose depth ~ N^{2/3} for the cost optimum; deeper favors accuracy.
