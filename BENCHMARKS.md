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
shell has no WebGPU adapter — WebGPU/D3D11, RTX 4070 SUPER, 2026-08-23).
Frames run uncapped (`?uncapped=1`; the page now defaults to the
vsync-locked loop), so the metric is true steps/sec.

**Environment caveat**: measured while a background process held ~80% GPU
utilization — absolute numbers are depressed vs an idle GPU; ratios between
configs measured in the same run remain meaningful. 5M rows ran each config
in an isolated browser process (`CONFIG=<label>`; the fifth in-process
navigation stalls on cumulative GPU memory at that size), 1 round each.

```
N=120k (3 rounds)          median steps/sec   rounds
fixed + counting-sort      236                226, 236, 240
fixed + open-addressing    214                211, 230, 214
fixed + funnel             226                234, 226, 213
adaptive + node-hash dir   356                358, 349, 356
adaptive + leafForParticle 509                509, 517, 478

N=500k (3 rounds)          median steps/sec   rounds
fixed + counting-sort      172                172, 163, 181
fixed + open-addressing    174                178, 174, 173
fixed + funnel             172                172, 170, 175
adaptive + node-hash dir   56                 62, 56, 52
adaptive + leafForParticle 45                 45, 44, 45

N=5M (1 round, isolated)   median steps/sec
fixed + counting-sort      12   (~60M particle-updates/sec)
fixed + open-addressing    10
fixed + funnel             10
adaptive + node-hash dir   7
adaptive + leafForParticle 5
```

Takeaways (details in [docs/GPU_NOTES.md](docs/GPU_NOTES.md) §7):

- **Near-field hash backends are now equal** (counting/open-addr/funnel
  within noise at every N). A new `materialize_ranges` pass resolves the
  hash table into the dense `cellStart`/`cellCount` arrays once per frame —
  one probe per leaf cell instead of one probe per neighbor-cell visit in
  every P2P consumer — so the hash modes' hot loops are the same two direct
  loads as the counting sort. The hash table remains the structure built
  each frame; its value is the worst-case probe bound and compactness, not
  throughput.
- **Adaptive FMM crosses over**: at 120k it is *faster* than the fixed grid
  (few tree nodes make the per-level chains cheap while the fixed grid
  always evaluates the full 128x128 lattice); at 500k+ the adaptive node
  count grows (~23k nodes) and fixed wins. Its value at large N is accuracy
  on clustered distributions, not throughput.
- **The funnel occupied-node directory pays off as the tree grows** (7 vs 5
  steps/sec at 5M, 56 vs 45 at 500k) but costs at small trees (356 vs 509
  at 120k) — probe vs one-indirection trade.
- The adaptive metadata rebuild now runs in a Web Worker sliced verbatim
  from the page's own script (see `getAdaptiveMetaWorker` in index.html),
  so the periodic refresh no longer hitches the render loop; the "adaptive
  is CPU-bound on its rebuild" caveat from earlier rounds is gone.

## Core FMM (2D log kernel)

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (exact direct)     57.14                     -  O(N^2) reference
+fmm (adaptive FMM)     1057.39 (0.1x)      1.974e-07  funnel-hash adaptive FMM, p=10; NOT faster than direct at N=2000 (Python tree traversal overhead)
+fmm (flat vectorized)    710.61 (0.1x)      7.522e-07  single-level vectorized adaptive FMM, depth=5 order=8; NOT faster than direct at N=2000 (K^2 M2L dominates at this scale)
+quantized (32-bit packed)     68.51 (0.8x)      3.891e-01  VoxelPackedTreeFreeFMM; module documents ~1.2e-1 rel-L2 packed cost (this clustered N=2000 distribution measures higher — see table)
```

Both FMM engines reach sub-1e-6 accuracy, but at N=2000 neither is faster than
the direct O(N^2) sum — the Python tree traversal and K^2 M2L constants dominate
at this scale; the FMM asymptotic win only appears at larger N (see the
scaling table below).

## Core FMM scaling

The same flat vectorized FMM (`FastVectorizedFMM(depth=5, order=8)`) and a
chunked vectorized direct O(N^2) sum, run on the clustered multi-scale
distribution at N in {2000, 8000, 32000}. The adaptive FMM engine is
omitted at these N (its Python tree traversal is even slower than the flat
scheme and would not change the crossover conclusion). Direct O(N^2) at
N=32000 is ~1e9 pairs; the chunked direct keeps memory bounded (block=2048
targets) and finished in ~36s, under the 120s budget so N=32000 was kept.

```
=== Core FMM scaling (clustered distribution; direct budget 120s) ===
N=  2000  direct=    148.86 ms  fmm=    680.06 ms  speedup= 0.22x  rel_l2=7.522e-07
N=  8000  direct=   6787.47 ms  fmm=   2182.33 ms  speedup= 3.11x  rel_l2=4.974e-07
N= 32000  direct=  35570.47 ms  fmm=   8165.55 ms  speedup= 4.36x  rel_l2=5.303e-07
```

Crossover observed at N=8000: the flat FMM becomes faster than direct
O(N^2) at N=8000 (3.11x, rel-L2 4.97e-7) and stays faster at N=32000
(4.36x, rel-L2 5.30e-7), while at N=2000 it is 0.22x (slower). The
asymptotic win is real and the accuracy stays sub-1e-6 throughout; the
"NOT faster at N=2000" row in the table above is the small-N constant-
factor regime (see [docs/GPU_NOTES.md](docs/GPU_NOTES.md) and
[docs/INAPPLICABILITY.md](docs/INAPPLICABILITY.md) Class D), not an
algorithmic fact.

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
