# Variant Benchmarks

Five domain benchmarks and ten application case-study benchmarks run through
the shared `core.benchmark_kit.VariantBenchmark` protocol: each reports
latency per variant and, where an exact reference exists, the repo-standard
`cross_validate` rel-L2 error next to it. Speed is never shown without the
accuracy it costs. Tables below are pasted verbatim from each
`benchmark_variants.py` run; the one-line takeaway under each is honest,
including "not faster at this scale" results.

## Core FMM (2D log kernel)

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (exact direct)     41.40                     -  O(N^2) reference
+fmm (CGR88 adaptive)     845.33 (0.0x)      1.974e-07  funnel-hash adaptive CGR88, p=10; NOT faster than direct at N=2000 (Python tree traversal overhead)
+fmm (flat vectorized)    478.87 (0.1x)      7.522e-07  single-level vectorized CGR88, depth=5 order=8; NOT faster than direct at N=2000 (K^2 M2L dominates at this scale)
+quantized (32-bit packed)     40.73 (1.0x)      3.891e-01  VoxelPackedTreeFreeFMM; module documents ~1.2e-1 rel-L2 packed cost (this clustered N=2000 distribution measures higher — see table)
```

Both FMM engines reach sub-1e-6 accuracy, but at N=2000 neither is faster than
the direct O(N^2) sum — the Python tree traversal and K^2 M2L constants dominate
at this scale; the FMM asymptotic win only appears at larger N.

## Physics — Tetrahedral contact broadphase

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (brute O(N^2))    841.49                     -  exact AABB overlap pair set
+elastichash (CellIndex ring-1)    211.06 (4.0x)              -  no missed collisions: True; 122433 exact pairs / 122433 broadphase candidates (filter superset, narrow-phase prunes false positives)
```

The CellIndex ring-1 broadphase is 4x faster than brute force on the demo tet
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
included only where the app's kernel is the 2D logarithmic CGR88 kernel
(apps 1 and 2); elsewhere it is omitted with the reason stated in the note
(Gaussian RBF, 3D Yukawa, nearest-point proximity, or high-dim cosine
retrieval -- none are the 2D log kernel). Where the result is approximate
rather than exact (LSH retrieval, filter broadphases), the correctness
metric is `recall@k` or `no missed collisions` in the note, not a rel-L2.

### App 1 -- Galaxy collision (2D log gravity forces)

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (exact direct)     17.23                     -  O(N^2) reference (app1 validate_against_direct path)
+elastichash (near only)     37.03 (0.5x)      2.610e-01  CellIndex ring-1 near-field, far-field SKIPPED (hash-truncated baseline)
+fmm (FastVectorizedFMM)     18.18 (0.9x)      2.386e-05  CGR88 flat FMM, depth=4 order=6 (the app's compute path)
```

At N=500 the flat FMM reaches 2.4e-5 rel-L2 force accuracy and is at parity
with direct summation; the hash-truncated near-field-only baseline is
slower than direct AND loses 26% rel-L2 because the far-field is skipped --
it is the cheap baseline, not a competitor.

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
standard (dense O(N^2))     70.02                     -  dense spatial RBF attention reference
+elastichash (near+far centroid)   1820.41 (0.0x)      9.392e-02  near exact (3x3 funnel hash) + far per-cell centroid; lossy far-field centroid approximation
```

The near-exact/far-centroid attention is NOT faster than dense O(N^2) at
N=1500 (per-cell Python loop overhead dominates) and pays a 9.4e-2 rel-L2
far-field centroid cost; +fmm is omitted because the Gaussian RBF is not
the 2D log kernel.

### App 4 -- Elastic-hash boids + 1euro (near-field boid rules)

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (brute near-field)     12.08                     -  O(N^2) near-field reference (separation + alignment, no far cohesion)
+elastichash (near+far+1euro)    257.31 (0.0x)              -  near-field exact vs brute (same 3x3 cell box); |far-field residual| = 0.501 (8.4% of total); +fmm axis omitted (not a 2D log kernel)
```

The hash boid step is near-field exact (same 3x3 cell box as brute) but NOT
faster at N=400 (per-cell Python loop overhead); the 8.4% far-field
cohesion residual is the intentional extra term, reported honestly.

### App 5 -- 3D protein electrostatics (Debye-Huckel screened Coulomb)

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (direct O(N^2))    333.51                     -  exact per-atom screened Coulomb reference
+elastichash (cluster O(K^2))     21.08 (15.8x)      5.682e-01  funnel-hash 3D Morton clusters, direct O(K^2) between centroids; lossy cluster-mean approximation
```

The funnel-hash cluster path is 15.8x faster than exact per-atom Debye-
Huckel at N=3000 but pays a 57% rel-L2 cluster-mean cost -- the speed/
accuracy tradeoff is explicit in the table; +fmm is omitted (3D Yukawa, not
2D log kernel).

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

### App 9 -- Streaming vector DB (multi-probe LSH ANN)

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (brute exact top-k)     66.66                     -  O(N*d) exact cosine top-10 per query
+elastichash (LSH multi-probe)     23.38 (2.9x)              -  recall@10 over 200 queries = 0.6%; zero-reorder funnel-hash ingestion; +fmm axis omitted (cosine ANN, not a kernel sum)
```

Multi-probe LSH is 2.9x faster than brute exact top-10 at N=10000/d=128
but recall@10 is only 0.6% -- 13-hyperplane LSH spreads 10k vectors across
8192 buckets so most buckets hold 1-2 vectors and the 6-bit multi-probe
cannot recover the true top-10. The speed is real; the recall cost is
reported honestly and is the known price of the fine-grained partition.

### App 10 -- Continuous spatial GNN (2D Gaussian message pass)

```
Variant                 Time (ms)  rel L2 vs ref  Note
------------------------------------------------------------------------------
standard (dense all-pairs)      4.49                     -  dense Gaussian message pass reference (no adjacency matrix)
+elastichash (near+far centroid)     65.93 (0.1x)      1.791e-01  near exact (3x3 funnel hash) + far per-cell centroid; lossy far-field centroid approximation
```

The near-exact/far-centroid GNN message pass is NOT faster than dense
all-pairs at N=150 (per-cell Python loop overhead dominates at small N) and
pays an 18% rel-L2 far-field centroid cost; +fmm is omitted (Gaussian
message kernel, not 2D log kernel).
