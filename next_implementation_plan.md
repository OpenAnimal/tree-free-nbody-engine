# Next Implementation Plan — Round 3 (for GLM-5.2 executor; review by GLM-5.3)

Working dir: repo root. All commands: `python -X utf8 ...` from repo root.
Rules: do NOT weaken any test/assertion; if an acceptance number moves,
STOP and report. Finish tasks in order. Where this plan gives exact math
formulas, transcribe them literally — do not re-derive or "simplify" signs.

## Round 2 status: DONE and verified (2026-08-18 review)

- Commits `565bfef` + `91844d9` in; all suites green (spatial_index 10/10,
  elastic_hash 10/10, cgr88 20/20, graphics/video suites, lint clean).
- BENCHMARKS.md: 5 domain tables + 10 app case-study tables, all with
  honest accuracy columns; app8 is the flagship win (28.6x, 100% recall).
- Uncommitted (executor round-2b): ten `apps/appN_benchmark_variants.py`,
  BENCHMARKS.md apps section, README/index.html updates — verified good.

## Round 3 tasks

### 3.1 Commit checkpoint (do FIRST, before any edits)
```
git add -A && git commit -m "Add per-app variant benchmarks and BENCHMARKS app case studies"
```

### 3.2 Inapplicability taxonomy + app9 decision
Create `docs/INAPPLICABILITY.md`:
- Class A — "not a kernel sum": app6 (nearest-point proximity), app7/app9
  (top-k retrieval). FMM approximates SUMS over sources; these are ARGMAX /
  nearest-neighbor queries. Closest fast technique: LSH / grid filters
  (already used).
- Class B — "kernel lacks FMM structure": softmax attention (not
  translation-invariant, not radial). Closest technique: linear-attention /
  random-feature kernelization (Performer-style) — documented, not
  implemented.
- Class C — "right kernel, our FMM is 2D-only": app5 3D Yukawa
  (FIXED in 3.4), volumetric AO 3D kernel (candidate for 3D FMM later).
- Class D — "right technique, wrong scale": Python per-cell loop constants
  dominate at demo N (flocking N=1000, app3 N=1500, app4 N=400, core FMM
  N=2000). Asymptotic win exists but needs larger N (3.3) or a compiled
  kernel (GPU notes in 3.6).
Each class entry: 2-4 sentences, one concrete falsifiable reason, link to
the BENCHMARKS.md table that demonstrates it.
Then app9: EITHER tune the LSH (fewer hyperplanes e.g. 6, multi-table x8,
keep multi-probe) until recall@10 >= 0.5 while still >= 1.5x faster than
brute, OR retitle its BENCHMARKS.md entry to "Cautionary: fine-grained LSH
partitions collapse recall" and add a pointer from INAPPLICABILITY.md
Class D-adjacent. Do not fake the middle ground; pick one and report it.
Acceptance: file exists, 4 classes, each with reason + table link; app9
section updated; `python -X utf8 tools/lint_claims.py` still exits 0.

### 3.3 Core FMM scaling table (show the O(N) crossover)
Extend `core/benchmark_variants.py` with `run_scaling()` (called from
`__main__` after the existing run): N in [2000, 8000, 32000]; for each N,
clustered distribution (reuse the round-1 clustered generator), variants:
`standard (direct)`, `+fmm (flat vectorized)` only (adaptive is too slow
in Python at these N — omit it and say so in a comment). Direct O(N^2) at
N=32000 is ~1e9 pairs: compute with the existing vectorized direct in
chunks (it already exists for the N=2000 table); if it takes > 120 s,
drop to [2000, 8000, 16000]. Add the table to BENCHMARKS.md under a new
"## Core FMM scaling" heading with one honest takeaway sentence stating
the observed crossover N (or that none appears up to N_max, with the
per-N time ratios so the trend is visible).
Acceptance: table prints; takeaway sentence states crossover or its
absence with numbers.

### 3.4 3D uniform-grid Yukawa FMM (the big one — exact math provided)
Create `core/yukawa3d_fmm.py`. Kernel: G(r) = exp(-kappa*r)/r (app5
Debye-Huckel). Single-level flat scheme on a uniform grid, exactly like
`FastVectorizedFMM` in 2D, indexed by `CellIndex(dims=3, grid_res=depth)`
+ funnel hash moments.

MATH (transcribe literally):

1. Radial functions. Define polynomials Q_n by
   Q_0(x) = 1;  Q_{n+1}(x) = (x + 2n + 1) * Q_n(x) - x * Q_n'(x).
   (Q_1 = x+1, Q_2 = x^2+3x+3.) Then
   G_n(r) = (-1)^n * exp(-kappa*r) * Q_n(kappa*r) / r^(2n+1).
   Sanity: kappa=0 gives G_n = (-1)^n (2n-1)!! / r^(2n+1) (Laplace).
   Implement Q_n as numpy poly1d so Q_n' is exact.

2. Derivative tensors. For displacement d (a 3-vector), the derivative
   d^alpha G / dx^alpha (multi-index alpha = (a,b,c), |alpha| = a+b+c) is
     D_alpha(d) = sum_n P_{alpha,n}(d) * G_n(|d|)
   where the polynomials P (in variables dx,dy,dz) follow
     P_{(0,0,0),0} = 1,  P_{alpha,n} = 0 if n<0 or n>|alpha|,
     P_{alpha+e_i, n} = d/dx_i [ P_{alpha,n} ]  +  x_i * P_{alpha,n-1}
   (e_i = unit multi-index on axis i). Derivation: for radial G,
   d/dx_i [P G_n] = (dP/dx_i) G_n + P (x_i/r) G_n'(r), and G_n'(r) =
   r * G_{n+1}(r) by definition of G_{n+1} = (1/r d/dr) G_n.
   Represent P as dict alpha -> dict n -> poly1d triples or a flattened
   coefficient array indexed by (alpha, n); build once per (|alpha| <= 2*p)
   at import, NOT per pair.
   MANDATORY GUARD TEST before anything else: validate D_alpha against
   central finite differences (order-2 for |alpha|<=2 with h=1e-4 on
   several non-axis-aligned d, rel tol 1e-5). If it fails, STOP and
   report — do not proceed to the FMM.

3. Flat FMM, grid spacing h = 1/depth, cell center c(cell):
   - Moments per occupied cell (|beta| <= p):
       M_beta(cell) = sum_{i in cell} q_i * (x_i - c)^beta / beta!
     (beta! = a!*b!*c! for beta=(a,b,c); (x_i-c)^beta is the product.)
   - Direct near field: for each target, sources in the target's
     ring-2 neighborhood (5x5x5 box, ring_direct=2) summed exactly via
     CellIndex neighborhood_indices(key, ring=2).
   - Far field: for each target cell t, over far source cells s
     (outside ring 2), local coefficients for |alpha| <= p:
       L_alpha(t) = sum_s sum_{|beta|<=p} D_{alpha+beta}(d_ts) * M_beta(s)
     with d_ts = c_t - c_s, and the SIGN convention absorbed by defining
     moments with (x_i - c) as above and evaluating
       u(x) = sum_{|alpha|<=p} L_alpha(t) * (x - c_t)^alpha      (NO /alpha!)
     — because alpha! was already folded into M_beta and D_alpha is the
     raw derivative. Verify against direct on a 2-cell toy case first.
   - Convergence geometry: ring-2 separation gives ratio
     (h*sqrt(3)) / (3h) ~ 0.58, so p=8 should reach ~1e-6 rel-L2 on
     clustered data. If accuracy < 1e-5 is not met: raise p to 10, then
     12; if still failing, STOP and report.

4. API: class Yukawa3DFMM(depth=6, p=8, kappa=1.0) with
   .evaluate(positions, charges) -> potentials (float64), occupying
   CellIndex for cells + funnel-hash cell->moments storage.

Create `core/test_yukawa3d_fmm.py`:
- derivative-vs-FD guard (from 3.4.2);
- kappa -> 0 limit vs exact 1/r Coulomb direct (rel-L2 < 1e-6);
- accuracy vs direct on the app5-style clustered distribution,
  N=2000, rel-L2 < 1e-5 (the acceptance number; if unreachable after
  p=12, STOP and report);
- occupied-cell set in the hash matches np.unique cell keys.
Run it standalone; add to `core/__init__.py` exports.
Then add the missing `+fmm (Yukawa3DFMM)` row to
`apps/app5_benchmark_variants.py` (accuracy_vs standard, p=8) and
REGENERATE the app5 BENCHMARKS.md table + takeaway sentence verbatim.
Acceptance: test file passes standalone; app5 table has the +fmm row with
its measured accuracy; INAPPLICABILITY.md Class C updated to "fixed by
core/yukawa3d_fmm.py".

### 3.5 GPU notes (documentation; optional stretch)
Create `docs/GPU_NOTES.md`:
- The append-only funnel hash cannot unlearn keys, so dynamic sims
  rebuild per frame. On CPU the repo uses two-pass sizing (count occupied
  cells first, then size capacity = max(16, 2*count)). On GPU the standard
  pattern is PING-PONG double buffering: two tables A/B; each step reads
  forces from A while building B, then swaps — the rebuild cost overlaps
  with compute and the "can't unlearn" limitation disappears. (This is the
  same two-structure trick as linear-time median via two lists / ping-pong
  buffers in graphics.)
- The "NOT faster at this scale" rows in BENCHMARKS.md are Python
  interpreter constants, not algorithmic facts; the compiled kernels
  (core/cuda_kernels, core/triton, native/zig, webgpu) are where
  funnel-hash constant factors vs dict/hashmap baselines can be measured.
- Optional stretch (only if 3.1-3.4 are done and green): implement the
  ping-pong rebuild in the Triton kernel path and report one timing row.
  If not done, write "not attempted this round" — do not fake it.

### 3.6 Optional stretch: Fast Gaussian Transform row (app3/app10)
Only if 3.1-3.4 done: add a `+fgt (Taylor order-4 Gaussian)` row to
app3's benchmark using the same machinery as 3.4 with kernel
G(r) = exp(-r^2/h^2) (the radial recursion in 3.4.1-3.4.2 works for ANY
radial kernel — recompute G_n's closed form: for a Gaussian,
G_n(r) = (-2/h^2)^n * exp(-r^2/h^2) / ... derive via the same
G_{n+1} = G_n'/r recursion numerically-symbolically with poly1d in
r^2, or evaluate G_n via the recursion on functions). If the derivation
gets complicated, SKIP and write one honest sentence in
INAPPLICABILITY.md Class B pointing at FGT literature instead.

### 3.7 Final verification matrix + commit
```
python -X utf8 -m core.test_spatial_index
python -X utf8 -m core.test_elastic_hash
python -X utf8 -m core.test_cgr88_cross_validation
python -X utf8 -m core.test_yukawa3d_fmm
python -X utf8 graphics_rendering/test_graphics_rendering.py
python -X utf8 video_streaming_codecs/test_video_streaming.py
python -X utf8 tools/lint_claims.py
python -X utf8 core/benchmark_variants.py
python -X utf8 apps/app5_benchmark_variants.py
python -X utf8 game_mechanics_spatial/benchmark_variants.py
python -X utf8 graphics_rendering/benchmark_variants.py
python -X utf8 video_streaming_codecs/benchmark_variants.py
```
Then update BENCHMARKS.md (scaling table, app5 +fmm row, app9 update)
and `git add -A && git commit -m "3D Yukawa FMM; inapplicability taxonomy; FMM scaling crossover; GPU notes"`.

## Known pitfalls (round 3 additions)

6. Taylor FMM convergence is governed by (source+target extent)/distance;
   with ring-1 only this ratio is ~0.87 and p=8 will NOT converge — that
   is why ring_direct=2 (5x5x5) is mandatory in 3.4.3.
7. The derivative-tensor sign conventions in 3.4.3 are the #1 risk;
   the toy 2-cell check (two cells, few particles, vs direct) is
   mandatory before scaling up. Signs: moments use (x_i - c), d_ts =
   c_t - c_s, no extra /alpha! at evaluation.
8. Do not evaluate G_n at r=0: derivative tensors are only ever queried
   at well-separated cell-center displacements (r >= 3h by construction).
9. Direct O(N^2) at N>=32000 in float64: chunk it; if runtime explodes,
   lower N_max rather than switching to float32.
10. Benchmark tables pasted into BENCHMARKS.md must be verbatim from a
    real run — re-run after any code change.
