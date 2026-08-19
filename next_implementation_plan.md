# Next Implementation Plan — Round 5 (for GLM-5.2 executor; review by GLM-5.3)

Working dir: repo root. All commands: `python -X utf8 ...` from repo root.
Rules: do NOT weaken any test/assertion; if an acceptance number moves,
STOP and report. Finish tasks in order. Math given here is to be
transcribed literally. Environment: no torch/triton/numba/wgpu-python;
Zig 0.16 available; WebGPU demo is browser-only.

## Round 4 status: DONE and verified (2026-08-19 review)

- Commits `721a6dd`, `c84eb8b`, `87ebf94`; run_all.py: 14 PASS, 1 SKIP
  (wgpu), 0 FAIL; lint + wgsl-sync clean.
- Gaussian2D FGT: app3 row 4.7e-7 rel-L2 (self-term correctly restored —
  G(0)=1 is finite for Gaussians, unlike Coulomb kernels). Approved.
- WGSL sync: executor found the .wgsl files were STALE and index.html
  carried the bug fixes; propagating index.html -> .wgsl was the correct
  reversal of the plan's direction. 12 shared functions now in sync.
- Zig funnel hash: membership equality on 1M keys PASS, 0 false hits,
  worst-case absent-key probes = probe_bound exactly (277). Honest.
- ONE OPEN DEFECT (round-5 task 5.2): the Yukawa3D error-vs-p table floors
  at ~6.3e-5 for p>=6, and the code attributes this to "ring-2 near field
  + f64 round-off". That attribution is WRONG: the near field is exact
  direct summation and f64 roundoff at N=2000 is ~1e-14. A true Taylor
  far-field error decays geometrically with p until it reaches roundoff.
  Something else stops it — most likely an off-by-one or silent drop of
  the highest-order derivative terms (|alpha| near 2p), or the moment set
  (|beta|<=p) truncating against the D_{alpha+beta} table bound.

## Round 5 tasks

### 5.1 Commit checkpoint (do FIRST)
`git status --short` should be clean or near-clean; if anything is
uncommitted, inspect it, then `git add -A && git commit -m "..."` with an
honest message. Do not push.

### 5.2 Diagnose and fix the Yukawa3D p-floor (correctness task, top priority)
Work in core/yukawa3d_fmm.py + apps/app5_benchmark_variants.py.
Experiments, in this order, reporting numbers for each:
a. Derivative-tensor audit: for the clustered N=2000 test distribution,
   verify that EVERY D_{alpha+beta} used in M2L has |alpha+beta| <= 2p
   AND that the P-tensor builder actually contains nonzero P_{alpha,n}
   for all |alpha| = 2p (print a count of nonzero entries per order;
   a sudden zero at |alpha| = 2p means an off-by-one in the builder).
b. Single-pair test: pick the worst-converging far cell pair (smallest
   |d_ts| among far pairs), compute its Taylor contribution at p = 4..12
   against the exact per-particle sum for just those two cells. If this
   single-pair error also floors, the bug is in the operator; if it decays
   geometrically, the bug is in the assembly (e.g. far-cell set changes
   with p, or a pair is double-counted / skipped at the ring boundary).
c. Sweep ring_direct=3 (7x7x7 direct) at p = 6, 8, 10: if the floor drops,
   the ring-2 separation assumption is being violated somewhere (find out
   where — report the violating pair's geometry).
Acceptance: EITHER rel-L2 at p=12 drops below 1e-8 (fix found — add a
regression test with the root cause named), OR a written root-cause
analysis in the module docstring + BENCHMARKS.md correcting the wrong
"near field + roundoff" attribution with the measured evidence. Do not
leave the wrong attribution in place either way.

### 5.3 Zig vs Python funnel-hash: probe counts and throughput
The Zig port reports mean ~33 probes/insert and 277 for absent keys. That
is honest but unexplained. Add a probe-count instrument to the Python
ElasticHashTable (a counter incremented in the probe loop, plus a
`mean_probes_last_op` stat — do NOT change probe behavior) and a script
`tools/compare_hash_python_zig.py` that: seeds the SAME 1M keys as the
Zig bench, inserts + probes in Python, prints mean probes for
insert/hit/absent next to the Zig numbers (hardcode the Zig numbers as a
commented reference from a fresh Zig run; re-run the Zig bench yourself:
see native/zig/ README). Acceptance: if Python mean probes match Zig
within 2x, record both in docs/GPU_NOTES.md with one sentence explaining
why ~30 probes is expected at this load factor (or, if it is NOT
expected, STOP and report — the schedule port may diverge); add a
Python-vs-Zig throughput row (expect ~100x; report the real number).

### 5.4 2D screened Yukawa (K0 kernel) Taylor FMM — math provided
Upgrade algorithm_theory/screened_yukawa_fmm.py from its honest order-0
tree-code (~1% error) to a true Taylor FMM. Kernel G(r) = K0(kappa*r),
z = kappa*r, K0/K1 from scipy.special.
MATH (transcribe literally; derived from dK0/dz = -K1,
dK1/dz = -K0 - K1/z):
  G_n(r) = kappa^(2n) * [ a_n(z) * K0(z) + b_n(z) * K1(z) ]
  a_0 = 1, b_0 = 0,
  a_{n+1}(z) = ( a_n'(z) - b_n(z) ) / z
  b_{n+1}(z) = ( b_n'(z) - a_n(z) - b_n(z)/z ) / z
(These are rational functions of z. Build a_n, b_n exactly once per p as
sympy expressions in z, then lambdify to fast callables; or represent as
Laurent polynomials in z. Verify with a numeric guard: compare G_1, G_2
against a central-difference (1/r d/dr) of G_0, G_1 at 5 radii, rel tol
1e-8.)
Everything else reuses the gaussian2d_fgt structure verbatim: 2D
P_{alpha,n} recursion, ring-2 direct near field, corrected sign/factorial
assembly (see pitfalls). NOTE: K0 has a log singularity at r=0 — never
evaluated, because G_n is only queried at far-cell centers (r >= 3h).
Tests (mirror test_gaussian2d_fgt.py): Bessel recursion guard, derivative
FD guard, 2-cell toy check, clustered N=2000 vs direct rel-L2 < 1e-6,
kappa -> small limit consistency (K0 -> -ln(r): compare against a direct
sum with -ln(r) kernel at small kappa, loose tol 1e-3).
Wire into the screened_yukawa demo/benchmark as a `+fmm (Taylor K0)` row
(accuracy_vs direct), KEEP the old tree-code row as a comparison, update
its honesty note to point at the new engine. Regenerate the relevant
BENCHMARKS.md table verbatim.

### 5.5 app10 +fmm row (cheap — same engine as app3)
app10 is a 2D Gaussian message pass. Add the `+fmm (Taylor FGT)` row to
apps/app10_benchmark_variants.py using core/gaussian2d_fgt.py, kernel
matched exactly as in app3 (read app10's kernel; assert equality on
r in linspace(0,3,50) first; restore self-terms if its dense reference
includes them). Regenerate the app10 BENCHMARKS.md table.

### 5.6 Final verification + docs + commit
- Run `python -X utf8 tools/run_all.py` (must stay 14+ PASS, 0 FAIL) plus
  `python -X utf8 apps/app5_benchmark_variants.py` and
  `python -X utf8 apps/app10_benchmark_variants.py` and
  `python -X utf8 tools/compare_hash_python_zig.py`.
- Update BENCHMARKS.md (5.2 outcome, 5.4 table, 5.5 table) and
  docs/GPU_NOTES.md (5.3 table).
- `git add -A && git commit -m "K0 screened Yukawa Taylor FMM; Yukawa3D p-floor diagnosis; Zig/Python hash comparison; app10 FGT row"`.

## For the MAIN agent (GLM-5.3) at next review — NOT for the executor

- Browser verification of the 5M instrumentation (index.html "5M Extreme
  Mode" sidebar block): serve repo root, drive a real browser, capture the
  per-pass timing tables at 1M and 5M presets, paste verbatim into
  docs/GPU_NOTES.md (the round-4 "pending manual browser run" note).
- Consider `git push` coordination with the user (CI workflow will run on
  GitHub once pushed).

## Deferred (round-6 candidates — do NOT start)

- Funnel-schedule WGSL port with ping-pong rebuilds (needs wgpu harness).
- AO-kernel 3D FMM: the AO kernel depends on per-source radius/opacity,
  not just displacement — a pure Taylor M2L does not apply without
  per-radius moment classes; needs a design decision from plan author.
- Adaptive (multi-level) 3D FMM.

## Known pitfalls (carry-forward, updated)

1. Unit mode (`grid_res=`) for [0,1) positions, world mode (`cell_size=`)
   for world units; mixing collapses everything to one cell.
2. `key_ints` returns (x, y) — arrays index `[y, x]`.
3. Never let a benchmark note claim a speedup the table doesn't show.
4. Broadphase accuracy = no missed collisions, not pair-set equality.
5. Rebuild CellIndex/elastic hash on every update (append-only).
6. Taylor FMM needs ring_direct=2 minimum; ring-1 will not converge.
7. Assembly (CORRECT form): M_beta = sum q (x_i-c)^beta/beta!;
   L_alpha = sum_s sum_beta (-1)^|beta| D_{alpha+beta}(d_ts) M_beta(s);
   u(x) = sum_alpha L_alpha (x-c_t)^alpha/alpha!. 2-cell toy check
   before scaling is mandatory.
8. Gaussian kernels have finite G(0): if the reference includes self
   terms, add them back explicitly (app3 learned this the hard way).
9. Never evaluate G_n at r=0 (K0 log-singular, 1/r singular); far-cell
   centers only (r >= 3h).
10. Benchmark/doc tables must be verbatim from real runs; re-run after
    any code change. Probe-count instrumentation must not alter probe
    behavior.
