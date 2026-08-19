# Next Implementation Plan — Round 4 (for GLM-5.2 executor; review by GLM-5.3)

Working dir: repo root. All commands: `python -X utf8 ...` from repo root.
Rules: do NOT weaken any test/assertion; if an acceptance number moves,
STOP and report. Finish tasks in order. Where math formulas are given,
transcribe literally. Environment facts: NO torch / triton / numba /
wgpu-python on this machine; Zig 0.16 IS installed; the WebGPU demo runs
in a browser (index.html), not headless.

## Round 3 status: DONE and verified (2026-08-19 review)

- Commits `83fe057` + `7fafe6b`; all suites green (spatial_index 10/10,
  elastic_hash 10/10, cgr88 20/20, yukawa3d 5/5, graphics/video suites,
  lint clean).
- 3D Yukawa FMM (`core/yukawa3d_fmm.py`): rel-L2 2.6e-8 at N=2000 p=8;
  app5 `+fmm` row 8.5e-5. The executor's sign correction to the plan's
  assembly formula ((-1)^|beta| + 1/alpha! at evaluation) was reviewed and
  is CORRECT — the plan text was wrong, the mandated 2-cell toy check
  caught it, and the module docstring documents the deviation. Approved.
- Scaling table: FMM crossover at N=8000 (3.11x), 4.36x at N=32000.
- INAPPLICABILITY.md (4 classes), GPU_NOTES.md, app9 cautionary retitle:
  all in place and consistent with BENCHMARKS.md.
- Uncommitted: only this plan file's round-3 addendum (web-demo tasks,
  now folded into 4.4-4.7 below).

## Round 4 tasks

### 4.1 Commit checkpoint (do FIRST)
```
git add -A && git commit -m "Round-3 plan addendum: web demo follow-up tasks"
```

### 4.2 One-command verification: tools/run_all.py + CI
Create `tools/run_all.py`: runs, in order, every entry of the 4.9 matrix
plus the five benchmark_variants.py; prints one PASS/FAIL/SKIP line per
item with elapsed time; exits nonzero if any non-skippable item fails;
treat `core.test_webgpu_parity` as skippable-with-reason. No new logic —
subprocess the existing commands, capture the tail line, parse pass/fail
from known markers ("tests passed", "PASSED", "no forbidden vocabulary",
empty-table failure = nonzero exit code). Acceptance:
`python -X utf8 tools/run_all.py` exits 0 and prints the summary table.
Then add `.github/workflows/ci.yml` (repo has a GitHub remote): ubuntu
runner, python 3.11, `pip install numpy scipy matplotlib`, run
`python -X utf8 tools/run_all.py`. Mark it done even if you cannot test
the workflow locally — keep the yaml minimal.

### 4.3 2D Gaussian Taylor FGT (eigenfunction kernel — math provided)
Create `core/gaussian2d_fgt.py`. Kernel: G(r) = exp(-r^2/h^2).
KEY IDENTITY (exact, all n): the Gaussian is an eigenfunction of the
radial operator, (1/r d/dr) G(r) = (-2/h^2) G(r), therefore
    G_n(r) = (-2/h^2)^n * exp(-r^2/h^2).
Derivative tensors: 2D multi-indices alpha=(a,b), SAME recursion as the
3D case in yukawa3d_fmm (identity is dimension-independent):
    P_(0,0),0 = 1;  P_{alpha,n} = 0 if n<0 or n>|alpha|;
    P_{alpha+e_i,n} = dP_{alpha,n}/dx_i + x_i * P_{alpha,n-1}
    D_alpha(d) = sum_n P_{alpha,n}(d) * G_n(|d|)
Flat scheme: reuse the structure of yukawa3d_fmm exactly — CellIndex
(dims=2, grid_res=depth), moments M_beta with (x_i - c)^beta/beta!,
local  L_alpha = sum_s sum_{|beta|<=p} (-1)^|beta| D_{alpha+beta}(d_ts)
M_beta(s)  and  u(x) = sum_alpha L_alpha (x - c_t)^alpha / alpha!
(THE CORRECTED SIGN/FACTORIAL FORM from round 3 — not the round-3 plan
text), exact direct near field over ring-2 (5x5 box).
Tests (`core/test_gaussian2d_fgt.py`, mirror yukawa3d tests):
- G_n eigenfunction sanity: numeric (1/r d/dr)G at 5 radii vs closed form;
- derivative tensor vs central FD (|alpha|<=2, h=3e-4, 4th-order stencil
  for pure second derivatives — copy the guard from yukawa3d);
- 2-cell toy check vs direct (rel-L2 < 1e-12);
- clustered N=2000 vs direct: rel-L2 < 1e-6 (Gaussian decay makes this
  easy; if not met raise p 8 -> 10, else STOP and report);
- kappa-free occupied-cell membership matches np.unique keys.
Wire into app3: FIRST read app3's actual RBF definition and choose h so
exp(-r^2/h^2) equals the app kernel EXACTLY (if the app uses
exp(-|x-y|^2/(2 sigma^2)) then h^2 = 2 sigma^2; assert equality of the
two kernel functions on r in linspace(0, 3, 50) before benchmarking).
Add the `+fmm (Taylor FGT)` row to apps/app3_benchmark_variants.py
(accuracy_vs standard), regenerate the app3 BENCHMARKS.md table +
takeaway verbatim, and update INAPPLICABILITY.md Class B: Gaussian RBFs
now have a fast transform (softmax attention still does not — the class
stays, the Gaussian entry moves to "served by core/gaussian2d_fgt.py").
Acceptance: tests pass; app3 table has the row; Class B updated.

### 4.4 Web demo honesty: the "+elastichash" axis (index.html)
The GPU hash in index.html (`eh_clear`/`eh_build`/`ehProbe`) is generic
open addressing (hashU32 + linear probe), NOT the funnel schedule of
core/elastic_hash.py. Since wgpu smoke-testing is unavailable here, take
the FALLBACK path: rename the axis everywhere in index.html to
"open-addressing hash (linear probe)" — UI labels, bench panel legend,
and any README/BENCHMARKS.md reference to the web demo — and add one
sentence to docs/GPU_NOTES.md: the funnel-schedule WGSL port (with
ping-pong rebuilds) is future work; the current demo hash shares only the
"hash-indexed cells, no pointers" idea. Do NOT attempt the funnel-schedule
WGSL port this round. Acceptance: grep for "elastichash" in index.html
shows only code identifiers (renaming the JS/WGSL identifiers is allowed
but optional); user-facing strings are honest; lint exits 0.

### 4.5 Single source of truth for WGSL: tools/check_wgsl_sync.py
index.html carries shaders inline; core/webgpu_kernels/ ships
tree_free_fmm.wgsl + adaptive_cgr88.wgsl consumed by webgpu_fmm_runner.py.
Create `tools/check_wgsl_sync.py`: extract every `@compute ... fn <name>`
block (whitespace-normalized) from index.html and from both .wgsl files;
for every function name present in BOTH sources, compare normalized
bodies; FAIL (exit 1, file:line of first difference) on divergence;
functions present in only one source are INFO, not failures (the demo
legitimately has extra UI kernels). If divergence is found: the
core/webgpu_kernels/ file is AUTHORITATIVE — update index.html's inline
copy to match, then re-run the check. Acceptance: exits 0; add to the
4.9 matrix.

### 4.6 WGSL parity test (skips without wgpu)
Create `core/test_webgpu_parity.py`: if `import wgpu` fails or no
adapter, print "SKIP: wgpu not installed" and exit 0. Otherwise: 2D
clustered N=2000, run the fixed-grid WGSL pipeline via
core/webgpu_kernels/webgpu_fmm_runner.py, compare per-particle force
vectors vs FastVectorizedFMM on identical inputs and softening, assert
rel-L2 < 1e-4 (f32 vs f64 floor — do not tighten). State in the
docstring that this covers the FILE kernels only (inline-demo parity is
4.5's sync check). Acceptance: passes or skips-with-reason.

### 4.7 5M-particle instrumentation (browser-assisted; merge-ready even unmeasured)
Add a per-pass timing breakdown to index.html's stats panel + console:
counting-sort (count/scan/scatter), hash build (if enabled), P2M,
M2M/L2L/M2L, L2P+P2P, render — CPU-side dispatch timing with a repeated
single-pass loop for stable numbers is acceptable (timestamp-query only
if trivially available). Print the table at the 1M and 5M presets. Do
NOT optimize anything this round — instrumentation only. If you cannot
run a browser, verify the code paths by review, mark the BENCHMARKS/
GPU_NOTES entry "measurements pending manual browser run", and STOP
after 4.8. Acceptance: instrumentation code merged; either two measured
tables (paste verbatim into docs/GPU_NOTES.md) or the honest pending note.

### 4.8 Yukawa3D error-vs-p convergence table
Add `run_convergence()` to core/benchmark_variants.py: clustered N=2000,
p in {4, 6, 8, 10, 12}, print rel-L2 vs direct per p (and wall time).
Paste verbatim into BENCHMARKS.md under "## Yukawa 3D FMM convergence"
with one sentence stating the observed geometric rate (expected: each +2
in p buys roughly a constant factor from the ~0.58 separation ratio).
Acceptance: table present; sentence states the measured rate.

### 4.9 Final verification matrix + commit
```
python -X utf8 -m core.test_spatial_index
python -X utf8 -m core.test_elastic_hash
python -X utf8 -m core.test_cgr88_cross_validation
python -X utf8 -m core.test_yukawa3d_fmm
python -X utf8 -m core.test_gaussian2d_fgt
python -X utf8 -m core.test_webgpu_parity
python -X utf8 graphics_rendering/test_graphics_rendering.py
python -X utf8 video_streaming_codecs/test_video_streaming.py
python -X utf8 tools/lint_claims.py
python -X utf8 tools/check_wgsl_sync.py
python -X utf8 tools/run_all.py
```
Update BENCHMARKS.md (app3 row, convergence table) and GPU_NOTES.md (4.4
sentence, 4.7 tables/pending-note). Then
`git add -A && git commit -m "Gaussian 2D FGT; web demo honesty; wgsl sync check; run_all + CI; convergence table"`.

### 4.10 Optional stretch (only if 4.1-4.9 are green)
Zig funnel-hash microbenchmark: port core/elastic_hash.py's funnel
schedule (insert + probe) to native/zig/src/funnel_hash.zig; acceptance =
identical (key -> value) membership vs Python on 1M seeded random keys;
report Python vs Zig insert+probe times in GPU_NOTES.md. If the schedule
port proves too risky, fall back to a Zig LINEAR-PROBE table, explicitly
labeled "linear probe, not funnel schedule" — still an honest
compiled-constants data point. If not attempted, write "not attempted".

## Deferred (round-5 candidates — do NOT start now)

- Funnel-schedule WGSL port with ping-pong rebuilds (needs wgpu harness).
- 2D screened Yukawa K0 FMM (requires Bessel-function G_n derivation —
  plan author will derive; executor must not improvise it).
- 3D FMM for the AO kernel (rational-kernel G_n recursion, same caveat).
- Adaptive (multi-level) 3D FMM.

## Known pitfalls (updated)

1. Unit mode (`grid_res=`) for [0,1) positions, world mode (`cell_size=`)
   for world units; mixing collapses everything to one cell.
2. `key_ints` returns (x, y) — arrays index `[y, x]`.
3. Never let a benchmark note claim a speedup the table doesn't show.
4. Broadphase "accuracy" = no missed collisions, not pair-set equality.
5. Rebuild CellIndex/elastic hash on every update (append-only).
6. Taylor FMM convergence is governed by (source+target extent)/distance:
   ring_direct=2 (5x5 in 2D, 5x5x5 in 3D) is mandatory; ring-1 will not
   converge at p=8.
7. CORRECTED round-3 convention (supersedes the old pitfall 7 text):
   moments M_beta = sum q (x_i - c)^beta / beta!;
   L_alpha = sum_s sum_beta (-1)^|beta| D_{alpha+beta}(d_ts) M_beta(s);
   u(x) = sum_alpha L_alpha (x - c_t)^alpha / alpha!.  The 2-cell toy
   check against direct summation is mandatory before anything scales.
8. Never evaluate G_n at r=0; derivative tensors are only queried at
   well-separated cell centers (r >= 3h by construction).
9. Benchmark tables pasted into docs must be verbatim from a real run;
   re-run after any code change.
10. index.html is ~4900 lines: verify line anchors with grep before
    editing; make only the string/label changes 4.4 requires.
