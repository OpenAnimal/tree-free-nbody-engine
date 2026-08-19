# Next Implementation Plan — Round 6 (for GLM-5.2 executor; review by GLM-5.3)

Working dir: repo root. All commands: `python -X utf8 ...` from repo root.
Rules: do NOT weaken any test/assertion; if an acceptance number moves,
STOP and report. Finish tasks in order. Environment: no
torch/triton/numba/wgpu-python; Zig 0.16 available; WebGPU demo is
browser-only (main agent verifies it — see the main-agent section).

## Round 5 status: DONE and verified (2026-08-19 review)

- Commits `c6471ba`, `e09305b`, plus main-agent `6ec52ae` (browser-verified
  1M/5M timings in docs/GPU_NOTES.md §4). run_all.py: 15 PASS, 1 SKIP.
- Yukawa3D p-floor ROOT CAUSE (approved): the app5 reference
  `_direct_debye_huckel` added `+1e-6` to ALL pairwise distances — a
  6.27e-5 systematic bias in the REFERENCE, not the FMM. After the fix the
  FMM converges geometrically to 1.5e-10 at p=12. Regression test pins it.
- Zig/Python funnel-hash: probe counts match to 4 decimals (same schedule);
  ~90-100x throughput gap explained as interpreter overhead; ~33 mean
  probes explained analytically (alpha=28 x beta=9 geometry at load 0.95).
- Screened Yukawa 2D K0 Taylor FMM: 6.2e-9 vs old tree-code 2.4e-2;
  kappa->log limit 2.9e-11. app10 FGT row 3.7e-5 (honestly attributed to
  the app's own +1e-4 reference regularization; true FGT error 2.9e-8).
- Browser measurement (main agent): at 5M, Main Compute (P2P near field)
  is 86% of the 68.6 ms GPU frame (Build+M2L+L2P = 7.5 ms total). The
  optimization target is unambiguous — see task 6.2.

## Round 6 tasks

### 6.1 Commit checkpoint (do FIRST)
`git status --short` should be clean; if not, inspect, then commit honestly.

### 6.2 WGSL P2P near-field optimization (evidence-driven — read GPU_NOTES §4 first)
Measured evidence: at 5M (256x256 grid, avg 76.3/cell), Main Compute is
59.3 ms of 68.6 ms. Per the round-3 addendum's evidence order for a
P2P-dominated profile:
a. Raise the default grid resolution at high N (leafBits): occupancy 76/cell
   is far above the ~8-16/cell sweet spot; at 5M try leafBits so that
   side^2 ~ N/12 (for 5M: side ~ 645 -> leafBits 10, 1024x1024). Expose it
   as an auto-tuned default with a manual override, and confirm the
   "Cell Occupancy" telemetry shows the drop.
b. If the P2P kernel loops all 9 neighbor cells unconditionally
   ("Full x9 cell lists"), consider skipping cells whose AABB does not
   intersect the P2P radius — only if this can be done WITHOUT changing
   results (the radius slider interacts with the x9 stencil; if skipping
   changes physics at small radius, do not do it — report instead).
Constraints: each optimization = one commit whose message contains the
before/after telemetry numbers; no accuracy regression in the demo's
validation overlay (FMM far-field error readout); do not touch the
counting-sort or M2L chain this round. Acceptance: run_all.py stays
15 PASS / 0 FAIL; commit messages contain numbers; the CPU-side
telemetry still reports all passes. Browser re-measurement is the MAIN
AGENT's job (see below) — do not claim GPU speedups without it.

### 6.3 URL params for hands-free browser verification (small, unblocks 6.2/6.4)
Add to index.html: query params `?preset=5m|500k|120k|15k`, `?n=1234567`
(equivalent to spinbutton+Set), `?scenario=galaxy|boids|vortex|sph`,
`?autoprint=1` (logs a single console line per 1Hz telemetry refresh in a
stable parseable format, e.g. `TELEM {json}` with all sidebar metrics).
No behavior change when params are absent. Acceptance: manual smoke —
open `index.html?preset=120k` and confirm 120,000 active particles
without clicking (executor may verify by code review if no browser;
the MAIN AGENT will exercise it for real at review time).

### 6.4 Funnel-schedule WGSL port with ping-pong rebuild (the big one)
The demo's hash (`eh_clear`/`eh_build`/`ehProbe`) is linear-probe open
addressing, honestly labeled since round 4. Port the REAL funnel schedule
(alpha slabs of beta-slot sub-arrays + overflow region, probe bound) from
core/elastic_hash.py into WGSL, with ping-pong double buffering (two
tables; each frame probes the read table while building the write table;
swap at frame end — see docs/GPU_NOTES.md §1). Keep the existing
linear-probe path as a toggle so both are benchmarkable in the live
micro-bench panel (axes: `+openaddr-hash` vs `+funnel-hash`).
Numerical acceptance (from the round-3 addendum, unchanged): with the
funnel path enabled, the sortedIndex contiguous ranges are IDENTICAL to
the counting-sort path's (same cell -> same particle set), verified by a
debug compare pass on N=100k particles (add a `?hashverify=1` mode that
runs the comparison once at startup and prints PASS/FAIL + mismatch count
to console). Also assert insert/probe counts stay within the funnel probe
bound (add a probe-counter to the WGSL, printed in TELEM).
If any part proves infeasible in WGSL (e.g. the paper's greedy insert
needs serialization the compute model can't express efficiently), STOP,
keep the honest `+openaddr-hash` label, and write the specific technical
blocker into docs/GPU_NOTES.md — do not ship a "funnel" that is not the
funnel schedule.
Note: keep WGSL single-source-of-truth — the sync check
(tools/check_wgsl_sync.py) compares index.html inline shaders against
core/webgpu_kernels/*.wgsl, so if new @compute functions are added to
index.html that have counterparts in the .wgsl files, update both.

### 6.5 Cleanup: unify the three radial Taylor FMM engines (refactor with guardrails)
core/ now has gaussian2d_fgt.py, yukawa3d_fmm.py, screened_yukawa2d_fmm.py
sharing the same P-tensor recursion and flat assembly, copy-pasted three
times. Extract core/radial_taylor.py with: the multi-index helpers, the
P_{alpha,n} polynomial-tensor builder (dims=2|3), the ring-2 flat scheme
driver (P2M / M2L / L2P / near-field via CellIndex), parameterized by a
`radial_fns: Callable[[float], float]` (the G_n family) and dims.
Then make the three engines thin wrappers that only supply their G_n
families. HARD CONSTRAINT: all existing tests
(test_gaussian2d_fgt, test_yukawa3d_fmm, test_screened_yukawa2d_fmm) must
pass UNCHANGED (same tolerances, same printed accuracy numbers within
noise) — they are the refactor guardrails. If extraction risks behavior,
STOP and report rather than forcing it. Acceptance: run_all.py 15 PASS;
the three test files untouched (git diff shows no changes to them).

### 6.6 Final verification + docs + commit
- `python -X utf8 tools/run_all.py` (15 PASS, 1 SKIP, 0 FAIL) plus
  `python -X utf8 tools/check_wgsl_sync.py` (should be included already)
  and `python -X utf8 tools/lint_claims.py`.
- Update docs/GPU_NOTES.md: funnel-port outcome (numbers or blocker),
  link the 6.2 commits to the §4 measurement.
- `git add -A && git commit -m "WGSL P2P tuning; funnel-hash WGSL port (or blocker report); radial Taylor unification; URL params for browser verification"`.

## For the MAIN agent (GLM-5.3) at next review — NOT for the executor

- Browser-verify 6.2: `?preset=5m&autoprint=1`, capture TELEM lines,
  compare Main Compute / Avg Step against the round-5 baseline in
  GPU_NOTES §4 (5M baseline: Main 59.3 ms, Avg Step 39.8 ms, Total 68.6 ms);
  paste before/after into GPU_NOTES and judge whether the commit-message
  claims hold.
- Browser-verify 6.4: `?preset=120k&hashverify=1` -> console PASS with
  0 mismatches; eyeball the live micro-bench panel's two hash axes.
- Sweep 1M/5M again with autoprint for a cleaner 1M steady-state row
  (the round-5 1M row was transient-contaminated — GPU_NOTES §4 caveat).

## Deferred (round-7 candidates — do NOT start)

- AO-kernel 3D FMM design decision (kernel depends on per-source radius/
  opacity — needs per-radius moment classes; plan author to design).
- Adaptive (multi-level) 3D FMM.
- Push/CI coordination (GitHub Actions will run on first push; the user
  decides when to push).

## Known pitfalls (carry-forward)

1. Unit mode (`grid_res=`) vs world mode (`cell_size=`); mixing collapses
   everything to one cell.
2. `key_ints` returns (x, y) — arrays index `[y, x]`.
3. Never claim a speedup without a measured table; never let a benchmark
   note claim more than the numbers show.
4. Broadphase accuracy = no missed collisions, not pair-set equality.
5. Rebuild CellIndex/elastic hash on every update (append-only).
6. Taylor FMM needs ring_direct=2 minimum; ring-1 will not converge.
7. Assembly (correct form): M_beta = sum q (x_i-c)^beta/beta!;
   L_alpha = sum_s sum_beta (-1)^|beta| D_{alpha+beta}(d_ts) M_beta(s);
   u(x) = sum_alpha L_alpha (x-c_t)^alpha/alpha!; 2-cell toy check first.
8. Gaussian kernels have finite G(0): add self terms back if the
   reference includes them.
9. Never evaluate G_n at r=0; far-cell centers only (r >= 3h).
10. Benchmark/doc tables verbatim from real runs; probe instrumentation
    must not alter probe behavior; WGSL changes must respect the sync
    check (index.html <-> core/webgpu_kernels/).
11. Check the REFERENCE before blaming the approximation (round-5 lesson:
    the +1e-6 bias lived in the "ground truth", not the FMM).
