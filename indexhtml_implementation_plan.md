# index.html Implementation Plan (benchmark-axis rework + core alignment)

> **UPDATE 2026-08-19 — implemented & reviewed.** T1–T6 were implemented in `index.html`
> and validated headlessly (zero WebGPU/JS errors, far-field probe unchanged at mean
> ≈ 2.1%, bench panel runs all axes, Export JSON produces `benchmark_kit`-shaped JSON
> under localStorage key `webgpu-index-bench`). Three follow-ups remain open — see
> "Review findings" at the bottom. The task list below is kept as the original spec;
> items marked ✅ are done as specified unless overridden by a review finding.

> Status: PLAN ONLY — do not treat any item below as already implemented.
> Target file: `index.html` (flagship WebGPU demo). Companion reference: `core/`.
> The static server on port 8642 serves the repo root; keep `index.html` self-contained.

## Context (verified current state)

- `index.html` now contains a **correct multi-level uniform-grid CGR88 FMM** (levels 0..K,
  K = leafBits−2; kernels `m2m_up`, per-level `m2l` with parity-checked V-lists, `l2l_down`,
  `l2p`). Validated in-browser via `?probe=1` (far-field vs JS brute force: mean rel ≈ 2.3%,
  the documented softening floor) and `?fmmdebug=1` (structural dump: per-level monopole
  conservation, per-level locals vs a float64 JS reference — all ≈ 2e-7 rel).
  Root cause of the previous 45% error was an element-wise `vec2` multiply standing in for
  complex multiplication in `l2l_down`/`m2m_up` — fixed; do not regress this. All complex
  coefficient math must go through `cmul`/`cdiv`/`cdivSoft`.
- Debug/validation URL params (keep, they are the regression harness):
  `?probe=1`, `?fmmdebug=1`, `?fmm2level=1` (force K=1), `?fmmK=n` (cap K).
- Adaptive FMM path (`useAdaptiveFmm`, `adaptiveMetadata`, WGSL `wgslAdaptiveFmmSource`)
  exists and is dispatchable from the UI (`selectFmmMode`: fixed / adaptive).
- The in-browser "Variant benchmark panel" (`bench-panel`, `runLiveBench()`, around line
  4200+) currently measures these axes: fmm order p=0/1/2/4, "elastichash fixed-grid",
  "elastichash adaptive CGR88", **1€ filter ON/OFF**, P2P radius tight/wide.
- The 1€ filter is a steering-smoothing side feature; it is not implemented uniformly
  across the 4 presets (scenario clusters) and is only meaningful for boid particles.
- Core benchmark framework: `core/benchmark_kit.py` (`VariantBenchmark`: standard /
  +elastichash / +fmm / +quantized axes, latency table + optional accuracy-vs + JSON
  export). Core algorithm modules: `core/elastic_hash.py` (ElasticHash dict), `core/
  cgr88_adaptive_fmm.py`, `core/fast_vectorized_fmm.py`, `core/spatial_index.py`,
  quantized/bit-packed variants per module where available.

## Goals

1. **Benchmark axes we actually care about**: `fmm` (fixed-grid multi-level),
   `adaptivefmm` (adaptive CGR88), `elastic hash`, `quantized_bitpacked`.
2. **Remove** `1€ filter` and generic `mode` as benchmark axes. The 1€ filter stays as a
   UI toggle (boids only); it must not appear in the benchmark table or its config
   permutations.
3. Align the in-browser panel with the core benchmark protocol (same axis names, same
   standard / +axis structure, exportable to JSON compatible with `benchmark_kit.py`
   conventions) so browser numbers and core numbers can be compared in one table.

## Task list

### T1 — Benchmark variant set ✅
Rows now: `standard`, `+fmm fixed-grid multi-level`, `+fmm p=0/1/2/4`, `+adaptivefmm`,
`+elastichash`, `+quantized_bitpacked`; 1€ and P2P-radius rows removed. Adaptive reports
an ERROR row when unavailable (verified available here). *Follow-up R1 below applies.*
New rows (each isolates one axis, baseline = current settings):
- `baseline (current settings)`
- `+fmm fixed-grid multi-level` (fixed mode, current order) — this is the standard
  far-field path (2-level via `?fmm2level` is NOT an axis; it is a debug mode).
- `+fmm order p=0/1/2/4` (keep; they are legitimate FMM-cost points) — label prefix `+fmm`.
- `+adaptivefmm adaptive CGR88` (`cfg.adaptive = true`; requires the adaptive metadata
  pipeline; if unavailable on the device, report row as ERROR like benchmark_kit does,
  not silently skip).
- `+elastichash` — measuring the elastic-hash cell structure itself. Implement as a
  variant that routes neighbor/cell lookup through the GPU elastic-hash path (see T3);
  if the hash path is not yet GPU-exposed, first land T3 and only then add this row.
- `+quantized_bitpacked` — quantized/bit-packed particle state variant (see T4).
- Keep `P2P radius` rows? → NO, remove (not a focus axis; noise in the table).

### T2 — Remove 1€ from the benchmark ✅
- Delete the `1€ filter ON/OFF` rows from `variants`.
- Remove `use1Euro` handling from `applyBenchConfig`/restore (keep the UI button itself).
- Rename color modes / legend strings that couple "FMM & 1€ Dynamics" so FMM coloring is
  described on its own (1€ may remain in the boid-only color mode label).

### T3 — Elastic hash axis (GPU-exposed) ✅ (kernels + bench row; GPU-timing columns follow R2)
- The fixed-grid FMM already uses a counting sort (`count/scan/scatter_cells`) — that is
  the "standard" path. Add a WGSL elastic-hash variant for cell membership
  (open-addressing with the same mix64 family as `core/elastic_hash.py`; see
  `core/webgpu_kernels/` for prior art) so `+elastichash` measures hash-build + probe
  cost vs counting sort at equal accuracy.
- Bench metrics per row (extend the table): median step ms (existing), plus
  `fmmBuild`/`fmmM2l`/`fmmL2p` pass timings from the existing timestamp-query telemetry,
  and GPU memory (existing `valGpuMemory` source).

### T4 — Quantized / bit-packed axis ✅
- Variant of the main state buffers: pack pos/vel as f16 (or 8.8 fixed-point) + Morton-
  style bit-packed cell keys, unpacked in the compute shader. Gate behind a flag; measure
  both latency and accuracy (probe `?probe=1` machinery already computes far-field rel
  error — reuse it to report an accuracy column, mirroring `accuracy_vs` in
  benchmark_kit).
- Reference implementations/prior art: `core/bitboard_morton_avx.py`, module-level
  `+quantized` variants in core.

### T5 — Export & parity with core framework ✅ (localStorage key: `webgpu-index-bench`)
- Add "Export JSON" button to the bench panel: writes the last run's rows as
  `{"title": "webgpu-index", "results": [{"variant", "time_ms", "accuracy_rel", "note"}]}`
  — the same shape `VariantBenchmark.save_json()` produces — and triggers a download
  (and/or `localStorage` copy). This lets `core/benchmark_kit.py` consumers merge browser
  results with core results.
- Axis naming: use exactly `standard`, `+fmm`, `+adaptivefmm`, `+elastichash`,
  `+quantized_bitpacked` (suffixes like ` p=4` allowed after the axis name).

### T6 — Small cleanups while in there ✅
- `applyBenchConfig`: the inner `const useAdaptive = cfg.adaptive;` shadows the outer
  `useAdaptiveFmm` logic confusingly — rename.
- The bench panel description text (line ~4209 comment and the visible section title)
  should list the new axes and stop mentioning 1€.
- Keep `?fmmK` documented as debug-only; it must never be an axis.

## Validation (must pass after implementation)

1. `?probe=1` → meanRel ≤ ~3%, maxRel ≤ ~12% at 120k (softening floor; unchanged by bench work).
2. `?fmmdebug=1` → all `badMomSlots`/`badSlots10pct` zero, `refVsBrute` ≈ 3%.
3. Bench run completes on galaxy preset with zero WebGPU validation errors, and produces
   rows for every axis above (or explicit ERROR rows).
4. `node --check` equivalent for inline scripts (extract `<script>` blocks and
   `new Function(...)`), plus the repo's Python regression tests still green.

## Constraints

- Single-file `index.html` (no build step, no external assets).
- Do not commit; keep the 8642 server working; keep all existing debug URL params.
- All complex arithmetic in WGSL via `cmul`/`cdiv` — never bare `*` between two
  coefficient `vec2`s (this was the multi-level correctness bug).

## Review findings (2026-08-19, after implementation) — OPEN follow-ups

Verified headlessly via CDP: page loads with 0 errors; `?probe=1` still at the softening
floor (mean 2.1%, max 8.6% at 120k, K=4); a full bench run completes and exports.

- **R1 — `+adaptivefmm` accuracy column reads ~3.9 rel (garbage-looking).** Almost
  certainly a *measurement artifact*, not a bug in the adaptive path: the probe's JS
  brute-force reference excludes only fixed-grid-zone pairs (leaf Chebyshev < 2), while
  the adaptive quadtree splits near/far differently, so the two sides of the comparison
  cover different pair sets. Fix: either (a) give the adaptive variant its own reference
  that excludes the adaptive near-zone pairs, or (b) report accuracy `n/a` with a note
  until (a) exists. Do not ship the current number — it looks like a correctness bug.
- **R2 — Step-latency metric cannot discriminate variants.** The table measures CPU
  `stepMs` (encode time, ~0.2–0.4 ms for every row), so "1.50x faster" style deltas are
  noise; GPU cost differences live in the timestamp-query telemetry. Fix: per row also
  record GPU timings (`webgpuPassTimings.fmmBuild/fmmM2l/fmmL2p/mainCompute/totalGpu`
  or the `onSubmittedWorkDone` fence) and show `totalGpu` (or pass columns) as the
  primary metric, keeping stepMs secondary.
- **R3 — `+elastichash` accuracy is *near*-equal, not equal** (0.040 vs 0.027 baseline
  mean rel). Expected cause: hash-probe ordering changes summation order (f32), or a
  rare collision-path difference. Confirm with `?fmmdebug=1` under `useElasticHash=1`
  (locals must still match the float64 reference at ~1e-5); if not, there is a real
  lookup defect. Update the row's note to reflect the measured delta.

Minor (informational): export JSON verified well-formed (`{title:"webgpu-index",
results:[{variant,time_ms,accuracy_rel,note}]}`); localStorage key is
`webgpu-index-bench`; the bench's per-variant probe logs appear on the console (useful,
keep).

---

# Follow-up implementation instructions (R1 / R2 / R3) — precise spec

Code anchors below reference current `index.html` line numbers (approximate; search by
identifier, they are unique).

## R1 — Adaptive-aware accuracy reference (or honest `n/a`)

Problem: `runFmmProbeCompare` (line ~2601) builds its brute-force reference by excluding
pairs with **fixed-grid leaf Chebyshev < 2** (`if (Math.max(|jcx-cx|,|jcy-cy|) < 2) continue`).
The adaptive FMM splits near/far by quadtree lists, not the fixed grid, so for
`+adaptivefmm` the GPU field and the reference cover different pair sets → accuracy
column reads ~3.9 rel.

Implement:
1. `runFmmProbeCompare(fieldF32, stateF32, N, benchResolve)` — add an optional
   `excludeNear` predicate parameter: `runFmmProbeCompare(field, state, N, resolve, isNear)`.
   Replace the hardcoded Chebyshev check with:
   `if (isNear ? isNear(i, j, stateF32) : fixedGridNear(i, j, stateF32)) continue;`
   where `fixedGridNear` is the existing inline check extracted into a closure.
2. For adaptive variants the CPU side already has everything: `adaptiveMetadata`
   exposes `leafForParticle` (Uint32Array), `listOffsets`, `listCounts`, `listData`
   (near/U/V/X list entries per adaptive leaf). Build the predicate:
   ```js
   const leafOf = adaptiveMetadata.leafForParticle;
   // nearSet[leafOf[i]] = Set of near-list SOURCE leaf ids for that leaf
   // (built once per bench variant from listOffsets/listCounts/listData).
   const isNear = (i, j) => nearSet[leafOf[i]].has(leafOf[j]);
   ```
   NOTE: check which list channel of `listCounts` (vec4: near/u/v/x) is the P2P/P2L
   near list in `adaptive_gpu_metadata.py` and use only that one — the reference must
   exclude exactly the pairs the adaptive shader evaluates directly.
3. Thread the predicate from `runLiveBench`: when applying a variant with
   `cfg.adaptive === true`, pass the adaptive predicate; otherwise pass `undefined`
   (fixed-grid rule). The probe is already variant-triggered via `benchProbeResolve`
   (lines ~4186/4265) — extend that call path with the predicate.
4. If step 2 turns out too invasive, the acceptable fallback is: for
   `+adaptivefmm` rows set `accuracy_rel: null` and note
   `'accuracy n/a: zone split differs from probe reference (R1)'`, and render `n/a`
   in the ACCURACY column and in the exported JSON. Do NOT keep the current number.

## R2 — GPU timing as the primary latency metric

Problem: the sample hook (line ~3989, inside `frameLoop`) pushes CPU `stepMs` only
(~0.2–0.4 ms, encode time; nearly identical across variants → "1.50x faster" noise).

Implement:
1. Extend `liveBenchState` with a second array `gpuSamples` and a pending-fence field.
   In `runWebGPUFrame`, right after `webgpuDevice.queue.submit(commandBuffer)` (the
   submit whose completion is already tracked at line ~4356), when
   `liveBenchState.running` record `t0 = performance.now()` and chain
   `webgpuDevice.queue.onSubmittedWorkDone().then(() => gpuSamples.push(performance.now() - t0))`.
   This measures real GPU frame latency per variant, works even when timestamp queries
   are unavailable, and needs no pass plumbing.
2. After each variant settles (in `runLiveBench`, where `baselineSamples` /
   per-variant samples are reduced with `median()`), also compute
   `gpuMs = median(liveBenchState.gpuSamples.slice(warmup))`. Reset both arrays in
   `waitForFrames()`.
3. Table: make GPU ms the primary column — reorder header to
   `AXIS VARIANT | GPU MS | STEP MS | VS BASELINE | ACCURACY (REL) | NOTE`, compute
   `VS BASELINE` from `gpuMs`. Keep step ms as the secondary column.
4. Additionally snapshot `{...webgpuPassTimings}` (declared line ~2458) after each
   variant settles and store `fmmBuild/fmmM2l/fmmL2p` in the row objects; export them
   in the JSON as extra fields (`benchmark_kit` consumers ignore unknown fields).
   Caveat: pass timings refresh asynchronously — take the snapshot after the last
   sample of the variant, and mark them absent if `!webgpuTimestampQueryEnabled`.

## R3 — Verify `+elastichash` exactness

Goal: distinguish "summation-order f32 noise" from a real lookup defect.

1. Run `?fmmdebug=1` with the elastic-hash path forced: add a temporary URL param
   (e.g. `?eh=1`) that sets `useElasticHash = true` for the FMM dispatch, or set
   `useElasticHash = true` from the console before frame 40 (`window` scope — check
   declaration at line ~2474; if it is `let` at script scope it is console-assignable).
2. The existing debug analysis recomputes locals in float64 FROM THE DUMPED MOMENTS.
   That validates m2l/l2l but not the hash lookup feeding `build_moments`. Two checks:
   a. `report.mono[*].sum` must equal N at every level (hash misses would corrupt
      P2M inputs — actually mono sums would still hold if particles land in the WRONG
      cell only when a0 aggregation is per-sorted-range; wrong placement WOULD show as
      bad `refVsBrute`).
   b. `report.field.refVsBrute` must stay ≈ 0.03 (as measured for the counting sort).
      If it degrades beyond ~0.05, the hash produces different cell membership →
      inspect `eh_build`'s CAS loop and `eh_scan`'s cursor reuse (`cellCursor[0]`).
3. Also verify equality of membership directly: the counting sort's `sortedIndex` and
   the hash path's `sortedIndex` ranges must contain identical particle sets per cell.
   Cheap check inside the debug analysis: after dumping, recompute cell membership in
   JS (`cellOf(pos)`) and compare against `sortedIndex[cellStart[c] .. +count[c]]`
   when the hash path is active — report a `sortedMismatch` count.
4. Update the `+elastichash` row note with the measured delta and, if R3.2/3.3 are
   clean, relabel from "at equal accuracy" to "equal membership; Δaccuracy = f32
   summation order" with the actual number.

## R0 — NEW (found 2026-08-19, second review pass): adaptive WGSL had the same cmul bug — FIXED, needs validation

The adaptive shader (source string "Flat adaptive CGR88 WebGPU kernel", from line ~1739)
contained three element-wise complex multiplies, the same bug class that broke the
fixed-grid FMM before:
- `m2m` line ~1866: `acc -= a0 * deltaPower / f32(k);` → now `cmul(a0, deltaPower)`
- `m2m` line ~1872: `acc += readc(&multipoles, child, j) * bj * dp;` → now `cmul(..., dp) * bj`
- `l2l` line ~1899: `acc += readc(&locals, parent, k) * binom * dp;` → now `cmul(..., dp) * binom`

These were silently corrupting adaptive M2M/L2L coefficients. Because the bench
accuracy for `+adaptivefmm` was measured through the fixed-grid probe (R1 artifact),
the corruption never showed up as an error — it likely contributed to the 3.9 number.

Status: fixed on disk. Sanity-checked headless (adaptive mode active, page responsive,
zero WebGPU errors). **Numeric validation is blocked on R1**: once the adaptive-aware
probe reference exists, re-run it and require mean rel error ≈ 0.02–0.03 (same band as
the fixed grid). If it is still high after R1, the adaptive shader needs a structural
dump analogous to `runFmmDebugAnalysis`.

Rule reminder for any future WGSL edits (now violated twice): complex vec2
coefficients must ONLY ever be multiplied via `cmul` (or divided via `cdiv`/`cdivSoft`).
A bare `*` between two `vec2<f32>` values is always a bug.
