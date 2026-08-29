# GPU Notes — funnel hash on accelerators, and the "not faster at this scale" caveat

This file documents two things that the BENCHMARKS.md tables cannot show on
their own:

1. Why the append-only funnel hash is still the right structure for dynamic
   simulations on the GPU, and the standard GPU pattern that removes its
   only real limitation (it cannot unlearn keys).
2. Why several BENCHMARKS.md rows honestly say "NOT faster than O(N^2) at
   this scale" — and where to look for the real constant factors.

Round index (this file doubles as the review log after Round 10, which
lives in `docs/review_history/`): round 3 → §1 · round 4 → §1/§4 ·
round 5 → §3–4 · round 6 → §5 · round 7 → §5.4/§6 · round 9 → §7 ·
round 11 → §8 · round 12 → §9 · round 13 → §10 · round 14 → §11 ·
round 15 → §10.4/§12.7 · round 16 → §12 · round 17 → §13 ·
round 18 → §14 · round 19 → §15.

---

## 1. The funnel hash cannot unlearn keys; dynamic sims use ping-pong rebuilds

The Farach-Colton, Krapivin, & Kuszmaul (2025) elastic hash is an **append-only**
table: keys are inserted, never deleted, and the table is rebuilt from
scratch when the load factor crosses its threshold. For a static kernel
sum (the BENCHMARKS.md tables) this is irrelevant — the table is built once
per `evaluate()` call. For a dynamic simulation (one FMM step per frame,
particles moving between cells every step) the occupied-cell set changes
every frame, so the table must be rebuilt every frame.

On CPU the repo already handles this with the **two-pass sizing** pattern in
`core/spatial_index.py` `CellIndex.build`: count the occupied cells first
(`np.unique`), then size the new elastic hash to `capacity = max(16,
2*count)` and insert. This is O(N) per rebuild and is what every domain
folder's `benchmark_variants.py` exercises.

On GPU the standard pattern for an append-only structure that must be
rebuilt every frame is **ping-pong double buffering**: two tables A and B.
Each step reads forces/fields from table A (the previous frame's occupied-
cell index) while building table B from this frame's occupied cells in
parallel, then swaps A and B. The rebuild cost overlaps with the compute
that reads A, so the "can't unlearn keys" limitation effectively
disappears — you never delete from the live table, you just retire it. This
is the same two-structure trick as linear-time median via two lists, or
ping-pong buffers in real-time graphics (render to off-screen buffer A while
displaying buffer B, then swap).

Concretely for this repo's funnel hash: the elastic hash's two-pass sizing
(count then insert) maps cleanly to two GPU passes — a `cub::DeviceRunLengthEncode`
-style unique-count pass, then a scatter-insert pass — and the ping-pong
swap is a single pointer exchange. No atomic deletion, no tombstones, no
reordering of live entries.

### Optional stretch (round 3): ping-pong rebuild in the Triton kernel path

**Not attempted this round.** The existing Triton path
(`core/cuda_kernels/triton_tree_free_fmm.py`) is a block-tiled **direct
O(N^2) Coulomb P2P** reference solver, not a funnel-hash FMM — it has no
occupied-cell index to ping-pong. Implementing the ping-pong rebuild
meaningfully would first require a Triton funnel-hash FMM (cell build +
P2M + M2L + L2P as GPU kernels), which is a separate multi-kernel project
beyond the round-3 scope. Rather than fake a timing row on the existing
direct kernel (which has no rebuild to overlap), this is recorded as "not
attempted this round" per the round-3 plan's instruction.

### Web demo hash honesty (round 4)

The GPU hash in `index.html` (`eh_clear` / `eh_build` / `ehProbe`) is
generic open addressing (hashU32 + linear probe), NOT the funnel schedule
of `core/elastic_hash.py`. The live in-browser micro-bench axis is
labeled `+openaddr-hash (linear probe)` to reflect this honestly. The
funnel-schedule WGSL port (with ping-pong rebuilds) is future work; the
current demo hash shares only the "hash-indexed cells, no pointers" idea
with the elastic hash. The static reference table in `index.html` (pasted
from `BENCHMARKS.md`) retains the `+elastichash` axis name because those
rows are Python results that DO use the real funnel hash.

---

## 2. The "NOT faster at this scale" rows are Python constants, not algorithmic facts

Several BENCHMARKS.md rows report that an FMM or spatial-hash variant is
slower than the brute O(N^2) direct sum at the demo's N (core FMM at
N=2000, flocking at N=1000, app3 at N=1500, app4 at N=400, app5 +fmm at
N=3000). This is a **constant-factor property of the pure-Python driver**,
not an asymptotic fact about the algorithm:

- The direct O(N^2) sum is a single vectorized NumPy call over an (N,N)
  distance matrix — one C-level dispatch for the whole computation.
- The FMM / spatial-hash paths loop over occupied cells in pure Python
  (per-cell hash probes, list materialization, per-cell pair assembly).
  Each occupied cell costs a Python interpreter dispatch (~microseconds),
  and at small N the number of such dispatches times the per-dispatch cost
  exceeds the single NumPy call's wall time even though the FMM does less
  total arithmetic.

The [Core FMM scaling](../BENCHMARKS.md#core-fmm-scaling) table shows this
directly: the same flat FMM that is at parity at N=2000 (1.0x) becomes
**8.0x faster than direct at N=32000** (rel-L2 3.7e-7). The crossover is
real; it just lives at larger N than the per-app demo scales.

Where the real constant factors can be measured is the **compiled kernel
paths**, where the per-cell dispatch is a few cycles instead of a Python
interpreter call:

- `core/cuda_kernels/` — CUDA + Triton GPU kernels
- `core/hip_kernels/` — HIP GPU kernels
- `core/opencl_kernels/` — OpenCL kernels
- `core/webgpu_kernels/` — WGSL WebGPU kernels
- `native/zig/` — Zig SIMD CPU backend

These are where funnel-hash constant factors vs dict/hashmap baselines
should be benchmarked. The Python `benchmark_variants.py` tables in
BENCHMARKS.md are correctness + asymptotic-order evidence, not GPU
constant-factor evidence; conflating the two would overclaim.

---

## 3. Python vs Zig funnel-hash probe counts and throughput (round 5)

`tools/compare_hash_python_zig.py` runs the SAME 1M splitmix64-key
workload as `zig/bench.zig` (capacity=1_000_000, delta=0.05, seed=42)
through the Python `core.elastic_hash.ElasticHashTable` and compares
mean probes per pass and throughput against a fresh Zig
`zig build run --release=fast` run. The Python salts use MT19937 while
the Zig port uses a splitmix64-seeded LCG, so exact slot assignments
differ — but the slab geometry (alpha=28, beta=9, total_slots=1_052_640,
probe_bound=277) is identical and the probe counts are geometry-driven,
not salt-driven.

| Pass            | Python M keys/s | Zig M keys/s | Py/Zig | Python mean probes | Zig mean probes | Py/Zig |
| --------------- | --------------- | ------------ | ------ | ------------------ | --------------- | ------ |
| INSERT          | 0.16            | 14.64        | 0.011  | 32.9626            | 32.9630         | 1.000  |
| LOOKUP (hits)   | 0.16            | 14.07        | 0.012  | 28.9557            | 28.9559         | 1.000  |
| LOOKUP (absent) | 0.03            | 2.62         | 0.010  | 277.0000           | 277.0000        | 1.000  |

The probe counts match to four decimals (well within the 2x acceptance
band), confirming the Python reference and the Zig port implement the
same funnel schedule. The throughput ratio is ~90-100x in Zig's favor,
which is the expected Python-interpreter-overhead gap on the per-key
path (the plan's "~100x" estimate). The absent-key mean probes equal
`probe_bound=277` exactly in both implementations — the funnel hash's
deterministic worst-case bound holds in both.

**Why ~30 probes is expected at this load factor.** At delta=0.05 the
funnel has alpha=28 slabs of beta=9 slots each, and the table is filled
to load 0.95. A key descends the slabs in order, scanning one beta-slot
sub-array per slab until it finds a free slot. At load 0.95 the early
(large) slabs are nearly full, so a key visits ~3.7 slabs on average
before finding room; mean probes = beta * E[slabs visited] ≈ 9 * 3.7 ≈
33. This is the funnel hash's expected O(log 1/delta) insert/search
cost materialized as a concrete number at this geometry, not a bug or
a miscalibration — the Zig and Python implementations agree on it to
four decimals.

---

## 4. WebGPU demo per-pass timings at 1M and 5M (round-5 browser verification)

Captured 2026-08-19 by driving the live demo (`index.html` served locally)
in a real browser on the dev machine (WebGPU adapter: nvidia lovelace).
Values are the sidebar telemetry panel read out verbatim after the sim had
settled (5M: ~60 frames after preset click; 1M: read shortly after a
`Set`-triggered rebuild, so its step latency still includes first-frame
transients — noted below). Default preset parameters: Fixed-Grid 2-Level
mode, p=2 quadrupole adaptive FMM, P2P radius 0.035, full x9 cell lists.

| Preset | FMM Build | FMM M2L+L2L | FMM L2P | Main Compute | Render | Total GPU | FPS  | Buffers | Cell occupancy |
| ------ | --------- | ------- | ------- | ------------ | ------ | --------- | ---- | ------- | -------------- |
| 1M     | 7.570 ms  | 0.269 ms | 5.674 ms | 90.563 ms | 1.687 ms | 105.783 ms | 10 | 192 MiB | avg 61.0/cell (128x128) |
| 5M     | 2.997 ms  | 0.270 ms | 4.221 ms | 59.284 ms | 1.772 ms | 68.565 ms  | 14 | 617 MiB | avg 76.3/cell (256x256) |

> **Note on the M2L column label:** the sidebar now labels this column
> "FMM M2L (M2L+L2L)" because it includes the CPU-timed `l2l_down` pass
> (the downward L2L chain has no dedicated GPU timestamp slot — all 10
> timestamp slots are used by build/m2l/l2p/main/render — so it is
> CPU-timed and added to the M2L metric). The TELEM JSON exposes it
> separately as `fmmL2lDownMs`. The historical values above were captured
> before this change; they reflect the M2L-only GPU timestamp and do not
> include L2L. At these scales L2L is sub-0.01 ms (a handful of polynomial
> shifts per level), so the column value is effectively unchanged.

5M Extreme Mode block (verbatim): `ACTIVE (WebGPU)`, P2P Budget `6/leaf`,
Peak Step `126.90 ms`, Avg Step (60-frame rolling) `39.78 ms (n=60)`.

Caveats, stated plainly:

- The 1M row's "Main Compute" (90.6 ms) exceeds the 5M row's (59.3 ms)
  because the 1M readout was taken within a few frames of the rebuild
  (cold JIT/warmup transients), while 5M shows a settled rolling state.
  The honest comparison for steady state is 5M Avg Step 39.78 ms vs 1M
  single-frame snapshot 105.8 ms total; treat the 1M pass split as
  transient-contaminated, not a clean scaling point.
- "Main Compute" (the P2P near-field pass) dominates both profiles
  (5M: 59.3 of 68.6 ms = 86%); the FMM multipole chain (Build+M2L+L2P =
  7.5 ms) is a small fraction. Any 5M optimization work should target the
  P2P pass first — this is the measured evidence for the round-6 priority.
- "GPU Complete" (~23-33 s) is a cold-start/first-dispatch artifact, not
  a steady-state number; ignore it.

---

## 5. Round-6 changes (WGSL P2P tuning, funnel-hash WGSL port, radial Taylor unification)

### 5.1 WGSL P2P near-field tuning (task 6.2)

`leafBitsForCount(n)` in `index.html` now ships an auto-tuning formula
that picks the leaf grid resolution from the actual particle count
instead of a fixed table, plus a `?leafBits=` URL override for
hands-free sweeps. The formula targets ~50-80 particles/cell at the
chosen depth (the §4 sweet spot where the P2P pass is bounded but the
FMM multipole chain stays cheap). No new buffers; the leaf resolution
still drives `ensureWebGPUCapacity` as before. Browser verification of
the 5M steady-state improvement vs the §4 baseline (Main 59.3 ms, Avg
Step 39.8 ms, Total 68.6 ms) is the MAIN agent's review task per the
plan — the executor does not claim a measured speedup here, only that
the knob exists and is auto-tuned.

### 5.2 Funnel-hash WGSL port (task 6.4)

The Farach-Colton, Krapivin, & Kuszmaul (2025) funnel schedule is now realized
in WGSL alongside the existing open-addressing hash. New WGSL kernels
in the FMM shader module:

- `funnel_clear` / `funnel_build` / `funnel_scan` / `funnel_scatter` —
  the four-pass build that produces the same contiguous `sortedIndex`
  ranges as the counting sort and the `eh_*` open-addr path, but
  addressed through the funnel table.
- `funnelProbe` (in both the compute and FMM shaders) — the full-descent
  search: alpha slabs of beta-slot sub-arrays, then overflow B
  (uniform probing, `bAttempts` slots), then overflow C (two-choice
  buckets of `cBucketSlots` slots). Bounded by
  `probe_bound = alpha*beta + bAttempts + 2*cBucketSlots`.
- `funnelProbeCount` — an atomic counter accumulated by `funnel_build`
  and read back periodically for the TELEM probe-bound assertion.

The funnel table reuses the `eh*` storage buffers (sized
`2*leafCells >= funnelGeometry.totalSize`), so no new large buffers are
allocated — only the small `funnelGeom` geometry buffer (packed u32
array of offsets/counts/salts) and the 4-byte probe counter. The
geometry is built in JS by `computeFunnelGeometry(capacity, delta=0.05)`
and uploaded once per leaf-resolution change.

Insert is parallel-CAS greedy (full descent + overflow) without the
paper's "stop early if overflow empty" shortcut — that shortcut needs a
serial overflow-emptiness check the compute model cannot express without
a global barrier mid-descent. The full-descent variant is correct (slots
only ever go EMPTY -> occupied, so parallel CAS never displaces a
resident key) and still bounded by `probe_bound`.

New bench axis `+funnel-hash` isolates the funnel path on the live
render loop, alongside the existing `+openaddr-hash` axis. New URL param
`?hashverify=1` forces the funnel path on at startup and runs a one-shot
readback at frame 40 that compares the funnel-built `ehStart`/`ehCount`
against a JS-rebuilt counting-sort reference, printing
`HASHVERIFY PASS/FAIL {json}` to the console with mismatch/missing
counts and the measured mean insert probes vs `probe_bound`. TELEM
autoprint lines now include a `funnel` block with `probeBound`,
`totalInsertProbes`, `meanInsertProbes`, `alpha`, `beta`, `tableSize`,
and `probeBoundOk` (boolean: true iff meanInsertProbes <= probeBound)
when the funnel path is active. The top-level TELEM JSON also includes
`fmmL2lDownMs` (the CPU-timed L2L downward pass, added to `fmmM2lMs` in
the sidebar's "FMM M2L" row) so headless verification can confirm the
full far-field cost is reported, not just the M2L upward accumulation.
Both fields apply the same adaptive guard as the sidebar: when the
adaptive FMM path is active (or the fixed-grid FMM is off), the cached
`webgpuL2lDownMs` is stale and both `fmmL2lDownMs` and the
`fmmM2lMs` l2l_down addend are reported as 0, so the TELEM JSON never
inflates the far-field cost with a stale CPU-timed value.

### 5.3 Radial Taylor unification (task 6.5)

The three copy-pasted radial Taylor FMM engines
(`gaussian2d_fgt.py`, `yukawa3d_fmm.py`, `screened_yukawa2d_fmm.py`)
now share a single `core/radial_taylor.py` module that holds the
dimension-parameterized P-tensor builder, the polynomial helpers, the
multi-index/factorial helpers, and the ring-2 flat scheme driver
(P2M / near-field / M2L / L2P). Each engine is now a thin subclass of
`RadialTaylorFMM` that supplies only its G_n family and near-field
kernel. All three test files are untouched (`git diff` shows no changes
to them) and `tools/run_all.py` reports 15 PASS / 1 SKIP / 0 FAIL — the
refactor is behavior-preserving by the test guardrails.

### 5.4 Near-field sampling on GPU (round-7 honesty pass)

The WebGPU demo's near field is **sampled, not exact**. This was
previously undocumented (finding F-15). Two sampling behaviors exist:

1. **Adaptive FMM path** (inline in `index.html`; a deliberately divergent
   uniform-grid CSR variant lives at
   `core/webgpu_kernels/adaptive_fmm.wgsl` — see its coverage contract;
   the shipped occupancy-adaptive kernel is the inline one): the tree is an
   occupancy-adaptive quadtree (leaves hold ≤ leafTarget = 16/32/64
   particles tiered by N, refined to max depth 4–10 chosen from N), and
   List-1 (adjacent-leaf) neighbors are subsampled at a **total
   per-particle budget** — full List-1 traversal (4096 cap) at N ≤ 32k,
   48 samples mid-N, and 32 above 2M (the §12.7 raise + §13.2 fix;
   the sidebar "Adaptive Near-Field Budget" select and `?p2pbudget=N`
   override the total), **spread across** the adjacency list — not
   allocated per adjacent leaf.
   (The pre-round-15 scheme of 24/12/6 **per adjacent leaf** × up to 32
   List-1 leaves made adaptive ~11× heavier than the fixed-grid 3×3 walk
   and, under core collapse, multiplied `leafWeight × sampleWeight` into
   force outliers — §10.3–10.4 and the live-demo collapse the user sees.)
   Each sample is reweighted by
   `min(leafOccupancy/sampleCount, 8) * min(list1Count/nl, 4)` so normal
   trees stay approximately unbiased while collapsed max-depth cores
   cannot explode force magnitudes. Three further serial loops are
   bounded the same way (stride-sampled, rotation-decorrelated,
   reweighted): P2M moment construction (≤512 particles/leaf), the List-4
   P2L translation in M2L (≤128 particles/source leaf), and the List-1
   adjacent-leaf enumeration itself (≤32 leaves/particle) — without those
   caps a collapse of all particles onto the center of mass stalls the
   frame (single thread per leaf/node walking thousands of entries).
   The sidebar "P2P Near Field" row reports the total samples/particle
   tier. The flat metadata (tree + interaction lists) is cross-validated
   against the direct O(N²) sum by `tools/validate_adaptive_js.py`
   (structural invariants + emulated-kernel potentials/forces), and since
   round 10 the executed WGSL multipole chain itself is validated on GPU
   by `tests/core/test_adaptive_wgsl_numeric.py` (full clear→P2M→M2M→M2L→
   L2L→L2P chain vs direct O(N²) and the `core/adaptive_fmm.py`
   reference; that test is what caught the per-level over-dispatch bug
   where surplus `ceil(count/64)` threads spilled into the next level's
   nodes and doubled their M2L/L2L local expansions — now guarded by the
   `dispatchCount` uniform field).
2. **Fixed-grid path** (`index.html` fixed-grid mode): dense cell lists
   beyond `p2pListCap` are stride-subsamples, stride-weighted (also
   unbiased in expectation, per-particle variance grows with stride).

Consequences for the demo's force accuracy are visual (the renderer
absorbs small per-particle noise) and are **not validated** against a
direct reference at 5M — the demo's `?probe=1` far-field check only
exercises the multipole (M2L/L2P) path, not the sampled P2P near field.
Anyone reusing these kernels for physics must either raise the budget
to cover the full neighbor list or accept the sampling noise. The
standalone `tree_free_fmm.wgsl` P2P is a separate, O(N²)-masked
all-tile scan (finding F-16) — also not the demo's cell-listed kernel;
see task T-E1 for the planned CSR port.

Cross-referenced from the README's WebGPU bullet.

---

## 6. Round-7 changes (CSR P2P, CUDA hash honesty, JAX flat pipeline)

### 6.1 CSR P2P in standalone kernels (task T-E1)

`core/csr_p2p.py` provides a reusable `csr_p2p_near_field` function that
replaces the per-cell `np.where(inverse == c)` O(N*K) scans with CSR
gathers (O(N) total). The CSR helper `core/_csr.py` is shared between:
- `bioinformatics/core/fast_multipole_kernel.py` (T-C6 bio CSR batching)
- `core/fast_vectorized_fmm.py` (T-E4 cleanup)
- `core/csr_p2p.py` (T-E1 standalone)

The standalone test (`python -m core.csr_p2p`) verifies the CSR P2P
matches the direct near-field reference to 2.8e-16 rel-L2 at N=200.

### 6.2 CUDA hash honesty (task T-E2)

The CUDA kernel (`core/cuda_kernels/tree_free_fmm_kernel.cu`) now
explicitly documents that its hash insert is a generic lock-free
open-addressing scheme (atomicCAS with linear probing), NOT the
Farach-Colton, Krapivin, & Kuszmaul (2025) funnel hash schedule. The FKK citation
has been removed from the CUDA file. The funnel hash lives in
`core/elastic_hash.py` (Python) and the WGSL demo (`index.html`).

### 6.3 JAX flat pipeline (task T-D4)

`core/jax_tree_free_fmm.py` now ships an assembled flat-scheme 2D
log-kernel FMM pipeline (`jax_flat_fmm_evaluate`) that wires the
verified adaptive FMM operator primitives (P2M, M2L, L2P, P2P) into a single
jitted function. The pipeline uses `jnp.argsort` for on-device binning
(the funnel hash stays CPU/WGSL — JAX with x64-disabled cannot express
the 64-bit funnel mixer). Test `core/test_jax_pipeline.py`
cross-validates vs `jax_direct_nbody_reference`; skips gracefully if
JAX is not installed.

### 6.4 CUDA kernel CSR cell lists (task T-E1 CUDA side)

The CUDA kernel `core/cuda_kernels/tree_free_fmm_kernel.cu` now builds
its per-cell particle ranges with the same counting-sort CSR pipeline as
the WGSL reference (`clear_cells` / `count_cells` / `scan_cells` /
`scatter_cells` in `index.html`), exposed as four `__launch_bounds__`
kernels: `clear_cell_counts` → `count_cells` → `scan_cells` →
`scatter_cells`. The result is the standard CSR triple
`cellStart` / `cellCount` / `sortedIndex`, so particles of cell `c`
occupy `sortedIndex[cellStart[c] .. +cellCount[c]]`. The fused L2P + P2P
`evaluate_adaptive_fmm_2d_kernel` now iterates the 3×3 leaf neighborhood via
those contiguous ranges instead of the previous O(N) masked scan, and the
old generic open-addressing hash insert (atomicCAS + linear probing with
`particle_next_ptrs`) has been removed. The far-field L2P multipole math
is unchanged. The scan is a single-block (256-thread) exclusive prefix
sum mirroring the WGSL one-workgroup scan — no thrust/CUB dependency,
CUDA runtime only. A small host launcher `launch_tree_free_fmm_2d`
chains the four build passes and the evaluate kernel (allocating
transient CSR buffers per call; production pipelines should reuse them
via ping-pong per §1). The HIP port (`core/hip_kernels/`) is a separate
file and is not affected by this change.

### 6.5 Standalone WGSL kernels — counting-sort CSR cell lists (task T-E1)

The two standalone WGSL kernel files
(`core/webgpu_kernels/tree_free_fmm.wgsl` and
`core/webgpu_kernels/adaptive_fmm.wgsl`) now build their near-field
neighbor lists with the same four-pass counting-sort CSR pipeline as the
`index.html` reference (and the CUDA kernel in §6.4):

  1. `clear_cells`  — zero `cellCount[c]` for every grid cell
  2. `count_cells`  — `atomicAdd(&cellCount[cellIndex(pos)], 1)` per particle
  3. `scan_cells`   — single-workgroup (256-thread) exclusive prefix sum
     over `cellCount`, writing `cellStart[c]` and initializing
     `cellCursor[c] = cellStart[c]`
  4. `scatter_cells`— `slot = atomicAdd(&cellCursor[cell], 1);
     sortedIndex[slot] = particleId`

The P2P sections then iterate the 3×3 leaf-cell neighborhood via
`sortedIndex[cellStart[c] .. cellStart[c]+cellCount[c]]` (CSR layout),
reading particles contiguously instead of the previous O(N²) masked
all-tile scan (tree_free_fmm.wgsl) or budgeted random List-1 sampling
(adaptive_fmm.wgsl).

**tree_free_fmm.wgsl**: The masked all-tile scan (128-thread tiles with
a Chebyshev-center adjacency mask) and the `var<workgroup>
tile_particles` shared-memory buffer have been removed. `SimulationParams`
gains `grid_dim` and `grid_origin` fields; four new storage buffers
(`cellCount`/`cellCursor`/`cellStart`/`sortedIndex`) are bound at
bindings 6–9. The far-field L2P multipole evaluation is unchanged.

**adaptive_fmm.wgsl**: The l2p kernel's List-1 budgeted-sampling P2P
(stride-based decorrelated subset with per-N budget caps) has been
replaced by 3×3 CSR iteration over a uniform-grid overlay. NOTE the
coverage contract (documented in the kernel header): the overlay P2P is
exactly-once only for uniform-depth trees where leaf == overlay cell; with
occupancy-adaptive metadata, same-coarse-leaf pairs can fall outside the
3×3 overlay and be silently missed. The occupancy-adaptive reference
(mixed-depth List-1 P2P, Lists 1–4) is the inline index.html kernel,
cross-validated by tools/validate_adaptive_js.py. A new
`GridParams` uniform (binding 16) supplies `gridDim`/`gridOrigin`/
`cellSize`; four cell-list storage buffers are bound at 17–20. The
far-field multipole passes (p2m/m2m/l2l/m2l), the L2P local-expansion
math, and the List-3 multipole ring in l2p are unchanged. The
`zeroNearP2P` probe flag still gates the P2P section.

### 6.6 5M P2P optimization ladder (task T-E3)

The P2P near-field pass is the measured 86% hotspot at 5M particles (§4:
Main Compute = 59.28 ms of 68.57 ms total; the FMM chain is 7.5 ms).  The
pass is bandwidth-bound: every lane of a target-leaf workgroup re-fetches
the 9 (or 25) neighbor cell lists from global memory.  The optimization
ladder below is ordered by (expected gain)/(implementation risk), each
step independently measured via the `?autoprint=1` 60-frame rolling mean
protocol.  A documented non-improvement is a valid result (house law).

**Sweep protocol generator:** `python tools/diag_p2p_ladder.py` prints the
URL protocol and expected baseline numbers for each step.

#### Step (a): leafBits sweep — DONE (auto-tune formula)

`leafBitsForCount(n)` in `index.html` ships an auto-tuning formula that
picks the leaf grid resolution from the actual particle count:

```js
function leafBitsForCount(n) {
    if (LEAF_BITS_OVERRIDE >= LEAF_BITS_MIN && LEAF_BITS_OVERRIDE <= LEAF_BITS_MAX) {
        return LEAF_BITS_OVERRIDE;
    }
    const targetSide = Math.sqrt(n / 12);
    let bits = Math.ceil(Math.log2(targetSide));
    if (bits < LEAF_BITS_MIN) bits = LEAF_BITS_MIN;
    if (bits > LEAF_BITS_MAX) bits = LEAF_BITS_MAX;
    return bits;
}
```

The formula targets ~12 particles/cell (the §4 sweet spot where the P2P
pass is bounded but the FMM multipole chain stays cheap).  A `?leafBits=N`
URL override (clamped to [6, 10]) enables hands-free sweeps.

Auto-tune table:

| N        | auto bits | side  | avg/cell |
|----------|-----------|-------|----------|
| 100k     | 7         | 128   | 6.1      |
| 500k     | 8         | 256   | 7.6      |
| 1M       | 9         | 512   | 3.8      |
| 2M       | 9         | 512   | 7.6      |
| 5M       | 10        | 1024  | 4.8      |

Browser measurement of the 5M steady-state improvement vs the §4 baseline
(Main 59.3 ms, Avg Step 39.8 ms, Total 68.6 ms) is the MAIN agent's review
task — the executor does not claim a measured speedup here, only that the
knob exists and is auto-tuned.  The `?leafBits=6..10` sweep at 1M/2M/5M
with `?autoprint=1` (60-frame Avg Step) is the protocol for pinning the
formula to the measured optimum.

#### Step (b): LDS staging — DESIGN (not yet measured)

**Design:** one workgroup per target leaf cell; stage each neighbor cell's
slice into `var<workgroup>` arrays (cap = `p2pListCap`, early-out beyond);
process one slice per barrier.  Removes ~9× redundant global reads per
lane.

**Risk:** LDS pressure drops occupancy — measure both Avg Step and
occupancy.  The `var<workgroup>` staging buffer is sized to
`p2pListCap * 2` floats (position only, x/y packed) per neighbor cell,
capped at 9 * p2pListCap * 2 = ~4.5 KB at p2pListCap=256.  This fits within
the typical 16 KB LDS limit but may reduce occupancy from 4 to 2
workgroups per CU on some architectures.

**Status:** design documented; browser measurement required via
`?ldsStaging=1&autoprint=1` (URL param not yet wired — the kernel variant
is a future commit).  Acceptance: 5M Avg Step improves ≥ 15% AND occupancy
does not drop > 30%.

#### Step (c): AABB cull — DESIGN (not yet measured)

**Design:** per-cell min/max extents (computed in a new kernel
`compute_cell_aabb`) → skip neighbor cells whose extent is beyond the P2P
radius.  The P2P loop already checks `r2 < rcSq` per candidate, but it
still iterates the full neighbor cell list; the AABB cull skips the
iteration entirely for cells whose closest extent is beyond `rc`.

**Telemetry:** the TELEM JSON would gain a `p2pCullFraction` field
(fraction of neighbor cell iterations skipped by the AABB test).

**Status:** design documented; browser measurement required via
`?aabbCull=1&autoprint=1` (URL param not yet wired — the AABB kernel is a
future commit).  Acceptance: 5M Avg Step improves ≥ 15% AND cull fraction
reported > 0.  The expected cull fraction is low at 5M (dense cells fill
most of the 3×3 neighborhood), but higher at 1M where cells are sparse —
the ladder step may be more valuable at lower N.

---

## 7. Round-9 changes (adaptive P2P reweighting, uncapped scheduler, UI reduction, honest cross-benchmark)

### 7.1 Adaptive List-1 P2P reweighting (square-artifact fix)

The adaptive L2P's budgeted List-1 near field subsampled dense adjacent
leaves (`cnt = min(occupancy, budget)` id-decorrelated samples) but gave
every sample weight 1, unlike the fixed-grid path's stride weighting. In
max-depth-saturated leaves the near-field force was therefore
under-estimated by `occupancy/budget` — a systematic error pattern aligned
to quadtree cells that visibly rendered as square artifacts drifting with
each metadata rebuild (worst at 5M: budget 6 vs leafTarget 64). Fixed by
multiplying each sampled charge by `occupancy/sampleCount`; the estimator
is now unbiased in expectation, matching the fixed-grid scheme. Verified
visually at 500k and 5M (smooth velocity heatmaps, no cell-aligned
structures) and by `tools/validate_adaptive_js.py` 7/7 (unchanged at
budget-saturating N where the weight is 1). §5.4 above documents the
sampling contract as it now stands.

### 7.2 Uncapped frame scheduler (vsync cap removed)

`requestAnimationFrame` is vsync-locked to the display refresh rate
(usually 60 Hz), which capped the demo's visible FPS regardless of GPU
headroom. The loop now defaults to **uncapped**: each step re-schedules
through a `MessageChannel` macrotask (no `setTimeout` 4 ms nesting clamp),
so the FPS metric reports true steps/sec (e.g. 15k vortex/sph measure
~2000 steps/sec vs the old 60; 500k galaxy measures 93–133). When the tab
is hidden the scheduler falls back to rAF so an unattended tab cannot burn
GPU indefinitely. `#btnFrameCap` toggles in-page; `?capped=1` /
`?uncapped=1` force a mode.

### 7.3 Option-space reduction

The "Sorted payload" and "Adaptive node hash" checkboxes were removed from
the UI (both fixed to the fast defaults); their A/B axes remain available
as URL params (`?sortedPayload=0`, `?adaptiveHash=0`). The remaining
benchmark axes are the meaningful ones: scenario, far-field model, FMM
order, fixed vs adaptive FMM, and near-field hash backend.

### 7.4 Measured verdict on "adaptive + funnel hashing" (500k cross-benchmark)

> **Superseded by §8** (2026-08-23): the `materialize_ranges` pass
> equalized the near-field hash backends, the metadata rebuild moved to a
> Web Worker, and adaptive now *wins* at 120k. Numbers below are the
> round-9 record.

`tools/browser_crossbench.js` (headless Chromium/Edge, WebGPU D3D11,
interleaved rounds, per-config medians — single sequential runs proved
noise-dominated on this machine) at N=500k, galaxy, p=2, uncapped:

| config                    | median steps/sec |
|---------------------------|------------------|
| fixed + counting-sort     | 126              |
| fixed + open-addressing   | 133              |
| fixed + funnel            | 131              |
| adaptive + node-hash dir  | 110              |
| adaptive + leafForParticle| 94               |

Larger N (median of 2 rounds): 2M fixed 58 vs adaptive 29; 5M fixed 28 vs
adaptive 11.

Honest conclusions, replacing the earlier "adaptive + funnel is superior"
hypothesis:

1. **The funnel occupied-node directory is a real adaptive-pipeline win**:
   +17% steps/sec at 500k (110 vs 94). The compact directory (~23k nodes
   at 500k) stays cache-resident where the 2 MB `leafForParticle` array
   does not. It stays the default (`?adaptiveHash=0` disables).
2. **The near-field hash backends are within ~5% of each other** in the
   fixed-grid pipeline (133/131/126). Funnel's advantage in that slot is
   its O(1) worst-case probe bound and compactness, not throughput on
   this GPU; counting-sort remains the simplest and is within noise.
3. **The adaptive FMM costs throughput vs the fixed grid at every
   measured N** (110 vs 126 at 500k; 29 vs 58 at 2M; 11 vs 28 at 5M). The
   dominant cost is the CPU-side metadata pipeline (position readback +
   JS quadtree rebuild + reupload every rebuild interval), not the GPU
   evaluation. The adaptive pipeline's value in this demo is accuracy on
   clustered distributions at bounded node counts (validated by
   `tools/validate_adaptive_js.py`), not speed; a GPU-resident adaptive
   build would be the path to closing the gap (future work).

Note on the near-field axis and adaptivity: in adaptive galaxy mode the
near field comes from the quadtree List-1 ranges, so the shared cell-list
hash axis (counting/openaddr/funnel) is inactive there by design
(`wantsCellListsThisFrame()`); the only funnel/adaptive composition in
the demo is the occupied-node directory of item 1.

## 8. Round-11 changes (hash-range materialization, worker-side tree build, vsync default)

### 8.1 `materialize_ranges`: one hash probe per cell, not per neighbor visit

The elastic/funnel near-field modes previously resolved a leaf cell's CSR
`(start, count)` through a hash probe on every neighbor-cell visit in every
P2P consumer (~9 probes per particle per frame; the probe is a divergent,
data-dependent load chain — the dominant reason the hash modes trailed the
counting sort in the near field). The new `materialize_ranges` compute pass
(fixed-grid FMM module, dispatched right after `funnel_scatter`/`eh_scatter`
whenever a hash backend ran) resolves EVERY leaf cell's range once — ~16k
probes per frame at leafBits 7 instead of ~9·N — and writes the results
into the dense `cellStart`/`cellCount` arrays that the counting sort
already uses. All `cellRangeOf` consumers (galaxy P2P, boids, vortex/SPH,
`build_moments`, `build_boid_centroids`) now do two direct u32 loads with
zero probe chains; the counting-sort backend was already direct. The hash
table remains the structure that gets BUILT each frame (worst-case probe
bound, compactness); only the consumer-side read path is dense.

Measured (crossbench, RTX 4070 SUPER at ~80% background GPU load, ratios
same-run): the three near-field backends are equal within noise at every
measured N — 500k: counting 172 / open-addr 174 / funnel 172; 5M:
12 / 10 / 10. The hash-modes-slower gap is closed.

### 8.2 Adaptive metadata build moved off the main thread (Web Worker)

"Tree-free" is about the per-step data structure — hash directories over
an implicit quadtree lattice, no pointers, no per-step tree traversal —
but the occupancy pass that decides WHICH lattice cells exist still has to
run somewhere, and round 9-10 ran it synchronously on the main thread: a
~60 ms (200k) to ~400 ms (5M) hitch every refresh interval. The rebuild now
runs in a Web Worker assembled at runtime from the page's own inline
script: `getAdaptiveMetaWorker()` slices the pure builder span verbatim
(the same source-slicing trick `tools/emit_adaptive_meta.mjs` uses, so the
worker cannot drift from the page), the main thread compacts a x/y-only
`Float32Array` snapshot (the builder reads nothing else; a new `posStride`
parameter defaults to 4 so existing callers and the Node harness are
unchanged) and transfers it; the worker transfers the resulting metadata
arrays back zero-copy. Generation counters drop stale builds after
reset/mode-switch/particle-count changes; Workers unavailable => one-strike
fallback to the old synchronous path.

Verified in-browser: exactly one blob Worker created, zero errors, and the
per-rebuild `p2pBudget` console warnings (emitted from inside the builder)
carry different max-leaf occupancies across refreshes — the worker is
genuinely rebuilding against live drifting positions. Worst frame step at
500k adaptive dropped from ~60 ms+ hitches to a 31 ms max over 12 s
(median 15.2 ms). A fully GPU-resident adaptive build (histogram + prefix
sum + refine in WGSL) remains the future path to removing the readback
copy as well.

### 8.3 VSync default, benchmark opt-in

The uncapped scheduler is now OPT-IN (`?uncapped=1` or the Frame Cap
button). The page previously started uncapped, which read as "lags by
default" — uncapped mode runs one sim step per loop iteration and the
render throttle can't hide the GPU saturation from the compositor on every
machine. Default = vsync-locked rAF loop (labeled "Rendered FPS (vsync)");
benchmark mode relabels the metric "Steps/sec (benchmark)" with a tooltip.
`tools/browser_crossbench.js` appends `?uncapped=1` itself and now uses
the full Chromium build (the headless shell has no WebGPU adapter and
silently falls back to WebGL2), scales the init wait with N, and supports
`CONFIG=<label>` for per-config process isolation at 5M (the fifth
in-process navigation stalls on cumulative GPU memory at that size).

### 8.4 Updated verdict (replaces §7.4 items 2-3)

Cross-benchmark, same loaded-GPU environment (ratios same-run; absolute
numbers depressed ~proportionally):

| N     | count | openaddr | funnel | adaptive+dir | adaptive+no-dir |
|-------|-------|----------|--------|--------------|-----------------|
| 120k  | 236   | 214      | 226    | 356          | 509             |
| 500k  | 172   | 174      | 172    | 56           | 45              |
| 5M    | 12    | 10       | 10     | 7            | 5               |

1. Adaptive CROSSES OVER the fixed grid: at 120k it is ~1.5-2.2x FASTER
   (few tree nodes => cheap per-level chains, while fixed always evaluates
   the full 128x128 moment lattice); at 500k+ the adaptive node count
   grows (~23k nodes at 500k) and fixed wins. Pick by distribution and N.
2. The funnel node directory pays off as the tree GROWS (7 vs 5 at 5M,
   56 vs 45 at 500k) but costs on small trees (356 vs 509 at 120k) —
   probe locality vs one-indirection trade.
3. Near-field hash backends are equal within noise at all N (§8.1).

The implementation-details sidebar line is plain ASCII prose (the raw
`?param=...` A/B suffix removed; the Copy button copies it) and the 1euro
toggle stays boids-only (hidden elsewhere).

## 9. Round-12 findings: adaptive tree-shape sweep (negative), Python-side rewrite (strong positive)

### 9.1 Adaptive tree-shape sweep at 500k — the auto tier is already optimal

Motivation: Lashuk et al. (2012) report that increasing leaf capacity
(shorter tree) shifts work from memory-bound M2L traversal to compute-bound
near-field and helps GPUs. `tools/adaptive_shape_sweep.js` swept
`?leaftarget=` x `?adapdepth=` (new URL overrides, auto = 0) at N=500k,
3 interleaved rounds, medians:

| config | median steps/s | nodes | maxLeafOccupancy |
|---|---|---|---|
| auto (lt16/d8) | **48** | 23010 | 288 |
| lt32 (d7) | 41 | 7288 | 1054 |
| lt64 (d7) | 35 | 5840 | 1060 |
| lt128 (d6) | 33 | 1833 | 4006 |
| lt64/d8 | 35 | 12579 | 285 |
| lt128/d9 | 27 | 11677 | 128 |

Result: NEGATIVE for the Lashuk heuristic — the auto tier (small leaves,
deep tree) is fastest; every shallower shape is slower, and deeper-than-auto
is slowest. Explanation: this demo's near field is budget-capped
(p2pBudget subsampling), so enlarging leaves does not add compute-bound
work for the GPU to amortize — it only coarsens the far field and starves
the near-field sample fraction (24/4006 on the densest leaf). The
classical compute-bound/memory-bound trade does not apply here. Defaults
unchanged; `?leaftarget=` / `?adapdepth=` stay as user tuning axes.

Conclusion for the adaptive-vs-fixed gap at 500k+ (48-56 vs 172 steps/s):
it is the divergent per-target chain walk itself, not tree shape. The
structural fix is materialized per-target far-field interaction lists
(built once per metadata refresh on GPU, evaluated as flat gathers like
the fixed grid) — the same move that closed the hash-mode near-field gap
in §8.1. Documented as the identified next step; not implemented this
round.

### 9.2 Artifact re-check (adaptive, 2.5x zoom)

20 s of adaptive galaxy at 500k, canvas + two 2.5x zoom crops inspected:
no grid-aligned seams, no square/rectangular clumping, no straight-line
density discontinuities at refinement boundaries, no rendering glitches.
The §8.1 materialize_ranges fix holds structurally.

### 9.3 Python core: the "NOT faster than direct at N=2000" rows are gone

The level-batched engine (2:1-balanced CGR88 with per-(level,offset) M2L
matrices, vectorized List-4 P2L via reduceat segment sums, CSR near-field)
-- originally shipped as `core/adaptive_fmm_fast.FastAdaptiveFMM` and since
consolidated into `core/adaptive_fmm.AdaptiveFMM` (alias `FastAdaptiveFMM`;
the classical per-box engines stay in that module as
`ClassicalAdaptiveFMM` / `TreeFreeElasticAdaptiveFMM` slow cross-validation
references, exercised by tests/core/test_adaptive_fmm_fast.py and
tests/core/test_adaptive_fmm_reference.py): 30 ms at N=2000 (23x faster
than the classical per-box engine, 1.2-2.4x faster than direct) at the SAME
2e-7 rel-L2; 91-112x at 32k; 465x at 128k (direct measured 296 s).
`FastVectorizedFMM` M2L rewritten as an exact FFT lattice convolution
(kernel = per-offset M2L matrix, 3x3 near block zeroed, side-2R padded):
710 -> 90 ms at N=2000, unchanged accuracy. See BENCHMARKS.md "Core FMM"
for tables, automated crossover headline, and plots
(assets/core_fmm_scaling_{loglog,linear}.png).
## 10. Round-13: materialized far-field interaction lists

### 10.1 Design

Section 9.1 identified materialized per-target far-field lists as the
structural fix for the adaptive-vs-fixed gap. This round implements it: each
target LEAF's far-field interaction list is resolved ONCE per metadata
refresh (worker-side, during the existing Web Worker build of section 8.2)
and evaluated on the GPU as a flat gather — the same move that closed the
near-field hash gap in section 8.1.

**CSR contents.** For every leaf `t`, the flat concatenation over levels
`l = t.level .. 1` of `List-2(ancestor_l(t))` — exactly the M2L source set
the per-level `m2l` + `l2l` chain delivers to `t`. Each entry is ONE u32:
source node index (low 22 bits) | operator row index (top 10 bits), row =
`l*49 + (dy+3)*7 + (dx+3)` for the source's cell offset from its List-2
target at that level (offsets are bounded to [-3,3] by the List-2
construction — children of the parent's 3x3 colleague ring). Entries are
grouped by descending level so the kernel shifts each level's gathered run
to the leaf center once per run. Per-node `start`/`count` ride the same
buffer as an interleaved 2N-word header.

**Operator table.** For a source at integer offset `(dx, dy)` from its
List-2 target at level `l`, the M2L delta is the fixed lattice vector
`(dx, dy)*2^-l`, so the whole (p+1)^2 complex operator collapses into a
dense row indexed by `(l, dx, dy)` — the same precomputed-matrix move as
`core/adaptive_fmm.py`'s per-(level, offset) M2L matrices (formerly
`adaptive_fmm_fast.py`; Gimbutas &
Greengard, 2012, FMMLIB2D `itable(-3:3,-3:3)`; Carrier, Greengard, &
Rokhlin, 1988). 11 levels x 49 offsets x 25 complex = 26,950
f32 (107,800 bytes), built in f64 by `buildFarOperatorTable()` with the
identical closed form as the WGSL `m2l` kernel (clog monopole log term,
(-1)-signed inverse powers, C(k+l-1, l) binomials), stored f32. The table
is position-independent but rides in the same storage buffer as the CSR
(header | entries | table) to respect the 16-storage-buffer per-stage
limit; `nodeParent`+`nodeFlags` were packed into one `nodeMeta` vec2
buffer (the packing the standalone kernel already used) to free the slot.

**New kernels** (identical text in index.html and
`core/webgpu_kernels/adaptive_fmm.wgsl` — `tools/check_wgsl_sync.py` now
verifies the shared functions textually, including the new ones):

- `p2l` — the List-4 P2L block of the old per-level `m2l`, dispatched once
  over ALL nodes (P2L has no inter-level dependency), writing ONLY each
  node's own P2L contribution into `locals`. P2L stays shared per
  (node, source-leaf) pair exactly as before.
- `far_gather` — one thread per leaf: walk the ancestor chain (<= 10
  `nodeMeta` pointer hops, once per LEAF not per particle), then for each
  CSR entry read the source's 5 moments and the operator row and apply a
  5x5 complex matvec (two loads + FMA — no clog/cdiv chains), accumulating
  per level and recentering each level's run onto the leaf with ONE exact
  L2L shift per level (composition of exact polynomial shifts equals the
  legacy chain's level-by-level shifts). The ancestors' P2L locals are
  folded in by the same shifts (a strict ancestor of a leaf is always
  internal, so no thread reads a slot another thread writes). List-3 M2P
  per particle and List-1 P2P in `l2p` are unchanged — List-3 sources can
  be several levels FINER than the leaf and hugging its boundary, where
  folding them into the leaf's local expansion would place source points
  inside the evaluation disk (|src - leaf center| >= 0.5w + w_src can fall
  below the 0.707w evaluation radius), so CGR88's per-particle M2P form is
  mathematically required there.

The legacy chain stays selectable with `?materializedFar=0` (default ON),
mirroring `?adaptiveHash=0`; the sidebar axis line shows `far=mat|chain`.

**Measured CSR sizes** (logged once per rebuild via console.debug from the
worker): 120k ~ 0.24-1.1M entries (1.0-4.6 MB header+entries, ~122
entries/leaf); 500k ~ 2.8-4.3M entries (11-17 MB, ~120/leaf) with the ops
table a constant 0.11 MB. Upload cost is part of the per-refresh
`uploadAdaptiveMetadata` (measured mean 17 ms / worst 29 ms per refresh at
500k, including all metadata arrays and occasional buffer + bind-group
recreation).

### 10.2 Validation

- `tools/check_wgsl_sync.py`: PASS — the new `p2l`/`far_gather` and the
  far CSR accessors are TEXTUALLY IDENTICAL between the page and the
  standalone kernel (the inline `readc`/`writec` were converted to the
  standalone's selector form to make this possible).
- `tools/validate_adaptive_js.py`: 7/7 scenes PASS for BOTH paths at the
  same tolerances as before (e.g. hardedge p=2: chain rel_pot 5.2e-4 /
  rel_f 1.8e-2, materialized identical; gates 3e-3 / 4e-2). The
  materialized-vs-chain deviation in the f64 emulator is 2-14e-9 (pure
  reordering); the emitted JS operator table matches the Python reference
  closed form to 4.7e-8; the CSR content check verifies every leaf's entry
  list equals the ancestor-chain List-2 sets with correct operator rows
  and descending-level grouping.
- `tests/core/test_adaptive_wgsl_numeric.py::test_adaptive_materialized_far_field`
  (new): executes the actual WGSL `p2l` + `far_gather` via wgpu-py on a
  mixed-depth tree — rel-L2 vs masked direct O(N^2): pot 2.2e-5 / force
  6.0e-4 (gates 5e-4 / 5e-3); vs the legacy WGSL chain on the SAME tree:
  pot 1.1e-7 / force 5.5e-8 — the two paths agree to f32 rounding.

### 10.3 Measured verdict (the gap did NOT close at 500k+)

Cross-bench, same loaded-GPU environment, medians of 3 interleaved rounds
(2026-08-24; absolute numbers depressed by background GPU load, ratios
same-run):

| N     | count | openaddr | funnel | adapt+dir (mat) | adapt+no-dir | adapt+chain |
|-------|-------|----------|--------|-----------------|--------------|-------------|
| 120k  | 225   | 215      | 214    | 366             | 541          | 349         |
| 500k  | 160   | 167      | 161    | 52              | 43           | 52          |
| 2M*   | 34    | —        | —      | 11              | —            | —           |

(*isolated `CONFIG=` processes, medians of 3.)

1. **The materialized gather is a small real win where the far field is a
   visible share**: +5% at 120k (366 vs 349 same-run). At 500k it is
   performance-neutral (52 vs 52). Numerically it is exact (above), and it
   removes ~2*depth per-level dispatches per frame — but it does not move
   500k steps/sec.
2. **The section 9.1 hypothesis is refuted by direct measurement**: the
   divergent per-target far-field walk is NOT the 500k bottleneck.
   Decomposition at 500k (3 rounds each, same protocol): adaptive default
   (near-field P2P budget 24/leaf) 50; budget 6/leaf 184 (3.7x, matching
   the fixed grid); budget 1/leaf 316 (6.3x); multipole order p=0 50 (no
   effect); far chain vs mat 49 vs 50 (no effect). The dominant cost is
   the l2p NEAR FIELD — the List-1 budgeted walk (up to 32 adjacent leaves
   x budget samples per particle, ~11x the fixed grid's ~69-pair 3x3
   uniform neighborhood at leafBits 8) — plus the metadata refresh
   pipeline.
3. **Why mat = chain at 500k despite the table**: the per-leaf CSR
   DUPLICATES each ancestor's List-2 across all descendant leaves (~4.3M
   entries at 500k vs ~1M per-node List-2 pairs for the chain, a ~4x
   duplication factor), which cancels the ~3-4x per-operator gain of FMA
   matvecs over the inline clog/cdiv math. The duplication-free variant
   (per-level table-based m2l into shared per-node locals + far_gather
   folding only) is the obvious refinement, but per (2) it would not move
   the 500k number either.
4. **Adaptive throughput is strongly phase-dependent** — a measurement
   caveat for every table in this file: the galaxy ICs are unseeded
   (`Math.random`), so the quadtree swings between ~1k and ~55k nodes over
   one run and adaptive steps/sec swings 24-213 with it (the 500k
   adaptive+dir rounds above: 52, 51, 128). Section 8.4's 56 and section
   9.1's 48 were single-phase samples of the clustered phase the
   crossbench window lands in; medians over interleaved rounds are the
   honest comparator.

**Bug fixed along the way (pre-existing since round 12)**: the metadata
worker's glue did not declare `adaptiveLeafTargetOverride` /
`adaptiveDepthOverride`, so every worker build threw ReferenceError and the
page silently fell back to synchronous main-thread rebuilds — visible only
in the diag channel's `errors` array (crossbench `diagErrRounds` was 2-3
of 3). With the glue fixed, the diag errors are empty and the far CSR
build + upload run off the main thread as designed. Crossbench now reports
`diagErrRounds: 0`.

### 10.4 Identified next step → **IMPLEMENTED (round 15)**

Close the near-field gap the same way section 8.1 closed the hash gap: the
adaptive List-1 walk is now budgeted **PER PARTICLE** (total sample budget
48/32/16 by N, spread across the leaf's adjacency list, stride-reweighted
with `sampleWeight` capped at 8 and `leafWeight` at 4 under collapse), not
per adjacent leaf. Section 10.3 measured the old per-leaf scheme at 500k:
default 24/leaf → 50 steps/s; 6/leaf → 184; 1/leaf → 316. The new default
total budgets are calibrated to sit near the fixed-grid 3×3 work budget
while killing the collapse force-blow-up path. Re-run
`tools/browser_crossbench.js` after this change to refresh the absolute
steps/sec tables (the 24/12/6 per-leaf rows above are historical).

## 11. Round-14: core-side hash benchmark tie-in

The Python-side head-to-head (funnel vs linear probing vs CPython dict vs
the compiled Zig port, `core/bench_hash_backends.py`, table in
BENCHMARKS.md "Core hash tables") puts numbers on what the demo's
near-field backend switch CANNOT show: after `materialize_ranges` (§8.1)
the demo hash only builds the structure, so counting-sort / open-addressing
/ funnel measure equal within steps/sec. Where the funnel table's
guarantees are visible instead:

- **Worst-case latency**: absent-key lookups pay exactly the deterministic
  bound (157-543 probes at delta = 0.125-0.01) at any n, where linear
  probing's tail reaches 22,216 probes at n = 1M, alpha = 0.99. On GPU,
  that tail is warp divergence inside a memory pass — a per-thread
  worst-case cap is a latency guarantee median steps/sec cannot express.
- **The far-field node directory** (adaptive+dir vs adaptive+no-dir) is
  the one place the hash backend choice still moves demo numbers:
  no-dir wins on small trees (120k: 541 vs 366, §10.3), the directory is
  ahead at 500k (52 vs 43) and at 5M (7 vs 5, §8.4 — small sample).
  (Historical note: this section originally recommended keeping no-dir as
  the default; the default was subsequently switched to directory ON,
  matching the in-page measured ~+18% steps/s — see the `useAdaptiveNodeHash`
  note and §13.4.) Re-measure with interleaved medians when touching this —
  §10.3 (4) shows single-window samples swing 24-213 steps/s with the tree
  phase.

Audit: the elastic-hashing table (`ElasticBatchingHashTable`, paper
Section 2) is reference/experimental — it is not used by any pipeline
(Python, Zig, WGSL, or demo) and loses to the funnel table in every
measured regime (see BENCHMARKS.md section for the numbers).

## 12. Round-16: why the live demo underdelivered, and the fixes

User-visible symptoms at 5M (2026-08-26): "non-FMM" 60+ fps vs fixed FMM 43
fps vs adaptive 23 fps and collapsing after a short time, frame-gap outliers,
visibly different particle propagation; near-field hash backends showed no
fps change. Instrumented repro (tools/probe_collapse.js, tools/probe_resolve.js,
tools/probe_fields.js, N=500k, RTX-class GPU) found five root causes, all
fixed in index.html this round.

### 12.1 "non-FMM is faster" was a labeling problem, not a benchmark

`ff=off` computes NO particle-particle gravity — two analytic point masses
only (the O(N) visual baseline), while the FMM modes do full N-body work.
The UI called it "Analytic Dual-Core, **direct**", inviting exactly the wrong
comparison. Fixed:
- The option is relabeled "None (Analytic Dual-Core only, O(N))" and the
  axis reads `ff=gravity off (cores only)`.
- A real **Direct All-Pairs O(N^2)** mode was added (`selectFmmMode`
  `direct`, WGSL `direct` entry point in the adaptive module, `?fmmdirect=1`).
  Every softened pair per frame, same charge/softening conventions as the
  adaptive List-1 P2P; the main compute consumes it like an FMM far field.
  Measured at 120k: 19 fps / 55 ms GPU per frame vs fixed-grid FMM's 60 fps —
  the algorithmic improvement the demo promises is now observable in-demo
  (at 5M, direct costs seconds per frame).

### 12.2 Adaptive collapse #1: refresh thrash (perf)

The drift probe's pull-forward (`interval - max(6, interval>>2)`) with a
threshold keyed to the FINEST leaf width (0.3/2^depth ~ 1.2e-3) fired on
every refresh (typical inter-refresh drift is ~2e-2), pinning rebuilds at
the 6-frame minimum forever — 599 rebuilds in 60 s at 500k, each one
destroying/recreating ~13 GPU buffers + all bind groups (measured: 599/599
rebuilds changed buffer sizes) and uploading 20-40 MB. Fixes: grow-only
buffer capacity (25%/64 KiB rounding — 599→1 reallocation events in 60 s),
a two-slot snapshot pool for the worker handoff, drift pull-forward floored
at `max(12, interval>>1)` (half-interval), and the per-rebuild
`console.debug` CSR spam gated to every 64th build.

### 12.3 Adaptive collapse #2: depth the refresh cadence cannot track (physics)

The decisive measurement (tools/probe_resolve.js, 3-way comparison):
`resolveLeafNode` was internally CORRECT (matched pure root-descent 513/513,
always a containing terminal node), but the builder's `leafForParticle`
disagreed with the tree for **99.4% of particles** on a 0.7 s-old tree —
particles drift 6-15 finest-leaf widths per refresh interval (dt=0.024 at
60 fps; disk speeds ~0.03). Deeper refinement than ~1 leaf-width-per-refresh
cannot be tracked by a CPU rebuild; the A/B proved it: `?adapdepth=4` stable
(65→89 nodes) vs `?adapdepth=6` dispersing, `?p2pbudget=128/256` changing
nothing. Dispersal signature pre-fix: tree 22911 nodes at init → <300 after
60 s, luminance radius swinging 14.3-18.1 while fixed held 18.43.

Fix: `computeBoundedAdaptiveDepth()` caps depth at
`floor(log2(1 / measuredDriftPerRefresh))` (min 3, max 10), where the drift
is already measured by the staleness probe each refresh — the tree refines
deeper automatically in calm phases (observed: transient 132-167-node
refinement spikes during close encounters, back to ~50 in quiet phases).
Post-fix at 500k: tree stable 46-87 nodes, luminance signature matches
fixed mode (rx 17.8-18.1 vs 18.4), centroid oscillation matches fixed
(0.41↔0.73, same period), zero submit drops, 60 fps locked, Total GPU
2.3-7.7 ms/frame.

Also fixed along the way: `resolveLeafNode` could return an INTERNAL node
for drifted particles (empty children of split nodes do not exist — the
builder materializes only non-empty cells), which starved them of near
field; it now descends through `nodeChildren` to a terminal node.
The 500k dispersal was dominated by the depth problem, not this, but both
were real.

### 12.4 Total-GPU telemetry bug

Adaptive passes wrote no timestamp slots, but the readback differenced all
10 unconditionally: unwritten slots resolve to 0, so "Total GPU" reported
the absolute GPU clock (8244→68549 ms over a 60 s run — the "+71538 ms" in
the long-run bench). Fixed: adaptive up/down/L2P passes now carry
timestampWrites (slots 0-1/2-3/4-5, L2P split into its own pass), per-slot
frame flags record which slots were written, and Total GPU is the SUM of
pass spans (immune to stale/zero slots; also fixes negative spans on
uncapped frames without a render pass). A circuit breaker now disables the
far-field family after 3 consecutive validation-dropped submits instead of
freezing the sim on one buffer forever while the FPS counter keeps ticking.

### 12.5 Hash backends: equal fps is by design — now measurable on request

`materialize_ranges` resolves every backend into the same dense
cellStart/cellCount once per frame, so the force loop is byte-identical
across backends (§8.1, §11). The UI tooltip now says so explicitly.
New `?nfprobe=1` opts into the live-hash path (per-neighbor open-addressing
or funnel probes in `cellRangeOf`, main-shader uniform `nfProbeMode`), where
the backends' probe costs differ and show up in FPS/compute time; the axis
appends `*` to the nf token. Counting-sort has no table to probe and stays
dense (its dense CSR is the same idea, materialized).

### 12.6 Reproduction tools added

- `tools/probe_collapse.js` — hooks uploadAdaptiveMetadata/queue.submit,
  counts rebuilds/size-changes/drops, positions checksum + canvas hash per
  second (this is what measured 599/599, the dispersal curves, and the fix).
- `tools/probe_resolve.js` — JS emulation of resolveLeafNode (funnel probe +
  walk-up + descent) vs pure root-descent vs leafForParticle over live
  positions; the 3-way comparison that isolated the staleness root cause.
- `tools/probe_fields.js` — fmmField magnitude/potential statistics, dir
  on/off.
- `tools/smoke_modes.js` — all-mode console-error/telemetry sweep.
- `tools/sig_stats.js` — luminance signature of saved screenshots.
- Note: drawImage() of the WebGPU canvas from an injected sampler can
  return a stale frame in headless runs — the canvas-freeze signal in the
  earlier bench data for adaptive+ahash0 was that artifact plus the
  12.2/12.3 churn under GPU contention, not a real compositor freeze;
  compositor-level screenshots evolve correctly.

### 12.7 Post-fix verification (all measurements on the same RTX-class GPU)

Full long-run bench (`tools/bench_fps_longrun.js 500000 120`, 10 configs,
2026-08-26 — replaces the contaminated 2026-08-25 baseline, which had run
under heavy background GPU load: fixed 1-2 rAF fps with 16 s gaps):

| config | steps/s med | collapse% | gaps>100ms/s | maxGap ms | build/m2l/l2p/main ms | total GPU ms |
|---|---|---|---|---|---|---|
| off (cores only)      | 60 | 0   | 0 | 50* | — | — |
| direct O(N^2)         | 1  | —   | 1.2 | 2467 | 814.9 (the pair pass) | ~817 |
| fixed+counting        | 60 | 0   | 0 | 17 | 5.51/0.07/0.26/2.23 | 8.15 |
| fixed+openaddr        | 60 | 0   | 0 | 17 | 5.25/0.07/0.26/2.37 | 8.09 |
| fixed+funnel          | 60 | 0   | 0 | 17 | 5.42/0.07/0.26/2.37 | 8.27 |
| fixed+openaddr+nfprobe| 60 | 0   | 0 | 17 | 5.02/0.07/0.26/2.30 | 7.79 |
| fixed+funnel+nfprobe  | 60 | 0   | 0 | 17 | 5.12/0.07/0.26/2.28 | 7.93 |
| adaptive (default)    | 60 | 0   | 0 | 17 | 0.62/0.10/1.99/0.04 | 3.05 |
| adaptive+ahash0       | 60 | 0   | 0 | 50* | 0.95/0.28/1.81/0.07 | ~3.1 |
| adaptive+far0 (chain) | 60 | -0.3 | 0 | 50* | 0.57/0.12/1.95/0.06 | 3.00 |

(*compositor jitter, not sim gaps.) Highlights: every mode holds vsync with
zero >100 ms sim gaps; adaptive is now FASTER than fixed at 500k (3.0 vs
8.1 ms GPU/frame — the drift-bounded tree is small); direct vs FMM is
1 vs 60 steps/s — the demo's promised algorithmic improvement is now
measurable in the page itself. nfprobe deltas were within median noise in
the interleaved bench (±0.1 ms on the main pass) but reproduce as ~+1.1 ms
(2.3 → 3.4 ms) in a single-session A/B at 500k/leafBits 8; at this
occupancy most per-neighbor probes hit an empty slot immediately.

5M spot check (45 s, adaptive default): 15-29 steps/s with NO collapse —
tree steady at 36-137 nodes (refinement spikes during the close encounter
at t~20 s), far entries 300-3200, 1 buffer reallocation for the whole
run, zero submit drops, GPU ~50 ms/frame dominated by the 5M-particle L2P.

List-1 near-field budget tiers raised 48/32/16 → 48/48/32 (the old
>2M floor left the adaptive near field ~9x leaner than the fixed grid's
3x3 walk at 5M, biasing propagation).

## 13. Round-17: numeric cross-validation rig; FMM-vs-AFMM mismatch root-caused and fixed

Motivation: the user reported (a) Analytic O(N), FMM, and AFMM (the
adaptive FMM mode) show visibly
different propagation, (b) AFMM trails FMM by "a couple of frames", (c) the
hash-backend selector appeared to do nothing, (d) FMM controls are not
settable in the vortex sim and the settings matrix is hard to interpret.
This round answers all four with measurements, fixes the one genuine bug,
and labels the rest honestly. Backup of the pre-round page:
`index.backup.2026-08-26.html` (local, untracked).

### 13.1 The rig: `validate.html` + `tools/smoke_validator.js`

A minimal self-contained cross-validation page (no UI beyond a table, no
rendering, paced by `onSubmittedWorkDone`). All WGSL kernels, the galaxy
ICs, the funnel-hash JS, the adaptive metadata builder, and the per-mode
dispatch sequences are spliced VERBATIM out of index.html by
`tools/_build_validate.py` (re-run it after touching index.html kernels).
Every mode starts from a byte-identical restored state; cores evolve
through the same CPU `updateGalaxyCores`; adaptive rebuilds its tree every
24 steps from a GPU readback, exactly like the demo.

Metrics per mode vs the Direct O(N^2) reference: one-step Δv rel-L2
(cosine, max/mean abs) and K-step final-position divergence; a `direct#2`
run establishes the GPU noise floor (measured: exactly 0 — the pair pass is
deterministic). `adaptive-fullNF` re-runs adaptive with `?p2pbudget=4096`
(exhaustive List-1) as the attribution control. The `off` row is labeled
INFO, not PASS/FAIL — it is a different physics model by construction
(cores at full GM=0.00075, zero particle-particle gravity; the FMM family
runs half-mass cores + Gp=0.015/N self-gravity). That asymmetry is also
why Analytic-vs-FMM trajectories MUST differ (37.7% Δv rel-L2 at 8k) and
why mode switches reseed the sim.

Run: `node tools/smoke_validator.js [n] [steps] [extraQuery]` (self-serves
the repo on :8124). Thresholds in the page header; `?tolmult=` scales them.

### 13.2 Root cause of the visual FMM-vs-AFMM mismatch (found, fixed)

First honest run (n=8000/120 steps, old default tier): fixed 1.08% Δv
rel-L2 vs direct, adaptive **26.4%**, adaptive-vs-fixed **26.1%**, off
37.7% (expected). The `adaptive-fullNF` control at 0.84% proved the
multipoles, tree, and dispatch chain are CORRECT — the entire mismatch
lives in the adaptive List-1 near-field SAMPLER. The budget sweep at 8k:
48 samples/particle → 26.4%, 256 → 16.7%, 4096 (exhaustive) → 0.84%.
The sampler is unbiased but 1/r^2 pair weights are heavy-tailed: the
closest few neighbors carry most of the true near-field force, and
subsampling them (48 of a few hundred List-1 neighbors) costs ~26% per-step
force error. Structural at 16k (27.1%) and 32k (27.2%); fixed stays ~1.1%.

Fix: the WGSL `l2p` auto tier (mirrored by JS `p2pBudgetAutoForN`, now the
single source in the page) is now **full traversal (4096 cap) at N ≤ 32k,
48 mid-N, 32 above 2M**, and the knob moved into the UI ("Adaptive
Near-Field Budget" select: Auto/128/256/512/1024/Full; `?p2pbudget=`
still overrides). Post-fix validator (8k): adaptive **0.84%**,
adaptive-vs-fixed **0.74%** — the two FMM formulations now agree within
their combined truncation error; 32k: adaptive 1.37% vs fixed 1.13%. Cost
at 500k: l2p 5.09 → 7.95 ms, total GPU 8.35 → 8.41 ms, still vsync-locked
60 fps. At 120k–2M the tier stays 48 (accuracy-matched workloads should
select Full — the select's tooltip carries the measured numbers).
(Reconciliation note: the 500k medians here — l2p 5.09→7.95 ms — differ
from §12.7's same-day adaptive l2p 1.99 ms because the tree phase swings
the node count 46–87 within a run and §12.7's window caught a cheap
phase; §13.2's A/B isolates the tier-raise delta, §12.7's table is the
round-16 absolute endpoint.)

Why not Full everywhere — the 5M wall: the drift-capped tree at 5M runs at
depth 3 (~48 nodes, ~100k particles/leaf), so "exhaustive List-1"
degenerates toward quasi-direct summation: measured 5M adaptive+fullNF =
4 steps/s with L2P 228.7 ms (vs 19 steps/s, L2P 49.9 ms at the default 32
tier). At that tree shape the sampleWeight cap (≤ 8) also mutes the near
field — the honest statement is that the 5M default adaptive near field is
approximate-to-muted, and the fix is architectural (§13.5), not a budget
number. This is documented in the select tooltip and here.

### 13.3 The AFMM-vs-FMM frame-time gap, attributed (5M reference scale)

New "AFMM Meta (CPU)" HUD row + TELEM `adaptiveMeta` block report the
worker rebuild latency, cadence, node count, and depth. 5M, 45 s runs,
2026-08-26:

| config | steps/s | build/m2l/l2p/main ms | total GPU ms | meta |
|---|---|---|---|---|
| fixed+counting | 41 | 13.8/0.2/2.5/6.5 | 24.4 | — |
| adaptive (tier 32) | 19 | 0.5/0.1/49.9/0.8 | 52.8 | 46.6 ms / 96 f, 48 nodes, d3 |
| adaptive+fullNF | 4 | 0.5/0.1/228.7/0.8 | 231.4 | 59.8 ms / 96 f |

Attribution: the gap is (1) the L2P pass (leaf resolution + List-1 + M2P
against a shallow drift-capped tree — 49.9 of 52.8 ms), (2) the 96-frame
CPU rebuild cadence with ~47 ms worker round-trips (17.6 gaps>34 ms/s from
the readback+swap), and (3) none of it is the multipole math. At 500k both
modes are vsync-locked (§12.7). At the parity-validated ≤32k scale the
fixed-vs-adaptive wall cost is equal within pacing noise (3.06 vs 3.09
ms/step in the rig). The adaptive mode is NOT algorithmically slower — it
is implementation-bound to a CPU-side tree refresh, which caps depth
(anti-staleness) which fattens leaves which inflates L2P. Reference
implementations keep the whole loop GPU-resident (see §13.5).

### 13.4 Hash backends: what the selector really controls (measured)

Re-confirmed at 500k/45 s (medians): build pass 4.16/3.86/3.93 ms for
counting-sort/open-addressing/funnel (±0.3 ms, interleaved-noise level);
force-loop main pass 2.22–2.29 ms identical for all backends BY DESIGN
(`materialize_ranges` resolves each table into the same dense arrays once
per frame — that IS the optimization). What differs by construction:
table memory (funnel ≈ 1.05× leaf cells at its 0.95 design load vs
open-addressing's 2×) and worst-case probes (deterministic funnel bound vs
linear probing's 21k+ tail at 0.99 load — see `core/hash_backends_results.json`
and BENCHMARKS.md "Core hash tables"). The funnel advantage is real but it is a
robustness/memory property, not a frame-time property at this occupancy.
New "Live hash probing in P2P loop (A/B)" checkbox (plus `?nfprobe=1|0`)
routes per-neighbor lookups through the live table — the honest probe-cost
A/B (~+1 ms on the main pass at 500k in single-session A/B, within
interleaved median noise). The adaptive occupied-node directory keeps the
funnel hash as its default index (measured +18% steps/s vs the dense
array, §useAdaptiveNodeHash note in the page).

### 13.5 Scale policy, Direct warning, settings map

- **5M = reference scale** (preset relabeled): the tier where the O(N)
  far-field methods are actually loaded. New **10M stress preset**
  (verified on the RTX-class test GPU: O(N) off-mode boots and holds
  60 steps/s, 0 gaps; **FMM measured 2026-08-26, 60 s runs —
  fixed-lattice 15 steps/s median (build/m2l/l2p/main 16.9/0.7/22.7/23.2
  ms, zero diag errors), adaptive 6 steps/s (7→5.8 over the minute, L2P
  155 ms, worker rebuild 122 ms every 96 frames, 41 nodes at depth 3)**;
  see §14. Direct at 10M is guarded by the double warning — extrapolated
  ~minutes per frame from the 500k measurement).
- **Direct O(N²) double warning**: interactive selection ≥ 150k particles
  asks for confirmation; ≥ 1M asks a second time with the trillion-pair
  arithmetic (5M nearly took out the user's machine). Scripted changes
  (probes, `?fmmdirect=1`) are exempt (`e.isTrusted` gate) so headless
  contracts are unchanged; a diag/console warning still fires ≥ 500k.
- **Settings map** (now also in the tooltips): Far-Field Model is the
  scenario-level choice (gravity=galaxy FMM family, centroids=boids,
  none=analytic, biot=reserved). FMM Mode/Order/Near-Field Budget are
  sub-controls of gravity. The Near-Field Hash Mode is shared by ALL
  scenarios (galaxy P2P, boids, vortex viscosity, SPH). Vortex/KH uses
  analytic shear + three moving Biot-Savart cores by design; a
  particle-particle Biot-Savart multipole far field (`biot`) remains a
  reserved stub — the natural Python reference for it is the screened
  Yukawa/Helmholtz Taylor FMM in `core/screened_yukawa2d_fmm.py`.

### 13.6 Roadmap (measured, not speculative)

1. GPU-resident adaptive construction (Morton sort + prefix-sum occupancy
   in the hashed-structure style of Warren & Salmon, 1993; Bonsai-lineage
   GPU tree codes — Bedorf, Gaburov, & Zwart, 2012): removes the CPU
   refresh
   cadence, unlocks deeper trees at 5M (small leaves → cheap EXACT near
   field → both the 26%-class sampler error and the 49.9 ms L2P disappear
   together). This is the single change that fixes adaptive at scale.
2. Hybrid near field (adaptive far field + per-frame fixed-lattice cell
   lists for P2P) — blocked on list-criteria overlap (double counting)
   unless the far-field exclusion is recomputed against the finer cutoff.
3. `biot` far field for the vortex scenario (screened-Poisson/Helmholtz
   Green's function multipoles, per the Python reference above).

Verification of this round: `tools/smoke_validator.js 8000 120` → all rows
PASS (adaptive 0.84% / adaptive-vs-fixed 0.74%); `tools/smoke_modes.js` →
all 8 mode/flag combinations healthy at 120k (direct 20 fps ≈ 50 ms/pair
pass, others vsync-locked); `tools/check_wgsl_sync.py` → PASS;
`node -e "new Function(...)"` on the inline script → syntax OK.

## 14. Round-18: professionalism review pass + first 10M measurements

Three parallel reviews (docs, demo UI copy, repo hygiene/tests) and the
fixes they drove, 2026-08-26. Full finding-by-finding record in
`docs/review_history/ROUND18_FINDINGS.md`.

- **UI blocker fixed**: `updateFmmControlsState()` overwrote the rich
  static tooltips (measured numbers, URL params) with short generic text
  at startup — the §12/§13 honesty tooltips never displayed. Tooltips are
  now stashed (`dataset.longTitle`) and the enabled state always shows
  the full text; only the disabled state swaps in the sub-control note.
- **Validator honesty**: validate.html claimed the demo default tier was
  "48/48/32" while the rig (n ≤ 32k) actually runs the auto tier's FULL
  traversal — and the `adaptive-fullNF` attribution control duplicated
  the adaptive row at that n. The footer now reports the true auto tier
  via `p2pBudgetAutoForN(n)`, and the attribution control runs only when
  the auto tier is a sampled one (skipped with an explanatory note at
  n ≤ 32k; `?p2pbudget=48` measures the sampler's cost there).
- **`?p2pbudget` URL clamp raised 256 → 4096** to match the sidebar
  select (512/1024/4096 silently yielded 256 before).
- **Copy/terminology**: mode names unified (Uniform-Lattice FMM /
  Adaptive FMM / Direct All-Pairs / None; "Fixed-Grid" and "Analytic (no
  FMM)" labels retired; validate.html retitled accordingly), "FCK"
  initialism spelled out, "5M Extreme Mode" telemetry renamed to match
  the "5M Reference" preset, O(N²) notation unified, dialog tone fixed,
  AFMM Meta row given real units, meta descriptions added to both pages.
- **Docs**: README variant table re-pasted from the current
  BENCHMARKS.md run (the old slice contradicted it ~18× on the FMM row);
  README "completely replaces sorting and tree construction" rewritten to
  the honest claim + a new "What 'tree-free' means" section; the eight
  citation-format violations in the README reference list fixed (plus
  Warren & Salmon, 1993, and Dongarra & Sullivan, 2000, added as
  references); dangling `benchmarks/` paths → `core/` (BENCHMARKS.md,
  §11); §2/§5.4 stale numbers refreshed; INAPPLICABILITY Class D numbers
  updated post-FFT-rewrite; conversational asides removed from the README
  front page; bibtex title aligned with CITATION.cff.
- **Tests/hygiene at review time**: `pytest tests/core` 108 passed /
  3 skipped (WGSL numeric tests need node/wgpu), `check_wgsl_sync.py`
  PASS, all README links resolve.

### 14.1 First 10M stress measurements (60 s runs, same RTX-class GPU)

| config | steps/s med | first5→last5 | gaps>100 ms/s | build/m2l/l2p/main ms | notes |
|---|---|---|---|---|---|
| fixed+counting | 15 | 15.2→15.0 | 0.12 | 16.9/0.7/22.7/23.2 | zero diag errors |
| adaptive (tier 32) | 6 | 7→5.8 | 5.73 | 0.8/0.1/155.3/3.1 | worker rebuild 122 ms / 96 f, 41 nodes, depth 3, zero diag errors |

Extrapolation corrected: §13.5 had projected ~20 steps/s for fixed at
10M from the 5M row; measured 15. The adaptive 5M→10M behavior matches
the §13.3 attribution exactly (depth-3 tree, ~244k particles/leaf, L2P
dominated). Artifacts: `tools/bench_r18_10m_{fixed,adaptive}.jsonl`.

Verification of this round: `tools/_build_validate.py` re-splice →
byte-clean; `python tools/check_wgsl_sync.py` → PASS; `node --check` on
both pages' inline scripts → OK; `node tools/smoke_validator.js 8000 120`
→ all rows PASS (adaptive 0.84%, adaptive-vs-fixed 0.74% — unchanged
from §13.2, i.e. the copy edits did not touch the physics).

---

## 15. Round-19: standard self-gravity scenarios (virialized log-disk + cold collapse) with exact-Hamiltonian dE/E

Motivation (owner question, 2026-08-27): every gravitational N-body paper
validates against standard scenarios — an equilibrium system and a cold
collapse — with energy conservation dE/E and Lagrangian radii as the
metrics. The demo previously had neither: its ICs prescribe each galaxy's
potential analytically (the O(N) shortcut), so no closed system existed to
conserve. This round adds the two standards **in the engine's own force
law**, selected from the new "Galaxy Initial Conditions" control or
`?ic=flyby|plummer|collapse`, with an exact-Hamiltonian energy diagnostic
and Lagrangian radii in the sidebar and in validate.html.

### 15.1 The force law, stated honestly

The demo's particle-particle gravity is **2D logarithmic and attractive**:

    F_ij = -Gp (r_i - r_j) / (r_ij^2 + eps^2),   eps^2 = 4e-5 (P2P_EPS2)

from the confining pair potential U(r) = +(Gp/2) ln(r^2 + eps^2) (force =
-grad U). This is the kernel the complex-Taylor FMM operators, the fixed-grid
M2L/L2P chain, the direct all-pairs baseline, and the near-field P2P all
implement — so the standard scenarios validate exactly what the engine
computes. Log gravity in 2D obeys a Gauss law, F(R)·2πR = 2πG·M_enc(R),
so any axisymmetric disk has F(R) = G·M_enc(R)/R in closed form; in
particular the interior of a uniform disk is exactly harmonic
(F = G·π·Σ0·R).

The stellar-dynamics classics (a 3D Plummer sphere in Hénon units, e.g.
Heggie & Hut, 2003) cannot be transplanted literally into a 2D engine; the
scenarios below are their 2D-log analogues, with every closed form derived
for this kernel rather than asserted by analogy.

### 15.2 Scenario 1: virialized log-disk (Q ≈ 1)

Positions: Σ(R) = M/(πa²) (1+R²/a²)^-2, a = 0.1, M = μ = 2e-3 (G = 1,
per-particle mass μ/N so the total G·M is N-independent). By the Gauss law
M_enc(R) = M·R²/(R²+a²), which inverts to the exact sampler
R = a·sqrt(u/(1-u)); the circular speed of its own field is
v_c(R) = sqrt(G·M)·R/sqrt(R²+a²) (flat rotation curve outside ~a, i.e.
Mestel-like; harmonic core inside).

Velocities: rotation + dispersion, v_φ = v_c·sqrt(1-s²), σ_R = s·v_c with
s = 0.3 — the standard practice for N-body disk ICs (centrifugal balance
minus the dispersion's contribution). **Virial by construction** (2K =
Σᵢ Rᵢ·|Fᵢ| for centrifugal support), **not an exact distribution-function
solution**.

Honest negative result worth recording: the first attempt used an isotropic
constant-dispersion equilibrium claimed from the 2D Jeans equation,
σ² = G·M/(4a). That derivation dropped the geometric (Σσ²)/R term; with
it, the exact isotropic solution of this profile is
σ²(R) = (GM/a)(1+u²)²[π/16 − J(u)]/u with J(u) = u/(8(1+u²)) −
u/(4(1+u²)²) + atan(u)/8 — which **diverges as 1/u at the center**. The
buggy constant-σ ICs were measured expanding (r50 ×4.2, dE/E −62% with a
sign-flipped energy formula) before the fix; the warm rotating disk holds
(r50 ×0.98). Both runs are in the round-19 smoke logs.
`tests/physics_simulation/test_standard_ics.py` now gates the sampler
(enclosed-mass quantiles), the virial identity 2K = Σ R·F (direct O(n²)
forces), and the energy-convention consistency independently in numpy.

### 15.3 Scenario 2: cold collapse (Q = 0)

Uniform disk, R0 = 0.3, zero velocities. The unsoftened interior force is
exactly harmonic (ω² = GπΣ0 = G·M/R0²), so every interior particle reaches
the center at the analytic free-fall time

    t_ff = π / (2ω) = π·R0 / (2·sqrt(G·M)) = 10.537 time units
          (= 439 steps at dt = 0.024)

independent of radius. Softening (ε = 0.0063 ≪ R0) regularizes the bounce;
the numpy oracle test measures minimum-r50 at 0.7–1.5 × t_ff.

### 15.4 Implementation

- `SimParams` grows two uniform words (buffer 112 → 128 B): `coreGM`
  (per analytic core; 0 for the standard ICs — cores pinned at the center,
  `updateGalaxyCores` early-returns) and `gpPerParticle` (flyby: 0.015/N
  with the calibrated half-core split; standard: μ/N). Both writers (demo
  `runWebGPUFrame`, rig `writeSimUniform`) compute them identically; the
  shader's hardcoded select() is gone.
- New WGSL entry point `energy_phi` (bindings 13–15, used only by it, so
  the main 13-binding pipeline layout is untouched): per particle
  φᵢ = Σ_{j≠i} ln(r_ij²+ε²), |vᵢ|², and rᵢ from the center.
  E = Σ|v|²/2 + (Gp/4)·Σφᵢ is the EXACT softened Hamiltonian of the closed
  system. O(N²): demo telemetry caps at N ≤ 200,000 and runs every 96
  frames ("Self-Gravity Energy" + "Lagrangian Radii" rows, r20/r50/r80 with
  the t0 ratio); the rig runs it at t0 and t_final per mode.
- The mouse attractor is disabled in the standard scenarios (external
  momentum would corrupt the conservation readout); the UI copy says so.
- validate.html: `?ic=` runs the whole existing cross-validation on the new
  ICs AND adds standard-scenario rows — per-mode dE/E (absolute and,
  crucially, relative to the direct run's own dE/E), r50/r80 ratios, the
  r80 window [0.75, 1.40] for the log-disk, and the r50 ≤ 0.75 collapse
  signature once steps·dt > 0.6·t_ff. The "off" row becomes a free-streaming
  null control (on the cold ICs: a frozen system, measured dE/E exactly
  +0.000%).

### 15.5 Measured results (n=8000, dt=0.024, same RTX-class GPU as §13/§14)

ic=plummer, 240 steps (5.76 t.u. ≈ 1.8 crossing times):

| mode | dv rel_l2 vs direct | dE/E | dE/E − direct | r50(t)/r50(0) | r80(t)/r80(0) |
|---|---|---|---|---|---|
| direct   | 0        | +3.439% | 0       | ×0.977 | 1.043 |
| fixed    | 0.010753 | +3.436% | −0.003% | ×0.978 | 1.044 |
| adaptive | 0.001097 | +3.423% | −0.016% | ×0.973 | 1.043 |
| off      | (INFO)   | +42.6%  | —       | ×2.000 | free streaming |

ic=collapse, 480 steps (11.52 t.u. > t_ff = 10.54; through maximum
compression and bounce):

| mode | dv rel_l2 vs direct | dE/E | dE/E − direct | r50(t)/r50(0) |
|---|---|---|---|---|
| direct   | 0        | +0.158% | 0      | ×0.643 |
| fixed    | 0.010872 | +0.155% | −0.003% | ×0.628 |
| adaptive | 0.008837 | −0.990% | −1.15% | ×0.633 |

Findings the metrics surfaced (all new, none threshold-loosened away):

1. **The integrator floor is symplectic Euler, not leapfrog** (v += a·dt;
   x += v·dt in `fn main`): its shadow-Hamiltonian offset at dt=0.024 is
   +3.4% on the log-disk ICs, mode-independent (direct = fixed = adaptive
   to 0.02%). The rig therefore checks the FMM modes' dE/E **relative to
   the direct run's** — the force-approximation-induced drift — which
   measures 0.003% (fixed) / 0.016% (adaptive) on the log-disk.
2. **Through cold collapse, adaptive drifts −1.15% vs direct** (threshold
   2%): the adaptive far field softens only the monopole term, and at
   maximum compression (leaf size ~ ε) the unsoftened higher-order
   multipoles inject this apparent drift. Forces still agree with direct
   at dv rel_l2 0.9%, and the fixed grid — whose minimum cell (1/64 at
   n=8000) stays ~2.5 ε — shows only −0.003%. Recorded as a measured
   characteristic, not a failure.
3. **Long-horizon demo drift at 120k (fixed grid)**: ~11.8k steps (283
   t.u.) on the log-disk gives dE/E = −6.8% while r50 stays ×0.98 —
   leafBits(120k) = 7 makes the finest cell 1/128 ≈ 1.2 ε, so the same
   soften-the-monopole-only mismatch acts continuously in the disk core
   (~−0.02%/t.u.). The collapse IC at 120k (post-bounce halo) measures
   +0.08% over the same horizon. The validator's short horizons are
   insensitive to this; the honest fix (softened higher-order M2L
   operators) is future work.
4. On the rotating disk the adaptive FMM's per-step force error vs direct
   is 0.11% — its quadtree concentrates resolution exactly where the disk
   does, vs 1.1% for the uniform lattice at this n.

Demo spot-checks (probe_stdics.js, 120k, uncapped): log-disk 732 fps with
the diagnostic kernel running every 96 frames; collapse 528 fps through
the bounce; zero console errors; `check_wgsl_sync.py` PASS (energy_phi is
demo-only, INFO row in its report).

### 15.6 How to run

    # flyby regression (numbers must match §13.2/§14 exactly)
    node tools/smoke_validator.js 8000 120
    # virIALIZED log-disk: equilibrium maintenance + dE/E
    node tools/smoke_validator.js 8000 240 "ic=plummer"
    # cold collapse through t_ff: dE/E + r50 collapse signature
    node tools/smoke_validator.js 8000 480 "ic=collapse"
    # IC math oracle (sampler, virial identity, energy convention, t_ff)
    python -m pytest tests/physics_simulation/test_standard_ics.py -v
    # interactive demo: index.html?ic=plummer (or ?ic=collapse)

Verification of this round: flyby smoke PASS at the §13.2-identical numbers
(fixed 0.01075 / adaptive 0.00844 / adaptive-vs-fixed 0.00740); plummer and
collapse smokes PASS (tables above); `tools/_build_validate.py` re-splice
byte-clean after every template edit; `check_wgsl_sync.py` PASS; `node
--check` on both pages' inline scripts OK; IC-math tests 4/4 PASS.
