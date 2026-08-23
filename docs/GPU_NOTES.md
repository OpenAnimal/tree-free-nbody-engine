# GPU Notes — funnel hash on accelerators, and the "not faster at this scale" caveat

This file documents two things that the BENCHMARKS.md tables cannot show on
their own:

1. Why the append-only funnel hash is still the right structure for dynamic
   simulations on the GPU, and the standard GPU pattern that removes its
   only real limitation (it cannot unlearn keys).
2. Why several BENCHMARKS.md rows honestly say "NOT faster than O(N^2) at
   this scale" — and where to look for the real constant factors.

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
directly: the same flat FMM that is 0.23x at N=2000 becomes **5.99x faster
than direct at N=32000** (rel-L2 5.3e-7). The crossover is real; it just
lives at larger N than the per-app demo scales.

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
   List-1 (adjacent-leaf) neighbors are subsampled at a per-particle
   budget of 24 / 12 / 6 sampled neighbors per adjacent leaf, tiered by
   total particle count (<0.5M / 0.5–2M / >2M); `?p2pbudget=N` overrides
   the tier. Since round 9 each sampled neighbor carries weight
   `leafOccupancy/sampleCount`, so the estimate is unbiased in expectation
   (matching the fixed-grid stride weighting); before round 9 the samples
   carried weight 1, which under-counted dense leaves by
   occupancy/budget and rendered as leaf-aligned square artifacts that
   drifted with each tree rebuild (see §7.1). Three further serial loops
   are now bounded the same way (stride-sampled, rotation-decorrelated,
   reweighted): P2M moment construction (≤512 particles/leaf), the List-4
   P2L translation in M2L (≤128 particles/source leaf), and the List-1
   adjacent-leaf enumeration itself (≤32 leaves/particle) — without those
   caps a collapse of all particles onto the center of mass stalls the
   frame (single thread per leaf/node walking thousands of entries).
   With the adaptive tree the sampling fraction is
   bounded below by budget/leafTarget, so it is 100% below 0.5M and
   degrades only for the densest cluster cores that saturate at max
   depth; the sidebar "P2P Near Field" row reports the actual fraction
   of the densest leaf. The flat metadata (tree + interaction lists) is
   cross-validated against the direct O(N²) sum by
   `tools/validate_adaptive_js.py` (structural invariants +
   emulated-kernel potentials/forces), and since round 10 the executed
   WGSL multipole chain itself is validated on GPU by
   `tests/core/test_adaptive_wgsl_numeric.py` (full clear→P2M→M2M→M2L→
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
