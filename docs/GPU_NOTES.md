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

The Farach-Colton / Krapivin / Kuszmaul elastic hash is an **append-only**
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
mode, p=2 quadrupole CGR88, P2P radius 0.035, full x9 cell lists.

| Preset | FMM Build | FMM M2L | FMM L2P | Main Compute | Render | Total GPU | FPS  | Buffers | Cell occupancy |
| ------ | --------- | ------- | ------- | ------------ | ------ | --------- | ---- | ------- | -------------- |
| 1M     | 7.570 ms  | 0.269 ms | 5.674 ms | 90.563 ms | 1.687 ms | 105.783 ms | 10 | 192 MiB | avg 61.0/cell (128x128) |
| 5M     | 2.997 ms  | 0.270 ms | 4.221 ms | 59.284 ms | 1.772 ms | 68.565 ms  | 14 | 617 MiB | avg 76.3/cell (256x256) |

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
