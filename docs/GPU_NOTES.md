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
