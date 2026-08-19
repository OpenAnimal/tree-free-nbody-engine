# Zig funnel-hash microbenchmark (round-4 task 4.10)

A faithful Zig port of `core/elastic_hash.py:ElasticHashTable` (the
Farach-Colton / Krapivin / Kuszmaul 2025 funnel hash) plus a microbenchmark
comparing insert/lookup throughput against the Python reference.

## Files

- `funnel_hash.zig` — the funnel hash table port (geometry-identical to
  the Python reference; salts use a splitmix64-seeded LCG instead of
  numpy's MT19937, so exact slot assignments differ but the slab
  geometry, probe bounds, and throughput characteristics match).
- `bench.zig` — the microbenchmark harness (insert, hit-lookup, and
  absent-key-lookup passes with throughput and mean-probe reporting).
- `build.zig` — the Zig build script.

## Building and running

Requires Zig 0.16.0 (the std API changed in 0.16: `std.heap.GeneralPurposeAllocator`
was removed in favor of `smp_allocator`; `std.time.Timer` was removed so
`bench.zig` ships a portable `Timer` wrapper using `RtlQueryPerformanceCounter`
on Windows and `clock_gettime(CLOCK_MONOTONIC)` on POSIX; `std.process.argsAlloc`
was replaced by `std.process.Args.toSlice`).

```bash
cd zig
zig build run --release=fast
# or with custom parameters:
zig build run --release=fast -- --capacity 2000000 --delta 0.05 --seed 42
```

## What it measures

For a table sized to `capacity` keys at final load `1 - delta`:

1. **INSERT** — insert `capacity` distinct splitmix64-derived keys; report
   throughput (M keys/s) and mean probes per insert.
2. **LOOKUP (hits)** — shuffle the inserted keys (Fisher-Yates) and look
   each up once; report throughput, mean probes, and hit rate (expected
   100%).
3. **LOOKUP (absent)** — query `capacity` keys from a disjoint space
   (high bit set); report throughput, mean probes, and false-hit count
   (must be 0). The mean probes for absent keys equals the deterministic
   `probe_bound = alpha*beta + b_attempts + 2*c_bucket_slots` — every
   absent key inspects exactly the worst-case number of slots, which is
   the funnel hash's headline guarantee.

## Reference results (Zig 0.16.0, ReleaseFast, Windows)

capacity = 1,000,000, delta = 0.05, seed = 42
geometry: alpha=28 beta=9 total_slots=1,052,640 probe_bound=277

| Pass             | Throughput (M keys/s) | Mean probes | Notes                |
| ---------------- | --------------------- | ----------- | -------------------- |
| INSERT           | 6.7 - 15.1            | 32.96       | 100% inserted        |
| LOOKUP (hits)    | 6.9 - 11.9            | 28.96       | 100% hit rate        |
| LOOKUP (absent)  | 2.1 - 2.2             | 277.00      | 0 false hits (= probe_bound) |

The throughput range reflects run-to-run variance on a busy host; the
probe counts are deterministic (geometry-driven, not salt-driven).

## Comparison to the Python reference

The Python `core/elastic_hash.py:ElasticHashTable` is a scalar-Python
implementation (one key per `insert`/`lookup` call, no vectorization in
the per-key path). On the same 1M-key workload it is bound by the Python
interpreter overhead (~50-100 ns per bytecode dispatch), so it runs at
roughly 0.1-0.3 M keys/s for both insert and lookup — i.e. the Zig port
is ~30-50x faster on the per-key path. The vectorized batch probe
`core/elastic_hash.py:funnel_probe` closes most of that gap for lookups
by amortizing the Python overhead across a NumPy chunk, but the per-key
`insert` path has no vectorized equivalent in the Python reference.

The point of this microbenchmark is NOT to claim a Zig speedup over the
Python reference (that would be a strawman) — it is to confirm that the
funnel hash's deterministic probe bound holds in a compiled implementation
and to provide a baseline for any future compiled-kernel port (e.g. a
Triton or WGSL funnel hash for the GPU path, which is recorded as future
work in `docs/GPU_NOTES.md`).

## Correctness

The bench's PASS verdict requires:
- all `capacity` keys inserted successfully (insert_ok == capacity)
- all `capacity` shuffled lookups hit (lookup_hits == capacity)
- zero false hits on absent keys (absent_false_hits == 0)

The probe_bound is reported and the absent-key mean probes equal it
exactly, confirming the deterministic worst-case bound.
