# Round-10 Deep Verification Instructions

**Status:** instructions only — do not implement findings in this pass
**Scope:** full repository correctness review
**Purpose:** find hidden numerical, algorithmic, edge-case, and validation-harness bugs that the existing smoke and acceptance tests can miss.

This is a future implementation brief. A later pass should execute the review, add focused regression tests, and fix verified defects. This document deliberately does not claim that the items below are bugs; they are hypotheses and review targets.

---

## 1. Non-negotiable review protocol

1. Start from the current working tree and record `git status`, commit, Python version, installed optional dependencies, GPU/runtime availability, and the exact pytest collection count.
2. Do not trust an existing test merely because it passes. For every numerical claim, build an independent oracle that does not share the implementation's indexing, recurrence, approximation, or helper functions.
3. Prefer small deterministic cases where every pair, cell, edge, and coefficient can be enumerated. Use fixed RNG seeds and save the seed, parameters, expected tolerance, and environment in the test/probe.
4. For approximation algorithms, compare against a direct O(N²) or analytic reference across structured and random inputs, not only one random cloud. Separate discretization error, truncation error, and implementation error.
5. Exercise both public APIs and low-level helpers. Verify output values, shapes, dtypes, ordering, mutation/aliasing behavior, metadata, counters, and error handling.
6. Test invariants independently: conservation, symmetry, reciprocity, translation/rotation behavior where promised, monotonicity, partition/exactly-once coverage, reproducibility, and finite outputs.
7. Include degenerate and boundary inputs: N=0, N=1, duplicate points, coincident targets/sources, empty cells, one occupied cell, exact cell boundaries, negative/out-of-domain coordinates, minimum/maximum depth, odd dimensions, zero/negative parameters, extreme magnitudes, and float32/float64.
8. A test must fail when the targeted implementation defect is reintroduced. Mutation-test each new gate by temporarily perturbing a sign, index, coefficient, boundary comparison, or fast path. Do not accept tests that only check shape, no-NaN, or that a function returns without raising.
9. Keep implementation changes separate from review. First write a failing reproducer/regression test and register the finding; only then fix it in a later implementation pass.
10. Never run JAX checks without disabling VRAM preallocation:
    ```powershell
    $env:XLA_PYTHON_CLIENT_PREALLOCATE = "false"
    # or set XLA_PYTHON_CLIENT_MEM_FRACTION = "0.10"
    ```
11. Do not treat a benchmark, self-test, or `__main__` printout as a correctness gate unless it has an independent expected value and a nonzero failure exit status.
12. Record skipped hardware-dependent checks explicitly (reason, required hardware/dependency, and what CPU/reference coverage substitutes for it).

### Required finding record

For every verified defect, record:

- ID, severity, and affected file/API;
- minimal reproducer and deterministic seed;
- expected result and independent oracle;
- observed result and why existing tests missed it;
- regression-test filename and acceptance tolerance;
- proposed fix (but do not apply it during the review-only pass);
- whether the defect affects numerical correctness, safety, performance, or claims/documentation.

---

## 2. Baseline and gates

Run from the repository root:

```powershell
$env:XLA_PYTHON_CLIENT_PREALLOCATE = "false"
python -m pytest --collect-only -q
python -m pytest tests/ -q --tb=short
python tools/lint_claims.py
python tools/check_wgsl_sync.py
python tools/validate_adaptive_js.py
```

Also inspect the CI workflow and `tools/run_all.py`; compare their file/package coverage with `tests/` and document any omitted gate. Run the relevant package test subset after each cluster review, then run the full suite at the end. The current baseline is 175 collected, with the last full run at 173 passed / 2 skipped; re-establish this rather than assuming it remains true.

Keep new probes and temporary outputs outside production packages (for example `tools/review_round10/` or a clearly marked temporary directory). Do not recreate root-level `test_pass*.py` scratch files. Promote only durable, independent regression tests into `tests/<package>/`.

---

## 3. Review waves and concrete targets

### Wave A — `core/`, browser FMM, native backends

Review:

- `core/adaptive_fmm.py`, `core/fast_vectorized_fmm.py`, `core/radial_taylor.py`, `core/yukawa3d_fmm.py`, `core/gaussian2d_fgt.py`, `core/screened_yukawa2d_fmm.py`;
- `core/spatial_index.py`, `core/elastic_hash.py`, `core/adaptive_gpu_metadata.py`, device/runtime code, CUDA/HIP/OpenCL/WebGPU kernels;
- `index.html` adaptive and uniform paths plus `tools/validate_adaptive_js.py`, `tools/emit_adaptive_meta.mjs`, and `tools/check_wgsl_sync.py`;
- native C/Zig and backend parity paths.

Build independent checks for:

- P2M/M2M/M2L/L2L/L2P coefficient signs, normalization, charge weighting, translation direction, and order-0/order-1 exact cases;
- exactly-once near/far coverage, list reciprocity, offset relocation, empty/coarse/deep cells, duplicate points, depth saturation, and particle-to-leaf resolution;
- CellIndex key encode/decode, world-mode bounds, ring neighborhoods, negative coordinates, exact boundaries, Morton overflow, and no missed/duplicated neighbors;
- float32/float64 and CPU/GPU/native parity, including non-unit charges and non-square/3D shapes;
- adaptive metadata buffer lengths, resize/rebuild stale data, bind-group consistency, and typed-array count/offset overflow;
- shipped-source validation soundness: prove the harness actually executes the shipped slice and fails if a kernel/source marker or result is deliberately corrupted;
- WGSL/CUDA/JS semantic drift and all fallback/toggle paths, including the documented uniform-only standalone kernel contract.

Do not claim browser performance correctness from a headless emulation; retain separate numerical and interactive gates.

### Wave B — `algorithm_theory/`

Review every module, grouped by mathematical family rather than filename:

- **Spatial/hash/graph:** `multipole_range_tree.py`, `spatial_graph_partitioning.py`, `spatial_disjoint_set_fmm.py`, `sublinear_distance_oracle.py`, `tree_free_geodesic_fmm.py`, `elastic_quotient_filter.py`.
- **Linear algebra/inference:** `localized_ensemble_kalman_fmm.py`, `matrix_free_gaussian_process.py`, `spectral_meshfree_laplacian.py`, `spectral_biclustering_fmm.py`, `functional_sobol_anova.py`, `kernel_causal_discovery.py`.
- **Transforms/kernels:** `non_uniform_fourier_hash.py`, `fractional_laplace_contour.py`, `fractional_volterra_memory.py`, `continuous_meshfree_wavelet.py`, `oscillatory_butterfly_kernel.py`, `screened_yukawa_fmm.py`.
- **Optimization/dynamics:** `optimal_transport_fmm.py`, `co_optimal_transport.py`, `opinion_dynamics_fmm.py`, `personalized_pagerank_fmm.py`, `sublinear_fast_dtw.py`, `sublinear_edit_distance.py`, `phase_space_attractor_fmm.py`, `spatial_voting_equilibrium.py`.
- **Tensor/multipole/physics:** `algebraic_multipole_tensor.py`, `quantum_fock_exchange_fmm.py`, `capacitance_boundary_bem.py`, `elastic_quotient_filter.py`, and remaining modules in the folder.

For each module, verify the actual algorithm against a simple reference (dense NumPy/SciPy, Dijkstra, dynamic programming, explicit kernel sum, direct matrix solve, finite differences, or an analytic solution). Specifically target transpose/orientation errors, off-by-one bands, empty reductions, normalization, cutoff tails, convergence criteria, damping, sign conventions, boundary inclusion, hash collisions, capacity/full-table behavior, and claimed complexity. Check that a documented “FMM” is not silently a tree-code or bucket approximation and that tests validate values rather than only execution.

Prioritize the known residual/open areas: the quotient-filter-to-funnel port, CUDA unit-charge behavior, PyTorch Taylor-FGT path, 5M browser ladder, and any Round-7 entries still marked OPEN/PARTIAL.

### Wave C — `neural_ops/`

Review `multipole_attention.py`, `flash_multipole_kernel.py`, `elastic_kv_cache.py`, `continuous_meshfree_gnn.py`, `equivariant_field_layer.py`, `taylor_fgt_attention.py`, diffusion/GP/cache modules, and all backend variants.

Required checks:

- exact dense attention parity for small N, including masking, normalization, duplicate/recent KV entries, bucket collisions, empty buckets, and cache rollover;
- approximation error versus N, grid depth, multipole order, feature scale, and cluster distribution;
- translation/rotation/equivariance claims with non-axis-aligned rotations and anisotropic/uneven data;
- gradients and adjoints against finite differences or an independent autodiff expression; verify no detached or stale buffers;
- dtype/device parity, batch-size one, non-multiple tile sizes, zero-length inputs, and invalid dimensions;
- confirm benchmark claims measure the intended path and that optional Torch/JAX paths are not silently skipped.

### Wave D — `bioinformatics/` and `environmental_modeling/`

For bioinformatics modules, compare sequence/k-mer, contact-map, binding, DDG, solvation, titration, chromatin, viral, and cellular calculations to toy analytic/known-property references. Check symmetry under sequence/entity permutation, conservation of totals, charge/pH limits, empty sequences, duplicate records, malformed inputs, units, and numerical stability. Separate scientific plausibility from algorithmic correctness.

For environmental modules, test groundwater diffusion/advection against analytic solutions and mass balance; airborne exposure against linearity, symmetry, and well-mixed limits; electrolyte screening against Debye asymptotics and neutral/charged cases; and radiotherapy against direct Gaussian convolution, superposition, and convergence. Check grid resolution, boundaries, source/target ordering, zero sources, and return-value test functions. Every pytest-collected `test_*` must use assertions rather than relying on a returned bool/float.

### Wave E — physics and quantized optimization

For `physics_simulation/`, verify broadphase completeness against all-pairs contacts, barrier energy/gradient/Hessian finite differences, line-search acceptance and last-chance behavior, collision symmetry, zero-contact scenes, coincident/near-coincident geometry, and solver convergence independent of the solver's own residual calculation.

For `quantized_bitpacked_optimization/`, compare encode/decode and Morton/bitboard operations against scalar references for every small bit width and boundary value. Test signedness, overflow, padding, run boundaries, duplicates, empty inputs, stable ordering, loss/lossless claims, and every optimization flag in isolation and combination. Benchmark only after byte-for-byte and numerical parity is established.

### Wave F — video, graphics, game mechanics, apps, tools, native/Zig

For video and graphics, use tiny hand-computable frames/scenes to test motion vectors, SAD/hash candidate selection, scene-cut frame placement, GOP/IDR behavior, rate-control bounds, codec round trips, color/channel ordering, alpha, degenerate geometry, and deterministic output. Stress uniform frames and hard cuts because they expose quadratic scans and deferred-state bugs.

For `game_mechanics_spatial/` and `apps/`, compare boids, proximity, attention, hydro, galaxy, and GNN paths to dense references for small scenes; test isolated entities, empty neighborhoods, negative/out-of-range coordinates, multiple entities per cell, and parameter extremes. Check that UI/demo labels and metrics describe the actual approximation.

For `native/`, `zig/`, and `tools/`, compile and compare native outputs with the Python scalar reference across small exhaustive inputs, malformed inputs, zero lengths, alignment/padding, endian behavior, and error returns. Audit validators for vacuous success: corrupt each input/source and confirm the gate fails.

---

## 4. Cross-cutting adversarial tests

Run these against every applicable API:

- empty input and singleton input;
- all points identical;
- two points exactly on a cell boundary and just below/above it;
- duplicate keys and maximum probe/capacity load;
- zero, negative, NaN, and infinity parameters where validation is promised;
- very small and very large scales, including underflow/overflow-sensitive kernels;
- non-contiguous NumPy arrays, read-only arrays, float32 versus float64;
- odd sizes and non-multiples of vector/tile/block dimensions;
- repeated calls after rebuild, resize, reset, or mutation;
- deterministic rerun with identical seed and changed seed;
- permutation of sources/targets and translation/rotation where the API promises invariance;
- forced fallback path versus optimized path;
- direct reference versus approximation as order/depth/resolution changes.

Also run a static claim audit: every complexity, “exact”, “lossless”, “equivariant”, “FMM”, “O(N)”, hardware, and accuracy claim must point to an executable gate that can falsify it.

---

## 5. Deliverables for the later implementation pass

1. `ROUND10_FINDINGS.md` with the finding register and explicit accepted residual risks.
2. One focused regression test per verified defect under `tests/<package>/`; no giant one-off pass script.
3. Fixes applied only after the reproducer is red, followed by the affected subset and full-suite run.
4. Mutation results for each new gate (or a documented reason mutation was infeasible).
5. Updated package docs/claims where the implementation is intentionally approximate.
6. Updated `tools/run_all.py` / CI coverage if a package or test path was omitted.
7. Final gate table containing collection, test result, optional-dependency skips, numerical cross-validation, validator-corruption checks, and browser/native measurements separately.
8. A final “not verified” section for interactive GPU, large-N, optional Torch, proprietary encoder, and other environment-limited checks.

Completion means every investigated claim has an independent falsifiable gate — not merely that the current suite is green.
