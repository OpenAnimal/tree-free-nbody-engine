# algorithm_theory/ — Round-7 status

Audited: 2026-08-18 (round-6) and 2026-08-20 (round-7). All modules import and
pass `test_basic_datatypes_fmm.py` and the folder benchmark runs end-to-end.

## Hash-table lineage (Round-7)

- `elastic_quotient_filter.py` is a **legacy pre-funnel scheme** — it is NOT
  the FKK funnel schedule implemented in `core/elastic_hash.py`. The
  geometric-levels + per-level linear-probing body predates the funnel port
  and is kept here as the AMQ / Jaccard-similarity reference baseline.
  Round-7 task T-A4 chose the docstring-banner path (porting the
  quotient/remainder + fingerprint semantics to `ElasticIntTable` is awkward
  because the funnel table stores `(key, int value)` pairs, not
  `(fingerprint, counter)` pairs). The real port — task T-A4b
  (`FunnelQuotientFilter`, implemented and covered in
  `tests/algorithm_theory/test_basic_datatypes_fmm.py`) — is DONE.
- New code should prefer `core.elastic_hash.ElasticHashTable` /
  `ElasticIntTable`, which expose a deterministic `probe_bound` and the
  A_1..A_alpha / B / C funnel geometry from Farach-Colton, Krapivin, & Kuszmaul (2025).

## Terminology caveat (read before citing)

Many files in this folder carry `fmm`/`multipole` in their names. After audit,
**most of them are order-0/order-1 cluster (Barnes–Hut-style tree-code)
approximations, not Greengard & Rokhlin translation-based FMMs** — there is no
M2M/M2L/L2L operator hierarchy in them. Where a module genuinely implements
operator-based FMM machinery, it says so. The two core engines with full
cross-validated FMM math live in `core/` (adaptive FMM, flat vectorized);
the radial-Taylor unification (round-6) added the dimension-parameterized
driver in `core/radial_taylor.py` that the Yukawa3D / Gaussian2D / screened-
Yukawa2D engines instantiate.

## Verified with numeric cross-validation

- `screened_yukawa_fmm.py` — real near/far tree-code with an order-1 dipole
  correction and screening truncation; ~1% rel L2 vs its own `direct_evaluate`
  (annotated in-file: tree-code, not translation-based FMM). Cross-validated
  against the round-6 `core/screened_yukawa2d_fmm.py` Taylor engine (6.2e-9)
  during the radial-Taylor unification — the two agree on the screened kernel
  to the tree-code's own tolerance, and the Taylor engine is the higher-
  accuracy reference.
- `packed_vectorized_fmm.py` (in `quantized_bitpacked_optimization/`) — genuine
  flat 2D log-kernel FMM (P2M/M2L/L2P + near-field P2P); loss/lossless behavior
  of each optimization is measured in its ablation benchmark.

## Naming caveats by pattern

- Files named `*_fmm.py` that aggregate per-cell centroids/mean quantities
  (e.g. pagerank, opinion dynamics, spectral biclustering, ensemble Kalman
  variants) implement **hash-bucketed cluster approximations**, not FMM.
  This is often a sensible design — the elastic-hash spatial index is the
  load-bearing part — but the name should not be read as a multipole claim.

## Exclusion candidates (future)

Modules whose "FMM" label adds no algorithmic content beyond a nearest-cell
hash lookup could be renamed or folded together at the next breaking revision.
None of them make numerical correctness claims beyond what their tests verify.

## Test gate

`python -X utf8 algorithm_theory/test_basic_datatypes_fmm.py` stays green.

## Round-8 honesty pass (2026-08-20)

A round-8 audit tightened correctness and retracted several overclaims that
the round-6/7 audits had left in place. Summary of the substantive changes
(each verified by re-running the affected module's `__main__` demo or its
`test_basic_datatypes_fmm.py` cases):

- `sublinear_edit_distance.py` — banded DP `j=0` column now initialised
  (pure-prefix-deletion paths were silently wrong); `max_k=0` truthiness
  fixed; sampling-recall trade-off documented; test slack tightened.
- `tree_free_geodesic_fmm.py` — local Bellman-Ford now iterates to
  convergence (`while new_local` with a node-count guard) instead of being
  capped at 8 sweeps (which stranded nodes); dead `ElasticHashTable` import
  removed; demo now asserts zero max-abs error vs Dijkstra on a path graph.
- `non_uniform_fourier_hash.py` — 3D gather branch for
  `type2_uniform_to_nonuniform` implemented (was missing); even-grid
  validation added; dead `self.eps` removed; Type-3 claim retracted
  (Type 1 and Type 2 only).
- `benchmark_screened_yukawa2d_variants.py` — K0 dipole term sign corrected
  (`+` -> `-`); measured accuracy improved from ~2.4e-2 to ~1.1e-3 and the
  in-file accuracy note updated to match.
- `elastic_quotient_filter.py` — `FunnelQuotientFilter.insert` now raises
  `RuntimeError` on table-full instead of silently dropping; FP-rate
  docstring corrected to `~n_stored / 2^64` (full 64-bit hash collision,
  not `2^(-r)`); deterministic FNV-1a-64 hash for `str`/`bytes` added for
  cross-process reproducibility; new tests cover both.
- `multipole_range_tree.py` — `morton_encode_nd` now raises on
  `bits_per_dim * D > 64` (was silent uint64 overflow); `max_depth`
  clamped to `64 // D`; range-query complexity and
  `compute_multipole_box_potential` (brute-force, not FMM) docstrings
  corrected; `query_range` unpacking fixed to include `val_sum` / `cnt`.
- `fractional_volterra_memory.py` — prefix sums introduced for `O(1)`
  per-block far-field aggregation (far field was `Theta(T^2)`, contradicting
  the `O(T log T)` claim; now genuinely `O(T log T)`); dead `order` param
  removed; T-doubling timing sanity added to the demo.
- `quantum_fock_exchange_fmm.py` — `np.vectorize(math.erf)` replaced with
  `scipy.special.erf`; dead `pair_indices` removed; dead `order` param
  removed; docstrings corrected (monopole-only far field, no exchange K
  matrix, cost is `O(N_pairs * K_cells)` not `O(N_basis)`).
- `matrix_free_gaussian_process.py` — cutoff-truncation tail corrected
  (`exp(-3.5^2/2) ~ 2.2e-3`, NOT `~1e-7`); variance path now scatters all
  `k_star` rows in a single pass over the test blocks (was an
  `O(n_test * n_blocks)` `np.where` scan); parity verified.
- `personalized_pagerank_fmm.py` — "exact O(|E|)" claim retracted
  (PCG residual-tolerance approximate, <=60 iters).
- `localized_ensemble_kalman_fmm.py` — "strictly O(N*M^2)" claim
  retracted (real cost is `O(N*(k_act^2*M + k_act^3))`); Gaspari-Cohn
  truncation noted as the first piece only (discontinuity at `r_loc`);
  hard-coded 0.85 posterior-spread shrink documented.
- `algebraic_multipole_tensor.py` — rebranded as a SYNTHETIC low-rank
  contraction demo (the dense baseline reconstructs the same
  `U * diag(s(r)) * U^T` the low-rank path uses, so the reported
  "Rel Error" is round-off, not a true M2L approximation error); dead
  `eps` param removed.
- `sublinear_distance_oracle.py` — "(1+eps) stretch" and
  "O(N log(1/eps)) space" claims retracted (first-point-per-bucket
  election has no stretch bound; level-0 SSSP tables are O(N^2)-class);
  dead `eps` param removed; dead `best_dist` removed.
- `phase_space_attractor_fmm.py` — Grassberger-Procaccia and motif
  discovery claims retracted (not implemented; only recurrence density
  and anomaly scoring).
- `spatial_graph_partitioning.py` — Polsby-Popper compactness documented
  as a dimensionally-heuristic proxy (variance-ellipse pseudo-area over
  cut-edge pseudo-perimeter, not a true isoperimetric quotient); Elastic
  Spatial Hashing claim retracted (perimeter counted via adjacency scan).
- `sublinear_fast_dtw.py` — "provable accuracy" claim retracted
  (Salvador-Chan FastDTW has no optimality guarantee; result is an upper
  bound); complexity corrected to `O(T * radius * log T)`.
- `spatial_point_cloud_compression.py` — "lossless integer" claim
  retracted (the implementation is LOSSY: coordinates are quantised to
  `precision_bits` and attributes are stored as float16).
- `spectral_meshfree_laplacian.py` — matvec vectorised via `np.add.at`
  (parity verified); "Galerkin coarse operator / strict SPD guarantee"
  claim retracted (coarse solve is diagonal-only).
- `spectral_biclustering_fmm.py` — `O(k * nnz(A))` claim retracted
  (A is dense here; per-iteration cost is `O(R*C)`).
- `__init__.py` — module list updated to flag each of the above
  deviations inline.

## Round-7 domain-expansion tasks (X-A7/A9/A10/A12, 2026-08-21)

- `capacitance_boundary_bem.py` — the GMRES matvec was an O(K^2) Python
  cell-pair double loop with a monopole-only far field (the X-A2 reword
  flagged this). Replaced with a `CellIndex`-backed near/far split: near
  field = direct block over `neighborhood_indices(key, ring=1)` (analytical
  disk self-potential on the i==j diagonal), far field = per-cell monopole
  + dipole moments about the unweighted cell centroid, evaluated as one
  vectorized `(n_target_in_cell, n_far_cells)` matrix op per target cell.
  This is a first-order Barnes-Hut-style tree code, NOT a translation-based
  FMM (no M2M/M2L/L2L hierarchy) — the module docstring and matvec
  docstring now say so. The dense O(N^2) direct matvec
  (`evaluate_boundary_potential_dense`) and the legacy O(K^2) cell-pair
  loop (`_evaluate_boundary_potential_cellpair`) are both retained for
  validation. Acceptance (self-test in `__main__` / `_self_test`):
  matvec rel-L2 vs dense reference 8.5e-4 (<=1e-3) on the smooth GMRES-solved
  sigma of a 2-sphere (R=2, eps_0=1 scaled units, kernel 1/(4*pi*r),
  analytic C=4*pi*R=25.1327); near/far-solved vs dense-solved capacitance
  rel err 8.0e-4 (<=1e-3); the residual vs the analytic continuum value
  (~3.4e-3 near/far, ~2.6e-3 dense) is BEM discretization error (the dense
  solve itself sits at that floor), not tree-code error. Wall-clock matvec
  scaling with N fixed and K varied 135->908 cells (x6.7): time x9.8 vs the
  quadratic bound x45.2 — sub-quadratic (the legacy O(K^2) Python loop
  would hit the x45 bound). The flat single-level scheme remains O(N*K)
  in flops for a surface BEM (the far field sees all K cells); the win is
  the O(K) Python iteration count plus vectorized inner work, not a
  Greengard & Rokhlin O(N) operator hierarchy.
- `screened_yukawa_fmm.py` (X-A9, option b) — `compute_screened_potential_field`
  now accepts `use_taylor_fmm=True` to delegate the full near+far evaluation
  to the verified core 3D Yukawa Taylor FMM (`core/yukawa3d_fmm.py:Yukawa3DFMM`,
  order-p M2L far field + exact ring-2 near field). Positions are affine-
  normalized into the unit cube with kappa rescaled by the domain span (the
  Yukawa kernel is not scale-invariant) and the result scaled back by 1/span.
  T-C8 was verified landed first (core/radial_taylor.py docstring: cells-per-
  side = depth, LINEAR; toy_2cell_check non-degenerate at 1.8e-13;
  core/test_yukawa3d_fmm.py all PASS at p=8 rel-L2 2.7e-8). Accuracy jump:
  the historical order-1 tree-code sat at ~1% rel-L2 vs direct; the Taylor
  FMM delegation reaches 3.6e-8 rel-L2 vs direct at p=8 on a 2k-particle
  cloud (X-A9 acceptance: <=1e-6). The tree-code path is retained as the
  honest order-0 comparison row and for backward compatibility (default
  `use_taylor_fmm=False`).

- `matrix_free_gaussian_process.py` (X-A10) — batch variance prediction:
  all test points are now solved in ONE multi-RHS PCG (block-Jacobi =
  scalar inverse-diagonal broadcast) against A = K + sigma_n^2 I, with
  chunked batches of 256 columns and per-column early-freeze, replacing
  the O(N_test) Python loop of separate 25-iteration PCG solves.
  Verification (2026-08-21, independent of the in-module acceptance):
  at matched TIGHT convergence (both paths tol 1e-12 / 300 iters) the
  batched variance matches the old per-point loop to rel-L2 5.6e-09 —
  the original <=1e-8 rel spec holds; the batching is numerically exact.
  At production settings (25 iters / tol 1e-4, identical to the
  pre-X-A10 code) both paths sit at the shared PCG-tolerance floor
  (max abs diff ~3e-3 on variance values of O(1e-3)) — this is a
  tolerance property, not a batching regression, and is what the
  in-module 5e-3 abs assertion measures. Spec-scale timing
  (N_train=4k, N_test=1k, production settings): old per-point loop
  91.2 s -> batched predict 12.4 s = 7.4x.
- X-A12 (unify spatial hashing on CellIndex, world mode, ring=1):
  migrated this round in `matrix_free_gaussian_process.py` (`10db41e`),
  `continuous_meshfree_wavelet.py` (`ea083f2`),
  `phase_space_attractor_fmm.py` (`0c43a6b`), `optimal_transport_fmm.py`
  (`1053189`), `opinion_dynamics_fmm.py` (`bf0bbc7`),
  `spatial_disjoint_set_fmm.py` (`0091843`, two methods, visited-cell
  guard preserved via Morton keys), plus unsolicited-but-verified
  `neural_ops/multipole_gaussian_process.py` (`d2e19cd`).
  `tree_free_geodesic_fmm.py` needed only removal of a dead
  `ElasticHashTable` import (done in the working tree; no hash rewrite —
  its geodesic hashing is not a spatial cell hash). Independently
  verified 2026-08-21: one module per commit, old-vs-new module mains
  byte-identical modulo timing lines (wavelet scalogram, attractor
  densities, OT u/v/cost/iters, opinion drift rel-L2 2.5e-16), and a
  direct old-vs-new partition A/B for SpatialDisjointSetFMM (3 radii:
  identical forests, weights, components, component counts).
  Gates: tools/run_all.py 24 PASS / 1 SKIP (wgpu absent) / 1 FAIL
  (tools/check_wgsl_sync.py — in-flight demo/file-kernel divergence
  from the parallel WebGPU workstream, not X-A12); lint_claims clean.
