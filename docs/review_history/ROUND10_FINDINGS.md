# Round-10 Deep Verification Findings

**Status:** Waves A–C complete; A1/A2/B1/B2/B3 all fixed and regression-tested
(2026-08-22). Waves D–E executed (see below); Wave F pending.
**Date:** 2026-08-22
**Method:** independent probes cross-validating against brute-force/analytic references

> **Note on probe scripts:** the one-off `tools/review_round10/probe_*.py`
> scripts that produced these findings have been removed; their checks were
> distilled into the durable regression tests listed in the "File" column of
> each table below (under `tests/<package>/`). The probe counts and results
> recorded here reflect what was observed during the review run.

---

## Wave A: `core/` — FMM engines, spatial index, elastic hash

### Probes run

| Probe | File | Checks | Result |
|---|---|---|---|
| CellIndex | `tests/core/test_spatial_index.py` | 60+ | ALL PASS |
| ElasticHashTable | `tests/core/test_elastic_hash.py` | 25+ | ALL PASS |
| Adaptive FMM operators | `tests/core/test_adaptive_fmm_reference.py` | 18 | ALL PASS |
| Radial Taylor engines | `tests/core/test_yukawa3d_fmm.py`, `test_screened_yukawa2d_fmm.py`, `test_gaussian2d_fgt.py` | 18 | ALL PASS |

### Findings

**R10-A1 [MINOR, performance] `_overflow_count` not decremented on `remove()`** — **FIXED**
- **File:** `core/elastic_hash.py` (`ElasticHashTable.remove`, `ElasticIntTable.remove`)
- **Fix:** `remove()` now decrements `_overflow_count` when the removed slot is
  in the overflow region. Safe: tombstones keep `occupied=True`, so the
  monotone-fill argument the insert early-stop relies on survives deletions.
- **Test:** `tests/core/test_elastic_hash.py::test_elastic_overflow_count_decremented_on_remove`
  (+ int-table variant). Natural overflow placement is a tail event random
  keys never trigger, so the test places the overflow entry white-box.

**R10-A2 [MINOR, documentation] Negative key sentinel collision** — **FIXED (docs)**
- **File:** `core/elastic_hash.py` class docstring
- **Fix:** docstring now states keys must be NON-NEGATIVE (-1/-2 are the
  reserved sentinels); all real callers pass Morton keys / bucket ids >= 0.

**R10-A3 [INFO] All adaptive FMM operators verified correct**
- P2M, M2M, M2L, L2L, L2P, P2L, M2P all match direct computation to within truncation-error tolerance.
- L2P force and M2P field verified against finite-difference gradients.
- Full pipeline (N=500, p=8) matches direct O(N²) to rel-L2 < 1e-4 for potential, < 1e-3 for forces.
- Convergence verified: error decreases monotonically with order p (p=2: 3.6e-3 → p=10: 2.7e-7).
- Edge cases pass: N=0, N=1, N=2 same cell, all-same-point (no inf/nan), symmetry.

**R10-A4 [INFO] All radial Taylor FMM engines verified correct**
- Yukawa3D: rel-L2 = 4.9e-8 at p=8, converges with order, N=1/N=2/symmetry/kappa=5.0/evaluate_targets all pass.
- Gaussian2D: rel-L2 < 1e-4 at p=8, N=1 passes.
- ScreenedYukawa2D: rel-L2 < 1e-3 at p=8, N=2 passes.
- All-same-point: no inf/nan (graceful degenerate handling).

**R10-A5 [INFO] CellIndex verified correct**
- Exhaustive key round-trip (2D 8×8, 3D 8×8×8).
- neighborhood_indices exactly-once coverage, no duplicates, no misses.
- far_keys is exact complement of neighbor_keys.
- World mode negative coordinates, boundary positions, grid_res overflow rejection.
- N=0, N=1, all-same-point, rebuild clears stale data, ring=0.

**R10-A6 [INFO] ElasticHashTable verified correct**
- 1000 keys at 95% load: all insert, all lookup correct, no false hits on absent keys.
- funnel_probe vectorized parity with scalar `_search` (550 queries, 0 mismatches).
- probe_bound respected by all lookups (present and absent keys).
- Duplicate key update, remove + re-insert, items() correctness, capacity boundary.
- ElasticIntTable insert_or_increment, ElasticBatchingHashTable basic.

---

## Wave B: `algorithm_theory/` — Koopman, LEnKF, GP, NUFFT, OT, Laplacian, Fock, etc.

### Probes run

| Probe | File | Checks | Result |
|---|---|---|---|
| algorithm_theory | `tests/algorithm_theory/test_round10_fixes.py` | 25+ | 2 FAIL (real bugs) |

### Findings

**R10-B1 [MAJOR, correctness] `SpectralMeshfreeLaplacian` crashes on 2D points** — **FIXED**
- Both operators in `algorithm_theory/spectral_meshfree_laplacian.py`
  (`MeshfreeGraphLaplacian`, `ConsistentMeshfreeLaplacian`) now accept
  (N, 2) and (N, 3) point sets with dimension-generic bucket keys; other
  shapes raise a clear `ValueError`.

**R10-B2 [MAJOR, correctness] `ContinuousFockExchangeFMM` far-field expansion is wrong** — **FIXED**
- **File:** `algorithm_theory/quantum_fock_exchange_fmm.py`
- **Fix:** the far field now uses the erf-screened monopole kernel
  `erf(omega_eff * R)/R` with `omega_eff = sqrt(g_t * g_bar / (g_t + g_bar))`
  (g_bar = far-cell charge-weighted mean gamma), matching the near-field /
  direct kernel family. Measured rel-L2 vs direct at `cell_size=0.3`
  dropped from ~0.7 to <0.1.
- **Test:** `tests/algorithm_theory/test_round10_fixes.py::test_fock_exchange_far_field_screened_kernel`.

**R10-B3 [MAJOR, correctness] `SpectralMeshfreeLaplacian` does not compute the continuous Laplacian** — **FIXED (redesign)**
- The old operator was a Wendland-weighted GRAPH Laplacian presented as a
  continuous-Laplacian discretization. It is kept (working PCG solver,
  demo unchanged in spirit) as `MeshfreeGraphLaplacian` — alias
  `SpectralMeshfreeLaplacian` — with honest graph-semantics docs, and
  `solve_meshfree_poisson` is documented as solving the GRAPH system.
- The continuous operator is now `ConsistentMeshfreeLaplacian`: RBF-FD
  with a Gaussian kernel + quadratic polynomial augmentation
  (Wright–Fornberg style). It reproduces nabla^2 EXACTLY on quadratics
  (measured 2.8e-12 on a 12^3 grid incl. one-sided boundary stencils) and
  is second-order on smooth fields (sin probe rel-L2 0.145 at h~0.55,
  vs ~1.0 pre-fix).
- Notable implementation findings on the way: (a) pure-polynomial
  min-norm moment matching is ill-conditioned on scattered clouds
  (weights O(1000)); (b) polyharmonic r3 saddle systems are EXACTLY
  singular on cospherical lattice stencils (edge points blew up to 1e18);
  (c) the Gaussian kernel keeps the saddle invertible for any distinct
  stencil, and the polynomial block alone guarantees quadratic exactness;
  (d) stencil selection must rank-check the polynomial basis — on
  lattices the nearest neighbors can all sit on {-1,0,1} offsets where
  x^2 = x identically (rank 9 of 10), so axis-2 offsets are grown in
  until the basis is spanned; (e) iterative/direct SOLVES of the
  non-symmetric consistent operator on bounded clouds are unstable
  (spurious near-null boundary modes) — the class is scoped to operator
  evaluation and says so.
- **Tests:** `tests/algorithm_theory/test_round10_fixes.py` (7 tests).

**R10-B4 [INFO] Koopman, LEnKF, GP, NUFFT, OT, SSSP, BEM, edit distance all verified correct**
- Koopman: recovers eigenvalues of linear dynamics (0.9, 0.8) from trajectory data; 1-step prediction matches.
- LEnKF: analysis mean matches direct Kalman, reduces ensemble spread, no NaN.
- GP: posterior mean matches dense Cholesky solve (rel-L2 < 0.1); variance is a sparse approximation (upper bound, by design).
- NUFFT: Type-1 and Type-2 match direct DFT (rel-L2 < 2e-4) with correct numpy FFT frequency ordering.
- OT: Sinkhorn plan satisfies marginal constraints (with appropriate gamma and cutoff).
- SSSP: FrontierClusteredSSSP matches Dijkstra baseline exactly.
- BEM: capacitance of sphere matches analytic 4πR to within 5%.
- Edit distance: sublinear approximation within bounds of exact Wagner-Fischer.

---

## Wave C: `neural_ops/` — attention, GNN, equivariant layers, KV cache

### Probes run

| Probe | File | Checks | Result |
|---|---|---|---|
| neural_ops | `tests/neural_ops/test_neural_ops_advanced.py` | 25+ | ALL PASS |

### Findings

**R10-C1 [INFO] All neural_ops modules verified functional**
- TreeFreeMultipoleAttention: output shape correct, no NaN, approximates direct attention.
- ElasticMultipoleKVCache: append/query works, no NaN.
- ContinuousMeshfreeGNNLayer: forward pass produces correct shape, no NaN/inf.
- EquivariantMultipoleLayer: rotation invariance confirmed for scalar features (rel-L2 < 0.5).
- FlashMultipoleAttentionEngine: output shape correct, no NaN.
- TaylorFGTAttention: output shape correct, no NaN.
- NeuralSPHIPCLayer: forward pass produces no NaN/inf.
- MultipoleAdjointEngine: numerical gradient check passes.
- HierarchicalElasticKVCache: query returns valid results, no NaN.
- SphericalMultipoleAttention: output shape correct, no NaN.

---

## Wave D: `bioinformatics/` + `environmental_modeling/`

### Probes run

| Probe | File | Checks | Result |
|---|---|---|---|
| Bio core (elastic hash + FMM) | `tests/bioinformatics/test_round10_wave_d.py` | 23 | ALL PASS after fixes |
| Bio modules (kmer, contact map, MD, pockets) | `tests/bioinformatics/test_round10_wave_d.py` | 25 | ALL PASS after fixes |
| Environmental (4 modules) | `tests/environmental_modeling/` | 18 | ALL PASS |
| Module `__main__` demo sweep | `tests/bioinformatics/test_round10_wave_d.py` | 29 modules | ALL PASS |
| Integrated benchmark | `tests/bioinformatics/test_round10_wave_d.py` | 1 | PASS (see below) |

Baseline before Wave D: `pytest tests/bioinformatics tests/environmental_modeling` = 37 passed.
After Wave D: 37 + 8 new bio tests + 4 new env tests = 49 passed.

### Findings

**R10-D1 [MAJOR, numerical] `ScreenedKernelType.YUKAWA` silently dropped the entire near field and all far-field forces** — **FIXED**
- **File:** `bioinformatics/core/fast_multipole_kernel.py` (`TreeFreeBioFMM.evaluate`)
- **What was wrong:** the near-field if/elif chain handled only `DEBYE_HUCKEL`,
  `COULOMB`, and `GENERALIZED_BORN`; a `YUKAWA` engine returned ~0.1% of the
  true potential (norm ratio 0.001 on the probe: far field only) and
  identically-zero forces. Additionally, the far-field `else` branch
  (GB/YUKAWA) computed potentials but never forces, so GB far-field forces
  were silently zero (same class as Round-7 finding P18-1, which had fixed
  only the GB *near* field).
- **Oracle:** independent O(N²) double-loop Debye-Hückel/Yukawa potential and
  force references; per-atom central finite differences of the exact
  potential (F_i = q_i · −∇V_i, seed 20260822). Before the fix: rel-L2 = 1.0
  for potential, forces all zero. After: rel-L2 < 5e-3 (near-field config) and
  far-field net forces nonzero with correct repulsive direction.
- **Fix:** the DH near-field branch now also covers `YUKAWA` (identical
  exp(−κr)/r kernel); the far `else` branch gained the monopole force term
  consistent with the DH-like kernel it evaluates.
- **Tests:** `tests/bioinformatics/test_round10_wave_d.py::test_yukawa_kernel_near_field_not_dropped`,
  `::test_gb_and_yukawa_far_field_forces_present`. Both were red before the
  fix (mutation evidence by construction).
- **Why existing tests missed it:** no test ever instantiated `kernel_type=YUKAWA`
  (only the enum was exported).

**R10-D2 [MINOR→MAJOR, algorithmic] Binding-pocket ray stencil was mirror-biased at the default 14 directions** — **FIXED**
- **File:** `bioinformatics/binding_pocket_detector.py` (`__init__`)
- **What was wrong:** the 26-direction cube stencil was enumerated with `dx`
  as the outer loop, so `dirs[:ray_directions]` with the default 14 kept 9
  rays pointing toward −x and none toward +x (stencil sum = [−6.14, −2.41, 0]).
  The concavity/burial score and the probe-point placement were therefore not
  invariant under reflection, biasing pocket detection toward one side of
  every protein.
- **Oracle:** stencil isotropy (sum of directions must vanish; every direction
  must have its antipode) and end-to-end mirror symmetry: reflecting a
  400-atom protein (x → −x) must reproduce mirrored pocket centers and equal
  druggability scores.
- **Fix:** stencil ordered faces (6) → corners (8) → edges (12); the default
  14 is now exactly the symmetric 6-face + 8-corner set.
- **Test:** `tests/bioinformatics/test_round10_wave_d.py::test_binding_pocket_ray_stencil_symmetric` (red before fix: sum ≠ 0, scores diverged).

**R10-D3 [MINOR, correctness-of-contract] `probe_radius` parameter stored but never used** — **FIXED**
- **File:** `bioinformatics/binding_pocket_detector.py` (`detect_pockets`)
- **What was wrong:** the documented water probe radius (1.4 Å) was accepted
  and stored but the clash test compared probe points against bare vdW radii
  (`d < vdw`), so the solvent-accessibility semantics advertised in the
  docstring were not implemented.
- **Oracle:** same structure probed with `probe_radius=0.01` vs `1.4` must
  reject strictly more candidate points with the larger probe (standard
  LIGSITE-style water-probe clearance `d ≥ vdw + probe_radius`).
- **Fix:** clash threshold is now `d < vdw + probe_radius`.
- **Test:** `::test_binding_pocket_probe_radius_used` (red before fix: 95 pocket points at both radii).

**R10-D4 [MAJOR, physics/claims] MD engine advertised LJ steric repulsion but computed none** — **FIXED**
- **File:** `bioinformatics/non_periodic_md_engine.py` (`compute_forces`)
- **What was wrong:** the documented force model
  "F_total = F_harmonic_bonds + F_LJ_sterics + F_FMM_electrostatics" declared
  σ=3.4 Å / ε=0.15 kcal/mol and then computed nothing: `e_lj` was always 0.0
  (and not even present in the returned dict), and there was no steric force
  at all — atoms could collapse onto each other; the pre-fix "stable ~300 K"
  trajectories on synthetic structures were an artifact of the missing term.
- **Oracle:** direct all-pairs 12-6 LJ sum (independent double loop) on a
  40-atom random cluster; analytic two-atom force; frictionless energy
  conservation; boundedness on a clashed structure (min pair distance 1.23 Å).
- **Fix:** truncated (unshifted) 12-6 LJ with 2.5σ cutoff via the
  Morton-elastic-hash cell lists (27-cell gather, same pattern as
  `contact_map_graph.py`), plus an energy-consistent repulsive force cap
  (`lj_fmax` = 200 kcal/mol/Å below the cap radius; V(r) = V(r_c) + fmax·(r_c−r)
  on the capped branch) so F = −∇V holds exactly and unrelaxed synthetic
  structures do not blow up the fixed-step integrator (uncapped r^−12 reached
  T ~ 1e14 K; capped it releases the stored clash energy at ≤ ~3.5e4 K with
  <0.3% energy drift). `e_lj` now reported in the energy dict.
- **Tests:** `::test_md_lj_forces_present_and_correct`,
  `::test_md_lj_many_body_vs_direct_all_pairs` (includes the cap in the
  reference via its own bisection), `::test_md_lj_capped_force_matches_gradient`,
  `::test_md_stable_with_lj` (clean chain stays 100–600 K with <5% drift;
  clashed protein stays finite/bounded). All red before the fix.

**R10-D5 [INFO] Verified-correct highlights (bioinformatics)**
- Morton encode/decode: exact match vs an independent scalar bit-interleave
  reference (500 random 21-bit triples) and roundtrip on boundary values
  {0, 1, 2^20, 2^21−1}.
- `ElasticSpatialHash3D` façade: 200-key insert/lookup, duplicate-update
  semantics, `lookup_with_probes` ≤ `probe_bound`, `build_from_coords`
  permutation-invariant key set and inverse (duplicates included).
- `TreeFreeBioFMM` DEBYE_HUCKEL/COULOMB potentials vs direct sums: rel-L2
  1.2e-5…9.8e-5 (monopole+dipole far field, exact near field); forces vs
  analytic direct reference 2.3e-5; near-field forces = q·(−∇V) to 2.3e-10 by
  FD; permutation symmetry 5.3e-17; N=0/N=1 edge cases correct (no
  self-interaction). GB near-field potential matches the Still/OBC pairwise
  closed form to 1.5e-16 and GB near-field force = q·(−∇V) to 2.2e-10.
  `TaylorYukawaBioFMM` unit mapping pinned by `toy_2cell_check_bio`
  (rel-L2 5.6e-11).
- `KmerElasticHashTable`: exact agreement with a string-based canonical-kmer
  reference counter for k=1/5/21 (counts, window totals, unique counts) on a
  3.5 kb sequence with an N-gap; canonical key idempotent; decode roundtrip;
  empty/short/all-N/lowercase inputs; spectrum binning conserves counts ≤
  max_depth.
- `ContactMapGraphBuilder`: CA contact edges, degrees, and distances match an
  O(N²) reference exactly (after mapping through the CA-subset index space —
  see R10-D6); duplicate-coordinate input produces no self-edges.
- `MacromolecularMDEngine` units: 1 kcal/mol = 418.4 Da·Å²/ps² conversion and
  σ_v = sqrt(k_B·T·418.4/m) verified; Maxwell-Boltzmann initial temperature
  within 1.2σ of the χ²(3N) distribution (deterministic RandomState(42) —
  identical velocities for identical systems by design); Langevin
  c1/c2 coefficients standard.

**R10-D6 [INFO, contract] `build_ca_contact_graph` returns edges in CA-subset index space**
- The `edges` / `edge_distances` / `degrees` keys index the CA-filtered
  subset (0..N_CA−1), not original atom indices. Internally consistent
  (distances, degrees, hubs), and the only in-repo consumer
  (`benchmark_bioinformatics.py`) uses counts/hubs, not raw edges. No fix
  applied (would change public API without an in-repo victim); external
  callers must map through the CA mask.

**R10-D7 [INFO] Verified-correct highlights (environmental_modeling)**
- `groundwater_plume`: screened-Helmholtz factorization re-derived by hand
  (correct for (−D∇² + v·∇ + λ)); superposition linearity across separate
  calls 2.0e-16; `flow_direction` normalized internally; empty sources →
  zeros. Module-embedded analytic anchors (pure diffusion and advected
  closed form, both asserting) re-confirmed via the suite run.
- `airborne_exposure_room_eigen`: matches an independently written 3D
  spectral reimplementation to 5.9e-16; the same spectral formula was
  validated against an independent 1D tridiagonal Neumann FD solve
  (rel 1.3e-4); superposition-linear; full-problem mirror-symmetric to
  1e-12; well-mixed mean matches Q/(V·λ) to <5%. A direct 3D FD comparison
  is not a usable oracle at affordable grids: the point-source 1/r
  singularity leaves O(10%) FD error with no monotone refinement trend
  (24³→48³ tested) — documented so nobody mistakes that gap for a defect.
- `airborne_exposure_room_images`: reciprocity C(a→b) = C(b→a) to 1e-12
  (module also asserts wall symmetry).
- `electrolyte_screening`: κ = 0.329·√I matches the textbook Debye length
  3.04 Å·√(1/I); K_E = 14.3996 eV·Å/e²; FMM vs direct Yukawa on a
  net-neutral configuration rel 2.3e-8; empty ions → zeros.
- `radiotherapy_dose`: `SuperpositionDoseEngine` vs fresh direct
  double-Gaussian sums rel 4.3e-7; `ray_trace_lazy` geometry and
  W/N·exp(−μt) weights exact, batching preserves order/count; module
  erf-anchor and 5k-point convergence tests re-confirmed via the suite.

**R10-D8 [INFO, harness] `PytestReturnNotNoneWarning` on two environmental tests** — **NOT-A-BUG**
- `tests/environmental_modeling/test_environmental_suite.py` re-exports
  module test functions that both `assert` and `return True`; pytest warns
  about the returned bool but the assertions are real gates (they can and do
  fail on regression). Left as-is to avoid touching the module-embedded
  test contract.

---

## Wave E: `physics_simulation/` + `quantized_bitpacked_optimization/`

### Probes run

| Probe | File | Checks | Result |
|---|---|---|---|
| Physics (IPC solver, broadphase, barriers, degenerate meshes) | `tests/physics_simulation/test_round10_wave_e.py` | 46 | ALL PASS after fixes |
| Quantized (Morton ops, pack/unpack, bitboards, FMM vs direct) | `tests/quantized_bitpacked_optimization/test_round10_wave_e.py` | 44 | ALL PASS after fixes |

Baseline before Wave E: `pytest tests/physics_simulation tests/quantized_bitpacked_optimization` = 10 passed.
After Wave E: 10 + 10 physics + 8 quantized = 28 passed.

### Findings

**R10-E1 [MINOR, robustness] `ClothMesh` crashed with `IndexError` on face-less / empty meshes** — **FIXED**
- **File:** `physics_simulation/ppf_contact_solver_fmm/matrix_free_ipc.py`
  (`ClothMesh._build_topology`, `find_broadphase_candidates`,
  `_find_broadphase_candidates_reference`)
- **Defect:** with no triangles (or 0 vertices), `edges = []` produced a 1-D
  `(0,)` `struct_edges` array and `struct_edges[:, 0]` raised
  `IndexError: too many indices for array`. Separately, both broadphase
  implementations called `positions.min(axis=0)` before checking emptiness,
  raising `ValueError: zero-size array to reduction operation minimum`.
- **Oracle:** construct `ClothMesh(3 verts, (0,3) triangles)` and
  `ClothMesh((0,3), (0,3))`; energies must be zero and finite;
  `find_broadphase_candidates((0,3))` must return shape (0,2). All three
  crashed before the fix.
- **Fix:** `np.array(edges, dtype=np.int32).reshape(-1, 2)` (no-op for
  non-empty meshes) + early `return np.empty((0, 2))` guards for empty
  positions in both broadphase implementations. `solve_step` now runs
  end-to-end on both degenerate meshes (verified: free-fall displacement
  dt²·g exact on the face-less mesh).
- **Tests:** `tests/physics_simulation/test_round10_wave_e.py::
  test_cloth_mesh_triangle_free_constructs_and_evaluates`,
  `::test_cloth_mesh_empty_constructs`, `::test_broadphase_empty_positions`
  (all RED before the fix — mutation evidence by construction).

**R10-E2 [MINOR, documentation/contract] `k_shear` stored but never consumed** — **FIXED (docs)**
- **File:** `physics_simulation/ppf_contact_solver_fmm/matrix_free_ipc.py`
- **Defect:** the `k_shear` constructor parameter (default 600, and
  `create_cloth_grid` passes `k_stretch * 0.35`) is stored and validated by
  `combine_cloth_meshes` but appears in no energy, force, or Hessian term.
  `_build_topology`'s docstring claimed it builds "shear cross-diagonals",
  which it does not (all triangle edges, including grid diagonals, are plain
  structural edges under `k_stretch`). Same class as R10-D3.
- **Fix:** honest docs on the class and `_build_topology`: the material model
  is stretch + hinge bending only; shear resistance comes from the stretch of
  the diagonal structural edges; `k_shear` is API-compat-only. Implementing a
  separate shear energy would double-count the diagonals, so docs (not code)
  is the correct fix.

**R10-E3 [MINOR, correctness] `pack_particles_64bit_3d` silently wrapped coordinates at depth > 8** — **FIXED**
- **File:** `quantized_bitpacked_optimization/packed_particle_types.py`
- **Defect:** the 64-bit layout has 8-bit per-axis integer fields, but the
  packer only masked (`& 0xFF`) instead of rejecting `depth > 8`:
  `pack(0.9, depth=9)` unpacked to 0.4 (silent 0.5 corruption). The 32-bit
  2D path already had the analogous guard in `VoxelPackedTreeFreeFMM`
  (raises for depth > 6); the 64-bit standalone path had none.
- **Oracle:** boundary-value roundtrip probe (depths 4/6/8 all pass with
  error <= 1.5/(256·grid_res)); depth=9 demonstrably corrupted positions.
- **Fix:** `ValueError` for `depth > 8` in `pack_particles_64bit_3d`
  (mirrors the engine's 32-bit guard message style).
- **Test:** `tests/quantized_bitpacked_optimization/test_round10_wave_e.py::
  test_pack64_depth_gt_8_raises` (RED before the fix).

**R10-E4 [INFO] Broadphase verified complete and reference-exact on adversarial scenes**
- Superset of all brute-force contact pairs (d < dhat, topo-excluded) AND
  exact set-parity with the CellIndex reference on: a 13³ lattice with every
  coordinate an exact multiple of dhat/2 (cell boundaries); boundary-
  straddling chains (distances just below dhat across cell edges); 25
  coincident + 2 nearby points (all C(25,2) coincident pairs emitted exactly
  once); 40 identical points; single point; two points exactly dhat apart
  (candidate but barrier-inactive, E = 0); Chebyshev-2 pairs with occupied
  midpoint (emitted) and without (not emitted); triplicated cloth vertices
  (dense same-cell duplicates). Deterministic on repeat calls.
- The 13-canonical-half-offset + 49-distance-2-with-midpoint vectorized
  scheme exactly reproduces the reference ring-1 closure in every case.

**R10-E5 [INFO] Shell elasticity verified against finite differences and invariances**
- Elastic forces = −∇E by central FD on folded random states
  (rel 1.2e-10) and smooth bend states (1.5e-9).
- Stretch+bend Hessian-vector products match FD at stretched states
  (3.2e-9); bending Hessian (exact quadratic) to 4.7e-9. (For compressed
  springs the PSD projection intentionally drops the negative tangential
  geometric stiffness — tested only where the projection is exact.)
- Zero ghost forces: on an IRREGULAR flat hinge mesh, energy and forces
  vanish under 20 random rotations+translations (worst 1.0e-9 ≈ roundoff at
  coords ~10, weights ~40). Elastic energy rigid-transform invariant to 1e-9.
- Face-less / single-triangle meshes evaluate to exactly zero energy.

**R10-E6 [INFO] IPC barrier verified (pairs, sphere, plane) — and a vacuous-test lesson**
- Pair gradient matches central FD at 9 distances across (0, dhat)
  (worst 4e-8 at eps=1e-9; the apparent error scales linearly with eps near
  d→0 and d→dhat — FD conditioning, not implementation error; the analytic
  2-point form is additionally pinned by the pre-existing
  `test_barrier_energy_gradient_analytic` to 1e-10).
- Antisymmetry (ΣF = 0, |ΣF| = 5e-12); exactly zero at d ≥ dhat; coincident
  pairs masked with no NaN; zero-contact scenes: barrier exactly zero and
  one implicit step reduces an independently computed incremental potential
  to 0.1% (with injected random velocities — note: with zero initial
  velocity, uniform gravity gives rigid free-fall and ψ ≡ 0 is the CORRECT
  solution; an oracle expecting decrease there would be wrong).
- Sphere+plane obstacle gradient/Hessian match FD (4e-9 / 4.5e-8) — but the
  first version of this check was VACUOUS: random probe points landed inside
  the sphere (negative gap is masked inactive) and beyond dhat of the plane,
  so F = 0 = FD and a planted sign-mutation passed unnoticed. The promoted
  regression test now constructs points at active gaps in (0.001·dhat,
  0.95·dhat) and asserts E > 0 first; under the same mutation it fails with
  rel = 0.21.

**R10-E7 [INFO] Quantized package verified against scalar references**
- `morton_inc/dec_{x,y}_2d`: exhaustive over all in-grid keys at depths 2–4
  vs independent decode→offset→encode (0 mismatches; bit-plane isolation
  verified; out-of-grid wraparound is the documented domain edge).
- `FastMortonNeighborTable2D.get_all_neighbors_batch`: exhaustive at depths
  2/3/4 (every key × 9 offsets) vs scalar reference, boundary → −1 exactly,
  center column = input key with depth bits preserved, empty input → (0, 9).
- `pack/unpack_particles_64bit_3d` (depths 4/6/8) and `pack/unpack_
  particles_32bit_2d` (depths 4/5/6): roundtrip position error within the
  floor+frac quantization bound at boundary values {0, cell edges, last
  cell}; fp16 charges ≤ 2^-10 rel; int8 charge grid multiples of 1/64
  roundtrip exactly with signedness preserved; ±2 clipping asymmetric per
  the documented int8 range (127/64 vs −2.0); out-of-domain positions clip
  (no wrap); duplicates → identical words; empty inputs roundtrip.
- Bitboards 2D/3D: occupancy sets and popcounts equal reference sets over
  random trials, full 64×64 grid, and empty input.
- Greedy aggregator: K ≤ 4 identity; sibling run + distant leaf → correct
  macro count/ratio/center (M2M moment math itself pinned by the pre-existing
  `tests/quantized_bitpacked_optimization/test_greedy_multipole_mesh.py`
  against direct parent P2M at 1e-12).
- `VoxelPackedTreeFreeFMM` vs direct O(N²) log-kernel sum: baseline (no
  lossy flags) rel-L2 1.31e-3; `enable_direct_strides` and
  `enable_bitboard_skip` byte-identical output vs disabled (lossless claim
  holds); packing/greedy measured lossy (0.15–0.29 / 0.31–0.55 depending on
  N — same order as the README's documented 0.12–0.14 / 0.23–0.30); N=0, N=1,
  coincident clusters finite; depth>6 + packing raises as documented.
- Tetra broadphase scaffold: occupied-neighbor tally equals a brute-force
  count (162/162, 174/174 across seeds).

**R10-E8 [INFO, residual risk] Coincident non-topological vertex pairs stall the line search**
- A pair at distance ~0 is excluded from the barrier (masked at d ≤ 1e-9),
  but the line-search validity floor (d < 1e-4) marks every trial invalid, so
  all 6 halvings fail and the solver keeps x unchanged (warning printed,
  `line_search_failures` counted). No NaN, no penetration, but the step is
  frozen for that Newton iteration. Inherent to vertex-vertex IPC with a
  discrete distance check (no CCD); documented in the solver docstring. Not
  fixed (design limitation, not a defect).

### Mutation evidence for Wave E gates

- Barrier pair-gradient term perturbed (`/d` → `/dhat`): caught by 2
  pre-existing tests (`test_barrier_energy_gradient_analytic`,
  `test_barrier_hessian_vector_product_fd`).
- Sphere obstacle gradient perturbed the same way: caught by the new
  `test_obstacle_barrier_gradient_finite_differences` ONLY after its vacuous
  construction was fixed (see R10-E6) — rel 0.21 vs threshold 1e-4.
- Neighbor-table boundary comparison off-by-one (`<` → `<=`): caught by the
  new exhaustive `test_neighbor_table_exhaustive_depth3`.
- R10-E1 / R10-E3 gates were RED against the pre-fix code (revert = the
  mutation).

---

## Wave F: pending

The following wave is defined in
`docs/ROUND10_DEEP_VERIFICATION_INSTRUCTIONS.md` but has not yet been executed:

- **Wave F:** `video_streaming_codecs/` + `graphics_rendering/` +
  `game_mechanics_spatial/` + `apps/` + `native/` + `tools/`

---

## Wave F: executed 2026-08-22 (round-9 session; supersedes "pending" above)

Wave F scope (`video_streaming_codecs/` + `graphics_rendering/` +
`game_mechanics_spatial/` + `apps/` + `native/` + `tools/`) was covered by
an independent adversarial agent pass plus main-agent gates. Findings (all
fixed and re-verified in the same session):

- **R10-F1 [CRITICAL, validation integrity] X-G3 ring-1 acceptance was
  vacuous** — `graphics_rendering/async_zerocopy_streaming.py` __main__
  compared a buffer view against itself (second render overwrote the
  first), printing rel-L2 0.000 and claiming "all 112 tiles within
  ring-1" for a torus that necessarily has Chebyshev-2+ pairs. True
  measured ring-1 vs all-tiles rel-L2: 3.07e-1. Fixed: outputs copied
  before the second render, honest 5e-1 regression gate, README line
  corrected. Same block printed "Real-time 60+ FPS verified" next to
  10.6–12.9 FPS measurements — replaced with measured values.
- **R10-F2 [MAJOR] `tools/run_all.py` release gate pointed at pre-move
  test paths** (~20 guaranteed FAILs after the tests/ restructure);
  re-pointed to `tests.*`, now exits 0 with 26 PASS / 1 SKIP / 0 FAIL.
- **R10-F3 [MAJOR] JAX pipeline silently vacuous**: `from jax.lax import
  segment_sum` ImportError was swallowed with JAX installed, disabling
  the whole pipeline and letting its 5 tests pass without executing.
  Fixed with a `jax._src.ops.scatter` fallback; tests now execute for
  real (rel-L2 vs direct 3.2e-10 at N=2000).
- **R10-F4 [MINOR] stale perf shares in
  `physics_simulation/.../benchmark_contact_scaling.py`** (broadphase
  28–38% measured vs claimed 15–25%) and a direction-broken
  "PPF is ~0.0x FASTER" print in `benchmark_vs_ppf.py` — both corrected.
- **R10-F5 [INFO] verified-correct**: CellIndex broadphase commit 16ff38f
  (parity exact on 4 scenes), X-G2 vectorized bounce (1.27e-2 vs exact,
  honest docs), greedy M2M rewrite (1e-15 vs direct), matrix-free IPC
  line-search crash fix + geometry caching, biosignal desync fix,
  naive-baseline de-inflation in the contact ladder.
- **Browser demo (main-agent)**: adaptive List-1 P2P subsampling was
  unweighted (leaf-aligned square artifacts; fixed by occupancy/cnt
  reweighting, verified visually at 500k and 5M), vsync cap removed
  (uncapped scheduler, default), UI option matrix reduced (two
  implementation-detail checkboxes removed; URL A/B retained), and the
  definitive 500k cross-benchmark recorded in BENCHMARKS.md +
  docs/GPU_NOTES.md §7 (funnel node-hash directory +17% on adaptive;
  near-field hash backends within ~5%; adaptive throughput behind fixed
  at all measured N due to CPU metadata rebuild).
