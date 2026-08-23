# bioinformatics — Honest Status & Naming Caveats

Audited: 2026-08-21. All modules import; `test_sota_modules.py` passes
(19 modules + k-mer vectorization parity check verified) and the folder
benchmark runs.

## Far-field engine (T-C1)

The verified 3D Yukawa FMM (`core.yukawa3d_fmm.Yukawa3DFMM`) is wrapped by
`TaylorYukawaBioFMM` in `core.fast_multipole_kernel`, which:

- Maps Ångström coordinates to the unit box, rescales κ accordingly, and
  converts unit-box potentials back to kcal/mol/e.
- Calls `evaluate_targets` (sources ≠ targets) and `evaluate_forces`
  (returns E = -dV/dx) on the underlying `Yukawa3DFMM`.
- Caches the FMM instance by `(depth, p, kappa_unit)` cache key so repeated
  evaluations at the same grid resolution reuse the precomputed Taylor
  tables instead of rebuilding them.

The `TreeFreeBioFMM` class (the older hash-bucketed cluster summation)
remains for backward compatibility; new code should prefer
`TaylorYukawaBioFMM` for 3D electrostatics.

## Complexity

The far-field engine is a **single-level flat scheme** with dense (K,K) M2L,
giving **O(N·K) far-field cost** where K is the cell count — **O(N) at fixed
cell count**. A multilevel O(N) FMM is future work.

## Terminology caveat (read before citing)

Several files reference FMM/multipoles (`solvation_free_energy`,
`eeg_source_localization_fmm`, `diff_fmm_guidance`, `whole_cell_viral_simulation`,
...). The physical kernels here are 3D Coulomb/Yukawa or Poisson-type kernels,
**not** the 2D logarithmic kernel solved by the repo's cross-validated adaptive FMM
core. `TaylorYukawaBioFMM` delegates to the verified `Yukawa3DFMM` (Taylor
expansion + M2L/L2L/P2L translations); the older `TreeFreeBioFMM` implements
hash-bucketed cluster summations (order-0/1 moments per spatial cell) — a
reasonable screened-far-field approximation, but not a translation-based FMM.
The `apps/app5_bioinformatics.py` demo states this explicitly and quantifies
the approximation.

## What is validated

- `test_sota_modules.py` exercises all 19 modules with synthetic-data checks.
  The cross-validation benchmark numbers (Pearson r, ROC-AUC) are computed
  on **synthetic self-generated data** — circular validation smoke tests,
  not external-accuracy figures (Round-7 honesty pass, finding F-12).
- `kmer_elastic_hash.py` uses the core `ElasticIntTable` (funnel hash) for
  real (queried, load-bearing) k-mer indexing (Round-7 task T-A3 fixed
  finding F-02). The `ingest_sequence` method is now vectorized (numpy
  2-bit packed rolling k-mers with vectorized canonical-form via bit-pair
  reversal XOR); a parity test against the legacy per-character loop is
  included in `test_sota_modules.py` (`test_kmer_vectorization_parity`).

## Recommendation

Where 3D electrostatics accuracy matters, pair these approximations with a
direct reference before drawing scientific conclusions. At the next breaking
revision, rename FMM-labeled modules to reflect the cluster-approximation
reality, or port them onto a true multilevel 3D FMM.
