# bioinformatics — Honest Status & Naming Caveats

Audited: 2026-08-18. All modules import; `test_sota_modules.py` passes
(19 modules verified) and the folder benchmark runs.

## Terminology caveat (read before citing)

Several files reference FMM/multipoles (`solvation_free_energy`,
`eeg_source_localization_fmm`, `diff_fmm_guidance`, `whole_cell_viral_simulation`,
...). The physical kernels here are 3D Coulomb/Yukawa or Poisson-type kernels,
**not** the 2D logarithmic kernel solved by the repo's cross-validated CGR88
core. What these modules actually implement are hash-bucketed cluster
summations (order-0/1 moments per spatial cell) — a reasonable screened-far-field
approximation, but not a translation-based FMM. The `apps/app5_bioinformatics.py`
demo states this explicitly and quantifies the approximation.

## What is validated

- `test_sota_modules.py` exercises all 19 modules with synthetic-data checks.
- `kmer_elastic_hash.py` uses the core `ElasticHashTable` for real
  (queried, load-bearing) k-mer indexing.

## Recommendation

Where 3D electrostatics accuracy matters, pair these approximations with a
direct reference before drawing scientific conclusions. At the next breaking
revision, rename FMM-labeled modules to reflect the cluster-approximation
reality, or port them onto a true 3D FMM (not currently in this repo).
