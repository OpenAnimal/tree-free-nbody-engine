# algorithm_theory — Honest Status & Naming Caveats

Audited: 2026-08-18. All modules import and pass `test_basic_datatypes_fmm.py`
and the folder benchmark runs end-to-end.

## Terminology caveat (read before citing)

Many files in this folder carry `fmm`/`multipole` in their names. After audit,
**most of them are order-0/order-1 cluster (Barnes–Hut-style tree-code)
approximations, not Greengard–Rokhlin translation-based FMMs** — there is no
M2M/M2L/L2L operator hierarchy in them. Where a module genuinely implements
operator-based FMM machinery, it says so. The two core engines with full
cross-validated FMM math live in `core/` (CGR88 adaptive, flat vectorized).

## Verified with numeric cross-validation

- `screened_yukawa_fmm.py` — real near/far tree-code with an order-1 dipole
  correction and screening truncation; ~1% rel L2 vs its own `direct_evaluate`
  (annotated in-file: tree-code, not translation-based FMM).
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
