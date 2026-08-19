# neural_ops — Honest Status & Naming Caveats

Audited: 2026-08-18. All modules import; `test_fmm_neural_ops.py` passes
(including SE(3) equivariance checks) and the folder benchmarks run.

## Terminology caveat (read before citing)

"Multipole" in this folder's layer names (`multipole_attention`,
`hyperbolic_multipole_attention`, `multipole_gaussian_process`,
`kernel_independent_fmm`, ...) denotes **spatially bucketed aggregation with
low-order per-cell moments** (typically mean / weighted centroid), used as a
far-field approximation inside neural operators. None of these implement
Greengard–Rokhlin FMM operator hierarchies, and none use the `core/` CGR88
engines (whose 2D logarithmic kernel does not match these learning kernels).
The elastic-hash-style spatial indexing is the load-bearing idea being
demonstrated.

## What is validated

- Forward passes are checked against dense/reference implementations in
  `test_fmm_neural_ops.py` (cosine-similarity / tolerance checks printed).
- Equivariant layers pass SE(3) symmetry tests (cosine similarity 0.9999).

## Recommendation

Treat the layer names as brand names for "hash-bucketed far-field
approximation," not as FMM claims. At the next breaking revision, rename
`fmm` → `far_field` / `cluster` where no operator hierarchy exists.
