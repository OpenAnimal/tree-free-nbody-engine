# neural_ops — Honest Status & Naming Caveats

Audited: 2026-08-18; drop-in pass 2026-08-22 (round 9). All modules import;
`python -m pytest tests/neural_ops/ -q` passes (including SE(3) equivariance
checks) and the folder benchmarks run.

## Drop-in status (round 9)

- `neural_ops/` is self-contained: copying the folder into another codebase
  works with numpy only (torch/jax acceleration optional). Inside the source
  repository the canonical `core/` engines are used; standalone,
  `_core_deps.py` substitutes fallbacks with identical outputs (dict-backed
  CellIndex; exact direct O(N²) Gaussian transform standing in for the FGT —
  it is the FGT's own accuracy reference, so results are exact but without
  the asymptotic speedup). Pinned by `tests/neural_ops/test_dropin_standalone.py`
  (fallback parity, no unguarded `core` imports, temp-copy import+forward,
  script-mode execution).
- Coordinate contract: spatial operators quantize onto `[0,1)^dims`;
  out-of-range coordinates now emit a `RuntimeWarning`
  (`_coord_contract.check_unit_coords`) instead of silently clipping.
- Two advanced paths still want the full repo (`tayloryukawa` kernel,
  `infinite_multipole_memory_network` example) and raise informative
  ImportErrors without it.
- `MultiHeadMultipoleAttention` passes `backend=`/`jit=` through to its
  heads (previously numpy-only regardless of argument).

## Terminology caveat (read before citing)

"Multipole" in this folder's layer names (`multipole_attention`,
`hyperbolic_multipole_attention`, `multipole_gaussian_process`,
`kernel_independent_fmm`, ...) denotes **spatially bucketed aggregation with
low-order per-cell moments** (typically mean / weighted centroid), used as a
far-field approximation inside neural operators. None of these implement
Greengard & Rokhlin FMM operator hierarchies, and none use the `core/` adaptive FMM
engines (whose 2D logarithmic kernel does not match these learning kernels).
The elastic-hash-style spatial indexing is the load-bearing idea being
demonstrated.

## What is validated

- Forward passes are checked against dense/reference implementations in
  `test_fmm_neural_ops.py` (cosine-similarity / tolerance checks printed).
- Equivariant layers pass SE(3) symmetry tests (cosine similarity 0.9999).

### Far-field error law (Round-7 task T-D1)

`tests/neural_ops/test_farfield_error.py` sweeps σ/cell × grid_depth on
`TreeFreeMultipoleAttention` (N=2000, D=64, randn features) and fits the
two-term error law `rel-L2 ≈ A·(cell/σ)² + B·s_qk` where `s_qk` = mean
in-cluster std of the scaled dot τq·k.

**Measured fit:** A = -1.265e-01, B = 5.980e-01 (RSS = 0.312).

**Finding:** term (a) — the feature-dot collapse `exp(τ q_i·k_j) →
exp(τ q_i·k̄_c)` — dominates (B·s_qk ≈ 0.60 for randn-K features with
s_qk ≈ 1.0). Term (b) — the spatial dipole truncation — fits to a small
*negative* A (the two-term LS fit is not constrained to non-negative
coefficients; the spatial term is sub-dominant and the fit allocates a
small negative coefficient to it). The const-K control (term (a) zeroed)
reaches rel-L2 < 0.1 at σ/cell ≥ 4 for depth 3-4, confirming the
decomposition.

**Claimed regime (lowered from v1's "σ ≥ 2·cell → rel-L2 < 0.1"):** the
far-field approximation's accuracy is bounded by the feature spread
within clusters. For randn features (s_qk ≈ 1.0), rel-L2 is O(0.5)
regardless of σ/cell. The approximation is accurate only when features
are near-constant within clusters (s_qk ≪ 1), or when the application
tolerates O(B·s_qk) rel-L2. The spatial geometry term (A·(cell/σ)²) is
benign for σ ≳ cell.

**Round-14 re-gate + refuted fix:** the claimed regime is now ASSERTED in
CI (`test_farfield_error_law`: const-K and small-K (s_qk ≈ 0.08) reach
rel-L2 < 0.1 for σ ≥ 2·cell at every depth; randn-K stays above 0.1,
monotonically worse than small-K — the finding itself is asserted so it
cannot silently regress into an over-claim). A moment-based repair was
implemented, measured, and REVERTED: the second-order cluster-covariance
weight correction exp(τ²/2 q^T Σ_c q) relieved rel-L2 by only ~5–10%
(B: 0.59 → 0.56), and adding the first-order value-feature cross moment
Σ_j δk_j ⊗ v_j was non-monotone (≈2× better at depth 3–4, ≈2× worse at
depth 5–6 with σ/cell = 8). Root cause: the exact per-cluster value sum
Σ_j exp(τ q·k_j) v_j is lognormal-concentrated — dominated by the
within-cluster max of q·k — so no finite-moment expansion recovers it at
O(1) in-cluster spread. (Module docstring of the test records the same.)

A third, smaller term — the collapse of v_j onto cluster moments in the
value aggregation — is absorbed into the fit's residual (RSS = 0.31).

### KV-cache recall frontier (Round-7 task T-D5)

`tests/neural_ops/test_kv_cache_recall.py` measures recall@10 (cosine similarity
between cache output and exact top-k attended output) on a 5k-token synthetic
stream, sweeping the LSH hyperplane count.

**ElasticMultipoleKVCache:**

| hyperplanes | recall@10 | mean exact tokens | compression |
|-------------|-----------|-------------------|-------------|
| 4           | 0.587     | 263               | 19x         |
| 8           | 0.620     | 275               | 18x         |
| 16          | 0.971     | 256               | 20x         |
| 32          | 1.000     | 256               | 20x         |

**HierarchicalElasticKVCache:**

| hyperplanes | recall@10 | mean exact tokens | compression |
|-------------|-----------|-------------------|-------------|
| 4           | 0.701     | 256               | 20x         |
| 8           | 0.599     | 256               | 20x         |
| 16          | 0.569     | 256               | 20x         |
| 32          | 0.568     | 256               | 20x         |

**Finding:** the Elastic cache is no longer recall-limited at the high-hyperplane
configs: recall reaches **0.97 at hp=16 and 1.00 at hp=32** (≈19.5x compression,
256 exact tokens) — the tier-1/tier-2 dedup fix (recent-window tokens no longer
double-counted in the target bucket) and the one-global-max rescale recovered
the exact set. Recall at low hp (4–8) is 0.59–0.62 because the LSH bucket is
coarse and most tokens fall outside the target bucket. The Hierarchical cache
reports 256 exact tokens at every config (the eviction path is exercised,
~4744 tokens evicted) with recall 0.57–0.70 — it relies more on far-field
cluster summaries, so recall stays moderate. The README cautionary framing
("experimental; recall varies with hp") remains accurate for the Hierarchical
cache and for low-hp Elastic configs.

### Taylor-FGT attention (Round-7 task T-D3)

`neural_ops/taylor_fgt_attention.py` implements exact spatial attention via
the Gaussian FGT, plus a feature-map path for the spatial × feature softmax.

**Layer 1 (pure spatial):** `out = FGT(x, v) / FGT(x, ones)` — two FGT calls,
exact to the FGT's truncation error. Measured rel-L2 = **1.5e-6** at N=1000,
D=4, σ=0.15, p=8 (below the 1e-4 target). `Gaussian3DFGT` added to
`core/gaussian2d_fgt.py` (the eigenfunction identity is dimension-independent).

**Layer 2 (spatial × feature softmax):** uses FAVOR+/Performer-style positive
random features to split the product kernel. Measured rel-L2 = **0.3951** at
N=500, m=16 — the expected Performer variance for small m. The feature-map
error is a ratio estimator whose variance is the known Performer pain point;
the deliverable is the measured rel-L2-vs-m curve, not an appeal to a bound.
The PyTorch autograd path (T-D3 step 2) is deferred — the NumPy-exact layer
is the verified foundation.

### Dense-check citation (Round-7 audit)

The `test_fmm_neural_ops.py` suite checks shape, NaN, Inf, and
CellIndex-occupancy consistency, but does NOT compare the
`TreeFreeMultipoleAttention` output against a dense O(N²) softmax reference.
The dense accuracy comparison lives in `test_farfield_error.py` (which reports
the rel-L2 error law, not a pass/fail gate). The `test_neural_ops_advanced.py`
suite checks shape/NaN/Inf for the 10 advanced modules; the
`MultipoleAdjointEngine` is the only module with a dense FD-verified accuracy
gate (max rel-error = 3.57e-10). The `benchmark_diffusion_and_gp.py` suite
includes a dense Cholesky comparison for the GP layer (mean error < 0.05,
variance error < 0.15).

### Near-field cell derivation fix (multipole_flow_drift, 2026-08-21)

`TreeFreeMultipoleFlowDrift.compute_drift` previously derived the source
bucket's near-field neighbor cell from the cluster **centroid**
(`floor(center * res)`). For a multi-particle bucket whose members hug a cell
boundary, floating-point rounding of the centroid can land it in a different
cell than every member, and the ring-1 neighborhood around the centroid cell
then misses a cell that is ring-1 adjacent to the bucket's actual cell —
demoting an exact near-field pair to an approximate far-field pair.

The fix derives the source cell from the bucket key `k_src` itself
(`k_src % res`, `(k_src // res) % res`, `k_src // res**2`), which is the
bucket's true cell by construction. Regression test
`_test_near_field_cell_from_bucket_key` (in `multipole_flow_drift.py`'s
`__main__`) constructs a 3-particle cluster hugging a cell boundary with a
probe in the ring-1-adjacent cell, and asserts (a) the new near field
includes the probe, (b) the old (centroid-shifted) near field would not, (c)
the new total near-field pair set equals brute-force ring-1 particle-cell
adjacency, and (d) `compute_drift` produces a non-zero repulsive force on the
cluster from the probe. The pre-existing `_test_dipole_sign_2charge` still
passes (rel-L2 = 9.9e-5 vs the exact 2-charge Coulomb field).

## Recommendation

Treat the layer names as brand names for "hash-bucketed far-field
approximation," not as FMM claims. At the next breaking revision, rename
`fmm` → `far_field` / `cluster` where no operator hierarchy exists.
