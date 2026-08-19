# Inapplicability Taxonomy — where the tree-free FMM does NOT apply

The tree-free FMM in `core/` is a *fast kernel-sum* engine: it approximates
SUMS of the form `u(x_i) = sum_j q_j G(x_i - x_j)` for translation-invariant
radial kernels `G` on a uniform grid, using CGR88 multipole/local expansions
over funnel-hash-indexed occupied cells. Not every problem in the repo is
such a sum. This file catalogs the cases where the FMM is inapplicable, so
the BENCHMARKS.md "omitted" rows have a single permanent home.

Each class gives: (a) what the problem actually is, (b) one concrete
falsifiable reason the FMM does not apply, (c) the closest fast technique
that DOES apply, and (d) a link to the BENCHMARKS.md table that demonstrates
the inapplicability.

---

## Class A — "not a kernel sum" (argmax / nearest-neighbor queries)

The FMM approximates a SUM over sources weighted by a kernel. These apps ask
an ARGMAX / nearest-neighbor question, which is not a sum at all — there is
no kernel to expand and no cancellation structure to exploit. Multipole
expansion of `max_j G(x_i, x_j)` is meaningless; the answer is a single
index, not an aggregate.

**Affected apps:** app6 (MuJoCo footpad nearest terrain point per probe),
app7 (high-dim LSH partition + cosine top-k retrieval), app9 (streaming
vector DB multi-probe LSH ANN).

**Falsifiable reason:** for app6, the per-probe output is `argmin_j |x_i -
p_j|` — a single index, not a sum; for app7/app9, the output is the top-k
indices ranked by cosine, again an argmax-style query. No `sum_j` appears in
either objective, so there is no multipole/local expansion to truncate.

**Closest fast technique:** spatial grid filters / LSH (already used in the
apps). For app6 the 3x3 `CellIndex` neighborhood is a correct filter (no
missed closest points); for app7/app9 random-hyperplane LSH + funnel-hash
bucketing is the standard ANN technique.

**Evidence tables:** [App 6](../BENCHMARKS.md#app-6-mujoco-footpad-proximity-3d-nearest-point-search),
[App 7](../BENCHMARKS.md#app-7-high-dim-lsh-partition-retrieval-cosine-top-k),
[App 9](../BENCHMARKS.md#app-9-streaming-vector-db-cautionary-fine-grained-lsh-partitions-collapse-recall).

---

## Class B — "kernel lacks FMM structure" (non-translation-invariant / non-radial)

The CGR88 expansion requires a kernel that is radial (a function of `|x-y|`
alone) and translation-invariant, so that `G(x-y)` admits a Taylor /
multipole expansion in `x-y`. Softmax attention `softmax(q·k / sqrt(d))` is
neither radial nor translation-invariant — it depends on the dot product
`q·k`, which is bilinear in the arguments, not a function of their
difference. There is no `G(r)` to expand.

**Affected apps:** app3 (spatial-hash attention with a 2D Gaussian RBF is
radial and DOES have FMM structure — see Class D for why it is still
omitted at demo scale; the softmax-attention variant documented here is the
non-radial case), app10 (continuous spatial GNN with a 2D Gaussian message
kernel — radial, see Class D). The pure softmax-attention case is documented
as the canonical Class B example.

**Falsifiable reason:** for a softmax-attention head, the contribution from
key `k_j` to query `q_i` is `exp(q_i·k_j / sqrt(d)) / sum_l exp(q_i·k_l /
sqrt(d))`. The numerator depends on `q_i·k_j`, not on `|q_i - k_j|`; a
multipole expansion in `q_i - k_j` does not exist because the kernel is not
a function of that difference.

**Closest fast technique:** linear-attention / random-feature kernelization
(Performer-style) approximates the softmax kernel with random features so
the attention sum becomes a linear-time matrix product. This is documented
here as the known alternative; it is NOT implemented in this repo (the apps
use spatial hashing instead, which is exact in the near field).

**FGT note (round-3 optional stretch, NOT attempted):** for the *radial
Gaussian RBF* variant of app3 (`G(r) = exp(-r^2/h^2)`, which IS radial and
so is technically a Class D "right technique, wrong scale" case rather than
a true Class B non-radial kernel), the corresponding fast transform is the
**Fast Gaussian Transform** (Greengard & Strain 1991), whose derivative-
tensor recursion `G_{n+1} = G_n'(r)/r` is the same one used in
`core/yukawa3d_fmm.py` — for a Gaussian it closes as
`G_n(r) = (-2/h^2)^n * exp(-r^2/h^2)`. A `+fgt (Taylor order-4 Gaussian)`
row for app3 was the round-3 optional stretch, but it was NOT implemented
this round: app3 is 2D and the round-3 derivative-tensor FMM machinery was
built for the 3D Yukawa case (3D multi-indices, `CellIndex(dims=3)`); a 2D
port of the P-tensor / M2L / L2P assembly is a separate piece of work, and
the plan permitted skipping it with an honest pointer to the FGT literature
rather than faking a half-derived row. The derivative-tensor machinery in
`core/yukawa3d_fmm.py` is kernel-agnostic (the `P_{alpha,n}` recursion only
uses `G_n'(r) = r G_{n+1}(r)`) and could be retargeted to the Gaussian in a
future round by swapping the `G_n` evaluator and dropping to 2D multi-
indices.

**Evidence table:** [App 3](../BENCHMARKS.md#app-3-spatial-hash-attention-2d-gaussian-rbf)
(Gaussian RBF, radial — Class D applies; the softmax variant is the
documented Class B non-radial case).

---

## Class C — "right kernel, our FMM is 2D-only" (3D radial kernels)

The kernel is a perfectly good radial translation-invariant 3D kernel, so an
FMM exists in principle — but the flagship `FastVectorizedFMM` /
`CGR88AdaptiveFMM` engines in `core/` are 2D (complex-CGR88 log kernel).
A 3D FMM needs real-space derivative tensors, not complex log expansions.

**Affected apps:** app5 (3D Debye-Huckel screened Coulomb / Yukawa
potential `G(r) = exp(-kappa r)/r`), volumetric AO (3D inverse-square
kernel, candidate for a future 3D FMM).

**Falsifiable reason:** the 2D CGR88 engine expands `log|x-y|` as a complex
Taylor series in `z = (x+iy)`; the 3D Yukawa kernel `exp(-kappa r)/r` has no
such holomorphic representation. A 3D FMM must instead use real derivative
tensors `d^alpha G / dx^alpha` over multi-indices `alpha = (a,b,c)` — a
different mathematical object.

**Status — PARTIALLY FIXED:** the 3D Yukawa case is now handled by
`core/yukawa3d_fmm.py` (`Yukawa3DFMM`, single-level flat scheme with the
exact derivative-tensor math from the round-3 plan). See the `+fmm
(Yukawa3DFMM)` row in the [App 5](../BENCHMARKS.md#app-5-3d-protein-electrostatics-debye-huckel-screened-coulomb)
table. The volumetric AO 3D inverse-square kernel remains a candidate for a
future 3D FMM (the derivative-tensor machinery in `yukawa3d_fmm.py` is
kernel-agnostic and could be retargeted).

**Evidence table:** [App 5](../BENCHMARKS.md#app-5-3d-protein-electrostatics-debye-huckel-screened-coulomb),
[Graphics rendering — Volumetric AO](../BENCHMARKS.md#graphics-rendering-volumetric-ao-3d-inverse-square-kernel).

---

## Class D — "right technique, wrong scale" (Python per-cell loop constants dominate at demo N)

The FMM / spatial-hash IS the right algorithm and IS asymptotically faster,
but at the demo's small N the Python interpreter overhead per occupied cell
(a per-cell loop in pure Python, hash probes, list materialization) is
larger than the O(N^2) vectorized NumPy direct sum. The asymptotic win
exists; it just does not appear at the demo's N. This is a constant-factor
issue of the Python driver, not an algorithmic fact — see
[GPU_NOTES.md](GPU_NOTES.md) for where the compiled kernels measure the real
constant factors.

**Affected apps/domains:** core FMM at N=2000 (the crossover table in
[Core FMM scaling](../BENCHMARKS.md#core-fmm-scaling) shows where the
crossover actually appears), flocking at N=1000, app3 at N=1500, app4 at
N=400.

**Falsifiable reason:** at N=2000 the direct vectorized sum is ~40 ms while
the flat FMM is ~480 ms — the FMM's per-cell Python loop over occupied
cells (K^2 M2L pairs in pure Python) costs more than the 4e6-pair vectorized
direct. As N grows the direct cost grows as N^2 while the FMM near-field
grows as N * neighbors, so a crossover exists; the scaling table locates it
(or reports its absence up to N_max with the per-N ratios).

**Closest fast technique:** the FMM itself, run at larger N, OR the compiled
kernels (`core/cuda_kernels`, `core/triton`, `native/zig`, `webgpu`) where
the per-cell constant is a few cycles instead of a Python interpreter
dispatch.

**Evidence tables:** [Core FMM (2D log kernel)](../BENCHMARKS.md#core-fmm-2d-log-kernel),
[Core FMM scaling](../BENCHMARKS.md#core-fmm-scaling),
[Game mechanics — flocking](../BENCHMARKS.md#game-mechanics-massive-crowd-flocking-2d-unit-mode),
[App 3](../BENCHMARKS.md#app-3-spatial-hash-attention-2d-gaussian-rbf),
[App 4](../BENCHMARKS.md#app-4-elastic-hash-boids-1euro-near-field-boid-rules).

### Class D-adjacent — app9 (LSH recall collapse, not just speed)

App9 is filed under Class A (not a kernel sum) but ALSO borders Class D: the
LSH is the right technique and IS faster (3.4x in the table), yet recall@10
collapses to 0.6% on this corpus. The cause is a *data-geometry* constant,
not a Python constant: the corpus is 20 tight Gaussian clusters, so the
true top-10 of a query near a cluster center is determined by fine
within-cluster noise alignment, which requires a fine LSH partition, which
empties the buckets. A hyperplane/table/probe sweep showed recall@10 >= 0.5
is reachable only at ~0.13x speed (8x slower than brute) and >= 1.5x speed
is reachable only at recall@10 <= ~5%. No middle ground exists on this
corpus. See the cautionary app9 table:
[App 9](../BENCHMARKS.md#app-9-streaming-vector-db-cautionary-fine-grained-lsh-partitions-collapse-recall).
