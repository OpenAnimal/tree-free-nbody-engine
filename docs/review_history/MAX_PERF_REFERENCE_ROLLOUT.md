# Max-Performance Reference Rollout Instructions

**Audience:** any implementer (including a less-capable coding model) tasked with
making `tree-free-nbody-engine` a **professional, fully functional reference**
for massive performance gains via:

1. **Near-field:** elastic / funnel hashing + cell lists / CSR P2P  
2. **Far-field:** the correct FMM / multipole / FGT / centroid model for the
   force kernel (not "enable multipoles everywhere")

**Working directory:** repository root  
`D:\Unreal Engine\FMM_Repos\tree-free-nbody-engine`  
(or the clone path you are using)

**Always run Python as:**

```bash
python -X utf8 <script>
```

**Full gate after meaningful changes:**

```bash
python -X utf8 tools/run_all.py
```

**Do not weaken tests or acceptance thresholds.** If a number must move, stop
and report why with evidence.

---

## 0. Mission (read this first)

### Goal

Every `apps/`, domain folder, `neural_ops/` example, browser demo path, and
core backend that **can** benefit from tree-free spatial structure must:

- Use the **canonical** near-field stack (`CellIndex` / funnel hash / CSR), not
  ad-hoc `dict` buckets or pure O(N²) Python double loops as the hot path.
- Use the **correct** far-field model for its kernel (or honestly document
  why multipoles do not apply — see §2 and `docs/INAPPLICABILITY.md`).
- Ship a `benchmark_variants.py` (or domain equivalent) that reports
  **latency AND accuracy** (`rel L2` or `recall@k`), including "NOT faster at
  this scale" when true.
- Prefer **vectorized NumPy**, then **compiled** paths (Zig / WebGPU / CUDA /
  JAX) where Python Class-D overhead dominates (see §2 Class D).

### Non-goals (do not do these)

- Do **not** bolt gravity adaptive FMM multipoles onto boids / SPH / ANN / argmax
  problems and claim "FMM everywhere."
- Do **not** hide slowdowns or accuracy loss. Honesty is part of the product.
- Do **not** rename open-addressing linear probe as "funnel hash."
- Do **not** change security, CI skip gates, or lower test tolerances to make
  green.
- Do **not** invent new hash implementations per app. Extend `core/`.

### Success definition (reference-grade)

A module is **reference-complete** when **all** of the following hold:

| # | Criterion |
|---|-----------|
| R1 | Hot path uses `core.spatial_index.CellIndex` and/or CSR (`core._csr` / `core.csr_p2p`) for local neighbors, **or** a documented Class A/B reason why not |
| R2 | Far-field uses the correct engine for the kernel (table in §3), or `farField=none` with a one-line kernel-class note |
| R3 | `standard` vs `+elastichash` (and `+fmm` / domain FGT / Yukawa when applicable) exists in `benchmark_variants.py` |
| R4 | At least one N where the accelerated path is **≥ 1.5×** faster than the exact/brute reference **or** a scaling table shows crossover (Class D honesty) |
| R5 | Accuracy gate: kernel sums `rel L2 ≤ 1e-3` (prefer ≤ 1e-5 for gravity/electrostatics at moderate p); filters `no missed`; ANN `recall@k` reported honestly |
| R6 | No forbidden vocabulary lies (run `tools/lint_claims.py`) |
| R7 | Demo / README pointers match reality (name "FMM" only if multipoles or FGT actually run) |

---

## 1. Architecture axioms (memorize)

These were established in the live WebGPU demo work and apply repo-wide.

### 1.1 Two independent axes

```
┌────────────────────────────────────────────────────────────────────┐
│  NEAR-FIELD membership backend                                     │
│  How do I find local neighbors?                                    │
│  counting-sort CSR | open-addr linear probe | funnel (FCK 2025)  │
│  SHARED wherever spatial P2P / ring-1 filters apply                │
└────────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────────┐
│  FAR-FIELD force model                                             │
│  What long-range approximation (if any)?                           │
│  none | gravity-adaptiveFMM | boid-centroids | FGT | Yukawa3D | biot…   │
│  CHOSEN BY KERNEL, not by folder name or marketing label           │
└────────────────────────────────────────────────────────────────────┘
```

**UI / browser reference:** `index.html` controls  
`#selectHashMode` (near) and `#selectFarField` (far).  
Gravity FMM mode/order are **sub-controls of farField=gravity only**.

### 1.2 What multipoles actually require

FMM multipoles approximate sums:

```text
u(x_i) = Σ_j q_j G(x_i − x_j)
```

for a **radial, translation-invariant** kernel `G`. If the problem is not that
sum, multipoles do not apply. Full taxonomy: `docs/INAPPLICABILITY.md`.

### 1.3 Funnel hash vs open addressing vs counting sort

| Backend | Module / demo | When to use |
|---------|---------------|-------------|
| **Funnel (FCK)** | `core/elastic_hash.py` `ElasticHashTable` (default); WGSL `funnel_*` in `index.html` | Occupied-cell directory, high load, lock-free CAS story, sparse cells |
| **Open-addr linear probe** | Demo `eh_*` path; some older GPU notes | Bench axis / fallback; do **not** call it funnel |
| **Counting-sort CSR** | `core/_csr.py` `build_csr`; demo `count/scan/scatter` | Dense uniform grids; contiguous `sortedIndex` ranges; max P2P bandwidth |
| **CellIndex** | `core/spatial_index.py` | **Canonical** Python near-field API (rebuilds funnel each `build()`) |

**GPU dynamic sims:** funnel tables are append-only → **rebuild every frame**
(ping-pong). See `docs/GPU_NOTES.md` §1.

### 1.4 Naming honesty

- `apps/app4_fmm_boids_1euro.py` historically used **centroids**, not multipoles.
  Far-field label must say centroids / order-0.
- Demo option labels already distinguish Gravity FMM vs Boid cell-centroids.
- `tools/lint_claims.py` catches some overclaims — keep it green.

---

## 2. Inapplicability classes (do not fight physics)

Copy of the decision tree. Full text: `docs/INAPPLICABILITY.md`.

| Class | Meaning | Fast technique | Multipoles? |
|-------|---------|----------------|-------------|
| **A** | Not a kernel sum (argmax / NN / top-k) | Grid filter, LSH + funnel buckets | **No** |
| **B** | Kernel not radial / not TI (e.g. softmax q·k) | Spatial hash near-field, linear attention elsewhere | **No** (Gaussian RBF is different — use FGT) |
| **C** | 3D radial but flagship engine is 2D log | `core/yukawa3d_fmm.py`, `TaylorYukawaBioFMM`, future 1/r² AO FMM | **Yes, with 3D engine** |
| **D** | Right algo, Python constants kill small-N | Larger N scaling table **or** Zig/WebGPU/CUDA | Yes, but measure crossover |

**Before implementing `+fmm` on any app:** classify A/B/C/D in a comment at
the top of its `benchmark_variants.py`.

---

## 3. Canonical building blocks (use these, don't reinvent)

### 3.1 Near-field (Python)

```python
from core.spatial_index import CellIndex
from core.csr_p2p import csr_p2p_near_field   # when you have inverse + kernel_fn
from core._csr import build_csr
from core.elastic_hash import ElasticHashTable  # only if you need raw table
```

**Preferred pattern for local neighbors:**

```python
idx = CellIndex(dims=2, grid_res=1 << depth)  # or dims=3, cell_size=...
idx.build(positions)  # rebuilds funnel-backed occupancy every call
# ring-1 (Chebyshev) candidates:
for key in idx.occupied_keys():  # OK for setup; hot path uses neighborhood_*
    neigh = idx.neighborhood_indices(key, ring=1)
    # evaluate local interactions on neigh
```

**CSR pattern (high performance Python / parity with engines):**

```python
# After mapping each particle -> cell id in 0..K-1:
cell_start, cell_particles, _ = build_csr(inverse, K)
# cell_particles[cell_start[c]:cell_start[c+1]] contiguous
```

See `core/csr_p2p.py` for a complete near-field kernel using CSR + CellIndex.

### 3.2 Far-field engines by kernel

| Kernel | Engine | Path |
|--------|--------|------|
| 2D log / 2D gravity / streamfunction | `FastVectorizedFMM` | `core/fast_vectorized_fmm.py` |
| 2D log adaptive | `AdaptiveFMM` / tree-free elastic adaptive | `core/adaptive_fmm.py`, `core/tree_free_fmm.py` |
| 2D Gaussian RBF | `Gaussian2DFGT` | `core/gaussian2d_fgt.py` |
| 2D screened Yukawa K0 | `ScreenedYukawa2DFMM` | `core/screened_yukawa2d_fmm.py` |
| 3D Yukawa / Debye–Hückel | `Yukawa3DFMM`, bio wrapper | `core/yukawa3d_fmm.py`, `bioinformatics/core/fast_multipole_kernel.py` |
| Boid far cohesion | Order-0 centroids (not multipoles) | Demo WGSL `build_boid_centroids`; Python app4 pattern |
| Compact SPH kernels | **none** far-field | Near-field cell lists only |
| Cosine top-k / ANN | LSH + funnel | `ElasticHashTable` buckets — Class A |

### 3.3 Compiled / browser backends

| Backend | Location | Notes |
|---------|----------|-------|
| WebGPU demo | `index.html` | Shared near-field hash; gravity FMM; boid centroids; funnel WGSL |
| File WGSL | `core/webgpu_kernels/adaptive_fmm.wgsl`, `tree_free_fmm.wgsl` | Keep in sync — `tools/check_wgsl_sync.py` |
| Zig native | `native/zig/` | SIMD P2P, multipole_2d/3d — prefer for Class D rescue |
| CUDA / HIP / OpenCL / Triton | `core/cuda_kernels/`, etc. | Many are **direct P2P** or partial; do not claim funnel-FMM unless true |
| JAX | `core/jax_tree_free_fmm.py` | Flat scheme + primitives; multi-level still partial |

### 3.4 Benchmark protocol (mandatory)

```python
from core.benchmark_kit import VariantBenchmark
```

Axes (use all that apply):

- `standard` — exact / dense / brute reference  
- `+elastichash` — CellIndex / funnel near (or full app path if hash-based)  
- `+fmm` / `+fgt` / `+yukawa3d` — only if kernel admits it  
- `+quantized` — only if pack path exists  

**Never report speed without accuracy.**

Regenerate docs tables into `BENCHMARKS.md` when numbers change meaningfully.

---

## 4. Repository map (what lives where)

```text
tree-free-nbody-engine/
├── core/                    # Canonical FMM + hash + CSR + tests + GPU kernels
├── apps/                    # app1..app10 case studies + *_benchmark_variants.py
├── index.html               # Live WebGPU/WebGL demo (flagship interactive ref)
├── native/zig/              # C-ABI high-perf backend
├── quantized_bitpacked_optimization/
├── neural_ops/              # Linear attention / FMM neural layers + examples/
├── bioinformatics/          # Structural bio + TreeFreeBioFMM / TaylorYukawaBioFMM
├── algorithm_theory/        # Many research kernels (screened, OT, geodesic, …)
├── environmental_modeling/  # Domain kernels (plume, dose, …)
├── physics_simulation/ppf_contact_solver_fmm/
├── graphics_rendering/      # AO, radiosity, …
├── video_streaming_codecs/  # Splats, motion, …
├── game_mechanics_spatial/  # Flocking, LOS, pathfinding, …
├── docs/                    # GPU_NOTES, INAPPLICABILITY, THIS FILE
├── tools/run_all.py         # Master verification matrix
├── BENCHMARKS.md            # Honest variant tables
├── OVERVIEW.md              # Architecture narrative
└── README.md                # Public-facing claims (must stay consistent)
```

---

## 5. Audit matrix (deep review snapshot)

Use this as the backlog. Status meanings:

- **REF** — already a good reference path (still check R4 scale)  
- **PARTIAL** — hash or FMM present but incomplete / slow Python / wrong label  
- **GAP** — missing accelerated path or wrong technique  
- **N/A** — Class A/B; hash/LSH only  

### 5.1 `apps/`

| App | Problem | Class | Near-field today | Far-field today | Status | Required work |
|-----|---------|-------|------------------|-----------------|--------|---------------|
| **app1** galaxy | 2D log gravity | FMM-OK | via FMM cells | `FastVectorizedFMM` | PARTIAL→REF | Ensure CellIndex/`+elastichash` near+far story; scale N past crossover; align with demo gravity path |
| **app2** hydro | 2D log streamfunction | FMM-OK | FMM cells | FMM ψ + FD velocity | PARTIAL | FD error floor honest; vectorize or Zig; larger grids; velocity from multipole grad if possible |
| **app3** spatial attention | Gaussian RBF / softmax | D / B | ElasticHash per-cell Python | FGT optional | PARTIAL | Hot path → CellIndex+vectorized or FGT; softmax stays near-hash only |
| **app4** boids | Reynolds rules | D (near) | ElasticHash 3×3 Python loops | **centroids** (not multipoles) | PARTIAL | Rewrite step with CellIndex + NumPy gather; match demo `centroids` far field; rename messaging |
| **app5** protein | 3D Yukawa | C (fixed) | funnel bio FMM | `TreeFreeBioFMM` / TaylorYukawa | REF-ish | Push N; ensure CSR near; keep p-convergence tests |
| **app6** MuJoCo proximity | nearest point | **A** | ElasticHash | none | PARTIAL | CellIndex world-mode filter; prove zero misses; vectorize distance |
| **app7** high-dim memory | cosine top-k | **A** | LSH+funnel | none | PARTIAL | Multi-probe + rerank; report recall@k vs speed Pareto |
| **app8** manifold k-NN | k-NN graph | **A**/graph | LSH+funnel | none | REF-ish | Keep 100% recall path; add larger-N bench |
| **app9** vector DB | ANN | **A** | LSH+funnel | none | GAP (recall) | Document corpus pathology; provide preset with usable recall OR different dataset |
| **app10** continuous GNN | Gaussian messages | D | ElasticHash Python | FGT available | PARTIAL | Wire FGT as default far/message path; vectorize |

### 5.2 Domain folders

| Folder | Role | Near-field | Far-field | Priority work |
|--------|------|------------|-----------|---------------|
| `game_mechanics_spatial/` | flocking, LOS, pathfinding | CellIndex / hash in variants | centroids / harmonic fields | **Port flocking hot path off pure Python loops**; scale N to show >1.5× |
| `graphics_rendering/` | volumetric AO | hash near/far | order-0 / quantized | **3D 1/r² multipole or FGT-style** for AO (Class C residue); fix slow near/far Python |
| `video_streaming_codecs/` | Gaussian splats | cell bucket | order-0 mean | Vectorize splat; optional better multipole SH if justified |
| `physics_simulation/ppf_contact_solver_fmm/` | IPC / tet contact | CellIndex broadphase | barrier field | Keep broadphase REF; ensure matrix-free path uses spatial bins |
| `environmental_modeling/` | plume, dose, electrolyte | varies | kernel-specific | Classify each module A–D; wire CellIndex + correct FMM |
| `bioinformatics/` | many engines | elastic_spatial_hash + FMM | Yukawa / contacts | Prefer `core` CellIndex + TaylorYukawa; audit modules still on O(N²) |
| `algorithm_theory/` | research zoo | mixed | many FMM-named modules | Each file: (1) kernel class (2) uses `core` engines (3) bench or STATUS skip reason |
| `neural_ops/` | TFMA, flash, GNN, KV | bucketing helpers | multipole attention | Ensure examples call fast kernels; watch `grid_depth` vs `grid_res` convention (§3.1 note in spatial_index) |
| `quantized_bitpacked_optimization/` | pack / Morton | bitboard | packed FMM | Keep ablation honest; integrate pack axis into more apps |
| `native/zig/` | C-ABI | simd_p2p | multipole_2d/3d | **Use as Class D escape hatch** for apps stuck in Python |

### 5.3 Browser `index.html` (flagship interactive)

| Feature | Status | Follow-up |
|---------|--------|-----------|
| Near-field hash shared (galaxy/boids/vortex/sph) | Done | Keep parity tests; stress funnel at 500k/5M |
| Far-field model select | Done | Wire vortex `biot` when kernel ready |
| Gravity fixed + adaptive FMM | Done | P2P still dominates 5M — leaf auto-tune + CSR coalescing |
| Boid centroids far field | Done | Match Python app4 coeffs; add accuracy note |
| Vortex/SPH cell lists | Done (near) | Optional true Biot-Savart FMM / SPH density CSR |
| WGSL sync with `core/webgpu_kernels/` | Partial | `tools/check_wgsl_sync.py` must stay PASS |
| Funnel = real FCK schedule | Done in demo | Never reintroduce labeling lies |

### 5.4 Core engines

| Module | Status | Work |
|--------|--------|------|
| `elastic_hash.py` | REF (funnel) | Keep tests; Zig parity tools |
| `spatial_index.py` | REF | Push all apps to use it |
| `csr_p2p.py` / `_csr.py` | REF | Wire into RadialTaylor / Yukawa near loops if ≥1.5× |
| `fast_vectorized_fmm.py` | REF Python | Class D — need Zig/GPU for product-scale |
| `adaptive_fmm.py` | correct, slow Py | GPU adaptive path is demo |
| `yukawa3d_fmm.py` | REF | More callers (graphics AO?) |
| `gaussian2d_fgt.py` | REF | app3/app10 default |
| CUDA/HIP/OpenCL/Triton | mixed honesty | Document each kernel’s real algorithm; implement true funnel-FMM only as explicit projects |
| `jax_tree_free_fmm.py` | partial | Multi-level assembly still future |

---

## 6. How to upgrade one module (step-by-step playbook)

Follow this **exactly** for each file or app. Do not skip steps.

### Step 0 — Classify

Write at top of the module or its `benchmark_variants.py`:

```text
Kernel: <formula or "argmax">
INAPPLICABILITY class: A | B | C | D | none(FMM-OK)
Near-field: required | optional | n/a
Far-field engine: none | FastVectorizedFMM | Gaussian2DFGT | Yukawa3DFMM | centroids | LSH | ...
```

### Step 1 — Establish exact reference

- Implement or keep `standard` brute / dense path.
- Freeze a RNG seed and N0 (small) where brute finishes < 2s.
- Record reference output `y_ref`.

### Step 2 — Near-field acceleration

1. Replace nested `for i: for j:` neighbor search with `CellIndex.build` +
   `neighborhood_indices(ring=…)`.
2. If particle lists per cell are hot, switch gathers to `build_csr`.
3. Validate: for filters, **zero misses** vs brute neighbor set; for forces,
   near-only rel L2 or exact match on same 3×3 box.
4. Add `+elastichash` row to variants.

### Step 3 — Far-field acceleration (only if class allows)

1. Pick engine from §3.2.
2. Wire evaluate() into the app step.
3. Cross-validate full field vs brute at N0: target rel L2 from R5.
4. Add `+fmm` / `+fgt` / etc. row with accuracy_vs=standard.

### Step 4 — Kill Class D (make it actually fast)

If accelerated path is slower at demo N:

1. Add scaling sweep N ∈ {N0, 4 N0, 16 N0} (cap by 120s budget).
2. Publish crossover in BENCHMARKS note.
3. If product needs small-N wins: port hot loop to **Zig** or reuse **WebGPU**
   patterns from `index.html` / `native/zig`.
4. Vectorize remaining Python with NumPy (no per-cell `for` over particles
   if avoidable).

### Step 5 — Reference polish

1. Docstring: algorithm, complexity, axes, honesty limits.  
2. README / OVERVIEW blurb matches reality.  
3. `python -X utf8 path/to/benchmark_variants.py` → paste table.  
4. `python -X utf8 tools/run_all.py` → no new FAIL.  
5. `python -X utf8 tools/lint_claims.py` → PASS.

### Step 6 — Commit discipline

One logical concern per commit. Message includes:

- before/after ms and accuracy  
- N and hardware class if GPU  

---

## 7. Priority task queue (execute in order)

A weaker model should work **top to bottom**, finishing acceptance before
moving on.

### P0 — Foundation + known correctness blockers

| ID | Task | Acceptance |
|----|------|------------|
| P0.1 | Confirm `tools/run_all.py` baseline green on your machine | Log PASS/SKIP counts |
| P0.2 | Read `docs/INAPPLICABILITY.md`, `docs/GPU_NOTES.md`, `BENCHMARKS.md` intro | No code |
| P0.3 | Ensure `core/test_elastic_hash.py`, `test_spatial_index.py`, `test_adaptive_fmm_*` pass | exit 0 |
| **P0.4** | **T-C8 / R7-F30 depth split-brain in `core/radial_taylor.py`:** docstring says grid is `2^depth` cells/side but code treats `depth` as **linear** cells/side. Same word means different things vs neural_ops (`1<<grid_depth`). **Fix docs + API to one convention; add a test that multi-cell M2L actually exercises far field (clusters in different cells).** | test proves M2L path; no silent same-cell collapse |
| **P0.5** | **Bio complexity honesty (`bioinformatics/core/fast_multipole_kernel.py`):** far loop is O(N·K) not O(N); `np.where(inverse==c)` rescans are hidden O(N·K). Switch far gather to CSR/`build_csr`; fix docstrings. | no false O(N) claims; lint_claims PASS; speed ≥ prior |
| **P0.6** | **`neural_ops/flash_multipole_kernel.py` far branch:** currently ~O(N²/B_c) tile loop, not true O(N). Either implement hierarchical far field or downgrade README "strict linear" claims and document complexity. | honest complexity in README/STATUS; preferably true O(N) far |

### P1 — Make near-field canonical in all apps

| ID | Task | Acceptance |
|----|------|------------|
| P1.1 | **app4** rewrite `ElasticHashBoidSwarm.step` to CellIndex + vectorized sep/align; centroids far field | variants: near exact on 3×3; full step faster or scaling crossover documented |
| P1.2 | **app3** hot path CellIndex or keep hash but vectorize per-cell; default `+fgt` when Gaussian | rel L2 ≤ 1e-5 for FGT; no softmax-as-FMM |
| P1.3 | **app6** CellIndex world-mode; zero missed closest points | `no missed == True` |
| P1.4 | **app10** CellIndex + Gaussian2DFGT messages | FGT error gate + bench row |
| P1.5 | **game_mechanics_spatial/massive_crowd_flocking.py** same as app4 pattern | >1.5× at N≥10k or scaling table |
| P1.6 | Grep apps/domains for raw `ElasticHashTable` + Python double loops; migrate to CellIndex | grep clean or justified |

### P2 — Far-field correctness where FMM applies

| ID | Task | Acceptance |
|----|------|------------|
| P2.1 | **app1** scale force validate to N where FMM ≥1.5× direct; keep rel L2 ≤ 1e-4 | bench table updated |
| P2.2 | **app2** reduce FD-dominated error or document grid-limited floor; speed path | honest note + improved N |
| P2.3 | **app5** ensure production path is TaylorYukawaBioFMM or Yukawa3DFMM not slow fallback | p-convergence test green |
| P2.4 | **graphics volumetric_fmm_ao.py** — raymarch currently O(R·S·K) all-cluster; FMM-accelerate 3D inverse-square far field (reuse yukawa tensor machinery with κ→0 or dedicated 1/r²) | +fmm row beats exact at chosen N with rel L2 ≤ 2e-2 or better |
| P2.5 | **environmental_modeling** each of 4 engines: classify + wire CellIndex + FMM/FGT if kernel-sum | tests in `test_environmental_suite.py` extended |
| P2.6 | **graphics surfel_radiosity_gi.py** vectorize near/far bounce (kill per-surfel Python O(N·K) loops) | variants show ≥1.5× or crossover |
| P2.7 | **algorithm_theory** quick wins: vectorize `quantum_fock_exchange_fmm.py` pair gen; BEM matvec true multipole if Coulomb | STATUS + bench note |
| P2.8 | Rename honesty: app3/app4/app10 filenames say FMM but use centroids — update module docstrings and README synonyms; optional file rename only if links updated | lint_claims PASS; no user-facing lie |

### P3 — Browser demo productization

| ID | Task | Acceptance |
|----|------|------------|
| P3.1 | P2P leaf auto-tune already present — re-measure 5M; document in GPU_NOTES | Main Compute ms down without accuracy loss |
| P3.2 | Implement vortex `farField=biot` **only if** particle–particle Biot-Savart is the product goal; else keep analytic + near lists | disabled reserved OR working with error gate |
| P3.3 | SPH true density/pressure via CSR cell lists (compact support only) | visual + telemetrized |
| P3.4 | Keep `tools/check_wgsl_sync.py` PASS between `index.html` embedded shaders and `core/webgpu_kernels/*` | PASS |
| P3.5 | Playwright or manual matrix: all scenarios × hash modes × far-field defaults | no WebGPU validation errors |

### P4 — Neural ops & bioinformatics reference depth

| ID | Task | Acceptance |
|----|------|------------|
| P4.1 | `neural_ops/examples/*`: each example runs and uses multipole/hash modules not dense O(N²) | script exits 0; timing print |
| P4.2 | Flash / TFMA: verify block sizes; add scaling bench row | `benchmark_neural_scaling.py` honest |
| P4.3 | bioinformatics modules still on naive pairs: migrate contacts to CellIndex | test_sota_modules PASS |
| P4.4 | algorithm_theory: STATUS.md inventory — REF / STUB / N/A per file | STATUS complete |

### P5 — Compiled backends honesty + Class D killers

| ID | Task | Acceptance |
|----|------|------------|
| P5.1 | Audit CUDA/HIP/OpenCL/Triton headers: one-line "algorithm = …" | no file claims funnel-FMM falsely |
| P5.2 | Zig binding used from at least one app hot path (app1 or flocking) | ≥5× vs pure Python at same N |
| P5.3 | Optional: Triton funnel-FMM project plan only if P0–P2 done | separate design doc, not fake rows |

---

## 8. Concrete recipes (copy-paste patterns)

### 8.1 Replace brute boids near-field

**Bad:**

```python
for i in range(N):
    for j in range(N):
        ...
```

**Good:**

```python
from core.spatial_index import CellIndex
import numpy as np

def step_boids(pos, vel, depth=5, sep_r=0.02, ali_r=0.05):
    N = pos.shape[0]
    grid = 1 << depth
    idx = CellIndex(dims=2, grid_res=grid)
    idx.build(pos)
    acc = np.zeros_like(pos)
    # Build dense cell lists once
    from collections import defaultdict
    # Prefer internal CSR if you extend CellIndex; minimal pattern:
    for key in idx.occupied_keys():
        members = idx.bucket(key)  # list of indices
        if not members:
            continue
        neigh = idx.neighborhood_indices(key, ring=1)
        if len(neigh) == 0:
            continue
        P = pos[members][:, None, :]
        Q = pos[neigh][None, :, :]
        d = P - Q
        r2 = np.sum(d * d, axis=-1) + 1e-8
        # separation / alignment masks ...
        ...
    return acc
```

Then add order-0 far field: mean of cell barycenters outside ring-1
(see `index.html` WGSL `build_boid_centroids` / app4 far loop).

### 8.2 Add FastVectorizedFMM to a 2D log force app

```python
from core.fast_vectorized_fmm import FastVectorizedFMM

fmm = FastVectorizedFMM(depth=5, order=6, softening=1e-4)
# positions (N,2) in [0,1)^2, charges (N,)
pot = fmm.evaluate(positions, charges)   # check API in file for forces vs pot
# For forces, use the app1 helper pattern in apps/app1_galaxy_collision.py
```

Cross-check:

```python
from core.adaptive_fmm import exact_direct_nbody_2d
# or app-local direct
rel = np.linalg.norm(fmm_out - direct_out) / (np.linalg.norm(direct_out) + 1e-30)
assert rel < 1e-4
```

### 8.3 Add Yukawa3D to a 3D electrostatics app

```python
from core.yukawa3d_fmm import Yukawa3DFMM
# or bioinformatics.core.fast_multipole_kernel.TaylorYukawaBioFMM for Å / kcal
```

Use existing app5 tests as golden.

### 8.4 Add Gaussian FGT

```python
from core.gaussian2d_fgt import Gaussian2DFGT
fgt = Gaussian2DFGT(...)  # read constructor in file
out = fgt.evaluate(pos, charges)
```

### 8.5 VariantBenchmark skeleton

```python
from core.benchmark_kit import VariantBenchmark

def main():
    bench = VariantBenchmark("appX_kernel_name")
    bench.add("standard (exact ...)", lambda: exact(), note="O(N^2) reference")
    bench.add("+elastichash (...)", lambda: hashed(), accuracy_vs="standard (exact ...)",
              note="CellIndex ring-1 ...")
    # only if Class allows:
    # bench.add("+fmm (...)", lambda: fmm(), accuracy_vs="standard (exact ...)", note="...")
    bench.run()  # prints table
if __name__ == "__main__":
    main()
```

Mirror flags from `apps/app1_benchmark_variants.py`.

---

## 9. Verification checklist (every PR)

```text
[ ] Classification comment present (kernel + class A–D)
[ ] No new pure-Python O(N²) hot path without standard baseline next to it
[ ] CellIndex or CSR used for spatial neighbors when applicable
[ ] Far-field engine matches kernel (or explicitly none)
[ ] benchmark_variants (or domain bench) updated
[ ] Accuracy metric reported beside latency
[ ] tools/run_all.py: no new FAIL
[ ] tools/lint_claims.py PASS
[ ] tools/check_wgsl_sync.py PASS if WGSL/index.html touched
[ ] README/OVERVIEW/BENCHMARKS updated if public numbers change
[ ] GPU note: XLA_PYTHON_CLIENT_PREALLOCATE=false if running JAX near other GPU jobs
```

---

## 10. Known footguns

1. **`grid_res` vs `grid_depth`:** `CellIndex(grid_res=32)` is linear cells/side.
   Neural ops often use `grid_res = 1 << grid_depth`. Mixing silently breaks
   occupancy. Read `core/spatial_index.py` docstring (R7-F30).

2. **Funnel rebuild:** never delete keys; `CellIndex.build` recreates table.

3. **Demo adaptive gravity** uses different metadata path than fixed-grid;
   near-field List-1 ≠ uniform 3×3 counting sort.

4. **Class D denial:** if FMM is 10× slower at N=2k, that is expected in
   pure Python — prove crossover or compile.

5. **App9 recall collapse:** speeding LSH further without geometry change
   can destroy recall; fix dataset or accept Pareto frontier.

6. **Claims lint:** words like "lock-free GPU funnel FMM" need to be true
   of the specific file you edit.

7. **JAX GPU memory:** follow repo `AGENTS.md` —
   `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

---

## 11. Definition of done (repo-level)

The repository is **reference-grade for max performance** when:

1. Every app1–10 meets R1–R7.  
2. Every domain `benchmark_variants.py` shows either a ≥1.5× win or a
   scaling crossover section.  
3. `docs/INAPPLICABILITY.md` classes have **no module wrongly claiming FMM**.  
4. Browser demo: all scenarios use shared near-field hash; far-field model
   correct per scenario; 500k interactive on mid GPU; 5M documented.  
5. `tools/run_all.py` green; BENCHMARKS.md regenerated; README tables match.  
6. At least one non-Python backend (Zig or WebGPU) is the default hot path
   for one Class D app (flocking or galaxy).

---

## 12. Suggested implementation order for a single agent session

If you only have one session:

1. **P0.1–P0.3** baseline  
2. **P0.4** radial_taylor depth fix (blocks bad FMM tests)  
3. **P0.5** bio CSR + honesty  
4. **P1.1** app4 boids CellIndex + centroids  
5. **P1.5** flocking domain  
6. **P1.2** app3 FGT default  
7. **P2.1** app1 scaling win  
8. Update `BENCHMARKS.md` + `run_all`

If you have a browser session after that: **P3.1 5M remeasure**.

Do not start P5 CUDA rewrites before P1 near-field is canonical — otherwise
you multiply incomplete stacks.

### Single-session anti-patterns

- Do not "enable FMM" on app6/7/8/9 (Class A).  
- Do not spend the session on algorithm_theory renames before P0.4.  
- Do not claim GPU funnel-FMM on Triton direct kernels.

---

## 13. File index for the executor

| Need | Open first |
|------|------------|
| Funnel algorithm | `core/elastic_hash.py` |
| Spatial API | `core/spatial_index.py` |
| CSR P2P | `core/csr_p2p.py`, `core/_csr.py` |
| 2D log FMM | `core/fast_vectorized_fmm.py` |
| 3D Yukawa | `core/yukawa3d_fmm.py` |
| Gaussian FGT | `core/gaussian2d_fgt.py` |
| Bench harness | `core/benchmark_kit.py` |
| App patterns | `apps/app1_*.py`, `app4_*.py`, `app5_*.py` |
| Inapplicability | `docs/INAPPLICABILITY.md` |
| GPU/demo notes | `docs/GPU_NOTES.md` |
| Numbers | `BENCHMARKS.md` |
| Live demo axes | `index.html` (search `farFieldModel`, `selectHashMode`, `wantsCellListsThisFrame`) |
| Gate | `tools/run_all.py` |

---

## 14. Document maintenance

When you complete a P-task:

1. Check the box in §7 (edit this file).  
2. If numbers changed, update `BENCHMARKS.md`.  
3. If architecture changed, add a dated one-paragraph note under §15 below.

### §15 Change log

| Date | Note |
|------|------|
| 2026-08-21 | Initial deep-review instruction file created. Captures near/far axis split, full app/domain audit, playbook, and priority queue aimed at max-performance reference implementations. |
| 2026-08-21 | Merged parallel audit findings: P0.4 radial_taylor depth split-brain (R7-F30), P0.5 bio O(N·K) honesty+CSR, P0.6 flash far complexity, P2.4–P2.8 graphics/algorithm_theory/naming. |

---

*End of instructions. Prefer small verified diffs over large untested rewrites.*
