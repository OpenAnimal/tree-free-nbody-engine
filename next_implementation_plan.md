# Next Implementation Plan — Round 2 (for GLM-5.2 executor; review by GLM-5.3)

Working dir: repo root. All commands: `python -X utf8 ...` from repo root.
Rules: do NOT weaken any test/assertion; if an acceptance number moves,
STOP and report. Finish tasks in order.

## Round 1 status: DONE and verified (2026-08-18 review)

- Core suites: spatial_index 10/10, elastic_hash 10/10, cgr88 20/20.
- Folder suites: graphics_rendering ALL PASS, video_streaming ALL PASS.
- All demos green with expected validation numbers: fog-of-war 0/200
  mismatches; harmonic flow 3.3e-05 mean; gaussian color err 0.306;
  tetrahedral broadphase 1,728 cells / 5,832 pairs; flocking near-field
  exact (0 missing neighbors); neuromorphic 0/200 mismatches.
- `core/elastic_hash.py` rewritten to the true funnel-hash schedule (FCK
  2025) + batched variant; `tools/lint_claims.py` exits 0; rename aliases
  `VolumetricMonopoleAO` / `ProceduralRBFMapGenerator` in place and importable.
- Variant benchmarks exist: `graphics_rendering/benchmark_variants.py`
  (standard 116ms / +elastichash 8.3e-04 / +quantized 2.0e-02) and
  `game_mechanics_spatial/benchmark_variants.py` (honest note: not faster
  than O(N²) at N=1000 — kept as-is, do not "fix").
- Apps migrated to FastVectorizedFMM + honest wording; assets regenerated.

## Round 2 tasks

### 2.1 Commit checkpoint (do FIRST, before any edits)
The whole round-1 work is uncommitted. Run:
```
git add -A && git commit -m "Unify spatial indexing on funnel hash + CellIndex; honest validations; variant benchmarks"
```
If `git add -A` picks up junk (__pycache__, *.pyc), add a `.gitignore` for
those first. Do not push.

### 2.2 Variant benchmark for the FMM core itself
Create `core/benchmark_variants.py` using `VariantBenchmark`:
- `standard`: `exact_direct_nbody_2d` (N=2000, clustered distribution like
  the one in test_flat_fmm_elastic_hash_occupancy in
  core/test_cgr88_cross_validation.py).
- `+fmm (CGR88 adaptive)`: `TreeFreeElasticAdaptiveFMM(p=10)` on the same
  inputs, accuracy_vs="standard".
- `+fmm (flat vectorized)`: `FastVectorizedFMM(depth=5, order=8)`,
  accuracy_vs="standard".
- `+quantized`: inspect `quantized_bitpacked_optimization/` for a packed FMM
  with a compatible evaluate API; adapt the CALLER inputs only — do NOT
  modify the quantized module. accuracy_vs="standard"; the note must state
  the known ~12% rel-L2 packed cost. If no compatible entry point exists,
  skip the row and say why in a comment.
Acceptance: table prints; adaptive FMM rel-L2 < 1e-6; if FMM is NOT the
fastest, print that honestly in the note — do not tune to win.

### 2.3 Variant benchmark for physics domain
Create `physics_simulation/ppf_contact_solver_fmm/benchmark_variants.py`:
- `standard`: brute-force O(N²) tet-tet AABB overlap count on the demo mesh.
- `+elastichash`: broadphase pair set via the existing CellIndex ring-1 code.
Accuracy semantics: broadphase is a filter — compare "every brute-force
colliding pair appears in the broadphase set" and print
`no missed collisions: True` in the note. Pair sets need NOT be equal.
Acceptance: table prints with `no missed collisions: True`.

### 2.4 Variant benchmark for video streaming domain
Create `video_streaming_codecs/benchmark_variants.py`:
- `standard`: exact per-pixel Gaussian splat of one frame (reuse
  volumetric_gaussian_stream's exact path).
- `+elastichash`: cell-bucketed splat (existing compress path).
- `+quantized`: existing quantized color path.
accuracy_vs standard on the reconstructed image; the known lossy color
quantization (~0.31 rel L2) must show up in the table, not be hidden.
Acceptance: table prints, three rows.

### 2.5 Aggregate numbers page
Create `BENCHMARKS.md` at repo root: run each of the five
`benchmark_variants.py` files, paste each table verbatim under a heading,
and add ONE prose sentence per table stating the honest takeaway (including
any "not faster at this scale" result). No editorializing.
Acceptance: file exists with 5 tables.

### 2.6 Re-run verification matrix (all must pass)
```
python -X utf8 -m core.test_spatial_index
python -X utf8 -m core.test_elastic_hash
python -X utf8 -m core.test_cgr88_cross_validation
python -X utf8 graphics_rendering/test_graphics_rendering.py
python -X utf8 video_streaming_codecs/test_video_streaming.py
python -X utf8 tools/lint_claims.py
python -X utf8 core/benchmark_variants.py
python -X utf8 physics_simulation/ppf_contact_solver_fmm/benchmark_variants.py
python -X utf8 video_streaming_codecs/benchmark_variants.py
python -X utf8 graphics_rendering/benchmark_variants.py
python -X utf8 game_mechanics_spatial/benchmark_variants.py
```
Then `git add -A && git commit -m "Add variant benchmarks for core/physics/video; BENCHMARKS.md"`.

## Known pitfalls (unchanged from round 1)

1. Unit mode (`grid_res=`) for [0,1) positions, world mode (`cell_size=`)
   for world units; mixing collapses everything to one cell.
2. `key_ints` returns (x, y) — arrays index `[y, x]`.
3. Never let a benchmark note claim a speedup the table doesn't show.
4. Broadphase "accuracy" = no missed collisions, not pair-set equality.
5. Rebuild CellIndex/elastic hash on every update (append-only, stale keys
   are never forgotten).
