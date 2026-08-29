# Round-18 Findings — Professionalism / Presentability Review

Date: 2026-08-26. Scope: is the repo professional and presentable to
outside scientists? Three parallel reviews (docs; demo UI copy; repo
hygiene + test suite) over README.md, BENCHMARKS.md, docs/GPU_NOTES.md,
docs/INAPPLICABILITY.md, OVERVIEW.md, index.html, validate.html(+template).
Fixes applied the same day; first 10M stress measurements added (§14.1 of
GPU_NOTES.md). Verification at the bottom.

## Blockers (all fixed)

1. **UI: the round-16/17 honesty tooltips never displayed.**
   `updateFmmControlsState()` (index.html) overwrote the rich static
   tooltips of `#selectFmmMode` / `#selectNfBudget` / `#selectHashMode` /
   `#selectFmmOrder` with short generic text at startup and on every state
   change. Fixed: tooltips stashed once (`dataset.longTitle`); the enabled
   state always shows the full authored text, only the disabled state
   swaps in the sub-control note.
2. **validate.html misreported its own near-field tier.** The footer said
   the demo default tier is "48/48/32" while the rig (n ≤ 32k) actually
   runs the auto tier's FULL traversal — and the `adaptive-fullNF`
   attribution control duplicated the adaptive row at that n. Fixed:
   footer reports the true tier via `p2pBudgetAutoForN(n)`; the
   attribution control now runs only when the auto tier is a sampled one
   and prints an explanatory skip-note otherwise (`?p2pbudget=48`
   measures the sampler's cost at small n).
3. **README variant table contradicted BENCHMARKS.md ~18×.** The
   "representative slice" was from an old run (adaptive 1057 ms "NOT
   faster" vs the current canonical 29.59 ms "1.2× faster"; stale
   broadphase/App-1 rows). Re-pasted from the current run.
4. **README over-claim**: "completely replaces sorting and tree
   construction" while the demo's adaptive mode rebuilds a quadtree on
   CPU every ~96 frames. Rewritten to the honest claim + new
   "What 'tree-free' means (and what it does not)" section (fixed lattice
   = no hierarchy build; adaptive = logical quadtree materialized into
   flat buffers + hash directories; near-field hash materialization is
   why backends measure equal FPS).

## Major (all fixed)

- Eight citation-format violations in the README reference list
  (missing "&" before last author — AGENTS.md hard rule); also the
  nested-paren year in GPU_NOTES §13.6 and Warren & Salmon / Dongarra &
  Sullivan added as proper references.
- `?p2pbudget=` URL override silently clamped at 256 while the sidebar
  select reaches 4096 (512/1024/4096 quietly yielded 256). Clamped at
  4096 now; the console suggestion cap raised likewise.
- Dangling paths after the `benchmarks/` → `core/` move
  (BENCHMARKS.md §hash reproduce command, GPU_NOTES §11) and after
  `core/test_amd_radeon_compliance.py` → `tests/core/` (OVERVIEW.md ×2).
- BENCHMARKS.md leftover second-person dialogue ("Your observation
  that ... is correct") → neutral phrasing.
- Stale numbers presented as current: GPU_NOTES §2 (5.99×→8.0×, updated),
  §5.4 (pre-raise 48/32/16 tier ladder → current full/48/32), §11
  recommendation contradicted by the actual directory-ON default
  (rewritten as historical note), INAPPLICABILITY Class D (~711 ms flat
  FMM → post-FFT-rewrite ~57 ms).
- README conversational asides ("poor as a church mouse") → factual
  support notice. (The "ask your favourite AI" line was KEPT — owner's
  voice; flag for the owner to reconsider.)
- Insider codes on the front door: "Round-17", "T-C2", "T-D4" removed
  from README (GPU_NOTES keeps round labels — it now has a round index).
- BENCHMARKS.md adaptive-at-500k takeaway marked superseded (2026-08-26)
  with pointer to GPU_NOTES §12.7/§13.3.
- §12.7 vs §13.2/§13.4 same-day 500k median discrepancy: reconciliation
  note added (tree-phase node swings 46–87; which table answers which
  question).
- Mode-name drift unified across UI + validate.html: "Fixed-Grid" →
  "Uniform-Lattice", "Analytic (no FMM)" → "Analytic cores only (no
  P2P)", validate rig retitled "Direct All-Pairs vs Uniform-Lattice FMM
  vs Adaptive FMM vs None"; "FCK" initialism spelled out; "5M Extreme
  Mode" telemetry renamed to match the "5M Reference" preset; O(N²)
  notation unified; Direct second-warning dialog tone fixed; AFMM Meta
  row given explicit units; meta descriptions added to both pages.

## Hygiene (at review time; .gitignore fixed, rest left to owner)

- `pytest tests/core`: 108 passed / 3 skipped (WGSL numeric tests need
  node/wgpu adapter), ~5 min. `check_wgsl_sync.py`: PASS. All README
  links/images resolve. `validate.html` matches its build template
  byte-for-byte.
- Untracked clutter (26 bench/probe artifacts, PNGs, the index.backup
  HTML) is now covered by `.gitignore` (kept on disk — docs reference
  them as local artifacts; GPU_NOTES marks the backup "local,
  untracked"). Harness .js files and validate rig remain intended for a
  future commit.
- CITATION.cff vs README bibtex title mismatch ("Spatial Field" vs
  "Spatial Computing") — bibtex aligned to the CFF.

## Deliberately NOT changed

- `core/benchmark_variants.py`'s "single-level vectorized adaptive FMM"
  row note (would require re-running the benchmark to keep the pasted
  tables "verbatim").
- `tests/core/test_jax_pipeline.py` has 6 bool-returning tests
  (PytestReturnNotNoneWarning — cosmetic; convert to asserts when next
  touched).
- Staging the `tools/review_round10/` deletions / committing the
  `benchmarks/`→`core/` move — owner's call; both are consistent with
  HEAD references now that docs point at `core/`.

## Verification

`tools/_build_validate.py` re-splice (after template edits) → clean;
`python tools/check_wgsl_sync.py` → PASS; `node --check` on both pages'
inline scripts → OK; `node tools/smoke_validator.js 8000 120` → all rows
PASS (adaptive 0.84% dvRelL2 vs direct, adaptive-vs-fixed 0.74% —
unchanged from round 17, i.e. the copy edits touched no physics);
`tools/bench_fps_longrun.js` 10M fixed/adaptive 60 s runs → GPU_NOTES
§14.1.
