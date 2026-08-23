# Project Knowledge & Guidelines

** Our aim is to create a professional, highly cross-validated codebase with minimal, self-contained, drop-in examples that provides magnitudes of performance gains and actual usefulness for science and humanity. 


## Test Layout

- All real tests live under `tests/<package>/` (mirrors the source package
  layout: `tests/core/`, `tests/neural_ops/`, `tests/bioinformatics/`, ...).
- Each `tests/<package>/` dir has an `__init__.py` so pytest's prepend import
  mode inserts the repo root into `sys.path` (lets tests do
  `from core.x import y` without each file needing its own path hack).
- A root `conftest.py` also inserts the repo root for standalone execution.
- `pyproject.toml` has `[tool.pytest.ini_options]` with `testpaths = ["tests"]`,
  so `python -m pytest` from the repo root collects the whole suite.
- Standalone: `python -m tests.core.test_elastic_hash` (run from repo root).
- `tools/browser_crossbench.js` is the Node/playwright browser cross-bench
  (needs `npm install` for playwright), not part of the pytest suite.

## Citation Conventions

Author-year citations follow one consistent standard across the whole repo
(docstrings, READMEs, code comments, plot titles, kernel headers):

- **Multi-author list**: comma-separated surnames with `&` before the last
  author. Never join distinct authors with a hyphen or slash.
  - 2 authors: `Greengard & Rokhlin (1987)`
  - 3 authors: `Carrier, Greengard, & Rokhlin (1988)`
  - 3 authors (flagship): `Farach-Colton, Krapivin, & Kuszmaul (2025)`
- **Year placement** (APA-style, avoids nested parens):
  - Narrative / standalone / reference list: year in parens —
    `Farach-Colton, Krapivin, & Kuszmaul (2025)`.
  - Inside an outer parenthetical that also holds other content: year by
    comma, no inner parens —
    `(Farach-Colton, Krapivin, & Kuszmaul, 2025, Section 3)`.
- **No `et al.`** for the flagship Farach-Colton citation — always write the
  full three-author list.
- **Hyphenated single surnames are kept** (e.g. `Farach-Colton` is one person,
  Martin Farach-Colton — the hyphen is correct and must not be split into
  `Farach & Colton`).
- **Eponym shorthand** (the technique referred to by the first author's
  surname alone, with no year and no co-author list, e.g.
  `Farach-Colton hash table`, `Barnes-Hut tree code`, `Biot-Savart law`,
  `Kelvin-Helmholtz instability`, `Debye-Hückel equation`) is left as-is.
  These are proper-name labels, not citations.

## JAX / GPU Execution Rules
- **Memory Allocation**: When running interactive checks, tests, or test compilations, **never** allow JAX to preallocate the default 75%-90% GPU VRAM. Always set:
  ```bash
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  # or
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
  ```
  to prevent OOM collisions with active background training sessions.
