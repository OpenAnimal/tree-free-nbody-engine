# Promotion venues & snippets (2026-08 research)

Screenable catalog for showing the repo to the communities that would
actually care. Two detailed catalogs (rules verified against official pages
where fetchable; unverifiable rules are flagged rather than guessed):

- **[venues_neural_graph.md](venues_neural_graph.md)** — neural ops (JAX /
  attention / KV-cache / SSM), graph & algorithm theory, general launch
  (Show HN mechanics, YC clarification, Product Hunt, awesome-lists).
- **[venues_bio_hpc_env.md](venues_bio_hpc_env.md)** — bioinformatics
  (Biostars Tools, SEQanswers, OpenMM/GROMACS/MDAnalysis/scverse/Biopython),
  physics/HPC/MD/astro, environmental (hydrology, radiotherapy, batteries),
  gamedev/graphics/video.

Both include per-module-family → venue mapping tables, account
requirements, self-promotion policies, and avoid-lists.

---

## Snippets

### Concepts-first, forum/Discord-generic

> We've been exploring a simple combination of two algorithms:
>
> - the **Fast Multipole Method** (Greengard & Rokhlin, 1987; adaptive form:
>   Carrier, Greengard, & Rokhlin, 1988) — replaces O(N²) all-pairs
>   interaction sums with O(N) by clustering distant sources into multipole
>   expansions; and
> - the **optimal open-addressing hash table** of Farach-Colton, Krapivin, &
>   Kuszmaul (2025) — broke a 40-year-old conjecture by Yao (1985) and gives
>   open addressing with O(1)-type lookups and no reordering.
>
> The idea: in FMM-type algorithms, replace the octree/kd-tree entirely with
> that hash table over an implicit lattice — pointerless, no per-step tree
> allocation, warp-friendly, worst-case-bounded cell lookups. The repo applies
> that one trick in ~175 self-contained modules across 11 domains: N-body and
> SPH, spatial attention and GNN layers and KV-caches, protein electrostatics
> and constant-pH MD, PageRank, optimal transport, geodesics, groundwater
> transport, video codecs, gamedev broadphase, and more.
>
> It runs on CPU (NumPy / Zig SIMD), JAX (differentiable), CUDA/OpenCL, and
> WebGPU (live browser N-body demo). MIT. Where feasible the modules are
> cross-validated against direct summation, and each module's README states
> which state it is in.
>
> It is fairly new and will contain bugs — but we hope it is interesting as
> inspiration for accelerating your own algorithms; in the right regime the
> approach can gain orders of magnitude. If one of these domains is yours,
> we'd value a critical look at that module.

Notes: swap the last module list sentence per venue (the detailed catalogs
suggest per-family angles). Do not use the repo-internal term "elastic hash"
outside the repo — "pointerless spatial hash with worst-case bounds" carries.

### Show HN variant (shorter, artifact-led)

> Show HN: Tree-Free N-Body Engine — FMM without the tree
>
> We replaced the octree of the Fast Multipole Method (Greengard & Rokhlin,
> 1987; Carrier, Greengard, & Rokhlin, 1988) with the 2025 optimal
> open-addressing hash table (Farach-Colton, Krapivin, & Kuszmaul) over an
> implicit lattice: no pointers, no per-step tree allocation, bounded cell
> lookups. The same trick is applied to ~175 modules in 11 domains (spatial
> attention, protein electrostatics, PageRank, optimal transport, ...),
> on CPU/JAX/CUDA/WebGPU. Live WebGPU demo on the page (500k particles in
> the browser); NumPy/JAX code is cross-validated against direct summation,
> and each module says which state it's in. Fairly new, will contain bugs —
> looking for what breaks.

---

## Cross-domain shortlist (merged top picks)

Highest value, lowest risk, ready today (details + caveats in the catalogs):

1. **Biostars `Tool` post** — sanctioned tool-announcement category; the 30
   bioinformatics modules (user's top human-benefit priority).
2. **OpenMM GitHub Discussions "Show and tell" / GROMACS third-party tools /
   MDAnalysis Discord** — where open-source MD package authors actually read.
3. **AboutHydrology list** — OSS announcements are explicitly in the charter.
4. **JAX GitHub Discussions "Show and Tell"** — the official JAX showcase
   channel; differentiable FMM ops.
5. **SEQanswers Bioinformatics forum** — "open source efforts" written into
   the subforum description; k-mer/minimizer/pangenome/CRISPR modules.
6. **Ziggit Showcase** — explicit showcase category for the Zig data
   structures; open signup (no AI-generated text allowed).
7. **WebGPU Matrix + Hugging Face Space + forum post** — official community
   solicits demos; the live sim gets a permanent URL.
8. **dev.to deep-dive series + awesome-list PRs** — owned platform + durable
   one-line PRs (lowest risk of all).
9. **Show HN** — the single big shot; hold until the demo is on GitHub Pages
   rather than the rate-limited proxy, then stay in the comments with
   benchmarks. (HN *is* Y Combinator's forum — see the YC clarification in
   venues_neural_graph.md §C.)
10. **scicomp / Matter Modeling SE** — answer-first presence, not launch
    posts; software authors are welcomed when they disclose affiliation and
    bring numbers.

**Avoid** (details in catalogs): Tildes, r/programming (self-promo removed),
cstheory/OR/SO as launch posts, LinkedIn group cold posts, lobste.rs `show`
within the first 70 days of a new account (mechanically blocked),
paperswithcode.com (shut down July 2025), vote-solicitation everywhere, and
AI-generated copy on Ziggit/HN.

**Timing note**: two credible 2025 analyses disagree on HN timing (Sunday
~midnight PT vs weekday 8–11 AM ET) — treat as heuristics; the stronger
lever is a fast-loading demo and the author answering every comment.
