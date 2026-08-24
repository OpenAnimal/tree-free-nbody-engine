# Promotion Venues — Neural Ops (A), Graph/Algorithms (B), General Launch (C)

**Scope:** promotion venues for the tree-free-nbody-engine, covering the neural-ops
modules (25), the algorithm_theory modules (35), and general launch channels.
Sibling docs cover other domains; this file owns A + B + C only.

**Researched:** 2026-08-24, via live web search and fetches of official rule pages
where reachable. Reddit rule pages are behind a login wall for crawlers — Reddit
entries below are verified from search-indexed rule text and are flagged
**[re-check sidebar before posting]**. Confidence labels:
- **[V]** = read from the venue's own pages this session
- **[V-3p]** = verified via third-party analyses of the venue (rule text quoted in indexed sources)
- **[U]** = unverified / ambiguous — stated as such, not invented

The load-bearing technical hook for every post: the optimal open-addressing hash
table of Farach-Colton, Krapivin, & Kuszmaul (2025, arXiv:2501.02305) replaces
pointer octrees/kd-trees, giving O(1) worst-case lookups over an implicit lattice.

---

## A) Neural-ops venues

### A1. JAX GitHub Discussions — "Show and Tell" — Fit: **High**
- URL: https://github.com/jax-ml/jax/discussions/categories/show-and-tell · type: official
  forum (GitHub Discussions) on the JAX repo
- Submission format: open a Discussion in the "Show and Tell" category; title + body +
  repo/demo links; code blocks and images supported.
- Self-promotion policy: **[V]** the JAX team stated they are "mostly trying to foster
  GitHub discussions as the main place for community discussion"
  (https://github.com/jax-ml/jax/discussions/14802), and the Show and Tell category
  exists exactly for showcasing user projects. This is the rare venue where your
  project showcase *is* the intended content.
- Account requirements: any GitHub account. No karma/age gates observed.
- Modules: every JAX backend module — flash_multipole_kernel, multipole_attention,
  autograd_adjoint_fmm (custom VJP), multipole_mamba_ssm, spectral_neural_pme,
  diffusion_policy_fmm, multipole_gaussian_process, equivariant_transformer /
  equivariant_field_layer, hyperbolic/spherical_multipole_attention,
  elastic_kv_cache + hierarchical_elastic_kv_cache, visual_transformer_ops,
  multipole_flow_drift, taylor_fgt_attention, continuous_meshfree_gnn,
  neural_sph_ipc, kernel_independent_fmm.
- Angle: "Show and Tell: pointerless multipole attention in pure JAX — O(N) attention
  without a tree, via Farach-Colton/Krapivin/Kuszmaul hashing."

### A2. r/MachineLearning ([P] posts) — Fit: **High** (moderation risk without benchmarks)
- URL: https://www.reddit.com/r/MachineLearning/ · type: subreddit (~research ML)
- Submission format: title must carry a bracket flair, e.g. **[P]** (project),
  [D] (discussion), [R] (research), [Q] (question); text or link post.
- Self-promotion policy: **[V-3p] "r/MachineLearning has held a long-standing, strict
  policy on spam. The posts deemed to be spam will be removed, and repeat offenders
  will be permanently banned."** Accepted [P] posts are substantive: methodology,
  benchmarks, honest scope. A bare repo link reads as promo; a benchmark table +
  "what's validated vs experimental" framing fits the sub's norms. **[re-check
  sidebar before posting]**
- Account requirements: subreddit enforces posting karma/account-age gates via
  AutoMod; exact thresholds unpublished. **[U]** Build karma first.
- Modules: the whole A list; strongest for flash_multipole_kernel (attention),
  multipole_mamba_ssm, diffusion_policy_fmm, equivariant_transformer,
  elastic_kv_cache, multipole_gaussian_process.
- Angle: "[P] We replaced octree pointer-chasing in multipole attention with an
  optimal hash table (FK&K 2025) — JAX kernels + scaling plots, validated vs direct
  summation."

### A3. Hugging Face — Spaces + community forum — Fit: **High**
- URLs: https://huggingface.co/spaces · https://discuss.huggingface.co/ · type: hosted
  demo platform + Discourse forum
- Submission format: deploy an interactive demo (Gradio/Streamlit/static) as a Space;
  announce it in the forum category for sharing and discussing projects
  ("Spaces, Models, Datasets ... getting feedback from the community") **[V-3p]**;
  a "Share your projects!" thread also exists.
- Self-promotion policy: **[V-3p]** project-sharing is an explicit, on-topic forum
  category — showing your Space is the purpose, not spam.
- Account requirements: free HF account; Spaces free tier (ZeroGPU quota) exists.
- Modules: any module with a visual/interactive face — the n-body/WebGPU live demo,
  diffusion_policy_fmm, visual_transformer_ops, neural_sph_ipc,
  continuous_meshfree_gnn, multipole_mamba_ssm.
- Angle: host the 5M-particle live sim as a Space; forum post: "Pointerless FMM demo
  Space — ask me anything about the hash-table octree replacement."
- Note (paper path): https://huggingface.co/papers/submit allows only **authors of
  papers < 7 days old (arXiv)** to submit to Daily Papers **[V]**. Not usable until
  you actually post an arXiv paper; listed here so nobody wastes time on it.

### A4. GPU MODE Discord — Fit: **High**
- URL: https://discord.gg/gpumode · type: Discord community (GPU programming;
  formerly CUDA MODE; lectures, hackathons, kernel leaderboard)
- Submission format: post in the demos/showcase channel ("to showcase cool projects
  and demos" **[V-3p]**); async chat norms, GIFs/benchmarks welcome.
- Self-promotion policy: **[U]** channel rules are only visible inside the server;
  observed norm = concrete kernels + numbers, not marketing. Join and read pins first.
- Account requirements: open Discord invite; no gates observed.
- Modules: flash_multipole_kernel (SRAM tiling), quantum_fock_exchange_fmm (CFMM
  Coulomb kernels), oscillatory_butterfly_kernel, non_uniform_fourier_hash (NUFFT
  kernels), spectral_neural_pme, screened_yukawa_fmm, capacitance_boundary_bem —
  anything with a GPU kernel and a speedup plot.
- Angle: "Hash-table-indexed multipole kernel: no tree build, no pointer chasing —
  SRAM-tiled attention/N-body kernel with numbers vs cuBLAS/Barnes-Hut."

### A5. EleutherAI Discord — Fit: **Med**
- URL: https://www.eleuther.ai/community · type: research Discord
- Submission format: hang out in research/paper channels; share work in-channel
  where on-topic. **[V-3p]** Norms: open discussion of papers and implementations;
  drive-by link drops without participation go badly.
- Account requirements: open join.
- Modules: multipole_attention theory, hyperbolic/spherical_multipole_attention,
  koopman_spectral_operator, spectral_neural_pme.
- Angle: discussion-first: "does O(1) worst-case lattice hashing break the
  tree-bottleneck in long-range attention? here's our kernel" in a relevant channel.

### A6. X / Twitter (ML communities + hashtags) — Fit: **Med**
- URL: https://x.com/i/communities/1509381007950204928 (Machine Learning community,
  ~66k members) and similar · type: social network + communities
- Submission format: thread with figures/GIFs; peer-reviewed evidence that images
  strongly boost engagement for research posts **[V-3p]**. Hashtags (#MachineLearning,
  #DeepLearning, #jax) + community posts.
- Self-promotion policy: none enforced; norms favor arXiv links, figures, benchmark
  charts, threads rather than bare links.
- Account requirements: any account; reach is algorithmic.
- Modules: whole A list; best: flash_multipole_kernel, elastic_kv_cache,
  diffusion_policy_fmm, multipole_mamba_ssm.
- Angle: "Octrees are pointers. We replaced them with an optimal hash table. 30-sec
  video of 5M particles + one benchmark chart + repo link."

### A7. dev.to — Fit: **High** (as long-form write-up, not announcement)
- URL: https://dev.to/ · type: blogging platform with moderated tags
- Submission format: Markdown article; tags (#machinelearning, #jax, #algorithms);
  canonical URL field exists for cross-posting your own content **[V-3p]**.
- Self-promotion policy: **[V-3p]** your own technical articles are the site's
  purpose; tag moderators remove off-tag/low-substance posts; official guidelines
  for AI-assisted articles exist — disclose AI assistance. Listicle-style
  "175 modules!!" posts are explicitly disliked community-wide.
- Account requirements: free account, no gates.
- Modules: any; best as per-family deep dives (one article per family: attention,
  KV-cache, GP, flow).
- Angle: "How we killed the octree: implementing Farach-Colton/Krapivin/Kuszmaul
  hashing for multipole attention (with JAX code)."

### A8. Papers With Code — status 2026 — Fit: **n/a (dead) / Med (revival)**
- paperswithcode.com was shut down by Meta (July 2025); the .com domain now
  redirects to Hugging Face Trending Papers **[V-3p]**. A community revival by
  HF engineer Niels Rogge is live at **paperswithcode.co** ("for the agentic era,"
  2500+ leaderboards) **[V-3p]**; alternatives (SOTAPapers, etc.) exist.
- Action: do **not** plan around classic PWC. If you write the arXiv paper, it will
  surface via HF Trending Papers automatically; consider adding results to the
  paperswithcode.co revival once its submission process stabilizes. **[U] on revival
  submission mechanics.**

### A9. PyTorch Forums — Fit: **Low**
- URL: https://discuss.pytorch.org/ · type: Discourse support/research forum
- Submission format: forum topic in a relevant category. No dedicated showcase
  category was verifiable; /guidelines is the pinned FAQ **[V-3p]**.
- Self-promotion policy: no explicit show-and-tell; governed by the PyTorch
  Foundation Code of Conduct; project posts appear occasionally in topical threads.
- Account requirements: free account.
- Modules: autograd_adjoint_fmm (custom autograd/VJP) is the only natural hook —
  a *question* about custom VJPs where the repo is the answer.
- Angle: question-first ("has anyone built tree-free hierarchical ops with custom
  VJPs?") — not a launch post.

### A10. Ziggit Showcase — Fit: **High** (Zig backends)
- URL: https://ziggit.dev/categories · type: official Zig forum (Discourse)
- Submission format: new topic in the **Showcase** category: "Post your latest Zig
  project and let the community know what you've been working on." **[V]**
- Self-promotion policy: **[V]** Showcase exists exactly for this; explicit mod
  rules: "Don't hijack an existing thread to show off your own project. Instead,
  make a new post under Showcase"; Showcase is "more for presenting concrete
  results." **[V]** Zero AI-generated text allowed on the forum itself (linking to
  AI-assisted projects is fine).
- Account requirements: open signup (no invite/karma gates observed). **[V]**
- Modules: the Zig/CPU-native backends — hash-table core, spatial_disjoint_set,
  multipole_range_tree, elastic_quotient_filter, sublinear_fast_dtw, n-body
  solvers, video codecs.
- Angle: "Showcase: tree-free FMM in pure Zig — octree replaced by an implicit
  lattice + optimal open addressing, benchmarks vs kd-tree baseline."

---

## B) Graph / algorithm-theory venues

### B1. Hacker News (general submissions + Show HN) — Fit: **High**
- URL: https://news.ycombinator.com · type: forum (Y Combinator's). Full mechanics
  under C1; here: the algorithm-theory angle.
- Submission format: link post (title = original article title, no editorializing);
  or Show HN with the live demo.
- Self-promotion policy: **[V]** see C1. For B, the low-risk route is a third party
  or a non-promotional technical essay ("Why Farach-Colton/Krapivin/Kuszmaul hashing
  ends the pointer era in FMM") that happens to link the repo.
- Modules: the hash-table core itself + spatial_disjoint_set, elastic_quotient_filter,
  sublinear_distance_oracle (Thorup-Zwick), optimal_transport_fmm,
  network_power_centrality, spectral_graph_sparsifier (Spielman-Srivastava).
- Angle: Show HN with the WebGPU lattice demo; the FK&K 2025 hook alone is
  front-page bait for this audience.

### B2. r/algorithms — Fit: **Med**
- URL: https://www.reddit.com/r/algorithms/ · type: subreddit
- Submission format: text or link post; no formal flair system; sidebar says it is
  "not for homework advice" **[V-3p]**; mods themselves say rules/sidebar are
  outdated and under revision. **[re-check sidebar before posting]**
- Self-promotion policy: no explicit, current written self-promo rule found **[U]**;
  Reddit-wide spam norms apply. A bare repo drop will likely be removed; a
  technical discussion of the *algorithmic claim* (worst-case O(1) lookups;
  sublinear oracles) with code as evidence is the accepted pattern.
- Account requirements: standard AutoMod gates, thresholds unpublished. **[U]**
- Modules: sublinear_distance_oracle, sublinear_edit_distance, sublinear_fast_dtw,
  spatial_disjoint_set, elastic_quotient_filter, multipole_range_tree,
  non_uniform_fourier_hash, personalized_pagerank_fmm, optimal_transport_fmm.
- Angle: "Worst-case O(1) spatial lookups via FK&K hashing — critique my
  implementation of a tree-free distance oracle (Thorup-Zwick without the tree)."

### B3. r/compsci — Fit: **Med**
- URL: https://www.reddit.com/r/compsci/ (~2.5M members) · type: subreddit
- Submission format: link or text; wiki "Submission Guidelines" exists
  (reddit.com/r/compsci/wiki/index) **[V-3p]**. **[re-check sidebar before posting]**
- Self-promotion policy: no explicit ban surfaced in this session **[U]**; community
  guidance is "keep self-promotion to a minimum"; homework/career posts off-topic.
- Account requirements: standard gates, unpublished. **[U]**
- Modules: same theory-facing set as B2, plus algorithm_theory cross-cutting posts
  (spectral_graph_sparsifier, koopman_spectral_operator,
  oscillatory_butterfly_kernel).
- Angle: text post on "the 2025 optimal hash table is quietly a spatial index —
  what else can we de-pointerize?" with the repo as the worked example.

### B4. cstheory Stack Exchange — Fit: **Low** (question-only; no cold promo)
- URL: https://cstheory.stackexchange.com/ · type: research-level Q&A
- Submission format: must be a genuine research-level question; answers, not
  announcements.
- Self-promotion policy: **[V]** Stack Exchange "Expected Behavior" (cstheory help
  center): "Avoid overt self-promotion. The community tends to vote down overt
  self-promotion and flag it as spam" — network-wide rule (meta.stackexchange.com
  q/57497): only mention your own work when it genuinely answers, disclose
  affiliation, and keep it a minority of contributions.
- Account requirements: open signup; question bans apply to low-quality askers.
- Modules (only as question fodder): the FK&K table's behavior under lattice access
  patterns, sublinear_distance_oracle, functional_sobol_anova,
  kernel_causal_discovery, opinion_dynamics_fmm, spatial_voting_equilibrium.
- Angle: a real question, e.g. "Amortized vs worst-case behavior of FK&K open
  addressing when the key universe is an adaptive octant lattice — any known
  analysis?" — repo link only if it's genuinely the context.
- **Cold tool promotion is NOT allowed here.**

### B5. lobste.rs — Fit: **Med** (mechanically gated)
- URL: https://lobste.rs/ · type: invite-only computing link-aggregator forum
- Submission format: link story with tags; `show` tag exists for one's own work.
- Self-promotion policy: **[V]** (lobste.rs/about): fine to participate as an
  author, not to treat the site "as a write-only tool for product announcements or
  driving traffic"; rule of thumb: "self-promo should be less than a quarter of
  one's stories and comments."
- Account requirements: **[V]** invitation-only ("The quickest way to receive an
  invitation is to talk to someone you recognize from the site"; chat is the other
  path; the old invitation queue is gone). **New users' first 70 days: cannot
  submit unseen domains, cannot flag, and cannot post to a tag list that includes
  `show`, `announce`, `ask`, `meta`, and others.** So you cannot self-post to `show`
  until day 71 — or an established user submits your work (authors of posted links
  are explicitly welcomed into the thread).
- Modules: hash-table core, elastic_quotient_filter, spatial_disjoint_set,
  sublinear_* family, NUFFT, koopman_spectral_operator — very much the lobste.rs
  palate.
- Angle: someone else submits the technical essay/repo under `show`; author joins
  comments for a deep technical exchange.

### B6. dev.to (#algorithms, #computerscience tags) — see A7 — Fit: **High**
- Same platform; one algorithm-theory deep-dive per family works better than one
  mega-post. Modules: entire B list via 5-6 articles (OT, spectral graph, sublinear,
  fractional operators, spatial data structures, Koopman/assimilation).

### B7. Ziggit Showcase — see A10 — Fit: **High**
- Native-Zig data-structure modules (spatial_disjoint_set, elastic_quotient_filter,
  multipole_range_tree, non_uniform_fourier_hash, sublinear_fast_dtw,
  spatial_point_cloud_compression) map 1:1 to "concrete results" Showcase posts.

### B8. OR Stack Exchange / Stack Overflow (tags) — Fit: **Low**
- URLs: or.stackexchange.com · stackoverflow.com (tags: `optimal-transport`,
  `nufft`, `disjoint-sets`) · type: Q&A
- Self-promotion policy: **[V]** same Stack Exchange rule as B4 — answers may cite
  your own library only if it genuinely answers, with disclosure.
- Modules: optimal_transport_fmm, co_optimal_transport (Gromov-Wasserstein),
  localized_ensemble_kalman_fmm, spatial_disjoint_set.
- Angle: answer real questions where the module is legitimately the best answer;
  zero-launch strategy, durable SEO.
- **Cold promotion NOT allowed.**

### B9. awesome-lists via pull request — Fit: **High** (durable, low-risk)
- Examples: awesome-webgpu (confirmed active **[V-3p]**), awesome-jax, awesome-gnn,
  awesome-optimal-transport, awesome-scientific-computing · type: GitHub curated lists
- Submission format: PR adding one line (name — one-line description); each list's
  CONTRIBUTING governs; lists like Zig's require you to have a Showcase thread
  first **[V]** (Zig awesome lists only include projects with their own Ziggit
  showcase topic).
- Self-promotion policy: PRs are the intended mechanism; quality bars vary.
- Account requirements: GitHub account.
- Modules: all; map each family to its matching list.
- Angle: 10 small PRs beat one big launch; permanent discoverability.

### B10. r/webgpu — Fit: **Med-High** (not rule-verified)
- URL: https://www.reddit.com/r/webgpu/ · type: subreddit. **[U]** no rules page
  verified this session; observed content = WebGPU project shares and discussion.
  Re-check sidebar. Companion to C9 (WebGPU Matrix).

---

## C) General launch venues

### C1. Hacker News "Show HN" — Fit: **High** (single best shot)
- URL: https://news.ycombinator.com/showhn.html · type: forum (HN is Y Combinator's
  platform)
- Submission format: **[V]** title begins "Show HN" (convention: `Show HN: Name –
  one-liner`); submit the URL of something runnable. "Show HN is for something
  you've made that other people can play with." Posts land on /shownew and
  graduate to /show after clearing a points threshold.
- Self-promotion policy: **[V]** self-showcase *is* the purpose, with conditions:
  "The project should be non-trivial"; "must be something you've worked on
  personally and which you're around to discuss"; off-topic: "blog posts, sign-up
  pages, newsletters, lists, and other reading material"; make it tryable
  "ideally without barriers such as signups or emails"; version bumps "generally
  aren't substantive enough"; and "Please don't ask friends to upvote or comment."
- Account requirements: **[V-3p]** no official karma minimum; users report ~no
  gate for submitting, but brand-new accounts posting video links/Show HN have been
  auto-killed by anti-spam filters (news.ycombinator.com/item?id=36844201).
  Recommended: account with some comment history first.
- Timing: two data-backed schools **[V-3p]** — (a) 23k-post analysis (June 2025):
  best odds Sunday ~midnight PT (low competition); (b) counter-analysis: weekdays
  8–11 AM ET (max audience). For a visual sim demo, weekday morning ET is the
  safer default; both are heuristics, not rules.
- What top commenters demand for sim/visual demos (community expectation, not a
  written rule **[U]**): instantly runnable demo, benchmark methodology vs named
  baselines (Barnes-Hut, direct O(N^2)), honesty about validation status (the
  repo's validated-vs-experimental split is an asset here), no marketing language,
  and the author replying substantively for hours. Historically the audience is
  hostile to hype and to undisclosed AI-generated text.
- Modules: everything; the WebGPU 5M-particle live demo is the asset.
- Angle: "Show HN: Tree-Free N-Body Engine – octree-free O(N) FMM via optimal
  hashing, live WebGPU demo."

### C2. Other YC-affiliated channels — Fit: **Low** (mostly not showcase venues)
- **YC Library** (library.ycombinator.com): curated hub of YC essays/talks **[V-3p]**;
  not submission-based. Not a venue.
- **Startup School** (startupschool.org): free course + YC Co-Founder Matching **[V-3p]**;
  the old Startup School forum was effectively superseded by Co-Founder Matching; not
  a project-showcase venue. Conference edition ran July 2026.
- **Launch YC** (ycombinator.com/launches): for YC-funded batch companies **[U — not
  verified this session; do not assume it accepts outside projects]**.
- Bottom line: **HN is the YC showcase channel.** No other YC property accepts
  community project showcases as of this research.

### C3. r/programming — Fit: **Low** (self-promo removed)
- URL: https://www.reddit.com/r/programming/ · type: subreddit (~6.9M)
- Submission format: link submissions (no self/text posts — widely known; treat as
  **[U]** and verify on submit).
- Self-promotion policy: **[V-3p]** rule text: "No blogspam. Don't post actual
  blogspam: blogposts that do nothing more than linking to a primary source" —
  and checker-verified: direct self-promotion is removed. Submitting your own repo
  yourself = removal risk. **[re-check sidebar before posting]**
- Account requirements: minimum karma enforced via AutoMod, thresholds unpublished
  **[V-3p]**.
- What IS allowed: a substantive technical article (not a link-wrapper) *submitted
  by someone else*, or major-version news once the project has an audience.
- Modules: none directly; the FK&K-hashing explainer article is the only viable
  artifact here, ideally third-party submitted.
- **Cold tool promotion is NOT allowed.**

### C4. r/MachineLearning — see A2 (also the flagship C venue for the whole repo).

### C5. Product Hunt — Fit: **Med**
- URL: https://www.producthunt.com · type: product-launch ranking site
- Submission format: product page with tagline, gallery, maker comment; launches go
  live 12:01 AM PT; can schedule up to ~1 month ahead; Tue-Thu recommended **[V-3p]**.
- Self-promotion policy: launching your own product is the purpose; OSS explicitly
  welcome (dedicated Open Source topic; "best open-source products launched on
  Product Hunt in 2026" discussions) **[V-3p]**.
- Account requirements: completed maker profile; aged profiles recommended by
  guides; "hunter" is now optional (self-launch supported) **[V-3p]**.
- Modules: the live demo framed as a product ("Pointerless spatial-computing engine
  — run 5M particles in your browser").
- Caveat: PH audience is product/indie-hacker flavored; research code without UX
  underperforms. Medium value, medium effort.

### C6. lobste.rs — see B5 (also a general-launch channel, gated by invite + 70 days).

### C7. Lemmy (programming communities) — Fit: **Low**
- URLs: lemmy.world/c/programming, lemm.ee, programming.dev (each instance/community
  has own rules) · type: federated link forums
- Self-promotion policy: **[V-3p]** many communities state "No self-promotion or
  upvote-farming of any kind" (lemmy.world example); rules are per-community and
  vary. **[U] overall — must read each community's sidebar.**
- Account requirements: open registration on most instances; some have application
  queues.
- Modules: none first-choice; possible echo of an HN-popular technical essay.
- Risk: fragmented moderation; low reach; not worth early effort.

### C8. Mastodon (tech instances) — Fit: **Med**
- URLs: fosstodon.org (FOSS-focused; invite/apply **[V-3p]**), hachyderm.io (tech
  professionals; OSS-project accounts allowed if they meet documented requirements
  **[V-3p]**) · type: federated microblog
- Submission format: post thread with media + hashtags (#opensource #julia-like
  research tags per instance culture).
- Self-promotion policy: no platform rules against sharing your own OSS; norms =
  contribute to the fediverse, don't broadcast-only. Project accounts explicitly
  accommodated at Hachyderm (with requirements) and Fosstodon (project accounts
  have migrated both ways — Fedora/XWiki precedent).
- Account requirements: account on an instance (some invite/approval).
- Modules: any; the visual demo carries.
- Angle: GIF + "we deleted the octree" one-paragraph explainer, link in reply.

### C9. WebGPU Matrix (#WebGPU:matrix.org) — Fit: **High** (WebGPU modules)
- URL: https://webgpu.org/ (community: #webgraphics:matrix.org; general channel
  #WebGPU:matrix.org) · type: official W3C community-group chat on Matrix
- Submission format: chat messages with demo links; webgpu.org lists samples/demos.
- Self-promotion policy: **[V]** the community explicitly encourages people to
  "contribute samples / demos / articles using WebGPU" and join Matrix chat
  (Khronos/webgpu.org community pages). Demo-sharing is welcomed.
- Account requirements: any Matrix account.
- Modules: all WebGPU backends — live n-body demo, video codecs, gamedev modules,
  attention kernels compiled to WGSL.
- Angle: "5M-particle browser FMM at 60fps — no octree, hash-lattice index; WGSL
  compute, works today."

### C10. X / Twitter — see A6; as a launch channel it is the echo amplifier for
every other venue (post the HN thread, the Space, the Ziggit post).

### C11. LinkedIn (posts + groups) — Fit: **Med-Low**
- URL: linkedin.com · type: professional network + groups
- Submission format: personal feed post (article/link + narrative); group posts
  subject to each group's rules and moderation queues.
- Self-promotion policy: **[V]** official guidance ("Self-Promotion in Groups",
  LinkedIn Help): explain relevance, invite discussion, follow group rules;
  best-practice docs say avoid self-promotion and add context. Pure promo posts in
  groups get removed and degrade algorithmic trust. Feed posts on your own profile
  are unrestricted.
- Account requirements: LinkedIn account; groups may screen members.
- Modules: none specifically; personal-story framing ("10 years of FMM pain → we
  replaced the tree with a hash table") performs; cold repo drops do not.
- **Group cold promotion: NOT allowed in most ML groups; personal feed is fine.**

### C12. dev.to — see A7/B6 (doubles as the canonical home for every deep-dive).

### C13. Papers With Code ecosystem — see A8. Dead since July 2025; revival at
paperswithcode.co; HF Trending Papers is the de facto successor (paper required).

### C14. Tildes (~comp) — Fit: **Avoid**
- URL: https://tildes.net · type: invite-only forum. **[V-3p]** "Tildes is a
  community, not a free advertising platform"; self-promotion "should be strongly
  discouraged ... treated harshly" in official discussions. Listed under Avoid.

---

## 1) Module-family → venue mapping

| Module family (modules) | Primary venues | Secondary |
|---|---|---|
| Multipole attention (multipole_attention, flash_multipole_kernel, taylor_fgt_attention, spherical/hyperbolic_multipole_attention) | HN Show HN (C1), r/MachineLearning [P] (A2), JAX Show and Tell (A1), GPU MODE (A4) | X (A6), HF forum (A3), dev.to (A7) |
| Equivariant / geometric (equivariant_transformer, equivariant_field_layer, continuous_meshfree_gnn) | r/MachineLearning (A2), JAX Show and Tell (A1) | HF Spaces (A3), EleutherAI (A5) |
| Sequence/state-space (multipole_mamba_ssm, elastic_kv_cache, hierarchical_elastic_kv_cache, visual_transformer_ops) | r/MachineLearning (A2), X (A6) | dev.to (A7), HF Spaces (A3) |
| Neural operators & flow (kernel_independent_fmm, spectral_neural_pme, diffusion_policy_fmm, multipole_flow_drift, neural_sph_ipc) | JAX Show and Tell (A1), r/MachineLearning (A2), HF Spaces (A3) | GPU MODE (A4), EleutherAI (A5) |
| Autodiff machinery (autograd_adjoint_fmm) | JAX Show and Tell (A1), PyTorch Forums as question (A9) | dev.to (A7) |
| GPs & inference (multipole_gaussian_process, matrix_free_gaussian_process, localized_ensemble_kalman_fmm) | r/MachineLearning (A2), JAX Show and Tell (A1) | dev.to (A7), OR SE answers (B8) |
| Graph learning on spatial data (personalized_pagerank_fmm, network_power_centrality, spectral_graph_sparsifier, spectral_biclustering_fmm, spatial_graph_partitioning) | r/algorithms (B2), r/compsci (B3), HN (B1) | awesome-lists (B9), dev.to (B6) |
| Sublinear algorithms (sublinear_distance_oracle, sublinear_edit_distance, sublinear_fast_dtw) | r/algorithms (B2), cstheory as question (B4), Ziggit (B7) | lobste.rs via third party (B5), HN (B1) |
| Optimal transport (optimal_transport_fmm, co_optimal_transport) | r/MachineLearning (A2) if ML-framed, r/algorithms (B2), OR SE answers (B8) | awesome-ot PR (B9), dev.to (B6) |
| Spatial data structures (spatial_disjoint_set, multipole_range_tree, elastic_quotient_filter, non_uniform_fourier_hash, algebraic_multipole_tensor, continuous_meshfree_wavelet) | Ziggit Showcase (A10/B7), r/algorithms (B2), HN (B1) | lobste.rs (B5), SO tag answers (B8) |
| Physics/analysis operators (tree_free_geodesic_fmm, koopman_spectral_operator, phase_space_attractor_fmm, oscillatory_butterfly_kernel, quantum_fock_exchange_fmm, capacitance_boundary_bem, screened_yukawa_fmm, fractional_laplace_contour, fractional_volterra_memory) | HN (B1), GPU MODE for kernels (A4), dev.to (B6) | r/compsci (B3), lobste.rs (B5) |
| Socio-dynamical & causal (opinion_dynamics_fmm, spatial_voting_equilibrium, kernel_causal_discovery, functional_sobol_anova) | r/compsci (B3), cstheory as question (B4) | dev.to (B6), X (A6) |
| Compression (spatial_point_cloud_compression) | Ziggit (A10), r/webgpu (B10) | HN (B1) |
| WebGPU/Zig demo surface (live sim, video codecs) | WebGPU Matrix (C9), r/webgpu (B10), Ziggit (A10), HF Spaces (A3) | X (A6/C10), Show HN (C1) |

---

## 2) Top 10 highest-value, lowest-risk (ranked)

1. **Hacker News Show HN** — zero gate, purpose-built for "made something people can
   play with," the live WebGPU demo is the ideal artifact. One shot; prepare
   benchmarks and author availability. (C1)
2. **JAX GitHub Discussions Show and Tell** — showcase is the explicit category;
   targets exactly the JAX audience for the neural-ops family. (A1)
3. **Ziggit Showcase** — explicit showcase category, open signup, concrete-results
   norm; covers all native-Zig data structures. No AI-generated text on-forum. (A10)
4. **dev.to deep-dive series** — your own platform, no gates, canonical-URL support;
   feeds every other venue and Google. (A7/B6/B12)
5. **Hugging Face: host demo Space + forum project-share post** — category exists for
   exactly this; demo gets a permanent URL. (A3)
6. **Awesome-list PRs (awesome-webgpu, awesome-jax, awesome-optimal-transport, Zig
   lists)** — lowest-risk durable channel; 10 one-line PRs. (B9)
7. **WebGPU Matrix + r/webgpu** — official community explicitly solicits demos;
   perfect for the browser FMM. (C9/B10)
8. **GPU MODE Discord demo channel** — kernel community; read channel rules inside
   the server first; bring numbers. (A4)
9. **r/MachineLearning [P]** — highest-value ML subreddit, but gated by AutoMod karma
   and strict spam policy; post only with benchmark tables and validated-vs-experimental
   honesty. Warm the account first. (A2)
10. **X/Twitter figure-first thread** — no rules risk, amplifies everything above;
    peer-reviewed evidence that images drive engagement. (A6/C10)

(11. lobste.rs — high fit culturally, but requires an invite and a 70-day wait before
you may use the `show` tag yourself; plan early or have an established user submit. B5.
12. Product Hunt — medium: OSS welcome, but product-flavored audience; only after the
demo is polished. C5.)

---

## 3) Avoid / hostile to this content

- **Tildes (~comp)** — "a community, not a free advertising platform"; harshest
  written anti-self-promo stance of any venue surveyed. (C14)
- **r/programming** — direct self-promotion removed; "no blogspam" rule; only viable
  via a substantial article submitted by someone else. (C3)
- **cstheory / OR SE / Stack Overflow as launch posts** — question-only Q&A; "avoid
  overt self-promotion" is written policy; announcements get closed/flagged. Usable
  only as genuine questions or genuine answers. (B4/B8)
- **LinkedIn ML groups (cold posts)** — removed per group rules and LinkedIn's own
  guidance; damages algorithmic trust. Personal feed is fine. (C11)
- **Many Lemmy programming communities** — per-community "no self-promotion or
  upvote-farming of any kind" rules; low reach for the effort. (C7)
- **lobste.rs `show` within first 70 days of membership** — mechanically blocked,
  not just discouraged. Also never treat it as "write-only announcement" space. (B5)
- **Papers With Code (.com)** — service shut down July 2025; do not plan around it. (A8/C13)
- **YC Library / Startup School / Launch YC** — not community showcase venues;
  HN is the only YC channel for this. (C2)
- **Vote-solicitation anywhere** — HN explicitly: "Please don't ask friends to upvote
  or comment"; Reddit bans it; it is the fastest way to get a launch killed.
- **AI-slop-flavored copy** — Ziggit bans AI-generated text outright; HN and dev.to
  communities are actively hostile to undisclosed AI text; disclose or write by hand.

---

## 4) Ambiguities / could not fully verify (do not rely without re-checking)

- Reddit per-sub karma/age gates (r/programming, r/MachineLearning, r/algorithms,
  r/compsci): enforced by AutoMod, thresholds intentionally unpublished. Read the
  sidebar + mod notes on the day.
- r/programming "link-only" restriction: widely reported but not confirmed from the
  live rules page this session (Reddit blocks rule-page fetches).
- GPU MODE Discord channel rules: only visible inside the server.
- r/webgpu rules: not fetched; community observed to accept project shares.
- paperswithcode.co revival: active and announced, but its submission/contribution
  process was not verifiable this session.
- Launch YC availability to non-YYC projects: unverified.
- HN "best time to post": two credible 2025 analyses disagree (Sunday midnight PT vs
  weekday 8–11 AM ET); treat as heuristics.
- EleutherAI Discord: norms observed, no written showcase policy exists to cite.

## Sources (primary)

- Show HN guidelines: https://news.ycombinator.com/showhn.html (fetched 2026-08-24)
- lobste.rs About: https://lobste.rs/about (fetched 2026-08-24)
- cstheory expected behavior: https://cstheory.stackexchange.com/help/behavior ;
  network rule: https://meta.stackexchange.com/questions/57497
- JAX community: https://github.com/jax-ml/jax/discussions/14802 ;
  https://github.com/jax-ml/jax/discussions/categories/show-and-tell
- Ziggit categories/moderation: https://ziggit.dev/categories ;
  https://ziggit.dev/t/ziggit-and-large-language-models/15068 ;
  https://ziggit.dev/t/introducing-zv-a-blazing-fast-zig-version-manager-project-starter/12315
- WebGPU community: https://webgpu.org/ ; https://github.com/gpuweb/gpuweb
- Hugging Face papers: https://huggingface.co/papers/submit ; forum:
  https://discuss.huggingface.co/t/where-to-share/176592
- PWC shutdown/revival: https://blog.tib.eu/2025/10/02/papers-with-code-went-offline-the-knowledge-doesnt-have-to/
  ; https://www.reddit.com/r/MachineLearning/comments/1tgmwqr/ (paperswithcode.co)
- HN timing analyses: https://blog.alcazarsec.com/tech/posts/best-time-to-post-on-hacker-news ;
  https://news.ycombinator.com/item?id=44625897
- New-account HN behavior: https://news.ycombinator.com/item?id=36844201
- LinkedIn self-promo guidance: https://www.linkedin.com/help/linkedin/answer/a569220
- Product Hunt: https://www.producthunt.com/launch/preparing-for-launch ;
  https://www.producthunt.com/topics/open-source
- Tildes self-promo: https://tildes.net/~tildes.official/3i5/daily_tildes_discussion_approaches_to_self_promotion
- Startup School: https://www.startupschool.org/
