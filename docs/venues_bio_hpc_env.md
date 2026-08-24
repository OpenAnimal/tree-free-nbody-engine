# Promotion Venue Research — Bioinformatics / Physics-HPC / Environmental / Gamedev

Compiled 2026-08-24. All rules below were web-verified on this date unless explicitly
marked **[UNVERIFIED]** or **[PARTIALLY VERIFIED]**. Quotes are from the venue's own
pages. Reddit rule pages frequently block crawlers — anything Reddit-side that I could
not read directly is flagged.

Project context used: *tree-free-nbody-engine* — octree-free FMM-family engine using
optimal non-reordering open addressing (Farach-Colton, Krapivin, & Kuszmaul, 2025,
arXiv:2501.02305) over an implicit lattice; pointerless, O(1) worst-case lookups;
~175 NumPy/JAX/Zig/WebGPU modules across 11 domains; MIT. `physics_simulation/`
contains the PPF contact-solver FMM suite (`matrix_free_ipc.py`,
`cloth_shell_simulation.py`, `tetrahedral_surgical_soft_robotics.py`, contact
benchmarks vs. PPF) — mapped below under Robotics/Sim.

Legend: **fit** = High/Med/Low for *this repo*. "Cold promo" = posting a link to your
tool without prior participation.

---

## A) Bioinformatics / Computational Biology (top human-benefit priority)

### A1. Biostars — https://www.biostars.org
- **Type**: Q&A/forum (Biostars engine), the largest bioinformatics Q&A site.
- **Submission format**: dedicated **`Tool` post type** — verified via the official
  tutorial "How to Use Biostars, Part II" (biostars.org/p/180013/), which lists post
  types and describes the Tool type; live tool announcements visible at
  https://www.biostars.org/t/Tools/ (e.g., "Tool: IsoMatch enables ..."). Tool posts
  use `Tool:` title prefixes and body = what it does, install, docs link.
- **Self-promo policy**: announcing your own tool is *explicitly sanctioned* via the
  `Tool` post type ("If you have discovered or designed a new tool, share it with a
  `Tool` post" — official tutorial p/180013). Community norms (from the guidelines
  post p/180747): substantive content, docs, no bare links; empty/one-line posts risk
  deletion.
- **Account**: free registration required to post.
- **Fit**: **High** — single best "bio people see it" venue.
- **Modules**: nearly all 30 bio modules; lead with electrostatics/GB solvation,
  CRISPR off-target, k-mer elastic hash, RNA folding, oncology/ddG.
- **Angle**: "Tool: tree-free-nbody-engine — linear-time electrostatics, solvation &
  spatial-graph kernels without octrees (pointerless hash FMM)."

### A2. SEQanswers — https://seqanswers.com/forum/
- **Type**: forum (vBulletin 6), genomics-sequencing community.
- **Submission format**: threads in subforums. Verified subforums: **Bioinformatics**
  ("Discussion of next-gen sequencing related bioinformatics: resources, algorithms,
  open source efforts, etc" — ~22k topics, active) and **General Genomics
  Discussion** ("any topics that don't fit into other categories").
- **Self-promo policy**: open-source efforts explicitly named on-topic in the
  Bioinformatics subforum description. Commercial content is restricted to sponsor
  areas / "no overtly commercial content permitted without prior approval" (forum
  rules text on Events/commercial posts) — non-commercial OSS announcements are fine.
- **Account**: free registration required (vBulletin login).
- **Fit**: **High** for sequence modules.
- **Modules**: k-mer elastic-hash counter, de Bruijn graph indexer, pan-genome search,
  minimizer seed-extend, CRISPR off-target, TCR-pMHC/neoantigens, RNA 3D folding.
- **Angle**: "Open-source: elastic k-mer hashing + de Bruijn indexing built on an O(1)
  worst-case hash table (2025 FOCS result) — feedback wanted."

### A3. Bioinformatics Stack Exchange — https://bioinformatics.stackexchange.com
- **Type**: Stack Exchange Q&A.
- **Submission format**: questions/answers only — **no announcements**.
- **Self-promo policy**: on-topic page (verified): bio = "intersection of biology,
  computer science, and maths or statistics"; "if a question involves a mixture of
  biology and computer science or math, it is probably appropriate." Self-answering
  acceptable; network-wide rules require disclosing affiliation when recommending
  your own software in answers.
- **Account**: Stack Exchange account; 100+ rep needed to self-answer with
  restrictions lifted? (standard SE mechanics — new accounts can ask/answer).
- **Fit**: **Med** — answer-first venue: answer k-mer/electrostatics/speed questions,
  cite the repo with disclosure.
- **Modules**: k-mer counting, PDB handling, MR, pharmacogenomics Q&A.
- **Angle**: answer "how to scale all-vs-all Coulomb / k-mer counting" questions,
  then self-answer one canonical question referencing the engine (with disclosure).

### A4. Matter Modeling Stack Exchange — https://mattermodeling.stackexchange.com
- **Type**: Stack Exchange Q&A (DFT/MD/comp-chem/materials).
- **Submission format**: Q&A; self-answers allowed ("also OK to ask and answer your
  own question" — on-topic page, verified).
- **Self-promo policy**: announcements not on-topic; the community is famously
  welcoming to software *authors* who answer and disclose affiliation. Scope
  (verified): "accepts all questions about the implementation and use of matter
  modeling software" plus theory.
- **Account**: SE account.
- **Fit**: **High** for MD/electrostatics/screening modules.
- **Modules**: GB solvation, Yukawa/Debye-Hückel FMM, screened Coulomb, constant-pH
  titration, non-periodic MD engine, LLPS/condensates, polypharmacology matrix,
  cryptic pockets, whole-cell sim.
- **Angle**: answer long-range electrostatics questions ("PME vs FMM for X"), disclose
  you built a pointerless FMM and show scaling numbers.

### A5. MDAnalysis — Discord https://discord.gg/fXTSfDJyxE + GitHub Discussions
https://github.com/MDAnalysis/mdanalysis/discussions
- **Type**: chat (Discord) + forum (GH Discussions). Mailing lists are **archived** —
  Discord/Discussions are the current channels (verified via mdanalysis.org/community).
- **Submission format**: Discord channels for casual/quick discussion ("quick
  questions, casual chat, and talking directly with core devs"); GH Discussions
  "Best for technical questions, feature discussions, and best practices."
- **Self-promo policy**: no explicit promo rule published; culture = ask/answer
  technical questions; a "would this kernel help MDAnalysis workloads?" framing
  lands better than a link drop.
- **Account**: Discord invite / GitHub account.
- **Fit**: **High** — exactly the "open-source MD package creators" audience the user
  wants.
- **Modules**: non-periodic MD engine, residue contact graphs, binding pockets, PDB
  loaders, NMA/allostery.
- **Angle**: "Pointerless O(N) neighbor-list/electrostatics kernel — useful for
  MDAnalysis trajectory analysis? Benchmarks inside."

### A6. OpenMM — GitHub Discussions "Show and tell" category
https://github.com/openmm/openmm/discussions/categories/show-and-tell
- **Type**: GitHub Discussions forum.
- **Submission format**: discussion post in the **"Show and tell"** category —
  verified to exist (categories: General, Ideas, Polls, Q&A, Show and tell).
  Plugin/tool authors announce here with description + install + repo link.
- **Self-promo policy**: showing your own work is the *purpose* of the category.
- **Account**: GitHub account.
- **Fit**: **High**.
- **Modules**: GB/solvation free energy, 3D molecular electrostatics, constant-pH
  titration, non-periodic MD, cryptic pockets.
- **Angle**: "Show and tell: octree-free FMM electrostatics prototype — could a
  CustomNonbondedForce-style path use hash-lattice multipoles?"

### A7. GROMACS forum — https://gromacs.bioexcel.eu
- **Type**: Discourse forum (replaced gmx-users mailing list; old list archived).
- **Submission format**: topic in the dedicated **third-party tools** category —
  verified description: "Discussion forum for third party tools and files useful for
  the GROMACS community." Announcements category is staff-only.
- **Self-promo policy**: third-party tool posts explicitly on-topic **in that
  category**; general etiquette ("use good etiquette, avoiding ALL CAPS..."; one
  topic per question; informative titles).
- **Account**: forum account (email posting possible but must match account address
  exactly).
- **Fit**: **High**.
- **Modules**: non-periodic MD, constant-pH, Yukawa/screened-Coulomb FMM, GB.
- **Angle**: "Third-party tool: pointerless FMM for non-periodic electrostatics —
  PME-free alternative benchmark."

### A8. AMBER mailing list — http://lists.ambermd.org/mailman/listinfo/amber
- **Type**: Mailman list ("forum for users of the AMBER Molecular Dynamics and
  related software"); archive back to 1999.
- **Submission format**: email; subscription required.
- **Self-promo policy**: **[PARTIALLY VERIFIED]** — list exists and third-party tool
  announcements have historically appeared (e.g., GridMAT-MD announcement posted by a
  GROMACS/AMBER dev in 2013); no explicit policy page found. One relevant,
  well-received announcement email is the safe pattern.
- **Account**: subscribe at the info page.
- **Fit**: **Med-High** for constant-pH MD (AMBER's cpHMD community) and GB models.
- **Angle**: "Constant-pH / GB practitioners: linear-time generalized Born without
  tree rebuilds — prototype + validation status."

### A9. NAMD-L / VMD-L — https://www.ks.uiuc.edu/Research/namd/mailing_list/
- **Type**: subscriber mailing lists (UIUC TCBG). Verified: "VMD-L is unmoderated,
  however posting is limited to subscribers only"; namd-l list server changed Sept
  2025 (re-subscribe may be needed).
- **Submission format**: email to namd-l@lists.ks.uiuc.edu.
- **Self-promo policy**: **[UNVERIFIED beyond subscriber-only rule]** — NAMD-L is "a
  forum for discussion and exchange of ideas between users and developers of NAMD";
  tool-relevant technical posts are normal.
- **Account**: subscription required.
- **Fit**: **Med** (NAMD is the tree-code lineage cousin — FMM comparison angle).
- **Angle**: technical comparison post: "Octree-free FMM on modern GPUs — has anyone
  benchmarked against NAMD's PME/full methods?"

### A10. scverse (single-cell ecosystem) — Discourse https://discourse.scverse.org ·
Zulip https://scverse.zulipchat.com · ecosystem registry
https://github.com/scverse/ecosystem-packages
- **Type**: Discourse forum + Zulip chat + biweekly community meetings (open agenda
  on HackMD) + annual conference (2026: Oct 12-14; **CfP opened Aug 20** —
  conference2026@scverse.org).
- **Submission format**: Discourse categories (verified): Help ("for help with a
  particular scverse package"), General ("topics broader in scope than one
  particular package"), Ecosystem, Announcements ("for scverse-related
  announcements" — i.e., internal, not for outside projects), Jobs. **No dedicated
  showcase category** — General is the right slot; ecosystem-packages listing via PR.
- **Self-promo policy**: **[PARTIALLY VERIFIED]** — no published promo rule found;
  "General" category + community-meeting "show your project"-style agenda slots are
  the observed mechanism (meetings have featured ecosystem package showcases).
- **Account**: Discourse/Zulip sign-up; GitHub for the registry PR.
- **Fit**: **Med** (modules are adjacent to single-cell, not core).
- **Modules**: 3D chromatin architecture, perturb-seq causal GRNs, GNN long-range
  layer, diff-FMM guidance, whole-cell sim.
- **Angle**: General-forum post: "Long-range interaction layer (FMM-based) for
  spatial omics / GRN inference — would this interest anyone here?"

### A11. Biopython mailing list — https://biopython.org/wiki/Mailing_lists
- **Type**: OBF-hosted mailing lists. Verified: `biopython@biopython.org` general
  discussion list (Python + bioinformatics, feature requests, help);
  `biopython-announce` is a low-volume **moderated list for Biopython release
  announcements only** — do NOT use it for third-party tools.
- **Submission format**: email (subscribe first; non-subscriber mail may be
  silently discarded due to past spam abuse).
- **Self-promo policy**: **[PARTIALLY VERIFIED]** — no explicit third-party promo
  rule; the general list historically tolerates "I wrote a package relevant to
  Biopython users" announcements when they invite collaboration (e.g., on PDB/SeqIO
  interoperability).
- **Fit**: **Med**.
- **Modules**: PDB loaders, k-mer/minimizer indexing, contact-map graphs.
- **Angle**: "PDB loader + elastic-hash k-mer modules — compatible with Biopython
  objects; feedback from maintainers welcome."

### A12. Bioconda — docs https://bioconda.github.io (Gitter/Matrix chat; PRs to
bioconda-recipes)
- **Type**: conda channel community; coordination via GitHub issues/PRs and a
  Gitter room (referenced by official FAQs as "the Bioconda Gitter"; exact current
  room URL **[UNVERIFIED]** — check the "Getting Help"/FAQ section of the docs).
- **Submission format**: conda recipe PR to `bioconda/bioconda-recipes` — the PR
  *is* the announcement; the package then appears to every `conda install` user.
- **Self-promo policy**: shipping a recipe is the sanctioned mechanism; chat is for
  recipe help.
- **Account**: GitHub; CLA/bot workflow.
- **Fit**: **High** (distribution = promotion) — the single most "package authors
  will see it" action in bio OSS.
- **Modules**: the whole Python-side repo (one `tree-free-nbody-engine` recipe).
- **Angle**: submit the recipe; ask in Gitter only for review help.

### A13. Galaxy Project — Help forum https://help.galaxyproject.org · Matrix chat
(see https://training.galaxyproject.org/training-material/faqs/galaxy/support_messaging_system.html)
· Community Hub news via PR
- **Type**: Discourse help forum ("A place for all Galaxy communities"), Matrix
  (primary chat, bridged from old Gitter), and a news site fed by PRs to
  `galaxyproject/galaxy-hub` (verified CONTRIBUTING: news items go in
  `content/news/<year>/<slug>/index.md`).
- **Submission format**: help-forum topic; Matrix message; Hub news PR.
- **Self-promo policy**: **[PARTIALLY VERIFIED]** — no explicit rule found; the
  Hub PR + relevant Matrix channel (dev/tools) is the observed route for tool
  news; wrapping a module as a Galaxy Tool Shed tool is the deep-integration path.
- **Account**: forum account / Matrix account / GitHub.
- **Fit**: **Med** (only worth it if you wrap 1-2 modules as Galaxy tools).
- **Modules**: CRISPR off-target scanner, k-mer counter, minimizer search,
  pharmacogenomics.
- **Angle**: tool-wrapper + short Hub news item: "Linear-time spatial kernels now
  wrapped for Galaxy."

### A14. MNE-Python community (EEG/MEG) — https://mne.tools (Discourse/lists)
**[UNVERIFIED — venue exists, current channel + promo rules not checked]**
- **Fit**: Med for `eeg_source_localization_fmm.py` + `biosignal_lsl_stream_engine.py`;
  ask as a technical question about FMM-accelerated forward solutions.
- Similar: LabStreamingLayer community (GitHub/forum) for the LSL engine —
  **[UNVERIFIED]**.

---

## B) Physics / HPC / MD / Astro

### B1. Computational Science Stack Exchange (scicomp) —
https://scicomp.stackexchange.com
- **Type**: SE Q&A.
- **Submission format**: questions/answers; no announcements.
- **Self-promo policy** (verified from /help/on-topic): scope = "questions and answers
  about computational methods used in technical disciplines"; on-topic: "Questions
  about software packages or languages used broadly in computational science."
  Page links developer-specific metas on "using this site as a resource" and
  "guidelines on disclosing project affiliations" — i.e., developers welcome but
  **must disclose affiliation** when citing their own work in answers.
- **Account**: SE account.
- **Fit**: **High** — the natural home for the algorithmic core (FMM, hash tables,
  Biot-Savart, FGT, Taylor FMM).
- **Modules**: radial Taylor FMM, Gaussian FGT, vortex Biot-Savart, N-body core,
  contact solver.
- **Angle**: self-answered canonical Q: "How to eliminate octree pointer-chasing in
  FMM? — non-reordering open addressing over a Morton lattice (results)."

### B2. r/HPC — https://www.reddit.com/r/HPC/
- **Type**: subreddit — "Subreddit for posting questions and asking for general
  advice about high performance computing" (verified description).
- **Submission format**: text/link post; sidebar demands reading rules & FAQ first.
- **Self-promo policy**: **[UNVERIFIED — rule page blocked to crawlers]**; Reddit
  sitewide wiki (r/reddit.com/wiki/selfpromotion) says self-promotion is "generally
  frowned upon" — participate before promoting.
- **Account**: Reddit account with karma history.
- **Fit**: **Med**.
- **Modules**: Zig SIMD backend, JAX FMM, lock-free CAS parallelism, GPU kernels.
- **Angle**: discussion post with benchmark plots: "Octree pointer-chasing vs flat
  hash lattice on SIMD/GPU — our numbers; what's yours?"

### B3. pynbody users — https://groups.google.com/g/pynbody-users
- **Type**: Google Group mailing list (verified active; official support channel
  per pynbody docs).
- **Submission format**: email/thread; Google account to join (or post via web).
- **Self-promo policy**: **[UNVERIFIED]** — general user-help list; relevant-tool
  posts historically tolerated.
- **Fit**: **Med-High** for the galaxy-collision module.
- **Angle**: "O(N) gravity backend you can point at Gadget/RAMSES snapshots —
  benchmark vs. pynbody's own tree."

### B4. yt project — Slack (invite at https://yt-project.org/slack.html) + yt-users
list
- **Type**: Slack org + two mailing lists (verified: "yt-users ... for asking for
  help, suggesting features and so on"; yt-dev for development chatter).
- **Submission format**: Slack message / list email.
- **Self-promo policy**: **[UNVERIFIED]**; open, community-run culture.
- **Fit**: **Med**.
- **Modules**: N-body galaxy collision, volumetric field evaluation.
- **Angle**: ask whether a pointerless O(N) gravity/field kernel is useful as a yt
  analysis backend; share benchmark.

### B5. Astropy / Open Astronomy — Discourse https://community.openastronomy.org
(category /c/astropy/8) + users list astropy@python.org
- **Type**: Discourse (Open Astronomy umbrella) + mailing list (verified:
  astropy-dev/astropy users lists; "Astropy Announcements" category is **read-only,
  for Astropy project announcements** — not for third-party).
- **Submission format**: Discourse topic in astropy help/dev categories; email to
  users list ("Post questions or start discussions about anything related to Python
  programming applied to astronomy").
- **Self-promo policy**: **[PARTIALLY VERIFIED]** — announcements category closed to
  outsiders; users list is discussion-oriented. Long-term: Affiliated Package
  program exists (astropy.org/affiliated).
- **Fit**: **Med**.
- **Modules**: N-body/galaxy modules, JAX kernels.
- **Angle**: users-list post framed as benchmark discussion + question.

### B6. JAX community — GitHub Discussions https://github.com/jax-ml/jax/discussions
- **Type**: discussions; **no official JAX Discord** — team "deliberately foster[s]
  GitHub Discussions as the main place for community discussion" (verified, jax-ml
  discussion #14802).
- **Submission format**: discussion post (Q&A / general categories).
- **Self-promo policy**: **[UNVERIFIED]**; treat as technical discussion, not promo.
- **Fit**: **Med-High** for `jax/` FMM modules (jit/vmap-friendly hash tables are a
  genuinely novel discussion topic).
- **Angle**: "Implementing a non-reordering hash table under jit/vmap for FMM —
  pitfalls and benchmarks" (the `docs/GPU_NOTES.md` content maps well).

### B7. MuJoCo discussions — https://github.com/google-deepmind/mujoco/discussions
**[UNVERIFIED — not checked]**
- **Fit**: Med-High for `physics_simulation/ppf_contact_solver_fmm` (matrix-free
  IPC, cloth, surgical soft robotics contact solver); angle: contact-solver scaling
  discussion vs. their island solver. Bevy/Jolt physics discussions (below) cover the
  game-side of the same modules.

---

## Amplifiers (cross-domain, archival)

### X1. Hacker News "Show HN" — https://news.ycombinator.com/showhn.html
- **Type**: link aggregator; Show HN is the sanctioned self-promo track.
- **Rules (verified quotes)**: "Show HN is for something you've made that other
  people can play with"; "The project must be something you've worked on personally
  and which you're around to discuss"; title must begin "Show HN"; no vote
  solicitation ("Please don't ask friends to upvote or comment"); no landing
  pages/fundraisers; must include a URL people can try.
- **Fit**: **High** — the live WebGL/WebGPU demo (index.html) is exactly "something
  people can play with".
- **Angle**: "Show HN: Octree-free N-body/FMM engine — 5M particles in the browser,
  pointerless hash-lattice core" — with honest "research prototype" framing.

### X2. arXiv — https://arxiv.org
- **Type**: preprint server (suggest physics.comp-ph, cs.DS, q-bio.QM/BM, astro-ph.IM
  depending on module family). Requires **endorsement** for first-time submitters in
  most categories. No promo rules — archival. Best paired with a validation-heavy
  methods paper. **Fit: High as an authority amplifier** (every venue above takes a
  preprint more seriously).

### X3. Zenodo — https://zenodo.org
- **Type**: DOI minting + versioned releases; community curation folders exist.
  Release each tagged version (repo already has CITATION.cff). No promo constraints.
  **Fit: High** (infrastructure, not discussion).

### X4. JOSS — https://joss.theoj.org
- **Type**: peer-reviewed software-paper journal (DOI, ~4-8 week review).
- **Rules (verified)**: must be "open source as per the OSI definition", have an
  "obvious research application", "you must be a major contributor", paper "must not
  focus on new research results". **New screening gates (verified)**: repo public
  >6 months with active development; evidence of research use/impact; open practices
  (issues/PRs/tests/CI); iterative history — concentrated commit bursts are
  desk-rejected.
- **Fit**: **High eventually, NOT NOW** — a fresh repo fails the 6-month/impact
  gates. Revisit after adoption evidence exists. (See Avoid list.)

---

## C) Environmental (4 modules)

### C1. AboutHydrology — https://groups.google.com/g/abouthydrology
- **Type**: Google Group announcement list (~6,000 hydrology researchers; weekly
  digest format since 2021).
- **Submission format**: announcement email to abouthydrology@googlegroups.com;
  subscribe first (abouthydrology+subscribe@googlegroups.com).
- **Self-promo policy** (verified from list's own description): the list exists to
  "send announces about events, Ph.D. and post-doc positions, new books, **open
  source software**, AGU and EGU sessions" — OSS announcements explicitly welcomed.
- **Account**: Google account / subscription.
- **Fit**: **High** — cleanest "explicitly allowed" venue found anywhere in this
  research.
- **Modules**: groundwater contaminant plume transport, airborne pollutant exposure.
- **Angle**: one concise announcement: "Open-source: linear-time particle-transport
  solver (no tree rebuilds) for plume/pollutant simulations — MIT, link."

### C2. MODFLOW Users Group — https://groups.google.com/g/modflow
- **Type**: Google Group mailing list (verified active: MODFLOW packages, GUIs,
  modeling scenarios).
- **Submission format**: email/thread after joining.
- **Self-promo policy**: **[UNVERIFIED]** — no formal policy page found; free-software
  guides have been shared there (e.g., "Free Groundwater Modeling Software Guide"
  thread), so OSS topic posts have precedent.
- **Account**: Google group membership.
- **Fit**: **High** for `groundwater_plume.py` (advection-dispersion, particle
  methods).
- **Angle**: "Free/OSS particle-tracking & dispersion kernel — could complement
  MT3D-style transport; benchmark inside."
- Amplifier: "MODFLOW and More" conference (Golden, CO; recurring) — abstract-level
  venue, not a forum.

### C3. CSDMS — Forum https://forum.csdms.io (docs: https://csdms.colorado.edu/wiki/CSDMS_Forum)
- **Type**: Discourse community forum — "A place for researchers to connect, share
  information, and discuss all things CSDMS"; runs "Software releases" round-up
  threads (e.g., Summer 2025); plus a model repository (469 open-source models) and
  `pymt` coupling toolkit.
- **Submission format**: forum topic; model submission to the repository.
- **Self-promo policy**: **[PARTIALLY VERIFIED]** — software-release threads exist as
  recurring formats; introducing a model in the forum + repository listing is the
  sanctioned path.
- **Account**: forum account.
- **Fit**: **High** for groundwater plume + airborne exposure.
- **Angle**: introduce-yourself + "should this become a pymt-couplable component?"

### C4. r/hydrology (r/Hydrology) — https://www.reddit.com/r/Hydrology/
- **Type**: small subreddit (~2.7k weekly visitors; ~49 weekly contributions —
  verified via search).
- **Self-promo policy**: **[UNVERIFIED — rules page not crawlable]**; AutoModerator
  enforces rules aggressively ("we remove all comments which break the rules").
- **Fit**: **Low-Med** (tiny audience; AboutHydrology/CSDMS are better).
- **Angle**: only as a question ("how do you speed up particle tracking?") with the
  repo as context.

### C5. r/medicalphysics — https://www.reddit.com/r/MedicalPhysics/
- **Type**: subreddit — "A place to discuss all things related to Medical Physics."
- **Self-promo policy**: **[UNVERIFIED — sidebar rules not crawlable]**; numbered
  rules exist (rule 1 begins "Do ..." — content cut off in crawl). Sitewide 10%
  self-promo guideline applies; surveys/vendors get redirected (AAPM BBS precedent)
  and a "don't brigade" mod post exists.
- **Fit**: **Low-Med** — clinical community, correctly skeptical of non-validated
  dose engines. The repo's own "research prototype" honesty is the only acceptable
  framing; radiotherapy dose distribution must not look like a clinical tool.
- **Angle**: question-format only: "Anyone benchmarking GPU FMM-style dose
  deposition? Prototype inside, validation status stated."

### C6. Battery / electrochemistry
- **PyBaMM** — Community https://pybamm.org/community (verified: Slack workspace +
  GitHub Discussions + a newer Discourse at pybamm.discourse.group). Norms
  **[PARTIALLY VERIFIED]**: Slack for community questions, Discussions/Discourse for
  technical threads. **Fit: High** for `electrolyte_screening.py` — electrolyte
  screening (Debye-Hückel lineage) is squarely in-topic. Angle: "fast continuum
  electrolyte screening (screened-Coulomb FMM) — useful for parameter sweeps over
  electrolyte recipes?"
- **r/batteries** — https://www.reddit.com/r/batteries/ — verified precedent:
  free-tool announcements well received (e.g., a free battery-pack sandbox tool
  thread). Rules page **[UNVERIFIED]**. Fit: Med (hobbyist-lean; frame as
  simulation/design tool).
- **Intercalation Station** (Substack newsletter) — publishes "Who's Who" guides to
  battery modeling software incl. OSS; pitching an inclusion is low-cost. Fit: Med.
  **[Pitch process unverified — use Substack contact.]**

---

## D) Gamedev / Graphics / Video (breadth, lower priority)

### D1. r/webgpu — https://www.reddit.com/r/webgpu/
- **Type**: subreddit — "News, information, and discussion about WebGPU."
- **Self-promo policy**: **[UNVERIFIED rule text]** but a strong caution signal:
  a user reported a **permanent ban for "LLM slop"** (an AI-heavy Godot WebGPU port).
  Moderators are demonstrably strict about AI-generated-looking content. The repo's
  README invites users to "ask your favourite AI to look at this repository" — do
  NOT lead with AI framing here; post dense technical content (WGSL kernel code,
  benchmarks) and answer questions.
- **Account**: Reddit account with history.
- **Fit**: **Med** (great relevance; hostile to low-effort/AI-flavored posts).
- **Modules**: WGSL/WebGPU kernels, WebGPU n-body demo, Gaussian splat streaming,
  crowd sim, boids.
- **Angle**: "Lock-free spatial-hash N-body in WGSL — compute-pass structure +
  5M-particle browser benchmark."

### D2. three.js forum — https://discourse.threejs.org (Showcase: /c/showcase/7)
- **Type**: Discourse forum.
- **Submission format**: Showcase topic. Verified category description: "Use this
  category to showcase any projects you have created using three.js. Showcases
  **require moderator approval**, so please be patient..."
- **Self-promo policy**: showcasing your three.js work is the category's purpose —
  but the project must **use three.js** (the repo demo is raw WebGL2/WebGPU, so this
  applies only if you ship a three.js demo/example).
- **Account**: Discourse account (trust levels apply).
- **Fit**: **Med** (conditional on a three.js-based demo).
- **Angle**: "5M-particle FMM n-body as a three.js scene — pointerless spatial hash."

### D3. Babylon.js forum — https://forum.babylonjs.com ("Demos and projects" /c/demos/9)
- **Type**: Discourse forum.
- **Submission format**: topic in Demos and projects. Verified description: "Use
  this section of the forum to showcase your work! Whether it is a fun playground
  ... or an entire website you have created..."
- **Self-promo policy**: explicitly the purpose of the section (Babylon-built work).
- **Account**: forum account.
- **Fit**: **Med** (needs a Babylon port/demo).
- **Angle**: WebGPU compute demo thread with playground link.

### D4. Godot forum — https://forum.godotengine.org (Showcase /c/showcase/14)
- **Type**: Discourse forum.
- **Submission format**: Showcase topic. Verified description: "Show everyone what
  you are working on and discover amazing projects! Show games you made with Godot!
  Show tools and other applications you made with Godot!"
- **Self-promo policy**: showcase is for things **made with Godot** — this repo is
  not, so a link-drop is off-topic. The annual showreel likewise requires
  "made in Godot."
- **Fit**: **Low** (unless you demonstrate a Godot integration, e.g., GDExtension
  crowd sim).
- **Angle**: engine-discussion post on pointerless broadphase for crowd sim/fog-of-war
  systems, citing the repo as reference implementation.

### D5. Bevy — Discord https://discord.com/invite/bevy (#showcase) · GitHub
Discussions (Q&A)
- **Type**: Discord (23k+ members) + GH Discussions ("best place for questions about
  Bevy").
- **Submission format**: #showcase channel post (weekly round-ups via "This Week in
  Bevy" newsletter — external visibility).
- **Self-promo policy**: **[UNVERIFIED — rules live inside the Discord]**; #showcase
  posts are aggregated publicly by thisweekinbevy.com, which is the real amplifier.
- **Fit**: **Low-Med** (Rust community; repo is Zig — frame as
  cross-language algorithm reference; Jolt/C++ physics discussions similar).
- **Modules**: crowd sim, flow-field pathfinding, LOD decimator, contact solver
  (IPC) for physics.
- **Angle**: "Showcase: pointerless FMM broadphase/neighbor solver (Zig core,
  algorithm portable) — benchmark discussion."

### D6. r/gamedev — https://www.reddit.com/r/gamedev/
- **Type**: subreddit.
- **Submission format**: weekly threads — **Feedback Friday** (playtesting/dev
  feedback), Screenshot Saturday, etc.
- **Self-promo policy (verified via subreddit wiki posting guidelines)**:
  self-promotional content is **only allowed in the weekly threads** or when
  accompanied by free assets for download. Cold tool-promo outside them is removed.
- **Account**: Reddit account; observe the 90/10 sitewide guideline.
- **Fit**: **Med** (browser demo is 'playable' — Feedback Friday compatible).
- **Angle**: Feedback Friday: "5M-particle browser N-body + spatial tools demo —
  perf feedback wanted."

### D7. Graphics Programming Discord — https://discord.com/invite/graphicsprogramming
- **Type**: Discord (~22k members; graphics-programming.org is the docs site).
- **Submission format**: showcase channel posts / technical discussion.
- **Self-promo policy**: **[UNVERIFIED — rules only visible after joining]**;
  community norm is dense technical content.
- **Fit**: **Med-High** for the graphics-research modules.
- **Modules**: volumetric FMM ambient occlusion, surfel radiosity GI, irradiance
  caches, mesh LOD decimator, Gaussian splat streaming.
- **Angle**: "Hierarchical irradiance caching without a tree — hash-lattice FMM
  results + shader code."

### D8. r/videography — https://www.reddit.com/r/videography/
- **Type**: subreddit for working video professionals.
- **Self-promo policy**: **[PARTIALLY VERIFIED]** — rule 1 is literally titled
  "Self Promotion ..." (truncated in crawl; historically strict, no-drive-by-promo).
- **Fit**: **Low** (pro shooters, not codec tooling users).
- **Modules**: AV1/VVC film grain synthesis, frame dedup, scenecut/GOP analysis,
  1€ stabilizer.
- **Angle**: at most a question about denoise/grain workflows; not a promo venue.

### D9. doom9 forum — https://forum.doom9.org
- **Type**: long-running video-coding forum.
- **Submission format**: dev threads in the appropriate coding subforums (e.g.,
  open-source tool threads like tsMuxer); the News/announcements section shows
  open-source announcements but new-thread creation there can be restricted
  ("You may not post new threads" permission notice observed on section pages).
- **Self-promo policy**: **[PARTIALLY VERIFIED]** — rules doc exists ("Before you
  start posting please read the forum rules"); OSS development threads are an
  established genre; pick the codec/dev subforum, not News.
- **Account**: forum registration (post-approval culture).
- **Fit**: **Med** for film grain synthesis (AV1/VVC film-grain synthesis is a known
  doom9-adjacent topic), scenecut/GOP analysis, motion estimation.
- **Angle**: dev thread: "Open-source FMM-accelerated motion estimation / film-grain
  parameter extraction — early prototype."

### D10. ffmpeg-devel — https://ffmpeg.org/mailman/listinfo/ffmpeg-devel
- **Type**: developer mailing list, patch-review culture.
- **Self-promo policy (verified norms)**: patches only ("patches should be posted to
  the ffmpeg-devel mailing list", via git send-email); netiquette — no top-posting,
  no thread hijacking; **subscribe first** or your mail sits in a rarely-checked
  moderation queue.
- **Fit**: **Low as promotion** (do NOT announce tools there). **High as
  contribution path**: an actual film-grain/dedup filter patch would be the credible
  way in.
- **Modules**: film grain synthesis, frame dedup, scenecut/GOP.

---

## 1) Module-family → venue mapping table

| Module family (repo dir) | Primary venues | Secondary |
| --- | --- | --- |
| Protein electrostatics & GB solvation (`solvation_free_energy.py`) | OpenMM Show and tell (A6), GROMACS 3rd-party (A7) | Biostars (A1), MMSE (A4) |
| 3D molecular electrostatics | MMSE (A4), OpenMM (A6) | Biostars (A1) |
| Constant-pH titration (`constant_ph_titration.py`) | AMBER list (A8), GROMACS (A7) | OpenMM (A6) |
| Non-periodic MD engine (`non_periodic_md_engine.py`) | MDAnalysis Discord/Discussions (A5), OpenMM (A6) | NAMD-L (A9), GROMACS (A7) |
| cryo-EM flexible fitting | Biostars Tool (A1) | MMSE (A4) |
| Macromolecular NMA & allostery (`macromolecular_nma_engine.py`) | Biostars (A1), MDAnalysis (A5) | MMSE (A4) |
| RNA 3D folding (`rna_tertiary_folding_engine.py`) | Biostars (A1), SEQanswers (A2) | — |
| TCR-pMHC immunogenicity / neoantigens | SEQanswers (A2), Biostars (A1) | — |
| Personalized oncology ddG (`personalized_oncology_ddg.py`) | Biostars (A1) | MMSE (A4) |
| Pharmacogenomics (`pharmacogenomics_metabolism.py`) | Biostars (A1) | BioSE answers (A3) |
| Polypharmacology affinity matrix | MMSE (A4), Biostars (A1) | OpenMM (A6) |
| Allosteric druggability / cryptic pockets | MMSE (A4), OpenMM (A6) | Biostars (A1) |
| CRISPR off-target (`crispr_offtarget_detector.py`) | SEQanswers Bioinformatics forum (A2) | Biostars (A1), Galaxy (A13) |
| k-mer elastic-hash + de Bruijn indexer (`kmer_elastic_hash.py`) | SEQanswers (A2), Biostars (A1) | Biopython list (A11) |
| Pan-genome colored de Bruijn search (`pangenome_search_engine.py`) | SEQanswers (A2) | Biostars (A1) |
| Minimizer indexing & seed-extend | SEQanswers (A2), Biopython list (A11) | Biostars (A1) |
| 3D chromatin architecture (`chromatin_expression_engine.py`) | scverse General/Zulip (A10) | Biostars (A1) |
| Perturb-seq causal GRNs (`causal_perturb_seq_grn.py`) | scverse (A10) | Biostars (A1) |
| Mendelian randomization | Biostars (A1), BioSE (A3) | — |
| Condensates / LLPS (`biomolecular_condensate_engine.py`) | MMSE (A4) | Biostars (A1) |
| Smart biologics / antibody engineering | Biostars (A1), SEQanswers (A2) | MMSE (A4) |
| EEG/MEG source localization (`eeg_source_localization_fmm.py`) | MNE community **[UNVERIFIED]** (A14) | Biostars (A1) |
| Biosignal LSL streaming | LSL community **[UNVERIFIED]** | MNE (A14) |
| Binding-pocket detector | MMSE (A4) | Biostars (A1) |
| Residue contact graphs (`contact_map_graph.py`) | MDAnalysis (A5) | Biostars (A1) |
| PDB loaders (`pdb_loader.py`) | Biopython list (A11), MDAnalysis (A5) | Biostars (A1) |
| GNN long-range layer / diff-FMM guidance / whole-cell sim | scverse General (A10) | Biostars (A1) |
| N-body galaxy collision | pynbody-users (B3), yt Slack (B4) | Show HN (X1), astropy users list (B5) |
| Vortex hydrodynamics (Biot-Savart) | scicomp SE (B1) | Graphics Programming Discord (D7) |
| 3D Yukawa / Debye-Hückel FMM, screened Coulomb | MMSE (A4), scicomp SE (B1) | PyBaMM (C6) |
| Radial Taylor FMM, Gaussian FGT | scicomp SE (B1) | MMSE (A4) |
| JAX FMM | JAX GH Discussions (B6) | r/HPC (B2) |
| Zig SIMD backend | Show HN (X1) | r/HPC (B2) |
| WGSL/WebGPU kernels + demos | r/webgpu (D1, caution) | Babylon/three.js (D2/D3), Show HN (X1) |
| PPF contact solver / matrix-free IPC / cloth / surgical robotics (`physics_simulation/`) | MuJoCo discussions **[UNVERIFIED]** (B7) | Bevy/Jolt (D5), r/gamedev FF (D6) |
| Groundwater plume (`groundwater_plume.py`) | AboutHydrology (C1), MODFLOW group (C2) | CSDMS forum (C3) |
| Airborne pollutant exposure (`airborne_exposure.py`) | AboutHydrology (C1), CSDMS (C3) | — |
| Radiotherapy dose (`radiotherapy_dose.py`) | r/medicalphysics (C5, caution) | MMSE (A4) |
| Electrolyte screening for batteries (`electrolyte_screening.py`) | PyBaMM Discourse/Slack (C6) | r/batteries (C6), MMSE (A4) |
| WebGPU n-body demo, boids, crowd sim, fog-of-war, WFC, LOD, flow-field, lasso, surfel GI, FMM AO, irradiance caches, splat streaming | r/webgpu (D1), Graphics Programming Discord (D7) | r/gamedev FF (D6), three.js/Babylon (D2/D3) |
| AV1/VVC film grain, motion estimation, frame dedup, scenecut/GOP | doom9 (D9) | ffmpeg-devel as code-contrib only (D10), r/videography (D8, caution) |
| Event cameras, 1€ stabilizer | r/videography (D8, caution) | doom9 (D9) |

Repo-wide amplifiers for everything: Bioconda recipe (A12), Zenodo DOI (X3), arXiv
preprint (X2), Show HN (X1).

---

## 2) Top-10 shortlist — human-benefit impact

1. **Biostars `Tool` post (A1)** — sanctioned tool-announcement channel in front of
   the whole bioinformatics Q&A crowd; covers the 30-module bio family.
2. **OpenMM GitHub Discussions "Show and tell" (A6)** — OpenMM's maintainers and
   plugin authors personally read this; electrostatics/GB/constant-pH modules.
3. **GROMACS forum third-party tools category (A7)** — dedicated, verified slot for
   exactly this kind of post.
4. **MDAnalysis Discord + Discussions (A5)** — "open-source MD package creators"
   audience the user named; frame as analysis-kernel discussion.
5. **Matter Modeling SE (A4)** — answer-first presence; software authors are welcomed
   when they disclose affiliation and bring numbers.
6. **SEQanswers Bioinformatics forum (A2)** — "open source efforts" literally written
   into the subforum description; k-mer/minimizer/pangenome/CRISPR modules.
7. **AboutHydrology (C1)** — OSS announcements explicitly in the list's charter;
   zero rule risk, ~6k researchers.
8. **PyBaMM Discourse/Slack (C6)** — electrolyte screening lands dead-center in the
   battery-modeling OSS community.
9. **CSDMS forum + model repository (C3)** — recurring software-release threads;
   groundwater/airborne exposure get durable, indexed visibility.
10. **Show HN (X1)** — one-shot cross-domain amplifier; the browser demo satisfies
    the "people can play with it" rule exactly.
- Honorable mentions: MODFLOW Google Group (C2); scverse Zulip + community meeting
  (A10); Bioconda recipe (A12); arXiv/Zenodo (X2/X3) under everything.

---

## 3) Avoid / hostile / rule-restricted list

- **r/webgpu** — permanent-ban precedent for "LLM slop." Only post dense, human,
  technical content; never lead with the README's "ask your favourite AI" framing.
- **r/comp_chem** — rule text (verified snippet): "No self-promotion or asking for
  collabs. If you made a free tool that you want to promote (and have a tip-jar)
  this is fine..." — ambiguous; treat as questions-only unless a mod confirms.
- **r/gamedev main feed** — self-promo allowed **only** in weekly threads
  (Feedback Friday / Screenshot Saturday); verified via subreddit wiki.
- **Reddit sitewide** — self-promotion "generally frowned upon" (r/reddit.com wiki);
  respect the ~90/10 participation ratio on every subreddit above.
- **ffmpeg-devel** — patch-review list; tool announcements are off-topic
  (contribute code instead).
- **JOSS right now** — verified screening gates require >6 months public history +
  demonstrated research impact; a fresh repo will be desk-rejected. Revisit later.
- **Godot Showcase / annual showreel** — "made with Godot" only; link-drops are
  off-topic.
- **Astropy "Announcements" + scverse "Announcements" categories** — internal,
  read-only / project-scoped; outsiders cannot post.
- **Biopython `biopython-announce`** — moderated, Biopython-release announcements
  only; use the general list instead.
- **r/medicalphysics** — clinical-validation culture; a non-validated dose engine
  presented as a tool will land badly (and surveys/vendors get routed to AAPM BBS).
  Question-format only, with the repo's own "research prototype" disclaimer.
- **doom9 News section** — new-thread creation restricted there ("You may not post
  new threads" observed); use the codec dev subforums.
- **Biostars bare-link posts** — Tool posts must be substantive (docs, usage,
  validation status) or risk deletion per the community guidelines.
- **MNE / LSL / MuJoCo / Bioconda-Gitter URL / most Discord channel rules** —
  **[UNVERIFIED]**; read in-venue rules before posting.

---

## Explicit unverified-rules register

The following could not be verified (blocked, login-walled, or not indexed) — check
in-venue before posting: r/HPC rules; r/Hydrology rules; r/medicalphysics full rule
text; r/webgpu rule text; r/batteries rule text; r/videography rule text beyond rule
title; AMBER list policy; NAMD-L promo norms; Biopython third-party promo norms;
scverse promo norms; Galaxy Matrix norms; Bioconda Gitter room URL; pynbody/yt/astropy
promo norms; JAX discussions promo norms; Bevy/Graphics-Programming/MuJoCo Discord
rules; doom9 full rules; Intercalation Station pitch process; MNE/LSL channels.
Everything else above was read from the venue's own pages on 2026-08-24.
