"""
Vocabulary lint: FAIL if a .py file contains an overclaimed phrase outside a
comment or an explicit allowlist.

Checked phrases (docstring/print/runtime claims only — historical class NAMES
like `ProceduralMultipoleMapGenerator` are allowed):
    "O(1) query"
    "Lock-Free CAS"
    "Lock-Free Motion Vector"
    "in O(1) without"
    "Multipole Radiance"
    "MULTIPOLE HARMONICS"
    "Strict O(N)"
    "end-to-end differentiable"
    "in strictly O(N)"
    "accelerated via Morton spatial hashing"

A line is exempt if it starts with "#" (comment). Files in ALLOWLIST are
exempt entirely. Exit code 1 on violations, printing file:line output.

Round-7 extension (task T-B6):
- SCAN_DIRS now also covers core, neural_ops, bioinformatics,
  algorithm_theory, and quantized_bitpacked_optimization (finding F-18).
- `core/elastic_hash.py` is allowlisted (it quotes the FKK paper
  legitimately, including the historical scheme it disavows).
- Structural rule: no `.py` outside `core/elastic_hash.py` may define a
  class whose name matches `Elastic*(Hash|Table|Filter)(\d*D)?` (the naming-ban regex below)
  (catches future legacy pre-funnel hash copies — finding F-01), except the
  two justified entries in STRUCTURAL_ALLOWLIST (T-A2 facade, T-A4 banner).
"""
import os
import re
import sys

FORBIDDEN = [
    "O(1) query",
    "Lock-Free CAS",
    "Lock-Free Motion Vector",
    "in O(1) without",
    "Multipole Radiance",
    "MULTIPOLE HARMONICS",
    "Strict O(N)",
    "end-to-end differentiable",
    "in strictly O(N)",
    "accelerated via Morton spatial hashing",
]

# Files exempt from the check (relative to repo root, forward slashes).
# core/elastic_hash.py quotes the FKK paper legitimately, including the
# historical pre-funnel scheme it documents and disavows.
ALLOWLIST = {
    "core/elastic_hash.py",
    # The linter itself defines the forbidden phrases in its docstring and
    # FORBIDDEN list — scanning it would flag its own vocabulary definition.
    "tools/lint_claims.py",
}

# Round-7 review fix (master-plan finding R7-F23): the structural Elastic-class
# rule below originally contradicted tasks T-A2/T-A4, which deliberately KEEP
# two Elastic-named classes, and turned the run_all gate RED. These two entries
# are the ONLY permitted exceptions; each must stay justified:
#   - elastic_spatial_hash.py: T-A2 facade -- `ElasticSpatialHash3D` delegates
#     to core.elastic_hash.ElasticHashTable (the funnel hash); verified by the
#     probe-bound assertion in bioinformatics/test_sota_modules.py.
#   - elastic_quotient_filter.py: T-A4 banner path -- `ElasticQuotientFilter`
#     is documented legacy (pre-funnel scheme, NOT FKK) in its module banner
#     and algorithm_theory/STATUS.md. Porting it to ElasticIntTable remains
#     open (plan task T-A4b).
# Phrase checks remain ACTIVE in these files (only the class-name rule is
# waived). Any NEW Elastic*Hash/Table/Filter class anywhere still fails.
STRUCTURAL_ALLOWLIST = {
    "bioinformatics/core/elastic_spatial_hash.py",
    "algorithm_theory/elastic_quotient_filter.py",
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_DIRS = [
    "apps",
    "game_mechanics_spatial",
    "graphics_rendering",
    "video_streaming_codecs",
    "physics_simulation",
    # Round-7: previously unchecked dirs (finding F-18)
    "core",
    "neural_ops",
    "bioinformatics",
    "algorithm_theory",
    "quantized_bitpacked_optimization",
    # Round-7 extension: tools/ and environmental_modeling/ were unchecked.
    "tools",
    "environmental_modeling",
]

# Round-7 task T-B6b: claim phrases also live in *.md (finding R7-F22 —
# OVERVIEW.md carried a stale "end-to-end differentiable pipeline" claim
# after the code was fixed). This second pass scans documentation files for
# a markdown-appropriate phrase subset.
MD_FORBIDDEN = [
    "Strict O(N)",
    "Strict Linear O(N)",
    "end-to-end differentiable",
    "End-to-end differentiable",
    "in strictly O(N)",
]

# Markdown files exempt from the MD pass (relative to repo root, forward
# slashes). Each entry is justified:
#   - docs/INAPPLICABILITY.md: quotes the disavowed claims it refutes.
#   - ROUND7_MASTER_PLAN.md: this plan file documents the claims it audits,
#     including the forbidden phrases as named findings (R7-F22, R7-F26).
#   - docs/GPU_NOTES.md: contains a historical "Round-X" narrative that
#     quotes prior wording it then corrects (history sections).
MD_ALLOWLIST = {
    "docs/INAPPLICABILITY.md",
    "docs/GPU_NOTES.md",
}

# Directory prefixes whose markdown is archived process material: the
# review plans quote the very claims they audit and ban (verbatim
# historical records, not live claims). Round-9 housekeeping moved
# ROUND{7,8,9}*.md and the implementation plans here.
MD_DIR_ALLOWLIST = (
    "docs/review_history/",
)

# Glob patterns for markdown files to scan (relative to repo root).
# BENCHMARKS.md is the canonical speed/accuracy source — scan it so a
# stale forbidden claim there is caught too.
#
# Round-8 audit fix: the previous globs used single-level `*/README.md` and
# `*/STATUS.md`, which silently skipped nested project docs such as
# `physics_simulation/ppf_contact_solver_fmm/README.md` (two levels deep).
# The globs are now recursive (`**/README.md`, `**/STATUS.md`) and a
# catch-all `**/*.md` is included so EVERY markdown file in the repo is
# scanned unless it lives under an excluded top-level dir (node_modules,
# .git, __pycache__, zig-cache) or is in MD_ALLOWLIST.  This catches
# future nested docs without needing a per-subproject glob entry.
MD_GLOBS = [
    "README.md",
    "OVERVIEW.md",
    "BENCHMARKS.md",
    "docs/*.md",
    "**/README.md",
    "**/STATUS.md",
    "**/*.md",
]

# Per-(file, phrase) allowlist for the case-insensitive MD pass: legitimate
# code identifiers or historical quotes that use a forbidden phrase in a
# justified context (e.g. a file quoting a disavowed claim it then refutes).
# Keys are (rel_path, phrase_lower) tuples.  Empty by default — add entries
# only when a non-exempt file legitimately needs a phrase.
MD_PHRASE_ALLOWLIST = set()

# Structural rule: no class named Elastic<...>Hash|Table|Filter[<digits>D]
# outside the allowlist. Catches future legacy pre-funnel hash copies
# (finding F-01). The (?:\d*D)? suffix allows `ElasticSpatialHash3D` while
# the word boundary avoids false positives like `ElasticHashBoidSwarm` or
# `ElasticMultipoleKVCache` (which use hashing but are not hash-table
# re-implementations).
_ELASTIC_CLASS_RE = re.compile(
    r"^\s*class\s+Elastic\w*(?:Hash|Table|Filter)(?:\d*D)?\b"
)


def _is_comment_line(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("#")


def scan_file(path: str, rel_path: str):
    violations = []
    if rel_path in ALLOWLIST:
        return violations
    try:
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if _is_comment_line(line):
                    continue
                for phrase in FORBIDDEN:
                    if phrase in line:
                        violations.append((rel_path, lineno, phrase, line.rstrip()))
                        break
                # Structural rule: class Elastic<Upper> outside allowlists
                if rel_path in STRUCTURAL_ALLOWLIST:
                    continue
                if _ELASTIC_CLASS_RE.match(line):
                    violations.append((
                        rel_path, lineno,
                        "class Elastic<Upper> (pre-funnel hash copy — finding F-01)",
                        line.rstrip(),
                    ))
    except (OSError, UnicodeDecodeError):
        pass
    return violations


def _expand_md_globs():
    """Yield (full_path, rel_path) for every markdown file matched by MD_GLOBS.

    Uses glob.glob(recursive=True) so that patterns like ``*/README.md``,
    ``**/README.md`` and ``**/*.md`` work correctly.  The previous
    hand-rolled splitter broke on ``*`` bases because
    ``os.path.isdir(os.path.join(REPO_ROOT, '*'))`` is always False,
    silently skipping 12+ sub-project docs; the single-level ``*/README.md``
    glob additionally missed nested docs two+ levels deep (e.g.
    ``physics_simulation/ppf_contact_solver_fmm/README.md``), which the
    recursive ``**`` patterns now catch.
    """
    import glob as _glob
    # Top-level dirs that are NOT project documentation and must never be
    # scanned (node_modules, .git, caches, build artifacts).
    _EXCLUDE_TOP = {"node_modules", ".git", "__pycache__", "zig-cache", ".zig-cache"}
    seen = set()
    for pattern in MD_GLOBS:
        for full in _glob.glob(os.path.join(REPO_ROOT, pattern), recursive=True):
            if not full.endswith(".md"):
                continue
            rel = os.path.relpath(full, REPO_ROOT).replace(os.sep, "/")
            # Skip excluded top-level directories.
            top = rel.split("/", 1)[0]
            if top in _EXCLUDE_TOP:
                continue
            if rel in seen:
                continue
            seen.add(rel)
            yield full, rel


def scan_md_file(path: str, rel_path: str):
    """Round-7 task T-B6b: scan a markdown file for forbidden claim phrases.

    Phrase matching is case-insensitive so that ``"strict o(n)"`` catches
    ``"Strict O(N)"``, ``"STRICT O(N)"``, etc.  A per-(file, phrase)
    allowlist (``MD_PHRASE_ALLOWLIST``) exempts legitimate code identifiers
    or historical quotes in non-exempt files.
    """
    violations = []
    if rel_path in MD_ALLOWLIST:
        return violations
    if rel_path.startswith(MD_DIR_ALLOWLIST):
        return violations
    try:
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line_lower = line.lower()
                for phrase in MD_FORBIDDEN:
                    phrase_lower = phrase.lower()
                    if phrase_lower not in line_lower:
                        continue
                    if (rel_path, phrase_lower) in MD_PHRASE_ALLOWLIST:
                        continue
                    violations.append((rel_path, lineno, phrase, line.rstrip()))
                    break
    except (OSError, UnicodeDecodeError):
        pass
    return violations


def main():
    all_violations = []
    for d in SCAN_DIRS:
        dir_path = os.path.join(REPO_ROOT, d)
        if not os.path.isdir(dir_path):
            continue
        for root, _dirs, files in os.walk(dir_path):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, REPO_ROOT).replace(os.sep, "/")
                all_violations.extend(scan_file(full, rel))

    # Round-7 task T-B6b: markdown claim-phrase pass.
    for full, rel in _expand_md_globs():
        all_violations.extend(scan_md_file(full, rel))

    if all_violations:
        for rel, lineno, phrase, line in all_violations:
            print(f"{rel}:{lineno}: forbidden phrase '{phrase}': {line}")
        print(f"\n{len(all_violations)} violation(s) found.")
        sys.exit(1)
    print("lint_claims: no forbidden vocabulary found.")
    sys.exit(0)


if __name__ == "__main__":
    main()
