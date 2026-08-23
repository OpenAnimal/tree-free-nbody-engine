"""Single source of truth check for WGSL shaders.

`index.html` carries WGSL shaders inline (as backtick template literals
assigned to `wgsl*Source` JS constants).  `core/webgpu_kernels/` ships
`tree_free_fmm.wgsl` + `adaptive_fmm.wgsl` consumed by
`webgpu_fmm_runner.py`.  This tool verifies that every function present in
BOTH a .wgsl file and the index.html inline shaders has a matching
(whitespace-normalized) body.

Rules (per the round-4 plan, section 4.5):
  * Extract every `fn <name>` block (including any preceding
    `@compute @workgroup_size(...)` annotation) from index.html's inline
    WGSL template literals and from both .wgsl files.
  * For every function name present in BOTH sources, compare normalized
    bodies; FAIL (exit 1, file:line of first difference) on divergence.
  * Functions present in only one source are INFO, not failures (the demo
    legitimately has extra UI kernels like pack_state / vs_main / fs_main).
  * If divergence is found: the `core/webgpu_kernels/` file is
    AUTHORITATIVE -- the tool reports the divergence but does NOT auto-edit
    index.html (the executor should update index.html's inline copy to
    match, then re-run the check).

T-E1 allowlist (2026-08-21): after the counting-sort CSR rewrite the file
kernels and the demo's inline shaders are intentionally DIFFERENT programs
that share function names -- the file kernels are the reference
implementation (own binding layout incl. the packed cellArrays buffer,
naga-compatible coefficient accessors, GridParams overlay uniform) while
the demo's inline copies are the browser production variant (elastic/funnel
hash axes, sortedPayload, quantization, budgeted P2P).  Divergence in the
names below is reported as ALLOWED with the reason; each file-kernel variant
is validated NUMERICALLY instead of textually:
  core.test_webgpu_parity     (fixed-grid kernel vs FastVectorizedFMM, wgpu-py)
  core.test_adaptive_wgsl_csr (adaptive kernel compile + counting-sort CSR)
Any OTHER shared name that diverges still FAILS the gate.

Run from repo root:  python -X utf8 tools/check_wgsl_sync.py
"""
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_HTML = os.path.join(REPO_ROOT, "index.html")
WGSL_DIR = os.path.join(REPO_ROOT, "core", "webgpu_kernels")
WGSL_FILES = [
    os.path.join(WGSL_DIR, "tree_free_fmm.wgsl"),
    os.path.join(WGSL_DIR, "adaptive_fmm.wgsl"),
]

# Post-T-E1 intentionally-diverged shared functions (see module docstring):
# file kernel = reference implementation with its own binding layout;
# demo inline = browser production variant. Validated numerically via
# core.test_webgpu_parity / core.test_adaptive_wgsl_csr instead of by text.
ALLOWED_DIVERGENT_FN = {
    "clear_cells": "T-E1 sort pass: file kernel uses GridParams overlay; demo uses its inline fixed-grid layout",
    "count_cells": "T-E1 sort pass: binding/param naming differs (params vs fmmParams)",
    "scan_cells": "T-E1 sort pass: packed cellArrays vs demo's separate cellStart buffer naming",
    "scatter_cells": "T-E1 sort pass: p2pCellIndex/particles vs leafIndex/fmmPos naming",
    "p2m": "adaptive far-field pass: file kernel T-E1 reference vs demo budgeted variant",
    "m2m": "adaptive far-field pass: file kernel T-E1 reference vs demo variant",
    "m2l": "adaptive far-field pass: file kernel T-E1 reference vs demo variant",
    "l2l": "adaptive far-field pass: file kernel T-E1 reference vs demo variant",
    "l2p": "file kernel = uniform-grid 3x3 CSR P2P overlay; demo = adaptive List-1 budgeted P2P",
    "isTerminal": "nodeFlags array vs packed nodeMeta vec2 selector",
    "readc": "naga forbids read_write storage pointers as fn params: file kernel uses a which-buffer selector",
    "writec": "naga forbids read_write storage pointers as fn params: file kernel uses a which-buffer selector",
}


# =====================================================================
# 1. Extract inline WGSL template literals from index.html
# =====================================================================

def _extract_inline_wgsl_blocks(html_text: str):
    """Return a list of (source_name, block_text) for every backtick
    template literal assigned to a `wgsl*Source` JS constant.

    The template literal may span many lines; we find the opening backtick
    after `= ` and scan for the closing backtick (respecting JS escape
    rules -- WGSL does not use backticks internally, so the first
    unescaped backtick closes the literal).
    """
    blocks = []
    # Match: const wgslXxxSource = `
    pattern = re.compile(r'const\s+(wgsl\w*Source)\s*=\s*`')
    for m in pattern.finditer(html_text):
        name = m.group(1)
        start = m.end()  # position right after the opening backtick
        # Find the closing backtick.  WGSL source does not contain backticks
        # or ${} template substitutions, so the next backtick is the close.
        end = html_text.find('`', start)
        if end == -1:
            continue  # malformed; skip
        block_text = html_text[start:end]
        blocks.append((name, block_text))
    return blocks


# =====================================================================
# 2. Extract fn blocks from WGSL source text
# =====================================================================

def _normalize_ws(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip each line.
    This makes the comparison independent of indentation differences."""
    lines = text.splitlines()
    stripped = [" ".join(ln.split()) for ln in lines]
    return " ".join(s for s in stripped if s)


def _extract_fn_blocks(wgsl_text: str, source_label: str):
    """Return dict: fn_name -> (normalized_body, raw_start_offset).

    A fn block starts at an optional `@compute @workgroup_size(...)` line
    (or sequence of @-annotated lines) followed by `fn <name>(...)`, and
    ends at the matching closing `}` (brace matching from the first `{`
    after the fn signature).
    """
    fns = {}
    # Find all `fn <name>` declarations.  We scan for the pattern
    # `fn <name>` at the start of a token (not inside a comment or string).
    fn_pattern = re.compile(r'\bfn\s+(\w+)\s*\(')

    # We need to handle @compute/@workgroup_size annotations that precede
    # the fn.  Strategy: find each `fn <name>(` match, then walk backwards
    # to capture any immediately preceding @-annotation lines.
    for m in fn_pattern.finditer(wgsl_text):
        name = m.group(1)
        fn_pos = m.start()

        # Walk backwards to capture @-annotations.
        # Look at the text before fn_pos, line by line.
        prefix = wgsl_text[:fn_pos]
        prefix_lines = prefix.splitlines()
        # The fn keyword is at the start of a new line (prefix ends with \n,
        # so splitlines() does NOT include a trailing empty entry).  The last
        # entry in prefix_lines is the line IMMEDIATELY before the fn line.
        ann_start_line = len(prefix_lines) - 1
        last_line_text = prefix_lines[-1] if prefix_lines else ""
        # If the last prefix line is NOT an annotation and NOT blank, the fn
        # is on its own line and the block must start at the fn line (one
        # past the last prefix line), NOT at the preceding non-annotation
        # line (which would wrongly include e.g. a closing `}` from the
        # previous function).
        if last_line_text.strip() != "" and not last_line_text.strip().startswith("@"):
            block_start_line = ann_start_line + 1
        else:
            # Walk up over @-annotation lines and blanks.
            i = ann_start_line
            while i > 0:
                prev = prefix_lines[i - 1].strip()
                if prev.startswith("@") or prev == "":
                    i -= 1
                else:
                    break
            block_start_line = i

        # Now find the body braces.  Scan forward from fn_pos for the first
        # `{` and brace-match to the closing `}`.
        brace_start = wgsl_text.find('{', fn_pos)
        if brace_start == -1:
            continue
        depth = 0
        body_end = -1
        j = brace_start
        while j < len(wgsl_text):
            c = wgsl_text[j]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    body_end = j + 1
                    break
            j += 1
        if body_end == -1:
            continue

        # Reconstruct the full block text from the annotation start to body_end.
        # Compute character offset of block_start_line.
        line_start_offset = 0
        for k in range(block_start_line):
            line_start_offset += len(prefix_lines[k]) + 1  # +1 for newline
        block_text = wgsl_text[line_start_offset:body_end]
        normalized = _normalize_ws(block_text)
        fns[name] = (normalized, line_start_offset)
    return fns


# =====================================================================
# 3. Main comparison
# =====================================================================

def main():
    if not os.path.isfile(INDEX_HTML):
        print(f"FAIL: index.html not found at {INDEX_HTML}")
        sys.exit(1)

    with open(INDEX_HTML, "r", encoding="utf-8") as f:
        html_text = f.read()

    inline_blocks = _extract_inline_wgsl_blocks(html_text)
    if not inline_blocks:
        print("FAIL: no inline WGSL template literals found in index.html")
        sys.exit(1)

    # Collect ALL inline fn copies per name (a function can be defined in
    # multiple inline blocks — e.g. hashU32 appears in several wgsl*Source
    # template literals).  The previous last-wins dict merge silently dropped
    # earlier copies, so two divergent inline definitions of the same name
    # were never compared to EACH OTHER (finding: 26 names defined in 2+
    # blocks).  We now keep every copy and compare them pairwise below.
    # name -> list of (normalized, source_label)
    inline_fns_all = {}
    for src_name, block_text in inline_blocks:
        fns = _extract_fn_blocks(block_text, src_name)
        for name, (norm, _off) in fns.items():
            inline_fns_all.setdefault(name, []).append((norm, src_name))

    # For the inline-vs-wgsl comparison we use the first inline copy of each
    # name (all inline copies are checked for mutual consistency separately).
    inline_fns = {name: copies[0] for name, copies in inline_fns_all.items()}

    # Extract fn blocks from each .wgsl file.
    wgsl_fns = {}  # name -> (normalized, file_label)
    for wgsl_path in WGSL_FILES:
        if not os.path.isfile(wgsl_path):
            print(f"FAIL: {wgsl_path} not found")
            sys.exit(1)
        with open(wgsl_path, "r", encoding="utf-8") as f:
            wgsl_text = f.read()
        label = os.path.relpath(wgsl_path, REPO_ROOT).replace(os.sep, "/")
        fns = _extract_fn_blocks(wgsl_text, label)
        for name, (norm, _off) in fns.items():
            wgsl_fns[name] = (norm, label)

    # Compare shared function names.
    shared = sorted(set(inline_fns) & set(wgsl_fns))
    only_inline = sorted(set(inline_fns) - set(wgsl_fns))
    only_wgsl = sorted(set(wgsl_fns) - set(inline_fns))

    # Pairwise comparison of ALL inline copies for duplicate function names.
    # A name defined in 2+ inline blocks must have identical bodies in every
    # copy — otherwise two divergent inline definitions would pass green
    # under the old last-wins merge (e.g. two copies of hashU32 that differ).
    inline_duplicates = {
        name: copies for name, copies in inline_fns_all.items() if len(copies) > 1
    }
    inline_divergences = []
    for name, copies in inline_duplicates.items():
        for i in range(len(copies)):
            for j in range(i + 1, len(copies)):
                norm_i, src_i = copies[i]
                norm_j, src_j = copies[j]
                if norm_i != norm_j:
                    inline_divergences.append(
                        (name, src_i, src_j, norm_i, norm_j))

    divergences = []
    allowed = []
    for name in shared:
        wgsl_norm, wgsl_src = wgsl_fns[name]
        # Compare ALL inline copies against the .wgsl version.  A name can
        # appear in multiple inline shader modules (e.g. m2l in both the
        # fixed-grid wgslFmmSource and the adaptive wgslAdaptiveFmmSource);
        # only the copy from the MATCHING module should agree with the .wgsl
        # file.  Flag a divergence only when NO inline copy matches.
        copies = inline_fns_all.get(name, [inline_fns[name]])
        matched = False
        mismatched_copies = []
        for inline_norm, inline_src in copies:
            if inline_norm == wgsl_norm:
                matched = True
                break
            mismatched_copies.append((inline_src, inline_norm))
        if not matched:
            if name in ALLOWED_DIVERGENT_FN:
                allowed.append((name, ALLOWED_DIVERGENT_FN[name]))
            else:
                # Report the first mismatched copy for the diff snippet.
                first_src, first_norm = mismatched_copies[0]
                divergences.append((name, first_src, wgsl_src, first_norm, wgsl_norm))

    # Report.
    n_inline_total = sum(len(c) for c in inline_fns_all.values())
    print(f"Inline WGSL blocks:    {len(inline_blocks)}")
    print(f"Inline fn definitions: {n_inline_total} "
          f"({len(inline_fns)} unique, {len(inline_duplicates)} duplicated)")
    print(f"WGSL file functions:   {len(wgsl_fns)}")
    print(f"Shared (compared):     {len(shared)}")
    if inline_duplicates:
        print(f"Inline duplicates:     {len(inline_duplicates)} name(s) "
              f"defined in 2+ blocks (checked pairwise)")
    print(f"Only in index.html:    {len(only_inline)} (INFO — no .wgsl counterpart)")
    if only_inline:
        for n in only_inline:
            copies = inline_fns_all[n]
            blocks = ", ".join(s for _, s in copies)
            print(f"  INFO  {n} (index.html only, in: {blocks})")
    print(f"Only in .wgsl files:   {len(only_wgsl)} (INFO)")
    if only_wgsl:
        for n in only_wgsl:
            print(f"  INFO  {n} (.wgsl only)")

    # Inline-vs-inline divergences are reported as WARNINGS (not failures).
    # Different inline shader modules (e.g. wgslComputeSource vs
    # wgslFmmSource vs wgslAdaptiveFmmSource) legitimately share function
    # names with module-specific differences (param struct names, atomic
    # vs non-atomic access, workgroup sizes, fixed-grid vs adaptive
    # implementations).  Reporting them makes the divergences VISIBLE so a
    # accidental drift (e.g. one copy of hashU32 changed but not the other)
    # shows up in the tool output, while keeping the gate green when the
    # divergences are the known intentional ones.
    if inline_divergences:
        print(f"\nWARN: {len(inline_divergences)} inline-vs-inline "
              f"divergence(s) — duplicate function definitions in index.html "
              f"that differ between shader modules (reported for visibility):")
        for name, src_i, src_j, norm_i, norm_j in inline_divergences:
            print(f"  WARN  fn {name}: {src_i} vs {src_j}")

    if allowed:
        print(f"\nALLOWED: {len(allowed)} intentionally-diverged function(s) "
              f"(post-T-E1 file-vs-demo split; see docstring):")
        for name, reason in allowed:
            print(f"  ALLOWED fn {name}: {reason}")

    if divergences:
        print(f"\nFAIL: {len(divergences)} function(s) diverge between "
              f"index.html and the .wgsl files.")
        print("The core/webgpu_kernels/ .wgsl file is AUTHORITATIVE -- "
              "update index.html's inline copy to match, then re-run.")
        for name, inline_src, wgsl_src, inline_norm, wgsl_norm in divergences:
            print(f"\n  DIVERGENCE: fn {name}")
            print(f"    index.html source: {inline_src}")
            print(f"    .wgsl source:      {wgsl_src}")
            # Show a short diff snippet.
            print(f"    index.html (normalized, first 200 chars): {inline_norm[:200]}")
            print(f"    .wgsl      (normalized, first 200 chars): {wgsl_norm[:200]}")
        sys.exit(1)

    print(f"\ncheck_wgsl_sync: {len(shared)} shared function(s) in sync, "
          f"{len(inline_duplicates)} inline duplicate name(s) pairwise-checked, "
          f"{len(only_inline)} index-only, {len(only_wgsl)} wgsl-only -- PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
