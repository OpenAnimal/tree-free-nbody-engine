"""Single source of truth check for WGSL shaders.

`index.html` carries WGSL shaders inline (as backtick template literals
assigned to `wgsl*Source` JS constants).  `core/webgpu_kernels/` ships
`tree_free_fmm.wgsl` + `adaptive_cgr88.wgsl` consumed by
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
    os.path.join(WGSL_DIR, "adaptive_cgr88.wgsl"),
]


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
        # The fn keyword is at the start of (or partway through) the last
        # prefix line.  Walk backwards over preceding lines that are
        # @-annotations or blank.
        ann_start_line = len(prefix_lines) - 1
        # The last prefix line contains text up to `fn ` -- if it's just
        # whitespace before `fn`, the annotation lines are above.
        last_line_text = prefix_lines[-1] if prefix_lines else ""
        if last_line_text.strip() == "":
            # fn is on its own line; annotations are above
            pass
        # Walk up over @-annotation lines.
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

    # Merge all inline fn blocks into one dict: name -> (normalized, source_label)
    inline_fns = {}
    for src_name, block_text in inline_blocks:
        fns = _extract_fn_blocks(block_text, src_name)
        for name, (norm, _off) in fns.items():
            inline_fns[name] = (norm, src_name)

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

    divergences = []
    for name in shared:
        inline_norm, inline_src = inline_fns[name]
        wgsl_norm, wgsl_src = wgsl_fns[name]
        if inline_norm != wgsl_norm:
            divergences.append((name, inline_src, wgsl_src, inline_norm, wgsl_norm))

    # Report.
    print(f"Inline WGSL functions: {len(inline_fns)}")
    print(f"WGSL file functions:   {len(wgsl_fns)}")
    print(f"Shared (compared):     {len(shared)}")
    print(f"Only in index.html:    {len(only_inline)} (INFO)")
    if only_inline:
        for n in only_inline:
            print(f"  INFO  {n} (index.html only)")
    print(f"Only in .wgsl files:   {len(only_wgsl)} (INFO)")
    if only_wgsl:
        for n in only_wgsl:
            print(f"  INFO  {n} (.wgsl only)")

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
          f"{len(only_inline)} index-only, {len(only_wgsl)} wgsl-only -- PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
