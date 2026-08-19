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

A line is exempt if it starts with "#" (comment). Files in ALLOWLIST are
exempt entirely. Exit code 1 on violations, printing file:line output.
"""
import os
import sys

FORBIDDEN = [
    "O(1) query",
    "Lock-Free CAS",
    "Lock-Free Motion Vector",
    "in O(1) without",
    "Multipole Radiance",
    "MULTIPOLE HARMONICS",
]

# Files exempt from the check (relative to repo root). Start empty.
ALLOWLIST = set()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_DIRS = [
    "apps",
    "game_mechanics_spatial",
    "graphics_rendering",
    "video_streaming_codecs",
    "physics_simulation",
]


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

    if all_violations:
        for rel, lineno, phrase, line in all_violations:
            print(f"{rel}:{lineno}: forbidden phrase '{phrase}': {line}")
        print(f"\n{len(all_violations)} violation(s) found.")
        sys.exit(1)
    print("lint_claims: no forbidden vocabulary found.")
    sys.exit(0)


if __name__ == "__main__":
    main()
