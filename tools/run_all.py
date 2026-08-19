"""One-command verification runner: subprocesses every entry of the round-4
final verification matrix (section 4.9) plus the five `benchmark_variants.py`
files, and prints one PASS/FAIL/SKIP line per item with elapsed wall time.

No new logic is exercised here -- each item is run as a subprocess invoking
the existing command, the tail line is captured, and pass/fail is parsed from
known markers:

  * "tests passed", "PASSED", "[PASS]"  -> PASS (also requires exit code 0)
  * empty-table / nonzero exit code     -> FAIL
  * `core.test_webgpu_parity` is treated as skippable-with-reason: if its
    subprocess prints "SKIP:" and exits 0, the runner reports SKIP (not FAIL).

The runner exits nonzero if any NON-skippable item fails.

Run from repo root:  python -X utf8 tools/run_all.py
"""
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (label, argv, skippable)
# argv is the list passed to subprocess; the python interpreter is prepended.
ITEMS = [
    ("core.test_spatial_index",
     ["-m", "core.test_spatial_index"], False),
    ("core.test_elastic_hash",
     ["-m", "core.test_elastic_hash"], False),
    ("core.test_cgr88_cross_validation",
     ["-m", "core.test_cgr88_cross_validation"], False),
    ("core.test_yukawa3d_fmm",
     ["-m", "core.test_yukawa3d_fmm"], False),
    ("core.test_gaussian2d_fgt",
     ["-m", "core.test_gaussian2d_fgt"], False),
    ("core.test_webgpu_parity",
     ["-m", "core.test_webgpu_parity"], True),
    ("graphics_rendering/test_graphics_rendering.py",
     ["graphics_rendering/test_graphics_rendering.py"], False),
    ("video_streaming_codecs/test_video_streaming.py",
     ["video_streaming_codecs/test_video_streaming.py"], False),
    ("tools/lint_claims.py",
     ["tools/lint_claims.py"], False),
    ("tools/check_wgsl_sync.py",
     ["tools/check_wgsl_sync.py"], False),
    # The five benchmark_variants.py files (core + 4 domain folders).
    ("core/benchmark_variants.py",
     ["core/benchmark_variants.py"], False),
    ("game_mechanics_spatial/benchmark_variants.py",
     ["game_mechanics_spatial/benchmark_variants.py"], False),
    ("graphics_rendering/benchmark_variants.py",
     ["graphics_rendering/benchmark_variants.py"], False),
    ("physics_simulation/ppf_contact_solver_fmm/benchmark_variants.py",
     ["physics_simulation/ppf_contact_solver_fmm/benchmark_variants.py"], False),
    ("video_streaming_codecs/benchmark_variants.py",
     ["video_streaming_codecs/benchmark_variants.py"], False),
]

PASS_MARKERS = ("tests passed", "PASSED", "[PASS]", "no forbidden vocabulary",
                "PASS")
SKIP_MARKERS = ("SKIP:", "SKIP ")


def _classify(label: str, exit_code: int, tail: str, skippable: bool):
    """Return (status, reason). status in {PASS, FAIL, SKIP}."""
    tail_stripped = (tail or "").strip()
    if skippable and exit_code == 0 and any(m in tail_stripped for m in SKIP_MARKERS):
        return "SKIP", tail_stripped
    if exit_code == 0:
        if any(m in tail_stripped for m in PASS_MARKERS):
            return "PASS", ""
        # benchmark_variants.py print a table, not a PASS marker; treat
        # exit 0 + non-empty output as PASS, exit 0 + empty output as FAIL
        # (the "empty-table failure" rule).
        if tail_stripped:
            return "PASS", ""
        return "FAIL", "exit 0 but no output (empty table)"
    return "FAIL", f"exit {exit_code}: {tail_stripped[-200:]}"


def main():
    print("=" * 78)
    print("tools/run_all.py -- round-4 verification matrix")
    print("=" * 78)
    py = sys.executable
    results = []
    any_fail = False
    for label, argv, skippable in ITEMS:
        cmd = [py, "-X", "utf8"] + argv
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=600,
            )
            exit_code = proc.returncode
            # Tail line of stdout (last non-empty line); fall back to stderr.
            out_lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
            err_lines = [ln for ln in (proc.stderr or "").splitlines() if ln.strip()]
            tail = out_lines[-1] if out_lines else (err_lines[-1] if err_lines else "")
        except subprocess.TimeoutExpired:
            exit_code = -1
            tail = "TIMEOUT (>600s)"
        except Exception as e:  # noqa: BLE001
            exit_code = -2
            tail = f"runner error: {e}"
        elapsed = time.perf_counter() - t0
        status, reason = _classify(label, exit_code, tail, skippable)
        if status == "FAIL" and not skippable:
            any_fail = True
        results.append((label, status, elapsed, reason))
        reason_str = f"  -- {reason}" if reason else ""
        print(f"  [{status:<4}] {label:<58} {elapsed:7.2f}s{reason_str}")
        sys.stdout.flush()

    print("-" * 78)
    n_pass = sum(1 for _, s, _, _ in results if s == "PASS")
    n_skip = sum(1 for _, s, _, _ in results if s == "SKIP")
    n_fail = sum(1 for _, s, _, _ in results if s == "FAIL")
    print(f"Summary: {n_pass} PASS, {n_skip} SKIP, {n_fail} FAIL "
          f"of {len(results)} items.")
    if any_fail:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
