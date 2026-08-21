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
    ("core.test_screened_yukawa2d_fmm",
     ["-m", "core.test_screened_yukawa2d_fmm"], False),
    ("core.test_uniform_multilevel_fmm",
     ["-m", "core.test_uniform_multilevel_fmm"], False),
    ("core.test_webgpu_parity",
     ["-m", "core.test_webgpu_parity"], True),
    # T-E1 file-kernel gate: adaptive_cgr88.wgsl compile (16-binding
    # consolidated layout) + counting-sort CSR validation via wgpu-py.
    # Skippable-with-reason like test_webgpu_parity when wgpu is absent.
    ("core.test_adaptive_wgsl_csr",
     ["-m", "core.test_adaptive_wgsl_csr"], True),
    ("core.test_jax_pipeline",
     ["-m", "core.test_jax_pipeline"], True),
    # Round-7 T-F1: neural_ops + bioinformatics + algorithm_theory coverage
    ("neural_ops.test_fmm_neural_ops",
     ["neural_ops/test_fmm_neural_ops.py"], False),
    ("neural_ops.test_neural_ops_advanced",
     ["neural_ops/test_neural_ops_advanced.py"], False),
    ("neural_ops.test_farfield_error",
     ["neural_ops/test_farfield_error.py"], False),
    ("neural_ops.test_kv_cache_recall",
     ["neural_ops/test_kv_cache_recall.py"], False),
    ("bioinformatics.test_sota_modules",
     ["bioinformatics/test_sota_modules.py"], False),
    ("algorithm_theory.test_basic_datatypes_fmm",
     ["algorithm_theory/test_basic_datatypes_fmm.py"], False),
    ("environmental_modeling.test_environmental_suite",
     ["-m", "environmental_modeling.test_environmental_suite"], False),
    ("graphics_rendering/test_graphics_rendering.py",
     ["graphics_rendering/test_graphics_rendering.py"], False),
    ("video_streaming_codecs/test_video_streaming.py",
     ["video_streaming_codecs/test_video_streaming.py"], False),
    ("tools/lint_claims.py",
     ["tools/lint_claims.py"], False),
    ("tools/check_wgsl_sync.py",
     ["tools/check_wgsl_sync.py"], False),
    # The five benchmark_variants.py files (core + 4 domain folders).
    # core/benchmark_variants.py is skippable: it runs the full scaling
    # sweep (direct O(N^2) at N=32000, ~36s) plus the adaptive CGR88 engine
    # (Python tree traversal, ~1s) on every invocation.  In CI / quick-check
    # contexts that only need the lint+sync+unit-test matrix, the SKIP note
    # below documents why it was omitted.  The full BENCHMARKS.md tables are
    # regenerated on demand by running the file directly.
    ("core/benchmark_variants.py",
     ["core/benchmark_variants.py"], True),
    ("game_mechanics_spatial/benchmark_variants.py",
     ["game_mechanics_spatial/benchmark_variants.py"], False),
    ("graphics_rendering/benchmark_variants.py",
     ["graphics_rendering/benchmark_variants.py"], False),
    ("physics_simulation/ppf_contact_solver_fmm/benchmark_variants.py",
     ["physics_simulation/ppf_contact_solver_fmm/benchmark_variants.py"], False),
    ("physics_simulation.test_matrix_free_ipc",
     ["physics_simulation/test_matrix_free_ipc.py"], False),
    ("video_streaming_codecs/benchmark_variants.py",
     ["video_streaming_codecs/benchmark_variants.py"], False),
]

PASS_MARKERS = ("tests passed", "PASSED", "[PASS]", "no forbidden vocabulary",
                "PASS")
SKIP_MARKERS = ("SKIP:", "SKIP ")


def _classify(label: str, exit_code: int, tail: str, skippable: bool):
    """Return (status, reason). status in {PASS, FAIL, SKIP}.

    SKIP-vs-PASS classification is independent of the skippable flag: a
    SKIP marker in the output means SKIP regardless of whether the item is
    skippable.  The skippable flag only controls whether a legitimate SKIP
    is acceptable (non-skippable SKIPs are still reported as SKIP, but they
    set any_fail in main because a non-skippable item should not be
    skipping).  Any non-SKIP failure (nonzero exit, no SKIP marker) sets
    any_fail regardless of the skippable flag.
    """
    tail_stripped = (tail or "").strip()
    has_skip_marker = any(m in tail_stripped for m in SKIP_MARKERS)
    # SKIP classification is independent of skippable: if the output says
    # SKIP and the process exited 0, classify as SKIP.
    if has_skip_marker and exit_code == 0:
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
    print("tools/run_all.py -- round-7 verification matrix (26 items)")
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
        # Any FAIL sets any_fail regardless of skippable (a genuinely failing
        # skippable item with nonzero exit and no SKIP marker is still a
        # failure).  A non-skippable item that prints SKIP and exits 0 is
        # classified SKIP but also sets any_fail (non-skippable items should
        # not be skipping).
        if status == "FAIL":
            any_fail = True
        elif status == "SKIP" and not skippable:
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
    # Document any SKIPs so the report explains why an item was omitted
    # rather than leaving the reader to guess.  Skippable items that printed
    # a SKIP marker and exited 0 are expected; non-skippable SKIPs would
    # have set any_fail above.
    skipped = [(label, reason) for label, status, _, reason in results
               if status == "SKIP"]
    if skipped:
        print("Skipped items (expected — see notes):")
        for label, reason in skipped:
            r = f" -- {reason}" if reason else ""
            print(f"  SKIP  {label}{r}")
    if any_fail:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
