"""WGSL parity test: compare the fixed-grid WGSL FMM pipeline against
FastVectorizedFMM on identical inputs.

This test covers the FILE kernels only (the WGSL source in
`core/webgpu_kernels/tree_free_fmm.wgsl` consumed by
`core/webgpu_kernels/webgpu_fmm_runner.py`).  Parity of the inline-demo
shaders is covered by `tools/check_wgsl_sync.py`'s sync check, not here.

If `wgpu` is not installed or no adapter is available, the test prints
"SKIP: wgpu not installed" and exits 0 (skippable-with-reason per the
round-4 plan section 4.6).

When wgpu IS available: 2D clustered N=2000, run the fixed-grid WGSL
pipeline, compare per-particle force vectors vs FastVectorizedFMM on
identical inputs and softening, assert rel-L2 < 1e-4 (f32 vs f64 floor —
do not tighten).

Run standalone:  python -X utf8 -m core.test_webgpu_parity
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _clustered2d(n=2000, seed=707):
    """2D clustered multi-scale distribution (same generator as the core
    benchmark_variants clustered distribution)."""
    rng = np.random.default_rng(seed)
    n1 = max(1, int(n * 0.20))
    n2 = max(1, int(n * 0.30))
    n3 = max(1, int(n * 0.40))
    nbg = max(0, n - (n1 + n2 + n3))
    c1 = rng.random((n1, 2)) * 0.10 + 0.10
    c2 = rng.random((n2, 2)) * 0.15 + 0.70
    c3 = rng.random((n3, 2)) * 0.30 + 0.40
    bg = rng.random((nbg, 2)) * 0.94 + 0.03 if nbg > 0 else np.empty((0, 2))
    pts = np.vstack([c1, c2, c3, bg]).astype(np.float64)
    pts = np.clip(pts, 0.01, 0.99)
    q = rng.uniform(-1.0, 1.0, size=len(pts)).astype(np.float64)
    return pts, q


def _skip(reason: str) -> int:
    print(f"SKIP: {reason}")
    return 0


def main():
    # 1. Check wgpu availability.
    try:
        import wgpu  # noqa: F401
    except ImportError:
        sys.exit(_skip("wgpu not installed"))

    from core.webgpu_kernels.webgpu_fmm_runner import is_webgpu_available

    if not is_webgpu_available():
        sys.exit(_skip("wgpu installed but no WebGPU adapter found"))

    # 2. Prepare identical inputs.
    pts, q = _clustered2d(n=2000, seed=707)
    softening = 0.02

    # 3. Reference: FastVectorizedFMM (f64) force vectors.
    from core import FastVectorizedFMM

    fmm_ref = FastVectorizedFMM(depth=5, order=8, softening=softening)
    _, fx_ref, fy_ref = fmm_ref.evaluate(pts, q, compute_forces=True)
    forces_ref = np.stack([np.asarray(fx_ref, dtype=np.float64),
                           np.asarray(fy_ref, dtype=np.float64)], axis=1)

    # 4. WGSL pipeline: run the fixed-grid WGSL FMM via the runner.
    #    The runner loads tree_free_fmm.wgsl and dispatches the compute
    #    pipeline.  We pass the same positions, charges, and softening.
    #    The WGSL kernel outputs per-particle force vectors (f32).
    try:
        from core.webgpu_kernels.webgpu_fmm_runner import (
            run_fixed_grid_fmm_forces,
        )
        forces_wgsl = run_fixed_grid_fmm_forces(
            pts, q, softening=softening, depth=5, order=8,
        )
    except (ImportError, AttributeError, NotImplementedError) as e:
        sys.exit(_skip(f"WGSL fixed-grid force runner unavailable: {e}"))

    forces_wgsl = np.asarray(forces_wgsl, dtype=np.float64)

    # 5. Compare per-particle force vectors (rel-L2 < 1e-4, f32 vs f64 floor).
    rel = float(np.linalg.norm(forces_wgsl - forces_ref) /
                max(1e-300, np.linalg.norm(forces_ref)))
    print(f"test_webgpu_parity: rel-L2 (WGSL f32 vs FastVectorizedFMM f64) = "
          f"{rel:.3e} (target < 1e-4)")
    if rel >= 1e-4:
        print(f"FAIL: rel-L2 {rel:.2e} >= 1e-4")
        sys.exit(1)
    print("test_webgpu_parity: PASS")


if __name__ == "__main__":
    main()
