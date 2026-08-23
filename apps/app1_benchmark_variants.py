"""Standardized variant benchmark for Application 1 (galaxy collision).

Variants (axes from the repo-wide `core.benchmark_kit` protocol):
  standard      -- exact direct O(N^2) gravitational force summation
                   (the reference used by the app's own validate_against_direct)
  +elastichash  -- near-field-only forces through the elastic-hash CellIndex
                   3x3 neighborhood (near-field exact, far-field SKIPPED --
                   the cheap "hash-truncated" baseline)
  +fmm          -- the app's actual compute path: FastVectorizedFMM (adaptive FMM 2D
                   log multipoles on the funnel-hash cell index)

Accuracy vs `standard` on the per-particle force vector (rel L2). The note
states honestly whether FMM is faster at this N -- no tuning to win.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.benchmark_kit import VariantBenchmark
from core.spatial_index import CellIndex


def _two_galaxies(n_per_galaxy: int = 250, seed: int = 42):
    """Same galaxy geometry as app1_galaxy_collision.run_galaxy_collision."""
    rng = np.random.RandomState(seed)

    def arm(center, velocity, radius=0.14):
        r = np.clip(rng.exponential(scale=radius / 2.5, size=n_per_galaxy), 0.01, radius)
        theta = rng.uniform(0, 2 * np.pi, size=n_per_galaxy)
        theta += 2.0 * np.log(r / radius + 1e-4)
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        v_mag = np.sqrt(1.0 / (r + 0.02)) * 0.04
        vx = velocity[0] - v_mag * np.sin(theta)
        vy = velocity[1] + v_mag * np.cos(theta)
        pos = np.stack([x, y], axis=1)
        vel = np.stack([vx, vy], axis=1)
        return pos, vel

    p1, v1 = arm(np.array([0.35, 0.4]), np.array([0.08, 0.05]))
    p2, v2 = arm(np.array([0.65, 0.6]), np.array([-0.08, -0.05]))
    pos = np.vstack([p1, p2]).astype(np.float64)
    masses = np.ones(len(pos)) / len(pos)
    return pos, masses


def _direct_forces(pos, masses, eps=1e-4):
    """Exact O(N^2) attractive 2D log-gravity force (app1 reference)."""
    delta = pos[:, None, :] - pos[None, :, :]
    r_sq = np.sum(delta * delta, axis=-1) + eps ** 2
    np.fill_diagonal(r_sq, np.inf)
    fx = -np.sum(masses[None, :, None] * delta / r_sq[:, :, None], axis=1)[:, 0]
    fy = -np.sum(masses[None, :, None] * delta / r_sq[:, :, None], axis=1)[:, 1]
    return np.stack([fx, fy], axis=1)


def _hash_near_field_forces(pos, masses, depth=4, eps=1e-4):
    """Near-field-only forces via the elastic-hash CellIndex 3x3 ring-1
    neighborhood. Far-field is SKIPPED (the hash-truncated baseline)."""
    idx = CellIndex(dims=2, grid_res=1 << depth)
    idx.build(pos)
    forces = np.zeros_like(pos)
    for i in range(len(pos)):
        key = idx.key_of(pos[i])
        neigh = idx.neighborhood_indices(key, ring=1)
        neigh = neigh[neigh != i]
        if len(neigh) == 0:
            continue
        d = pos[i] - pos[neigh]
        r_sq = np.sum(d * d, axis=1) + eps ** 2
        forces[i] = -np.sum(masses[neigh, None] * d / r_sq[:, None], axis=0)
    return forces


def _fmm_forces(pos, masses, depth=4, order=6, eps=1e-4):
    """The app's actual path: FastVectorizedFMM forces.

    No force-magnitude clamping (unlike the simulation's
    compute_fmm_gravitational_forces which clamps to 50.0 for stability):
    the benchmark measures raw FMM accuracy vs the unclamped direct
    reference, so clamping only the FMM path would inject a false
    discrepancy on any particle whose direct force exceeds 50.0.
    """
    from core.fast_vectorized_fmm import FastVectorizedFMM
    fmm = FastVectorizedFMM(depth=depth, order=order, softening=eps)
    _, fx, fy = fmm.evaluate(pos, masses, compute_forces=True)
    return np.stack([fx, fy], axis=1)


def run_app1_variants(n_per_galaxy: int = 250):
    pos, masses = _two_galaxies(n_per_galaxy=n_per_galaxy)
    n = len(pos)

    bench = VariantBenchmark(
        f"App 1 -- Galaxy collision 2D log-gravity forces (N={n}, two spirals)"
    )
    bench.add(
        "standard (exact direct)",
        lambda: _direct_forces(pos, masses),
        note="O(N^2) reference (app1 validate_against_direct path)",
    )
    bench.add(
        "+elastichash (near only)",
        lambda: _hash_near_field_forces(pos, masses, depth=4),
        accuracy_vs="standard (exact direct)",
        note="CellIndex ring-1 near-field, far-field SKIPPED (hash-truncated baseline)",
    )
    bench.add(
        "+fmm (FastVectorizedFMM)",
        lambda: _fmm_forces(pos, masses, depth=4, order=6),
        accuracy_vs="standard (exact direct)",
        note="adaptive FMM flat FMM, depth=4 order=6 (the app's compute path)",
    )
    bench.run()

    # Scaling table: show FMM crossover at larger N (Class D honesty —
    # FMM is not faster at small N, but wins decisively past N~1000).
    print("\n  Scaling table (depth=4, order=6):")
    print(f"  {'N':>6s}  {'direct (ms)':>12s}  {'fmm (ms)':>12s}  {'speedup':>8s}  {'rel L2':>10s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*10}")
    import time as _time
    for nper in (250, 500, 1000, 2000):
        p, m = _two_galaxies(n_per_galaxy=nper)
        nn = len(p)
        t0 = _time.perf_counter(); ref = _direct_forces(p, m); td = (_time.perf_counter() - t0) * 1000
        t0 = _time.perf_counter(); fmm = _fmm_forces(p, m, depth=4, order=6); tf = (_time.perf_counter() - t0) * 1000
        rel = float(np.linalg.norm(fmm - ref) / (np.linalg.norm(ref) + 1e-30))
        print(f"  {nn:6d}  {td:12.1f}  {tf:12.1f}  {td/tf:8.2f}x  {rel:10.2e}")


if __name__ == "__main__":
    run_app1_variants()
