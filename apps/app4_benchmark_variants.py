"""Standardized variant benchmark for Application 4 (elastic-hash boids + 1euro).

Variants:
  standard      -- brute-force O(N^2) near-field boid forces (separation +
                   alignment over all pairs within the interaction radius)
  +elastichash  -- the app's ElasticHashBoidSwarm.step: near-field via the
                   3x3 funnel-hash neighborhood (same neighbor set as brute =>
                   near-field exact) PLUS a far-field per-cell centroid
                   cohesion term that is an intentional extra

The +fmm axis is OMITTED with reason: the boid interaction kernel is a
piecewise near-field rule, not the 2D logarithmic CGR88 kernel.

Accuracy semantics (matching game_mechanics_spatial/benchmark_variants.py):
the near-field is exact (same 3x3 cell box => identical near-field forces),
but the full step is NOT cross-validated against the brute near-field because
the far-field cohesion term is an intentional extra; the residual is reported
in the note.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.benchmark_kit import VariantBenchmark


def _init_state(n_boids: int = 400, seed: int = 42):
    np.random.seed(seed)
    pos = np.random.uniform(0.15, 0.85, size=(n_boids, 2))
    theta = np.random.uniform(0, 2 * np.pi, size=n_boids)
    vel = np.stack([np.cos(theta), np.sin(theta)], axis=1) * 0.05
    return pos, vel


def _brute_near_field_accel(pos, vel, radius=0.05,
                            w_sep=1.8, w_ali=1.0):
    """Brute O(N^2) near-field boid acceleration (separation + alignment)
    over all pairs within `radius`. No far-field cohesion (the intentional
    extra in the hash path)."""
    n = len(pos)
    acc = np.zeros_like(vel)
    for i in range(n):
        diff = pos[i] - pos
        d = np.linalg.norm(diff, axis=1) + 1e-6
        mask = (d < radius) & (np.arange(n) != i)
        if not np.any(mask):
            continue
        sep = np.sum((diff[mask] / (d[mask, None] ** 2)) * 0.001, axis=0)
        ali = (np.mean(vel[mask], axis=0) - vel[i])
        acc[i] = w_sep * sep + w_ali * ali
    return acc


def run_app4_variants(n_boids: int = 400):
    from apps.app4_fmm_boids_1euro import ElasticHashBoidSwarm
    pos0, vel0 = _init_state(n_boids=n_boids)

    swarm = ElasticHashBoidSwarm(n_boids=n_boids, depth=4)
    # Reset to the benchmark initial state (the constructor seeds np.random
    # itself; overwrite so standard and +elastichash see identical inputs).
    swarm.pos = pos0.copy()
    swarm.vel = vel0.copy()

    dt_v = 1e-6
    stats = swarm.step(dt=dt_v)
    got_accel = (stats[1].astype(np.float64) - vel0) / dt_v
    ref_accel = _brute_near_field_accel(pos0, vel0)
    far_res = np.linalg.norm(got_accel - ref_accel)
    note = (f"near-field exact vs brute (same 3x3 cell box); "
            f"|far-field residual| = {far_res:.3f} "
            f"({100 * far_res / max(np.linalg.norm(got_accel), 1e-12):.1f}% of total); "
            f"+fmm axis omitted (not a 2D log kernel)")

    bench = VariantBenchmark(
        f"App 4 -- Elastic-hash boids + 1euro (N={n_boids}, near-field boid rules + far centroid cohesion)"
    )
    bench.add(
        "standard (brute near-field)",
        lambda: _brute_near_field_accel(pos0, vel0),
        note="O(N^2) near-field reference (separation + alignment, no far cohesion)",
    )
    bench.add(
        "+elastichash (near+far+1euro)",
        lambda: ElasticHashBoidSwarm(n_boids=n_boids, depth=4).step(dt=0.016)[1],
        note=note,
    )
    return bench.run()


if __name__ == "__main__":
    run_app4_variants()
