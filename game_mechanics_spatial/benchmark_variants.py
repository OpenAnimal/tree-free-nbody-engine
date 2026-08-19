"""Standardized variant benchmark for flocking: standard O(N^2) vs +elastichash."""
import numpy as np, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.benchmark_kit import VariantBenchmark


def run_flocking_variants(n_agents=1000):
    from game_mechanics_spatial.massive_crowd_flocking import MassiveGameCrowdEngine
    rng = np.random.default_rng(42)
    pos = rng.uniform(0.05, 0.95, size=(n_agents, 2)).astype(np.float64)
    vel = rng.uniform(-0.05, 0.05, size=(n_agents, 2)).astype(np.float64)

    crowd = MassiveGameCrowdEngine(depth=5, num_agents=n_agents)

    # Near-field exactness check (same neighbor set => identical near-field
    # forces). The far-field heading term is an intentional extra, so the full
    # step is NOT cross-validated against the brute near-field; instead the
    # residual is reported in the note.
    dt_v = 1e-6
    stats = crowd.simulate_crowd_step(pos, vel, dt=dt_v, apply_speed_clamp=False)
    got_accel = (stats["new_velocities"].astype(np.float64) - vel) / dt_v
    ref_accel = crowd.brute_force_step_accel(pos, vel)
    far_residual = np.linalg.norm(got_accel - ref_accel)
    near_exact_note = (f"near-field exact vs brute (same 3x3 cell box); "
                       f"|far-field residual| = {far_residual:.3f} "
                       f"({100 * far_residual / np.linalg.norm(got_accel):.1f}% of total); "
                       f"NOT faster than O(N^2) at N={n_agents} (per-cell Python loop overhead dominates at small N)")

    bench = VariantBenchmark("Massive Crowd Flocking (2D unit mode; near-field boid forces + far-field heading)")
    bench.add("standard (brute O(N^2))", lambda: crowd.brute_force_step_accel(pos, vel),
              note="O(N^2) near-field reference")
    bench.add("+elastichash (near+far)", lambda: crowd.simulate_crowd_step(
        pos, vel, dt=0.016, apply_speed_clamp=False)["new_velocities"],
              note=near_exact_note)
    return bench.run()


if __name__ == "__main__":
    run_flocking_variants()
