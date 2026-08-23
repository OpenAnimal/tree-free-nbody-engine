"""Standardized variant benchmark for Application 6 (MuJoCo-style proximity).

Variants:
  standard      -- brute-force O(N_terrain) nearest-terrain-point search per
                   footpad probe (the exact closest point + contact force)
  +elastichash  -- the app's compute path: 3x3 funnel-hash neighborhood
                   nearest-point search per probe

The +fmm axis is OMITTED with reason: proximity queries are exact
nearest-point searches, not kernel sums, so multipole math does not apply
(per the app's own header).

Accuracy semantics: a 3x3 neighborhood is a FILTER, not an exact solver.
The correctness check is "every probe's true closest terrain point lies
inside the probed 3x3 neighborhood" (`no missed closest points: True`).
The candidate set need not equal the exact set; the contact force from the
neighborhood-closest point may differ from the global-closest force, and
that residual is reported in the note.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.benchmark_kit import VariantBenchmark


def _terrain_and_probes(n_terrain: int = 2500):
    """Same geometry as app6_mujoco_proximity.generate_terrain_and_robot_contacts."""
    x = np.linspace(0.05, 0.95, int(np.sqrt(n_terrain)))
    y = np.linspace(0.05, 0.95, int(np.sqrt(n_terrain)))
    X, Y = np.meshgrid(x, y)
    Z = 0.2 + 0.08 * np.sin(3 * np.pi * X) * np.cos(3 * np.pi * Y) + 0.04 * np.sin(8 * np.pi * X)
    terrain = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)
    foot_x = np.linspace(0.42, 0.58, 12)
    foot_y = np.linspace(0.45, 0.55, 8)
    FX, FY = np.meshgrid(foot_x, foot_y)
    FZ = np.full_like(FX, 0.24)
    probes = np.stack([FX.ravel(), FY.ravel(), FZ.ravel()], axis=1).astype(np.float64)
    return terrain, probes


def _contact_force(probe, closest, min_dist, k_contact=500.0, d_margin=0.05):
    penetration = max(0.0, d_margin - min_dist)
    normal = (probe - closest) / (min_dist + 1e-6)
    return normal * (k_contact * penetration + 5.0 * np.exp(-min_dist / 0.02))


def _brute_proximity(terrain, probes):
    """Exact O(N_terrain) nearest point + contact force per probe."""
    forces = np.zeros_like(probes)
    depths = np.zeros(len(probes))
    closest_pts = np.zeros_like(probes)
    for i, p in enumerate(probes):
        d = np.linalg.norm(terrain - p, axis=1)
        j = int(np.argmin(d))
        min_dist = float(d[j])
        closest_pts[i] = terrain[j]
        depths[i] = max(0.0, 0.05 - min_dist)
        forces[i] = _contact_force(p, terrain[j], min_dist)
    return forces, depths, closest_pts


def _hash_proximity(terrain, probes, grid_res=16):
    """The app's path: 3x3 CellIndex neighborhood nearest-point search.
    Uses CellIndex (canonical spatial index) instead of raw ElasticHashTable."""
    from core.spatial_index import CellIndex
    cell_index = CellIndex(dims=2, grid_res=grid_res)
    cell_index.build(terrain[:, :2])
    forces = np.zeros_like(probes)
    depths = np.zeros(len(probes))
    closest_pts = np.zeros_like(probes)
    for i, p in enumerate(probes):
        key = cell_index.key_of(p[:2])
        near_idx = cell_index.neighborhood_indices(int(key), ring=1)
        min_dist = 1e9
        closest = p.copy()
        if len(near_idx) > 0:
            t = terrain[near_idx]
            d = np.linalg.norm(t - p, axis=1)
            j = int(np.argmin(d))
            if d[j] < min_dist:
                min_dist = float(d[j])
                closest = t[j]
        closest_pts[i] = closest
        depths[i] = max(0.0, 0.05 - min_dist)
        forces[i] = _contact_force(p, closest, min_dist)
    return forces, depths, closest_pts


def run_app6_variants(n_terrain: int = 2500):
    terrain, probes = _terrain_and_probes(n_terrain=n_terrain)

    # One-off correctness check: every probe's global-closest terrain point
    # must lie inside the probed 3x3 neighborhood (filter completeness).
    _, _, brute_closest = _brute_proximity(terrain, probes)
    _, _, hash_closest = _hash_proximity(terrain, probes)
    no_missed = bool(np.all(np.all(brute_closest == hash_closest, axis=1)))
    f_brute, _, _ = _brute_proximity(terrain, probes)
    f_hash, _, _ = _hash_proximity(terrain, probes)
    force_res = float(np.linalg.norm(f_hash - f_brute))
    note = (f"no missed closest points: {no_missed}; "
            f"|contact-force residual| = {force_res:.3f} "
            f"(same closest point per probe => residual is numerical only); "
            f"+fmm axis omitted (proximity is nearest-point, not a kernel sum)")

    bench = VariantBenchmark(
        f"App 6 -- MuJoCo footpad proximity (n_terrain={n_terrain}, "
        f"{len(probes)} footpad probes, 3D nearest-point search)"
    )
    bench.add(
        "standard (brute O(N))",
        lambda: _brute_proximity(terrain, probes)[0],
        note="exact nearest terrain point per probe",
    )
    bench.add(
        "+elastichash (3x3 neighborhood)",
        lambda: _hash_proximity(terrain, probes)[0],
        note=note,
    )
    return bench.run()


if __name__ == "__main__":
    run_app6_variants()
