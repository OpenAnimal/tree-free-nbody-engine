"""
Test suite for the Matrix-Free IPC cloth solver (physics_simulation).

Covers the three areas from Round-7 plan task X-P2:
  1. Broadphase set-equality vs brute force on a 200-vertex random cloth patch.
  2. Analytic 2-point barrier energy / gradient check.
  3. 5-step drape smoke test: min distance > 0, energy monotone decrease.

Run:  python -X utf8 physics_simulation/test_matrix_free_ipc.py
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics_simulation.ppf_contact_solver_fmm.matrix_free_ipc import (
    MatrixFreeIPCSolver,
    ClothMesh,
    create_cloth_grid,
    combine_cloth_meshes,
)


def _brute_force_pairs(positions, cloth, dhat):
    """All (i<j) vertex pairs with dhat > dist > 1e-9, excluding topo neighbors."""
    N = len(positions)
    pairs = set()
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            if d < dhat and d > 1e-9:
                k64 = (i << 32) | j
                if k64 not in cloth.topo_exclusion_set:
                    pairs.add((i, j))
    return pairs


def test_broadphase_captures_all_contacts():
    """Broadphase candidate set must be a superset of all brute-force contact
    pairs (dist < dhat) on a 200-vertex random cloth patch."""
    rng = np.random.default_rng(2024)
    # 200-vertex cloth grid (10 x 20)
    cloth = create_cloth_grid(nx=10, ny=20, width=0.3, height=0.6,
                              center=(0.5, 0.5, 0.5), k_stretch=1000.0, density=0.2)
    assert cloth.num_vertices == 200

    dhat = 0.05
    solver = MatrixFreeIPCSolver(dhat=dhat, stiffness=4e3, cell_size=0.05)

    # Randomly perturb positions to create close pairs across the mesh.
    positions = cloth.rest_positions + rng.standard_normal((200, 3)) * 0.02

    candidates = solver.find_broadphase_candidates(positions, cloth)
    cand_set = set(map(tuple, candidates.tolist()))
    bf_pairs = _brute_force_pairs(positions, cloth, dhat)

    missed = bf_pairs - cand_set
    assert len(missed) == 0, (
        f"Broadphase missed {len(missed)} brute-force contact pairs: {sorted(missed)[:5]}"
    )
    # Also verify that active broadphase pairs (dist < dhat) exactly equal brute force.
    if len(candidates) > 0:
        dists = np.linalg.norm(positions[candidates[:, 0]] - positions[candidates[:, 1]], axis=-1)
        active = set(map(tuple, candidates[dists < dhat].tolist()))
        assert active == bf_pairs, (
            f"Active broadphase pairs != brute force: "
            f"extra={active - bf_pairs}, missing={bf_pairs - active}"
        )
    print(f"[PASS] test_broadphase_captures_all_contacts: "
          f"bf={len(bf_pairs)} cands={len(cand_set)} missed=0")


def test_barrier_energy_gradient_analytic():
    """Analytic 2-point barrier energy and gradient check."""
    dhat = 0.02
    stiffness = 5e3
    solver = MatrixFreeIPCSolver(dhat=dhat, stiffness=stiffness, cell_size=0.02)

    # Two points at a known distance d < dhat.
    d = 0.012
    x_i = np.array([0.0, 0.0, 0.0])
    x_j = np.array([d, 0.0, 0.0])
    positions = np.stack([x_i, x_j])
    candidates = np.array([[0, 1]], dtype=np.int32)

    E, forces = solver.compute_barrier_energy_and_forces(positions, candidates)

    # Analytic energy: E = -kappa * (d - dhat)^2 * ln(d / dhat)
    E_analytic = -stiffness * (d - dhat) ** 2 * np.log(d / dhat)
    assert abs(E - E_analytic) < 1e-10, f"Energy mismatch: {E} vs {E_analytic}"

    # Analytic gradient: g = dE/dd = -kappa * [2*(d-dhat)*ln(d/dhat) + (d-dhat)^2/d]
    g = -stiffness * (2.0 * (d - dhat) * np.log(d / dhat) + (d - dhat) ** 2 / d)
    # Force on i: f_i = -g * (x_i - x_j) / d = -g * (-1, 0, 0) = (g, 0, 0)
    # Force on j: f_j = +g * (x_i - x_j) / d = +g * (-1, 0, 0) = (-g, 0, 0)
    f_i_analytic = np.array([-g * (-1.0), 0.0, 0.0])  # = (g, 0, 0)
    f_j_analytic = np.array([+g * (-1.0), 0.0, 0.0])  # = (-g, 0, 0)

    assert np.allclose(forces[0], f_i_analytic, atol=1e-10), (
        f"Force on vertex 0 mismatch: {forces[0]} vs {f_i_analytic}"
    )
    assert np.allclose(forces[1], f_j_analytic, atol=1e-10), (
        f"Force on vertex 1 mismatch: {forces[1]} vs {f_j_analytic}"
    )

    # Sanity: energy should be positive (barrier pushes apart), repulsive forces.
    # Vertex 0 at x=0 is pushed left (-x) away from vertex 1 at x=d>0.
    assert E > 0, f"Barrier energy should be positive, got {E}"
    assert forces[0, 0] < 0, f"Force on vertex 0 should be repulsive (-x), got {forces[0]}"
    assert forces[1, 0] > 0, f"Force on vertex 1 should be repulsive (+x), got {forces[1]}"

    print(f"[PASS] test_barrier_energy_gradient_analytic: "
          f"E={E:.6e} (analytic={E_analytic:.6e}) "
          f"f_i={forces[0]} f_j={forces[1]}")


def test_drape_smoke_5_steps():
    """5-step drape smoke test: no interpenetration (min distance > 0) and
    total incremental-potential energy monotonically non-increasing under
    Newton line search."""
    cloth1 = create_cloth_grid(nx=10, ny=10, width=0.3, height=0.3,
                               center=(0.5, 0.5, 0.38), k_stretch=1500.0,
                               k_bend=0.05, density=0.25)
    cloth2 = create_cloth_grid(nx=10, ny=10, width=0.28, height=0.28,
                               center=(0.505, 0.495, 0.44), k_stretch=1400.0,
                               k_bend=0.04, density=0.22)
    cloth = combine_cloth_meshes([cloth1, cloth2])

    dhat = 0.015
    dt = 0.012
    solver = MatrixFreeIPCSolver(dhat=dhat, stiffness=4e3, cell_size=0.03,
                                 max_newton_iters=5, cg_max_iters=16,
                                 cg_tol=1e-4, damp_coef=0.20)
    solver.add_sphere_obstacle(center=np.array([0.5, 0.5, 0.2]), radius=0.12)
    solver.add_plane_obstacle(point=np.array([0.0, 0.0, 0.05]),
                              normal=np.array([0.0, 0.0, 1.0]))

    positions = cloth.rest_positions.copy()
    velocities = np.zeros_like(positions)
    gravity = np.array([0.0, 0.0, -9.81])

    def incremental_potential(x, x_tilde, cloth, candidates):
        """psi = 0.5 * ||M^{1/2}(x - x_tilde)||^2 / dt^2 + E_elastic + E_barrier"""
        e_inertial = 0.5 * float(np.sum(
            cloth.masses[:, None] * ((x - x_tilde) ** 2))) / (dt ** 2)
        e_el, _ = solver.compute_elastic_energy_and_forces(x, cloth)
        e_bar, _ = solver.compute_barrier_energy_and_forces(x, candidates)
        return e_inertial + e_el + e_bar

    energies = []
    min_dists = []
    for step in range(5):
        # Compute x_tilde (the unconstrained prediction) for this step.
        f_ext = cloth.masses[:, None] * gravity
        x_tilde = positions + velocities * dt + ((dt ** 2) / cloth.masses[:, None]) * f_ext
        candidates = solver.find_broadphase_candidates(x_tilde, cloth)

        psi_before = incremental_potential(x_tilde, x_tilde, cloth, candidates)

        positions, velocities, metrics = solver.solve_step(
            positions, velocities, cloth, dt=dt, gravity=gravity
        )

        psi_after = incremental_potential(positions, x_tilde, cloth, candidates)
        energies.append((psi_before, psi_after))

        # Check min distance among candidate pairs (no interpenetration).
        cands = solver.find_broadphase_candidates(positions, cloth)
        if len(cands) > 0:
            dists = np.linalg.norm(positions[cands[:, 0]] - positions[cands[:, 1]], axis=-1)
            min_d = float(np.min(dists))
        else:
            min_d = float('inf')
        min_dists.append(min_d)

        # Check obstacle non-penetration.
        for sphere in solver.spheres:
            gap = float(np.min(np.linalg.norm(positions - sphere["center"], axis=-1)
                               - sphere["radius"]))
            min_d = min(min_d, gap)
        for plane in solver.planes:
            gap = float(np.min(np.sum((positions - plane["point"]) * plane["normal"], axis=-1)))
            min_d = min(min_d, gap)
        min_dists[-1] = min_d

    # Assert no interpenetration.
    assert all(md > 0 for md in min_dists), (
        f"Interpenetration detected: min distances = {min_dists}"
    )

    # Assert energy monotone decrease under Newton line search (within tolerance).
    for step, (psi_b, psi_a) in enumerate(energies):
        assert psi_a <= psi_b + 1e-6, (
            f"Step {step}: energy increased {psi_b:.6e} -> {psi_a:.6e} "
            f"(delta={psi_a - psi_b:.6e})"
        )

    print(f"[PASS] test_drape_smoke_5_steps: "
          f"min_dists={[f'{md:.4f}' for md in min_dists]} "
          f"energy_decreases={all(a <= b + 1e-6 for b, a in energies)}")


def main():
    print("=" * 70)
    print("Matrix-Free IPC Solver Test Suite (X-P2)")
    print("=" * 70)
    test_broadphase_captures_all_contacts()
    test_barrier_energy_gradient_analytic()
    test_drape_smoke_5_steps()
    print("=" * 70)
    print("All tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
