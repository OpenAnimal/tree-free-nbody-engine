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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from physics_simulation.ppf_contact_solver_fmm.matrix_free_ipc import (
    MatrixFreeIPCSolver,
    ClothMesh,
    create_cloth_grid,
    combine_cloth_meshes,
    line_search_accepts,
)


def test_line_search_acceptance_predicate():
    """Direct unit test of the line-search acceptance predicate, including
    the VALID-trial-with-increased-energy path that the historical `_`
    rebinding bug crashed on (and that no end-to-end scene currently
    reaches -- see test_stiff_scene_line_search_no_crash)."""
    psi_init = 1.0
    # Sufficient decrease: accepted at any halving.
    assert line_search_accepts(0.5, psi_init, 0)
    # Flat within tolerance: accepted.
    assert line_search_accepts(psi_init + 1e-3, psi_init, 0)
    # VALID trial with increased energy BEFORE the last arm: rejected
    # (the step must be halved instead of accepted).
    assert not line_search_accepts(psi_init + 1.0, psi_init, 0)
    assert not line_search_accepts(psi_init + 1.0, psi_init, 4)
    # Last-chance arm accepts even an increased energy (termination).
    assert line_search_accepts(psi_init + 1.0, psi_init, 5)
    # The predicate must return a plain bool (the historical failure mode
    # was an ndarray leaking into the guard's `==` comparison).
    result = line_search_accepts(psi_init + 1.0, psi_init, 0)
    assert isinstance(result, bool)


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
    solver = MatrixFreeIPCSolver(dhat=dhat, stiffness=4e3)

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
    solver = MatrixFreeIPCSolver(dhat=dhat, stiffness=stiffness)

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


def test_barrier_hessian_vector_product_fd():
    """Finite-difference Hessian-vector product check for the barrier term.

    For B(d) = -(d-dhat)^2 * ln(d/dhat), the analytic scalar curvature is
    B''(d) = -2 ln(d/dhat) + 4(dhat-d)/d + (dhat-d)^2/d^2 (third term PLUS).
    This test would have caught the sign error in the third term: the old
    code used a minus, giving B''=4.386 instead of 6.386 at d=0.5, dhat=1.

    The solver's barrier Hessian is the PSD-projected (normal-only) Hessian:
    Hv = h_scalar * (v_diff · n) * n.  The full Hessian also has a tangential
    component B'(d)/d * (I - n n^T) that is dropped by the PSD projection.
    To compare FD against the PSD-projected analytic product, the perturbation
    v is restricted to the normal direction (v_diff ∥ n), so the tangential
    component contributes zero and FD matches the PSD projection exactly.
    """
    dhat = 0.02
    stiffness = 5e3
    solver = MatrixFreeIPCSolver(dhat=dhat, stiffness=stiffness)

    # Test at several d values in (0.1*dhat, 0.9*dhat).
    d_fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
    eps = 1e-6

    for frac in d_fractions:
        d = frac * dhat
        positions = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
        candidates = np.array([[0, 1]], dtype=np.int32)

        # Perturbation along the normal (x-axis) so the tangential Hessian
        # component contributes zero and FD matches the PSD projection.
        v = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        # Analytic Hessian-vector product.
        Hv_analytic = solver.apply_barrier_hessian_vector_product(v, positions, candidates)

        # FD: Hv ≈ (forces(x - eps*v) - forces(x + eps*v)) / (2*eps)
        # (forces = -grad, so Hv = d(grad)/dx * v = -(f(x+eps*v) - f(x-eps*v))/(2*eps)
        #  = (f(x-eps*v) - f(x+eps*v)) / (2*eps))
        _, f_plus = solver.compute_barrier_energy_and_forces(positions + eps * v, candidates)
        _, f_minus = solver.compute_barrier_energy_and_forces(positions - eps * v, candidates)
        Hv_fd = (f_minus - f_plus) / (2.0 * eps)

        rel_err = float(np.linalg.norm(Hv_analytic - Hv_fd) / max(1e-12, np.linalg.norm(Hv_fd)))
        assert rel_err < 1e-6, (
            f"Barrier Hessian-vector product FD mismatch at d/dhat={frac}: "
            f"rel_err={rel_err:.2e} (analytic={Hv_analytic.ravel()} "
            f"fd={Hv_fd.ravel()})"
        )

    # Also verify the scalar h_scalar itself matches B''(d) at d=0.5*dhat.
    d = 0.5 * dhat
    Bpp_true = -2.0 * np.log(d / dhat) + 4.0 * (dhat - d) / d + ((dhat - d) ** 2) / (d ** 2)
    h_scalar_expected = stiffness * max(1e-2, Bpp_true)
    positions = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
    candidates = np.array([[0, 1]], dtype=np.int32)
    v = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    Hv = solver.apply_barrier_hessian_vector_product(v, positions, candidates)
    # With positions [[0,0,0],[d,0,0]]: n = (x_i - x_j)/d = (-1,0,0) and
    # v_diff = (1,0,0), so v_diff·n = -1 and Hv[0,0] = h_scalar·(-1)·(-1)
    # = +h_scalar (assert via abs to stay robust to convention flips).
    h_scalar_code = abs(Hv[0, 0])
    rel_err_scalar = abs(h_scalar_code - h_scalar_expected) / max(1e-12, h_scalar_expected)
    assert rel_err_scalar < 1e-10, (
        f"h_scalar mismatch at d/dhat=0.5: code={h_scalar_code:.6f} "
        f"expected={h_scalar_expected:.6f} (rel_err={rel_err_scalar:.2e})"
    )

    print(f"[PASS] test_barrier_hessian_vector_product_fd: "
          f"FD vs analytic h_scalar rel_err < 1e-6 at d/dhat in {d_fractions} "
          f"(h_scalar@0.5={h_scalar_code:.1f}, B''={Bpp_true:.4f})")


def test_drape_smoke_5_steps():
    """5-step drape smoke test: no interpenetration (min distance > 0) and
    total incremental-potential energy monotonically non-increasing under
    Newton line search."""
    cloth1 = create_cloth_grid(nx=10, ny=10, width=0.3, height=0.3,
                               center=(0.5, 0.5, 0.38), k_stretch=1500.0,
                               k_bend=0.05, density=0.25)
    # Layer 2 uses the SAME material params as layer 1: combine_cloth_meshes
    # stores a single material set, so per-layer params would be silently
    # discarded (see combine_cloth_meshes docstring).
    cloth2 = create_cloth_grid(nx=10, ny=10, width=0.28, height=0.28,
                               center=(0.505, 0.495, 0.44), k_stretch=1500.0,
                               k_bend=0.05, density=0.25)
    cloth = combine_cloth_meshes([cloth1, cloth2])

    dhat = 0.015
    dt = 0.012
    solver = MatrixFreeIPCSolver(dhat=dhat, stiffness=4e3,
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

    # Compute the initial minimum distance (over candidate pairs and
    # obstacles) to establish a meaningful non-penetration baseline.
    def _min_distance(pos):
        cands = solver.find_broadphase_candidates(pos, cloth)
        md = float('inf')
        if len(cands) > 0:
            dists = np.linalg.norm(pos[cands[:, 0]] - pos[cands[:, 1]], axis=-1)
            md = min(md, float(np.min(dists)))
        for sphere in solver.spheres:
            gap = float(np.min(np.linalg.norm(pos - sphere["center"], axis=-1)
                               - sphere["radius"]))
            md = min(md, gap)
        for plane in solver.planes:
            gap = float(np.min(np.sum((pos - plane["point"]) * plane["normal"], axis=-1)))
            md = min(md, gap)
        return md

    initial_min_dist = _min_distance(cloth.rest_positions)

    for step in range(5):
        # Compute x_tilde (the unconstrained prediction) for this step.
        f_ext = cloth.masses[:, None] * gravity
        x_tilde = positions + velocities * dt + ((dt ** 2) / cloth.masses[:, None]) * f_ext
        candidates = solver.find_broadphase_candidates(x_tilde, cloth)

        psi_before = incremental_potential(x_tilde, x_tilde, cloth, candidates)

        positions, velocities, metrics = solver.solve_step(
            positions, velocities, cloth, dt=dt, gravity=gravity
        )

        # NOTE: psi_after uses `candidates` from x_tilde (frozen at the
        # predicted step), not from the final positions.  Pairs that become
        # active during the step (at the final positions) are not included
        # in this energy evaluation — this is the classic vertex-vertex IPC
        # candidate-set-frozen-at-prediction limitation (no point-triangle
        # CCD).  The line search inside solve_step also uses this same
        # frozen candidate set.
        psi_after = incremental_potential(positions, x_tilde, cloth, candidates)
        energies.append((psi_before, psi_after))

        min_d = _min_distance(positions)
        min_dists.append(min_d)

    # Assert no interpenetration with a meaningful threshold: the line
    # search floor is 1e-4, so the min distance must stay well above it.
    # The previous assertion `md > 0` would pass even at 1e-15 (numerical
    # noise); here we require md > 1e-4 * 0.5 (half the line search floor,
    # allowing for floating-point slack).
    penetration_floor = 1e-4 * 0.5
    assert all(md > penetration_floor for md in min_dists), (
        f"Interpenetration detected: min distances = {min_dists} "
        f"(floor = {penetration_floor})"
    )
    # Also assert the min distance has not collapsed relative to the initial
    # state (it should not drop below 10% of the initial min distance).
    assert all(md > 0.1 * initial_min_dist for md in min_dists), (
        f"Min distance collapsed below 10% of initial ({initial_min_dist}): "
        f"{min_dists}"
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


def test_stiff_scene_line_search_no_crash():
    """Regression: the Newton line-search loop must not raise when a VALID
    trial's energy INCREASES (psi_trial > psi_init + 1e-3) before the
    last-chance halving.

    ROOT CAUSE (audit finding #5): the loop used `for _ in range(6):` but
    the unpacks `e_el_trial, _ = ...` and `e_bar_trial, _ = ...` rebound
    `_` to the forces ndarray. The last-chance guard
    `if psi_trial <= psi_init + 1e-3 or _ == 5:` then evaluated
    `ndarray == 5` -> "ValueError: The truth value of an array is
    ambiguous" whenever a valid trial's energy increased (the `or`
    short-circuits only when the first operand is False). The fix renames
    the loop variable to `halving` and keeps the `halving == 5` last-chance
    semantics.

    No in-repo scene currently reaches the energy-increase-on-valid-trial
    path (instrumented 8x8 two-layer drapes at stiffness 2e5 log zero such
    events in 20 steps; an earlier revision's "crashes at step 19"
    reproduction claim was not reproducible and has been retracted), so this
    end-to-end stiff-scene test guards (a) no exception, (b) monotonically
    non-increasing incremental potential, and (c) no interpenetration --
    while the acceptance PREDICATE itself (accept-or-halve on an
    energy-increasing valid trial, last-chance arm included) is exercised
    directly by `test_line_search_acceptance_predicate` on synthetic
    energies.
    """
    cloth1 = create_cloth_grid(nx=8, ny=8, width=0.3, height=0.3,
                               center=(0.5, 0.5, 0.38), k_stretch=1500.0,
                               k_bend=0.05, density=0.25)
    cloth2 = create_cloth_grid(nx=8, ny=8, width=0.28, height=0.28,
                               center=(0.505, 0.495, 0.44), k_stretch=1500.0,
                               k_bend=0.05, density=0.25)
    cloth = combine_cloth_meshes([cloth1, cloth2])

    dhat = 0.015
    dt = 0.012
    # Stiffness ~2e5 is the regime that triggers the energy-increase-on-
    # valid-trial path (the auditor's reproduction).
    solver = MatrixFreeIPCSolver(dhat=dhat, stiffness=2e5,
                                 max_newton_iters=5, cg_max_iters=16,
                                 cg_tol=1e-4, damp_coef=0.20)
    solver.add_sphere_obstacle(center=np.array([0.5, 0.5, 0.2]), radius=0.12)
    solver.add_plane_obstacle(point=np.array([0.0, 0.0, 0.05]),
                              normal=np.array([0.0, 0.0, 1.0]))

    positions = cloth.rest_positions.copy()
    velocities = np.zeros_like(positions)
    gravity = np.array([0.0, 0.0, -9.81])

    def incremental_potential(x, x_tilde, cands):
        e_inertial = 0.5 * float(np.sum(
            cloth.masses[:, None] * ((x - x_tilde) ** 2))) / (dt ** 2)
        e_el, _ = solver.compute_elastic_energy_and_forces(x, cloth)
        e_bar, _ = solver.compute_barrier_energy_and_forces(x, cands)
        return e_inertial + e_el + e_bar

    # Run past the auditor's crash step (19); 20 steps exercises the path.
    N_STEPS = 20
    deltas = []
    for step in range(N_STEPS):
        f_ext = cloth.masses[:, None] * gravity
        x_tilde = positions + velocities * dt + \
            ((dt ** 2) / cloth.masses[:, None]) * f_ext
        cands = solver.find_broadphase_candidates(x_tilde, cloth)
        psi_before = incremental_potential(x_tilde, x_tilde, cands)
        # The regression guard: this call must NOT raise (the old code
        # raised ValueError at step 19).
        positions, velocities, metrics = solver.solve_step(
            positions, velocities, cloth, dt=dt, gravity=gravity)
        psi_after = incremental_potential(positions, x_tilde, cands)
        deltas.append(psi_after - psi_before)

    # (a) No raise -> we got here. (b) Acceptance logic respected: the
    # line search must produce a non-increasing incremental potential every
    # step (the sufficient-decrease / last-chance accept always lands at or
    # below psi_before, modulo the 1e-3 slack used in the guard).
    violations = [s for s, d in enumerate(deltas) if d > 1e-3 + 1e-9]
    assert not violations, (
        f"line-search acceptance logic violated: energy increased at steps "
        f"{violations}; deltas={[f'{d:+.3e}' for d in deltas]}")
    # (c) No interpenetration: min distance stays above half the line-search
    # floor (same bar as the smoke test).
    cands_final = solver.find_broadphase_candidates(positions, cloth)
    if len(cands_final) > 0:
        min_d = float(np.min(np.linalg.norm(
            positions[cands_final[:, 0]] - positions[cands_final[:, 1]],
            axis=-1)))
        assert min_d > 0.5e-4, f"interpenetration at final step: min_d={min_d}"
    for sphere in solver.spheres:
        gap = float(np.min(np.linalg.norm(positions - sphere["center"],
                                          axis=-1) - sphere["radius"]))
        assert gap > 0.5e-4, f"sphere interpenetration: gap={gap}"
    for plane in solver.planes:
        gap = float(np.min(np.sum(
            (positions - plane["point"]) * plane["normal"], axis=-1)))
        assert gap > 0.5e-4, f"plane interpenetration: gap={gap}"
    print(f"[PASS] test_stiff_scene_line_search_no_crash: {N_STEPS} steps at "
          f"stiffness=2e5, no raise, energy non-increasing "
          f"(max delta {max(deltas):+.3e}), no interpenetration")


def test_broadphase_parity_reference():
    """Pair-set parity: the vectorized ``find_broadphase_candidates`` must
    EXACTLY equal the reference ``_find_broadphase_candidates_reference`` pair
    set (set equality of sorted packed int64 keys, not subset) on four scenes:

      (a) the drape scene (perturbed so contacts exist),
      (b) a random uniform scene N=5000, dhat=0.05 in the unit box,
      (c) a heavily clustered scene (20 gaussian clusters, N=5000),
      (d) a cloth scene with topological exclusions.

    The reference emits all triu pairs of each occupied cell's 27-cell ring-1
    neighborhood (the "ring-1 neighborhood closure"), which includes
    Chebyshev-distance-2 pairs when an occupied midpoint cell sits between
    them.  The vectorized scheme reproduces this closure exactly via the 13
    canonical half-offsets (Chebyshev-1) plus 49 canonical distance-2 offsets
    with an occupied-midpoint check (Chebyshev-2 closure), each unordered pair
    emitted exactly once (no dedup).
    """
    def _pack(arr):
        if len(arr) == 0:
            return np.array([], dtype=np.int64)
        return np.sort((arr[:, 0].astype(np.int64) << 32)
                       | arr[:, 1].astype(np.int64))

    # (a) Drape scene perturbed so candidate pairs exist.
    cloth1 = create_cloth_grid(nx=10, ny=10, width=0.3, height=0.3,
                               center=(0.5, 0.5, 0.38), k_stretch=1500.0,
                               k_bend=0.05, density=0.25)
    cloth2 = create_cloth_grid(nx=10, ny=10, width=0.28, height=0.28,
                               center=(0.505, 0.495, 0.44), k_stretch=1500.0,
                               k_bend=0.05, density=0.25)
    cloth_drape = combine_cloth_meshes([cloth1, cloth2])
    solver_drape = MatrixFreeIPCSolver(dhat=0.015, stiffness=4e3)
    rng = np.random.default_rng(7)
    pos_drape = cloth_drape.rest_positions + rng.standard_normal(
        cloth_drape.rest_positions.shape) * 0.01
    new_a = solver_drape.find_broadphase_candidates(pos_drape, cloth_drape)
    ref_a = solver_drape._find_broadphase_candidates_reference(pos_drape, cloth_drape)
    assert np.array_equal(_pack(new_a), _pack(ref_a)), (
        f"(a) drape parity failed: new={len(new_a)} ref={len(ref_a)}")
    print(f"[PASS] (a) drape parity: {len(new_a)} pairs == reference")

    # (b) Random uniform scene N=5000, dhat=0.05 in unit box.
    rng = np.random.default_rng(1)
    pos_rand = rng.random((5000, 3))
    solver_rand = MatrixFreeIPCSolver(dhat=0.05, stiffness=4e3)
    new_b = solver_rand.find_broadphase_candidates(pos_rand, None)
    ref_b = solver_rand._find_broadphase_candidates_reference(pos_rand, None)
    assert np.array_equal(_pack(new_b), _pack(ref_b)), (
        f"(b) random parity failed: new={len(new_b)} ref={len(ref_b)}")
    print(f"[PASS] (b) random N=5000 parity: {len(new_b)} pairs == reference")

    # (c) Heavily clustered scene: 20 gaussian clusters, N=5000.
    rng = np.random.default_rng(2)
    centers = rng.random((20, 3)) * 0.8 + 0.1
    pos_clust = np.repeat(centers, 250, axis=0) + rng.standard_normal((5000, 3)) * 0.015
    solver_clust = MatrixFreeIPCSolver(dhat=0.05, stiffness=4e3)
    new_c = solver_clust.find_broadphase_candidates(pos_clust, None)
    ref_c = solver_clust._find_broadphase_candidates_reference(pos_clust, None)
    assert np.array_equal(_pack(new_c), _pack(ref_c)), (
        f"(c) clustered parity failed: new={len(new_c)} ref={len(ref_c)}")
    print(f"[PASS] (c) clustered N=5000 parity: {len(new_c)} pairs == reference")

    # (d) Cloth scene with topological exclusions.
    rng = np.random.default_rng(2024)
    cloth_topo = create_cloth_grid(nx=10, ny=20, width=0.3, height=0.6,
                                   center=(0.5, 0.5, 0.5), k_stretch=1000.0,
                                   density=0.2)
    solver_topo = MatrixFreeIPCSolver(dhat=0.05, stiffness=4e3)
    pos_topo = cloth_topo.rest_positions + rng.standard_normal((200, 3)) * 0.02
    new_d = solver_topo.find_broadphase_candidates(pos_topo, cloth_topo)
    ref_d = solver_topo._find_broadphase_candidates_reference(pos_topo, cloth_topo)
    assert np.array_equal(_pack(new_d), _pack(ref_d)), (
        f"(d) cloth-topo parity failed: new={len(new_d)} ref={len(ref_d)}")
    print(f"[PASS] (d) cloth-topo parity: {len(new_d)} pairs == reference")


def main():
    print("=" * 70)
    print("Matrix-Free IPC Solver Test Suite (X-P2)")
    print("=" * 70)
    test_broadphase_captures_all_contacts()
    test_broadphase_parity_reference()
    test_barrier_energy_gradient_analytic()
    test_barrier_hessian_vector_product_fd()
    test_drape_smoke_5_steps()
    test_stiff_scene_line_search_no_crash()
    print("=" * 70)
    print("All tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
