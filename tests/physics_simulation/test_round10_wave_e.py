"""Round-10 Wave E regression + durable gates for physics_simulation.

Regression tests (were RED before the fixes):
  - R10-E1: ClothMesh with zero faces (or zero vertices) crashed with
    IndexError because struct_edges was a 1-D (0,) array; the solver's
    broadphase also crashed on (0, 3) positions at positions.min().

Durable independent-oracle gates promoted from the Round-10 Wave E
physics probe (one-off review scaffolding, since removed):
  - elastic energy gradient vs central finite differences (folded state)
  - bending stencil annihilates flat configurations under random rigid
    transforms (zero ghost forces)
  - sphere+plane obstacle barrier energy gradient vs finite differences
  - broadphase completeness + reference parity on an exact cell-boundary
    lattice and on coincident duplicate points
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from physics_simulation.ppf_contact_solver_fmm.matrix_free_ipc import (
    MatrixFreeIPCSolver,
    ClothMesh,
    create_cloth_grid,
)


# ---------------------------------------------------------------------------
# R10-E1 regression: degenerate meshes must construct and evaluate
# ---------------------------------------------------------------------------

def test_cloth_mesh_triangle_free_constructs_and_evaluates():
    """A mesh with vertices but no triangles (no structural edges, no hinges)
    must construct and evaluate to zero energy/forces instead of crashing
    with `IndexError: too many indices for array` (struct_edges was 1-D)."""
    pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    mesh = ClothMesh(pts, np.empty((0, 3), dtype=np.int32),
                     k_stretch=100.0, k_bend=0.01, density=0.2)
    assert mesh.struct_edges.shape == (0, 2)
    assert mesh.hinges.shape == (0, 4)
    solver = MatrixFreeIPCSolver(dhat=0.02, stiffness=1e3)
    E, F = solver.compute_elastic_energy_and_forces(pts, mesh)
    assert E == 0.0 and np.all(F == 0.0)
    Hv = solver.apply_elastic_hessian_vector_product(
        np.ones_like(pts), pts, mesh)
    assert np.all(Hv == 0.0)
    # masses fall back to the per-vertex minimum
    assert np.allclose(mesh.masses, 1e-4)


def test_cloth_mesh_empty_constructs():
    """A fully empty mesh (0 vertices, 0 triangles) must construct."""
    mesh = ClothMesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.int32))
    assert mesh.num_vertices == 0
    assert mesh.struct_edges.shape == (0, 2)


def test_broadphase_empty_positions():
    """Broadphase on a (0, 3) position array must return an empty candidate
    set instead of raising at positions.min(axis=0)."""
    solver = MatrixFreeIPCSolver(dhat=0.02, stiffness=1e3)
    cand = solver.find_broadphase_candidates(np.empty((0, 3)), None)
    assert cand.shape == (0, 2)


# ---------------------------------------------------------------------------
# Durable oracle gates (were verified green by the Wave E probe)
# ---------------------------------------------------------------------------

def _fd_grad(E, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in range(x.shape[0]):
        for c in range(3):
            xp = x.copy(); xp[i, c] += eps
            xm = x.copy(); xm[i, c] -= eps
            g[i, c] = (E(xp) - E(xm)) / (2 * eps)
    return g


def test_elastic_energy_gradient_finite_differences():
    """Forces must equal -grad(E) by central FD on a folded random state
    (stretch + bending both active). Independent of the analytic derivation."""
    rng = np.random.default_rng(20260822)
    cloth = create_cloth_grid(nx=5, ny=5, width=0.3, height=0.3,
                              center=(0.5, 0.5, 0.5), k_stretch=900.0,
                              k_bend=0.05, density=0.2)
    solver = MatrixFreeIPCSolver(dhat=0.02, stiffness=4e3)
    x = cloth.rest_positions + rng.standard_normal(cloth.rest_positions.shape) * 0.02
    x[12, 2] += 0.06  # fold one vertex out of plane
    _, F = solver.compute_elastic_energy_and_forces(x, cloth)
    g_fd = _fd_grad(lambda p: solver.compute_elastic_energy_and_forces(p, cloth)[0], x)
    rel = np.linalg.norm(F - (-g_fd)) / np.linalg.norm(g_fd)
    assert rel < 1e-6, f"elastic force != -grad E: rel={rel:.3e}"


def test_bending_flat_state_annihilated_under_rigid_transforms():
    """Zero ghost forces: on an IRREGULAR flat hinge mesh, the bending energy
    and forces must vanish in any 3D rotation/translation."""
    rng = np.random.default_rng(7)
    pts2d = np.array([[0.0, 0.0], [1.0, 0.05], [2.1, -0.04], [3.0, 0.02],
                      [0.4, 1.0], [1.5, 1.1], [2.6, 0.95], [3.4, 1.05]])
    tris = np.array([[0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5],
                     [2, 3, 6], [3, 7, 6]], dtype=np.int32)
    flat = ClothMesh(np.column_stack([pts2d, np.zeros(len(pts2d))]), tris,
                     k_stretch=500.0, k_bend=0.1, density=0.2)
    solver = MatrixFreeIPCSolver(dhat=0.02, stiffness=4e3)
    worst = 0.0
    for _ in range(10):
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        ang = rng.uniform(0, 2 * np.pi)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
        x = flat.rest_positions @ R.T + rng.standard_normal(3) * 10
        E, F = solver.compute_elastic_energy_and_forces(x, flat)
        worst = max(worst, abs(E), float(np.max(np.abs(F))))
    assert worst < 1e-8, f"flat-state ghost energy/forces: {worst:.3e}"


def test_obstacle_barrier_gradient_finite_differences():
    """Sphere + plane obstacle barrier forces vs central FD.  Points are
    constructed at ACTIVE gaps in (0.001*dhat, 0.95*dhat) — negative
    (penetrating) or >= dhat gaps are masked out and would make the test
    vacuous."""
    rng = np.random.default_rng(11)
    dhat = 0.02
    solver = MatrixFreeIPCSolver(dhat=dhat, stiffness=5e3)
    solver.add_sphere_obstacle(center=np.array([0.0, 0.0, 0.0]), radius=0.1)
    solver.add_plane_obstacle(point=np.array([0.0, 0.0, -0.05]),
                              normal=np.array([0.2, -0.1, 1.0]))
    # 6 points just OUTSIDE the sphere surface, far from the plane
    dirs = rng.standard_normal((6, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    gap_s = 0.001 * dhat + rng.random(6) * 0.94 * dhat
    near_sphere = dirs * (0.1 + gap_s)[:, None]
    # 6 points just above the plane (positive gap), far from the sphere:
    # base = p0 + gap*n, then tangential offsets ~0.5 in the plane's span
    n = np.array([0.2, -0.1, 1.0]); n /= np.linalg.norm(n)
    e1 = np.cross(n, [1.0, 0.0, 0.0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    gap_p = 0.001 * dhat + rng.random(6) * 0.94 * dhat
    tang = rng.standard_normal((6, 2)) * 0.5
    near_plane = (np.array([0.0, 0.0, -0.05])[None, :] + gap_p[:, None] * n[None, :]
                  + tang[:, 0:1] * e1[None, :] + tang[:, 1:2] * e2[None, :])
    x = np.vstack([near_sphere, near_plane])
    empty = np.empty((0, 2), dtype=np.int32)
    E, F = solver.compute_barrier_energy_and_forces(x, empty)
    assert E > 0, "obstacle barrier contributed nothing (test vacuous)"
    g_fd = _fd_grad(
        lambda p: solver.compute_barrier_energy_and_forces(p, empty)[0], x,
        eps=1e-7)
    rel = np.linalg.norm(F - (-g_fd)) / np.linalg.norm(g_fd)
    assert rel < 1e-4, f"obstacle barrier force != -grad E: rel={rel:.3e}"


def _all_pairs_active(pos, dhat):
    N = len(pos)
    return {(i, j) for i in range(N) for j in range(i + 1, N)
            if 1e-9 < float(np.linalg.norm(pos[i] - pos[j])) < dhat}


def test_broadphase_exact_boundary_lattice_complete():
    """Adversarial scene: every coordinate is an exact multiple of dhat/2
    (cell boundaries). The candidate set must be a superset of all true
    contact pairs and exactly match the CellIndex reference closure."""
    g = np.arange(-6, 7)
    dhat = 0.05
    X, Y, Z = np.meshgrid(g * 0.5 * dhat, g * 0.5 * dhat, g * 0.5 * dhat)
    lat = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    solver = MatrixFreeIPCSolver(dhat=dhat, stiffness=1e3)
    cand = set(map(tuple, solver.find_broadphase_candidates(lat, None).tolist()))
    bf = _all_pairs_active(lat, dhat)
    missed = bf - cand
    assert not missed, f"broadphase missed {len(missed)} boundary pairs"
    ref = set(map(tuple,
                  solver._find_broadphase_candidates_reference(lat, None).tolist()))
    assert cand == ref, (
        f"parity broken on boundary lattice: only_new={len(cand - ref)} "
        f"only_ref={len(ref - cand)}")


def test_broadphase_coincident_duplicates():
    """25 coincident + 2 nearby points: every coincident pair must be a
    candidate (same cell), no duplicates emitted, reference parity holds."""
    dhat = 0.05
    pos = np.array([[0.1, 0.1, 0.1]] * 25
                   + [[0.1 + 0.04, 0.1, 0.1], [0.1, 0.1 + 0.045, 0.1]])
    solver = MatrixFreeIPCSolver(dhat=dhat, stiffness=1e3)
    arr = solver.find_broadphase_candidates(pos, None)
    cand = set(map(tuple, arr.tolist()))
    assert len(cand) == len(arr), "duplicate pairs emitted"
    # all C(25,2) coincident pairs plus the two cross pairs must be present
    for i in range(25):
        for j in range(i + 1, 25):
            assert (i, j) in cand
    ref = set(map(tuple,
                  solver._find_broadphase_candidates_reference(pos, None).tolist()))
    assert cand == ref


def test_barrier_coincident_pair_no_nan():
    """A coincident non-topological pair (d=0) is masked out of the barrier:
    zero energy, zero forces, no NaN."""
    solver = MatrixFreeIPCSolver(dhat=0.02, stiffness=5e3)
    x = np.array([[0.3, 0.3, 0.3], [0.3, 0.3, 0.3]])
    E, F = solver.compute_barrier_energy_and_forces(
        x, np.array([[0, 1]], dtype=np.int32))
    assert np.isfinite(E) and E == 0.0
    assert np.all(np.isfinite(F)) and np.all(F == 0.0)


def test_zero_contact_scene_energy_consistent_step():
    """Zero-contact scene with injected random velocities: one implicit step
    must reduce the independently computed incremental potential."""
    rng = np.random.default_rng(3)
    cloth = create_cloth_grid(nx=4, ny=4, width=0.2, height=0.2,
                              center=(0.5, 0.5, 0.5))
    solver = MatrixFreeIPCSolver(dhat=0.005, stiffness=1e3)  # tiny dhat
    p = cloth.rest_positions.copy()
    v = rng.standard_normal(p.shape) * 0.05
    dt = 0.012
    grav = np.array([0.0, 0.0, -9.81])
    x_t = p + v * dt + dt * dt * grav

    def psi(xx):
        return (0.5 * np.sum(cloth.masses[:, None] * (xx - x_t) ** 2) / dt ** 2
                + solver.compute_elastic_energy_and_forces(xx, cloth)[0])

    psi0 = psi(x_t)
    pn, _, _ = solver.solve_step(p, v, cloth, dt=dt, gravity=grav)
    assert psi(pn) < 0.2 * psi0, (
        f"zero-contact step failed to minimize psi: {psi0:.3e} -> {psi(pn):.3e}")
    # barrier is exactly zero with no candidates
    E, F = solver.compute_barrier_energy_and_forces(
        pn, np.empty((0, 2), dtype=np.int32))
    assert E == 0.0 and np.all(F == 0.0)
