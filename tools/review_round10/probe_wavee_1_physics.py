"""Wave E probe 1: physics_simulation/ppf_contact_solver_fmm deep verification.

Independent oracles:
  - central finite differences for elastic + barrier energies/gradients/Hessians
  - all-pairs O(N^2) contact enumeration (incl. exact cell boundaries, coincident
    points, adversarial lattices)
  - reference CellIndex broadphase parity on adversarial scenes
  - rotation/translation invariance of shell energies
  - zero-contact / degenerate-mesh behavior
Run: python tools/review_round10/probe_wavee_1_physics.py
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
from physics_simulation.ppf_contact_solver_fmm.tetrahedral_surgical_soft_robotics import (
    TetrahedralSoftRoboticsSolver,
)

PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def rot(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


# ---------------------------------------------------------------- FD helpers
def fd_grad(E, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in range(x.shape[0]):
        for c in range(3):
            xp = x.copy(); xp[i, c] += eps
            xm = x.copy(); xm[i, c] -= eps
            g[i, c] = (E(xp) - E(xm)) / (2 * eps)
    return g


def fd_hess_vec(F_of_x, x, v, eps=1e-6):
    """H v = (grad(x+eps v) - grad(x-eps v))/(2 eps); grad = -F."""
    Fp = F_of_x(x + eps * v)
    Fm = F_of_x(x - eps * v)
    return (Fm - Fp) / (2 * eps)


# =========================================================================
print("== 1. Elastic energy gradient vs central finite differences ==")
rng = np.random.default_rng(20260822)
cloth = create_cloth_grid(nx=5, ny=5, width=0.3, height=0.3, center=(0.5, 0.5, 0.5),
                          k_stretch=900.0, k_bend=0.05, density=0.2)
solver = MatrixFreeIPCSolver(dhat=0.02, stiffness=4e3)

# (a) rest state: energies/forces must vanish to roundoff (coords ~0.5,
#     hinge weights ~40, so roundoff floor is ~1e-9, not 1e-16)
x = cloth.rest_positions.copy()
E, F = solver.compute_elastic_energy_and_forces(x, cloth)
check("flat rest state has zero elastic energy (roundoff)", E < 1e-15, f"E={E:.2e}")
check("flat rest state has zero forces (roundoff)", np.linalg.norm(F) < 1e-8,
      f"|F|={np.linalg.norm(F):.2e}")

# (b) randomly perturbed, folded state (bending active, mixed stretch)
x = cloth.rest_positions + rng.standard_normal(cloth.rest_positions.shape) * 0.02
x[12, 2] += 0.06  # fold one vertex out of plane
E, F = solver.compute_elastic_energy_and_forces(x, cloth)
g_fd = fd_grad(lambda p: solver.compute_elastic_energy_and_forces(p, cloth)[0], x)
err = np.linalg.norm(F - (-g_fd)) / max(1e-12, np.linalg.norm(g_fd))
check("elastic gradient FD @ folded random state", err < 1e-6, f"rel={err:.2e} E={E:.4e}")

# (c) bent-only state: displace out of plane but keep edge lengths ~constant
x = cloth.rest_positions.copy()
w = 0.3
x[:, 2] += 0.02 * np.sin(np.pi * (x[:, 0] - 0.35) / w)
E, F = solver.compute_elastic_energy_and_forces(x, cloth)
g_fd = fd_grad(lambda p: solver.compute_elastic_energy_and_forces(p, cloth)[0], x)
err = np.linalg.norm(F - (-g_fd)) / max(1e-12, np.linalg.norm(g_fd))
check("elastic gradient FD @ smooth bend state", err < 1e-6, f"rel={err:.2e}")

# =========================================================================
print("== 2. Bending: flat-state annihilation in random orientations ==")
# Irregular flat mesh (random triangulated quad strip), then random 3D rotation:
# the discrete mean-curvature vector H must vanish in ANY orientation.
pts2d = np.array([[0.0, 0.0], [1.0, 0.05], [2.1, -0.04], [3.0, 0.02],
                  [0.4, 1.0], [1.5, 1.1], [2.6, 0.95], [3.4, 1.05]])
tris = np.array([[0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5], [2, 3, 6], [3, 7, 6]], dtype=np.int32)
flat = ClothMesh(np.column_stack([pts2d, np.zeros(len(pts2d))]), tris,
                 k_stretch=500.0, k_bend=0.1, density=0.2)
worst = 0.0
for trial in range(20):
    R = rot(rng.standard_normal(3), rng.uniform(0, 2 * np.pi))
    t = rng.standard_normal(3) * 10
    x = (flat.rest_positions @ R.T) + t
    E, F = solver.compute_elastic_energy_and_forces(x, flat)
    worst = max(worst, abs(E), float(np.max(np.abs(F))))
check("bending energy+forces vanish on rotated/translated flat mesh", worst < 1e-8,
      f"worst |E|,|F| = {worst:.2e} over 20 random rigid transforms")

# Folded-state rigid invariance of the total elastic energy
x_fold = flat.rest_positions.copy(); x_fold[4, 2] = 0.3
E0, _ = solver.compute_elastic_energy_and_forces(x_fold, flat)
R = rot(np.array([0.3, 0.7, -0.2]), 0.9)
E1, _ = solver.compute_elastic_energy_and_forces(x_fold @ R.T + 3.0, flat)
check("elastic energy rigid-transform invariant", abs(E1 - E0) < 1e-9 * max(1, abs(E0)),
      f"E0={E0:.6e} E1={E1:.6e}")

# =========================================================================
print("== 3. Elastic Hessian-vector products vs finite differences ==")
# (a) stretched springs (d > L0): the PSD projection equals the true Hessian.
xs = cloth.rest_positions.copy()
stretch_dir = (xs[0] - xs[1]); stretch_dir /= np.linalg.norm(stretch_dir)
xs[0] += 0.3 * stretch_dir  # clearly stretched
v = rng.standard_normal(xs.shape)
Hv = solver.apply_elastic_hessian_vector_product(v, xs, cloth)
F_of = lambda p: solver.compute_elastic_energy_and_forces(p, cloth)[1]
Hv_fd = fd_hess_vec(F_of, xs, v, eps=2e-6)
err = np.linalg.norm(Hv - Hv_fd) / max(1e-12, np.linalg.norm(Hv_fd))
check("stretch+bend Hessian FD @ stretched state", err < 5e-4, f"rel={err:.2e}")

# (b) bending-only Hessian (exact quadratic -> FD to machine precision)
xb = cloth.rest_positions.copy(); xb[12, 2] += 0.05
vb = np.zeros_like(xb); vb[:, 2] = rng.standard_normal(xb.shape[0])
Hv_b = solver.apply_elastic_hessian_vector_product(vb, xb, cloth)
Hv_b_fd = fd_hess_vec(F_of, xb, vb, eps=1e-6)
err = np.linalg.norm(Hv_b - Hv_b_fd) / max(1e-12, np.linalg.norm(Hv_b_fd))
check("bending Hessian FD (exact quadratic)", err < 1e-6, f"rel={err:.2e}")

# =========================================================================
print("== 4. Barrier pair energy/gradient vs FD at many distances ==")
dhat = 0.02
solver2 = MatrixFreeIPCSolver(dhat=dhat, stiffness=5e3)
worst = 0.0
for frac in [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999]:
    d = frac * dhat
    x = np.array([[0.0, 0.0, 0.0], [d * 0.7, d * 0.7, d * np.sqrt(0.02)]])
    x[1] *= d / np.linalg.norm(x[1])  # exact distance d along a skew direction
    cand = np.array([[0, 1]], dtype=np.int32)
    E, F = solver2.compute_barrier_energy_and_forces(x, cand)
    # eps=1e-9: near d->0 and d->dhat the FD error scales linearly with eps
    # (verified: rel error 9e-6/4e-4 at eps=1e-7 -> 9e-10/4e-8 at eps=1e-9),
    # so a smaller eps is required there; this is FD conditioning, not code error.
    g_fd = fd_grad(lambda p: solver2.compute_barrier_energy_and_forces(p, cand)[0], x, eps=1e-9)
    err = np.linalg.norm(F - (-g_fd)) / max(1e-12, np.linalg.norm(g_fd))
    worst = max(worst, err)
check("barrier pair gradient FD across d in (0, dhat)", worst < 1e-6, f"worst rel={worst:.2e}")

# inactive (d >= dhat): energy and forces exactly zero
x = np.array([[0.0, 0, 0], [dhat * 1.0000001, 0, 0]])
E, F = solver2.compute_barrier_energy_and_forces(x, np.array([[0, 1]], dtype=np.int32))
check("barrier zero at d >= dhat", E == 0.0 and np.all(F == 0.0))

# coincident pair (d = 0): masked out, no NaN
x = np.array([[0.3, 0.3, 0.3], [0.3, 0.3, 0.3]])
E, F = solver2.compute_barrier_energy_and_forces(x, np.array([[0, 1]], dtype=np.int32))
check("barrier coincident pair: no NaN, zero contribution",
      np.isfinite(E) and np.all(np.isfinite(F)) and E == 0.0 and np.all(F == 0.0))

# pair force antisymmetry (Newton's third law) on a random multi-pair scene
x = rng.random((30, 3)) * 0.01  # everything within dhat of everything
cand = np.array([[i, j] for i in range(30) for j in range(i + 1, 30)], dtype=np.int32)
E, F = solver2.compute_barrier_energy_and_forces(x, cand)
check("barrier pair forces antisymmetric (sum F = 0)",
      np.linalg.norm(F.sum(axis=0)) < 1e-9 * max(1.0, np.linalg.norm(F)),
      f"|sum F|={np.linalg.norm(F.sum(axis=0)):.2e}")

# =========================================================================
print("== 5. Obstacle barrier (sphere + plane) energy/gradient/Hessian vs FD ==")
solver3 = MatrixFreeIPCSolver(dhat=0.02, stiffness=5e3)
solver3.add_sphere_obstacle(center=np.array([0.0, 0.0, 0.0]), radius=0.1)
solver3.add_plane_obstacle(point=np.array([0.0, 0.0, -0.05]), normal=np.array([0.2, -0.1, 1.0]))
# Points at ACTIVE gaps in (0.001*dhat, 0.95*dhat): penetrating (negative) or
# >= dhat gaps are masked inactive and would make the check vacuous.
dirs = rng.standard_normal((6, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
gap_s = 0.001 * 0.02 + rng.random(6) * 0.94 * 0.02
near_sphere = dirs * (0.1 + gap_s)[:, None]
nrm = np.array([0.2, -0.1, 1.0]); nrm /= np.linalg.norm(nrm)
e1 = np.cross(nrm, [1.0, 0.0, 0.0]); e1 /= np.linalg.norm(e1)
e2 = np.cross(nrm, e1)
gap_p = 0.001 * 0.02 + rng.random(6) * 0.94 * 0.02
tang = rng.standard_normal((6, 2)) * 0.5
near_plane = (np.array([0.0, 0.0, -0.05])[None, :] + gap_p[:, None] * nrm[None, :]
              + tang[:, 0:1] * e1[None, :] + tang[:, 1:2] * e2[None, :])
x = np.vstack([near_sphere, near_plane])
E, F = solver3.compute_barrier_energy_and_forces(x, np.empty((0, 2), dtype=np.int32))
check("obstacle scene actually active (E > 0)", E > 0, f"E={E:.4e}")
g_fd = fd_grad(lambda p: solver3.compute_barrier_energy_and_forces(p, np.empty((0, 2), dtype=np.int32))[0],
               x, eps=1e-7)
err = np.linalg.norm(F - (-g_fd)) / max(1e-12, np.linalg.norm(g_fd))
check("sphere+plane barrier gradient FD", err < 1e-4, f"rel={err:.2e} E={E:.4e}")

v = rng.standard_normal(x.shape)
Hv = solver3.apply_barrier_hessian_vector_product(v, x, np.empty((0, 2), dtype=np.int32))
F_of = lambda p: solver3.compute_barrier_energy_and_forces(p, np.empty((0, 2), dtype=np.int32))[1]
Hv_fd = fd_hess_vec(F_of, x, v, eps=2e-7)
err = np.linalg.norm(Hv - Hv_fd) / max(1e-12, np.linalg.norm(Hv_fd))
check("sphere+plane barrier Hessian FD (normal-projected)",
      err < 5e-3, f"rel={err:.2e}")

# =========================================================================
print("== 6. Broadphase completeness vs all-pairs on adversarial scenes ==")


def all_pairs_active(pos, dhat_v):
    N = len(pos)
    out = set()
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < dhat_v and d > 1e-9:
                out.add((i, j))
    return out


def run_scene(name, pos, dhat_v, cloth=None):
    s = MatrixFreeIPCSolver(dhat=dhat_v, stiffness=1e3)
    cand = s.find_broadphase_candidates(pos, cloth)
    cset = set(map(tuple, cand.tolist()))
    bf = all_pairs_active(pos, dhat_v)
    if cloth is not None:
        bf = {p for p in bf if ((p[0] << 32) | p[1]) not in cloth.topo_exclusion_set}
    missed = bf - cset
    check(f"broadphase superset: {name}", len(missed) == 0,
          f"bf={len(bf)} cand={len(cset)} missed={sorted(missed)[:3]}")
    # reference parity
    ref = s._find_broadphase_candidates_reference(pos, cloth)
    rset = set(map(tuple, ref.tolist()))
    check(f"broadphase parity vs reference: {name}", cset == rset,
          f"only_new={len(cset - rset)} only_ref={len(rset - cset)}")
    return cset


# (a) exact cell-boundary lattice: coords are exact multiples of dhat/2
g = np.arange(-6, 7)
X, Y, Z = np.meshgrid(g * 0.5 * 0.05, g * 0.5 * 0.05, g * 0.5 * 0.05)
lat = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
run_scene("exact-boundary 13^3 lattice", lat, 0.05)

# (b) pairs straddling a cell boundary just below dhat
pts = []
for k in range(-5, 6):
    base = k * 0.05
    pts.append([base + 0.02, 0.0, 0.0])   # cell k
    pts.append([base + 0.049999, 0.0, 0.0])  # same cell, d=0.029999 < dhat
    pts.append([base + 0.050001, 0.0, 0.0])  # cell k+1, d=0.030001 < dhat from first
    pts.append([base + 0.07, 0.0, 0.0])    # cell k+1
strad = np.array(pts)
run_scene("boundary-straddling chains", strad, 0.05)

# (c) coincident duplicate points (non-topological)
dup = np.array([[0.1, 0.1, 0.1]] * 25 + [[0.1 + 0.04, 0.1, 0.1], [0.1, 0.1 + 0.045, 0.1]])
run_scene("25 coincident + 2 near points", dup, 0.05)

# (d) all points identical
run_scene("all-identical 40 points", np.tile([0.2, -0.3, 0.4], (40, 1)), 0.05)

# (e) single point
c1 = run_scene("single point", np.array([[0.05, 0.05, 0.05]]), 0.05)
check("single point -> no candidates", len(c1) == 0)

# (f) two points at exactly dhat apart (not active, but maybe candidates)
x2 = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
c2 = run_scene("two points exactly dhat apart (cell boundary)", x2, 0.05)
E2, F2 = MatrixFreeIPCSolver(dhat=0.05, stiffness=1e3).compute_barrier_energy_and_forces(
    x2, np.array([[0, 1]], dtype=np.int32))
check("two points exactly dhat apart: candidate but barrier inactive (E=0)",
      (0, 1) in c2 and E2 == 0.0)

# (g) Chebyshev-2 pair with occupied midpoint cell must appear (reference closure)
m3 = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.1, 0.0, 0.0]])
c3 = run_scene("collinear A-mid-B (Chebyshev-2 with midpoint)", m3, 0.05)
check("dist-2 pair with occupied midpoint emitted", (0, 2) in c3)

# (h) Chebyshev-2 pair WITHOUT midpoint must NOT appear in either impl
m4 = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
c4 = run_scene("collinear A-B no midpoint", m4, 0.05)
check("dist-2 pair without midpoint not emitted", (0, 1) not in c4)

# (i) random cloth scene with topo exclusions, dense (many same-cell duplicates)
clothb = create_cloth_grid(nx=8, ny=8, width=0.2, height=0.2, center=(0.4, 0.4, 0.4))
posb = np.repeat(clothb.rest_positions, 3, axis=0) + rng.standard_normal((192, 3)) * 1e-4
run_scene("triplicated cloth vertices (dense duplicates)", posb, 0.05)

# =========================================================================
print("== 7. Zero-contact scene: solve_step reduces incremental potential ==")
# NOTE: with zero initial velocity and uniform gravity, x_tilde is a RIGID
# translation of the rest state (no stretch, no bend) and the exact solution
# is free fall, so psi stays ~0 and nothing should move relative to x_tilde.
# To exercise the solver we start from random velocities instead.
zc = create_cloth_grid(nx=4, ny=4, width=0.2, height=0.2, center=(0.5, 0.5, 0.5))
sz = MatrixFreeIPCSolver(dhat=0.005, stiffness=1e3)  # tiny dhat -> no contacts
p = zc.rest_positions.copy()
vv = rng.standard_normal(p.shape) * 0.05
dt = 0.012
grav = np.array([0.0, 0.0, -9.81])
x_t = p + vv * dt + dt * dt * grav


def psi(xx):
    return (0.5 * np.sum(zc.masses[:, None] * (xx - x_t) ** 2) / dt ** 2
            + sz.compute_elastic_energy_and_forces(xx, zc)[0])


psi0 = psi(x_t)
pn, vn, m = sz.solve_step(p, vv, zc, dt=dt, gravity=grav)
psi1 = psi(pn)
check("zero-contact step lowers oracle incremental potential", psi1 < psi0,
      f"psi {psi0:.6e} -> {psi1:.6e}")
check("zero-contact step kills most injected velocity error", psi1 < 0.2 * psi0,
      f"reduction to {psi1 / psi0:.3f} of initial")
check("zero-contact: barrier energy exactly 0 at all states",
      sz.compute_barrier_energy_and_forces(pn, np.empty((0, 2), dtype=np.int32))[0] == 0.0)

# =========================================================================
print("== 8. Degenerate meshes ==")
tri = np.array([[0, 1, 2]], dtype=np.int32)
pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
try:
    m1 = ClothMesh(pts, tri, k_stretch=100.0, k_bend=0.01, density=0.2)
    E, F = solver.compute_elastic_energy_and_forces(pts, m1)
    check("single-triangle mesh: works, zero energy at rest", E < 1e-12)
except Exception as e:
    check("single-triangle mesh construction", False, repr(e))

try:
    m0 = ClothMesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.int32))
    check("empty mesh (0 verts, 0 faces) constructs with (0,2) edges",
          m0.struct_edges.shape == (0, 2))
except Exception as e:
    check("empty mesh (0 verts, 0 faces) constructs", False, f"{type(e).__name__}: {e}")

try:
    mn = ClothMesh(pts + 0.5, np.empty((0, 3), dtype=np.int32))
    E, F = solver.compute_elastic_energy_and_forces(pts + 0.5, mn)
    check("triangle-free mesh energies", np.isfinite(E))
except Exception as e:
    check("triangle-free mesh (no faces)", False, f"{type(e).__name__}: {e}")

# =========================================================================
print("== 9. Tetra broadphase neighbor tally vs brute force ==")
ts = TetrahedralSoftRoboticsSolver()
verts = rng.random((150, 3))
tets = np.zeros((10, 4), dtype=np.int32)
stats = ts.solve_deformable_step(verts.astype(np.float32), tets)
cells = np.clip(np.floor(verts * 32), 0, 31).astype(np.int64)
occ = set(map(tuple, cells.tolist()))
bf = 0
for c in occ:
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (c[0] + dx, c[1] + dy, c[2] + dz) in occ:
                    bf += 1
check("tetra broadphase tally == brute-force occupied-neighbor count",
      stats["broadphase_neighbor_cell_pairs"] == bf,
      f"got={stats['broadphase_neighbor_cell_pairs']} want={bf}")

# =========================================================================
print("== 10. Determinism / repeated calls after mutation ==")
s = MatrixFreeIPCSolver(dhat=0.03, stiffness=1e3)
pos = rng.random((60, 3))
a1 = s.find_broadphase_candidates(pos, None)
a2 = s.find_broadphase_candidates(pos, None)
check("broadphase deterministic on repeat call",
      np.array_equal(a1, a2))
pos2 = pos.copy(); pos2[0] += 0.5  # rebuild after mutation
a3 = s.find_broadphase_candidates(pos2, None)
check("broadphase rebuild after mutation changes/differs sanely",
      a3.shape[1] == 2)

# =========================================================================
print()
print(f"PHYSICS PROBE: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILURES:")
    for f in FAIL:
        print("  -", f)
sys.exit(1 if FAIL else 0)
