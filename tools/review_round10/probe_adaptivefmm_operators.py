"""Round-10 Wave A probe: adaptive FMM operators vs direct computation.

Tests each operator (P2M, M2M, M2L, L2L, L2P, P2L, M2P) independently
against a direct O(N^2) reference, then tests the full composition chain.
This is where sign errors, coefficient errors, and transpose errors hide.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import cmath
from core.adaptive_fmm import (
    p2m, m2m, m2l, l2l, l2p, l2p_force, p2l, m2p,
    p2p_potential_and_force, exact_direct_nbody_2d, exact_direct_nbody_forces_2d,
)

FAIL = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(name)

print("=" * 70)
print("PROBE: adaptive FMM operators vs direct computation")
print("=" * 70)

rng = np.random.RandomState(42)
p = 8  # expansion order

# ============================================================
# 1. P2M -> M2P vs direct (single source cluster, far target)
# ============================================================
print(f"\n[1] P2M -> M2P vs direct (p={p})")
# Source cluster centered at z0=0, particles within |z|<0.3
n_src = 20
src_pts = rng.uniform(-0.3, 0.3, size=(n_src, 2))
src_q = rng.uniform(-1, 1, size=n_src)
z0 = 0.0 + 0.0j

# Target far away (|z - z0| > 2*r_src)
target = np.array([[3.0, 0.0]])
z_target = 3.0 + 0.0j

# Direct potential at target
direct_pot = 0.0
for i in range(n_src):
    dx = target[0, 0] - src_pts[i, 0]
    dy = target[0, 1] - src_pts[i, 1]
    r2 = dx * dx + dy * dy
    direct_pot += src_q[i] * 0.5 * np.log(r2)

# FMM: P2M then M2P
m_coeffs = p2m(src_pts, src_q, z0, p)
fmm_pot, fmm_field = m2p(m_coeffs, z0, z_target, p)

check("P2M->M2P potential matches direct", abs(fmm_pot - direct_pot) < 1e-6 * abs(direct_pot),
      f"direct={direct_pot:.10f}, fmm={fmm_pot:.10f}, diff={abs(fmm_pot - direct_pot):.2e}")

# ============================================================
# 2. P2M -> M2M -> M2P vs direct (shift expansion center)
# ============================================================
print(f"\n[2] P2M -> M2M -> M2P vs direct (center shift)")
z1 = 1.0 + 0.5j  # new center
# Direct is the same (potential is translation-invariant)
# FMM: P2M at z0, M2M to z1, M2P from z1
m_at_z0 = p2m(src_pts, src_q, z0, p)
m_at_z1 = m2m(m_at_z0, z0, z1, p)
fmm_pot2, _ = m2p(m_at_z1, z1, z_target, p)
# After M2M, convergence radius = |delta| + r_src = 1.118 + 0.3 = 1.418.
# Target at |z_target - z1| = 2.06. Ratio = 0.69, so p=8 truncation ~ 3%.
check("P2M->M2M->M2P matches direct (truncation-aware)", abs(fmm_pot2 - direct_pot) < 0.05 * abs(direct_pot),
      f"direct={direct_pot:.10f}, fmm={fmm_pot2:.10f}, diff={abs(fmm_pot2 - direct_pot):.2e}")

# ============================================================
# 3. P2M -> M2L -> L2P vs direct (well-separated clusters)
# ============================================================
print(f"\n[3] P2M -> M2L -> L2P vs direct")
# Source cluster at z0=0, target cluster at z1=5
z_src = 0.0 + 0.0j
z_dst = 5.0 + 0.0j
# Target points near z_dst
n_tgt = 10
tgt_pts = z_dst.real + rng.uniform(-0.2, 0.2, size=(n_tgt, 2))
tgt_pts[:, 1] = z_dst.imag + rng.uniform(-0.2, 0.2, size=n_tgt)

# Direct potential at each target
direct_pots = np.zeros(n_tgt)
for j in range(n_tgt):
    for i in range(n_src):
        dx = tgt_pts[j, 0] - src_pts[i, 0]
        dy = tgt_pts[j, 1] - src_pts[i, 1]
        r2 = dx * dx + dy * dy
        direct_pots[j] += src_q[i] * 0.5 * np.log(r2)

# FMM: P2M at z_src, M2L to z_dst, L2P at each target
m_coeffs3 = p2m(src_pts, src_q, z_src, p)
l_coeffs3 = m2l(m_coeffs3, z_src, z_dst, p)
fmm_pots3 = np.array([l2p(l_coeffs3, tgt_pts[j, 0] + 1j * tgt_pts[j, 1], z_dst, p) for j in range(n_tgt)])

max_err = np.max(np.abs(fmm_pots3 - direct_pots))
rel_err = max_err / max(1e-30, np.max(np.abs(direct_pots)))
check("P2M->M2L->L2P matches direct", rel_err < 1e-6,
      f"max_abs={max_err:.2e}, rel={rel_err:.2e}")

# ============================================================
# 4. P2M -> M2L -> L2L -> L2P vs direct (local expansion shift)
# ============================================================
print(f"\n[4] P2M -> M2L -> L2L -> L2P vs direct")
z_child = 5.3 + 0.1j  # child center near z_dst
# FMM: P2M, M2L to z_dst, L2L to z_child, L2P from z_child
l_at_parent = m2l(m_coeffs3, z_src, z_dst, p)
l_at_child = l2l(l_at_parent, z_dst, z_child, p)
fmm_pots4 = np.array([l2p(l_at_child, tgt_pts[j, 0] + 1j * tgt_pts[j, 1], z_child, p) for j in range(n_tgt)])
max_err4 = np.max(np.abs(fmm_pots4 - direct_pots))
rel_err4 = max_err4 / max(1e-30, np.max(np.abs(direct_pots)))
check("P2M->M2L->L2L->L2P matches direct", rel_err4 < 1e-5,
      f"max_abs={max_err4:.2e}, rel={rel_err4:.2e}")

# ============================================================
# 5. P2L -> L2P vs direct (distant particles to local)
# ============================================================
print(f"\n[5] P2L -> L2P vs direct")
# Distant sources, local expansion around z_dst
far_src = rng.uniform(8, 10, size=(5, 2))
far_q = rng.uniform(-1, 1, size=5)
# Direct
direct_pots5 = np.zeros(n_tgt)
for j in range(n_tgt):
    for i in range(len(far_src)):
        dx = tgt_pts[j, 0] - far_src[i, 0]
        dy = tgt_pts[j, 1] - far_src[i, 1]
        r2 = dx * dx + dy * dy
        direct_pots5[j] += far_q[i] * 0.5 * np.log(r2)
# FMM: P2L at z_dst, L2P
l_coeffs5 = p2l(far_src, far_q, z_dst, p)
fmm_pots5 = np.array([l2p(l_coeffs5, tgt_pts[j, 0] + 1j * tgt_pts[j, 1], z_dst, p) for j in range(n_tgt)])
max_err5 = np.max(np.abs(fmm_pots5 - direct_pots5))
rel_err5 = max_err5 / max(1e-30, np.max(np.abs(direct_pots5)))
check("P2L->L2P matches direct", rel_err5 < 1e-6,
      f"max_abs={max_err5:.2e}, rel={rel_err5:.2e}")

# ============================================================
# 6. L2P force vs finite-difference gradient
# ============================================================
print(f"\n[6] L2P force vs finite-difference gradient")
# Use the local expansion from test 3
l_coeffs6 = l_coeffs3
z_center6 = z_dst
# Pick a target point
z_t = 5.1 + 0.05j
fx_fmm, fy_fmm = l2p_force(l_coeffs6, z_t, z_center6, p)
# Finite difference: phi(z + h) - phi(z - h) / (2h)
h = 1e-7
phi_xp = l2p(l_coeffs6, (z_t + h).real + 1j * (z_t + h).imag, z_center6, p)
phi_xm = l2p(l_coeffs6, (z_t - h).real + 1j * (z_t - h).imag, z_center6, p)
phi_yp = l2p(l_coeffs6, z_t.real + 1j * (z_t.imag + h), z_center6, p)
phi_ym = l2p(l_coeffs6, z_t.real + 1j * (z_t.imag - h), z_center6, p)
fx_fd = -(phi_xp - phi_xm) / (2 * h)  # F = -grad phi
fy_fd = -(phi_yp - phi_ym) / (2 * h)
check("L2P force x vs FD", abs(fx_fmm - fx_fd) < 1e-4 * max(1, abs(fx_fd)),
      f"fmm={fx_fmm:.8f}, fd={fx_fd:.8f}")
check("L2P force y vs FD", abs(fy_fmm - fy_fd) < 1e-4 * max(1, abs(fy_fd)),
      f"fmm={fy_fmm:.8f}, fd={fy_fd:.8f}")

# ============================================================
# 7. M2P field vs finite-difference gradient
# ============================================================
print(f"\n[7] M2P field vs finite-difference gradient")
# Use multipole from test 1
m_coeffs7 = m_coeffs
z_center7 = z0
z_t7 = 3.0 + 0.0j
pot7, field7 = m2p(m_coeffs7, z_center7, z_t7, p)
# FD of potential
h = 1e-7
pot_xp = m2p(m_coeffs7, z_center7, z_t7 + h, p)[0]
pot_xm = m2p(m_coeffs7, z_center7, z_t7 - h, p)[0]
pot_yp = m2p(m_coeffs7, z_center7, z_t7 + 1j * h, p)[0]
pot_ym = m2p(m_coeffs7, z_center7, z_t7 - 1j * h, p)[0]
# F = -grad phi, and field = dPhi/dz = dphi/dx - i*dphi/dy
# So Re(field) = dphi/dx, Im(field) = -dphi/dy
# F_x = -dphi/dx = -Re(field), F_y = -dphi/dy = Im(field)
fd_dphi_dx = (pot_xp - pot_xm) / (2 * h)
fd_dphi_dy = (pot_yp - pot_ym) / (2 * h)
check("M2P field Re (=dphi/dx) vs FD", abs(field7.real - fd_dphi_dx) < 1e-4 * max(1, abs(fd_dphi_dx)),
      f"field_re={field7.real:.8f}, fd={fd_dphi_dx:.8f}")
check("M2P field Im (=-dphi/dy) vs FD", abs(field7.imag - (-fd_dphi_dy)) < 1e-4 * max(1, abs(fd_dphi_dy)),
      f"field_im={field7.imag:.8f}, fd={-fd_dphi_dy:.8f}")

# ============================================================
# 8. Full pipeline: TreeFreeElasticAdaptiveFMM vs direct
# ============================================================
print(f"\n[8] Full FMM pipeline vs direct O(N^2)")
from core.adaptive_fmm import TreeFreeElasticAdaptiveFMM
N = 500
pts = rng.uniform(0, 1, size=(N, 2))
charges = rng.uniform(-1, 1, size=N)
fmm = TreeFreeElasticAdaptiveFMM(p=8, max_depth=8, max_leaf_particles=32)
pot_fmm, fx_fmm, fy_fmm = fmm.evaluate(pts, charges)
pot_dir = exact_direct_nbody_2d(pts, charges)
fx_dir, fy_dir = exact_direct_nbody_forces_2d(pts, charges)
rel_pot = np.linalg.norm(pot_fmm - pot_dir) / max(1e-30, np.linalg.norm(pot_dir))
rel_fx = np.linalg.norm(fx_fmm - fx_dir) / max(1e-30, np.linalg.norm(fx_dir))
rel_fy = np.linalg.norm(fy_fmm - fy_dir) / max(1e-30, np.linalg.norm(fy_dir))
check(f"FMM potential vs direct (N={N}, p=8)", rel_pot < 1e-4,
      f"rel_L2={rel_pot:.2e}")
check(f"FMM force_x vs direct", rel_fx < 1e-3,
      f"rel_L2={rel_fx:.2e}")
check(f"FMM force_y vs direct", rel_fy < 1e-3,
      f"rel_L2={rel_fy:.2e}")

# ============================================================
# 9. Edge case: N=1 (single particle)
# ============================================================
print(f"\n[9] N=1 single particle")
pts1 = np.array([[0.5, 0.5]])
q1 = np.array([1.0])
fmm9 = TreeFreeElasticAdaptiveFMM(p=4, max_depth=4, max_leaf_particles=4)
pot9, fx9, fy9 = fmm9.evaluate(pts1, q1)
pot_dir9 = exact_direct_nbody_2d(pts1, q1)
fx_dir9, fy_dir9 = exact_direct_nbody_forces_2d(pts1, q1)
check("N=1: potential matches", np.allclose(pot9, pot_dir9, atol=1e-10),
      f"fmm={pot9}, dir={pot_dir9}")
check("N=1: force matches", np.allclose(fx9, fx_dir9, atol=1e-10) and np.allclose(fy9, fy_dir9, atol=1e-10),
      f"fmm_fx={fx9}, dir_fx={fx_dir9}")

# ============================================================
# 10. Edge case: N=2 (two particles in same cell)
# ============================================================
print(f"\n[10] N=2 same cell")
pts2 = np.array([[0.5, 0.5], [0.51, 0.51]])
q2 = np.array([1.0, -1.0])
fmm10 = TreeFreeElasticAdaptiveFMM(p=8, max_depth=8, max_leaf_particles=4)
pot10, fx10, fy10 = fmm10.evaluate(pts2, q2)
pot_dir10 = exact_direct_nbody_2d(pts2, q2)
check("N=2: potential matches", np.allclose(pot10, pot_dir10, atol=1e-6),
      f"fmm={pot10}, dir={pot_dir10}")

# ============================================================
# 11. Convergence: error should decrease with order p
# ============================================================
print(f"\n[11] Convergence with order p")
N11 = 300
pts11 = rng.uniform(0, 1, size=(N11, 2))
q11 = rng.uniform(-1, 1, size=N11)
pot_dir11 = exact_direct_nbody_2d(pts11, q11)
errors = []
for pp in [2, 4, 6, 8, 10]:
    fmm11 = TreeFreeElasticAdaptiveFMM(p=pp, max_depth=8, max_leaf_particles=32)
    pot11, _, _ = fmm11.evaluate(pts11, q11)
    rel = np.linalg.norm(pot11 - pot_dir11) / max(1e-30, np.linalg.norm(pot_dir11))
    errors.append(rel)
    print(f"    p={pp}: rel-L2 = {rel:.2e}")
# Error should generally decrease (not necessarily monotonically for adaptive)
check("convergence: p=10 better than p=2", errors[-1] < errors[0],
      f"p=2: {errors[0]:.2e}, p=10: {errors[-1]:.2e}")

# ============================================================
# 12. Symmetry: same charges at symmetric positions => symmetric potential
# ============================================================
print(f"\n[12] Symmetry test")
pts_sym = np.array([[0.3, 0.5, ], [0.7, 0.5]])
q_sym = np.array([1.0, 1.0])
fmm12 = TreeFreeElasticAdaptiveFMM(p=8, max_depth=8, max_leaf_particles=4)
pot12, fx12, fy12 = fmm12.evaluate(pts_sym, q_sym)
check("symmetric: potentials equal", abs(pot12[0] - pot12[1]) < 1e-10,
      f"pot0={pot12[0]:.10f}, pot1={pot12[1]:.10f}")
check("symmetric: forces opposite x", abs(fx12[0] + fx12[1]) < 1e-8,
      f"fx0={fx12[0]:.10f}, fx1={fx12[1]:.10f}")
check("symmetric: forces zero y", abs(fy12[0]) < 1e-6 and abs(fy12[1]) < 1e-6,
      f"fy0={fy12[0]:.10f}, fy1={fy12[1]:.10f}")

print("\n" + "=" * 70)
if FAIL:
    print(f"RESULT: {len(FAIL)} FAILURES")
    for f in FAIL:
        print(f"  - {f}")
else:
    print("RESULT: ALL CHECKS PASSED")
print("=" * 70)
