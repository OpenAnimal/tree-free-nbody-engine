"""Round-10 Wave A probe: Yukawa3D + Gaussian2D + ScreenedYukawa2D FMM engines.

Cross-validates each engine against direct O(N^2) computation with
multiple configurations, edge cases, and convergence checks.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from core.yukawa3d_fmm import Yukawa3DFMM
from core.gaussian2d_fgt import Gaussian2DFGT
from core.screened_yukawa2d_fmm import ScreenedYukawa2DFMM

FAIL = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(name)

print("=" * 70)
print("PROBE: Radial Taylor FMM engines vs direct computation")
print("=" * 70)

rng = np.random.RandomState(42)

# ============================================================
# 1. Yukawa3D: basic accuracy vs direct
# ============================================================
print("\n[1] Yukawa3D vs direct (N=200, p=8, kappa=1.0)")
N = 200
pts = rng.uniform(0, 1, size=(N, 3))
charges = rng.uniform(-1, 1, size=N)
kappa = 1.0

fmm = Yukawa3DFMM(depth=4, p=8, kappa=kappa)
pot_fmm = fmm.evaluate(pts, charges)

# Direct
pot_dir = np.zeros(N)
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        r = np.linalg.norm(pts[i] - pts[j])
        pot_dir[i] += charges[j] * np.exp(-kappa * r) / r

rel = np.linalg.norm(pot_fmm - pot_dir) / np.linalg.norm(pot_dir)
check(f"Yukawa3D rel-L2 < 1e-4", rel < 1e-4, f"rel={rel:.2e}")

# ============================================================
# 2. Yukawa3D: convergence with order
# ============================================================
print("\n[2] Yukawa3D convergence with order")
errors = []
for p in [2, 4, 6, 8]:
    fmm_p = Yukawa3DFMM(depth=4, p=p, kappa=kappa)
    pot_p = fmm_p.evaluate(pts, charges)
    rel_p = np.linalg.norm(pot_p - pot_dir) / np.linalg.norm(pot_dir)
    errors.append(rel_p)
    print(f"    p={p}: rel-L2 = {rel_p:.2e}")
check("Yukawa3D: p=8 better than p=2", errors[-1] < errors[0],
      f"p=2: {errors[0]:.2e}, p=8: {errors[-1]:.2e}")

# ============================================================
# 3. Yukawa3D: N=1 (no interactions)
# ============================================================
print("\n[3] Yukawa3D N=1")
pts1 = np.array([[0.5, 0.5, 0.5]])
q1 = np.array([1.0])
fmm1 = Yukawa3DFMM(depth=2, p=4, kappa=1.0)
pot1 = fmm1.evaluate(pts1, q1)
check("N=1: potential is 0 (no self-interaction)", abs(pot1[0]) < 1e-10, f"pot={pot1[0]}")

# ============================================================
# 4. Yukawa3D: N=2 (single pair)
# ============================================================
print("\n[4] Yukawa3D N=2")
pts2 = np.array([[0.3, 0.5, 0.5], [0.7, 0.5, 0.5]])
q2 = np.array([1.0, 1.0])
fmm2 = Yukawa3DFMM(depth=4, p=8, kappa=1.0)
pot2 = fmm2.evaluate(pts2, q2)
r = np.linalg.norm(pts2[0] - pts2[1])
expected0 = q2[1] * np.exp(-1.0 * r) / r
expected1 = q2[0] * np.exp(-1.0 * r) / r
check("N=2: pot[0] matches direct", abs(pot2[0] - expected0) < 1e-6 * abs(expected0),
      f"fmm={pot2[0]:.10f}, direct={expected0:.10f}")
check("N=2: pot[1] matches direct", abs(pot2[1] - expected1) < 1e-6 * abs(expected1),
      f"fmm={pot2[1]:.10f}, direct={expected1:.10f}")

# ============================================================
# 5. Yukawa3D: different kappa
# ============================================================
print("\n[5] Yukawa3D kappa=5.0 (short-range)")
fmm_k5 = Yukawa3DFMM(depth=4, p=8, kappa=5.0)
pot_k5 = fmm_k5.evaluate(pts, charges)
pot_dir_k5 = np.zeros(N)
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        r = np.linalg.norm(pts[i] - pts[j])
        pot_dir_k5[i] += charges[j] * np.exp(-5.0 * r) / r
rel_k5 = np.linalg.norm(pot_k5 - pot_dir_k5) / np.linalg.norm(pot_dir_k5)
check("Yukawa3D kappa=5 rel-L2 < 1e-4", rel_k5 < 1e-4, f"rel={rel_k5:.2e}")

# ============================================================
# 6. Yukawa3D: evaluate_targets (distinct targets)
# ============================================================
print("\n[6] Yukawa3D evaluate_targets")
targets = rng.uniform(0, 1, size=(50, 3))
fmm_t = Yukawa3DFMM(depth=4, p=8, kappa=1.0)
pot_targets = fmm_t.evaluate_targets(pts, charges, targets)
# Direct
pot_targets_dir = np.zeros(50)
for i in range(50):
    for j in range(N):
        r = np.linalg.norm(targets[i] - pts[j])
        if r > 1e-15:
            pot_targets_dir[i] += charges[j] * np.exp(-r) / r
rel_t = np.linalg.norm(pot_targets - pot_targets_dir) / np.linalg.norm(pot_targets_dir)
check("evaluate_targets rel-L2 < 1e-4", rel_t < 1e-4, f"rel={rel_t:.2e}")

# ============================================================
# 7. Gaussian2D: basic accuracy vs direct
# ============================================================
print("\n[7] Gaussian2D FGT vs direct (N=200, p=8)")
N2d = 200
pts2d = rng.uniform(0, 1, size=(N2d, 2))
charges2d = rng.uniform(-1, 1, size=N2d)
h = 0.15

fgt = Gaussian2DFGT(depth=4, p=8, h=h)
pot_fgt = fgt.evaluate(pts2d, charges2d)

# Direct: sum q_j * exp(-|r_i - r_j|^2 / h^2)
pot_dir2d = np.zeros(N2d)
for i in range(N2d):
    for j in range(N2d):
        if i == j:
            continue
        r2 = np.sum((pts2d[i] - pts2d[j]) ** 2)
        pot_dir2d[i] += charges2d[j] * np.exp(-r2 / (h * h))

rel2d = np.linalg.norm(pot_fgt - pot_dir2d) / max(1e-30, np.linalg.norm(pot_dir2d))
check("Gaussian2D rel-L2 < 1e-4", rel2d < 1e-4, f"rel={rel2d:.2e}")

# ============================================================
# 8. Gaussian2D: N=1
# ============================================================
print("\n[8] Gaussian2D N=1")
pts1_2d = np.array([[0.5, 0.5]])
q1_2d = np.array([1.0])
fgt1 = Gaussian2DFGT(depth=2, p=4, h=0.2)
pot1_2d = fgt1.evaluate(pts1_2d, q1_2d)
check("Gaussian2D N=1: potential is 0", abs(pot1_2d[0]) < 1e-10, f"pot={pot1_2d[0]}")

# ============================================================
# 9. ScreenedYukawa2D: basic accuracy vs direct
# ============================================================
print("\n[9] ScreenedYukawa2D vs direct (N=200, p=8, kappa=2.0)")
from scipy.special import k0
kappa2d = 2.0
fmm_sy2d = ScreenedYukawa2DFMM(depth=4, p=8, kappa=kappa2d)
pot_sy2d = fmm_sy2d.evaluate(pts2d, charges2d)

# Direct: sum q_j * K0(kappa * |r_i - r_j|)
pot_dir_sy2d = np.zeros(N2d)
for i in range(N2d):
    for j in range(N2d):
        if i == j:
            continue
        r = np.linalg.norm(pts2d[i] - pts2d[j])
        if r > 1e-15:
            pot_dir_sy2d[i] += charges2d[j] * k0(kappa2d * r)

rel_sy2d = np.linalg.norm(pot_sy2d - pot_dir_sy2d) / max(1e-30, np.linalg.norm(pot_dir_sy2d))
check("ScreenedYukawa2D rel-L2 < 1e-3", rel_sy2d < 1e-3, f"rel={rel_sy2d:.2e}")

# ============================================================
# 10. ScreenedYukawa2D: N=2
# ============================================================
print("\n[10] ScreenedYukawa2D N=2")
pts2_sy = np.array([[0.3, 0.5], [0.7, 0.5]])
q2_sy = np.array([1.0, 1.0])
fmm2_sy = ScreenedYukawa2DFMM(depth=4, p=8, kappa=2.0)
pot2_sy = fmm2_sy.evaluate(pts2_sy, q2_sy)
r2_sy = np.linalg.norm(pts2_sy[0] - pts2_sy[1])
exp0 = q2_sy[1] * k0(2.0 * r2_sy)
exp1 = q2_sy[0] * k0(2.0 * r2_sy)
check("ScreenedYukawa2D N=2: pot[0]", abs(pot2_sy[0] - exp0) < 1e-4 * max(1, abs(exp0)),
      f"fmm={pot2_sy[0]:.8f}, direct={exp0:.8f}")
check("ScreenedYukawa2D N=2: pot[1]", abs(pot2_sy[1] - exp1) < 1e-4 * max(1, abs(exp1)),
      f"fmm={pot2_sy[1]:.8f}, direct={exp1:.8f}")

# ============================================================
# 11. Yukawa3D: all-same-point (degenerate)
# ============================================================
print("\n[11] Yukawa3D all-same-point")
pts_same = np.full((10, 3), 0.5)
q_same = rng.uniform(-1, 1, size=10)
fmm_same = Yukawa3DFMM(depth=3, p=6, kappa=1.0)
pot_same = fmm_same.evaluate(pts_same, q_same)
# Direct: all pairs have r=0, so exp(-kappa*0)/0 = inf. With self-exclusion,
# each particle sees 0 distance to others but self is excluded.
# Actually r=0 for all pairs, so the direct sum diverges. The FMM should
# handle this gracefully (not crash, not produce inf/nan).
check("all-same-point: no inf", not np.any(np.isinf(pot_same)), f"pot={pot_same}")
check("all-same-point: no nan", not np.any(np.isnan(pot_same)), f"pot={pot_same}")

# ============================================================
# 12. Yukawa3D: symmetry (equal charges at symmetric positions)
# ============================================================
print("\n[12] Yukawa3D symmetry")
pts_sym = np.array([[0.3, 0.5, 0.5], [0.7, 0.5, 0.5]])
q_sym = np.array([1.0, 1.0])
fmm_sym = Yukawa3DFMM(depth=4, p=8, kappa=1.0)
pot_sym = fmm_sym.evaluate(pts_sym, q_sym)
check("Yukawa3D symmetric: potentials equal", abs(pot_sym[0] - pot_sym[1]) < 1e-10,
      f"pot0={pot_sym[0]:.10f}, pot1={pot_sym[1]:.10f}")

print("\n" + "=" * 70)
if FAIL:
    print(f"RESULT: {len(FAIL)} FAILURES")
    for f in FAIL:
        print(f"  - {f}")
else:
    print("RESULT: ALL CHECKS PASSED")
print("=" * 70)
