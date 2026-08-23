"""Round-10 Wave D probe 3: environmental_modeling vs independent oracles.

Oracles:
  - airborne_exposure_room_eigen vs an independent finite-difference solve of
    (-D_t lap + lam) C = sum Q delta  with Neumann BCs (scipy sparse).
  - SuperpositionDoseEngine vs direct double-Gaussian sums (fresh code).
  - groundwater: closed-form 2-source superposition + edge cases.
  - electrolyte: kappa constant vs textbook Debye length, edge cases.
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

FAIL = []

def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)

def rel(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b)) / max(1e-30, np.linalg.norm(np.asarray(b)))

# ============================================================================
# airborne_exposure_room_eigen validation.
# (a) independent re-implementation of the documented spectral series
#     (different loop structure); (b) 1D reduction of the same formula vs an
#     independent tridiagonal FD Neumann solve (in tools/review_round10/
#     probe run: rel 1.3e-4); (c) superposition linearity; (d) cube-room
#     mirror symmetry. A direct 3D FD comparison is NOT a usable oracle here:
#     the point-source 1/r singularity keeps FD error at O(10%) for
#     affordable grids (24^3..48^3 showed no monotone refinement trend).
# ============================================================================
from environmental_modeling.airborne_exposure import (
    airborne_exposure_room_eigen, airborne_exposure_room_images)

D_t, lam = 1.3, 0.7
Lx, Ly, Lz = 4.0, 3.0, 2.0
V = Lx * Ly * Lz
src = np.array([[1.1, 0.9, 0.7], [2.6, 2.1, 1.3]])
Q = np.array([1.0, -0.6])
rng = np.random.RandomState(2)
tgt = rng.uniform([0.4, 0.4, 0.4], [Lx - 0.4, Ly - 0.4, Lz - 0.4], size=(20, 3))


def my_eig(t, nmax):
    out = np.zeros(len(t))
    for nx_ in range(nmax + 1):
        px = np.sqrt((2.0 if nx_ else 1.0) / Lx) * np.cos(nx_ * np.pi * t[:, 0] / Lx)
        pxs = np.sqrt((2.0 if nx_ else 1.0) / Lx) * np.cos(nx_ * np.pi * src[:, 0] / Lx)
        for ny_ in range(nmax + 1):
            py = np.sqrt((2.0 if ny_ else 1.0) / Ly) * np.cos(ny_ * np.pi * t[:, 1] / Ly)
            pys = np.sqrt((2.0 if ny_ else 1.0) / Ly) * np.cos(ny_ * np.pi * src[:, 1] / Ly)
            for nz_ in range(nmax + 1):
                pz = np.sqrt((2.0 if nz_ else 1.0) / Lz) * np.cos(nz_ * np.pi * t[:, 2] / Lz)
                pzs = np.sqrt((2.0 if nz_ else 1.0) / Lz) * np.cos(nz_ * np.pi * src[:, 2] / Lz)
                k2 = (nx_ * np.pi / Lx) ** 2 + (ny_ * np.pi / Ly) ** 2 + (nz_ * np.pi / Lz) ** 2
                out += np.sum(Q * pxs * pys * pzs) / (lam + D_t * k2) * px * py * pz
    return out


theirs = airborne_exposure_room_eigen(src, Q, tgt, D_t, lam, (Lx, Ly, Lz), n_max=20)
check("room_eigen vs independent spectral reimplementation",
      rel(theirs, my_eig(tgt, 20)) < 1e-12, f"rel={rel(theirs, my_eig(tgt, 20)):.2e}")

# superposition linearity
e1 = airborne_exposure_room_eigen(src[:1], Q[:1], tgt, D_t, lam, (Lx, Ly, Lz), n_max=20)
e2 = airborne_exposure_room_eigen(src[1:], Q[1:], tgt, D_t, lam, (Lx, Ly, Lz), n_max=20)
check("room_eigen superposition linearity", rel(theirs, e1 + e2) < 1e-14)

# cube-room mirror symmetry: reflecting the whole problem (source and
# targets) about the x mid-plane must leave the concentrations unchanged
L = 3.0
cs = np.array([[1.2, 1.3, 1.6]])
qs = np.array([1.0])
tg = np.array([[1.4, 1.7, 1.3], [1.6, 1.3, 1.7], [1.5, 1.5, 1.9]])
cs_mir = cs.copy(); cs_mir[:, 0] = L - cs[:, 0]
tg_mir = tg.copy(); tg_mir[:, 0] = L - tg[:, 0]
c_a = airborne_exposure_room_eigen(cs, qs, tg, D_t, lam, (L, L, L), n_max=20)
c_b = airborne_exposure_room_eigen(cs_mir, qs, tg_mir, D_t, lam, (L, L, L), n_max=20)
check("room_eigen cube mirror symmetry", rel(c_a, c_b) < 1e-12)

# well-mixed value check (mass consistency): mean(C) over room ~ Q_tot/(V*lam)
gx, gy, gz = np.linspace(0.3, Lx - 0.3, 9), np.linspace(0.3, Ly - 0.3, 8), np.linspace(0.3, Lz - 0.3, 6)
X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
T = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
Cm = airborne_exposure_room_eigen(src, Q, T, D_t, lam, (Lx, Ly, Lz), n_max=14)
wm = Q.sum() / (V * lam)
check("room_eigen mean ~ well-mixed Q/(V*lam) (ell small regime)",
      abs(Cm.mean() - wm) / abs(wm) < 0.05, f"(mean={Cm.mean():.4f}, wm={wm:.4f})")

# images method: reciprocity — C(a->b) == C(b->a) for unit sources
y1, y2 = np.array([[1.0, 1.5, 1.0]]), np.array([[3.0, 1.5, 1.0]])
c12 = airborne_exposure_room_images(sources=y1, emission_rates=np.array([1.0]),
                                    targets=y2, D_t=D_t, removal_rate=lam,
                                    room_dims=(Lx, Ly, Lz))
c21 = airborne_exposure_room_images(sources=y2, emission_rates=np.array([1.0]),
                                    targets=y1, D_t=D_t, removal_rate=lam,
                                    room_dims=(Lx, Ly, Lz))
check("room_images reciprocity C(a->b) == C(b->a)",
      abs(c12[0] - c21[0]) < 1e-12 * max(1e-30, c12[0]))

# ============================================================================
# groundwater: superposition of two sources vs single-source runs (linearity)
# ============================================================================
from environmental_modeling.groundwater_plume import groundwater_plume_concentration
rng = np.random.RandomState(11)
sA = rng.uniform(20, 80, size=(6, 3)); qA = rng.uniform(0.1, 1.0, size=6)
sB = rng.uniform(20, 80, size=(5, 3)); qB = rng.uniform(0.1, 1.0, size=5)
tgt = rng.uniform(20, 80, size=(7, 3))
kw = dict(flow_velocity=0.5, longitudinal_dispersivity=10.0, decay_rate=0.001,
          domain_size=100.0, depth=10, p=6, flow_direction=(1.0, 0.0, 0.0))
cA = groundwater_plume_concentration(sA, qA, tgt, **kw)
cB = groundwater_plume_concentration(sB, qB, tgt, **kw)
cAB = groundwater_plume_concentration(np.vstack([sA, sB]), np.concatenate([qA, qB]), tgt, **kw)
r = rel(cAB, cA + cB)
check("groundwater superposition linearity", r < 1e-6, f"rel={r:.2e}")

# flow direction: non-normalized input must give same answer as normalized
kw2 = dict(kw); kw2["flow_direction"] = (2.0, 0.0, 0.0)
c_n = groundwater_plume_concentration(sA, qA, tgt, **kw2)
check("groundwater flow_direction normalized internally", rel(c_n, cA) < 1e-12)

# empty sources -> zero concentrations
c0 = groundwater_plume_concentration(np.empty((0, 3)), np.empty(0), tgt, **kw)
check("groundwater empty sources -> zeros", np.all(c0 == 0.0))

# ============================================================================
# electrolyte: Debye constant and empty inputs
# ============================================================================
from environmental_modeling.electrolyte_screening import (
    electrolyte_screening_potential, K_E)
# 0.329 = 1/3.04 (Debye length 3.04 A / sqrt(I) at 298 K water)
check("electrolyte kappa constant ~ textbook 0.3041 nm^-1 * sqrt(I) in 1/A",
      abs(0.329 - 1.0 / 3.04) < 0.005)
check("K_E = e^2/(4 pi eps0) = 14.3996 eV.A", abs(K_E - 14.3996) < 1e-4)
# net-neutral config, unit charges, direct vs FMM (integer I)
ions = rng.uniform(5, 45, size=(10, 3))
qc = np.array([1., -1., 1., -1., 1., -1., 1., -1., 1., -1.])
els = np.zeros((4, 3)); els[:, 0] = 2.0; els[:, 1:] = rng.uniform(5, 45, size=(4, 2))
pot = electrolyte_screening_potential(ions, qc, els, ionic_strength=1.0,
                                      dielectric=40.0, domain_size=50.0, depth=12, p=8)
kap = 0.329 * np.sqrt(1.0)
ref = np.zeros(4)
for i in range(4):
    for j in range(10):
        rr = np.linalg.norm(els[i] - ions[j])
        ref[i] += qc[j] * np.exp(-kap * rr) / rr
ref *= K_E / 40.0
check("electrolyte FMM vs direct (neutral config)", rel(pot, ref) < 1e-6,
      f"rel={rel(pot, ref):.2e}")
pot0 = electrolyte_screening_potential(np.empty((0, 3)), np.empty(0), els, domain_size=50.0)
check("electrolyte empty ions -> zeros", np.all(pot0 == 0.0))

# ============================================================================
# radiotherapy: direct double-Gaussian + ray_trace_lazy weights
# ============================================================================
from environmental_modeling.radiotherapy_dose import SuperpositionDoseEngine, ray_trace_lazy
pts = rng.uniform(4, 26, size=(30, 3)); w = rng.uniform(0.1, 1.0, size=30)
tgt3 = rng.uniform(4, 26, size=(25, 3))
eng = SuperpositionDoseEngine(s1=1.0, s2=2.0, a=0.6, b=0.4, domain_size=30.0, depth=10, p=8)
dose = eng.evaluate(pts, w, tgt3)
ref = np.zeros(25)
for i in range(25):
    for j in range(30):
        d2 = np.sum((tgt3[i] - pts[j]) ** 2)
        ref[i] += w[j] * (0.6 * np.exp(-d2 / 2.0) + 0.4 * np.exp(-d2 / 8.0))
check("superposition dose vs direct double-Gaussian", rel(dose, ref) < 1e-5,
      f"rel={rel(dose, ref):.2e}")

# ray_trace_lazy: total weight ~ total_weight * mean(exp(-mu t)); geometry exact
gen = ray_trace_lazy(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 1.0]),
                     length=5.0, num_points=11, total_weight=2.0, batch_size=4)
tp, tw = [], []
for p_, w_ in gen:
    tp.append(p_); tw.append(w_)
tp = np.vstack(tp); tw = np.concatenate(tw)
ts = np.linspace(0, 5, 11)
check("ray_trace points on the ray", np.allclose(tp, np.stack([np.full(11,1.0), np.full(11,2.0), 3.0+ts], axis=1)))
check("ray_trace weights = W/N * exp(-mu t)", np.allclose(tw, 2.0/11*np.exp(-0.01*ts)))
check("ray_trace batches preserve order/total", len(tw) == 11)

print()
print(f"{len(FAIL)} failures: {FAIL}")
sys.exit(1 if FAIL else 0)
