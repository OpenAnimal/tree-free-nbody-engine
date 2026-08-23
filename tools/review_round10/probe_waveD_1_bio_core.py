"""Round-10 Wave D probe 1: bioinformatics/core/elastic_spatial_hash.py and
bioinformatics/core/fast_multipole_kernel.py against independent oracles.

Oracle strategy:
  - Morton: scalar bit-interleave reference written independently (loop over
    bit positions, not the magic-constant Part1By2 chain).
  - TreeFreeBioFMM: direct O(N^2) double loop written from the kernel
    definitions; forces vs central finite differences of the exact potential.
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from bioinformatics.core.elastic_spatial_hash import (
    morton_encode_3d, morton_decode_3d, ElasticSpatialHash3D)
from bioinformatics.core.fast_multipole_kernel import (
    TreeFreeBioFMM, ScreenedKernelType, TaylorYukawaBioFMM,
    COULOMB_CONSTANT_KCAL, toy_2cell_check_bio)

rng = np.random.default_rng(20260822)
FAIL = []

def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)

# ---------------------------------------------------------------- morton
def ref_encode(ix, iy, iz):
    """Independent scalar reference: interleave bit i of z,y,x (x lowest)."""
    code = 0
    for b in range(21):
        code |= ((ix >> b) & 1) << (3 * b)
        code |= ((iy >> b) & 1) << (3 * b + 1)
        code |= ((iz >> b) & 1) << (3 * b + 2)
    return code

xs = rng.integers(0, 1 << 21, size=500)
ys = rng.integers(0, 1 << 21, size=500)
zs = rng.integers(0, 1 << 21, size=500)
got = morton_encode_3d(xs, ys, zs)
want = np.array([ref_encode(int(a), int(b), int(c)) for a, b, c in zip(xs, ys, zs)])
check("morton_encode_3d vs scalar bit-interleave", np.array_equal(got, want))

# boundary values
edge = [0, 1, (1 << 21) - 1, 1 << 20]
ok_rt = True
for a in edge:
    for b in edge:
        c = int(morton_encode_3d(np.int64(a), np.int64(b), np.int64(0)))
        dx, dy, dz = morton_decode_3d(c)
        if (dx, dy, dz) != (a, b, 0):
            ok_rt = False
rr = rng.integers(0, 1 << 21, size=(3, 300))
codes = morton_encode_3d(rr[0], rr[1], rr[2])
for cd, a, b, c in zip(codes, rr[0], rr[1], rr[2]):
    if morton_decode_3d(int(cd)) != (int(a), int(b), int(c)):
        ok_rt = False
check("morton encode/decode roundtrip (edges+random)", ok_rt)

# ------------------------------------------------- ElasticSpatialHash3D facade
h = ElasticSpatialHash3D(cell_size=2.0, capacity_hint=256)
keys = list(rng.integers(0, 1 << 40, size=200).astype(np.int64))
ok_ins = ok_look = ok_dup = True
for i, k in enumerate(keys):
    if not h.insert(int(k), i):
        ok_ins = False
for i, k in enumerate(keys):
    if h.lookup(int(k)) != i:
        ok_look = False
h.insert(int(keys[0]), "dup")           # duplicate key updates value
ok_dup = (h.lookup(int(keys[0])) == "dup")
absent = h.lookup(123456789012345)
val, probes = h.lookup_with_probes(int(keys[5]))
check("facade insert/lookup all keys", ok_ins and ok_look)
check("facade duplicate insert updates", ok_dup, f"(absent->{absent})")
check("facade probe count <= probe_bound", probes <= h.probe_bound,
      f"(probes={probes}, bound={h.probe_bound})")

# build_from_coords: duplicates, permutation symmetry, cluster inverse
coords = rng.uniform(0, 20, size=(120, 3))
coords[50:60] = coords[0:10]  # duplicate block
mk, uk, inv = h.build_from_coords(coords)
n_unique_ref = len(set(map(int, mk)))
check("build_from_coords unique keys count", len(uk) == n_unique_ref,
      f"({len(uk)} vs {n_unique_ref})")
# permutation symmetry
perm = rng.permutation(120)
mk2, uk2, inv2 = h.build_from_coords(coords[perm])
check("build_from_coords permutation-invariant key set",
      set(map(int, uk)) == set(map(int, uk2)))
check("inverse consistent under permutation",
      all(inv2[i] == inv[j] for i, j in enumerate(perm)))

# ------------------------------------------------- TreeFreeBioFMM oracles
def direct_dh_potential(coords, charges, kappa, eps):
    N = len(coords)
    out = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            r = np.linalg.norm(coords[i] - coords[j])
            out[i] += charges[j] * np.exp(-kappa * r) / r
    return out * COULOMB_CONSTANT_KCAL / eps

def direct_dh_force(coords, charges, kappa, eps):
    """F_i = -q_i d/dx_i sum_{j!=i} q_j K K_const with K=exp(-kr)/r."""
    N = len(coords)
    out = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = coords[i] - coords[j]
            r = np.linalg.norm(d)
            e = np.exp(-kappa * r)
            dK = -e * (kappa / r + 1.0 / r**2)     # dK/dr
            out[i] += charges[j] * dK * d / r * (-charges[i])
    return out * COULOMB_CONSTANT_KCAL / eps

def rel(a, b):
    return np.linalg.norm(a - b) / max(1e-30, np.linalg.norm(b))

# (a) coarse cell -> mostly far-field: 8 clusters spread out, 4 atoms each
centers = np.array([[0, 0, 0], [40, 0, 0], [0, 40, 0], [0, 0, 40],
                    [40, 40, 0], [40, 0, 40], [0, 40, 40], [40, 40, 40]], dtype=float)
pts = np.vstack([c + rng.uniform(-2, 2, size=(4, 3)) for c in centers])
q = rng.uniform(-1, 1, size=len(pts))
for kappa, eps, tag in [(0.127, 78.5, "DH"), (0.0, 4.0, "kappa=0")]:
    eng = TreeFreeBioFMM(cell_size=8.0, kappa=kappa, dielectric_water=eps,
                         kernel_type=ScreenedKernelType.DEBYE_HUCKEL)
    pot, _, meta = eng.evaluate(pts, q)
    ref = direct_dh_potential(pts, q, kappa, eps)
    check(f"TreeFreeBioFMM DH potential vs direct ({tag})", rel(pot, ref) < 1e-3,
          f"rel={rel(pot, ref):.2e} K={meta['num_clusters']}")

# kappa=0 uses eps_w for DH; redo reference with eps_w
eng = TreeFreeBioFMM(cell_size=8.0, kappa=0.0, dielectric_water=78.5,
                     kernel_type=ScreenedKernelType.DEBYE_HUCKEL)
pot, _, _ = eng.evaluate(pts, q)
ref = direct_dh_potential(pts, q, 0.0, 78.5)
check("TreeFreeBioFMM DH kappa=0 potential vs direct (eps_w)",
      rel(pot, ref) < 1e-3, f"rel={rel(pot, ref):.2e}")

# (b) COULOMB kernel (uses eps_p)
eng = TreeFreeBioFMM(cell_size=8.0, kernel_type=ScreenedKernelType.COULOMB,
                     dielectric_protein=4.0)
pot, _, _ = eng.evaluate(pts, q)
ref = np.zeros(len(pts))
for i in range(len(pts)):
    for j in range(len(pts)):
        if i != j:
            ref[i] += q[j] / np.linalg.norm(pts[i] - pts[j])
ref *= COULOMB_CONSTANT_KCAL / 4.0
check("TreeFreeBioFMM COULOMB potential vs direct", rel(pot, ref) < 1e-3,
      f"rel={rel(pot, ref):.2e}")

# (c) forces vs direct analytic force (DH)
eng = TreeFreeBioFMM(cell_size=8.0, kappa=0.127, dielectric_water=78.5)
pot, F, _ = eng.evaluate(pts, q, compute_forces=True)
refF = direct_dh_force(pts, q, 0.127, 78.5)
check("TreeFreeBioFMM DH forces vs direct analytic", rel(F, refF) < 0.02,
      f"rel={rel(F, refF):.2e}")

# (d) forces vs per-atom central finite differences of the exact potential
#     (all-near configuration isolates the near-field force path).
#     NOTE: V_i is per unit TARGET charge, so F_i = q_i * (-grad V_i).
def fd_force(eval_fn, pts, q, h_=1e-5):
    num = np.zeros((len(pts), 3))
    for i in range(len(pts)):
        for d in range(3):
            ep = pts.copy(); ep[i, d] += h_
            em = pts.copy(); em[i, d] -= h_
            num[i, d] = -(eval_fn(ep)[0][i] - eval_fn(em)[0][i]) / (2 * h_)
    return num * q[:, None]

eng2 = TreeFreeBioFMM(cell_size=100.0, kappa=0.127, dielectric_water=78.5)  # all near
pot2, F2, _ = eng2.evaluate(pts, q, compute_forces=True)
num = fd_force(lambda p: eng2.evaluate(p, q), pts, q)
check("TreeFreeBioFMM near-field forces = -grad V (per-atom FD)",
      rel(F2, num) < 1e-4, f"rel={rel(F2, num):.2e}")

# (e) permutation symmetry of potentials
pperm = rng.permutation(len(pts))
pot_p, _, _ = eng.evaluate(pts[pperm], q[pperm])
check("TreeFreeBioFMM potential permutation symmetry",
      rel(pot_p, pot[pperm]) < 1e-9, f"rel={rel(pot_p, pot[pperm]):.2e}")

# (f) N=0 and N=1 edge cases
p0, f0, m0 = TreeFreeBioFMM().evaluate(np.empty((0, 3)), np.empty(0))
check("N=0 evaluate", p0.shape == (0,) and (f0 is None))
p1, f1, _ = TreeFreeBioFMM().evaluate(np.zeros((1, 3)), np.array([2.0]),
                                      compute_forces=True)
check("N=1 single atom: no self-interaction",
      p1[0] == 0.0 and np.all(f1 == 0.0))

# (g) YUKAWA kernel: near field covered?
small = pts[:8]  # all within one 8 A cell cluster region -> interactions near-field
eng_y = TreeFreeBioFMM(cell_size=8.0, kappa=0.127,
                       kernel_type=ScreenedKernelType.YUKAWA)
pot_y, Fy, _ = eng_y.evaluate(small, q[:8], compute_forces=True)
ref_y = direct_dh_potential(small, q[:8], 0.127, 78.5)
check("YUKAWA kernel potential vs direct (near-field)",
      rel(pot_y, ref_y) < 0.05, f"rel={rel(pot_y, ref_y):.2e} "
      f"(norm ratio {np.linalg.norm(pot_y)/max(1e-30,np.linalg.norm(ref_y)):.3f})")
check("YUKAWA kernel forces nonzero", np.any(Fy != 0))

# (h) GENERALIZED_BORN near field vs direct GB formula (single cell => all near)
br = rng.uniform(1.4, 2.0, size=len(small))
eng_gb = TreeFreeBioFMM(cell_size=100.0, kappa=0.127,
                        kernel_type=ScreenedKernelType.GENERALIZED_BORN)
pot_gb, Fgb, _ = eng_gb.evaluate(small, q[:8], born_radii=br, compute_forces=True)
ref_gb = np.zeros(len(small))
for i in range(len(small)):
    for j in range(len(small)):
        if i == j:
            continue
        r = np.linalg.norm(small[i] - small[j])
        ap = br[i] * br[j]
        f_gb = np.sqrt(r * r + ap * np.exp(-r * r / (4.0 * ap + 1e-8)))
        ref_gb[i] += q[:8][j] * (1.0 / 4.0 - np.exp(-0.127 * f_gb) / 78.5) \
            * COULOMB_CONSTANT_KCAL / f_gb
check("GB near-field potential vs direct GB formula", rel(pot_gb, ref_gb) < 1e-6,
      f"rel={rel(pot_gb, ref_gb):.2e}")
# GB force vs per-atom FD of own potential (times target charge)
numF = fd_force(lambda p: eng_gb.evaluate(p, q[:8], born_radii=br), small, q[:8])
check("GB near-field force = -grad V (per-atom FD)", rel(Fgb, numF) < 1e-3,
      f"rel={rel(Fgb, numF):.2e}")

# (i) TaylorYukawaBioFMM toy check (units wrapper)
check("toy_2cell_check_bio", toy_2cell_check_bio(p=6))

print()
print(f"{len(FAIL)} failures: {FAIL}")
sys.exit(1 if FAIL else 0)
