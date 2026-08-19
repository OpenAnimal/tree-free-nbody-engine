"""2D Screened Yukawa (Debye-Huckel) Taylor Fast Multipole Method.

Kernel:  G(r) = K0(kappa * r)   (modified Bessel function of the second
                                 kind, order 0; the 2D screened Poisson
                                 / Helmholtz Green's function).

Single-level flat scheme on a uniform 2D grid, indexed by
`core.spatial_index.CellIndex(dims=2, grid_res=depth)` + funnel-hash
cell->moments storage.  Structured verbatim like `core/gaussian2d_fgt.py`
(2D P_{alpha,n} recursion, ring-2 direct near field, corrected
sign/factorial assembly); only the radial functions G_n differ.

------------------------------------------------------------------------------
MATH (transcribed literally from the round-5 implementation plan, 5.4).
------------------------------------------------------------------------------
The kernel K0(z), z = kappa*r, satisfies
    dK0/dz = -K1,    dK1/dz = -K0 - K1/z.
The radial operator (1/r d/dr) acts on G_n as G_{n+1} = (1/r d/dr) G_n,
and since d/dr = kappa d/dz and r = z/kappa,
    (1/r d/dr) G = (kappa/z) * kappa * dG/dz = kappa^2 (1/z d/dz) G.
Writing G_n(r) = kappa^(2n) * [ a_n(z) K0(z) + b_n(z) K1(z) ] and applying
(1/z d/dz) to a_n K0 + b_n K1, using the two Bessel recurrences above,
yields the literal plan recursions:
    a_0 = 1, b_0 = 0,
    a_{n+1}(z) = ( a_n'(z) - b_n(z) ) / z
    b_{n+1}(z) = ( b_n'(z) - a_n(z) - b_n(z)/z ) / z
These a_n, b_n are rational functions of z (Laurent polynomials in z,
i.e. integer powers of z, possibly negative).  They are built EXACTLY
ONCE per p as Laurent-polynomial dicts {exponent: coef} and evaluated at
far-cell centers only (z = kappa*r, r >= 3*h_grid), never at r=0 (K0 has
a log singularity at r=0).

A numeric guard (`bessel_recursion_guard`) compares G_1, G_2 against a
central-difference (1/r d/dr) of G_0, G_1 at 5 radii, rel tol 1e-8.

Everything else (P_{alpha,n} recursion, M2L, near field, L2P) reuses the
gaussian2d_fgt structure verbatim.  See that module for the sign /
factorial convention (the corrected round-3 form, verified by the
mandatory 2-cell toy check below).

------------------------------------------------------------------------------
API: class ScreenedYukawa2DFMM(depth=6, p=8, kappa=1.0) with
.evaluate(positions, charges) -> potentials (float64).
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.special import kn  # kn(n, z) = K_n(z)

try:
    from .spatial_index import CellIndex
except ImportError:  # pragma: no cover - direct module execution
    from spatial_index import CellIndex

try:
    from .radial_taylor import RadialTaylorFMM, multi_indices as _rt_multi_indices, factorial as _rt_factorial
except ImportError:  # pragma: no cover - direct module execution
    from radial_taylor import RadialTaylorFMM, multi_indices as _rt_multi_indices, factorial as _rt_factorial


# =====================================================================
# 1. Laurent polynomials a_n(z), b_n(z) and radial functions G_n(r)
# =====================================================================
# A Laurent polynomial is represented as {exponent: coef} (exponent any int).

def _lp_deriv(p: Dict[int, float]) -> Dict[int, float]:
    """d/dz of a Laurent polynomial {exp: coef}."""
    out: Dict[int, float] = {}
    for e, c in p.items():
        if e != 0:
            out[e - 1] = out.get(e - 1, 0.0) + c * e
    return out


def _lp_shift(p: Dict[int, float], k: int) -> Dict[int, float]:
    """Multiply by z^k (shift every exponent by +k)."""
    return {e + k: c for e, c in p.items()}


def _lp_sub(a: Dict[int, float], b: Dict[int, float]) -> Dict[int, float]:
    """a - b for Laurent polynomials (drops exact-zero coefs)."""
    out = dict(a)
    for e, c in b.items():
        out[e] = out.get(e, 0.0) - c
    return {e: c for e, c in out.items() if c != 0.0}


def _lp_eval(p: Dict[int, float], z: np.ndarray) -> np.ndarray:
    """Evaluate a Laurent polynomial at z (scalar or array)."""
    if not p:
        return np.zeros_like(np.asarray(z, dtype=np.float64))
    out = np.zeros_like(np.asarray(z, dtype=np.float64))
    for e, c in p.items():
        out = out + c * (z ** e)
    return out


def _build_ab_polynomials(max_n: int) -> Tuple[List[Dict[int, float]],
                                               List[Dict[int, float]]]:
    """Build a_n, b_n for n = 0..max_n via the literal plan recursion:
        a_0 = 1, b_0 = 0,
        a_{n+1} = (a_n' - b_n) / z
        b_{n+1} = (b_n' - a_n - b_n/z) / z
    """
    a: List[Dict[int, float]] = [{0: 1.0}]
    b: List[Dict[int, float]] = [{}]
    for n in range(max_n):
        an, bn = a[n], b[n]
        # a_{n+1} = (a_n' - b_n) * z^{-1}
        an1 = _lp_shift(_lp_sub(_lp_deriv(an), bn), -1)
        # b_{n+1} = (b_n' - a_n - b_n * z^{-1}) * z^{-1}
        bn1 = _lp_shift(_lp_sub(_lp_sub(_lp_deriv(bn), an), _lp_shift(bn, -1)), -1)
        a.append(an1)
        b.append(bn1)
    return a, b


def _make_G_n_evaluator(kappa: float,
                        a_polys: List[Dict[int, float]],
                        b_polys: List[Dict[int, float]]):
    """Return G_n(r, n) for r an array, n an int.

    G_n(r) = kappa^(2n) * [ a_n(z) * K0(z) + b_n(z) * K1(z) ],  z = kappa*r.
    """
    k2 = [float(kappa) ** (2 * n) for n in range(len(a_polys))]

    def G_n(r: np.ndarray, n: int) -> np.ndarray:
        z = kappa * r
        # K0, K1 from scipy.special.kn.  r (hence z) is > 0 at far centers.
        K0 = kn(0, z)
        K1 = kn(1, z)
        return k2[n] * (_lp_eval(a_polys[n], z) * K0
                        + _lp_eval(b_polys[n], z) * K1)

    return G_n


def _make_near_field_kernel(kappa: float):
    """Near-field kernel for the 2D screened Yukawa: G(r) = K0(kappa*r)."""
    k = float(kappa)

    def kernel(diff: np.ndarray) -> np.ndarray:
        r = np.sqrt(np.sum(diff * diff, axis=-1))
        # K0 is log-singular at r=0; the driver masks self pairs to 0
        # afterwards, so the r_safe value at r=0 is never used.
        r_safe = np.where(r < 1e-30, 1.0, r)
        return kn(0, k * r_safe)

    return kernel


class ScreenedYukawa2DFMM(RadialTaylorFMM):
    """Single-level flat 2D screened Yukawa (K0) Taylor FMM.

    Thin wrapper over RadialTaylorFMM supplying the Bessel K0 G_n family
    (a_n/b_n Laurent polynomials) and the K0(kappa*r) near-field kernel.
    """

    def __init__(self, depth: int = 6, p: int = 8, kappa: float = 1.0,
                 ring_direct: int = 2):
        self.kappa = float(kappa)
        # Laurent polynomials a_n, b_n for n = 0..2p (needed for D_{alpha+beta}).
        self._a, self._b = _build_ab_polynomials(max_n=2 * int(p))
        G_n = _make_G_n_evaluator(self.kappa, self._a, self._b)
        near_field = _make_near_field_kernel(self.kappa)
        super().__init__(depth=depth, p=p, dims=2, G_n=G_n,
                         near_field_kernel=near_field,
                         ring_direct=ring_direct)


# =====================================================================
# Helpers (thin wrappers over radial_taylor for backward compat)
# =====================================================================

def _multi_indices(order: int) -> List[Tuple[int, int]]:
    return _rt_multi_indices(order, 2)


def _factorial(alpha: Tuple[int, int]) -> int:
    return _rt_factorial(alpha)


# =====================================================================
# Guard 1: Bessel recursion (a_n, b_n) vs central-difference (1/r d/dr)
# =====================================================================

def bessel_recursion_guard(kappa: float = 1.0, p: int = 8,
                           rel_tol: float = 1e-8) -> bool:
    """Compare G_1, G_2 (from the a_n/b_n Laurent recursion) against a
    central-difference (1/r d/dr) of G_0, G_1 at 5 radii.  rel tol 1e-8."""
    fmm = ScreenedYukawa2DFMM(depth=6, p=p, kappa=kappa)
    radii = np.array([0.05, 0.13, 0.27, 0.41, 0.55], dtype=np.float64)
    dr = 1e-6
    worst = 0.0
    for r in radii:
        G0 = float(fmm._G_n(np.array([r]), 0)[0])
        G0p = float(fmm._G_n(np.array([r + dr]), 0)[0])
        G0m = float(fmm._G_n(np.array([r - dr]), 0)[0])
        G1_num = (G0p - G0m) / (2.0 * r * dr)
        G1_ana = float(fmm._G_n(np.array([r]), 1)[0])
        denom = max(1e-12, abs(G1_num), abs(G1_ana))
        rel = abs(G1_num - G1_ana) / denom
        worst = max(worst, rel)
        if rel > rel_tol:
            print(f"BESSEL GUARD FAIL (G1): r={r} num={G1_num:.6e} "
                  f"ana={G1_ana:.6e} rel={rel:.2e}")
            return False
        G1p = float(fmm._G_n(np.array([r + dr]), 1)[0])
        G1m = float(fmm._G_n(np.array([r - dr]), 1)[0])
        G2_num = (G1p - G1m) / (2.0 * r * dr)
        G2_ana = float(fmm._G_n(np.array([r]), 2)[0])
        denom = max(1e-12, abs(G2_num), abs(G2_ana))
        rel = abs(G2_num - G2_ana) / denom
        worst = max(worst, rel)
        if rel > rel_tol:
            print(f"BESSEL GUARD FAIL (G2): r={r} num={G2_num:.6e} "
                  f"ana={G2_ana:.6e} rel={rel:.2e}")
            return False
    print(f"bessel_recursion_guard: worst rel err = {worst:.2e} "
          f"(tol {rel_tol:.0e}) -- PASS")
    return True


# =====================================================================
# Guard 2: derivative tensor vs central finite differences (|alpha|<=2)
# =====================================================================

def _finite_diff_D(fmm: "ScreenedYukawa2DFMM", d: np.ndarray,
                   alpha: Tuple[int, int], h: float = 3e-4) -> float:
    """Central finite difference of G(r)=K0(kappa*r) for the multi-index
    alpha, |alpha|<=2.  Uses 4th-order stencils for first and pure-second
    derivatives (same rationale as gaussian2d_fgt.py: the K0 kernel's
    higher derivatives grow with n, so the O(h^2) 2-point/3-point stencils
    would false-fail at h=3e-4)."""
    a, b = alpha
    order = a + b
    kappa = fmm.kappa

    def G(vec):
        r = float(np.linalg.norm(vec))
        return float(kn(0, kappa * r))

    if order == 0:
        return G(d)
    if order == 1:
        axis = [a, b].index(1)
        e = np.zeros(2); e[axis] = h
        # 4th-order central first derivative.
        return float((G(d - 2 * e) - 8 * G(d - e)
                      + 8 * G(d + e) - G(d + 2 * e)) / (12 * h))
    if order == 2:
        if a == 2 or b == 2:
            axis = 0 if a == 2 else 1
            e = np.zeros(2); e[axis] = h
            return float((-G(d + 2 * e) + 16 * G(d + e) - 30 * G(d)
                          + 16 * G(d - e) - G(d - 2 * e)) / (12 * h * h))
        # mixed: 4th-order 16-point stencil.
        offsets = (-2, -1, 1, 2)
        coeffs = (1, -8, 8, -1)
        s = 0.0
        for ci, ai in zip(coeffs, offsets):
            for cj, aj in zip(coeffs, offsets):
                e = np.array([ai * h, aj * h], dtype=np.float64)
                s += ci * cj * G(d + e)
        return float(s / (144.0 * h * h))
    raise ValueError("finite-difference guard only supports |alpha|<=2")


def derivative_fd_guard(kappa: float = 1.0, p: int = 8,
                        h_fd: float = 3e-4, rel_tol: float = 1e-5) -> bool:
    """Validate D_alpha against central finite differences for |alpha|<=2 on
    several non-axis-aligned displacements.  Returns True iff all pass."""
    fmm = ScreenedYukawa2DFMM(depth=6, p=p, kappa=kappa)
    test_ds = [
        np.array([0.3, 0.17]),
        np.array([0.55, 0.21]),
        np.array([-0.27, 0.62]),
        np.array([0.11, -0.73]),
        np.array([0.42, 0.48]),
    ]
    alphas = _multi_indices(2)
    worst = 0.0
    for d in test_ds:
        for alpha in alphas:
            ana = fmm.D_alpha(d, alpha)
            fd = _finite_diff_D(fmm, d, alpha, h=h_fd)
            denom = max(1e-12, abs(fd), abs(ana))
            rel = abs(ana - fd) / denom
            worst = max(worst, rel)
            if rel > rel_tol:
                print(f"FD GUARD FAIL: alpha={alpha} d={d} ana={ana:.6e} "
                      f"fd={fd:.6e} rel={rel:.2e}")
                return False
    print(f"derivative_fd_guard: worst rel err = {worst:.2e} "
          f"(tol {rel_tol:.0e}) -- PASS")
    return True


# =====================================================================
# 2-cell toy check (sign/factorial convention verification)
# =====================================================================

def toy_2cell_check(kappa: float = 1.0, p: int = 8) -> bool:
    """Two cells, a handful of particles, compare FMM vs exact direct.
    Mandatory sign-convention check before scaling up."""
    rng = np.random.default_rng(0)
    depth = 4
    h_grid = 1.0 / depth
    c1 = (np.array([3, 3]) + 0.5) * h_grid
    c2 = (np.array([10, 10]) + 0.5) * h_grid
    n1, n2 = 4, 5
    pts1 = c1 + rng.uniform(-h_grid * 0.4, h_grid * 0.4, size=(n1, 2))
    pts2 = c2 + rng.uniform(-h_grid * 0.4, h_grid * 0.4, size=(n2, 2))
    pts = np.vstack([pts1, pts2])
    q = rng.uniform(-1.0, 1.0, size=len(pts))
    pot_exact = np.zeros(len(pts))
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i == j:
                continue
            r = np.linalg.norm(pts[i] - pts[j])
            pot_exact[i] += q[j] * float(kn(0, kappa * r))
    fmm = ScreenedYukawa2DFMM(depth=depth, p=p, kappa=kappa)
    pot_fmm = fmm.evaluate(pts, q)
    rel = np.linalg.norm(pot_fmm - pot_exact) / max(1e-30, np.linalg.norm(pot_exact))
    print(f"toy_2cell_check: rel-L2 = {rel:.3e} (target < 1e-10) "
          f"{'PASS' if rel < 1e-10 else 'FAIL'}")
    return rel < 1e-10


if __name__ == "__main__":
    ok_be = bessel_recursion_guard()
    ok_fd = derivative_fd_guard()
    ok_toy = toy_2cell_check()
    if not (ok_be and ok_fd and ok_toy):
        raise SystemExit(1)
    print("screened_yukawa2d_fmm guards: PASS")
