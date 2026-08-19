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


# =====================================================================
# 2. Derivative-tensor polynomials P_{alpha, n}  (2D multi-indices)
# =====================================================================
# Representation: P[alpha_tuple] -> {n: {monomial_exp_tuple: coef}}
# Identical recursion to gaussian2d_fgt.py (dimension-independent).

def _deriv_xi(poly: Dict[Tuple[int, int], float],
              axis: int) -> Dict[Tuple[int, int], float]:
    """d/dx_axis of a polynomial represented as {exp: coef}."""
    out: Dict[Tuple[int, int], float] = {}
    for (i, j), c in poly.items():
        exp = [i, j]
        e = exp[axis]
        if e > 0:
            exp2 = list(exp)
            exp2[axis] = e - 1
            out[tuple(exp2)] = out.get(tuple(exp2), 0.0) + c * e
    return out


def _mul_xi(poly: Dict[Tuple[int, int], float],
            axis: int) -> Dict[Tuple[int, int], float]:
    """Multiply by x_axis (shift exponent on `axis` by +1)."""
    out: Dict[Tuple[int, int], float] = {}
    for (i, j), c in poly.items():
        exp = [i, j]
        exp[axis] += 1
        out[tuple(exp)] = c
    return out


def _build_P_tensors(max_order: int) -> Dict[Tuple[int, int], Dict[int, Dict[Tuple[int, int], float]]]:
    """Build P_{alpha, n} for all |alpha| <= max_order, 0 <= n <= |alpha|.
    Recurrence (identical to gaussian2d_fgt.py):
        P_{(0,0), 0} = 1
        P_{alpha+e_i, n} = d/dx_i[P_{alpha,n}] + x_i * P_{alpha, n-1}
        P_{alpha, n} = 0 if n<0 or n>|alpha|.
    """
    P: Dict[Tuple[int, int], Dict[int, Dict[Tuple[int, int], float]]] = {}
    zero = (0, 0)
    P[zero] = {0: {(0, 0): 1.0}}

    all_alphas: List[Tuple[int, int]] = []
    for total in range(0, max_order + 1):
        for a in range(total + 1):
            b = total - a
            all_alphas.append((a, b))

    for alpha in all_alphas:
        if alpha == zero:
            continue
        P.setdefault(alpha, {})
        a, b = alpha
        order = a + b
        chosen_axis = 0 if a > 0 else 1
        alpha_prev = list(alpha)
        alpha_prev[chosen_axis] -= 1
        alpha_prev = tuple(alpha_prev)
        P_prev = P.get(alpha_prev, {})
        for n in range(order + 1):
            poly_n = P_prev.get(n)
            if poly_n:
                d = _deriv_xi(poly_n, axis=chosen_axis)
                if d:
                    P[alpha].setdefault(n, {})
                    for exp, cc in d.items():
                        P[alpha][n][exp] = P[alpha][n].get(exp, 0.0) + cc
            poly_nm1 = P_prev.get(n - 1)
            if poly_nm1:
                m = _mul_xi(poly_nm1, axis=chosen_axis)
                if m:
                    P[alpha].setdefault(n, {})
                    for exp, cc in m.items():
                        P[alpha][n][exp] = P[alpha][n].get(exp, 0.0) + cc
    return P


# =====================================================================
# Tensor evaluation over a batch of displacements
# =====================================================================

def _eval_poly(poly: Dict[Tuple[int, int], float],
               dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    out = np.zeros_like(dx)
    for (i, j), c in poly.items():
        out += c * (dx ** i) * (dy ** j)
    return out


class ScreenedYukawa2DFMM:
    """Single-level flat 2D screened Yukawa (K0) Taylor FMM."""

    def __init__(self, depth: int = 6, p: int = 8, kappa: float = 1.0,
                 ring_direct: int = 2):
        self.depth = int(depth)
        self.p = int(p)
        self.kappa = float(kappa)
        self.ring_direct = int(ring_direct)
        self.grid_res = 1 << self.depth
        # Laurent polynomials a_n, b_n for n = 0..2p (needed for D_{alpha+beta}).
        self._a, self._b = _build_ab_polynomials(max_n=2 * self.p)
        self._G_n = _make_G_n_evaluator(self.kappa, self._a, self._b)
        # Derivative-tensor polynomials P_{alpha, n} for |alpha| <= 2*p.
        self._P = _build_P_tensors(max_order=2 * self.p)
        self._alphas_p = _multi_indices(self.p)
        self._alphas_2p = _multi_indices(2 * self.p)
        self._alpha_p_index = {a: i for i, a in enumerate(self._alphas_p)}
        self._alpha_fact = np.array([_factorial(a) for a in self._alphas_p])

    # ------------------------------------------------------------------

    def evaluate(self, positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float64)
        charges = np.asarray(charges, dtype=np.float64)
        N = len(positions)
        if N == 0:
            return np.empty(0, dtype=np.float64)
        p = self.p
        depth = self.depth
        h_grid = 1.0 / depth
        kappa = self.kappa

        # 1. Build the CellIndex over occupied cells.
        cell_index = CellIndex(dims=2, grid_res=depth)
        unique_keys, inverse = cell_index.build(positions)
        K = len(unique_keys)
        inverse = np.asarray(inverse, dtype=np.int64)

        cell_ints = np.array([cell_index.key_ints(int(k)) for k in unique_keys], dtype=np.int64)  # (K,2)
        centers = (cell_ints.astype(np.float64) + 0.5) * h_grid  # (K,2)

        # 2. P2M: M_beta(cell) = sum_i q_i (x_i - c)^beta / beta!
        n_mom = len(self._alphas_p)
        pc = centers[inverse]
        disp = positions - pc
        dx_p = disp[:, 0]; dy_p = disp[:, 1]
        M = np.zeros((n_mom, K), dtype=np.float64)
        for bi, beta in enumerate(self._alphas_p):
            a, b = beta
            w = charges * (dx_p ** a) * (dy_p ** b) / _factorial(beta)
            M[bi] = np.bincount(inverse, weights=w, minlength=K)

        # 3. Near field: exact direct over ring-2 neighborhood, K0(kappa*r).
        pot = np.zeros(N, dtype=np.float64)
        for c_id, key in enumerate(unique_keys):
            idx_t = cell_index.bucket(int(key))
            if len(idx_t) == 0:
                continue
            near_idx = cell_index.neighborhood_indices(int(key), ring=self.ring_direct)
            if len(near_idx) == 0:
                continue
            xt = positions[idx_t]
            xs = positions[near_idx]
            qs = charges[near_idx]
            diff = xt[:, None, :] - xs[None, :, :]
            r = np.sqrt(np.sum(diff * diff, axis=-1))
            # K0 is log-singular at r=0; mask self pairs (r=0) to 0.
            r_safe = np.where(r < 1e-30, 1.0, r)
            g = kn(0, kappa * r_safe)
            id_t = idx_t[:, None]
            id_s = near_idx[None, :]
            self_mask = (id_t == id_s)
            g = np.where(self_mask, 0.0, g)
            pot[idx_t] += np.sum(qs[None, :] * g, axis=1)

        # 4. Far field: M2L over well-separated pairs, then L2P.
        if K > 1:
            ci = cell_ints.astype(np.int64)
            dci = ci[:, None, :] - ci[None, :, :]
            cheb = np.max(np.abs(dci), axis=-1)
            far_mask = cheb > self.ring_direct
            d_ts = centers[:, None, :] - centers[None, :, :]
            dx = d_ts[:, :, 0]; dy = d_ts[:, :, 1]
            r_ts = np.sqrt(dx * dx + dy * dy)
            r_far = np.where(far_mask, r_ts, 1.0)
            Gn = np.stack([self._G_n(r_far, n) for n in range(2 * p + 1)], axis=-1)
            Gn = np.where(far_mask[:, :, None], Gn, 0.0)

            n_loc = n_mom
            L = np.zeros((n_loc, K), dtype=np.float64)
            decomps = self._decompositions()
            for gamma in self._alphas_2p:
                d_list = decomps.get(gamma)
                if not d_list:
                    continue
                D_gamma = self._eval_D_tensor(gamma, dx, dy, Gn)
                D_gamma = np.where(far_mask, D_gamma, 0.0)
                betas_idx = np.array([bi for (_, bi, _) in d_list], dtype=np.int64)
                signs = np.array([s for (_, _, s) in d_list], dtype=np.float64)
                Mstack = M[betas_idx] * signs[:, None]
                contrib = D_gamma @ Mstack.T
                for j, (ai, _, _) in enumerate(d_list):
                    L[ai] += contrib[:, j]

            one_over_fact = 1.0 / self._alpha_fact
            Lp = L[:, inverse]
            far_pot = np.zeros(N, dtype=np.float64)
            for ai, alpha in enumerate(self._alphas_p):
                a, b = alpha
                far_pot += one_over_fact[ai] * Lp[ai] * (dx_p ** a) * (dy_p ** b)
            pot += far_pot

        self.cell_index = cell_index
        self._last_M = M
        return pot

    # ------------------------------------------------------------------

    def _eval_D_tensor(self, gamma: Tuple[int, int],
                       dx: np.ndarray, dy: np.ndarray,
                       Gn: np.ndarray) -> np.ndarray:
        """D_gamma(d) = sum_n P_{gamma,n}(d) * G_n(|d|)."""
        out = np.zeros_like(dx)
        Pgamma = self._P.get(gamma, {})
        for n, poly in Pgamma.items():
            Pval = _eval_poly(poly, dx, dy)
            out += Pval * Gn[:, :, n]
        return out

    def _decompositions(self) -> Dict[Tuple[int, int], List[Tuple[int, int, float]]]:
        """For each gamma with |gamma|<=2p, list (alpha_idx, beta_idx, sign)
        with |alpha|<=p, |beta|<=p, alpha+beta=gamma, sign=(-1)^|beta|."""
        p = self.p
        out: Dict[Tuple[int, int], List[Tuple[int, int, float]]] = {}
        for ai, alpha in enumerate(self._alphas_p):
            for bi, beta in enumerate(self._alphas_p):
                gamma = (alpha[0] + beta[0], alpha[1] + beta[1])
                if gamma[0] + gamma[1] > 2 * p:
                    continue
                out.setdefault(gamma, []).append(
                    (ai, bi, (-1.0) ** (beta[0] + beta[1])))
        return out

    # ------------------------------------------------------------------

    def D_alpha(self, d: np.ndarray, alpha: Tuple[int, int]) -> float:
        """Single-point derivative tensor D_alpha(d) for a 2-vector d.
        Used by the FD guard test."""
        d = np.asarray(d, dtype=np.float64).reshape(2)
        r = float(np.linalg.norm(d))
        if r < 1e-30:
            return 0.0
        val = 0.0
        Palpha = self._P.get(alpha, {})
        for n, poly in Palpha.items():
            Pval = 0.0
            for (i, j), c in poly.items():
                Pval += c * (d[0] ** i) * (d[1] ** j)
            val += Pval * float(self._G_n(np.array([r]), n)[0])
        return val


# =====================================================================
# Helpers
# =====================================================================

def _multi_indices(order: int) -> List[Tuple[int, int]]:
    out = []
    for total in range(order + 1):
        for a in range(total + 1):
            b = total - a
            out.append((a, b))
    return out


def _factorial(alpha: Tuple[int, int]) -> int:
    from math import factorial
    return factorial(alpha[0]) * factorial(alpha[1])


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
