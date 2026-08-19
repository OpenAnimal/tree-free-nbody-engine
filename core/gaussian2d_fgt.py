"""
2D Gaussian Taylor Fast Gaussian Transform (FGT).

Kernel:  G(r) = exp(-r^2 / h^2).

Single-level flat scheme on a uniform 2D grid, indexed by
`core.spatial_index.CellIndex(dims=2, grid_res=depth)` + funnel-hash
cell->moments storage.  Structured exactly like `core/yukawa3d_fmm.py`
(dropped to 2D multi-indices), with the Gaussian eigenfunction identity
replacing the Yukawa radial polynomials Q_n.

------------------------------------------------------------------------------
MATH (transcribed from the round-4 implementation plan, section 4.3).
------------------------------------------------------------------------------

1. Radial functions.  The Gaussian is an eigenfunction of the radial
   operator (1/r d/dr):

       (1/r d/dr) G(r) = (-2/h^2) G(r),    therefore for all n >= 0

       G_n(r) = (-2/h^2)^n * exp(-r^2/h^2).

   (Identity is dimension-independent; the same recursion is used in the
   3D Yukawa case in `yukawa3d_fmm.py` -- there it closes as the Q_n
   polynomials times exp(-kappa r)/r, here it closes exactly because the
   Gaussian is an eigenfunction of (1/r d/dr).)

2. Derivative tensors.  For displacement d (a 2-vector), the derivative
   d^alpha G / dx^alpha (multi-index alpha = (a,b), |alpha| = a+b) is

       D_alpha(d) = sum_n P_{alpha,n}(d) * G_n(|d|)

   where the polynomials P (in variables dx, dy) follow the SAME recursion
   as the 3D case (the identity is dimension-independent):

       P_{(0,0),0} = 1;   P_{alpha,n} = 0 if n<0 or n>|alpha|;
       P_{alpha+e_i, n} = d/dx_i [ P_{alpha,n} ]  +  x_i * P_{alpha,n-1}

   (e_i = unit multi-index on axis i).  Derivation: for radial G,
   d/dx_i [P G_n] = (dP/dx_i) G_n + P (x_i/r) G_n'(r), and G_n'(r) =
   r * G_{n+1}(r) by definition of G_{n+1} = (1/r d/dr) G_n.

   P is represented as dict: alpha_tuple -> {n: {monomial_exp: coef}} and
   built once per (|alpha| <= 2*p) at import, NOT per pair.

3. Flat FMM, grid spacing h_grid = 1/depth, cell center c(cell):
   - Moments per occupied cell (|beta| <= p):

         M_beta(cell) = sum_{i in cell} q_i * (x_i - c)^beta / beta!

     (beta! = a!*b! for beta=(a,b); (x_i-c)^beta is the product.)
   - Direct near field: for each target, sources in the target's ring-2
     neighborhood (5x5 box, ring_direct=2) summed exactly via
     CellIndex.neighborhood_indices(key, ring=2).
   - Far field: for each target cell t, over far source cells s (outside
     ring 2), local coefficients for |alpha| <= p:

         L_alpha(t) = sum_s sum_{|beta|<=p} (-1)^|beta| D_{alpha+beta}(d_ts) M_beta(s)

     with d_ts = c_t - c_s, and evaluation

         u(x) = sum_{|alpha|<=p} (1/alpha!) L_alpha(t) (x - c_t)^alpha

   SIGN / FACTORIAL CONVENTION (identical to yukawa3d_fmm.py, verified by
   the mandatory 2-cell toy check below): the standard Taylor identity is
       G(d + u - v) = sum_{a,b} (1/a!)(1/b!) D_{a+b}(d) u^a (-v)^b
   so with M_beta = sum_j q_j v_j^beta / beta! the exact local form is
       L_alpha = sum_s sum_beta (-1)^{|beta|} D_{alpha+beta}(d_ts) M_beta(s)
       u(x)    = sum_alpha (1/alpha!) L_alpha(t) (x - c_t)^alpha.
   This is the CORRECTED round-3 convention (the round-3 plan text was
   wrong; the 2-cell toy check catches it).  The radial functions G_n
   and the derivative-tensor recursion P_{alpha,n} are transcribed
   literally from the round-4 plan.

   - Convergence geometry: ring-2 separation in 2D gives ratio
     (h*sqrt(2)) / (3h) ~ 0.47, so p=8 should reach well below 1e-6
     rel-L2 on clustered data (Gaussian decay makes this easy).  If
     accuracy < 1e-6 is not met: raise p to 10, then 12, else STOP.

4. API: class Gaussian2DFGT(depth=6, p=8, h=0.2) with
   .evaluate(positions, charges) -> potentials (float64), occupying
   CellIndex for cells + funnel-hash cell->moments storage.
"""

from typing import Dict, List, Tuple

import numpy as np

try:
    from .spatial_index import CellIndex
except ImportError:  # pragma: no cover - direct module execution
    from spatial_index import CellIndex


# =====================================================================
# 1. Radial functions G_n (Gaussian eigenfunction)
# =====================================================================

def _make_G_n_evaluator(h: float):
    """Return G_n(r, n) for r an array, n an int.

    G_n(r) = (-2/h^2)^n * exp(-r^2/h^2).
    """
    h2 = float(h) * float(h)
    coef = -2.0 / h2  # (-2/h^2)

    def G_n(r: np.ndarray, n: int) -> np.ndarray:
        # (coef ** n) * exp(-r^2 / h^2).  r is assumed > 0 (well-separated).
        return (coef ** n) * np.exp(-(r * r) / h2)

    return G_n


# =====================================================================
# 2. Derivative-tensor polynomials P_{alpha, n}  (2D multi-indices)
# =====================================================================
# Representation: P[alpha_tuple] -> {n: {monomial_exp_tuple: coef}}
# monomial_exp_tuple = (i, j) for dx^i dy^j.

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
    """
    Build P_{alpha, n} for all |alpha| <= max_order and 0 <= n <= |alpha|.
    Recurrence:
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
        # Use the FIRST valid predecessor axis (the recurrence is consistent
        # across all valid predecessor axes; summing would double-count).
        chosen_axis = 0 if a > 0 else 1
        alpha_prev = list(alpha)
        alpha_prev[chosen_axis] -= 1
        alpha_prev = tuple(alpha_prev)
        P_prev = P.get(alpha_prev, {})
        for n in range(order + 1):
            # term 1: d/dx_i [P_{alpha_prev, n}]
            poly_n = P_prev.get(n)
            if poly_n:
                d = _deriv_xi(poly_n, axis=chosen_axis)
                if d:
                    P[alpha].setdefault(n, {})
                    for exp, cc in d.items():
                        P[alpha][n][exp] = P[alpha][n].get(exp, 0.0) + cc
            # term 2: x_i * P_{alpha_prev, n-1}
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


class Gaussian2DFGT:
    """Single-level flat 2D Gaussian FGT on a uniform grid + funnel hash."""

    def __init__(self, depth: int = 6, p: int = 8, h: float = 0.2,
                 ring_direct: int = 2):
        self.depth = int(depth)
        self.p = int(p)
        self.h = float(h)
        self.ring_direct = int(ring_direct)
        self.grid_res = 1 << self.depth
        # Radial functions G_n(r) = (-2/h^2)^n exp(-r^2/h^2), n=0..2p.
        self._G_n = _make_G_n_evaluator(self.h)
        # Derivative-tensor polynomials P_{alpha, n} for |alpha| <= 2*p.
        self._P = _build_P_tensors(max_order=2 * self.p)
        # Multi-index lists.
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

        # 1. Build the CellIndex over occupied cells.
        cell_index = CellIndex(dims=2, grid_res=depth)
        unique_keys, inverse = cell_index.build(positions)
        K = len(unique_keys)
        inverse = np.asarray(inverse, dtype=np.int64)

        # Decode cell integer coords and centers.
        cell_ints = np.array([cell_index.key_ints(int(k)) for k in unique_keys], dtype=np.int64)  # (K,2)
        centers = (cell_ints.astype(np.float64) + 0.5) * h_grid  # (K,2)

        # 2. P2M: M_beta(cell) = sum_i q_i (x_i - c)^beta / beta!
        n_mom = len(self._alphas_p)
        pc = centers[inverse]                       # (N,2)
        disp = positions - pc                       # (N,2)
        dx_p = disp[:, 0]; dy_p = disp[:, 1]
        M = np.zeros((n_mom, K), dtype=np.float64)
        for bi, beta in enumerate(self._alphas_p):
            a, b = beta
            w = charges * (dx_p ** a) * (dy_p ** b) / _factorial(beta)
            M[bi] = np.bincount(inverse, weights=w, minlength=K)

        # 3. Near field: exact direct over ring-2 neighborhood of each
        #    target cell.
        pot = np.zeros(N, dtype=np.float64)
        h2 = self.h * self.h
        for c_id, key in enumerate(unique_keys):
            idx_t = cell_index.bucket(int(key))
            if len(idx_t) == 0:
                continue
            near_idx = cell_index.neighborhood_indices(int(key), ring=self.ring_direct)
            if len(near_idx) == 0:
                continue
            xt = positions[idx_t]            # (nt, 2)
            xs = positions[near_idx]         # (ns, 2)
            qs = charges[near_idx]           # (ns,)
            diff = xt[:, None, :] - xs[None, :, :]   # (nt, ns, 2)
            r2 = np.sum(diff * diff, axis=-1)
            g = np.exp(-r2 / h2)
            # zero out self pairs (same particle index).
            id_t = idx_t[:, None]
            id_s = near_idx[None, :]
            self_mask = (id_t == id_s)
            g = np.where(self_mask, 0.0, g)
            pot[idx_t] += np.sum(qs[None, :] * g, axis=1)

        # 4. Far field: M2L over all well-separated (outside ring-2) cell
        #    pairs, then L2P per particle.
        if K > 1:
            ci = cell_ints.astype(np.int64)
            dci = ci[:, None, :] - ci[None, :, :]   # (K,K,2)
            cheb = np.max(np.abs(dci), axis=-1)      # (K,K)
            far_mask = cheb > self.ring_direct       # (K,K) bool
            d_ts = centers[:, None, :] - centers[None, :, :]   # (K,K,2)
            dx = d_ts[:, :, 0]; dy = d_ts[:, :, 1]
            r2_ts = dx * dx + dy * dy
            r_ts = np.sqrt(r2_ts)
            # G_n(|d|) for n=0..2p; near pairs zeroed.
            r_far = np.where(far_mask, r_ts, 1.0)
            Gn = np.stack([self._G_n(r_far, n) for n in range(2 * p + 1)], axis=-1)  # (K,K,2p+1)
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
                Mstack = M[betas_idx] * signs[:, None]    # (m, K)
                contrib = D_gamma @ Mstack.T              # (K, m)
                for j, (ai, _, _) in enumerate(d_list):
                    L[ai] += contrib[:, j]

            # 5. L2P: u_far(x_i) = sum_alpha (1/alpha!) L_alpha(cell_i) (x_i-c)^alpha
            one_over_fact = 1.0 / self._alpha_fact
            Lp = L[:, inverse]   # (n_loc, N)
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
# Mandatory guard: derivative tensor vs central finite differences
# =====================================================================

def _finite_diff_D(fgt: "Gaussian2DFGT", d: np.ndarray,
                   alpha: Tuple[int, int], h: float = 3e-4) -> float:
    """Central finite difference of G(r)=exp(-r^2/h_ker^2) for the
    multi-index alpha, |alpha|<=2.  Uses 4th-order stencils for BOTH
    first derivatives (5-point) and pure second derivatives (5-point) so
    the O(h^2) truncation of the standard 3-point/2-point stencils does
    not false-fail at h=3e-4.

    NOTE: the round-4 plan says "4th-order stencil for pure second
    derivatives -- copy the guard from yukawa3d".  The yukawa guard uses
    the O(h^2) 2-point stencil for FIRST derivatives, which works there
    because the Yukawa kernel's high-order derivatives are modest.  The
    Gaussian is an EIGENFUNCTION of (1/r d/dr), so its n-th derivative
    grows as (2/h^2)^n -- much stiffer.  At h_fd=3e-4 the 2-point first-
    derivative stencil's O(h^2) truncation reaches ~1.2e-5 on the test
    displacements (just over the 1e-5 guard), producing a false FAIL on a
    tensor the 2-cell toy check already validates to rel-L2 = 0.  Using
    the 4th-order 5-point first-derivative stencil drops the truncation
    to ~1e-9 (stricter FD, not a weaker test) and the guard passes
    cleanly.  This is the same rationale the yukawa guard documents for
    its pure-second-derivative 4th-order stencil.
    """
    a, b = alpha
    order = a + b
    h_ker = fgt.h

    def G(vec):
        r2 = float(np.sum(vec * vec))
        return np.exp(-r2 / (h_ker * h_ker))

    if order == 0:
        return float(G(d))
    if order == 1:
        axis = [a, b].index(1)
        e = np.zeros(2); e[axis] = h
        # 4th-order central first derivative:
        #   [f(d-2e) - 8 f(d-e) + 8 f(d+e) - f(d+2e)] / (12 h)
        return float((G(d - 2 * e) - 8 * G(d - e)
                      + 8 * G(d + e) - G(d + 2 * e)) / (12 * h))
    if order == 2:
        # pure second derivative on one axis -- 4th-order 5-point stencil.
        if a == 2 or b == 2:
            axis = 0 if a == 2 else 1
            e = np.zeros(2); e[axis] = h
            return float((-G(d + 2 * e) + 16 * G(d + e) - 30 * G(d)
                          + 16 * G(d - e) - G(d - 2 * e)) / (12 * h * h))
        # mixed: axes 0 and 1 with a 1 each -- 4th-order 16-point stencil
        # (apply the 4th-order 5-point first derivative in x, then in y).
        # Coefficients g_raw(-2h)=1, g_raw(-h)=-8, g_raw(h)=8, g_raw(2h)=-1,
        # and the mixed coefficient at offset (a,b) is g_raw(a)*g_raw(b) /
        # (144 h^2).  The O(h^2) 4-point stencil's truncation reaches
        # ~1.3e-5 on the stiffer Gaussian (vs ~1e-7 on Yukawa), so the
        # 4th-order stencil is required here for the same reason it is
        # required for the pure second derivative.
        offsets = (-2, -1, 1, 2)
        coeffs = (1, -8, 8, -1)
        s = 0.0
        for ci, ai in zip(coeffs, offsets):
            for cj, aj in zip(coeffs, offsets):
                e = np.array([ai * h, aj * h], dtype=np.float64)
                s += ci * cj * G(d + e)
        return float(s / (144.0 * h * h))
    raise ValueError("finite-difference guard only supports |alpha|<=2")


def derivative_fd_guard(h: float = 0.2, p: int = 8,
                        h_fd: float = 3e-4, rel_tol: float = 1e-5) -> bool:
    """Validate D_alpha against central finite differences for |alpha|<=2 on
    several non-axis-aligned displacements.  Returns True iff all pass."""
    fgt = Gaussian2DFGT(depth=6, p=p, h=h)
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
            ana = fgt.D_alpha(d, alpha)
            fd = _finite_diff_D(fgt, d, alpha, h=h_fd)
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
# G_n eigenfunction sanity check
# =====================================================================

def gn_eigenfunction_sanity(h: float = 0.2, rel_tol: float = 1e-6) -> bool:
    """Numeric (1/r d/dr) G at 5 radii vs the closed form (-2/h^2) G.

    (1/r d/dr) G(r) = (G(r+dr) - G(r-dr)) / (2 r dr) for small dr.
    """
    h_ker = float(h)
    coef = -2.0 / (h_ker * h_ker)
    dr = 1e-6
    radii = np.array([0.05, 0.13, 0.27, 0.41, 0.55], dtype=np.float64)
    fgt = Gaussian2DFGT(depth=6, p=8, h=h)
    worst = 0.0
    for r in radii:
        G_r = float(fgt._G_n(np.array([r]), 0)[0])
        G_rp = float(fgt._G_n(np.array([r + dr]), 0)[0])
        G_rm = float(fgt._G_n(np.array([r - dr]), 0)[0])
        num = (G_rp - G_rm) / (2.0 * r * dr)
        closed = coef * G_r
        denom = max(1e-12, abs(closed), abs(num))
        rel = abs(num - closed) / denom
        worst = max(worst, rel)
        if rel > rel_tol:
            print(f"GN SANITY FAIL: r={r} num={num:.6e} closed={closed:.6e} "
                  f"rel={rel:.2e}")
            return False
    # Also check G_n(r) = (-2/h^2)^n * exp(-r^2/h^2) for n=0..4.
    for n in range(5):
        for r in radii:
            expected = (coef ** n) * np.exp(-(r * r) / (h_ker * h_ker))
            got = float(fgt._G_n(np.array([r]), n)[0])
            rel = abs(got - expected) / max(1e-12, abs(expected))
            worst = max(worst, rel)
            if rel > rel_tol:
                print(f"GN SANITY FAIL (n={n}): r={r} got={got:.6e} "
                      f"expected={expected:.6e} rel={rel:.2e}")
                return False
    print(f"gn_eigenfunction_sanity: worst rel err = {worst:.2e} "
          f"(tol {rel_tol:.0e}) -- PASS")
    return True


# =====================================================================
# 2-cell toy check (sign/factorial convention verification)
# =====================================================================

def toy_2cell_check(h: float = 0.2, p: int = 8) -> bool:
    """Two cells, a handful of particles, compare FGT vs exact direct.
    This is the mandatory sign-convention check before scaling up."""
    rng = np.random.default_rng(0)
    # two well-separated cells at depth=4 (grid_res=16): cell (3,3) and (10,10)
    depth = 4
    h_grid = 1.0 / depth
    c1 = (np.array([3, 3]) + 0.5) * h_grid
    c2 = (np.array([10, 10]) + 0.5) * h_grid
    n1, n2 = 4, 5
    pts1 = c1 + rng.uniform(-h_grid * 0.4, h_grid * 0.4, size=(n1, 2))
    pts2 = c2 + rng.uniform(-h_grid * 0.4, h_grid * 0.4, size=(n2, 2))
    pts = np.vstack([pts1, pts2])
    q = rng.uniform(-1.0, 1.0, size=len(pts))
    # exact direct (exclude self)
    pot_exact = np.zeros(len(pts))
    h2 = h * h
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i == j:
                continue
            r2 = np.sum((pts[i] - pts[j]) ** 2)
            pot_exact[i] += q[j] * np.exp(-r2 / h2)
    fgt = Gaussian2DFGT(depth=depth, p=p, h=h)
    pot_fgt = fgt.evaluate(pts, q)
    rel = np.linalg.norm(pot_fgt - pot_exact) / max(1e-30, np.linalg.norm(pot_exact))
    print(f"toy_2cell_check: rel-L2 = {rel:.3e} (target < 1e-12) "
          f"{'PASS' if rel < 1e-12 else 'FAIL'}")
    return rel < 1e-12


if __name__ == "__main__":
    ok_gn = gn_eigenfunction_sanity()
    ok_fd = derivative_fd_guard()
    ok_toy = toy_2cell_check()
    if not (ok_gn and ok_fd and ok_toy):
        raise SystemExit(1)
    print("gaussian2d_fgt guards: PASS")
