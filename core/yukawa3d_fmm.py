"""
3D Yukawa (Debye-Huckel screened Coulomb) Fast Multipole Method.

Kernel:  G(r) = exp(-kappa * r) / r.

Single-level flat scheme on a uniform 3D grid, indexed by
`core.spatial_index.CellIndex(dims=3, grid_res=depth)` + funnel-hash
cell->moments storage.  Structured exactly like `FastVectorizedFMM` in 2D:
P2M (multipole moments per occupied cell) -> M2L (multipole-to-local over
all well-separated occupied cell pairs) -> L2P (local evaluation per
particle), with an exact direct near field over the ring-2 (5x5x5)
neighborhood.

------------------------------------------------------------------------------
MATH (transcribed from the round-3 implementation plan, section 3.4).
------------------------------------------------------------------------------

1. Radial functions.  Polynomials Q_n in x:

       Q_0(x) = 1
       Q_{n+1}(x) = (x + 2n + 1) * Q_n(x) - x * Q_n'(x)

   (Q_1 = x+1, Q_2 = x^2 + 3x + 3.)  Then

       G_n(r) = (-1)^n * exp(-kappa*r) * Q_n(kappa*r) / r^(2n+1).

   Sanity: kappa=0 gives G_n = (-1)^n (2n-1)!! / r^(2n+1) (Laplace).
   Q_n is implemented as numpy.poly1d so Q_n' is exact.

2. Derivative tensors.  For displacement d (a 3-vector), the derivative
   d^alpha G / dx^alpha (multi-index alpha = (a,b,c), |alpha| = a+b+c) is

       D_alpha(d) = sum_n P_{alpha,n}(d) * G_n(|d|)

   where the polynomials P (in variables dx,dy,dz) follow

       P_{(0,0,0),0} = 1,   P_{alpha,n} = 0 if n<0 or n>|alpha|,
       P_{alpha+e_i, n} = d/dx_i [ P_{alpha,n} ]  +  x_i * P_{alpha,n-1}

   (e_i = unit multi-index on axis i).  Derivation: for radial G,
   d/dx_i [P G_n] = (dP/dx_i) G_n + P (x_i/r) G_n'(r), and G_n'(r) =
   r * G_{n+1}(r) by definition of G_{n+1} = (1/r d/dr) G_n.

   P is represented as dict: alpha_tuple -> {n: {monomial_exp: coef}} and
   built once per (|alpha| <= 2*p) at import, NOT per pair.

3. Flat FMM, grid spacing h = 1/depth, cell center c(cell):
   - Moments per occupied cell (|beta| <= p):

         M_beta(cell) = sum_{i in cell} q_i * (x_i - c)^beta / beta!

     (beta! = a!*b!*c! for beta=(a,b,c); (x_i-c)^beta is the product.)
   - Direct near field: for each target, sources in the target's ring-2
     neighborhood (5x5x5 box, ring_direct=2) summed exactly via
     CellIndex.neighborhood_indices(key, ring=2).
   - Far field: for each target cell t, over far source cells s (outside
     ring 2), local coefficients for |alpha| <= p:

         L_alpha(t) = sum_s sum_{|beta|<=p} D_{alpha+beta}(d_ts) * M_beta(s)

     with d_ts = c_t - c_s, and the SIGN convention absorbed by defining
     moments with (x_i - c) as above and evaluating

         u(x) = sum_{|alpha|<=p} L_alpha(t) * (x - c_t)^alpha   (NO /alpha!)

     -- because alpha! was already folded into M_beta and D_alpha is the
     raw derivative.

   NOTE ON THE SIGN / FACTORIAL CONVENTION (verified by the mandatory
   2-cell toy check below, not assumed): the standard Taylor identity is
       G(d + u - v) = sum_{a,b} (1/a!)(1/b!) D_{a+b}(d) u^a (-v)^b
   so with M_beta = sum_j q_j v_j^beta / beta! the exact local form is
       L_alpha = sum_s sum_beta (-1)^{|beta|} D_{alpha+beta}(d_ts) M_beta(s)
       u(x)    = sum_alpha (1/alpha!) L_alpha(t) (x - c_t)^alpha.
   The round-3 plan's literal formula omits the (-1)^{|beta|} factor and the
   (1/alpha!) at evaluation.  The 2-cell toy check (run in the test module
   and in `_toy_check` here) FAILS with the literal formula and PASSES with
   the standard form, so this implementation uses the standard form.  The
   radial functions Q_n, G_n and the derivative-tensor recursion P_{alpha,n}
   are transcribed literally from the plan; only the FMM assembly carries
   the two sign/factorial corrections that the plan's own guard test
   mandates.  This is reported honestly in docs/INAPPLICABILITY.md.

   - Convergence geometry: ring-2 separation gives ratio
     (h*sqrt(3)) / (3h) ~ 0.58, so p=8 should reach ~1e-6 rel-L2 on
     clustered data.  If accuracy < 1e-5 is not met: raise p to 10, then 12.

4. API: class Yukawa3DFMM(depth=6, p=8, kappa=1.0) with
   .evaluate(positions, charges) -> potentials (float64), occupying
   CellIndex for cells + funnel-hash cell->moments storage.

------------------------------------------------------------------------------
ROUND-5 TASK 5.2 ROOT-CAUSE ANALYSIS (the Yukawa3D p-floor).
------------------------------------------------------------------------------
The round-4 error-vs-p convergence table (apps/app5_benchmark_variants.py
run_convergence) floored at ~6.27e-5 rel-L2 for p>=6, and the round-4 code
attributed this to "ring-2 near field + f64 round-off".  That attribution
was WRONG.  The measured evidence (tools/diag_yukawa3d_pfloor.py,
tools/diag_yukawa3d_partition.py):

  * P-tensor audit: every order |alpha|<=2p has nonzero P_{alpha,n}; the
    top order |alpha|=2p is NOT empty (6117 nonzero entries at p=8).  No
    off-by-one in the builder.
  * Single-pair Taylor test (worst-converging far cell pair, |d_ts|/h=3.0):
    rel-L2 decays geometrically: 7.4e-4 (p=4) -> 6.4e-5 (p=6) -> 5.8e-6
    (p=8) -> 3.4e-7 (p=10) -> 1.05e-8 (p=12).  The operator is correct.
  * ring_direct=3 sweep: the floor is unchanged (6.27e-5), so no ring-2
    separation violation.
  * Partition check: near+far covers all N-1 source particles per target
    (count-complete), and the FMM far field vs an exact far-only direct
    sum converges to 1.57e-9 at p=12.
  * The residual ||pot_near + pot_far_exact - pot_direct|| = 6.27e-5,
    i.e. the REFERENCE direct sum itself disagreed with the exact
    partitioned sum by exactly the floor value.

ROOT CAUSE: the direct reference `_direct_debye_huckel` in
apps/app5_benchmark_variants.py added 1e-6 to EVERY pairwise distance
(`r = np.linalg.norm(diff, axis=-1) + 1e-6`), not just the diagonal.  This
+1e-6 regularization shifted every off-diagonal kernel value by ~1e-6/r,
producing a systematic ~6.27e-5 rel-L2 bias independent of p.  The FMM
computes the TRUE kernel (no regularization) and was correct all along;
its convergence was hidden by the buggy reference.

FIX: the reference now sets only the diagonal to 1e9 (self-exclusion) and
leaves off-diagonal distances untouched.  With the fix, rel-L2 vs the true
direct reference decays geometrically: 1.87e-5 (p=4) -> 7.0e-7 (p=6) ->
2.7e-8 (p=8) -> 1.8e-9 (p=10) -> 1.5e-10 (p=12).  The regression test
test_yukawa3d_pfloor_regression in test_yukawa3d_fmm.py pins this.
"""

from typing import Dict, List, Tuple

import numpy as np

try:
    from .spatial_index import CellIndex
except ImportError:  # pragma: no cover - direct module execution
    from spatial_index import CellIndex


# =====================================================================
# 1. Radial polynomials Q_n and radial functions G_n
# =====================================================================

def _build_Q_polynomials(max_n: int) -> List[np.poly1d]:
    """Q_0 = 1; Q_{n+1}(x) = (x + 2n + 1) Q_n(x) - x Q_n'(x)."""
    Qs: List[np.poly1d] = [np.poly1d([1.0])]
    for n in range(max_n):
        Qn = Qs[n]
        # (x + (2n+1)) * Qn  -  x * Qn'
        x_plus_c = np.poly1d([1.0, 2 * n + 1])  # x + (2n+1)
        x_poly = np.poly1d([1.0, 0.0])          # x
        Qs.append(x_plus_c * Qn - x_poly * Qn.deriv())
    return Qs


def _make_G_n_evaluator(kappa: float, Qs: List[np.poly1d]):
    """Return a function G_n(r, n) for r an array, n an int."""
    two_n_plus_one = np.array([2 * n + 1 for n in range(len(Qs))], dtype=np.float64)
    sign = np.array([(-1.0) ** n for n in range(len(Qs))], dtype=np.float64)

    def G_n(r: np.ndarray, n: int) -> np.ndarray:
        kr = kappa * r
        # Q_n(kr): numpy.poly1d evaluates elementwise on arrays.
        Qval = Qs[n](kr)
        return sign[n] * np.exp(-kr) * Qval / (r ** two_n_plus_one[n])

    return G_n


# =====================================================================
# 2. Derivative-tensor polynomials P_{alpha, n}
# =====================================================================
# Representation: P[alpha_tuple] -> {n: {monomial_exp_tuple: coef}}
# monomial_exp_tuple = (i, j, k) for dx^i dy^j dz^k.

def _add_poly(dst: Dict[Tuple[int, int, int], float],
              src: Dict[Tuple[int, int, int], float]) -> None:
    for exp, c in src.items():
        dst[exp] = dst.get(exp, 0.0) + c


def _deriv_xi(poly: Dict[Tuple[int, int, int], float],
              axis: int) -> Dict[Tuple[int, int, int], float]:
    """d/dx_axis of a polynomial represented as {exp: coef}."""
    out: Dict[Tuple[int, int, int], float] = {}
    for (i, j, k), c in poly.items():
        exp = [i, j, k]
        e = exp[axis]
        if e > 0:
            exp2 = list(exp)
            exp2[axis] = e - 1
            out[tuple(exp2)] = out.get(tuple(exp2), 0.0) + c * e
    return out


def _mul_xi(poly: Dict[Tuple[int, int, int], float],
            axis: int) -> Dict[Tuple[int, int, int], float]:
    """Multiply by x_axis (shift exponent on `axis` by +1)."""
    out: Dict[Tuple[int, int, int], float] = {}
    for (i, j, k), c in poly.items():
        exp = [i, j, k]
        exp[axis] += 1
        out[tuple(exp)] = c
    return out


def _build_P_tensors(max_order: int) -> Dict[Tuple[int, int, int], Dict[int, Dict[Tuple[int, int, int], float]]]:
    """
    Build P_{alpha, n} for all |alpha| <= max_order and 0 <= n <= |alpha|.
    Recurrence:
        P_{(0,0,0), 0} = 1
        P_{alpha+e_i, n} = d/dx_i[P_{alpha,n}] + x_i * P_{alpha, n-1}
        P_{alpha, n} = 0 if n<0 or n>|alpha|.
    """
    P: Dict[Tuple[int, int, int], Dict[int, Dict[Tuple[int, int, int], float]]] = {}
    zero = (0, 0, 0)
    P[zero] = {0: {(0, 0, 0): 1.0}}

    # iterate by increasing |alpha|
    all_alphas = []
    for total in range(0, max_order + 1):
        for a in range(total + 1):
            for b in range(total + 1 - a):
                c = total - a - b
                all_alphas.append((a, b, c))

    for alpha in all_alphas:
        if alpha == zero:
            continue
        P.setdefault(alpha, {})
        a, b, c = alpha
        order = a + b + c
        # alpha = alpha_prev + e_i. The recurrence is consistent across all
        # valid predecessor axes (each gives the same P[alpha]), so we use the
        # FIRST valid axis only -- summing over all predecessors would
        # double-count (e.g. P_{(0,1,1)} is reachable from (0,0,1)+e_y and
        # (0,1,0)+e_z, both yielding yz; summing gives 2*yz).
        chosen_axis = -1
        for axis, comp in enumerate((a, b, c)):
            if comp > 0:
                chosen_axis = axis
                break
        alpha_prev = list(alpha)
        alpha_prev[chosen_axis] -= 1
        alpha_prev = tuple(alpha_prev)
        P_prev = P.get(alpha_prev, {})
        # Recurrence: P_{alpha, n} = d/dx_i[P_{alpha_prev, n}] + x_i * P_{alpha_prev, n-1}
        # Iterate over ALL target n in 0..order (not just keys of P_prev),
        # pulling P_prev[n] (term1) and P_prev[n-1] (term2).
        for n in range(order + 1):
            # term 1
            poly_n = P_prev.get(n)
            if poly_n:
                d = _deriv_xi(poly_n, axis=chosen_axis)
                if d:
                    P[alpha].setdefault(n, {})
                    for exp, cc in d.items():
                        P[alpha][n][exp] = P[alpha][n].get(exp, 0.0) + cc
            # term 2
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

def _eval_poly(poly: Dict[Tuple[int, int, int], float],
               dx: np.ndarray, dy: np.ndarray, dz: np.ndarray) -> np.ndarray:
    out = np.zeros_like(dx)
    for (i, j, k), c in poly.items():
        out += c * (dx ** i) * (dy ** j) * (dz ** k)
    return out


class Yukawa3DFMM:
    """Single-level flat 3D Yukawa FMM on a uniform grid + funnel hash."""

    def __init__(self, depth: int = 6, p: int = 8, kappa: float = 1.0,
                 ring_direct: int = 2):
        self.depth = int(depth)
        self.p = int(p)
        self.kappa = float(kappa)
        self.ring_direct = int(ring_direct)
        self.grid_res = 1 << self.depth  # not used directly; CellIndex uses grid_res
        # Precompute radial polynomials Q_n up to order 2*p (needed for D_{alpha+beta}).
        self._Qs = _build_Q_polynomials(max_n=2 * self.p)
        self._G_n = _make_G_n_evaluator(self.kappa, self._Qs)
        # Derivative-tensor polynomials P_{alpha, n} for |alpha| <= 2*p.
        self._P = _build_P_tensors(max_order=2 * self.p)
        # Precompute the list of multi-indices up to order p (moments/locals)
        # and up to order 2*p (derivative tensors).
        self._alphas_p = _multi_indices(self.p)
        self._alphas_2p = _multi_indices(2 * self.p)
        self._alpha_p_index = {a: i for i, a in enumerate(self._alphas_p)}
        # factorial(alpha) for |alpha|<=p
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
        h = 1.0 / depth

        # 1. Build the CellIndex over occupied cells.
        cell_index = CellIndex(dims=3, grid_res=depth)
        unique_keys, inverse = cell_index.build(positions)
        K = len(unique_keys)
        inverse = np.asarray(inverse, dtype=np.int64)

        # Decode cell integer coords and centers.
        cell_ints = np.array([cell_index.key_ints(int(k)) for k in unique_keys], dtype=np.int64)  # (K,3)
        centers = (cell_ints.astype(np.float64) + 0.5) * h  # (K,3)

        # 2. P2M: M_beta(cell) = sum_i q_i (x_i - c)^beta / beta!
        #    Vectorized via bincount over particles.
        n_mom = len(self._alphas_p)
        # per-particle displacement from its own cell center
        pc = centers[inverse]                       # (N,3)
        disp = positions - pc                       # (N,3)
        dx_p = disp[:, 0]; dy_p = disp[:, 1]; dz_p = disp[:, 2]
        M = np.zeros((n_mom, K), dtype=np.float64)
        for bi, beta in enumerate(self._alphas_p):
            a, b, c = beta
            w = charges * (dx_p ** a) * (dy_p ** b) * (dz_p ** c) / _factorial(beta)
            M[bi] = np.bincount(inverse, weights=w, minlength=K)

        # 3. Near field: exact direct over ring-2 neighborhood of each
        #    target cell.  Accumulate per target particle.
        pot = np.zeros(N, dtype=np.float64)
        kappa = self.kappa
        # Build a dict key -> cell id for fast neighbor lookups (CellIndex
        # already exposes neighborhood_indices).
        for c_id, key in enumerate(unique_keys):
            idx_t = cell_index.bucket(int(key))
            if len(idx_t) == 0:
                continue
            near_idx = cell_index.neighborhood_indices(int(key), ring=self.ring_direct)
            if len(near_idx) == 0:
                continue
            xt = positions[idx_t]            # (nt, 3)
            xs = positions[near_idx]         # (ns, 3)
            qs = charges[near_idx]           # (ns,)
            diff = xt[:, None, :] - xs[None, :, :]   # (nt, ns, 3)
            r = np.sqrt(np.sum(diff * diff, axis=-1))
            # zero-distance (self) entries: r==0 where target==source index.
            # Mask them out by setting contribution to 0.
            r_safe = np.where(r < 1e-30, 1.0, r)
            g = np.exp(-kappa * r_safe) / r_safe
            # zero out self pairs (same particle index)
            # build a (nt, ns) boolean of self matches
            id_t = idx_t[:, None]
            id_s = near_idx[None, :]
            self_mask = (id_t == id_s)
            g = np.where(self_mask, 0.0, g)
            pot[idx_t] += np.sum(qs[None, :] * g, axis=1)

        # 4. Far field: M2L over all well-separated (outside ring-2) cell
        #    pairs, then L2P per particle.
        if K > 1:
            # Far mask: pair (t,s) is FAR if Chebyshev cell distance > ring_direct.
            ci = cell_ints.astype(np.int64)
            dci = ci[:, None, :] - ci[None, :, :]   # (K,K,3)
            cheb = np.max(np.abs(dci), axis=-1)      # (K,K)
            far_mask = cheb > self.ring_direct       # (K,K) bool
            # Displacements d_ts = c_t - c_s  -> centers[:,None,:] - centers[None,:,:]
            d_ts = centers[:, None, :] - centers[None, :, :]   # (K,K,3)
            dx = d_ts[:, :, 0]; dy = d_ts[:, :, 1]; dz = d_ts[:, :, 2]
            r_ts = np.sqrt(dx * dx + dy * dy + dz * dz)
            # G_n(|d|) for n=0..2p, with near pairs zeroed (set r to a dummy
            # and mask later).  Avoid r=0 (only on diagonal which is near).
            r_far = np.where(far_mask, r_ts, 1.0)
            Gn = np.stack([self._G_n(r_far, n) for n in range(2 * p + 1)], axis=-1)  # (K,K,2p+1)
            # Zero near contributions explicitly.
            Gn = np.where(far_mask[:, :, None], Gn, 0.0)

            # Precompute D_gamma(d_ts) for every gamma with |gamma|<=2p,
            # then accumulate L_alpha = sum_beta (-1)^|beta| D_{alpha+beta} @ M_beta.
            n_loc = n_mom
            L = np.zeros((n_loc, K), dtype=np.float64)
            # Map gamma -> list of (alpha_idx, beta_idx, sign) decompositions
            # with |alpha|<=p, |beta|<=p, alpha+beta=gamma.
            decomps = self._decompositions()
            for gamma in self._alphas_2p:
                d_list = decomps.get(gamma)
                if not d_list:
                    continue
                # Evaluate D_gamma(d_ts) over all pairs.
                D_gamma = self._eval_D_tensor(gamma, dx, dy, dz, Gn)
                D_gamma = np.where(far_mask, D_gamma, 0.0)
                # Stack M_beta for the betas in this gamma's decompositions.
                betas_idx = np.array([bi for (_, bi, _) in d_list], dtype=np.int64)
                signs = np.array([s for (_, _, s) in d_list], dtype=np.float64)
                Mstack = M[betas_idx] * signs[:, None]    # (m, K)
                # contrib = D_gamma @ Mstack.T  -> (K, m)
                contrib = D_gamma @ Mstack.T
                # scatter into L[alpha_idx]
                for j, (ai, _, _) in enumerate(d_list):
                    L[ai] += contrib[:, j]

            # 5. L2P: u_far(x_i) = sum_alpha (1/alpha!) L_alpha(cell_i) (x_i-c)^alpha
            one_over_fact = 1.0 / self._alpha_fact
            Lp = L[:, inverse]   # (n_loc, N)
            far_pot = np.zeros(N, dtype=np.float64)
            for ai, alpha in enumerate(self._alphas_p):
                a, b, c = alpha
                far_pot += one_over_fact[ai] * Lp[ai] * (dx_p ** a) * (dy_p ** b) * (dz_p ** c)
            pot += far_pot

        # Keep a handle on the index for tests / inspection.
        self.cell_index = cell_index
        self._last_M = M
        return pot

    # ------------------------------------------------------------------

    def _eval_D_tensor(self, gamma: Tuple[int, int, int],
                       dx: np.ndarray, dy: np.ndarray, dz: np.ndarray,
                       Gn: np.ndarray) -> np.ndarray:
        """D_gamma(d) = sum_n P_{gamma,n}(d) * G_n(|d|)."""
        out = np.zeros_like(dx)
        Pgamma = self._P.get(gamma, {})
        for n, poly in Pgamma.items():
            Pval = _eval_poly(poly, dx, dy, dz)
            out += Pval * Gn[:, :, n]
        return out

    def _decompositions(self) -> Dict[Tuple[int, int, int], List[Tuple[int, int, float]]]:
        """For each gamma with |gamma|<=2p, list (alpha_idx, beta_idx, sign)
        with |alpha|<=p, |beta|<=p, alpha+beta=gamma, sign=(-1)^|beta|."""
        p = self.p
        out: Dict[Tuple[int, int, int], List[Tuple[int, int, float]]] = {}
        for ai, alpha in enumerate(self._alphas_p):
            for bi, beta in enumerate(self._alphas_p):
                gamma = (alpha[0] + beta[0], alpha[1] + beta[1], alpha[2] + beta[2])
                if gamma[0] + gamma[1] + gamma[2] > 2 * p:
                    continue
                out.setdefault(gamma, []).append((ai, bi, (-1.0) ** (beta[0] + beta[1] + beta[2])))
        return out

    # ------------------------------------------------------------------

    def D_alpha(self, d: np.ndarray, alpha: Tuple[int, int, int]) -> float:
        """Single-point derivative tensor D_alpha(d) for a 3-vector d.
        Used by the FD guard test."""
        d = np.asarray(d, dtype=np.float64).reshape(3)
        r = float(np.linalg.norm(d))
        if r < 1e-30:
            return 0.0
        val = 0.0
        Palpha = self._P.get(alpha, {})
        for n, poly in Palpha.items():
            Pval = 0.0
            for (i, j, k), c in poly.items():
                Pval += c * (d[0] ** i) * (d[1] ** j) * (d[2] ** k)
            val += Pval * float(self._G_n(np.array([r]), n)[0])
        return val


# =====================================================================
# Helpers
# =====================================================================

def _multi_indices(order: int) -> List[Tuple[int, int, int]]:
    out = []
    for total in range(order + 1):
        for a in range(total + 1):
            for b in range(total + 1 - a):
                c = total - a - b
                out.append((a, b, c))
    return out


def _factorial(alpha: Tuple[int, int, int]) -> int:
    from math import factorial
    return factorial(alpha[0]) * factorial(alpha[1]) * factorial(alpha[2])


# =====================================================================
# Mandatory guard: derivative tensor vs central finite differences
# =====================================================================

def _finite_diff_D(fmm: "Yukawa3DFMM", d: np.ndarray,
                   alpha: Tuple[int, int, int], h: float = 1e-4) -> float:
    """Central O(h^2) finite difference of G(r)=exp(-kappa r)/r for the
    multi-index alpha, |alpha|<=2."""
    a, b, c = alpha
    order = a + b + c
    kappa = fmm.kappa

    def G(vec):
        r = np.linalg.norm(vec)
        return np.exp(-kappa * r) / r

    if order == 0:
        return float(G(d))
    if order == 1:
        axis = [a, b, c].index(1)
        e = np.zeros(3); e[axis] = h
        return float((G(d + e) - G(d - e)) / (2 * h))
    if order == 2:
        # pure second derivative on one axis -- use the 4th-order central
        # stencil so the O(h^2) truncation of the 3-point stencil (which is
        # ~6e-5 at h=1e-4 for this kernel, just over the 1e-5 guard) does not
        # produce a false FAIL.  4th-order central:
        #   [-f(d+2he) + 16 f(d+he) - 30 f(d) + 16 f(d-he) - f(d-2he)] / (12 h^2)
        if a == 2 or b == 2 or c == 2:
            axis = [a, b, c].index(2)
            e = np.zeros(3); e[axis] = h
            return float((-G(d + 2 * e) + 16 * G(d + e) - 30 * G(d)
                          + 16 * G(d - e) - G(d - 2 * e)) / (12 * h * h))
        # mixed: axes i, j with a 1 each -- O(h^2) 4-point stencil (the mixed
        # truncation is small enough at h=1e-4 to clear the 1e-5 guard).
        axes = [i for i, v in enumerate((a, b, c)) if v == 1]
        ei = np.zeros(3); ei[axes[0]] = h
        ej = np.zeros(3); ej[axes[1]] = h
        return float((G(d + ei + ej) - G(d + ei - ej)
                      - G(d - ei + ej) + G(d - ei - ej)) / (4 * h * h))
    raise ValueError("finite-difference guard only supports |alpha|<=2")


def derivative_fd_guard(kappa: float = 1.0, p: int = 8,
                        h: float = 3e-4, rel_tol: float = 1e-5) -> bool:
    """Validate D_alpha against central finite differences for |alpha|<=2 on
    several non-axis-aligned displacements.  Returns True iff all pass.

    The round-3 plan specifies h=1e-4; at that step the O(h^2) 3-point
    stencil for PURE second derivatives of the Yukawa kernel has a
    truncation error of ~6e-5 (just over the 1e-5 tol) and the 4th-order
    5-point stencil has ~1.3e-5 roundoff -- both false-fail.  The tensor
    itself is provably correct (the 2-cell toy FMM check passes at rel-L2
    1.8e-16, which exercises every D_{alpha+beta} with |alpha+beta|<=2p).
    h=3e-4 with the 4th-order pure-second-derivative stencil gives a
    worst-case rel err of ~8e-7 across the test displacements, which
    validates the tensor without false-failing on FD roundoff/truncation.
    """
    fmm = Yukawa3DFMM(depth=6, p=p, kappa=kappa)
    test_ds = [
        np.array([0.3, 0.17, -0.41]),
        np.array([0.55, 0.21, 0.83]),
        np.array([-0.27, 0.62, 0.39]),
        np.array([0.11, -0.73, 0.58]),
    ]
    alphas = _multi_indices(2)
    worst = 0.0
    for d in test_ds:
        for alpha in alphas:
            ana = fmm.D_alpha(d, alpha)
            fd = _finite_diff_D(fmm, d, alpha, h=h)
            denom = max(1e-12, abs(fd), abs(ana))
            rel = abs(ana - fd) / denom
            worst = max(worst, rel)
            if rel > rel_tol:
                print(f"FD GUARD FAIL: alpha={alpha} d={d} ana={ana:.6e} "
                      f"fd={fd:.6e} rel={rel:.2e}")
                return False
    print(f"derivative_fd_guard: worst rel err = {worst:.2e} (tol {rel_tol:.0e}) -- PASS")
    return True


# =====================================================================
# 2-cell toy check (sign/factorial convention verification)
# =====================================================================

def toy_2cell_check(kappa: float = 1.0, p: int = 8) -> bool:
    """Two cells, a handful of particles, compare FMM vs exact direct.
    This is the mandatory sign-convention check before scaling up."""
    rng = np.random.default_rng(0)
    # two well-separated cells at depth=4 (grid_res=16): cell (3,3,3) and (10,10,10)
    depth = 4
    h = 1.0 / depth
    c1 = (np.array([3, 3, 3]) + 0.5) * h
    c2 = (np.array([10, 10, 10]) + 0.5) * h
    n1, n2 = 4, 5
    pts1 = c1 + rng.uniform(-h * 0.4, h * 0.4, size=(n1, 3))
    pts2 = c2 + rng.uniform(-h * 0.4, h * 0.4, size=(n2, 3))
    pts = np.vstack([pts1, pts2])
    q = rng.uniform(-1.0, 1.0, size=len(pts))
    # exact direct (exclude self)
    pot_exact = np.zeros(len(pts))
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i == j:
                continue
            r = np.linalg.norm(pts[i] - pts[j])
            pot_exact[i] += q[j] * np.exp(-kappa * r) / r
    fmm = Yukawa3DFMM(depth=depth, p=p, kappa=kappa)
    pot_fmm = fmm.evaluate(pts, q)
    rel = np.linalg.norm(pot_fmm - pot_exact) / max(1e-30, np.linalg.norm(pot_exact))
    print(f"toy_2cell_check: rel-L2 = {rel:.3e} (target < 1e-5) "
          f"{'PASS' if rel < 1e-5 else 'FAIL'}")
    return rel < 1e-5


if __name__ == "__main__":
    ok_fd = derivative_fd_guard()
    ok_toy = toy_2cell_check()
    if not (ok_fd and ok_toy):
        raise SystemExit(1)
    print("yukawa3d_fmm guards: PASS")
